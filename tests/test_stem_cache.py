from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio2gp import separate
from audio2gp.config import Options
from audio2gp.types import PreprocessedAudio
from audio2gp.utils import write_json


def _audio(tmp_path: Path, payload: bytes = b"RIFF-test-audio") -> PreprocessedAudio:
    job = tmp_path / "work" / "job-1"
    job.mkdir(parents=True)
    path = job / "song.wav"
    path.write_bytes(payload)
    return PreprocessedAudio(path=path, duration_s=20.0, samplerate=44100)


def _options(tmp_path: Path, **kwargs) -> Options:
    return Options(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        **kwargs,
    )


def _fill_cache(cache_dir: Path, *, warnings: list[str] | None = None) -> None:
    cache_dir.mkdir(parents=True)
    for index, name in enumerate(separate.STEM_ORDER):
        (cache_dir / f"{name}.wav").write_bytes(f"stem-{name}-{index}".encode())
    write_json(
        cache_dir / "meta.json",
        {"cache_version": separate.CACHE_VERSION, "warnings": warnings or []},
    )


def test_cache_key_is_content_addressed_and_model_specific(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    key = separate.stem_cache_dir(audio, _options(tmp_path))
    assert key.parent.name == ".stem_cache"
    assert key.parent.parent == (tmp_path / "work").resolve()
    assert len(key.name) == 32

    same = separate.stem_cache_dir(audio, _options(tmp_path))
    assert same == key

    other_model = separate.stem_cache_dir(
        audio, _options(tmp_path, demucs_model="htdemucs")
    )
    assert other_model != key

    audio.path.write_bytes(b"different audio")
    changed = separate.stem_cache_dir(audio, _options(tmp_path))
    assert changed != key


def test_cache_hit_copies_stems_and_returns_warnings(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    options = _options(tmp_path)
    cache_dir = separate.stem_cache_dir(audio, options)
    _fill_cache(cache_dir, warnings=["bass stem is silent"])

    progress: list[tuple[float, str]] = []
    paths, warnings, used_device = separate.separate(
        audio,
        options,
        "cuda",
        progress_cb=lambda pct, message: progress.append((pct, message)),
    )
    assert used_device == "cuda"
    assert set(paths) == set(separate.STEM_ORDER)
    for name, path in paths.items():
        assert path == audio.path.parent / "stems" / f"{name}.wav"
        assert path.read_bytes() == (cache_dir / f"{name}.wav").read_bytes()
    assert "bass stem is silent" in warnings
    assert any("复用" in warning for warning in warnings)
    assert progress[-1][0] == 100
    assert any("缓存" in message for _, message in progress)


def test_incomplete_or_empty_cache_is_ignored(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    cache_dir = separate.stem_cache_dir(audio, _options(tmp_path))
    _fill_cache(cache_dir)
    (cache_dir / "bass.wav").write_bytes(b"")
    assert separate._cached_paths(cache_dir) is None
    (cache_dir / "bass.wav").unlink()
    assert separate._cached_paths(cache_dir) is None


def test_publish_cache_does_not_replace_existing_entry(tmp_path: Path) -> None:
    target = tmp_path / "cache" / "entry"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "marker").write_text("first")
    (second / "marker").write_text("second")

    separate._publish_cache(first, target)
    assert (target / "marker").read_text() == "first"
    separate._publish_cache(second, target)
    assert (target / "marker").read_text() == "first"
    assert not second.exists()


def test_cache_metadata_is_json(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    cache_dir = separate.stem_cache_dir(audio, _options(tmp_path))
    _fill_cache(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["cache_version"] == separate.CACHE_VERSION


@pytest.mark.parametrize("name", separate.STEM_ORDER)
def test_cached_paths_require_each_stem(tmp_path: Path, name: str) -> None:
    audio = _audio(tmp_path)
    cache_dir = separate.stem_cache_dir(audio, _options(tmp_path))
    _fill_cache(cache_dir)
    (cache_dir / f"{name}.wav").unlink()
    assert separate._cached_paths(cache_dir) is None
