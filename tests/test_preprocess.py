from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio2gp.config import Options
from audio2gp.preprocess import _fold_bpm, estimate_bpm, prepare
from audio2gp.utils import read_audio

SR = 44100


def _write_sine(path: Path, duration: float = 2.0, sr: int = 48000) -> None:
    t = np.arange(int(duration * sr)) / sr
    y = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    sf.write(path, y, sr)


def _write_click_track(path: Path, bpm: float = 120.0, duration: float = 6.0) -> None:
    rng = np.random.default_rng(3)
    n = int(duration * SR)
    y = np.zeros(n, dtype=np.float64)
    period = 60.0 / bpm
    hit_len = int(0.035 * SR)
    decay = np.exp(-np.arange(hit_len) / (0.008 * SR))
    for at in np.arange(0.1, duration, period):
        start = int(at * SR)
        end = min(n, start + hit_len)
        y[start:end] += 0.9 * rng.normal(0.0, 1.0, end - start) * decay[: end - start]
    sf.write(path, y, SR)


def test_prepare_normalizes_and_overrides_bpm(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    _write_sine(source, duration=2.0, sr=48000)
    work = tmp_path / "work"
    progress: list[tuple[float, str]] = []

    result = prepare(
        source,
        work,
        Options(bpm_override=120.0),
        progress_cb=lambda pct, msg: progress.append((pct, msg)),
    )

    assert result.path == work / "song.wav"
    assert result.path.is_file()
    assert result.samplerate == 44100
    assert result.bpm == 120.0
    assert result.duration_s == pytest.approx(2.0, abs=0.05)
    assert progress and progress[-1][0] == 100.0
    assert all(0.0 <= pct <= 100.0 for pct, _ in progress)

    data, sr = read_audio(result.path)
    assert sr == 44100
    assert data.shape[1] == 2
    assert len(data) == pytest.approx(2.0 * SR, abs=2000)


def test_prepare_estimates_click_bpm(tmp_path: Path) -> None:
    source = tmp_path / "click.wav"
    _write_click_track(source, bpm=120.0)
    result = prepare(source, tmp_path / "work", Options(bpm_override=None))
    assert result.bpm == pytest.approx(120.0, abs=3.0)


def test_estimate_bpm_silent_falls_back_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(SR, dtype=np.float32), SR)
    bpm, warnings = estimate_bpm(path)
    assert bpm == 120.0
    assert any("BPM estimation failed" in warning for warning in warnings)


def test_fold_bpm_into_musical_range() -> None:
    assert _fold_bpm(50.0) == 100.0
    assert _fold_bpm(240.0) == 120.0
    assert _fold_bpm(300.0) == 150.0
    assert _fold_bpm(220.0) == 110.0


def test_prepare_missing_empty_and_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / "missing.wav", tmp_path / "work", Options())

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        prepare(empty, tmp_path / "work", Options())

    invalid = tmp_path / "not_audio.txt"
    invalid.write_text("definitely not audio", encoding="utf-8")
    with pytest.raises(RuntimeError):
        prepare(invalid, tmp_path / "work", Options(bpm_override=120.0))
