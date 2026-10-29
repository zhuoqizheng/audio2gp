from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .config import Options
from .types import PreprocessedAudio
from .utils import ensure_dir, is_silent, read_json, write_json

LocalCB = Callable[[float, str], None]

STEM_ORDER = ("vocals", "drums", "bass", "other")
# Bump this whenever normalization or the Demucs call parameters change; old
# cached stems then stop matching instead of silently bypassing the new code.
CACHE_VERSION = 1


def _emit(cb: LocalCB | None, pct: float, message: str) -> None:
    if cb:
        cb(max(0.0, min(100.0, float(pct))), message)


def _save_stem(tensor, path: Path, samplerate: int) -> None:
    """Prefer Demucs' save_audio; fall back to soundfile for API changes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from demucs.api import save_audio

        save_audio(tensor, str(path), samplerate=samplerate, bits_per_sample=16)
        return
    except Exception:
        pass
    import numpy as np
    import soundfile as sf

    data = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    if data.ndim == 1:
        data = data[None, :]
    sf.write(str(path), data.T, samplerate, subtype="PCM_16")


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text and "memory" in text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stem_cache_dir(audio: PreprocessedAudio, options: Options) -> Path:
    """Content-addressed cache shared by all jobs under the same work root.

    The normalized WAV content, model and fixed separator settings form the
    key. Two uploads of the same song with a different filename therefore hit
    the same cache entry.
    """
    material = "|".join(
        (
            f"audio2gp-stems-v{CACHE_VERSION}",
            str(options.demucs_model),
            "shifts=1",
            "overlap=0.25",
            _sha256_file(Path(audio.path)),
        )
    )
    key = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return Path(audio.path).resolve().parent.parent / ".stem_cache" / key


def _cached_paths(cache_dir: Path) -> dict[str, Path] | None:
    paths: dict[str, Path] = {}
    for name in STEM_ORDER:
        path = cache_dir / f"{name}.wav"
        if not path.is_file() or path.stat().st_size == 0:
            return None
        paths[name] = path
    return paths


def _copy_cached_stems(cache_dir: Path, stems_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in STEM_ORDER:
        target = stems_dir / f"{name}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_dir / f"{name}.wav", target)
        paths[name] = target
    return paths


def _publish_cache(temp_dir: Path, cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(temp_dir, cache_dir)
    except OSError:
        # Another process may have published the same entry concurrently.
        shutil.rmtree(temp_dir, ignore_errors=True)


def separate(
    audio: PreprocessedAudio,
    options: Options,
    device: str,
    logger: logging.Logger | None = None,
    progress_cb: LocalCB | None = None,
) -> tuple[dict[str, Path], list[str], str]:
    warnings: list[str] = []
    stems_dir = Path(audio.path).parent / "stems"
    ensure_dir(stems_dir)

    cache_dir: Path | None = None
    if options.cache_stems:
        _emit(progress_cb, 1, "检查分轨缓存")
        try:
            cache_dir = stem_cache_dir(audio, options)
            cached = _cached_paths(cache_dir)
        except Exception as exc:
            warnings.append(f"无法读取 Demucs 分轨缓存，将重新分离：{exc}")
            if logger:
                logger.warning("stem cache lookup failed: %s", exc)
            cache_dir = None
            cached = None
        if cached is not None:
            try:
                _emit(progress_cb, 20, "复用已缓存的分轨")
                paths = _copy_cached_stems(cache_dir, stems_dir)  # type: ignore[arg-type]
                try:
                    meta = read_json(cache_dir / "meta.json")
                    cached_warnings = [
                        str(item) for item in (meta.get("warnings") or [])
                    ]
                except Exception:
                    cached_warnings = []
                warnings.extend(cached_warnings)
                warnings.append(f"复用了 Demucs 分轨缓存（{cache_dir.name[:8]}）。")
                _emit(progress_cb, 100, "分轨完成（缓存）")
                if logger:
                    logger.info("separate: cache hit %s", cache_dir)
                return paths, warnings, device
            except Exception as exc:
                warnings.append(f"分轨缓存复制失败，将重新分离：{exc}")
                if logger:
                    logger.warning("stem cache copy failed: %s", exc)

    from demucs.api import Separator

    _emit(progress_cb, 2, "加载 Demucs 模型")
    if logger:
        logger.info("separate: model=%s device=%s", options.demucs_model, device)

    def run_on(run_device: str) -> tuple[dict[str, Path], str]:
        last_emit = [-1.0]

        def callback(data: dict) -> None:
            try:
                models = max(1, int(data.get("models", 1)))
                model_idx = int(data.get("model_idx_in_bag", 0))
                segment = float(data.get("segment_offset", 0.0))
                length = float(data.get("audio_length", 0.0)) or 1.0
                pct = 5.0 + 90.0 * min(1.0, (model_idx + min(1.0, segment / length)) / models)
                if pct - last_emit[0] >= 0.7:
                    last_emit[0] = pct
                    _emit(progress_cb, pct, f"分离中（模型 {model_idx + 1}/{models}）")
            except Exception:
                pass

        separator = Separator(
            model=options.demucs_model,
            device=run_device,
            shifts=1,
            overlap=0.25,
            split=True,
            jobs=0,
            progress=False,
            callback=callback,
        )
        origin, demucs_stems = separator.separate_audio_file(Path(audio.path))
        _emit(progress_cb, 96, "保存分轨")
        samplerate = int(getattr(separator, "samplerate", 44100))
        paths: dict[str, Path] = {}
        for name in STEM_ORDER:
            if name not in demucs_stems:
                raise RuntimeError(f"Demucs did not return the '{name}' stem")
            path = stems_dir / f"{name}.wav"
            _save_stem(demucs_stems[name], path, samplerate)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Demucs stem was not written: {path}")
            if is_silent(path):
                warnings.append(f"{name} stem is silent")
            paths[name] = path
            if logger:
                logger.info("separate: wrote %s (%s)", path, name)
        return paths, run_device

    try:
        paths, used_device = run_on(device)
    except Exception as exc:
        if device != "cpu" and _is_oom(exc):
            warnings.append("CUDA out of memory during separation; retrying on CPU (slow).")
            if logger:
                logger.warning("CUDA OOM, retrying Demucs on CPU: %s", exc)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            paths, used_device = run_on("cpu")
        else:
            raise RuntimeError(f"Demucs separation failed on {device}: {exc}") from exc
    if cache_dir is not None:
        try:
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(tempfile.mkdtemp(prefix=".stems-", dir=cache_dir.parent))
            for name in STEM_ORDER:
                shutil.copy2(paths[name], temp_dir / f"{name}.wav")
            write_json(
                temp_dir / "meta.json",
                {
                    "cache_version": CACHE_VERSION,
                    "model": options.demucs_model,
                    "warnings": warnings,
                },
            )
            _publish_cache(temp_dir, cache_dir)
            if logger:
                logger.info("separate: cached stems at %s", cache_dir)
        except Exception as exc:
            warnings.append(f"无法写入 Demucs 分轨缓存（不影响本次结果）：{exc}")
            if logger:
                logger.warning("stem cache write failed: %s", exc)
    _emit(progress_cb, 100, "分轨完成")
    return paths, warnings, used_device
