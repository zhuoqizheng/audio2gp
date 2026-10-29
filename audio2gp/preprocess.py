from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Options, bar_seconds
from .types import PreprocessedAudio, ProgressCB
from .utils import ensure_dir, ffprobe_duration, normalize_audio

# A stage-local callback takes 0..100; pipeline maps it to the overall budget.
LocalCB = Callable[[float, str], None]


def _emit(cb: LocalCB | None, pct: float, message: str) -> None:
    if cb:
        cb(max(0.0, min(100.0, float(pct))), message)


def _fold_bpm(bpm: float) -> float:
    while bpm < 60:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return bpm


def estimate_bpm(wav_path: Path, logger: logging.Logger | None = None) -> tuple[float, list[str]]:
    import librosa

    warnings: list[str] = []
    try:
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        if y.size < sr // 2:
            raise ValueError("audio is shorter than 0.5 s")
        y, _ = librosa.effects.trim(y, top_db=35)
        if y.size < sr // 2:
            raise ValueError("audio is effectively silent")
        # Percussion is the most reliable tempo source in a mixed song.
        perc = librosa.effects.percussive(y, margin=(1.0, 1.0))
        if float(np.max(np.abs(perc))) < 1e-5:
            perc = y
        tempo, _ = librosa.beat.beat_track(y=perc, sr=sr, units="frames")
        bpm = float(np.atleast_1d(tempo)[0])
        if not math.isfinite(bpm) or bpm <= 0:
            raise ValueError(f"invalid tempo {bpm!r}")
        bpm = _fold_bpm(bpm)
        return round(bpm, 2), warnings
    except Exception as exc:
        warnings.append(f"BPM estimation failed ({exc}); using 120 BPM. Adjust in the UI.")
        if logger:
            logger.warning("BPM estimation failed: %s", exc)
        return 120.0, warnings


def prepare(
    input_path: Path,
    work_dir: Path,
    options: Options,
    logger: logging.Logger | None = None,
    progress_cb: LocalCB | None = None,
) -> PreprocessedAudio:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {input_path}")
    if input_path.stat().st_size == 0:
        raise ValueError(f"Input audio is empty: {input_path}")

    ensure_dir(work_dir)
    wav_path = work_dir / "song.wav"
    _emit(progress_cb, 5, "检查音频")
    normalize_audio(input_path, wav_path, logger=logger)
    _emit(progress_cb, 35, "已转换为 44.1 kHz 立体声 WAV")

    try:
        duration = ffprobe_duration(wav_path)
    except Exception:
        duration = 0.0
    if duration <= 0:
        raise RuntimeError("Normalized audio has no readable duration")
    _emit(progress_cb, 50, f"时长 {duration:.1f} 秒")

    if options.bpm_override:
        bpm = float(options.bpm_override)
        bpm_warnings: list[str] = []
    else:
        _emit(progress_cb, 60, "估计 BPM")
        bpm, bpm_warnings = estimate_bpm(wav_path, logger=logger)
    _emit(progress_cb, 100, f"BPM {bpm:.1f}")

    warnings = list(bpm_warnings)
    if duration / max(bar_seconds(bpm), 1e-9) > 2048:
        warnings.append("Song is longer than 2048 bars; GP5 output may be unwieldy.")

    return PreprocessedAudio(
        path=wav_path,
        duration_s=float(duration),
        samplerate=44100,
        bpm=float(bpm),
        warnings=warnings,
    )
