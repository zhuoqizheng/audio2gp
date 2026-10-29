from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import BEATS_PER_BAR, bar_seconds

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


@dataclass
class RhythmInfo:
    """Beat grid and bar phase estimated from the mixed audio."""

    bpm: float
    beats_per_bar: int
    beat_times_s: list[float] = field(default_factory=list)
    downbeat_times_s: list[float] = field(default_factory=list)
    method: str = "none"
    tempo_source: str = "mix"
    warnings: list[str] = field(default_factory=list)

    @property
    def first_downbeat_s(self) -> float | None:
        return self.downbeat_times_s[0] if self.downbeat_times_s else None


def _analyze_madmom(
    wav_path: Path,
    logger: logging.Logger | None,
    progress_cb: LocalCB | None,
) -> RhythmInfo:
    """Joint beat/downbeat tracking with madmom's BLSTM ensemble.

    The source audio is already normalized to 44.1 kHz by preprocess.prepare,
    which RNNDownBeatProcessor requires. We deliberately request a four-beat
    bar: this app writes 4/4 GP5 files, and letting a generic meter detector
    choose between 3 and 4 can pick 3/4 for a swung 4/4 song.
    """
    from madmom_infer.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    _emit(progress_cb, 8, "加载强拍模型")
    processor = RNNDownBeatProcessor()
    _emit(progress_cb, 30, "提取节拍特征")
    activations = processor(str(wav_path))
    if activations is None or len(activations) < 2:
        raise ValueError("downbeat model returned no activations")
    _emit(progress_cb, 80, "解码节拍与强拍")
    tracker = DBNDownBeatTrackingProcessor(
        beats_per_bar=[BEATS_PER_BAR],
        fps=100,
    )
    tracked = tracker(activations)
    if tracked is None or len(tracked) < 2:
        raise ValueError("downbeat decoder returned no beats")

    tracked = np.asarray(tracked, dtype=float)
    beats = tracked[:, 0]
    positions = np.rint(tracked[:, 1]).astype(int)
    downbeats = beats[positions == 1]
    if downbeats.size == 0:
        raise ValueError("downbeat decoder found no bar starts")

    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
    if intervals.size == 0:
        raise ValueError("beat intervals are not usable")
    bpm = _fold_bpm(60.0 / float(np.median(intervals)))
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError(f"invalid detected BPM {bpm!r}")

    if logger:
        logger.info(
            "rhythm: madmom-infer %d beats, %d downbeats, %.2f BPM",
            len(beats),
            len(downbeats),
            bpm,
        )
    _emit(progress_cb, 100, f"强拍分析完成（{bpm:.1f} BPM）")
    return RhythmInfo(
        bpm=round(float(bpm), 2),
        beats_per_bar=BEATS_PER_BAR,
        beat_times_s=[float(value) for value in beats],
        downbeat_times_s=[float(value) for value in downbeats],
        method="madmom-infer",
    )


def _accent_phase_score(
    beat_times: np.ndarray,
    low_energy: np.ndarray,
    onset_strength: np.ndarray,
    sr: int,
    hop_length: int,
) -> int:
    """Choose the most accented beat phase as the (fallback) bar start.

    A downbeat usually carries the strongest low-frequency attack in
    pop/rock. Normalizing each feature first keeps a loud chorus from
    overwhelming the rest of the song.
    """
    frames = np.clip(
        np.rint(beat_times * sr / hop_length).astype(int),
        0,
        max(0, len(low_energy) - 1),
    )
    low = np.asarray(low_energy[frames], dtype=float)
    onset = np.asarray(onset_strength[frames], dtype=float)

    def robust_z(values: np.ndarray) -> np.ndarray:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if scale < 1e-9:
            scale = float(np.std(values))
        if scale < 1e-9:
            return np.zeros_like(values)
        return (values - median) / scale

    score = robust_z(low) + 0.35 * robust_z(onset)
    phase_scores = [float(np.mean(score[phase::BEATS_PER_BAR])) for phase in range(BEATS_PER_BAR)]
    return int(np.argmax(phase_scores))


def _analyze_librosa(
    wav_path: Path,
    logger: logging.Logger | None,
    progress_cb: LocalCB | None,
) -> RhythmInfo:
    """Dependency-light fallback: librosa beats + low-band accent phase."""
    import librosa

    _emit(progress_cb, 10, "分析节拍（librosa）")
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    if y.size < sr // 2:
        raise ValueError("audio is shorter than 0.5 s")
    perc = librosa.effects.percussive(y, margin=(1.0, 1.0))
    if float(np.max(np.abs(perc))) < 1e-5:
        perc = y
    tempo, frames = librosa.beat.beat_track(y=perc, sr=sr, units="frames")
    beat_times = np.asarray(librosa.frames_to_time(frames, sr=sr), dtype=float)
    if beat_times.size < BEATS_PER_BAR:
        raise ValueError("librosa found too few beats")
    bpm = _fold_bpm(float(np.atleast_1d(tempo)[0]))
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError(f"invalid tempo {bpm!r}")

    _emit(progress_cb, 65, "估计强拍相位")
    hop_length = 512
    spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = (freqs >= 20.0) & (freqs < 150.0)
    if np.any(low_mask):
        low_energy = np.sqrt(np.mean(spectrum[low_mask] ** 2, axis=0))
    else:
        low_energy = np.zeros(spectrum.shape[1], dtype=float)
    onset_strength = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length
    )
    phase = _accent_phase_score(
        beat_times, low_energy, onset_strength, sr, hop_length
    )
    downbeats = beat_times[phase::BEATS_PER_BAR]
    if downbeats.size == 0:
        raise ValueError("no downbeat phase found")

    if logger:
        logger.info(
            "rhythm: librosa fallback %d beats, %d downbeats, %.2f BPM",
            len(beat_times),
            len(downbeats),
            bpm,
        )
    _emit(progress_cb, 100, f"强拍分析完成（librosa，{bpm:.1f} BPM）")
    return RhythmInfo(
        bpm=round(float(bpm), 2),
        beats_per_bar=BEATS_PER_BAR,
        beat_times_s=[float(value) for value in beat_times],
        downbeat_times_s=[float(value) for value in downbeats],
        method="librosa-accent",
    )


def analyze_rhythm(
    wav_path: Path | str,
    logger: logging.Logger | None = None,
    progress_cb: LocalCB | None = None,
) -> RhythmInfo:
    """Analyze beats/downbeats, preferring madmom-infer and falling back.

    Model weights are fetched and cached by madmom-infer on first use. Any
    model/network problem is non-fatal: librosa's beat tracker plus a
    low-band accent phase keeps the app usable offline.
    """
    wav_path = Path(wav_path)
    warnings: list[str] = []
    try:
        return _analyze_madmom(wav_path, logger, progress_cb)
    except Exception as exc:
        message = f"madmom-infer downbeat analysis failed ({type(exc).__name__}: {exc}); using librosa fallback"
        warnings.append(message)
        if logger:
            logger.warning(message)

    try:
        info = _analyze_librosa(wav_path, logger, progress_cb)
        info.warnings = warnings + info.warnings
        return info
    except Exception as exc:
        message = f"beat/downbeat analysis failed ({exc}); bar 1 starts at the audio start"
        warnings.append(message)
        if logger:
            logger.warning(message)
        _emit(progress_cb, 100, "强拍分析失败，保持原始时间轴")
        return RhythmInfo(
            bpm=0.0,
            beats_per_bar=BEATS_PER_BAR,
            warnings=warnings,
            method="none",
        )


_TEMPO_RATIOS = (1.0, 2.0, 0.5, 1.5, 2.0 / 3.0)


def refine_tempo_from_drums(
    drums_path: Path | str,
    reference_bpm: float,
    logger: logging.Logger | None = None,
) -> tuple[float, list[str]]:
    """Use the separated drum stem for a more reliable global tempo.

    librosa can report half/double time on sparse or programmed drums. The
    candidate closest in log-space to the mix/downbeat tempo is preferred when
    it agrees within 4%; otherwise the drum estimate wins and a warning is
    emitted. A failure is always non-fatal: the caller keeps the mix tempo.
    """
    import librosa

    warnings: list[str] = []
    path = Path(drums_path)
    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        if y.size < sr // 2:
            raise ValueError("drum stem is shorter than 0.5 s")
        if float(np.max(np.abs(y))) < 1e-5:
            raise ValueError("drum stem is silent")
        tempo, frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(frames, sr=sr)
        if beat_times.size >= 4:
            intervals = np.diff(beat_times)
            intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
            if intervals.size >= 3:
                bpm = 60.0 / float(np.median(intervals))
            else:
                bpm = float(np.atleast_1d(tempo)[0])
        else:
            bpm = float(np.atleast_1d(tempo)[0])
        bpm = _fold_bpm(bpm)
        if not math.isfinite(bpm) or bpm <= 0:
            raise ValueError(f"invalid drum tempo {bpm!r}")

        reference = _fold_bpm(float(reference_bpm)) if reference_bpm > 0 else 0.0
        if reference > 0:
            candidates = [bpm * ratio for ratio in _TEMPO_RATIOS]
            candidates = [value for value in candidates if 30.0 <= value <= 300.0]
            chosen = min(
                candidates,
                key=lambda value: abs(math.log(value / reference)),
            )
            if abs(chosen - reference) / reference <= 0.04:
                # Keep the bar-grid tempo when the drums agree within a few
                # percent: even a 1% tempo error drifts by seconds over a full
                # song, and the downbeat model already gives us the grid phase.
                if abs(bpm - reference) / reference > 0.02:
                    warnings.append(
                        f"drum tempo {bpm:.1f} BPM agrees with the bar grid "
                        f"({reference:.1f} BPM); keeping the grid tempo"
                    )
                bpm = reference
            elif abs(bpm - reference) / reference > 0.08:
                warnings.append(
                    f"drum tempo {bpm:.1f} BPM differs from the mix tempo "
                    f"{reference:.1f} BPM; using the drum stem"
                )
        if logger:
            logger.info("tempo refine: drums=%.2f BPM reference=%.2f", bpm, reference_bpm)
        return round(float(bpm), 2), warnings
    except Exception as exc:
        message = (
            f"drum-stem tempo estimation failed ({type(exc).__name__}: {exc}); "
            "keeping the mix/downbeat tempo"
        )
        warnings.append(message)
        if logger:
            logger.warning(message)
        return float(reference_bpm), warnings


def choose_grid_offset(
    first_downbeat_s: float | None,
    first_note_s: float | None,
    bpm: float,
    *,
    beats_per_bar: int = BEATS_PER_BAR,
    denominator: int = 4,
    tolerance_s: float = 0.05,
) -> float:
    """Return the time of bar 1 tick 0 so bar lines land on downbeats.

    If the transcription has material before the first downbeat (a pickup),
    the first downbeat is placed at the start of bar 2 (or a later bar) and
    the leading bars contain that material plus rests. This preserves pickup
    notes without inventing a short, non-representable first measure.

    ``beats_per_bar`` is the meter numerator and ``denominator`` its lower
    number, so 6/8 and 12/8 produce the correct bar length.
    """
    if first_downbeat_s is None or bpm <= 0:
        return 0.0
    bar_s = bar_seconds(
        bpm, beats_per_bar=beats_per_bar, denominator=denominator
    )
    if not math.isfinite(bar_s) or bar_s <= 0 or first_downbeat_s < 0:
        return 0.0
    note_s = max(0.0, float(first_note_s or 0.0))
    if note_s >= first_downbeat_s - tolerance_s:
        return float(first_downbeat_s)
    lead_bars = max(
        1,
        int(
            math.ceil(
                (first_downbeat_s - note_s - tolerance_s) / bar_s
            )
        ),
    )
    return float(first_downbeat_s - lead_bars * bar_s)
