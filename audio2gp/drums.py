from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .config import grid_seconds
from .types import NoteEvent

LocalCB = Callable[[float, str], None]

HOP = 256
N_FFT = 2048
SR = 22050

# GM percussion keys. Toms intentionally share a small, conventional set.
KICK = 36
SNARE = 38
CLOSED_HAT = 42
OPEN_HAT = 46
CRASH = 49
TOMS = (41, 45, 47, 48, 50)


def _emit(cb: LocalCB | None, pct: float, message: str) -> None:
    if cb:
        cb(max(0.0, min(100.0, float(pct))), message)


def _band_sum(spec: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spec[mask]))


def _merge_times(times: list[float], window: float = 0.04) -> list[float]:
    if not times:
        return []
    times = sorted(times)
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] > window:
            merged.append(t)
    return merged


def _tom_for_centroid(centroid: float) -> int:
    if centroid < 180:
        return 41
    if centroid < 260:
        return 45
    if centroid < 400:
        return 47
    if centroid < 600:
        return 48
    return 50


def transcribe_drums(
    stem_path: Path,
    bpm: float,
    quantize: int,
    logger: logging.Logger | None = None,
    progress_cb: LocalCB | None = None,
) -> tuple[list[NoteEvent], list[str]]:
    import librosa

    warnings: list[str] = []
    _emit(progress_cb, 3, "分析鼓轨")
    try:
        y, sr = librosa.load(str(stem_path), sr=SR, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Could not read drum stem: {exc}") from exc
    if y.size < int(0.2 * sr) or float(np.max(np.abs(y))) < 1e-5:
        warnings.append("drum stem is silent; wrote an empty drum track")
        return [], warnings

    y_perc = librosa.effects.percussive(y, margin=(1.0, 1.0))
    if float(np.max(np.abs(y_perc))) < 1e-6:
        y_perc = y
    S = np.abs(librosa.stft(y_perc, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    n_frames = S.shape[1]
    _emit(progress_cb, 18, "检测鼓点")

    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP)
    full_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP,
        units="frames",
        backtrack=True,
        delta=0.10,
        wait=2,
    )
    bands = {
        "low": (20.0, 150.0),
        "body": (150.0, 500.0),
        "snare": (1500.0, 8000.0),
        "high": (8000.0, 16000.0),
    }
    full_times = [
        float(t)
        for t in librosa.frames_to_time(full_frames, sr=sr, hop_length=HOP)
    ]
    # onset_detect ignores frame zero; keep an initial transient (common in
    # full mixes that start on a downbeat) as a candidate.
    initial = float(np.max(np.abs(y[: max(1, int(0.02 * sr))])))
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6 and initial >= 0.2 * peak:
        full_times.append(0.0)
    # Band detectors often fire a few frames before the full-band attack.
    # Only use them for events the full-band detector missed, so their early
    # timestamps do not win over the backtracked full-band onset.
    extra_times: list[float] = []
    for low, high in bands.values():
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            continue
        energy = np.log1p(S[mask].sum(axis=0))
        flux = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
        if float(np.max(flux)) <= 0:
            continue
        frames = librosa.onset.onset_detect(
            onset_envelope=librosa.util.normalize(flux),
            sr=sr,
            hop_length=HOP,
            units="frames",
            backtrack=False,
            delta=0.08,
            wait=2,
        )
        for t in librosa.frames_to_time(frames, sr=sr, hop_length=HOP):
            t = float(t)
            if all(abs(t - full_t) > 0.04 for full_t in full_times):
                extra_times.append(t)
    all_times = _merge_times(full_times + extra_times)
    # onset_detect can report an edge artifact on the final frame.
    usable_end = max(0.0, len(y) / sr - 3 * HOP / sr)
    times = [t for t in all_times if 0.0 <= t < usable_end]
    if logger:
        logger.info("drums: %d candidate onsets", len(times))
    if not times:
        warnings.append("no drum onsets detected; wrote an empty drum track")
        return [], warnings
    _emit(progress_cb, 35, f"分类 {len(times)} 个鼓点")

    raw_hits: list[tuple[float, int, float]] = []
    refr = {KICK: 0.085, SNARE: 0.075, CLOSED_HAT: 0.035, OPEN_HAT: 0.13, CRASH: 0.28}
    last_hit: dict[int, float] = {}
    eps = 1e-9

    for t in times:
        frame = int(round(t * sr / HOP))
        frame = max(0, min(n_frames - 1, frame))
        # Backtracking returns the start of the attack, which can be a few
        # frames before the spectral peak. Pick the strongest nearby frame so
        # the attack features are measured on the actual transient.
        look_end = min(n_frames, frame + 5)
        if look_end > frame:
            offsets = np.argmax(S[:, frame:look_end].sum(axis=0))
            frame = min(n_frames - 1, frame + int(offsets))
        t = float(librosa.frames_to_time(frame, sr=sr, hop_length=HOP))
        if t >= usable_end:
            continue
        attack_end = min(n_frames, frame + 4)
        attack = S[:, frame:attack_end].mean(axis=1)
        total = _band_sum(attack, freqs, 20.0, 16000.0) + eps
        low_e = _band_sum(attack, freqs, *bands["low"])
        body_e = _band_sum(attack, freqs, *bands["body"])
        snare_e = _band_sum(attack, freqs, *bands["snare"])
        high_e = _band_sum(attack, freqs, *bands["high"])
        low_ratio = low_e / total
        body_ratio = body_e / total
        snare_ratio = snare_e / total
        high_ratio = high_e / total
        centroid = float(np.sum(freqs * attack) / (np.sum(attack) + eps))
        spec = np.maximum(attack, eps)
        flatness = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + eps))

        high_attack = _band_sum(attack, freqs, *bands["high"]) + eps
        # Measure the high-frequency tail before the next grid-level hit can
        # leak in; a wider window turns a closed hat followed by a snare into
        # a false crash/open hat.
        late_start = min(n_frames, frame + 5)
        late_end = min(n_frames, frame + 10)
        high_late = (
            _band_sum(S[:, late_start:late_end].mean(axis=1), freqs, *bands["high"])
            if late_end > late_start
            else 0.0
        )
        high_decay = high_late / high_attack

        strength = float(onset_env[min(len(onset_env) - 1, frame)]) if len(onset_env) else 1.0
        strength = max(strength, float(np.sum(attack)) * 0.01, 1e-4)

        classes: list[int] = []
        # A kick's click can lift its centroid, so use low-band dominance
        # rather than a very low centroid; the snare/hat checks below exclude
        # the low-band cases.
        is_kick = low_ratio >= 0.30 and centroid < 4000.0 and snare_ratio < 0.40
        if is_kick:
            classes.append(KICK)
        # A snare has noise plus a little body. Body energy is the useful
        # discriminator: a bare hi-hat has low≈body≈0 and a very high centroid.
        is_snare = (
            not is_kick
            and snare_ratio >= 0.10
            and body_ratio >= 0.02
            and 600.0 <= centroid <= 7000.0
            and low_ratio < 0.55
        )
        if is_snare:
            classes.append(SNARE)

        # Hats are high-frequency and have essentially no body energy.
        # Sustained high-band decay separates open hats; very long decays are
        # treated as cymbals. Prefer open-hat/closed-hat when ambiguous.
        if (
            high_ratio >= 0.25
            and centroid >= 6000.0
            and low_ratio < 0.35
            and body_ratio < 0.02
        ):
            if high_decay >= 0.85:
                classes.append(CRASH)
            elif high_decay >= 0.50:
                classes.append(OPEN_HAT)
            else:
                classes.append(CLOSED_HAT)

        if (
            not is_kick
            and not is_snare
            and body_ratio >= 0.30
            and 120.0 <= centroid <= 1300.0
            and low_ratio < 0.45
        ):
            classes.append(_tom_for_centroid(centroid))

        if not classes:
            if low_ratio >= 0.24 and centroid < 4000:
                classes.append(KICK)
            elif (
                snare_ratio >= 0.05
                and body_ratio >= 0.02
                and centroid < 7000.0
            ):
                classes.append(SNARE)
            elif high_ratio >= 0.05 and centroid >= 4000.0 and body_ratio < 0.02:
                classes.append(CLOSED_HAT)

        for pitch in dict.fromkeys(classes):
            if t - last_hit.get(pitch, -1e9) < refr.get(pitch, 0.05):
                continue
            last_hit[pitch] = t
            raw_hits.append((t, pitch, strength))

    if not raw_hits:
        warnings.append("drum onsets produced no usable GM classes; empty drum track")
        return [], warnings

    # Normalize attack strength per class, then map onto a musical velocity range.
    by_pitch: dict[int, list[float]] = {}
    for _t, pitch, strength in raw_hits:
        by_pitch.setdefault(pitch, []).append(strength)
    ranges = {
        pitch: (min(values), max(values)) for pitch, values in by_pitch.items()
    }
    grid_s = grid_seconds(bpm, quantize)
    quantized: dict[tuple[int, int], tuple[float, float]] = {}
    for t, pitch, strength in raw_hits:
        lo, hi = ranges[pitch]
        norm = 0.6 if hi - lo < 1e-9 else (strength - lo) / (hi - lo)
        velocity = 50.0 + 65.0 * max(0.0, min(1.0, norm))
        if pitch in (CLOSED_HAT, OPEN_HAT, CRASH):
            velocity -= 8.0
        velocity = max(35.0, min(120.0, velocity))
        tick = max(0, int(round(t / grid_s)) * (3840 // quantize))
        key = (tick, pitch)
        old = quantized.get(key)
        if old is None or velocity > old[1]:
            quantized[key] = (t, velocity)

    events = [
        NoteEvent(
            start_s=float(t),
            end_s=float(t) + max(0.08, grid_s * 0.5),
            pitch=int(pitch),
            amplitude=float(velocity / 120.0),
        )
        for (tick, pitch), (t, velocity) in sorted(quantized.items())
    ]
    _emit(progress_cb, 100, f"鼓点完成（{len(events)} 个）")
    if logger:
        logger.info("drums: %d GM hits from %d onsets", len(events), len(times))
    return events, warnings
