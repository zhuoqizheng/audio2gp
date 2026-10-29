"""Chord analysis from Demucs harmonic stems.

The detector is deliberately conservative: it works on beat-synchronous chroma,
suppresses the most common harmonic-overtones (a fifth and a major third above
each sounding note), and applies a Viterbi smoother so a single noisy beat does
not create a new chord.  It reports the chord root and a small quality set; it
does not try to infer inversions, slash chords, or sophisticated voicings.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .config import TICKS_PER_QUARTER, parse_time_signature
from .types import ChordEvent

# quality -> (semitone offsets from the root, template weights)
QUALITY_SPECS: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {
    "major": ((0, 4, 7), (1.0, 0.9, 0.8)),
    "minor": ((0, 3, 7), (1.0, 0.9, 0.8)),
    "7": ((0, 4, 7, 10), (1.0, 0.9, 0.8, 0.35)),
    "maj7": ((0, 4, 7, 11), (1.0, 0.9, 0.8, 0.35)),
    "m7": ((0, 3, 7, 10), (1.0, 0.9, 0.8, 0.35)),
    "sus4": ((0, 5, 7), (1.0, 0.9, 0.8)),
    "dim": ((0, 3, 6), (1.0, 0.85, 0.8)),
    "power": ((0, 7), (1.0, 0.9)),
}
QUALITY_SUFFIX = {
    "major": "",
    "minor": "m",
    "7": "7",
    "maj7": "maj7",
    "m7": "m7",
    "sus4": "sus4",
    "dim": "dim",
    "power": "5",
}
QUALITIES = tuple(QUALITY_SPECS)
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# Stems that carry harmony.  Vocals are down-weighted because a melody can
# imply extensions without the accompaniment actually playing them.
STEM_WEIGHTS = {"other": 1.0, "bass": 0.8, "vocals": 0.3}
SAMPLE_RATE = 22050
HOP_LENGTH = 1024
TRANSITION_PENALTY = 0.65
OVERTONE_FIFTH = 0.35
OVERTONE_THIRD = 0.30
OUTSIDE_PENALTY = 0.25
ROOT_BONUS = 0.15
EXTENSION_PENALTY = 0.10
SILENCE_RATIO = 0.04
# Major/minor keys whose conventional spelling uses flats instead of sharps.
FLAT_MAJOR_TONICS = {1, 3, 5, 8, 10}
FLAT_MINOR_TONICS = {3, 5, 8, 10}


def _states() -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]], np.ndarray]:
    rows: list[np.ndarray] = []
    names: list[str] = []
    tone_sets: list[tuple[int, ...]] = []
    roots: list[int] = []
    for root in range(12):
        for quality in QUALITIES:
            offsets, weights = QUALITY_SPECS[quality]
            row = np.zeros(12, dtype=float)
            tones = tuple((root + offset) % 12 for offset in offsets)
            for pitch, weight in zip(tones, weights):
                row[pitch] = weight
            rows.append(row)
            names.append(f"{root}:{quality}")
            tone_sets.append(tones)
            roots.append(root)
    return (
        np.asarray(rows, dtype=float),
        np.asarray(roots, dtype=int),
        tone_sets,
        np.asarray(names, dtype=object),
    )


_TEMPLATES, _ROOTS, _TONE_SETS, _STATE_KEYS = _states()
_STATE_LOOKUP = {key: index for index, key in enumerate(_STATE_KEYS.tolist())}


def _state_name(root: int, quality: str, flat: bool) -> str:
    names = FLAT_NAMES if flat else PITCH_NAMES
    return f"{names[int(root) % 12]}{QUALITY_SUFFIX[quality]}"


def _scale_pitch_classes(tonic: int, mode: str) -> set[int]:
    intervals = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    return {(int(tonic) + interval) % 12 for interval in intervals}


def _spelling(key_info: Any) -> tuple[bool, set[int] | None]:
    """Return (use_flats, diatonic_pitch_classes) for the detected key."""
    tonic = getattr(key_info, "tonic", None)
    mode = getattr(key_info, "mode", None)
    if tonic is None or mode not in {"major", "minor"}:
        return False, None
    tonic = int(tonic) % 12
    flat = (
        tonic in FLAT_MAJOR_TONICS
        if mode == "major"
        else tonic in FLAT_MINOR_TONICS
    )
    return flat, _scale_pitch_classes(tonic, mode)


def _load_chroma(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (frame-normalized chroma, raw frame energy) for one stem."""
    import librosa

    y, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if y.size < SAMPLE_RATE // 4:
        raise ValueError("audio is too short for chord analysis")
    harmonic = librosa.effects.harmonic(y, margin=3.0)
    if float(np.max(np.abs(harmonic))) < 1e-6:
        harmonic = y
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        bins_per_octave=12,
    )
    energy = np.linalg.norm(chroma, axis=0)
    norms = np.maximum(energy, 1e-9)
    return chroma / norms[None, :], energy


def _mixed_chroma(
    stems: Mapping[str, Path | str] | Iterable[Path | str],
    warnings: list[str],
    progress_cb: Callable[[float, str], None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Average frame-normalized chroma from the harmonic stems.

    Returns (chroma, frame times, mean frame energy).  The mean energy is on
    the same frame grid as ``chroma`` and is used for silence detection.
    """
    import librosa

    if isinstance(stems, Mapping):
        candidates = [(str(name), Path(path)) for name, path in stems.items()]
    else:
        candidates = [
            (f"source{index}", Path(path)) for index, path in enumerate(stems)
        ]
    sources: list[tuple[float, np.ndarray, np.ndarray]] = []
    total_weight = 0.0
    for index, (name, path) in enumerate(candidates):
        weight = STEM_WEIGHTS.get(name, 0.65)
        if not path.is_file():
            warnings.append(f"和弦分析跳过 {name}: 文件不存在")
            continue
        if progress_cb:
            progress_cb(
                100.0 * index / max(1, len(candidates)),
                f"读取 {name} 声部（和弦分析）",
            )
        try:
            loaded = _load_chroma(path)
        except Exception as exc:
            warnings.append(f"和弦分析跳过 {name}: {exc}")
            continue
        if loaded is None:
            continue
        chroma, energy = loaded
        if not np.all(np.isfinite(chroma)) or float(np.sum(energy)) <= 1e-9:
            warnings.append(f"和弦分析跳过 {name}: 没有音高内容")
            continue
        sources.append((weight, chroma, energy))
        total_weight += weight
    if not sources:
        return None
    frames = min(source[1].shape[1] for source in sources)
    mix = np.zeros((12, frames), dtype=float)
    energy = np.zeros(frames, dtype=float)
    for weight, chroma, frame_energy in sources:
        mix += weight * chroma[:, :frames]
        energy += weight * frame_energy[:frames]
    mix /= max(1e-9, total_weight)
    energy /= max(1e-9, total_weight)
    times = librosa.frames_to_time(
        np.arange(frames), sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    if progress_cb:
        progress_cb(100.0, "计算和弦特征")
    return mix, times, energy


def _segment_emissions(
    chroma: np.ndarray,
    times: np.ndarray,
    energy: np.ndarray,
    bpm: float,
    offset_s: float,
    duration_s: float,
    diatonic: set[int] | None,
) -> list[tuple[int, int, float, float, np.ndarray, float]]:
    """Build quarter-note emissions.

    Each item is (start_tick, end_tick, start_s, end_s, emission, confidence).
    """
    beat_s = 60.0 / float(bpm)
    if not math.isfinite(beat_s) or beat_s <= 0:
        return []
    prefix = np.concatenate(([0.0], np.cumsum(energy)))
    positive = energy[energy > 0]
    reference = float(np.percentile(positive, 75)) if positive.size else 0.0
    silence = max(1e-8, SILENCE_RATIO * reference)
    first_index = max(0, int(math.ceil((0.0 - float(offset_s)) / beat_s)))
    first_start = float(offset_s) + first_index * beat_s
    if first_start >= duration_s:
        return []
    count = int(math.ceil((duration_s - first_start) / beat_s)) + 1
    out: list[tuple[int, int, float, float, np.ndarray, float]] = []
    for step in range(count):
        index = first_index + step
        start_s = float(offset_s) + index * beat_s
        end_s = min(float(duration_s), start_s + beat_s)
        if end_s <= 0 or start_s >= duration_s:
            continue
        start_s = max(0.0, start_s)
        mask = (times >= start_s) & (times < end_s)
        if not np.any(mask):
            continue
        mean_energy = float(np.mean(energy[mask]))
        if mean_energy < silence:
            continue
        vector = np.mean(chroma[:, mask], axis=1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            continue
        vector = vector / norm

        # Remove the two strongest overtones of the sounding chord tones.  This
        # prevents a plain major triad from being reported as maj7 because the
        # third's third harmonic sits a major seventh above the root.
        direct = np.clip(
            vector
            - OVERTONE_FIFTH * np.roll(vector, 7)
            - OVERTONE_THIRD * np.roll(vector, 4),
            0.0,
            None,
        )
        salience = np.clip(direct - float(np.median(direct)), 0.0, None)
        salience_norm = float(np.linalg.norm(salience))
        if salience_norm <= 1e-9:
            continue
        salience = salience / salience_norm

        scores = _TEMPLATES @ salience
        outside = np.asarray(
            [1.0 - float(np.sum(salience[list(tones)])) for tones in _TONE_SETS]
        )
        scores += ROOT_BONUS * salience[_ROOTS]
        scores -= OUTSIDE_PENALTY * np.clip(outside, 0.0, None)
        notes_per_state = np.asarray([len(tones) for tones in _TONE_SETS])
        scores -= EXTENSION_PENALTY * np.maximum(0, notes_per_state - 3)
        if diatonic is not None:
            for state_index, tones in enumerate(_TONE_SETS):
                if _ROOTS[state_index] in diatonic:
                    scores[state_index] += 0.05
                if all(pitch in diatonic for pitch in tones):
                    scores[state_index] += 0.03
        ordered = np.partition(scores, -2)
        margin = float(ordered[-1] - ordered[-2])
        confidence = 1.0 / (1.0 + math.exp(-3.0 * max(-6.0, min(6.0, margin))))
        start_tick = max(0, index * TICKS_PER_QUARTER)
        end_tick = max(start_tick + TICKS_PER_QUARTER // 2, (index + 1) * TICKS_PER_QUARTER)
        out.append((start_tick, end_tick, start_s, end_s, scores, confidence))
    return out


def _viterbi(emissions: np.ndarray) -> np.ndarray:
    if emissions.shape[0] == 0:
        return np.empty(0, dtype=int)
    count, states = emissions.shape
    dp = emissions[0].copy()
    back = np.zeros((count, states), dtype=np.int32)
    for step in range(1, count):
        candidate = np.tile(dp[:, None] - TRANSITION_PENALTY, (1, states))
        candidate[np.arange(states), np.arange(states)] = dp
        back[step] = np.argmax(candidate, axis=0)
        dp = candidate[back[step], np.arange(states)] + emissions[step]
    path = [int(np.argmax(dp))]
    for step in range(count - 1, 0, -1):
        path.append(int(back[step, path[-1]]))
    path.reverse()
    return np.asarray(path, dtype=int)


def _decode_blocks(
    emissions: list[np.ndarray],
) -> list[int]:
    """Decode contiguous blocks; -1 marks a silence gap."""
    path = [-1] * len(emissions)
    start = 0
    while start < len(emissions):
        while start < len(emissions) and emissions[start] is None:  # type: ignore[comparison-overlap]
            start += 1
        end = start
        while end < len(emissions) and emissions[end] is not None:
            end += 1
        if end > start:
            block = np.stack(emissions[start:end])  # type: ignore[arg-type]
            decoded = _viterbi(block)
            for offset, state in enumerate(decoded):
                path[start + offset] = int(state)
        start = end
    return path


def analyze_chords(
    stems: Mapping[str, Path | str] | Iterable[Path | str],
    bpm: float,
    *,
    offset_s: float = 0.0,
    duration_s: float | None = None,
    key_info: Any = None,
    warnings: list[str] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[ChordEvent]:
    """Estimate chord segments from harmonic stems.

    ``offset_s`` is the audio time represented by tick 0, exactly as used by
    note quantization.  Segments are quarter-note aligned, but reported on the
    same integer tick grid as the score, so chord changes and bar lines agree.
    """
    warning_list = warnings if warnings is not None else []
    if not math.isfinite(float(bpm)) or float(bpm) <= 0:
        warning_list.append("和弦分析跳过：速度无效")
        return []
    mixed = _mixed_chroma(stems, warning_list, progress_cb)
    if mixed is None:
        warning_list.append("和弦分析跳过：没有可用的和声声部")
        return []
    chroma, times, energy = mixed
    if duration_s is None:
        duration_s = float(times[-1]) if times.size else 0.0
    duration_s = max(0.0, float(duration_s))
    if duration_s <= 0:
        warning_list.append("和弦分析跳过：音频时长无效")
        return []
    flat, diatonic = _spelling(key_info)
    segments = _segment_emissions(
        chroma,
        times,
        energy,
        float(bpm),
        float(offset_s),
        duration_s,
        diatonic,
    )
    if not segments:
        warning_list.append("和弦分析未找到有音高的段落")
        return []
    emissions = [segment[4] for segment in segments]
    path = _decode_blocks(emissions)
    events: list[ChordEvent] = []
    for (start_tick, end_tick, start_s, end_s, _scores, confidence), state in zip(
        segments, path
    ):
        if state < 0:
            continue
        root = int(_ROOTS[state])
        quality = str(_STATE_KEYS[state]).split(":", 1)[1]
        label = _state_name(root, quality, flat)
        if events and events[-1].label == label and events[-1].end_tick == start_tick:
            previous = events[-1]
            events[-1] = ChordEvent(
                start_tick=previous.start_tick,
                end_tick=end_tick,
                start_s=previous.start_s,
                end_s=end_s,
                label=label,
                root=root,
                quality=quality,
                confidence=round(
                    (previous.confidence + float(confidence)) / 2.0, 3
                ),
            )
        else:
            events.append(
                ChordEvent(
                    start_tick=start_tick,
                    end_tick=end_tick,
                    start_s=round(start_s, 3),
                    end_s=round(end_s, 3),
                    label=label,
                    root=root,
                    quality=quality,
                    confidence=round(float(confidence), 3),
                )
            )
    if not events:
        warning_list.append("和弦分析未识别出和弦")
    return events


def chords_to_dict(
    events: list[ChordEvent],
    *,
    bpm: float,
    offset_s: float,
    duration_s: float,
    key: dict | None = None,
    time_signature: str = "4/4",
) -> dict:
    """Serialize chord segments plus a compact summary for JSON/UI.

    ``bar``/``beat`` are 1-based. In compound meters (6/8, 9/8, 12/8) a
    musical beat is a dotted quarter, matching the pulse grouping, so 12/8
    reports four beats per bar rather than twelve. Ticks themselves are
    unaffected: they always count quarter notes (960 per quarter).
    """
    meter = parse_time_signature(time_signature)
    segments = []
    counts: dict[str, int] = {}
    labels: set[str] = set()
    progression: list[str] = []
    for event in events:
        start_tick = int(event.start_tick)
        segments.append(
            {
                "start_tick": start_tick,
                "end_tick": int(event.end_tick),
                "start_s": round(float(event.start_s), 3),
                "end_s": round(float(event.end_s), 3),
                "label": event.label,
                "root": int(event.root),
                "quality": event.quality,
                "confidence": round(float(event.confidence), 3),
                "bar": start_tick // meter.bar_ticks + 1,
                "beat": (start_tick % meter.bar_ticks) // meter.pulse_ticks + 1,
            }
        )
        counts[event.quality] = counts.get(event.quality, 0) + 1
        labels.add(event.label)
        if not progression or progression[-1] != event.label:
            progression.append(event.label)
    return {
        "version": 1,
        "method": "beat-synchronous chroma + Viterbi",
        "bpm": round(float(bpm), 3),
        "time_signature": meter.label,
        "bar_ticks": meter.bar_ticks,
        "offset_s": round(float(offset_s), 3),
        "duration_s": round(float(duration_s), 3),
        "key": dict(key or {}),
        "segments": segments,
        "summary": {
            "chord_count": len(segments),
            "unique_chords": sorted(labels),
            "quality_counts": counts,
            "progression": progression,
        },
    }


def chords_from_state(entries: Iterable[dict]) -> list[ChordEvent]:
    """Rebuild ChordEvents from the score IR's stored chord list."""
    events: list[ChordEvent] = []
    for entry in entries:
        try:
            events.append(
                ChordEvent(
                    start_tick=int(entry["start_tick"]),
                    end_tick=int(entry["end_tick"]),
                    start_s=float(entry.get("start_s", 0.0)),
                    end_s=float(entry.get("end_s", 0.0)),
                    label=str(entry["label"]),
                    root=int(entry["root"]),
                    quality=str(entry["quality"]),
                    confidence=float(entry.get("confidence", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid chord entry: {entry!r}") from exc
    return events
