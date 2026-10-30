"""Notation-agnostic note metrics for transcription evaluation.

The matcher is deliberately simple and dependency-free: greedy one-to-one
matching with an onset tolerance. "Strict" matches need the same pitch and an
onset within the tolerance; "pitch-only" ignores onsets. Octave errors are
unmatched notes that line up in time but are 12 or 24 semitones away. Leak is
an unmatched note in this stem that matches a note from another stem's truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .midifile import MidiNote, load_midi_notes

DEFAULT_TOLERANCE_S = 0.08


@dataclass(frozen=True)
class Note:
    start_s: float
    end_s: float
    pitch: int


def _as_notes(notes: Iterable[object], pitch_attr: str = "pitch") -> list[Note]:
    out: list[Note] = []
    for note in notes:
        out.append(
            Note(
                start_s=float(getattr(note, "start_s")),
                end_s=float(getattr(note, "end_s")),
                pitch=int(getattr(note, pitch_attr)),
            )
        )
    out.sort(key=lambda item: (item.start_s, item.pitch))
    return out


def _match(
    predicted: list[Note],
    truth: list[Note],
    tolerance_s: float | None,
    pitch_ok: Callable[[int, int], bool],
) -> tuple[int, set[int]]:
    used: set[int] = set()
    matched = 0
    for note in predicted:
        best_index = None
        best_distance = float("inf")
        for index, candidate in enumerate(truth):
            if index in used or not pitch_ok(note.pitch, candidate.pitch):
                continue
            distance = abs(note.start_s - candidate.start_s)
            if tolerance_s is not None and distance > tolerance_s:
                continue
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None:
            used.add(best_index)
            matched += 1
    return matched, used


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_notes(
    predicted: Iterable[object],
    truth: Iterable[object],
    *,
    other_truth: Iterable[Iterable[object]] | None = None,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> dict:
    pred = _as_notes(predicted)
    expected = _as_notes(truth)
    others = [_as_notes(items) for items in (other_truth or [])]

    matched, used = _match(
        pred, expected, tolerance_s, lambda a, b: a == b
    )
    pitch_only, _ = _match(pred, expected, None, lambda a, b: a == b)

    matched_indices = _predicted_matched_indices(pred, expected, tolerance_s)
    octave_errors = 0
    leaked = 0
    for index, note in enumerate(pred):
        if index in matched_indices:
            continue
        octave_ok = False
        for truth_index, candidate in enumerate(expected):
            if truth_index in used:
                continue
            delta = abs(note.pitch - candidate.pitch)
            if delta in {12, 24} and abs(note.start_s - candidate.start_s) <= tolerance_s:
                octave_ok = True
                break
        if octave_ok:
            octave_errors += 1
            continue
        for other in others:
            other_matched, _ = _match(
                [note], other, tolerance_s, lambda a, b: a == b
            )
            if other_matched:
                leaked += 1
                break

    unmatched = max(0, len(pred) - matched - octave_errors - leaked)
    precision = _ratio(matched, len(pred))
    recall = _ratio(matched, len(expected))
    pitch_precision = _ratio(pitch_only, len(pred))
    pitch_recall = _ratio(pitch_only, len(expected))
    denominator = matched + octave_errors
    return {
        "truth_notes": len(expected),
        "predicted_notes": len(pred),
        "matched": matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "pitch_only": {
            "matched": pitch_only,
            "precision": round(pitch_precision, 4),
            "recall": round(pitch_recall, 4),
            "f1": round(_f1(pitch_precision, pitch_recall), 4),
        },
        "octave_errors": octave_errors,
        "octave_error_rate": round(_ratio(octave_errors, denominator), 4),
        "leaked": leaked,
        "leak_rate": round(_ratio(leaked, len(pred)), 4),
        "spurious": unmatched,
    }


def _predicted_matched_indices(
    predicted: list[Note],
    truth: list[Note],
    tolerance_s: float,
) -> set[int]:
    """Indices of predicted notes consumed by the strict match.

    ``_match`` counts matches but does not return which predicted notes won;
    this helper replays the same greedy order so octave/leak accounting does
    not double-count a note that already matched exactly.
    """
    used_truth: set[int] = set()
    matched_pred: set[int] = set()
    for pred_index, note in enumerate(predicted):
        best_index = None
        best_distance = float("inf")
        for truth_index, candidate in enumerate(truth):
            if truth_index in used_truth or note.pitch != candidate.pitch:
                continue
            distance = abs(note.start_s - candidate.start_s)
            if distance > tolerance_s:
                continue
            if distance < best_distance:
                best_distance = distance
                best_index = truth_index
        if best_index is not None:
            used_truth.add(best_index)
            matched_pred.add(pred_index)
    return matched_pred


def evaluate_stems(
    predicted: dict[str, list],
    truth: dict[str, list],
    *,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> dict:
    stems = [name for name in sorted(set(predicted) & set(truth))]
    per_stem: dict[str, dict] = {}
    total_pred = total_truth = total_matched = 0
    for stem in stems:
        others = [notes for name, notes in truth.items() if name != stem]
        result = evaluate_notes(
            predicted.get(stem, []),
            truth.get(stem, []),
            other_truth=others,
            tolerance_s=tolerance_s,
        )
        result["stem"] = stem
        per_stem[stem] = result
        total_pred += result["predicted_notes"]
        total_truth += result["truth_notes"]
        total_matched += result["matched"]
    precision = _ratio(total_matched, total_pred)
    recall = _ratio(total_matched, total_truth)
    return {
        "tolerance_s": tolerance_s,
        "stems": per_stem,
        "micro": {
            "predicted_notes": total_pred,
            "truth_notes": total_truth,
            "matched": total_matched,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_f1(precision, recall), 4),
        },
    }


def load_midi_dir(directory: Path | str) -> dict[str, list[MidiNote]]:
    path = Path(directory)
    if (path / "midi").is_dir():
        path = path / "midi"
    return {
        file.stem: load_midi_notes(file)
        for file in sorted(path.glob("*.mid"))
        if file.stem not in {"full", "reference"}
    }


def evaluate_dirs(
    predicted_dir: Path | str,
    truth_dir: Path | str,
    *,
    stems: Iterable[str] | None = None,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> dict:
    predicted = load_midi_dir(predicted_dir)
    truth = load_midi_dir(truth_dir)
    if stems is not None:
        wanted = {str(name) for name in stems}
        predicted = {name: notes for name, notes in predicted.items() if name in wanted}
        truth = {name: notes for name, notes in truth.items() if name in wanted}
    return evaluate_stems(predicted, truth, tolerance_s=tolerance_s)
