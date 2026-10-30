from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from audio2gp.evaluation import evaluate_dirs, evaluate_notes, evaluate_stems
from audio2gp.midifile import (
    first_tempo_bpm,
    first_time_signature,
    load_midi_notes,
    midi_duration,
    tempo_change_count,
)


@dataclass
class _N:
    start_s: float
    end_s: float
    pitch: int


def _n(start: float, pitch: int, length: float = 0.4) -> _N:
    return _N(start, start + length, pitch)


def test_strict_match_is_one_to_one_and_uses_tolerance() -> None:
    truth = [_n(0.0, 60), _n(0.5, 64)]
    predicted = [_n(0.03, 60), _n(0.54, 64), _n(1.0, 67)]
    result = evaluate_notes(predicted, truth, tolerance_s=0.08)
    assert result["matched"] == 2
    assert result["precision"] == round(2 / 3, 4)
    assert result["recall"] == 1.0
    assert result["spurious"] == 1

    outside = evaluate_notes([_n(0.2, 60)], [_n(0.0, 60)], tolerance_s=0.08)
    assert outside["matched"] == 0
    assert outside["pitch_only"]["matched"] == 1
    assert outside["pitch_only"]["f1"] == 1.0


def test_octave_errors_and_leak_are_counted() -> None:
    truth = [_n(0.0, 60), _n(1.0, 64)]
    predicted = [_n(0.0, 72), _n(1.0, 64)]
    result = evaluate_notes(predicted, truth)
    assert result["matched"] == 1
    assert result["octave_errors"] == 1
    assert result["octave_error_rate"] == 0.5
    assert result["leaked"] == 0

    leak = evaluate_notes(
        [_n(0.0, 40)],
        [],
        other_truth=[[_n(0.0, 40)]],
    )
    assert leak["leaked"] == 1
    assert leak["leak_rate"] == 1.0
    # A note matching its own stem is not a leak, even if another stem agrees.
    own = evaluate_notes(
        [_n(0.0, 40)],
        [_n(0.0, 40)],
        other_truth=[[_n(0.0, 40)]],
    )
    assert own["matched"] == 1
    assert own["leaked"] == 0


def test_evaluate_stems_uses_micro_totals() -> None:
    report = evaluate_stems(
        {"other": [_n(0.0, 64)], "bass": [_n(0.0, 40)]},
        {"other": [_n(0.0, 64), _n(0.5, 67)], "bass": [_n(0.0, 40)]},
    )
    assert set(report["stems"]) == {"other", "bass"}
    assert report["micro"]["matched"] == 2
    assert report["micro"]["truth_notes"] == 3
    assert report["micro"]["f1"] == round(2 * (2 / 2) * (2 / 3) / (2 / 2 + 2 / 3), 4)


def _write_midi(path: Path, tempo: int = 500000) -> None:
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    meta.append(
        mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0)
    )
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)
    notes = mido.MidiTrack()
    notes.append(mido.Message("program_change", program=25, channel=0, time=0))
    notes.append(mido.Message("note_on", note=60, velocity=90, channel=0, time=0))
    notes.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
    notes.append(mido.Message("note_on", note=64, velocity=80, channel=0, time=0))
    notes.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=480))
    notes.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(notes)
    mid.save(str(path))


def test_midi_reader_honors_tempo_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "truth.mid"
    _write_midi(path)
    notes = load_midi_notes(path)
    assert [(round(n.start_s, 3), n.pitch, n.program) for n in notes] == [
        (0.0, 60, 25),
        (0.5, 64, 25),
    ]
    assert round(midi_duration(path), 3) == 1.0
    assert first_tempo_bpm(path) == 120.0
    assert first_time_signature(path) == (3, 4)
    assert tempo_change_count(path) == 1
    assert notes[0].name == ""


def test_evaluate_dirs_detects_job_output_midi_folder(tmp_path: Path) -> None:
    reference = tmp_path / "golden"
    reference.mkdir()
    _write_midi(reference / "other.mid")
    job = tmp_path / "job"
    (job / "midi").mkdir(parents=True)
    _write_midi(job / "midi" / "other.mid")
    report = evaluate_dirs(job, reference)
    assert report["micro"]["f1"] == 1.0
    assert report["stems"]["other"]["matched"] == 2
