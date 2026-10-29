from __future__ import annotations

from pathlib import Path

import mido
import pytest

from audio2gp.config import Options, grid_ticks, grid_seconds
from audio2gp.midi_post import clean, write_full_midi, write_stem_midi
from audio2gp.types import ChordEvent, Note, NoteEvent

BPM = 120.0


def _event(start: float, end: float, pitch: int, amplitude: float = 0.7) -> NoteEvent:
    return NoteEvent(start, end, pitch, amplitude)


def _notes_for_stem(stem: str, events: list[NoteEvent], quantize: int = 16) -> list[Note]:
    options = Options(quantize=quantize)
    return clean({stem: events}, options, BPM)[stem]


def test_clustering_preserves_chord_and_snaps_to_grid() -> None:
    # Three attacks within the 35 ms cluster window become one onset.
    events = [
        _event(0.501, 0.900, 60, 0.7),
        _event(0.509, 0.910, 64, 0.8),
        _event(0.512, 0.920, 67, 0.6),
    ]
    notes = _notes_for_stem("other", events)
    assert len(notes) == 3
    assert {note.pitch for note in notes} == {60, 64, 67}
    assert len({note.start_tick for note in notes}) == 1
    start = notes[0].start_tick
    assert start % grid_ticks(16) == 0


def test_attacks_outside_cluster_window_remain_separate() -> None:
    notes = _notes_for_stem(
        "other",
        [_event(0.0, 0.20, 60), _event(0.20, 0.45, 62)],
    )
    assert len(notes) == 2
    assert len({note.start_tick for note in notes}) == 2


@pytest.mark.parametrize("quantize", [8, 16, 32])
def test_quantization_grids_and_end_snapping(quantize: int) -> None:
    grid_t = grid_ticks(quantize)
    grid_s = grid_seconds(BPM, quantize)
    notes = _notes_for_stem(
        "other",
        [_event(0.11, 0.11 + 3.2 * grid_s, 60)],
        quantize=quantize,
    )
    assert len(notes) == 1
    note = notes[0]
    assert note.start_tick % grid_t == 0
    assert note.end_tick % grid_t == 0
    assert note.end_tick >= note.start_tick + grid_t
    assert note.start_tick == round(0.11 / grid_s) * grid_t


def test_min_duration_filters_melodic_but_not_drums() -> None:
    short = _event(0.0, 0.01, 60)
    options = Options(min_note_sec=0.03)
    melodic = clean({"other": [short]}, options, BPM)["other"]
    drums = clean({"drums": [short]}, options, BPM)["drums"]
    assert melodic == []
    assert len(drums) == 1
    assert drums[0].end_tick == drums[0].start_tick + grid_ticks(options.quantize)


def test_bass_keeps_lowest_and_vocals_keep_strongest() -> None:
    events = [_event(0.0, 0.5, 48, 0.7), _event(0.0, 0.5, 43, 0.5)]
    bass = _notes_for_stem("bass", events)
    assert [note.pitch for note in bass] == [43]

    events = [_event(0.0, 0.5, 60, 0.8), _event(0.0, 0.5, 64, 0.3)]
    vocals = _notes_for_stem("vocals", events)
    assert [note.pitch for note in vocals] == [60]


def test_duplicate_pitch_and_overlap_trim() -> None:
    duplicate = _notes_for_stem(
        "other",
        [_event(0.0, 0.5, 60, 0.7), _event(0.001, 0.6, 60, 0.9)],
    )
    assert len(duplicate) == 1
    assert duplicate[0].velocity >= 100

    overlap = _notes_for_stem(
        "other",
        [_event(0.0, 0.5, 60), _event(0.3, 0.8, 60)],
        quantize=16,
    )
    assert len(overlap) == 2
    first, second = sorted(overlap, key=lambda note: note.start_tick)
    assert first.end_tick == second.start_tick
    assert second.end_tick > second.start_tick


def _note(start: int, end: int, pitch: int, velocity: int = 90) -> Note:
    return Note(start, end, pitch, velocity, "other")


def test_stem_midi_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "guitar.mid"
    write_stem_midi(
        [_note(0, 480, 60), _note(480, 960, 64)],
        BPM,
        path,
        program=27,
        channel=3,
        name="Guitar",
    )
    mid = mido.MidiFile(str(path))
    assert mid.ticks_per_beat == 960
    meta = mid.tracks[0]
    assert any(msg.type == "set_tempo" and msg.tempo == mido.bpm2tempo(BPM) for msg in meta)
    assert any(
        msg.type == "time_signature" and (msg.numerator, msg.denominator) == (4, 4)
        for msg in meta
    )
    track = mid.tracks[1]
    assert any(msg.type == "track_name" and msg.name == "Guitar" for msg in track)
    assert any(
        msg.type == "program_change" and msg.program == 27 and msg.channel == 3
        for msg in track
    )
    on = [msg for msg in track if msg.type == "note_on" and msg.velocity > 0]
    off = [msg for msg in track if msg.type == "note_off"]
    assert len(on) == 2 and len(off) == 2
    assert all(msg.channel == 3 for msg in on + off)
    assert [msg.note for msg in on] == [60, 64]


def test_full_midi_channels_programs_and_names(tmp_path: Path) -> None:
    path = tmp_path / "full.mid"
    write_full_midi(
        {
            "vocals": [_note(0, 480, 72)],
            "drums": [_note(0, 120, 36)],
            "bass": [_note(0, 480, 40)],
            "other": [_note(0, 480, 60)],
        },
        BPM,
        path,
    )
    mid = mido.MidiFile(str(path))
    parsed: dict[str, tuple[int, int]] = {}
    for track in mid.tracks[1:]:
        name = next(msg.name for msg in track if msg.type == "track_name")
        program = next(
            (msg.channel, msg.program)
            for msg in track if msg.type == "program_change"
        )
        parsed[name] = program
    assert parsed == {
        "Vocals": (0, 53),
        "Drums": (9, 0),
        "Bass": (2, 33),
        "Guitar": (1, 27),
    }


def test_full_midi_can_limit_tracks(tmp_path: Path) -> None:
    path = tmp_path / "full_subset.mid"
    write_full_midi(
        {
            "vocals": [_note(0, 480, 72)],
            "other": [_note(0, 480, 60)],
        },
        BPM,
        path,
        stems=("vocals", "other"),
    )
    mid = mido.MidiFile(str(path))
    names = [
        next(msg.name for msg in track if msg.type == "track_name")
        for track in mid.tracks[1:]
    ]
    assert names == ["Vocals", "Guitar"]


def test_full_midi_writes_sorted_chord_markers(tmp_path: Path) -> None:
    path = tmp_path / "chords.mid"
    write_full_midi(
        {"other": [_note(0, 1920, 60), _note(1920, 3840, 67)]},
        BPM,
        path,
        stems=("other",),
        chords=[
            ChordEvent(1920, 3840, 1.0, 2.0, "G", 7, "major", 0.6),
            ChordEvent(0, 1920, 0.0, 1.0, "Cmaj7", 0, "maj7", 0.7),
        ],
    )
    track = mido.MidiFile(str(path)).tracks[0]
    tick = 0
    markers: list[tuple[int, str]] = []
    for message in track:
        tick += int(message.time)
        if message.type == "marker":
            markers.append((tick, message.text))
    assert markers == [(0, "Cmaj7"), (1920, "G")]


def test_midi_note_off_precedes_note_on_at_same_tick(tmp_path: Path) -> None:
    path = tmp_path / "retrigger.mid"
    write_stem_midi(
        [_note(0, 960, 60), _note(960, 1920, 60)],
        BPM,
        path,
        program=27,
        channel=0,
        name="Guitar",
    )
    track = mido.MidiFile(str(path)).tracks[1]
    # Gather delta-accumulated ticks for the same pitch around the boundary.
    tick = 0
    at_960: list[tuple[str, int]] = []
    for msg in track:
        tick += msg.time
        if tick == 960 and msg.type in {"note_on", "note_off"}:
            at_960.append((msg.type, msg.velocity))
    assert at_960 == [("note_off", 0), ("note_on", 90)]
