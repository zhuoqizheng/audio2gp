from __future__ import annotations

import json
import zipfile
from pathlib import Path

import guitarpro as gp
import mido
import pytest

from audio2gp import score_ir
from audio2gp.config import TUNING_BASS, TUNING_GUITAR
from audio2gp.types import ChordEvent, FingeredNote, Note, TabTrack
from audio2gp.utils import write_json


def _tracks() -> list[TabTrack]:
    guitar = TabTrack(
        name="Guitar",
        stem="other",
        tuning=[TUNING_GUITAR[i] for i in range(1, 7)],
        notes=[
            FingeredNote(0, 960, 64, 90, 1, 0),
            FingeredNote(0, 960, 59, 90, 2, 0),
            FingeredNote(960, 1920, 67, 90, 1, 3),
        ],
        channel=0,
        instrument=27,
    )
    bass = TabTrack(
        name="Bass",
        stem="bass",
        tuning=[TUNING_BASS[i] for i in range(1, 5)],
        notes=[FingeredNote(0, 960, 28, 90, 4, 0)],
        channel=2,
        instrument=33,
    )
    drums = TabTrack(
        name="Drums",
        stem="drums",
        tuning=[],
        is_percussion=True,
        channel=9,
        instrument=0,
        drum_notes=[Note(0, 240, 36, 100, "drums")],
    )
    return [guitar, bass, drums]


def _state() -> dict:
    return score_ir.build_state(
        _tracks(),
        bpm=120.0,
        measure_count=1,
        job_id="job-1",
        quantize=16,
        layout="auto",
        max_fret=15,
        key={"name": "C major", "gp_key": "CMajor"},
    )


def _midi_pitches(path: Path) -> set[int]:
    mid = mido.MidiFile(str(path))
    return {
        message.note
        for track in mid.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    }


def test_build_state_has_stable_ids_and_round_trips(tmp_path: Path) -> None:
    state = _state()
    score_ir.validate_state(state)
    assert state["format"] == score_ir.FORMAT
    assert state["revision"] == 0
    assert [track["id"] for track in state["tracks"]] == ["guitar", "bass", "drums"]
    assert [note["id"] for note in state["tracks"][0]["notes"]] == [
        "n000001",
        "n000002",
        "n000003",
    ]
    assert state["tracks"][2]["drum_notes"][0]["id"] == "n000005"

    path = tmp_path / "tracks.json"
    write_json(path, state)
    loaded = score_ir.load_state(path)
    assert loaded == state

    tracks = score_ir.ir_to_tracks(loaded)
    assert [track.name for track in tracks] == ["Guitar", "Bass", "Drums"]
    assert [note.pitch for note in tracks[0].notes] == [64, 59, 67]
    assert tracks[2].is_percussion
    assert tracks[2].drum_notes is not None
    assert tracks[2].drum_notes[0].pitch == 36


def test_move_string_keeps_pitch_and_derives_fret() -> None:
    state = _state()
    result = score_ir.update_note_position(
        state, "guitar", "n000003", string=2
    )
    assert result["changed"] == 1
    note = score_ir.get_note(score_ir.get_track(state, "guitar"), "n000003")
    assert (note["string"], note["fret"], note["pitch"]) == (2, 8, 67)

    # Changing only the fret keeps the string and moves the pitch.
    score_ir.update_note_position(state, "guitar", "n000003", fret=10)
    note = score_ir.get_note(score_ir.get_track(state, "guitar"), "n000003")
    assert (note["string"], note["fret"], note["pitch"]) == (2, 10, 69)

    # No-op edit reports zero changes but remains valid.
    result = score_ir.update_note_position(
        state, "guitar", "n000003", string=2, fret=10, pitch=69
    )
    assert result["changed"] == 0


def test_move_rejects_impossible_and_colliding_positions() -> None:
    state = _state()
    with pytest.raises(score_ir.StateError, match="fret must be between"):
        score_ir.update_note_position(
            state, "guitar", "n000001", string=6
        )
    with pytest.raises(score_ir.StateError, match="does not match"):
        score_ir.update_note_position(
            state, "guitar", "n000001", fret=3, pitch=64
        )
    # n000002 already occupies string 1 at tick 0.
    with pytest.raises(score_ir.StateError, match="already has a note"):
        score_ir.update_note_position(
            state, "guitar", "n000002", string=1, fret=0
        )
    with pytest.raises(score_ir.StateError, match="unknown note"):
        score_ir.update_note_position(
            state, "guitar", "missing", fret=1
        )
    with pytest.raises(score_ir.StateError, match="drum tracks"):
        score_ir.update_note_position(
            state, "drums", "n000005", string=1, fret=0
        )


def test_bulk_transpose_refingers_and_reports_drops() -> None:
    state = _state()
    result = score_ir.transpose_track(state, "guitar", 1)
    assert result["changed"] == 3
    assert result["dropped"] == 0
    track = score_ir.get_track(state, "guitar")
    assert [note["pitch"] for note in track["notes"]] == [65, 60, 68]
    for note in track["notes"]:
        assert note["pitch"] == track["tuning"][note["string"] - 1] + note["fret"]
        assert 0 <= note["fret"] <= state["max_fret"]

    with pytest.raises(score_ir.StateError, match="drum tracks"):
        score_ir.transpose_track(state, "drums", 1)
    with pytest.raises(score_ir.StateError, match="not be zero"):
        score_ir.transpose_track(state, "guitar", 0)
    with pytest.raises(score_ir.StateError, match="between -24 and 24"):
        score_ir.transpose_track(state, "guitar", 25)
    with pytest.raises(score_ir.StateError, match="no playable notes"):
        score_ir.transpose_track(_state(), "guitar", 24)


def test_undo_restores_the_previous_committed_state(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = _state()
    write_json(
        out_dir / "meta.json",
        {"job_id": "job-1", "status": "ok", "files": {"midi": {}, "meta": "meta.json"}},
    )
    write_json(out_dir / "tracks.json", state)
    score_ir.update_note_position(state, "guitar", "n000001", fret=2)
    first = score_ir.commit_state(state, out_dir)
    assert first["state"]["revision"] == 1
    assert (out_dir / score_ir.PREVIOUS_STATE).is_file()

    undone = score_ir.undo_last_edit(first["state"], out_dir)
    assert undone["state"]["revision"] == 2
    note = score_ir.get_note(score_ir.get_track(undone["state"], "guitar"), "n000001")
    assert (note["string"], note["fret"], note["pitch"]) == (1, 0, 64)
    assert not (out_dir / score_ir.PREVIOUS_STATE).exists()
    with pytest.raises(score_ir.StateError, match="没有可撤销"):
        score_ir.undo_last_edit(undone["state"], out_dir)


def test_flags_are_metadata_and_ignore_stale_notes(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = _state()
    first = score_ir.set_note_flag(
        state, out_dir, "guitar", "n000003", flagged=True, reason="sounds wrong"
    )
    assert first["flagged"] is True
    assert first["flags"][0]["reason"] == "sounds wrong"
    assert first["flags"][0]["bar"] == 1
    assert state["revision"] == 0  # flags are not score edits

    second = score_ir.set_note_flag(
        state, out_dir, "bass", "n000004", flagged=True, reason=""
    )
    assert {
        flag["note_id"]: flag["reason"] for flag in second["flags"]
    } == {"n000003": "sounds wrong", "n000004": "需要人工检查"}

    off = score_ir.set_note_flag(state, out_dir, "guitar", "n000003", flagged=False)
    assert [flag["note_id"] for flag in off["flags"]] == ["n000004"]

    score_ir.delete_note(state, "bass", "n000004")
    assert score_ir.load_flags(out_dir, state) == []
    with pytest.raises(score_ir.StateError, match="unknown note"):
        score_ir.set_note_flag(state, out_dir, "guitar", "missing", flagged=True)


def test_delete_note_and_mute_track() -> None:
    state = _state()
    deleted = score_ir.delete_note(state, "guitar", "n000001")
    assert deleted["deleted"] == 1
    assert len(score_ir.get_track(state, "guitar")["notes"]) == 2
    with pytest.raises(score_ir.StateError, match="unknown note"):
        score_ir.delete_note(state, "guitar", "n000001")

    muted = score_ir.set_track_muted(state, "bass", True)
    assert muted == {"track": "bass", "muted": True}
    assert score_ir.get_track(state, "bass")["muted"] is True
    with pytest.raises(score_ir.StateError, match="unknown track"):
        score_ir.set_track_muted(state, "piano", True)


def test_commit_regenerates_gp5_midi_meta_and_zip(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = _state()
    write_json(
        out_dir / "meta.json",
        {
            "job_id": "job-1",
            "status": "ok",
            "files": {
                "midi": {
                    "other": "midi/other.mid",
                    "bass": "midi/bass.mid",
                    "drums": "midi/drums.mid",
                },
                "full_midi": "midi/full.mid",
                "meta": "meta.json",
            },
        },
    )
    score_ir.update_note_position(state, "guitar", "n000001", fret=2)
    result = score_ir.commit_state(state, out_dir)

    new_state = result["state"]
    assert new_state["revision"] == 1
    assert state["revision"] == 0
    assert (out_dir / "tracks.json").is_file()
    assert json.loads((out_dir / "tracks.json").read_text())["revision"] == 1
    assert result["gp5"].is_file()
    assert result["midi"]["other"].is_file()
    assert result["midi"]["bass"].is_file()
    assert result["midi"]["drums"].is_file()
    assert result["full_midi"].is_file()

    song = gp.parse(str(result["gp5"]))
    assert len(song.measureHeaders) == 1
    assert [track.name for track in song.tracks] == ["Guitar", "Bass", "Drums"]
    assert _midi_pitches(result["midi"]["other"]) == {66, 59, 67}

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["revision"] == 1
    assert meta["note_counts"] == {"other": 3, "bass": 1, "drums": 1}
    assert meta["files"]["tracks"] == "tracks.json"
    assert meta["files"]["gp5"] == "song.gp5"

    with zipfile.ZipFile(result["zip"]) as zf:
        names = set(zf.namelist())
    assert {
        "tracks.json",
        "song.gp5",
        "midi/other.mid",
        "midi/bass.mid",
        "midi/drums.mid",
        "midi/full.mid",
        "meta.json",
    } <= names

    # A second commit publishes the next revision and reflects deleted notes.
    score_ir.delete_note(new_state, "guitar", "n000003")
    second = score_ir.commit_state(new_state, out_dir)
    assert second["state"]["revision"] == 2
    assert _midi_pitches(second["midi"]["other"]) == {66, 59}
    assert not list(out_dir.glob(".edit-*"))


def _chords() -> list[ChordEvent]:
    return [
        ChordEvent(0, 1920, 0.0, 1.0, "C", 0, "major", 0.8),
        ChordEvent(1920, 3840, 1.0, 2.0, "Am", 9, "minor", 0.6),
    ]


def test_chords_persist_validate_and_regenerate_annotations(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = score_ir.build_state(
        _tracks(),
        bpm=120.0,
        measure_count=1,
        job_id="job-chords",
        quantize=16,
        layout="auto",
        max_fret=15,
        key={"name": "C major", "gp_key": "CMajor"},
        chords=_chords(),
    )
    score_ir.validate_state(state)
    assert [chord.label for chord in score_ir.ir_to_chords(state)] == ["C", "Am"]
    assert state["chords"][0] == {
        "start_tick": 0,
        "end_tick": 1920,
        "start_s": 0.0,
        "end_s": 1.0,
        "label": "C",
        "root": 0,
        "quality": "major",
        "confidence": 0.8,
    }

    write_json(
        out_dir / "meta.json",
        {
            "job_id": "job-chords",
            "status": "ok",
            "files": {
                "midi": {"other": "midi/other.mid"},
                "full_midi": "midi/full.mid",
            },
        },
    )
    write_json(
        out_dir / "chords.json",
        {"version": 1, "segments": [], "summary": {}},
    )
    result = score_ir.commit_state(state, out_dir)
    new_state = result["state"]
    assert [chord["label"] for chord in new_state["chords"]] == ["C", "Am"]

    song = gp.parse(str(result["gp5"]))
    assert [
        beat.effect.chord.name
        for measure in song.tracks[0].measures
        for beat in measure.voices[0].beats
        if beat.effect.chord is not None
    ] == ["C", "Am"]
    markers: list[tuple[int, str]] = []
    tick = 0
    for message in mido.MidiFile(str(result["full_midi"])).tracks[0]:
        tick += int(message.time)
        if message.type == "marker":
            markers.append((tick, message.text))
    assert markers == [(0, "C"), (1920, "Am")]
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["files"]["chords"] == "chords.json"

    # Editor moves regenerate GP5 and MIDI but keep the chord progression.
    score_ir.update_note_position(new_state, "guitar", "n000001", fret=2)
    second = score_ir.commit_state(new_state, out_dir)
    assert [chord["label"] for chord in second["state"]["chords"]] == ["C", "Am"]
    markers = []
    tick = 0
    for message in mido.MidiFile(str(second["full_midi"])).tracks[0]:
        tick += int(message.time)
        if message.type == "marker":
            markers.append((tick, message.text))
    assert markers == [(0, "C"), (1920, "Am")]

    broken = json.loads(json.dumps(second["state"]))
    broken["chords"][1]["start_tick"] = 0
    with pytest.raises(score_ir.StateError, match="unsorted or overlapping"):
        score_ir.validate_state(broken)
    broken = json.loads(json.dumps(second["state"]))
    broken["chords"][0]["quality"] = "quartal"
    with pytest.raises(score_ir.StateError, match="unknown chord quality"):
        score_ir.validate_state(broken)
