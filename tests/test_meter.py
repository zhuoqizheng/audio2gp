from __future__ import annotations

from pathlib import Path

import mido
import pytest

from audio2gp import score_ir
from audio2gp.chords import chords_to_dict
from audio2gp.config import (
    TICKS_PER_QUARTER,
    TUNING_GUITAR,
    Meter,
    Options,
    grid_seconds,
    grid_ticks,
    parse_time_signature,
)
from audio2gp.gp_writer import parse_gp5, verify_gp5, write_gp5
from audio2gp.midi_post import write_stem_midi, write_tab_tracks_midi
from audio2gp.rhythm import choose_grid_offset
from audio2gp.types import ChordEvent, FingeredNote, Note, TabTrack

BPM = 120.0


@pytest.mark.parametrize(
    "signature, bar_ticks, pulses, pulse_ticks, compound",
    [
        ("4/4", 3840, 4, 960, False),
        ("3/4", 2880, 3, 960, False),
        ("2/4", 1920, 2, 960, False),
        ("6/8", 2880, 2, 1440, True),
        ("9/8", 4320, 3, 1440, True),
        ("12/8", 5760, 4, 1440, True),
    ],
)
def test_meter_geometry(
    signature: str,
    bar_ticks: int,
    pulses: int,
    pulse_ticks: int,
    compound: bool,
) -> None:
    meter = parse_time_signature(signature)
    assert meter.label == signature
    assert meter.bar_ticks == bar_ticks
    assert meter.pulses_per_bar == pulses
    assert meter.pulse_ticks == pulse_ticks
    assert meter.compound is compound
    assert meter.beams[0] == (3 if compound else 2)


@pytest.mark.parametrize("value", ["5/4", "4/3", "0/4", "4", "", "4.4.4", "17/4"])
def test_invalid_time_signatures_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        parse_time_signature(value)


def test_meter_accepts_instance_and_options_normalize() -> None:
    meter = Meter(12, 8)
    assert parse_time_signature(meter) is meter

    options = Options(time_signature="12/8", manual_downbeat_s=1.25)
    assert options.meter.bar_ticks == 5760
    assert options.manual_downbeat_s == 1.25

    normalized = Options(manual_downbeat_s=0)
    assert normalized.manual_downbeat_s is None
    assert normalized.meter.label == "4/4"


def test_quantize_grid_names_note_values_not_bar_fractions() -> None:
    # 1/8, 1/8 triplet, 1/16, 1/32 are independent of the bar length.
    assert grid_ticks(8) == 480
    assert grid_ticks(12) == 320
    assert grid_ticks(16) == 240
    assert grid_ticks(32) == 120
    assert grid_seconds(BPM, 16) == pytest.approx(0.125)


def _guitar(notes: list[FingeredNote]) -> TabTrack:
    return TabTrack(
        name="Guitar",
        stem="other",
        tuning=[TUNING_GUITAR[i] for i in range(1, 7)],
        notes=notes,
        channel=0,
        instrument=27,
    )


@pytest.mark.parametrize(
    "signature, bar_ticks", [("4/4", 3840), ("3/4", 2880), ("6/8", 2880), ("12/8", 5760)]
)
@pytest.mark.parametrize("layout", ["onset", "ties"])
def test_gp5_headers_and_measure_sums_follow_meter(
    tmp_path: Path, signature: str, bar_ticks: int, layout: str
) -> None:
    guitar = _guitar(
        [
            FingeredNote(0, bar_ticks // 2, 64, 90, 1, 0),
            FingeredNote(bar_ticks // 2, bar_ticks, 59, 90, 2, 0),
        ]
    )
    path = tmp_path / f"meter-{signature.replace('/', '_')}-{layout}.gp5"
    warnings = write_gp5(
        [guitar],
        BPM,
        2,
        path,
        layout=layout,
        quantize=16,
        time_signature=signature,
    )
    assert warnings == []
    assert verify_gp5(path, 2, 1, time_signature=signature) == []

    song = parse_gp5(path)
    assert [int(header.start) for header in song.measureHeaders] == [
        TICKS_PER_QUARTER,
        TICKS_PER_QUARTER + bar_ticks,
    ]
    for header in song.measureHeaders:
        assert (
            header.timeSignature.numerator,
            header.timeSignature.denominator.value,
        ) == tuple(int(part) for part in signature.split("/"))
    for measure in song.tracks[0].measures:
        assert sum(beat.duration.time for beat in measure.voices[0].beats) == bar_ticks


def test_verify_gp5_rejects_a_mismatched_time_signature(tmp_path: Path) -> None:
    path = tmp_path / "wrong.gp5"
    write_gp5([_guitar([])], BPM, 1, path, layout="onset", time_signature="3/4")
    issues = verify_gp5(path, 1, time_signature="4/4")
    assert any("time signature" in issue for issue in issues)


def test_midi_writers_store_the_selected_meter(tmp_path: Path) -> None:
    note = Note(0, 480, 60, 90, "other")
    stem_path = tmp_path / "stem.mid"
    write_stem_midi(
        [note],
        BPM,
        stem_path,
        program=27,
        channel=0,
        name="Guitar",
        time_signature="12/8",
    )
    stem_meta = mido.MidiFile(str(stem_path)).tracks[0]
    assert any(
        message.type == "time_signature"
        and (message.numerator, message.denominator) == (12, 8)
        for message in stem_meta
    )

    full_path = tmp_path / "full.mid"
    write_tab_tracks_midi([_guitar([FingeredNote(0, 480, 64, 90, 1, 0)])], BPM, full_path,
                          time_signature="6/8")
    full_meta = mido.MidiFile(str(full_path)).tracks[0]
    assert any(
        message.type == "time_signature"
        and (message.numerator, message.denominator) == (6, 8)
        for message in full_meta
    )


def test_chord_positions_use_meter_pulses() -> None:
    events = [
        ChordEvent(0, 1440, 0.0, 0.75, "C", 0, "major", 0.8),
        ChordEvent(4320, 5760, 2.25, 3.0, "G", 7, "major", 0.7),
    ]
    doc = chords_to_dict(
        events,
        bpm=BPM,
        offset_s=0.0,
        duration_s=3.0,
        time_signature="12/8",
    )
    assert doc["time_signature"] == "12/8"
    assert doc["bar_ticks"] == 5760
    first, second = doc["segments"]
    assert (first["bar"], first["beat"]) == (1, 1)
    # Tick 4320 is the fourth dotted-quarter beat of a 12/8 bar.
    assert (second["bar"], second["beat"]) == (1, 4)

    six_eight = chords_to_dict(
        events,
        bpm=BPM,
        offset_s=0.0,
        duration_s=3.0,
        time_signature="6/8",
    )
    # In 6/8 the same tick is bar 2, second pulse.
    second = six_eight["segments"][1]
    assert (second["bar"], second["beat"]) == (2, 2)


def test_choose_grid_offset_uses_compound_bar_length() -> None:
    # 12/8 at 120 BPM = 3.0 s per bar. A pickup longer than one bar pushes
    # the first downbeat into bar 3 (two leading bars).
    offset = choose_grid_offset(
        3.25,
        0.05,
        BPM,
        beats_per_bar=12,
        denominator=8,
    )
    assert offset == pytest.approx(-2.75)

    # 3/4 at 120 BPM = 1.5 s per bar.
    offset = choose_grid_offset(1.6, 0.0, BPM, beats_per_bar=3, denominator=4)
    assert offset == pytest.approx(-1.4)


def test_score_ir_round_trips_meter_and_commits_gp5(tmp_path: Path) -> None:
    tracks = [_guitar([FingeredNote(0, 480, 64, 90, 1, 0)])]
    state = score_ir.build_state(
        tracks,
        BPM,
        measure_count=2,
        job_id="meter-job",
        quantize=16,
        layout="auto",
        max_fret=15,
        time_signature="12/8",
    )
    assert state["time_signature"] == "12/8"
    assert state["bar_ticks"] == 5760
    assert state["beats_per_bar"] == 12
    score_ir.validate_state(state)

    # One bar of 12/8 is longer than one 4/4 bar, so a tick inside bar 2 must
    # remain valid; a tick beyond the last measure must not.
    state["tracks"][0]["notes"][0]["end_tick"] = 5760
    score_ir.validate_state(state)
    state["tracks"][0]["notes"][0]["end_tick"] = 11520 + 1
    with pytest.raises(score_ir.StateError, match="invalid tick range"):
        score_ir.validate_state(state)
    state["tracks"][0]["notes"][0]["end_tick"] = 5760

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    committed = score_ir.commit_state(state, out_dir)
    song = parse_gp5(committed["gp5"])
    header = song.measureHeaders[0]
    assert (header.timeSignature.numerator, header.timeSignature.denominator.value) == (
        12,
        8,
    )
    assert committed["state"]["time_signature"] == "12/8"
    assert committed["state"]["bar_ticks"] == 5760
