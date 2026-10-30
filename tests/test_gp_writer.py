from __future__ import annotations

from pathlib import Path

import guitarpro as gp

from audio2gp.config import (
    DRUM_LANE_BY_PITCH,
    TICKS_PER_BAR,
    TUNING_BASS,
    TUNING_GUITAR,
    VELOCITY_STEPS,
)
from audio2gp.gp_writer import _decompose, parse_gp5, verify_gp5, write_gp5
from audio2gp.types import ChordEvent, FingeredNote, Note, TabTrack


def _fnote(
    start: int,
    end: int,
    pitch: int,
    string: int,
    fret: int,
    velocity: int = 90,
) -> FingeredNote:
    return FingeredNote(start, end, pitch, velocity, string, fret)


def _guitar(notes: list[FingeredNote] | None = None) -> TabTrack:
    return TabTrack(
        name="Guitar",
        stem="other",
        tuning=[TUNING_GUITAR[i] for i in range(1, 7)],
        notes=notes or [],
        channel=0,
        instrument=27,
    )


def _bass(notes: list[FingeredNote] | None = None) -> TabTrack:
    return TabTrack(
        name="Bass",
        stem="bass",
        tuning=[TUNING_BASS[i] for i in range(1, 5)],
        notes=notes or [],
        channel=2,
        instrument=33,
    )


def _drums(notes: list[Note] | None = None) -> TabTrack:
    return TabTrack(
        name="Drums",
        stem="drums",
        tuning=[],
        is_percussion=True,
        channel=9,
        instrument=0,
        drum_notes=notes if notes is not None else [],
    )


def _assert_measure_sums(song: gp.Song) -> None:
    for track in song.tracks:
        for measure in track.measures:
            assert len(measure.voices) == 2
            assert sum(beat.duration.time for beat in measure.voices[0].beats) == TICKS_PER_BAR
            assert all(beat.status != gp.BeatStatus.empty for beat in measure.voices[0].beats)


def test_onset_layout_round_trip(tmp_path: Path) -> None:
    guitar = _guitar(
        [
            _fnote(0, 960, 64, 1, 0),
            _fnote(0, 960, 59, 2, 0),
            _fnote(960, 1440, 64, 1, 0),
        ]
    )
    bass = _bass([_fnote(0, 3840, 28, 4, 0)])
    drums = _drums(
        [
            Note(0, 240, 36, 100, "drums"),
            Note(0, 240, 38, 90, "drums"),
        ]
    )
    path = tmp_path / "onset.gp5"
    warnings = write_gp5(
        [guitar, bass, drums],
        bpm=120.0,
        measure_count=2,
        path=path,
        layout="onset",
        quantize=16,
    )
    assert warnings == []
    assert verify_gp5(path, expected_measures=2, expected_tracks=3) == []

    song = gp.parse(str(path))
    assert song.tempo == 120
    assert len(song.measureHeaders) == 2
    assert len(song.tracks) == 3
    _assert_measure_sums(song)
    assert [track.name for track in song.tracks] == ["Guitar", "Bass", "Drums"]

    guitar_track, bass_track, drum_track = song.tracks
    assert (guitar_track.channel.channel, guitar_track.channel.instrument) == (0, 27)
    assert (bass_track.channel.channel, bass_track.channel.instrument) == (2, 33)
    assert drum_track.isPercussionTrack
    assert drum_track.channel.channel == 9
    assert drum_track.channel.instrument == 0
    assert len(drum_track.strings) == 6
    assert all(string.value == 0 for string in drum_track.strings)

    first_measure = guitar_track.measures[0]
    assert len(first_measure.voices[1].beats) == 1
    assert first_measure.voices[1].beats[0].status == gp.BeatStatus.empty
    normal = [b for b in first_measure.voices[0].beats if b.status == gp.BeatStatus.normal]
    assert len(normal) == 2
    assert {note.string for note in normal[0].notes} == {1, 2}


def test_onset_chord_duration_and_rests(tmp_path: Path) -> None:
    guitar = _guitar([_fnote(960, 1440, 72, 1, 8)])
    path = tmp_path / "rests.gp5"
    write_gp5([guitar], bpm=100.0, measure_count=1, path=path, layout="onset")
    song = gp.parse(str(path))
    beats = song.tracks[0].measures[0].voices[0].beats
    assert sum(beat.duration.time for beat in beats) == TICKS_PER_BAR
    assert beats[0].status == gp.BeatStatus.rest
    assert beats[0].duration.time == 960
    assert beats[1].status == gp.BeatStatus.normal
    assert beats[1].duration.time == 2880
    assert beats[1].notes[0].value == 8


def test_onset_layout_splits_a_gap_that_is_not_a_single_note(tmp_path: Path) -> None:
    # Tick 0 to 1320 is legal on the 1/32 grid but 1320 is not one note value;
    # the old code wrote 1280 and left an unwritable 40-tick rest.
    guitar = _guitar(
        [
            _fnote(0, 480, 64, 1, 0),
            _fnote(1320, 1800, 67, 3, 3),
        ]
    )
    path = tmp_path / "gap.gp5"
    write_gp5(
        [guitar], bpm=120.0, measure_count=1, path=path, layout="onset", quantize=32
    )
    assert verify_gp5(path, expected_measures=1, expected_tracks=1) == []
    song = gp.parse(str(path))
    beats = song.tracks[0].measures[0].voices[0].beats
    assert sum(beat.duration.time for beat in beats) == TICKS_PER_BAR


def test_drum_hit_is_clamped_to_the_9_8_bar_line(tmp_path: Path) -> None:
    # 9/8 is 4320 ticks; on the 1/8-triplet grid (320) the last hit starts at
    # 4160 and must not spill 160 ticks past the bar line.
    path = tmp_path / "drums_9_8.gp5"
    write_gp5(
        [_drums([Note(4160, 4480, 36, 100, "drums")])],
        bpm=120.0,
        measure_count=1,
        path=path,
        layout="onset",
        quantize=12,
        time_signature="9/8",
    )
    assert (
        verify_gp5(
            path, expected_measures=1, expected_tracks=1, time_signature="9/8"
        )
        == []
    )
    song = gp.parse(str(path))
    beats = song.tracks[0].measures[0].voices[0].beats
    assert sum(beat.duration.time for beat in beats) == 4320


def test_empty_track_filled_with_eight_measures_of_rests(tmp_path: Path) -> None:
    path = tmp_path / "empty.gp5"
    write_gp5([_guitar()], bpm=90.0, measure_count=8, path=path, layout="ties")
    assert verify_gp5(path, expected_measures=8, expected_tracks=1) == []
    song = gp.parse(str(path))
    _assert_measure_sums(song)
    track = song.tracks[0]
    assert len(track.measures) == 8
    for measure in track.measures:
        assert measure.voices[0].beats
        assert all(beat.status == gp.BeatStatus.rest for beat in measure.voices[0].beats)
        assert all(not beat.notes for beat in measure.voices[0].beats)
        assert len(measure.voices[1].beats) == 1
        assert measure.voices[1].beats[0].status == gp.BeatStatus.empty


def test_in_bar_tie(tmp_path: Path) -> None:
    # The second attack splits the bar into chunks while string 1 sustains.
    guitar = _guitar(
        [
            _fnote(0, 2880, 64, 1, 0),
            _fnote(1920, 2880, 59, 2, 0),
        ]
    )
    path = tmp_path / "tie.gp5"
    write_gp5([guitar], bpm=120.0, measure_count=1, path=path, layout="ties")
    assert verify_gp5(path, expected_measures=1) == []
    beats = gp.parse(str(path)).tracks[0].measures[0].voices[0].beats
    assert [(beat.status, beat.duration.time) for beat in beats] == [
        (gp.BeatStatus.normal, 1920),
        (gp.BeatStatus.normal, 960),
        (gp.BeatStatus.rest, 960),
    ]
    assert beats[0].notes[0].type == gp.NoteType.normal
    first_by_string = {note.string: note for note in beats[0].notes}
    second_by_string = {note.string: note for note in beats[1].notes}
    assert first_by_string[1].type == gp.NoteType.normal
    assert second_by_string[1].type == gp.NoteType.tie
    assert second_by_string[1].value == first_by_string[1].value
    assert second_by_string[2].type == gp.NoteType.normal


def test_cross_bar_tie(tmp_path: Path) -> None:
    guitar = _guitar([_fnote(2880, 4800, 64, 1, 0)])
    path = tmp_path / "cross_tie.gp5"
    write_gp5([guitar], bpm=120.0, measure_count=2, path=path, layout="ties")
    assert verify_gp5(path, expected_measures=2) == []
    song = gp.parse(str(path))
    _assert_measure_sums(song)
    first = song.tracks[0].measures[0].voices[0].beats
    second = song.tracks[0].measures[1].voices[0].beats
    normal_first = [b for b in first if b.status == gp.BeatStatus.normal]
    assert len(normal_first) == 1
    assert normal_first[0].duration.time == 960
    assert normal_first[0].notes[0].type == gp.NoteType.normal
    assert second[0].status == gp.BeatStatus.normal
    assert second[0].duration.time == 960
    assert second[0].notes[0].type == gp.NoteType.tie
    assert all(b.status == gp.BeatStatus.rest for b in second[1:])


def test_mixed_sustain_and_string_change(tmp_path: Path) -> None:
    guitar = _guitar(
        [
            _fnote(0, 3840, 64, 1, 0),
            _fnote(0, 1920, 59, 2, 0),
            _fnote(1920, 3840, 61, 2, 2),
        ]
    )
    path = tmp_path / "mixed.gp5"
    write_gp5([guitar], bpm=120.0, measure_count=1, path=path, layout="ties")
    assert verify_gp5(path, expected_measures=1) == []
    beats = gp.parse(str(path)).tracks[0].measures[0].voices[0].beats
    assert [(b.status, b.duration.time) for b in beats] == [
        (gp.BeatStatus.normal, 1920),
        (gp.BeatStatus.normal, 1920),
    ]
    first_by_string = {note.string: note for note in beats[0].notes}
    second_by_string = {note.string: note for note in beats[1].notes}
    assert first_by_string[1].type == gp.NoteType.normal
    assert first_by_string[2].type == gp.NoteType.normal
    assert second_by_string[1].type == gp.NoteType.tie
    assert second_by_string[1].value == 0
    assert second_by_string[2].type == gp.NoteType.normal
    assert second_by_string[2].value == 2


def test_percussion_lanes_values_and_velocity_snapping(tmp_path: Path) -> None:
    drum_notes = [
        Note(0, 240, 36, 93, "drums"),
        Note(0, 240, 38, 87, "drums"),
        Note(0, 240, 42, 81, "drums"),
        Note(0, 240, 46, 75, "drums"),
        Note(0, 240, 49, 69, "drums"),
        Note(0, 240, 41, 63, "drums"),
    ]
    path = tmp_path / "drums.gp5"
    warnings = write_gp5(
        [_drums(drum_notes)],
        bpm=120.0,
        measure_count=1,
        path=path,
        layout="ties",
        quantize=16,
    )
    assert warnings == []
    assert verify_gp5(path, expected_measures=1) == []
    song = gp.parse(str(path))
    track = song.tracks[0]
    assert track.isPercussionTrack
    assert track.channel.channel == 9
    assert (track.clefTranspose, track.clefTransposeSecondary) == (0, 0)
    _assert_measure_sums(song)
    beats = track.measures[0].voices[0].beats
    normal = [b for b in beats if b.status == gp.BeatStatus.normal]
    assert len(normal) == 1
    notes = normal[0].notes
    assert {note.value for note in notes} == {36, 38, 42, 46, 49, 41}
    assert {note.string for note in notes} == {1, 2, 3, 4, 5, 6}
    for note in notes:
        assert note.string == DRUM_LANE_BY_PITCH[note.value]
        assert note.velocity in VELOCITY_STEPS
        assert note.type == gp.NoteType.normal
    assert normal[0].duration.time == 240


def test_empty_drum_track_warns_and_fills_rests(tmp_path: Path) -> None:
    path = tmp_path / "empty_drums.gp5"
    warnings = write_gp5(
        [_guitar(), _drums([])],
        bpm=120.0,
        measure_count=2,
        path=path,
        layout="onset",
    )
    assert any("empty drum track" in warning for warning in warnings)
    song = gp.parse(str(path))
    _assert_measure_sums(song)
    drum_track = song.tracks[1]
    for measure in drum_track.measures:
        assert all(beat.status == gp.BeatStatus.rest for beat in measure.voices[0].beats)


def test_32nd_grid_layout(tmp_path: Path) -> None:
    guitar = _guitar([_fnote(120, 240, 64, 1, 0)])
    path = tmp_path / "32nd.gp5"
    write_gp5(
        [guitar], bpm=120.0, measure_count=1, path=path, layout="ties", quantize=32
    )
    assert verify_gp5(path, expected_measures=1) == []
    song = gp.parse(str(path))
    beats = song.tracks[0].measures[0].voices[0].beats
    assert sum(beat.duration.time for beat in beats) == TICKS_PER_BAR
    duration_120 = [beat for beat in beats if beat.duration.time == 120]
    assert len(duration_120) >= 1
    assert any(
        note.type == gp.NoteType.normal and note.value == 0
        for beat in duration_120
        for note in beat.notes
    )


def test_duration_decomposition_for_all_grid_units() -> None:
    for quantize in (8, 12, 16, 32):
        grid = TICKS_PER_BAR // quantize
        for units in range(1, quantize + 1):
            length = units * grid
            parts = _decompose(length)
            assert sum(part[0] for part in parts) == length
            assert all(part[0] > 0 for part in parts)


def test_eighth_triplet_grid_round_trip(tmp_path: Path) -> None:
    # 12 units per bar = eighth-note triplets. Three triplet eighths fill a
    # quarter note and twelve fill the 4/4 bar.
    guitar = _guitar(
        [
            _fnote(0, 320, 64, 1, 0),
            _fnote(320, 640, 67, 1, 3),
            _fnote(640, 960, 71, 1, 7),
        ]
    )
    path = tmp_path / "triplet.gp5"
    write_gp5(
        [guitar],
        bpm=120.0,
        measure_count=1,
        path=path,
        layout="ties",
        quantize=12,
    )
    assert verify_gp5(path, expected_measures=1, expected_tracks=1) == []
    song = gp.parse(str(path))
    beats = song.tracks[0].measures[0].voices[0].beats
    assert sum(beat.duration.time for beat in beats) == TICKS_PER_BAR
    sounding = [beat for beat in beats if beat.status == gp.BeatStatus.normal]
    assert [beat.duration.time for beat in sounding[:3]] == [320, 320, 320]
    for beat in sounding[:3]:
        assert beat.duration.tuplet.enters == 3
        assert beat.duration.tuplet.times == 2
    assert [note.value for beat in sounding[:3] for note in beat.notes] == [0, 3, 7]


def test_duplicate_strings_do_not_produce_an_invalid_gp5(tmp_path: Path) -> None:
    guitar = _guitar(
        [
            _fnote(0, 960, 64, 1, 0),
            _fnote(0, 960, 65, 1, 1),
        ]
    )
    path = tmp_path / "duplicate_strings.gp5"
    # PyGuitarPro itself cannot retain two notes on one string; the writer
    # still has to produce a file that re-parses cleanly.
    write_gp5([guitar], bpm=120.0, measure_count=1, path=path, layout="onset")
    assert verify_gp5(path, expected_measures=1) == []
    beats = gp.parse(str(path)).tracks[0].measures[0].voices[0].beats
    seen = [note.string for beat in beats for note in beat.notes]
    assert len(seen) == len(set(seen))


def test_verify_detects_missing_measures(tmp_path: Path) -> None:
    guitar = _guitar([_fnote(0, 960, 64, 1, 0)])
    path = tmp_path / "short.gp5"
    write_gp5([guitar], bpm=120.0, measure_count=1, path=path, layout="onset")
    issues = verify_gp5(path, expected_measures=2, expected_tracks=1)
    assert issues


def test_chord_symbols_attach_to_the_first_melodic_track(tmp_path: Path) -> None:
    guitar = _guitar(
        [
            _fnote(0, 1920, 64, 1, 0),
            _fnote(1920, 3840, 67, 1, 3),
        ]
    )
    bass = _bass([_fnote(0, 3840, 28, 4, 0)])
    path = tmp_path / "chords.gp5"
    warnings = write_gp5(
        [guitar, bass],
        bpm=120.0,
        measure_count=1,
        path=path,
        layout="onset",
        chords=[
            ChordEvent(0, 1920, 0.0, 1.0, "C", 0, "major", 0.8),
            ChordEvent(1920, 3840, 1.0, 2.0, "Gm7", 7, "m7", 0.6),
        ],
    )
    assert warnings == []
    song = gp.parse(str(path))
    guitar_chords = [
        beat.effect.chord.name
        for beat in song.tracks[0].measures[0].voices[0].beats
        if beat.effect.chord is not None
    ]
    assert guitar_chords == ["C", "Gm7"]
    assert all(
        beat.effect.chord is None
        for measure in song.tracks[1].measures
        for beat in measure.voices[0].beats
    )


def test_out_of_range_chords_are_skipped_with_a_warning(tmp_path: Path) -> None:
    guitar = _guitar([_fnote(0, 3840, 64, 1, 0)])
    path = tmp_path / "late_chord.gp5"
    warnings = write_gp5(
        [guitar],
        bpm=120.0,
        measure_count=1,
        path=path,
        layout="onset",
        chords=[ChordEvent(3840, 7680, 2.0, 3.0, "C", 0, "major", 0.8)],
    )
    assert any("超出来谱面范围" in warning for warning in warnings)
    assert not any(
        beat.effect.chord is not None
        for measure in gp.parse(str(path)).tracks[0].measures
        for beat in measure.voices[0].beats
    )


def test_repeated_tie_patterns_reconstruct_the_right_pitch(tmp_path: Path) -> None:
    # PyGuitarPro 0.11 looks up tied-note values with Beat equality, so this
    # repeated pattern (a huge sustain plus a one-off note, as real audio often
    # produces) makes the raw parser return fret 0 for valid ties. parse_gp5
    # and verify_gp5 must recover the writer's intended values.
    song = gp.Song(tracks=[], measureHeaders=[])
    song.title = "Tie Regression"
    song.tempo = 120
    song.measureHeaders.append(
        gp.MeasureHeader(
            number=1,
            start=960,
            timeSignature=gp.TimeSignature(4, gp.Duration(4)),
        )
    )
    track = gp.Track(
        song,
        number=1,
        name="Guitar",
        strings=[gp.GuitarString(i, TUNING_GUITAR[i]) for i in range(1, 7)],
        channel=gp.MidiChannel(channel=0, effectChannel=1, instrument=27),
        clefTranspose=0,
        clefTransposeSecondary=0,
    )
    song.tracks.append(track)
    voice = track.measures[0].voices[0]
    voice.beats = []
    scheme = [
        [(1, 9, gp.NoteType.normal), (2, 3, gp.NoteType.normal), (4, 0, gp.NoteType.normal)],
        [(1, 9, gp.NoteType.tie), (2, 3, gp.NoteType.tie), (4, 0, gp.NoteType.tie)],
        [(1, 9, gp.NoteType.tie), (2, 3, gp.NoteType.tie), (4, 4, gp.NoteType.normal)],
        [(1, 9, gp.NoteType.tie), (2, 3, gp.NoteType.tie), (4, 4, gp.NoteType.tie)],
        [(1, 9, gp.NoteType.tie), (2, 3, gp.NoteType.tie), (4, 4, gp.NoteType.tie)],
    ]
    for index in range(16):
        if index < len(scheme):
            beat = gp.Beat(
                voice, status=gp.BeatStatus.normal, duration=gp.Duration(16)
            )
            beat.notes = [
                gp.Note(beat, value=value, velocity=90, string=string, type=note_type)
                for string, value, note_type in scheme[index]
            ]
        else:
            beat = gp.Beat(voice, status=gp.BeatStatus.rest, duration=gp.Duration(16))
            beat.notes = []
        voice.beats.append(beat)

    path = tmp_path / "repeated_ties.gp5"
    gp.write(song, str(path), version=(5, 1, 0), encoding="cp1252")

    raw_beats = gp.parse(str(path)).tracks[0].measures[0].voices[0].beats
    raw_values = {
        note.string: note.value for note in raw_beats[3].notes if note.string == 4
    }
    assert raw_values == {4: 0}

    fixed_beats = parse_gp5(path).tracks[0].measures[0].voices[0].beats
    for beat in fixed_beats[3:5]:
        note = next(note for note in beat.notes if note.string == 4)
        assert (note.type, note.value) == (gp.NoteType.tie, 4)

    assert verify_gp5(path, expected_measures=1, expected_tracks=1) == []
