from __future__ import annotations

from collections import defaultdict

import pytest

from audio2gp.config import TUNING_BASS, TUNING_GUITAR
from audio2gp.fingering import (
    _trim_string_conflicts,
    arrange_bass,
    arrange_guitar,
    arrange_vocals,
)
from audio2gp.types import FingeredNote, Note


def _note(start: int, end: int, pitch: int, velocity: int = 90) -> Note:
    return Note(start, end, pitch, velocity, "other")


def _assert_legal(notes: list[FingeredNote], tuning: dict[int, int], max_fret: int) -> None:
    for note in notes:
        assert 1 <= note.string <= len(tuning)
        assert 0 <= note.fret <= max_fret
        assert note.pitch == tuning[note.string] + note.fret
        assert note.end_tick > note.start_tick
    by_onset: dict[int, list[FingeredNote]] = defaultdict(list)
    for note in notes:
        by_onset[note.start_tick].append(note)
    for onset_notes in by_onset.values():
        assert len({note.string for note in onset_notes}) == len(onset_notes)
        fretted = [note.fret for note in onset_notes if note.fret > 0]
        if fretted:
            assert max(fretted) - min(fretted) <= 5


def test_open_e_chord_uses_all_open_strings() -> None:
    notes = [_note(0, 960, pitch) for pitch in (40, 45, 50, 55, 59, 64)]
    result, warnings = arrange_guitar(notes)
    assert warnings == []
    _assert_legal(result, TUNING_GUITAR, 15)
    assert {(note.string, note.fret) for note in result} == {
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)
    }
    assert sorted(note.pitch for note in result) == [40, 45, 50, 55, 59, 64]


def test_single_note_prefers_open_string() -> None:
    result, warnings = arrange_guitar([_note(0, 960, 64)])
    assert warnings == []
    assert len(result) == 1
    assert (result[0].string, result[0].fret, result[0].pitch) == (1, 0, 64)


def test_out_of_range_note_folds_an_octave_with_warning() -> None:
    result, warnings = arrange_guitar([_note(0, 960, 30)])
    assert len(result) == 1
    # 30 is below low E2: fold to 42 (E2 + 2).
    assert (result[0].string, result[0].fret, result[0].pitch) == (6, 2, 42)
    assert any("octave-folded" in warning for warning in warnings)
    _assert_legal(result, TUNING_GUITAR, 15)


def test_more_notes_than_strings_is_trimmed_with_warning() -> None:
    notes = [_note(0, 960, pitch) for pitch in (40, 43, 47, 50, 52, 55, 59)]
    result, warnings = arrange_guitar(notes)
    assert len(result) <= 6
    assert any("dropped" in warning for warning in warnings)
    _assert_legal(result, TUNING_GUITAR, 15)


def test_chord_spans_never_exceed_limit() -> None:
    # Wide, awkward pairs are folded or trimmed until legal.
    for pitches in ((41, 75), (40, 76), (43, 71), (42, 69)):
        result, _warnings = arrange_guitar(
            [_note(0, 960, pitch) for pitch in pitches]
        )
        _assert_legal(result, TUNING_GUITAR, 15)
        assert result


def test_drop_when_folding_cannot_make_a_legal_chord() -> None:
    # Six strings already occupied plus one extra; at least one is dropped.
    notes = [_note(0, 960, pitch) for pitch in (40, 45, 50, 55, 59, 64, 67)]
    result, warnings = arrange_guitar(notes)
    assert len(result) <= 6
    assert any("dropped" in warning for warning in warnings)
    _assert_legal(result, TUNING_GUITAR, 15)


def test_fallback_drops_the_quietest_note_not_the_most_versatile() -> None:
    # 88 and 89 only fit on string 1 (folded), so the full chord is illegal.
    # The quiet 89 must go first; the strong 62 is playable and must stay.
    notes = [
        _note(0, 960, 88, 120),
        _note(0, 960, 62, 90),
        _note(0, 960, 89, 30),
    ]
    result, warnings = arrange_guitar(notes)
    pitches = {note.pitch for note in result}
    assert 76 in pitches  # the strong 88 survives (folded)
    assert 77 not in pitches  # the quiet 89 was dropped
    assert pitches & {50, 62, 74}  # the 62 note survives, possibly folded
    assert any("dropped the weakest note" in warning for warning in warnings)
    _assert_legal(result, TUNING_GUITAR, 15)


def test_sequence_keeps_a_narrow_hand_position() -> None:
    notes = [_note(i * 480, (i + 1) * 480, pitch) for i, pitch in enumerate(range(60, 68))]
    result, warnings = arrange_guitar(notes, max_fret=15)
    assert warnings == []
    _assert_legal(result, TUNING_GUITAR, 15)
    frets = [note.fret for note in result]
    assert max(frets) - min(frets) <= 5
    assert len(result) == len(notes)


def test_bass_uses_bass_tuning_and_lowest_note() -> None:
    result, warnings = arrange_bass([_note(0, 960, 28)])
    assert warnings == []
    assert len(result) == 1
    assert (result[0].string, result[0].fret, result[0].pitch) == (4, 0, 28)
    _assert_legal(result, TUNING_BASS, 15)


def test_vocals_are_monophonic_and_keep_strongest_note() -> None:
    result, warnings = arrange_vocals(
        [_note(0, 480, 60, 80), _note(0, 480, 64, 40)], max_fret=24
    )
    assert warnings == []
    assert len(result) == 1
    assert result[0].pitch == 60
    _assert_legal(result, TUNING_GUITAR, 24)


def test_string_conflict_trimming() -> None:
    notes = [
        FingeredNote(0, 960, 60, 90, 1, 0),
        FingeredNote(480, 1440, 64, 90, 1, 0),
    ]
    warnings: list[str] = []
    trimmed = _trim_string_conflicts(notes, warnings)
    assert len(trimmed) == 2
    assert trimmed[0].end_tick == 480
    assert any("overlapping" in warning for warning in warnings)


def test_max_fret_is_respected_or_note_skipped() -> None:
    result, warnings = arrange_guitar([_note(0, 960, 64)], max_fret=0)
    assert result and all(note.fret == 0 for note in result)
    # A note that cannot be played at fret 0 may fold/skip; never emit an
    # illegal position.
    result, _warnings = arrange_guitar([_note(0, 960, 41)], max_fret=0)
    for note in result:
        assert 0 <= note.fret <= 0
        assert note.pitch == TUNING_GUITAR[note.string]
