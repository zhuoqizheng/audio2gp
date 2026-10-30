from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable, Iterator, Literal

import guitarpro as gp

from .config import (
    DRUM_LANE_BY_PITCH,
    DURATION_SPECS,
    PROGRAMS,
    VELOCITY_STEPS,
    Meter,
    grid_ticks,
    parse_time_signature,
)
from .types import ChordEvent, FingeredNote, Note, TabTrack
from .utils import ensure_dir, safe_ascii

Layout = Literal["auto", "onset", "ties"]

GP_CHORD_TYPES = {
    "major": gp.ChordType.major,
    "minor": gp.ChordType.minor,
    "7": gp.ChordType.seventh,
    "maj7": gp.ChordType.majorSeventh,
    "m7": gp.ChordType.minorSeventh,
    "sus4": gp.ChordType.suspendedFourth,
    "dim": gp.ChordType.diminished,
    "power": gp.ChordType.power,
}


def _gp_chord(chord: ChordEvent, string_count: int) -> gp.Chord:
    """Build a display-only GP5 chord symbol (no fingering diagram)."""
    match = re.match(r"^[A-G]([#b])?", str(chord.label))
    flat = bool(match and match.group(1) == "b")
    intonation = "flat" if flat else "sharp"
    root = gp.PitchClass(int(chord.root) % 12, intonation=intonation)
    return gp.Chord(
        length=max(1, int(string_count)),
        newFormat=True,
        sharp=not flat,
        root=root,
        type=GP_CHORD_TYPES.get(str(chord.quality), gp.ChordType.major),
        extension=gp.ChordExtension.none,
        bass=root,
        tonality=gp.ChordAlteration.perfect,
        add=False,
        name=safe_ascii(str(chord.label), fallback="chord")[:32],
        fifth=gp.ChordAlteration.perfect,
        ninth=gp.ChordAlteration.perfect,
        eleventh=gp.ChordAlteration.perfect,
        firstFret=0,
        strings=[-1] * max(1, int(string_count)),
        omissions=[True] * 7,
        show=True,
    )


def _annotate_chords(song: gp.Song, chords: Iterable[ChordEvent] | None) -> list[str]:
    """Attach chord symbols to the first melodic track, at beat starts."""
    chord_list = sorted(chords or [], key=lambda item: item.start_tick)
    if not chord_list:
        return []
    target = next((track for track in song.tracks if not track.isPercussionTrack), None)
    if target is None:
        return ["和弦未写入 GP5：没有可标注的弦乐轨"]
    beats: list[tuple[int, int, gp.Beat]] = []
    tick = 0
    for measure in target.measures:
        for beat in measure.voices[0].beats:
            end = tick + int(beat.duration.time)
            beats.append((tick, end, beat))
            tick = end
    if not beats:
        return ["和弦未写入 GP5：目标轨没有可标注的拍"]
    annotated = 0
    collisions = 0
    outside = 0
    for chord in chord_list:
        start = max(0, int(chord.start_tick))
        chosen: gp.Beat | None = None
        for beat_start, beat_end, beat in beats:
            if beat_start <= start < beat_end:
                chosen = beat
                break
        if chosen is None:
            outside += 1
            continue
        if getattr(chosen.effect, "chord", None) is not None:
            collisions += 1
            continue
        chosen.effect.chord = _gp_chord(chord, len(target.strings))
        annotated += 1
    warnings: list[str] = []
    if collisions:
        warnings.append(
            f"GP5 和弦标注：{collisions} 个和弦落在已有和弦符号的同一拍，已保留第一个"
        )
    if outside:
        warnings.append(f"GP5 和弦标注：{outside} 个和弦起点超出来谱面范围，已跳过")
    if annotated == 0:
        warnings.append("GP5 和弦标注：没有和弦被写入")
    return warnings


class GP5ValidationError(RuntimeError):
    pass


def _snap_velocity(velocity: int | float) -> int:
    value = max(1, min(127, int(round(float(velocity)))))
    return min(VELOCITY_STEPS, key=lambda step: abs(step - value))


DurationSpec = tuple[int, int, bool, int, int]


@lru_cache(maxsize=None)
def _decompose_cached(length: int) -> tuple[DurationSpec, ...] | None:
    if length == 0:
        return ()
    if length < 0:
        return None
    for spec in DURATION_SPECS:
        ticks = spec[0]
        if ticks > length:
            continue
        rest = length - ticks
        if rest == 0:
            return (spec,)
        sub = _decompose_cached(rest)
        if sub is not None:
            return (spec,) + sub
    return None


def _decompose(length: int) -> list[DurationSpec]:
    if length == 0:
        return []
    result = _decompose_cached(length)
    if result is None:
        raise ValueError(f"GP tick length {length} is not representable")
    return list(result)


def _largest_fittable(length: int) -> DurationSpec:
    """Largest single beat that leaves a rest GP can represent."""
    for spec in DURATION_SPECS:
        ticks = spec[0]
        if ticks > length:
            continue
        if _decompose_cached(length - ticks) is not None:
            return spec
    raise ValueError(f"GP tick length {length} cannot be split into a beat and a rest")


def _duration(spec: DurationSpec) -> gp.Duration:
    _ticks, value, dotted, enters, times = spec
    if enters != 1 or times != 1:
        return gp.Duration(
            value,
            isDotted=dotted,
            tuplet=gp.Tuplet(enters=enters, times=times),
        )
    return gp.Duration(value, isDotted=dotted)


def _new_note(
    beat: gp.Beat,
    *,
    value: int,
    velocity: int,
    string: int,
    note_type: gp.NoteType,
    hammer: bool = False,
    slide: int = 0,
) -> gp.Note:
    note = gp.Note(
        beat,
        value=int(value),
        velocity=_snap_velocity(velocity),
        string=int(string),
        type=note_type,
    )
    note.type = note_type
    note.string = int(string)
    note.value = int(value)
    note.velocity = _snap_velocity(velocity)
    # Articulations belong only on the attack; a tie continues the note.
    if note_type == gp.NoteType.normal:
        if hammer:
            note.effect.hammer = True
        if slide > 0:
            note.effect.slides = [gp.SlideType.outUpwards]
        elif slide < 0:
            note.effect.slides = [gp.SlideType.outDownwards]
    return note


def _add_sounding(
    voice: gp.Voice,
    length: int,
    notes: list[tuple[int, int, int, gp.NoteType, bool, int]],
) -> None:
    # Length must be exactly representable here; callers use _decompose first.
    spec = next((s for s in DURATION_SPECS if s[0] == length), None)
    if spec is None:
        raise ValueError(f"length {length} is not a single representable beat")
    beat = gp.Beat(voice, status=gp.BeatStatus.normal, duration=_duration(spec))
    beat.notes = [
        _new_note(
            beat,
            value=value,
            velocity=velocity,
            string=string,
            note_type=note_type,
            hammer=hammer,
            slide=slide,
        )
        for value, velocity, string, note_type, hammer, slide in notes
    ]
    voice.beats.append(beat)


def _add_rests(voice: gp.Voice, length: int) -> None:
    for spec in _decompose(length):
        beat = gp.Beat(
            voice,
            status=gp.BeatStatus.rest,
            duration=_duration(spec),
        )
        beat.notes = []
        voice.beats.append(beat)


def _note_tuple(
    note: FingeredNote, note_type: gp.NoteType
) -> tuple[int, int, int, gp.NoteType, bool, int]:
    effects = note_type == gp.NoteType.normal
    return (
        note.fret,
        note.velocity,
        note.string,
        note_type,
        bool(note.hammer) and effects,
        int(note.slide) if effects else 0,
    )


def _drum_tuple(
    note: Note, lane: int
) -> tuple[int, int, int, gp.NoteType, bool, int]:
    return (note.pitch, note.velocity, lane, gp.NoteType.normal, False, 0)


def _add_onset_layout(
    voice: gp.Voice,
    notes: list[FingeredNote],
    measure_index: int,
    grid_t: int,
    bar_ticks: int,
) -> None:
    bar_start = measure_index * bar_ticks
    bar_end = bar_start + bar_ticks
    groups: dict[int, list[FingeredNote]] = {}
    for note in notes:
        if bar_start <= note.start_tick < bar_end:
            groups.setdefault(note.start_tick, []).append(note)
    onsets = sorted(groups)
    pos = bar_start
    for index, onset in enumerate(onsets):
        if onset < pos:
            continue
        if onset > pos:
            _add_rests(voice, onset - pos)
            pos = onset
        next_onset = onsets[index + 1] if index + 1 < len(onsets) else bar_end
        # The largest note value can leave an unrepresentable rest (a 1320-tick
        # gap: 1280 + 40). Pick a value whose remainder GP can write.
        interval = min(max(grid_t, next_onset - onset), bar_end - onset)
        duration_ticks = _largest_fittable(interval)[0]
        note_tuples = [
            _note_tuple(n, gp.NoteType.normal)
            for n in sorted(groups[onset], key=lambda n: n.string)
        ]
        _add_sounding(voice, duration_ticks, note_tuples)
        pos = onset + duration_ticks
        if pos < next_onset:
            _add_rests(voice, next_onset - pos)
            pos = next_onset
    if pos < bar_end:
        _add_rests(voice, bar_end - pos)


def _add_drum_layout(
    voice: gp.Voice,
    notes: list[Note],
    measure_index: int,
    grid_t: int,
    bar_ticks: int,
) -> None:
    bar_start = measure_index * bar_ticks
    bar_end = bar_start + bar_ticks
    groups: dict[int, list[Note]] = {}
    for note in notes:
        if bar_start <= note.start_tick < bar_end:
            groups.setdefault(note.start_tick, []).append(note)
    onsets = sorted(groups)
    pos = bar_start
    for onset in onsets:
        if onset > pos:
            _add_rests(voice, onset - pos)
            pos = onset
        if onset < pos:
            continue
        used_lanes: set[int] = set()
        note_tuples: list[tuple[int, int, int, gp.NoteType, bool, int]] = []
        for note in sorted(groups[onset], key=lambda n: (-n.velocity, n.pitch)):
            preferred = DRUM_LANE_BY_PITCH.get(note.pitch, 6)
            lane = preferred if preferred not in used_lanes else next(
                (candidate for candidate in range(1, 7) if candidate not in used_lanes), None
            )
            if lane is None:
                continue
            used_lanes.add(lane)
            note_tuples.append(_drum_tuple(note, lane))
        if not note_tuples:
            continue
        length = _largest_fittable(min(grid_t, bar_end - onset))[0]
        _add_sounding(voice, length, note_tuples)
        pos = onset + length
    if pos < bar_end:
        _add_rests(voice, bar_end - pos)


def _add_tie_layout(
    voice: gp.Voice,
    notes: list[FingeredNote],
    measure_index: int,
    state: dict[int, int],
    bar_ticks: int,
) -> None:
    """Lay out sustains, chaining ties when the previous GP note supports it.

    ``state`` maps string -> fret of the last emitted voice-0 beat and is
    carried across beats, measures, and bars. If the source notes overlap on a
    string (which a tablature cannot represent), the newer note wins and is
    written as a normal attack rather than an unbacked tie.
    """
    bar_start = measure_index * bar_ticks
    bar_end = bar_start + bar_ticks
    overlapping = [
        note for note in notes if note.start_tick < bar_end and note.end_tick > bar_start
    ]
    if not overlapping:
        _add_rests(voice, bar_ticks)
        state.clear()
        return
    boundaries = {bar_start, bar_end}
    for note in overlapping:
        boundaries.add(max(bar_start, note.start_tick))
        boundaries.add(min(bar_end, note.end_tick))
    points = sorted(boundaries)
    for left, right in zip(points, points[1:]):
        length = right - left
        if length <= 0:
            continue
        active_by_string: dict[int, FingeredNote] = {}
        for note in sorted(
            (n for n in overlapping if n.start_tick <= left < n.end_tick),
            key=lambda n: (n.start_tick, n.velocity, n.end_tick),
        ):
            active_by_string[int(note.string)] = note
        if not active_by_string:
            _add_rests(voice, length)
            state.clear()
            continue
        first_chunk = True
        for spec in _decompose(length):
            ticks = spec[0]
            note_tuples: list[tuple[int, int, int, gp.NoteType, bool, int]] = []
            next_state: dict[int, int] = {}
            for string, note in sorted(active_by_string.items()):
                is_attack = note.start_tick == left and first_chunk
                # A tie must continue the immediately preceding note on the
                # same string. If the source data has a gap or an overlapping
                # string, restart the note instead of emitting a broken tie.
                can_tie = state.get(string) == note.fret
                note_type = (
                    gp.NoteType.normal
                    if is_attack or not can_tie
                    else gp.NoteType.tie
                )
                note_tuples.append(_note_tuple(note, note_type))
                next_state[string] = note.fret
            _add_sounding(voice, ticks, note_tuples)
            state.clear()
            state.update(next_state)
            first_chunk = False


def _ensure_measures(track: gp.Track, headers: list[gp.MeasureHeader]) -> None:
    while len(track.measures) < len(headers):
        track.measures.append(gp.Measure(track, headers[len(track.measures)]))
    if len(track.measures) > len(headers):
        del track.measures[len(headers):]
    for measure in track.measures:
        while len(measure.voices) < 2:
            measure.voices.append(gp.Voice(measure))
        if len(measure.voices) > 2:
            del measure.voices[2:]
        measure.voices[0].beats = []
        # alphaTab 1.4 crashes on a completely empty second voice. A single
        # empty beat is the same padding TabForge uses and does not affect the
        # voice-0 duration or playback. Voice 1 is never used for our music.
        measure.voices[1].beats = [
            gp.Beat(
                measure.voices[1],
                status=gp.BeatStatus.empty,
                duration=gp.Duration(1),
            )
        ]


def key_signature(name: str | None) -> gp.KeySignature:
    """Resolve a PyGuitarPro KeySignature member from a canonical key name."""
    if not name:
        return gp.KeySignature.CMajor
    member = getattr(gp.KeySignature, str(name), None)
    if isinstance(member, gp.KeySignature):
        return member
    return gp.KeySignature.CMajor


def _build_song(
    tracks: list[TabTrack],
    bpm: float,
    measure_count: int,
    layout: Literal["onset", "ties"],
    quantize: int,
    key_name: str | None = None,
    time_signature: str = "4/4",
) -> gp.Song:
    meter: Meter = parse_time_signature(time_signature)
    song = gp.Song(tracks=[], measureHeaders=[])
    song.title = "audio2gp draft"
    song.artist = "audio2gp"
    song.tempo = max(20, min(300, int(round(float(bpm)))))
    song.key = key_signature(key_name)
    headers = [
        gp.MeasureHeader(
            number=index + 1,
            start=960 + index * meter.bar_ticks,
            keySignature=song.key,
            timeSignature=gp.TimeSignature(
                meter.numerator,
                gp.Duration(meter.denominator),
                beams=list(meter.beams),
            ),
        )
        for index in range(measure_count)
    ]
    song.measureHeaders.extend(headers)
    grid_t = grid_ticks(quantize)

    for index, tab_track in enumerate(tracks):
        is_percussion = bool(tab_track.is_percussion)
        channel = 9 if is_percussion else int(tab_track.channel)
        effect = 9 if is_percussion else min(15, channel + 1)
        if is_percussion:
            strings = [gp.GuitarString(number, 0) for number in range(1, 7)]
        else:
            strings = [
                gp.GuitarString(number, int(tab_track.tuning[number - 1]))
                for number in range(1, len(tab_track.tuning) + 1)
            ]
        track = gp.Track(
            song,
            number=index + 1,
            name=safe_ascii(tab_track.name, fallback=f"Track {index + 1}"),
            fretCount=24,
            isPercussionTrack=is_percussion,
            strings=strings,
            channel=gp.MidiChannel(
                channel=channel,
                effectChannel=effect,
                instrument=PROGRAMS["drums"] if is_percussion else int(tab_track.instrument),
            ),
            clefTranspose=0,
            clefTransposeSecondary=0,
        )
        _ensure_measures(track, headers)
        song.tracks.append(track)
        tie_state: dict[int, int] = {}
        for measure_index, measure in enumerate(track.measures):
            voice = measure.voices[0]
            if is_percussion:
                _add_drum_layout(
                    voice,
                    tab_track.drum_notes or [],
                    measure_index,
                    grid_t,
                    meter.bar_ticks,
                )
            elif layout == "onset":
                _add_onset_layout(
                    voice, tab_track.notes, measure_index, grid_t, meter.bar_ticks
                )
            else:
                _add_tie_layout(
                    voice, tab_track.notes, measure_index, tie_state, meter.bar_ticks
                )
    return song


def _tied_note_value_by_identity(self: object, note: gp.Note) -> int:
    """Reconstruct a GP5 tie's value by walking back by object identity.

    GP5 does not store a fret for a tied note; the reader must find the most
    recent note on the same string. PyGuitarPro 0.11 uses
    ``voice.beats.index(note.beat)`` while the current beat is still being
    filled, and ``Beat.__eq__`` compares note contents. A repeated pattern can
    therefore match an earlier beat and make the reader search the wrong
    prefix, turning valid tied notes into fret 0. Walking back with ``is``
    avoids that ambiguity.
    """
    measure = note.beat.voice.measure
    voice_index = measure.voices.index(note.beat.voice)
    for depth, candidate in enumerate(reversed(measure.track.measures)):
        voice = candidate.voices[voice_index]
        if depth == 0:
            beats: list[gp.Beat] = []
            for beat in voice.beats:
                if beat is note.beat:
                    break
                beats.append(beat)
        else:
            beats = voice.beats
        for beat in reversed(beats):
            if beat.status != gp.BeatStatus.empty:
                for previous in beat.notes:
                    if previous.string == note.string:
                        return previous.value
    return -1


@contextmanager
def _identity_tie_parser() -> Iterator[None]:
    from guitarpro.gp3 import GP3File

    original = GP3File.getTiedNoteValue
    GP3File.getTiedNoteValue = _tied_note_value_by_identity  # type: ignore[method-assign]
    try:
        yield
    finally:
        GP3File.getTiedNoteValue = original  # type: ignore[method-assign]


def parse_gp5(path: Path | str) -> gp.Song:
    """Parse a GP5 file with reliable tied-note values."""
    with _identity_tie_parser():
        return gp.parse(str(path))


def verify_gp5(
    path: Path,
    expected_measures: int,
    expected_tracks: int | None = None,
    *,
    time_signature: str = "4/4",
) -> list[str]:
    issues: list[str] = []
    try:
        song = parse_gp5(path)
    except Exception as exc:
        return [f"GP5 parse failed: {exc}"]
    meter = parse_time_signature(time_signature)
    if len(song.measureHeaders) != expected_measures:
        issues.append(
            f"measure header count {len(song.measureHeaders)} != {expected_measures}"
        )
    if expected_tracks is not None and len(song.tracks) != expected_tracks:
        issues.append(f"track count {len(song.tracks)} != {expected_tracks}")
    if not (20 <= int(song.tempo) <= 300):
        issues.append(f"tempo out of range: {song.tempo}")
    for header_index, header in enumerate(song.measureHeaders):
        signature = header.timeSignature
        # GP5 stores the lower number in Duration.value (4, 8), not in the
        # note-length value used for beats.
        actual = (int(signature.numerator), int(signature.denominator.value))
        if actual != (meter.numerator, meter.denominator):
            issues.append(
                f"measure header {header_index}: time signature {actual[0]}/{actual[1]} "
                f"!= {meter.label}"
            )
        expected_start = 960 + header_index * meter.bar_ticks
        if int(header.start) != expected_start:
            issues.append(
                f"measure header {header_index}: start {header.start} != {expected_start}"
            )
    for track_index, track in enumerate(song.tracks):
        if len(track.measures) != expected_measures:
            issues.append(
                f"track {track_index} measure count {len(track.measures)} != {expected_measures}"
            )
        if track.isPercussionTrack:
            if track.channel.channel != 9:
                issues.append(f"percussion track {track_index} channel {track.channel.channel} != 9")
            if getattr(track, "clefTranspose", 0) not in (0, None):
                issues.append(f"percussion track {track_index} clefTranspose != 0")
        # Tie continuity: a tie on a string must continue a note of the same
        # fret/value from the immediately preceding beat. A rest or an absent
        # string ends the previous note.
        active: dict[int, int] = {}
        for measure_index, measure in enumerate(track.measures):
            if len(measure.voices) != 2:
                issues.append(
                    f"track {track_index} measure {measure_index}: {len(measure.voices)} voices != 2"
                )
                continue
            beats = measure.voices[0].beats
            total = sum(int(beat.duration.time) for beat in beats)
            if total != meter.bar_ticks:
                issues.append(
                    f"track {track_index} measure {measure_index}: voice0 ticks "
                    f"{total} != {meter.bar_ticks} ({meter.label})"
                )
            for beat_index, beat in enumerate(beats):
                where = f"track {track_index} measure {measure_index} beat {beat_index}"
                if int(beat.duration.time) <= 0:
                    issues.append(f"{where}: zero duration")
                if beat.status not in (gp.BeatStatus.normal, gp.BeatStatus.rest):
                    issues.append(f"{where}: unusual status {beat.status}")
                if beat.status == gp.BeatStatus.rest and beat.notes:
                    issues.append(f"{where}: rest beat has notes")
                if beat.status == gp.BeatStatus.normal and not beat.notes:
                    issues.append(f"{where}: normal beat has no notes")
                seen_strings: set[int] = set()
                next_active: dict[int, int] = {}
                for note in beat.notes:
                    string = int(note.string)
                    if not 1 <= string <= len(track.strings):
                        issues.append(f"{where}: invalid string {note.string}")
                    if string in seen_strings:
                        issues.append(f"{where}: duplicate note on string {string}")
                    seen_strings.add(string)
                    if not 1 <= int(note.velocity) <= 127:
                        issues.append(f"{where}: invalid velocity {note.velocity}")
                    if note.type not in (gp.NoteType.normal, gp.NoteType.tie):
                        issues.append(f"{where}: invalid note type {note.type}")
                    if not track.isPercussionTrack:
                        if note.type == gp.NoteType.tie:
                            previous = active.get(string)
                            if previous is None or previous != int(note.value):
                                issues.append(
                                    f"{where}: unbacked tie on string {string} "
                                    f"(value {note.value}, previous {previous})"
                                )
                        next_active[string] = int(note.value)
                active = {} if track.isPercussionTrack else next_active
    return issues


def _write_and_validate(
    tracks: list[TabTrack],
    bpm: float,
    measure_count: int,
    path: Path,
    layout: Literal["onset", "ties"],
    quantize: int,
    key_name: str | None = None,
    chords: Iterable[ChordEvent] | None = None,
    *,
    time_signature: str = "4/4",
) -> list[str]:
    song = _build_song(
        tracks,
        bpm,
        measure_count,
        layout,
        quantize,
        key_name,
        time_signature=time_signature,
    )
    chord_warnings = _annotate_chords(song, chords)
    ensure_dir(path.parent)
    gp.write(song, str(path), version=(5, 1, 0), encoding="cp1252")
    issues = verify_gp5(
        path,
        expected_measures=measure_count,
        expected_tracks=len(tracks),
        time_signature=time_signature,
    )
    if issues:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise GP5ValidationError("; ".join(issues[:12]))
    warnings: list[str] = list(chord_warnings)
    for track in tracks:
        warnings.extend(track.warnings)
        if track.is_percussion and not track.drum_notes:
            warnings.append(f"{track.name}: empty drum track")
    return warnings


def write_gp5(
    tracks: list[TabTrack],
    bpm: float,
    measure_count: int,
    path: Path,
    layout: Layout = "auto",
    quantize: int = 16,
    key_name: str | None = None,
    chords: Iterable[ChordEvent] | None = None,
    *,
    time_signature: str = "4/4",
) -> list[str]:
    measure_count = max(1, int(measure_count))
    warnings: list[str] = []
    if layout == "onset":
        return _write_and_validate(
            tracks,
            bpm,
            measure_count,
            path,
            "onset",
            quantize,
            key_name,
            chords,
            time_signature=time_signature,
        )
    if layout == "ties":
        return _write_and_validate(
            tracks,
            bpm,
            measure_count,
            path,
            "ties",
            quantize,
            key_name,
            chords,
            time_signature=time_signature,
        )
    try:
        return _write_and_validate(
            tracks,
            bpm,
            measure_count,
            path,
            "ties",
            quantize,
            key_name,
            chords,
            time_signature=time_signature,
        )
    except Exception as exc:
        warnings.append(f"tie layout failed; using onset layout: {exc}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        result = _write_and_validate(
            tracks,
            bpm,
            measure_count,
            path,
            "onset",
            quantize,
            key_name,
            chords,
            time_signature=time_signature,
        )
        return warnings + result
