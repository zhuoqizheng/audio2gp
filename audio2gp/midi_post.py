from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import mido

from .config import (
    PROGRAMS,
    TICKS_PER_QUARTER,
    Options,
    grid_seconds,
    grid_ticks,
    parse_time_signature,
    ticks_to_seconds,
)
from . import techniques
from .types import ChordEvent, Note, NoteEvent, TabTrack

CLUSTER_SECONDS = 0.035


def _velocity(amplitude: float) -> int:
    try:
        amp = float(amplitude)
    except (TypeError, ValueError):
        amp = 0.7
    if not (amp == amp):  # NaN
        amp = 0.7
    amp = max(0.0, min(1.0, amp))
    return max(1, min(127, int(round(20 + 107 * amp))))


def _cluster(events: Iterable[NoteEvent], threshold_s: float = CLUSTER_SECONDS) -> list[tuple[float, list[NoteEvent]]]:
    """Group near-simultaneous attacks before any grid snapping."""
    valid = [
        e
        for e in events
        if e.end_s > e.start_s and 0 <= int(e.pitch) <= 127
    ]
    valid.sort(key=lambda e: (float(e.start_s), int(e.pitch)))
    clusters: list[tuple[float, list[NoteEvent]]] = []
    current: list[NoteEvent] = []
    anchor = 0.0
    for event in valid:
        if not current:
            current = [event]
            anchor = float(event.start_s)
        elif float(event.start_s) - anchor <= threshold_s:
            current.append(event)
        else:
            clusters.append((sum(float(e.start_s) for e in current) / len(current), current))
            current = [event]
            anchor = float(event.start_s)
    if current:
        clusters.append((sum(float(e.start_s) for e in current) / len(current), current))
    return clusters


def _dedupe_same_pitch(notes: list[Note]) -> list[Note]:
    best: dict[tuple[int, int], Note] = {}
    for note in notes:
        key = (note.start_tick, note.pitch)
        old = best.get(key)
        if old is None:
            best[key] = note
        elif (note.velocity, note.end_tick) > (old.velocity, old.end_tick):
            best[key] = note
    return list(best.values())


def _trim_pitch_overlaps(notes: list[Note]) -> list[Note]:
    by_pitch: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        by_pitch[note.pitch].append(note)
    result: list[Note] = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda n: (n.start_tick, n.end_tick, n.velocity))
        kept: list[Note] = []
        for note in pitch_notes:
            if kept and note.start_tick < kept[-1].end_tick:
                prev = kept[-1]
                if note.start_tick > prev.start_tick:
                    prev.end_tick = note.start_tick
                    kept.append(note)
                else:
                    # Same start: keep the stronger/longer replacement.
                    if (note.velocity, note.end_tick) >= (prev.velocity, prev.end_tick):
                        kept[-1] = note
            else:
                kept.append(note)
        result.extend(n for n in kept if n.end_tick > n.start_tick)
    return result


def _limit_polyphony(notes: list[Note], limit: int) -> list[Note]:
    """Keep at most ``limit`` notes per attack, preferring strong notes."""
    if limit <= 0:
        return notes
    by_start: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        by_start[note.start_tick].append(note)
    kept: list[Note] = []
    for start in sorted(by_start):
        candidates = sorted(
            by_start[start],
            key=lambda n: (-n.velocity, n.pitch, -n.end_tick),
        )
        kept.extend(candidates[:limit])
    kept.sort(key=lambda n: (n.start_tick, n.pitch))
    return kept


def _enforce_monophonic(notes: list[Note], stem: str) -> list[Note]:
    by_start: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        by_start[note.start_tick].append(note)
    chosen: list[Note] = []
    for start in sorted(by_start):
        candidates = by_start[start]
        if stem == "bass":
            chosen.append(min(candidates, key=lambda n: (n.pitch, -n.velocity)))
        else:  # vocals
            chosen.append(max(candidates, key=lambda n: (n.velocity, n.pitch)))
    chosen.sort(key=lambda n: (n.start_tick, n.pitch))
    for prev, curr in zip(chosen, chosen[1:]):
        if curr.start_tick < prev.end_tick:
            prev.end_tick = curr.start_tick
    return [n for n in chosen if n.end_tick > n.start_tick]


def clean(
    events: dict[str, list[NoteEvent]],
    options: Options,
    bpm: float,
    logger: logging.Logger | None = None,
    offset_s: float = 0.0,
) -> dict[str, list[Note]]:
    """Quantize raw events into integer GP ticks for every stem.

    ``offset_s`` is the audio time represented by tick 0. It is set to the
    downbeat grid chosen by ``rhythm.choose_grid_offset`` so quantized notes
    and GP5 bar lines agree with the recording's bar phase. Material before
    the offset (a few milliseconds of attack or a pickup bar) is clamped to
    tick 0; pickups are normally handled by placing the first downbeat at
    the start of bar 2.
    """
    grid_t = grid_ticks(options.quantize)
    grid_s = grid_seconds(bpm, options.quantize)
    offset_s = float(offset_s) if math.isfinite(float(offset_s)) else 0.0
    out: dict[str, list[Note]] = {}

    for stem, raw_events in events.items():
        clusters = _cluster(raw_events)
        notes: list[Note] = []
        dropped_short = 0
        for mean_start, members in clusters:
            start_tick = max(
                0, int(round((mean_start - offset_s) / grid_s)) * grid_t
            )
            for event in members:
                duration = float(event.end_s) - float(event.start_s)
                if stem != "drums" and duration < options.min_note_sec:
                    dropped_short += 1
                    continue
                if stem == "drums":
                    end_tick = start_tick + grid_t
                else:
                    # Snap both ends to the same grid. Non-grid ends make the
                    # GP5 tie layout unrepresentable and force the onset
                    # fallback, so this is deliberate, not just cosmetic.
                    end_tick = (
                        int(round((float(event.end_s) - offset_s) / grid_s))
                        * grid_t
                    )
                    end_tick = max(end_tick, start_tick + grid_t)
                slide = (
                    techniques.event_slide(event)
                    if options.detect_techniques
                    else 0
                )
                notes.append(
                    Note(
                        start_tick=start_tick,
                        end_tick=end_tick,
                        pitch=int(event.pitch),
                        velocity=_velocity(event.amplitude),
                        stem=stem,
                        slide=slide,
                    )
                )
        before = len(notes)
        notes = _dedupe_same_pitch(notes)
        if stem == "drums":
            notes.sort(key=lambda n: (n.start_tick, n.pitch))
        else:
            notes = _trim_pitch_overlaps(notes)
            poly_before = len(notes)
            notes = _limit_polyphony(notes, int(options.max_polyphony))
            if stem in {"bass", "vocals"}:
                notes = _enforce_monophonic(notes, stem)
            else:
                notes.sort(key=lambda n: (n.start_tick, n.pitch))
            if logger and poly_before != len(notes):
                logger.info(
                    "midi_post %s: polyphony cap dropped %d note(s)",
                    stem,
                    poly_before - len(notes),
                )
        out[stem] = notes
        if logger:
            logger.info(
                "midi_post %s: %d events -> %d notes (%d short dropped, %d duplicates)",
                stem,
                len(raw_events),
                len(notes),
                dropped_short,
                max(0, before - len(notes)),
            )
    return out


def _write_midi(
    tracks: list[tuple[str, list[Note], int, int, bool]],
    bpm: float,
    path: Path,
    markers: list[tuple[int, str]] | None = None,
    *,
    time_signature: str = "4/4",
) -> None:
    """Write a type-1 MIDI with explicit channels, including empty tracks."""
    meter = parse_time_signature(time_signature)
    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_QUARTER)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="audio2gp", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=0))
    meta.append(
        mido.MetaMessage(
            "time_signature",
            numerator=meter.numerator,
            denominator=meter.denominator,
            time=0,
        )
    )
    marker_tick = 0
    for tick, label in sorted(
        ((max(0, int(tick)), str(label)) for tick, label in (markers or [])),
        key=lambda item: item[0],
    ):
        delta = max(0, tick - marker_tick)
        meta.append(mido.MetaMessage("marker", text=label[:120], time=delta))
        marker_tick = tick
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    for name, notes, program, channel, is_drum in tracks:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name[:64], time=0))
        track.append(
            mido.Message("program_change", program=int(program), channel=int(channel), time=0)
        )
        events: list[tuple[int, int, int, int]] = []
        for note in notes:
            start = max(0, int(note.start_tick))
            end = max(start + 1, int(note.end_tick))
            velocity = max(1, min(127, int(note.velocity)))
            pitch = max(0, min(127, int(note.pitch)))
            # note_off before note_on at the same tick avoids stuck/retrigger issues
            events.append((start, 1, pitch, velocity))
            events.append((end, 0, pitch, 0))
        events.sort(key=lambda item: (item[0], item[1], item[2]))
        last_tick = 0
        for tick, kind, pitch, velocity in events:
            delta = max(0, tick - last_tick)
            if kind == 1:
                msg = mido.Message(
                    "note_on", note=pitch, velocity=velocity, channel=int(channel), time=delta
                )
            else:
                msg = mido.Message(
                    "note_off", note=pitch, velocity=0, channel=int(channel), time=delta
                )
            track.append(msg)
            last_tick = tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(track)

    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))


def write_stem_midi(
    notes: list[Note],
    bpm: float,
    path: Path,
    program: int,
    channel: int,
    name: str,
    drums: bool = False,
    *,
    time_signature: str = "4/4",
) -> None:
    _write_midi(
        [(name, notes, program, channel, drums)],
        bpm,
        path,
        time_signature=time_signature,
    )


def write_full_midi(
    all_notes: dict[str, list[Note]],
    bpm: float,
    path: Path,
    stems: Iterable[str] | None = None,
    chords: Iterable[ChordEvent] | None = None,
    *,
    time_signature: str = "4/4",
) -> None:
    channels = {"vocals": 0, "drums": 9, "bass": 2, "other": 1}
    programs = {
        "vocals": PROGRAMS["vocals"],
        "drums": PROGRAMS["drums"],
        "bass": PROGRAMS["bass"],
        "other": PROGRAMS["guitar"],
    }
    labels = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "other": "Guitar"}
    order = tuple(stems) if stems is not None else ("vocals", "drums", "bass", "other")
    unknown = set(order) - set(labels)
    if unknown:
        raise ValueError(f"unknown stems for full.mid: {sorted(unknown)}")
    tracks = []
    for stem in order:
        tracks.append(
            (
                labels[stem],
                all_notes.get(stem, []),
                programs[stem],
                channels[stem],
                stem == "drums",
            )
        )
    markers = [
        (int(chord.start_tick), str(chord.label)) for chord in (chords or [])
    ]
    _write_midi(
        tracks,
        bpm,
        path,
        markers=markers,
        time_signature=time_signature,
    )


def _track_midi_data(
    track: TabTrack,
) -> tuple[str, list[Note], int, int, bool]:
    if track.is_percussion:
        notes = list(track.drum_notes or [])
    else:
        notes = [
            Note(
                start_tick=note.start_tick,
                end_tick=note.end_tick,
                pitch=note.pitch,
                velocity=note.velocity,
                stem=track.stem,
            )
            for note in track.notes
        ]
    return (
        track.name,
        notes,
        int(track.instrument),
        int(track.channel),
        bool(track.is_percussion),
    )


def write_tab_track_midi(
    track: TabTrack,
    bpm: float,
    path: Path,
    *,
    time_signature: str = "4/4",
) -> None:
    """Write one GP track as MIDI, preserving its channel/program.

    This is the editor rebuild path: it reflects exactly the fingered notes
    that the GP5 writer sees, including any octave folds or string edits.
    """
    _write_midi(
        [_track_midi_data(track)],
        bpm,
        path,
        time_signature=time_signature,
    )


def write_tab_tracks_midi(
    tracks: list[TabTrack],
    bpm: float,
    path: Path,
    chords: Iterable[ChordEvent] | None = None,
    *,
    time_signature: str = "4/4",
) -> None:
    if not tracks:
        raise ValueError("cannot write a MIDI file with no tracks")
    markers = [
        (int(chord.start_tick), str(chord.label)) for chord in (chords or [])
    ]
    _write_midi(
        [_track_midi_data(track) for track in tracks],
        bpm,
        path,
        markers=markers,
        time_signature=time_signature,
    )


def note_seconds(note: Note, bpm: float) -> tuple[float, float]:
    return (
        ticks_to_seconds(note.start_tick, bpm),
        ticks_to_seconds(note.end_tick, bpm),
    )
