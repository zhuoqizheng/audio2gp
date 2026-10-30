"""Small MIDI reader shared by evaluation, external engines, and MIDI input.

It converts any Standard MIDI File into absolute-time notes. Tempo changes are
honored, programs are tracked per channel inside each track, and channel 9 is
left for the caller to interpret as drums (GM convention).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import mido

DEFAULT_TEMPO = 500000  # microseconds per quarter = 120 BPM


@dataclass(frozen=True)
class MidiNote:
    start_s: float
    end_s: float
    pitch: int
    velocity: int
    channel: int
    program: int
    track: int
    name: str


class TempoMap:
    def __init__(self, mid: mido.MidiFile) -> None:
        changes: dict[int, int] = {0: DEFAULT_TEMPO}
        for track in mid.tracks:
            tick = 0
            for message in track:
                tick += int(message.time)
                if message.type == "set_tempo":
                    changes[int(tick)] = int(message.tempo)

        self.ticks_per_beat = int(mid.ticks_per_beat)
        self.ticks: list[int] = []
        self.tempos: list[int] = []
        self.seconds: list[float] = []
        previous_tick = 0
        previous_tempo = DEFAULT_TEMPO
        elapsed = 0.0
        for tick in sorted(changes):
            elapsed += (
                (tick - previous_tick) * previous_tempo / 1e6 / self.ticks_per_beat
            )
            self.ticks.append(tick)
            self.tempos.append(changes[tick])
            self.seconds.append(elapsed)
            previous_tick = tick
            previous_tempo = changes[tick]

    def at(self, tick: int) -> float:
        index = max(0, bisect_right(self.ticks, int(tick)) - 1)
        delta = int(tick) - self.ticks[index]
        return (
            self.seconds[index]
            + delta * self.tempos[index] / 1e6 / self.ticks_per_beat
        )


def load_midi_notes(path: Path | str) -> list[MidiNote]:
    mid = mido.MidiFile(str(path))
    tempo = TempoMap(mid)
    notes: list[MidiNote] = []
    for index, track in enumerate(mid.tracks):
        tick = 0
        name = ""
        program = 0
        pending: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for message in track:
            tick += int(message.time)
            if message.type == "track_name" and not name:
                name = str(message.name)
            elif message.type == "program_change":
                program = int(message.program)
            elif message.type == "note_on" and int(message.velocity) > 0:
                key = (int(message.channel), int(message.note))
                pending.setdefault(key, []).append(
                    (tick, int(message.velocity), program)
                )
            elif message.type in {"note_off", "note_on"}:
                key = (int(message.channel), int(message.note))
                starts = pending.get(key)
                if not starts:
                    continue
                start_tick, velocity, start_program = starts.pop(0)
                start_s = tempo.at(start_tick)
                end_s = tempo.at(tick)
                if end_s <= start_s:
                    continue
                notes.append(
                    MidiNote(
                        start_s=start_s,
                        end_s=end_s,
                        pitch=key[1],
                        velocity=velocity,
                        channel=key[0],
                        program=start_program,
                        track=index,
                        name=name,
                    )
                )
    notes.sort(key=lambda note: (note.start_s, note.pitch))
    return notes


def midi_duration(path: Path | str) -> float:
    notes = load_midi_notes(path)
    return max((note.end_s for note in notes), default=0.0)


def first_tempo_bpm(path: Path | str) -> float:
    mid = mido.MidiFile(str(path))
    for track in mid.tracks:
        for message in track:
            if message.type == "set_tempo":
                return float(mido.tempo2bpm(int(message.tempo)))
    return float(mido.tempo2bpm(DEFAULT_TEMPO))


def tempo_change_count(path: Path | str) -> int:
    mid = mido.MidiFile(str(path))
    ticks: set[int] = set()
    for track in mid.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.type == "set_tempo":
                ticks.add(tick)
    return len(ticks)


def first_time_signature(path: Path | str) -> tuple[int, int] | None:
    mid = mido.MidiFile(str(path))
    for track in mid.tracks:
        for message in track:
            if message.type == "time_signature":
                return int(message.numerator), int(message.denominator)
    return None
