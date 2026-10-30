"""Import an existing MIDI file as the transcription source.

MIDI input skips Demucs separation and Basic Pitch: tracks are mapped to the
app's four stems by GM program, channel and track name, then re-quantized,
re-fingered and exported like any other job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Options, parse_time_signature
from .midifile import (
    first_tempo_bpm,
    first_time_signature,
    load_midi_notes,
    midi_duration,
    tempo_change_count,
)
from .types import NoteEvent

MIDI_EXTENSIONS = {".mid", ".midi"}
DRUM_CHANNEL = 9
BASS_PROGRAMS = range(32, 40)  # GM 0-based: Acoustic Bass .. Synth Bass 2
GUITAR_PROGRAMS = range(24, 32)
VOCAL_PROGRAMS = range(52, 55)

_NAME_HINTS = (
    ("drums", ("drum", "perc", "鼓")),
    ("bass", ("bass", "贝斯", "贝司")),
    ("vocals", ("vocal", "voice", "sing", "人声", "唱")),
    ("other", ("guitar", "gtr", "key", "piano", "吉他", "钢琴")),
)


@dataclass
class MidiInput:
    bpm: float
    duration_s: float
    events: dict[str, list[NoteEvent]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _stem_for_note(note) -> str:
    if int(note.channel) == DRUM_CHANNEL:
        return "drums"
    name = str(note.name or "").lower()
    for stem, hints in _NAME_HINTS:
        if any(hint in name for hint in hints):
            return stem
    program = int(note.program)
    if program in BASS_PROGRAMS:
        return "bass"
    if program in GUITAR_PROGRAMS:
        return "other"
    if program in VOCAL_PROGRAMS:
        return "vocals"
    # Piano, keys, strings and unknown melodic instruments are the GP5 guitar
    # source by default; the user can still change the selected stems.
    return "other"


def prepare(
    path: Path | str,
    options: Options,
    logger: logging.Logger | None = None,
) -> MidiInput:
    path = Path(path)
    notes = load_midi_notes(path)
    if not notes:
        raise ValueError("MIDI 文件中没有音符")
    events: dict[str, list[NoteEvent]] = {}
    for note in notes:
        if not 0 <= int(note.pitch) <= 127:
            continue
        end_s = max(float(note.end_s), float(note.start_s) + 0.01)
        stem = _stem_for_note(note)
        events.setdefault(stem, []).append(
            NoteEvent(
                start_s=float(note.start_s),
                end_s=end_s,
                pitch=int(note.pitch),
                amplitude=max(0.05, min(1.0, float(note.velocity) / 127.0)),
            )
        )
    for notes_for_stem in events.values():
        notes_for_stem.sort(key=lambda event: (event.start_s, event.pitch))

    bpm = first_tempo_bpm(path)
    warnings: list[str] = []
    changes = tempo_change_count(path)
    if changes > 1:
        warnings.append(
            f"MIDI 含 {changes} 个速度事件；当前仅使用首个速度 {bpm:.2f} BPM，"
            "整曲按固定速度重新量化。"
        )
    signature = first_time_signature(path)
    meter = parse_time_signature(options.time_signature)
    if signature is not None and f"{signature[0]}/{signature[1]}" != meter.label:
        warnings.append(
            f"MIDI 文件拍号为 {signature[0]}/{signature[1]}，当前选择 {meter.label}；"
            "小节线会按所选拍号重新划分。"
        )
    if logger:
        logger.info(
            "MIDI input: %d notes, %d stems, %.2f BPM, %.2fs",
            len(notes),
            len(events),
            bpm,
            midi_duration(path),
        )
    return MidiInput(
        bpm=bpm,
        duration_s=midi_duration(path),
        events=events,
        warnings=warnings,
    )
