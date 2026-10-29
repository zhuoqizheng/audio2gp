from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

ProgressCB = Callable[[str, float, str], None]
"""stage, overall 0..100, message"""


@dataclass(frozen=True)
class NoteEvent:
    """A raw transcription event; times are seconds from the start of the file."""

    start_s: float
    end_s: float
    pitch: int
    amplitude: float
    # Basic Pitch contour bins (3 bins per semitone) when available.
    bends: tuple[int, ...] = ()


@dataclass
class Note:
    """A cleaned, quantized note in integer GP ticks (quarter note = 960)."""

    start_tick: int
    end_tick: int
    pitch: int
    velocity: int
    stem: str
    # Audio-derived out-slide: +1 upwards, -1 downwards, 0 none.
    slide: int = 0


@dataclass
class FingeredNote:
    start_tick: int
    end_tick: int
    pitch: int
    velocity: int
    string: int
    fret: int
    # Inferred during fingering/technique post-processing. The GP5 writer
    # emits these only on attack notes.
    hammer: bool = False
    slide: int = 0


@dataclass
class TabTrack:
    name: str
    stem: str
    tuning: list[int]
    notes: list[FingeredNote] = field(default_factory=list)
    is_percussion: bool = False
    channel: int = 0
    instrument: int = 25
    warnings: list[str] = field(default_factory=list)
    drum_notes: list[Note] | None = None


@dataclass
class ChordEvent:
    """A detected chord over a tick range; ticks match the score grid."""

    start_tick: int
    end_tick: int
    start_s: float
    end_s: float
    label: str
    root: int
    quality: str
    confidence: float


@dataclass
class PreprocessedAudio:
    path: Path
    duration_s: float
    samplerate: int = 44100
    bpm: float = 120.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    job_id: str
    status: Literal["ok", "partial"]
    out_dir: Path
    work_dir: Path
    stems: dict[str, Path]
    midi: dict[str, Path]
    full_midi: Path
    gp5: Path | None
    zip_path: Path
    meta: dict
    warnings: list[str]
    note_counts: dict[str, int]
    timings: dict[str, float]
    device: str
    bpm: float
    duration_s: float
    rhythm: dict = field(default_factory=dict)
    key: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    chords: list[ChordEvent] = field(default_factory=list)
    chords_path: Path | None = None
    output_format: str = "gp5_midi"
