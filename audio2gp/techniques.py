from __future__ import annotations

import math
from collections import defaultdict

from .config import TICKS_PER_QUARTER
from .types import FingeredNote, NoteEvent

# Basic Pitch contours have three bins per semitone.
CONTOUR_BINS_PER_SEMITONE = 3.0
SLIDE_MIN_SEMITONES = 0.7
SLIDE_JITTER_SEMITONES = 0.2
LEGATO_MAX_GAP_S = 0.06
LEGATO_MIN_INTERVAL = 1
LEGATO_MAX_INTERVAL = 4


def slide_direction(bends: tuple[int, ...] | list[int] | None) -> int:
    """Detect a conservative out-slide from a Basic Pitch bend contour.

    This mirrors TabForge's shape test: the contour must keep moving in one
    direction and end at least 0.7 semitones away, with no span that cannot be
    explained by the net movement. It intentionally returns 0 for bends
    (trajectories that return to their starting pitch) and vibrato.
    """
    if not bends or len(bends) < 3:
        return 0
    try:
        values = [float(value) / CONTOUR_BINS_PER_SEMITONE for value in bends]
    except (TypeError, ValueError):
        return 0
    if not all(math.isfinite(value) for value in values):
        return 0
    net = values[-1] - values[0]
    if abs(net) < SLIDE_MIN_SEMITONES:
        return 0
    span = max(values) - min(values)
    if span > abs(net) + 2.0 * SLIDE_JITTER_SEMITONES:
        return 0
    # Require most of the trajectory to move in the net direction; a wide
    # vibrato with an accidental final offset should not become a slide.
    moving = 0
    useful = 0
    for left, right in zip(values, values[1:]):
        delta = right - left
        if abs(delta) < 0.05:
            continue
        moving += 1
        if (delta > 0) == (net > 0):
            useful += 1
    return (1 if net > 0 else -1) if moving and useful / moving >= 0.75 else 0


def mark_legato(
    notes: list[FingeredNote],
    bpm: float,
    *,
    max_gap_s: float = LEGATO_MAX_GAP_S,
    max_interval: int = LEGATO_MAX_INTERVAL,
) -> tuple[list[FingeredNote], dict[str, int]]:
    """Mark conservative hammer-on/pull-off attacks on a fingered part.

    Only isolated single-note onsets on the same string, separated by at most
    60 ms, a 1-4 semitone interval, and a non-accented second note qualify.
    The flag goes on the first note, where Guitar Pro stores the articulation.
    Notes that already have an audio slide are left alone.
    """
    by_start: dict[int, list[FingeredNote]] = defaultdict(list)
    for note in notes:
        by_start[int(note.start_tick)].append(note)
    onsets = sorted(by_start)
    tick_seconds = 60.0 / (TICKS_PER_QUARTER * max(1e-9, float(bpm)))
    counts = {"hammer_on": 0, "pull_off": 0}
    for first_tick, second_tick in zip(onsets, onsets[1:]):
        first_group = by_start[first_tick]
        second_group = by_start[second_tick]
        if len(first_group) != 1 or len(second_group) != 1:
            continue
        first = first_group[0]
        second = second_group[0]
        if first.string != second.string or first.slide or second.slide:
            continue
        interval = abs(int(second.pitch) - int(first.pitch))
        if not LEGATO_MIN_INTERVAL <= interval <= max_interval:
            continue
        gap_s = (int(second.start_tick) - int(first.end_tick)) * tick_seconds
        if not 0.0 <= gap_s <= max_gap_s:
            continue
        # A hammer-on/pull-off is an ornament, not a second accent.
        if int(second.velocity) > int(first.velocity):
            continue
        first.hammer = True
        if second.pitch > first.pitch:
            counts["hammer_on"] += 1
        else:
            counts["pull_off"] += 1
    return notes, counts


def event_slide(event: NoteEvent) -> int:
    return slide_direction(event.bends)
