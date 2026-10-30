from __future__ import annotations

import importlib.util
from pathlib import Path

from audio2gp.types import NoteEvent

# "scripts" is not a package and the ROS installation ships a same-named one,
# so load the file directly instead of importing scripts.*.
_SPEC = importlib.util.spec_from_file_location(
    "six_stem_experiment",
    Path(__file__).resolve().parent.parent / "scripts" / "six_stem_experiment.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
events_to_notes = _MODULE.events_to_notes


def test_events_to_notes_converts_seconds_to_ticks() -> None:
    events = [NoteEvent(0.0, 0.5, 64, 0.8), NoteEvent(1.25, 1.25, 67, 0.2)]
    notes = events_to_notes(events, "guitar", bpm=120.0)
    assert [(note.start_tick, note.end_tick) for note in notes] == [(0, 960), (2400, 2401)]
    assert [note.pitch for note in notes] == [64, 67]
    assert notes[0].velocity == 102
    assert all(note.stem == "guitar" for note in notes)
