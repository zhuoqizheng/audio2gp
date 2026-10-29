from __future__ import annotations

import copy
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .chords import QUALITIES, chords_from_state
from .config import parse_time_signature
from .gp_writer import write_gp5
from .midi_post import write_tab_track_midi, write_tab_tracks_midi
from .types import ChordEvent, FingeredNote, Note, TabTrack
from .utils import read_json, safe_ascii, slugify, write_json, zip_directory

FORMAT = 1
STEMS = ("vocals", "drums", "bass", "other")


class StateError(ValueError):
    """A user-editable score state is invalid."""


class RevisionConflict(StateError):
    pass


def _track_id(track: TabTrack, used: set[str]) -> str:
    base = {
        "Guitar": "guitar",
        "Bass": "bass",
        "Drums": "drums",
        "Vocals": "vocals",
    }.get(track.name)
    if not base:
        base = slugify(track.name, max_len=24) or "track"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _chord_entries(chords: list[ChordEvent] | None) -> list[dict]:
    return [
        {
            "start_tick": int(chord.start_tick),
            "end_tick": int(chord.end_tick),
            "start_s": round(float(chord.start_s), 3),
            "end_s": round(float(chord.end_s), 3),
            "label": str(chord.label),
            "root": int(chord.root),
            "quality": str(chord.quality),
            "confidence": round(float(chord.confidence), 3),
        }
        for chord in (chords or [])
    ]


def build_state(
    tracks: list[TabTrack],
    bpm: float,
    measure_count: int,
    *,
    job_id: str,
    quantize: int,
    layout: str,
    max_fret: int,
    key: dict | None = None,
    chords: list[ChordEvent] | None = None,
    time_signature: str = "4/4",
) -> dict:
    """Project the GP5 track model into the browser-editable score IR."""
    meter = parse_time_signature(time_signature)
    used: set[str] = set()
    counter = 0
    out_tracks: list[dict] = []
    for track in tracks:
        notes: list[dict] = []
        for note in track.notes:
            counter += 1
            notes.append(
                {
                    "id": f"n{counter:06d}",
                    "start_tick": int(note.start_tick),
                    "end_tick": int(note.end_tick),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "string": int(note.string),
                    "fret": int(note.fret),
                    "hammer": bool(note.hammer),
                    "slide": int(note.slide),
                }
            )
        drum_notes: list[dict] = []
        for note in track.drum_notes or []:
            counter += 1
            drum_notes.append(
                {
                    "id": f"n{counter:06d}",
                    "start_tick": int(note.start_tick),
                    "end_tick": int(note.end_tick),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                }
            )
        out_tracks.append(
            {
                "id": _track_id(track, used),
                "name": safe_ascii(track.name, fallback="Track"),
                "stem": str(track.stem),
                "tuning": [int(value) for value in track.tuning],
                "is_percussion": bool(track.is_percussion),
                "channel": int(track.channel),
                "instrument": int(track.instrument),
                "muted": False,
                "warnings": list(track.warnings),
                "notes": notes,
                "drum_notes": drum_notes,
            }
        )
    return {
        "format": FORMAT,
        "job_id": str(job_id),
        "revision": 0,
        "bpm": round(float(bpm), 3),
        "time_signature": meter.label,
        "beats_per_bar": meter.numerator,
        "bar_ticks": meter.bar_ticks,
        "measure_count": max(1, int(measure_count)),
        "quantize": int(quantize),
        "layout": str(layout),
        "max_fret": int(max_fret),
        "string_numbering": "1=highest",
        "key": dict(key or {}),
        "chords": _chord_entries(chords),
        "tracks": out_tracks,
    }


def validate_state(state: dict) -> None:
    if not isinstance(state, dict) or int(state.get("format", 0)) != FORMAT:
        raise StateError("unsupported score state format")
    if not isinstance(state.get("tracks"), list) or not state["tracks"]:
        raise StateError("score state has no tracks")
    if not 20.0 <= float(state.get("bpm", 0)) <= 300.0:
        raise StateError("score state BPM is out of range")
    if int(state.get("measure_count", 0)) < 1:
        raise StateError("score state has no measures")
    try:
        meter = parse_time_signature(str(state.get("time_signature") or "4/4"))
    except ValueError as exc:
        raise StateError(f"score state has an invalid time signature: {exc}") from exc
    max_tick = int(state.get("measure_count", 1)) * meter.bar_ticks
    previous_end = 0
    for chord in state.get("chords") or []:
        if not isinstance(chord, dict):
            raise StateError("score state has an invalid chord entry")
        start = int(chord.get("start_tick", -1))
        end = int(chord.get("end_tick", -1))
        if not 0 <= start < end <= max_tick:
            raise StateError("score state has a chord with an invalid tick range")
        if start < previous_end:
            raise StateError("score state chords are unsorted or overlapping")
        previous_end = end
        if not 0 <= int(chord.get("root", -1)) <= 11:
            raise StateError("score state has a chord with an invalid root")
        quality = str(chord.get("quality") or "")
        if quality not in QUALITIES:
            raise StateError(f"score state has an unknown chord quality: {quality}")
        if not str(chord.get("label") or "").strip():
            raise StateError("score state has a chord without a label")
        confidence = float(chord.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise StateError("score state has a chord confidence outside 0..1")
    track_ids: set[str] = set()
    for track in state["tracks"]:
        track_id = str(track.get("id") or "")
        if not track_id or track_id in track_ids:
            raise StateError("score state has duplicate or empty track id")
        track_ids.add(track_id)
        if not track.get("is_percussion"):
            tuning = track.get("tuning") or []
            if not 1 <= len(tuning) <= 7:
                raise StateError(f"track {track_id}: invalid tuning")
        tuning = [int(value) for value in track.get("tuning") or []]
        if any(not 0 <= value <= 127 for value in tuning):
            raise StateError(f"track {track_id}: tuning is out of MIDI range")
        seen: set[tuple[int, int]] = set()
        note_ids: set[str] = set()
        for note in track.get("notes") or []:
            if track.get("is_percussion"):
                raise StateError(f"track {track_id}: percussion track has fretted notes")
            note_id = str(note.get("id") or "")
            if not note_id or note_id in note_ids:
                raise StateError(f"track {track_id}: duplicate or empty note id")
            note_ids.add(note_id)
            start = int(note["start_tick"])
            end = int(note["end_tick"])
            if not 0 <= start < end <= max_tick:
                raise StateError(f"track {track_id}: note has invalid tick range")
            string = int(note["string"])
            fret = int(note["fret"])
            pitch = int(note["pitch"])
            if not 1 <= string <= len(tuning):
                raise StateError(f"track {track_id}: note has invalid string")
            if not 0 <= fret <= int(state.get("max_fret", 24)):
                raise StateError(f"track {track_id}: note has invalid fret")
            if pitch != tuning[string - 1] + fret:
                raise StateError(
                    f"track {track_id}: pitch {pitch} does not match string "
                    f"{string} fret {fret}"
                )
            key = (start, string)
            if key in seen:
                raise StateError(
                    f"track {track_id}: two notes on string {string} at tick {start}"
                )
            seen.add(key)
        for note in track.get("drum_notes") or []:
            if not track.get("is_percussion"):
                raise StateError(f"track {track_id}: melodic track has drum notes")
            note_id = str(note.get("id") or "")
            if not note_id or note_id in note_ids:
                raise StateError(f"track {track_id}: duplicate or empty note id")
            note_ids.add(note_id)
            if not 0 <= int(note["start_tick"]) < int(note["end_tick"]) <= max_tick:
                raise StateError(f"track {track_id}: drum note has invalid tick range")
            if not 0 <= int(note["pitch"]) <= 127:
                raise StateError(f"track {track_id}: drum note pitch is out of range")


def load_state(path: Path) -> dict:
    state = read_json(path)
    validate_state(state)
    return state


def get_track(state: dict, track_id: str) -> dict:
    for track in state["tracks"]:
        if str(track.get("id")) == str(track_id):
            return track
    raise StateError(f"unknown track: {track_id}")


def get_note(track: dict, note_id: str) -> dict:
    for note in track.get("notes") or []:
        if str(note.get("id")) == str(note_id):
            return note
    raise StateError(f"unknown note: {note_id}")


def _open_pitch(track: dict, string: int) -> int:
    tuning = track.get("tuning") or []
    if not 1 <= int(string) <= len(tuning):
        raise StateError(f"string must be between 1 and {len(tuning)}")
    return int(tuning[int(string) - 1])


def update_note_position(
    state: dict,
    track_id: str,
    note_id: str,
    *,
    string: int | None = None,
    fret: int | None = None,
    pitch: int | None = None,
) -> dict:
    """Move a note to another string/fret.

    - string only: keep the sounding pitch and derive the fret.
    - fret only: keep the string; the sounding pitch follows the fret.
    - string + fret: the pitch follows the fret unless an explicit pitch is
      supplied, which must then match.
    """
    track = get_track(state, track_id)
    if track.get("is_percussion"):
        raise StateError("drum tracks cannot be repositioned")
    note = get_note(track, note_id)
    max_fret = int(state.get("max_fret", 24))
    new_string = int(note["string"]) if string is None else int(string)
    open_pitch = _open_pitch(track, new_string)

    if fret is None:
        target_pitch = int(note["pitch"]) if pitch is None else int(pitch)
        new_fret = target_pitch - open_pitch
    else:
        new_fret = int(fret)
        derived_pitch = open_pitch + new_fret
        if pitch is not None and int(pitch) != derived_pitch:
            raise StateError(
                f"pitch {pitch} does not match string {new_string} fret {new_fret} "
                f"(sounds {derived_pitch})"
            )
    if not 0 <= new_fret <= max_fret:
        raise StateError(f"fret must be between 0 and {max_fret}")
    new_pitch = open_pitch + new_fret
    if not 0 <= new_pitch <= 127:
        raise StateError("note pitch is out of MIDI range")
    for other in track.get("notes") or []:
        if other is note:
            continue
        if (
            int(other["start_tick"]) == int(note["start_tick"])
            and int(other["string"]) == new_string
        ):
            raise StateError(
                f"string {new_string} already has a note at tick {note['start_tick']}"
            )

    changed = (
        new_string != int(note["string"])
        or new_fret != int(note["fret"])
        or new_pitch != int(note["pitch"])
    )
    note["string"] = new_string
    note["fret"] = new_fret
    note["pitch"] = new_pitch
    note["hammer"] = False
    note["slide"] = 0
    validate_state(state)
    return {"changed": int(changed), "note": dict(note)}


def delete_note(state: dict, track_id: str, note_id: str) -> dict:
    track = get_track(state, track_id)
    if track.get("is_percussion"):
        raise StateError("drum notes cannot be edited in this editor")
    notes = track.get("notes") or []
    for index, note in enumerate(notes):
        if str(note.get("id")) == str(note_id):
            del notes[index]
            validate_state(state)
            return {"deleted": 1, "track": track_id, "note_id": note_id}
    raise StateError(f"unknown note: {note_id}")


def set_track_muted(state: dict, track_id: str, muted: bool) -> dict:
    track = get_track(state, track_id)
    track["muted"] = bool(muted)
    return {"track": track_id, "muted": bool(muted)}


def ir_to_chords(state: dict) -> list[ChordEvent]:
    validate_state(state)
    return chords_from_state(state.get("chords") or [])


def ir_to_tracks(state: dict) -> list[TabTrack]:
    validate_state(state)
    tracks: list[TabTrack] = []
    for entry in state["tracks"]:
        percussion = bool(entry.get("is_percussion"))
        notes = [
            FingeredNote(
                start_tick=int(note["start_tick"]),
                end_tick=int(note["end_tick"]),
                pitch=int(note["pitch"]),
                velocity=int(note["velocity"]),
                string=int(note["string"]),
                fret=int(note["fret"]),
                hammer=bool(note.get("hammer", False)),
                slide=int(note.get("slide", 0)),
            )
            for note in entry.get("notes") or []
        ]
        drum_notes = [
            Note(
                start_tick=int(note["start_tick"]),
                end_tick=int(note["end_tick"]),
                pitch=int(note["pitch"]),
                velocity=int(note["velocity"]),
                stem=str(entry.get("stem", "drums")),
            )
            for note in entry.get("drum_notes") or []
        ]
        tracks.append(
            TabTrack(
                name=str(entry.get("name", "Track")),
                stem=str(entry.get("stem", "other")),
                tuning=[int(value) for value in entry.get("tuning") or []],
                notes=notes,
                is_percussion=percussion,
                channel=int(entry.get("channel", 0)),
                instrument=int(entry.get("instrument", 27)),
                warnings=list(entry.get("warnings") or []),
                drum_notes=drum_notes if percussion else None,
            )
        )
    return tracks


def _note_counts(tracks: list[TabTrack]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for track in tracks:
        stem = track.stem
        count = len(track.drum_notes or []) if track.is_percussion else len(track.notes)
        counts[stem] = counts.get(stem, 0) + count
    return counts


def commit_state(
    state: dict,
    out_dir: Path | str,
    *,
    logger: logging.Logger | None = None,
) -> dict:
    """Regenerate GP5/MIDI from the IR and atomically publish the new revision."""
    validate_state(state)
    out_dir = Path(out_dir)
    tracks = ir_to_tracks(state)
    chords = ir_to_chords(state)
    bpm = float(state["bpm"])
    measure_count = max(1, int(state["measure_count"]))
    meter = parse_time_signature(str(state.get("time_signature") or "4/4"))
    tmp_dir = Path(tempfile.mkdtemp(prefix=".edit-", dir=out_dir))
    tmp_gp5 = tmp_dir / "song.gp5"
    gp5_path = out_dir / "song.gp5"
    midi_paths: dict[str, Path] = {}
    stagged_midi: list[tuple[Path, Path]] = []
    used_stems: set[str] = set()
    try:
        gp5_warnings = write_gp5(
            tracks,
            bpm,
            measure_count,
            tmp_gp5,
            layout=str(state.get("layout", "auto")),
            quantize=int(state.get("quantize", 16)),
            key_name=str((state.get("key") or {}).get("gp_key") or "") or None,
            chords=chords,
            time_signature=meter.label,
        )
        # Per-stem files keep the names the pipeline/UI already expose.
        for track in tracks:
            stem = track.stem if track.stem in STEMS else "other"
            if stem in used_stems:
                continue
            used_stems.add(stem)
            target = out_dir / "midi" / f"{stem}.mid"
            temp = tmp_dir / f"{stem}.mid"
            write_tab_track_midi(track, bpm, temp, time_signature=meter.label)
            midi_paths[stem] = target
            stagged_midi.append((temp, target))
        temp_full = tmp_dir / "full.mid"
        full_midi = out_dir / "midi" / "full.mid"
        write_tab_tracks_midi(
            tracks, bpm, temp_full, chords=chords, time_signature=meter.label
        )

        # Publish files only after every writer has succeeded.
        gp5_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_gp5, gp5_path)
        for temp, target in stagged_midi:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp, target)
        full_midi.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_full, full_midi)

        new_state = copy.deepcopy(state)
        new_state["revision"] = int(state.get("revision", 0)) + 1
        new_state["edited_at"] = datetime.now(timezone.utc).isoformat()
        state_path = out_dir / "tracks.json"
        write_json(state_path, new_state)
        meta = _update_meta(
            out_dir,
            new_state,
            tracks,
            midi_paths,
            full_midi,
        )
        zip_name = str((meta.get("files") or {}).get("zip") or f"audio2gp_{new_state['job_id']}.zip")
        zip_path = zip_directory(out_dir, out_dir / zip_name)
        if logger:
            logger.info(
                "score edit: revision %s -> %s; %d tracks",
                state.get("revision"),
                new_state["revision"],
                len(tracks),
            )
        return {
            "state": new_state,
            "gp5": gp5_path,
            "midi": midi_paths,
            "full_midi": full_midi,
            "zip": zip_path,
            "warnings": list(gp5_warnings),
            "meta": meta,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _update_meta(
    out_dir: Path,
    state: dict,
    tracks: list[TabTrack],
    midi_paths: dict[str, Path],
    full_midi: Path,
) -> dict:
    meta_path = out_dir / "meta.json"
    try:
        meta = read_json(meta_path)
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}
    files = dict(meta.get("files") or {})
    files["tracks"] = "tracks.json"
    files["gp5"] = "song.gp5"
    files["full_midi"] = "midi/full.mid"
    files["midi"] = {
        name: f"midi/{name}.mid" for name in sorted(midi_paths)
    }
    if (out_dir / "chords.json").is_file():
        files["chords"] = "chords.json"
    files.setdefault("meta", "meta.json")
    files.setdefault(
        "zip", f"audio2gp_{state.get('job_id', 'job')}.zip"
    )
    meta.update(
        {
            "status": "ok",
            "revision": int(state["revision"]),
            "edited_at": state.get("edited_at"),
            "tracks": [track.name for track in tracks],
            "note_counts": _note_counts(tracks),
            "files": files,
        }
    )
    meta.pop("gp5_error", None)
    write_json(meta_path, meta)
    return meta
