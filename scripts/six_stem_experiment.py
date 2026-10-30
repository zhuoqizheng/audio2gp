#!/usr/bin/env python3
"""Experiment: is htdemucs_6s's dedicated guitar stem better than "other"?

The product pipeline uses htdemucs_ft/htdemucs (four stems, guitar lives in
"other"). This script separates one song with htdemucs_6s, transcribes both the
"guitar" and "other" stems with the same Basic Pitch settings, and scores each
prediction against a reference MIDI file (for example a hand-checked guitar
line). It is an experiment only: no product code path changes.

Example:
    python scripts/six_stem_experiment.py song.wav \
        --reference eval/golden/song/guitar.mid --out-dir eval/six_stem/song
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audio2gp import midi_post, transcribe
from audio2gp.config import Options
from audio2gp.evaluation import evaluate_notes, load_midi_notes
from audio2gp.types import Note, NoteEvent
from audio2gp.utils import ensure_dir, run_checked, write_json

CANDIDATES = ("guitar", "other")
# MIDI ticks are only a transport here; scoring converts notes back to seconds.
WRITE_BPM = 120.0


def events_to_notes(events: list[NoteEvent], stem: str, bpm: float = WRITE_BPM) -> list[Note]:
    scale = float(bpm) / 60.0 * 960.0
    notes: list[Note] = []
    for event in events:
        start = round(event.start_s * scale)
        end = max(start + 1, round(event.end_s * scale))
        notes.append(
            Note(
                start_tick=start,
                end_tick=end,
                pitch=int(event.pitch),
                velocity=max(1, min(127, round(event.amplitude * 127))),
                stem=stem,
            )
        )
    return notes


def separate_six(input_path: Path, out_dir: Path, device: str) -> dict[str, Path]:
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs_6s",
        "--shifts",
        "1",
        "--overlap",
        "0.25",
        "-o",
        str(out_dir),
    ]
    if device in {"cpu", "cuda"}:
        cmd += ["-d", device]
    run_checked(cmd + [str(input_path)])
    track_dir = out_dir / "htdemucs_6s" / input_path.stem
    stems = {name: track_dir / f"{name}.wav" for name in CANDIDATES}
    missing = [name for name, path in stems.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"htdemucs_6s did not produce: {', '.join(missing)}")
    return stems


def evaluate_candidate(
    stem: str,
    wav_path: Path,
    out_dir: Path,
    options: Options,
    reference: Path | None,
) -> dict:
    # transcribe_stems only knows product stems; the 6s guitar stem uses the
    # same melodic settings as "other".
    events, warnings = transcribe.transcribe_stems({"other": wav_path}, options, WRITE_BPM)
    notes = events_to_notes(events.get("other", []), stem)
    midi_path = out_dir / f"{stem}.mid"
    midi_post.write_stem_midi(
        notes,
        WRITE_BPM,
        midi_path,
        program=25,
        channel=0,
        name=f"6s {stem}",
    )
    entry: dict = {
        "stem": stem,
        "wav": str(wav_path),
        "midi": str(midi_path),
        "events": len(events.get("other", [])),
        "notes": len(notes),
        "warnings": warnings,
    }
    if reference is not None:
        predicted = load_midi_notes(midi_path)
        truth = load_midi_notes(reference)
        metrics = evaluate_notes(predicted, truth)
        entry["metrics"] = {
            key: metrics[key]
            for key in (
                "truth_notes",
                "predicted_notes",
                "matched",
                "precision",
                "recall",
                "f1",
                "pitch_only",
                "octave_error_rate",
                "leak_rate",
            )
        }
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, help="input mix")
    parser.add_argument("--reference", type=Path, help="reference MIDI for the guitar line")
    parser.add_argument("--out-dir", type=Path, required=True, help="experiment output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--stems", default=",".join(CANDIDATES),
        help="comma-separated candidate stems (default: guitar,other)",
    )
    args = parser.parse_args(argv)
    if not args.audio.is_file():
        parser.error(f"audio not found: {args.audio}")
    if args.reference is not None and not args.reference.is_file():
        parser.error(f"reference MIDI not found: {args.reference}")
    candidates = tuple(name.strip() for name in args.stems.split(",") if name.strip())
    unknown = set(candidates) - set(CANDIDATES)
    if unknown:
        parser.error(f"unknown stems: {', '.join(sorted(unknown))}")

    out_dir = ensure_dir(args.out_dir)
    stems = separate_six(args.audio, out_dir, args.device)
    options = Options()
    results = [
        evaluate_candidate(name, stems[name], out_dir, options, args.reference)
        for name in candidates
    ]
    report = {
        "input": str(args.audio),
        "separator": "htdemucs_6s",
        "reference": str(args.reference) if args.reference else None,
        "candidates": results,
    }
    write_json(out_dir / "report.json", report)

    print("| stem | events | notes | truth | matched | P | R | F1 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for entry in results:
        metrics = entry.get("metrics") or {}
        print(
            "| {stem} | {events} | {notes} | {truth} | {matched} | {p} | {r} | {f1} |".format(
                stem=entry["stem"],
                events=entry["events"],
                notes=entry["notes"],
                truth=metrics.get("truth_notes", "—"),
                matched=metrics.get("matched", "—"),
                p=metrics.get("precision", "—"),
                r=metrics.get("recall", "—"),
                f1=metrics.get("f1", "—"),
            )
        )
    print(f"\nWrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
