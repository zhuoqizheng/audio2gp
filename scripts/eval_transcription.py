#!/usr/bin/env python3
"""Compare per-stem MIDI output with reference MIDI truth.

Reference layout (one file per stem):

    eval/golden/song-a/other.mid
    eval/golden/song-a/bass.mid
    eval/golden/song-a/drums.mid

Prediction layout: a job output directory (uses its midi/ subdirectory) or a
directory of <stem>.mid files. Only stems present in both sides are scored.

Example:
    python scripts/eval_transcription.py \
        --reference eval/golden/song-a \
        --prediction output/20261030-120000_song-a_ab12cd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audio2gp.evaluation import evaluate_dirs


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _table(report: dict) -> str:
    lines = [
        "| stem | truth | pred | matched | P | R | F1 | pitch-only F1 | octave err | leak |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stem, row in report["stems"].items():
        lines.append(
            "| {stem} | {truth} | {pred} | {matched} | {p} | {r} | {f1} | "
            "{pf1} | {octave} | {leak} |".format(
                stem=stem,
                truth=row["truth_notes"],
                pred=row["predicted_notes"],
                matched=row["matched"],
                p=_fmt(row["precision"]),
                r=_fmt(row["recall"]),
                f1=_fmt(row["f1"]),
                pf1=_fmt(row["pitch_only"]["f1"]),
                octave=_fmt(row["octave_error_rate"]),
                leak=_fmt(row["leak_rate"]),
            )
        )
    micro = report["micro"]
    lines.append(
        "| **micro** | {truth} | {pred} | {matched} | {p} | {r} | {f1} | — | — | — |".format(
            truth=micro["truth_notes"],
            pred=micro["predicted_notes"],
            matched=micro["matched"],
            p=_fmt(micro["precision"]),
            r=_fmt(micro["recall"]),
            f1=_fmt(micro["f1"]),
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score predicted per-stem MIDI against reference MIDI."
    )
    parser.add_argument("--reference", required=True, type=Path, help="reference MIDI directory")
    parser.add_argument(
        "--prediction",
        required=True,
        type=Path,
        help="job output directory (midi/ is auto-detected) or a per-stem MIDI directory",
    )
    parser.add_argument(
        "--stems",
        default="",
        metavar="LIST",
        help="comma-separated stems to score (default: every stem on both sides)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.08,
        help="onset tolerance in seconds for strict matching (default: 0.08)",
    )
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit 1 when micro F1 is below this value",
    )
    args = parser.parse_args(argv)

    stems = [part.strip() for part in args.stems.split(",") if part.strip()]
    report = evaluate_dirs(
        args.prediction,
        args.reference,
        stems=stems or None,
        tolerance_s=args.tolerance,
    )
    if not report["stems"]:
        print(
            "error: no stem MIDI files found on both sides; "
            "check --reference and --prediction",
            file=sys.stderr,
        )
        return 1
    report["reference"] = str(args.reference)
    report["prediction"] = str(args.prediction)
    print(_table(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON: {args.json}")
    if args.fail_under is not None and report["micro"]["f1"] < args.fail_under:
        print(
            f"\nFAIL: micro F1 {report['micro']['f1']:.3f} < {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
