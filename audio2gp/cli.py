from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import (
    BASS_TUNINGS,
    GUITAR_TUNINGS,
    STEM_NAMES,
    TIME_SIGNATURE_CHOICES,
    Options,
)
from .pipeline import PipelineError, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio2gp",
        description=(
            "Convert a mixed MP3/WAV into Demucs stems, per-stem MIDI, "
            "full.mid, chord labels, and (optionally) an editable Guitar Pro 5 "
            "draft. Use --midi-only to skip GP5 and the score editor. All "
            "processing is local."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="input audio (MP3/WAV/etc.) or MIDI (.mid/.midi); MIDI skips separation and transcription",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="output root")
    parser.add_argument("--work-dir", type=Path, default=Path("work"), help="work root")
    parser.add_argument(
        "--model",
        choices=("htdemucs_ft", "htdemucs"),
        default="htdemucs_ft",
        help="Demucs model (ft is slower and generally better)",
    )
    parser.add_argument(
        "--engine",
        choices=("basic_pitch", "external"),
        default="basic_pitch",
        help=(
            "melodic transcriber; external runs AUDIO2GP_EXTERNAL_CMD per stem "
            "and falls back to Basic Pitch on failure"
        ),
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=(8, 12, 16, 32),
        default=16,
        help="grid: 8=1/8, 12=1/8 triplet, 16=1/16, 32=1/32",
    )
    parser.add_argument(
        "--guitar-stem",
        choices=("other", "vocals"),
        default="other",
        help="stem used for the guitar tab",
    )
    parser.add_argument(
        "--stems",
        default="all",
        metavar="LIST",
        help=(
            "comma-separated source stems to transcribe and output "
            "(vocals,drums,bass,other); default: all. Example for a "
            "guitar+vocal song: --stems other,vocals"
        ),
    )
    parser.add_argument("--max-fret", type=int, default=15, help="guitar/bass max fret (0-24)")
    parser.add_argument(
        "--min-note-sec",
        type=float,
        default=0.03,
        help="drop melodic notes shorter than this",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Demucs device; cpu is extremely slow",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=0.0,
        help="BPM override (20-300); 0 estimates from the audio",
    )
    parser.add_argument(
        "--time-signature",
        choices=TIME_SIGNATURE_CHOICES,
        default="4/4",
        help=(
            "user-selected meter; there is no automatic meter detection. "
            "BPM is always quarter-note BPM. Shuffle is usually notated 12/8 "
            "or 4/4 with --quantize 12 (1/8 triplets)"
        ),
    )
    parser.add_argument(
        "--manual-downbeat",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "time of the first downbeat in the source audio; overrides "
            "automatic analysis and works with every meter (0 = not set; "
            "ignored for MIDI input)"
        ),
    )
    parser.add_argument(
        "--no-align-downbeats",
        action="store_true",
        help=(
            "do not analyze downbeats; keep bar 1 at the audio start"
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "onset", "ties"),
        default="auto",
        help="GP5 rhythm layout; auto tries sustains/ties then falls back",
    )
    parser.add_argument(
        "--no-keep-stems",
        action="store_true",
        help="keep stems only in work/ for playback, not in output/ or the zip",
    )
    parser.add_argument(
        "--no-vocals-track",
        action="store_true",
        help="omit the optional vocals melody tab track in song.gp5",
    )
    parser.add_argument(
        "--guitar-tuning",
        choices=tuple(GUITAR_TUNINGS),
        default="standard",
        help="guitar tuning preset",
    )
    parser.add_argument(
        "--bass-tuning",
        choices=tuple(BASS_TUNINGS),
        default="standard",
        help="bass tuning preset",
    )
    parser.add_argument(
        "--suggest-tuning",
        action="store_true",
        help="try every preset and keep the one that preserves the most pitches",
    )
    parser.add_argument(
        "--ghost-amplitude",
        type=float,
        default=0.15,
        help="suppress low-amplitude ghost notes (0 disables the gate)",
    )
    parser.add_argument(
        "--max-polyphony",
        type=int,
        default=6,
        help="maximum simultaneous notes considered per onset",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Basic Pitch onset threshold",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="Basic Pitch frame threshold",
    )
    parser.add_argument(
        "--no-detect-key",
        action="store_true",
        help="skip key detection; write C major in the GP5",
    )
    parser.add_argument(
        "--no-chords",
        action="store_true",
        help="skip chord analysis (chords.json, MIDI markers, GP5 chord symbols)",
    )
    parser.add_argument(
        "--midi-only",
        action="store_true",
        help=(
            "MIDI-only output: skip fingering, song.gp5, tracks.json, and the "
            "browser score editor; still write MIDI, chords, stems, meta, and ZIP"
        ),
    )
    parser.add_argument(
        "--no-techniques",
        action="store_true",
        help="do not infer hammer-ons, pull-offs, or slides",
    )
    parser.add_argument(
        "--no-tempo-from-drums",
        action="store_true",
        help="keep the mix-level tempo estimate instead of refining from drums",
    )
    parser.add_argument(
        "--no-cache-stems",
        action="store_true",
        help="always run Demucs; do not reuse content-addressed stem caches",
    )
    parser.add_argument("--version", action="version", version=f"audio2gp {__version__}")
    return parser


def _selected_stems(value: str) -> tuple[str, ...]:
    text = (value or "").strip().lower()
    if text in {"", "all"}:
        return STEM_NAMES
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = set(names) - set(STEM_NAMES)
    if unknown:
        raise ValueError(
            "unknown --stems values: "
            + ", ".join(sorted(unknown))
            + f"; choose from {', '.join(STEM_NAMES)}"
        )
    return names


def _options_from_args(args: argparse.Namespace) -> Options:
    return Options(
        output_dir=args.output,
        work_dir=args.work_dir,
        demucs_model=args.model,
        transcription_engine=args.engine,
        quantize=args.quantize,
        guitar_stem=args.guitar_stem,
        keep_stems=not args.no_keep_stems,
        max_fret=args.max_fret,
        min_note_sec=args.min_note_sec,
        device=args.device,
        include_vocals=not args.no_vocals_track,
        bpm_override=args.bpm if args.bpm and args.bpm > 0 else None,
        time_signature=args.time_signature,
        manual_downbeat_s=(
            args.manual_downbeat if args.manual_downbeat and args.manual_downbeat > 0 else None
        ),
        align_downbeats=not args.no_align_downbeats,
        selected_stems=_selected_stems(args.stems),
        layout=args.layout,
        guitar_tuning=args.guitar_tuning,
        bass_tuning=args.bass_tuning,
        suggest_tuning=args.suggest_tuning,
        ghost_amplitude=args.ghost_amplitude,
        max_polyphony=args.max_polyphony,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        detect_key=not args.no_detect_key,
        detect_chords=not args.no_chords,
        export_gp5=not args.midi_only,
        detect_techniques=not args.no_techniques,
        tempo_from_drums=not args.no_tempo_from_drums,
        cache_stems=not args.no_cache_stems,
    )


def _progress_printer():
    last_pct = [-10.0]
    last_stage = [""]

    def callback(stage: str, pct: float, message: str) -> None:
        if stage != last_stage[0] or pct - last_pct[0] >= 2.0 or pct >= 100.0:
            print(f"[{pct:5.1f}%] {stage}: {message}", file=sys.stderr, flush=True)
            last_pct[0] = pct
            last_stage[0] = stage

    return callback


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = _options_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.input.is_file():
        print(f"error: input file does not exist: {args.input}", file=sys.stderr)
        return 1

    try:
        result = run(args.input, options, progress_cb=_progress_printer())
    except PipelineError as exc:
        print(f"error [{exc.stage}]: {exc.message}", file=sys.stderr)
        print(f"work: {exc.work_dir}", file=sys.stderr)
        print(f"log:  {exc.log_path}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    print()
    print(f"status: {result.status}")
    print(f"output format: {result.output_format}")
    print(f"bpm: {result.bpm:.2f}")
    rhythm = result.rhythm or {}
    print(f"time signature: {rhythm.get('time_signature', '4/4')}")
    if rhythm.get("mode") == "manual":
        print(
            "downbeats: manual "
            f"first={rhythm.get('first_downbeat_s')}s "
            f"offset={rhythm.get('grid_offset_s')}s"
        )
    elif rhythm.get("enabled") and rhythm.get("mode") == "off":
        print("downbeats: alignment off")
    elif rhythm.get("enabled") and rhythm.get("method") in {"madmom-infer", "librosa-accent"}:
        print(
            "downbeats: "
            f"{rhythm.get('method')} first={rhythm.get('first_downbeat_s')}s "
            f"offset={rhythm.get('grid_offset_s')}s"
        )
    else:
        print("downbeats: not aligned (bar 1 starts at the audio start)")
    print(f"device: {result.device}")
    print(f"selected stems: {', '.join(result.meta.get('selected_stems', []))}")
    print(f"note counts: {result.note_counts}")
    print(f"output: {result.out_dir}")
    for name in result.stems:
        print(f"stem {name}: {result.stems[name]}")
    for name, path in result.midi.items():
        print(f"midi {name}: {path}")
    print(f"midi full: {result.full_midi}")
    if result.output_format == "midi_only":
        print("gp5: (not requested; MIDI-only mode)")
    else:
        print(f"gp5: {result.gp5 if result.gp5 else '(failed; partial success)'}")
    if result.chords_path is not None:
        print(f"chords: {result.chords_path}")
        labels = [chord.label for chord in result.chords]
        if labels:
            print(f"chord progression: {' - '.join(labels)}")
        else:
            print("chord progression: (none detected)")
    print(f"zip: {result.zip_path}")
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
