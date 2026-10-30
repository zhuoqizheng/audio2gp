# Transcription evaluation

Two scripts measure transcription quality against reference MIDI. They never
modify product code or outputs.

## Per-stem scoring

`audio2gp/evaluation.py` matches predicted notes to reference notes using onset
tolerance and pitch, with optional octave folding, and reports:

- **precision / recall / F1** — strict onset + pitch matches.
- **pitch-only F1** — ignores timing; useful for checking note choice.
- **octave error rate** — matched pitch classes one or more octaves away.
- **leak rate** — predicted notes with no reference counterpart (false positives).

Predictions and references are directories of `<stem>.mid` files. A job output
directory is also accepted; its `midi/` subdirectory is detected automatically.
Only stems present on both sides are scored.

```bash
.venv/bin/python scripts/eval_transcription.py \
  --reference eval/golden/song-a \
  --prediction output/<job_id> \
  --json eval/report.json --fail-under 0.60
```

Reference layout:

```text
eval/golden/song-a/
  other.mid
  bass.mid
  drums.mid
```

`--stems other,bass` restricts scoring. `--tolerance` changes the onset window
(default 0.08 s). `--fail-under` exits non-zero below the requested micro F1,
which makes the script usable in a local regression loop.

## Six-stem experiment

The product separates with `htdemucs_ft`/`htdemucs` and treats guitar as part
of `other`. `scripts/six_stem_experiment.py` tests whether `htdemucs_6s`'s
dedicated `guitar` stem transcribes better than `other` for one song:

```bash
.venv/bin/python scripts/six_stem_experiment.py song.wav \
  --reference eval/golden/song/guitar.mid \
  --out-dir eval/six_stem/song
```

It runs Demucs 6s once, transcribes both candidate stems with Basic Pitch, writes
`htdemucs_6s/<stem>.mid`, scores each against the reference, and writes
`report.json` plus a markdown table. This is an experiment only; neither the
separation model nor the stem mapping in the product pipeline changes.

## Limitations

- Reference MIDI is assumed correct; hand-check ambiguous parts first.
- Strict matching does not model sustained-note segmentation or repeated
  onsets, so read precision/recall alongside a listening pass.
- The six-stem experiment uses one song per run; conclusions need multiple
  songs and consistent Basic Pitch settings.
