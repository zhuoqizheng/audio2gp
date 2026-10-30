# audio2gp

**English** | [简体中文](README.zh-CN.md)

> **Local-first audio → stems → MIDI / multitrack Guitar Pro draft**
>
> Turn a mixed song into editable guitar, bass, and drum tabs — without uploading audio.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Local](https://img.shields.io/badge/Audio-never_uploaded-success)

audio2gp separates a mixed song (MP3 / WAV / FLAC / M4A / OGG) into `vocals`, `drums`, `bass`, and `other`, transcribes it to MIDI, estimates the chord progression, assigns playable guitar/bass string-and-fret positions, and exports a multitrack `.gp5`, `tracks.json`, and per-stem MIDI files. The result can be previewed, played, and lightly edited in the browser with alphaTab, after which the server regenerates the score.

**All audio processing stays on your machine.** There are no cloud inference APIs, no LLM-generated notation, no audio uploads, and no scraping of commercial tab sites. The FastAPI server binds to `127.0.0.1` only.

> The output is an **editable draft**, not a professional transcription. Drum parts, chords, and high-position fingerings may need manual correction.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Time signatures and shuffle](#time-signatures-and-shuffle)
- [Output files](#output-files)
- [Browser score editor](#browser-score-editor)
- [HTTP API](#http-api)
- [Architecture](#architecture)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Privacy and license](#privacy-and-license)
- [References](#references)

## Features

- **Local stem separation**: Demucs `htdemucs_ft` (cleaner, slower) or `htdemucs`. The four-stem result is content-addressed and cached, so changing the selected stems does not rerun Demucs.
- **Melodic transcription**: Basic Pitch 0.4.0 with its TFLite model on CPU; ghost-note filtering, short-note filtering, a polyphony cap, and forced monophony for vocals and bass.
- **Drum track**: heuristic onset and spectral-band classification (kick, snare, closed/open hi-hat, crash, toms), plus drums-first tempo correction with half/double-time disambiguation.
- **MIDI input**: upload a `.mid`/`.midi` file to skip Demucs, transcription, key detection, and audio chord analysis. Tracks are mapped to drums/bass/vocals/other by channel 9, track name, and GM program, then quantized and fingered like an audio job.
- **User-selected meter**: 4/4, 3/4, 2/4, 6/8, 9/8, or 12/8. There is no automatic meter detection. Automatic downbeat analysis runs for every meter: madmom for 4/4 and a librosa accent-phase fallback for the others; a manual first downbeat always wins.
- **Chord analysis**: beat-synchronous chroma, harmonic suppression, and Viterbi smoothing; results go to `chords.json`, GP5 chord symbols, and MIDI markers in `full.mid`.
- **Playable fingering**: candidate position enumeration plus a Viterbi search with hand-position, string-change, and large-jump costs; standard, Drop D, and half-step-down tunings, with optional tuning suggestions.
- **Conservative techniques**: only high-confidence hammer-ons, pull-offs, and slides are marked. Bends, vibrato, palm mutes, and harmonics are not guessed.
- **Multitrack GP5**: Guitar, Bass, Drums, and an optional Vocals track. Time signatures, bar lines, rests, ties, percussion, and key signatures all follow the selected meter.
- **Two output modes**: `gp5_midi` (default; GP5 plus the editor) or `midi_only` (skip fingering, GP5, and the editor while still delivering MIDI, stems, chords, metadata, and a ZIP).
- **Browser editing**: alphaTab preview and playback; click a note to change its string/fret, delete it, mute or solo a track, transpose a whole track by ±1/±12 semitones, undo the last edit, or flag a note for manual review. Every note edit carries a revision; stale writes receive HTTP 409. Flags and solo do not consume the undo step.
- **Pluggable transcription**: Basic Pitch by default, or a local external engine through `AUDIO2GP_EXTERNAL_CMD` (per-stem command → MIDI, with automatic Basic Pitch fallback).
- **Single-machine UI**: FastAPI + static HTML with no frontend build step. Completed jobs are re-registered from `output/*/meta.json` at startup, so `?job=` links survive a restart. A legacy Gradio entry point remains available for audio input.

## Quick start

### Requirements

- **Linux** (developed on Ubuntu 22.04)
- **Python 3.10**: Basic Pitch 0.4.0 currently ships its `tflite-runtime` wheel for Linux / Python 3.10 only; `scripts/setup_env.sh` rejects other versions
- `ffmpeg` and `ffprobe` on `PATH` (Debian/Ubuntu: `sudo apt install ffmpeg`)
- An NVIDIA GPU is recommended; CPU works but is significantly slower for both Demucs and Basic Pitch
- A modern browser for the alphaTab score view

Basic Pitch runs on CPU via TFLite, and the madmom downbeat model also runs on CPU. The GPU mainly accelerates Demucs.

### Installation

`scripts/setup_env.sh` reuses a **torch/torchaudio installation already present in the user site** and prevents pip from downloading a second CPU-only torch. First install a torch build matching your CUDA environment in the user site. This is the combination tested on the development machine:

```bash
python3.10 -m pip install --user \
  torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

> A CPU-only torch build also works, but processing a full song will be very slow. Do not install torch or Demucs through a plain `pip install -r requirements.txt`; the setup script handles Demucs as an isolated install.

Then, from the repository root:

```bash
git clone https://github.com/zhuoqizheng/audio2gp.git
cd audio2gp
scripts/setup_env.sh
```

The first setup downloads the Demucs and madmom downbeat weights. Basic Pitch ships its model in the wheel, so there is no separate download for it.

### Start the web UI

```bash
.venv/bin/audio2gp-web
# open http://127.0.0.1:7860
```

The server binds to loopback only and rejects public addresses. It has no share or tunnel mode.

### Command line

```bash
.venv/bin/audio2gp song.mp3 -o output
.venv/bin/audio2gp song.mp3 -o output --stems other,vocals
.venv/bin/audio2gp song.mp3 -o output --time-signature 12/8 \
  --manual-downbeat 0.85 --quantize 8
.venv/bin/audio2gp song.mp3 -o output --midi-only
.venv/bin/audio2gp song.mid -o output --stems bass,other
AUDIO2GP_EXTERNAL_CMD='my-engine {wav} {out} --bpm {bpm}' \
  .venv/bin/audio2gp song.mp3 -o output --engine external
```

Show every option:

```bash
.venv/bin/audio2gp --help
```

The most useful options:

| Option | Description |
|---|---|
| `--time-signature {4/4,3/4,2/4,6/8,9/8,12/8}` | Fixed meter; the user must choose it, as there is no automatic meter detection |
| `--manual-downbeat SECONDS` | Time of the first downbeat in the source audio; works for every meter. `0` means “not set” |
| `--quantize {8,12,16,32}` | Quantization grid: 1/8, 1/8 triplet, 1/16, 1/32. Defined by note value, independent of meter |
| `--stems LIST` | Stems to transcribe and deliver, e.g. `other,vocals`; Demucs still separates all four. For MIDI input, only stems present in the file are exported |
| `--engine {basic_pitch,external}` | Melodic transcription engine. `external` requires `AUDIO2GP_EXTERNAL_CMD`; failure falls back to Basic Pitch with a warning |
| `--midi-only` | Export MIDI only; skip fingering, GP5, `tracks.json`, and the editor |
| `--no-align-downbeats` | Disable automatic downbeat analysis; a manual downbeat still applies. MIDI input always uses the file timeline |
| `--bpm 0` | `0` = estimate automatically; a value from 20 to 300 overrides the BPM (quarter-note BPM) |
| `--no-chords` | Disable chord analysis and chord artifacts |

For both the web UI and CLI, the recommended workflow is to listen to the song, determine the meter and the first downbeat, and then upload it. This gives the most reliable bar lines.

## Usage

1. **Upload** a full mix or a MIDI file. The default limit is 200 MB / 600 seconds, adjustable through CLI flags or environment variables.
2. **Select stems.** Choose only what you need; for a guitar-and-vocal song, select `other` + `vocals`.
3. **Set the meter** (and optionally a manual downbeat, BPM, quantization, and tuning). Defaults are `4/4`, automatic BPM estimation, and a 1/16 grid.
4. **Start conversion.** Audio stages are: normalization → rhythm/downbeats → Demucs separation → Basic Pitch/drum transcription → cleanup and quantization → chords → fingering → GP5/MIDI → editor state. MIDI input skips directly from file import to cleanup/quantization (its first tempo event is used; the page meter governs bar lines).
5. Preview the stems, inspect the chord timeline, open the alphaTab score, and download the files.
6. To adjust the score, click notes to change their string/fret or delete them, or use the mixer to mute, solo, or transpose a track. Every note edit regenerates GP5/MIDI/ZIP; the toolbar can undo one edit, and flagged notes are saved to `flags.json`.

If GP5 generation fails in GP5 mode, the job is marked `partial` and the generated MIDI, stems, and ZIP are retained. A MIDI-only job is not considered failed merely because no GP5 exists.

## Time signatures and shuffle

**Shuffle/swing is a feel, not a time signature.** Choose the meter using the way the song is actually counted:

| Musical feel | Time signature | Quantization grid |
|---|---|---|
| Slow blues, or the band counts in 12/8 | **12/8** | **1/8** (12 eighth notes per bar) |
| Counted in 4/4 with swung eighth notes (shuffle/swing) | **4/4** | **1/8 triplet** (quantize value 12) |
| A slow song counted in 6/8 | **6/8** | **1/8** |

Key points:

- A 12/8 bar has four dotted-quarter pulses, each divided into three eighth notes. This is the cleanest notation for a slow shuffle or blues.
- In 4/4 swing, eighth notes are played in pairs at roughly a 2:1 long–short ratio. Use the 1/8 triplet grid to preserve the swing.
- Do not transcribe a shuffle with a straight 1/8 or 1/16 grid; the swing will be rounded into straight or dotted rhythms.
- BPM is always **quarter-note BPM** here. The pulse in 12/8 is a dotted quarter (1440 ticks), so the displayed BPM does not necessarily match the common “dotted-quarter BPM” convention.

### Bar alignment

- **4/4** uses madmom downbeat detection by default. Other meters use a librosa beat tracker plus a meter-aware low-frequency accent score to choose the downbeat phase. If analysis fails, bar 1 starts at the audio origin with a warning.
- If the song starts with a pickup, the first downbeat is placed at bar 2 or later, and the pickup material occupies complete leading bars instead of a short anacrusis measure.
- Compound meters use their pulse grouping (6/8 = 2 pulses, 9/8 = 3, 12/8 = 4). The librosa path infers the beat count per bar from the detected median beat interval; if the tracker locks onto dotted-quarter pulses instead of eighths, the phase may be offset — supply a **manual first downbeat (seconds)** to correct it.
- A manual downbeat takes precedence over automatic analysis and still applies when “align to detected downbeats” is disabled.
- MIDI input skips downbeat analysis entirely and uses the file’s own note times; the selected meter is still used to split bars and write the time signature.

## Output files

```text
output/<job_id>/
  stems/<selected stems>.wav   # when keep_stems=true
  midi/<selected stem>.mid
  midi/full.mid
  midi/reference.mid           # untouched transcription snapshot for comparison
  song.gp5                     # GP5 mode; may be absent if generation failed
  tracks.json                  # GP5 mode; editor IR
  flags.json                   # notes flagged for manual review (when any)
  chords.json                  # when audio chord analysis ran
  meta.json
  audio2gp_<job_id>.zip
work/<job_id>/
  song.wav                     # audio input: 44.1 kHz / stereo / PCM s16
  stems/                       # all four stems (still used for UI playback if not kept)
  pipeline.log
work/.stem_cache/<hash>/       # Demucs cache, never included in the ZIP
```

For MIDI input there are no WAV stems and no Demucs cache entry; `midi/reference.mid` is a copy of the uploaded file.

- `stems/` contains only the selected stems. Unselected stems remain in `work/` for diagnostics.
- `chords.json` contains chord segments (start/end ticks and times, label, root, quality, confidence, bar, beat) and a summary.
- `tracks.json` is the stable editor intermediate representation (IR), storing the revision, note IDs, string/fret positions, chords, and mute state.
- `flags.json` stores user-flagged notes (track/note ID, optional reason, timestamp). Flags do not bump the revision; deleting a note removes its flag.
- `meta.json` records status, BPM, meter, bar length, options, warnings, timings, the file list, and any error. On startup the API rebuilds completed jobs from these files.
- Jobs are not cleaned up automatically; `output/` and `work/` are yours to manage. A full song can use hundreds of MB.

## Browser score editor

- alphaTab renders and plays the score in the browser. alphaTab, the Bravura font, and the Sonivox SoundFont are vendored and work offline.
- Click a note → the right panel lists other `(string, fret)` positions that preserve its pitch; positions already occupied by another note at the same onset are disabled.
- **Apply** a change, **delete a note**, **mute/solo** a track, or **transpose** a track by −12/−1/+1/+12 semitones (notes that cannot fit the fretboard are dropped with a warning). Every note edit carries a revision. If two tabs edit at once, the stale tab receives HTTP 409 and must reload.
- **Undo** restores the previous `tracks.json` and regenerates GP5/MIDI once. Note edits, transposes, and mute changes create an undo step; solo and flags do not.
- **Flags** mark a note for later review with an optional reason (200 chars). They are stored in `flags.json`, do not block editing, and appear in the download list.
- The browser **never writes GP5 directly**. Edit requests only modify the server-side `tracks.json`; the same PyGuitarPro/MIDI export path then atomically rebuilds `song.gp5`, MIDI, `meta.json`, and the ZIP.
- MIDI-only results have no score or editor, but still provide the chord timeline and all MIDI downloads.

## HTTP API

The web UI is a client of this API:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Upload `file` + JSON `options`; returns 202 and a job ID |
| `GET` | `/api/jobs/{id}` | Status, progress, log, and completed result |
| `GET` | `/api/jobs/{id}/state` | Read the editor IR (GP5 mode only) |
| `GET` | `/api/jobs/{id}/score.gp5` | Current GP5 file |
| `GET` | `/api/jobs/{id}/audio/{stem}` | Stream a stem for preview |
| `GET` | `/api/jobs/{id}/files/{path}` | Whitelisted file downloads |
| `PATCH` | `/api/jobs/{id}/tracks/{tid}/notes/{nid}` | Change a note’s string/fret |
| `DELETE` | `/api/jobs/{id}/tracks/{tid}/notes/{nid}` | Delete a note |
| `PUT` | `/api/jobs/{id}/tracks/{tid}/mute` | Set track mute state |
| `POST` | `/api/jobs/{id}/tracks/{tid}/transpose` | Transpose a track by −24…+24 semitones |
| `PUT` | `/api/jobs/{id}/tracks/{tid}/notes/{nid}/flag` | Flag/unflag a note with an optional reason |
| `POST` | `/api/jobs/{id}/undo` | Restore the previous score revision (one step) |
| `GET` | `/api/health`, `/api/limits` | Health check and UI limits |

Key `POST /api/jobs` options:

```json
{
  "time_signature": "12/8",
  "manual_downbeat_s": 0.85,
  "align_downbeats": true,
  "quantize": 8,
  "selected_stems": ["other", "vocals"],
  "detect_chords": true,
  "export_gp5": true
}
```

Results additionally include `time_signature`, `bar_ticks`, `manual_downbeat_s`, chord segments, `output_format`, `input_kind`, flags, and undo availability. `/api/limits` advertises the available meters, `manual_downbeat` support, all `auto_downbeat_time_signatures`, transcription engines, and whether an external engine is configured. Completed jobs are restored from `output/<job_id>/meta.json` at startup (newest 200), so `?job=` links work after a restart.

## Architecture

```text
uploaded audio                         uploaded .mid/.midi
  → ffmpeg normalization + BPM          → track → stem mapping (GM/channel/name)
  → user meter / downbeat analysis      → first tempo event, file timeline
  → Demucs four-stem separation         → (separation/transcription skipped)
  → Basic Pitch/external engine + drums
  → onset clustering, quantization, ghost-note cleanup, monophony
  → chord analysis (chroma + Viterbi)   → (audio analysis skipped)
  → hand-position candidates + DP fingering (GP5 mode)
  → GP5 / per-stem MIDI / full.mid / reference.mid / tracks.json
  → FastAPI + alphaTab editing, undo, transpose, flags, re-export
```

Key design decisions:

- **Batch processing and editing are decoupled**: `pipeline.py` produces the draft and IR; `score_ir.py` only reads/mutates the IR and calls the same export functions.
- **Single GPU worker**: the API runs jobs serially to avoid multiple Demucs instances competing for VRAM.
- **Content-addressed stem cache**: the key includes the normalized WAV hash, model, and separation parameters; changing the stem selection only reruns transcription.
- **MIDI-only is a separate path**: it does not build editor state, run fingering, or produce `song.gp5` / `tracks.json`.
- **MIDI input reuses the same post-processing**: only import/mapping replaces the audio front end; quantization, chords (empty), fingering, GP5, editor state, and exports are unchanged.
- **Writes are published atomically**: temporary files are written first, then `os.replace` publishes them; a failure does not destroy the previous score.
- **External transcription is a subprocess contract**: the command template is shell-split and executed without a shell, once per melodic stem; placeholders `{wav}`, `{stem}`, `{out}`, `{bpm}`, `{device}` are substituted, and a timeout (`AUDIO2GP_EXTERNAL_TIMEOUT_S`, default 600 s) bounds each call.

For the complete stage descriptions, module signatures, and algorithm details, see [`system_design.md`](system_design.md).

## Tests

```bash
# Fast unit/integration tests (no real models)
.venv/bin/python -m pytest -q

# Real Demucs / Basic Pitch end-to-end tests (downloads weights on first run)
.venv/bin/python -m pytest -q -m slow
```

Current results on the development machine: **166 passed** in the fast suite and **3 passed** in the slow suite. The slow tests cover real separation, transcription, GP5 round-trip, ZIP output, and the legacy Gradio smoke test.

GP5 acceptance is a PyGuitarPro parse plus structural checks: track count, two voices, every voice-0 measure summing to the selected meter’s bar length, ties, percussion channels, and time-signature headers. There is no Guitar Pro/TuxGuitar on this machine, so open generated files in a score editor for a final visual review.

### Quality evaluation

`scripts/eval_transcription.py` scores per-stem MIDI against reference MIDI (precision, recall, F1, pitch-only F1, octave errors, leak rate) and can fail below a micro-F1 threshold:

```bash
.venv/bin/python scripts/eval_transcription.py \
  --reference eval/golden/song-a --prediction output/<job_id> --fail-under 0.60
```

`scripts/six_stem_experiment.py` tests whether `htdemucs_6s`'s dedicated guitar stem beats `other` for one song. Both are experiments and do not change product outputs. See [`docs/eval.md`](docs/eval.md).

## Known limitations

- **No automatic meter detection**: each job uses one user-selected meter; there is no mid-song meter change or tempo map.
- **Automatic downbeat analysis for non-4/4 meters is heuristic**: the librosa path infers pulse grouping from the beat interval and can lock onto the wrong phase on free or heavily swung material. Use a manual first downbeat when it does.
- BPM is always quarter-note BPM; compound-meter display tempo may differ from the usual “dotted-quarter BPM” convention.
- Basic Pitch is the only built-in melodic model; an external command engine can be plugged in, but drum transcription remains heuristic. Toms and cymbals are especially approximate.
- Fingering is a playability approximation. Difficult chords may be octave-folded, reduced, or skipped (with warnings).
- Chords estimate only root + quality; inversions, slash chords, voicings, and chord diagrams are not recognized. Power chords, extended harmonies, and distorted mixes may be simplified.
- Techniques are limited to conservative hammer-on/pull-off and slide detection; bends, vibrato, palm mutes, harmonics, and pitch-bend lanes are not supported.
- Only GP5 (5.1) is written, not GPX/GP7/MusicXML. The editor supports string/fret changes, deletion, mute, solo, one-step undo, whole-track transpose, and note flags; there is no multi-step undo or lead/rhythm split.
- MIDI input uses a single fixed tempo (the first tempo event) and the page-selected meter; tempo maps and in-file meter changes are not preserved. Track mapping is heuristic (channel 9 → drums, names and GM programs → the other stems).
- Completed jobs are restored from `output/*/meta.json` on startup (newest 200). Jobs that were running during a restart are lost, and output files are never cleaned up automatically.
- Basic Pitch runs on CPU, so full-song processing is the main bottleneck; the first run also needs to download model weights.
- Output is a draft and must not be represented as professional-grade notation.

## Privacy and license

### Privacy

- Audio is processed locally and is never uploaded to a remote service. There are no cloud inference APIs and no LLM.
- The web server binds to `127.0.0.1` only, rejects non-loopback addresses, and offers no `--share` or public tunnel.
- The first installation downloads Demucs/madmom model weights from upstream; this is a model download, not an audio upload.
- The alphaTab, font, and SoundFont assets used by the result page are vendored in the repository and work offline.
- The optional external transcription engine runs a command you configure locally. Whether that command sends audio anywhere is outside this app’s control; inspect third-party engines before use.

### License

The project code is released under the **MIT** license; see [`LICENSE`](LICENSE) and `pyproject.toml`.

| Component | License / notes |
|---|---|
| audio2gp | MIT |
| Demucs | MIT; pretrained weights follow the upstream release terms |
| Basic Pitch | Apache-2.0; ICASSP 2022 TFLite model bundled |
| madmom-infer | Code BSD-2-Clause; **downbeat weights CC BY-NC-SA 4.0 (non-commercial, share-alike)** |
| PyGuitarPro | LGPL-3.0 |
| FastAPI / uvicorn | MIT / BSD |
| alphaTab (vendored) | MPL-2.0; license file included with the assets |
| Bravura font (vendored) | SIL OFL 1.1 |
| Sonivox SoundFont (vendored) | Apache-2.0 |
| Gradio (legacy UI) | Apache-2.0 |

**Commercial use note**: the default downbeat alignment loads madmom’s CC BY-NC-SA weights. For commercial use, disable downbeat alignment (`--no-align-downbeats` or the UI switch) or replace it with a commercially usable downbeat model; when disabled, no madmom weights are loaded.

Guitar Pro is a trademark of Arobas Music. This project does not embed Guitar Pro and is not affiliated with it; score preview uses the MPL-2.0 alphaTab library, and the server writes GP5 with PyGuitarPro.

## References

- [system_design.md](system_design.md): full algorithm, stages, module interfaces, and API design
- [TabForge](https://github.com/rayBastard/tabforge): architecture and engineering reference (no code copied)
- [Demucs](https://github.com/facebookresearch/demucs)
- [Basic Pitch](https://github.com/spotify/basic-pitch)
- [PyGuitarPro](https://github.com/Perlence/PyGuitarPro)
- [alphaTab](https://www.alphatab.net/)
- [madmom](https://github.com/CPJKU/madmom)
