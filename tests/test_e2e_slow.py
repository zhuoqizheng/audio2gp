from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import guitarpro as gp
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from audio2gp import pipeline
from audio2gp.config import STEM_NAMES, TICKS_PER_BAR, Options
from audio2gp.gp_writer import parse_gp5

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path.home() / "Downloads" / "demo.mp3"


def _make_synthetic_mix(path: Path, seconds: float = 6.0, bpm: float = 100.0) -> Path:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_test_audio.py"),
            str(path),
            "--seconds",
            str(seconds),
            "--bpm",
            str(bpm),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def _assert_gp5_and_outputs(result) -> None:
    assert result.status == "ok", result.warnings
    assert result.gp5 is not None and result.gp5.is_file()
    assert not any("tie layout failed" in warning for warning in result.warnings)

    for name in STEM_NAMES:
        stem = Path(result.stems[name])
        assert stem.is_file() and stem.stat().st_size > 0
        info = sf.info(stem)
        assert info.frames > 0
        assert result.midi[name].is_file()
        assert result.midi[name].stat().st_size > 0
    assert result.full_midi.is_file() and result.full_midi.stat().st_size > 0
    assert result.zip_path.is_file() and result.zip_path.stat().st_size > 0

    song = parse_gp5(result.gp5)
    assert [track.name for track in song.tracks] == ["Guitar", "Bass", "Drums", "Vocals"]
    for track in song.tracks:
        assert len(track.measures) == len(song.measureHeaders)
        for measure in track.measures:
            assert len(measure.voices) == 2
            assert sum(beat.duration.time for beat in measure.voices[0].beats) == TICKS_PER_BAR
            assert all(
                beat.status != gp.BeatStatus.empty for beat in measure.voices[0].beats
            )

    full = pretty_midi.PrettyMIDI(str(result.full_midi))
    assert len(full.instruments) == 4
    assert sum(len(instrument.notes) for instrument in full.instruments) > 0

    meta = json.loads((result.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ok"
    assert meta["device"] in {"cuda", "cpu"}
    assert 40.0 <= float(meta["audio"]["bpm"]) <= 240.0
    assert set(meta["note_counts"]) == set(STEM_NAMES)

    rhythm = meta["rhythm"]
    assert rhythm["enabled"] is True
    assert rhythm["beats_per_bar"] == 4
    assert rhythm["method"] in {"madmom-infer", "librosa-accent"}
    first_downbeat = rhythm["first_downbeat_s"]
    assert first_downbeat is not None
    # The bar grid must land on an integer bar boundary at the detected
    # downbeat; this is the invariant behind the alignment fix.
    bar_s = 60.0 / float(meta["audio"]["bpm"]) * 4
    bar_position = (first_downbeat - rhythm["grid_offset_s"]) / bar_s
    assert abs(bar_position - round(bar_position)) < 0.02

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
    assert {"meta.json", "song.gp5", "midi/full.mid"} <= names
    assert {f"stems/{name}.wav" for name in STEM_NAMES} <= names
    assert {f"midi/{name}.mid" for name in STEM_NAMES} <= names


def test_synthetic_mix_end_to_end(tmp_path: Path) -> None:
    input_path = _make_synthetic_mix(tmp_path / "mix.wav")
    options = Options(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        demucs_model="htdemucs",
        quantize=16,
        device="auto",
        keep_stems=True,
    )
    result = pipeline.run(input_path, options)

    _assert_gp5_and_outputs(result)
    assert result.note_counts["drums"] > 0
    assert result.note_counts["bass"] > 0
    assert result.note_counts["other"] > 0
    # The synthetic bass and drum parts must actually survive separation.
    for name in ("drums", "bass", "other"):
        audio, _rate = sf.read(result.stems[name], always_2d=True)
        assert float(np.max(np.abs(audio))) > 1e-5


@pytest.mark.skipif(not DEMO.is_file(), reason=f"real demo clip not found: {DEMO}")
def test_real_demo_clip_end_to_end(tmp_path: Path) -> None:
    clip = tmp_path / "demo60.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "45",
            "-t",
            "60",
            "-i",
            str(DEMO),
            "-vn",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    options = Options(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        demucs_model="htdemucs",
        quantize=16,
        device="auto",
        keep_stems=True,
    )
    result = pipeline.run(clip, options)

    _assert_gp5_and_outputs(result)
    assert 60.0 <= result.bpm <= 180.0
    assert 59.0 <= result.duration_s <= 61.0
    assert result.note_counts["drums"] > 0
    assert result.note_counts["other"] > 0
    for name in STEM_NAMES:
        audio, _rate = sf.read(result.stems[name], always_2d=True)
        assert float(np.max(np.abs(audio))) > 1e-5
