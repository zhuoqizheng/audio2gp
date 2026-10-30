from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio2gp import engines
from audio2gp.config import Options

FAKE_ENGINE = """
import sys
import mido

out = sys.argv[sys.argv.index("--out") + 1]
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.Message("note_on", note=64, velocity=100, time=0))
track.append(mido.Message("note_off", note=64, velocity=0, time=480))
mid.save(out)
"""


def _wav(path: Path) -> Path:
    sf.write(str(path), np.full(22050, 0.1, dtype=np.float32), 22050)
    return path


def test_external_engine_reads_generated_midi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "fake_engine.py"
    script.write_text(FAKE_ENGINE, encoding="utf-8")
    monkeypatch.setenv(
        engines.ENV_COMMAND,
        f"{sys.executable} {script} --wav {{wav}} --stem {{stem}} --out {{out}}",
    )
    events, warnings = engines.transcribe_external(
        {"other": _wav(tmp_path / "other.wav")}, Options(), 120.0
    )
    assert warnings == []
    assert len(events["other"]) == 1
    note = events["other"][0]
    assert note.pitch == 64
    assert note.start_s == pytest.approx(0.0)
    assert note.end_s == pytest.approx(0.5)
    assert note.amplitude == pytest.approx(100 / 127)


def test_external_engine_reports_configuration_and_placeholder_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(engines.ENV_COMMAND, raising=False)
    with pytest.raises(engines.EngineError, match="未配置"):
        engines.transcribe_external({"other": _wav(tmp_path / "other.wav")}, Options(), 120.0)

    monkeypatch.setenv(engines.ENV_COMMAND, "engine --unknown {missing}")
    with pytest.raises(engines.EngineError, match="未知占位符"):
        engines.transcribe_external({"other": _wav(tmp_path / "other.wav")}, Options(), 120.0)
