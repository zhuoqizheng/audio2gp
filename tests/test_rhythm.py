from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from audio2gp import rhythm
from audio2gp.config import Options
from audio2gp.midi_post import clean
from audio2gp.types import NoteEvent


def test_fold_bpm_rejects_zero_without_hanging() -> None:
    result: dict[str, BaseException] = {}

    def call() -> None:
        try:
            rhythm._fold_bpm(0.0)
        except BaseException as exc:  # noqa: BLE001 - captured for the assertion
            result["error"] = exc

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(1.0)
    assert not thread.is_alive(), "_fold_bpm(0.0) did not terminate"
    assert isinstance(result.get("error"), ValueError)


def test_choose_grid_offset_without_pickup() -> None:
    # The first musical event is the downbeat: bar 1 starts there.
    assert rhythm.choose_grid_offset(0.02, 0.0, 120.0) == pytest.approx(0.02)


def test_choose_grid_offset_keeps_a_pickup_bar() -> None:
    # 76 BPM 4/4: bar = 3.16 s. A note at 0.2 s and downbeat at 0.95 s
    # means the first downbeat must move to bar 2.
    offset = rhythm.choose_grid_offset(0.95, 0.20, 76.0)
    bar_s = 60.0 / 76.0 * 4
    assert offset == pytest.approx(0.95 - bar_s)
    assert offset < 0.0


def test_choose_grid_offset_handles_a_long_intro() -> None:
    # 120 BPM: bar = 2 s. A downbeat at 5 s needs three leading bars.
    offset = rhythm.choose_grid_offset(5.0, 0.0, 120.0)
    assert offset == pytest.approx(-1.0)
    beat_s = 0.125  # 1/16 grid at 120 BPM
    assert round((0.0 - offset) / beat_s) * 240 > 0


def test_choose_grid_offset_without_a_downbeat_is_a_noop() -> None:
    assert rhythm.choose_grid_offset(None, 0.0, 120.0) == 0.0
    assert rhythm.choose_grid_offset(1.0, 0.0, 0.0) == 0.0


def test_accent_phase_prefers_the_strong_beat_phase() -> None:
    sr = 100
    hop = 1
    beat_times = np.arange(16, dtype=float) * 0.1
    low_energy = np.zeros(200, dtype=float)
    onset_strength = np.zeros(200, dtype=float)
    for index, beat in enumerate(beat_times):
        frame = int(round(beat * sr / hop))
        if index % 4 == 2:
            low_energy[frame] = 10.0
            onset_strength[frame] = 5.0
        else:
            low_energy[frame] = 0.2
            onset_strength[frame] = 0.2
    assert rhythm._accent_phase_score(
        beat_times, low_energy, onset_strength, sr, hop
    ) == 2


def test_analyze_rhythm_falls_back_and_reports_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "song.wav"
    path.write_bytes(b"fake")

    def broken_madmom(*args, **kwargs):
        raise RuntimeError("model unavailable")

    def fake_librosa(*args, **kwargs):
        return rhythm.RhythmInfo(
            bpm=120.0,
            beats_per_bar=4,
            beat_times_s=[0.0, 0.5, 1.0, 1.5],
            downbeat_times_s=[0.0, 2.0],
            method="librosa-accent",
        )

    monkeypatch.setattr(rhythm, "_analyze_madmom", broken_madmom)
    monkeypatch.setattr(rhythm, "_analyze_librosa", fake_librosa)
    info = rhythm.analyze_rhythm(path)
    assert info.method == "librosa-accent"
    assert any("madmom-infer" in warning for warning in info.warnings)


def test_non_4_4_meter_skips_madmom_and_passes_the_meter_to_librosa(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "song.wav"
    path.write_bytes(b"fake")
    seen: dict[str, str] = {}

    def unexpected_madmom(*args, **kwargs):
        raise AssertionError("madmom must not run for 6/8")

    def fake_librosa(*args, **kwargs):
        seen["meter"] = kwargs["meter"].label
        return rhythm.RhythmInfo(
            bpm=120.0,
            beats_per_bar=3,
            beat_times_s=[0.0, 0.5, 1.0, 1.5],
            downbeat_times_s=[0.0, 1.5],
            method="librosa-accent",
        )

    monkeypatch.setattr(rhythm, "_analyze_madmom", unexpected_madmom)
    monkeypatch.setattr(rhythm, "_analyze_librosa", fake_librosa)
    info = rhythm.analyze_rhythm(path, meter="6/8")
    assert seen["meter"] == "6/8"
    assert info.beats_per_bar == 3
    assert info.downbeat_times_s == [0.0, 1.5]
    assert info.warnings == []


def test_clean_applies_a_positive_downbeat_offset() -> None:
    notes = clean(
        {"other": [NoteEvent(0.5, 0.9, 60, 0.8)]},
        Options(quantize=16),
        120.0,
        offset_s=0.5,
    )["other"]
    assert len(notes) == 1
    assert notes[0].start_tick == 0
    assert notes[0].end_tick % 240 == 0


def test_clean_applies_a_negative_offset_for_a_pickup_bar() -> None:
    # offset=-2.0 maps audio time 0 to bar 2 start (tick 3840 at 4/4).
    notes = clean(
        {"other": [NoteEvent(0.0, 0.5, 60, 0.8)]},
        Options(quantize=16),
        120.0,
        offset_s=-2.0,
    )["other"]
    assert len(notes) == 1
    assert notes[0].start_tick == 3840
    assert notes[0].end_tick == 3840 + 4 * 240
