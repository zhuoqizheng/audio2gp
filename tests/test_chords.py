from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from audio2gp.chords import analyze_chords, chords_from_state, chords_to_dict
from audio2gp.config import TICKS_PER_BAR, TICKS_PER_QUARTER
from audio2gp.keydetect import KeyInfo

SR = 22050
BPM = 120.0


def _synth(prog: list[tuple[int, str]], *, lead_s: float = 0.0) -> np.ndarray:
    """Simple additive triads, one bar each, with a beat-rate amplitude pulse."""
    beat_s = 60.0 / BPM
    total = lead_s + len(prog) * 4 * beat_s
    audio = np.zeros(int(total * SR), dtype=float)
    times = np.arange(audio.size) / SR
    triads = {"major": (0, 4, 7), "minor": (0, 3, 7)}
    for index, (root, quality) in enumerate(prog):
        start = lead_s + index * 4 * beat_s
        end = start + 4 * beat_s
        mask = (times >= start) & (times < end)
        local = np.zeros(int(mask.sum()))
        local_t = times[mask]
        for offset in triads[quality]:
            midi = 48 + root + offset
            frequency = 440.0 * 2 ** ((midi - 69) / 12)
            for harmonic in (1, 2, 3):
                local += (0.25 / harmonic) * np.sin(
                    2 * np.pi * frequency * harmonic * local_t
                )
        local *= 0.5 + 0.5 * np.exp(-3.0 * ((local_t - start) % beat_s))
        audio[mask] += local
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    return audio.astype(np.float32)


def _write(path: Path, audio: np.ndarray) -> Path:
    wavfile.write(path, SR, audio)
    return path


def test_major_minor_progression_segments_and_summary(tmp_path: Path) -> None:
    audio = _synth([(0, "major"), (7, "major"), (9, "minor"), (5, "major")])
    stem = _write(tmp_path / "other.wav", audio)
    events = analyze_chords(
        {"other": stem},
        BPM,
        offset_s=0.0,
        duration_s=len(audio) / SR,
        key_info=KeyInfo(
            tonic=0,
            mode="major",
            name="C major",
            confidence=0.8,
            method="synthetic",
        ),
    )

    assert [event.label for event in events] == ["C", "G", "Am", "F"]
    assert [event.start_tick for event in events] == [
        0,
        TICKS_PER_BAR,
        2 * TICKS_PER_BAR,
        3 * TICKS_PER_BAR,
    ]
    assert [event.start_s for event in events] == [0.0, 2.0, 4.0, 6.0]
    assert all(event.confidence > 0.35 for event in events)

    doc = chords_to_dict(
        events,
        bpm=BPM,
        offset_s=0.0,
        duration_s=len(audio) / SR,
        key={"name": "C major"},
    )
    assert doc["segments"][0]["bar"] == 1
    assert doc["segments"][0]["beat"] == 1
    assert doc["segments"][2]["bar"] == 3
    assert doc["summary"] == {
        "chord_count": 4,
        "unique_chords": ["Am", "C", "F", "G"],
        "quality_counts": {"major": 3, "minor": 1},
        "progression": ["C", "G", "Am", "F"],
    }
    assert chords_from_state(doc["segments"])[1].label == "G"


def test_offset_places_the_first_chord_at_tick_zero(tmp_path: Path) -> None:
    # The score grid starts 0.5 s into the audio, mirroring a downbeat offset;
    # chord ticks must use that same origin as note quantization.
    lead_s = 0.5
    audio = _synth([(0, "major"), (7, "major")], lead_s=lead_s)
    stem = _write(tmp_path / "other.wav", audio)
    events = analyze_chords(
        {"other": stem},
        BPM,
        offset_s=lead_s,
        duration_s=len(audio) / SR,
    )

    assert events[0].label in {"C", "G"}
    assert events[0].start_tick == 0
    assert events[0].start_s == lead_s
    assert all(event.start_tick % TICKS_PER_QUARTER == 0 for event in events)


def test_analysis_without_a_harmonic_stem_warns_and_skips(tmp_path: Path) -> None:
    warnings: list[str] = []
    events = analyze_chords(
        {"other": tmp_path / "missing.wav"},
        BPM,
        duration_s=2.0,
        warnings=warnings,
    )
    assert events == []
    assert any("文件不存在" in warning for warning in warnings)
    assert any("没有可用的和声声部" in warning for warning in warnings)
