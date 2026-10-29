from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from audio2gp.drums import transcribe_drums

SR = 44100


def _highpassed_noise(rng: np.random.Generator, n: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, "highpass", fs=SR, output="sos")
    return sosfilt(sos, rng.normal(0.0, 1.0, n))


def _kick(rng: np.random.Generator) -> np.ndarray:
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    freq = np.linspace(95.0, 45.0, n)
    body = np.sin(2.0 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 13.0)
    click = rng.normal(0.0, 1.0, n) * np.exp(-t * 250.0)
    return body + 0.4 * click


def _snare(rng: np.random.Generator) -> np.ndarray:
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    body = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t * 16.0)
    noise = _highpassed_noise(rng, n, 1200.0) * np.exp(-t * 14.0)
    return 0.5 * body + noise


def _hat(rng: np.random.Generator, open_hat: bool = False) -> np.ndarray:
    duration = 0.5 if open_hat else 0.09
    n = int(duration * SR)
    t = np.arange(n) / SR
    decay = 6.0 if open_hat else 55.0
    return _highpassed_noise(rng, n, 7000.0) * np.exp(-t * decay)


def _place(track: np.ndarray, signal: np.ndarray, at: float, gain: float) -> None:
    start = int(at * SR)
    n = min(len(signal), len(track) - start)
    if n > 0:
        track[start : start + n] += gain * signal[:n]


@pytest.fixture(scope="module")
def synthetic_pattern(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(7)
    track = np.zeros(int(4.0 * SR), dtype=np.float64)
    # Deliberately simple, unambiguous pattern: kicks on quarter-ish beats,
    # snares off the beat, eighth-note closed hats, and one open hat.
    for at in (0.25, 1.25, 2.25):
        _place(track, _kick(rng), at, 0.9)
    for at in (0.75, 1.75, 2.75):
        _place(track, _snare(rng), at, 0.55)
    for at in (0.125, 0.375, 0.625, 1.0, 1.5, 2.0, 2.5, 3.0):
        _place(track, _hat(rng), at, 0.25)
    _place(track, _hat(rng, open_hat=True), 3.25, 0.3)
    track *= 0.9 / np.max(np.abs(track))
    path = tmp_path_factory.mktemp("drums") / "pattern.wav"
    sf.write(path, track, SR)
    return path


def _hits_by_pitch(events) -> dict[int, list[float]]:
    hits: dict[int, list[float]] = {}
    for event in events:
        hits.setdefault(event.pitch, []).append(event.start_s)
    return hits


def _assert_hit(hits: dict[int, list[float]], pitch: int, at: float) -> None:
    assert pitch in hits, f"missing GM pitch {pitch} at {at:.3f}s; got {hits}"
    assert any(abs(t - at) <= 0.08 for t in hits[pitch]), (
        f"pitch {pitch} not near {at:.3f}s; got {sorted(hits[pitch])}"
    )


def test_synthetic_hits_classified(synthetic_pattern: Path) -> None:
    events, warnings = transcribe_drums(synthetic_pattern, bpm=120.0, quantize=16)
    assert warnings == []
    hits = _hits_by_pitch(events)

    for at in (0.25, 1.25, 2.25):
        _assert_hit(hits, 36, at)
    for at in (0.75, 1.75, 2.75):
        _assert_hit(hits, 38, at)
    for at in (0.125, 0.375, 0.625, 1.0, 1.5, 2.0, 2.5, 3.0):
        _assert_hit(hits, 42, at)
    _assert_hit(hits, 46, 3.25)

    # Only the intended GM classes, and no duplicate (time, pitch) hits.
    assert set(hits) == {36, 38, 42, 46}
    assert len(events) == 15
    assert len({(round(e.start_s, 3), e.pitch) for e in events}) == len(events)
    for event in events:
        assert 0.2 <= event.amplitude <= 1.0
        assert event.end_s > event.start_s


def test_silent_stem_returns_empty_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(SR, dtype=np.float32), SR)
    events, warnings = transcribe_drums(path, bpm=120.0, quantize=16)
    assert events == []
    assert any("silent" in warning for warning in warnings)


def test_slow_fade_returns_no_onsets(tmp_path: Path) -> None:
    duration = 2.0
    n = int(duration * SR)
    t = np.arange(n) / SR
    # A full-length linear fade has no transient, so the onset detector
    # should not invent drum hits.
    path = tmp_path / "fade.wav"
    sf.write(path, 0.1 * (t / duration) * np.sin(2.0 * np.pi * 220.0 * t), SR)
    events, warnings = transcribe_drums(path, bpm=120.0, quantize=16)
    assert events == []
    assert warnings
