#!/usr/bin/env python3
"""Build a small, fully local synthetic mix for slow tests and demos.

The mix contains simple drums, a bass line, and guitar-ish chords. It is not
meant to sound like a real recording; it gives Demucs/Basic Pitch a repeatable
input with activity in the low, mid, and high bands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def _midi_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def _envelope(n: int, attack: float = 0.005, decay: float = 6.0) -> np.ndarray:
    t = np.arange(n) / SR
    attack_n = max(1, int(attack * SR))
    env = np.exp(-t * decay)
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)
    return env


def _pluck(pitch: float, duration: float, gain: float = 1.0) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    f = _midi_hz(pitch)
    env = _envelope(n, attack=0.004, decay=6.5)
    # A few harmonics make the chord audible to Basic Pitch without being a
    # pure sine.
    signal = (
        np.sin(2.0 * np.pi * f * t)
        + 0.45 * np.sin(2.0 * np.pi * 2.0 * f * t)
        + 0.22 * np.sin(2.0 * np.pi * 3.0 * f * t)
    )
    return gain * signal * env


def _bass_note(pitch: float, duration: float, gain: float = 1.0) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    f = _midi_hz(pitch)
    env = _envelope(n, attack=0.008, decay=2.8)
    return gain * (np.sin(2.0 * np.pi * f * t) + 0.15 * np.sin(2.0 * np.pi * 2 * f * t)) * env


def _kick(rng: np.random.Generator) -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = np.linspace(95.0, 45.0, n)
    body = np.sin(2.0 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 13.0)
    click = rng.normal(0.0, 1.0, n) * np.exp(-t * 250.0)
    return 0.9 * body + 0.3 * click


def _snare(rng: np.random.Generator) -> np.ndarray:
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    noise = rng.normal(0.0, 1.0, n) * np.exp(-t * 18.0)
    body = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t * 16.0)
    return 0.55 * noise + 0.35 * body


def _highpassed_noise(rng: np.random.Generator, n: int, cutoff: float = 7000.0) -> np.ndarray:
    # A simple difference high-pass is enough for a test mix and avoids
    # scipy.signal as a runtime dependency of this script.
    noise = rng.normal(0.0, 1.0, n)
    return np.diff(noise, prepend=noise[0]) * 0.5


def _hat(rng: np.random.Generator, duration: float = 0.07) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    return 0.22 * _highpassed_noise(rng, n) * np.exp(-t * 55.0)


def _place(track: np.ndarray, signal: np.ndarray, at_s: float, gain: float = 1.0) -> None:
    start = int(at_s * SR)
    if start >= len(track):
        return
    n = min(len(signal), len(track) - start)
    if n > 0:
        track[start : start + n] += gain * signal[:n]


def make_mix(seconds: float = 9.6, bpm: float = 100.0, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    beat = 60.0 / bpm
    bar = 4.0 * beat
    stems = {
        "drums": np.zeros(n, dtype=np.float64),
        "bass": np.zeros(n, dtype=np.float64),
        "chords": np.zeros(n, dtype=np.float64),
    }

    # Four-bar loop: C, Am, F, G.
    chords = [
        ((48, 52, 55), 36),   # C3 E3 G3, C2 bass
        ((45, 48, 52), 33),   # A2 C3 E3, A1 bass
        ((41, 45, 48), 29),   # F2 A2 C3, F1 bass
        ((43, 47, 50), 31),   # G2 B2 D3, G1 bass
    ]
    n_bars = max(1, int(round(seconds / bar)))
    for bar_index in range(n_bars):
        bar_start = bar_index * bar
        chord, root = chords[bar_index % len(chords)]
        for pitch in chord:
            _place(stems["chords"], _pluck(pitch, min(bar * 0.95, 2.6), 0.23), bar_start)
        for beat_index in range(0, 4, 2):
            _place(stems["chords"], _pluck(chord[0] + 12, min(beat * 0.9, 1.2), 0.10),
                   bar_start + beat_index * beat)
        for eighth in range(8):
            at = bar_start + eighth * beat / 2.0
            pitch = root if eighth % 4 != 3 else root + 7
            _place(stems["bass"], _bass_note(pitch, beat * 0.42, 0.38), at)

        for beat_index in range(4):
            at = bar_start + beat_index * beat
            if beat_index in (0, 2):
                _place(stems["drums"], _kick(rng), at, 0.7)
            if beat_index in (1, 3):
                _place(stems["drums"], _snare(rng), at, 0.45)
            _place(stems["drums"], _hat(rng), at + beat / 2.0, 0.8)
            _place(stems["drums"], _hat(rng), at, 0.6)

    # A quiet sine melody keeps the upper midrange active for Basic Pitch.
    melody = np.zeros(n, dtype=np.float64)
    melody_notes = ((64, 1), (67, 2), (69, 1), (67, 2), (64, 1), (62, 2), (60, 2))
    step = beat / 2.0
    at = 0.0
    for pitch, steps in melody_notes * max(1, n_bars // 2):
        if at >= seconds:
            break
        dur = step * steps * 0.9
        _place(melody, _pluck(pitch, dur, 0.11), at)
        at += step * steps

    mono = stems["drums"] + stems["bass"] + stems["chords"] + melody
    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono *= 0.85 / peak
    # Keep stereo so the normalization path and Demucs see a normal mix.
    left = mono + 0.03 * np.roll(mono, int(0.003 * SR))
    right = mono
    return np.stack([left, right], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("work/test_mix.wav"),
        help="output WAV path (default: work/test_mix.wav)",
    )
    parser.add_argument("--seconds", type=float, default=9.6, help="duration in seconds")
    parser.add_argument("--bpm", type=float, default=100.0, help="tempo")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    mix = make_mix(seconds=max(2.0, float(args.seconds)), bpm=float(args.bpm), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, mix, SR, subtype="PCM_16")
    print(f"wrote {args.output} ({mix.shape[0] / SR:.2f}s, {SR} Hz, stereo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
