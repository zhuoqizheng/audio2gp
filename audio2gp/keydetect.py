from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# Krumhansl-Kessler (1982) key profiles, rotated to C.
_KK_MAJOR = np.asarray(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
_KK_MINOR = np.asarray(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)

PITCH_NAMES = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)
FLAT_NAMES = (
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
)

# PyGuitarPro names its enum values by "major/minor + accidental count",
# which is easy to get backwards (BMajorFlat = B-flat major). Keep an explicit
# map from pitch-class/mode to the enum member's name.
_GP_KEY_NAMES = {
    (0, "major"): "CMajor",
    (1, "major"): "DMajorFlat",
    (2, "major"): "DMajor",
    (3, "major"): "EMajorFlat",
    (4, "major"): "EMajor",
    (5, "major"): "FMajor",
    (6, "major"): "FMajorSharp",
    (7, "major"): "GMajor",
    (8, "major"): "AMajorFlat",
    (9, "major"): "AMajor",
    (10, "major"): "BMajorFlat",
    (11, "major"): "BMajor",
    (0, "minor"): "CMinor",
    (1, "minor"): "CMinorSharp",
    (2, "minor"): "DMinor",
    (3, "minor"): "EMinorFlat",
    (4, "minor"): "EMinor",
    (5, "minor"): "FMinor",
    (6, "minor"): "FMinorSharp",
    (7, "minor"): "GMinor",
    (8, "minor"): "GMinorSharp",
    (9, "minor"): "AMinor",
    (10, "minor"): "BMinorFlat",
    (11, "minor"): "BMinor",
}
_FLAT_MAJOR_TONICS = {1, 3, 8, 10}
_FLAT_MINOR_TONICS = {3, 5, 8, 10}


@dataclass
class KeyInfo:
    tonic: int
    mode: str
    name: str
    confidence: float
    method: str = "krumhansl-kessler"
    warnings: list[str] = field(default_factory=list)

    @property
    def gp_key_name(self) -> str:
        return _GP_KEY_NAMES[(int(self.tonic) % 12, self.mode)]


def _chroma_for_file(path: Path, hop_length: int = 4096) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if y.size < sr // 2:
        raise ValueError("audio is too short for key detection")
    # Separate the harmonic part so drums/percussion do not smear the chroma.
    harmonic = librosa.effects.harmonic(y, margin=3.0)
    if float(np.max(np.abs(harmonic))) < 1e-6:
        harmonic = y
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=12,
    )
    # Loud frames carry more of the harmony than silence/noise.
    energy = np.linalg.norm(chroma, axis=0)
    if float(np.sum(energy)) > 1e-9:
        weights = energy / np.sum(energy)
        return np.asarray(chroma @ weights, dtype=float)
    return np.mean(chroma, axis=1)


def _correlate(chroma: np.ndarray, profile: np.ndarray) -> np.ndarray:
    scores = np.zeros(12, dtype=float)
    for tonic in range(12):
        rotated = np.roll(profile, tonic)
        if np.std(chroma) < 1e-9 or np.std(rotated) < 1e-9:
            scores[tonic] = 0.0
        else:
            scores[tonic] = float(np.corrcoef(chroma, rotated)[0, 1])
    return scores


def detect_key(
    paths: Iterable[Path | str],
    warnings: list[str] | None = None,
) -> KeyInfo | None:
    """Estimate the global key from one or more harmonic sources.

    Multiple stems are averaged so a quiet guitar stem cannot outweigh a
    bass/other mix. Failures are non-fatal; callers should record a warning and
    leave the GP5 in C major (its neutral default).
    """
    warning_list = warnings if warnings is not None else []
    summed: np.ndarray | None = None
    used = 0
    for raw_path in paths:
        path = Path(raw_path)
        try:
            chroma = _chroma_for_file(path)
        except Exception as exc:
            warning_list.append(f"key detection skipped {path.name}: {exc}")
            continue
        if not np.all(np.isfinite(chroma)) or float(np.sum(chroma)) <= 1e-9:
            warning_list.append(f"key detection skipped {path.name}: no pitch content")
            continue
        summed = chroma if summed is None else summed + chroma
        used += 1
    if summed is None or used == 0:
        return None

    chroma = summed / used
    chroma = chroma / max(1e-9, float(np.sum(chroma)))
    major_scores = _correlate(chroma, _KK_MAJOR)
    minor_scores = _correlate(chroma, _KK_MINOR)
    candidates = [
        (float(score), tonic, "major") for tonic, score in enumerate(major_scores)
    ] + [(float(score), tonic, "minor") for tonic, score in enumerate(minor_scores)]
    candidates.sort(reverse=True)
    best_score, tonic, mode = candidates[0]
    second = candidates[1][0]
    confidence = max(0.0, min(1.0, (best_score - second) if best_score > 0 else 0.0))
    names = FLAT_NAMES if (
        (mode == "major" and tonic in _FLAT_MAJOR_TONICS)
        or (mode == "minor" and tonic in _FLAT_MINOR_TONICS)
    ) else PITCH_NAMES
    name = f"{names[tonic]} {mode}"
    return KeyInfo(
        tonic=tonic,
        mode=mode,
        name=name,
        confidence=round(float(confidence), 4),
        warnings=warning_list,
    )
