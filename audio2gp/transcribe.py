from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .config import Options
from .types import NoteEvent
from .utils import is_silent

LocalCB = Callable[[float, str], None]

_MODEL = None


def _emit(cb: LocalCB | None, pct: float, message: str) -> None:
    if cb:
        cb(max(0.0, min(100.0, float(pct))), message)


def _get_model():
    global _MODEL
    if _MODEL is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        _MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _MODEL


FREQ_RANGES = {
    "vocals": (80.0, 1200.0),
    "bass": (30.0, 400.0),
    "other": (60.0, 2000.0),
}


def transcribe_stems(
    stems: dict[str, Path],
    options: Options,
    bpm: float,
    logger: logging.Logger | None = None,
    progress_cb: LocalCB | None = None,
) -> tuple[dict[str, list[NoteEvent]], list[str]]:
    from basic_pitch.inference import predict

    warnings: list[str] = []
    result: dict[str, list[NoteEvent]] = {}
    melodic = [name for name in ("vocals", "bass", "other") if name in stems]
    _emit(progress_cb, 1, "加载 Basic Pitch（CPU）")
    model = _get_model()

    for index, name in enumerate(melodic):
        stem_path = stems[name]
        _emit(progress_cb, 5 + 90 * index / max(1, len(melodic)), f"转录 {name}")
        if is_silent(stem_path):
            result[name] = []
            warnings.append(f"{name} stem is silent; wrote an empty track")
            continue
        low, high = FREQ_RANGES[name]
        try:
            _, _midi, note_events = predict(
                str(stem_path),
                model_or_model_path=model,
                onset_threshold=float(options.onset_threshold),
                frame_threshold=float(options.frame_threshold),
                minimum_note_length=max(30.0, float(options.min_note_sec) * 1000.0),
                minimum_frequency=low,
                maximum_frequency=high,
                multiple_pitch_bends=False,
                melodia_trick=name in {"vocals", "bass"},
                midi_tempo=float(bpm),
            )
        except Exception as exc:
            if is_silent(stem_path):
                result[name] = []
                warnings.append(f"{name} could not be transcribed (silent); empty track")
                continue
            raise RuntimeError(f"Basic Pitch failed for {name}: {exc}") from exc

        events: list[NoteEvent] = []
        for item in note_events:
            try:
                start, end, pitch, amplitude = item[:4]
                start = float(start)
                end = float(end)
                pitch = int(round(float(pitch)))
                amplitude = float(amplitude)
                raw_bends = item[4] if len(item) > 4 else None
                bends = tuple(int(value) for value in raw_bends) if raw_bends else ()
            except Exception:
                continue
            if end > start and 0 <= pitch <= 127:
                events.append(NoteEvent(start, end, pitch, amplitude, bends))
        raw_count = len(events)
        if events and options.ghost_amplitude > 0:
            # AI-generated mixes and Demucs bleed often leave short, quiet
            # ghost octaves. Filter those only when both amplitude and duration
            # are implausibly small, so genuinely soft playing survives.
            peak = max(event.amplitude for event in events)
            gate = max(0.05, float(options.ghost_amplitude) * peak)
            events = [
                event
                for event in events
                if not (
                    event.amplitude < gate
                    and (event.end_s - event.start_s) < 0.15
                )
            ]
        result[name] = events
        if not events:
            warnings.append(f"{name} produced no notes; wrote an empty track")
        elif raw_count != len(events):
            warnings.append(
                f"{name}: suppressed {raw_count - len(events)} short/quiet ghost note(s)"
            )
        if logger:
            logger.info(
                "transcribe: %s -> %d note events (%d ghost suppressed)",
                name,
                len(events),
                max(0, raw_count - len(events)),
            )
        _emit(progress_cb, 5 + 90 * (index + 1) / max(1, len(melodic)), f"{name} 转录完成")

    _emit(progress_cb, 100, "旋律转录完成")
    return result, warnings
