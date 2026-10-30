"""Optional external transcription engine.

The engine is a local command configured through AUDIO2GP_EXTERNAL_CMD. Each
melodic stem is transcribed by one subprocess:

    AUDIO2GP_EXTERNAL_CMD="/path/to/engine --wav {wav} --stem {stem} --out {out} --bpm {bpm}"

The command must write a Standard MIDI File to {out}. Tokens are split with
shlex and never passed through a shell. Placeholders: {wav}, {stem}, {out},
{bpm}, {device}. Audio never leaves the machine.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import Callable

from .config import Options
from .midifile import load_midi_notes
from .types import NoteEvent
from .utils import is_silent, run_checked

LocalCB = Callable[[float, str], None]

ENV_COMMAND = "AUDIO2GP_EXTERNAL_CMD"
ENV_TIMEOUT = "AUDIO2GP_EXTERNAL_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 600.0
MELODIC_STEMS = ("vocals", "bass", "other")


class EngineError(RuntimeError):
    pass


def _emit(cb: LocalCB | None, pct: float, message: str) -> None:
    if cb:
        cb(max(0.0, min(100.0, float(pct))), message)


def engine_command() -> list[str]:
    template = (os.environ.get(ENV_COMMAND) or "").strip()
    if not template:
        raise EngineError(
            f"未配置外部转录引擎：请设置环境变量 {ENV_COMMAND}"
        )
    try:
        return shlex.split(template)
    except ValueError as exc:
        raise EngineError(f"{ENV_COMMAND} 无法解析：{exc}") from exc


def _expand(tokens: list[str], values: dict[str, str]) -> list[str]:
    try:
        return [token.format(**values) for token in tokens]
    except (KeyError, IndexError) as exc:
        raise EngineError(
            f"{ENV_COMMAND} 含未知占位符 {exc}；可用："
            "{wav} {stem} {out} {bpm} {device}"
        ) from exc


def _timeout() -> float:
    try:
        value = float(os.environ.get(ENV_TIMEOUT) or DEFAULT_TIMEOUT_S)
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return value if value > 0 else DEFAULT_TIMEOUT_S


def transcribe_external(
    stems: dict[str, Path],
    options: Options,
    bpm: float,
    logger: logging.Logger | None = None,
    progress_cb: LocalCB | None = None,
) -> tuple[dict[str, list[NoteEvent]], list[str]]:
    """Run the external engine once per melodic stem and read its MIDI."""
    base = engine_command()
    warnings: list[str] = []
    result: dict[str, list[NoteEvent]] = {}
    melodic = [name for name in MELODIC_STEMS if name in stems]
    for index, name in enumerate(melodic):
        stem_path = Path(stems[name])
        _emit(progress_cb, 5 + 90 * index / max(1, len(melodic)), f"外部引擎转录 {name}")
        if is_silent(stem_path):
            result[name] = []
            warnings.append(f"{name} stem is silent; wrote an empty track")
            continue
        out_path = stem_path.parent / "external" / f"{name}.mid"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        cmd = _expand(
            base,
            {
                "wav": str(stem_path),
                "stem": name,
                "out": str(out_path),
                "bpm": f"{float(bpm):.6f}",
                "device": str(options.device),
            },
        )
        if logger:
            logger.info("external engine: %s", " ".join(cmd))
        try:
            run_checked(cmd, timeout=_timeout(), logger=logger)
        except RuntimeError as exc:
            raise EngineError(f"外部引擎转录 {name} 失败：{exc}") from exc
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise EngineError(f"外部引擎没有为 {name} 写出 MIDI：{out_path}")
        notes = [
            NoteEvent(
                start_s=float(note.start_s),
                end_s=float(note.end_s),
                pitch=int(note.pitch),
                amplitude=max(0.05, min(1.0, float(note.velocity) / 127.0)),
            )
            for note in load_midi_notes(out_path)
        ]
        if not notes:
            warnings.append(f"外部引擎对 {name} 返回了空 MIDI")
        result[name] = notes
    return result, warnings
