from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import unicodedata
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "audio")


def safe_ascii(text: str, fallback: str = "audio2gp", max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\x20-\x7e]+", " ", text).strip()
    return (text[:max_len] or fallback)


def new_job_id(input_path: Path | str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(Path(input_path).stem)
    return f"{stamp}_{slug}_{uuid.uuid4().hex[:6]}"


def run_checked(
    cmd: list[str],
    *,
    timeout: float | None = None,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess[str]:
    if logger:
        logger.info("run: %s", " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout}s: {cmd[0]}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if logger:
            logger.error("command failed (%s): %s", proc.returncode, detail[-4000:])
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"{detail[-4000:]}"
        )
    return proc


def ffprobe_duration(path: Path) -> float:
    proc = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    try:
        return float(proc.stdout.strip().splitlines()[0])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Could not read duration for {path}") from exc


def normalize_audio(input_path: Path, output_path: Path, logger: logging.Logger | None = None) -> None:
    ensure_dir(output_path.parent)
    run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            "-map_metadata",
            "-1",
            str(output_path),
        ],
        logger=logger,
    )


def read_audio(path: Path, *, mono: bool = False) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        data = data.mean(axis=1)
    return data, int(sr)


def is_silent(path: Path, threshold: float = 1e-5) -> bool:
    try:
        data, _ = read_audio(path)
        if data.size == 0:
            return True
        return float(np.max(np.abs(data))) < threshold
    except Exception:
        return True


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_job_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    name = f"audio2gp.job.{abs(hash(str(log_path.resolve())))}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def zip_directory(directory: Path, zip_path: Path) -> Path:
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.resolve() == zip_path.resolve():
                continue
            relative = path.relative_to(directory)
            if any(part.startswith(".") for part in relative.parts):
                continue  # undo snapshots and temp files stay out of the ZIP
            zf.write(path, relative.as_posix())
    return zip_path


def copy_file(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def human_bytes(n: int | float) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out
