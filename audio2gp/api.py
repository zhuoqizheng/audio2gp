from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import pipeline, score_ir
from .config import (
    BASS_TUNINGS,
    GUITAR_TUNINGS,
    QUANTIZE_VALUES,
    STEM_NAMES,
    TIME_SIGNATURE_CHOICES,
    TIME_SIGNATURE_LABELS,
    Options,
)
from .utils import (
    ensure_dir,
    ffprobe_duration,
    new_job_id,
    read_json,
    slugify,
    write_json,
)

logger = logging.getLogger("audio2gp.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".aif",
    ".aiff",
}
STAGE_LABELS = {
    "queued": "排队中",
    "preprocess": "预处理与节拍",
    "rhythm": "强拍对齐",
    "separate": "Demucs 分轨",
    "analyze": "速度与调性",
    "transcribe": "音符转录",
    "post": "量化与指法",
    "gp5": "导出结果",
    "done": "完成",
    "error": "失败",
}
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class JobOptions(BaseModel):
    """The subset of Options that a browser client may choose."""

    model_config = ConfigDict(extra="forbid")

    demucs_model: Literal["htdemucs_ft", "htdemucs"] = "htdemucs_ft"
    quantize: int = 16
    guitar_stem: Literal["other", "vocals"] = "other"
    keep_stems: bool = True
    max_fret: int = Field(default=15, ge=0, le=24)
    min_note_sec: float = Field(default=0.03, ge=0.0, le=1.0)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    include_vocals: bool = True
    bpm_override: float | None = Field(default=None, gt=0, le=300)
    time_signature: str = "4/4"
    manual_downbeat_s: float | None = Field(default=None, gt=0, le=24 * 3600)
    align_downbeats: bool = True
    guitar_tuning: str = "standard"
    bass_tuning: str = "standard"
    suggest_tuning: bool = False
    ghost_amplitude: float = Field(default=0.15, ge=0.0, le=1.0)
    max_polyphony: int = Field(default=6, ge=1, le=12)
    onset_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    frame_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    detect_techniques: bool = True
    detect_key: bool = True
    detect_chords: bool = True
    export_gp5: bool = True
    tempo_from_drums: bool = True
    cache_stems: bool = True
    selected_stems: list[str] = Field(default_factory=lambda: list(STEM_NAMES))
    layout: Literal["auto", "onset", "ties"] = "auto"


class NoteEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=0)
    string: int | None = Field(default=None, ge=1, le=7)
    fret: int | None = Field(default=None, ge=0, le=24)
    pitch: int | None = Field(default=None, ge=0, le=127)


class MuteEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=0)
    muted: bool


@dataclass
class JobRecord:
    id: str
    original_name: str
    input_path: Path
    work_dir: Path
    out_dir: Path
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "等待处理"
    log: list[dict[str, Any]] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished: str | None = None
    error: str | None = None
    partial: bool = False
    warnings: list[str] = field(default_factory=list)
    result: dict | None = None
    edit_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def state_path(self) -> Path:
        return self.out_dir / "tracks.json"


class JobManager:
    """Single-process job registry with a one-worker GPU-friendly executor.

    Jobs live in memory; their output directories remain on disk. This is
    deliberately a localhost application, not a multi-user service.
    """

    def __init__(
        self,
        output_root: Path,
        work_root: Path,
        *,
        runner: Callable[..., Any] | None = None,
        max_upload_mb: int = 200,
        max_duration_s: float = 600.0,
    ) -> None:
        self.output_root = ensure_dir(Path(output_root).expanduser().resolve())
        self.work_root = ensure_dir(Path(work_root).expanduser().resolve())
        self.runner = runner or pipeline.run
        self.max_upload_bytes = int(max_upload_mb) * 1024 * 1024
        self.max_duration_s = float(max_duration_s)
        self.jobs: dict[str, JobRecord] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio2gp-job")

    def allocate(self, original_name: str) -> JobRecord:
        original = Path(original_name or "audio").name
        extension = Path(original).suffix.lower()
        base = slugify(Path(original).stem, max_len=36)
        with self.lock:
            job_id = new_job_id(original)
            while job_id in self.jobs:
                job_id = new_job_id(original)
            work_dir = ensure_dir(self.work_root / job_id)
            out_dir = self.output_root / job_id
            record = JobRecord(
                id=job_id,
                original_name=original,
                input_path=work_dir / f"{base}{extension}",
                work_dir=work_dir,
                out_dir=out_dir,
            )
            self.jobs[job_id] = record
        return record

    def discard(self, record: JobRecord) -> None:
        with self.lock:
            self.jobs.pop(record.id, None)
        shutil.rmtree(record.work_dir, ignore_errors=True)
        shutil.rmtree(record.out_dir, ignore_errors=True)

    def submit(self, record: JobRecord, options: Options) -> None:
        with self.lock:
            record.status = "queued"
            record.stage = "queued"
            record.progress = 0.0
            record.message = "等待 GPU 空闲"
            record.log.append(self._entry("queued", 0.0, record.message))
        self.executor.submit(self._run, record, options)

    def _entry(self, stage: str, pct: float, message: str) -> dict[str, Any]:
        return {
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "stage_label": STAGE_LABELS.get(stage, stage),
            "progress": round(max(0.0, min(100.0, float(pct))), 1),
            "message": str(message),
        }

    def _run(self, record: JobRecord, options: Options) -> None:
        with self.lock:
            record.status = "running"
            record.stage = "preprocess"
            record.message = "开始处理"
            record.log.append(self._entry("preprocess", 0.0, record.message))

        def progress(stage: str, overall: float, message: str) -> None:
            with self.lock:
                record.stage = stage
                record.progress = max(record.progress, float(overall))
                record.message = str(message)
                record.log.append(self._entry(stage, overall, message))
                if len(record.log) > 80:
                    del record.log[: len(record.log) - 80]

        try:
            self.runner(
                record.input_path,
                options,
                progress_cb=progress,
                job_id=record.id,
            )
            result = self._build_result(record)
            with self.lock:
                record.status = "done"
                record.stage = "done"
                record.progress = 100.0
                record.result = result
                record.partial = result.get("status") == "partial"
                record.warnings = list(result.get("warnings") or [])
                if record.partial:
                    record.message = "完成（GP5 部分失败）"
                elif result.get("output_format") == "midi_only":
                    record.message = "完成（仅 MIDI）"
                else:
                    record.message = "完成"
                record.finished = datetime.now(timezone.utc).isoformat()
                record.log.append(self._entry("done", 100.0, record.message))
        except Exception as exc:  # keep the local server alive
            logger.exception("job %s failed", record.id)
            with self.lock:
                record.status = "error"
                record.stage = "error"
                record.message = "处理失败"
                record.error = str(exc)
                record.finished = datetime.now(timezone.utc).isoformat()
                record.log.append(self._entry("error", record.progress, str(exc)))
        finally:
            if record.finished is None:
                record.finished = datetime.now(timezone.utc).isoformat()

    def get(self, job_id: str) -> JobRecord:
        with self.lock:
            record = self.jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="没有这个任务")
        return record

    def _build_result(self, record: JobRecord) -> dict:
        meta: dict[str, Any] = {}
        meta_path = record.out_dir / "meta.json"
        if meta_path.is_file():
            try:
                loaded = read_json(meta_path)
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                logger.warning("job %s: unreadable meta.json", record.id, exc_info=True)
        state: dict | None = None
        if record.state_path.is_file():
            try:
                state = score_ir.load_state(record.state_path)
            except Exception:
                logger.warning("job %s: unreadable tracks.json", record.id, exc_info=True)
        chords_doc: dict[str, Any] = {}
        if (record.out_dir / "chords.json").is_file():
            try:
                loaded = read_json(record.out_dir / "chords.json")
                if isinstance(loaded, dict):
                    chords_doc = loaded
            except Exception:
                logger.warning("job %s: unreadable chords.json", record.id, exc_info=True)

        def url(relative: str) -> str:
            return f"/api/jobs/{record.id}/files/{quote(relative, safe='/')}"

        files: dict[str, Any] = {}
        if (record.out_dir / "song.gp5").is_file():
            files["gp5"] = url("song.gp5")
        if (record.out_dir / "meta.json").is_file():
            files["meta"] = url("meta.json")
        if state is not None:
            files["tracks"] = url("tracks.json")
        if (record.out_dir / "chords.json").is_file():
            files["chords"] = url("chords.json")
        if (record.out_dir / "midi" / "full.mid").is_file():
            files["full_midi"] = url("midi/full.mid")
        midi_names = []
        if state is not None:
            for track in state.get("tracks") or []:
                stem = str(track.get("stem") or "")
                if stem in STEM_NAMES and stem not in midi_names:
                    midi_names.append(stem)
        else:
            midi_names = [name for name in (meta.get("selected_stems") or []) if name in STEM_NAMES]
        files["midi"] = {
            name: url(f"midi/{name}.mid")
            for name in midi_names
            if (record.out_dir / "midi" / f"{name}.mid").is_file()
        }
        files["stems"] = {
            name: url(f"stems/{name}.wav")
            for name in midi_names
            if (record.out_dir / "stems" / f"{name}.wav").is_file()
        }
        zip_name = str((meta.get("files") or {}).get("zip") or "")
        if zip_name and (record.out_dir / zip_name).is_file():
            files["zip"] = url(zip_name)
        meta_files = meta.get("files") or {}
        if not files["stems"] and isinstance(meta_files.get("stems"), dict):
            files["stems"] = {
                name: url(relative)
                for name, relative in meta_files["stems"].items()
                if (record.out_dir / relative).is_file()
            }

        audio: dict[str, str] = {}
        for name in STEM_NAMES:
            if (
                (record.out_dir / "stems" / f"{name}.wav").is_file()
                or (record.work_dir / "stems" / f"{name}.wav").is_file()
            ):
                audio[name] = f"/api/jobs/{record.id}/audio/{name}"

        revision = int((state or {}).get("revision", meta.get("revision", 0)) or 0)
        track_summary: list[dict] = []
        if state is not None:
            for track in state.get("tracks") or []:
                track_summary.append(
                    {
                        "id": track.get("id"),
                        "name": track.get("name"),
                        "stem": track.get("stem"),
                        "tuning": track.get("tuning") or [],
                        "is_percussion": bool(track.get("is_percussion")),
                        "muted": bool(track.get("muted")),
                        "notes": len(track.get("notes") or []),
                        "drum_notes": len(track.get("drum_notes") or []),
                    }
                )
        else:
            track_summary = [
                {"id": None, "name": name, "stem": None, "notes": None, "muted": False}
                for name in (meta.get("tracks") or [])
            ]

        result_status = str(meta.get("status") or "ok")
        if result_status not in {"ok", "partial"}:
            result_status = "ok"
        key = meta.get("key") or (state or {}).get("key") or {}
        score_url = (
            f"/api/jobs/{record.id}/score.gp5?rev={revision}"
            if (record.out_dir / "song.gp5").is_file()
            else None
        )
        output_format = str(
            meta.get("output_format")
            or ("gp5_midi" if score_url else "midi_only")
        )
        if output_format not in {"gp5_midi", "midi_only"}:
            output_format = "gp5_midi"
        chord_segments = chords_doc.get("segments") or []
        chord_summary = chords_doc.get("summary") or {}
        time_signature = str(
            (state or {}).get("time_signature")
            or meta.get("time_signature")
            or (meta.get("options") or {}).get("time_signature")
            or "4/4"
        )
        bar_ticks = int(
            (state or {}).get("bar_ticks")
            or meta.get("bar_ticks")
            or 3840
        )
        return {
            "status": result_status,
            "output_format": output_format,
            "chords": chord_segments,
            "chord_summary": chord_summary,
            "chord_detection": bool(meta.get("options", {}).get("detect_chords", True)),
            "bpm": (state or {}).get("bpm", (meta.get("audio") or {}).get("bpm")),
            "time_signature": time_signature,
            "bar_ticks": bar_ticks,
            "manual_downbeat_s": (meta.get("options") or {}).get("manual_downbeat_s"),
            "align_downbeats": bool(
                (meta.get("options") or {}).get("align_downbeats", True)
            ),
            "duration_s": (meta.get("audio") or {}).get("duration_s"),
            "device": meta.get("device"),
            "key": key,
            "measure_count": (state or {}).get("measure_count", meta.get("measure_count")),
            "revision": revision,
            "tracks": track_summary,
            "note_counts": meta.get("note_counts") or {},
            "warnings": list(meta.get("warnings") or []),
            "files": files,
            "audio": audio,
            "state_url": f"/api/jobs/{record.id}/state" if state is not None else None,
            "score_url": score_url,
            "can_edit": state is not None,
            "rhythm": meta.get("rhythm") or {},
        }

    def refresh_after_edit(self, record: JobRecord) -> dict:
        result = self._build_result(record)
        with self.lock:
            record.result = result
            record.partial = result.get("status") == "partial"
            record.warnings = list(result.get("warnings") or [])
            record.status = "done"
            record.stage = "done"
            record.progress = 100.0
            record.message = "已更新乐谱" if not record.partial else "已更新（GP5 部分失败）"
        return result

    def public(self, record: JobRecord) -> dict:
        with self.lock:
            return {
                "id": record.id,
                "status": record.status,
                "stage": record.stage,
                "stage_label": STAGE_LABELS.get(record.stage, record.stage),
                "progress": round(record.progress, 1),
                "message": record.message,
                "log": list(record.log[-30:]),
                "created": record.created,
                "finished": record.finished,
                "error": record.error,
                "partial": record.partial,
                "original_name": record.original_name,
                "warnings": list(record.warnings),
                "result": record.result if record.status == "done" else None,
            }


def _options_from_payload(raw: str, manager: JobManager) -> Options:
    try:
        payload = JobOptions.model_validate_json(raw or "{}")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"参数无效：{exc}") from exc
    data = payload.model_dump()
    data["selected_stems"] = tuple(data["selected_stems"])
    data["output_dir"] = manager.output_root
    data["work_dir"] = manager.work_root
    try:
        return Options(**data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {max_bytes // (1024 * 1024)} MB 上限",
                    )
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="上传文件为空")
    return size


def _safe_output_file(record: JobRecord, relative: str) -> Path:
    base = record.out_dir.resolve()
    candidate = (base / relative).resolve()
    try:
        resolved_relative = candidate.relative_to(base).as_posix()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="文件不存在") from exc
    parts = resolved_relative.split("/")
    allowed = (
        resolved_relative in {"song.gp5", "meta.json", "tracks.json", "chords.json"}
        or (parts[0] in {"midi", "stems"} and len(parts) == 2)
        or (
            len(parts) == 1
            and resolved_relative.startswith("audio2gp_")
            and resolved_relative.endswith(".zip")
        )
    )
    if not allowed or not candidate.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return candidate


def _media_type(path: Path) -> str:
    return {
        ".gp5": "application/octet-stream",
        ".mid": "audio/midi",
        ".wav": "audio/wav",
        ".zip": "application/zip",
        ".json": "application/json",
    }.get(path.suffix.lower(), "application/octet-stream")


def create_app(
    output_root: Path | str | None = None,
    work_root: Path | str | None = None,
    *,
    runner: Callable[..., Any] | None = None,
    max_upload_mb: int | None = None,
    max_duration_s: float | None = None,
) -> FastAPI:
    output = Path(
        output_root
        or os.environ.get("AUDIO2GP_OUTPUT_DIR")
        or "output"
    )
    work = Path(
        work_root
        or os.environ.get("AUDIO2GP_WORK_DIR")
        or "work"
    )
    upload_mb = int(
        max_upload_mb
        if max_upload_mb is not None
        else os.environ.get("AUDIO2GP_MAX_UPLOAD_MB", "200")
    )
    duration_s = float(
        max_duration_s
        if max_duration_s is not None
        else os.environ.get("AUDIO2GP_MAX_DURATION_S", "600")
    )
    manager = JobManager(
        output,
        work,
        runner=runner,
        max_upload_mb=upload_mb,
        max_duration_s=duration_s,
    )

    app = FastAPI(
        title="audio2gp",
        description="本地混音 → Demucs 分轨 → MIDI/GP5 与在线改谱",
        version="0.2.0",
    )
    app.state.manager = manager
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": "0.2.0"}

    @app.get("/api/limits")
    def limits() -> dict:
        return {
            "max_upload_mb": upload_mb,
            "max_duration_s": duration_s,
            "quantize": list(QUANTIZE_VALUES),
            "guitar_tunings": GUITAR_TUNINGS,
            "bass_tunings": BASS_TUNINGS,
            "stems": list(STEM_NAMES),
            "output_formats": ["gp5_midi", "midi_only"],
            "chord_analysis": True,
            "time_signatures": [
                {"value": value, "label": TIME_SIGNATURE_LABELS[value]}
                for value in TIME_SIGNATURE_CHOICES
            ],
            "manual_downbeat": True,
            # The automatic downbeat detector is deliberately 4/4-only.
            "auto_downbeat_time_signatures": ["4/4"],
        }

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        file: UploadFile = File(...),
        options: str = Form("{}"),
    ) -> dict:
        parsed = _options_from_payload(options, manager)
        original = Path(file.filename or "audio").name
        extension = Path(original).suffix.lower()
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail="支持的音频格式：" + ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS)),
            )
        record = manager.allocate(original)
        try:
            await _save_upload(file, record.input_path, manager.max_upload_bytes)
            try:
                duration = await asyncio.to_thread(ffprobe_duration, record.input_path)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"无法读取音频：{exc}") from exc
            if duration > manager.max_duration_s:
                raise HTTPException(
                    status_code=413,
                    detail=f"音频时长 {duration:.1f}s 超过 {manager.max_duration_s:.0f}s 上限",
                )
            manager.submit(record, parsed)
        except Exception:
            manager.discard(record)
            raise
        finally:
            await file.close()
        return {"id": record.id, "status": "queued"}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return manager.public(manager.get(job_id))

    @app.get("/api/jobs/{job_id}/state")
    def get_state(job_id: str) -> dict:
        record = manager.get(job_id)
        if not record.state_path.is_file():
            raise HTTPException(status_code=404, detail="该任务还没有可编辑的乐谱状态")
        try:
            return score_ir.load_state(record.state_path)
        except score_ir.StateError as exc:
            raise HTTPException(status_code=500, detail=f"乐谱状态损坏：{exc}") from exc

    @app.get("/api/jobs/{job_id}/score.gp5")
    def get_score(job_id: str) -> FileResponse:
        record = manager.get(job_id)
        path = record.out_dir / "song.gp5"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="GP5 尚未生成")
        return FileResponse(
            path,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/jobs/{job_id}/audio/{stem}")
    def get_audio(job_id: str, stem: str) -> FileResponse:
        record = manager.get(job_id)
        if stem not in STEM_NAMES:
            raise HTTPException(status_code=404, detail="未知分轨")
        for candidate in (
            record.out_dir / "stems" / f"{stem}.wav",
            record.work_dir / "stems" / f"{stem}.wav",
        ):
            if candidate.is_file():
                return FileResponse(
                    candidate,
                    media_type="audio/wav",
                    headers={"Cache-Control": "no-store"},
                )
        raise HTTPException(status_code=404, detail="该分轨不可用")

    @app.get("/api/jobs/{job_id}/files/{relative:path}")
    def get_file(job_id: str, relative: str) -> FileResponse:
        record = manager.get(job_id)
        path = _safe_output_file(record, relative)
        return FileResponse(
            path,
            media_type=_media_type(path),
            filename=path.name,
            headers={"Cache-Control": "no-store"},
        )

    def _edit_context(record: JobRecord, revision: int) -> dict:
        if not record.state_path.is_file():
            raise HTTPException(status_code=409, detail="该任务尚无可编辑的乐谱")
        try:
            state = score_ir.load_state(record.state_path)
        except score_ir.StateError as exc:
            raise HTTPException(status_code=500, detail=f"乐谱状态损坏：{exc}") from exc
        if int(state.get("revision", 0)) != int(revision):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"乐谱已更新（服务器版本 {state.get('revision', 0)}，"
                    f"页面版本 {revision}），请刷新后重试"
                ),
            )
        return state

    def _commit_edit(record: JobRecord, state: dict, changed: int) -> dict:
        try:
            committed = score_ir.commit_state(
                state,
                record.out_dir,
                logger=logging.getLogger(f"audio2gp.edit.{record.id}"),
            )
        except score_ir.StateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("job %s: score rebuild failed", record.id)
            raise HTTPException(status_code=500, detail=f"重新生成 GP5 失败：{exc}") from exc
        result = manager.refresh_after_edit(record)
        new_state = committed["state"]
        return {
            "job": manager.public(record),
            "state": new_state,
            "revision": int(new_state["revision"]),
            "changed": int(changed),
            "warnings": list(committed["warnings"]),
            "gp5_url": result.get("score_url"),
            "files": result.get("files"),
        }

    @app.patch("/api/jobs/{job_id}/tracks/{track_id}/notes/{note_id}")
    def edit_note(
        job_id: str,
        track_id: str,
        note_id: str,
        payload: NoteEdit,
    ) -> dict:
        record = manager.get(job_id)
        with record.edit_lock:
            state = _edit_context(record, payload.revision)
            if payload.string is None and payload.fret is None and payload.pitch is None:
                raise HTTPException(status_code=422, detail="至少提供 string、fret 或 pitch 之一")
            try:
                edited = score_ir.update_note_position(
                    state,
                    track_id,
                    note_id,
                    string=payload.string,
                    fret=payload.fret,
                    pitch=payload.pitch,
                )
            except score_ir.StateError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _commit_edit(record, state, int(edited["changed"]))

    @app.delete("/api/jobs/{job_id}/tracks/{track_id}/notes/{note_id}")
    def delete_note(
        job_id: str,
        track_id: str,
        note_id: str,
        revision: int = Query(..., ge=0),
    ) -> dict:
        record = manager.get(job_id)
        with record.edit_lock:
            state = _edit_context(record, revision)
            try:
                edited = score_ir.delete_note(state, track_id, note_id)
            except score_ir.StateError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _commit_edit(record, state, int(edited["deleted"]))

    @app.put("/api/jobs/{job_id}/tracks/{track_id}/mute")
    def mute_track(job_id: str, track_id: str, payload: MuteEdit) -> dict:
        record = manager.get(job_id)
        with record.edit_lock:
            state = _edit_context(record, payload.revision)
            try:
                edited = score_ir.set_track_muted(state, track_id, payload.muted)
            except score_ir.StateError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _commit_edit(record, state, int(bool(edited["muted"])))

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="audio2gp FastAPI web UI (localhost only)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="loopback bind address")
    parser.add_argument("--port", type=int, default=7860, help="port (default 7860)")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("AUDIO2GP_OUTPUT_DIR", "output"),
        help="job output root",
    )
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("AUDIO2GP_WORK_DIR", "work"),
        help="job work root",
    )
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=int(os.environ.get("AUDIO2GP_MAX_UPLOAD_MB", "200")),
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=float(os.environ.get("AUDIO2GP_MAX_DURATION_S", "600")),
    )
    args = parser.parse_args(argv)
    if args.host not in LOOPBACK_HOSTS:
        print(
            f"Refusing to bind {args.host!r}: audio2gp is localhost-only. "
            "Use 127.0.0.1."
        )
        return 2
    app = create_app(
        args.output_dir,
        args.work_dir,
        max_upload_mb=args.max_upload_mb,
        max_duration_s=args.max_duration_s,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch
    raise SystemExit(main())
