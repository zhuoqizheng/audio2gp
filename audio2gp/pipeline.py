from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import chords as chords_mod
from . import drums as drums_mod
from . import engines
from . import fingering as fingering_mod
from . import keydetect as keydetect_mod
from . import midi_in
from . import midi_post
from . import preprocess as preprocess_mod
from . import rhythm as rhythm_mod
from . import score_ir
from . import separate as separate_mod
from . import techniques as techniques_mod
from . import transcribe as transcribe_mod
from .config import (
    BASS_TUNINGS,
    GUITAR_TUNINGS,
    PROGRAMS,
    STEM_NAMES,
    TUNING_GUITAR,
    Meter,
    Options,
    resolve_device,
)
from .gp_writer import write_gp5
from .types import (
    ChordEvent,
    FingeredNote,
    Note,
    NoteEvent,
    PreprocessedAudio,
    RunResult,
    TabTrack,
)
from .utils import (
    copy_file,
    ensure_dir,
    get_job_logger,
    new_job_id,
    write_json,
    zip_directory,
)

LocalCB = Callable[[float, str], None]

STAGE_BUDGETS = {
    "preprocess": (0.0, 5.0),
    "rhythm": (5.0, 10.0),
    "separate": (10.0, 50.0),
    "analyze": (50.0, 55.0),
    "transcribe": (55.0, 80.0),
    "post": (80.0, 90.0),
    "gp5": (90.0, 100.0),
}


class PipelineError(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        job_id: str,
        work_dir: Path,
        out_dir: Path,
        log_path: Path,
    ) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message
        self.job_id = job_id
        self.work_dir = work_dir
        self.out_dir = out_dir
        self.log_path = log_path


class _Reporter:
    """Maps a stage-local 0..100 callback into the overall 0..100 budget."""

    def __init__(self, cb: Callable[[str, float, str], None] | None) -> None:
        self.cb = cb
        self.last = -1.0

    def emit_stage(self, stage: str, local_pct: float, message: str) -> None:
        start, end = STAGE_BUDGETS[stage]
        local = max(0.0, min(100.0, float(local_pct)))
        overall = start + (end - start) * local / 100.0
        overall = max(self.last, min(100.0, overall))
        self.last = overall
        if self.cb is None:
            return
        try:
            self.cb(stage, overall, message)
        except Exception:
            # A broken UI callback must never break a long-running job.
            pass

    def stage_cb(self, stage: str) -> LocalCB:
        def callback(pct: float, message: str) -> None:
            self.emit_stage(stage, pct, message)

        return callback


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _guitar_track(
    cleaned: dict[str, list[Note]],
    options: Options,
    bpm: float,
) -> tuple[TabTrack, dict]:
    source = options.guitar_stem
    source_notes = cleaned.get(source, [])
    scores: list[dict[str, float | str]] = []
    if source == "vocals":
        notes, warnings = fingering_mod.arrange_vocals(
            source_notes, max_fret=options.max_fret
        )
        used_tuning = "standard"
    elif options.suggest_tuning:
        used_tuning, scores, notes, warnings = fingering_mod.suggest_tuning(
            source_notes,
            GUITAR_TUNINGS,
            max_fret=options.max_fret,
            kind="guitar",
        )
        if used_tuning != options.guitar_tuning:
            warnings.append(
                f"suggested guitar tuning '{used_tuning}' instead of "
                f"'{options.guitar_tuning}'"
            )
    else:
        used_tuning = options.guitar_tuning
        notes, warnings = fingering_mod.arrange_guitar(
            source_notes,
            max_fret=options.max_fret,
            tuning=GUITAR_TUNINGS[used_tuning],
        )
    techniques: dict[str, int] = {"hammer_on": 0, "pull_off": 0}
    if options.detect_techniques and source != "vocals":
        notes, techniques = techniques_mod.mark_legato(notes, bpm)
    tuning = GUITAR_TUNINGS.get(used_tuning, TUNING_GUITAR)
    track = TabTrack(
        name="Guitar",
        stem=source,
        tuning=[tuning[number] for number in sorted(tuning)],
        notes=notes,
        channel=0,
        instrument=PROGRAMS["guitar"],
        warnings=warnings,
    )
    info = {
        "configured": options.guitar_tuning,
        "used": used_tuning,
        "suggested": used_tuning if options.suggest_tuning else None,
        "candidates": scores,
        "techniques": techniques,
    }
    return track, info


def _bass_track(
    cleaned: dict[str, list[Note]],
    options: Options,
    bpm: float,
) -> tuple[TabTrack, dict]:
    source_notes = cleaned.get("bass", [])
    scores: list[dict[str, float | str]] = []
    if options.suggest_tuning:
        used_tuning, scores, notes, warnings = fingering_mod.suggest_tuning(
            source_notes,
            BASS_TUNINGS,
            max_fret=options.max_fret,
            kind="bass",
        )
        if used_tuning != options.bass_tuning:
            warnings.append(
                f"suggested bass tuning '{used_tuning}' instead of "
                f"'{options.bass_tuning}'"
            )
    else:
        used_tuning = options.bass_tuning
        notes, warnings = fingering_mod.arrange_bass(
            source_notes,
            max_fret=options.max_fret,
            tuning=BASS_TUNINGS[used_tuning],
        )
    tuning = BASS_TUNINGS.get(used_tuning, BASS_TUNINGS["standard"])
    track = TabTrack(
        name="Bass",
        stem="bass",
        tuning=[tuning[number] for number in sorted(tuning)],
        notes=notes,
        channel=2,
        instrument=PROGRAMS["bass"],
        warnings=warnings,
    )
    info = {
        "configured": options.bass_tuning,
        "used": used_tuning,
        "suggested": used_tuning if options.suggest_tuning else None,
        "candidates": scores,
    }
    return track, info


def _drum_track(cleaned: dict[str, list[Note]]) -> TabTrack:
    return TabTrack(
        name="Drums",
        stem="drums",
        tuning=[],
        notes=[],
        is_percussion=True,
        channel=9,
        instrument=PROGRAMS["drums"],
        drum_notes=cleaned.get("drums", []),
    )


def _vocals_track(cleaned: dict[str, list[Note]], options: Options) -> TabTrack:
    notes, warnings = fingering_mod.arrange_vocals(
        cleaned.get("vocals", []), max_fret=options.max_fret
    )
    return TabTrack(
        name="Vocals",
        stem="vocals",
        tuning=[TUNING_GUITAR[number] for number in sorted(TUNING_GUITAR)],
        notes=notes,
        channel=4,
        instrument=PROGRAMS["vocals"],
        warnings=warnings,
    )


def _build_tracks(
    cleaned: dict[str, list[Note]],
    options: Options,
    bpm: float,
) -> tuple[list[TabTrack], dict]:
    tracks: list[TabTrack] = []
    tunings: dict[str, dict] = {}
    if options.guitar_stem in options.selected_stems:
        track, info = _guitar_track(cleaned, options, bpm)
        tracks.append(track)
        tunings["guitar"] = info
    if "bass" in options.selected_stems:
        track, info = _bass_track(cleaned, options, bpm)
        tracks.append(track)
        tunings["bass"] = info
    if "drums" in options.selected_stems:
        tracks.append(_drum_track(cleaned))
    if (
        options.include_vocals
        and "vocals" in options.selected_stems
        and options.guitar_stem != "vocals"
    ):
        tracks.append(_vocals_track(cleaned, options))
    if not tracks:
        raise ValueError("the selected stems produce no GP5 track")
    return tracks, tunings


def _measure_count(
    cleaned: dict[str, list[Note]],
    bpm: float,
    duration_s: float,
    offset_s: float = 0.0,
    *,
    meter: Meter | None = None,
) -> int:
    meter = meter or Meter(4, 4)
    all_notes = [note for notes in cleaned.values() for note in notes]
    max_start = max((note.start_tick for note in all_notes), default=0)
    max_end = max((note.end_tick for note in all_notes), default=0)
    timeline_s = max(0.0, duration_s - offset_s)
    by_duration = max(
        1,
        int(math.ceil(timeline_s / max(1e-9, meter.bar_seconds(bpm)))),
    )
    by_start = max_start // meter.bar_ticks + 1
    # A note ending exactly on a bar line is fully covered by the preceding
    # measure; ceil handles that boundary without adding a blank bar.
    by_end = max(1, int(math.ceil(max_end / meter.bar_ticks)))
    return max(1, by_duration, by_start, by_end)


def _write_midi_files(
    cleaned: dict[str, list[Note]],
    bpm: float,
    out_dir: Path,
    stems: tuple[str, ...] = STEM_NAMES,
    chords: list[ChordEvent] | None = None,
    *,
    time_signature: str = "4/4",
) -> tuple[dict[str, Path], Path]:
    midi_dir = ensure_dir(out_dir / "midi")
    specs = {
        "vocals": (PROGRAMS["vocals"], 2, "Vocals", False),
        "drums": (PROGRAMS["drums"], 9, "Drums", True),
        "bass": (PROGRAMS["bass"], 1, "Bass", False),
        "other": (PROGRAMS["guitar"], 0, "Guitar (other)", False),
    }
    paths: dict[str, Path] = {}
    for stem in stems:
        program, channel, name, is_drum = specs[stem]
        path = midi_dir / f"{stem}.mid"
        midi_post.write_stem_midi(
            cleaned.get(stem, []),
            bpm,
            path,
            program=program,
            channel=channel,
            name=name,
            drums=is_drum,
            time_signature=time_signature,
        )
        paths[stem] = path
    full_path = midi_dir / "full.mid"
    midi_post.write_full_midi(
        cleaned,
        bpm,
        full_path,
        stems=stems,
        chords=chords,
        time_signature=time_signature,
    )
    return paths, full_path


def run(
    input_path: Path | str,
    options: Options | None = None,
    progress_cb: Callable[[str, float, str], None] | None = None,
    *,
    job_id: str | None = None,
) -> RunResult:
    options = options or Options()
    input_path = Path(input_path).expanduser().resolve()
    out_root = Path(options.output_dir).expanduser().resolve()
    work_root = Path(options.work_dir).expanduser().resolve()
    job_id = job_id or new_job_id(input_path)
    work_dir = ensure_dir(work_root / job_id)
    out_dir = ensure_dir(out_root / job_id)
    log_path = work_dir / "pipeline.log"
    logger = get_job_logger(log_path)
    logger.info("job %s input=%s", job_id, input_path)
    logger.info("options: %s", options.to_meta())

    reporter = _Reporter(progress_cb)
    warnings: list[str] = []
    stage_times: dict[str, float] = {}
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    input_size = input_path.stat().st_size if input_path.exists() else 0
    is_midi = input_path.suffix.lower() in midi_in.MIDI_EXTENSIONS
    input_kind = "midi" if is_midi else "audio"

    def error_meta(stage: str, exc: BaseException) -> PipelineError:
        timings["total"] = round(time.perf_counter() - total_start, 3)
        meta = {
            "job_id": job_id,
            "status": "error",
            "stage": stage,
            "error": str(exc),
            "input_kind": input_kind,
            "created": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(input_path),
                "name": input_path.name,
                "size_bytes": input_size,
            },
            "options": options.to_meta(),
            "warnings": _dedupe(warnings),
            "timings": timings,
        }
        try:
            write_json(out_dir / "meta.json", meta)
        except Exception:
            logger.exception("could not write error meta.json")
        return PipelineError(
            stage,
            str(exc),
            job_id=job_id,
            work_dir=work_dir,
            out_dir=out_dir,
            log_path=log_path,
        )

    if not input_path.is_file():
        raise error_meta("preprocess", FileNotFoundError(f"Input file does not exist: {input_path}"))
    if input_size == 0:
        raise error_meta("preprocess", ValueError(f"Input file is empty: {input_path}"))

    midi_events: dict[str, list[NoteEvent]] = {}
    reporter.emit_stage("preprocess", 0, "开始预处理")
    t0 = time.perf_counter()
    if is_midi:
        # A MIDI file is already the transcription; skip WAV conversion too.
        try:
            imported = midi_in.prepare(input_path, options, logger=logger)
        except Exception as exc:
            logger.exception("MIDI import failed")
            raise error_meta("preprocess", exc) from exc
        warnings.extend(imported.warnings)
        midi_events = imported.events
        audio = PreprocessedAudio(
            input_path,
            imported.duration_s,
            samplerate=0,
            bpm=(
                float(options.bpm_override)
                if options.bpm_override is not None
                else imported.bpm
            ),
        )
        reporter.emit_stage("preprocess", 100, "MIDI 读取完成")
    else:
        # Normalize to 44.1 kHz WAV + BPM estimate.
        try:
            audio = preprocess_mod.prepare(
                input_path,
                work_dir,
                options,
                logger=logger,
                progress_cb=reporter.stage_cb("preprocess"),
            )
        except Exception as exc:
            logger.exception("preprocess failed")
            raise error_meta("preprocess", exc) from exc
        warnings.extend(audio.warnings)
        reporter.emit_stage("preprocess", 100, "预处理完成")
    stage_times["preprocess"] = round(time.perf_counter() - t0, 3)
    logger.info("preprocess: bpm=%.2f duration=%.2f", audio.bpm, audio.duration_s)

    # Rhythm phase. The meter is always user-selected in this build. Automatic
    # downbeat analysis is meter-aware: madmom for 4/4, librosa accent phase
    # for every other signature.
    meter = options.meter
    manual_downbeat_s = options.manual_downbeat_s
    if is_midi:
        manual_downbeat_s = None
    elif (
        manual_downbeat_s is not None
        and manual_downbeat_s > audio.duration_s + 0.05
    ):
        warnings.append(
            f"手动强拍位置 {manual_downbeat_s:.2f}s 超出音频时长 "
            f"{audio.duration_s:.2f}s，已忽略"
        )
        manual_downbeat_s = None
    auto_alignment = (
        not is_midi
        and options.align_downbeats
        and manual_downbeat_s is None
    )
    reporter.emit_stage(
        "rhythm",
        0,
        "分析节拍与强拍" if auto_alignment else "设置节奏",
    )
    t0 = time.perf_counter()
    rhythm_info: rhythm_mod.RhythmInfo | None = None
    if is_midi:
        rhythm_mode = "midi"
        reporter.emit_stage("rhythm", 100, "MIDI 输入：沿用文件时间轴")
    elif manual_downbeat_s is not None:
        rhythm_mode = "manual"
        reporter.emit_stage(
            "rhythm", 100, f"手动强拍 {manual_downbeat_s:.2f}s"
        )
    elif options.align_downbeats:
        rhythm_mode = "auto"
        try:
            rhythm_info = rhythm_mod.analyze_rhythm(
                audio.path,
                logger=logger,
                progress_cb=reporter.stage_cb("rhythm"),
                meter=meter,
            )
        except Exception as exc:  # defensive: analyze_rhythm is non-fatal itself
            logger.exception("rhythm analysis failed; continuing without alignment")
            warnings.append(f"强拍分析失败；第 1 小节从音频开头开始：{exc}")
            rhythm_mode = "auto-failed"
        if rhythm_info is not None:
            warnings.extend(rhythm_info.warnings)
            if (
                options.bpm_override is None
                and rhythm_info.bpm > 0
                and rhythm_info.downbeat_times_s
            ):
                audio.bpm = float(rhythm_info.bpm)
        reporter.emit_stage(
            "rhythm", 100, "节拍分析完成" if rhythm_info else "未对齐强拍"
        )
    else:
        rhythm_mode = "off"
        reporter.emit_stage("rhythm", 100, "已关闭强拍对齐")
    stage_times["rhythm"] = round(time.perf_counter() - t0, 3)
    logger.info(
        "rhythm: meter=%s mode=%s method=%s bpm=%.2f first_downbeat=%s manual=%s",
        meter.label,
        rhythm_mode,
        "midi" if is_midi else rhythm_info.method if rhythm_info else "off",
        audio.bpm,
        rhythm_info.first_downbeat_s if rhythm_info else None,
        manual_downbeat_s,
    )

    t0 = time.perf_counter()
    playback_stems: dict[str, Path] = {}
    if is_midi:
        used_device = "midi"
        stems_work: dict[str, Path] = {}
        selected = tuple(
            name
            for name in STEM_NAMES
            if name in options.selected_stems and name in midi_events
        )
        if not selected:
            raise error_meta(
                "separate",
                ValueError(
                    "MIDI 中没有与所选声部匹配的音符；请勾选文件中存在的声部"
                ),
            )
        reporter.emit_stage("separate", 100, "MIDI 输入：跳过分轨")
        logger.info("MIDI input; selected stems: %s", selected)
    else:
        # Resolve the device before Demucs starts; a missing torch is a setup error.
        try:
            device, device_warnings = resolve_device(options.device)
        except Exception as exc:
            logger.exception("device resolution failed")
            raise error_meta("separate", exc) from exc
        warnings.extend(device_warnings)
        logger.info("device: %s", device)

        # Demucs separation.
        reporter.emit_stage("separate", 0, "开始分离")
        try:
            stems_work, sep_warnings, used_device = separate_mod.separate(
                audio,
                options,
                device,
                logger=logger,
                progress_cb=reporter.stage_cb("separate"),
            )
        except Exception as exc:
            logger.exception("separation failed")
            raise error_meta("separate", exc) from exc
        warnings.extend(sep_warnings)
        selected = options.selected_stems
        missing = [name for name in selected if name not in stems_work]
        if missing:
            raise error_meta(
                "separate",
                RuntimeError(f"Demucs did not return: {', '.join(missing)}"),
            )
        logger.info("separation device: %s; selected stems: %s", used_device, selected)

        playback_stems = {
            name: Path(stems_work[name]).resolve() for name in selected
        }
        if options.keep_stems:
            try:
                for name in selected:
                    playback_stems[name] = copy_file(
                        stems_work[name],
                        out_dir / "stems" / f"{name}.wav",
                    ).resolve()
            except Exception as exc:
                logger.exception("could not copy stems to output")
                raise error_meta("separate", exc) from exc
        reporter.emit_stage("separate", 100, "分轨完成")
    stage_times["separate"] = round(time.perf_counter() - t0, 3)

    # Musical context from the separated stems: the drum stem is a cleaner
    # tempo reference than the full mix, and harmonic stems give a useful key.
    reporter.emit_stage(
        "analyze", 0, "读取 MIDI 时间轴" if is_midi else "分析速度与调性"
    )
    t0 = time.perf_counter()
    tempo_source = (
        "midi"
        if is_midi
        else rhythm_info.tempo_source if rhythm_info is not None else "mix"
    )
    if options.bpm_override is not None:
        tempo_source = "override"
    elif (
        options.tempo_from_drums
        and options.bpm_override is None
        and "drums" in stems_work
    ):
        refined_bpm, tempo_warnings = rhythm_mod.refine_tempo_from_drums(
            stems_work["drums"],
            audio.bpm,
            logger=logger,
        )
        warnings.extend(tempo_warnings)
        tempo_source = "drums" if abs(refined_bpm - audio.bpm) > 0.005 else "drums-verified"
        audio.bpm = float(refined_bpm)
        if rhythm_info is not None:
            rhythm_info.bpm = audio.bpm
            rhythm_info.tempo_source = tempo_source
        logger.info("tempo: %.2f BPM (source=%s)", audio.bpm, tempo_source)
    reporter.emit_stage(
        "analyze", 45, "MIDI 输入：跳过调性分析" if is_midi else "估计调性"
    )

    key_info: keydetect_mod.KeyInfo | None = None
    if options.detect_key and not is_midi:
        harmonic_sources = [
            stems_work[name]
            for name in ("other", "bass", "vocals")
            if name in stems_work and Path(stems_work[name]).is_file()
        ]
        if harmonic_sources:
            try:
                key_info = keydetect_mod.detect_key(
                    harmonic_sources,
                    warnings=warnings,
                )
            except Exception as exc:  # defensive: detection must stay optional
                logger.exception("key detection failed")
                warnings.append(f"key detection failed ({exc}); GP5 stays in C major")
        elif not is_midi:
            warnings.append("key detection skipped: no harmonic stems available")
    if is_midi:
        warnings.append("MIDI 输入：跳过调性检测、鼓速校正与音频和弦分析")
    key_meta = {
        "enabled": bool(options.detect_key) and not is_midi,
        "detected": key_info is not None,
        "name": key_info.name if key_info is not None else None,
        "tonic": int(key_info.tonic) if key_info is not None else None,
        "mode": key_info.mode if key_info is not None else None,
        "confidence": float(key_info.confidence) if key_info is not None else None,
        "gp_key": key_info.gp_key_name if key_info is not None else "CMajor",
        "method": key_info.method if key_info is not None else None,
    }
    stage_times["analyze"] = round(time.perf_counter() - t0, 3)
    reporter.emit_stage(
        "analyze",
        100,
        "MIDI 输入：跳过速度/调性分析"
        if is_midi
        else f"调性：{key_info.name}"
        if key_info is not None
        else "调性未确定（保持 C 大调）",
    )

    # Basic Pitch (vocals/bass/other) plus the drum heuristic. Only selected
    # source stems are transcribed; unselected Demucs stems remain in work/.
    melodic_selected = tuple(
        name for name in ("vocals", "bass", "other") if name in selected
    )
    drums_selected = "drums" in selected
    melodic_span = 75.0 if melodic_selected and drums_selected else 100.0
    reporter.emit_stage("transcribe", 0, "开始转录")
    t0 = time.perf_counter()

    def melodic_progress(pct: float, message: str) -> None:
        reporter.emit_stage(
            "transcribe",
            melodic_span * min(100.0, float(pct)) / 100.0,
            message,
        )

    def drum_progress(pct: float, message: str) -> None:
        base = melodic_span if melodic_selected and drums_selected else 0.0
        span = 100.0 - base
        reporter.emit_stage(
            "transcribe",
            base + span * min(100.0, float(pct)) / 100.0,
            message,
        )

    melodic_events: dict[str, list[NoteEvent]] = {}
    drum_events: list[NoteEvent] = []
    tr_warnings: list[str] = []
    drum_warnings: list[str] = []
    if is_midi:
        melodic_events = {
            name: list(midi_events.get(name, [])) for name in melodic_selected
        }
        if drums_selected:
            drum_events = list(midi_events.get("drums", []))
        reporter.emit_stage("transcribe", 100, "MIDI 输入：跳过转录")
    else:
        if melodic_selected:
            melodic_stems = {name: stems_work[name] for name in melodic_selected}
            try:
                if options.transcription_engine == "external":
                    try:
                        melodic_events, tr_warnings = engines.transcribe_external(
                            melodic_stems,
                            options,
                            audio.bpm,
                            logger=logger,
                            progress_cb=melodic_progress,
                        )
                    except Exception as exc:
                        logger.exception("external transcription engine failed")
                        warnings.append(
                            f"外部转录引擎失败，已回退 Basic Pitch：{exc}"
                        )
                        melodic_events, tr_warnings = transcribe_mod.transcribe_stems(
                            melodic_stems,
                            options,
                            audio.bpm,
                            logger=logger,
                            progress_cb=melodic_progress,
                        )
                else:
                    melodic_events, tr_warnings = transcribe_mod.transcribe_stems(
                        melodic_stems,
                        options,
                        audio.bpm,
                        logger=logger,
                        progress_cb=melodic_progress,
                    )
            except Exception as exc:
                logger.exception("Basic Pitch transcription failed")
                raise error_meta("transcribe", exc) from exc
        elif drums_selected:
            reporter.emit_stage("transcribe", 0, "未选择旋律分轨，跳过 Basic Pitch")
        warnings.extend(tr_warnings)

        if drums_selected:
            try:
                drum_events, drum_warnings = drums_mod.transcribe_drums(
                    stems_work["drums"],
                    audio.bpm,
                    options.quantize,
                    logger=logger,
                    progress_cb=drum_progress,
                )
            except Exception as exc:
                logger.exception("drum transcription failed; continuing with empty drums")
                warnings.append(f"drum transcription failed; wrote an empty drum track: {exc}")
                drum_events = []
                drum_warnings = []
        elif melodic_selected:
            reporter.emit_stage("transcribe", 100, "未选择鼓轨，跳过鼓点转录")
        warnings.extend(drum_warnings)
        reporter.emit_stage("transcribe", 100, "转录完成")
    events = {
        name: list(melodic_events.get(name, [])) for name in melodic_selected
    }
    if drums_selected:
        events["drums"] = list(drum_events)
    stage_times["transcribe"] = round(time.perf_counter() - t0, 3)

    # Snap the bar grid to the detected downbeats. If transcription found
    # material before the first downbeat, choose_grid_offset keeps it in
    # leading pickup bars instead of discarding it.
    all_events = [event for event_list in events.values() for event in event_list]
    first_note_s = min(
        (float(event.start_s) for event in all_events if event.end_s > event.start_s),
        default=None,
    )
    grid_offset_s = 0.0
    reference_downbeat_s: float | None = None
    if manual_downbeat_s is not None:
        reference_downbeat_s = float(manual_downbeat_s)
    elif rhythm_info is not None and rhythm_info.downbeat_times_s:
        reference_downbeat_s = rhythm_info.first_downbeat_s
    if reference_downbeat_s is not None:
        grid_offset_s = rhythm_mod.choose_grid_offset(
            reference_downbeat_s,
            first_note_s,
            audio.bpm,
            beats_per_bar=meter.numerator,
            denominator=meter.denominator,
        )
    logger.info(
        "alignment: meter=%s first_note=%.3f reference_downbeat=%s offset=%.3f",
        meter.label,
        first_note_s if first_note_s is not None else -1.0,
        reference_downbeat_s,
        grid_offset_s,
    )

    # Quantize/clean, optional fingering, chord analysis, and MIDI.
    reporter.emit_stage("post", 0, "整理音符")
    t0 = time.perf_counter()
    chords: list[ChordEvent] = []
    chords_doc: dict = {}
    chords_path: Path | None = None
    try:
        cleaned = midi_post.clean(
            events,
            options,
            audio.bpm,
            logger=logger,
            offset_s=grid_offset_s,
        )
        cleaned = {name: cleaned.get(name, []) for name in selected}
        note_counts = {name: len(cleaned[name]) for name in selected}
        tracks: list[TabTrack] = []
        tuning_meta: dict = {}
        if options.export_gp5:
            reporter.emit_stage("post", 25, "计算指法")
            tracks, tuning_meta = _build_tracks(cleaned, options, audio.bpm)
            for track in tracks:
                warnings.extend(track.warnings)
        else:
            reporter.emit_stage("post", 25, "仅 MIDI：跳过指法与 GP5")
        measure_count = _measure_count(
            cleaned,
            audio.bpm,
            audio.duration_s,
            offset_s=grid_offset_s,
            meter=meter,
        )

        if options.detect_chords and not is_midi:
            reporter.emit_stage("post", 50, "分析和弦")
            harmonic_stems = {
                name: stems_work[name]
                for name in ("other", "bass", "vocals")
                if name in selected
                and name in stems_work
                and Path(stems_work[name]).is_file()
            }
            if harmonic_stems:
                def chord_progress(pct: float, message: str) -> None:
                    local = 50.0 + 35.0 * max(0.0, min(100.0, float(pct))) / 100.0
                    reporter.emit_stage("post", local, message)

                try:
                    chords = chords_mod.analyze_chords(
                        harmonic_stems,
                        audio.bpm,
                        offset_s=grid_offset_s,
                        duration_s=audio.duration_s,
                        key_info=key_info,
                        warnings=warnings,
                        progress_cb=chord_progress,
                    )
                except Exception as exc:  # optional analysis must not fail the job
                    logger.exception("chord analysis failed")
                    warnings.append(f"和弦分析失败；继续导出 MIDI/GP5：{exc}")
                    chords = []
            else:
                warnings.append(
                    "和弦分析跳过：所选声部中没有 other、bass 或 vocals 和声声部"
                )
            chords_doc = chords_mod.chords_to_dict(
                chords,
                bpm=audio.bpm,
                offset_s=grid_offset_s,
                duration_s=audio.duration_s,
                key=key_meta,
                time_signature=meter.label,
            )
            try:
                chords_path = out_dir / "chords.json"
                write_json(chords_path, chords_doc)
            except Exception as exc:
                chords_path = None
                logger.exception("could not write chords.json")
                warnings.append(f"和弦结果写入失败；MIDI/GP5 不受影响：{exc}")
            logger.info(
                "chords: %d segments; progression=%s",
                len(chords),
                chords_doc["summary"]["progression"],
            )
        elif is_midi:
            reporter.emit_stage("post", 85, "MIDI 输入：跳过和弦分析")
        else:
            reporter.emit_stage("post", 85, "和弦分析已关闭")

        reporter.emit_stage("post", 90, "写入 MIDI")
        midi_paths, full_midi = _write_midi_files(
            cleaned,
            audio.bpm,
            out_dir,
            stems=selected,
            chords=chords if options.detect_chords else None,
            time_signature=meter.label,
        )
        # Snapshot the untouched transcription so the editor can compare edits
        # against the reference after full.mid has been regenerated. For MIDI
        # input the uploaded file itself is that reference.
        copy_file(
            input_path if is_midi else full_midi,
            out_dir / "midi" / "reference.mid",
        )
    except Exception as exc:
        logger.exception("MIDI cleanup/post-processing failed")
        raise error_meta("post", exc) from exc
    stage_times["post"] = round(time.perf_counter() - t0, 3)
    logger.info("note counts: %s; measures: %d", note_counts, measure_count)
    reporter.emit_stage("post", 100, "后处理完成")

    # GP5. Failure here is explicitly a partial success in full-output mode.
    reporter.emit_stage("gp5", 0, "写入 GP5" if options.export_gp5 else "仅 MIDI：跳过 GP5")
    t0 = time.perf_counter()
    gp5_path: Path | None = None
    gp5_error: str | None = None
    if options.export_gp5:
        gp5_path = out_dir / "song.gp5"
        try:
            gp5_warnings = write_gp5(
                tracks,
                audio.bpm,
                measure_count,
                gp5_path,
                layout=options.layout,
                quantize=options.quantize,
                key_name=key_info.gp_key_name if key_info is not None else None,
                chords=chords if options.detect_chords else None,
                time_signature=meter.label,
            )
            warnings.extend(gp5_warnings)
            for track in tracks:
                if track.is_percussion and not track.drum_notes:
                    warnings.append("drum track is empty")
            logger.info("GP5 written: %s", gp5_path)
        except Exception as exc:
            gp5_path = None
            gp5_error = str(exc)
            logger.exception("GP5 generation failed; delivering partial success")
            warnings.append(f"GP5 generation failed; MIDI and stems are still available: {exc}")
            try:
                (out_dir / "song.gp5").unlink(missing_ok=True)
            except Exception:
                pass
    stage_times["gp5"] = round(time.perf_counter() - t0, 3)
    if not options.export_gp5:
        reporter.emit_stage("gp5", 100, "MIDI-only 完成")
    else:
        reporter.emit_stage("gp5", 100, "GP5 完成" if gp5_path else "GP5 失败，已保留 MIDI")

    # Browser editor state. It is deliberately a separate artifact from the
    # batch pipeline: the API/UI mutate this IR and regenerate GP5/MIDI through
    # the same export functions used here, never by touching pipeline code.
    state: dict = {}
    state_error: str | None = None
    if options.export_gp5:
        try:
            state = score_ir.build_state(
                tracks,
                audio.bpm,
                measure_count,
                job_id=job_id,
                quantize=options.quantize,
                layout=options.layout,
                max_fret=options.max_fret,
                key=key_meta,
                chords=chords,
                time_signature=meter.label,
            )
            score_ir.validate_state(state)
            write_json(out_dir / "tracks.json", state)
        except Exception as exc:
            state = {}
            state_error = str(exc)
            logger.exception("could not write editable score state")
            warnings.append(f"在线编辑状态生成失败；仍可下载 MIDI/GP5：{exc}")

    warnings = _dedupe(warnings)
    status = "ok" if (not options.export_gp5 or gp5_path) else "partial"
    timings.update(stage_times)
    timings["total"] = round(time.perf_counter() - total_start, 3)

    zip_name = f"audio2gp_{job_id}.zip"
    files: dict[str, object] = {
        "midi": {name: f"midi/{name}.mid" for name in selected},
        "full_midi": "midi/full.mid",
        "reference_midi": "midi/reference.mid",
        "meta": "meta.json",
        "zip": zip_name,
    }
    if gp5_path is not None:
        files["gp5"] = "song.gp5"
    if state:
        files["tracks"] = "tracks.json"
    if chords_path is not None:
        files["chords"] = "chords.json"
    if options.keep_stems and playback_stems:
        files["stems"] = {name: f"stems/{name}.wav" for name in selected}

    manual_reference = manual_downbeat_s if rhythm_mode == "manual" else None
    rhythm_meta = {
        "enabled": bool(options.align_downbeats),
        "mode": rhythm_mode,
        "method": (
            "midi" if is_midi
            else "manual" if rhythm_mode == "manual"
            else rhythm_info.method if rhythm_info
            else "off"
        ),
        "time_signature": meter.label,
        "beats_per_bar": meter.pulses_per_bar,
        "beat_unit": meter.denominator,
        "bar_ticks": meter.bar_ticks,
        "bpm": (
            round(float(rhythm_info.bpm), 3)
            if rhythm_info is not None and rhythm_info.bpm > 0
            else None
        ),
        "tempo_source": tempo_source,
        "first_downbeat_s": (
            round(float(manual_reference), 4)
            if manual_reference is not None
            else round(float(rhythm_info.first_downbeat_s), 4)
            if rhythm_info is not None and rhythm_info.first_downbeat_s is not None
            else None
        ),
        "first_note_s": (
            round(float(first_note_s), 4) if first_note_s is not None else None
        ),
        "grid_offset_s": round(float(grid_offset_s), 4),
        "downbeat_count": (
            len(rhythm_info.downbeat_times_s) if rhythm_info else 0
        ),
        "auto_alignment_supported": True,
    }

    meta = {
        "job_id": job_id,
        "status": status,
        "created": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(input_path),
            "name": input_path.name,
            "size_bytes": input_size,
        },
        "audio": {
            "bpm": round(float(audio.bpm), 3),
            "tempo_source": tempo_source,
            "duration_s": round(float(audio.duration_s), 3),
            "sample_rate": audio.samplerate,
        },
        "options": options.to_meta(),
        "output_format": options.output_format,
        "input_kind": input_kind,
        "device": used_device,
        "selected_stems": list(selected),
        "time_signature": meter.label,
        "bar_ticks": meter.bar_ticks,
        "rhythm": rhythm_meta,
        "key": key_meta,
        "warnings": warnings,
        "note_counts": note_counts,
        "measure_count": measure_count,
        "chords": chords_doc if options.detect_chords else {},
        "tracks": [track.name for track in tracks],
        "tunings": tuning_meta,
        "revision": 0,
        "timings": timings,
        "files": files,
    }
    if gp5_error:
        meta["gp5_error"] = gp5_error
    if state_error:
        meta["state_error"] = state_error
    zip_path = out_dir / zip_name
    try:
        write_json(out_dir / "meta.json", meta)
        zip_directory(out_dir, zip_path)
    except Exception as exc:
        logger.exception("finalize failed")
        raise error_meta("finalize", exc) from exc
    reporter.emit_stage("gp5", 100, "完成")
    logger.info(
        "done status=%s bpm=%.2f device=%s gp5=%s zip=%s timings=%s",
        status,
        audio.bpm,
        used_device,
        gp5_path,
        zip_path,
        timings,
    )
    return RunResult(
        job_id=job_id,
        status=status,
        out_dir=out_dir,
        work_dir=work_dir,
        stems=playback_stems,
        midi=midi_paths,
        full_midi=full_midi,
        gp5=gp5_path,
        zip_path=zip_path,
        meta=meta,
        warnings=warnings,
        note_counts=note_counts,
        timings=timings,
        device=used_device,
        bpm=float(audio.bpm),
        duration_s=float(audio.duration_s),
        rhythm=rhythm_meta,
        key=key_meta,
        state=state,
        chords=chords,
        chords_path=chords_path,
        output_format=options.output_format,
    )
