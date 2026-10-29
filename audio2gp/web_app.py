from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from .config import (  # noqa: E402
    STEM_NAMES,
    TIME_SIGNATURE_CHOICES,
    TIME_SIGNATURE_LABELS,
    Options,
)
from .pipeline import PipelineError, run  # noqa: E402

OUTPUT_ROOT = Path(os.environ.get("AUDIO2GP_OUTPUT_DIR", "output")).expanduser().resolve()
WORK_ROOT = Path(os.environ.get("AUDIO2GP_WORK_DIR", "work")).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
WORK_ROOT.mkdir(parents=True, exist_ok=True)

JOB_LOCK = threading.Lock()
STAGE_LABELS = {
    "preprocess": "预处理",
    "rhythm": "强拍分析",
    "separate": "分轨",
    "analyze": "速度/调性分析",
    "transcribe": "转录",
    "post": "后处理",
    "gp5": "GP5",
}
STEM_LABELS = {
    "vocals": "人声",
    "drums": "鼓",
    "bass": "贝斯",
    "other": "吉他/其他",
}
STEM_CHOICES = [
    ("吉他/其他", "other"),
    ("人声", "vocals"),
    ("贝斯", "bass"),
    ("鼓", "drums"),
]

INITIAL_SUMMARY = "### 结果\n等待任务。上传音频后点击「开始转换」。"
INITIAL_LOG = (
    "等待任务。\n"
    "所有处理都在本机完成；首次运行会加载（必要时下载）Demucs 与强拍模型权重。\n"
    "拍号由你选择（本工具不做自动拍号识别）；默认 4/4。\n"
    "4/4 会自动按检测到的强拍对齐小节，其他拍号请填写手动强拍位置。\n"
)

HELP_MARKDOWN = """\
### 处理流程

1. **预处理**：把上传文件统一为 44.1 kHz 立体声 WAV，并估计速度。
2. **节奏与强拍**：拍号由你在参数中指定，不做自动拍号识别。选 4/4 且开启「按检测到的强拍对齐小节」时，在本机用 madmom 强拍模型检测拍点与小节起点；其他拍号不会自动检测，可填写手动强拍位置。
3. **Demucs 分轨**：在后台分离出人声、鼓、贝斯、吉他/其他四条源分轨；只有你勾选的分轨会继续转录和输出。
4. **转录**：选中的旋律分轨用 Basic Pitch 转成音高；选中的鼓轨用频谱与起音启发式映射到 GM 鼓件。
5. **量化与对齐**：把音符吸附到量化网格，并让第 1 小节第 1 拍落在检测到的强拍上。
6. **指法与输出**：搜索吉他和贝斯把位，写入选中分轨的 MIDI、`full.mid` 与可编辑的 `song.gp5`。

### 分轨选择

- 勾选「原曲包含的乐器」决定哪些源分轨被转录、写入 MIDI/full.mid、写入 GP5，以及复制到输出和 ZIP。
- Demucs 仍然会在后台分离四条源分轨，以保证分离质量；未勾选的轨不会出现在最终结果里（仍保留在工作目录，便于排查）。
- 例子：原曲只有吉他和人声时，只勾选「吉他/其他」和「人声」，GP5 就只生成 Guitar 和 Vocals 两轨。
- 吉他轨使用「吉他轨来源」指定的分轨；要让吉他轨出现，来源分轨必须被勾选。例如把来源设为 `vocals` 时，必须勾选「人声」。
- 至少勾选一个分轨；勾选组合必须能产生至少一条 GP5 轨，否则会给出参数错误。
- 「GP5 增加人声旋律轨」仍然控制是否额外写入独立的人声旋律轨；当吉他轨来源就是 `vocals` 时，默认不再重复生成同样的人声轨。

### 拍号、shuffle 与小节对齐

- **拍号由用户指定**：可选 4/4、3/4、2/4、6/8、9/8、12/8。拍号决定小节长度、MIDI/GP5 的拍号标记和小节线位置；本工具不做自动拍号识别。
- **BPM 一律按四分音符计算**（MIDI 标准）。6/8、9/8、12/8 的一小节分别是 2、3、4 个附点四分音符拍；实际听感仍是每小节 2/3/4 个大拍。
- **shuffle 不是一种拍号**，而是一种演奏感觉。常见写法有两种：4/4 加摇摆八分（本工具可把量化网格设为「1/8 三连音」），或者直接记成 12/8（推荐，记谱更干净）。慢速布鲁斯也常用 12/8。
- **自动强拍分析只支持 4/4**：这与 madmom 模型按四拍一小节解码有关。其他拍号若开启对齐，会给出警告并让第 1 小节从音频开头开始；请填写「手动强拍位置」。
- **手动强拍位置**：填写音频中第一个强拍出现的时间（秒）。填写后不再运行自动强拍检测，适用于所有拍号。第一个强拍之前已有音符（弱起 / pickup）时，弱起保留在前置小节，第一个强拍作为第 2 小节（或更后）的第 1 拍。
- 没有手动值且自动分析成功时：第一个检测到的强拍就是第 1 小节第 1 拍。
- 自动分析失败或关闭对齐时：第 1 小节从音频开头开始，其它步骤照常进行。
- 如果小节线整体错位，通常是强拍检测错误：可以先用 `htdemucs_ft` 重跑、填写手动强拍位置，或填入实际 BPM 后再转换。BPM 覆盖只改速度和网格，不改强拍相位。

### 参数建议

- **分轨选择**：按原曲实际乐器勾选；例如吉他+人声只勾两项，贝斯/鼓独奏可以只勾一项。
- **Demucs 模型**：`htdemucs_ft` 分离质量更好但约慢 4 倍；`htdemucs` 适合快速预览。
- **拍号**：按原曲选择。普通歌曲选 4/4；华尔兹/三拍子选 3/4；进行曲选 2/4；三连音感的 shuffle、慢布鲁斯优先选 12/8；复合二拍可选 6/8。拍号必须由你知道或听出，工具不会替你判断。
- **量化网格**：表示音符时值，与拍号无关。默认 1/16；速度快的独奏可试 1/32，慢歌可试 1/8；shuffle/swing 的八分请选「1/8 三连音」（12）。
- **手动强拍位置**：知道第一个小节第 1 拍的时间时填写（秒），可替代自动检测并支持所有拍号；`0` 表示不使用。
- **吉他轨来源**：`other` 通常包含吉他、键盘等；`vocals` 会生成单音人声旋律。
- **最大品格**：限制指法搜索范围；原曲高把位独奏可调到 19–24。
- **BPM 覆盖**：自动估计明显错误时填写实际 BPM；`0` 表示自动。
- **保留 stems**：关闭后仍可在页面试听，但输出目录与 ZIP 不含 WAV，节省磁盘。

### 输出是什么

- `stems/*.wav`：选中分轨的 Demucs 分离结果，可继续在 DAW 里使用。
- `midi/*.mid`：选中分轨各自的 MIDI 与合并的 `full.mid`，速度、拍号已写入。
- `song.gp5`：Guitar Pro 5 草稿，只包含选中分轨能生成的乐器轨。
- `meta.json`：BPM、强拍、选中分轨、选项、警告、耗时等信息。
- ZIP：上述文件的打包。

**注意**：结果是可编辑草稿，不是专业扒谱。鼓点、指法、八度与弱起判断都可能有近似或错误；建议在 Guitar Pro / TuxGuitar 中核对后修改。
"""

OUTPUT_NOTE = """\
**下载内容**：选中分轨各自的 MIDI、合并的 `full.mid`、`song.gp5`。ZIP 还包含选中分轨的 stems WAV（除非关闭保留）和 `meta.json`。
若 GP5 显示生成失败，MIDI 与 stems 仍可下载；详情见结果里的警告与运行日志。
"""

CSS = """
.small-note {
    font-size: 0.85rem;
    line-height: 1.5;
    color: var(--body-text-color-subdued);
    margin-top: -0.25rem;
}
.help-box {
    font-size: 0.92rem;
    line-height: 1.6;
}
"""


def _zip_update(path: Path | str | None) -> dict:
    if path is None:
        return gr.update(value=None, interactive=False)
    return gr.update(value=str(path), interactive=True)


def _summary(result) -> str:
    counts = "，".join(f"{STEM_LABELS.get(k, k)} {v}" for k, v in result.note_counts.items())
    selected = result.meta.get("selected_stems") or list(result.stems)
    selected_text = "、".join(STEM_LABELS.get(str(name), str(name)) for name in selected)
    tracks = "、".join(result.meta.get("tracks", []))
    rhythm = result.rhythm or result.meta.get("rhythm") or {}
    signature = str(rhythm.get("time_signature") or "4/4")
    lines = [
        f"**状态**: `{result.status}`",
        f"**BPM**: {result.bpm:.2f}   **拍号**: {signature}   "
        f"**设备**: {result.device}   **时长**: {result.duration_s:.1f}s",
        f"**分轨**: {selected_text}",
        f"**音符数**: {counts}",
        f"**GP5 轨道**: {tracks}",
    ]
    if rhythm.get("mode") == "manual":
        offset = rhythm.get("grid_offset_s")
        pickup = "，含弱起小节" if offset is not None and float(offset) < -1e-6 else ""
        lines.append(
            f"**小节对齐**: 手动强拍 {float(rhythm['first_downbeat_s']):.2f}s"
            f"（{signature}）{pickup}"
        )
    elif not rhythm.get("enabled"):
        lines.append("**小节对齐**: 关闭，第 1 小节从音频开头开始")
    elif rhythm.get("mode") == "unsupported-meter":
        lines.append(
            f"**小节对齐**: 自动分析仅支持 4/4，{signature} 从音频开头开始；"
            "可在「手动强拍位置（秒）」填写第 1 小节第 1 拍"
        )
    elif rhythm.get("first_downbeat_s") is None:
        lines.append("**小节对齐**: 未检测到强拍，第 1 小节从音频开头开始")
    else:
        method = {
            "madmom-infer": "madmom 强拍模型",
            "librosa-accent": "librosa 重音估计",
        }.get(str(rhythm.get("method")), str(rhythm.get("method")))
        offset = rhythm.get("grid_offset_s")
        pickup = "，含弱起小节" if offset is not None and float(offset) < -1e-6 else ""
        lines.append(
            f"**小节对齐**: {signature} 强拍 {float(rhythm['first_downbeat_s']):.2f}s"
            f"（{method}）{pickup}"
        )
    lines.append(f"**输出目录**: `{result.out_dir}`")
    if result.gp5 is not None:
        lines.append(f"**GP5**: `{result.gp5.name}`")
    else:
        lines.append("**GP5**: 生成失败（仍可下载 MIDI / stems）")
    if result.warnings:
        lines.append("")
        lines.append("**警告**:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n\n".join(lines)


def _error_summary(message: str) -> str:
    return "### 失败\n\n" + message.replace("\n", "\n\n")


def _run_job(
    audio_path: str | None,
    model: str,
    quantize: int,
    guitar_stem: str,
    max_fret: int,
    bpm: float,
    align_downbeats: bool,
    keep_stems: bool,
    include_vocals: bool,
    device: str,
    selected_stems: list[str] | tuple[str, ...] | None,
    time_signature: str = "4/4",
    manual_downbeat: float = 0.0,
    progress: gr.Progress = gr.Progress(),
):
    """Generator handler.

    The pipeline runs in a daemon worker, never in the UI thread. The handler
    drains a queue and yields complete output tuples so Gradio can stream
    progress without the pipeline touching Gradio objects.
    """
    stems: dict[str, str | None] = {name: None for name in STEM_NAMES}
    files: list[str] = []
    zip_update = _zip_update(None)
    summary = INITIAL_SUMMARY
    log_lines = [INITIAL_LOG.rstrip()]
    log_lines.append("准备任务…")

    def emit_state() -> tuple:
        return (
            stems["vocals"],
            stems["drums"],
            stems["bass"],
            stems["other"],
            list(files),
            zip_update,
            summary,
            "\n".join(log_lines[-80:]),
        )

    if not audio_path:
        log_lines.append("错误：请先上传音频文件。")
        summary = _error_summary("请先上传音频文件。")
        yield emit_state()
        return

    if not JOB_LOCK.acquire(blocking=False):
        log_lines.append("已有任务正在运行；请等待完成后再试。")
        summary = _error_summary("已有任务正在运行，请等待完成。")
        yield emit_state()
        return

    try:
        try:
            options = Options(
                output_dir=OUTPUT_ROOT,
                work_dir=WORK_ROOT,
                demucs_model=str(model),
                quantize=int(quantize),
                guitar_stem=str(guitar_stem),
                keep_stems=bool(keep_stems),
                max_fret=int(max_fret),
                device=str(device),
                include_vocals=bool(include_vocals),
                bpm_override=(float(bpm) if bpm is not None and float(bpm) > 0 else None),
                time_signature=str(time_signature or "4/4"),
                manual_downbeat_s=(
                    float(manual_downbeat)
                    if manual_downbeat is not None and float(manual_downbeat) > 0
                    else None
                ),
                align_downbeats=bool(align_downbeats),
                selected_stems=tuple(selected_stems or ()),
                layout="auto",
            )
        except Exception as exc:
            log_lines.append(f"参数错误：{exc}")
            summary = _error_summary(f"参数错误：{exc}")
            yield emit_state()
            return
        log_lines.append(
            "分轨: "
            + "、".join(STEM_LABELS.get(name, name) for name in options.selected_stems)
        )

        q: queue.Queue = queue.Queue()

        def progress_cb(stage: str, pct: float, message: str) -> None:
            q.put(("progress", stage, pct, message))

        def worker() -> None:
            try:
                result = run(audio_path, options, progress_cb=progress_cb)
                q.put(("done", result))
            except BaseException as exc:  # noqa: BLE001 - conveyed to the handler thread
                q.put(("error", exc))

        thread = threading.Thread(target=worker, name="audio2gp-job", daemon=True)
        thread.start()
        log_lines.append("任务已启动。")
        yield emit_state()

        last_pct = 0.0
        last_stage = "preprocess"
        last_yield = time.monotonic()
        while True:
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() - last_yield >= 2.0:
                    try:
                        progress(last_pct / 100.0, desc=f"{STAGE_LABELS.get(last_stage, last_stage)}")
                    except Exception:
                        pass
                    last_yield = time.monotonic()
                    yield emit_state()
                continue

            kind = item[0]
            if kind == "progress":
                _kind, stage, pct, message = item
                last_stage = str(stage)
                last_pct = max(0.0, min(100.0, float(pct)))
                label = STAGE_LABELS.get(last_stage, last_stage)
                log_lines.append(f"{last_pct:5.1f}%  {label}  {message}")
                summary = (
                    f"### 处理中\n\n**阶段**: {label}  \n"
                    f"**进度**: {last_pct:.1f}%  \n**输出目录**: `{OUTPUT_ROOT}`"
                )
                try:
                    progress(last_pct / 100.0, desc=f"{label}: {message}")
                except Exception:
                    pass
                last_yield = time.monotonic()
                yield emit_state()
                continue

            if kind == "done":
                result = item[1]
                for name in STEM_NAMES:
                    path = result.stems.get(name)
                    stems[name] = str(path) if path else None
                files = [str(result.midi[name]) for name in result.midi]
                files.append(str(result.full_midi))
                if result.gp5 is not None:
                    files.append(str(result.gp5))
                zip_update = _zip_update(result.zip_path)
                summary = _summary(result)
                rhythm = result.rhythm or {}
                signature = str(rhythm.get("time_signature") or "4/4")
                if rhythm.get("mode") == "manual":
                    alignment = f"手动强拍 {float(rhythm['first_downbeat_s']):.2f}s"
                elif not rhythm.get("enabled"):
                    alignment = "关闭（第 1 小节从音频开头开始）"
                elif rhythm.get("mode") == "unsupported-meter":
                    alignment = f"{signature} 未自动对齐（第 1 小节从音频开头开始）"
                elif rhythm.get("first_downbeat_s") is None:
                    alignment = "未检测到强拍"
                else:
                    alignment = f"强拍 {float(rhythm['first_downbeat_s']):.2f}s"
                log_lines.append(
                    f"完成：{result.status}，BPM {result.bpm:.2f}，拍号 {signature}，"
                    f"对齐 {alignment}，设备 {result.device}，"
                    f"用时 {result.timings.get('total', 0):.1f}s"
                )
                for warning in result.warnings:
                    log_lines.append(f"警告：{warning}")
                try:
                    progress(1.0, desc="完成")
                except Exception:
                    pass
                yield emit_state()
                return

            # error
            exc = item[1]
            if isinstance(exc, PipelineError):
                message = (
                    f"[{exc.stage}] {exc.message}\n"
                    f"工作目录: {exc.work_dir}\n日志: {exc.log_path}"
                )
            else:
                message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            summary = _error_summary(message)
            log_lines.append("任务失败：")
            log_lines.extend(message.splitlines()[-40:])
            yield emit_state()
            return
    finally:
        JOB_LOCK.release()


def _reset():
    return (
        gr.update(value=None),
        None,
        None,
        None,
        None,
        [],
        _zip_update(None),
        INITIAL_SUMMARY,
        INITIAL_LOG,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        analytics_enabled=False,
        delete_cache=(86400, 86400),
        title="audio2gp — 混音转 GP5 草稿",
    ) as demo:
        gr.Markdown(
            "# audio2gp\n"
            "把任意混音 MP3/WAV 在本机拆成 **Demucs 分轨**、逐轨 MIDI、完整 MIDI，"
            "并生成可编辑的 Guitar Pro 5 草稿。无需云端 API。\n\n"
            "**流程**：预处理 → 用户指定拍号/节奏对齐 → Demucs 分轨 → "
            "音高/鼓点转录 → 量化与小节对齐 → 指法 → GP5。"
        )
        with gr.Accordion("使用说明：对齐规则、参数含义与输出格式（点击展开）", open=False):
            gr.Markdown(HELP_MARKDOWN, elem_classes=["help-box"])
        with gr.Row():
            with gr.Column(scale=5):
                audio_in = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    format="wav",
                    label="上传混音音频（MP3 / WAV）",
                )
                gr.Markdown(
                    "**上传**：支持 MP3、WAV、FLAC、M4A 等 ffmpeg 能解码的格式；"
                    "上传后会先统一转为 44.1 kHz 立体声 WAV。文件越长处理越久，"
                    "建议先用 30–60 秒片段确认参数。",
                    elem_classes=["small-note"],
                )
                with gr.Accordion("转换参数（下方逐项说明）", open=True):
                    with gr.Row():
                        model = gr.Dropdown(
                            choices=["htdemucs_ft", "htdemucs"],
                            value="htdemucs_ft",
                            label="Demucs 模型",
                            info="htdemucs_ft 质量更好但约慢 4 倍；htdemucs 适合快速预览。",
                        )
                        device = gr.Dropdown(
                            choices=["auto", "cuda", "cpu"],
                            value="auto",
                            label="设备",
                            info="auto 会优先使用 CUDA；CPU 回退会很慢。",
                        )
                    gr.Markdown(
                        "**Demucs 模型**：`htdemucs_ft` 是 4 个模型的集成，人声/鼓/贝斯分离更细，"
                        "但耗时约为 `htdemucs` 的 4 倍；`htdemucs` 适合先快速试听。"
                        "**设备**：`auto` 自动使用可用的 CUDA；`cuda` 强制 GPU；"
                        "`cpu` 仅用于没有显卡或排查 CUDA 问题，速度很慢。",
                        elem_classes=["small-note"],
                    )
                    selected_stems = gr.CheckboxGroup(
                        choices=STEM_CHOICES,
                        value=[name for _label, name in STEM_CHOICES],
                        label="原曲包含的乐器（分轨选择）",
                        info="只勾选原曲中真实存在的乐器；至少选择一项。",
                    )
                    gr.Markdown(
                        "**分轨选择**：只勾选原曲中真实存在的乐器。Demucs 仍会在后台分离四条源分轨，"
                        "以保证分离质量；未勾选的轨不会被转录、不会写入 MIDI/GP5，也不会复制到输出/ZIP。"
                        "例如原曲只有主唱和木吉他，勾选「吉他/其他」+「人声」即可得到两轨。"
                        "至少选一项；「吉他轨来源」所在的 stem 必须勾选（来源为 `other` 就勾吉他/其他，"
                        "来源为 `vocals` 就勾人声）。",
                        elem_classes=["small-note"],
                    )
                    with gr.Row():
                        quantize = gr.Dropdown(
                            choices=[8, 12, 16, 32],
                            value=16,
                            label="量化网格（1/8、1/8 三连、1/16、1/32）",
                            info="音符时值网格，与拍号无关；默认 1/16，12 表示每拍三个八分音符。",
                        )
                        guitar_stem = gr.Dropdown(
                            choices=["other", "vocals"],
                            value="other",
                            label="吉他轨来源",
                            info="other 通常包含吉他/键盘；vocals 生成单音人声旋律。",
                        )
                    with gr.Row():
                        time_signature = gr.Dropdown(
                            choices=[
                                (TIME_SIGNATURE_LABELS[value], value)
                                for value in TIME_SIGNATURE_CHOICES
                            ],
                            value="4/4",
                            label="拍号（由你指定，不自动识别）",
                            info="决定小节长度、GP5/MIDI 拍号与小节线；BPM 一律按四分音符计。",
                        )
                        manual_downbeat = gr.Number(
                            value=0,
                            precision=2,
                            label="手动强拍位置（秒，0 = 不手动指定）",
                            info="第一个强拍的时间；支持所有拍号，优先于自动强拍分析。",
                        )
                    gr.Markdown(
                        "**量化网格**：`1/16` 适合大多数流行、摇滚和布鲁斯；`1/32` 保留更快的音符，"
                        "但可能把演奏噪声变成碎拍；`1/8` 更干净，却会丢掉较短的装饰音。"
                        "**吉他轨来源**：`other` 通常包含吉他、键盘和少量残余人声，适合作为吉他轨；"
                        "`vocals` 生成单音人声旋律，适合想弹唱主旋律的情况。\n\n"
                        "**拍号**：本工具不做自动拍号识别，必须由你选择。普通歌曲 4/4；三拍子/华尔兹 3/4；"
                        "二拍子/进行曲 2/4；三连音感的 shuffle 与慢布鲁斯优先 12/8；复合二拍可用 6/8。"
                        "shuffle 不是拍号而是演奏感觉，也可以选 4/4 并把量化网格设为 `1/8 三连音`。"
                        "BPM 始终按四分音符计算：6/8、9/8、12/8 每小节分别等于 2、3、4 个附点四分音符拍。\n\n"
                        "**手动强拍位置**：填写第一个小节第 1 拍在音频中的时间（秒）。填写后不再运行自动强拍检测，"
                        "适用于所有拍号；开头弱起仍会保留在前置小节。`0` 表示不使用。",
                        elem_classes=["small-note"],
                    )
                    with gr.Row():
                        max_fret = gr.Slider(
                            minimum=5,
                            maximum=24,
                            step=1,
                            value=15,
                            label="最大品格",
                            info="限制指法搜索的最高品格；高把位独奏可调高。",
                        )
                        bpm = gr.Number(
                            value=0,
                            precision=1,
                            label="BPM 覆盖（0 = 自动估计）",
                            info="自动 BPM 明显错误时填写实际值；只改速度和网格，不改强拍相位。",
                        )
                    gr.Markdown(
                        "**最大品格**：限制指法搜索范围。15 品覆盖大部分常用把位；"
                        "原曲有高把位独奏时可调到 19–24；调低可减少高把位结果。"
                        "**BPM 覆盖**：`0` 表示自动估计。自动 BPM 明显错误（例如把 80 听成 160）时，"
                        "填写实际 BPM；它改变速度和小节长度，但不会移动第一个强拍的位置。",
                        elem_classes=["small-note"],
                    )
                    with gr.Row():
                        align_downbeats = gr.Checkbox(
                            value=True,
                            label="按检测到的强拍对齐小节（自动分析仅 4/4）",
                            info="让第 1 小节第 1 拍落在强拍；其他拍号请填手动强拍位置。",
                        )
                        keep_stems = gr.Checkbox(
                            value=True,
                            label="在输出和 ZIP 中保留 stems WAV",
                            info="关闭后仍可在页面试听，但输出目录/ZIP 不含 WAV。",
                        )
                        include_vocals = gr.Checkbox(
                            value=True,
                            label="GP5 增加人声旋律轨",
                            info="额外写入人声音高旋律，不包含歌词。",
                        )
                    gr.Markdown(
                        "**按检测到的强拍对齐小节**：在本机用 madmom 模型检测强拍，"
                        "让小节线落在重拍上。自动强拍分析目前只支持 4/4；其他拍号会给出警告，"
                        "并让第 1 小节从音频开头开始，除非填写了手动强拍位置。"
                        "没有弱起时第一个强拍是第 1 小节第 1 拍；"
                        "有弱起时弱起保留在前置小节，第一个强拍从第 2 小节（或更后）开始。"
                        "模型不可用时自动回退到 librosa 重音估计。"
                        "**保留 stems WAV**：开启时输出目录和 ZIP 含所选分轨的 WAV，便于在 DAW 中继续使用；"
                        "关闭后页面仍可试听所选分轨，但只保留 MIDI/GP5/元数据，可节省磁盘。"
                        "**GP5 增加人声旋律轨**：在 GP5 中额外写入一条人声音高旋律轨，"
                        "不包含歌词或咬字；关闭后 GP5 只含吉他、贝斯、鼓。"
                        "首次运行会下载约 3 MB 的强拍模型权重；所有音频处理都在本机完成。",
                        elem_classes=["small-note"],
                    )
                with gr.Row():
                    run_button = gr.Button("开始转换", variant="primary", scale=2)
                    reset_button = gr.Button("清空", scale=1)
                log_box = gr.Textbox(
                    value=INITIAL_LOG,
                    label="运行日志",
                    lines=14,
                    max_lines=26,
                    autoscroll=True,
                    interactive=False,
                )
            with gr.Column(scale=7):
                gr.Markdown("### 分轨试听")
                gr.Markdown(
                    "转换完成后可在这里试听**所选的** Demucs 分轨；未勾选的轨不会出现。"
                    "它们是 GP5 转录的输入，也可单独下载到 DAW 中使用。",
                    elem_classes=["small-note"],
                )
                with gr.Row():
                    audio_vocals = gr.Audio(label="人声", type="filepath", interactive=False)
                    audio_drums = gr.Audio(label="鼓", type="filepath", interactive=False)
                with gr.Row():
                    audio_bass = gr.Audio(label="贝斯", type="filepath", interactive=False)
                    audio_other = gr.Audio(label="吉他/其他", type="filepath", interactive=False)
                gr.Markdown("### 下载")
                gr.Markdown(OUTPUT_NOTE, elem_classes=["small-note"])
                with gr.Row():
                    download_files = gr.File(
                        label="逐轨 MIDI / full.mid / song.gp5",
                        file_count="multiple",
                        interactive=False,
                    )
                    download_zip = gr.DownloadButton(
                        label="下载全部（ZIP）",
                        variant="primary",
                        interactive=False,
                    )
                summary = gr.Markdown(INITIAL_SUMMARY)
                gr.Markdown(
                    "**说明**：输出使用你选择的拍号；自动拍号识别不在本工具范围内。"
                    "4/4 会自动按检测到的强拍对齐小节，其他拍号可填写手动强拍位置。"
                    "开头有弱起时会保留为前置小节。BPM 为估计值，调式默认 C 大调。"
                    "指法与鼓点为算法近似，建议用 Guitar Pro / TuxGuitar 打开后编辑。"
                )

        outputs = [
            audio_vocals,
            audio_drums,
            audio_bass,
            audio_other,
            download_files,
            download_zip,
            summary,
            log_box,
        ]
        run_button.click(
            fn=_run_job,
            inputs=[
                audio_in,
                model,
                quantize,
                guitar_stem,
                max_fret,
                bpm,
                align_downbeats,
                keep_stems,
                include_vocals,
                device,
                selected_stems,
                time_signature,
                manual_downbeat,
            ],
            outputs=outputs,
            api_name="convert",
            concurrency_limit=1,
            concurrency_id="audio2gp",
            trigger_mode="once",
        )
        reset_button.click(
            fn=_reset,
            inputs=[],
            outputs=[audio_in] + outputs,
            queue=False,
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="audio2gp Gradio web UI")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    parser.add_argument("--port", type=int, default=7860, help="port (default: 7860)")
    args = parser.parse_args(argv)

    demo = build_demo()
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        max_threads=4,
        max_file_size="1gb",
        enable_monitoring=False,
        footer_links=[],
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
        allowed_paths=[str(OUTPUT_ROOT), str(WORK_ROOT)],
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
