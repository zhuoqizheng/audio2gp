# audio2gp 算法与系统设计

**版本**：2.2  
**更新日期**：2026-10-29  
**硬件假设**：本机 NVIDIA RTX 4090（24 GB，CUDA），也可回退 CPU  
**输入**：一首完整的 MP3 / WAV / FLAC / M4A / OGG 混音  
**输出**：Demucs 分轨、每轨 MIDI、`full.mid`、和弦分析；可继续编辑的多轨 `.gp5`/`tracks.json`，或仅 MIDI（不生成 GP5/编辑状态）  
**运行方式**：仅监听本机回环地址的 FastAPI + 静态 HTML；另有 CLI。推理、量化、指法和导出均在本机完成。

---

## 1. 目标、原则与非目标

### 1.1 目标

把一首没有分轨、没有 MIDI 的完整混音转成一份**可编辑的 Guitar Pro 草稿**：

1. 先分离 `vocals`、`drums`、`bass`、`other` 四类声部；
2. 按用户选择只转录和交付需要的声部；
3. 生成每轨 MIDI 与一份合并 MIDI；
4. 从和声声部分析逐拍和弦进行；
5. 推断吉他/贝斯弦品，写为多轨 GP5，并把和弦符号写入谱面/full MIDI 标记；
6. 在浏览器中直接预览、播放，并支持改弦/品、删除音符、轨道静音；
7. 所有编辑都写回中间表示，再经同一条 PyGuitarPro 导出链路生成新的 GP5/MIDI。

用户也可以选择 **MIDI-only**：仍完成分轨、清理/量化、和弦分析、MIDI 与 ZIP，但跳过指法、GP5、`tracks.json` 和编辑器，任务状态为 `ok`。

**拍号由用户选择**（4/4、3/4、2/4、6/8、9/8、12/8），本版本不做自动拍号识别；所选拍号贯穿小节长度、量化、GP5/MIDI 拍号、和弦小节/拍标签和编辑器 IR。4/4 默认使用自动强拍分析；其他拍号可手动输入首个强拍时间，否则第 1 小节从音频开头开始并警告。shuffle 是节奏感觉而非拍号，常用 12/8，或用 4/4 + 1/8 三连音量化表达 swing。

输出是**草稿**。算法优先避免“看起来合理但不可弹”的位置，而不是追求专业制谱师级别的细节。

### 1.2 核心原则

- **不用 LLM**：记谱和指法全部由专用模型、信号处理和搜索构成。
- **不上传音频**：默认只绑定 `127.0.0.1`，不调用云端推理 API，不抓取商业谱库。
- **模型可替换**：算法接口与模型依赖分离；没有 CUDA 时明确警告并降级。
- **批处理与编辑器解耦**：pipeline 只负责产出草稿与 `tracks.json`；编辑器只修改该状态并调用导出模块。MIDI-only 不构造编辑状态。
- **失败可降级**：GP5 生成失败仍交付 MIDI/分轨/ZIP，任务标记为部分成功；和弦分析/写 `chords.json` 失败只警告，不影响主产物。

### 1.3 明确不做

- LLM 生成谱面/指法；
- 云端账号、公共部署、多用户鉴权；
- 抓取或分发商业吉他谱；
- 内嵌 Guitar Pro；
- 完整 DAW/制谱器、撤销栈、多轨自动重排、PWA、移动端专门适配；
- 自动拍号识别、曲中变拍号、变速曲/速度图、雕琢级表情记号；
- 复音音乐理论级纠错。

Guitar Pro 本身也不内嵌：网页使用 MPL-2.0 的 alphaTab 阅读/播放 GP5；服务器使用 PyGuitarPro 写文件。

---

## 2. 对 TabForge 的参考与取舍

参考项目：[rayBastard/tabforge](https://github.com/rayBastard/tabforge)（MIT）。本项目**没有复制其代码**，也没有引入它的 ML 依赖或多引擎体系；下面这些工程思路被吸收并按本项目规模简化。

| TabForge 的做法 | audio2gp 的落地 | 差异/简化 |
|---|---|---|
| Demucs → Basic Pitch → librosa → 指法 → GP5 的本地管线 | 相同的主干；FastAPI + 静态前端 + alphaTab | 本项目的模型栈固定为 Demucs 4.1 + Basic Pitch；不接其他转录引擎 |
| 鼓轨优先的全局速度、半/倍速消歧、120 BPM 回退 | `rhythm.refine_tempo_from_drums`：鼓点中位拍间隔；2:1/1:2/3:2 等候选按 log 距离贴近参考速度；4% 内保留小节网格速度；失败保留混音/强拍速度 | 未做 TabForge 的完整置信度评估；静音/异常时保留参考值并警告 |
| 先聚类起音再吸附网格，避免和弦被拆散 | `midi_post._cluster` 在 35 ms 窗口内聚类，用簇平均起音量化 | 固定窗口；鼓点单独按 `(tick, pitch)` 去重 |
| 鬼音/短音/复音清理 | Basic Pitch 幅度门限 + 短音 0.15 s 联合过滤；`min_note_sec`；每起音最多 `max_polyphony` 个同时音；贝斯/人声单音化 | 未按乐器分档的完整阈值表；以保守为优先 |
| 直拍与三连音分开量化 | 网格 8 / 12 / 16 / 32；12 = 1/8 三连音；三连音使用 GP5 tuplet；网格按音符时值定义，与拍号无关 | 拍号由用户固定选择（4/4、3/4、2/4、6/8、9/8、12/8）；不做自动拍号识别和曲中变拍号；不写混合节奏族 |
| 手型位置上的 Viterbi 指法 | 候选位置枚举 + 跨度限制 + 手位/换弦/大跳转移代价的 DP；支持预设调弦、调弦建议 | 代价函数和候选剪枝为本项目实现；不做 TabForge 的 lead/rhythm 自动拆分 |
| 保守的技巧标记 | 仅对同弦、间隔 0–60 ms、1–4 半音、非重音的相邻单音标记 hammer-on/pull-off；Basic Pitch 音高轮廓净移动 ≥0.7 半音且单调时标出滑音 | 不识别推弦、颤音、闷音、泛音；没有音高弯音轨 |
| PyGuitarPro 导出 + 回读校验 | `gp_writer` 写 GP5，`parse_gp5` 对象身份回读校验每小节时值 | 只写 GP5（5.1），不写 GPX/GP7/MusicXML |
| FastAPI jobs、内存任务、进度轮询、文件下载 | 本项目的 `api.py`：单 worker、轮询、本地文件白名单、上传大小/时长限制 | 任务只在内存中；服务重启后需重新上传；输出目录仍在磁盘 |
| 分轨缓存与“改选择不重跑 Demucs” | 内容寻址缓存（归一化 WAV 内容 + 模型 + 分离参数）位于 `work/.stem_cache/` | 缓存粒度是完整四轨；没有分阶段 UI |
| alphaTab 读取 GP5，在线播放与改谱 | 浏览器预览/播放；语义编辑（弦/品/删除/静音）回传服务器 | 浏览器不写 GP5；没有完整撤销栈 |

TabForge 的许可证是 MIT；本项目的 alphaTab（MPL-2.0）、Bravura（SIL OFL）、Sonivox 音色（Apache-2.0）、PyGuitarPro（LGPL-3.0）的许可证文件随 vendored 资源分发。

---

## 3. 系统架构

```
浏览器（localhost）
  index.html + style.css + app.js
  ├─ 上传、参数、轮询 GET /api/jobs/{id}
  ├─ alphaTab 读取 /score.gp5，播放、点击音符
  └─ PATCH/DELETE/PUT 语义编辑 → 服务器
                    │ HTTP
┌───────────────────▼───────────────────────────────────────────┐
│ FastAPI（audio2gp/api.py）                                     │
│  JobManager：内存任务 + 单 worker ThreadPoolExecutor            │
│  上传限制、ffprobe 时长校验、文件白名单、per-job edit_lock      │
└───────────────────┬───────────────────────────────────────────┘
                    │ pipeline.run(...)
┌───────────────────▼───────────────────────────────────────────┐
│ Python 管线（audio2gp/pipeline.py）                            │
│ Stage 0 preprocess → Stage 1 rhythm → Stage 2 separate        │
│ → Stage 3 tempo/key → Stage 4 transcribe → Stage 5 clean       │
│ → Stage 5 chords → Stage 6 fingering → Stage 7 GP5/export       │
│ → Stage 8 editor IR (GP5 mode; MIDI-only skips 6/7/8 exports)    │
└───────────────────┬───────────────────────────────────────────┘
                    │ tracks.json
┌───────────────────▼───────────────────────────────────────────┐
│ 编辑器重建（audio2gp/score_ir.py）                             │
│  校验 IR → ir_to_tracks → write_gp5 / write_tab_track_midi    │
│  → 原子替换 song.gp5、MIDI、tracks.json、meta.json、ZIP        │
└───────────────────────────────────────────────────────────────┘
```

要点：

- **进度**：`progress_cb(stage, overall_pct, message)`；API 将其写入任务日志并给浏览器轮询。没有 WebSocket/SSE。
- **并发**：一个 GPU worker 串行执行任务，避免两个 Demucs 同时占显存。编辑请求由每个任务的 `edit_lock` 串行化。
- **编辑与批处理解耦**：`pipeline.py` 不导入 API；`score_ir.py` 不运行 Demucs/Basic Pitch；两者只共享 `types.py`、`gp_writer.py`、`midi_post.py`。
- **缓存**：`work/.stem_cache/<hash>/` 保存完整四轨；选择不同的 `selected_stems` 只改变转录与交付，不重新分离。

---

## 4. 阶段（Stage 0–8）

阶段名称与 API 进度条对应关系：`preprocess`、`rhythm`、`separate`、`analyze`、`transcribe`、`post`、`gp5`；Stage 8 不占用任务进度，由后续编辑端点触发。

### Stage 0：输入校验、归一化与初始速度

**输入**：上传的音频文件。  
**输出**：`work/<job_id>/song.wav`（44.1 kHz、双声道、PCM s16）、时长、初始 BPM。

步骤：

1. API 校验扩展名、非空、上传大小（默认 200 MB）和时长（默认 600 s）；CLI 也校验文件存在和非空。
2. `ffmpeg -vn -ac 2 -ar 44100 -c:a pcm_s16le -map_metadata -1` 归一化；去除封面/视频流。
3. `ffprobe` 读取时长。
4. 若 `bpm_override > 0` 直接采用；否则 librosa 在 HPSS 打击成分上估计速度，折叠到 60–180，失败回退 120 BPM 并警告。
5. 超过 2048 小节时警告（GP5 文件会很大）。

失败策略：归一化失败是致命错误（`PipelineError("preprocess", ...)`），写出错误 `meta.json`。首次 API 任务在 `work/<job>/` 下保存上传文件。

### Stage 1：拍点、强拍与小节网格

**输入**：归一化 WAV、用户所选拍号 `Options.time_signature`、可选手动强拍 `Options.manual_downbeat_s`。  
**输出**：`RhythmInfo`：BPM、强拍时间、拍号（用户选择）、节拍方法、警告。

步骤：

1. **拍号**：`config.parse_time_signature` 只接受 4/4、3/4、2/4、6/8、9/8、12/8，计算 `bar_ticks = numerator * 3840 // denominator`（四分音符 = 960 ticks）。4/4 = 3840、3/4 = 2880、2/4 = 1920、6/8 = 2880、9/8 = 4320、12/8 = 5760。复合拍（6/8、9/8、12/8）的律动单位是附点四分（1440 ticks），每小节分别 2/3/4 个 pulse；单拍子拍号每拍 = 四分音符。
2. **手动强拍**：若 `manual_downbeat_s` 已设置，直接作为第一个强拍，跳过自动分析，适用于任何拍号。
3. **4/4 自动分析**：未手动指定时，仅 4/4 使用 `madmom-infer` 的 `DOWNBEATS_BLSTM`，强制 4/4 解码；得到 beat/downbeat 时间。根据强拍中位间隔计算 BPM；若与 Stage 0 的估计相差过大，使用强拍 BPM。madmom 不可用时回退 librosa 拍点 + 低频重音相位。仍失败则关闭对齐并警告，小节从音频 0 秒开始。
4. **非 4/4 且未手动指定**：自动强拍模型只支持 4/4，直接跳过自动分析，记录“自动强拍分析目前只支持 4/4”警告，第 1 小节从音频 0 秒开始。
5. `--no-align-downbeats` 关闭以上自动分析；手动强拍优先于该开关，仍然生效。没有手动值时第 1 小节从音频 0 秒开始。

**小节原点**（`choose_grid_offset(first_downbeat, first_note, bpm, beats_per_bar=meter.numerator, denominator=meter.denominator)`）：

- 若第一个音符不早于首个强拍（允许 50 ms 容差），tick 0 = 首个强拍；
- 若第一个音符早于首个强拍（弱起/pickup），把首个强拍放到第 2 小节或更后，前面的完整小节放弱起材料；**不写短小的 anacrusis 小节**；
- 偏移量按所选拍号的小节长度计算，所有音符相对该偏移量化，GP5 小节线与强拍对齐；
- 没有可用强拍时，偏移为 0，第 1 小节从音频开头开始。

BPM 始终按**四分音符**计（MIDI 标准）。12/8 中一个附点四分 pulse 是 1.5 个四分音符（1440 ticks）；记谱软件的拍号/小节正确，但显示的 BPM 不等同于“每分钟附点四分”的常见记法。  

注意：madmom 权重是 **CC BY-NC-SA 4.0**，非商用。若需要商用，关闭强拍对齐（`--no-align-downbeats` 或 UI 开关），或替换权重。代码为 BSD-2-Clause。

### Stage 2：Demucs 分轨与缓存

**输入**：归一化 WAV、模型名、设备、是否缓存。  
**输出**：`work/<job>/stems/{vocals,drums,bass,other}.wav`；输出目录的 stems 仅包含 `selected_stems`（当 `keep_stems`）。

- 模型默认 `htdemucs_ft`（四模型集成，较慢、较干净）；`htdemucs` 更快。
- `Separator(..., shifts=1, overlap=0.25, split=True, jobs=0)`。
- 保存为 16-bit、Demucs samplerate（通常 44.1 kHz）。
- 检测静音轨并警告，不中断。
- CUDA OOM：清空缓存，降级到 CPU 重跑一次；CPU 很慢。其他错误让 `separate` 阶段失败。
- **缓存**：`cache_stems=true` 时，以 `sha256(归一化 WAV) + 模型 + 分离参数 + 缓存版本` 作为键，在 `work/.stem_cache/<key>/` 保存完整四轨和 `meta.json`。命中时直接复制到当前任务，日志/警告提示“复用缓存”。缓存条目先写临时目录再原子改名；不完整条目不使用。缓存目录不进入 `output/` 或 ZIP。关闭缓存则总是重跑。

### Stage 3：全局速度校正与可选调性

**输入**：四个分轨、Stage 1 的速度/网格、用户覆盖。  
**输出**：最终 BPM、速度来源、`key_meta`。

**速度**：

1. 若 `bpm_override` 存在，来源为 `override`，不再估计。
2. 否则若 `tempo_from_drums=true` 且鼓轨存在：在鼓轨上用 librosa 拍点，优先用拍间隔中位数；失败保留参考速度。
3. 半/倍速消歧：在 2:1、1:2、3:2、2:3 等候选里，取 log 距离最接近参考速度的值。若与强拍网格相差 ≤4%，保留网格 BPM（长曲 1% 误差会漂移数秒）；否则采用鼓速并警告。
4. 最终 BPM 用于所有 MIDI/GP5 时值。

**调性**（可选，默认开启）：

- 使用 Krumhansl–Kessler 音级轮廓；对和声分轨（`other`、`bass`、`vocals`）做 HPSS 谐波分离与 chroma 平均。
- 输出调名、tonic、mode、置信度、GP5 调号枚举名；失败/关闭时默认 C 大调。
- 检测结果写入 GP5 调号与 `meta.json`/`tracks.json`。本工具不写临时转调或音阶指板图。

### Stage 4：分轨音符转录

**输入**：`selected_stems` 对应的 WAV、速度、阈值。  
**输出**：每轨 `NoteEvent` 列表（秒为单位）。

**旋律轨（vocals / bass / other）**：

- Basic Pitch ICASSP 2022（TFLite，0.4.0 在 Linux/Python 3.10 为 CPU 推理），复用同一个模型实例。
- `minimum_note_length = max(30 ms, min_note_sec * 1000)`；`onset_threshold`、`frame_threshold` 可调（UI 高级参数）。
- 频率范围：vocals 80–1200 Hz、bass 30–400 Hz、other 60–2000 Hz。
- vocals/bass 开 melodia trick（单旋律倾向）；other 关闭以保留和弦。
- 鬼音过滤：若音符幅度 < `max(0.05, ghost_amplitude * 峰值)` **且**持续时间 < 150 ms，则丢弃；计数写入警告。
- 贝斯在 Stage 5 进一步单音化；other 保留和弦/复音。

**鼓轨**：

- HPSS 打击成分；STFT 2048/256（22.05 kHz）。
- 全频 onset（`delta=0.10`、`wait=2`、backtrack）为主；低频/中频/军鼓/高频 flux onset 只补全频漏掉的点，避免时间提前。
- 每起音提取低频能量、body、snare 频段、高频衰减、谱质心等特征，映射 GM：
  - 底鼓 36、军鼓 38、闭镲 42、开镲 46、吊镲 49；
  - 桶鼓 41/45/47/48/50（按质心近似）。
- 每类有 refractory；重叠候选去重。鼓组识别是启发式，桶鼓/镲片尤其容易混淆，会在警告中说明。
- 空鼓轨写为空任务轨道并警告，不阻止任务完成。

失败策略：任一旋律轨 Basic Pitch 异常通常是致命错误；静音轨转空轨并警告。鼓转录异常降级为空鼓轨并警告。

### Stage 5：清理、量化、MIDI

**输入**：原始 `NoteEvent`、网格、偏移、最短音长、复音上限。  
**输出**：整数 tick 的 `Note`、每轨 MIDI、`full.mid`。

- tick 基准：四分音符 960；小节长度 = `numerator * 3840 // denominator`（4/4 = 3840、3/4 = 2880、2/4 = 1920、6/8 = 2880、9/8 = 4320、12/8 = 5760）。网格值按音符时值定义，与拍号无关：
  - 8 → 480 ticks（1/8）
  - 12 → 320 ticks（1/8 三连音）
  - 16 → 240 ticks（1/16）
  - 32 → 120 ticks（1/32）
- **先聚类再量化**：35 ms 内的起音归为一簇，用簇平均开始时间吸附网格，避免和弦音被分到不同拍。
- 短音过滤（`min_note_sec`，鼓除外）；结束时间也吸附网格，至少保留一个网格单位，保证 GP5 tie 布局可表示。
- 去重（同音高、同网格）、结束时间早于下一同音高起的重叠修剪。
- 每起音复音上限 `max_polyphony`（默认 6）；按音符重要性保留。
- bass / vocals 单音化；other 保留和弦。
- 速度值由 Basic Pitch amplitude 映射为 8 档 GP 力度。
- 鼓：每个候选一拍网格，按 `(tick, pitch)` 去重。
- MIDI：`mido` 写 type-1；每轨显式通道和 program；所有 MIDI 写出函数写入所选拍号的 `time_signature` meta（默认 4/4）；`full.mid` 只包含所选分轨。单轨 MIDI 的通道为 Guitar 0、Bass 1、Vocals 2、Drums 9；`full.mid` 为 Guitar 1、Bass 2、Vocals 0、Drums 9。GP5 使用 Guitar 0/1、Bass 2/3、Vocals 4/5、Drums 9/9；编辑重建时 MIDI 跟随各 GP5 轨的实际通道。检测到和弦时，`full.mid` 的 conductor track 写入有序 MIDI marker（和弦标签，tick = 和弦起点）。

### Stage 5b：和弦分析（`chords.py`）

**输入**：所选 `other`/`bass`/`vocals` 和声分轨、BPM、网格偏移/时长、key（可选）。  
**输出**：`ChordEvent(start_tick, end_tick, start_s, end_s, label, root, quality, confidence)`；`chords.json`。

- 22.05 kHz 读取分轨，HPSS harmonic 后取 CQT chroma；每帧先 L2 归一化再加权平均：`other=1.0`、`bass=0.8`、`vocals=0.3`，未知来源 0.65，避免高能声部主导。
- 分析仍按四分音符（960 ticks）同步求均值；以 75 分位帧能量为参考做静音判断，静音段不产生和弦，允许进行中存在空拍。`bar`/`beat` 标签使用所选拍号：复合拍的 `beat` 是附点四分 pulse（见 `chords_to_dict`），tick 坐标本身不变。
- 谐波抑制：从 chroma 中减去上方五度/大三度的谐波能量，再减中位数、归一化，降低把大三和弦误判为 maj7 的概率。
- 12 根 × 8 品质模板（major/minor/7/maj7/m7/sus4/dim/power）打分；加入根音、模板外能量、扩展音惩罚；key 可用时给音阶内根音/和弦音小幅加权。
- 每个连续非静音块做 Viterbi（切换惩罚 0.65）平滑；置信度由最高与次高发射分数差经 logistic 映射，写入 `confidence`。相邻同标签段合并。
- 起始 tick 使用与音符相同的 `grid_offset_s` 及 960 ticks/四分音符，故和弦变化与小节线一致；`chords.json` 还带 1-based `bar`/`beat` 和 `summary`（`chord_count`、`unique_chords`、`quality_counts`、`progression`）。
- 只做根音+品质估计，不做转位/斜杠和弦/voicing/和弦图；密集失真、power chord、暴露的七度可能被保守简化。分析失败、没有和声声部或写入失败都只记录警告，不使任务失败。

失败策略：清理/MIDI 写入异常是致命 `post` 阶段错误；和弦分析是可降级的可选结果。

### Stage 6：指法、调弦与技巧（仅 GP5 模式）

**输入**：每轨量化 `Note`、最高品、调弦预设、`suggest_tuning`。  
**输出**：`FingeredNote(string, fret, hammer, slide)` 的 `TabTrack`。

`export_gp5=false`（MIDI-only）时整段跳过，不枚举候选、不做 DP，也不产生 GP5 轨。

**轨道来源**：

- Guitar ← `guitar_stem`（`other` 或 `vocals`）；
- Bass ← `bass`；
- Drums ← `drums`（打击乐轨，独立记谱）；
- Vocals 旋律轨 ← `vocals`（当人声不是吉他来源且 `include_vocals`）。

**吉他/贝斯指法**：

1. 按起音分组；每个音高枚举所有合法 `(string, fret)`：`0 <= fret <= max_fret`，音高 = 空弦 + 品。
2. 同一拍每个弦只能有一个音；吉他手型跨度上限 5 品，贝斯 4 品。
3. 静态代价偏好低把位、窄跨度、空弦；每层保留前 K 个候选组合。
4. 动态规划（Viterbi 风格）跨起音选择，转移代价包括手位移动、换弦数、大跳惩罚、同音高换位置惩罚。
5. 回溯得到整段指法。候选位置本来就包含 ±12 八度折叠，但附加 1000 的极高代价，正常范围内有解时不会被选中；仍无解则丢弃重要性最低的音。两者都记录警告，不悄悄改音高。

**调弦**：

- 吉他预设：标准 EADGBE、Drop D、降半音；贝斯：标准 EADG、Drop D。
- `suggest_tuning=true` 时对所有预设分别指法，按“保留原音高数量优先、总品位低优先”选择；否则固定用户选择，不静默换调弦。
- 弦号约定：**1 = 最高音弦**（与 Guitar Pro/alphaTab 的显示习惯相反；API/UI 负责映射）。

**技巧**：

- hammer-on/pull-off：同弦、相邻两个单音起音、间隔 ≤60 ms、音程 1–4 半音、第二个音不是重音；写在第一个音上。
- 滑音：Basic Pitch 轮廓（每半音 3 个 bin）净移动 ≥0.7 半音、总跨度接近净移动、≥75% 的移动方向一致 → 出滑音 ±1；不把往返轨迹/颤音当滑音。
- 不做推弦、颤音、闷音、泛音、击勾弦跨弦等。

失败策略：无解时跳过/八度折叠并警告；空轨仍生成空 GP5 轨。

### Stage 7：GP5 组装与导出（可选）

**输入**：`TabTrack` 列表、BPM、小节数、拍号、key、layout、quantize、`ChordEvent` 列表。  
**输出**：`song.gp5`、`tracks.json`、`meta.json`、ZIP、每轨 MIDI。

- 小节数：`max(1, ceil(duration / meter.bar_seconds), max_note_end // meter.bar_ticks + 1)`；恰好落在小节线的结束不额外增小节。
- 拍号由用户选择；每个 `MeasureHeader` 写 `gp.TimeSignature(numerator, gp.Duration(denominator), beams=meter.beams)`，`start = 960 + index * meter.bar_ticks`。GP5 版本 5.1，cp1252，轨名 ASCII。
- 每小节两个 voice；voice 0 必须完整覆盖 `meter.bar_ticks`，使用音符/休止填充；voice 1 放一个空 beat（合法性要求）。复合拍的 beams 按三连音组（每 3 个八分）分组，简单拍每 2 个一组。
- 布局：
  - `ties`：以所有起止点为边界切分，跨 beat/小节用 tie 连接同弦同音高；能表达长音；
  - `onset`：每个起音一个 beat，最保守，适合无法表示时；
  - `auto`：先尝试 ties，校验失败自动回退 onset，并警告。
- 时值分解支持：全音符、附点二分、二分、附点四分、四分、附点八分、八分、附点十六分、十六分、附点三十二分、三十二分，以及 1/8/1/16/1/32 三连音（GP5 tuplet）。
- 打击乐：`isPercussionTrack=True`，通道 9；GM 音高作为 Note.value；六条固定 lane 1–6；不使用 ties。
- 调号：Stage 3 的 key 枚举；调性关闭/失败时 C 大调。
- 和弦符号：仅当和弦分析启用时，把每个和弦起点映射到第一条非打击乐轨中包含该 tick 的 beat，设置 `BeatEffect.chord`。GP5 只存显示用符号（名称/根音/品质），不生成指法图；同一拍多个和弦保留第一个，超范围/无弦乐轨只警告。
- **回读校验**：用 `gp_writer.parse_gp5`（对象身份解析）重新读取，检查轨数、小节数、两 voice、每个 header 的 `(numerator, denominator.value)` 与 `start = 960 + index * bar_ticks`、每小节 voice 0 总时值 = `meter.bar_ticks`、Note string/type、打击乐通道/移调、tempo、调号。和弦效果不参与时值校验。
- 写 GP5 失败 → 任务 `partial`，仍保留 MIDI/分轨/ZIP；不要因为制谱失败丢掉转录结果。
- `export_gp5=false` 时不执行本阶段：不写 `song.gp5`/`tracks.json`，任务仍为 `ok`；`meta.json` 的 `output_format` 为 `midi_only`。

### Stage 8：编辑状态、API 与浏览器

**状态**：`output/<job_id>/tracks.json`（Score IR），由 pipeline 在 GP5 成功后写入；失败不影响下载。MIDI-only 不写该文件，也不提供 `/state` 或 `/score.gp5`。

**IR 结构**（`format=1`）：

```json
{
  "format": 1,
  "job_id": "20261028-...",
  "revision": 0,
  "bpm": 88.24,
  "time_signature": "4/4",
  "bar_ticks": 3840,
  "beats_per_bar": 4,
  "measure_count": 8,
  "quantize": 16,
  "layout": "auto",
  "max_fret": 15,
  "string_numbering": "1=highest",
  "key": {"name": "B major", "gp_key": "BMajor"},
  "chords": [
    {"start_tick": 0, "end_tick": 1920, "start_s": 0.0, "end_s": 1.0,
     "label": "B", "root": 11, "quality": "major", "confidence": 0.71}
  ],
  "tracks": [
    {
      "id": "guitar",
      "name": "Guitar",
      "stem": "other",
      "tuning": [64, 59, 55, 50, 45, 40],
      "is_percussion": false,
      "channel": 0,
      "instrument": 27,
      "muted": false,
      "warnings": [],
      "notes": [
        {"id": "n000001", "start_tick": 1680, "end_tick": 1920,
         "pitch": 63, "velocity": 90, "string": 2, "fret": 4,
         "hammer": false, "slide": 0}
      ],
      "drum_notes": []
    }
  ]
}
```

约束：

- 音符 ID 稳定（`n000001...`），前端点击映射使用 ID，不用索引；
- `1=highest`：弦 1 是最高音弦；
- 非鼓轨始终满足 `pitch == tuning[string-1] + fret`；
- `time_signature` 是所选拍号；`bar_ticks = numerator * 3840 // denominator`；`beats_per_bar` 存分子（12/8 = 12），鼓/和弦标签使用复合拍的 pulse 概念；
- 每轨同一时间/弦不重复；音符 tick 落在 `[0, measure_count*bar_ticks]`；
- `revision` 单调递增；编辑请求带 revision，不一致返回 **409**，避免覆盖别的标签页；
- 鼓轨只存 `drum_notes`，没有 string/fret；
- `chords` 可缺省（旧 IR 兼容）；存在时必须按 tick 有序、不重叠、位于 `[0, measure_count*bar_ticks]`，root 0–11、quality 属于 8 种模板、置信度 0–1。编辑器目前不修改和弦，但每次 commit 都会从 IR 重建 GP5 和弦符号和 `full.mid` marker；
- 不缩减小节（删除音符留空小节）；GP5/MIDI 导出保持一致的小节数。

**编辑 API**：

| 方法 | 路径 | 作用 |
|---|---|---|
| `POST` | `/api/jobs` | multipart：`file` + JSON `options`；返回 202 `{id,status}` |
| `GET` | `/api/jobs/{id}` | 状态、进度、日志、完成后 `result` |
| `GET` | `/api/jobs/{id}/state` | 读取 IR |
| `GET` | `/api/jobs/{id}/score.gp5` | 当前 GP5（no-store） |
| `GET` | `/api/jobs/{id}/audio/{stem}` | 试听分轨（output 或 work） |
| `GET` | `/api/jobs/{id}/files/{path}` | 白名单下载（GP5/MIDI/stems/meta/tracks/chords/ZIP） |
| `PATCH` | `/api/jobs/{id}/tracks/{tid}/notes/{nid}` | `{revision,string?,fret?,pitch?}` 改位置 |
| `DELETE` | `/api/jobs/{id}/tracks/{tid}/notes/{nid}?revision=N` | 删除音符 |
| `PUT` | `/api/jobs/{id}/tracks/{tid}/mute` | `{revision,muted}` 持久化静音 |

语义：

- **只改 string**：保持音高，推导新品；**只改 fret**：保持弦，音高随之改变；**string+fret**：音高随之改变；若同时给 pitch 且不匹配则 422。
- **删除**：移除 IR 音符，下一 revision 重新生成 GP5/MIDI。
- **静音**：写入 IR、重新生成并刷新结果；导出文件中仍保留该轨，取消静音恢复。alphaTab 预览用它实时静音。
- 所有写操作先校验，再在 `output/` 内暂存临时 GP5/MIDI，成功后用 `os.replace` 原子发布；失败不会破坏旧文件。成功后更新 `tracks.json`、`meta.json`、ZIP，revision+1。

**浏览器**：

- alphaTab 读取 GP5（**浏览器不能写 GP5**），`core.includeNoteBounds=true` 才能收到 `noteMouseDown`；本地 vendored alphaTab/SoundFont/Bravura 字体，离线可用。
- 点击音符：alphaTab 的弦号 `a` 与 IR 的弦号换算为 `IR = N - a + 1`；按 `start_tick + pitch + string` 映射到稳定 ID。
- 编辑成功：刷新下载链接，使用 revision/cache buster 重新加载同一个 AlphaTabApi；失败显示服务器返回原因（例如“同拍该弦已占用”）。
- 页面 URL 带 `?job=<id>`，刷新可恢复仍在内存中的任务和已完成结果；服务重启后任务内存丢失，但磁盘输出仍在。
- 结果页根据 `output_format` 切换布局：MIDI-only 隐藏 alphaTab/编辑/混音工作区，显示“仅 MIDI”说明和和弦时间线；GP5 模式显示谱面、编辑器、和弦时间线。和弦时间线标出低置信度段（<0.35）。

---

## 5. 模块划分与接口

```
audio2gp/
  api.py          FastAPI、JobManager、静态文件、编辑端点
  pipeline.py     Stage 0–7 编排、进度预算、meta/ZIP
  preprocess.py   ffmpeg 归一化、初始 BPM
  rhythm.py       madmom 强拍、librosa 回退、鼓速校正、网格偏移
  separate.py     Demucs、CUDA/OOM、内容寻址分轨缓存（实现）
  separation.py   稳定的分轨接口别名，重导出 separate
  transcribe.py   Basic Pitch 旋律转录与鬼音过滤
  drums.py        onset + 频段鼓件分类
  midi_post.py    聚类、量化、清理、MIDI 写出、编辑重建 MIDI
  chords.py       拍同步 chroma、谐波抑制、Viterbi 和弦检测与 JSON
  keydetect.py    Krumhansl–Kessler 调性
  fingering.py    候选位置、手型 DP、调弦建议
  techniques.py   滑音/击勾弦推断
  gp_writer.py    GP5 写入、tuplet、tie、打击乐、回读校验（实现）
  export_gp5.py   稳定的导出接口别名，重导出 gp_writer
  score_ir.py     编辑 IR、校验、提交重建
  config.py       Options、调弦、网格、速度时值
  types.py        NoteEvent/Note/FingeredNote/TabTrack/ChordEvent/PreprocessedAudio/RunResult
  utils.py        任务 id、ffmpeg/ffprobe、JSON、ZIP、日志
  cli.py          命令行入口
  web_app.py      旧版 Gradio UI（保留，不是主入口）
  static/         index.html、style.css、app.js、vendored alphaTab
```

关键签名：

```python
pipeline.run(input_path, options, progress_cb=None, *, job_id=None) -> RunResult

preprocess.prepare(input_path, work_dir, options, logger=None, progress_cb=None)
    -> PreprocessedAudio

rhythm.analyze_rhythm(wav_path, logger=None, progress_cb=None) -> RhythmInfo
rhythm.refine_tempo_from_drums(path, reference_bpm, logger=None) -> (bpm, warnings)
rhythm.choose_grid_offset(first_downbeat_s, first_note_s, bpm, *,
                          beats_per_bar=4, denominator=4) -> float

separation.separate(audio, options, device, logger=None, progress_cb=None)
    -> (stems, warnings, used_device)          # separate.py also exports this
separation.stem_cache_dir(audio, options) -> Path

transcribe.transcribe_stems(stems, options, bpm, logger=None, progress_cb=None)
    -> (events, warnings)
drums.transcribe_drums(path, bpm, quantize, logger=None, progress_cb=None)
    -> (events, warnings)

midi_post.clean(events, options, bpm, logger=None, offset_s=0.0)
    -> {stem: [Note]}
midi_post.write_stem_midi(..., *, time_signature="4/4")
midi_post.write_full_midi(all_notes, bpm, path, stems=None, chords=None,
                          *, time_signature="4/4")
midi_post.write_tab_track_midi(track, bpm, path, *, time_signature="4/4")
midi_post.write_tab_tracks_midi(tracks, bpm, path, chords=None,
                                *, time_signature="4/4")

chords.analyze_chords(stems, bpm, *, offset_s=0.0, duration_s=None,
                      key_info=None, warnings=None, progress_cb=None)
    -> [ChordEvent]
chords.chords_to_dict(events, *, bpm, offset_s, duration_s, key=None,
                      time_signature="4/4") -> dict
chords.chords_from_state(entries) -> [ChordEvent]

fingering.arrange_guitar(notes, max_fret, ...)
    -> (fingered_notes, warnings)
fingering.arrange_bass(notes, max_fret, ...)
fingering.suggest_tuning(notes, ...)

export_gp5.write_gp5(tracks, bpm, measure_count, path, layout="auto",
                     quantize=16, key_name=None, chords=None,
                     *, time_signature="4/4") -> warnings
export_gp5.verify_gp5(path, expected_measures, *, time_signature="4/4") -> warnings
export_gp5.parse_gp5(path) -> guitarpro.Song

score_ir.build_state(tracks, bpm, measure_count, *, job_id, quantize,
                     layout, max_fret, key=None, chords=None,
                     time_signature="4/4") -> dict
score_ir.ir_to_chords(state) -> [ChordEvent]
score_ir.validate_state(state); score_ir.load_state(path)
score_ir.update_note_position(state, track_id, note_id, *, string=None,
                              fret=None, pitch=None) -> dict
score_ir.delete_note(state, track_id, note_id) -> dict
score_ir.set_track_muted(state, track_id, muted) -> dict
score_ir.commit_state(state, out_dir) -> {state, gp5, midi, full_midi, zip, warnings, meta}
```

---

## 6. API 参数

`POST /api/jobs` 的 `options` JSON（Pydantic `extra="forbid"`，非法键 422）：

```json
{
  "demucs_model": "htdemucs_ft",
  "quantize": 16,
  "guitar_stem": "other",
  "keep_stems": true,
  "max_fret": 15,
  "min_note_sec": 0.03,
  "device": "auto",
  "include_vocals": true,
  "bpm_override": null,
  "time_signature": "4/4",
  "manual_downbeat_s": null,
  "align_downbeats": true,
  "guitar_tuning": "standard",
  "bass_tuning": "standard",
  "suggest_tuning": false,
  "ghost_amplitude": 0.15,
  "max_polyphony": 6,
  "onset_threshold": 0.5,
  "frame_threshold": 0.3,
  "detect_techniques": true,
  "detect_key": true,
  "tempo_from_drums": true,
  "cache_stems": true,
  "selected_stems": ["vocals", "drums", "bass", "other"],
  "layout": "auto",
  "detect_chords": true,
  "export_gp5": true
}
```

`time_signature` 只能是 4/4、3/4、2/4、6/8、9/8、12/8（其他值 422）；这是用户选择，不做自动拍号识别。`manual_downbeat_s` 是源音频秒数，`> 0` 或 `null`（0 在 CLI 中表示未设置）；设置后覆盖自动强拍分析，适用于任何拍号。自动强拍分析仅 4/4；其他拍号未设置手动强拍时第 1 小节从音频开头开始。`/api/limits` 的 `time_signatures`、`manual_downbeat`、`auto_downbeat_time_signatures` 描述这些能力；任务结果额外返回 `time_signature`、`bar_ticks`、`manual_downbeat_s`。

`selected_stems` 只控制转录/交付，不影响 Demucs 始终分离四轨。至少要选一轨。`export_gp5=true` 时组合还必须能产生至少一条 GP5 轨（例如只选 vocals、关闭人声旋律轨、吉他来源是 other → 422）；`export_gp5=false` 不受此限制。`detect_chords=false` 时不分析、不写 `chords.json`、不写 MIDI marker；GP5 模式下也不写和弦符号。

---

## 7. 输出目录

```text
output/<job_id>/
  stems/<selected>.wav       # keep_stems=true
  midi/<selected>.mid
  midi/full.mid
  song.gp5                   # 仅 GP5 模式；失败时可能不存在
  tracks.json                # 仅 GP5 模式；Score IR（可编辑）
  chords.json                # 和弦分析启用时；可能为空结果
  meta.json
  audio2gp_<job_id>.zip
work/<job_id>/
  <upload>.*
  song.wav
  stems/                     # 未 keep 时仍在此供试听
  pipeline.log
work/.stem_cache/<hash>/     # 四轨 + meta.json，不进入输出
```

`meta.json` 包含：状态、`output_format`、输入信息、BPM/来源/时长、`time_signature`/`bar_ticks`、设备、选项、`selected_stems`、`rhythm`（含 `mode`、`time_signature`、`beats_per_bar`、`beat_unit`、`bar_ticks`、`first_downbeat_s`、`auto_alignment_supported`）、`key`、`chords`（segments/summary，segment 带拍号相关 `bar`/`beat`）、警告、每轨音符数、小节数、轨道名、调弦、revision、耗时、文件路径、GP5 错误（若有）。MIDI-only 的 `files` 不含 `gp5`/`tracks`。ZIP 使用相对路径，排除 `.stem_cache`。

---

## 8. CUDA 与性能假设

- 系统：Ubuntu 22.04 / Python 3.10 / RTX 4090 24 GB；CUDA torch 2.6.0+cu124（用户 site，`.venv` 通过 `.pth` 复用，禁止 pip 再装 torch）。
- Demucs 默认 `htdemucs_ft`，约 321 MB 权重，首次从 Hugging Face 下载；`htdemucs` 约 80 MB，更快。缓存到 `~/.cache/huggingface`。
- 分离在 CUDA；Basic Pitch 0.4.0 的 TFLite 后端在 CPU；madmom 强拍在 CPU；指法/导出在 CPU。因此 GPU 只加速分轨，整曲耗时由 Basic Pitch CPU 推理主导。
- 20 秒测试片段在 RTX 4090 上约十余秒完成（含分轨、转录、GP5）。
- 4 分钟完整歌曲需要数百 MB 临时空间（WAV/stems/cache/ZIP）；输出目录不会自动清理。
- OOM 只对 Demucs 捕获并降级 CPU；Basic Pitch 内存不足会让 `transcribe` 失败并写入错误 meta。

---

## 9. 依赖与安装

系统依赖：`ffmpeg`、`ffprobe`。Python 3.10。

`requirements.txt` 包含 Basic Pitch、librosa、madmom-infer、PyGuitarPro、FastAPI、uvicorn、python-multipart、alphaTab 所在静态资源不需要 pip、Gradio（旧 UI 兼容）、pytest/httpx（测试）。torch/torchaudio 和 Demucs **不在** requirements 中：

```bash
python3 -m venv .venv
scripts/setup_env.sh
# 或手动：
.venv/bin/pip install -r requirements.txt -c constraints.txt
.venv/bin/pip install --no-deps demucs==4.1.0
.venv/bin/pip install -e . --no-deps
```

`setup_env.sh` 在最后才把用户 site 的 CUDA torch 加进 `.venv`，避免 pip 下载第二个 CPU torch；Demucs 的运行时依赖单独安装。首次 setup 会下载 Demucs 与 madmom 权重。详细故障排查见 README。

启动：

```bash
.venv/bin/audio2gp-web                  # http://127.0.0.1:7860
.venv/bin/python -m audio2gp.api --port 7860
.venv/bin/audio2gp song.mp3 -o output --device cuda
```

`main()` 拒绝绑定非 loopback 地址；不提供 `--share`/公网隧道。

---

## 10. 测试策略

**快速单元测试**（不加载模型）：

- `test_preprocess.py`：ffmpeg 归一化、时长/BPM、失败路径；
- `test_rhythm.py`：强拍相位、弱起偏移、librosa 回退、鼓速消歧、网格量化；
- `test_meter.py`：六种拍号的 `bar_ticks`/pulse/beams、非法拍号、Options 规范化、音符时值网格与拍号无关、各拍号的 GP5 header/小节总时值/MIDI time-signature 往返、拍号不匹配回读失败、和弦 bar/beat（含复合拍 pulse）、复合拍 `choose_grid_offset`、12/8 IR 校验/提交；
- `test_midi_post.py`：先聚类后量化、1/8/三连音/1/16/1/32 网格、短音/去重/单音化、MIDI tempo/通道、和弦 marker；
- `test_chords.py`：合成 C/G/Am/F 进行的标签与 tick、网格偏移、`chords.json` summary、缺失和声声部警告；
- `test_fingering.py`：合法位置、跨度、空弦优先、DP 连续性、调弦/调弦建议、技巧；
- `test_drums.py`：合成 kick/snare/hat 的 GM 映射、力度、去重、空轨；
- `test_gp_writer.py`：onset/tie、跨小节 tie、三连音、打击乐、每小节 voice 0 = 3840、回读、和弦符号与超范围警告；
- `test_score_ir.py`：稳定 ID、往返、改弦品、冲突、删除/静音、原子重建 GP5/MIDI/meta/ZIP、和弦持久化/重建/校验；
- `test_stem_cache.py`：内容寻址键、模型隔离、缓存命中复制、缺轨/空轨失效、不覆盖已有条目；
- `test_api.py`：上传校验、任务生命周期、下载、IR 编辑、revision 409、路径白名单、选项透传、任务恢复、MIDI-only 结果/和弦下载；
- `test_pipeline.py`：阶段顺序/单调进度、选择分轨组合、空轨、GP5 部分成功、MIDI-only 不写 GP5/IR、和弦 JSON/GP5/marker/meta、3/4 手动强拍不触发自动分析、非 4/4 无手动强拍时的警告；
- `test_web_app.py`：旧 Gradio 冒烟。

**慢测试**（`-m slow`，真实模型）：合成混音与真实歌曲片段；分轨非空、MIDI 音符数、GP5 回读每小节等于所选拍号小节长度（默认 4/4 为 3840）、ZIP、第二次运行结构一致。

快速套件在本机为 139 passed、3 deselected。

**浏览器测试**（手动/开发）：

- 用真实任务打开 `/?job=<id>`，确认 alphaTab 渲染、播放器就绪；
- 确认和弦时间线/下载 `chords.json`；MIDI-only 任务隐藏谱面并显示说明；
- 点击吉他音符 → 右栏出现候选弦品；确认已占用弦位被禁用；
- 应用候选 → revision+1，`song.gp5`/MIDI/ZIP 更新，alphaTab 重载；
- 删除音符、静音/取消静音；
- 两个标签页用旧 revision 编辑 → 第二个得到 409。

**验收门槛**：PyGuitarPro 回读 + 结构校验；本机没有 Guitar Pro/TuxGuitar，视觉最终验收需用户在制谱软件中打开。不能声称“专业级记谱”。

---

## 11. 许可证

| 组件 | 许可/备注 |
|---|---|
| 本项目代码 | MIT（见仓库） |
| TabForge 设计参考 | MIT；仅参考思路，不复制代码 |
| Demucs | MIT；模型权重按其发布条款 |
| Basic Pitch | Apache-2.0；ICASSP 2022 权重随包 |
| madmom-infer 代码 | BSD-2-Clause；**强拍权重 CC BY-NC-SA 4.0（非商用）** |
| PyGuitarPro | LGPL-3.0 |
| FastAPI/uvicorn | MIT/BSD |
| alphaTab | MPL-2.0（vendored `alphaTab.LICENSE`） |
| Bravura 字体 | SIL OFL 1.1（vendored `Bravura-OFL.txt`） |
| Sonivox SoundFont | Apache-2.0（vendored `sonivox.LICENSE`） |

商用前必须处理 madmom 强拍权重的非商用条款（关闭强拍对齐可避免加载）。

---

## 12. 已知限制与路线

**当前限制**：

- 每个任务只能固定一种用户选择的拍号（4/4、3/4、2/4、6/8、9/8、12/8），**无自动拍号识别**，无曲中变拍号/速度图；BPM 始终按四分音符计，复合拍的显示速度与“每分钟附点四分”的记法不同；
- 自动强拍模型仅支持 4/4；非 4/4 未手动指定首强拍时第 1 小节从音频开头开始并警告；强拍模型可能在硬摇摆、长自由前奏、串烧中锁错相位，用户可改用手动强拍或关闭对齐；
- 弱起写成完整前置小节，不是短 anacrusis；
- 旋律转录仅 Basic Pitch，未做多引擎/乐器特定模型；
- 鼓为启发式分类，桶鼓/镲片最不可靠；
- 指法是近似最优，和弦/高把位/特殊调弦可能八度折叠或丢音（有警告）；
- 和弦检测只给根音+品质，不识别转位/斜杠和弦/voicing/和弦图；power chord、暴露七度、密集失真可能被简化，低置信度会在 UI 标出；
- MIDI-only 没有谱面/编辑器；GP5 和弦符号只标注第一条非打击乐轨；
- 不检测推弦、颤音、闷音、泛音；
- 只写 GP5，不写 GPX/GP7/MusicXML；
- 编辑仅限改弦/品、删除、静音；无撤销、无批量移调、无 lead/rhythm 拆分；
- 任务是进程内内存记录，服务重启后无法从 UI 恢复（磁盘文件仍在）；
- alphaTab 只读 GP5，所有编辑必须经服务器；MIDI-only 不提供编辑。

**后续优先级**：

1. 手型搜索增加“保持音/和弦形状”约束，减少八度折叠；
2. 非 4/4 的自动强拍/拍号分析，或曲中拍号变化支持（当前由用户固定拍号并可手动指定首强拍）；
3. 更细的鼓件/力度模型；
4. 编辑器加入一次撤销与批量移调；
5. 多引擎转录接口（保持默认本地、无 LLM）；
6. 任务持久化/恢复。

---

## 13. 参考

- TabForge: https://github.com/rayBastard/tabforge
- Demucs: https://github.com/facebookresearch/demucs
- Basic Pitch: https://github.com/spotify/basic-pitch
- PyGuitarPro: https://github.com/Perlence/PyGuitarPro
- alphaTab: https://www.alphatab.net/
- madmom: https://github.com/CPJKU/madmom
- Krumhansl–Kessler key profiles（keydetect.py 中的实现说明）
