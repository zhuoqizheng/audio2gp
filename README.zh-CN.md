# audio2gp

[English](README.md) | **简体中文**

> **本地优先的「音频 → 分轨 → MIDI / 多轨 Guitar Pro 草稿」工具**
>
> Turn a mixed song into editable guitar/bass/drum tabs — locally.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Local](https://img.shields.io/badge/Audio-never_uploaded-success)

audio2gp 把一个完整混音（MP3 / WAV / FLAC / M4A / OGG）拆成 `vocals`、`drums`、`bass`、`other` 四个声部，转录为 MIDI，估计和弦进行，分配可弹的吉他/贝斯弦品，并导出多轨 `.gp5`、`tracks.json` 和每轨 MIDI。结果可以在浏览器里用 alphaTab 预听、播放和做轻量编辑，再让服务器重新生成谱面。

**所有音频处理都在本机完成。** 没有云端推理、没有 LLM 生成谱面、不上传音频、不抓取商业谱库；FastAPI 只监听 `127.0.0.1`。

> 输出是**可继续编辑的草稿**，不是专业制谱师级别的成品。鼓件、和弦与高把位指法都可能需要人工修正。

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [拍号与 shuffle](#拍号与-shuffle)
- [输出文件](#输出文件)
- [在线谱面编辑器](#在线谱面编辑器)
- [HTTP API](#http-api)
- [实现架构](#实现架构)
- [测试](#测试)
- [已知限制](#已知限制)
- [隐私与许可证](#隐私与许可证)
- [参考](#参考)

## 特性

- **本地分轨**：Demucs `htdemucs_ft`（更干净、较慢）或 `htdemucs`。完整四轨使用内容寻址缓存，换分轨选择不会重新跑 Demucs。
- **旋律转录**：Basic Pitch 0.4.0 TFLite 在 CPU 上运行；鬼音过滤、短音过滤、复音上限，人声/贝斯强制单音。
- **鼓轨**：基于 onset 和频段的启发式分类（底鼓、军鼓、闭镲、开镲、吊镲、桶鼓），鼓点优先的速度校正与半/倍速消歧。
- **拍号由用户选择**：4/4、3/4、2/4、6/8、9/8、12/8；不做自动拍号识别。4/4 可使用自动强拍分析，所有拍号都支持手动首强拍。
- **和弦分析**：拍同步 chroma + 谐波抑制 + Viterbi 平滑，输出 `chords.json`，写入 GP5 和弦符号和 `full.mid` marker。
- **可弹指法**：吉他/贝斯候选位置枚举 + 手位/换弦/大跳代价的 Viterbi 搜索；标准、Drop D、降半音调弦与调弦建议。
- **保守技巧**：只标记高置信度的 hammer-on / pull-off / 滑音，不猜测推弦、颤音、闷音、泛音。
- **多轨 GP5**：Guitar、Bass、Drums、可选 Vocals；拍号、小节线、休止、tie、打击乐和调号均按所选拍号生成。
- **两种输出模式**：`gp5_midi`（默认，GP5 + 编辑器）或 `midi_only`（跳过指法/GP5/编辑器，仍交付 MIDI、stems、和弦、元数据、ZIP）。
- **在线编辑**：alphaTab 预览/播放；点击音符可换弦/换品、删除、静音；每次编辑带 revision，服务器重新生成文件，旧页面写入返回 409。
- **单机本地 UI**：FastAPI + 静态 HTML，无需前端构建步骤；另保留一个旧版 Gradio 入口供兼容使用。

## 快速开始

### 环境要求

- **Linux**（开发机为 Ubuntu 22.04）
- **Python 3.10**：Basic Pitch 0.4.0 的 `tflite-runtime` wheel 目前限定 Linux / Python 3.10；`scripts/setup_env.sh` 会拒绝其他版本
- `ffmpeg` 和 `ffprobe` 在 `PATH` 中（Debian/Ubuntu：`sudo apt install ffmpeg`）
- NVIDIA GPU 推荐；CPU 也能运行，但 Demucs 与 Basic Pitch 都会明显变慢
- 一个现代浏览器（结果页使用 alphaTab）

Basic Pitch 在 CPU 上通过 TFLite 运行；madmom 强拍模型也在 CPU 上运行。GPU 主要加速 Demucs。

### 安装

`scripts/setup_env.sh` 会复用 **user site 中已安装的 torch/torchaudio**，并避免 pip 下载第二份 CPU torch。先按 PyTorch 官方方式在 user site 安装与本机 CUDA 匹配的 torch，例如开发机测试过的组合：

```bash
python3.10 -m pip install --user \
  torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

> CPU-only 环境也可以安装 CPU 版 torch，只是处理整首歌会非常慢。不要在普通 `pip install -r requirements.txt` 中安装 torch 或 Demucs，setup 脚本会处理 Demucs 的隔离安装。

然后在仓库根目录运行：

```bash
git clone https://github.com/zhuoqizheng/audio2gp.git
cd audio2gp
scripts/setup_env.sh
```

首次 setup 会下载 Demucs 权重和 madmom 强拍权重；Basic Pitch 的模型随 wheel 提供，无需单独下载。

### 启动网页

```bash
.venv/bin/audio2gp-web
# 打开 http://127.0.0.1:7860
```

服务只绑定 loopback，拒绝公网地址，也不提供 share/隧道模式。

### 命令行

```bash
.venv/bin/audio2gp song.mp3 -o output
.venv/bin/audio2gp song.mp3 -o output --stems other,vocals
.venv/bin/audio2gp song.mp3 -o output --time-signature 12/8 \
  --manual-downbeat 0.85 --quantize 8
.venv/bin/audio2gp song.mp3 -o output --midi-only
```

完整参数：

```bash
.venv/bin/audio2gp --help
```

最常用的选项：

| 选项 | 作用 |
|---|---|
| `--time-signature {4/4,3/4,2/4,6/8,9/8,12/8}` | 固定拍号；必须由用户指定，本工具不做自动拍号识别 |
| `--manual-downbeat SECONDS` | 第一个强拍在音频中的时间；支持所有拍号，`0` 表示不手动指定 |
| `--quantize {8,12,16,32}` | 量化网格：1/8、1/8 三连音、1/16、1/32；按音符时值定义，与拍号无关 |
| `--stems LIST` | 只转录/交付的声部，例如 `other,vocals`；Demucs 仍会分离四轨 |
| `--midi-only` | 只导出 MIDI，跳过指法、GP5、`tracks.json` 和编辑器 |
| `--no-align-downbeats` | 关闭自动强拍分析；手动强拍仍然生效 |
| `--bpm 0` | `0` = 自动估计，否则覆盖 BPM（四分音符 BPM） |
| `--no-chords` | 关闭和弦分析及和弦产物 |

网页和 CLI 的推荐流程都是：先听原曲，确定拍号与第一个强拍，再上传；这样小节线最可靠。

## 使用方式

1. **上传**一首完整混音。默认限制为 200 MB / 600 秒，可通过 CLI 参数或环境变量调整。
2. **选择声部**。只勾选需要的声部；例如只有吉他和人声时选 `other` + `vocals`。
3. **设置拍号**（可选手动强拍、BPM、量化、调弦）。默认 `4/4`、自动估计 BPM、1/16 网格。
4. **开始转换**。流程为：归一化 → 节奏/强拍 → Demucs 分轨 → Basic Pitch/鼓转录 → 清理量化 → 和弦 → 指法 → GP5/MIDI → 编辑状态。
5. 在结果页试听各分轨、查看和弦时间线、打开 alphaTab 谱面、下载文件。
6. 如需调整，点击音符修改弦/品或删除，也可以静音轨道；每次保存都会重新生成 GP5/MIDI/ZIP。

GP5 模式失败时任务会标记为 `partial`，已经生成的 MIDI、stems 和 ZIP 仍会保留；只导出 MIDI 的任务不会因为“没有 GP5”而失败。

## 拍号与 shuffle

**shuffle/swing 不是拍号，而是一种演奏感觉。** 判定拍号时看歌曲实际怎么数拍：

| 音乐感觉 | 选什么拍号 | 量化网格 |
|---|---|---|
| 慢布鲁斯，或乐队按 12/8 数 | **12/8** | **1/8**（每小节 12 个八分音符） |
| 按 4/4 数、八分音符摇摆（shuffle/swing） | **4/4** | **1/8 三连音**（量化值 12） |
| 6/8 的慢歌 | **6/8** | **1/8** |

要点：

- 12/8 每小节有 4 个附点四分 pulse，每个 pulse 分成三个八分音符，最适合慢速 shuffle/布鲁斯。
- 4/4 swing 的八分音符成对演奏，时值约 2:1；使用 1/8 三连音网格才能保留摇摆感。
- 不要用普通 1/8 或 1/16 网格转录 shuffle，否则摇摆会被量化成直音或附点节奏。
- 本工具的 BPM 一律是**四分音符 BPM**。12/8 的律动单位是附点四分（1440 ticks），所以谱面显示的 BPM 不一定等于“每分钟附点四分”的常见记法。

### 小节对齐

- **4/4** 默认用 madmom 检测强拍；如果开头是弱起，第一个强拍会放到第 2 小节或更后，前面用完整小节保留弱起材料，而不是写一个短小的 anacrusis 小节。
- **其他拍号**不会自动检测强拍，第 1 小节默认从音频开头开始并给出警告。填写**手动强拍（秒）**可把首强拍放在任意拍号的小节线上。
- 自动强拍模型本身只支持 4/4；选择 12/8 并不会让它分析 12/8。
- 手动强拍优先于自动分析，即使关闭了“自动强拍对齐”也仍然生效。

## 输出文件

```text
output/<job_id>/
  stems/<selected stems>.wav   # keep_stems=true 时
  midi/<selected stem>.mid
  midi/full.mid
  song.gp5                     # GP5 模式；生成失败时可能缺失
  tracks.json                  # GP5 模式；编辑器 IR
  chords.json                  # 和弦分析启用时
  meta.json
  audio2gp_<job_id>.zip
work/<job_id>/
  song.wav                     # 44.1 kHz / stereo / PCM s16
  stems/                       # 完整四轨（未保留时仍供 UI 试听）
  pipeline.log
work/.stem_cache/<hash>/       # Demucs 分轨缓存，不进入 ZIP
```

- `stems/` 只包含勾选的声部；未勾选的轨仍保留在 `work/` 供诊断。
- `chords.json` 包含和弦段（起止 tick/时间、label、root、quality、confidence、bar、beat）和摘要。
- `tracks.json` 是稳定的编辑器中间表示（IR），保存 revision、音符 ID、弦品、和弦和静音状态。
- `meta.json` 记录状态、BPM、拍号、小节长度、选项、警告、耗时、文件清单和错误信息。
- 任务不会自动清理；`output/` 和 `work/` 由用户自行管理。首次运行和整曲处理会占用数百 MB 空间。

## 在线谱面编辑器

- 谱面由 alphaTab 在浏览器中渲染和播放；alphaTab、Bravura 字体和 Sonivox SoundFont 都已 vendored，可离线使用。
- 点击音符 → 右侧列出保持音高的其他 `(string, fret)` 位置；被同一时刻其他音占用的弦位会禁用。
- **应用**修改、**删除音符**、**静音轨道**。所有写操作都带 revision；两个标签页同时编辑时，旧页面会收到 409，需要刷新。
- 浏览器**不直接写 GP5**。编辑请求只修改服务器上的 `tracks.json`，再由同一条 PyGuitarPro/MIDI 导出链路原子重建 `song.gp5`、MIDI、`meta.json` 和 ZIP。
- MIDI-only 结果没有谱面和编辑器，但仍有和弦时间线与全部 MIDI 下载。

## HTTP API

网页是这个 API 的客户端：

| 方法 | 路径 | 作用 |
|---|---|---|
| `POST` | `/api/jobs` | 上传 `file` + JSON `options`，返回 202 和任务 ID |
| `GET` | `/api/jobs/{id}` | 状态、进度、日志和完成结果 |
| `GET` | `/api/jobs/{id}/state` | 读取编辑器 IR（仅 GP5 模式） |
| `GET` | `/api/jobs/{id}/score.gp5` | 当前 GP5 |
| `GET` | `/api/jobs/{id}/audio/{stem}` | 试听分轨 |
| `GET` | `/api/jobs/{id}/files/{path}` | 白名单文件下载 |
| `PATCH` | `/api/jobs/{id}/tracks/{tid}/notes/{nid}` | 修改音符弦/品 |
| `DELETE` | `/api/jobs/{id}/tracks/{tid}/notes/{nid}` | 删除音符 |
| `PUT` | `/api/jobs/{id}/tracks/{tid}/mute` | 设置轨道静音 |
| `GET` | `/api/health`, `/api/limits` | 健康检查与 UI 限制 |

`POST /api/jobs` 的关键选项：

```json
{
  "time_signature": "12/8",
  "manual_downbeat_s": 0.85,
  "align_downbeats": true,
  "quantize": 8,
  "selected_stems": ["other", "vocals"],
  "detect_chords": true,
  "export_gp5": true
}
```

结果额外返回 `time_signature`、`bar_ticks`、`manual_downbeat_s`、和弦片段和 `output_format`。`/api/limits` 会公布可选拍号、`manual_downbeat` 支持以及 `auto_downbeat_time_signatures: ["4/4"]`。

## 实现架构

```text
上传音频
  → ffmpeg 归一化与初始 BPM
  → 用户拍号 / 手动首强拍（4/4 可自动检测）
  → Demucs 四轨分离（内容寻址缓存）
  → Basic Pitch 旋律 + 鼓点启发式
  → 起音聚类、量化、鬼音清理、单音化
  → 和弦分析（chroma + Viterbi）
  → 手型候选 + DP 指法（GP5 模式）
  → GP5 / 每轨 MIDI / full.mid / tracks.json
  → FastAPI + alphaTab 在线编辑与重导出
```

关键设计：

- **批处理与编辑器解耦**：`pipeline.py` 产出初稿和 IR；`score_ir.py` 只读/改 IR 并调用同一套导出函数。
- **单一 GPU worker**：API 串行执行任务，避免多个 Demucs 同时占显存。
- **内容寻址分轨缓存**：缓存键包含归一化 WAV 哈希、模型和分离参数；改分轨选择只重跑转录。
- **MIDI-only 是独立路径**：不构造编辑器状态，不运行指法搜索，也不生成 `song.gp5`/`tracks.json`。
- **所有写操作原子发布**：临时文件成功后再 `os.replace`，失败不会破坏上一版谱面。

完整阶段说明、模块签名和算法细节见 [`system_design.md`](system_design.md)。

## 测试

```bash
# 快速单元/集成测试（不加载真实模型）
.venv/bin/python -m pytest -q

# 真实 Demucs / Basic Pitch 端到端测试（首次会下载权重，耗时较长）
.venv/bin/python -m pytest -q -m slow
```

开发机当前结果：快速套件 **139 passed**，慢速套件 **3 passed**。慢速测试覆盖真实分轨、转录、GP5 回读、ZIP 和旧版 Gradio 冒烟。

GP5 的验收方式是 PyGuitarPro 回读 + 结构校验：轨道数、两个 voice、每个 voice 0 合计等于所选拍号的小节长度、tie、打击乐通道和拍号 header。本机没有 Guitar Pro/TuxGuitar，因此建议在制谱软件里打开生成文件做最终视觉检查。

## 已知限制

- **没有自动拍号识别**：每个任务只能固定一种用户选择的拍号；没有曲中变拍号或速度图。
- **自动强拍分析只支持 4/4**：非 4/4 必须手动指定首强拍，否则第 1 小节从音频开头开始。
- BPM 始终按四分音符计算；复合拍的显示速度可能与“每分钟附点四分”的习惯不同。
- 转录只使用 Basic Pitch；不是多引擎/乐器特定模型。鼓件、桶鼓和镲片尤其依赖启发式。
- 指法是可弹性近似值；困难和弦可能被八度折叠、裁掉声部或跳过（都有警告）。
- 和弦只估计根音 + 品质，不识别转位/斜杠和弦/voicing/和弦图；power chord、复杂延伸音和失真混音可能被简化。
- 技巧仅限保守的 hammer-on/pull-off 与滑音；没有推弦、颤音、闷音、泛音或弯音轨。
- 只写 GP5（5.1），不写 GPX/GP7/MusicXML；编辑器只支持改弦/品、删除和静音，没有撤销栈或批量移调。
- 任务是进程内内存记录；服务重启后 UI 任务列表丢失，但 `output/` 文件仍在。
- Basic Pitch 在 CPU 上推理，整曲处理时间是主要瓶颈；首次运行还需要下载模型权重。
- 输出是草稿，不能声称达到专业制谱质量。

## 隐私与许可证

### 隐私

- 音频只在本地处理，不会上传到任何远端服务；没有云端推理 API，也没有 LLM。
- Web 服务只监听 `127.0.0.1`，拒绝非 loopback 地址，不提供 `--share` 或公网隧道。
- 首次安装会从上游下载 Demucs/madmom 权重；这是模型文件下载，不是音频上传。
- 结果页的 alphaTab、字体和音色资源已随仓库提供，离线可用。

### 许可证

本项目代码使用 **MIT** 许可，见 [`LICENSE`](LICENSE) 和 `pyproject.toml`。

| 组件 | 许可 / 说明 |
|---|---|
| audio2gp | MIT |
| Demucs | MIT；预训练权重按其上游发布条款使用 |
| Basic Pitch | Apache-2.0；ICASSP 2022 TFLite 模型随包 |
| madmom-infer | 代码 BSD-2-Clause；**强拍权重 CC BY-NC-SA 4.0（非商用、相同方式共享）** |
| PyGuitarPro | LGPL-3.0 |
| FastAPI / uvicorn | MIT / BSD |
| alphaTab（vendored） | MPL-2.0，许可证文件随资源分发 |
| Bravura 字体（vendored） | SIL OFL 1.1 |
| Sonivox SoundFont（vendored） | Apache-2.0 |
| Gradio（旧 UI） | Apache-2.0 |

**商用注意**：默认强拍对齐会加载 madmom 的 CC BY-NC-SA 权重。需要商用时应关闭强拍对齐（`--no-align-downbeats` 或网页开关），或者替换为可商用的强拍模型；关闭后没有任何 madmom 权重被加载。

Guitar Pro 是 Arobas Music 的商标。本项目不内嵌 Guitar Pro，也不与之关联；谱面预览使用 MPL-2.0 的 alphaTab，服务器端使用 PyGuitarPro 写 GP5。

## 参考

- [system_design.md](system_design.md)：完整算法、阶段、模块接口与 API 设计
- [TabForge](https://github.com/rayBastard/tabforge)：架构与工程思路参考（未复制代码）
- [Demucs](https://github.com/facebookresearch/demucs)
- [Basic Pitch](https://github.com/spotify/basic-pitch)
- [PyGuitarPro](https://github.com/Perlence/PyGuitarPro)
- [alphaTab](https://www.alphatab.net/)
- [madmom](https://github.com/CPJKU/madmom)
