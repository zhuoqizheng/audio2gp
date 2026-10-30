"use strict";

const $ = (selector) => document.querySelector(selector);
const POLL_MS = 1500;

const STEM_LABELS = { vocals: "人声", drums: "鼓", bass: "贝斯", other: "吉他 / 其他" };
const STAGE_ORDER = ["preprocess", "rhythm", "separate", "analyze", "transcribe", "post", "gp5"];
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function outputMode() {
  return $("#exportFormatSelect") && $("#exportFormatSelect").value === "midi_only"
    ? "midi_only"
    : "gp5_midi";
}

function startLabel() {
  return outputMode() === "midi_only" ? "开始分析并仅导出 MIDI" : "开始分析并生成 GP5";
}

function updateOutputFormatUi() {
  const hint = $("#outputFormatHint");
  if (hint) {
    hint.textContent = outputMode() === "midi_only"
      ? "仅 MIDI 模式跳过指法搜索、song.gp5、tracks.json 和在线改谱；仍导出分轨 MIDI、full.mid、和弦、分轨、元数据与 ZIP。"
      : "GP5 模式需要所选声部能产生至少一条弦乐/鼓/人声轨，并额外生成 song.gp5、tracks.json 与在线改谱。";
  }
  const button = $("#startBtn");
  if (button && !button.disabled) button.textContent = startLabel();
}

function doneMessage(job) {
  if (job && job.partial) return "完成，但 GP5 部分失败；MIDI、和弦与 ZIP 仍可下载。";
  if (job && job.result && job.result.output_format === "midi_only") {
    return "任务完成：已生成 MIDI 与和弦（未生成 GP5）。";
  }
  return "任务完成。";
}

const ui = {
  jobId: null,
  job: null,
  result: null,
  scoreState: null,
  selected: null,
  at: null,
  solo: new Set(),
  flags: [],
  pollTimer: null,
  pollFailures: 0,
  toastTimer: null,
  loadingResult: false,
  maxUploadBytes: 200 * 1024 * 1024,
};

function noteName(pitch) {
  const p = Number(pitch);
  if (!Number.isFinite(p) || p < 0 || p > 127) return "—";
  return `${NOTE_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
}

function toInt(id, fallback) {
  const value = Number.parseInt($(id).value, 10);
  return Number.isFinite(value) ? value : fallback;
}

function toFloat(id, fallback) {
  const value = Number.parseFloat($(id).value);
  return Number.isFinite(value) ? value : fallback;
}

// Meter geometry mirrors audio2gp.config.Meter: ticks always count quarter
// notes (960 each), and compound eighth meters group three eighths per beat.
function meterInfo(result) {
  const barTicks = Number(result && result.bar_ticks) || 3840;
  const signature = String((result && result.time_signature) || "4/4");
  const [numerator, denominator] = signature.split("/").map(Number);
  const safeNumerator = Number.isFinite(numerator) && numerator > 0 ? numerator : 4;
  const safeDenominator = Number.isFinite(denominator) && denominator > 0 ? denominator : 4;
  const compound = safeDenominator === 8 && safeNumerator >= 6 && safeNumerator % 3 === 0;
  const pulses = compound ? safeNumerator / 3 : safeNumerator;
  return {
    signature,
    barTicks,
    pulses,
    pulseTicks: Math.max(1, Math.round(barTicks / Math.max(1, pulses))),
  };
}

function barNumber(tick) {
  const meter = meterInfo(ui.result || ui.scoreState);
  return Math.floor(Number(tick) / meter.barTicks) + 1;
}

function mmss(seconds) {
  if (!Number.isFinite(Number(seconds))) return "—";
  const total = Math.max(0, Math.round(Number(seconds)));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function errDetail(payload, fallback) {
  const detail = payload && payload.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((entry) => entry && (entry.msg || entry.message) ? (entry.msg || entry.message) : String(entry)).join("；");
  }
  if (detail) return JSON.stringify(detail);
  return fallback;
}

async function readJson(response) {
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (!response.ok) {
    throw new Error(errDetail(data, `HTTP ${response.status}`));
  }
  return data;
}

function setFormStatus(message, isError = false) {
  const node = $("#formStatus");
  node.textContent = message;
  node.classList.toggle("error", Boolean(isError));
}

function showToast(message, isError = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", Boolean(isError));
  node.hidden = false;
  window.clearTimeout(ui.toastTimer);
  ui.toastTimer = window.setTimeout(() => { node.hidden = true; }, 4200);
}

function setBusy(busy) {
  $("#startBtn").disabled = busy;
  $("#startBtn").textContent = busy ? "处理中…" : startLabel();
  const format = $("#exportFormatSelect");
  if (format) format.disabled = busy;
}

function resetForNewJob() {
  window.clearTimeout(ui.pollTimer);
  ui.jobId = null;
  ui.job = null;
  ui.result = null;
  ui.scoreState = null;
  ui.selected = null;
  ui.solo.clear();
  ui.flags = [];
  ui.loadingResult = false;
  ui.pollFailures = 0;
  renderUndo(false);
  renderFlags();
  $("#resultPanel").hidden = true;
  $("#jobPanel").hidden = true;
  $("#scoreView").innerHTML = "";
  $("#chordPanel").hidden = true;
  $("#noScoreNote").hidden = true;
  $("#scoreWorkspace").hidden = false;
  $("#chordTimeline").textContent = "";
  $("#chordProgression").textContent = "";
  $("#playBtn").disabled = true;
  $("#playPos").textContent = "尚未加载乐谱";
  closeNoteEditor();
  if (ui.at) {
    try { ui.at.destroy(); } catch { /* ignore */ }
    ui.at = null;
  }
}

function optionsPayload() {
  const selected = [];
  if ($("#stemOther").checked) selected.push("other");
  if ($("#stemBass").checked) selected.push("bass");
  if ($("#stemDrums").checked) selected.push("drums");
  if ($("#stemVocals").checked) selected.push("vocals");
  const bpm = toFloat("#bpmOverride", 0);
  const manualDownbeat = toFloat("#manualDownbeat", 0);
  return {
    demucs_model: $("#modelSelect").value,
    transcription_engine: $("#engineSelect").value,
    quantize: toInt("#quantizeSelect", 16),
    time_signature: $("#timeSignatureSelect").value,
    manual_downbeat_s: manualDownbeat > 0 ? manualDownbeat : null,
    guitar_stem: $("#guitarStemSelect").value,
    keep_stems: $("#keepStems").checked,
    max_fret: toInt("#maxFret", 15),
    min_note_sec: toFloat("#minNoteSec", 0.03),
    device: $("#deviceSelect").value,
    include_vocals: $("#includeVocals").checked,
    bpm_override: bpm > 0 ? bpm : null,
    align_downbeats: $("#alignDownbeats").checked,
    guitar_tuning: $("#guitarTuningSelect").value,
    bass_tuning: $("#bassTuningSelect").value,
    suggest_tuning: $("#suggestTuning").checked,
    ghost_amplitude: toFloat("#ghostAmplitude", 0.15),
    max_polyphony: toInt("#maxPolyphony", 6),
    onset_threshold: toFloat("#onsetThreshold", 0.5),
    frame_threshold: toFloat("#frameThreshold", 0.3),
    detect_techniques: $("#detectTechniques").checked,
    detect_key: $("#detectKey").checked,
    tempo_from_drums: $("#tempoFromDrums").checked,
    cache_stems: $("#cacheStems").checked,
    selected_stems: selected,
    layout: $("#layoutSelect").value,
    export_gp5: outputMode() === "gp5_midi",
    detect_chords: $("#detectChords").checked,
  };
}

async function loadLimits() {
  try {
    const response = await fetch("/api/limits", { cache: "no-store" });
    const limits = await readJson(response);
    const mb = Number(limits.max_upload_mb);
    const seconds = Number(limits.max_duration_s);
    if (Number.isFinite(mb) && mb > 0) {
      ui.maxUploadBytes = mb * 1024 * 1024;
      const minutes = Number.isFinite(seconds) ? Math.round(seconds / 60) : null;
      $("#limitsNote").textContent = `当前上限 ${mb} MB${minutes ? ` / 约 ${minutes} 分钟` : ""}，可在启动参数中调整。`;
    }
    const external = $("#engineSelect") && $("#engineSelect").querySelector('option[value="external"]');
    if (external && limits.external_engine_configured === false) {
      external.disabled = true;
      const hint = $("#engineHint");
      if (hint) {
        hint.textContent = "当前未配置 AUDIO2GP_EXTERNAL_CMD，外部引擎不可选；设置该环境变量并重启服务后可选用。";
      }
    }
  } catch { /* keep the documented defaults when the endpoint is unavailable */ }
}

function validateForm(file) {
  if (!file) return "请先选择一个音频或 MIDI 文件。";
  if (file.size > ui.maxUploadBytes) {
    return `文件为 ${(file.size / (1024 * 1024)).toFixed(1)} MB，超过当前上传上限。请缩短文件或调整启动参数。`;
  }
  const selected = ["stemOther", "stemBass", "stemDrums", "stemVocals"]
    .filter((id) => $(`#${id}`).checked);
  if (!selected.length) return "请至少勾选一个参与转谱的声部。";
  if (outputMode() === "gp5_midi") {
    const source = $("#guitarStemSelect").value;
    const gp5Tracks = [];
    if (selected.includes("stemOther") && source === "other") gp5Tracks.push("Guitar");
    if (selected.includes("stemVocals") && source === "vocals") gp5Tracks.push("Guitar");
    if (selected.includes("stemBass")) gp5Tracks.push("Bass");
    if (selected.includes("stemDrums")) gp5Tracks.push("Drums");
    if (selected.includes("stemVocals") && source !== "vocals" && $("#includeVocals").checked) {
      gp5Tracks.push("Vocals");
    }
    if (!gp5Tracks.length) {
      return "GP5 模式下所选声部不会产生任何 GP5 轨道；请勾选能生成 Guitar/Bass/Drums/Vocals 的声部，或把输出格式改为仅 MIDI。";
    }
  }
  return "";
}

function renderJob(job) {
  $("#jobPanel").hidden = false;
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  $("#stageLabel").textContent = job.stage_label || job.stage || "处理中";
  $("#progressPct").textContent = `${Math.round(progress)}%`;
  $("#progressFill").style.width = `${progress}%`;
  $("#progressBar").setAttribute("aria-valuenow", String(Math.round(progress)));
  $("#jobMessage").textContent = job.message || "";
  $("#jobIdLabel").textContent = job.id ? `任务 ${job.id}` : "";
  const lines = (job.log || []).map((entry) => {
    const stamp = entry.time ? new Date(entry.time).toLocaleTimeString() : "";
    const label = entry.stage_label || entry.stage || "";
    return `${stamp}  [${label}] ${entry.message || ""}`;
  });
  const box = $("#logBox");
  $("#logWrap").hidden = lines.length === 0;
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 12;
  box.textContent = lines.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

async function startJob(event) {
  event.preventDefault();
  const file = $("#fileInput").files[0];
  const problem = validateForm(file);
  if (problem) {
    setFormStatus(problem, true);
    return;
  }
  resetForNewJob();
  setFormStatus("正在上传并排队…");
  setBusy(true);
  $("#jobPanel").hidden = false;
  $("#jobIdLabel").textContent = "";
  $("#stageLabel").textContent = "上传中";
  $("#progressPct").textContent = "0%";
  $("#progressFill").style.width = "0%";
  $("#jobMessage").textContent = "文件正在写入本机工作目录…";
  $("#logBox").textContent = "";

  const form = new FormData();
  form.append("file", file);
  form.append("options", JSON.stringify(optionsPayload()));
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    const data = await readJson(response);
    ui.jobId = data.id;
    ui.pollFailures = 0;
    rememberJob(data.id);
    setFormStatus("任务已创建，可继续向下查看进度。");
    schedulePoll(300);
  } catch (error) {
    setBusy(false);
    setFormStatus(`启动失败：${error.message}`, true);
    $("#jobMessage").textContent = error.message;
    $("#stageLabel").textContent = "失败";
  }
}

function rememberJob(jobId) {
  try {
    const url = new URL(window.location.href);
    url.searchParams.set("job", jobId);
    window.history.replaceState({}, "", url);
  } catch { /* URL persistence is optional */ }
}

async function restoreJobFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const jobId = (params.get("job") || "").trim();
  if (!jobId) return;
  ui.jobId = jobId;
  setBusy(true);
  $("#jobPanel").hidden = false;
  $("#jobIdLabel").textContent = `任务 ${jobId}`;
  $("#stageLabel").textContent = "读取任务";
  $("#jobMessage").textContent = "正在从本机任务内存中恢复进度…";
  $("#resultPanel").hidden = false;
  $("#scoreHint").textContent = "任务恢复后将在这里加载乐谱。";
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
    const job = await readJson(response);
    ui.job = job;
    renderJob(job);
    if (job.status === "done" && job.result) {
      setBusy(false);
      setFormStatus(job.partial ? "任务已完成（GP5 部分失败）。" : "已恢复完成的任务。");
      await loadResult(job);
      return;
    }
    if (job.status === "error") {
      setBusy(false);
      setFormStatus(`该任务处理失败：${job.error || "未知错误"}`, true);
      return;
    }
    setFormStatus("任务仍在处理，已继续接收进度。");
    schedulePoll(300);
  } catch (error) {
    setBusy(false);
    $("#resultPanel").hidden = true;
    setFormStatus(`无法恢复该任务：${error.message}。服务器重启会丢失任务内存记录，请重新上传。`, true);
  }
}

function schedulePoll(delay = POLL_MS) {
  window.clearTimeout(ui.pollTimer);
  ui.pollTimer = window.setTimeout(pollJob, delay);
}

async function pollJob() {
  if (!ui.jobId) return;
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(ui.jobId)}`, { cache: "no-store" });
    const job = await readJson(response);
    ui.pollFailures = 0;
    ui.job = job;
    renderJob(job);
    if (job.status === "done") {
      setBusy(false);
      setFormStatus(doneMessage(job));
      if (job.result && !ui.loadingResult && !ui.result) {
        await loadResult(job);
      }
      return;
    }
    if (job.status === "error") {
      setBusy(false);
      setFormStatus(`处理失败：${job.error || "未知错误"}`, true);
      return;
    }
    schedulePoll();
  } catch (error) {
    ui.pollFailures += 1;
    if (ui.pollFailures >= 4) {
      setBusy(false);
      setFormStatus(`无法读取任务进度：${error.message}`, true);
      return;
    }
    schedulePoll(Math.min(8000, POLL_MS * (ui.pollFailures + 1)));
  }
}

async function loadResult(job) {
  ui.loadingResult = true;
  ui.job = job;
  ui.result = job.result;
  ui.jobId = job.id;
  // Keep the form consistent with a result restored from ?job=<id>, so the
  // visible output mode and start-button label are not stale.
  const formatSelect = $("#exportFormatSelect");
  if (formatSelect && job.result.output_format) {
    formatSelect.value = job.result.output_format;
  }
  if ($("#detectChords") && typeof job.result.chord_detection === "boolean") {
    $("#detectChords").checked = job.result.chord_detection;
  }
  if ($("#timeSignatureSelect") && job.result.time_signature) {
    $("#timeSignatureSelect").value = job.result.time_signature;
  }
  if ($("#manualDownbeat")) {
    $("#manualDownbeat").value = job.result.manual_downbeat_s
      ? Number(job.result.manual_downbeat_s).toFixed(2)
      : "0";
  }
  if ($("#alignDownbeats") && typeof job.result.align_downbeats === "boolean") {
    $("#alignDownbeats").checked = job.result.align_downbeats;
  }
  updateOutputFormatUi();
  $("#resultPanel").hidden = false;
  renderSummary(job.result);
  renderChords(job.result);
  renderDownloads(job.result);
  renderWarnings(job.result);
  renderStemPlayers(job.result);
  try {
    if (job.result.state_url) {
      const stateResponse = await fetch(job.result.state_url, { cache: "no-store" });
      ui.scoreState = await readJson(stateResponse);
    } else {
      ui.scoreState = null;
    }
  } catch (error) {
    ui.scoreState = null;
    showToast(`读取编辑状态失败：${error.message}`, true);
  }
  ui.flags = Array.isArray(job.result.flags) ? job.result.flags : [];
  renderMixer();
  renderFlags();
  renderUndo(Boolean(job.result.can_undo));
  const canScore = Boolean(job.result.state_url && job.result.score_url && ui.scoreState);
  $("#scoreWorkspace").hidden = !canScore;
  $("#noScoreNote").hidden = canScore;
  $("#resultTitle").textContent = canScore ? "预览与轻量改谱" : "结果与下载";
  $("#resultCardNote").innerHTML = canScore
    ? "alphaTab 负责在线显示与播放；浏览器不能写 GP5。每次修改都先写入 <code>tracks.json</code>，再由服务器的 PyGuitarPro 导出链路重新生成 <code>song.gp5</code>。"
    : "当前任务没有可在线预览的 GP5。下载区仍提供 MIDI、和弦结果与 ZIP；需要 GP5 与在线改谱时，请选择“GP5 + MIDI”输出格式并重新处理。";
  if (canScore) {
    $("#noScoreNote").hidden = true;
    loadScore(job.result.score_url);
  } else {
    if (job.result.output_format === "midi_only") {
      $("#noScoreTitle").textContent = "仅 MIDI 模式：本次没有生成 GP5。";
      $("#noScoreText").textContent = "分轨 MIDI、full.mid、和弦分析、分轨试听和 ZIP 仍可下载；和弦时间线显示在下方。若需要多轨 GP5 与在线改谱，请改用“GP5 + MIDI”输出格式。";
    } else {
      $("#noScoreTitle").textContent = "本次 GP5 或在线编辑状态未能生成。";
      $("#noScoreText").textContent = "MIDI、和弦结果、分轨与 ZIP 仍可下载；请查看下方处理警告了解失败原因，或重新处理。";
    }
    $("#playBtn").disabled = true;
    if (ui.at) {
      try { ui.at.destroy(); } catch { /* ignore */ }
      ui.at = null;
    }
  }
  ui.loadingResult = false;
  if (ui.jobId) rememberJob(ui.jobId);
  $("#resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSummary(result) {
  const summary = $("#summary");
  summary.textContent = "";
  const trackNames = (result.tracks || []).map((track) => track.name).filter(Boolean);
  const midiOnly = result.output_format === "midi_only";
  const chordSegments = result.chords || [];
  const chordSummary = result.chord_summary || {};
  const uniqueChords = chordSummary.unique_chords || [];
  let chordText = "—";
  if (result.input_kind === "midi") chordText = "MIDI 输入跳过";
  else if (result.chord_detection === false) chordText = "已关闭";
  else if (chordSegments.length) chordText = `${chordSegments.length} 段 · ${uniqueChords.length} 种`;
  else if (result.chord_detection) chordText = "未识别";
  const items = [
    ["状态", result.status === "partial" ? "部分成功" : "完成"],
    ["输入", result.input_kind === "midi" ? "MIDI" : "音频"],
    ["输出", midiOnly ? "仅 MIDI" : "GP5 + MIDI"],
    ["拍号", meterInfo(result).signature],
    ["速度", result.bpm ? `${Number(result.bpm).toFixed(2)} BPM` : "—"],
    ["调性", (result.key && result.key.name) || "未检测"],
    ["时长 / 小节", `${mmss(result.duration_s)} · ${result.measure_count || "—"} 小节`],
    ["和弦", chordText],
    ["轨道", trackNames.length ? trackNames.join(" / ") : (midiOnly ? "仅 MIDI 声部" : "—")],
  ];
  for (const [key, value] of items) {
    const box = document.createElement("div");
    box.className = "summary-item";
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = key;
    const v = document.createElement("span");
    v.className = "v";
    v.textContent = value;
    box.append(k, v);
    summary.appendChild(box);
  }
}

function revisioned(url, revision) {
  if (!url) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}rev=${encodeURIComponent(revision)}`;
}

function downloadLabel(name, ext) {
  return `${name} (${ext})`;
}

function renderDownloads(result) {
  const box = $("#downloadLinks");
  box.textContent = "";
  const files = result.files || {};
  const revision = (ui.scoreState && ui.scoreState.revision) ?? result.revision ?? 0;
  const add = (url, label, ext) => {
    if (!url) return;
    const a = document.createElement("a");
    a.href = revisioned(url, revision);
    a.download = "";
    a.textContent = downloadLabel(label, ext);
    box.appendChild(a);
  };
  add(files.gp5, "多轨 Guitar Pro", "gp5");
  add(files.zip, "全部结果", "zip");
  add(files.full_midi, "多轨 MIDI", "mid");
  add(files.reference_midi, "原始转录参考", "mid");
  add(files.chords, "和弦分析", "json");
  add(files.flags, "人工检查标记", "json");
  for (const [stem, url] of Object.entries(files.midi || {})) {
    add(url, `${STEM_LABELS[stem] || stem} MIDI`, "mid");
  }
  for (const [stem, url] of Object.entries(files.stems || {})) {
    add(url, `${STEM_LABELS[stem] || stem} 分轨`, "wav");
  }
  if (!box.children.length) {
    const span = document.createElement("span");
    span.className = "hint";
    span.textContent = "暂无可下载文件。";
    box.appendChild(span);
  }
}

function renderChords(result) {
  const panel = $("#chordPanel");
  const timeline = $("#chordTimeline");
  const meta = $("#chordMeta");
  const progression = $("#chordProgression");
  timeline.textContent = "";
  progression.textContent = "";
  meta.textContent = "";
  const segments = result.chords || [];
  const summary = result.chord_summary || {};
  if (result.chord_detection === false) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const unique = summary.unique_chords || [];
  if (!segments.length) {
    meta.textContent = "未识别出和弦";
    progression.textContent = "没有可显示的和弦进行。可查看处理警告，或关闭和弦分析后重试。";
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "常见原因：所选声部没有和声内容、音频过短，或分离后的 other/bass/vocals 太安静。仅鼓轨不会产生和弦。";
    timeline.appendChild(empty);
    return;
  }
  meta.textContent = `${segments.length} 段 · ${unique.length} 种和弦`;
  const progressionLabels = summary.progression || [];
  progression.textContent = progressionLabels.length
    ? `进行：${progressionLabels.join(" → ")}`
    : "";
  for (const segment of segments) {
    const row = document.createElement("div");
    const confidence = Number(segment.confidence);
    row.className = `chord-segment${confidence < 0.35 ? " low-confidence" : ""}`;
    const label = document.createElement("span");
    label.className = "chord-label";
    label.textContent = segment.label || "—";
    const position = document.createElement("span");
    position.className = "chord-position";
    const meter = meterInfo(result);
    const bar = segment.bar || (Math.floor(Number(segment.start_tick) / meter.barTicks) + 1);
    const beat = segment.beat
      || (Math.floor((Number(segment.start_tick) % meter.barTicks) / meter.pulseTicks) + 1);
    position.textContent = `第 ${bar} 小节 · 第 ${beat} 拍`;
    const time = document.createElement("span");
    time.className = "chord-time";
    time.textContent = `${Number(segment.start_s || 0).toFixed(1)}s`;
    const conf = document.createElement("span");
    conf.className = "chord-confidence";
    conf.textContent = Number.isFinite(confidence) ? `置信度 ${(confidence * 100).toFixed(0)}%` : "置信度 —";
    row.append(label, position, time, conf);
    timeline.appendChild(row);
  }
}

function renderWarnings(result) {
  const panel = $("#warningsPanel");
  const list = $("#warningList");
  list.textContent = "";
  const warnings = result.warnings || [];
  panel.hidden = warnings.length === 0;
  $("#warningCount").textContent = warnings.length ? `(${warnings.length})` : "";
  warnings.forEach((warning) => {
    const li = document.createElement("li");
    li.textContent = warning;
    list.appendChild(li);
  });
}

function renderStemPlayers(result) {
  const box = $("#stemPlayers");
  box.textContent = "";
  const audio = result.audio || {};
  const names = Object.keys(audio);
  if (!names.length) return;
  for (const stem of names) {
    const card = document.createElement("div");
    card.className = "stem-player";
    const label = document.createElement("span");
    label.className = "stem-name";
    label.textContent = `${STEM_LABELS[stem] || stem} · 分轨试听`;
    const player = document.createElement("audio");
    player.controls = true;
    player.preload = "none";
    player.src = audio[stem];
    card.append(label, player);
    box.appendChild(card);
  }
}

function renderMixer() {
  const box = $("#trackMixer");
  box.textContent = "";
  if (!ui.scoreState) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "没有可用的轨道状态。";
    box.appendChild(hint);
    return;
  }
  for (const track of ui.scoreState.tracks || []) {
    const soloed = ui.solo.has(track.id);
    const chip = document.createElement("span");
    chip.className = `track-chip${track.muted ? " muted" : ""}${soloed ? " soloed" : ""}`;
    const name = document.createElement("strong");
    name.textContent = track.name || "Track";
    const count = track.is_percussion ? (track.drum_notes || []).length : (track.notes || []).length;
    const meta = document.createElement("span");
    meta.className = "track-meta";
    meta.textContent = `${count} 个音符`;
    const mute = document.createElement("button");
    mute.type = "button";
    mute.textContent = track.muted ? "取消静音" : "静音";
    mute.addEventListener("click", () => toggleMute(track, mute));
    const solo = document.createElement("button");
    solo.type = "button";
    solo.textContent = soloed ? "取消 solo" : "solo";
    solo.title = "只影响当前页面试听，不写入 tracks.json 或导出文件。";
    solo.addEventListener("click", () => toggleSolo(track));
    chip.append(name, meta, mute, solo);
    if (!track.is_percussion && track.name !== "Vocals" && (track.tuning || []).length) {
      const group = document.createElement("span");
      group.className = "transpose-group";
      for (const semitones of [-12, -1, 1, 12]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = semitones > 0 ? `+${semitones}` : String(semitones);
        button.title = `整轨移调 ${semitones} 半音；重新指法，可撤销一步。`;
        button.addEventListener("click", () => transposeTrack(track, semitones));
        group.appendChild(button);
      }
      chip.appendChild(group);
    }
    box.appendChild(chip);
  }
}

function applyAlphaTabMutes() {
  if (!ui.at || !ui.at.score || !ui.scoreState) return;
  const tracks = ui.at.score.tracks || [];
  const soloing = ui.solo.size > 0;
  for (let index = 0; index < tracks.length; index += 1) {
    const stateTrack = (ui.scoreState.tracks || [])[index];
    if (!stateTrack) continue;
    // Solo is preview-only and overrides the persisted mute while active.
    const muted = soloing ? !ui.solo.has(stateTrack.id) : Boolean(stateTrack.muted);
    try {
      ui.at.changeTrackMute([tracks[index]], muted);
    } catch { /* the player may not be ready yet */ }
  }
}

function initOrLoadAlphaTab(url) {
  if (!window.alphaTab) {
    $("#scoreHint").textContent = "alphaTab 脚本未加载，无法在线预览；仍可下载 GP5/MIDI。";
    $("#playBtn").disabled = true;
    return;
  }
  if (!url) {
    $("#scoreView").innerHTML = "";
    $("#scoreHint").textContent = "GP5 未生成；可在下载区获取 MIDI，或在支持的任务中继续编辑。";
    $("#playBtn").disabled = true;
    if (ui.at) {
      try { ui.at.destroy(); } catch { /* ignore */ }
      ui.at = null;
    }
    return;
  }
  $("#scoreView").hidden = false;
  $("#playBtn").disabled = true;
  $("#scoreHint").textContent = "正在加载乐谱…";
  if (ui.at) {
    try {
      ui.at.load(revisioned(url, Date.now()));
      return;
    } catch {
      try { ui.at.destroy(); } catch { /* ignore */ }
      ui.at = null;
      $("#scoreView").innerHTML = "";
    }
  }
  const api = new window.alphaTab.AlphaTabApi($("#scoreView"), {
    file: revisioned(url, Date.now()),
    core: {
      includeNoteBounds: true,
      fontDirectory: "/static/vendor/alphatab/font/",
    },
    player: {
      enablePlayer: true,
      soundFont: "/static/vendor/alphatab/sonivox.sf2",
      scrollMode: window.alphaTab.ScrollMode.Off,
    },
  });
  ui.at = api;
  api.scoreLoaded.on((score) => {
    for (const track of score.tracks || []) {
      const percussion = (track.staves || []).some((stave) => stave.isPercussion)
        || (track.playbackInfo && track.playbackInfo.primaryChannel === 9);
      const notationOnly = percussion || track.name === "Vocals";
      for (const stave of track.staves || []) {
        stave.showTablature = !notationOnly;
        stave.showStandardNotation = notationOnly;
      }
    }
    try {
      if (api.render) api.render();
    } catch { /* newer alphaTab renders automatically */ }
    applyAlphaTabMutes();
    $("#scoreHint").textContent = `已加载 ${(score.masterBars || []).length} 小节。点击吉他或贝斯音符可改弦/品；人声与鼓不支持弦位编辑。`;
    $("#playPos").textContent = `共 ${(score.masterBars || []).length} 小节`;
    if (api.isReadyForPlayback) $("#playBtn").disabled = false;
  });
  api.playerReady.on(() => {
    $("#playBtn").disabled = false;
    applyAlphaTabMutes();
  });
  api.playerStateChanged.on((event) => {
    const playing = window.alphaTab.synth
      && event.state === window.alphaTab.synth.PlayerState.Playing;
    $("#playBtn").textContent = playing ? "⏸ 暂停" : "▶ 播放";
  });
  api.playedBeatChanged.on((beat) => {
    if (!beat) return;
    const bar = beat.voice && beat.voice.bar ? beat.voice.bar.index + 1 : "?";
    const total = api.score && api.score.masterBars ? api.score.masterBars.length : "?";
    $("#playPos").textContent = `第 ${bar} / ${total} 小节`;
  });
  api.noteMouseDown.on((note) => selectNote(note));
  api.error.on((error) => {
    const message = error && error.message ? error.message : "未知错误";
    $("#scoreHint").textContent = `alphaTab 加载失败：${message}`;
    $("#playBtn").disabled = true;
  });
  $("#playBtn").onclick = () => {
    if (ui.at && ui.at.isReadyForPlayback) ui.at.playPause();
  };
}

function loadScore(url) {
  const keepY = window.scrollY;
  if (ui.at && url) {
    const restore = () => {
      window.scrollTo({ top: keepY });
      try { ui.at.renderFinished.off(restore); } catch { /* ignore */ }
    };
    try { ui.at.renderFinished.on(restore); } catch { /* ignore */ }
    initOrLoadAlphaTab(url);
    window.setTimeout(restore, 600);
  } else {
    initOrLoadAlphaTab(url);
  }
}

function trackForNote(note, alphaTrack) {
  if (!ui.scoreState) return null;
  const tracks = ui.scoreState.tracks || [];
  if (alphaTrack && Number.isInteger(alphaTrack.index) && tracks[alphaTrack.index]) {
    return tracks[alphaTrack.index];
  }
  return tracks.find((track) => track.name === (alphaTrack && alphaTrack.name)) || null;
}

function selectNote(note) {
  if (!ui.scoreState) {
    showToast("编辑状态尚未加载，请稍候。", true);
    return;
  }
  const staff = note.beat.voice && note.beat.voice.bar ? note.beat.voice.bar.staff : null;
  const alphaTrack = staff ? staff.track : null;
  const track = trackForNote(note, alphaTrack);
  if (!track) {
    showToast("无法把该音符映射到编辑状态。", true);
    return;
  }
  if (track.is_percussion || track.name === "Vocals") {
    $("#scoreHint").textContent = track.is_percussion
      ? "鼓轨的每个音符是鼓件，不存在“换弦”概念，本编辑器不修改鼓谱。"
      : "人声轨是旋律记谱，不映射到真实吉他弦位；请下载后在制谱软件中编辑。";
    return;
  }
  const tuning = track.tuning || [];
  if (!tuning.length) {
    showToast("该轨没有可用的调弦信息。", true);
    return;
  }
  const start = note.beat.absolutePlaybackStart ?? note.beat.playbackStart ?? 0;
  const pitch = note.realValue;
  const stringNumber = tuning.length - note.string + 1;
  let target = (track.notes || []).find((item) =>
    Number(item.start_tick) === Number(start)
    && Number(item.pitch) === Number(pitch)
    && Number(item.string) === Number(stringNumber));
  if (!target) {
    const candidates = (track.notes || []).filter((item) =>
      Number(item.start_tick) === Number(start) && Number(item.pitch) === Number(pitch));
    if (candidates.length === 1) target = candidates[0];
  }
  if (!target) {
    $("#scoreHint").textContent = "这个位置没有可编辑的起音（可能是延音线、装饰音，或谱面量化后的非起音位置）。";
    closeNoteEditor();
    return;
  }
  ui.selected = { track, note: target };
  renderNoteEditor();
  $("#noteEditor").hidden = false;
  $("#noteEditor").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderNoteEditor() {
  if (!ui.selected) return;
  const { track, note } = ui.selected;
  const tuning = track.tuning || [];
  const n = tuning.length;
  $("#noteTitle").textContent = `${track.name} · ${noteName(note.pitch)} · 第 ${barNumber(note.start_tick)} 小节`;
  const alts = $("#noteAlts");
  alts.textContent = "";
  // A pitch may be playable on several strings, but a string can only sound
  // once at a given onset. Hide positions already used by a chord tone of the
  // same attack so the browser cannot submit a guaranteed server-side 422.
  const occupied = new Map();
  for (const item of track.notes || []) {
    if (
      item.id !== note.id
      && Number(item.start_tick) === Number(note.start_tick)
    ) {
      occupied.set(Number(item.string), item);
    }
  }
  let alternatives = 0;
  for (let string = 1; string <= n; string += 1) {
    const fret = Number(note.pitch) - Number(tuning[string - 1]);
    if (fret < 0 || fret > Number(ui.scoreState.max_fret || 24)) continue;
    const button = document.createElement("button");
    button.type = "button";
    const current = string === Number(note.string);
    const blocked = !current && occupied.has(string);
    const openName = noteName(tuning[string - 1]);
    button.textContent = `弦 ${string} · 第 ${fret} 品（空弦 ${openName}）${current ? " ← 当前" : ""}${blocked ? " · 本拍已占用" : ""}`;
    if (current) {
      button.classList.add("current");
      button.disabled = true;
    } else if (blocked) {
      button.disabled = true;
    } else {
      alternatives += 1;
      button.addEventListener("click", () => patchPosition({ string }, button));
    }
    alts.appendChild(button);
  }
  if (!alternatives) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = occupied.size
      ? "这个音高在当前调弦下没有空余的其他弦位；可尝试修改品位，或删除同一拍上的另一个音。"
      : "当前调弦下，这个音高没有其他可弹位置。";
    alts.appendChild(hint);
  }
  const select = $("#manualString");
  select.textContent = "";
  for (let string = 1; string <= n; string += 1) {
    const option = document.createElement("option");
    option.value = String(string);
    const blocked = string !== Number(note.string) && occupied.has(string);
    option.textContent = `弦 ${string} — ${noteName(tuning[string - 1])}${blocked ? "（本拍已占用）" : ""}`;
    option.disabled = blocked;
    if (string === Number(note.string)) option.selected = true;
    select.appendChild(option);
  }
  $("#manualFret").value = String(note.fret);
  const flag = (ui.flags || []).find(
    (item) => item.track_id === track.id && item.note_id === note.id
  );
  $("#flagNote").checked = Boolean(flag);
  $("#flagReason").value = flag ? flag.reason || "" : "";
  $("#noteStatus").textContent = "";
  $("#noteStatus").classList.remove("error");
}

function closeNoteEditor() {
  ui.selected = null;
  $("#noteEditor").hidden = true;
  $("#noteStatus").textContent = "";
}

async function patchPosition(payload, sourceButton) {
  if (!ui.selected || !ui.scoreState || !ui.jobId) return;
  const { track, note } = ui.selected;
  const status = $("#noteStatus");
  status.textContent = "正在重新生成 GP5…";
  status.classList.remove("error");
  const buttons = document.querySelectorAll("#noteAlts button, #applyPositionBtn, #deleteNoteBtn");
  buttons.forEach((button) => { button.disabled = true; });
  try {
    const response = await fetch(
      `/api/jobs/${encodeURIComponent(ui.jobId)}/tracks/${encodeURIComponent(track.id)}/notes/${encodeURIComponent(note.id)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ revision: Number(ui.scoreState.revision), ...payload }),
      },
    );
    const data = await readJson(response);
    handleEditResponse(data, `已把 ${noteName(note.pitch)} 移到新位置。`, true);
  } catch (error) {
    status.textContent = `修改失败：${error.message}`;
    status.classList.add("error");
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
    if (sourceButton) sourceButton.disabled = false;
  }
}

async function deleteSelectedNote() {
  if (!ui.selected || !ui.scoreState || !ui.jobId) return;
  const { track, note } = ui.selected;
  if (!window.confirm(`删除 ${track.name} 中 ${noteName(note.pitch)} 这个音符？可用“撤销上一处修改”恢复一步。`)) return;
  const status = $("#noteStatus");
  status.textContent = "正在删除并重新生成 GP5…";
  status.classList.remove("error");
  $("#deleteNoteBtn").disabled = true;
  try {
    const response = await fetch(
      `/api/jobs/${encodeURIComponent(ui.jobId)}/tracks/${encodeURIComponent(track.id)}/notes/${encodeURIComponent(note.id)}?revision=${encodeURIComponent(ui.scoreState.revision)}`,
      { method: "DELETE" },
    );
    const data = await readJson(response);
    handleEditResponse(data, "音符已删除，GP5/MIDI 已重新生成。", true);
  } catch (error) {
    status.textContent = `删除失败：${error.message}`;
    status.classList.add("error");
  } finally {
    $("#deleteNoteBtn").disabled = false;
  }
}

async function toggleMute(track, button) {
  if (!ui.scoreState || !ui.jobId) return;
  const nextMuted = !track.muted;
  button.disabled = true;
  button.textContent = "更新中…";
  try {
    const response = await fetch(
      `/api/jobs/${encodeURIComponent(ui.jobId)}/tracks/${encodeURIComponent(track.id)}/mute`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ revision: Number(ui.scoreState.revision), muted: nextMuted }),
      },
    );
    const data = await readJson(response);
    handleEditResponse(data, nextMuted ? `${track.name} 已静音。` : `${track.name} 已恢复。`, false);
    applyAlphaTabMutes();
  } catch (error) {
    showToast(`静音操作失败：${error.message}`, true);
    renderMixer();
  }
}

async function toggleSolo(track) {
  if (!ui.scoreState) return;
  if (ui.solo.has(track.id)) ui.solo.delete(track.id);
  else ui.solo.add(track.id);
  renderMixer();
  applyAlphaTabMutes();
  showToast(ui.solo.size
    ? `试听 solo：${[...ui.solo].map((id) => {
        const item = (ui.scoreState.tracks || []).find((candidate) => candidate.id === id);
        return item ? item.name : id;
      }).join(" / ")}（不改变导出文件）`
    : "已退出 solo，恢复各轨静音状态。");
}

async function transposeTrack(track, semitones) {
  if (!ui.scoreState || !ui.jobId) return;
  const direction = semitones > 0 ? `+${semitones}` : String(semitones);
  if (!window.confirm(
    `把“${track.name}”整轨移调 ${direction} 半音？会重新搜索指法；超出最大品位的音将被删除，可用撤销恢复一步。`
  )) return;
  $("#scoreHint").textContent = `正在把 ${track.name} 移调 ${direction} 半音并重新生成 GP5…`;
  try {
    const response = await fetch(
      `/api/jobs/${encodeURIComponent(ui.jobId)}/tracks/${encodeURIComponent(track.id)}/transpose`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ revision: Number(ui.scoreState.revision), semitones }),
      },
    );
    const data = await readJson(response);
    const dropped = Number(data.dropped || 0);
    const message = dropped
      ? `${track.name} 已移调 ${direction} 半音；${dropped} 个音超出可弹范围被删除，可撤销一步。`
      : `${track.name} 已移调 ${direction} 半音。`;
    handleEditResponse(data, message, true);
  } catch (error) {
    showToast(`移调失败：${error.message}`, true);
    $("#scoreHint").textContent = "移调未完成，谱面保持原样。";
    renderMixer();
  }
}

async function undoLastEdit() {
  if (!ui.scoreState || !ui.jobId) return;
  const button = $("#undoBtn");
  button.disabled = true;
  $("#scoreHint").textContent = "正在撤销最近一步修改并重新生成 GP5…";
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(ui.jobId)}/undo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: Number(ui.scoreState.revision) }),
    });
    const data = await readJson(response);
    handleEditResponse(data, "已撤销最近一步修改。", true);
  } catch (error) {
    showToast(`撤销失败：${error.message}`, true);
    $("#scoreHint").textContent = "撤销未完成，谱面保持原样。";
    renderUndo(true);
  }
}

function renderUndo(canUndo) {
  const button = $("#undoBtn");
  if (!button) return;
  button.disabled = !canUndo;
  button.title = canUndo
    ? "恢复最近一次修改前的乐谱；只能撤销一步，撤销后这一步不能用“重做”恢复。"
    : "没有可撤销的修改。";
}

function renderFlags() {
  const panel = $("#flagPanel");
  const list = $("#flagList");
  if (!panel || !list) return;
  list.textContent = "";
  const flags = ui.flags || [];
  $("#flagMeta").textContent = flags.length ? `${flags.length} 个标记` : "";
  panel.hidden = flags.length === 0;
  for (const flag of flags) {
    const li = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = `${flag.track_name || flag.track_id} · ${noteName(flag.pitch)}`;
    const position = document.createElement("span");
    position.textContent = `第 ${flag.bar} 小节`;
    const reason = document.createElement("span");
    reason.className = "flag-reason";
    reason.textContent = flag.reason || "";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "取消标记";
    button.addEventListener("click", () => {
      sendFlag(flag.track_id, flag.note_id, false, "", button);
    });
    li.append(name, position, reason, button);
    list.appendChild(li);
  }
}

async function sendFlag(trackId, noteId, flagged, reason, sourceButton) {
  if (!ui.jobId || !ui.scoreState) return;
  if (sourceButton) sourceButton.disabled = true;
  try {
    const response = await fetch(
      `/api/jobs/${encodeURIComponent(ui.jobId)}/tracks/${encodeURIComponent(trackId)}/notes/${encodeURIComponent(noteId)}/flag`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ flagged, reason }),
      },
    );
    const data = await readJson(response);
    ui.flags = data.flags || [];
    if (data.job && data.job.result) {
      ui.result = data.job.result;
      renderDownloads(ui.result);
    }
    renderFlags();
    const status = $("#noteStatus");
    if (ui.selected) renderNoteEditor();
    if (status) {
      status.textContent = flagged ? "已保存人工检查标记。" : "已取消该音符的标记。";
      status.classList.remove("error");
    }
    if (data.warning) showToast(data.warning, true);
  } catch (error) {
    showToast(`标记保存失败：${error.message}`, true);
    if (sourceButton) sourceButton.disabled = false;
  }
}

function saveSelectedFlag() {
  if (!ui.selected) return;
  const { track, note } = ui.selected;
  sendFlag(
    track.id,
    note.id,
    $("#flagNote").checked,
    $("#flagReason").value,
    $("#applyFlagBtn"),
  );
}

function handleEditResponse(data, message, reloadScore) {
  if (data.state) ui.scoreState = data.state;
  if (Array.isArray(data.flags)) ui.flags = data.flags;
  if (typeof data.can_undo === "boolean") renderUndo(data.can_undo);
  if (data.job) {
    ui.job = data.job;
    ui.result = data.job.result;
    renderSummary(ui.result);
    renderChords(ui.result);
    renderWarnings(ui.result);
    renderStemPlayers(ui.result);
  }
  renderMixer();
  renderFlags();
  renderDownloads(ui.result);
  closeNoteEditor();
  if (reloadScore) {
    const url = (ui.result && ui.result.score_url) || data.gp5_url;
    $("#scoreHint").textContent = "已更新，正在重新加载乐谱…";
    loadScore(url);
  }
  showToast(message);
}

function setupFileInput() {
  const input = $("#fileInput");
  const drop = $(".file-drop");
  const hint = $("#fileHint");
  const update = () => {
    const file = input.files[0];
    if (!file) {
      hint.textContent = "点击选择，或把文件拖到这里。音频请使用包含全部乐器的成品混音；也可以直接上传 .mid/.midi 跳过分离与转录。";
      drop.classList.remove("has-file");
      return;
    }
    hint.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    drop.classList.add("has-file");
    setFormStatus("");
  };
  input.addEventListener("change", update);
  ["dragenter", "dragover"].forEach((eventName) => {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      drop.classList.remove("dragover");
    });
  });
  drop.addEventListener("drop", (event) => {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (!files || !files.length) return;
    input.files = files;
    update();
  });
}

function bindEvents() {
  setupFileInput();
  $("#jobForm").addEventListener("submit", startJob);
  $("#exportFormatSelect").addEventListener("change", updateOutputFormatUi);
  updateOutputFormatUi();
  $("#closeNoteBtn").addEventListener("click", closeNoteEditor);
  $("#applyPositionBtn").addEventListener("click", () => {
    const string = Number.parseInt($("#manualString").value, 10);
    const fret = Number.parseInt($("#manualFret").value, 10);
    if (!Number.isFinite(string) || !Number.isFinite(fret)) {
      $("#noteStatus").textContent = "请输入有效的弦号和品位。";
      $("#noteStatus").classList.add("error");
      return;
    }
    patchPosition({ string, fret });
  });
  $("#deleteNoteBtn").addEventListener("click", deleteSelectedNote);
  $("#applyFlagBtn").addEventListener("click", saveSelectedFlag);
  $("#undoBtn").addEventListener("click", undoLastEdit);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("#noteEditor").hidden) closeNoteEditor();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadLimits();
  restoreJobFromUrl();
});
