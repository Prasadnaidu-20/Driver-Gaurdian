// DriveGuardian dashboard — stream frames to /ws/stream; show performance (Phase 1)
// and face landmarks / features with an EAR-MAR chart (Phase 2).
//
// Wire format (client → server), little-endian:
//   float64 timestamp_ms | uint32 frame_id | JPEG bytes
// Flow control: exactly one frame in flight. The next frame is captured only after the
// result for the previous one has arrived, so latency never builds up.
"use strict";

const HEADER_BYTES = 12;

const el = {
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  placeholder: document.getElementById("video-placeholder"),
  btnWebcam: document.getElementById("btn-webcam"),
  btnStop: document.getElementById("btn-stop"),
  fileInput: document.getElementById("file-input"),
  status: document.getElementById("status"),
  fps: document.getElementById("m-fps"),
  rtt: document.getElementById("m-rtt"),
  backend: document.getElementById("m-backend"),
  backendPct: document.getElementById("m-backend-pct"),
  frames: document.getElementById("m-frames"),
  lastStatus: document.getElementById("m-status"),
  showLandmarks: document.getElementById("toggle-landmarks"),
  faceStatus: document.getElementById("face-status"),
  earLR: document.getElementById("f-ear-lr"),
  ear: document.getElementById("f-ear"),
  mar: document.getElementById("f-mar"),
  pose: document.getElementById("f-pose"),
  gaze: document.getElementById("f-gaze"),
  blink: document.getElementById("f-blink"),
  jaw: document.getElementById("f-jaw"),
  chartCanvas: document.getElementById("chart-ear-mar"),
};

const OVERLAY_COLORS = {
  box: "#22c55e",
  right_eye: "#38bdf8",
  left_eye: "#38bdf8",
  mouth: "#f472b6",
  right_iris: "#facc15",
  left_iris: "#facc15",
};

const state = {
  config: null,          // stream settings from /api/config
  capture: null,         // offscreen canvas at the configured frame size
  ws: null,
  source: null,          // "webcam" | "file" | null
  mediaStream: null,
  objectUrl: null,
  streaming: false,
  inFlight: false,       // a frame has been captured/sent and its result not yet received
  scheduled: false,      // a requestAnimationFrame for the next send is pending
  frameId: 0,
  sentAt: 0,             // performance.now() when the in-flight frame was sent
  rtts: [],              // rolling client round-trip latencies (ms)
  framesDone: 0,
  chart: null,           // Chart.js instance (null if the CDN failed to load)
  chartLastTs: null,     // last frame timestamp added to the chart (ms)
  lastFace: null,        // last face block, redrawn when the landmark toggle changes
};

// ---------- helpers ----------

function setStatus(text) {
  el.status.textContent = text;
}

function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function fmtMs(v) {
  return `${v.toFixed(1)} ms`;
}

function setButtons() {
  const active = state.source !== null;
  el.btnStop.disabled = !active;
  el.btnWebcam.disabled = state.config === null;
}

function resetMetrics() {
  state.rtts = [];
  state.framesDone = 0;
  for (const node of [el.fps, el.rtt, el.backend, el.backendPct, el.lastStatus]) node.textContent = "–";
  el.frames.textContent = "0";
}

// Draw the video into the w×h capture canvas keeping its aspect ratio (black bars).
// The <video> and the overlay both use object-fit: contain in a 4:3 box, so the captured
// frame, the displayed video and the overlay share the same geometry for any file.
function drawLetterboxed(ctx, video, w, h) {
  const vw = video.videoWidth || w;
  const vh = video.videoHeight || h;
  const scale = Math.min(w / vw, h / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(video, (w - dw) / 2, (h - dh) / 2, dw, dh);
}

// ---------- face: overlay, Driver State card, chart ----------

function clearOverlay() {
  el.overlay.getContext("2d").clearRect(0, 0, el.overlay.width, el.overlay.height);
}

function drawOverlay(face) {
  clearOverlay();
  if (!face || !face.detected) return;
  const ctx = el.overlay.getContext("2d");
  const W = el.overlay.width;
  const H = el.overlay.height;
  if (face.bbox) {
    const [x, y, bw, bh] = face.bbox;
    ctx.strokeStyle = OVERLAY_COLORS.box;
    ctx.lineWidth = 2;
    ctx.strokeRect(x * W, y * H, bw * W, bh * H);
  }
  if (!el.showLandmarks.checked || !face.landmarks) return;
  for (const [group, points] of Object.entries(face.landmarks)) {
    ctx.fillStyle = OVERLAY_COLORS[group] || "#fff";
    const r = group.endsWith("iris") ? 1.5 : 1.8;
    for (const [px, py] of points) {
      ctx.beginPath();
      ctx.arc(px * W, py * H, r, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

const FEATURE_NODES = () => [el.earLR, el.ear, el.mar, el.pose, el.gaze, el.blink, el.jaw];

function setFaceStatus(text, cls) {
  el.faceStatus.textContent = text;
  el.faceStatus.className = `badge ${cls}`;
}

function resetDriverState() {
  setFaceStatus("–", "");
  for (const node of FEATURE_NODES()) node.textContent = "–";
}

function renderDriverState(result) {
  const face = result.face;
  if (!face || !face.available) {
    setFaceStatus("model unavailable", "bad");
    return;
  }
  if (!face.detected) {
    setFaceStatus("no face", "warn");
    for (const node of FEATURE_NODES()) node.textContent = "–";
    return;
  }
  const f = face.features;
  const n = (v, d = 3) => v.toFixed(d);
  setFaceStatus("face detected", "ok");
  el.earLR.textContent = `${n(f.ear_left)} / ${n(f.ear_right)}`;
  el.ear.textContent = n(f.ear_mean);
  el.mar.textContent = n(f.mar);
  el.pose.textContent = `${n(f.yaw, 1)} / ${n(f.pitch, 1)} / ${n(f.roll, 1)}`;
  el.gaze.textContent = `${n(f.gaze_h, 2)} / ${n(f.gaze_v, 2)}`;
  el.blink.textContent = `${n(f.eye_blink_left, 2)} / ${n(f.eye_blink_right, 2)}`;
  el.jaw.textContent = n(f.jaw_open, 2);
}

function initChart() {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not loaded (CDN unreachable?) — EAR/MAR chart disabled");
    return;
  }
  const dataset = (label, color) => ({
    label, data: [], borderColor: color, backgroundColor: color,
    borderWidth: 1.5, pointRadius: 0, spanGaps: false,
  });
  const axisColor = "#8b96a1";
  const gridColor = "#2a333d";
  state.chart = new Chart(el.chartCanvas, {
    type: "line",
    data: { datasets: [dataset("EAR mean", "#38bdf8"), dataset("MAR", "#f472b6")] },
    options: {
      animation: false,
      maintainAspectRatio: false,
      parsing: false,
      scales: {
        x: { type: "linear", ticks: { color: axisColor, stepSize: 1, maxTicksLimit: 6, callback: (v) => `${Number(v).toFixed(0)} s` },
             grid: { color: gridColor } },
        y: { min: 0, suggestedMax: 0.6, ticks: { color: axisColor }, grid: { color: gridColor } },
      },
      plugins: { legend: { labels: { color: "#e6e9ec", boxWidth: 12 } }, tooltip: { enabled: false } },
    },
  });
}

function resetChart() {
  state.chartLastTs = null;
  if (!state.chart) return;
  for (const ds of state.chart.data.datasets) ds.data = [];
  state.chart.update("none");
}

function updateChart(result) {
  if (!state.chart) return;
  const ts = result.timestamp_ms;
  if (state.chartLastTs !== null && ts < state.chartLastTs) resetChart();  // video seek / replay
  state.chartLastTs = ts;
  const x = ts / 1000;
  const f = result.face && result.face.detected ? result.face.features : null;
  const [earDs, marDs] = state.chart.data.datasets;
  earDs.data.push({ x, y: f ? f.ear_mean : null });  // null leaves a gap while there is no face
  marDs.data.push({ x, y: f ? f.mar : null });
  const minX = x - state.config.chart_window_s;
  for (const ds of state.chart.data.datasets) {
    while (ds.data.length && ds.data[0].x < minX) ds.data.shift();
  }
  state.chart.options.scales.x.min = minX;
  state.chart.options.scales.x.max = x;
  state.chart.update("none");
}

// ---------- WebSocket ----------

function openSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  state.inFlight = false;

  ws.onopen = () => {
    if (state.ws !== ws) return;
    setStatus(`Streaming (${state.source})`);
    scheduleNext();
  };
  ws.onmessage = (event) => {
    if (state.ws !== ws) return;
    onResult(JSON.parse(event.data));
  };
  ws.onerror = () => {
    if (state.ws !== ws) return;
    setStatus("Connection error");
  };
  ws.onclose = () => {
    if (state.ws !== ws) return;  // ignore close events from sockets we replaced
    state.ws = null;
    state.streaming = false;
    state.inFlight = false;
    if (state.source !== null) setStatus("Disconnected from server");
  };
}

function closeSocket() {
  const ws = state.ws;
  state.ws = null;
  if (ws && ws.readyState <= WebSocket.OPEN) ws.close(1000, "client stop");
}

// ---------- frame loop ----------

function readyToSend() {
  const v = el.video;
  if (!state.streaming || !state.ws || state.ws.readyState !== WebSocket.OPEN) return false;
  if (v.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return false;
  if (state.source === "file" && (v.paused || v.ended || v.seeking)) return false;
  return true;
}

function scheduleNext() {
  if (!state.streaming || state.inFlight || state.scheduled) return;
  state.scheduled = true;
  requestAnimationFrame(() => {
    state.scheduled = false;
    sendFrame();
  });
}

function sendFrame() {
  if (state.inFlight) return;
  if (!readyToSend()) {
    // Webcam still warming up: keep polling. A paused/seeking file resumes via its events.
    if (state.streaming && state.source === "webcam") scheduleNext();
    return;
  }
  state.inFlight = true;

  const timestampMs = state.source === "file" ? el.video.currentTime * 1000 : performance.now();
  const frameId = state.frameId;
  state.frameId = (state.frameId + 1) >>> 0;  // uint32 wrap

  const { frame_width: w, frame_height: h, jpeg_quality: q } = state.config;
  drawLetterboxed(state.capture.getContext("2d"), el.video, w, h);
  state.capture.toBlob(async (blob) => {
    const ws = state.ws;
    if (!blob || !ws || ws.readyState !== WebSocket.OPEN || !state.streaming) {
      state.inFlight = false;
      scheduleNext();
      return;
    }
    const jpeg = new Uint8Array(await blob.arrayBuffer());
    const buf = new ArrayBuffer(HEADER_BYTES + jpeg.byteLength);
    const view = new DataView(buf);
    view.setFloat64(0, timestampMs, true);
    view.setUint32(8, frameId, true);
    new Uint8Array(buf, HEADER_BYTES).set(jpeg);
    if (state.ws !== ws || ws.readyState !== WebSocket.OPEN) {
      state.inFlight = false;
      return;
    }
    state.sentAt = performance.now();
    ws.send(buf);
  }, "image/jpeg", q);
}

function onResult(result) {
  const rtt = performance.now() - state.sentAt;
  state.inFlight = false;
  state.framesDone += 1;

  state.rtts.push(rtt);
  if (state.rtts.length > state.config.perf_window_frames) state.rtts.shift();

  const perf = result.perf;
  el.fps.textContent = perf.fps.toFixed(1);
  el.rtt.textContent = `${fmtMs(percentile(state.rtts, 50))} / ${fmtMs(percentile(state.rtts, 95))}`;
  el.backend.textContent = fmtMs(perf.backend_ms);
  el.backendPct.textContent = `${fmtMs(perf.backend_p50_ms)} / ${fmtMs(perf.backend_p95_ms)}`;
  el.frames.textContent = String(state.framesDone);
  el.lastStatus.textContent = result.status;

  if (result.status !== "bad_message" && result.status !== "decode_error") {
    state.lastFace = result.face;
    drawOverlay(result.face);
    renderDriverState(result);
    updateChart(result);
  }

  scheduleNext();
}

// ---------- sources ----------

function stop(message = "Stopped") {
  state.streaming = false;
  closeSocket();
  if (state.mediaStream) {
    for (const track of state.mediaStream.getTracks()) track.stop();
    state.mediaStream = null;
  }
  el.video.pause();
  el.video.srcObject = null;
  el.video.removeAttribute("src");
  el.video.load();
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
  state.source = null;
  state.inFlight = false;
  state.lastFace = null;
  clearOverlay();
  el.placeholder.hidden = false;
  setStatus(message);
  setButtons();
}

function beginStreaming(source) {
  state.source = source;
  state.streaming = true;
  state.frameId = 0;
  resetMetrics();
  resetDriverState();
  resetChart();
  el.placeholder.hidden = true;
  setButtons();
  openSocket();
}

async function startWebcam() {
  stop("Starting webcam…");
  try {
    const { frame_width: w, frame_height: h } = state.config;
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: w }, height: { ideal: h } },
      audio: false,
    });
  } catch (err) {
    stop(`Webcam unavailable: ${err.name}`);
    return;
  }
  el.video.srcObject = state.mediaStream;
  await el.video.play();
  beginStreaming("webcam");
}

async function startFile(file) {
  stop("Loading video…");
  state.objectUrl = URL.createObjectURL(file);
  el.video.src = state.objectUrl;
  try {
    await el.video.play();
  } catch (err) {
    stop(`Cannot play file: ${err.name}`);
    return;
  }
  beginStreaming("file");
}

// ---------- wiring ----------

el.btnWebcam.addEventListener("click", startWebcam);
el.btnStop.addEventListener("click", () => stop());
el.fileInput.addEventListener("change", () => {
  const file = el.fileInput.files[0];
  el.fileInput.value = "";  // allow re-loading the same file
  if (file && state.config) startFile(file);
});

// File playback: resume sending after pause/seek; finish cleanly at the end.
el.video.addEventListener("play", scheduleNext);
el.video.addEventListener("seeked", scheduleNext);
el.video.addEventListener("ended", () => {
  if (state.source !== "file") return;
  state.streaming = false;
  closeSocket();
  setStatus("Video ended");
});

el.showLandmarks.addEventListener("change", () => drawOverlay(state.lastFace));

window.addEventListener("pagehide", () => closeSocket());

async function init() {
  setButtons();
  try {
    const resp = await fetch("/api/config");
    state.config = await resp.json();
  } catch (err) {
    setStatus("Could not load /api/config");
    return;
  }
  const { frame_width: w, frame_height: h } = state.config;
  state.capture = document.createElement("canvas");
  state.capture.width = w;
  state.capture.height = h;
  el.overlay.width = w;
  el.overlay.height = h;
  initChart();
  setButtons();
  setStatus("Idle");
}

init();
