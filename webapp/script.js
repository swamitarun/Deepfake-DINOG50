/**
 * DeepShield AI — script.js
 * Handles video upload, API call, results rendering, and per-frame chart.
 *
 * ⚙️  HF Space URL — update if your space name changes.
 */
const API_URL = "https://mrtsp-deepfake-dinov2-api.hf.space";

// ── Keep-Alive: Ping HF Space every 9 min to prevent free-tier sleep ──
// Free HF Spaces sleep after 15min idle — this ping prevents that!
(function startKeepAlive() {
  const ping = () => fetch(`${API_URL}/`, { method: 'GET', mode: 'no-cors' }).catch(() => {});
  ping(); // Wake Space immediately on page load
  setInterval(ping, 9 * 60 * 1000); // Ping every 9 minutes
})();

const MAX_FILE_MB = 30;
const MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024;

let currentFile = null;
let lastResult = null;

// ─────────────────────────────────────────────────────
// Section switcher
// ─────────────────────────────────────────────────────
function showSection(id) {
  const sections = ["upload-section", "loading-section", "results-section", "error-section"];
  sections.forEach(s => {
    document.getElementById(s).classList.toggle("hidden", s !== id);
  });
}

// ─────────────────────────────────────────────────────
// Drag & Drop handlers
// ─────────────────────────────────────────────────────
function onDragOver(e) {
  e.preventDefault();
  document.getElementById("drop-zone").classList.add("dragging");
}

function onDragLeave() {
  document.getElementById("drop-zone").classList.remove("dragging");
}

function onDrop(e) {
  e.preventDefault();
  document.getElementById("drop-zone").classList.remove("dragging");
  const file = e.dataTransfer?.files?.[0];
  if (file) processFile(file);
}

function onFileSelected(e) {
  const file = e.target.files?.[0];
  if (file) processFile(file);
}

// ─────────────────────────────────────────────────────
// File Processing
// ─────────────────────────────────────────────────────
function processFile(file) {
  const allowedExt = [".mp4", ".mov", ".avi", ".mkv"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowedExt.includes(ext)) {
    showError(`❌ Unsupported file type: "${ext}". Please upload MP4, MOV, AVI, or MKV.`);
    return;
  }

  if (file.size > MAX_FILE_BYTES) {
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    showError(`❌ File too large (${sizeMB} MB). Maximum allowed size is ${MAX_FILE_MB} MB.`);
    return;
  }

  currentFile = file;

  document.getElementById("file-name").textContent = file.name;
  document.getElementById("file-size").textContent = formatBytes(file.size);

  const url = URL.createObjectURL(file);
  const video = document.getElementById("video-preview");
  video.src = url;

  document.getElementById("drop-zone").classList.add("hidden");
  document.getElementById("file-preview").classList.remove("hidden");
}

function resetUpload() {
  currentFile = null;
  lastResult = null;

  const fileInput = document.getElementById("file-input");
  fileInput.value = "";

  const video = document.getElementById("video-preview");
  if (video.src) URL.revokeObjectURL(video.src);
  video.src = "";

  document.getElementById("ring-fill").style.strokeDashoffset = "314";
  document.getElementById("frame-chart").innerHTML = "";

  document.getElementById("drop-zone").classList.remove("hidden");
  document.getElementById("file-preview").classList.add("hidden");
  showSection("upload-section");

  const btn = document.getElementById("analyze-btn");
  if (btn) { btn.disabled = false; btn.innerHTML = '<span class="btn-icon">🔍</span><span>Analyze for Deepfakes</span>'; }
}

// ─────────────────────────────────────────────────────
// Main: Analyze Video
// ─────────────────────────────────────────────────────
async function analyzeVideo() {
  if (!currentFile) return;

  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span><span>Uploading...</span>';

  showSection("loading-section");
  animateLoadingSteps();

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ detail: "Unknown server error." }));
      throw new Error(errData.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    lastResult = data;
    renderResults(data);
    showSection("results-section");

  } catch (err) {
    console.error("Analysis failed:", err);
    let msg = err.message || "Could not connect to the analysis server.";
    if (msg.includes("fetch") || msg.includes("NetworkError") || msg.includes("Failed to fetch")) {
      msg = `⚠️ Cannot reach the DINO-G50 server.\n\nThe Hugging Face Space might be waking up (takes ~60 sec on first request). Please wait a moment and try again.\n\nServer: ${API_URL}`;
    }
    showError(msg);
  }
}

// ─────────────────────────────────────────────────────
// Render Results
// ─────────────────────────────────────────────────────
function renderResults(data) {
  const isFake = data.verdict === "FAKE";
  const fakePct = data.fake_probability;
  const realPct = data.real_probability;

  const card = document.getElementById("verdict-card");
  card.classList.remove("is-fake", "is-real");
  card.classList.add(isFake ? "is-fake" : "is-real");

  const ring = document.getElementById("ring-fill");
  const circumference = 314;
  const dashOffset = circumference - (fakePct / 100) * circumference;
  ring.style.stroke = isFake ? "#ef4444" : "#22c55e";
  setTimeout(() => { ring.style.strokeDashoffset = dashOffset; }, 100);

  document.getElementById("verdict-pct").textContent = `${fakePct}%`;
  const lbl = document.getElementById("verdict-label");
  lbl.textContent = data.verdict;
  lbl.style.color = isFake ? "#f87171" : "#4ade80";

  const badge = document.getElementById("verdict-badge");
  badge.textContent = isFake ? "⚠️  FAKE DETECTED" : "✅ REAL VIDEO";
  badge.className = "verdict-badge " + (isFake ? "fake" : "real");

  document.getElementById("stat-fake").textContent = `${fakePct}%`;
  document.getElementById("stat-real").textContent = `${realPct}%`;
  document.getElementById("stat-frames").textContent = `${data.frame_count} frames`;
  document.getElementById("stat-size").textContent = `${data.file_size_mb} MB`;

  renderFrameChart(data.per_frame_scores || []);
}

function renderFrameChart(scores) {
  const container = document.getElementById("frame-chart");
  container.innerHTML = "";

  if (!scores.length) {
    container.innerHTML = '<p style="color:var(--text-sub);font-size:13px;padding:20px 0;">No per-frame data available.</p>';
    return;
  }

  scores.forEach((score, i) => {
    const isFakeBar = score > 50;
    const height = Math.max(4, score);

    const wrap = document.createElement("div");
    wrap.className = "bar-wrap";

    const bar = document.createElement("div");
    bar.className = `bar ${isFakeBar ? "bar-fake" : "bar-real"}`;
    bar.style.height = "0%";
    bar.setAttribute("data-tip", `Frame ${i + 1}: ${score}%`);

    wrap.appendChild(bar);
    container.appendChild(wrap);

    setTimeout(() => {
      bar.style.height = `${height}%`;
    }, 100 + i * 30);
  });
}

// ─────────────────────────────────────────────────────
// Error display
// ─────────────────────────────────────────────────────
function showError(msg) {
  document.getElementById("error-msg").textContent = msg;
  showSection("error-section");
}

// ─────────────────────────────────────────────────────
// Loading step animation
// ─────────────────────────────────────────────────────
function animateLoadingSteps() {
  const steps = ["step-1", "step-2", "step-3"];
  steps.forEach(s => {
    const el = document.getElementById(s);
    el.classList.remove("active", "done");
  });

  let i = 0;
  function next() {
    if (i > 0) {
      document.getElementById(steps[i - 1]).classList.remove("active");
      document.getElementById(steps[i - 1]).classList.add("done");
    }
    if (i < steps.length) {
      document.getElementById(steps[i]).classList.add("active");
      i++;
      setTimeout(next, i < steps.length ? 4000 : 99999);
    }
  }
  next();
}

// ─────────────────────────────────────────────────────
// Copy result to clipboard
// ─────────────────────────────────────────────────────
function copyResult() {
  if (!lastResult) return;
  const { verdict, fake_probability, real_probability, frame_count, filename } = lastResult;
  const text =
    `DeepShield AI — DINO-G50 Result\n` +
    `File: ${filename}\n` +
    `Verdict: ${verdict}\n` +
    `Fake Probability: ${fake_probability}%\n` +
    `Real Probability: ${real_probability}%\n` +
    `Frames Analyzed: ${frame_count}`;

  navigator.clipboard?.writeText(text).then(() => {
    const btn = document.querySelector(".action-secondary");
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = "✅ Copied!";
      setTimeout(() => { btn.textContent = orig; }, 2000);
    }
  }).catch(() => alert(text));
}

// ─────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}
