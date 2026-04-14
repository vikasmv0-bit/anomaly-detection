/**
 * app.js — Surveillance AI Dashboard (Multi-Stream)
 * --------------------------------------------------
 * Handles dynamic stream cards, per-stream telemetry,
 * priority-sorting, global chart, and aggregated alerts.
 */

// ── State ────────────────────────────────────────────────────────────────
const activeCards = {};   // { streamId: DOMElement }
let anomalyChart = null;
let pollingInterval = null;
let dismissedAlerts = new Set(); // for the clear button

// ── Initialization ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    setupDragDrop();
    updateTime();
    setInterval(updateTime, 1000);
    pollingInterval = setInterval(fetchTelemetry, 1000);
});

function updateTime() {
    const now = new Date();
    document.getElementById("header-time").textContent = now.toLocaleTimeString("en-US", {
        hour12: true, hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
}

// ── Chart Setup ───────────────────────────────────────────────────────────
function initChart() {
    const ctx = document.getElementById("anomaly-chart").getContext("2d");

    const gradient = ctx.createLinearGradient(0, 0, 0, 180);
    gradient.addColorStop(0, "rgba(0, 240, 255, 0.25)");
    gradient.addColorStop(1, "rgba(0, 240, 255, 0.0)");

    anomalyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Anomaly Score",
                    data: [],
                    borderColor: "#00f0ff",
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                },
                {
                    label: "Threshold",
                    data: [],
                    borderColor: "rgba(239, 68, 68, 0.5)",
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    tension: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            interaction: { mode: "nearest", axis: "x", intersect: false },
            plugins: {
                legend: {
                    labels: { color: "#94a3b8", font: { size: 11, family: "Inter" }, boxWidth: 12, padding: 16 },
                },
                tooltip: {
                    backgroundColor: "rgba(17, 24, 39, 0.95)",
                    titleColor: "#f1f5f9", bodyColor: "#94a3b8",
                    borderColor: "rgba(0, 240, 255, 0.2)", borderWidth: 1, padding: 10,
                    callbacks: { label: (ctx) => `Score: ${ctx.parsed.y.toFixed(4)}` },
                },
            },
            scales: {
                x: {
                    ticks: { color: "#64748b", maxTicksLimit: 10, font: { size: 10 } },
                    grid: { color: "rgba(148, 163, 184, 0.06)" },
                },
                y: {
                    min: 0, max: 1,
                    ticks: { color: "#64748b", font: { size: 10 }, stepSize: 0.2 },
                    grid: { color: "rgba(148, 163, 184, 0.06)" },
                },
            },
        },
    });
}

function updateChart(scores) {
    if (!anomalyChart || !scores || scores.length === 0) return;
    anomalyChart.data.labels = scores.map((s) => s.frame);
    anomalyChart.data.datasets[0].data = scores.map((s) => s.score);
    anomalyChart.data.datasets[1].data = new Array(scores.length).fill(0.55);
    anomalyChart.update("none");
}

// ── Main Polling Loop ─────────────────────────────────────────────────────
function fetchTelemetry() {
    fetch("/alerts")
        .then((res) => res.json())
        .then((data) => {
            if (!data.streams) return;

            const streamIds = Object.keys(data.streams);

            // 1. Create or update stream cards
            for (const sid of streamIds) {
                if (!activeCards[sid]) createStreamCard(sid, data.streams[sid]);
                updateStreamCard(sid, data.streams[sid]);
            }

            // 2. Remove cards for dead streams
            for (const sid in activeCards) {
                if (!streamIds.includes(sid)) removeStreamCard(sid);
            }

            // 3. Show/hide placeholders
            const noPlaceholder = document.getElementById("no-streams-placeholder");
            const globalDash = document.getElementById("global-dashboard");
            if (streamIds.length > 0) {
                if (noPlaceholder) noPlaceholder.style.display = "none";
                if (globalDash) globalDash.style.display = "block";
            } else {
                if (noPlaceholder) noPlaceholder.style.display = "flex";
                if (globalDash) globalDash.style.display = "none";
            }

            // 4. Sort grid by priority
            sortByPriority(data.streams);

            // 5. Update the aggregated stats, alerts, and chart
            updateGlobalStats(data.streams);
            updateGlobalAlerts(data.streams);
            updateGlobalStatusBadge(data.streams);
        })
        .catch((err) => console.error("Polling error:", err));
}

// ── Priority Engine ──────────────────────────────────────────────────────
const PRIORITY_KEYWORDS = ["ACCIDENT", "FIGHTING", "THEFT"];

function getPriorityScore(streamData) {
    let score = (streamData.stats?.peak_score || 0) * 10;
    let isCritical = false;
    if (streamData.alerts && streamData.alerts.length > 0) {
        const latest = streamData.alerts[streamData.alerts.length - 1];
        const desc = (latest.description || "").toUpperCase();
        if (PRIORITY_KEYWORDS.some((k) => desc.includes(k))) { score += 1000; isCritical = true; }
        else if (latest.type === "danger") { score += 500; }
    }
    return { score, isCritical };
}

function sortByPriority(streamsData) {
    const container = document.getElementById("streams-container");
    const cards = Array.from(container.querySelectorAll(".stream-card"));

    cards.sort((a, b) => {
        const sa = parseFloat(a.dataset.priorityScore || "0");
        const sb = parseFloat(b.dataset.priorityScore || "0");
        return sb - sa;
    });

    // Also update card styles and priority badge
    for (const card of cards) {
        const sid = card.dataset.streamId;
        const sd = streamsData[sid];
        if (!sd) continue;

        const { score, isCritical } = getPriorityScore(sd);
        card.dataset.priorityScore = score;

        const badge = card.querySelector(".priority-badge");

        // Determine badge
        const hasAlerts = sd.alerts && sd.alerts.length > 0;
        const latestDesc = hasAlerts ? (sd.alerts[sd.alerts.length - 1].description || "").toUpperCase() : "";
        const specificEvent = PRIORITY_KEYWORDS.find((k) => latestDesc.includes(k));

        if (specificEvent) {
            card.className = "stream-card priority-critical";
            if (badge) { badge.className = "priority-badge critical"; badge.textContent = specificEvent; }
        } else if (isCritical) {
            card.className = "stream-card priority-critical";
            if (badge) { badge.className = "priority-badge critical"; badge.textContent = "⚠ DANGER"; }
        } else if (score > 3) {
            card.className = "stream-card priority-high";
            if (badge) { badge.className = "priority-badge high"; badge.textContent = "Warning"; }
        } else {
            card.className = "stream-card";
            if (badge) { badge.className = "priority-badge normal"; badge.textContent = "Normal"; }
        }

        container.appendChild(card);
    }
}

// ── Stream Card Management ────────────────────────────────────────────────
function createStreamCard(streamId, streamData) {
    const sourceName = escapeHtml(streamData.source || "Stream");
    const isUpload = streamData.is_upload;

    const card = document.createElement("div");
    card.className = "stream-card";
    card.dataset.streamId = streamId;
    card.dataset.priorityScore = "0";

    card.innerHTML = `
        <div class="stream-card-header">
            <div class="stream-card-title">
                <span class="stream-dot"></span>
                <span class="stream-source-name" title="${sourceName}">${isUpload ? "📹 " : "📷 "}${sourceName}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span class="priority-badge normal">Normal</span>
                <button class="close-stream-btn" onclick="stopStream('${streamId}')">✕ Stop</button>
            </div>
        </div>

        <div style="position:relative;">
            <div class="stream-event-toast" id="toast-${streamId}"></div>
            <div class="stream-video-wrap">
                <img src="/video_feed/${streamId}" alt="Stream ${streamId}"
                     onerror="this.src=''; this.style.display='none';">
            </div>
        </div>

        <div class="stream-mini-stats">
            <div class="stream-mini-stat">
                <div class="stream-mini-stat-val red" id="card-peak-${streamId}">0.00</div>
                <div class="stream-mini-stat-label">Peak Score</div>
            </div>
            <div class="stream-mini-stat">
                <div class="stream-mini-stat-val cyan" id="card-alerts-${streamId}">0</div>
                <div class="stream-mini-stat-label">Alerts</div>
            </div>
            <div class="stream-mini-stat">
                <div class="stream-mini-stat-val green" id="card-frames-${streamId}">0</div>
                <div class="stream-mini-stat-label">Frames</div>
            </div>
        </div>

        ${isUpload ? `
        <div class="stream-upload-progress" id="uprog-${streamId}" style="display:none;">
            <div class="upload-prog-bar"><div class="upload-prog-fill" id="upfill-${streamId}"></div></div>
            <div class="upload-prog-row">
                <span id="uptext-${streamId}">Processing…</span>
                <a class="upload-download-link" id="updl-${streamId}" href="#">⬇ Download</a>
            </div>
        </div>` : ""}

        <div class="stream-alert-log" id="alog-${streamId}">
            <span style="color:rgba(255,255,255,0.2);">Waiting for events…</span>
        </div>
    `;

    document.getElementById("streams-container").appendChild(card);
    activeCards[streamId] = card;
}

function updateStreamCard(streamId, streamData) {
    const card = activeCards[streamId];
    if (!card) return;

    const stats = streamData.stats || {};
    const peakEl   = document.getElementById(`card-peak-${streamId}`);
    const alertsEl = document.getElementById(`card-alerts-${streamId}`);
    const framesEl = document.getElementById(`card-frames-${streamId}`);

    if (peakEl)   peakEl.textContent   = (stats.peak_score || 0).toFixed(2);
    if (alertsEl) alertsEl.textContent = stats.total_alerts || 0;
    if (framesEl) framesEl.textContent = (stats.total_frames || 0).toLocaleString();

    // Event toast
    const toast = document.getElementById(`toast-${streamId}`);
    if (toast && streamData.alerts && streamData.alerts.length > 0) {
        const latest = streamData.alerts[streamData.alerts.length - 1];
        const desc = (latest.description || "").toUpperCase();
        if (PRIORITY_KEYWORDS.some((k) => desc.includes(k))) {
            toast.textContent = "⚠ " + latest.description;
            toast.style.display = "block";
        } else {
            toast.style.display = "none";
        }
    }

    // Alert log
    const logEl = document.getElementById(`alog-${streamId}`);
    if (logEl && streamData.alerts && streamData.alerts.length > 0) {
        let html = "";
        const recent = streamData.alerts.slice(-8).reverse();
        for (const a of recent) {
            const color = a.type === "danger" ? "#ef4444" : "#fbbf24";
            html += `<div class="stream-alert-entry">
                <span style="color:${color}; font-weight:600;">[${a.timestamp}]</span>
                <span style="opacity:0.85;"> ${escapeHtml(a.description)}</span>
            </div>`;
        }
        logEl.innerHTML = html;
    }

    // Upload progress
    if (streamData.is_upload && streamData.upload_state) {
        const progDiv = document.getElementById(`uprog-${streamId}`);
        const fill    = document.getElementById(`upfill-${streamId}`);
        const text    = document.getElementById(`uptext-${streamId}`);
        const dl      = document.getElementById(`updl-${streamId}`);
        if (progDiv) progDiv.style.display = "block";
        if (fill) fill.style.width = (streamData.upload_state.progress || 0) + "%";

        if (streamData.upload_state.processing) {
            const cur = streamData.upload_state.current_frame || 0;
            const tot = streamData.upload_state.total_frames  || 0;
            if (text) text.textContent = `Frame ${cur} / ${tot} (${streamData.upload_state.progress || 0}%)`;
        } else {
            if (fill) { fill.style.background = "#06d6a0"; }
            if (text) text.textContent = streamData.upload_state.results?.error
                ? "Error: " + streamData.upload_state.results.error
                : "Processing Complete ✓";
            if (dl && streamData.upload_state.output_file) {
                dl.href = "/download/" + streamData.upload_state.output_file;
                dl.style.display = "inline-block";
            }
        }
    }
}

function removeStreamCard(streamId) {
    const card = activeCards[streamId];
    if (card) { card.remove(); delete activeCards[streamId]; }
}

// ── Global Stats, Chart & Alerts ──────────────────────────────────────────
function updateGlobalStats(streamsData) {
    let totalFrames = 0, totalAnomalies = 0, peakScore = 0;

    for (const sid in streamsData) {
        const s = streamsData[sid].stats || {};
        totalFrames    += s.total_frames    || 0;
        totalAnomalies += s.anomaly_frames  || 0;
        peakScore = Math.max(peakScore, s.peak_score || 0);
    }

    const pct = totalFrames > 0 ? ((totalAnomalies / totalFrames) * 100).toFixed(1) : "0.0";
    const nStreams = Object.keys(streamsData).length;

    document.getElementById("stat-frames").textContent     = totalFrames.toLocaleString();
    document.getElementById("stat-anomalies").textContent  = totalAnomalies.toLocaleString();
    document.getElementById("stat-percentage").textContent = pct + "%";
    document.getElementById("stat-peak").textContent       = peakScore.toFixed(4);
    document.getElementById("stream-count-badge").textContent = nStreams + (nStreams === 1 ? " stream" : " streams");

    // Drive chart from highest-priority stream
    const prioritized = Object.entries(streamsData)
        .map(([sid, sd]) => ({ sid, sd, ps: getPriorityScore(sd).score }))
        .sort((a, b) => b.ps - a.ps);

    if (prioritized.length > 0) {
        const best = prioritized[0];
        updateChart(best.sd.scores || []);
        const label = document.getElementById("chart-stream-label");
        if (label) label.textContent = `(${best.sd.source || best.sid})`;
    }
}

function updateGlobalAlerts(streamsData) {
    // Aggregate all alerts across streams
    let allAlerts = [];
    for (const sid in streamsData) {
        const sd = streamsData[sid];
        if (sd.alerts) {
            for (const a of sd.alerts) {
                allAlerts.push({ ...a, stream_id: sid, source: sd.source });
            }
        }
    }

    // Sort by frame number descending to get newest first
    allAlerts.sort((a, b) => b.frame - a.frame);

    const listEl  = document.getElementById("alerts-list");
    const countEl = document.getElementById("alert-count");

    // Filter out dismissed
    const visible = allAlerts.filter((a) => !dismissedAlerts.has(`${a.stream_id}-${a.frame}`));
    countEl.textContent = visible.length;

    if (visible.length === 0) {
        listEl.innerHTML = `<div class="empty-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
            <p>No anomalies detected</p>
        </div>`;
        return;
    }

    let html = "";
    for (const a of visible.slice(0, 40)) {
        html += `
        <div class="alert-item">
            <div class="alert-indicator ${a.type || "warning"}"></div>
            <div class="alert-content">
                <div class="alert-title">${escapeHtml(a.description)}</div>
                <div class="alert-meta">
                    <span class="alert-source">${escapeHtml(a.source || a.stream_id)}</span>
                    <span>Frame #${a.frame}</span>
                    <span>${a.timestamp || ""}</span>
                    <span class="alert-score">Score: ${(a.score || 0).toFixed(4)}</span>
                </div>
            </div>
        </div>`;
    }
    listEl.innerHTML = html;
}

function clearGlobalAlerts() {
    // Mark all current alerts as dismissed
    fetch("/alerts")
        .then((r) => r.json())
        .then((data) => {
            if (!data.streams) return;
            for (const sid in data.streams) {
                for (const a of (data.streams[sid].alerts || [])) {
                    dismissedAlerts.add(`${sid}-${a.frame}`);
                }
            }
        });
}

function updateGlobalStatusBadge(streamsData) {
    const badge = document.getElementById("system-status");
    let anyDanger = false;
    let anyActive = Object.keys(streamsData).length > 0;

    for (const sid in streamsData) {
        const sd = streamsData[sid];
        if (sd.alerts && sd.alerts.length > 0) {
            const latest = sd.alerts[sd.alerts.length - 1];
            if (latest.type === "danger" || PRIORITY_KEYWORDS.some(k => (latest.description || "").toUpperCase().includes(k))) {
                anyDanger = true;
            }
        }
    }

    if (anyDanger) {
        badge.className = "status-badge danger";
        badge.querySelector(".status-text").textContent = "⚠ Critical";
    } else if (anyActive) {
        badge.className = "status-badge active";
        badge.querySelector(".status-text").textContent = "Monitoring";
    } else {
        badge.className = "status-badge";
        badge.querySelector(".status-text").textContent = "Standby";
    }
}

// ── Stream Controls ──────────────────────────────────────────────────────
function startCamera() {
    const url = document.getElementById("camera-url").value.trim() || "0";
    fetch("/start_camera", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: url }),
    })
    .then((r) => r.json())
    .then((data) => {
        if (data.error) alert("Could not start stream: " + data.error);
    })
    .catch((e) => alert("Connection error: " + e));
}

function stopStream(streamId) {
    fetch("/stop_camera/" + streamId, { method: "GET" })
        .catch((e) => console.error("Stop error:", e));
}

// ── File Upload ──────────────────────────────────────────────────────────
function handleFileSelect(e) {
    const files = e.target.files;
    for (let i = 0; i < files.length; i++) {
        uploadFile(files[i]);
    }
    e.target.value = "";
}

function uploadFile(file) {
    const status = document.getElementById("upload-status-text");
    const zoneText = document.getElementById("upload-zone-text");
    if (status)   { status.style.color = "#00f0ff"; status.textContent = "Uploading…"; }
    if (zoneText) zoneText.textContent = `Uploading ${file.name}…`;

    const fd = new FormData();
    fd.append("video", file);

    fetch("/upload", { method: "POST", body: fd })
        .then((r) => r.json())
        .then((data) => {
            if (data.error) {
                if (status) { status.style.color = "#ef4444"; status.textContent = data.error; }
            } else {
                if (status) { status.style.color = "#06d6a0"; status.textContent = "Uploaded!"; }
                if (zoneText) zoneText.textContent = "Drag & drop or click to test multiple videos (MP4, AVI, MOV, MKV, WMV)";
                setTimeout(() => { if (status) status.textContent = ""; }, 4000);
            }
        })
        .catch(() => {
            if (status) { status.style.color = "#ef4444"; status.textContent = "Upload failed."; }
        });
}

// ── Drag & Drop on Upload Target ─────────────────────────────────────────
function setupDragDrop() {
    const target = document.getElementById("upload-drop-target");
    if (!target) return;

    target.addEventListener("dragover",  (e) => { e.preventDefault(); target.classList.add("drag-over"); });
    target.addEventListener("dragleave", ()  => target.classList.remove("drag-over"));
    target.addEventListener("drop",      (e) => {
        e.preventDefault();
        target.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            for (let i = 0; i < e.dataTransfer.files.length; i++) {
                uploadFile(e.dataTransfer.files[i]);
            }
        }
    });

    // Allow dropping anywhere on the body (as fallback)
    document.body.addEventListener("dragover",  (e) => e.preventDefault());
    document.body.addEventListener("drop",      (e) => {
        e.preventDefault();
        if (e.target === target || target.contains(e.target)) return;
        if (e.dataTransfer.files.length > 0) {
            for (let i = 0; i < e.dataTransfer.files.length; i++) {
                const f = e.dataTransfer.files[i];
                const ext = f.name.split(".").pop().toLowerCase();
                if (["mp4","avi","mov","mkv","wmv"].includes(ext)) uploadFile(f);
            }
        }
    });
}

// ── Utilities ────────────────────────────────────────────────────────────
function escapeHtml(text) {
    const d = document.createElement("div");
    d.textContent = text || "";
    return d.innerHTML;
}
