"use strict";

// ── DOM refs ───────────────────────────────────────────────────
const $frame = document.getElementById("frame");
const $frameEmpty = document.getElementById("frame-empty");
const $frameMeta = document.getElementById("frame-meta");
const $btnPrev = document.getElementById("btn-prev");
const $btnNext = document.getElementById("btn-next");
const $btnLive = document.getElementById("btn-live");

const $chatLog = document.getElementById("chat-log");
const $chatForm = document.getElementById("chat-form");
const $chatInput = document.getElementById("chat-input");
const $btnSend = document.getElementById("btn-send");
const $optDryRun = document.getElementById("opt-dry-run");
const $optNoFocus = document.getElementById("opt-no-focus");
const $optPlatform = document.getElementById("opt-platform");

const $logView = document.getElementById("log-view");
const $btnClearLogs = document.getElementById("btn-clear-logs");

// ── state ──────────────────────────────────────────────────────
const state = {
  liveMode: true,
  currentId: null,
  knownIds: [],          // ascending order, fed by /api/frames listings
  busy: false,
  logsAtBottom: true,
  globalSrc: null,
  runChats: new Map(),   // run_id -> chat <li>
};

// ── frame fetching ─────────────────────────────────────────────
function setFrameSrc(id) {
  if (id === null || id === undefined) {
    $frame.classList.add("empty");
    $frameEmpty.style.display = "";
    $frameMeta.textContent = "—";
    return;
  }
  $frame.classList.remove("empty");
  $frameEmpty.style.display = "none";
  $frame.src = `/api/frames/${id}`;
  state.currentId = id;
  updateFrameMeta();
}

function updateFrameMeta() {
  const id = state.currentId;
  if (id === null) {
    $frameMeta.textContent = "—";
    return;
  }
  const idx = state.knownIds.indexOf(id);
  const total = state.knownIds.length;
  if (idx >= 0 && total > 0) {
    $frameMeta.textContent =
      `${idx + 1}/${total}` + (state.liveMode ? "  • live" : "");
  } else {
    $frameMeta.textContent = state.liveMode ? "live" : "history";
  }
  $btnPrev.disabled = idx <= 0;
  $btnNext.disabled = idx < 0 || idx >= total - 1;
  $btnLive.classList.toggle("active", state.liveMode);
}

async function refreshKnownIds() {
  // Fetch up to 500 frames (newest-first), reverse to ascending.
  try {
    const r = await fetch("/api/frames?limit=500");
    if (!r.ok) return;
    const data = await r.json();
    const ids = (data.items || []).map(m => m.id).reverse();
    state.knownIds = ids;
    if (state.liveMode && ids.length > 0) {
      const latest = ids[ids.length - 1];
      if (latest !== state.currentId) setFrameSrc(latest);
      else updateFrameMeta();
    } else {
      updateFrameMeta();
    }
  } catch (_) {}
}

async function pollLatest() {
  // Long-poll: server holds for up to 10s if no change.
  while (true) {
    try {
      const since = state.currentId ?? "";
      const url = `/api/frames/latest?wait=1&since=${encodeURIComponent(since)}`;
      const r = await fetch(url);
      if (r.ok) {
        const { item } = await r.json();
        if (item && item.id !== state.currentId) {
          if (!state.knownIds.includes(item.id)) {
            state.knownIds.push(item.id);
            // Cap mirror of server's ring buffer.
            if (state.knownIds.length > 600) {
              state.knownIds.splice(0, state.knownIds.length - 500);
            }
          }
          if (state.liveMode) setFrameSrc(item.id);
          else updateFrameMeta();
        }
      }
    } catch (_) {
      await sleep(1000);
    }
  }
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ── frame controls ─────────────────────────────────────────────
$btnPrev.addEventListener("click", async () => {
  if (state.knownIds.length === 0) return;
  const idx = state.knownIds.indexOf(state.currentId);
  if (idx > 0) {
    state.liveMode = false;
    setFrameSrc(state.knownIds[idx - 1]);
  } else if (idx < 0 && state.knownIds.length > 0) {
    state.liveMode = false;
    setFrameSrc(state.knownIds[state.knownIds.length - 1]);
  }
});

$btnNext.addEventListener("click", () => {
  if (state.knownIds.length === 0) return;
  const idx = state.knownIds.indexOf(state.currentId);
  if (idx >= 0 && idx < state.knownIds.length - 1) {
    const next = state.knownIds[idx + 1];
    setFrameSrc(next);
    if (idx + 1 === state.knownIds.length - 1) state.liveMode = true;
    updateFrameMeta();
  }
});

$btnLive.addEventListener("click", () => {
  state.liveMode = true;
  if (state.knownIds.length > 0) {
    setFrameSrc(state.knownIds[state.knownIds.length - 1]);
  } else {
    updateFrameMeta();
  }
});

// ── chat / runs ────────────────────────────────────────────────
function appendChat({ runId, intent, status, reason }) {
  let li = state.runChats.get(runId);
  if (!li) {
    li = document.createElement("li");
    li.dataset.runId = runId;
    const intentEl = document.createElement("div");
    intentEl.className = "intent";
    intentEl.textContent = intent;
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    li.append(intentEl, metaEl);
    $chatLog.appendChild(li);
    state.runChats.set(runId, li);
  }
  const meta = li.querySelector(".meta");
  meta.textContent = `${runId.slice(0, 6)} · ${status}` +
    (reason ? ` — ${reason}` : "");
  li.classList.toggle("outcome-ok", status === "succeeded");
  li.classList.toggle("outcome-bad", status === "failed");
  li.classList.toggle("outcome-err", status === "error");
  $chatLog.scrollTop = $chatLog.scrollHeight;
}

async function pollRunStatus(runId) {
  while (true) {
    try {
      const r = await fetch(`/api/runs/${runId}`);
      if (!r.ok) return;
      const rec = await r.json();
      appendChat({
        runId, intent: rec.intent,
        status: rec.status, reason: rec.reason,
      });
      if (rec.status !== "running" && rec.status !== "pending") {
        state.busy = false;
        $btnSend.disabled = false;
        return;
      }
    } catch (_) {}
    await sleep(700);
  }
}

$chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const intent = $chatInput.value.trim();
  if (!intent || state.busy) return;
  state.busy = true;
  $btnSend.disabled = true;
  const body = {
    intent,
    no_focus: $optNoFocus.checked,
    dry_run: $optDryRun.checked,
    platform: $optPlatform.value,
  };
  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (r.status === 409) {
      appendSystemLog("ERROR", "another run is already in progress");
      state.busy = false;
      $btnSend.disabled = false;
      return;
    }
    if (!r.ok) {
      const t = await r.text();
      appendSystemLog("ERROR", `run rejected: ${t}`);
      state.busy = false;
      $btnSend.disabled = false;
      return;
    }
    const rec = await r.json();
    $chatInput.value = "";
    appendChat({
      runId: rec.run_id, intent: rec.intent, status: rec.status,
    });
    pollRunStatus(rec.run_id);
  } catch (err) {
    appendSystemLog("ERROR", String(err));
    state.busy = false;
    $btnSend.disabled = false;
  }
});

$chatInput.addEventListener("keydown", (e) => {
  // ⌘/Ctrl+Enter sends.
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    $chatForm.requestSubmit();
  }
});

// ── logs ───────────────────────────────────────────────────────
function appendSystemLog(level, msg) {
  appendLogLine({ ts: Date.now() / 1000, level, source: "system", msg });
}

function tsString(ts) {
  const d = new Date(ts * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function appendLogLine(ev) {
  const line = document.createElement("div");
  line.className = `log-line lvl-${ev.level} src-${ev.source}`;
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = tsString(ev.ts) + "  ";
  line.appendChild(ts);
  line.appendChild(document.createTextNode(ev.msg));
  $logView.appendChild(line);
  // Trim DOM if it grows huge.
  while ($logView.childNodes.length > 1500) {
    $logView.removeChild($logView.firstChild);
  }
  if (state.logsAtBottom) {
    $logView.scrollTop = $logView.scrollHeight;
  }
}

$logView.addEventListener("scroll", () => {
  const nearBottom =
    $logView.scrollHeight - $logView.scrollTop - $logView.clientHeight < 30;
  state.logsAtBottom = nearBottom;
});

$btnClearLogs.addEventListener("click", () => {
  $logView.innerHTML = "";
});

function connectGlobalLogs() {
  if (state.globalSrc) state.globalSrc.close();
  const src = new EventSource("/api/logs?tail=200");
  src.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      appendLogLine(ev);
    } catch (_) {}
  };
  src.onerror = () => {
    src.close();
    setTimeout(connectGlobalLogs, 1500);
  };
  state.globalSrc = src;
}

// ── boot ───────────────────────────────────────────────────────
async function init() {
  await refreshKnownIds();
  connectGlobalLogs();
  pollLatest();
  // Periodic re-list to catch ring-buffer evictions / deep history.
  setInterval(refreshKnownIds, 5000);
}

init();
