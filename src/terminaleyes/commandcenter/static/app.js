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
const $optVault = document.getElementById("opt-vault");
const $btnLock = document.getElementById("btn-lock");
const $btnUnlock = document.getElementById("btn-unlock");

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
$frame.addEventListener("load", () => {
  $frame.classList.remove("empty");
  $frameEmpty.style.display = "none";
});
// One-shot resync guard so we don't spin forever if every frame
// 404s (e.g. server has zero frames).
let _imageErrorResyncing = false;
$frame.addEventListener("error", async () => {
  $frame.classList.add("empty");
  $frameEmpty.style.display = "";
  $frameEmpty.textContent =
    `image load failed for id ${state.currentId} — refreshing frame list…`;
  if (_imageErrorResyncing) return;
  _imageErrorResyncing = true;
  try {
    // Stale id: the FrameStore was rebuilt (cc restart, ring-buffer
    // eviction, etc.) so our cached id no longer maps to anything on
    // disk. Re-list from the server and jump to the new latest.
    const before = state.currentId;
    state.currentId = null;     // force setFrameSrc to actually swap
    await refreshKnownIds();
    if (state.currentId === before || state.currentId === null) {
      $frameEmpty.textContent =
        "no frames available — send an instruction.";
    }
  } finally {
    _imageErrorResyncing = false;
  }
});

function setFrameSrc(id) {
  if (id === null || id === undefined) {
    $frame.classList.add("empty");
    $frameEmpty.style.display = "";
    $frameMeta.textContent = "—";
    return;
  }
  // Cache-bust per id so a re-render picks up a fresh fetch even if
  // the URL is the same.
  $frame.src = `/api/frames/${id}?t=${Date.now()}`;
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
    if (!r.ok) {
      $frameEmpty.textContent =
        `frames API returned ${r.status} ${r.statusText}`;
      return;
    }
    const data = await r.json();
    const ids = (data.items || []).map(m => m.id).reverse();
    state.knownIds = ids;
    if (ids.length === 0) {
      $frameEmpty.textContent = "No frames yet — send an instruction.";
      return;
    }
    if (state.liveMode) {
      const latest = ids[ids.length - 1];
      if (latest !== state.currentId) setFrameSrc(latest);
      else updateFrameMeta();
    } else {
      updateFrameMeta();
    }
  } catch (e) {
    $frameEmpty.textContent = "frames API failed: " + (e && e.message);
  }
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
        if ($btnLock) $btnLock.disabled = false;
        if ($btnUnlock) $btnUnlock.disabled = false;
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
      if ($btnLock) $btnLock.disabled = false;
      if ($btnUnlock) $btnUnlock.disabled = false;
      return;
    }
    if (!r.ok) {
      const t = await r.text();
      appendSystemLog("ERROR", `run rejected: ${t}`);
      state.busy = false;
      $btnSend.disabled = false;
      if ($btnLock) $btnLock.disabled = false;
      if ($btnUnlock) $btnUnlock.disabled = false;
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
    if ($btnLock) $btnLock.disabled = false;
    if ($btnUnlock) $btnUnlock.disabled = false;
  }
});

// ── quick actions: Lock / Unlock ──────────────────────────────
async function startRun(body, fallbackIntent) {
  if (state.busy) return;
  state.busy = true;
  $btnSend.disabled = true;
  $btnLock.disabled = true;
  $btnUnlock.disabled = true;
  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (r.status === 409) {
      appendSystemLog("ERROR", "another run is already in progress");
      return;
    }
    if (!r.ok) {
      const t = await r.text();
      appendSystemLog("ERROR", `run rejected: ${t}`);
      return;
    }
    const rec = await r.json();
    appendChat({
      runId: rec.run_id,
      intent: rec.intent || fallbackIntent,
      status: rec.status,
    });
    pollRunStatus(rec.run_id);
  } catch (err) {
    appendSystemLog("ERROR", String(err));
  } finally {
    // pollRunStatus re-enables Send on terminal status, but for the
    // synchronous error paths above we have to release the buttons
    // here. pollRunStatus also re-enables, so calling twice is fine.
    if (state.busy) {
      // poll is still running; leave buttons disabled.
    } else {
      $btnSend.disabled = false;
      $btnLock.disabled = false;
      $btnUnlock.disabled = false;
    }
  }
}

$btnLock.addEventListener("click", () => {
  startRun({
    intent: "lock the screen",
    no_focus: true,
    dry_run: $optDryRun.checked,
    platform: $optPlatform.value,
  }, "lock the screen");
});

$btnUnlock.addEventListener("click", () => {
  const vault = ($optVault.value || "").trim();
  if (!vault) {
    appendSystemLog("ERROR", "Unlock requires a vault entry name");
    return;
  }
  startRun({
    intent: "unlock the screen",
    no_focus: true,
    dry_run: $optDryRun.checked,
    platform: $optPlatform.value,
    vault,
  }, "unlock the screen");
});

$chatInput.addEventListener("keydown", (e) => {
  // Enter sends. Shift+Enter inserts a newline so multi-line intents
  // are still possible. ⌘/Ctrl+Enter is kept for muscle memory.
  if (e.key === "Enter" && !e.isComposing) {
    if (e.shiftKey) return;            // Shift+Enter → newline
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

// ── chat history (replay prior runs on load) ─────────────────────
async function loadChatHistory() {
  try {
    const r = await fetch("/api/runs?limit=50");
    if (!r.ok) return;
    const data = await r.json();
    const items = data.items || [];
    // The server returns newest-first; chat reads top-to-bottom
    // oldest-to-newest, so reverse before appending.
    for (const rec of items.reverse()) {
      appendChat({
        runId: rec.run_id,
        intent: rec.intent,
        status: rec.status,
        reason: rec.reason,
      });
      // If a run somehow ended up still "running" while the page
      // was reloaded, resume polling its status so the chat updates
      // when it eventually completes.
      if (rec.status === "running" || rec.status === "pending") {
        pollRunStatus(rec.run_id);
      }
    }
  } catch (_) {}
}

// ── boot ───────────────────────────────────────────────────────
async function init() {
  await loadChatHistory();
  await refreshKnownIds();
  connectGlobalLogs();
  pollLatest();
  // Periodic re-list to catch ring-buffer evictions / deep history.
  setInterval(refreshKnownIds, 5000);
}

init();
