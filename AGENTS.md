# AGENTS.md

terminaleyes is built around a tiered agent architecture. Each agent
is a small, testable unit with a typed `Outcome` return; higher-tier
agents compose lower-tier ones. The deep technical reference lives in
[CLAUDE.md](./CLAUDE.md); this file is the index.

## At a glance

```
agents/
├── base.py / context.py   — Agent ABC, Outcome, AgentContext
├── vault.py               — AES-256-GCM encrypted credential store
│
├── verify.py              — tier-1: visual yes/no oracle
├── cursor.py              — tier-1: locate cursor (HSV / variance / diff)
├── target.py              — tier-1: locate target by description
│
├── wake.py                — tier-2: wake screen sequence
├── type_text.py           — tier-2: text input (with secret mode)
├── scroll.py              — tier-2: mouse-wheel scroll
│
├── focus.py               — tier-3: centre + maximise the foreground app
├── login.py               — tier-3: wake + verify-login + type secret
├── navigate.py            — tier-3: browser-aware URL bar typing + OCR oracle
├── click.py               — tier-3: find-and-click with scroll-and-retry (was SearchAgent)
│
└── controller.py          — top-level: rules + LLM-planner fallback

commandcenter/             — web UI + REST/SSE backend
├── server.py              — FastAPI app
├── runner.py              — one-at-a-time ControllerAgent runner
├── frame_store.py         — indexes frames written by AgentContext.record_frame()
├── factory.py             — make_default_context_factory(settings, base_dir, bus)
├── log_bus.py             — pub/sub for logs + redirected stdout/stderr
└── static/                — mobile-first SPA
```

Each user-facing CLI subcommand maps to one agent or to the controller.

## Tier 1 — atomic primitives

| Agent | Job | Notes |
|-------|-----|-------|
| `VerifyAgent` | "Does the screen look like X?" — yes/no | Visual-only steering; refuses to hallucinate on dark/asleep screens. JSON-mode + free-form retry. Records the captured frame via `record_label`. |
| `CursorAgent` | Locate the cursor in the current frame | HSV (motion-verified) → oscillation-variance → ROI-prior diff. Variance now uses centroid-of-mask (handles disconnected cursor positions in the trail). No model calls. |
| `TargetAgent` | Locate a target by description | Cascade: OCR (quoted-token primary) → scene-map+ShowUI grounding → ShowUI on cropped sidebar/footer. |

## Tier 2 — actions

| Agent | Job |
|-------|-----|
| `WakeAgent` | Mouse jiggle + Down arrow + click (idempotent). |
| `TypeAgent` | Send text via the keyboard backend. `secret=True` redacts from local logs; `submit=True` follows with Enter. |
| `ScrollAgent` | Mouse-wheel scroll. `direction=up\|down`, `amount=N`, optional `hover_at=(x_pct, y_pct)` so the wheel scrolls the right pane on layouts with independently-scrolling sidebars and main content. |

## Tier 3 — workflows

| Agent | Job | Notes |
|-------|-----|-------|
| `FocusAgent` | Verify the foreground app is centred/maximised; if not, drive `Super+Up` (GNOME) or `Cmd+Ctrl+F` (macOS). | Awake pre-check refuses to act on dark/asleep screens. |
| `LoginAgent` | `Wake → poll-VerifyAgent("login screen?") → TypeAgent(secret) → Enter`. | Password from explicit arg, vault, file, env var, or `getpass`. Visual-only verification — no reliance on the literal word "password" being on screen. |
| `NavigateAgent` | URL-bar navigation with browser activation + post-flight oracle. | Pre-flight `VerifyAgent("is foreground a browser?")`; on miss, GNOME activities → type "firefox"/"google-chrome"/"chromium" → Enter. Post-flight: OCR URL bar + fuzzy match (`difflib.SequenceMatcher` ratio ≥ 0.75) accepts OCR substitutions like `localilama` ↔ `localllama`. |
| `ClickAgent` (alias `SearchAgent`) | Find a target by description and click it. | Wraps `VisualServoHomer`. **Scroll-aware**: if the homer can't locate the target, calls `ScrollAgent` and retries up to `scroll_attempts=3`. Other failure reasons (validator held, etc.) bypass the scroll fallback. |

## Tier 4 — storage

| `Vault` | AES-256-GCM at `~/.config/terminaleyes/vault.enc`, mode `0600`. Scrypt KDF (`N=2^15`, `r=8`, `p=1`). Master passphrase via `getpass` or `TERMINALEYES_VAULT_PASSPHRASE` env var. CLI: `vault add/get/list/remove/status`. |

## Top level — `ControllerAgent`

The orchestrator. Takes a free-form intent, plans a sequence of agent
calls, runs them with audit trail, fails safe.

**Two-phase planning:**

1. **Rule-based router** (default, fast, no LLM):
   - `login` → `[LoginAgent]`
   - `focus` / `center` / `maximise` → `[FocusAgent]`
   - `go to URL` / `open URL` → `[FocusAgent, NavigateAgent]`
   - `click X` → `[FocusAgent, ClickAgent(target=X)]`
   - `type X` → `[TypeAgent(text=X)]`
   - `scroll [up|down] [N]` → `[ScrollAgent]`
   - `A and B` / `A then B` → chain of plans; adjacent-duplicate `focus` steps deduplicated
2. **LLM-planner fallback**: when no rule matches, prompt the
   multimodal model with the registry, expect a JSON plan, validate
   every step's agent name against the registry, reject malformed or
   unknown actions. Disable with `--no-llm-fallback`.

**Defaults:**
- Click-like steps are prefixed with `FocusAgent` unless `--no-focus`.
- Hard cap on total steps (`MAX_STEPS = 12`).
- Each step's `Outcome` is collected; final outcome surfaces the full
  audit trail.

**Usage:**

```bash
terminaleyes do "click the Run button"
terminaleyes do "go to reddit.com/r/LocalLLaMA"
terminaleyes do "login and open reddit.com" --vault myhost
terminaleyes do "scroll down 6"
terminaleyes do --dry-run "wake the screen and centre the browser"
terminaleyes do --no-focus "click X"        # skip the auto-focus prefix
terminaleyes do --no-llm-fallback "..."     # rules-only, refuses unknown intents
```

## Session output dir + frame recording

Every captured frame is written to a per-invocation directory so
runs are visually replayable.

**Location** (priority order):
1. `--output-dir PATH` CLI flag
2. `TERMINALEYES_OUTPUT_DIR` env var (loadable from `.env`)
3. `~/.local/share/terminaleyes/runs/`

**Layout:**
```
<session_dir>/
├── 0001_174330_navigate_browser_check.png    # VerifyAgent capture
├── 0002_174331_homer_capture.png             # ClickAgent per-step
├── 0003_174333_navigate_postflight_full.png
├── 0004_174333_navigate_postflight_urlbar.png
└── homer/                                    # debug subdir
    └── 174337_vs/
        ├── step_01.png                       # annotated step
        ├── oscillation_init_variance.png
        └── ...
```

Filenames: `^\d{4}_\d{6}_[A-Za-z0-9_-]+\.png$`. Sequential `seq` +
`HHMMSS` + agent label.

## Command Center (`commandcenter/`)

A FastAPI server + SPA that exposes the agent layer over HTTP/SSE.

```bash
terminaleyes commandcenter             # http://0.0.0.0:8765
terminaleyes cc --port 8888
```

**How it composes the agent layer:**
- The runner is one-at-a-time: each `POST /api/run` builds a fresh
  `AgentContext` via `make_default_context_factory(settings, base_dir,
  bus)`, calls `ControllerAgent.run(intent=...)`, then closes
  mouse/keyboard/capture. The webcam is held only during a run.
- Per-run output dir = `<watch_dir>/<run_id>/`. `bus.active_run(run_id)`
  is set before the factory runs, so the factory reads
  `bus.current_run_id()` to name the dir. UI's `FrameMeta.run_id`
  matches the runner's `RunRecord.run_id`.
- `FrameStore` polls `<watch_dir>` every 250 ms, indexes new
  images, serves bytes via `/api/frames`. Default watch dir resolves
  from `TERMINALEYES_OUTPUT_DIR` then falls back to the agent default.
- `LogBus` captures the `terminaleyes` logger AND redirects
  `stdout`/`stderr` of the active run; SSE subscribers per-run +
  global get `LogEvent { ts, level, source, msg, run_id }`.

**Endpoints:** see CLAUDE.md "Command Center" for the full table.

## Where to look first

- **Click behaviour** → `commander/visual_servo_homer.py` (the loop), surfaced via `agents/click.py`
- **Cursor detection knobs** → `agents/cursor.py` + `commander/cursor_finder.py`
- **OCR target locator** → `agents/target.py` + `commander/ocr_finder.py`
- **Browser activation logic** → `agents/navigate.py` `_activate_browser`
- **Login wake / verify polling** → `agents/login.py`
- **Vault crypto** → `agents/vault.py`
- **Web UI ↔ agent layer wiring** → `commandcenter/factory.py`, `commandcenter/runner.py`
- **Pi BT/USB HID** → `raspi/server.py`, `raspi/bt_hid.py`, plus
  CLAUDE.md "Pi Zero 2 W — Critical lessons learned"
- **CLI subcommand wiring** → `cli.py`
- **Defaults (model name, base URLs)** → `config/settings.py`

## CLI surface

```bash
# Controller (high-level)
terminaleyes do "<intent>"                  # rules → LLM fallback

# Direct agent invocations (low-level)
terminaleyes login [--vault NAME] [--password-file F] [--password-env V]
terminaleyes focus [--platform linux|macos]
terminaleyes interact                       # legacy REPL, routes through homer

# Vault
terminaleyes vault add NAME
terminaleyes vault get NAME
terminaleyes vault list
terminaleyes vault remove NAME
terminaleyes vault status

# Command Center web UI
terminaleyes commandcenter [--port 8765] [--frames-dir PATH]

# Common to all subcommands
--output-dir PATH                           # per-session frame dir override
TERMINALEYES_OUTPUT_DIR=PATH                # same, via env

# Other (Pi-side / agent loop / capture / etc.)
terminaleyes-pi                             # Pi REST endpoint
terminaleyes run --goal "..."               # older goal-driven loop
terminaleyes watch                          # passive screen observer
terminaleyes endpoint                       # local dev terminal display
terminaleyes capture-test                   # save one webcam frame
```

## Adding a new agent

1. Create `agents/<name>.py` subclassing `Agent` with an `async def run(...)`.
2. Make sure any `await self.ctx.capture.capture_frame()` is followed by `self.ctx.record_frame(image, label="...")` so the capture surfaces in the session output dir.
3. If the agent should be reachable via the controller, add it to `REGISTRY` in `agents/controller.py` with a one-line capability description.
4. Add a rule to `_plan_one` if there's a natural English shape; the LLM-planner fallback handles long-tail intents automatically.
5. Direct CLI subcommand wiring in `cli.py` if it's standalone-useful.
