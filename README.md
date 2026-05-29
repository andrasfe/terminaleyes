# terminaleyes

Vision-based remote control of a real computer. A webcam watches the
target's screen, classical CV + multimodal LLMs locate the cursor and
click targets, and HID commands flow over Bluetooth (or USB) via a
Raspberry Pi. The agent layer composes everything into a single
high-level interface — `terminaleyes do "<intent>"`.

```
[Dev Mac / agent layer] --USB ECM Ethernet--> [Pi Zero 2 W] --BT HID--> [Target Mac/Ubuntu]
        10.0.0.1                                  10.0.0.2                keyboard + mouse
                                                                              ^
                                                                              |
                                                  webcam pointed here  -------+
```

Detailed architecture and Pi setup live in [CLAUDE.md](./CLAUDE.md);
agent index in [AGENTS.md](./AGENTS.md).

## What you can do

```bash
# Top-level controller — rule planner with LLM-planner fallback
terminaleyes do "click the Run button"
terminaleyes do "go to reddit.com/r/LocalLLaMA"
terminaleyes do "login and open reddit.com" --vault myhost
terminaleyes do "scroll down 6"
terminaleyes do --dry-run "wake the screen and centre the browser"

# Direct agent invocations
terminaleyes login              # wake → visually verify login → type from vault
terminaleyes focus              # centre + maximise the foreground app
terminaleyes vault add NAME     # encrypted local credential store

# Web UI (FastAPI + SPA, mobile-friendly)
terminaleyes commandcenter      # http://0.0.0.0:8765 — see frames + run intents

# Legacy REPL (still works)
terminaleyes interact -m "click X"
```

## Agent layer

```
agents/
├── base.py / context.py   — Agent ABC, Outcome, AgentContext (output_dir + record_frame)
├── vault.py               — AES-256-GCM credential store (scrypt KDF)
│
├── verify.py              — tier-1: visual yes/no oracle
├── cursor.py              — tier-1: locate cursor (HSV / variance / diff)
├── target.py              — tier-1: locate target by description
│
├── wake.py                — tier-2: wake screen
├── type_text.py           — tier-2: text input (with secret mode)
├── scroll.py              — tier-2: mouse-wheel scroll
│
├── focus.py               — tier-3: centre + maximise the foreground app
├── login.py               — tier-3: wake + verify-login + type secret
├── navigate.py            — tier-3: browser-aware URL bar typing + OCR oracle
├── click.py               — tier-3: find-and-click + scroll-and-retry (was SearchAgent)
│
└── controller.py          — top-level: rules + LLM-planner fallback

commandcenter/             — web UI + REST/SSE backend (FastAPI + SPA)
```

Each agent is a small testable unit returning a typed
`Outcome { success, reason, data }`. Higher tiers compose lower tiers
through the shared `AgentContext`. The `ControllerAgent` decomposes
free-form English intents into agent sequences:

```
"click the Run button"      → [FocusAgent, ClickAgent(target=...)]
"go to URL"                 → [FocusAgent, NavigateAgent(url=...)]
"login and open reddit.com" → [LoginAgent, FocusAgent, NavigateAgent]
"scroll down 6"             → [ScrollAgent(direction=down, amount=6)]
"wake then centre browser"  → [WakeAgent, FocusAgent]   (LLM fallback)
```

Safe defaults:
- Click-like intents auto-prefix `FocusAgent` (skip with `--no-focus`).
- `NavigateAgent` refuses to send keystrokes until verifier confirms a browser is foreground; activates one via GNOME activities (Super → type browser name → Enter) → ClickAgent on dock icon → Super+N sweep.
- `LoginAgent` refuses to type until verifier confirms a login screen — visual cues only, NOT keyword matching for "password".
- `FocusAgent` refuses to act on dark/asleep frames.
- LLM-planner output is validated against the registry; unknown agent names reject the plan.

## How clicking works

The visual servo homer behind `ClickAgent`:

1. **Slam to corner** — many `(-20,-20)` mouse moves; cursor is now at top-left of screen.
2. **Detect cursor** —
   - HSV thresholding for a saturated red `redglass`-style cursor
     (motion-verified to reject static red UI elements like the Reddit logo).
   - Falls back to **oscillation-variance**: jiggle 6 short HID bursts,
     pick the pixel cluster with highest std-dev across captured frames.
     Robust on any default cursor.
3. **Locate target** — cascade:
   - Tesseract OCR (multi-pass scales / PSMs / both polarities for dark mode), restricted to the user's quoted token when present.
   - Scene-map (multimodal) + ShowUI grounding of the matched label.
   - ShowUI on focused crops (sidebar, footer).
4. **Visual servo loop** — proportional HID move + ROI-prior frame diff to track the cursor; ratio learned online with floor/ceil clamps. First iteration is seeded by a tiny open-loop pointer-acceleration model (see "Open-loop pointer-accel seed" below) so step 1 lands within ~2 px of target on Ubuntu, before the closed-loop fine-tuning kicks in.
5. **Click gate** — geometric: cursor within ~1.2% of aim point for 2 consecutive frames.
6. **Click retry diamond** — first click often overshoots by ~1%; if the post-click oracle says nothing changed, nudge in 4 directions and re-verify.
7. **Post-click navigation oracle** — capture ~2.5s after click, OCR the URL bar / page header, look for the target's distinguishing keywords.

### Open-loop pointer-accel seed (Ubuntu)

Ubuntu's libinput "adaptive" pointer-acceleration profile is
non-linear and velocity-dependent, so a single multiplicative ratio
(`pct_per_hid`) is wrong almost everywhere — the closed-loop homer
compensates by iterating, which works but costs camera frames.

`src/terminaleyes/commander/pointer_accel.py` loads a tiny 2-layer
MLP (`data/ml/checkpoints/pointer_accel-v2/`, ~20 KB) that maps
`(hid_dx, hid_dy, cursor_x_pct, cursor_y_pct)` → observed cursor
delta under the acceleration curve. The homer inverts it with
Newton's method to get an open-loop HID seed for iteration 1; the
closed-loop ratio still owns iterations 2+. Median seed error on
held-out trajectories is ~2 px (90th pct ~140 px), well inside the
~12 px click gate.

Reproduce:

```bash
# Webcam pointed at the target; commander up on http://127.0.0.1:8765.
scripts/collect_pointer_accel.sh --grid 6        # 6×4 click grid → 24 probes
scripts/build_pointer_accel_dataset.py           # → data/ml/pointer_accel/{train,val,test}.jsonl
scripts/train_pointer_accel.py --output data/ml/checkpoints/pointer_accel-v3
```

The checked-in v2 model is Ubuntu-specific (libinput adaptive curve).
macOS uses different pointer accel — retrain for that target if you
want the seed to help there too. If the checkpoint is missing the
homer silently falls back to closed-loop-only behaviour.

## How login works

`LoginAgent`:

1. **Wake** — mouse jiggle + Down arrow + click; dismisses GDM clock overlay.
2. **Polled visual verification** — `VerifyAgent` asks the multimodal model "does this LOOK like a login/password screen?" using **visual cues only** (centred input, hidden-character dots, avatar, clock, dark blurred background) — never relies on the literal word "password" appearing. Default 6 polls × 1.0s with mouse / arrow nudges between.
3. **Type password** — `keyboard.send_text(pw, secret=True)` so the dev-side log records only `length=N`. Pi side already only logs length.
4. **Submit** — Enter (skippable with `--no-submit`).

Password sources, in priority order:
- `--vault NAME` — read from local AES-GCM vault
- `--password-file PATH` — path visible in `ps`, contents not
- `--password-env VAR` — variable name visible, value not
- Interactive `getpass.getpass()` (default)

The password is **never** a positional CLI argument.

## Session output (every captured frame is saved)

Every screenshot the agents take is persisted to a single per-invocation
directory so a run can be replayed visually after the fact. Resolution
order:

1. `--output-dir PATH` CLI flag
2. `TERMINALEYES_OUTPUT_DIR` env var (loadable from `.env`)
3. `~/.local/share/terminaleyes/runs/`

Filenames: `NNNN_HHMMSS_<agent_label>.png`, sequentially numbered so an
`ls` lists captures in the order they were taken.

```
$ ls ~/.local/share/terminaleyes/runs/2026-05-09_17-43-30/
0001_174330_navigate_browser_check.png
0002_174331_homer_capture.png
0003_174333_navigate_postflight_full.png
0004_174333_navigate_postflight_urlbar.png
homer/
└── 174337_vs/
    ├── step_01.png
    ├── step_02.png
    └── ...
```

The Command Center web UI watches this directory and streams frames +
logs to the browser; see the next section.

## Command Center (web UI)

```bash
terminaleyes commandcenter             # http://0.0.0.0:8765 (LAN-reachable)
terminaleyes cc --port 8888
```

A FastAPI app + mobile-first SPA exposing the agent layer over HTTP/SSE.

- `GET /` — the SPA
- `POST /api/run` — start a `ControllerAgent` intent (one at a time)
- `GET /api/runs[/{id}/logs]` — recent runs + per-run SSE log stream
- `GET /api/frames[/latest|/{id}]` — newest-first frame index + bytes + long-poll
- `GET /api/state` — `{busy, latest_id, frame_count, active_run}`

Each `POST /api/run` builds a fresh `AgentContext` with `output_dir =
<watch_dir>/<run_id>/` so frames in the UI cleanly map to runner
records (`FrameMeta.run_id == RunRecord.run_id`). The webcam is held
only during a run, exactly matching `terminaleyes do`.

Manual mouse actions from the UI (click/move/scroll) and the
on-screen ⟳ refresh button use a **before-and-after** capture loop:
save one frame immediately after the HID event, then poll the camera
every `TERMINALEYES_CC_POLL_INTERVAL_S` (default 1.5s) and stop once
the screen has been *region-scale stable* for two consecutive polls
or the `TERMINALEYES_CC_MAX_WAIT_S` budget (default 15s) is up. A
single "final" frame is then saved if-and-only-if it differs from
the initial one. Intermediary frames are dropped so the UI shows
before-and-after only.

Stability is measured by downsampling each frame to
`TERMINALEYES_CC_DOWNSAMPLE`×`…` (default 64×64) grayscale, abs-diffing,
and counting cells whose delta exceeds `TERMINALEYES_CC_CELL_DELTA`
(default 16/255). If the fraction of changed cells is below
`TERMINALEYES_CC_CHANGE_FRACTION` (default 0.5%) the poll counts as
stable. This is robust to the cursor moving a few pixels and to
webcam shimmer (both invisible after downsample) but trips on any
popup / menu / page load / focus-highlight that paints a region.

## Vault

```bash
# Store a secret (value via getpass; passphrase via getpass too)
terminaleyes vault add github

# Use it in a login flow
terminaleyes login --vault github

# Manage
terminaleyes vault list           # entry names only
terminaleyes vault status         # backend + path + mode
terminaleyes vault remove github
```

Format: AES-256-GCM at `~/.config/terminaleyes/vault.enc`, mode
`0600`. Scrypt KDF (`N=2^15, r=8, p=1`). Master passphrase via
`getpass` or `TERMINALEYES_VAULT_PASSPHRASE` env var.

## Models

- **`nvidia/nemotron-3-nano-omni`** on LM Studio (port 1234) —
  default multimodal: scene-map enumeration, login-screen verifier,
  post-click navigation oracle, LLM-planner fallback. Override with
  `TERMINALEYES_COMMANDER__LMSTUDIO_MODEL`.
- **`ShowUI-2B`** on llama.cpp (port 1235) — fast UI grounding (~0.1s/query). Used as fallback when OCR misses.
- **tesseract** (system binary) — primary text-target locator AND
  post-click URL-bar oracle. Cheapest and most reliable signal when
  the target is named text.

```bash
# ShowUI llama.cpp server
brew install llama.cpp
llama-server \
  --hf-repo localattention/ShowUI-2B-Q4_K_M-GGUF \
  --hf-file showui-2b-q4_k_m.gguf \
  --mmproj <mmproj-Qwen2-VL-2B-Instruct-f16.gguf> \
  -c 4096 --port 1235
# Note: do NOT use --image-min-tokens — it breaks ShowUI output.

# Tesseract (OCR backend)
brew install tesseract
pip install pytesseract
```

## Cursor on the target machine

Detection works on any default cursor via oscillation-variance, but a
high-contrast cursor lets HSV detection take over (faster, no
jiggle).

- **Ubuntu / GNOME**:
  ```bash
  sudo apt install -y xcursor-themes
  gsettings set org.gnome.desktop.interface cursor-theme 'redglass'
  gsettings set org.gnome.desktop.interface cursor-size 96
  ```
  Log out / open a new app for the theme to apply.
- **macOS**: System Settings → Accessibility → Display → Pointer
  (set saturated colours, max size).

## Calibrating the homer for your setup

terminaleyes ships **one pre-trained homer setup**:
`pointer_accel-v5` + `longjump-v2`, calibrated against an Ubuntu
target running the `redglass` cursor at size 96 driven over BT HID
from a Pi. If your rig matches that, you're done — the homer
converges in 2–5 closed-loop iterations out of the box. If you're
targeting macOS, Windows, a different webcam position, or a
different cursor theme, the shipped models are an *approximation*
that the closed-loop servo still works through (just with more
iterations); for tight, fast clicks on your own setup, follow the
runbook below to retrain.

**Privacy note.** The training rows are pure motion telemetry —
`(hid_dx, hid_dy, measured_dx_pct, measured_dy_pct, cursor_x_pct,
cursor_y_pct)`. No screenshots, no keystrokes, no document content,
no app names, no anything tied to what was on the screen. The model
learns "send X HID → cursor moves Y pixels on screen" — only the
target OS's pointer-acceleration curve. The webcam frames captured
during collection are stored locally in
`~/.local/share/terminaleyes/runs/` (gitignored, never uploaded)
and the training scripts ONLY consume the numerical
`history.jsonl`, never the PNGs. Inspect any
`data/ml/{pointer_accel,longjump}/*.jsonl` to verify — they're
plain JSON lines, six numbers per row.

### From-scratch training (any target OS)

Prerequisites: hardware + Pi BT-HID setup per the Pi section
below; Command Center reachable at `http://0.0.0.0:8765`; the
target's pointer made high-contrast (see "Cursor on the target
machine" — Ubuntu redglass / macOS Accessibility coloured pointer
/ Windows mouse-pointer colour). The collection clicks at a grid
of pixel positions, so first park the target on a *quiet* screen
— a plain desktop or a maximised text editor. Anything with
clickable toolbars (LibreOffice menubar, browser tabs) will fire
their actions during collection and pollute the screen state.

```bash
# 1. Verify the cursor is visible in the webcam frame.
curl -sX POST http://127.0.0.1:8765/api/snapshot
# Open the saved PNG; you should see a clear red blob on screen.

# 2. Collect ~7 min of probes. The script clicks an 8×6 grid
#    (48 positions), and each click_at runs slam → detect →
#    long-jump chain → closed-loop refinement. Every step is
#    logged to ~/.local/share/terminaleyes/runs/<id>/homer/.../history.jsonl.
date +%s > /tmp/train_start.txt
scripts/collect_pointer_accel.sh --grid 8

# 3. Build the per-step (pointer-accel) dataset. HSV-only keeps only
#    the rows where cursor detection was pixel-accurate; --since
#    discards trajectories from before this collection.
scripts/build_pointer_accel_dataset.py \
    --hsv-only --since "$(cat /tmp/train_start.txt)"

# 4. Train the per-step model.
scripts/train_pointer_accel.py \
    --output data/ml/checkpoints/pointer_accel-v6

# 5. Build the per-trajectory (long-jump) dataset from the SAME
#    collection run.
scripts/build_longjump_dataset.py --since "$(cat /tmp/train_start.txt)"

# 6. Train the long-jump model.
scripts/train_longjump.py \
    --output data/ml/checkpoints/longjump-v3

# 7. Wire the new checkpoints in: prepend their paths to
#    _POINTER_ACCEL_CHECKPOINT_CANDIDATES and
#    _LONGJUMP_CHECKPOINT_CANDIDATES near the top of
#    src/terminaleyes/commander/visual_servo_homer.py.

# 8. Restart Command Center and smoke-test:
terminaleyes commandcenter
# In another shell:
curl -sX POST http://127.0.0.1:8765/api/mouse/click_at \
    -H 'Content-Type: application/json' \
    -d '{"x_pct":0.5,"y_pct":0.5,"button":"left"}'
# Should land in 2-5 closed-loop steps, geometric_confirm.
```

**Per-OS notes:**
- **Ubuntu (libinput-adaptive):** what the shipped models target.
  Redglass theme at size 96 is the easiest detection setup.
- **macOS:** different pointer-accel curve and no redglass theme.
  Crank the Accessibility pointer outline + fill to saturated red
  at max size before collecting. The pointer shape isn't an
  asymmetric arrow, so the `arrow-shape` filter in
  `find_cursor_hsv_motion_directed` (aspect 1.2–2.5, solidity ≥
  0.45) may need loosening for your specific pointer. If the
  homer's cross-check log says "no motion-diff red blob near osc
  result", the shape filter is too strict — sweep the bounds wider
  in `cursor_finder.py` and re-collect.
- **Windows:** Settings → Mouse Pointer → Custom → pick a high-
  contrast colour and max size. Same caveats as macOS for shape.
- The collection grid clicks at pixel positions on the screen. On
  any OS, those positions will activate whatever UI happens to be
  there. Use a plain background.
- Always wait for `--since` to filter out pre-recalibration runs
  when retraining; mixing old and new pointer-accel responses
  produces a model that fits neither distribution.

How much data is enough? With `--grid 8` (48 probes) you'll get
~30–45 useful trajectories. The shipped models trained on this
amount; numbers improve with `--grid 10` (75 probes) or two
back-to-back grid runs. Held-out HID error of `median 5 px / p90
10 px` is the target.

## Webcam vs capture card

The webcam path works; the homer compensates for perspective, glare,
bezel, and small-text OCR limits with a fallback chain. A USB capture
card (HDMI → UVC) is a strict upgrade: image-px = screen-px, no
bezel/glare, OCR reads any on-screen text. Most CV / perspective
compensation in the homer becomes redundant. Same `cv2.VideoCapture`
interface — typically a config-only swap (different `device_index`).

## Raspberry Pi remote keyboard

```
[Dev Mac]  --USB ECM Ethernet-->  [Pi Zero 2 W]  --Bluetooth HID-->  [Target]
 10.0.0.1                            10.0.0.2                       keyboard + mouse
```

- USB ECM Ethernet for the Pi REST API (no WiFi needed)
- Bluetooth HID to drive the target — the Pi's BCM43436s shares one
  radio between WiFi and BT, so freeing WiFi (ECM mode) gives BT a
  stable channel
- USB HID gadget mode (`/dev/hidg0`, `/dev/hidg1`) is a viable
  fallback when BT can't be used

### Pi REST API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/health` | Server + connection status |
| POST   | `/bt/keystroke` | BT keyboard key `{"key": "Enter"}` |
| POST   | `/bt/key-combo` | BT keyboard combo `{"modifiers": ["ctrl"], "key": "c"}` |
| POST   | `/bt/text` | BT keyboard text `{"text": "hello"}` |
| POST   | `/bt/mouse/move` | BT mouse move `{"x": 10, "y": -5}` |
| POST   | `/bt/mouse/click` | BT mouse click `{"button": "left"}` |
| POST   | `/bt/mouse/scroll` | BT mouse scroll `{"amount": -3}` |

USB HID variants (`/keystroke`, `/text`, `/key-combo`, `/mouse/*`)
exist when using `hid` or `all` gadget mode.

### Quick start (Pi)

```bash
# On the Pi (via SSH over USB ECM at 10.0.0.2):
sudo bash scripts/setup_usb_gadget.sh ecm   # USB Ethernet gadget
sudo bash scripts/setup_bt_hid.sh           # one-time BT HID config
sudo bash scripts/radio_mode.sh bt          # switch to Bluetooth mode
sudo systemctl start terminaleyes-pi        # start REST API

# On dev Mac:
curl http://10.0.0.2:8080/health

# After pairing target Mac/Ubuntu via Bluetooth Settings:
curl -X POST -H 'Content-Type: application/json' \
  -d '{"text":"hello from pi"}' http://10.0.0.2:8080/bt/text
```

CLAUDE.md has the full pairing checklist, debugging commands, and the
hard-won lessons (band steering, persistent NetworkManager
connections, BlueZ `--noplugin=input`, etc.).

### Keeping the target BT connection alive

Every `systemctl restart terminaleyes-pi` closes the L2CAP sockets,
so the target sees the Pi HID device disappear. Bluez's autoconnect
is unreliable — without help, you'd manually toggle Bluetooth on
the target each time.

[`scripts/target_bt_reconnect.sh`](./scripts/target_bt_reconnect.md)
runs on the **target machine** and keeps the connection alive. It
polls every 5 minutes (configurable), and if the Pi's HID device is
paired-but-disconnected, runs `bluetoothctl connect`. If pairing
itself has been lost, it scans + pairs + trusts + connects
automatically (relies on the Pi's auto-accept agent —
[`scripts/bt-agent.py`](./scripts/bt-agent.py)).

**Run it once-off (foreground):**

```bash
# On the target — the repo's already cloned here.
./scripts/target_bt_reconnect.sh --probe   # see what bluez reports
./scripts/target_bt_reconnect.sh --pair    # one-shot scan + pair + connect
./scripts/target_bt_reconnect.sh           # forever loop in foreground
```

**Important**: run it OUTSIDE the foreground terminal the controller
types into. If the script logs into the same terminal where
`terminaleyes do` directs keystrokes, the script's `[timestamp]`
log lines interleave with command output and the final-state
verifier can't isolate one from the other.

**Detached run (tmux):**

```bash
tmux new-session -d -s btka './scripts/target_bt_reconnect.sh'
# attach later: tmux attach -t btka      detach: Ctrl+B then d
```

**Systemd-user service** (survives reboot, logs to journal — preferred):

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/bt-keepalive.service <<'UNIT'
[Unit]
Description=Keep devmouse BT connection alive
After=bluetooth.service

[Service]
ExecStart=%h/terminaleyes/scripts/target_bt_reconnect.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now bt-keepalive
# tail logs:
journalctl --user -fu bt-keepalive
# Optional: keep running after logout
sudo loginctl enable-linger $USER
```

Full reference (env knobs, modes, troubleshooting):
[`scripts/target_bt_reconnect.md`](./scripts/target_bt_reconnect.md).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
brew install tesseract
pip install pytesseract
```

## Configuration

Most defaults are sensible. Common overrides via environment variables:

```bash
# Model used by the agent layer (multimodal calls)
export TERMINALEYES_COMMANDER__LMSTUDIO_MODEL='nvidia/nemotron-3-nano-omni'

# Pi base URL (defaults to http://10.0.0.2:8080)
export TERMINALEYES_COMMANDER__PI_BASE_URL='http://10.0.0.2:8080'

# Vault master passphrase (only set this for unattended scripting)
export TERMINALEYES_VAULT_PASSPHRASE='...'
```

For the older terminal-display loop (`terminaleyes run`) configuration
lives in `config/terminaleyes.yaml`; see CLAUDE.md.

## Architecture overview

```
src/terminaleyes/
├── agents/             # Tiered agent layer (see AGENTS.md)
├── commandcenter/      # Web UI + REST/SSE backend (FastAPI + SPA)
├── commander/          # Implementation modules: visual servo homer, OCR, cursor finder, scene-map
├── capture/            # Webcam / capture-card (cv2.VideoCapture wrapper)
├── interpreter/        # MLLM provider clients (OpenAI-compatible)
├── keyboard/           # Abstract keyboard + HTTP/USB backends
├── mouse/              # Abstract mouse + HTTP backend
├── raspi/              # Pi-side: HID codes, BT HID, REST server
├── endpoint/           # Local dev terminal display
├── agent/              # Older goal-driven loop (terminaleyes run)
├── watcher/            # Passive screen observer (terminaleyes watch)
├── config/             # Pydantic settings from YAML + env
└── cli.py              # All subcommand wiring
```

## Other commands

```bash
terminaleyes-pi                       # start Pi REST endpoint
terminaleyes capture-test             # save one webcam frame
terminaleyes endpoint                 # local dev terminal display
terminaleyes watch                    # passive screen observer
terminaleyes run --goal "..."         # older goal-driven agent loop
terminaleyes validate                 # MLLM-vs-actual-screen check
```

## License

MIT
