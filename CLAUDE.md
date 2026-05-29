# CLAUDE.md

## Project

terminaleyes — vision-based agentic terminal controller. A webcam captures the target's screen; classical CV + multimodal LLMs locate the cursor and the click target; HID commands flow over BT (or USB) via a Raspberry Pi to drive the target machine. Target OS may be macOS or Ubuntu/GNOME.

## Agent architecture

Tiered agents under `src/terminaleyes/agents/` (full index in [AGENTS.md](./AGENTS.md)):

- **Tier 1 (atomic)** — `VerifyAgent` (visual yes/no), `CursorAgent` (locate cursor), `TargetAgent` (locate target by description)
- **Tier 2 (actions)** — `WakeAgent` (jiggle/arrow/click), `TypeAgent` (text input with `secret=True`), `ScrollAgent` (mouse-wheel scroll, optional approximate-hover)
- **Tier 3 (workflows)** — `FocusAgent` (centre app), `LoginAgent` (wake+verify+type), `NavigateAgent` (browser-aware URL bar typing with post-OCR oracle), `ClickAgent` (find-and-click via visual servo, with scroll-and-retry)
- **Tier 4 (storage)** — `Vault` (AES-256-GCM file with scrypt KDF)
- **Top level** — `ControllerAgent` decomposes free-form intents into agent sequences via a rule planner; falls back to an LLM planner (validated against the registry) when no rule matches. CLI `terminaleyes do "<intent>"`.

Each agent returns a typed `Outcome { success: bool, reason: str, data: dict }`. Higher-tier agents construct lower-tier agents with the same `AgentContext` so I/O resources (capture, mouse, keyboard, vision client, vault, output dir) are wired once per session.

Defaults that make the controller "safe":
- Click-like intents are prefixed with `FocusAgent` unless `--no-focus`.
- `NavigateAgent` refuses to send keystrokes until `VerifyAgent` confirms a browser is the foreground app — falls back to GNOME activities (Super → type "firefox" → Enter) → Chrome → Chromium → visual icon click → Super+N sweep.
- `LoginAgent` refuses to type when `VerifyAgent` doesn't see a login screen — visual-only judgement, NOT keyword matching for "password".
- `FocusAgent` refuses to act on dark/asleep frames.

## Session output dir

Every captured frame is written to a single per-invocation directory so the run can be replayed visually after the fact. Resolution order:

1. `--output-dir PATH` CLI flag
2. `TERMINALEYES_OUTPUT_DIR` env var (loaded from `.env` by `load_settings()` before agent imports)
3. `~/.local/share/terminaleyes/runs/`

In all three cases a fresh subdirectory is created per invocation. Filename shape: `NNNN_HHMMSS_<agent_label>.png`, sequentially numbered so an `ls` lists captures in capture order. Sources currently wired to `AgentContext.record_frame()`:

- `VerifyAgent` — labelled by caller (`focus_awake_check`, `navigate_browser_check`, etc.)
- `FocusAgent` — `focus_awake_check`, `focus_initial_check`, `focus_recheck_NN`
- `NavigateAgent` — `navigate_browser_check`, `navigate_browser_recheck_NN`, `navigate_postflight_full`, `navigate_postflight_urlbar`
- `VisualServoHomer` (`ClickAgent`) — every `_capture_color`/`_capture_gray` records a `homer_capture` frame; the homer's annotated debug step images go to `<session>/homer/<run-id>/`

The Command Center web UI (`terminaleyes commandcenter`) watches this directory and streams frames + logs over SSE; see "Command Center" below.

## Raspberry Pi Keyboard Architecture

### Primary: USB ECM + Bluetooth HID
```
[Dev Mac / Agent] --USB ECM Ethernet--> [Pi Zero 2 W] --BT HID--> [Target Mac]
     10.0.0.1                              10.0.0.2
```
- Dev Mac runs capture/MLLM/agent loop, sends commands via `HttpKeyboardOutput`
- Pi's USB data port provides ECM (USB Ethernet) to the dev Mac — no WiFi needed for API
- Pi runs `terminaleyes-pi` (raspi/server.py) listening on `0.0.0.0:8080`
- Pi sends keystrokes/mouse to the target Mac over Bluetooth HID (`bt_hid.py`)
- WiFi radio is freed for Bluetooth (BCM43436s shares one radio)

### Fallback: USB HID (direct)
```
[Camera+Agent Machine] --HTTP/WiFi--> [Pi Zero REST API] --USB HID--> [Target Machine]
```
- Pi's USB data port acts as HID keyboard+mouse (gadget mode `hid`)
- Agent communicates over WiFi, Pi writes HID reports to /dev/hidg0

### USB gadget modes (`setup_usb_gadget.sh`)
- `hid` — Keyboard + mouse HID only (default, backward compatible)
- `ecm` — ECM USB Ethernet only (avoids "Keyboard Setup Assistant" on Mac)
- `all` — Both HID + ECM in one composite gadget

## Key directories

- `src/terminaleyes/agents/` — **Tiered agent layer**. See AGENTS.md for the index.
  - `base.py` / `context.py` — Agent ABC, Outcome dataclass, AgentContext (with `output_dir` + `record_frame()`)
  - `vault.py` — AES-256-GCM credential store (scrypt KDF, atomic write, 0600)
  - `verify.py` — tier-1: visual yes/no oracle (JSON-mode + retry, visual-only steering, `record_label` arg)
  - `cursor.py` — tier-1: locate cursor (HSV / oscillation-variance / ROI-diff). Standalone primitive, also embedded in the homer.
  - `target.py` — tier-1: locate target by description (OCR → scene-map+ShowUI → cropped ShowUI)
  - `wake.py` — tier-2: jiggle + Down arrow + click; idempotent
  - `type_text.py` — tier-2: text input; `secret=True` redacts logs; optional Enter
  - `scroll.py` — tier-2: mouse-wheel scroll, optional `hover_at` to land scroll on the right pane
  - `focus.py` — tier-3: awake check → centred check → Super+Up / click+Super+Up corrective combos
  - `login.py` — tier-3: Wake + poll-Verify(login screen) + Type(secret) + Enter
  - `navigate.py` — tier-3: browser-aware URL navigation. Pre-flight Verify(browser?) → activate via GNOME activities (Super+type browser name) → Type URL → post-flight OCR oracle on URL bar with fuzzy match
  - `click.py` — tier-3: find-and-click (wraps `VisualServoHomer`); scroll-and-retry when target not located. `SearchAgent` is an alias.
  - `controller.py` — top-level orchestrator: rule planner + LLM-planner fallback (validated against REGISTRY)
- `src/terminaleyes/commandcenter/` — **Web UI + REST/SSE backend**.
  - `server.py` — FastAPI app. Endpoints: `GET /api/frames[/latest|/{id}]`, `POST /api/run`, `GET /api/runs[/{id}/logs]`, `GET /api/state`, plus the SPA at `/`.
  - `runner.py` — one-at-a-time `ControllerAgent` runner with per-run resource lifecycle (matches `terminaleyes do`).
  - `frame_store.py` — polls the agent layer's output dir, indexes new images, serves bytes. Default watch dir reads from `TERMINALEYES_OUTPUT_DIR` then falls back to the agent default — guaranteed to agree with what the agents write.
  - `factory.py` — `make_default_context_factory(settings, base_dir, bus)`: builds a fresh `AgentContext` per run with `output_dir = base_dir / <run_id>` so frames map cleanly to runner records.
  - `log_bus.py` — pub/sub for log records + redirected stdout/stderr; SSE streams subscribe.
  - `static/` — mobile-first SPA (`index.html`, `app.js`, `styles.css`).
- `src/terminaleyes/commander/` — Implementation modules backing the agents
  - `visual_servo_homer.py` — closed-loop CV homer; the click engine ClickAgent wraps. Persists every step to `<run>/homer/<id>/history.jsonl` for training.
  - `pointer_accel.py` — open-loop pointer-acceleration MLP. v3+ checkpoints are *direct inverse* models — `(measured_dx_pct, measured_dy_pct, cursor) → hid` — so inference is a single forward pass with no Newton iteration. v1/v2 checkpoints are forward models that the runtime Newton-inverts (kept for backwards compat — `config.json:"direction"` selects). Used by the homer as the first-iteration HID seed; falls back silently to closed-loop if no checkpoint is present. Ubuntu-libinput-adaptive specific.
  - `cursor_finder.py` — HSV finder (saturated red `redglass` cursor) + variance fallback
  - `ocr_finder.py` — tesseract wrapper with multi-pass preprocessing
  - `login.py` — backwards-compat shim; routes to `agents.login.LoginAgent`
  - `closed_loop_homer.py` — older static-calibration homer (kept as helper for scene-map + keyword extraction)
  - `interactive.py` — REPL session dispatching to the homer
- `src/terminaleyes/mouse/` — Abstract mouse interface + HTTP backend (BT/USB transport)
- `src/terminaleyes/raspi/` — Pi-specific: HID codes, HID writer, REST server
- `src/terminaleyes/keyboard/` — Abstract keyboard interface + backends (HTTP, USB HID). `send_text(secret=True)` redacts content from local logs (used for password input).
- `src/terminaleyes/endpoint/` — Local dev endpoint (shell + pygame display)
- `scripts/setup_usb_gadget.sh` — Pi USB gadget setup: `hid`, `ecm`, or `all` mode (run with sudo)
- `scripts/setup_bt_hid.sh` — One-time BT HID setup: bluetoothd override, adapter config, agent install
- `scripts/bt-agent.py` — Python D-Bus pairing agent (auto-accepts, runs as systemd service)
- `scripts/radio_mode.sh` — Switch between WiFi and BT modes (persists across reboot)
- `scripts/deploy_pi.sh` — Full deploy: rsync, pip install, test endpoints, install services
- `scripts/collect_pointer_accel.sh` — fire `/api/mouse/click_at` at an N×M grid to collect homer step records for training the open-loop pointer-accel MLP
- `scripts/build_pointer_accel_dataset.py` — walk `<run>/homer/*/history.jsonl` → `data/ml/pointer_accel/{train,val,test}.jsonl` (80/10/10 split by trajectory)
- `scripts/train_pointer_accel.py` — MLX-backed MLP trainer; emits `data/ml/checkpoints/pointer_accel-vN/{weights.npz,config.json}` (load via `commander.pointer_accel.PointerAccelModel`)

## Commands

```bash
pip install -e ".[dev]"                    # install
brew install tesseract                     # OCR backend (system binary)
pip install pytesseract                    # python binding
python -m pytest tests/ -v                 # run all tests
python -m pytest tests/unit/test_raspi/ -v # run raspi tests only
terminaleyes-pi                            # start Pi REST API server

# ── Common flags (all subcommands) ──
terminaleyes --output-dir PATH ...         # override session output dir
TERMINALEYES_OUTPUT_DIR=PATH terminaleyes ...   # same, via env (works in .env)

# ── Controller (top-level) ──
terminaleyes do "click the Run button"
terminaleyes do "go to reddit.com/r/LocalLLaMA"
terminaleyes do "login and open reddit.com" --vault myhost
terminaleyes do "scroll down 6"
terminaleyes do --dry-run "<intent>"       # show plan without executing
terminaleyes do --no-focus "click X"       # skip auto-focus prefix
terminaleyes do --no-llm-fallback "..."    # rules-only; refuses unknown intents

# ── Direct agent invocations (low-level) ──
terminaleyes focus [--platform linux|macos] [--max-attempts N]
terminaleyes login                         # interactive getpass prompt
terminaleyes login --vault NAME            # password from local vault
terminaleyes login --password-file PATH    # path visible in `ps`, contents not
terminaleyes login --password-env VAR      # var name visible, value not
terminaleyes login --click-input           # visually click password field first
terminaleyes login --no-verify             # skip the visual login-screen check
terminaleyes login --verify-attempts 12 --verify-interval 1.5

# ── Vault ──
terminaleyes vault add NAME                # prompt for value via getpass
terminaleyes vault get NAME                # print to stdout (warns if TTY)
terminaleyes vault list                    # entry names only
terminaleyes vault remove NAME
terminaleyes vault status                  # backend + path + mode

# ── Command Center (web UI) ──
terminaleyes commandcenter                 # FastAPI on 0.0.0.0:8765
terminaleyes cc --port 8888                # alias
terminaleyes cc --frames-dir PATH          # override watch dir

# ── Legacy / direct ──
terminaleyes interact                      # REPL routes through the homer
terminaleyes interact -m "click X"         # single-command mode
```

## Interactive Visual Commander

### Models
- **nemotron-3-nano-omni** on LM Studio (port 1234) — default multimodal: scene-map enumeration, login-screen verification, post-click navigation oracle. Override with `TERMINALEYES_COMMANDER__LMSTUDIO_MODEL`.
- **ShowUI-2B** on llama.cpp (port 1235) — fast UI grounding (~0.1s). Used as fallback when OCR doesn't find the target.
- **tesseract** (system binary) — primary text-target locator AND post-click URL-bar oracle. The cheapest and most reliable signal when the target is named text.

### Visual servo cursor homing (`commander/visual_servo_homer.py`)
Replaces the older static-calibration homer entirely. Per run:

1. **Slam to corner** — many `(-20, -20)` mouse moves; cursor is now at top-left of screen.
2. **Detect cursor**:
   - HSV thresholding for a saturated red `redglass`-style cursor (motion-verified to reject static red UI elements like the Reddit logo)
   - Falls back to **oscillation-variance**: jiggle 6 short HID bursts, find the pixel cluster with highest std-dev across captured frames. Robust on any default cursor.
3. **Locate target** (cascade, first hit wins):
   - Tesseract OCR with multi-pass preprocessing (scales 3–5, PSM 6+11, both polarities for dark mode), restricted to user-quoted tokens when present so generic words like "subreddit"/"entry" don't match
   - Scene-map (multimodal) + ShowUI grounding of the matched label
   - ShowUI on focused crops (left sidebar, footer)
4. **Visual servo loop** — proportional HID move from current ratio + ROI-prior frame diff to track cursor. Ratio learned online with floor/ceil clamps so a bad sample can't run away. Hard cap on HID per axis. **First iteration is seeded** by an open-loop pointer-accel MLP (`commander/pointer_accel.py`, checkpoint at `data/ml/checkpoints/pointer_accel-v2/`) which inverts a learned forward model `(hid, cursor_pos) → measured_pct_delta` via Newton's method. Median seed error ~2 px on Ubuntu libinput-adaptive; absence of the checkpoint is harmless (homer logs and falls back to ratio-only seed). Every step (sent HID + measured delta + cursor position) is persisted to `<run>/homer/<id>/history.jsonl` for future retraining.
5. **Click gate** — geometric: visually-detected cursor within `CLICK_TOL_PCT=1.2%` of aim point for 2 consecutive frames. Hotspot offset compensates for centroid-vs-tip on the default arrow.
6. **Click retry pattern** — first click overshoots by ~1% on most cursors; if the post-click oracle says nothing changed, nudge through a small diamond (5 attempts) and re-verify.
7. **Post-click navigation oracle** — captures a frame ~2.5s after click, OCRs the URL bar and page header, looks for the target's distinguishing keywords (quoted text wins over generic descriptors).

### Cursor on the target machine
- Detection works on any default cursor via oscillation-variance.
- Optional: switch to a high-contrast theme so HSV detection takes over (faster, no jiggle).
  - **Ubuntu/GNOME**: `sudo apt install -y xcursor-themes; gsettings set org.gnome.desktop.interface cursor-theme 'redglass'; gsettings set org.gnome.desktop.interface cursor-size 96`. Log out / open a new app for the theme to apply.
  - **macOS**: System Settings → Accessibility → Display → Pointer (set saturated colours, max size).

### Webcam vs capture card
- Webcam works; the homer compensates for perspective, glare, bezel, and small-text OCR limits (with a fallback chain).
- A USB capture card (HDMI→UVC) is a strict upgrade: image-px = screen-px, no bezel/glare, OCR reads any on-screen text. Most CV/perspective compensation in the homer becomes redundant. Same `cv2.VideoCapture` interface — typically a config-only swap (different `device_index`).

### ShowUI llama.cpp server
```bash
brew install llama.cpp
llama-server \
  --hf-repo localattention/ShowUI-2B-Q4_K_M-GGUF \
  --hf-file showui-2b-q4_k_m.gguf \
  --mmproj <mmproj-Qwen2-VL-2B-Instruct-f16.gguf> \
  -c 4096 --port 1235
```
Note: do NOT use `--image-min-tokens` — it breaks ShowUI output.

## Login flow (`commander/login.py`)

End-to-end remote login that **never sees the literal word "password"** to decide whether to type — verification is purely visual.

1. **Wake** — mouse jiggle (×4) + Down arrow + left click. Wakes the monitor and dismisses GDM clock overlays.
2. **Polled visual verification** — captures a frame, asks the multimodal model "does this LOOK like a login/password screen?" (centred input, hidden-character dots, avatar/clock, dark blurred bg — NOT keyword matching). On miss, alternates between mouse jiggle and Down arrow nudges and re-checks. Default 6 polls × 1.0s.
3. **Type password** — `keyboard.send_text(pw, secret=True)` so the dev-side log records only `length=N`. Pi side already only logs length.
4. **Submit** — Enter (skippable with `--no-submit`).

**Password sources** (priority): `--password-file PATH` > `--password-env VAR` > interactive `getpass.getpass()`. **Never** a positional arg (would leak via `ps`).

**`--click-input`** uses the visual homer to click an input field by *visual* description ("the centred text input", "the highlighted input box") — no reliance on the word "password" appearing on screen.

## Command-line gotchas
- BT HID transit between dev Mac and Pi is over USB ECM (10.0.0.0/24 link-local) — fine for the cable, don't pipe credentials over an untrusted network leg.
- Python doesn't zero string memory; the password reference is dropped immediately after submit but a memory dump could still recover it. Treat the dev Mac as trusted.

## Command Center (`commandcenter/`)

A FastAPI app that exposes the agent layer over HTTP/SSE plus a mobile-first SPA. Boot:

```bash
terminaleyes commandcenter                 # http://0.0.0.0:8765
terminaleyes cc --port 8888 --frames-dir /tmp/foo
```

**Wiring at startup:**
- `FrameStore` polls the configured watch dir (`TERMINALEYES_OUTPUT_DIR` or default) every 250 ms, indexes new images, surfaces `FrameMeta { id, ts, run_id, filename }`. The `run_id` matches the runner's record id so the UI correlates frames ↔ runs without ambiguity.
- `LogBus` captures the `terminaleyes` logger AND redirects `stdout`/`stderr` of the active run; SSE subscribers (per-run + global) get `LogEvent { ts, level, source, msg, run_id }`.
- `Runner` is one-at-a-time: each `POST /api/run` builds a fresh `AgentContext` via `make_default_context_factory(settings, base_dir=watch_dir, bus)`, invokes `ControllerAgent.run(intent=...)`, then closes mouse/keyboard/capture. The webcam is held only during a run, exactly matching `terminaleyes do`.
- Per-run output dir = `<watch_dir>/<run_id>/`. `bus.active_run(run_id)` is set before the factory runs, so `bus.current_run_id()` is read in the factory to name the dir. UI's `FrameMeta.run_id == RunRecord.run_id`.
- Manual mouse actions (`/api/mouse/click_at`, `/move`, `/scroll`, `/button`) and `/api/snapshot` use a **before-and-after** capture loop: save an initial frame, then poll every `TERMINALEYES_CC_POLL_INTERVAL_S` (default 1.5s) and stop once the screen has been region-scale stable for two consecutive polls or `TERMINALEYES_CC_MAX_WAIT_S` (default 15s) is exhausted. A single final frame is saved iff it differs from the initial one — intermediary frames are dropped. Stability metric is **fraction of changed cells in a downsampled grayscale grid**: each frame is resized to `TERMINALEYES_CC_DOWNSAMPLE` square (default 64), abs-diffed, and cells whose delta exceeds `TERMINALEYES_CC_CELL_DELTA` (default 16/255) are counted; the poll is "stable" if that fraction is below `TERMINALEYES_CC_CHANGE_FRACTION` (default 0.005 = 0.5%). This is deliberately not pixel-MSE — a moving cursor or a few jittered webcam pixels collapse to <1 cell after downsample, but any region-scale UI change (popup, menu, page load, focus highlight) trips it. Webcam stays open across polls; serialized with `_manual_capture_lock`; refuses while a `ControllerAgent` run holds the device.
- `POST /api/snapshot?dedup=1` (used by the UI's **Active Refresh** checkbox and the typing loop) skips the poll loop entirely: grab one frame, compare against the most recent stored frame, persist only if changed. Uses a tighter threshold than the post-mouse-action stability check — `TERMINALEYES_CC_DEDUP_FRACTION` (default 0.001 = ~4 cells of the 64×64 grid) catches a single typed character; the post-action 0.5% threshold would miss it. The UI fires dedup snapshots in two contexts: (a) every 60s when Active Refresh is armed, (b) every 2s while keystrokes are flowing to the host (debounced 5s after the last keystroke). Idle screens produce zero writes in both cases.

**Endpoints:**
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | SPA |
| GET | `/api/state` | `{busy, latest_id, frame_count, active_run}` |
| GET | `/api/frames[?limit=N&before=ID]` | newest-first index |
| GET | `/api/frames/latest[?wait=1&since=ID]` | one-shot or long-poll |
| GET | `/api/frames/{id}` | image bytes |
| GET | `/api/frames/{id}/neighbours` | `{prev, next}` |
| POST | `/api/run` | start a controller intent (409 if busy) |
| GET | `/api/runs[?limit=N]` | recent run records |
| GET | `/api/runs/{id}` | one record |
| GET | `/api/runs/{id}/logs` | SSE log stream (replays buffer) |
| GET | `/api/logs[?tail=N]` | global SSE log stream |

## Pi setup sequence

1. Flash Raspberry Pi OS Lite (64-bit) with Raspberry Pi Imager
2. In Imager settings: hostname `keyboarder`, user `andras`, WiFi (2.4GHz SSID!), SSH enabled
3. On boot partition, edit `user-data` to add persistent NM connection via `write_files` + `wifi-ensure.service`
4. Boot, SSH in, deploy code: `rsync` then `pip install -e ".[rpi]"`
5. Add `dwc2` and `libcomposite` to `/etc/modules`
6. Add `dtoverlay=dwc2` to `/boot/config.txt`
7. Run `sudo bash scripts/setup_usb_gadget.sh ecm` (or `hid`/`all`)
8. Create NM connection for USB ethernet: `sudo nmcli connection add type ethernet con-name usb-ecm ifname usb0 ipv4.addresses 10.0.0.2/24 ipv4.method manual ipv4.never-default yes`
9. Enable systemd service: `sudo systemctl enable terminaleyes-pi`
10. On dev Mac: configure USB ethernet interface (`en10` or similar) with IP `10.0.0.1/24`
11. Test: `curl http://10.0.0.2:8080/health`

## Pi Zero 2 W — Critical lessons learned

### WiFi setup (MUST follow or WiFi will break)
- **Router band steering MUST be disabled** — Pi Zero 2 W only supports 2.4GHz. ASUS RT-AX3000 Smart Connect steers to 5GHz. Disable it or use a dedicated 2.4GHz SSID.
- **Cloud-init `network-config` puts WiFi in `/run/` (tmpfs)** — connections vanish on reboot. Fix: write a persistent `.nmconnection` file to `/etc/NetworkManager/system-connections/` via cloud-init `write_files`, and disable cloud-init network with `99-disable-network-config.cfg`.
- **`wifi-ensure.service` is required** — NM autoconnect alone is unreliable on Pi Zero 2 W. A systemd service that retries `nmcli connection up` after boot is needed.
- **Never create these files** (they break WiFi permanently):
  - `/etc/network/interfaces.d/wlan0` — makes NetworkManager ignore wlan0
  - `/etc/dhcpcd.exit-hook` — interferes with NetworkManager

### Boot config pitfalls (DO NOT do these)
- `dtoverlay=disable-bt` — **breaks boot** (UART conflict with `enable_uart=1`)
- `systemd.run=` in cmdline.txt — unreliable, caused boot failures
- `systemd.mask=` in cmdline.txt — not a valid kernel parameter

### Bluetooth HID — Architecture (verified working 2026-02-20)
- BCM43436s shares one radio for WiFi and BT Classic
- **`bluetoothctl connect` from Pi will crash WiFi permanently** — never do this while WiFi is active
- With ECM mode, WiFi is no longer needed for API → radio is free for BT HID
- USB BT dongle (Edimax BT-8500) prevents Pi Zero 2 W from booting — don't plug into data port

**How BT HID works (the proven pattern from all Pi BT HID projects):**
1. `RegisterProfile` via D-Bus publishes the SDP record so the host (Mac) discovers HID
2. Raw L2CAP sockets on PSM 17 (control) and PSM 19 (interrupt) handle actual data
3. The Profile1 `NewConnection` callback is NOT used for HID data transfer
4. HID reports are sent on the interrupt channel prefixed with `0xA1` (DATA|INPUT)

**Critical: `--noplugin=input` is MANDATORY on bluetoothd:**
- The BlueZ `input` plugin binds to PSM 17/19 itself — our sockets get `EADDRINUSE`
- Override file: `/etc/systemd/system/bluetooth.service.d/override.conf`
- Content: `ExecStart=` then `ExecStart=/usr/libexec/bluetooth/bluetoothd --compat --noplugin=input`
- `--compat` enables the SDP server socket for remote device discovery
- Without `--noplugin=input`: "Connect" button does nothing, or connection resets immediately

**Pairing agent — must be a Python D-Bus agent (`scripts/bt-agent.py`):**
- `bluetoothctl agent NoInputNoOutput` as a systemd service exits immediately — DO NOT use
- Heredoc/pipe tricks (`{ echo ...; sleep infinity } | bluetoothctl`) are fragile
- The Python agent registers on D-Bus, implements all Agent1 methods, runs GLib main loop
- Must start AFTER `terminaleyes-pi` (which restarts bluetooth via `radio_mode.sh`)
- Service: `bt-agent.service` with `After=bluetooth.service terminaleyes-pi.service`

### Bluetooth HID — Debugging checklist
If BT HID stops working, check in this order:

1. **Is bluetoothd running with correct flags?**
   ```bash
   cat /proc/$(pgrep -x bluetoothd)/cmdline | tr '\0' ' '
   # Must show: --compat --noplugin=input
   ```

2. **Is the SDP record visible?**
   ```bash
   sudo sdptool browse local | grep -A5 "devmouse"
   # Must show: "Human Interface Device" (0x1124)
   ```

3. **Is the pairing agent running?**
   ```bash
   sudo pgrep -fa bt-agent.py
   sudo cat /tmp/bt-agent.log
   # Must show: "Agent registered" and "Waiting for pairing requests"
   ```

4. **Is the adapter discoverable?**
   ```bash
   hciconfig hci0
   # Must show: UP RUNNING PSCAN ISCAN
   bluetoothctl show | grep -E "Powered|Discoverable|Pairable"
   ```

5. **Are L2CAP sockets listening?**
   ```bash
   curl -s http://10.0.0.2:8080/bt/keystroke -X POST -H 'Content-Type: application/json' -d '{"key":"a"}'
   # "No Bluetooth client connected" = sockets OK, no client yet
   # "Bluetooth HID not initialized" (503) = sockets failed to bind
   ```

6. **Is the Mac paired?**
   ```bash
   bluetoothctl devices  # or: bluetoothctl info <MAC>
   ```

7. **Control channel messages?** Check service logs for SET_PROTOCOL handling:
   ```bash
   sudo journalctl -u terminaleyes-pi -f
   # Look for: "Control channel msg: 0x71" and "SET_PROTOCOL: Report mode"
   ```

**Common failure modes:**
- "Connect does nothing" on Mac → `input` plugin is loaded (check step 1)
- "Connection reset by peer" on send → control channel SET_PROTOCOL not handled, or stale pairing
- "Address already in use" on startup → `input` plugin loaded, or previous process didn't clean up (SO_REUSEADDR helps)
- Agent keeps dying → `radio_mode.sh` restarted bluetooth, agent must restart after
- Mac shows "Keyboard Setup Assistant: cannot be identified" → dismiss it, one-time only on first pair
- `bt_hid_connected: false` after pairing → Mac paired but didn't open L2CAP channels; try Forget + re-pair
- **Mac pair spinner times out with ZERO Pi-side log activity** → stale Pi-side bond record blocking it at HCI level. The Pi auto-rejects pair attempts from devices it thinks are already paired, **before bluetoothd or any agent log fires**. Fix: `sudo bluetoothctl devices` to list, then `sudo bluetoothctl remove <MAC>` for each. `setup_bt_hid.sh` step [4.25/6] does this on every install (added 2026-05-29).
- **Mac shows old device name even after rename** → macOS caches the BT device name per MAC address forever. Toggle Mac Bluetooth off/on (menu bar icon) or `sudo killall bluetoothd` to flush the cache. Forget+Repair alone doesn't always do it.
- **`Pairable: no` even though `main.conf` has `Pairable = true`** → BlueZ 5.82 logs `Unknown key Pairable for group General` and silently ignores the line. `Pairable` is NOT a valid main.conf key in this BlueZ version. The runtime path `bluetoothctl pairable on` is the only thing that works; `bt-strip-audio-sdp.sh` re-asserts it on every `bluetooth.service` cycle via `PartOf=bluetooth.service`.
- **Mac auto-routes audio to the Pi after pairing, kills Mac speaker output** → bluetoothd's protocol stack publishes Hands-Free / SIM Access / etc SDP records *underneath* the `--noplugin=` filter. Those flags are plugin-level; the records are baked into the daemon itself. Cannot be turned off from main.conf or noplugin alone. `bt-strip-audio-sdp.sh` walks `sdptool browse local` and deletes each by handle after every `bluetoothd` start. Without it the Pi advertises as a HID *and* audio device and macOS grabs the audio profile silently.
- **Pairing agent registration fails: `Failed to register agent object` / `No agent is registered`** → the bash-shim agent (`exec bluetoothctl <<EOF agent NoInputNoOutput`) doesn't work on BlueZ 5.82 — interactive bluetoothctl exits before its D-Bus session finishes registering the agent. Must use the Python D-Bus agent (`scripts/bt-agent.py`) which stays in a GLib main loop. `setup_bt_hid.sh` step [4/6] installs it (verified working pattern; same approach every working BT HID emulator project converges on).

### USB gadget
- `dtoverlay=dwc2` IS required in config.txt (switches from dwc_otg host-mode to dwc2 gadget-capable driver)
- Also need `dwc2` and `libcomposite` in `/etc/modules`
- `modprobe` lives at `/usr/sbin/modprobe` — service PATH must include it, or use full path
- ECM mode: NetworkManager will take over `usb0` and flush static IPs — use `nmcli connection add` for persistent config (not `ip addr add`)
- Pi 5 has NO USB gadget support (USB-C is power-only, RP1 chip is host-only)
- Pi 4 USB-C is shared power/data — needs alternate power source for gadget mode

## Pi setup sequence — BT HID (after initial setup)

12. Run `sudo bash scripts/setup_bt_hid.sh` (one-time: installs override, configures adapter)
13. Run `sudo bash scripts/radio_mode.sh bt` (switch to BT mode, disables WiFi)
14. Reboot Pi, reconfigure Mac en10 IP if needed
15. On target Mac: Bluetooth Settings → Connect to "devmouse" (or "keyboarder")
16. Dismiss Keyboard Setup Assistant if it appears
17. Test: `curl -X POST -H 'Content-Type: application/json' -d '{"text":"hello"}' http://10.0.0.2:8080/bt/text`

## Pending

- Integration tests for HID writer with real /dev/hidg0
- Integration tests for REST API end-to-end with target machine
- Make bt-agent.service restart-proof (currently needs manual restart if bluetooth restarts)
- Pi 4 migration: dual-band WiFi + separate BT chip (no radio mode switching needed)
- Webcam mirror detection (auto-detect if image is flipped)
- Self-improving HID ratio cache (persist learned `pct_per_hid` per session/screen so future runs start from a better prior)
- Tests for the agent layer (vault round-trip, mock-context controller dry-runs, etc.)
- Vault: optional OS-keychain backend via `keyring` (macOS Keychain / Secret Service / Credential Manager)
- Refactor `VisualServoHomer` internals to use `CursorAgent` + `TargetAgent` directly (currently they wrap the same helpers in parallel)
- Command Center: FrameStore should optionally recurse into `<session>/homer/<run-id>/` so annotated debug step images surface in the UI; OR flatten homer debug output to the session top level
- Event-driven frame notifications (FrameStore polls every 250ms; could subscribe to AgentContext.record_frame() emissions to push frames to subscribers without polling lag)
