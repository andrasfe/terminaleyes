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
terminaleyes do --dry-run "wake the screen and centre the browser"

# Direct agent invocations
terminaleyes login              # wake → visually verify login → type from vault
terminaleyes focus              # centre + maximise the foreground app
terminaleyes vault add NAME     # encrypted local credential store

# Legacy REPL (still works)
terminaleyes interact -m "click X"
```

## Agent layer

```
agents/
├── base.py / context.py   — Agent ABC, Outcome, AgentContext
├── vault.py               — AES-256-GCM credential store (scrypt KDF)
│
├── verify.py              — tier-1: visual yes/no oracle
├── cursor.py              — tier-1: locate cursor (HSV / variance / diff)
├── target.py              — tier-1: locate target by description
│
├── wake.py                — tier-2: wake screen
├── type_text.py           — tier-2: text input (with secret mode)
│
├── focus.py               — tier-3: centre + maximise the foreground app
├── login.py               — tier-3: wake + verify-login + type secret
├── navigate.py            — tier-3: URL-bar typing
├── click.py               — tier-3: find-and-click (was SearchAgent)
│
└── controller.py          — top-level: rules + LLM-planner fallback
```

Each agent is a small testable unit returning a typed
`Outcome { success, reason, data }`. Higher tiers compose lower tiers
through the shared `AgentContext`. The `ControllerAgent` decomposes
free-form English intents into agent sequences:

```
"click the Run button"      → [FocusAgent, ClickAgent(target=...)]
"go to URL"                 → [FocusAgent, NavigateAgent(url=...)]
"login and open reddit.com" → [LoginAgent, FocusAgent, NavigateAgent]
"wake then centre browser"  → [WakeAgent, FocusAgent]   (LLM fallback)
```

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
4. **Visual servo loop** — proportional HID move + ROI-prior frame diff to track the cursor; ratio learned online with floor/ceil clamps.
5. **Click gate** — geometric: cursor within ~1.2% of aim point for 2 consecutive frames.
6. **Click retry diamond** — first click often overshoots by ~1%; if the post-click oracle says nothing changed, nudge in 4 directions and re-verify.
7. **Post-click navigation oracle** — capture ~2.5s after click, OCR the URL bar / page header, look for the target's distinguishing keywords.

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
├── commander/          # Implementation modules: visual servo homer, OCR, cursor finder, scene-map
├── capture/            # Webcam capture (cv2.VideoCapture wrapper)
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
