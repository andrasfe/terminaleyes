# AGENTS.md

terminaleyes ships several user-facing agents that share the same Pi
HID + webcam infrastructure but specialise to different jobs. This
file is the index; deep architecture details live in
[CLAUDE.md](./CLAUDE.md).

## Agents at a glance

| Command                  | Purpose                                                       | Module                          |
|--------------------------|---------------------------------------------------------------|---------------------------------|
| `terminaleyes interact`  | Single-shot or REPL: "click X", "type Y", "press Enter"       | `commander/interactive.py`      |
| `terminaleyes login`     | Wake remote screen + poll-verify it's a login + type password | `commander/login.py`            |
| `terminaleyes run`       | Goal-driven agent loop                                        | `agent/loop.py`                 |
| `terminaleyes watch`     | Passive screen observer (build a session summary)             | `watcher/`                      |
| `terminaleyes-pi`        | REST endpoint on the Pi (HID gateway, BT or USB)              | `raspi/server.py`               |
| MCP server               | Exposes screen control as MCP tools to external clients       | `mcp_server.py`                 |

## Visual servo homer

The click engine for `interact` and `login --click-input`. End-to-end:

```
slam to corner → detect cursor (HSV / oscillation-variance)
              → locate target  (OCR → scene-map+ShowUI → cropped ShowUI)
              → servo (proportional HID + ROI-prior diff, online ratio)
              → click + diamond retry
              → post-click URL-bar oracle
```

Models: `nvidia/nemotron-3-nano-omni` (default, scene-map + verifier),
`ShowUI-2B` (fast UI grounding), `tesseract` (OCR primary + oracle).
Source: `commander/visual_servo_homer.py`. CLAUDE.md "Interactive
Visual Commander" has the full architecture.

## Login agent

`commander/login.py`. A reliable wake-and-type sequence with a polled
multimodal "is this a login/password screen?" check before any
keystroke. Verification is **visual** — centred input, hidden dots,
avatar/clock, dark blurred background — so it works even on label-less
GDM lock screens that don't say the word "password" anywhere.

Password sources: `--password-file`, `--password-env`, or interactive
getpass. Never a positional CLI arg. The keyboard backend redacts the
password from local logs (`secret=True`); the Pi side logs length only.

## MCP server

`mcp_server.py` exposes screen control as MCP tools (`mouse_click`,
`mouse_move`, `screenshot`, `type_text`, `key_combo`, `pi_health`,
etc.) so an external Claude Code or other MCP client can drive the
same Pi pipeline without going through the CLI.

## Pi-side daemon

`raspi/server.py` — FastAPI app exposing `/keystroke`, `/text`,
`/mouse/move`, `/mouse/click`, `/key-combo` over both USB HID (gadget
mode) and BT HID. Same endpoint surface either way; BT routes are
prefixed with `/bt/...`. See CLAUDE.md "Bluetooth HID — Architecture"
for the (extensive) hard-won setup details.

## Where to look first

- New behaviour for clicks → `commander/visual_servo_homer.py`
- Cursor detection knobs → `commander/cursor_finder.py` (HSV thresholds,
  variance percentile)
- OCR target locator → `commander/ocr_finder.py`
- Login wake/verify polling → `commander/login.py`
- Pi BT/USB HID quirks → `raspi/server.py`, `raspi/bt_hid.py`,
  CLAUDE.md "Pi Zero 2 W — Critical lessons learned"
- CLI subcommand wiring → `cli.py`
- Defaults (model name, base URLs) → `config/settings.py`
