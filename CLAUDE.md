# CLAUDE.md

## Project

terminaleyes — vision-based agentic terminal controller. Webcam captures a terminal, MLLM interprets the screen, agent decides actions, keyboard output types them.

## Raspberry Pi Keyboard Architecture

The Pi Zero acts as a USB HID keyboard gadget. Two deployment modes:

### Mode 1: Agent on separate machine (primary)
```
[Camera+Agent Machine] --HTTP--> [Pi Zero REST API] --USB HID--> [Target Machine]
```
- Agent machine runs capture/MLLM/agent loop, sends commands via `HttpKeyboardOutput`
- Pi runs `terminaleyes-pi` (raspi/server.py) listening on port 8080
- Pi translates HTTP requests to HID reports on /dev/hidg0

### Mode 2: Agent directly on Pi
```
[Pi Zero] --USB HID--> [Target Machine]
```
- Agent runs on Pi itself, uses `UsbHidKeyboardOutput` (keyboard/usb_hid_backend.py)
- Writes directly to /dev/hidg0, no HTTP layer

## Key directories

- `src/terminaleyes/raspi/` — Pi-specific: HID codes, HID writer, REST server
- `src/terminaleyes/keyboard/` — Abstract keyboard interface + backends (HTTP, USB HID)
- `src/terminaleyes/endpoint/` — Local dev endpoint (shell + pygame display)
- `scripts/setup_usb_gadget.sh` — One-time Pi USB gadget setup (run with sudo)

## Commands

```bash
pip install -e ".[dev]"                    # install
python -m pytest tests/ -v                 # run all tests
python -m pytest tests/unit/test_raspi/ -v # run raspi tests only
terminaleyes-pi                            # start Pi REST API server
```

## Pi setup sequence (when hardware arrives)

1. Flash Raspberry Pi OS Lite to SD card
2. Enable USB OTG: add `dtoverlay=dwc2` to /boot/config.txt, add `dwc2` and `libcomposite` to /etc/modules
3. Run `sudo bash scripts/setup_usb_gadget.sh`
4. Install terminaleyes: `pip install -e ".[rpi]"`
5. Run `terminaleyes-pi` to start the REST API
6. From agent machine: configure `keyboard.http_base_url` to point to Pi's IP

## Pending (blocked on hardware)

- Integration tests for HID writer with real /dev/hidg0
- Integration tests for REST API end-to-end with target machine
- Boot-time auto-start (systemd service)
- Network configuration (static IP / mDNS)
