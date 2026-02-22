# CLAUDE.md

## Project

terminaleyes — vision-based agentic terminal controller. Webcam captures a terminal, MLLM interprets the screen, agent decides actions, keyboard output types them.

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

- `src/terminaleyes/raspi/` — Pi-specific: HID codes, HID writer, REST server
- `src/terminaleyes/keyboard/` — Abstract keyboard interface + backends (HTTP, USB HID)
- `src/terminaleyes/endpoint/` — Local dev endpoint (shell + pygame display)
- `scripts/setup_usb_gadget.sh` — Pi USB gadget setup: `hid`, `ecm`, or `all` mode (run with sudo)
- `scripts/setup_bt_hid.sh` — One-time BT HID setup: bluetoothd override, adapter config, agent install
- `scripts/bt-agent.py` — Python D-Bus pairing agent (auto-accepts, runs as systemd service)
- `scripts/radio_mode.sh` — Switch between WiFi and BT modes (persists across reboot)
- `scripts/deploy_pi.sh` — Full deploy: rsync, pip install, test endpoints, install services

## Commands

```bash
pip install -e ".[dev]"                    # install
python -m pytest tests/ -v                 # run all tests
python -m pytest tests/unit/test_raspi/ -v # run raspi tests only
terminaleyes-pi                            # start Pi REST API server
```

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
   sudo sdptool browse local | grep -A5 "TerminalEyes HID"
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
15. On target Mac: Bluetooth Settings → Connect to "TerminalEyes HID" (or "keyboarder")
16. Dismiss Keyboard Setup Assistant if it appears
17. Test: `curl -X POST -H 'Content-Type: application/json' -d '{"text":"hello"}' http://10.0.0.2:8080/bt/text`

## Pending

- Integration tests for HID writer with real /dev/hidg0
- Integration tests for REST API end-to-end with target machine
- Make bt-agent.service restart-proof (currently needs manual restart if bluetooth restarts)
- Pi 4 migration: dual-band WiFi + separate BT chip (no radio mode switching needed)
