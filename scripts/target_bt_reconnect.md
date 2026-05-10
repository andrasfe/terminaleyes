# `target_bt_reconnect.sh`

Keeps the **target machine's** Bluetooth HID connection to the Pi
alive without manual intervention. Polls every few minutes; when the
Pi's HID device is paired-but-not-connected, runs
`bluetoothctl connect`. Already-connected? No-op. Never-paired?
Warns and keeps watching.

Symptom this fixes: every time the Pi service restarts
(`systemctl restart terminaleyes-pi`) the L2CAP sockets close, the
target sees the BT HID device disappear, and the target's BT stack
doesn't always auto-reconnect. Without this script you have to
manually toggle Bluetooth or click "Connect" each time.

The script runs on the **target** machine (the Ubuntu / Kali / Mac
that the webcam is watching). It does not run on the dev Mac, and
does not run on the Pi.

---

## Install

### Option 1 — one-liner from the running cc

The Command Center serves the script at
`/scripts/target_bt_reconnect.sh`. On the target:

```bash
# Replace 192.168.50.251 with the cc host's LAN IP (it's printed
# when you start `terminaleyes cc` on the dev Mac).
mkdir -p ~/bin
curl -fsSL http://192.168.50.251:8765/scripts/target_bt_reconnect.sh \
     -o ~/bin/target_bt_reconnect.sh
chmod +x ~/bin/target_bt_reconnect.sh
```

If the target shares the USB-ECM segment with the dev Mac, the
`10.0.0.1:8765` URL works too.

### Option 2 — scp from the repo

```bash
scp scripts/target_bt_reconnect.sh user@target:~/bin/
```

---

## Run

### Foreground (logs to stdout)

```bash
~/bin/target_bt_reconnect.sh
```

Each line of output is timestamped. Ctrl+C to stop.

### Background — systemd user service (recommended)

Survives logout / reboot. Installs once, runs forever.

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/bt-keepalive.service <<'UNIT'
[Unit]
Description=Keep TerminalEyes HID BT connection alive
After=bluetooth.service

[Service]
ExecStart=%h/bin/target_bt_reconnect.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now bt-keepalive
```

Tail the log:

```bash
journalctl --user -fu bt-keepalive
```

Stop / disable:

```bash
systemctl --user disable --now bt-keepalive
```

> **Note**: `systemctl --user` services normally stop when you log
> out. To keep them running after logout, enable user lingering:
> `sudo loginctl enable-linger $USER`.

---

## Configuration

Everything is overridable via environment variables — set them in
the `[Service]` block (`Environment="KEY=value"`) or before
launching from a shell.

| var | default | what it does |
|---|---|---|
| `PI_BT_NAME` | `TerminalEyes HID` | Device name as it appears in `bluetoothctl devices`. Used to look up the MAC. |
| `PI_BT_MAC` | *(auto-discovered)* | Explicit Pi MAC, e.g. `AA:BB:CC:DD:EE:FF`. Skips the by-name lookup — set it if the Pi's name changes or to shave a few `bluetoothctl` calls per loop. |
| `INTERVAL` | `300` | Seconds between checks. Drop to `60` for snappier recovery during active development. |
| `LOG_FILE` | *(unset)* | Optional path; output is also appended to this file. Useful when running under tmux. |

Example with all knobs:

```bash
PI_BT_MAC=A1:B2:C3:D4:E5:F6 INTERVAL=60 LOG_FILE=/tmp/bt-keepalive.log \
    ~/bin/target_bt_reconnect.sh
```

---

## How it decides

Each iteration:

1. Look up the Pi's MAC. Uses `bluetoothctl devices Paired` (bluez
   5.65+) and falls back to `bluetoothctl devices` filtered by
   name on older versions. Re-runs each loop so a freshly-paired
   device gets picked up without a script restart.
2. `bluetoothctl info <mac>` → check for `Connected: yes`. If so,
   continue silently.
3. Otherwise: `power on`, `trust <mac>`, `connect <mac>` (with a
   20s timeout). Logs the outcome:
   - `✓ connected` — success
   - `✓ already connected (no-op)` — `br-connection-already-connected`
   - `✗ connect failed: <tail of bluetoothctl output>` — investigate

The script never **pairs** a new device — only reconnects already-
paired ones. Initial pairing is a one-time manual step (Bluetooth
settings → `TerminalEyes HID` → Pair).

---

## Troubleshooting

**`bluetoothctl: command not found`**
Install bluez: `sudo apt install -y bluez` (Debian/Ubuntu/Kali).

**`device $PI_BT_NAME not paired`**
The script is reading bluetoothctl correctly but no paired device
matches the name. On the target: open Bluetooth settings, pair
with `TerminalEyes HID`, then check
`bluetoothctl paired-devices` shows it. The script will pick it
up automatically on the next loop.

**`✗ connect failed: br-connection-create-socket`**
The Pi's L2CAP listener isn't open yet. Common right after
`systemctl restart terminaleyes-pi`. The script will retry on
the next interval — usually within one cycle.

**`✗ connect failed: br-connection-page-timeout`**
Pi adapter isn't discoverable / powered. On the Pi:
`hciconfig hci0` should show `UP RUNNING PSCAN ISCAN`. If not:
`sudo systemctl restart bluetooth && sudo systemctl restart terminaleyes-pi`.

**Reconnects, but the keyboard / mouse still doesn't work**
Pairing is fine but the Pi service isn't serving HID reports.
Hit the Pi's `/health` from the dev Mac:
`curl http://10.0.0.2:8080/health` — `bt_hid_connected: true`
means the Pi's interrupt-channel socket has an active client.
If it stays `false` after a successful `bluetoothctl connect`,
the Pi side needs a restart.
