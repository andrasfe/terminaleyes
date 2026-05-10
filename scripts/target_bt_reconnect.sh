#!/usr/bin/env bash
# target_bt_reconnect.sh — keep the target's BT HID connection alive.
#
# Runs on the TARGET machine (the Ubuntu/Kali box the webcam watches).
# Polls every INTERVAL seconds; when bluetoothctl reports the Pi's BT
# HID device as paired-but-not-connected, runs `bluetoothctl connect`.
# Already-connected? No-op. Never-paired? Logs a warning and waits.
#
# Designed to run forever in a tmux pane or as a systemd --user unit.
# No daemonisation — keep it in the foreground so its output is
# inspectable.
#
# Configuration via env:
#   PI_BT_NAME — device name as it appears in `bluetoothctl devices`.
#                Default: TerminalEyes HID.
#   PI_BT_MAC  — explicit MAC (skips the by-name lookup).
#   INTERVAL   — seconds between checks. Default: 300 (5 min).
#   LOG_FILE   — optional path to also append output to.
#
# Usage:
#   ./target_bt_reconnect.sh
#   PI_BT_MAC=AA:BB:CC:DD:EE:FF INTERVAL=60 ./target_bt_reconnect.sh
#
# To run forever after reboot, drop into ~/.config/systemd/user/
# as a unit named e.g. bt-keepalive.service with:
#   [Service]
#   ExecStart=/home/<user>/bin/target_bt_reconnect.sh
#   Restart=always
#   [Install]
#   WantedBy=default.target
# then `systemctl --user enable --now bt-keepalive`.

set -u

PI_BT_NAME="${PI_BT_NAME:-TerminalEyes HID}"
INTERVAL="${INTERVAL:-300}"
LOG_FILE="${LOG_FILE:-}"

log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    [ -n "$LOG_FILE" ] && echo "$msg" >> "$LOG_FILE"
}

# Resolve the Pi's BT MAC. ``bluetoothctl devices Paired`` was added
# in bluez 5.65; older versions need plain ``bluetoothctl devices``
# plus a separate Paired check. We try the modern form first and fall
# back gracefully.
find_mac() {
    local out
    out=$(bluetoothctl devices Paired 2>/dev/null) \
        || out=$(bluetoothctl devices 2>/dev/null) \
        || return 1
    awk -v name="$PI_BT_NAME" '
        # Each line: "Device AA:BB:CC:DD:EE:FF <Name>"
        index($0, name) > 0 { print $2; exit }
    ' <<< "$out"
}

is_connected() {
    local mac="$1"
    bluetoothctl info "$mac" 2>/dev/null \
        | grep -q "Connected: yes"
}

attempt_connect() {
    local mac="$1"
    log "Pi BT HID ($mac) disconnected — attempting reconnect..."
    # Trust + Power on first; if the adapter is soft-blocked or the
    # device isn't trusted, a bare `connect` silently fails.
    bluetoothctl power on >/dev/null 2>&1
    bluetoothctl trust "$mac" >/dev/null 2>&1
    local result
    result=$(timeout 20 bluetoothctl connect "$mac" 2>&1 || true)
    if echo "$result" | grep -q "Connection successful"; then
        log "  ✓ connected"
    elif echo "$result" | grep -q "br-connection-already-connected"; then
        log "  ✓ already connected (no-op)"
    else
        log "  ✗ connect failed: $(echo "$result" | tail -1)"
    fi
}

main_loop() {
    local mac="${PI_BT_MAC:-}"
    if [ -z "$mac" ]; then
        mac=$(find_mac || true)
    fi
    if [ -z "$mac" ]; then
        log "WARN: device $PI_BT_NAME not paired — waiting and retrying"
    else
        log "monitoring $PI_BT_NAME ($mac), every ${INTERVAL}s"
    fi

    while true; do
        # Re-discover the MAC each loop so a freshly-paired device
        # is picked up without needing to restart the script.
        if [ -z "$mac" ]; then
            mac=$(find_mac || true)
            if [ -n "$mac" ]; then
                log "discovered $PI_BT_NAME ($mac)"
            fi
        fi

        if [ -n "$mac" ]; then
            if is_connected "$mac"; then
                : # already connected — nothing to do
            else
                attempt_connect "$mac"
            fi
        fi

        sleep "$INTERVAL"
    done
}

# Sanity check on dependencies.
command -v bluetoothctl >/dev/null 2>&1 || {
    echo "ERROR: bluetoothctl not found. Install bluez:"
    echo "  sudo apt install -y bluez"
    exit 1
}

main_loop
