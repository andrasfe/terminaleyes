#!/usr/bin/env bash
# setup_bt_hid.sh — Configure Raspberry Pi as a Bluetooth HID keyboard + mouse.
#
# Installs BlueZ, configures the adapter for HID, and prepares the Pi
# to act as a Bluetooth combo device (keyboard + mouse) that can be
# paired with a host.
#
# CRITICAL: Disables BlueZ's `input` plugin so our L2CAP sockets on
# PSM 17/19 aren't stolen.  This is required by ALL BT HID emulator
# projects (PiKVM, Bluetooth_HID, Pi-Bluetooth-Keyboard, etc.).
#
# Prerequisites:
#   - Raspberry Pi with built-in Bluetooth (Pi Zero 2 W, Pi 3/4/5)
#   - terminaleyes installed with: pip install -e ".[rpi]"
#
# Usage:
#   sudo bash scripts/setup_bt_hid.sh          # set up
#   sudo bash scripts/setup_bt_hid.sh teardown  # remove configuration

set -euo pipefail

CONF_FILE="/etc/bluetooth/main.conf"
CONF_BACKUP="/etc/bluetooth/main.conf.bak"
OVERRIDE_DIR="/etc/systemd/system/bluetooth.service.d"
OVERRIDE_FILE="$OVERRIDE_DIR/override.conf"
AGENT_SCRIPT="/usr/local/bin/bt-pairing-agent.sh"
AGENT_CONF="/etc/systemd/system/bt-agent.service"

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
teardown() {
    echo "Tearing down Bluetooth HID configuration..."

    # Restore original bluetooth config
    if [ -f "$CONF_BACKUP" ]; then
        cp "$CONF_BACKUP" "$CONF_FILE"
        echo "Restored original $CONF_FILE"
    fi

    # Remove bluetoothd override
    if [ -f "$OVERRIDE_FILE" ]; then
        rm -f "$OVERRIDE_FILE"
        rmdir "$OVERRIDE_DIR" 2>/dev/null || true
        echo "Removed bluetoothd override"
    fi

    # Remove the auto-accept agent service
    if [ -f "$AGENT_CONF" ]; then
        systemctl stop bt-agent 2>/dev/null || true
        systemctl disable bt-agent 2>/dev/null || true
        rm -f "$AGENT_CONF"
        rm -f "$AGENT_SCRIPT"
        systemctl daemon-reload
        echo "Removed bt-agent service"
    fi

    # Restart bluetooth
    systemctl daemon-reload
    systemctl restart bluetooth
    echo "Bluetooth HID configuration removed."
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup() {
    # Ensure we're root
    if [ "$(id -u)" -ne 0 ]; then
        echo "ERROR: Must run as root (sudo)" >&2
        exit 1
    fi

    echo "=== Setting up Bluetooth HID (Keyboard + Mouse) ==="
    echo ""

    # -----------------------------------------------------------------------
    # Step 1: Install dependencies
    # -----------------------------------------------------------------------
    echo "[1/6] Installing Bluetooth dependencies..."

    apt-get update -qq
    apt-get install -y -qq bluez python3-dbus bluetooth pi-bluetooth

    echo "  BlueZ version: $(bluetoothctl --version 2>/dev/null || echo 'unknown')"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 2: Configure bluetoothd with --compat and --noplugin=input
    # -----------------------------------------------------------------------
    echo "[2/6] Configuring bluetoothd (disable input plugin)..."

    # CRITICAL: The BlueZ 'input' plugin binds to PSM 17/19 (HID control
    # and interrupt channels).  If loaded, our application gets EADDRINUSE
    # when trying to bind its own L2CAP sockets.  Every working BT HID
    # project requires disabling it.
    #
    # --compat enables the SDP server socket so sdptool and
    # RegisterProfile SDP records are visible to remote devices.

    mkdir -p "$OVERRIDE_DIR"

    # Find the actual bluetoothd binary path
    BTDAEMON="/usr/libexec/bluetooth/bluetoothd"
    if [ ! -f "$BTDAEMON" ]; then
        BTDAEMON="/usr/lib/bluetooth/bluetoothd"
    fi

    cat > "$OVERRIDE_FILE" <<EOF
[Service]
ExecStart=
ExecStart=$BTDAEMON --compat --noplugin=input
EOF

    echo "  Override: $OVERRIDE_FILE"
    echo "  Flags: --compat --noplugin=input"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 3: Configure Bluetooth adapter
    # -----------------------------------------------------------------------
    echo "[3/6] Configuring Bluetooth adapter..."

    # Backup original config
    if [ -f "$CONF_FILE" ] && [ ! -f "$CONF_BACKUP" ]; then
        cp "$CONF_FILE" "$CONF_BACKUP"
    fi

    # Device class 0x0025C0:
    #   Major class: Peripheral (0x0500)
    #   Minor class: Combo keyboard+pointing (0x00C0)
    #   Service class: none extra
    cat > "$CONF_FILE" <<'BTCONF'
[General]
# TerminalEyes Bluetooth HID Configuration (Keyboard + Mouse)
Name = TerminalEyes HID
Class = 0x0025C0
DiscoverableTimeout = 0
PairableTimeout = 0
Discoverable = true
Pairable = true

[Policy]
AutoEnable=true
BTCONF

    echo "  Set device class to 0x0025C0 (Keyboard + Mouse combo)"
    echo "  Discoverable: always, no timeout"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 4: Set up auto-accept pairing agent
    # -----------------------------------------------------------------------
    echo "[4/6] Setting up pairing agent..."

    # Create a script that runs bluetoothctl in a persistent session
    # with the agent registered.  Plain `bluetoothctl agent NoInputNoOutput`
    # exits immediately — we need to keep the process alive.
    cat > "$AGENT_SCRIPT" <<'AGENTSCRIPT'
#!/usr/bin/env bash
# Persistent Bluetooth pairing agent — auto-accepts all pair requests.
# Uses bluetoothctl in interactive mode to keep the agent alive.
exec bluetoothctl <<EOF
agent NoInputNoOutput
default-agent
EOF
# bluetoothctl stays in interactive mode reading from stdin (which is
# the heredoc).  After the heredoc ends, it keeps running as a
# foreground process until killed.  If it exits, systemd restarts it.
AGENTSCRIPT
    chmod +x "$AGENT_SCRIPT"

    cat > "$AGENT_CONF" <<AGENTSERVICE
[Unit]
Description=Bluetooth Auto-Accept Pairing Agent
After=bluetooth.service
Requires=bluetooth.service

[Service]
Type=simple
ExecStartPre=/bin/sleep 2
ExecStart=$AGENT_SCRIPT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
AGENTSERVICE

    systemctl daemon-reload
    systemctl enable bt-agent
    echo "  Pairing agent: $AGENT_SCRIPT"
    echo "  Auto-accept: NoInputNoOutput (Just Works pairing)"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 5: Apply configuration and power on adapter
    # -----------------------------------------------------------------------
    echo "[5/6] Applying configuration..."

    # Unblock rfkill (common on Pi — BT starts soft-blocked)
    rfkill unblock bluetooth 2>/dev/null || true
    sleep 1

    # Restart bluetooth to pick up new config + override
    systemctl daemon-reload
    systemctl restart bluetooth
    sleep 2

    # Start the pairing agent
    systemctl restart bt-agent 2>/dev/null || true
    sleep 1

    # Power on via hciconfig (more reliable than bluetoothctl for class/name)
    hciconfig hci0 up 2>/dev/null || true
    hciconfig hci0 class 0x0025C0 2>/dev/null || true
    hciconfig hci0 name "TerminalEyes HID" 2>/dev/null || true
    hciconfig hci0 piscan 2>/dev/null || true
    sleep 1

    echo "  Adapter powered on and discoverable"
    echo "  OK"

    # -----------------------------------------------------------------------
    # Step 6: Verify
    # -----------------------------------------------------------------------
    echo "[6/6] Verifying setup..."

    # Check bluetoothd flags
    BTPID=$(pgrep -x bluetoothd 2>/dev/null || echo "")
    if [ -n "$BTPID" ]; then
        BTCMDLINE=$(cat "/proc/$BTPID/cmdline" 2>/dev/null | tr '\0' ' ' || echo "")
        if echo "$BTCMDLINE" | grep -q "noplugin=input"; then
            echo "  bluetoothd: input plugin disabled (correct)"
        else
            echo "  WARNING: input plugin may still be loaded!"
            echo "  cmdline: $BTCMDLINE"
        fi
        if echo "$BTCMDLINE" | grep -q "compat"; then
            echo "  bluetoothd: --compat mode (SDP server enabled)"
        fi
    fi

    # Check agent
    if systemctl is-active --quiet bt-agent 2>/dev/null; then
        echo "  bt-agent: running"
    else
        echo "  WARNING: bt-agent not running"
    fi

    # Check adapter status
    ADAPTER_INFO=$(bluetoothctl show 2>/dev/null || echo "")
    if echo "$ADAPTER_INFO" | grep -q "Powered: yes"; then
        echo "  Adapter: powered on"
    else
        echo "  WARNING: Adapter may not be powered on"
    fi

    if echo "$ADAPTER_INFO" | grep -q "Discoverable: yes"; then
        echo "  Discoverable: yes"
    else
        echo "  WARNING: Adapter may not be discoverable"
    fi

    BT_ADDR=$(echo "$ADAPTER_INFO" | grep "Controller" | awk '{print $2}')
    if [ -n "$BT_ADDR" ]; then
        echo "  Address: $BT_ADDR"
    fi

    echo ""
    echo "=== Bluetooth HID Setup Complete ==="
    echo ""
    echo "The Pi is now discoverable as 'TerminalEyes HID'."
    echo "It presents as a combo keyboard + mouse device."
    echo ""
    echo "Next steps:"
    echo "  1. Start the REST API: sudo systemctl start terminaleyes-pi"
    echo "  2. On the host machine, pair with 'TerminalEyes HID'"
    echo "  3. The REST API accepts Bluetooth commands at:"
    echo ""
    echo "  Keyboard:"
    echo "     POST /bt/keystroke  {\"key\": \"Enter\"}"
    echo "     POST /bt/key-combo  {\"modifiers\": [\"ctrl\"], \"key\": \"c\"}"
    echo "     POST /bt/text       {\"text\": \"hello\"}"
    echo ""
    echo "  Mouse:"
    echo "     POST /bt/mouse/move   {\"x\": 10, \"y\": 0}"
    echo "     POST /bt/mouse/click  {\"button\": \"left\"}"
    echo "     POST /bt/mouse/scroll {\"amount\": -3}"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-setup}" in
    teardown|remove|down)
        teardown
        ;;
    setup|up|"")
        setup
        ;;
    *)
        echo "Usage: $0 [setup|teardown]" >&2
        exit 1
        ;;
esac
