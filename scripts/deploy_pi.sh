#!/usr/bin/env bash
# deploy_pi.sh — Deploy terminaleyes to Raspberry Pi, test endpoints, install service.
#
# Reads PI_HOST, PI_USERNAME, PI_PASSWORD from .env in the project root.
#
# Usage:
#   bash scripts/deploy_pi.sh              # full deploy + test
#   bash scripts/deploy_pi.sh --skip-test  # deploy only, skip endpoint tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[SKIP]${NC} $1"; }
info() { echo -e "  ${BLUE}[..]${NC} $1"; }
header() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE" >&2
    echo "Create it with PI_HOST, PI_USERNAME, PI_PASSWORD" >&2
    exit 1
fi

# Source .env (strip leading/trailing whitespace, skip blanks/comments)
_CLEANED_ENV=$(mktemp)
sed 's/^[[:space:]]*//; s/[[:space:]]*$//' "$ENV_FILE" | grep -v '^$' | grep -v '^#' > "$_CLEANED_ENV"
set -a
# shellcheck disable=SC1090
source "$_CLEANED_ENV"
set +a
rm -f "$_CLEANED_ENV"

PI_HOST="${PI_HOST:?PI_HOST not set in .env}"
PI_USERNAME="${PI_USERNAME:?PI_USERNAME not set in .env}"
PI_PASSWORD="${PI_PASSWORD:?PI_PASSWORD not set in .env}"

REMOTE="$PI_USERNAME@$PI_HOST"
REMOTE_DIR="/home/$PI_USERNAME/terminaleyes"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
SKIP_TEST=false

for arg in "$@"; do
    case "$arg" in
        --skip-test) SKIP_TEST=true ;;
    esac
done

# Helper: run command on Pi via SSH
pi_run() {
    ssh $SSH_OPTS "$REMOTE" "$@"
}

# Helper: run command on Pi as root
pi_sudo() {
    ssh $SSH_OPTS "$REMOTE" "echo '$PI_PASSWORD' | sudo -S bash -c '$*' 2>/dev/null"
}

# ---------------------------------------------------------------------------
# Phase 1: SSH Setup
# ---------------------------------------------------------------------------
header "Phase 1: SSH Connectivity"

# Check if sshpass is available for key setup
if ! command -v sshpass &>/dev/null; then
    info "sshpass not found — install with: brew install hudochenkov/sshpass/sshpass"
    info "Assuming SSH keys are already set up..."
else
    # Set up SSH key auth if not already done
    if ! ssh $SSH_OPTS -o BatchMode=yes "$REMOTE" "true" 2>/dev/null; then
        info "Setting up SSH key authentication..."
        # Generate key if needed
        if [ ! -f "$HOME/.ssh/id_ed25519" ] && [ ! -f "$HOME/.ssh/id_rsa" ]; then
            ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" -q
        fi
        sshpass -p "$PI_PASSWORD" ssh-copy-id $SSH_OPTS "$REMOTE" 2>/dev/null
        ok "SSH key copied"
    fi
fi

# Verify connectivity
if pi_run "hostname" &>/dev/null; then
    HOSTNAME=$(pi_run "hostname")
    ok "Connected to $HOSTNAME ($PI_HOST)"
else
    fail "Cannot connect to $PI_HOST"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 2: Deploy Code
# ---------------------------------------------------------------------------
header "Phase 2: Deploy Code"

info "Syncing project files to $REMOTE:$REMOTE_DIR ..."

rsync -az --delete \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='captured_*' \
    --exclude='*.jpg' \
    --exclude='*.png' \
    --exclude='.mypy_cache' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='*.egg-info' \
    -e "ssh $SSH_OPTS" \
    "$PROJECT_DIR/src" \
    "$PROJECT_DIR/scripts" \
    "$PROJECT_DIR/pyproject.toml" \
    "$PROJECT_DIR/README.md" \
    "$PROJECT_DIR/tests" \
    "$REMOTE:$REMOTE_DIR/"

ok "Files synced to $REMOTE_DIR"

# ---------------------------------------------------------------------------
# Phase 3: Install on Pi
# ---------------------------------------------------------------------------
header "Phase 3: Install Dependencies"

info "Creating venv and installing package..."

# Use --system-site-packages so python3-dbus (apt) is accessible
pi_run "cd ~/terminaleyes && ( test -d .venv || python3 -m venv --system-site-packages .venv ) && .venv/bin/pip install --quiet --upgrade pip && .venv/bin/pip install --quiet -e '.[rpi]' && .venv/bin/python -c 'from terminaleyes.raspi.server import main; print(\"INSTALL_OK\")'"

ok "Package installed with [rpi] extras"

# Verify entry point exists
if pi_run "test -f $REMOTE_DIR/.venv/bin/terminaleyes-pi" 2>/dev/null; then
    ok "terminaleyes-pi entry point exists"
else
    warn "Entry point not found (may need pip install -e .)"
fi

# ---------------------------------------------------------------------------
# Phase 4: Setup USB Gadget (ECM Ethernet)
# ---------------------------------------------------------------------------
header "Phase 4: USB ECM Gadget Setup"

info "Running setup_usb_gadget.sh ecm on Pi..."

GADGET_RESULT=$(pi_sudo "bash $REMOTE_DIR/scripts/setup_usb_gadget.sh ecm" 2>&1) || true

if echo "$GADGET_RESULT" | grep -q "enabled"; then
    ok "USB ECM gadget configured"
elif echo "$GADGET_RESULT" | grep -q "No USB Device Controller"; then
    warn "No UDC found — USB cable may not be connected to target"
else
    warn "Gadget setup had issues: $(echo "$GADGET_RESULT" | tail -1)"
fi

# Check for usb0 interface
if pi_run "ip link show usb0" &>/dev/null; then
    ok "usb0 interface exists"
else
    warn "usb0 not present (expected if USB not connected to host)"
fi

# ---------------------------------------------------------------------------
# Phase 5: Start Server & Test Endpoints
# ---------------------------------------------------------------------------
if [ "$SKIP_TEST" = false ]; then
    header "Phase 5: Start Server & Test Endpoints"

    # Kill any existing server
    pi_run "pkill -f terminaleyes-pi || true" || true
    sleep 1

    # Start server in background (close all fds so SSH disconnects)
    info "Starting terminaleyes-pi server on port 8080..."
    ssh $SSH_OPTS "$REMOTE" "cd $REMOTE_DIR && nohup .venv/bin/terminaleyes-pi > /tmp/terminaleyes-pi.log 2>&1 < /dev/null &" || true
    sleep 2

    # Wait for server to be ready (up to 15 seconds)
    info "Waiting for server to be ready..."
    READY=false
    for i in $(seq 1 15); do
        if curl -s --connect-timeout 2 "http://$PI_HOST:8080/health" &>/dev/null; then
            READY=true
            break
        fi
        sleep 1
    done

    if [ "$READY" = true ]; then
        ok "Server is running"
    else
        fail "Server did not start within 15 seconds"
        info "Log output:"
        pi_run "cat /tmp/terminaleyes-pi.log 2>/dev/null" || true
        # Continue anyway to see what happens
    fi

    # Test endpoints
    echo ""
    info "Testing endpoints..."

    # GET /health
    HTTP_CODE=$(curl -s -o /tmp/te_health.json -w "%{http_code}" "http://$PI_HOST:8080/health" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        ok "GET /health -> $HTTP_CODE"
    else
        fail "GET /health -> $HTTP_CODE"
    fi

    # POST /keystroke
    HTTP_CODE=$(curl -s -o /tmp/te_keystroke.json -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"key": "Enter"}' \
        "http://$PI_HOST:8080/keystroke" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ]; then
        ok "POST /keystroke -> $HTTP_CODE ($([ "$HTTP_CODE" = "400" ] && echo "HID not open — expected" || echo "success"))"
    else
        fail "POST /keystroke -> $HTTP_CODE"
    fi

    # POST /key-combo
    HTTP_CODE=$(curl -s -o /tmp/te_combo.json -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"modifiers": ["ctrl"], "key": "c"}' \
        "http://$PI_HOST:8080/key-combo" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ]; then
        ok "POST /key-combo -> $HTTP_CODE ($([ "$HTTP_CODE" = "400" ] && echo "HID not open — expected" || echo "success"))"
    else
        fail "POST /key-combo -> $HTTP_CODE"
    fi

    # POST /text
    HTTP_CODE=$(curl -s -o /tmp/te_text.json -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"text": "hello"}' \
        "http://$PI_HOST:8080/text" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ]; then
        ok "POST /text -> $HTTP_CODE ($([ "$HTTP_CODE" = "400" ] && echo "HID not open — expected" || echo "success"))"
    else
        fail "POST /text -> $HTTP_CODE"
    fi

    # --- Bluetooth HID endpoints (keyboard + mouse) ---
    echo ""
    info "Testing Bluetooth HID endpoints..."

    # POST /bt/keystroke
    HTTP_CODE=$(curl -s -o /tmp/te_bt_ks.json -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"key": "Enter"}' \
        "http://$PI_HOST:8080/bt/keystroke" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ] || [ "$HTTP_CODE" = "503" ]; then
        ok "POST /bt/keystroke -> $HTTP_CODE ($([ "$HTTP_CODE" = "503" ] && echo "BT not set up — expected" || echo "tested"))"
    else
        fail "POST /bt/keystroke -> $HTTP_CODE"
    fi

    # POST /bt/key-combo
    HTTP_CODE=$(curl -s -o /tmp/te_bt_combo.json -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"modifiers": ["ctrl"], "key": "c"}' \
        "http://$PI_HOST:8080/bt/key-combo" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ] || [ "$HTTP_CODE" = "503" ]; then
        ok "POST /bt/key-combo -> $HTTP_CODE ($([ "$HTTP_CODE" = "503" ] && echo "BT not set up — expected" || echo "tested"))"
    else
        fail "POST /bt/key-combo -> $HTTP_CODE"
    fi

    # POST /bt/text
    HTTP_CODE=$(curl -s -o /tmp/te_bt_text.json -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"text": "hello"}' \
        "http://$PI_HOST:8080/bt/text" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ] || [ "$HTTP_CODE" = "503" ]; then
        ok "POST /bt/text -> $HTTP_CODE ($([ "$HTTP_CODE" = "503" ] && echo "BT not set up — expected" || echo "tested"))"
    else
        fail "POST /bt/text -> $HTTP_CODE"
    fi

    # POST /bt/mouse/move
    HTTP_CODE=$(curl -s -o /tmp/te_bt_mouse.json -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"x": 10, "y": 0}' \
        "http://$PI_HOST:8080/bt/mouse/move" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ] || [ "$HTTP_CODE" = "503" ]; then
        ok "POST /bt/mouse/move -> $HTTP_CODE ($([ "$HTTP_CODE" = "503" ] && echo "BT not set up — expected" || echo "tested"))"
    else
        fail "POST /bt/mouse/move -> $HTTP_CODE"
    fi

    # POST /bt/mouse/click
    HTTP_CODE=$(curl -s -o /tmp/te_bt_click.json -w "%{http_code}" \
        -X POST -H "Content-Type: application/json" \
        -d '{"button": "left"}' \
        "http://$PI_HOST:8080/bt/mouse/click" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "400" ] || [ "$HTTP_CODE" = "503" ]; then
        ok "POST /bt/mouse/click -> $HTTP_CODE ($([ "$HTTP_CODE" = "503" ] && echo "BT not set up — expected" || echo "tested"))"
    else
        fail "POST /bt/mouse/click -> $HTTP_CODE"
    fi

    # Kill the test server
    pi_run "pkill -f 'terminaleyes-pi' 2>/dev/null || true"
    ok "Test server stopped"

    # -------------------------------------------------------------------
    # Phase 6: Run Unit Tests on Pi
    # -------------------------------------------------------------------
    header "Phase 6: Run Unit Tests on Pi"

    info "Installing test dependencies..."
    pi_run "cd $REMOTE_DIR && .venv/bin/pip install --quiet pytest pytest-asyncio pytest-mock"

    info "Running unit tests..."
    TEST_OUTPUT=$(pi_run "cd $REMOTE_DIR && .venv/bin/python -m pytest tests/unit/test_raspi/ -v 2>&1") || true

    if echo "$TEST_OUTPUT" | grep -q "passed"; then
        PASSED=$(echo "$TEST_OUTPUT" | grep -oE '[0-9]+ passed' || echo "? passed")
        ok "Unit tests: $PASSED"
    else
        fail "Unit tests failed"
        echo "$TEST_OUTPUT" | tail -20
    fi
fi

# ---------------------------------------------------------------------------
# Phase 7: Install systemd Service
# ---------------------------------------------------------------------------
header "Phase 7: Install systemd Service"

info "Deploying terminaleyes-pi.service..."

# Generate service file with correct username
SERVICE_CONTENT=$(cat <<EOF
[Unit]
Description=TerminalEyes Pi Keyboard REST API
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$REMOTE_DIR
ExecStartPre=/bin/bash $REMOTE_DIR/scripts/setup_usb_gadget.sh ecm
ExecStartPre=/bin/bash $REMOTE_DIR/scripts/radio_mode.sh apply
ExecStart=$REMOTE_DIR/.venv/bin/terminaleyes-pi
Restart=on-failure
RestartSec=5
Environment=PATH=$REMOTE_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
EOF
)

# Write service file to Pi
echo "$SERVICE_CONTENT" | pi_run "cat > /tmp/terminaleyes-pi.service"
pi_sudo "mv /tmp/terminaleyes-pi.service /etc/systemd/system/terminaleyes-pi.service"
pi_sudo "systemctl daemon-reload"
pi_sudo "systemctl enable terminaleyes-pi.service"

ok "systemd service installed and enabled"
info "Start with: ssh $REMOTE 'sudo systemctl start terminaleyes-pi'"
info "Logs with:  ssh $REMOTE 'sudo journalctl -u terminaleyes-pi -f'"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Deployment Complete"
echo ""
echo "  Pi:       $PI_HOST"
echo "  User:     $PI_USERNAME"
echo "  Code:     $REMOTE_DIR"
echo "  Service:  terminaleyes-pi.service (enabled, not started)"
echo "  API:      http://$PI_HOST:8080"
echo ""
echo "  Next steps:"
echo "    1. Connect Pi USB data port to target machine"
echo "    2. sudo systemctl start terminaleyes-pi"
echo "    3. curl http://$PI_HOST:8080/health"
echo ""
