#!/usr/bin/env bash
# setup_usb_gadget.sh — Configure Raspberry Pi Zero as a USB HID keyboard gadget.
#
# This script uses Linux USB ConfigFS to set up the Pi as a composite
# USB device that presents itself as a standard keyboard to whatever
# machine the Pi's USB data port is plugged into.
#
# Prerequisites:
#   - Raspberry Pi Zero (W/2W) or Pi 4 with USB OTG support
#   - dwc2 overlay enabled in /boot/config.txt
#   - libcomposite kernel module available
#
# Usage:
#   sudo bash scripts/setup_usb_gadget.sh          # set up
#   sudo bash scripts/setup_usb_gadget.sh teardown  # tear down
#
# After setup, the HID device is available at /dev/hidg0.

set -euo pipefail

GADGET_DIR="/sys/kernel/config/usb_gadget/terminaleyes_kb"
UDC_PATH="/sys/class/udc"

# HID keyboard report descriptor (standard 8-byte boot keyboard)
# Modifier byte + reserved + 6 key codes
REPORT_DESCRIPTOR=$(printf '%b' \
    '\x05\x01' \
    '\x09\x06' \
    '\xa1\x01' \
    '\x05\x07' \
    '\x19\xe0' \
    '\x29\xe7' \
    '\x15\x00' \
    '\x25\x01' \
    '\x75\x01' \
    '\x95\x08' \
    '\x81\x02' \
    '\x95\x01' \
    '\x75\x08' \
    '\x81\x01' \
    '\x95\x06' \
    '\x75\x08' \
    '\x15\x00' \
    '\x25\x65' \
    '\x05\x07' \
    '\x19\x00' \
    '\x29\x65' \
    '\x81\x00' \
    '\xc0')

teardown() {
    echo "Tearing down USB gadget..."
    if [ -d "$GADGET_DIR" ]; then
        # Disable the gadget
        echo "" > "$GADGET_DIR/UDC" 2>/dev/null || true
        # Remove function from configuration
        rm -f "$GADGET_DIR/configs/c.1/hid.usb0" 2>/dev/null || true
        # Remove strings
        rmdir "$GADGET_DIR/configs/c.1/strings/0x409" 2>/dev/null || true
        rmdir "$GADGET_DIR/configs/c.1" 2>/dev/null || true
        rmdir "$GADGET_DIR/functions/hid.usb0" 2>/dev/null || true
        rmdir "$GADGET_DIR/strings/0x409" 2>/dev/null || true
        rmdir "$GADGET_DIR" 2>/dev/null || true
        echo "Gadget removed."
    else
        echo "No gadget found at $GADGET_DIR"
    fi
}

setup() {
    # Ensure we're root
    if [ "$(id -u)" -ne 0 ]; then
        echo "ERROR: Must run as root (sudo)" >&2
        exit 1
    fi

    # Check for existing gadget
    if [ -d "$GADGET_DIR" ]; then
        echo "Gadget already exists. Tearing down first..."
        teardown
    fi

    # Load the required kernel module
    modprobe libcomposite || {
        echo "ERROR: Cannot load libcomposite. Ensure dwc2 overlay is enabled:" >&2
        echo "  Add 'dtoverlay=dwc2' to /boot/config.txt" >&2
        echo "  Add 'dwc2' to /etc/modules" >&2
        echo "  Add 'libcomposite' to /etc/modules" >&2
        exit 1
    }

    echo "Creating USB HID keyboard gadget..."

    # Create the gadget
    mkdir -p "$GADGET_DIR"

    # USB device descriptor
    echo 0x1d6b > "$GADGET_DIR/idVendor"   # Linux Foundation
    echo 0x0104 > "$GADGET_DIR/idProduct"   # Multifunction Composite Gadget
    echo 0x0100 > "$GADGET_DIR/bcdDevice"   # v1.0.0
    echo 0x0200 > "$GADGET_DIR/bcdUSB"      # USB 2.0

    # Device strings
    mkdir -p "$GADGET_DIR/strings/0x409"
    echo "terminaleyes0001"   > "$GADGET_DIR/strings/0x409/serialnumber"
    echo "terminaleyes"       > "$GADGET_DIR/strings/0x409/manufacturer"
    echo "Pi USB Keyboard"    > "$GADGET_DIR/strings/0x409/product"

    # Configuration
    mkdir -p "$GADGET_DIR/configs/c.1/strings/0x409"
    echo "Keyboard Configuration" > "$GADGET_DIR/configs/c.1/strings/0x409/configuration"
    echo 250 > "$GADGET_DIR/configs/c.1/MaxPower"  # 250mA

    # HID function
    mkdir -p "$GADGET_DIR/functions/hid.usb0"
    echo 1   > "$GADGET_DIR/functions/hid.usb0/protocol"     # Keyboard
    echo 1   > "$GADGET_DIR/functions/hid.usb0/subclass"     # Boot interface
    echo 8   > "$GADGET_DIR/functions/hid.usb0/report_length"
    echo -ne "$REPORT_DESCRIPTOR" > "$GADGET_DIR/functions/hid.usb0/report_desc"

    # Link function to configuration
    ln -s "$GADGET_DIR/functions/hid.usb0" "$GADGET_DIR/configs/c.1/"

    # Enable the gadget by binding to the UDC (USB Device Controller)
    UDC=$(ls "$UDC_PATH" | head -1)
    if [ -z "$UDC" ]; then
        echo "ERROR: No USB Device Controller found. Is dwc2 loaded?" >&2
        teardown
        exit 1
    fi
    echo "$UDC" > "$GADGET_DIR/UDC"

    echo "USB HID keyboard gadget enabled."
    echo "  Device: /dev/hidg0"
    echo "  UDC:    $UDC"
    echo ""
    echo "The Pi now appears as a keyboard to the connected machine."
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
