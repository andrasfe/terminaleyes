#!/usr/bin/env python3
"""Bluetooth auto-accept pairing agent using D-Bus.

Registers as a NoInputNoOutput agent and auto-accepts all pairing
requests. More reliable than bluetoothctl's agent mode.
"""
import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

AGENT_PATH = "/org/bluez/terminaleyes_agent"
CAPABILITY = "NoInputNoOutput"

class Agent(dbus.service.Object):
    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Release(self):
        print("Agent released")

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def AuthorizeService(self, device, uuid):
        print(f"AuthorizeService: {device} {uuid} -> accepted")

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="s")
    def RequestPinCode(self, device):
        print(f"RequestPinCode: {device} -> 0000")
        return "0000"

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="u")
    def RequestPasskey(self, device):
        print(f"RequestPasskey: {device} -> 0")
        return dbus.UInt32(0)

    @dbus.service.method("org.bluez.Agent1", in_signature="ouq", out_signature="")
    def DisplayPasskey(self, device, passkey, entered):
        print(f"DisplayPasskey: {device} passkey={passkey}")

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def DisplayPinCode(self, device, pincode):
        print(f"DisplayPinCode: {device} pin={pincode}")

    @dbus.service.method("org.bluez.Agent1", in_signature="ou", out_signature="")
    def RequestConfirmation(self, device, passkey):
        print(f"RequestConfirmation: {device} passkey={passkey} -> confirmed")
        # Auto-accept by returning without error

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="")
    def RequestAuthorization(self, device):
        print(f"RequestAuthorization: {device} -> authorized")
        # Auto-accept by returning without error

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Cancel(self):
        print("Pairing cancelled")


if __name__ == "__main__":
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    agent = Agent(bus, AGENT_PATH)

    manager = dbus.Interface(
        bus.get_object("org.bluez", "/org/bluez"),
        "org.bluez.AgentManager1",
    )

    manager.RegisterAgent(AGENT_PATH, CAPABILITY)
    print(f"Agent registered: {AGENT_PATH} ({CAPABILITY})")

    manager.RequestDefaultAgent(AGENT_PATH)
    print("Set as default agent")

    # Also ensure adapter is discoverable/pairable
    adapter = dbus.Interface(
        bus.get_object("org.bluez", "/org/bluez/hci0"),
        "org.freedesktop.DBus.Properties",
    )
    for prop, val in [
        ("Powered", dbus.Boolean(True)),
        ("Discoverable", dbus.Boolean(True)),
        ("DiscoverableTimeout", dbus.UInt32(0)),
        ("Pairable", dbus.Boolean(True)),
    ]:
        try:
            adapter.Set("org.bluez.Adapter1", prop, val)
        except Exception:
            pass

    # Auto-trust newly-paired devices. Without this, macOS hosts pair
    # OK but the resulting bond is Trusted=no, which on some macOS
    # versions blocks the Mac from auto-opening the L2CAP HID channels
    # (PSM 17/19) when it next comes in range. The HID server reports
    # bt_hid_connected=false and the operator has to manually click
    # Connect in System Settings → Bluetooth every session.
    #
    # We listen for PropertiesChanged on org.bluez.Device1 and the
    # moment a device flips to Paired=True we mark it Trusted=True via
    # the same Properties interface. Idempotent: re-trust calls are a
    # no-op if the device is already trusted.
    def _on_device_props_changed(interface, changed, invalidated, path=None):
        if interface != "org.bluez.Device1":
            return
        if "Paired" not in changed:
            return
        if not changed["Paired"]:
            return
        try:
            dev_props = dbus.Interface(
                bus.get_object("org.bluez", path),
                "org.freedesktop.DBus.Properties",
            )
            trusted = bool(dev_props.Get("org.bluez.Device1", "Trusted"))
            if trusted:
                print(f"auto-trust: {path} already trusted, skipping")
                return
            dev_props.Set("org.bluez.Device1", "Trusted", dbus.Boolean(True))
            print(f"auto-trust: {path} → Trusted=True")
        except Exception as exc:
            print(f"auto-trust failed for {path}: {exc}")

    bus.add_signal_receiver(
        _on_device_props_changed,
        dbus_interface="org.freedesktop.DBus.Properties",
        signal_name="PropertiesChanged",
        arg0="org.bluez.Device1",
        path_keyword="path",
    )
    print("Auto-trust handler registered on Device1.PropertiesChanged")

    print("Waiting for pairing requests...")
    GLib.MainLoop().run()
