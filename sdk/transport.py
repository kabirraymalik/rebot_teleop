"""Transport resolution for the reBotArm SDK: channel/transport -> motorbridge Controller.

Officially supported adapter: a PCAN-firmware USB-CAN adapter (PCBUSB library on
macOS, PCAN-Basic on Windows, socketcan on Linux). The CH340/GD32 RobStride
"AT-mode" debug board is NOT currently reachable via motorbridge: its AT text
protocol does not match motorbridge's mcu-serial framing (empirically verified
2026-10-08). The mcu-serial auto-pick below is therefore a best-effort for
bridge boards flashed with motorbridge-compatible MCU firmware, not for the
stock RobStride debug board.

Resolution rules (make_controller):
  transport explicit:
    "socketcan"  -> Controller(channel)                       # e.g. "can0" (Linux)
    "pcan"       -> Controller(channel + "@1000000")          # bitrate appended if missing
    "mcu-serial" -> Controller.from_mcu_serial(channel, baud)
    "dm-serial"  -> Controller.from_dm_serial(channel, baud)
  transport == "auto":
    channel startswith "/dev/" -> dm-serial if vendor == "damiao" else mcu-serial
    channel == "auto"          -> platform scan (see _resolve_auto_channel)
    anything else              -> socketcan on Linux, pcan elsewhere
"""
from __future__ import annotations

import glob
import os
import sys

from motorbridge import Controller
from motorbridge.platform_hints import can_load_pcbusb, macos_pcbusb_hint

_EXPLICIT_TRANSPORTS = ("socketcan", "pcan", "mcu-serial", "dm-serial")


def _serial_scan() -> list[str]:
    """Candidate serial bridge nodes, WCH (CH340-family) nodes first."""
    wch = sorted(glob.glob("/dev/cu.wchusbserial*"))
    usb = sorted(glob.glob("/dev/cu.usbserial*"))
    return wch + usb


def _make_explicit(transport: str, channel: str, baud: int) -> Controller:
    if transport == "socketcan":
        return Controller(channel)
    if transport == "pcan":
        if "@" not in channel:
            channel = f"{channel}@1000000"
        return Controller(channel)
    if transport == "mcu-serial":
        return Controller.from_mcu_serial(channel, baud)
    if transport == "dm-serial":
        return Controller.from_dm_serial(channel, baud)
    raise ValueError(
        f"Unknown transport {transport!r}; expected one of "
        f"{_EXPLICIT_TRANSPORTS} or 'auto'"
    )


def _resolve_auto_channel(baud: int, vendor: str) -> Controller:
    if sys.platform.startswith("linux"):
        if os.path.exists("/sys/class/net/can0"):
            return Controller("can0")
        nodes = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
        if nodes:
            if vendor == "damiao":
                return Controller.from_dm_serial(nodes[0], baud)
            return Controller.from_mcu_serial(nodes[0], baud)
        raise RuntimeError(
            "channel='auto': no socketcan interface 'can0' and no serial "
            "bridge node found (/dev/ttyACM*, /dev/ttyUSB*)"
        )
    if sys.platform == "darwin":
        if can_load_pcbusb():
            return Controller("can0@1000000")
        nodes = _serial_scan()
        if nodes:
            if vendor == "damiao":
                return Controller.from_dm_serial(nodes[0], baud)
            return Controller.from_mcu_serial(nodes[0], baud)
        raise RuntimeError(
            "channel='auto': no PCBUSB library and no serial bridge node "
            "(/dev/cu.wchusbserial*, /dev/cu.usbserial*) found.\n"
            + macos_pcbusb_hint("rebot-teleop")
        )
    raise RuntimeError(
        f"channel='auto' is not supported on platform {sys.platform!r}; "
        "set channel/transport explicitly in the config"
    )


def make_controller(
    channel: str,
    transport: str = "auto",
    baud: int = 921600,
    vendor: str = "robstride",
) -> Controller:
    """Open a motorbridge Controller for the given channel/transport.

    See the module docstring for the resolution rules and adapter caveats.
    """
    transport = (transport or "auto").lower()
    if transport != "auto":
        return _make_explicit(transport, channel, baud)

    if channel.startswith("/dev/"):
        if vendor == "damiao":
            return Controller.from_dm_serial(channel, baud)
        return Controller.from_mcu_serial(channel, baud)
    if channel == "auto":
        return _resolve_auto_channel(baud, vendor)
    # A bare CAN interface name: socketcan on Linux, PCAN elsewhere.
    if sys.platform.startswith("linux"):
        return Controller(channel)
    return _make_explicit("pcan", channel, baud)
