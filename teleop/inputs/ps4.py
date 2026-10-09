"""Pygame PS4 (DualShock 4) gamepad input device.

Mapping (SDL game-controller layout used by pygame 2 on macOS):
  left stick x/y -> dx/dy, right stick vertical -> dz, right stick
  horizontal -> dyaw, L1/R1 roll -/+, L2/R2 analog triggers -> dgrip
  (L2 close, R2 open), dpad up/down -> dpitch, TRIANGLE reset, OPTIONS quit.
Stick deadzone 0.12.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# On macOS (mjpython) Cocoa belongs to the MuJoCo viewer's main thread; the
# dummy video driver keeps SDL off Cocoa entirely — joystick still works.
if sys.platform == "darwin":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from teleop.inputs.base import NUM_AXES

DEADZONE = 0.12

# SDL game-controller indices for DualShock 4 under pygame 2.
AX_LX, AX_LY, AX_RX, AX_RY, AX_L2, AX_R2 = 0, 1, 2, 3, 4, 5
BTN_TRIANGLE = 3
BTN_OPTIONS = 6
BTN_L1, BTN_R1 = 9, 10
BTN_DPAD_UP, BTN_DPAD_DOWN = 11, 12


def _deadzone(v: float) -> float:
    if abs(v) < DEADZONE:
        return 0.0
    # Rescale so output is continuous at the deadzone edge.
    s = (abs(v) - DEADZONE) / (1.0 - DEADZONE)
    return s if v > 0 else -s


class PS4Input:
    def __init__(self, index: int = 0) -> None:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No joystick detected. Pair/connect the PS4 controller "
                "(USB or Bluetooth) and retry, or use --input keyboard."
            )
        self._js = pygame.joystick.Joystick(index)
        self._js.init()
        print(f"[ps4] using joystick {index}: {self._js.get_name()} "
              f"({self._js.get_numaxes()} axes, {self._js.get_numbuttons()} buttons)")
        self._axes = np.zeros(NUM_AXES)
        # SDL reports trigger axes as 0 (center) until first touched; treat
        # them as released (-1) until we see them move.
        self._trigger_seen = [False, False]

    def poll(self) -> tuple[np.ndarray, dict[str, bool]]:
        events = {"reset": False, "quit": False, "slow": False}
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                events["quit"] = True
            elif ev.type == pygame.JOYBUTTONDOWN:
                if ev.button == BTN_OPTIONS:
                    events["quit"] = True
                elif ev.button == BTN_TRIANGLE:
                    events["reset"] = True

        js = self._js
        a = self._axes
        # Stick up gives a negative SDL value; invert so up/forward is positive.
        a[0] = -_deadzone(js.get_axis(AX_LY))        # dx: left stick vertical
        a[1] = -_deadzone(js.get_axis(AX_LX))        # dy: left stick horizontal (left = +y)
        a[2] = -_deadzone(js.get_axis(AX_RY))        # dz: right stick vertical
        a[5] = -_deadzone(js.get_axis(AX_RX))        # dyaw: right stick horizontal
        a[3] = float(js.get_button(BTN_R1)) - float(js.get_button(BTN_L1))  # droll
        a[4] = self._dpad_pitch()                    # dpitch
        a[6] = self._trigger(AX_R2, 1) - self._trigger(AX_L2, 0)  # dgrip: R2 open, L2 close
        return a, events

    def close(self) -> None:
        pygame.quit()

    def _dpad_pitch(self) -> float:
        js = self._js
        if js.get_numhats() > 0:
            return float(js.get_hat(0)[1])
        if js.get_numbuttons() > BTN_DPAD_DOWN:
            return float(js.get_button(BTN_DPAD_UP)) - float(js.get_button(BTN_DPAD_DOWN))
        return 0.0

    def _trigger(self, axis: int, slot: int) -> float:
        raw = self._js.get_axis(axis)
        if not self._trigger_seen[slot]:
            if abs(raw) < 0.01:
                return 0.0
            self._trigger_seen[slot] = True
        return (raw + 1.0) * 0.5  # [-1, 1] -> [0, 1]
