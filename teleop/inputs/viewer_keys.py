"""Keyboard input via the MuJoCo viewer's key_callback (no extra window).

This is the default keyboard backend on macOS: under mjpython, pygame cannot
open its own window (Cocoa windows must be created on the main thread, which
the viewer owns), so key presses are taken from the viewer window itself.

The viewer only reports key PRESSES (no releases), so this backend is
discrete-step only: each press emits one unit step; fine mode (half steps) is
a TOGGLE on F rather than a held modifier. Note some letters also trigger the
viewer's own visualization toggles (harmless, cosmetic).

Wire-up: pass .key_callback to mujoco.viewer.launch_passive(...), then call
.poll() each frame like any other InputDevice.
"""
from __future__ import annotations

import threading

import numpy as np

from teleop.inputs.base import NUM_AXES

# GLFW keycodes (letters are their ASCII uppercase).
_K = {
    "UP": 265, "DOWN": 264, "LEFT": 263, "RIGHT": 262,
    "W": 87, "S": 83, "A": 65, "D": 68, "Q": 81, "E": 69,
    "Z": 90, "X": 88, "O": 79, "C": 67, "H": 72, "F": 70,
    "SPACE": 32, "ESC": 256,
}

# keycode -> (axis index, sign); axes = [dx, dy, dz, droll, dpitch, dyaw, dgrip]
_KEY_AXIS = {
    _K["UP"]: (0, +1), _K["DOWN"]: (0, -1),
    _K["LEFT"]: (1, +1), _K["RIGHT"]: (1, -1),
    _K["W"]: (2, +1), _K["S"]: (2, -1),
    _K["Q"]: (3, +1), _K["E"]: (3, -1),
    _K["Z"]: (4, +1), _K["X"]: (4, -1),
    _K["A"]: (5, +1), _K["D"]: (5, -1),
    _K["O"]: (6, +1), _K["C"]: (6, -1),
}

BINDINGS = (
    "Keys go to the VIEWER window (no separate key window):",
    "  Up / Down : x fwd/back      Left/Right: y +/-",
    "  W / S     : z up/down       Q / E     : roll +/-",
    "  Z / X     : pitch +/-       A / D     : yaw +/-",
    "  O / C     : gripper open/close",
    "  F         : toggle fine mode (half steps)",
    "  SPACE     : HOLD/TRACK clutch (real arm, --mirror)",
    "  H         : return goal to start pose   ESC: quit",
    "(some letters also flip viewer display toggles - cosmetic only)",
)


class ViewerKeyInput:
    """InputDevice fed by mujoco.viewer key callbacks (discrete steps only)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps = np.zeros(NUM_AXES, dtype=int)
        self._reset = False
        self._clutch = False
        self._quit = False
        self._fine = False
        print("\n".join(BINDINGS))

    def key_callback(self, keycode: int) -> None:
        with self._lock:
            if keycode == _K["ESC"]:
                self._quit = True
            elif keycode == _K["H"]:
                self._reset = True
            elif keycode == _K["SPACE"]:
                self._clutch = True
            elif keycode == _K["F"]:
                self._fine = not self._fine
                print(f"[viewer_keys] fine mode {'ON (half steps)' if self._fine else 'OFF'}")
            elif keycode in _KEY_AXIS:
                axis, sign = _KEY_AXIS[keycode]
                self._steps[axis] += sign

    def poll(self) -> tuple[np.ndarray, dict]:
        with self._lock:
            steps = self._steps.copy()
            self._steps[:] = 0
            events = {"reset": self._reset, "quit": self._quit, "slow": self._fine,
                      "clutch": self._clutch, "steps": steps}
            self._reset = False
            self._clutch = False
        return np.zeros(NUM_AXES), events

    def close(self) -> None:
        pass
