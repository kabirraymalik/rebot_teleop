"""Input device protocol for teleop.

Axes order (all in [-1, 1]): [dx, dy, dz, droll, dpitch, dyaw, dgrip].
Events dict keys (all bool):
  - "reset": return EE command to the home pose (edge-triggered)
  - "quit":  stop teleop
  - "slow":  slow-mode modifier held; the consumer halves speeds
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

AXIS_NAMES: tuple[str, ...] = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgrip")
NUM_AXES: int = len(AXIS_NAMES)


class InputDevice(Protocol):
    def poll(self) -> tuple[np.ndarray, dict[str, bool]]:
        """Return (axes (7,) float in [-1, 1], events dict). Non-blocking."""
        ...
