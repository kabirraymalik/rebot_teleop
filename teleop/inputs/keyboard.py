"""Pygame keyboard input device.

Opens a small window that must be FOCUSED to receive keys (noted in its title).

Default mode is "step": each key PRESS emits one discrete unit step per axis
(events["steps"], an int array in axis order); the main loop scales it by its
--lin-step / --ang-step / --grip-step sizes. Pass mode="hold" for the legacy
held-key velocity axes.
"""
from __future__ import annotations

import numpy as np
import pygame

from teleop.inputs.base import NUM_AXES

BINDINGS = (
    "Up / Down : x fwd/back",
    "Left/Right: y +/-",
    "W / S     : z up/down",
    "Q / E     : roll +/-",
    "Z / X     : pitch +/-",
    "A / D     : yaw +/-",
    "O / C     : gripper open/close",
    "SPACE     : fine mode (half step)",
    "H         : reset to home",
    "ESC       : quit",
)

# key -> (axis index, sign); axes = [dx, dy, dz, droll, dpitch, dyaw, dgrip]
_KEY_AXIS = {
    pygame.K_UP: (0, +1), pygame.K_DOWN: (0, -1),
    pygame.K_LEFT: (1, +1), pygame.K_RIGHT: (1, -1),
    pygame.K_w: (2, +1), pygame.K_s: (2, -1),
    pygame.K_q: (3, +1), pygame.K_e: (3, -1),
    pygame.K_z: (4, +1), pygame.K_x: (4, -1),
    pygame.K_a: (5, +1), pygame.K_d: (5, -1),
    pygame.K_o: (6, +1), pygame.K_c: (6, -1),
}


class KeyboardInput:
    def __init__(self, mode: str = "step") -> None:
        if mode not in ("step", "hold"):
            raise ValueError(f"mode must be 'step' or 'hold', got {mode!r}")
        self.mode = mode
        pygame.init()
        self._screen = pygame.display.set_mode((420, 260))
        pygame.display.set_caption("reBot teleop keys - FOCUS THIS WINDOW")
        self._font = pygame.font.Font(None, 20)
        self._axes = np.zeros(NUM_AXES)
        self._draw()

    def poll(self) -> tuple[np.ndarray, dict]:
        events: dict = {"reset": False, "quit": False, "slow": False}
        steps = np.zeros(NUM_AXES, dtype=int)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                events["quit"] = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    events["quit"] = True
                elif ev.key == pygame.K_h:
                    events["reset"] = True
                elif self.mode == "step" and ev.key in _KEY_AXIS:
                    axis, sign = _KEY_AXIS[ev.key]
                    steps[axis] += sign

        keys = pygame.key.get_pressed()
        events["slow"] = bool(keys[pygame.K_SPACE])
        a = self._axes
        if self.mode == "hold":
            a[:] = 0.0
            for key, (axis, sign) in _KEY_AXIS.items():
                if keys[key]:
                    a[axis] += sign
            np.clip(a, -1.0, 1.0, out=a)
        else:
            a[:] = 0.0
            events["steps"] = steps

        self._draw()
        return a, events

    def close(self) -> None:
        pygame.quit()

    def _draw(self) -> None:
        self._screen.fill((24, 26, 28))
        title = f"mode: {self.mode}"
        surf = self._font.render(title, True, (140, 200, 140))
        self._screen.blit(surf, (14, 8))
        for i, line in enumerate(BINDINGS):
            surf = self._font.render(line, True, (220, 225, 220))
            self._screen.blit(surf, (14, 32 + 22 * i))
        pygame.display.flip()
