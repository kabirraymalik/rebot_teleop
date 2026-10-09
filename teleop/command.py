"""End-effector pose + gripper command model for teleop.

Orientation convention: rpy = (roll, pitch, yaw) composed as R = Rx(roll) *
Ry(pitch) * Rz(yaw), which matches ``mujoco.mju_euler2Quat(quat, rpy, "xyz")``
(verified numerically). Quaternions are wxyz (MuJoCo convention).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

# Reachable-and-safe EE workspace box, meters (base frame = world).
WORKSPACE_LO = np.array([0.10, -0.40, 0.03])
WORKSPACE_HI = np.array([0.55, 0.40, 0.60])


class EECommand:
    """Integrates normalized input axes into a clamped EE pose + gripper command.

    Fields: ``pos`` (3,) meters, ``rpy`` (3,) radians, ``grip`` float in [0, 1]
    (0 = closed).
    """

    def __init__(self, pos: np.ndarray, rpy: np.ndarray, grip: float) -> None:
        # Deliberately NOT clamped: a pose latched from the real arm may start
        # outside the workspace box, and snapping it inside would command a
        # jump. Increments may only move back toward the box (_move_pos).
        self.pos = np.asarray(pos, dtype=float).copy()
        self.rpy = np.asarray(rpy, dtype=float).copy()
        self.grip = float(np.clip(grip, 0.0, 1.0))

    def _move_pos(self, delta: np.ndarray) -> None:
        """Translate, clamped so motion can re-enter but never further exit
        the workspace box."""
        lo = np.minimum(WORKSPACE_LO, self.pos)
        hi = np.maximum(WORKSPACE_HI, self.pos)
        self.pos = np.clip(self.pos + delta, lo, hi)

    def apply_axes(
        self,
        axes: np.ndarray,
        dt: float,
        lin_speed: float = 0.15,
        ang_speed: float = 0.9,
        grip_speed: float = 1.0,
    ) -> None:
        """Integrate axes = [dx, dy, dz, droll, dpitch, dyaw, dgrip] in [-1, 1]."""
        self._move_pos(np.asarray(axes[:3], dtype=float) * (lin_speed * dt))
        self.rpy += np.asarray(axes[3:6], dtype=float) * (ang_speed * dt)
        self.grip = float(np.clip(self.grip + float(axes[6]) * grip_speed * dt, 0.0, 1.0))

    def apply_delta(self, dpos: np.ndarray, drpy: np.ndarray, dgrip: float) -> None:
        """Apply one discrete increment (meters / radians / grip fraction)."""
        self._move_pos(np.asarray(dpos, dtype=float))
        self.rpy += np.asarray(drpy, dtype=float)
        self.grip = float(np.clip(self.grip + float(dgrip), 0.0, 1.0))

    @property
    def quat_wxyz(self) -> np.ndarray:
        """rpy -> quaternion (wxyz), q = qx(roll) * qy(pitch) * qz(yaw)."""
        hr, hp, hy = 0.5 * self.rpy
        cr, sr = math.cos(hr), math.sin(hr)
        cp, sp = math.cos(hp), math.sin(hp)
        cy, sy = math.cos(hy), math.sin(hy)
        return np.array([
            cr * cp * cy - sr * sp * sy,
            sr * cp * cy + cr * sp * sy,
            cr * sp * cy - sr * cp * sy,
            cr * cp * sy + sr * sp * cy,
        ])

    @classmethod
    def from_fk(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        grip: float,
        site: str = "tcp",
    ) -> "EECommand":
        """Seed pos/rpy from the current site pose (kinematics must be up to date)."""
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            raise ValueError(f"site {site!r} not found in model")
        R = data.site_xmat[sid].reshape(3, 3)
        return cls(data.site_xpos[sid], _mat_to_rpy(R), grip)


def _mat_to_rpy(R: np.ndarray) -> np.ndarray:
    """Inverse of quat_wxyz's convention: R = Rx(roll) * Ry(pitch) * Rz(yaw)."""
    sp = float(np.clip(R[0, 2], -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(sp) < 0.9999:
        roll = math.atan2(-R[1, 2], R[2, 2])
        yaw = math.atan2(-R[0, 1], R[0, 0])
    else:
        # Gimbal lock: roll/yaw are degenerate; fold everything into roll.
        roll = math.atan2(sp * R[1, 0], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])
