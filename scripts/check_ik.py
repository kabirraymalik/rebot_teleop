#!/usr/bin/env python
"""Headless IK regression checks (no viewer; run with the env's plain python).

    python scripts/check_ik.py [--scene sim/models/rs_grasp_scene.xml]

Tests: (0) rpy<->quat convention consistency, (a) circular-trajectory tracking,
(b) 15 cm step-target convergence, (c) unreachable-target limit safety.
Exits nonzero on failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.ik import ARM_JOINT_NAMES, IKTracker  # noqa: E402
from teleop.command import EECommand, _mat_to_rpy  # noqa: E402

# Mid-workspace, well-conditioned pose used when the scene has no "home" keyframe.
Q_HOME = np.array([0.0, 1.0, 0.8, 0.0, 0.6, 0.0])
DT = 0.01


def default_scene() -> str:
    teleop_scene = REPO_ROOT / "sim" / "models" / "teleop_scene.xml"
    if teleop_scene.exists():
        return str(teleop_scene)
    return str(REPO_ROOT / "sim" / "models" / "rs_grasp_scene.xml")


class Rig:
    """Kinematic-mode test rig: qpos is written directly from the IK state."""

    def __init__(self, scene: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in ARM_JOINT_NAMES]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jids])
        self.jnt_range = self.model.jnt_range[jids]
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.ik = IKTracker(self.model)

    def reset(self) -> None:
        if self.key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.qadr] = Q_HOME
        mujoco.mj_forward(self.model, self.data)
        self.ik.reset(self.data.qpos[self.qadr])

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.sid])
        return self.data.site_xpos[self.sid].copy(), quat

    def track(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """One kinematic step: IK then qpos teleport + FK."""
        q6 = self.ik.step(self.data, pos, quat, DT)
        self.data.qpos[self.qadr] = q6
        mujoco.mj_kinematics(self.model, self.data)

    def errors(self, pos: np.ndarray, quat: np.ndarray) -> tuple[float, float]:
        p, q = self.tcp_pose()
        neg = np.zeros(4)
        dq = np.zeros(4)
        vel = np.zeros(3)
        mujoco.mju_negQuat(neg, q)
        mujoco.mju_mulQuat(dq, quat, neg)
        if dq[0] < 0:
            dq *= -1
        mujoco.mju_quat2Vel(vel, dq, 1.0)
        return float(np.linalg.norm(pos - p)), float(np.linalg.norm(vel))


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def test_quat_convention() -> bool:
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        rpy = rng.uniform(-1.4, 1.4, 3)
        cmd = EECommand(np.array([0.3, 0.0, 0.3]), rpy, 0.0)
        ref = np.zeros(4)
        mujoco.mju_euler2Quat(ref, rpy, "xyz")
        worst = max(worst, float(np.linalg.norm(cmd.quat_wxyz - ref)))
        # round trip through the matrix extraction used by from_fk
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, ref)
        worst = max(worst, float(np.linalg.norm(_mat_to_rpy(R.reshape(3, 3)) - rpy)))
    return check("quat convention", worst < 1e-9,
                 f"max deviation vs mju_euler2Quat('xyz') = {worst:.2e}")


def test_circle(rig: Rig) -> bool:
    rig.reset()
    p0, quat0 = rig.tcp_pose()
    radius, period = 0.04, 10.0  # 8 cm diameter circle in the y-z plane
    pos_errs, ori_errs = [], []
    target = np.zeros(3)
    for i in range(int(period / DT)):
        th = 2.0 * np.pi * (i + 1) * DT / period
        target[:] = p0
        target[1] += radius * np.sin(th)
        target[2] += radius * (1.0 - np.cos(th))
        rig.track(target, quat0)
        pe, oe = rig.errors(target, quat0)
        pos_errs.append(pe)
        ori_errs.append(oe)
    mean_pe, max_pe, max_oe = np.mean(pos_errs), np.max(pos_errs), np.max(ori_errs)
    ok = mean_pe < 0.005 and max_pe < 0.015 and max_oe < 0.1
    return check("circle tracking", ok,
                 f"pos err mean {mean_pe * 1e3:.2f} mm (<5), max {max_pe * 1e3:.2f} mm (<15), "
                 f"ori err max {max_oe:.4f} rad (<0.1)")


def test_step_target(rig: Rig) -> bool:
    rig.reset()
    p0, quat0 = rig.tcp_pose()
    direction = np.array([0.3, -1.0, -0.3])
    target = p0 + 0.15 * direction / np.linalg.norm(direction)
    for _ in range(int(2.0 / DT)):
        rig.track(target, quat0)
    pe, oe = rig.errors(target, quat0)
    ok = pe < 0.003
    return check("15 cm step convergence", ok,
                 f"final pos err {pe * 1e3:.2f} mm (<3), ori err {oe:.4f} rad")


def test_limits(rig: Rig) -> bool:
    rig.reset()
    _, quat0 = rig.tcp_pose()
    target = np.array([0.9, 0.0, 0.30])  # beyond max reach
    qs = []
    for _ in range(int(2.0 / DT)):
        rig.track(target, quat0)
        qs.append(rig.data.qpos[rig.qadr].copy())
    qs = np.array(qs)
    finite = bool(np.all(np.isfinite(qs)))
    within = bool(np.all(qs >= rig.jnt_range[:, 0]) and np.all(qs <= rig.jnt_range[:, 1]))
    ok = finite and within
    return check("unreachable-target limits", ok,
                 f"finite={finite}, within joint ranges={within}, "
                 f"final q={np.round(qs[-1], 3).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=default_scene())
    args = ap.parse_args()
    print(f"scene: {args.scene}")
    rig = Rig(args.scene)
    results = [
        test_quat_convention(),
        test_circle(rig),
        test_step_target(rig),
        test_limits(rig),
    ]
    if not all(results):
        sys.exit(1)
    print("all IK checks passed")


if __name__ == "__main__":
    main()
