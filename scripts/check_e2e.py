#!/usr/bin/env python
"""Headless end-to-end smoke test of the teleop sim loop (no viewer).

Replicates scripts/teleop_sim.py's frame loop for ~3 s on teleop_scene.xml:
home keyframe -> EECommand.from_fk -> synthetic axes (slow +x, then slow
+yaw, then settle) -> IKTracker.step -> gravity-compensated PD physics.
Asserts the tcp moved in +x, the orientation yawed, the mirror hand-off hook
carries the latest (q6, grip_norm), and nothing went non-finite or unstable.

Run:  /path/to/env/python scripts/check_e2e.py     (exit 0 pass / 1 fail)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.ik import ARM_JOINT_NAMES, IKTracker  # noqa: E402
from teleop.command import EECommand  # noqa: E402

SCENE = REPO_ROOT / "sim" / "models" / "teleop_scene.xml"

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def load_teleop_sim():
    """Import scripts/teleop_sim.py as a module (headless import-path check)."""
    spec = importlib.util.spec_from_file_location(
        "teleop_sim", REPO_ROOT / "scripts" / "teleop_sim.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["teleop_sim"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ts = load_teleop_sim()
    check(True, "teleop_sim imports headlessly")
    check(ts.get_target() is None, "get_target() is None before the first frame",
          repr(ts.get_target()))

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    check(key >= 0, 'keyframe "home" exists')
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    arm_qadr = np.array([model.jnt_qposadr[j] for j in jids])
    arm_dadr = np.array([model.jnt_dofadr[j] for j in jids])
    arm_range = model.jnt_range[jids]
    grip_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in ("joint7", "joint_left", "joint_right")]
    grip_qadr = np.array([model.jnt_qposadr[j] for j in grip_jids])
    grip_dadr = np.array([model.jnt_dofadr[j] for j in grip_jids])
    arm_act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
                        for n in ARM_JOINT_NAMES])
    grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint7_motor")
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    check(tcp >= 0, 'site "tcp" exists')

    ik = IKTracker(model)
    ik.reset(data.qpos[arm_qadr])
    grip = float(np.clip(data.qpos[grip_qadr[0]] / ts.GRIP_SIM_SCALE, 0.0, 1.0))
    cmd = EECommand.from_fk(model, data, grip)

    tcp_home = data.site_xpos[tcp].copy()
    quat_home = np.zeros(4)
    mujoco.mju_mat2Quat(quat_home, data.site_xmat[tcp])

    timestep = model.opt.timestep
    n_sub = max(1, round((1.0 / 60.0) / timestep))
    frame_dt = n_sub * timestep
    n_frames = round(3.0 / frame_dt)
    slow = 0.5  # slow-mode speed scale, as teleop_sim applies for events["slow"]

    max_qvel = 0.0
    last_q6 = None
    for frame in range(n_frames):
        t = frame * frame_dt
        axes = np.zeros(7)
        if t < 1.25:
            axes[0] = 1.0   # +x
        elif t < 2.5:
            axes[5] = 1.0   # +yaw
        # else: settle with zero axes
        cmd.apply_axes(axes, frame_dt,
                       lin_speed=ts.LIN_SPEED * slow,
                       ang_speed=ts.ANG_SPEED * slow,
                       grip_speed=ts.GRIP_SPEED * slow)
        q6_target = ik.step(data, cmd.pos, cmd.quat_wxyz, frame_dt)
        ts.target_state.set(q6_target, cmd.grip)
        last_q6 = q6_target

        grip_target = ts.GRIP_SIM_SCALE * cmd.grip
        for _ in range(n_sub):
            tau = (data.qfrc_bias[arm_dadr]
                   + ts.ARM_KP * (q6_target - data.qpos[arm_qadr])
                   - ts.ARM_KD * data.qvel[arm_dadr])
            data.ctrl[arm_act] = np.clip(tau, -ts.ARM_TAU_LIMIT, ts.ARM_TAU_LIMIT)
            gtau = (ts.GRIPPER_KP * (grip_target - data.qpos[grip_qadr[0]])
                    - ts.GRIPPER_KD * data.qvel[grip_dadr[0]])
            data.ctrl[grip_act] = np.clip(gtau, -ts.GRIPPER_TAU_LIMIT, ts.GRIPPER_TAU_LIMIT)
            mujoco.mj_step(model, data)

        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            check(False, "state stayed finite", f"NaN/inf at t={t:.2f} s")
            return 1
        max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel[arm_dadr]))))
        if not np.all((q6_target >= arm_range[:, 0] - 1e-9)
                      & (q6_target <= arm_range[:, 1] + 1e-9)):
            check(False, "IK targets within joint ranges", f"violated at t={t:.2f} s")
            return 1

    check(True, "state stayed finite for 3 s")
    check(True, "IK targets stayed within joint ranges")

    tcp_now = data.site_xpos[tcp]
    dx = float(tcp_now[0] - tcp_home[0])
    # 1.25 s at 0.5 * 0.15 m/s commands ~0.094 m of +x travel.
    check(dx > 0.05, "tcp moved in +x", f"dx = {dx * 1000:.1f} mm")

    quat_now = np.zeros(4)
    mujoco.mju_mat2Quat(quat_now, data.site_xmat[tcp])
    dq = np.zeros(3)
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, quat_home)
    rel = np.zeros(4)
    mujoco.mju_mulQuat(rel, quat_now, neg)
    mujoco.mju_quat2Vel(dq, rel, 1.0)
    ori_change = float(np.linalg.norm(dq))
    # 1.25 s at 0.5 * 0.9 rad/s commands ~0.56 rad of yaw.
    check(ori_change > 0.2, "tcp orientation yawed", f"|dtheta| = {ori_change:.3f} rad")

    check(max_qvel < 10.0, "no instability (arm qvel bounded)",
          f"max |qvel| = {max_qvel:.2f} rad/s")

    track_err = float(np.max(np.abs(last_q6 - data.qpos[arm_qadr])))
    check(track_err < 0.05, "PD settled on the final IK target",
          f"max joint err = {track_err * 1000:.1f} mrad")

    tgt = ts.get_target()
    check(tgt is not None and np.allclose(tgt[0], last_q6) and tgt[1] == cmd.grip,
          "mirror hook returns the latest (q6, grip_norm)")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        return 1
    print("\nALL GREEN: e2e teleop sim loop OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
