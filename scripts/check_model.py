"""Headless self-test for sim/models/teleop_scene.xml.

Compiles the scene, resets to the "home" keyframe, verifies the arm is
collision-free there, then holds the pose for 500 steps with the
mujoco_sync-style gravity-compensated PD loop and asserts the arm stays put.

Run with the rebot_teleop env python (no viewer needed):
    python scripts/check_model.py
Exits nonzero on any failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "sim" / "models" / "teleop_scene.xml"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))

# PD gains copied from dev_helpers .../rebotarm_mujoco_rs/mujoco_sync.py
ARM_KP = np.array([80.0, 100.0, 100.0, 35.0, 25.0, 18.0])
ARM_KD = np.array([8.0, 10.0, 10.0, 4.0, 3.0, 2.5])
ARM_TAU_LIMIT = np.array([36.0, 36.0, 36.0, 14.0, 14.0, 14.0])
GRIPPER_KP = 1800.0
GRIPPER_KD = 18.0
GRIPPER_TAU_LIMIT = 64.0

HOLD_STEPS = 500
HOLD_TOL_RAD = 0.05
LIMIT_MARGIN_RAD = 0.15


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def arm_body_ids(model: mujoco.MjModel) -> set[int]:
    """base_link and every body in its subtree (links, gripper, fingers)."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        fail("body 'base_link' not found")
    ids = {base_id}
    for body in range(model.nbody):
        parent = body
        while parent != 0:
            if parent in ids:
                ids.add(body)
                break
            parent = int(model.body_parentid[parent])
    return ids


def arm_contacts(model: mujoco.MjModel, data: mujoco.MjData, ids: set[int]) -> list[str]:
    out = []
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        if b1 in ids or b2 in ids:
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
            out.append(f"{n1} <-> {n2}")
    return out


def main() -> None:
    if not MODEL_PATH.is_file():
        fail(f"model not found: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 6 arm + joint7 + joint_left + joint_right = 9, plus two free bodies (7 each).
    if model.nq != 23:
        fail(f"nq = {model.nq}, expected 23")
    if model.nu != 7:
        fail(f"nu = {model.nu}, expected 7")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        fail("keyframe 'home' not found")

    joint_ids = []
    for name in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            fail(f"joint '{name}' not found")
        joint_ids.append(jid)
    joint_ids = np.array(joint_ids)
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    act_ids = []
    for name in ARM_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
        if aid < 0:
            fail(f"actuator '{name}_motor' not found")
        act_ids.append(aid)
    act_ids = np.array(act_ids)

    grip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
    grip_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint7_motor")
    if grip_jid < 0 or grip_aid < 0:
        fail("gripper joint7 / joint7_motor not found")
    grip_qadr = int(model.jnt_qposadr[grip_jid])
    grip_dadr = int(model.jnt_dofadr[grip_jid])

    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if tcp_site < 0:
        fail("site 'tcp' not found")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    q_home = data.qpos[qpos_adr].copy()
    grip_home = float(data.qpos[grip_qadr])

    # Home must sit comfortably inside the joint limits.
    for name, jid, q in zip(ARM_JOINTS, joint_ids, q_home):
        lo, hi = model.jnt_range[jid]
        if q < lo + LIMIT_MARGIN_RAD or q > hi - LIMIT_MARGIN_RAD:
            fail(f"{name} home q={q:.3f} within {LIMIT_MARGIN_RAD} rad of range [{lo}, {hi}]")

    # No contact involving any arm geom at the home pose (object-table contacts are fine).
    ids = arm_body_ids(model)
    bad = arm_contacts(model, data, ids)
    if bad:
        fail("arm contacts at home pose: " + "; ".join(bad))

    tcp_home = data.site_xpos[tcp_site].copy()
    print(f"home q: {np.array2string(q_home, precision=3)}")
    print(f"tcp world position at home: {np.array2string(tcp_home, precision=4)}")

    # Hold home with the mujoco_sync PD loop.
    max_err = 0.0
    for _ in range(HOLD_STEPS):
        q = data.qpos[qpos_adr]
        qd = data.qvel[dof_adr]
        bias = data.qfrc_bias[dof_adr]
        tau = bias + ARM_KP * (q_home - q) - ARM_KD * qd
        data.ctrl[act_ids] = np.clip(tau, -ARM_TAU_LIMIT, ARM_TAU_LIMIT)

        gq = float(data.qpos[grip_qadr])
        gqd = float(data.qvel[grip_dadr])
        gtau = GRIPPER_KP * (grip_home - gq) - GRIPPER_KD * gqd
        data.ctrl[grip_aid] = float(np.clip(gtau, -GRIPPER_TAU_LIMIT, GRIPPER_TAU_LIMIT))

        mujoco.mj_step(model, data)
        max_err = max(max_err, float(np.max(np.abs(data.qpos[qpos_adr] - q_home))))

    print(f"max |q - q_home| over {HOLD_STEPS} steps: {max_err:.4f} rad")
    if max_err > HOLD_TOL_RAD:
        fail(f"arm drifted {max_err:.4f} rad from home (> {HOLD_TOL_RAD})")

    bad = arm_contacts(model, data, ids)
    if bad:
        fail("arm contacts after hold: " + "; ".join(bad))

    drift = np.linalg.norm(data.site_xpos[tcp_site] - tcp_home)
    print(f"tcp drift after hold: {drift * 1000:.2f} mm")

    check_empty_scene()
    print("OK")


def check_empty_scene() -> None:
    """teleop_empty.xml: floor + arm + goal triad only, 'rest' keyframe = zeros,
    PD holds the rest pose."""
    path = MODEL_PATH.parent / "teleop_empty.xml"
    if not path.exists():
        fail(f"{path} not found")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    print(f"[empty scene] nq={model.nq} nu={model.nu}")
    if model.nq != 9 or model.nu != 7:
        fail(f"empty scene expected nq=9/nu=7, got nq={model.nq}/nu={model.nu}")

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest")
    if key < 0:
        fail("keyframe 'rest' not found in teleop_empty.xml")
    if np.any(model.key_qpos[key] != 0.0):
        fail("'rest' keyframe is not all zeros")

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
    if body < 0 or model.body_mocapid[body] < 0:
        fail("ik_target mocap body missing from empty scene")
    for g in ("axis_x", "axis_y", "axis_z"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) < 0:
            fail(f"goal-triad geom {g!r} missing")

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 7)]
    qpos_adr = np.array([model.jnt_qposadr[j] for j in jids])
    dof_adr = np.array([model.jnt_dofadr[j] for j in jids])
    act_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                          f"joint{i}_motor") for i in range(1, 7)])
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    ids = arm_body_ids(model)
    bad = arm_contacts(model, data, ids)
    if bad:
        fail("arm contacts at rest pose: " + "; ".join(bad))
    print(f"[empty scene] tcp at rest: {np.round(data.site_xpos[tcp], 4)}")

    q_rest = np.zeros(6)
    max_err = 0.0
    for _ in range(HOLD_STEPS):
        tau = (data.qfrc_bias[dof_adr]
               + ARM_KP * (q_rest - data.qpos[qpos_adr])
               - ARM_KD * data.qvel[dof_adr])
        data.ctrl[act_ids] = np.clip(tau, -ARM_TAU_LIMIT, ARM_TAU_LIMIT)
        mujoco.mj_step(model, data)
        max_err = max(max_err, float(np.max(np.abs(data.qpos[qpos_adr] - q_rest))))
    print(f"[empty scene] max |q| over {HOLD_STEPS} PD-hold steps: {max_err:.4f} rad")
    if max_err > HOLD_TOL_RAD:
        fail(f"empty scene: arm drifted {max_err:.4f} rad from rest (> {HOLD_TOL_RAD})")


if __name__ == "__main__":
    main()
