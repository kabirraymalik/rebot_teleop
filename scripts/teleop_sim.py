#!/usr/bin/env python
"""Interactive MuJoCo teleop of the reBot B601-RS arm.

On macOS the MuJoCo viewer requires mjpython:

    mjpython scripts/teleop_sim.py                 # keyboard (default)
    mjpython scripts/teleop_sim.py --input ps4
    mjpython scripts/teleop_sim.py --input kb --hold

On macOS keyboard presses go to the viewer window itself; on Linux (or with
--hold) a small pygame key window opens and must be focused to receive keys.

--mirror additionally streams the live joint targets to the real arm at
125 Hz (teleop.mirror.RealArmMirror). On start the sim is
latched to the measured real pose (no jump) and the HOLD clutch is engaged:
the real arm parks while the sim previews your commands. H toggles
HOLD <-> TRACK (rate-limited catch-up; sim reset is disabled while
mirroring). On exit (window close or Ctrl-C) the real arm ramps to its rest
pose before torque-off.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.ik import ARM_JOINT_NAMES, IKTracker  # noqa: E402
from teleop.command import EECommand  # noqa: E402
from teleop.motion_profiles import advance_velocity_limited_reference  # noqa: E402

DEFAULT_SCENE = REPO_ROOT / "sim" / "models" / "teleop_empty.xml"
FALLBACK_SCENES = (REPO_ROOT / "sim" / "models" / "teleop_scene.xml",
                   REPO_ROOT / "sim" / "models" / "rs_grasp_scene.xml")

# Sim PD gains (validated reference: rebotarm_mujoco_rs/mujoco_sync.py).
ARM_KP = np.array([80.0, 100.0, 100.0, 35.0, 25.0, 18.0])
ARM_KD = np.array([8.0, 10.0, 10.0, 4.0, 3.0, 2.5])
ARM_TAU_LIMIT = np.array([36.0, 36.0, 36.0, 14.0, 14.0, 14.0])
GRIPPER_KP, GRIPPER_KD, GRIPPER_TAU_LIMIT = 1800.0, 18.0, 64.0

# Gripper convention: grip_norm in [0,1]; sim drive joint7 = 0.05 * grip_norm m.
GRIP_SIM_SCALE = 0.05

LIN_SPEED, ANG_SPEED, GRIP_SPEED = 0.15, 0.9, 1.0  # m/s, rad/s, 1/s (hold mode)

# Discrete step sizes per key press (step mode, the default).
LIN_STEP, ANG_STEP, GRIP_STEP = 0.01, math.radians(5.0), 0.1

# Joint-space reference slew limit, rad/s — the same low-level shaping the
# real-arm mirror applies (motion_profiles.advance_velocity_limited_reference),
# so the sim validates the exact control loop used on hardware.
REF_SPEED_LIMIT = 0.3
GRIP_REF_SPEED = 0.6  # grip_norm/s == 3 rad/s in gripper motor space (3/5)


class TargetState:
    """Thread-safe (q6_target, grip_norm) hand-off point for a mirror loop.

    Returns None until the first set(), and again when the last set() is older
    than ``stale_after`` seconds (sim loop hung or died) — the mirror holds
    its current reference in both cases instead of tracking a dead value.
    """

    def __init__(self, stale_after: float = 0.5) -> None:
        self._lock = threading.Lock()
        self._q6: np.ndarray | None = None
        self._grip = 0.0
        self._stamp = 0.0
        self._stale_after = stale_after

    def set(self, q6: np.ndarray, grip: float) -> None:
        with self._lock:
            if self._q6 is None:
                self._q6 = np.array(q6, dtype=np.float64).copy()
            else:
                self._q6[:] = q6
            self._grip = float(grip)
            self._stamp = time.monotonic()

    def get_target(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            if self._q6 is None or time.monotonic() - self._stamp > self._stale_after:
                return None
            return self._q6.copy(), self._grip


target_state = TargetState()


def get_target() -> tuple[np.ndarray, float] | None:
    """Latest (q6_target radians, grip_norm in [0,1]), or None before the
    first sim frame; safe from any thread."""
    return target_state.get_target()


def _make_input(kind: str, hold: bool):
    # Imported lazily so unused backends (and any pygame window) never load.
    if kind in ("kb", "keyboard"):
        if sys.platform == "darwin" and not hold:
            # Under mjpython, pygame cannot open a window off the main thread;
            # take key presses from the viewer window instead.
            from teleop.inputs.viewer_keys import ViewerKeyInput
            return ViewerKeyInput()
        from teleop.inputs.keyboard import KeyboardInput
        return KeyboardInput(mode="hold" if hold else "step")
    from teleop.inputs.ps4 import PS4Input
    return PS4Input()


def _resolve_scene(arg: str) -> str:
    p = Path(arg)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    if not p.exists() and Path(arg) == Path(str(DEFAULT_SCENE)):
        for fb in FALLBACK_SCENES:
            if fb.exists():
                print(f"[teleop_sim] {DEFAULT_SCENE} not found, falling back to {fb}")
                p = fb
                break
    if not p.exists():
        sys.exit(f"scene not found: {arg}")
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE), help="MJCF scene path")
    ap.add_argument("--input", choices=("kb", "keyboard", "ps4"), default="kb",
                    help="input device: kb (keyboard, default) or ps4")
    ap.add_argument("--hold", action="store_true",
                    help="keyboard hold-to-move velocity mode instead of "
                         "discrete per-press steps")
    ap.add_argument("--lin-step", type=float, default=LIN_STEP,
                    help="EE position step per key press, meters")
    ap.add_argument("--ang-step", type=float, default=ANG_STEP,
                    help="EE orientation step per key press, radians")
    ap.add_argument("--grip-step", type=float, default=GRIP_STEP,
                    help="gripper step per key press, fraction of full range")
    ap.add_argument("--speed-limit", type=float, default=REF_SPEED_LIMIT,
                    help="joint reference slew limit, rad/s (matches the "
                         "real-arm mirror's low-level shaping)")
    ap.add_argument("--kinematic", action="store_true",
                    help="teleport qpos to the shaped reference instead of PD physics")
    ap.add_argument("--mirror", action="store_true",
                    help="stream joint targets to the real arm at 125 Hz; "
                         "starts parked in HOLD, H toggles HOLD/TRACK")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(_resolve_scene(args.scene))
    data = mujoco.MjData(model)

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    arm_qadr = np.array([model.jnt_qposadr[j] for j in jids])
    arm_dadr = np.array([model.jnt_dofadr[j] for j in jids])
    grip_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in ("joint7", "joint_left", "joint_right")]
    grip_qadr = np.array([model.jnt_qposadr[j] for j in grip_jids])
    grip_dadr = np.array([model.jnt_dofadr[j] for j in grip_jids])
    arm_act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
                        for n in ARM_JOINT_NAMES])
    grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint7_motor")
    mocap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
    mocap_id = model.body_mocapid[mocap_body] if mocap_body >= 0 else -1
    # Start pose preference: "rest" (arm's physical resting pose, q=0,
    # matching the real arm's motor zeros) over the older elbow-up "home".
    home_key = -1
    for key_name in ("rest", "home"):
        home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if home_key >= 0:
            break

    ik = IKTracker(model)
    arm_range = model.jnt_range[jids]  # (6, 2)

    # Shaped joint reference — the same velocity-limited low-level stage the
    # real-arm mirror runs at 125 Hz; PD tracks this, not the raw IK output.
    q_ref = np.zeros(6)
    q_ref_vel = np.zeros(6)
    grip_ref = np.zeros(1)
    grip_ref_vel = np.zeros(1)
    ref_vlim = np.full(6, args.speed_limit)
    grip_vlim = np.array([GRIP_REF_SPEED])

    def seed_reference(q6: np.ndarray, grip: float) -> None:
        q_ref[:] = q6
        q_ref_vel[:] = 0.0
        grip_ref[:] = grip
        grip_ref_vel[:] = 0.0

    def reset_home() -> EECommand:
        if home_key >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_key)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        ik.reset(data.qpos[arm_qadr])
        grip = float(np.clip(data.qpos[grip_qadr[0]] / GRIP_SIM_SCALE, 0.0, 1.0))
        seed_reference(data.qpos[arm_qadr], grip)
        return EECommand.from_fk(model, data, grip)

    def latch_pose(q6: np.ndarray, grip: float) -> EECommand:
        """Set the sim to an externally measured pose (the real arm's)."""
        q6 = np.clip(q6, arm_range[:, 0], arm_range[:, 1])
        grip = float(np.clip(grip, 0.0, 1.0))
        data.qpos[arm_qadr] = q6
        data.qpos[grip_qadr] = GRIP_SIM_SCALE * grip
        data.qvel[arm_dadr] = 0.0
        data.qvel[grip_dadr] = 0.0
        mujoco.mj_forward(model, data)
        ik.reset(q6)
        seed_reference(q6, grip)
        return EECommand.from_fk(model, data, grip)

    cmd = reset_home()

    try:
        dev = _make_input(args.input, args.hold)
    except RuntimeError as exc:
        sys.exit(str(exc))

    mirror = None
    if args.mirror:
        from teleop.mirror import RealArmMirror  # lazy: only touch sdk paths when asked

        mirror = RealArmMirror(get_target=get_target)
        try:
            # Start with the hold clutch engaged: the real arm parks at its
            # latched pose while the sim previews; H releases it.
            q6_real, grip_norm_real = mirror.start(holding=True)
        except RuntimeError as exc:
            close = getattr(dev, "close", None)
            if close is not None:
                close()
            sys.exit(str(exc))
        # Sim follows the real arm's pose so the first targets hold, not jump.
        cmd = latch_pose(q6_real, grip_norm_real)
        print(f"[teleop_sim] mirror latched to real pose q6={np.round(q6_real, 3)}, "
              f"grip={grip_norm_real:.2f}")
        print("[teleop_sim] HOLD engaged: real arm is parked. Check the sim pose "
              "matches the physical arm, position the goal triad, then press H "
              "to let the real arm track (H toggles HOLD/TRACK; sim reset is "
              "disabled while mirroring).")

    timestep = model.opt.timestep
    n_sub = max(1, round((1.0 / 60.0) / timestep))  # physics steps per viewer frame
    frame_dt = n_sub * timestep

    key_cb = getattr(dev, "key_callback", None)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()
                axes, events = dev.poll()
                if events["quit"]:
                    break
                if events["reset"]:
                    if mirror is not None:
                        # While mirroring, H is the HOLD/TRACK clutch, never a
                        # sim reset (a reset would command a jump to the rest
                        # pose on the real arm).
                        if mirror.hold:
                            tgt = target_state.get_target()
                            ref_q, _ = mirror.reference
                            delta = (float(np.abs(tgt[0] - ref_q).max())
                                     if tgt is not None else 0.0)
                            print(f"[teleop_sim] TRACK: real arm slews to the sim "
                                  f"target (max joint delta {delta:.2f} rad, "
                                  f"rate-limited)")
                            mirror.set_hold(False)
                        else:
                            mirror.set_hold(True)
                            print("[teleop_sim] HOLD: real arm parked; sim keeps "
                                  "previewing")
                    else:
                        cmd = reset_home()
                scale = 0.5 if events["slow"] else 1.0
                steps = events.get("steps")
                if steps is not None:
                    # Discrete increments: one fixed step per key press.
                    cmd.apply_delta(steps[:3] * args.lin_step * scale,
                                    steps[3:6] * args.ang_step * scale,
                                    float(steps[6]) * args.grip_step * scale)
                else:
                    cmd.apply_axes(axes, frame_dt,
                                   lin_speed=LIN_SPEED * scale,
                                   ang_speed=ANG_SPEED * scale,
                                   grip_speed=GRIP_SPEED * scale)
                quat = cmd.quat_wxyz
                q6_target = ik.step(data, cmd.pos, quat, frame_dt)
                target_state.set(q6_target, cmd.grip)

                if mocap_id >= 0:
                    data.mocap_pos[mocap_id] = cmd.pos
                    data.mocap_quat[mocap_id] = quat

                if args.kinematic:
                    q_ref[:], q_ref_vel[:], _ = advance_velocity_limited_reference(
                        q_ref, q_ref_vel, q6_target, ref_vlim, frame_dt)
                    grip_ref[:], grip_ref_vel[:], _ = advance_velocity_limited_reference(
                        grip_ref, grip_ref_vel, np.array([cmd.grip]), grip_vlim, frame_dt)
                    data.qpos[arm_qadr] = q_ref
                    data.qpos[grip_qadr] = GRIP_SIM_SCALE * grip_ref[0]
                    data.qvel[arm_dadr] = 0.0
                    data.qvel[grip_dadr] = 0.0
                    mujoco.mj_forward(model, data)
                else:
                    for _ in range(n_sub):
                        # Same low-level stack as the real arm: velocity-limited
                        # reference (mirror.py's shaping) + gravity-compensated
                        # PD (reference: mujoco_sync.py).
                        q_ref[:], q_ref_vel[:], _ = advance_velocity_limited_reference(
                            q_ref, q_ref_vel, q6_target, ref_vlim, timestep)
                        grip_ref[:], grip_ref_vel[:], _ = advance_velocity_limited_reference(
                            grip_ref, grip_ref_vel, np.array([cmd.grip]), grip_vlim, timestep)
                        tau = (data.qfrc_bias[arm_dadr]
                               + ARM_KP * (q_ref - data.qpos[arm_qadr])
                               - ARM_KD * data.qvel[arm_dadr])
                        data.ctrl[arm_act] = np.clip(tau, -ARM_TAU_LIMIT, ARM_TAU_LIMIT)
                        grip_target = GRIP_SIM_SCALE * grip_ref[0]
                        gtau = (GRIPPER_KP * (grip_target - data.qpos[grip_qadr[0]])
                                - GRIPPER_KD * data.qvel[grip_dadr[0]])
                        data.ctrl[grip_act] = np.clip(gtau, -GRIPPER_TAU_LIMIT, GRIPPER_TAU_LIMIT)
                        mujoco.mj_step(model, data)

                viewer.sync()
                leftover = frame_dt - (time.perf_counter() - t0)
                if leftover > 0:
                    time.sleep(leftover)
    except KeyboardInterrupt:
        pass  # fall through to the safe mirror stop below
    finally:
        close = getattr(dev, "close", None)
        if close is not None:
            close()
        if mirror is not None:
            print("[teleop_sim] stopping mirror: ramping real arm to rest pose...")
            mirror.stop(safe=True)


if __name__ == "__main__":
    main()
