"""Stream teleop joint targets to the real reBot B601-RS arm at 125 Hz.

RealArmMirror pulls (q6_target, grip_norm) from a caller-supplied
``get_target`` callable, advances a velocity-limited reference toward it
(teleop.motion_profiles) and sends MIT position commands with v_des = 0 and
the validated stiffness gains from the sdk config. The hardware surface is
injected through ``arm_factory`` so tests run against a fake with no sdk
import and no hardware.

Safety model:
- start(holding=True) parks the arm at its latched pose until set_hold(False)
  releases it — the deliberate step between enabling torque and motion.
- start() latches to the measured real pose and returns it so the caller can
  initialize the sim/teleop state to the real arm: the first commands hold
  the arm exactly where it is (no jumps).
- The 125 Hz loop never reads positions (get_positions does blocking
  feedback/SDO traffic) and never switches control mode; both happen once in
  start().
- Watchdog: a get_target failure keeps tracking the last good target; after
  1 s without a good target the reference freezes (PD hold, torque stays on).
- stop(safe=True) ramps the reference to q6 = zeros (rest pose, the arm
  extended onto its support) at 0.15 rad/s before disabling.
- stop(safe=False) holds the current pose for 0.5 s then cuts torque: the
  arm free-falls under gravity from any raised pose. Use only when the arm
  is already at rest or supported.

Typical use::

    mirror = RealArmMirror(get_target=teleop_state.target)
    q6_real, grip_norm_real = mirror.start()   # latch sim to this pose
    try:
        ...                                    # run the teleop/sim loop
    except KeyboardInterrupt:
        pass
    finally:
        mirror.stop(safe=True)                 # Ctrl-C lands here; a second
                                               # Ctrl-C skips the ramp and
                                               # disables immediately
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import yaml

from teleop.motion_profiles import advance_velocity_limited_reference


# MJCF joint ranges (contract): J2/J3 widened to [-0.2, 3.14].
JOINT_LIMITS_LOW = np.array([-2.8, -0.2, -0.2, -1.57, -1.57, -3.14])
JOINT_LIMITS_HIGH = np.array([2.8, 3.14, 3.14, 1.57, 1.57, 3.14])

GRIPPER_MOTOR_RANGE_RAD = 5.0      # grip_norm in [0,1] -> motor 0..5 rad
GRIPPER_SPEED_LIMIT_RAD_S = 3.0    # motor space, matches validated leader cfg

DEFAULT_RATE_HZ = 125.0
DEFAULT_SPEED_LIMIT_RAD_S = 0.3
STALE_TIMEOUT_S = 1.0
SAFE_STOP_SPEED_RAD_S = 0.15
SAFE_STOP_TIMEOUT_S = 30.0
UNSAFE_STOP_HOLD_S = 0.5
ESTOP_AFTER_CONSECUTIVE_FAILURES = 20
REST_TOLERANCE_RAD = 1e-6


@dataclass(frozen=True)
class MirrorGains:
    """MIT gains used for every send; v_des is always 0."""

    arm_kp: np.ndarray   # (6,)
    arm_kd: np.ndarray   # (6,)
    gripper_kp: float
    gripper_kd: float


def load_mit_gains(config_dir: str | Path | None = None) -> MirrorGains:
    """Read per-joint MIT kp/kd from the sdk config yaml files.

    Resolves sdk/configs/rebotarm.yaml -> hardware_yaml -> joints[*].MIT, so
    the mirror always streams with the same gains the sdk is configured for.
    """
    cfg_dir = (
        Path(config_dir)
        if config_dir is not None
        else Path(__file__).resolve().parent.parent / "sdk" / "configs"
    )
    top = yaml.safe_load((cfg_dir / "rebotarm.yaml").read_text())
    hw = yaml.safe_load((cfg_dir / str(top["hardware_yaml"])).read_text())
    joints = {j["name"]: j for j in hw["joints"]}
    arm_names = hw["groups"]["arm"]["joints"]
    if len(arm_names) != 6:
        raise RuntimeError(f"expected 6 arm joints in sdk config, got {arm_names}")
    grip_names = hw["groups"].get("gripper", {}).get("joints", [])
    if not grip_names:
        raise RuntimeError("sdk config has no gripper group; mirror requires one")
    arm_kp = np.array([float(joints[n]["MIT"]["kp"]) for n in arm_names])
    arm_kd = np.array([float(joints[n]["MIT"]["kd"]) for n in arm_names])
    grip = joints[grip_names[0]]
    return MirrorGains(
        arm_kp=arm_kp,
        arm_kd=arm_kd,
        gripper_kp=float(grip["MIT"]["kp"]),
        gripper_kd=float(grip["MIT"]["kd"]),
    )


def _coerce_gains(config: Any) -> MirrorGains:
    if config is None:
        return load_mit_gains()
    if isinstance(config, MirrorGains):
        return config
    if isinstance(config, dict):
        return MirrorGains(
            arm_kp=np.asarray(config["arm_kp"], dtype=np.float64).reshape(6),
            arm_kd=np.asarray(config["arm_kd"], dtype=np.float64).reshape(6),
            gripper_kp=float(config["gripper_kp"]),
            gripper_kd=float(config["gripper_kd"]),
        )
    raise TypeError(f"config must be None, dict or MirrorGains, got {type(config)!r}")


def _default_arm_factory() -> Any:
    # Imported lazily: the sdk package (and motorbridge) must not be a hard
    # dependency of importing this module or of running the headless tests.
    from sdk import RebotArm

    return RebotArm()


class RealArmMirror:
    """125 Hz joint-target mirror from a teleop source to the real arm.

    Parameters:
        get_target: callable returning (q6 radians (6,), grip_norm [0,1]),
            called from the mirror thread every tick; may return None or
            raise to signal "no fresh target". Must be cheap and non-blocking.
        arm_factory: zero-arg callable returning a RebotArm-shaped object
            (connect/disconnect/disable_all, groups .arm/.gripper with
            enable/mode_mit/send_mit/get_positions). Defaults to the real sdk.
        rate_hz: send rate (125 Hz is the sustainable CAN cadence).
        speed_limit: per-joint tracking velocity limit, rad/s (scalar or (6,)).
        config: MIT gains - None (load from sdk configs), a MirrorGains, or a
            dict with arm_kp/arm_kd/gripper_kp/gripper_kd.
    """

    def __init__(
        self,
        get_target: Callable[[], Optional[tuple[np.ndarray, float]]],
        arm_factory: Optional[Callable[[], Any]] = None,
        rate_hz: float = DEFAULT_RATE_HZ,
        speed_limit: float = DEFAULT_SPEED_LIMIT_RAD_S,
        config: Any = None,
    ) -> None:
        self._get_target = get_target
        self._arm_factory = arm_factory if arm_factory is not None else _default_arm_factory
        self._rate_hz = float(rate_hz)
        self._track_vlim = np.broadcast_to(
            np.asarray(speed_limit, dtype=np.float64), (6,)
        ).copy()
        self._gains = _coerce_gains(config)
        self._grip_kp = np.array([self._gains.gripper_kp])
        self._grip_kd = np.array([self._gains.gripper_kd])

        self._arm: Any = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._rest_evt = threading.Event()
        self._ramping = False

        # Reference state (owned by the mirror thread after start()).
        self._ref_q = np.zeros(6)
        self._ref_qd = np.zeros(6)
        self._ref_grip = np.zeros(1)   # motor radians, 0..5
        self._ref_grip_vel = np.zeros(1)
        self._target_q = np.zeros(6)
        self._target_grip = 0.0
        self._last_fresh = 0.0

        self.send_error_count = 0
        self.estopped = False
        self._consecutive_failures = 0

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self, holding: bool = False) -> tuple[np.ndarray, float]:
        """Connect, enable, enter MIT mode once, latch and start streaming.

        Returns (q6_real, grip_norm_real) measured from the arm so the caller
        can initialize sim/teleop state to the real pose; the internal
        reference starts there, so the arm holds its pose with no jump.

        holding=True starts with the hold clutch engaged (see set_hold): the
        arm PD-holds its latched pose and ignores targets until released.
        """
        print("[mirror] enabling motor torque on the real arm")
        if self._thread is not None:
            raise RuntimeError("RealArmMirror.start() called twice")

        arm = self._arm_factory()
        try:
            arm.connect()
            arm.arm.enable()
            arm.gripper.enable()
            # The only mode switches of the whole session.
            arm.arm.mode_mit()
            arm.gripper.mode_mit()
            # Latch read: retry — the sdk returns NaN for any joint whose
            # position could not be read, and latching a wrong pose means the
            # first MIT frame commands a snap.
            q6 = grip = None
            for attempt in range(3):
                q6 = np.asarray(arm.arm.get_positions(), dtype=np.float64).reshape(-1)
                grip = np.asarray(arm.gripper.get_positions(), dtype=np.float64).reshape(-1)
                if (q6.shape[0] >= 6 and np.all(np.isfinite(q6[:6]))
                        and grip.size and math.isfinite(float(grip[0]))):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(
                    f"could not read a complete arm pose after 3 attempts "
                    f"(arm={q6!r}, gripper={grip!r}); refusing to stream"
                )
            q6 = q6[:6].copy()
            grip_rad = float(grip[0])
        except Exception:
            for cleanup in (getattr(arm, "disable_all", None), getattr(arm, "disconnect", None)):
                if cleanup is not None:
                    try:
                        cleanup()
                    except Exception:
                        pass
            raise

        self._arm = arm
        self._ref_q = q6.copy()
        self._ref_qd = np.zeros(6)
        self._ref_grip = np.array([min(max(grip_rad, 0.0), GRIPPER_MOTOR_RANGE_RAD)])
        self._ref_grip_vel = np.zeros(1)
        # Initial target is the measured pose itself (unclipped): hold exactly
        # where the arm is until the first good external target arrives.
        self._target_q = q6.copy()
        self._target_grip = float(self._ref_grip[0])
        self._last_fresh = time.monotonic()
        self.send_error_count = 0
        self.estopped = False
        self._consecutive_failures = 0
        self._stop_evt.clear()
        self._rest_evt.clear()
        self._ramping = False
        self._hold = bool(holding)

        self._thread = threading.Thread(target=self._loop, name="rebot-mirror", daemon=True)
        self._thread.start()
        grip_norm_real = min(max(grip_rad / GRIPPER_MOTOR_RANGE_RAD, 0.0), 1.0)
        return q6.copy(), grip_norm_real

    def stop(self, safe: bool = True, timeout: float = SAFE_STOP_TIMEOUT_S) -> None:
        """Stop streaming and disable the arm.

        safe=True: ramp the joint reference to q6 = zeros (rest pose, arm
        extended onto its support) at 0.15 rad/s and wait until reached, then
        disable and disconnect. If the ramp is not done after ``timeout``
        seconds the arm KEEPS HOLDING (torque is never cut on a timer) and the
        operator is prompted; a KeyboardInterrupt (second Ctrl-C) abandons the
        ramp and disables immediately. The gripper holds its position during
        the ramp.

        safe=False: PD-hold the current reference for 0.5 s, then cut torque.
        WARNING: with torque cut the arm free-falls under gravity from any
        raised pose; use only at rest or with the arm supported.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            self._stop_evt.set()
            self._shutdown_arm()
            return
        try:
            if safe:
                with self._lock:
                    self._ramping = True
                deadline = time.monotonic() + timeout
                warned = False
                while thread.is_alive() and not self._rest_evt.is_set():
                    if not warned and time.monotonic() >= deadline:
                        # Never cut torque mid-ramp on a timer: keep holding
                        # and let the operator decide.
                        print(
                            "[mirror] rest ramp not finished after "
                            f"{timeout:.0f}s - HOLDING position. Press Ctrl-C "
                            "to cut torque now (arm will fall if unsupported)."
                        )
                        warned = True
                    self._rest_evt.wait(0.05)
            else:
                time.sleep(UNSAFE_STOP_HOLD_S)
        except KeyboardInterrupt:
            pass  # operator override: abandon the ramp, disable now
        finally:
            self._stop_evt.set()
            try:
                thread.join(2.0)
            finally:
                self._thread = None
                try:
                    self._shutdown_arm()
                except KeyboardInterrupt:
                    # A last Ctrl-C mid-shutdown must not leave torque on.
                    self._shutdown_arm()
                    raise

    @property
    def reference(self) -> tuple[np.ndarray, float]:
        """Snapshot of the current (q6 reference, gripper motor radians)."""
        with self._lock:
            return self._ref_q.copy(), float(self._ref_grip[0])

    @property
    def hold(self) -> bool:
        """True while the hold clutch is engaged (PD-holding, ignoring targets)."""
        with self._lock:
            return self._hold

    def set_hold(self, hold: bool) -> None:
        """Engage/release the hold clutch.

        Engaged: the arm PD-holds the current reference and ignores incoming
        targets. Released: the reference resumes slewing toward the latest
        target at the normal rate limit (no jump — the velocity-limited
        reference bridges any gap accumulated while holding).
        """
        with self._lock:
            self._hold = bool(hold)

    # ── internals ──────────────────────────────────────────────────────

    def _shutdown_arm(self) -> None:
        arm, self._arm = self._arm, None
        if arm is None:
            return
        try:
            arm.disable_all()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass

    def _poll_target(self, now: float) -> None:
        """Pull one target; invalid/failed pulls leave the last good target."""
        try:
            result = self._get_target()
        except Exception:
            return
        if result is None:
            return
        try:
            q6t, grip_norm = result
            q6t = np.asarray(q6t, dtype=np.float64).reshape(-1)
            gn = float(grip_norm)
        except Exception:
            return
        if q6t.shape != (6,) or not np.all(np.isfinite(q6t)) or not math.isfinite(gn):
            return
        self._target_q = np.clip(q6t, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        self._target_grip = GRIPPER_MOTOR_RANGE_RAD * min(max(gn, 0.0), 1.0)
        self._last_fresh = now

    def _loop(self) -> None:
        period = 1.0 / self._rate_hz
        arm_group = self._arm.arm
        grip_group = self._arm.gripper
        zeros6 = np.zeros(6)
        zeros1 = np.zeros(1)
        ramp_vlim = np.full(6, SAFE_STOP_SPEED_RAD_S)
        grip_vlim = np.array([GRIPPER_SPEED_LIMIT_RAD_S])

        last = time.monotonic()
        next_t = last + period
        while not self._stop_evt.is_set():
            now = time.monotonic()
            dt = now - last
            last = now

            with self._lock:
                ramping = self._ramping
                holding = self._hold

            if ramping:
                goal_q = zeros6
                goal_grip = float(self._ref_grip[0])  # hold gripper on ramp
                vlim = ramp_vlim
            elif holding:
                # Hold clutch: PD-hold the frozen reference, ignore targets.
                goal_q = self._ref_q
                goal_grip = float(self._ref_grip[0])
                vlim = self._track_vlim
            else:
                self._poll_target(now)
                if now - self._last_fresh > STALE_TIMEOUT_S:
                    # Stale: freeze the reference where it is (PD hold).
                    goal_q = self._ref_q
                    goal_grip = float(self._ref_grip[0])
                else:
                    goal_q = self._target_q
                    goal_grip = self._target_grip
                vlim = self._track_vlim

            with self._lock:
                self._ref_q, self._ref_qd, _ = advance_velocity_limited_reference(
                    self._ref_q, self._ref_qd, goal_q, vlim, dt
                )
                self._ref_grip, self._ref_grip_vel, _ = advance_velocity_limited_reference(
                    self._ref_grip, self._ref_grip_vel, np.array([goal_grip]), grip_vlim, dt
                )

            ok = True
            try:
                arm_group.send_mit(
                    self._ref_q, vel=zeros6, kp=self._gains.arm_kp, kd=self._gains.arm_kd
                )
                # The sdk swallows per-motor CallError to keep the other
                # joints updating; it reports the count here instead.
                if getattr(arm_group, "last_send_failures", 0):
                    ok = False
                    self.send_error_count += arm_group.last_send_failures
            except Exception:
                ok = False
                self.send_error_count += 1
            try:
                grip_group.send_mit(
                    self._ref_grip, vel=zeros1, kp=self._grip_kp, kd=self._grip_kd
                )
                if getattr(grip_group, "last_send_failures", 0):
                    ok = False
                    self.send_error_count += grip_group.last_send_failures
            except Exception:
                ok = False
                self.send_error_count += 1

            if ok:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= ESTOP_AFTER_CONSECUTIVE_FAILURES:
                    # Sustained bus failure: cut torque rather than keep
                    # commanding blind.
                    self.estopped = True
                    try:
                        self._arm.disable_all()
                    except Exception:
                        pass
                    self._stop_evt.set()
                    return

            if ramping and not self._rest_evt.is_set() and np.all(
                np.abs(self._ref_q) <= REST_TOLERANCE_RAD
            ):
                self._rest_evt.set()

            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()  # fell behind; do not spiral
