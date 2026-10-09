#!/usr/bin/env python
"""Headless check of teleop.mirror.RealArmMirror against a fake arm.

No hardware, no sdk import, no viewer. Run with the project interpreter:

    /Users/krm/miniconda3/envs/rebot_teleop/bin/python scripts/check_mirror.py

Covers: ungated start, the real-pose latch, the hold clutch, velocity-bounded
monotone tracking with v_des == 0 on every send, staleness freeze, resume,
stop(safe=True) rest ramp with disable last, stop(safe=False) hold+disable,
loop hygiene (no get_positions / mode switches inside the loop), and the sdk
config gain loader.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teleop.mirror import (  # noqa: E402
    GRAVITY_FF_CLIP,
    MirrorGains,
    RealArmMirror,
    load_mit_gains,
)
import teleop.leader  # noqa: E402,F401  (importability: no hardware at import)
import teleop.leader_policy  # noqa: E402,F401
import teleop.motion_profiles  # noqa: E402,F401

GAINS = {
    "arm_kp": [80.0, 150.0, 150.0, 50.0, 50.0, 50.0],
    "arm_kd": [5.0, 10.0, 10.0, 5.0, 4.0, 4.0],
    "gripper_kp": 50.0,
    "gripper_kd": 4.0,
}
Q0 = np.array([0.05, 0.10, 0.15, -0.05, 0.05, 0.0])
G0_RAD = 2.5  # gripper motor radians -> grip_norm 0.5
SPEED_LIMIT = 0.3
DT_SLACK_S = 0.01  # scheduling jitter allowance on send-to-send dt

_checks = 0


def check(cond: bool, msg: str) -> None:
    global _checks
    _checks += 1
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


class FakeGroup:
    def __init__(self, owner: "FakeArm", name: str, positions) -> None:
        self.owner = owner
        self.name = name
        self._positions = np.asarray(positions, dtype=np.float64)
        self.sends: list[tuple] = []  # (t, pos, vel, kp, kd)
        self.get_positions_calls = 0
        self.mode_mit_calls = 0
        self.enable_calls = 0

    def enable(self) -> None:
        self.enable_calls += 1

    def disable(self) -> None:
        pass

    def mode_mit(self, kp=None, kd=None) -> bool:
        self.mode_mit_calls += 1
        return True

    def send_mit(self, pos, vel=None, kp=None, kd=None, tau=None) -> None:
        self.sends.append(
            (
                time.monotonic(),
                np.array(pos, dtype=np.float64).copy(),
                None if vel is None else np.array(vel, dtype=np.float64).copy(),
                None if kp is None else np.array(kp, dtype=np.float64).copy(),
                None if kd is None else np.array(kd, dtype=np.float64).copy(),
                None if tau is None else np.array(tau, dtype=np.float64).copy(),
            )
        )
        self.owner.events.append(("send", self.name))

    def get_positions(self) -> np.ndarray:
        self.get_positions_calls += 1
        return self._positions.copy()


class FakeArm:
    def __init__(self, q0, g0_rad) -> None:
        self.events: list[tuple] = []
        self.arm = FakeGroup(self, "arm", q0)
        self.gripper = FakeGroup(self, "gripper", [g0_rad])
        self.connected = False

    def connect(self) -> None:
        self.connected = True
        self.events.append(("connect",))

    def disable_all(self) -> None:
        self.events.append(("disable_all",))

    def disconnect(self) -> None:
        self.connected = False
        self.events.append(("disconnect",))


class TargetSource:
    """Mutable get_target provider; optionally supplies gravity tau."""

    def __init__(self, q: np.ndarray, grip_norm: float) -> None:
        self.q = q.copy()
        self.grip_norm = grip_norm
        self.tau = None
        self.fail = False

    def __call__(self):
        if self.fail:
            raise RuntimeError("input source dead")
        if self.tau is not None:
            return self.q.copy(), self.grip_norm, self.tau.copy()
        return self.q.copy(), self.grip_norm


def arm_sends(fake: FakeArm):
    return fake.arm.sends


def assert_speed_bound(sends, vlim: float, label: str) -> None:
    worst = 0.0
    for (t0, p0, *_), (t1, p1, *_) in zip(sends, sends[1:]):
        dt = t1 - t0
        allowed = vlim * (min(dt, 0.1) + DT_SLACK_S)
        step = float(np.max(np.abs(p1 - p0)))
        worst = max(worst, step - allowed)
        if step > allowed:
            raise AssertionError(
                f"{label}: step {step:.5f} rad exceeds {vlim}*dt bound "
                f"{allowed:.5f} (dt={dt * 1e3:.2f} ms)"
            )
    check(True, f"{label}: per-tick steps bounded by {vlim}*dt ({len(sends)} sends)")


def assert_monotone_toward(sends, target: np.ndarray, label: str) -> None:
    for (_, p0, *_), (_, p1, *_) in zip(sends, sends[1:]):
        d0 = np.abs(target - p0)
        d1 = np.abs(target - p1)
        if np.any(d1 > d0 + 1e-9):
            raise AssertionError(f"{label}: reference moved away from target")
    check(True, f"{label}: references move monotonically toward target")


def main() -> int:
    print("[1] start without any env gate")
    fake = FakeArm(Q0, G0_RAD)
    factory_calls: list[int] = []

    def factory() -> FakeArm:
        factory_calls.append(1)
        return fake

    src = TargetSource(Q0, G0_RAD / 5.0)
    mirror = RealArmMirror(src, arm_factory=factory, config=GAINS)

    print("[2] latch to real pose")
    q_latch, g_latch = mirror.start()
    check(len(factory_calls) == 1, "start() constructs the arm exactly once (no env gate)")
    check(np.allclose(q_latch, Q0), "start() returns the arm's measured q6")
    check(abs(g_latch - 0.5) < 1e-12, "start() returns grip_norm from motor radians / 5")
    check(fake.arm.mode_mit_calls == 1 and fake.gripper.mode_mit_calls == 1,
          "mode_mit called exactly once per group at startup")
    check(fake.arm.enable_calls == 1 and fake.gripper.enable_calls == 1,
          "enable called once per group")
    time.sleep(0.2)
    check(len(arm_sends(fake)) >= 10, "loop is streaming")
    for _, p, *_ in arm_sends(fake):
        if not np.allclose(p, Q0, atol=1e-12):
            raise AssertionError("reference moved while target == latched pose")
    check(True, "no jump: holds the latched pose before any new target")

    print("[3] step target tracking")
    step = np.array([0.10, -0.10, 0.08, 0.06, -0.06, 0.04])
    q1 = Q0 + step
    i0 = len(arm_sends(fake))
    g0_grip = len(fake.gripper.sends)
    t_change = time.monotonic()
    src.q = q1
    src.grip_norm = 0.9
    time.sleep(0.8)
    sends = arm_sends(fake)[i0:]
    tracked = [s for s in sends if s[0] > t_change + 0.05]
    check(len(tracked) > 20, "enough sends recorded during tracking")
    assert_speed_bound(sends, SPEED_LIMIT, "tracking")
    assert_monotone_toward(tracked, q1, "tracking")
    check(np.allclose(sends[-1][1], q1, atol=1e-9), "arm reference converged to the step target")
    grip_sends = fake.gripper.sends[g0_grip:]
    assert_speed_bound(grip_sends, 3.0, "gripper tracking")
    check(abs(float(fake.gripper.sends[-1][1][0]) - 4.5) < 1e-9,
          "gripper reference converged to 5.0*grip_norm")
    for group in (fake.arm, fake.gripper):
        for _, _, vel, kp, kd, *_tau in group.sends:
            if vel is None or np.any(vel != 0.0):
                raise AssertionError(f"{group.name}: send with v_des != 0")
            if kp is None or kd is None:
                raise AssertionError(f"{group.name}: send without explicit gains")
    check(True, "every send has v_des == 0 and explicit gains")
    check(np.allclose(fake.arm.sends[-1][3], GAINS["arm_kp"])
          and np.allclose(fake.arm.sends[-1][4], GAINS["arm_kd"])
          and float(fake.gripper.sends[-1][3][0]) == GAINS["gripper_kp"]
          and float(fake.gripper.sends[-1][4][0]) == GAINS["gripper_kd"],
          "sends carry the configured MIT gains")

    print("[4] staleness freeze")
    q2 = q1 + np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    src.q = q2
    time.sleep(0.25)          # move toward q2 for a while...
    src.fail = True           # ...then the input source dies
    t_fail = time.monotonic()
    time.sleep(1.4)           # < 1 s: keeps tracking last target; > 1 s: freeze
    frozen = [s for s in arm_sends(fake) if s[0] > t_fail + 1.15]
    check(len(frozen) > 10, "loop keeps streaming (PD hold) while stale")
    ref_frozen = frozen[0][1]
    for _, p, *_ in frozen:
        if not np.array_equal(p, ref_frozen):
            raise AssertionError("reference changed while frozen")
    check(True, "reference frozen after 1 s without a fresh target")
    check(abs(ref_frozen[0] - q2[0]) > 0.05, "freeze happened before reaching the stale target")
    check(not mirror.estopped and mirror.send_error_count == 0,
          "staleness did not trip the estop or count send errors")

    print("[5] resume after staleness")
    src.q = q1
    src.fail = False
    # The reference kept tracking toward q2 for the 1 s pre-freeze window,
    # so it is ~0.375 rad past q1 on J1: allow 1.25 s + margin to come back.
    time.sleep(1.6)
    check(np.allclose(arm_sends(fake)[-1][1], q1, atol=1e-9),
          "tracking resumes smoothly after the source recovers")

    print("[6] stop(safe=True) rest ramp")
    i_stop = len(arm_sends(fake))
    grip_before_stop = float(fake.gripper.sends[-1][1][0])
    t0 = time.monotonic()
    mirror.stop(safe=True, timeout=10.0)
    stop_duration = time.monotonic() - t0
    ramp = arm_sends(fake)[max(i_stop - 1, 0):]
    check(len(ramp) > 20, "ramp streamed")
    assert_speed_bound(ramp, 0.15, "rest ramp")
    assert_monotone_toward(ramp, np.zeros(6), "rest ramp")
    check(np.allclose(ramp[-1][1], np.zeros(6), atol=1e-9), "final arm reference is q6 = zeros")
    check(abs(float(fake.gripper.sends[-1][1][0]) - grip_before_stop) < 1e-9,
          "gripper held (not ramped) during the rest ramp")
    expected = float(np.max(np.abs(q1))) / 0.15
    check(expected * 0.8 < stop_duration < expected + 3.0,
          f"stop took {stop_duration:.2f} s (ramp at 0.15 rad/s needs ~{expected:.2f} s)")
    event_names = [e[0] for e in fake.events]
    check("disable_all" in event_names and "disconnect" in event_names,
          "disable_all and disconnect were called")
    i_disable = event_names.index("disable_all")
    check(i_disable > 0 and event_names[i_disable - 1] == "send"
          and "send" not in event_names[i_disable:],
          "disable_all comes after the last send; nothing sent afterwards")
    check(event_names.index("disconnect") > i_disable, "disconnect after disable_all")
    check(fake.arm.get_positions_calls == 1 and fake.gripper.get_positions_calls == 1,
          "get_positions never called inside the 125 Hz loop")
    check(fake.arm.mode_mit_calls == 1 and fake.gripper.mode_mit_calls == 1,
          "no mode switches inside the loop")
    rate = len(arm_sends(fake)) / (arm_sends(fake)[-1][0] - arm_sends(fake)[0][0])
    check(100.0 < rate < 150.0, f"send cadence ~125 Hz (measured {rate:.1f} Hz)")

    print("[7] stop(safe=False) hold then disable")
    fake2 = FakeArm(Q0, G0_RAD)
    src2 = TargetSource(Q0, 0.5)
    mirror2 = RealArmMirror(src2, arm_factory=lambda: fake2, config=GAINS)
    mirror2.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    mirror2.stop(safe=False)
    held = time.monotonic() - t0
    check(0.45 <= held < 2.0, f"unsafe stop held for ~0.5 s ({held:.2f} s)")
    check(np.allclose(fake2.arm.sends[-1][1], Q0, atol=1e-12),
          "unsafe stop never ramped (torque cut at current pose)")
    names2 = [e[0] for e in fake2.events]
    check("disable_all" in names2 and "send" not in names2[names2.index("disable_all"):],
          "unsafe stop disables after the last send")

    print("[8] sdk config gain loader")
    gains = load_mit_gains()
    check(isinstance(gains, MirrorGains)
          and gains.arm_kp.shape == (6,) and gains.arm_kd.shape == (6,)
          and np.all(gains.arm_kp > 0) and np.all(gains.arm_kd > 0)
          and gains.gripper_kp > 0 and gains.gripper_kd > 0
          and math.isfinite(gains.gripper_kp),
          "load_mit_gains() resolves 6 arm + 1 gripper MIT gain sets from sdk/configs")

    print("[9] hold clutch (start holding / H-toggle semantics)")
    fake3 = FakeArm(Q0, G0_RAD)
    src3 = TargetSource(Q0, G0_RAD / 5.0)
    mirror3 = RealArmMirror(src3, arm_factory=lambda: fake3, config=GAINS)
    _q, _g = mirror3.start(holding=True)
    check(mirror3.hold, "start(holding=True) engages the clutch")
    held_target = Q0 + np.array([0.20, -0.20, 0.15, 0.10, -0.10, 0.05])
    src3.q = held_target
    time.sleep(0.3)
    for _, p, *_ in arm_sends(fake3):
        if not np.allclose(p, Q0, atol=1e-12):
            raise AssertionError("holding: reference moved despite target change")
    check(len(arm_sends(fake3)) > 10, "holding: streams a frozen PD-hold reference")
    i0 = len(arm_sends(fake3))
    mirror3.set_hold(False)
    time.sleep(0.8)
    sends3 = arm_sends(fake3)[i0:]
    assert_speed_bound(sends3, SPEED_LIMIT, "post-release catch-up")
    check(float(np.max(np.abs(sends3[-1][1] - Q0))) > 0.05,
          "released: reference slews toward the pending target")
    mirror3.set_hold(True)
    time.sleep(0.05)
    i1 = len(arm_sends(fake3))
    src3.q = Q0
    time.sleep(0.25)
    refrozen = [p for _, p, *_ in arm_sends(fake3)[i1:]]
    for p in refrozen[1:]:
        if not np.allclose(p, refrozen[0], atol=1e-12):
            raise AssertionError("re-hold: reference moved")
    check(len(refrozen) > 5, "re-hold freezes the reference again")
    mirror3.stop(safe=False)

    print("[10] gravity feedforward pass-through + clip")
    fake4 = FakeArm(Q0, G0_RAD)
    src4 = TargetSource(Q0, G0_RAD / 5.0)
    mirror4 = RealArmMirror(src4, arm_factory=lambda: fake4, config=GAINS)
    mirror4.start()
    time.sleep(0.15)
    last_tau = arm_sends(fake4)[-1][5]
    check(last_tau is not None and np.allclose(last_tau, 0.0),
          "2-tuple source: tau feedforward defaults to zeros")
    tau_in = np.array([5.0, 10.0, -10.0, 3.0, -3.0, 1.0])
    src4.tau = tau_in
    time.sleep(0.2)
    check(np.allclose(arm_sends(fake4)[-1][5], tau_in),
          "3-tuple source: gravity tau reaches send_mit unchanged (within clip)")
    src4.tau = np.array([100.0, -100.0, 100.0, 50.0, -50.0, 50.0])
    time.sleep(0.2)
    check(np.allclose(np.abs(arm_sends(fake4)[-1][5]), GRAVITY_FF_CLIP),
          "oversized tau is clipped to GRAVITY_FF_CLIP")
    mirror4.stop(safe=False)

    print(f"\nALL GREEN: {_checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
