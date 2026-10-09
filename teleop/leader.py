"""Phase-2 leader-arm input source: a second arm hand-guided as the leader.

STATUS: SKELETON - NOT YET TESTED ON HARDWARE. The structure (sampling
thread, mapping, get_target contract) is in place; the degree/radian
conventions of the leader readback and the leader's zero/direction pairing
must be verified against the physical leader before first use.

A LeaderArm owns a second RebotArm on its own transport/channel (separate
from the follower the mirror drives), samples its joint positions at ~50 Hz,
runs them through teleop.leader_policy.LeaderMapping (direction signs,
relative latching, jump rejection, gripper scaling) and exposes the result
through ``get_target()`` in exactly the shape RealArmMirror consumes:

    leader = LeaderArm(arm_factory=lambda: RebotArm("rebotarm_leader.yaml"))
    mirror = RealArmMirror(get_target=leader.get_target)
    q6_real, grip_norm_real = mirror.start()
    leader.start(follower_q6=q6_real, follower_grip_norm=grip_norm_real)

Reading positions at 50 Hz does blocking feedback/SDO traffic, which is why
the leader must be on its own channel: it never shares a bus with the
125 Hz mirror stream. The leader's motors are never enabled (no torque);
the arm is moved by hand.

No hardware is touched at import time; the sdk import happens lazily inside
the default factory.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from teleop.leader_policy import LeaderMapping

DEFAULT_RATE_HZ = 50.0
GRIPPER_MOTOR_RANGE_RAD = 5.0
# get_target() reports None once its newest mapped sample is older than this,
# so RealArmMirror's own staleness watchdog (1 s) takes over if the sampling
# thread stalls or dies.
SAMPLE_TTL_S = 0.5


def _default_arm_factory() -> Any:
    # Lazy: importing teleop.leader must not require sdk/motorbridge.
    from sdk import RebotArm

    return RebotArm()


class LeaderArm:
    """Samples a hand-guided leader arm and maps it to follower targets.

    NOT YET TESTED ON HARDWARE.

    Parameters:
        arm_factory: zero-arg callable returning the leader RebotArm-shaped
            object (connect/disconnect, groups .arm/.gripper with
            get_positions). It must address the leader's own channel.
        rate_hz: leader sampling rate (~50 Hz).
        absolute: passed to LeaderMapping; False (default) latches
            relative motion from the poses given to start().
    """

    def __init__(
        self,
        arm_factory: Optional[Callable[[], Any]] = None,
        rate_hz: float = DEFAULT_RATE_HZ,
        absolute: bool = False,
    ) -> None:
        self._arm_factory = arm_factory if arm_factory is not None else _default_arm_factory
        self._rate_hz = float(rate_hz)
        self._absolute = bool(absolute)

        self._arm: Any = None
        self._mapping: Optional[LeaderMapping] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

        self._latest_q6: Optional[np.ndarray] = None
        self._latest_grip_norm = 0.0
        self._latest_time = 0.0
        self.rejected_samples = 0  # mapping jump rejections (>25 deg/frame)

    def start(self, follower_q6: np.ndarray, follower_grip_norm: float) -> None:
        """Connect the leader, latch the mapping, start the sampling thread.

        ``follower_q6`` / ``follower_grip_norm`` are the follower's real pose
        as returned by RealArmMirror.start(), so relative leader motion is
        applied on top of where the follower actually is (no jump on engage).

        The leader's motors are intentionally never enabled.
        """
        if self._thread is not None:
            raise RuntimeError("LeaderArm.start() called twice")
        arm = self._arm_factory()
        arm.connect()
        try:
            leader_deg = self._sample_leader_degrees(arm)
            self._mapping = LeaderMapping(
                leader=leader_deg,
                follower=np.asarray(follower_q6, dtype=np.float64).reshape(6),
                gripper=GRIPPER_MOTOR_RANGE_RAD
                * min(max(float(follower_grip_norm), 0.0), 1.0),
                absolute=self._absolute,
            )
        except Exception:
            try:
                arm.disconnect()
            except Exception:
                pass
            raise
        self._arm = arm
        # Before the first mapped sample, feed the latched follower pose.
        with self._lock:
            self._latest_q6 = np.asarray(follower_q6, dtype=np.float64).reshape(6).copy()
            self._latest_grip_norm = min(max(float(follower_grip_norm), 0.0), 1.0)
            self._latest_time = time.monotonic()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="rebot-leader", daemon=True)
        self._thread.start()

    def get_target(self) -> Optional[tuple[np.ndarray, float]]:
        """Latest mapped (q6_target, grip_norm), or None when stale/not ready.

        Shape-compatible with RealArmMirror(get_target=...). Returning None
        on staleness lets the mirror's watchdog freeze the follower.
        """
        with self._lock:
            if self._latest_q6 is None:
                return None
            if time.monotonic() - self._latest_time > SAMPLE_TTL_S:
                return None
            return self._latest_q6.copy(), self._latest_grip_norm

    def stop(self) -> None:
        """Stop sampling and disconnect the leader (its motors were never on)."""
        self._stop_evt.set()
        thread = self._thread
        if thread is not None:
            thread.join(2.0)
            self._thread = None
        arm, self._arm = self._arm, None
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass

    # ── internals ──────────────────────────────────────────────────────

    @staticmethod
    def _sample_leader_degrees(arm: Any) -> list[float]:
        """Read the 7 leader angles (6 arm + gripper) in degrees.

        UNVERIFIED ON HARDWARE: LeaderMapping works in the Wiki leader's
        degree convention; a RebotArm leader reads back radians, converted
        here. Zero offsets and direction signs must be checked on the
        physical pairing before trusting this.
        """
        q_arm = np.asarray(arm.arm.get_positions(), dtype=np.float64).reshape(-1)[:6]
        q_grip = np.asarray(arm.gripper.get_positions(), dtype=np.float64).reshape(-1)
        grip = float(q_grip[0]) if q_grip.size else 0.0
        values = [math.degrees(v) for v in q_arm] + [math.degrees(grip)]
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError(f"non-finite leader readback: {values!r}")
        return values

    def _loop(self) -> None:
        period = 1.0 / self._rate_hz
        next_t = time.monotonic() + period
        while not self._stop_evt.is_set():
            try:
                leader_deg = self._sample_leader_degrees(self._arm)
                q6_list, grip_rad = self._mapping.map(leader_deg)
            except ValueError:
                # Jump rejection (reset/turn discontinuity): keep the last
                # good target; the mirror freezes if this persists.
                self.rejected_samples += 1
            except Exception:
                pass  # transient read failure; staleness handles persistence
            else:
                with self._lock:
                    self._latest_q6 = np.asarray(q6_list, dtype=np.float64)
                    self._latest_grip_norm = min(
                        max(grip_rad / GRIPPER_MOTOR_RANGE_RAD, 0.0), 1.0
                    )
                    self._latest_time = time.monotonic()
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
                next_t += period
            else:
                next_t = time.monotonic() + period
