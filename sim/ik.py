"""Velocity-level damped-least-squares CLIK for the reBot B601-RS arm.

Tracks a 6D pose target for the MJCF site "tcp" using the 6 arm joints only
(the model also contains gripper and object joints, which are ignored).
Quaternions are MuJoCo convention: wxyz.
"""
from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
)


class IKTracker:
    """Closed-loop IK integrating its own joint state (never writes the caller's MjData).

    Per :meth:`step`, a few Gauss-Newton inner iterations of
    ``dq = J^T (J J^T + lam^2 I)^-1 e`` are applied to an internal ``q_ik``
    state, with adaptive damping near singularities (pattern from the browser
    reference simulator: quadratic blend between min/max damping by
    singularity proximity), a per-joint velocity cap, and joint-range
    clamping with a safety margin.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str = "tcp",
        joint_names: Sequence[str] = ARM_JOINT_NAMES,
        iters: int = 4,
        vel_limit: float = 2.8,
        limit_margin: float = 0.02,
        min_damping: float = 0.018,
        max_damping: float = 0.075,
        singularity_threshold: float = 0.08,
    ) -> None:
        self._model = model
        self.iters = int(iters)
        self.vel_limit = float(vel_limit)
        self.min_damping = float(min_damping)
        self.max_damping = float(max_damping)
        self.singularity_threshold = float(singularity_threshold)

        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self._site_id < 0:
            raise ValueError(f"site {site!r} not found in model")

        joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} not found in model")
            joint_ids.append(jid)
        self._qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self._dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
        n = len(joint_ids)

        rng = model.jnt_range[joint_ids]
        self._lo = rng[:, 0] + limit_margin
        self._hi = rng[:, 1] - limit_margin

        # Private FK scratch: step() must not touch the caller's physics state.
        self._data = mujoco.MjData(model)
        mujoco.mj_resetData(model, self._data)

        # Preallocated work arrays (step() is called every frame).
        self._q = np.zeros(n)
        self._q_start = np.zeros(n)
        self._step_lo = np.zeros(n)
        self._step_hi = np.zeros(n)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._jac = np.zeros((6, n))
        self._err = np.zeros(6)
        self._target_quat = np.zeros(4)
        self._site_quat = np.zeros(4)
        self._neg_quat = np.zeros(4)
        self._err_quat = np.zeros(4)

    @property
    def q(self) -> np.ndarray:
        """Current internal joint state (copy), radians, joint1..joint6."""
        return self._q.copy()

    def reset(self, q6: np.ndarray) -> None:
        """Re-seed the internal IK state (e.g. from measured/sim joint positions)."""
        np.clip(q6, self._lo, self._hi, out=self._q)

    def step(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance the internal joint state toward the 6D target; return q6 target.

        ``data`` is accepted for API stability but not read: FK runs on the
        internal integrated state so IK tracking is independent of physics lag.
        """
        self._target_quat[:] = target_quat_wxyz
        mujoco.mju_normalize4(self._target_quat)

        # Velocity cap is enforced on the total step, not per inner iteration.
        self._q_start[:] = self._q

        for _ in range(self.iters):
            self._fk()
            err = self._pose_error(target_pos)
            if np.dot(err[:3], err[:3]) < 1e-10 and np.dot(err[3:], err[3:]) < 1e-8:
                break

            mujoco.mj_jacSite(self._model, self._data, self._jacp, self._jacr, self._site_id)
            J = self._jac
            J[:3] = self._jacp[:, self._dof_adr]
            J[3:] = self._jacr[:, self._dof_adr]

            JJT = J @ J.T
            lam = self._adaptive_damping(J)
            JJT.flat[:: JJT.shape[0] + 1] += lam * lam
            try:
                y = np.linalg.solve(JJT, err)
            except np.linalg.LinAlgError:
                break
            self._q += J.T @ y
            # Scale the total frame excursion uniformly to the velocity cap:
            # corner-clamping each joint distorts the step direction and
            # causes period-2 limit cycles on unreachable targets.
            excursion = self._q - self._q_start
            worst = np.abs(excursion).max()
            cap = self.vel_limit * dt
            if worst > cap > 0.0:
                self._q[:] = self._q_start + excursion * (cap / worst)
            np.clip(self._q, self._lo, self._hi, out=self._q)

        return self._q.copy()

    def _fk(self) -> None:
        self._data.qpos[self._qpos_adr] = self._q
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)  # cdof, needed by mj_jacSite

    def _pose_error(self, target_pos: np.ndarray) -> np.ndarray:
        """6D error [p_err; rot_err], world frame; rot_err is axis*angle."""
        err = self._err
        err[:3] = target_pos - self._data.site_xpos[self._site_id]
        mujoco.mju_mat2Quat(self._site_quat, self._data.site_xmat[self._site_id])
        mujoco.mju_negQuat(self._neg_quat, self._site_quat)
        mujoco.mju_mulQuat(self._err_quat, self._target_quat, self._neg_quat)
        if self._err_quat[0] < 0.0:  # shorter arc
            self._err_quat *= -1.0
        mujoco.mju_quat2Vel(err[3:], self._err_quat, 1.0)
        return err

    def _adaptive_damping(self, J: np.ndarray) -> float:
        """Blend damping up as the smallest singular value approaches zero."""
        sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
        proximity = 1.0 - min(sigma_min / self.singularity_threshold, 1.0)
        return self.min_damping + (self.max_damping - self.min_damping) * proximity * proximity
