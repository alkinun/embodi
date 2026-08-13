from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .environment import SO101PickPlaceEnv


class Phase(str, Enum):
    OPEN = "open"
    PREGRASP = "pregrasp"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    TRANSIT = "transit"
    RELEASE = "release"
    RETREAT = "retreat"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ExpertStatus:
    phase: Phase
    position_error: float
    ticks_in_phase: int


class PrivilegedPickPlaceExpert:
    def __init__(self, env: SO101PickPlaceEnv, frequency: float = 30.0) -> None:
        self.env = env
        self.frequency = frequency
        mujoco = env.mujoco
        self.site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.arm_joint_ids = env.joint_ids[:5]
        self.arm_dofs = env.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_qpos = env.qpos_addresses[:5]
        self.lower = env.model.jnt_range[self.arm_joint_ids, 0] + np.deg2rad(2.0)
        self.upper = env.model.jnt_range[self.arm_joint_ids, 1] - np.deg2rad(2.0)
        self.reference_rotation = env.data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.reference_approach = self.reference_rotation[:, 0].copy()
        self.phase = Phase.OPEN
        self.phase_tick = 0
        self.total_tick = 0
        self.last_error = float("inf")
        self.grasp_site = np.zeros(3)
        self.release_site = np.zeros(3)

    @property
    def terminal(self) -> bool:
        return self.phase in (Phase.DONE, Phase.FAILED)

    def status(self) -> ExpertStatus:
        return ExpertStatus(self.phase, self.last_error, self.phase_tick)

    def _transition(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_tick = 0

    @staticmethod
    def _rotation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.cross(current[:, 0], desired[:, 0])
            + np.cross(current[:, 1], desired[:, 1])
            + np.cross(current[:, 2], desired[:, 2])
        )

    def _ik_action(
        self,
        goal: np.ndarray,
        gripper: float,
        max_speed: float = 0.10,
        preserve_tilt: bool = True,
    ) -> np.ndarray:
        env = self.env
        mujoco = env.mujoco
        site_position = env.data.site_xpos[self.site_id]
        error = np.asarray(goal) - site_position
        self.last_error = float(np.linalg.norm(error))
        velocity = 5.0 * error
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity *= max_speed / speed

        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, self.site_id)
        position_jacobian = jacp[:, self.arm_dofs]
        if preserve_tilt:
            rotation = env.data.site_xmat[self.site_id].reshape(3, 3)
            rotation_error = self._rotation_error(rotation, self.reference_rotation)
            characteristic_length = 0.06
            jacobian = np.vstack((position_jacobian, characteristic_length * jacr[:2, self.arm_dofs]))
            task_velocity = np.concatenate((velocity, characteristic_length * 3.0 * rotation_error[:2]))
        else:
            jacobian = position_jacobian
            task_velocity = velocity
        damping = 0.025
        qdot = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0]),
            task_velocity,
        )
        qdot = np.clip(qdot, -1.0, 1.0)
        measured = env.data.qpos[self.arm_qpos]
        command = np.clip(measured + qdot / self.frequency, self.lower, self.upper)
        action = np.empty(6, dtype=np.float32)
        action[:5] = np.rad2deg(command)
        action[5] = gripper
        return action

    def action(self) -> np.ndarray:
        env = self.env
        cube = env.data.xpos[env.cube_body_id].copy()
        target = env.data.xpos[env.target_body_id].copy()
        site = env.data.site_xpos[self.site_id].copy()
        action: np.ndarray

        if self.phase == Phase.OPEN:
            action = env.state()
            action[5] = 16.0
            self.last_error = 0.0
            if self.phase_tick >= 18:
                self._transition(Phase.PREGRASP)

        elif self.phase == Phase.PREGRASP:
            goal = cube - self.reference_approach * 0.065
            action = self._ik_action(goal, 16.0, max_speed=0.12)
            if self.last_error < 0.012:
                self._transition(Phase.DESCEND)
            elif self.phase_tick > 120:
                self._transition(Phase.FAILED)

        elif self.phase == Phase.DESCEND:
            goal = cube + np.array([0.0, 0.0, 0.002])
            action = self._ik_action(goal, 16.0, max_speed=0.045)
            if self.last_error < 0.008:
                self.grasp_site = site.copy()
                self._transition(Phase.CLOSE)
            elif self.phase_tick > 90:
                self._transition(Phase.FAILED)

        elif self.phase == Phase.CLOSE:
            action = self._ik_action(self.grasp_site, 0.0, max_speed=0.03)
            if self.phase_tick >= 24:
                self._transition(Phase.LIFT)

        elif self.phase == Phase.LIFT:
            goal = np.array([site[0], site[1], max(0.12, cube[2] + 0.075)])
            action = self._ik_action(goal, 0.0, max_speed=0.07, preserve_tilt=False)
            if cube[2] > 0.065:
                self._transition(Phase.TRANSIT)
            elif self.phase_tick > 100:
                self._transition(Phase.FAILED)

        elif self.phase == Phase.TRANSIT:
            goal = np.array([target[0], target[1], 0.10])
            action = self._ik_action(goal, 0.0, max_speed=0.08, preserve_tilt=False)
            if self.last_error < 0.012:
                self.release_site = site.copy()
                self._transition(Phase.RELEASE)
            elif self.phase_tick > 150:
                self._transition(Phase.FAILED)

        elif self.phase == Phase.RELEASE:
            action = self._ik_action(self.release_site, 16.0, max_speed=0.03, preserve_tilt=False)
            if self.phase_tick >= 30:
                self._transition(Phase.RETREAT)

        elif self.phase == Phase.RETREAT:
            goal = self.release_site + np.array([0.0, 0.0, 0.06])
            action = self._ik_action(goal, 16.0, max_speed=0.06, preserve_tilt=False)
            cube_offset = env.data.xpos[env.cube_body_id] - target
            contained = abs(cube_offset[0]) < 0.014 and abs(cube_offset[1]) < 0.014
            settled = np.linalg.norm(env.data.cvel[env.cube_body_id, 3:]) < 0.05
            if contained and settled and cube_offset[2] < 0.06:
                self._transition(Phase.DONE)
            elif self.phase_tick > 75:
                self._transition(Phase.FAILED)

        else:
            action = env.state()

        self.phase_tick += 1
        self.total_tick += 1
        if self.total_tick > 750 and not self.terminal:
            self._transition(Phase.FAILED)
        return action.astype(np.float32)


def run_expert_episode(
    env: SO101PickPlaceEnv,
    max_steps: int = 750,
    capture: bool = False,
) -> tuple[bool, list[dict[str, np.ndarray]]]:
    expert = PrivilegedPickPlaceExpert(env)
    frames: list[dict[str, np.ndarray]] = []
    for _ in range(max_steps):
        state = env.state()
        action = expert.action()
        if capture:
            top, _ = env.render_cameras()
            frames.append({"image": top, "state": state, "action": action.copy()})
        env.apply_action(action)
        env.step_control_period(30.0)
        if expert.terminal:
            break
    return expert.phase == Phase.DONE, frames
