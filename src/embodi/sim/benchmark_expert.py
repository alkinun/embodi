from __future__ import annotations

from enum import Enum

import numpy as np

from .benchmark_environment import BenchmarkManipulationEnv
from .morphology import SO101


class BenchmarkPhase(str, Enum):
    OPEN = "open"
    PRECONTACT = "precontact"
    CONTACT = "contact"
    CLOSE = "close"
    LIFT = "lift"
    TRANSIT = "transit"
    RELEASE = "release"
    RETREAT = "retreat"
    PUSH = "push"
    DONE = "done"
    FAILED = "failed"


class PrivilegedBenchmarkExpert:
    def __init__(self, env: BenchmarkManipulationEnv, frequency: float = 30.0) -> None:
        self.env = env
        self.frequency = frequency
        self.phase = BenchmarkPhase.OPEN
        self.phase_tick = 0
        self.total_tick = 0
        self.last_error = float("inf")
        self.reference_rotation = env.data.site_xmat[env.tool_site_id].reshape(3, 3).copy()
        self.reference_approach = self.reference_rotation[:, 0].copy()
        self.contact_site = np.zeros(3)
        self.release_site = np.zeros(3)
        self.grasp_offset = np.zeros(3)

    @property
    def terminal(self) -> bool:
        return self.phase in {BenchmarkPhase.DONE, BenchmarkPhase.FAILED}

    @property
    def open_gripper(self) -> float:
        return 16.0 if self.env.morphology.id == SO101.id else 0.08

    @staticmethod
    def _rotation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        return 0.5 * sum(
            np.cross(current[:, axis], desired[:, axis]) for axis in range(3)
        )

    def _transition(self, phase: BenchmarkPhase) -> None:
        self.phase = phase
        self.phase_tick = 0

    def _ik_action(
        self,
        goal: np.ndarray,
        gripper: float,
        *,
        max_speed: float,
        preserve_rotation: bool = True,
    ) -> np.ndarray:
        env = self.env
        position = env.data.site_xpos[env.tool_site_id]
        error = np.asarray(goal, dtype=np.float64) - position
        self.last_error = float(np.linalg.norm(error))
        if env.morphology.id != SO101.id:
            return self._panda_position_action(np.asarray(goal), gripper)
        velocity = 5.0 * error
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity *= max_speed / speed
        jacobian_position = np.zeros((3, env.model.nv))
        jacobian_rotation = np.zeros((3, env.model.nv))
        env.mujoco.mj_jacSite(
            env.model,
            env.data,
            jacobian_position,
            jacobian_rotation,
            env.tool_site_id,
        )
        position_jacobian = jacobian_position[:, env.model.jnt_dofadr[env.arm_joint_ids]]
        if preserve_rotation:
            rotation = env.data.site_xmat[env.tool_site_id].reshape(3, 3)
            rotation_error = self._rotation_error(rotation, self.reference_rotation)
            characteristic_length = 0.06
            rotation_rows = 2 if env.morphology.id == SO101.id else 3
            jacobian = np.vstack(
                (
                    position_jacobian,
                    characteristic_length
                    * jacobian_rotation[:rotation_rows, env.model.jnt_dofadr[env.arm_joint_ids]],
                )
            )
            task_velocity = np.concatenate(
                (velocity, characteristic_length * 3.0 * rotation_error[:rotation_rows])
            )
        else:
            jacobian = position_jacobian
            task_velocity = velocity
        damping = 0.025
        qdot = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0]),
            task_velocity,
        )
        qdot = np.clip(qdot, -1.0, 1.0)
        measured = env.data.qpos[env.arm_qpos_addresses]
        lower = env.model.jnt_range[env.arm_joint_ids, 0] + 0.01
        upper = env.model.jnt_range[env.arm_joint_ids, 1] - 0.01
        command = np.clip(measured + qdot / self.frequency, lower, upper)
        action = env.state()
        action[: len(command)] = (
            np.rad2deg(command) if env.morphology.id == SO101.id else command
        )
        action[-1] = gripper
        return action.astype(np.float32)

    def _panda_position_action(self, goal: np.ndarray, gripper: float) -> np.ndarray:
        env = self.env
        candidate = env.data.qpos[env.arm_qpos_addresses].copy()
        arm_dofs = env.model.jnt_dofadr[env.arm_joint_ids]
        lower = env.model.jnt_range[env.arm_joint_ids, 0] + 0.01
        upper = env.model.jnt_range[env.arm_joint_ids, 1] - 0.01
        for _ in range(60):
            env.fk_data.qpos[:] = env.data.qpos
            env.fk_data.qpos[env.arm_qpos_addresses] = candidate
            env.mujoco.mj_forward(env.model, env.fk_data)
            error = goal - env.fk_data.site_xpos[env.tool_site_id]
            if np.linalg.norm(error) < 1e-4:
                break
            jacobian_position = np.zeros((3, env.model.nv))
            jacobian_rotation = np.zeros((3, env.model.nv))
            env.mujoco.mj_jacSite(
                env.model,
                env.fk_data,
                jacobian_position,
                jacobian_rotation,
                env.tool_site_id,
            )
            jacobian = jacobian_position[:, arm_dofs]
            damping = 0.01
            correction = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(3), error
            )
            candidate = np.clip(candidate + np.clip(correction, -0.12, 0.12), lower, upper)
        action = env.state()
        measured = env.data.qpos[env.arm_qpos_addresses]
        action[:7] = measured + np.clip(candidate - measured, -0.04, 0.04)
        action[7] = gripper
        return action.astype(np.float32)

    def _precontact_goal(self, object_position: np.ndarray) -> np.ndarray:
        if self.env.morphology.id == SO101.id:
            return object_position - self.reference_approach * 0.065
        return object_position + np.array([0.0, 0.0, 0.10])

    def _contact_goal(self, object_position: np.ndarray) -> np.ndarray:
        if self.env.morphology.id == SO101.id:
            return object_position + np.array([0.0, 0.0, 0.002])
        return object_position + np.array([0.0, 0.0, 0.005])

    def action(self) -> np.ndarray:
        env = self.env
        object_position = env.data.xpos[env.object_body_id].copy()
        target_position = env.data.xpos[env.target_body_id].copy()
        tool_position = env.data.site_xpos[env.tool_site_id].copy()
        action: np.ndarray

        if self.phase == BenchmarkPhase.OPEN:
            action = env.state()
            action[-1] = self.open_gripper
            self.last_error = 0.0
            if self.phase_tick >= 18:
                self._transition(BenchmarkPhase.PRECONTACT)

        elif self.phase == BenchmarkPhase.PRECONTACT:
            if env.task.id == "push_to_zone":
                direction = target_position[:2] - object_position[:2]
                direction /= max(np.linalg.norm(direction), 1e-8)
                goal = object_position.copy()
                goal[:2] -= direction * 0.055
                goal[2] += 0.012
                preserve = env.morphology.id != SO101.id
            else:
                goal = self._precontact_goal(object_position)
                preserve = env.morphology.id == SO101.id
            action = self._ik_action(
                goal,
                self.open_gripper,
                max_speed=0.15,
                preserve_rotation=preserve,
            )
            if env.task.id == "push_to_zone":
                threshold = 0.04
            else:
                threshold = 0.003 if env.morphology.id != SO101.id else 0.015
            if self.last_error < threshold:
                self._transition(
                    BenchmarkPhase.PUSH
                    if env.task.id == "push_to_zone"
                    else BenchmarkPhase.CONTACT
                )
            elif self.phase_tick > 300:
                self._transition(BenchmarkPhase.FAILED)

        elif self.phase == BenchmarkPhase.PUSH:
            direction = target_position[:2] - object_position[:2]
            target_distance = np.linalg.norm(direction)
            direction /= max(target_distance, 1e-8)
            if target_distance <= env.scenario.target_tolerance * 0.75:
                action = env.state()
            else:
                goal = np.array(
                    [
                        object_position[0] + direction[0] * 0.015,
                        object_position[1] + direction[1] * 0.015,
                        object_position[2] + 0.012,
                    ]
                )
                action = self._ik_action(
                    goal,
                    self.open_gripper,
                    max_speed=min(0.06, max(0.01, target_distance * 0.5)),
                    preserve_rotation=env.morphology.id != SO101.id,
                )
            if env.success:
                self._transition(BenchmarkPhase.DONE)
                action = env.state()
            elif self.phase_tick > 480:
                self._transition(BenchmarkPhase.FAILED)

        elif self.phase == BenchmarkPhase.CONTACT:
            action = self._ik_action(
                self._contact_goal(object_position),
                self.open_gripper,
                max_speed=0.02,
                preserve_rotation=env.morphology.id == SO101.id,
            )
            contact_threshold = 0.005 if env.morphology.id != SO101.id else 0.01
            if self.last_error < contact_threshold:
                self.contact_site = tool_position.copy()
                self._transition(BenchmarkPhase.CLOSE)
            elif self.phase_tick > 240:
                self._transition(BenchmarkPhase.FAILED)

        elif self.phase == BenchmarkPhase.CLOSE:
            action = self._ik_action(
                self.contact_site,
                0.0,
                max_speed=0.03,
                preserve_rotation=True,
            )
            close_ticks = 75 if env.morphology.id != SO101.id else 30
            if self.phase_tick >= close_ticks:
                self._transition(BenchmarkPhase.LIFT)

        elif self.phase == BenchmarkPhase.LIFT:
            goal = np.array([tool_position[0], tool_position[1], 0.14])
            action = self._ik_action(
                goal,
                0.0,
                max_speed=0.08,
                preserve_rotation=env.morphology.id != SO101.id,
            )
            if env.task.id == "lift_object" and env.success:
                self._transition(BenchmarkPhase.DONE)
                action = env.state()
            elif object_position[2] > 0.07 and env.task.id != "lift_object":
                self.grasp_offset = object_position - tool_position
                self._transition(BenchmarkPhase.TRANSIT)
            elif self.phase_tick > 150:
                self._transition(BenchmarkPhase.FAILED)

        elif self.phase == BenchmarkPhase.TRANSIT:
            height = 0.13 if env.task.id == "stack_object" else 0.10
            goal = np.array(
                [
                    target_position[0] - self.grasp_offset[0],
                    target_position[1] - self.grasp_offset[1],
                    height,
                ]
            )
            action = self._ik_action(
                goal,
                0.0,
                max_speed=0.10,
                preserve_rotation=env.morphology.id != SO101.id,
            )
            if self.last_error < 0.015:
                self.release_site = tool_position.copy()
                self._transition(BenchmarkPhase.RELEASE)
            elif self.phase_tick > 220:
                self._transition(BenchmarkPhase.FAILED)

        elif self.phase == BenchmarkPhase.RELEASE:
            action = self._ik_action(
                self.release_site,
                self.open_gripper,
                max_speed=0.03,
                preserve_rotation=env.morphology.id != SO101.id,
            )
            if self.phase_tick >= 35:
                self._transition(BenchmarkPhase.RETREAT)

        elif self.phase == BenchmarkPhase.RETREAT:
            action = self._ik_action(
                self.release_site + np.array([0.0, 0.0, 0.08]),
                self.open_gripper,
                max_speed=0.08,
                preserve_rotation=env.morphology.id != SO101.id,
            )
            if env.success:
                self._transition(BenchmarkPhase.DONE)
                action = env.state()
            elif self.phase_tick > 120:
                self._transition(BenchmarkPhase.FAILED)

        else:
            action = env.state()

        self.phase_tick += 1
        self.total_tick += 1
        if self.total_tick > 500 and not self.terminal:
            self._transition(BenchmarkPhase.FAILED)
        return action.astype(np.float32)


def run_benchmark_expert_episode(
    env: BenchmarkManipulationEnv,
    max_steps: int = 500,
) -> tuple[bool, BenchmarkPhase]:
    expert = PrivilegedBenchmarkExpert(env)
    for _ in range(max_steps):
        action = expert.action()
        env.apply_action(action)
        env.step_control_period(expert.frequency)
        if expert.terminal:
            break
    return expert.phase == BenchmarkPhase.DONE and env.success, expert.phase
