from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .environment import SO101PickPlaceEnv
from .morphology import (
    PANDA,
    SO101,
    SO101_BODY_LIMITS_DEGREES,
    SO101_GRIPPER_NATIVE_RANGE,
    MorphologySpec,
    morphology_by_id,
)
from .tasks import TaskSpec, TaskStatus, task_by_id


@dataclass(frozen=True)
class BenchmarkScenario:
    object_xy: tuple[float, float]
    target_xy: tuple[float, float]
    object_half_extents: tuple[float, float, float] = (0.0165, 0.0165, 0.0165)
    object_mass: float = 0.0149
    friction: float = 1.2
    target_tolerance: float = 0.055
    distractor_xy: tuple[tuple[float, float], ...] = ()
    visual_domain: str = "default"

    def __post_init__(self) -> None:
        values = (
            *self.object_xy,
            *self.target_xy,
            *self.object_half_extents,
            self.object_mass,
            self.friction,
            self.target_tolerance,
            *(coordinate for position in self.distractor_xy for coordinate in position),
        )
        if not np.isfinite(values).all():
            raise ValueError("benchmark scenario values must be finite")
        if any(value <= 0 for value in self.object_half_extents):
            raise ValueError("object half extents must be positive")
        if self.object_mass <= 0 or self.friction <= 0 or self.target_tolerance <= 0:
            raise ValueError("mass, friction, and target tolerance must be positive")
        if self.visual_domain != "default":
            raise ValueError("benchmark runtime currently supports only the default visual domain")


def default_scenario(morphology_id: str, task_id: str | None = None) -> BenchmarkScenario:
    if morphology_id == SO101.id:
        if task_id == "push_to_zone":
            return BenchmarkScenario(object_xy=(0.30, 0.0), target_xy=(0.25, -0.04))
        return BenchmarkScenario(object_xy=(0.30, 0.0), target_xy=(0.10, -0.18))
    if morphology_id == PANDA.id:
        if task_id == "push_to_zone":
            return BenchmarkScenario(object_xy=(0.45, 0.0), target_xy=(0.35, 0.0))
        return BenchmarkScenario(object_xy=(0.45, 0.0), target_xy=(0.40, -0.20))
    raise ValueError(f"unknown morphology: {morphology_id}")


def _floats(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _task_geometry(task: TaskSpec, scenario: BenchmarkScenario) -> str:
    tolerance = scenario.target_tolerance
    if task.id == "pick_place_bin":
        wall = 0.002
        height = 0.025
        return f"""
      <geom name="target_floor" type="box" size="{tolerance} {tolerance} 0.003" rgba="0.82 0.86 0.92 1"/>
      <geom type="box" pos="{tolerance + wall} 0 {height}" size="{wall} {tolerance + wall} {height}" rgba="0.92 0.94 0.98 1"/>
      <geom type="box" pos="{-tolerance - wall} 0 {height}" size="{wall} {tolerance + wall} {height}" rgba="0.92 0.94 0.98 1"/>
      <geom type="box" pos="0 {tolerance + wall} {height}" size="{tolerance} {wall} {height}" rgba="0.92 0.94 0.98 1"/>
      <geom type="box" pos="0 {-tolerance - wall} {height}" size="{tolerance} {wall} {height}" rgba="0.92 0.94 0.98 1"/>"""
    if task.id == "stack_object":
        return """
      <geom name="support_geom" type="box" size="0.06 0.06 0.025" pos="0 0 0.025" mass="0.2" rgba="0.15 0.35 0.85 1" friction="1.2 0.01 0.001"/>"""
    return f"""
      <geom name="target_floor" type="cylinder" size="{tolerance} 0.002" rgba="0.2 0.7 0.3 0.45" contype="0" conaffinity="0"/>"""


def _scene_xml(robot_filename: str, task: TaskSpec, scenario: BenchmarkScenario, morphology: MorphologySpec) -> str:
    object_z = scenario.object_half_extents[2] + 0.001
    camera = (
        '<camera name="top" pos="0.31 0 0.68" xyaxes="0 1 0 -1 0 0" fovy="62"/>'
        if morphology.id == SO101.id
        else '<camera name="top" pos="0.48 0 1.15" xyaxes="0 1 0 -1 0 0" fovy="55"/>'
    )
    distractors = "\n".join(
        f'<body name="distractor_{index}" pos="{x} {y} 0.012"><geom type="box" '
        'size="0.012 0.012 0.012" rgba="0.55 0.55 0.58 1"/></body>'
        for index, (x, y) in enumerate(scenario.distractor_xy)
    )
    return f"""<mujoco model="embodi_{morphology.id}_{task.id}">
  <include file="{robot_filename}"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  </visual>
  <worldbody>
    <light pos="0.3 0.2 1.4" dir="0 0 -1" diffuse="0.95 0.95 0.95"/>
    <geom name="table" type="plane" size="1 1 0.05" rgba="0.09 0.10 0.12 1" friction="1 0.01 0.001"/>
    {camera}
    <body name="target" pos="{scenario.target_xy[0]} {scenario.target_xy[1]} 0.003">
      {_task_geometry(task, scenario)}
    </body>
    <body name="object" pos="{scenario.object_xy[0]} {scenario.object_xy[1]} {object_z}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{_floats(scenario.object_half_extents)}"
        mass="{scenario.object_mass}" rgba="0.9 0.055 0.035 1"
        friction="{scenario.friction} 0.01 0.001" condim="4" solref="0.008 1"/>
    </body>
    {distractors}
  </worldbody>
</mujoco>
"""


def _prepare_panda_xml(root: Path) -> str:
    source = root / PANDA.model_filename
    destination = root / "panda_embodi.xml"
    tree = ET.parse(source)
    link0 = tree.find(".//body[@name='link0']")
    hand = tree.find(".//body[@name='hand']")
    if link0 is None or hand is None:
        raise RuntimeError("pinned Panda XML is missing link0 or hand")
    # Mount the arm base below the work surface so low tabletop grasps remain
    # reachable with the gripper's downward home orientation.
    link0.set("pos", "-0.1 0 -0.1")
    if link0.find("./site[@name='baseframe']") is None:
        ET.SubElement(link0, "site", name="baseframe", size="0.003", rgba="0 0 0 0")
    if hand.find("./site[@name='gripperframe']") is None:
        ET.SubElement(
            hand,
            "site",
            name="gripperframe",
            pos="0 0 0.1034",
            size="0.003",
            rgba="0 0 0 0",
        )
    text = ET.tostring(tree.getroot(), encoding="unicode")
    if not destination.exists() or destination.read_text() != text:
        destination.write_text(text)
    return destination.name


class BenchmarkManipulationEnv:
    def __init__(
        self,
        morphology: str,
        task: str,
        scenario: BenchmarkScenario | None = None,
        *,
        width: int = 640,
        height: int = 480,
        asset_dir: Path | None = None,
        render: bool = True,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.morphology = morphology_by_id(morphology)
        self.task = task_by_id(task)
        self.scenario = scenario or default_scenario(morphology, task)
        self.width = width
        self.height = height
        root = self.morphology.ensure_assets(asset_dir)
        if self.morphology.id == SO101.id:
            SO101PickPlaceEnv._write_scaled_robot_xml(root)
            robot_filename = "so101_visual_1_2x.xml"
        else:
            robot_filename = _prepare_panda_xml(root)
        scene_xml = _scene_xml(robot_filename, self.task, self.scenario, self.morphology)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=root,
            prefix="embodi_benchmark_",
            suffix=".xml",
            delete=False,
        ) as stream:
            stream.write(scene_xml)
            scene_path = Path(stream.name)
        try:
            self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        finally:
            scene_path.unlink(missing_ok=True)
        self.data = mujoco.MjData(self.model)
        self.fk_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width) if render else None
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.morphology.sim_joint_names
            ]
        )
        if (self.joint_ids < 0).any():
            raise RuntimeError("morphology XML is missing configured joints")
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.morphology.arm_joint_names
            ]
        )
        self.arm_qpos_addresses = self.model.jnt_qposadr[self.arm_joint_ids]
        self.base_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.morphology.base_frame
        )
        self.tool_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.morphology.tool_frame
        )
        self.object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free"
        )
        self.object_qpos_address = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_dof_address = int(self.model.jnt_dofadr[self.object_joint_id])
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.control_ranges = self.model.actuator_ctrlrange.copy()
        self.control_time = 0.0
        self.dwell = 0
        self.lifted = False
        self.success = False
        self.reset()

    @property
    def instruction(self) -> str:
        return self.task.instruction

    def native_to_qpos(self, native: np.ndarray) -> np.ndarray:
        native = np.asarray(native, dtype=np.float32)
        if native.shape != (self.morphology.native_state_dim,) or not np.isfinite(native).all():
            raise ValueError("native state must match morphology dimensions and be finite")
        if self.morphology.id == SO101.id:
            body = np.deg2rad(
                np.clip(native[:5], SO101_BODY_LIMITS_DEGREES[:, 0], SO101_BODY_LIMITS_DEGREES[:, 1])
            )
            fraction = np.clip(
                (native[5] - SO101_GRIPPER_NATIVE_RANGE[0])
                / np.ptp(SO101_GRIPPER_NATIVE_RANGE),
                0.0,
                1.0,
            )
            gripper = -0.174533 + fraction * (1.7453292 + 0.174533)
            return np.concatenate((body, [gripper])).astype(np.float32)
        arm_lower = self.model.jnt_range[self.arm_joint_ids, 0]
        arm_upper = self.model.jnt_range[self.arm_joint_ids, 1]
        arm = np.clip(native[:7], arm_lower, arm_upper)
        finger = np.clip(native[7] / 2.0, 0.0, 0.04)
        return np.concatenate((arm, [finger, finger])).astype(np.float32)

    def qpos_to_native(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (len(self.morphology.sim_joint_names),) or not np.isfinite(qpos).all():
            raise ValueError("qpos must match morphology simulation joints and be finite")
        if self.morphology.id == SO101.id:
            body = np.rad2deg(qpos[:5])
            fraction = np.clip((qpos[5] + 0.174533) / (1.7453292 + 0.174533), 0.0, 1.0)
            gripper = SO101_GRIPPER_NATIVE_RANGE[0] + fraction * np.ptp(
                SO101_GRIPPER_NATIVE_RANGE
            )
            return np.concatenate((body, [gripper])).astype(np.float32)
        return np.concatenate((qpos[:7], [np.clip(qpos[7] + qpos[8], 0.0, 0.08)])).astype(
            np.float32
        )

    def native_to_control(self, native: np.ndarray) -> np.ndarray:
        qpos = self.native_to_qpos(native)
        if self.morphology.id == SO101.id:
            return np.clip(qpos, self.control_ranges[:, 0], self.control_ranges[:, 1])
        control = np.concatenate((qpos[:7], [np.clip(native[7] / 0.08, 0.0, 1.0) * 255.0]))
        return np.clip(control, self.control_ranges[:, 0], self.control_ranges[:, 1])

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        qpos = self.native_to_qpos(self.morphology.home_native_state)
        self.data.qpos[self.qpos_addresses] = qpos
        self.data.ctrl[:] = self.native_to_control(self.morphology.home_native_state)
        object_z = self.scenario.object_half_extents[2] + 0.001
        self.data.qpos[self.object_qpos_address : self.object_qpos_address + 7] = [
            *self.scenario.object_xy,
            object_z,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.mujoco.mj_forward(self.model, self.data)
        self.control_time = 0.0
        self.dwell = 0
        self.lifted = False
        self.success = False

    def state(self) -> np.ndarray:
        return self.qpos_to_native(self.data.qpos[self.qpos_addresses]).copy()

    @staticmethod
    def _rotation_6d(rotation: np.ndarray) -> np.ndarray:
        return np.asarray(rotation, dtype=np.float32)[:, :2].T.reshape(-1)

    @staticmethod
    def _rotation_6d_matrix(rotation: np.ndarray) -> np.ndarray:
        first = np.asarray(rotation[:3], dtype=np.float64)
        first /= max(np.linalg.norm(first), 1e-8)
        second = np.asarray(rotation[3:6], dtype=np.float64)
        second -= first * np.dot(first, second)
        second /= max(np.linalg.norm(second), 1e-8)
        return np.stack((first, second, np.cross(first, second)), axis=1)

    def _end_effector_in_base(self, data) -> tuple[np.ndarray, np.ndarray]:
        base_position = data.site_xpos[self.base_site_id]
        base_rotation = data.site_xmat[self.base_site_id].reshape(3, 3)
        tool_position = data.site_xpos[self.tool_site_id]
        tool_rotation = data.site_xmat[self.tool_site_id].reshape(3, 3)
        return base_rotation.T @ (tool_position - base_position), base_rotation.T @ tool_rotation

    def _opening_fraction(self, native: np.ndarray) -> float:
        if self.morphology.id == SO101.id:
            return float(
                np.clip(
                    (native[5] - SO101_GRIPPER_NATIVE_RANGE[0])
                    / np.ptp(SO101_GRIPPER_NATIVE_RANGE),
                    0.0,
                    1.0,
                )
            )
        return float(np.clip(native[7] / 0.08, 0.0, 1.0))

    def canonical_state(self) -> np.ndarray:
        position, rotation = self._end_effector_in_base(self.data)
        state = np.zeros((1, 10), dtype=np.float32)
        state[0] = np.concatenate(
            (position.astype(np.float32), self._rotation_6d(rotation), [self._opening_fraction(self.state())])
        )
        return state

    def canonical_arm_action(self, native: np.ndarray) -> np.ndarray:
        native = np.asarray(native, dtype=np.float32)
        current_position, current_rotation = self._end_effector_in_base(self.data)
        self.fk_data.qpos[:] = self.data.qpos
        self.fk_data.qpos[self.qpos_addresses] = self.native_to_qpos(native)
        self.mujoco.mj_kinematics(self.model, self.fk_data)
        target_position, target_rotation = self._end_effector_in_base(self.fk_data)
        canonical = np.zeros((1, 10), dtype=np.float32)
        canonical[0] = np.concatenate(
            (
                (target_position - current_position).astype(np.float32),
                self._rotation_6d(current_rotation.T @ target_rotation),
                [self._opening_fraction(native)],
            )
        )
        return canonical

    def native_action_from_canonical(self, canonical: np.ndarray, iterations: int = 20) -> np.ndarray:
        canonical = np.asarray(canonical, dtype=np.float64)
        if canonical.shape == (1, 10):
            canonical = canonical[0]
        if canonical.shape != (10,) or not np.isfinite(canonical).all():
            raise ValueError("canonical command must be a finite width-10 block")
        current_position, current_rotation = self._end_effector_in_base(self.data)
        base_position = self.data.site_xpos[self.base_site_id].copy()
        base_rotation = self.data.site_xmat[self.base_site_id].reshape(3, 3).copy()
        target_position = base_position + base_rotation @ (current_position + canonical[:3])
        target_rotation = base_rotation @ (
            current_rotation @ self._rotation_6d_matrix(canonical[3:9])
        )
        candidate = self.data.qpos[self.arm_qpos_addresses].copy()
        home = self.native_to_qpos(self.morphology.home_native_state)[: len(candidate)]
        arm_dofs = self.model.jnt_dofadr[self.arm_joint_ids]
        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        for _ in range(iterations):
            self.fk_data.qpos[:] = self.data.qpos
            self.fk_data.qpos[self.arm_qpos_addresses] = candidate
            self.mujoco.mj_forward(self.model, self.fk_data)
            position = self.fk_data.site_xpos[self.tool_site_id]
            rotation = self.fk_data.site_xmat[self.tool_site_id].reshape(3, 3)
            position_error = target_position - position
            rotation_error = 0.5 * sum(
                np.cross(rotation[:, axis], target_rotation[:, axis]) for axis in range(3)
            )
            if np.linalg.norm(position_error) < 1e-5 and np.linalg.norm(rotation_error) < 1e-4:
                break
            jacobian_position = np.zeros((3, self.model.nv))
            jacobian_rotation = np.zeros((3, self.model.nv))
            self.mujoco.mj_jacSite(
                self.model,
                self.fk_data,
                jacobian_position,
                jacobian_rotation,
                self.tool_site_id,
            )
            characteristic_length = 0.08
            jacobian = np.vstack(
                (
                    jacobian_position[:, arm_dofs],
                    characteristic_length * jacobian_rotation[:, arm_dofs],
                )
            )
            error = np.concatenate((position_error, characteristic_length * rotation_error))
            damping = 0.01
            inverse = np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(6), np.eye(6))
            pseudoinverse = jacobian.T @ inverse
            correction = pseudoinverse @ error
            if len(candidate) > 6:
                nullspace = np.eye(len(candidate)) - pseudoinverse @ jacobian
                correction += nullspace @ (0.03 * (home - candidate))
            candidate = np.clip(candidate + np.clip(correction, -0.12, 0.12), lower, upper)
        native = self.state()
        native[: len(candidate)] = (
            np.rad2deg(candidate) if self.morphology.id == SO101.id else candidate
        )
        if self.morphology.id == SO101.id:
            native[5] = np.clip(canonical[9], 0.0, 1.0) * np.ptp(SO101_GRIPPER_NATIVE_RANGE)
        else:
            native[7] = np.clip(canonical[9], 0.0, 1.0) * 0.08
        return native.astype(np.float32)

    def apply_action(self, native: np.ndarray) -> None:
        self.data.ctrl[:] = self.native_to_control(native)

    def diagnostic_state(self) -> dict[str, float]:
        object_position = self.data.xpos[self.object_body_id]
        target_position = self.data.xpos[self.target_body_id]
        tool_position = self.data.site_xpos[self.tool_site_id]
        speed = np.linalg.norm(
            self.data.qvel[self.object_dof_address : self.object_dof_address + 6]
        )
        return {
            "ee_object_distance_m": float(np.linalg.norm(tool_position - object_position)),
            "object_target_xy_distance_m": float(
                np.linalg.norm(object_position[:2] - target_position[:2])
            ),
            "object_height_m": float(object_position[2]),
            "object_speed": float(speed),
            "gripper_opening_fraction": self._opening_fraction(self.state()),
        }

    def task_status(self) -> TaskStatus:
        diagnostics = self.diagnostic_state()
        near = diagnostics["ee_object_distance_m"] < 0.06
        release_threshold = 0.4 if self.morphology.id == SO101.id else 0.7
        released = diagnostics["gripper_opening_fraction"] >= release_threshold
        settled = diagnostics["object_speed"] < 0.05
        if self.task.id == "push_to_zone":
            predicate = (
                diagnostics["object_target_xy_distance_m"] <= self.scenario.target_tolerance
                and diagnostics["object_height_m"] < 0.06
                and settled
            )
            stage = "push" if near else "approach"
        elif self.task.id == "lift_object":
            predicate = (
                diagnostics["object_height_m"] >= 0.08
                and diagnostics["ee_object_distance_m"] < 0.06
                and not released
            )
            stage = "lift" if near else "approach"
        elif self.task.id == "pick_place_bin":
            predicate = (
                self.lifted
                and diagnostics["object_target_xy_distance_m"] <= self.scenario.target_tolerance
                and diagnostics["object_height_m"] < 0.09
                and settled
                and released
            )
            stage = "release" if self.lifted else ("lift" if near else "approach")
        else:
            support_top = 0.05
            expected_height = support_top + self.scenario.object_half_extents[2]
            predicate = (
                diagnostics["object_target_xy_distance_m"] <= self.scenario.target_tolerance
                and abs(diagnostics["object_height_m"] - expected_height) < 0.015
                and settled
                and released
            )
            stage = "stack" if self.lifted else ("lift" if near else "approach")
        self.dwell = self.dwell + 1 if predicate else 0
        self.success = self.success or self.dwell >= self.task.dwell_steps
        return TaskStatus(success=self.success, terminal=self.success, stage="success" if self.success else stage)

    def step_control_period(self, frequency: float = 30.0) -> TaskStatus:
        self.control_time += 1.0 / frequency
        while self.data.time + 1e-9 < self.control_time:
            self.mujoco.mj_step(self.model, self.data)
        self.lifted = self.lifted or self.data.xpos[self.object_body_id][2] > 0.07
        return self.task_status()

    def render_top(self) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("environment was created without rendering")
        self.renderer.update_scene(self.data, camera="top")
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
