from __future__ import annotations

from pathlib import Path
import random
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .assets import ensure_so101_assets


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
BODY_LIMITS_DEGREES = np.array(
    [
        [-110.0, 110.0],
        [-100.0, 100.0],
        [-96.8, 96.8],
        [-95.0, 95.0],
        [-157.2, 157.2],
    ],
    dtype=np.float32,
)
GRIPPER_DATASET_RANGE = np.array([0.0, 35.0], dtype=np.float32)
HOME_STATE = np.array([0.3516, -86.8359, 75.0586, 72.0703, 82.9688, 0.3453], dtype=np.float32)
TASK = "Pick up the cube and place it in the box."


SCENE_XML = """<mujoco model="embodi_so101_pickplace">
  <include file="so101_visual_1_2x.xml"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.65 0.72 0.82" rgb2="0.12 0.15 0.2" width="512" height="3072"/>
    <texture type="2d" name="table_tex" file="table.png"/>
    <material name="table_mat" texture="table_tex" texrepeat="1 1" rgba="1 1 1 1" specular="0.05" shininess="0.1" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0.12 0.35 0.9" dir="0 -0.15 -1" diffuse="0.95 0.95 0.95" specular="0.7 0.7 0.7"/>
    <geom name="table" type="plane" size="1 1 0.05" material="table_mat" friction="1 0.01 0.001"/>
    <camera name="top" pos="0.31 0 0.68" xyaxes="0 1 0 -1 0 0" fovy="62"/>
    <body name="target" pos="0.10 -0.18 0.008">
      <geom name="target_floor" type="box" size="0.0325 0.0325 0.004" rgba="0.92 0.93 0.94 1" contype="1" conaffinity="1"/>
      <geom type="box" pos="0.034 0 0.018" size="0.002 0.034 0.020" rgba="0.96 0.97 0.98 1"/>
      <geom type="box" pos="-0.034 0 0.018" size="0.002 0.034 0.020" rgba="0.96 0.97 0.98 1"/>
      <geom type="box" pos="0 0.034 0.018" size="0.0325 0.002 0.020" rgba="0.96 0.97 0.98 1"/>
      <geom type="box" pos="0 -0.034 0.018" size="0.0325 0.002 0.020" rgba="0.96 0.97 0.98 1"/>
    </body>
    <body name="cube" pos="0.23 0 0.018">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.0165 0.0165 0.0165" mass="0.0149" rgba="0.9 0.055 0.035 1"
        friction="1.2 0.01 0.001" condim="4" solref="0.008 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class SO101PickPlaceEnv:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        asset_dir: Path | None = None,
        seed: int = 0,
        cube_x_range: tuple[float, float] = (0.28, 0.32),
        cube_y_range: tuple[float, float] = (-0.025, 0.025),
        render: bool = True,
    ) -> None:
        import mujoco

        if (
            len(cube_x_range) != 2
            or len(cube_y_range) != 2
            or not np.isfinite((*cube_x_range, *cube_y_range)).all()
            or cube_x_range[0] > cube_x_range[1]
            or cube_y_range[0] > cube_y_range[1]
        ):
            raise ValueError("cube ranges must contain two finite, ascending values")
        self.mujoco = mujoco
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.cube_x_range = tuple(cube_x_range)
        self.cube_y_range = tuple(cube_y_range)
        root = ensure_so101_assets(asset_dir)
        self._write_scaled_robot_xml(root)
        self._write_table_texture(root / "table.png")
        scene_path = root / "embodi_pickplace.xml"
        if not scene_path.exists() or scene_path.read_text() != SCENE_XML:
            scene_path.write_text(SCENE_XML)
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._apply_robot_colors()
        self._calibrate_wrist_camera()
        self.data = mujoco.MjData(self.model)
        self.fk_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width) if render else None
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        )
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_qpos_address = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.base_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "baseframe")
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.control_ranges = self.model.actuator_ctrlrange.copy()
        self.episode = 0
        self.lifted = False
        self.success = False
        self.control_time = 0.0
        self.reset()

    def _calibrate_wrist_camera(self) -> None:
        camera_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )
        # Fit against dataset frame 0 at its recorded six-joint state.
        self.model.cam_pos[camera_id] = [0.0, 0.055, -0.045]
        self.model.cam_quat[camera_id] = [0.9950042, -0.0998334, 0.0, 0.0]
        self.model.cam_intrinsic[camera_id] = [0.002, 0.002, 0.0, 0.0]

    @staticmethod
    def _write_scaled_robot_xml(root: Path) -> None:
        source = root / "so101.xml"
        destination = root / "so101_visual_1_2x.xml"
        tree = ET.parse(source)
        for mesh in tree.findall("./asset/mesh"):
            filename = mesh.get("file", "")
            if filename == "wrist_roll_follower_so101_v1.stl":
                mesh.set("scale", "-1.2 1.2 1.2")
            elif filename == "wrist_roll_follower_so101_gripper_part0_v1.stl":
                mesh.set("scale", "-1 1 1")
            elif "gripper_part" not in filename:
                mesh.set("scale", "1.2 1.2 1.2")
        moving_jaw = tree.find(".//body[@name='moving_jaw_so101_v1']")
        if moving_jaw is not None:
            position = [float(value) for value in moving_jaw.get("pos", "0 0 0").split()]
            position[0] *= -1
            position[1] *= -1
            moving_jaw.set("pos", " ".join(str(value) for value in position))
            moving_jaw.set("quat", "0 0 1 1")
        for geom in tree.findall(".//geom"):
            if not geom.get("name", "").startswith("fixed_jaw_") or geom.get("pos") is None:
                continue
            position = [float(value) for value in geom.get("pos", "").split()]
            position[0] *= -1
            geom.set("pos", " ".join(str(value) for value in position))
        gripper_site = tree.find(".//site[@name='gripperframe']")
        if gripper_site is not None:
            position = [float(value) for value in gripper_site.get("pos", "0 0 0").split()]
            position[0] *= -1
            gripper_site.set("pos", " ".join(str(value) for value in position))
        tree.write(destination, encoding="unicode")

    @staticmethod
    def _write_table_texture(path: Path) -> None:
        from PIL import Image

        size = 512
        y, x = np.mgrid[0:size, 0:size]
        random = np.random.default_rng(0)
        grain = random.normal(0.0, 0.007, (size, size))
        laminate = 0.006 * np.sin(x / 8.0) + 0.004 * np.sin((x + y) / 27.0)
        surface = (grain + laminate)[..., None]
        base = np.array([0.055, 0.058, 0.065], dtype=np.float32)
        texture = np.clip(base + surface, 0.0, 1.0)
        image = Image.fromarray((texture * 255).astype(np.uint8), mode="RGB")
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
            temporary_path = Path(stream.name)
        try:
            image.save(temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _apply_robot_colors(self) -> None:
        white = (0.94, 0.95, 0.96, 1.0)
        red_orange = (0.96, 0.36, 0.24, 1.0)
        material_colors = {
            "base_motor_holder_so101_v1_material": red_orange,
            "base_so101_v2_material": red_orange,
            "waveshare_mounting_plate_so101_v2_material": red_orange,
            "motor_holder_so101_base_v1_material": white,
            "rotation_pitch_so101_v1_material": white,
            "upper_arm_so101_v1_material": red_orange,
            "under_arm_so101_v1_material": red_orange,
            "motor_holder_so101_wrist_v1_material": red_orange,
            "wrist_roll_pitch_so101_v2_material": white,
            "wrist_roll_follower_so101_v1_material": red_orange,
            "moving_jaw_so101_v1_material": red_orange,
        }
        for name, color in material_colors.items():
            material_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_MATERIAL, name
            )
            if material_id >= 0:
                self.model.mat_rgba[material_id] = color
        for geom_id in range(self.model.ngeom):
            mesh_id = int(self.model.geom_dataid[geom_id])
            if self.model.geom_type[geom_id] != self.mujoco.mjtGeom.mjGEOM_MESH or mesh_id < 0:
                continue
            mesh_name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_MESH, mesh_id
            )
            if mesh_name == "wrist_roll_follower_so101_gripper_part0_v1":
                self.model.geom_matid[geom_id] = -1
                self.model.geom_rgba[geom_id] = white
                self.model.geom_group[geom_id] = 2

    @staticmethod
    def dataset_to_sim(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        body = np.deg2rad(np.clip(action[:5], BODY_LIMITS_DEGREES[:, 0], BODY_LIMITS_DEGREES[:, 1]))
        gripper_fraction = np.clip(
            (action[5] - GRIPPER_DATASET_RANGE[0]) / np.ptp(GRIPPER_DATASET_RANGE), 0.0, 1.0
        )
        gripper = -0.174533 + gripper_fraction * (1.7453292 + 0.174533)
        return np.concatenate((body, np.array([gripper], dtype=np.float32)))

    @staticmethod
    def sim_to_dataset(qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32)
        body = np.rad2deg(qpos[:5])
        gripper_fraction = np.clip((qpos[5] + 0.174533) / (1.7453292 + 0.174533), 0.0, 1.0)
        gripper = GRIPPER_DATASET_RANGE[0] + gripper_fraction * np.ptp(GRIPPER_DATASET_RANGE)
        return np.concatenate((body, np.array([gripper], dtype=np.float32)))

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        home = self.dataset_to_sim(HOME_STATE)
        self.data.qpos[self.qpos_addresses] = home
        self.data.ctrl[:] = np.clip(home, self.control_ranges[:, 0], self.control_ranges[:, 1])
        cube_x = self.random.uniform(*self.cube_x_range)
        cube_y = self.random.uniform(*self.cube_y_range)
        cube = self.cube_qpos_address
        self.data.qpos[cube : cube + 7] = [cube_x, cube_y, 0.018, 1.0, 0.0, 0.0, 0.0]
        self.mujoco.mj_forward(self.model, self.data)
        self.episode += 1
        self.lifted = False
        self.success = False
        self.control_time = 0.0

    def state(self) -> np.ndarray:
        return self.sim_to_dataset(self.data.qpos[self.qpos_addresses]).copy()

    def diagnostic_state(self) -> dict[str, float]:
        cube = self.data.xpos[self.cube_body_id]
        target = self.data.xpos[self.target_body_id]
        gripper = self.data.site_xpos[self.gripper_site_id]
        return {
            "ee_cube_distance_m": float(np.linalg.norm(gripper - cube)),
            "cube_target_xy_distance_m": float(np.linalg.norm(cube[:2] - target[:2])),
            "cube_height_m": float(cube[2]),
            "gripper_opening": float(self.state()[5]),
        }

    @staticmethod
    def _rotation_6d(rotation: np.ndarray) -> np.ndarray:
        return np.asarray(rotation, dtype=np.float32)[:, :2].T.reshape(-1)

    def _end_effector_in_base(self, data) -> tuple[np.ndarray, np.ndarray]:
        base_position = data.site_xpos[self.base_site_id]
        base_rotation = data.site_xmat[self.base_site_id].reshape(3, 3)
        ee_position = data.site_xpos[self.gripper_site_id]
        ee_rotation = data.site_xmat[self.gripper_site_id].reshape(3, 3)
        return base_rotation.T @ (ee_position - base_position), base_rotation.T @ ee_rotation

    def canonical_state(self) -> np.ndarray:
        position, rotation = self._end_effector_in_base(self.data)
        gripper = np.clip(
            (self.state()[5] - GRIPPER_DATASET_RANGE[0]) / np.ptp(GRIPPER_DATASET_RANGE), 0.0, 1.0
        )
        state = np.zeros((1, 10), dtype=np.float32)
        state[0] = np.concatenate(
            (position.astype(np.float32), self._rotation_6d(rotation), np.array([gripper], dtype=np.float32))
        )
        return state

    def canonical_arm_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite six-dimensional vector")
        current_position, current_rotation = self._end_effector_in_base(self.data)
        self.fk_data.qpos[:] = self.data.qpos
        target = np.clip(self.dataset_to_sim(action), self.control_ranges[:, 0], self.control_ranges[:, 1])
        self.fk_data.qpos[self.qpos_addresses] = target
        self.mujoco.mj_kinematics(self.model, self.fk_data)
        target_position, target_rotation = self._end_effector_in_base(self.fk_data)
        relative_rotation = current_rotation.T @ target_rotation
        gripper = np.clip(
            (action[5] - GRIPPER_DATASET_RANGE[0]) / np.ptp(GRIPPER_DATASET_RANGE), 0.0, 1.0
        )
        canonical = np.zeros((1, 10), dtype=np.float32)
        canonical[0] = np.concatenate(
            (
                (target_position - current_position).astype(np.float32),
                self._rotation_6d(relative_rotation),
                np.array([gripper], dtype=np.float32),
            )
        )
        return canonical

    @staticmethod
    def _rotation_6d_matrix(rotation: np.ndarray) -> np.ndarray:
        first = np.asarray(rotation[:3], dtype=np.float64)
        first /= max(np.linalg.norm(first), 1e-8)
        second = np.asarray(rotation[3:6], dtype=np.float64)
        second -= first * np.dot(first, second)
        second /= max(np.linalg.norm(second), 1e-8)
        third = np.cross(first, second)
        return np.stack((first, second, third), axis=1)

    def native_action_from_canonical(self, canonical: np.ndarray, iterations: int = 12) -> np.ndarray:
        """Decode one canonical command with deterministic damped least-squares IK."""
        canonical = np.asarray(canonical, dtype=np.float64)
        if canonical.shape == (1, 10):
            canonical = canonical[0]
        if canonical.shape != (10,) or not np.isfinite(canonical).all():
            raise ValueError("canonical command must be a finite width-10 block")
        current_position_base, current_rotation_base = self._end_effector_in_base(self.data)
        target_position_base = current_position_base + canonical[:3]
        target_rotation_base = current_rotation_base @ self._rotation_6d_matrix(canonical[3:9])
        base_position = self.data.site_xpos[self.base_site_id].copy()
        base_rotation = self.data.site_xmat[self.base_site_id].reshape(3, 3).copy()
        target_position_world = base_position + base_rotation @ target_position_base
        target_rotation_world = base_rotation @ target_rotation_base

        candidate = self.data.qpos[self.qpos_addresses[:5]].copy()
        arm_joint_ids = self.joint_ids[:5]
        arm_dofs = self.model.jnt_dofadr[arm_joint_ids]
        lower = self.model.jnt_range[arm_joint_ids, 0]
        upper = self.model.jnt_range[arm_joint_ids, 1]
        for _ in range(iterations):
            self.fk_data.qpos[:] = self.data.qpos
            self.fk_data.qpos[self.qpos_addresses[:5]] = candidate
            self.mujoco.mj_forward(self.model, self.fk_data)
            position = self.fk_data.site_xpos[self.gripper_site_id]
            rotation = self.fk_data.site_xmat[self.gripper_site_id].reshape(3, 3)
            position_error = target_position_world - position
            rotation_error = 0.5 * sum(
                np.cross(rotation[:, axis], target_rotation_world[:, axis]) for axis in range(3)
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
                self.gripper_site_id,
            )
            characteristic_length = 0.06
            jacobian = np.vstack(
                (
                    jacobian_position[:, arm_dofs],
                    characteristic_length * jacobian_rotation[:, arm_dofs],
                )
            )
            error = np.concatenate((position_error, characteristic_length * rotation_error))
            damping = 0.01
            correction = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            candidate = np.clip(candidate + np.clip(correction, -0.12, 0.12), lower, upper)
        action = np.empty(6, dtype=np.float32)
        action[:5] = np.rad2deg(candidate)
        action[5] = np.clip(canonical[9], 0.0, 1.0) * np.ptp(GRIPPER_DATASET_RANGE)
        return action

    def apply_action(self, action: np.ndarray) -> None:
        target = self.dataset_to_sim(action)
        self.data.ctrl[:] = np.clip(target, self.control_ranges[:, 0], self.control_ranges[:, 1])

    def step_control_period(self, frequency: float = 30.0) -> None:
        self.control_time += 1.0 / frequency
        while self.data.time + 1e-9 < self.control_time:
            self.mujoco.mj_step(self.model, self.data)
        cube_position = self.data.xpos[self.cube_body_id]
        target_position = self.data.xpos[self.target_body_id]
        self.lifted = self.lifted or cube_position[2] > 0.07
        in_target = np.linalg.norm(cube_position[:2] - target_position[:2]) < 0.055
        self.success = self.lifted and in_target and cube_position[2] < 0.09

    def render_top(self) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("environment was created without rendering")
        self.renderer.update_scene(self.data, camera="top")
        return self.renderer.render().copy()

    def render_wrist(self) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("environment was created without rendering")
        self.renderer.update_scene(self.data, camera="wrist_cam")
        # The physical feed is vertically inverted; preserve claw handedness.
        return np.flipud(self.renderer.render()).copy()

    def render_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        return self.render_top(), self.render_wrist()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
