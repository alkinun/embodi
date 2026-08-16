from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .assets import MENAGERIE_REVISION, ensure_panda_assets, ensure_so101_assets


SO101_BODY_LIMITS_DEGREES = np.array(
    [
        [-110.0, 110.0],
        [-100.0, 100.0],
        [-96.8, 96.8],
        [-95.0, 95.0],
        [-157.2, 157.2],
    ],
    dtype=np.float32,
)
SO101_GRIPPER_NATIVE_RANGE = np.array([0.0, 35.0], dtype=np.float32)
SO101_HOME_STATE = np.array(
    [0.3516, -86.8359, 75.0586, 72.0703, 82.9688, 0.3453], dtype=np.float32
)
PANDA_HOME_STATE = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.08], dtype=np.float32
)


@dataclass(frozen=True)
class MorphologySpec:
    id: str
    menagerie_directory: str
    model_filename: str
    native_state_names: tuple[str, ...]
    native_action_names: tuple[str, ...]
    arm_joint_names: tuple[str, ...]
    sim_joint_names: tuple[str, ...]
    home_native_state: NDArray[np.float32]
    base_frame: str
    tool_frame: str
    asset_loader: Callable[[Path | None], Path]
    asset_revision: str = MENAGERIE_REVISION

    def __post_init__(self) -> None:
        state = np.asarray(self.home_native_state, dtype=np.float32)
        if not self.id or not self.native_state_names or not self.arm_joint_names:
            raise ValueError("morphology ids and native/arm joint names must not be empty")
        if len(set(self.native_state_names)) != len(self.native_state_names):
            raise ValueError("native state names must be unique")
        if len(set(self.native_action_names)) != len(self.native_action_names):
            raise ValueError("native action names must be unique")
        if len(set(self.sim_joint_names)) != len(self.sim_joint_names):
            raise ValueError("simulation joint names must be unique")
        if state.shape != (len(self.native_state_names),) or not np.isfinite(state).all():
            raise ValueError("home state must match native state names and be finite")
        if len(self.native_state_names) != len(self.native_action_names):
            raise ValueError("benchmark morphologies require equal native state/action dimensions")
        object.__setattr__(self, "home_native_state", state)

    @property
    def native_state_dim(self) -> int:
        return len(self.native_state_names)

    @property
    def native_action_dim(self) -> int:
        return len(self.native_action_names)

    def ensure_assets(self, cache_dir: Path | None = None) -> Path:
        return self.asset_loader(cache_dir)


SO101 = MorphologySpec(
    id="so101",
    menagerie_directory="robotstudio_so101",
    model_filename="so101.xml",
    native_state_names=(
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ),
    native_action_names=(
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ),
    arm_joint_names=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
    sim_joint_names=(
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ),
    home_native_state=SO101_HOME_STATE,
    base_frame="baseframe",
    tool_frame="gripperframe",
    asset_loader=ensure_so101_assets,
)

PANDA = MorphologySpec(
    id="panda",
    menagerie_directory="franka_emika_panda",
    model_filename="panda.xml",
    native_state_names=(
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "gripper_width",
    ),
    native_action_names=(
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "gripper_width",
    ),
    arm_joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
    sim_joint_names=(
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    ),
    home_native_state=PANDA_HOME_STATE,
    base_frame="baseframe",
    tool_frame="gripperframe",
    asset_loader=ensure_panda_assets,
)

MORPHOLOGIES = {morphology.id: morphology for morphology in (SO101, PANDA)}


def morphology_by_id(morphology_id: str) -> MorphologySpec:
    try:
        return MORPHOLOGIES[morphology_id]
    except KeyError as error:
        raise ValueError(f"unknown morphology: {morphology_id}") from error
