from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_POSITION_NAMES = tuple(f"{name}.pos" for name in SO101_JOINT_NAMES)
BODY_LIMITS_DEGREES = np.array(
    [[-110.0, 110.0], [-100.0, 100.0], [-96.8, 96.8], [-95.0, 95.0], [-157.2, 157.2]],
    dtype=np.float32,
)


def resolve_so101_order(names: Sequence[str]) -> tuple[int, ...]:
    normalized = [name.removesuffix(".pos") for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError("SO101 feature names must be unique")
    missing = [name for name in SO101_JOINT_NAMES if name not in normalized]
    extra = [name for name in normalized if name not in SO101_JOINT_NAMES]
    if missing or extra:
        raise ValueError(f"SO101 feature names mismatch: missing={missing} extra={extra}")
    return tuple(normalized.index(name) for name in SO101_JOINT_NAMES)


@dataclass(frozen=True)
class SO101CanonicalCalibration:
    joint_signs: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1)
    joint_offsets_degrees: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0)
    gripper_closed_native: float = 0.0
    gripper_open_native: float = 100.0
    source_calibration_sha256: str = ""
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("unsupported SO101 canonical calibration format")
        if len(self.joint_signs) != 5 or any(sign not in (-1, 1) for sign in self.joint_signs):
            raise ValueError("joint_signs must contain five values chosen from -1 and 1")
        if len(self.joint_offsets_degrees) != 5 or not np.isfinite(self.joint_offsets_degrees).all():
            raise ValueError("joint_offsets_degrees must contain five finite values")
        if not np.isfinite((self.gripper_closed_native, self.gripper_open_native)).all():
            raise ValueError("gripper endpoints must be finite")
        if self.gripper_closed_native == self.gripper_open_native:
            raise ValueError("gripper endpoints must differ")

    @classmethod
    def load(cls, path: str | Path) -> SO101CanonicalCalibration:
        values = json.loads(Path(path).read_text())
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown SO101 calibration fields: {unknown}")
        for name in ("joint_signs", "joint_offsets_degrees"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def to_model_state(self, values: Sequence[float], names: Sequence[str]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("SO101 state/action must be a finite six-dimensional vector")
        ordered = values[list(resolve_so101_order(names))]
        result = np.empty(6, dtype=np.float32)
        result[:5] = ordered[:5] * np.asarray(self.joint_signs) + np.asarray(
            self.joint_offsets_degrees
        )
        if np.any(result[:5] < BODY_LIMITS_DEGREES[:, 0]) or np.any(
            result[:5] > BODY_LIMITS_DEGREES[:, 1]
        ):
            raise ValueError("calibrated SO101 body state is outside the model limits")
        opening = (ordered[5] - self.gripper_closed_native) / (
            self.gripper_open_native - self.gripper_closed_native
        )
        result[5] = np.clip(opening, 0.0, 1.0) * 35.0
        return result

    def provenance(self, source_path: str | Path) -> dict:
        path = Path(source_path)
        return {
            **asdict(self),
            "calibration_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "joint_names": list(SO101_POSITION_NAMES),
            "base_frame": "baseframe",
            "tool_frame": "gripperframe",
            "rotation6d": "first_two_columns",
        }

    def verify_source_calibration(self, source_path: str | Path) -> str:
        path = Path(source_path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(self.source_calibration_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_calibration_sha256.lower()
        ):
            raise ValueError("source_calibration_sha256 must be a 64-character sha-256")
        if actual != self.source_calibration_sha256.lower():
            raise ValueError("canonical calibration does not match the LeRobot calibration file")
        return actual


class SO101Canonicalizer:
    """Generate canonical labels with the same pinned FK convention as simulation."""

    def __init__(self, calibration: SO101CanonicalCalibration) -> None:
        from embodi.sim.environment import SO101PickPlaceEnv

        self.calibration = calibration
        self.environment = SO101PickPlaceEnv(width=64, height=48, seed=0, render=False)

    def labels(
        self,
        state: Sequence[float],
        action: Sequence[float],
        state_names: Sequence[str],
        action_names: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        model_state = self.calibration.to_model_state(state, state_names)
        model_action = self.calibration.to_model_state(action, action_names)
        env = self.environment
        env.data.qpos[env.qpos_addresses] = env.dataset_to_sim(model_state)
        env.mujoco.mj_forward(env.model, env.data)
        return env.canonical_state(), env.canonical_arm_action(model_action)

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> SO101Canonicalizer:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
