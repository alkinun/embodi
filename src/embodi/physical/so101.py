from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import sys
import tempfile
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
GRIPPER_ENDPOINT_TOLERANCE = 1.0


def resolve_so101_order(names: Sequence[str]) -> tuple[int, ...]:
    names = list(names)
    if len(names) != len(set(names)):
        raise ValueError("SO101 feature names must be unique")
    missing = [name for name in SO101_POSITION_NAMES if name not in names]
    extra = [name for name in names if name not in SO101_POSITION_NAMES]
    if missing or extra:
        raise ValueError(f"SO101 feature names mismatch: missing={missing} extra={extra}")
    return tuple(names.index(name) for name in SO101_POSITION_NAMES)


@dataclass(frozen=True)
class SO101CanonicalCalibration:
    joint_signs: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1)
    joint_offsets_degrees: tuple[float, float, float, float, float] = (0, 0, 0, 0, 0)
    gripper_closed_native: float = 0.0
    gripper_open_native: float = 100.0
    source_calibration_sha256: str = ""
    physical_validation_complete: bool = False
    format_version: int = 1

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("unsupported SO101 canonical calibration format")
        if not isinstance(self.joint_signs, tuple) or len(self.joint_signs) != 5 or any(
            type(sign) is not int or sign not in (-1, 1) for sign in self.joint_signs
        ):
            raise ValueError("joint_signs must contain five values chosen from -1 and 1")
        if (
            not isinstance(self.joint_offsets_degrees, tuple)
            or len(self.joint_offsets_degrees) != 5
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in self.joint_offsets_degrees)
            or not np.isfinite(self.joint_offsets_degrees).all()
        ):
            raise ValueError("joint_offsets_degrees must contain five finite values")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (self.gripper_closed_native, self.gripper_open_native)
        ) or not np.isfinite((self.gripper_closed_native, self.gripper_open_native)).all():
            raise ValueError("gripper endpoints must be finite")
        if not 0.0 <= self.gripper_closed_native <= 100.0 or not 0.0 <= self.gripper_open_native <= 100.0:
            raise ValueError("gripper endpoints must be in the native [0, 100] range")
        if abs(self.gripper_closed_native - self.gripper_open_native) < 5.0:
            raise ValueError("gripper endpoints must differ by at least 5 native units")
        if type(self.physical_validation_complete) is not bool:
            raise ValueError("physical_validation_complete must be boolean")
        if not isinstance(self.source_calibration_sha256, str):
            raise ValueError("source_calibration_sha256 must be a string")

    @classmethod
    def from_json(cls, content: str) -> SO101CanonicalCalibration:
        def reject_duplicates(pairs):
            values = {}
            for key, value in pairs:
                if key in values:
                    raise ValueError(f"duplicate SO101 calibration field: {key}")
                values[key] = value
            return values

        values = json.loads(content, object_pairs_hook=reject_duplicates)
        if not isinstance(values, dict):
            raise ValueError("SO101 calibration must be a JSON object")
        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - field_names)
        if unknown:
            raise ValueError(f"unknown SO101 calibration fields: {unknown}")
        missing = sorted(field_names - set(values))
        if missing:
            raise ValueError(f"missing SO101 calibration fields: {missing}")
        for name in ("joint_signs", "joint_offsets_degrees"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> SO101CanonicalCalibration:
        return cls.from_json(Path(path).read_text())

    @classmethod
    def load_snapshot(cls, path: str | Path) -> tuple[SO101CanonicalCalibration, str]:
        content = Path(path).read_bytes()
        return cls.from_json(content.decode()), hashlib.sha256(content).hexdigest()

    def require_physical_validation(self) -> None:
        if not self.physical_validation_complete:
            raise ValueError("SO101 canonical calibration has not completed physical validation")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(asdict(self), stream, indent=2)
            stream.write("\n")
            stream.flush()
        try:
            temporary_path.replace(path)
        finally:
            active_error = sys.exception()
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if active_error is None:
                    raise
                active_error.add_note(f"calibration save cleanup failed: {cleanup_error!r}")

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
        if not -GRIPPER_ENDPOINT_TOLERANCE / abs(
            self.gripper_open_native - self.gripper_closed_native
        ) <= opening <= 1.0 + GRIPPER_ENDPOINT_TOLERANCE / abs(
            self.gripper_open_native - self.gripper_closed_native
        ):
            raise ValueError("SO101 gripper state is outside calibrated endpoints")
        result[5] = np.clip(opening, 0.0, 1.0) * 35.0
        return result

    def provenance(self, source_path: str | Path, file_sha256: str | None = None) -> dict:
        path = Path(source_path)
        return {
            **asdict(self),
            "calibration_file_sha256": file_sha256
            if file_sha256 is not None
            else hashlib.sha256(path.read_bytes()).hexdigest(),
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
