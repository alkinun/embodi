from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .so101 import BODY_LIMITS_DEGREES, SO101_POSITION_NAMES


def observation_vector(observation: Mapping[str, object]) -> np.ndarray:
    missing = [name for name in SO101_POSITION_NAMES if name not in observation]
    if missing:
        raise ValueError(f"SO101 observation is missing positions: {missing}")
    values = np.asarray([observation[name] for name in SO101_POSITION_NAMES], dtype=np.float32)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("SO101 observation positions must be finite scalars")
    if np.any(values[:5] < BODY_LIMITS_DEGREES[:, 0]) or np.any(
        values[:5] > BODY_LIMITS_DEGREES[:, 1]
    ):
        raise ValueError("SO101 body observation is outside conservative limits")
    if not 0.0 <= values[5] <= 100.0:
        raise ValueError("SO101 gripper observation must be in [0, 100]")
    return values


@dataclass(frozen=True)
class SO101SafetyLimits:
    body_delta_degrees: float = 0.5
    gripper_delta_percent: float = 1.0
    body_session_envelope_degrees: float = 5.0
    gripper_session_envelope_percent: float = 5.0

    def __post_init__(self) -> None:
        values = (
            self.body_delta_degrees,
            self.gripper_delta_percent,
            self.body_session_envelope_degrees,
            self.gripper_session_envelope_percent,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("SO101 safety limits must be finite and positive")


def limit_action(
    requested: np.ndarray,
    current: np.ndarray,
    session_start: np.ndarray,
    limits: SO101SafetyLimits = SO101SafetyLimits(),
) -> tuple[np.ndarray, tuple[str, ...]]:
    vectors = [np.asarray(value, dtype=np.float32) for value in (requested, current, session_start)]
    if any(value.shape != (6,) or not np.isfinite(value).all() for value in vectors):
        raise ValueError("requested, current, and session_start must be finite six-dimensional vectors")
    requested, current, session_start = vectors
    if np.any(session_start[:5] < BODY_LIMITS_DEGREES[:, 0]) or np.any(
        session_start[:5] > BODY_LIMITS_DEGREES[:, 1]
    ) or not 0.0 <= session_start[5] <= 100.0:
        raise ValueError("SO101 session start is outside conservative limits")
    lower = np.concatenate(
        (
            np.maximum(
                BODY_LIMITS_DEGREES[:, 0],
                session_start[:5] - limits.body_session_envelope_degrees,
            ),
            [max(0.0, session_start[5] - limits.gripper_session_envelope_percent)],
        )
    )
    upper = np.concatenate(
        (
            np.minimum(
                BODY_LIMITS_DEGREES[:, 1],
                session_start[:5] + limits.body_session_envelope_degrees,
            ),
            [min(100.0, session_start[5] + limits.gripper_session_envelope_percent)],
        )
    )
    delta = np.array([limits.body_delta_degrees] * 5 + [limits.gripper_delta_percent])
    if np.any(current < lower) or np.any(current > upper):
        raise ValueError("SO101 feedback left the startup-centered session envelope")
    command_lower = np.maximum(lower, current - delta)
    command_upper = np.minimum(upper, current + delta)
    if np.any(command_lower > command_upper):
        raise ValueError("SO101 safety limits have no valid command intersection")
    bounded = np.clip(requested, command_lower, command_upper)
    clipped = tuple(
        name
        for name, before, after in zip(SO101_POSITION_NAMES, requested, bounded, strict=True)
        if not np.isclose(before, after)
    )
    return bounded.astype(np.float32), clipped
