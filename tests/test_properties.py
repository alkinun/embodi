from math import cos, pi, sin

from hypothesis import assume, given, settings
from hypothesis import strategies as st
import numpy as np
import torch

from embodi.canonical import matrix_to_rotation_6d, rotation_6d_to_matrix
from embodi.physical.safety import SO101SafetyLimits, limit_action
from embodi.physical.so101 import SO101_POSITION_NAMES


finite_float = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=32,
)


@given(st.lists(finite_float, min_size=6, max_size=6))
@settings(max_examples=100, deadline=None)
def test_rotation_6d_always_produces_a_proper_rotation(values: list[float]) -> None:
    rotation = torch.tensor(values, dtype=torch.float32)
    first = rotation[:3]
    second = rotation[3:]
    assume(torch.linalg.vector_norm(first) > 1e-3)
    first = first / torch.linalg.vector_norm(first)
    assume(torch.linalg.vector_norm(second - torch.dot(first, second) * first) > 1e-3)

    matrix = rotation_6d_to_matrix(rotation)

    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, torch.eye(3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(matrix), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


@given(
    st.floats(-pi, pi, allow_nan=False, allow_infinity=False),
    st.floats(-pi, pi, allow_nan=False, allow_infinity=False),
    st.floats(-pi, pi, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_rotation_matrix_round_trip_preserves_arbitrary_euler_rotation(
    x: float,
    y: float,
    z: float,
) -> None:
    cx, sx = cos(x), sin(x)
    cy, sy = cos(y), sin(y)
    cz, sz = cos(z), sin(z)
    rotation_x = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    rotation_y = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    rotation_z = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    matrix = rotation_z @ rotation_y @ rotation_x

    restored = rotation_6d_to_matrix(matrix_to_rotation_6d(matrix))

    torch.testing.assert_close(restored, matrix, atol=1e-5, rtol=1e-5)


@given(
    st.lists(finite_float, min_size=5, max_size=5),
    st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False, width=32),
    st.lists(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False, width=32), min_size=5, max_size=5),
    st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False, width=32),
    st.lists(finite_float, min_size=6, max_size=6),
)
@settings(max_examples=100, deadline=None)
def test_so101_action_filter_never_exceeds_step_or_session_limits(
    start_body: list[float],
    start_gripper: float,
    body_offset: list[float],
    gripper_offset: float,
    requested_values: list[float],
) -> None:
    limits = SO101SafetyLimits()
    session_start = np.array([*start_body, start_gripper], dtype=np.float32)
    current = session_start + np.array([*body_offset, gripper_offset], dtype=np.float32)
    current[5] = np.clip(current[5], 0.0, 100.0)
    requested = np.array(requested_values, dtype=np.float32)

    bounded, clipped = limit_action(requested, current, session_start, limits)

    session_lower = np.concatenate(
        (session_start[:5] - limits.body_session_envelope_degrees, [max(0.0, session_start[5] - 5.0)])
    )
    session_upper = np.concatenate(
        (session_start[:5] + limits.body_session_envelope_degrees, [min(100.0, session_start[5] + 5.0)])
    )
    step = np.array([limits.body_delta_degrees] * 5 + [limits.gripper_delta_percent])
    tolerance = 1e-5
    assert np.all(bounded >= session_lower - tolerance)
    assert np.all(bounded <= session_upper + tolerance)
    assert np.all(bounded >= current - step - tolerance)
    assert np.all(bounded <= current + step + tolerance)
    assert np.isfinite(bounded).all()
    assert set(clipped) == {
        name
        for name, before, after in zip(SO101_POSITION_NAMES, requested, bounded, strict=True)
        if not np.isclose(before, after)
    }
