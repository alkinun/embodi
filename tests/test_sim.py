import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from embodi.sim.environment import HOME_STATE, SO101PickPlaceEnv


def test_state_action_conversion_round_trip() -> None:
    simulated = SO101PickPlaceEnv.dataset_to_sim(HOME_STATE)
    restored = SO101PickPlaceEnv.sim_to_dataset(simulated)
    np.testing.assert_allclose(restored, HOME_STATE, atol=1e-4)


def test_state_action_conversion_clips_limits() -> None:
    simulated = SO101PickPlaceEnv.dataset_to_sim(np.array([999, -999, 999, -999, 999, 999]))
    assert np.isfinite(simulated).all()
    assert simulated[-1] <= 1.7453292


def test_rotation_6d_uses_matrix_columns() -> None:
    rotation = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_array_equal(SO101PickPlaceEnv._rotation_6d(rotation), [1, 4, 7, 2, 5, 8])


def test_deterministic_decoder_reconstructs_canonical_fk_target() -> None:
    env = SO101PickPlaceEnv(width=64, height=48, seed=3)
    try:
        action = env.state()
        action[:5] += np.array([0.5, -0.4, 0.3, -0.2, 0.1], dtype=np.float32)
        action[5] = 16.0
        canonical = env.canonical_arm_action(action)
        decoded = env.native_action_from_canonical(canonical)
        reconstructed = env.canonical_arm_action(decoded)
        np.testing.assert_allclose(reconstructed[0, :3], canonical[0, :3], atol=5e-4)
        np.testing.assert_allclose(reconstructed[0, 9], canonical[0, 9], atol=1e-6)
    finally:
        env.close()


def test_environment_uses_configured_cube_range() -> None:
    env = SO101PickPlaceEnv(
        width=64,
        height=48,
        seed=3,
        cube_x_range=(0.25, 0.25),
        cube_y_range=(0.05, 0.05),
    )
    try:
        cube_position = env.data.xpos[env.cube_body_id]
        np.testing.assert_allclose(cube_position[:2], [0.25, 0.05], atol=1e-6)
    finally:
        env.close()


def test_environment_rejects_invalid_cube_range() -> None:
    with pytest.raises(ValueError, match="cube ranges"):
        SO101PickPlaceEnv(cube_x_range=(0.32, 0.28))
