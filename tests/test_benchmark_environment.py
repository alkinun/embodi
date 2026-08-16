import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from embodi.sim.benchmark_environment import BenchmarkManipulationEnv, default_scenario
from embodi.sim.benchmark_expert import (
    BenchmarkPhase,
    PrivilegedBenchmarkExpert,
    run_benchmark_expert_episode,
)


@pytest.mark.parametrize("morphology", ["so101", "panda"])
@pytest.mark.parametrize("task", ["push_to_zone", "lift_object", "pick_place_bin", "stack_object"])
def test_benchmark_cells_compile_and_reset(morphology: str, task: str) -> None:
    env = BenchmarkManipulationEnv(morphology, task, width=64, height=48, render=False)
    try:
        assert env.instruction
        assert env.state().shape == (6 if morphology == "so101" else 8,)
        assert env.canonical_state().shape == (1, 10)
        assert np.isfinite(env.canonical_state()).all()
        assert env.task_status().success is False
        assert np.isfinite(list(env.diagnostic_state().values())).all()
    finally:
        env.close()


@pytest.mark.parametrize("morphology", ["so101", "panda"])
def test_native_state_round_trip(morphology: str) -> None:
    env = BenchmarkManipulationEnv(morphology, "pick_place_bin", render=False)
    try:
        native = env.morphology.home_native_state
        np.testing.assert_allclose(env.qpos_to_native(env.native_to_qpos(native)), native, atol=1e-4)
    finally:
        env.close()


@pytest.mark.parametrize("morphology", ["so101", "panda"])
def test_deterministic_ik_reconstructs_small_canonical_target(morphology: str) -> None:
    env = BenchmarkManipulationEnv(morphology, "pick_place_bin", render=False)
    try:
        target = env.state()
        if morphology == "so101":
            target[:5] += np.array([0.3, -0.25, 0.2, -0.15, 0.1], dtype=np.float32)
            target[5] = 16.0
        else:
            target[:7] += np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01])
            target[7] = 0.04
        canonical = env.canonical_arm_action(target)
        decoded = env.native_action_from_canonical(canonical)
        reconstructed = env.canonical_arm_action(decoded)
        np.testing.assert_allclose(reconstructed[0, :3], canonical[0, :3], atol=2e-3)
        np.testing.assert_allclose(reconstructed[0, 9], canonical[0, 9], atol=1e-5)
    finally:
        env.close()


def test_default_scenarios_are_morphology_specific() -> None:
    assert default_scenario("so101").object_xy != default_scenario("panda").object_xy
    assert default_scenario("so101", "push_to_zone").target_xy != default_scenario(
        "so101", "pick_place_bin"
    ).target_xy
    with pytest.raises(ValueError, match="unknown morphology"):
        default_scenario("missing")


@pytest.mark.parametrize("morphology", ["so101", "panda"])
def test_privileged_expert_emits_finite_native_actions(morphology: str) -> None:
    env = BenchmarkManipulationEnv(morphology, "pick_place_bin", render=False)
    try:
        expert = PrivilegedBenchmarkExpert(env)
        for _ in range(5):
            action = expert.action()
            assert action.shape == env.state().shape
            assert np.isfinite(action).all()
            env.apply_action(action)
            env.step_control_period()
        assert expert.phase == BenchmarkPhase.OPEN
    finally:
        env.close()


@pytest.mark.parametrize("morphology", ["so101", "panda"])
@pytest.mark.parametrize("task", ["push_to_zone", "lift_object", "pick_place_bin", "stack_object"])
def test_default_benchmark_cell_passes_privileged_admission(morphology: str, task: str) -> None:
    env = BenchmarkManipulationEnv(morphology, task, render=False)
    try:
        success, phase = run_benchmark_expert_episode(env, max_steps=500)
        assert success, (morphology, task, phase, env.diagnostic_state())
        assert env.success
    finally:
        env.close()
