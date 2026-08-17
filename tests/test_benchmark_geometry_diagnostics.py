import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodi.sim.benchmark import BenchmarkDefinition, ScenarioManifest
from embodi.sim.benchmark_geometry_diagnostics import (
    HORIZONS,
    _append_outcome,
    _lead_accumulators,
    _late_initial_ratio,
    _load_or_create_report,
    _metric_group,
    _new_report,
    _outcome_chain,
    _registered_output,
    _shard_aggregate,
    _validate_outcomes,
    accumulate_chunk_consistency,
    aggregate_registered_shards,
    classify_distribution_shift,
    classify_horizon_mechanism,
    compact_accumulator,
    compose_absolute_targets,
    diagnostic_scenarios,
    expected_outcome_ids,
    merge_compact,
    parse_args,
    registered_shard_filename,
    run_diagnostic_episode,
    update_compact,
    validate_h16_reproduction,
)
from embodi.sim.tasks import TaskStatus


ROOT = Path(__file__).parents[1]


def _rotation_6d(rotation: np.ndarray) -> np.ndarray:
    return rotation[:, :2].T.reshape(-1)


def test_exact_diagnostic_subset_has_seven_axes_per_cell() -> None:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    selected = diagnostic_scenarios(manifest, "so101", registered=True)
    assert len(selected) == 21
    assert expected_outcome_ids(selected)[0].endswith("push_to_zone-0000@h1")
    assert expected_outcome_ids(selected)[-1].endswith("pick_place_bin-0006@h16")
    for task in ("push_to_zone", "lift_object", "pick_place_bin"):
        cell = [scenario for scenario in selected if scenario.task == task]
        assert [scenario.scenario_id[-4:] for scenario in cell] == [f"{index:04d}" for index in range(7)]
        axes = [
            name
            for scenario in cell
            for name, value in scenario.factors.items()
            if value["bin"].startswith("development:")
        ]
        assert len(set(axes)) == len(axes) == 7


def test_absolute_targets_compose_translation_rotation_and_native_gripper() -> None:
    angle = np.pi / 2
    anchor_rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    relative_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    anchor = np.r_[np.array([1.0, 2.0, 3.0]), _rotation_6d(anchor_rotation), 0.25]
    chunk = np.zeros((32, 1, 10))
    chunk[:, 0, :3] = [0.1, -0.2, 0.3]
    chunk[:, 0, 3:9] = _rotation_6d(relative_rotation)
    chunk[:, 0, 9] = 0.5
    targets = compose_absolute_targets(anchor, chunk, gripper_native_scale=35.0)
    assert targets["translation_m"][0] == pytest.approx([1.1, 1.8, 3.3])
    assert targets["rotation"][0] == pytest.approx(anchor_rotation @ relative_rotation)
    assert targets["gripper_native"][0] == pytest.approx(17.5)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_overlap_uses_old_h_plus_k_against_new_k(horizon: int) -> None:
    previous = {
        "translation_m": np.zeros((32, 3)),
        "rotation": np.repeat(np.eye(3)[None], 32, axis=0),
        "gripper_native": np.zeros(32),
    }
    current = {
        "translation_m": np.zeros((32, 3)),
        "rotation": np.repeat(np.eye(3)[None], 32, axis=0),
        "gripper_native": np.zeros(32),
    }
    for lead in range(32 - horizon):
        previous["translation_m"][horizon + lead, 0] = lead
        previous["gripper_native"][horizon + lead] = lead
        current["translation_m"][lead, 0] = lead
        current["gripper_native"][lead] = lead
    overlap = _metric_group(32 - horizon)
    accumulate_chunk_consistency(_metric_group(31), overlap, current, previous, horizon)
    assert all(value["count"] == 1 for value in overlap["translation_m"].values())
    assert all(value["maximum"] == 0 for value in overlap["translation_m"].values())
    assert all(value["maximum"] == 0 for value in overlap["gripper_native"].values())


class FakeProcessor:
    def __call__(self, images, instructions, state, **kwargs):
        assert state is None
        return kwargs


class FakePolicy:
    config = SimpleNamespace(
        image_do_rescale=False,
        backbone_name="fake",
        backbone_revision="fake-revision",
    )

    def __init__(self) -> None:
        self.reset_calls = 0
        self.predict_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_canonical_action_chunk(self, _batch):
        self.predict_calls += 1
        chunk = torch.zeros((1, 32, 1, 10), dtype=torch.float32)
        chunk[..., 3] = 1
        chunk[..., 7] = 1
        return chunk


class FakeExpert:
    def __init__(self, env) -> None:
        self.env = env
        self.calls = 0

    @property
    def terminal(self) -> bool:
        return False

    def action(self):
        self.calls += 1
        value = np.zeros(10, dtype=np.float32)
        value[3] = value[7] = 1
        return value


class FakeEpisodeEnv:
    instruction = "test"
    lifted = False

    def __init__(self, terminal_step: int = 5, task: str = "lift_object") -> None:
        self.morphology = SimpleNamespace(id="panda", native_action_dim=10)
        self.task = SimpleNamespace(id=task)
        self.scenario = SimpleNamespace(target_tolerance=0.055)
        self.terminal_step = terminal_step
        self.steps = 0
        self.decode_anchor_steps = []
        self.status_calls = 0
        self.applied = 0

    def render_top(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def canonical_state(self):
        state = np.zeros((1, 10), dtype=np.float32)
        state[0, :3] = self.steps
        state[0, 3] = state[0, 7] = 1
        return state

    def native_action_from_canonical(self, canonical):
        self.decode_anchor_steps.append(self.steps)
        return np.asarray(canonical, dtype=np.float32).reshape(10)

    def canonical_arm_action(self, native):
        return np.asarray(native, dtype=np.float32).reshape(1, 10)

    @staticmethod
    def native_to_qpos(native):
        return np.asarray(native)

    @staticmethod
    def qpos_to_native(qpos):
        return np.asarray(qpos)

    def apply_action(self, _native):
        self.applied += 1

    def step_control_period(self, frequency):
        assert frequency == 30.0
        self.steps += 1
        self.status_calls += 1
        terminal = self.steps == self.terminal_step
        return TaskStatus(success=terminal, terminal=terminal, stage="success" if terminal else "lift")

    def task_status(self):
        raise AssertionError("task_status must not be called outside step_control_period")

    def diagnostic_state(self):
        return {
            "ee_object_distance_m": 0.05 if self.steps >= 2 else 0.1,
            "object_target_xy_distance_m": 0.04,
            "object_height_m": 0.09 if self.steps >= 3 else 0.02,
            "object_speed": 0.01,
            "gripper_opening_fraction": 0.1,
        }


def test_episode_decodes_before_step_calls_shadow_once_and_uses_returned_status() -> None:
    experts = []

    def expert_factory(env):
        expert = FakeExpert(env)
        experts.append(expert)
        return expert

    env = FakeEpisodeEnv(terminal_step=5)
    policy = FakePolicy()
    result = run_diagnostic_episode(
        env,
        policy,
        FakeProcessor(),
        "cpu",
        execution_horizon=4,
        expert_factory=expert_factory,
    )
    assert result["success"] is True
    assert result["steps"] == result["commands"] == env.status_calls == env.applied == 5
    assert result["shadow_expert"]["calls"] == experts[0].calls == 5
    assert env.decode_anchor_steps == [0] * 32 + [4] * 32
    assert policy.predict_calls == 2
    assert result["milestones"]["near_ee_object"] == {"attained": True, "first_step": 2}
    assert result["milestones"]["lifted"] == {"attained": True, "first_step": 3}
    assert result["milestones"]["lift_predicate"] == {"attained": True, "first_step": 3}
    assert result["milestones"]["success"] == {"attained": True, "first_step": 5}


def test_compact_accumulator_math_and_invalid_values() -> None:
    first = compact_accumulator()
    for value in (1.0, 2.0, 3.0):
        update_compact(first, value)
    second = compact_accumulator()
    update_compact(second, 4.0)
    assert first == {"count": 3, "sum": 6.0, "sum_squared": 14.0, "maximum": 3.0}
    assert merge_compact([first, second]) == {
        "count": 4,
        "sum": 10.0,
        "sum_squared": 30.0,
        "maximum": 4.0,
    }
    with pytest.raises(ValueError, match="finite and nonnegative"):
        update_compact(first, float("nan"))


def test_resume_rejects_contract_mismatch_and_chain_corruption(tmp_path: Path) -> None:
    provenance = {"git_dirty": True, "git_commit": "test", "evaluator_sha256": "hash"}
    expected = _new_report(
        registered=False,
        contract={"contract": "first"},
        identity={"checkpoint": "same"},
        expected_ids=[],
        provenance=provenance,
    )
    output = tmp_path / "report.json"
    output.write_text(json.dumps(expected))
    changed = _new_report(
        registered=False,
        contract={"contract": "changed"},
        identity={"checkpoint": "same"},
        expected_ids=[],
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="contract, checkpoint, or evaluator"):
        _load_or_create_report(output, changed)
    expected["outcome_chain_sha256"] = "0" * 64
    output.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match="hash chain"):
        _load_or_create_report(output, expected)


def test_h16_reproduction_mismatch_rejected() -> None:
    source = {
        "success": True,
        "steps": 1,
        "chunks": 1,
        "final_stage": "success",
        "failure_reason": None,
        "commands": 1,
        "clipped_commands": 0,
    }
    outcome = {
        "scenario_id": "scenario",
        "execution_horizon": 16,
        "success": False,
        "steps": 1,
        "chunks": 1,
        "final_stage": "failure",
        "failure_reason": "failure",
        "commands": 1,
        "clipped_commands": 0,
        "exp37_expected_success": True,
        "exp37_success_reproduced": False,
        "exp37_expected_trajectory": source,
        "exp37_trajectory_reproduced": False,
    }
    with pytest.raises(ValueError, match="exactly reproduce"):
        validate_h16_reproduction([outcome], {"scenario": source})


def test_h16_trajectory_length_difference_is_descriptive() -> None:
    source = {
        "success": True,
        "steps": 481,
        "chunks": 31,
        "final_stage": "success",
        "failure_reason": None,
        "commands": 481,
        "clipped_commands": 0,
    }
    outcome = {
        **source,
        "scenario_id": "scenario",
        "execution_horizon": 16,
        "steps": 467,
        "chunks": 30,
        "commands": 467,
        "exp37_expected_success": True,
        "exp37_success_reproduced": True,
        "exp37_expected_trajectory": source,
        "exp37_trajectory_reproduced": False,
    }
    validate_h16_reproduction([outcome], {"scenario": source})


def test_shard_aggregate_retains_chunk_consistency_by_lead() -> None:
    outcome = run_diagnostic_episode(
        FakeEpisodeEnv(terminal_step=5),
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        execution_horizon=4,
        expert_factory=FakeExpert,
    )
    outcome["task"] = "lift_object"
    aggregate = _shard_aggregate([outcome])
    adjacent = aggregate["chunk_consistency"]["h4"]["adjacent"]
    overlap = aggregate["chunk_consistency"]["h4"]["cross_replan_overlap"]
    assert adjacent["by_lead"]["translation_m"]["0"]["count"] == 2
    assert overlap["by_lead"]["translation_m"]["0"]["count"] == 1
    assert aggregate["native_clipping"]["decoded_commands"] == 64


def test_outcome_validation_rejects_lost_reconstruction_samples() -> None:
    outcome = run_diagnostic_episode(
        FakeEpisodeEnv(terminal_step=5),
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        execution_horizon=4,
        expert_factory=FakeExpert,
    )
    outcome.update(
        outcome_id="development-panda-lift_object-0000@h4",
        scenario_id="development-panda-lift_object-0000",
        scenario_seed=1,
        morphology="panda",
        task="lift_object",
        active_factor_axes=["workspace"],
    )
    _validate_outcomes({"outcomes": [outcome]})
    outcome["policy_reconstruction"]["by_lead"]["translation_m"]["0"]["count"] -= 1
    with pytest.raises(ValueError, match="counts do not match chunk decoding"):
        _validate_outcomes({"outcomes": [outcome]})
    outcome["policy_reconstruction"]["by_lead"]["translation_m"]["0"]["count"] += 1
    outcome["policy_reconstruction"]["distribution"]["translation_m"]["counts"][0] -= 1
    with pytest.raises(ValueError, match="histogram count"):
        _validate_outcomes({"outcomes": [outcome]})


def test_zero_initial_shadow_error_ratio_is_explicit() -> None:
    assert _late_initial_ratio(0.0, 1.0) == "infinite"
    assert _late_initial_ratio(0.0, 0.0) == "undefined"


def test_aggregate_requires_all_twelve_shards() -> None:
    with pytest.raises(ValueError, match="all 12 shards"):
        aggregate_registered_shards(
            [],
            SimpleNamespace(scenarios=()),
            source_reports={},
            exp36_summary={"experiment": 36, "status": "completed_gate_passed", "fixed_step": 1200},
        )


def test_classification_arithmetic() -> None:
    strong = classify_horizon_mechanism(
        {
            36001: {"h1_success": 0.0, "h16_success": 0.1, "h1_milestones": 0.2, "h16_milestones": 0.3},
            36002: {"h1_success": 0.1, "h16_success": 0.15, "h1_milestones": 0.2, "h16_milestones": 0.25},
            36003: {"h1_success": 0.2, "h16_success": 0.2, "h1_milestones": 0.2, "h16_milestones": 0.2},
        }
    )
    assert strong["classification"] == "strong"
    assert strong["three_seed_mean_paired_success_gain_percentage_points"] == pytest.approx(5.0)
    partial = classify_horizon_mechanism(
        {
            seed: {"h1_success": 0.1, "h16_success": 0.11, "h1_milestones": 0.2, "h16_milestones": 0.2}
            for seed in (36001, 36002, 36003)
        }
    )
    assert partial["classification"] == "partial"
    association = classify_distribution_shift({36001: 2.0, 36002: 2.5, 36003: 1.0})
    assert association["classification"] == "strong_association"
    assert association["causal_claim"] is False


def test_cli_and_registered_output_contracts(tmp_path: Path) -> None:
    args = parse_args(
        [
            "shard",
            "--condition",
            "matched",
            "--seed",
            "36001",
            "--morphology",
            "panda",
            "--smoke",
            "--limit-scenarios",
            "1",
            "--output",
            str(tmp_path / "smoke.json"),
        ]
    )
    assert args.limit_scenarios == 1
    filename = registered_shard_filename("matched", 36001, "panda")
    assert filename == "benchmark-exp38-matched-seed36001-panda-diagnostic.json"
    assert _registered_output("matched", 36001, "panda", tmp_path / filename).name == filename
    with pytest.raises(ValueError, match="output filename"):
        _registered_output("matched", 36001, "panda", tmp_path / "wrong.json")


def test_outcome_chain_changes_when_compact_diagnostic_changes() -> None:
    report = {"outcomes": [], "outcome_chain_sha256": _outcome_chain([])}
    outcome = {"value": _lead_accumulators(1)}
    _append_outcome(report, outcome)
    original = report["outcome_chain_sha256"]
    outcome["value"]["0"]["sum"] = 1.0
    assert _outcome_chain(report["outcomes"]) != original
