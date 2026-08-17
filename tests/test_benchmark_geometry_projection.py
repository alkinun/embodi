import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodi.sim.benchmark import BenchmarkDefinition, Scenario, ScenarioManifest
from embodi.sim.benchmark import file_sha256
import embodi.sim.benchmark_geometry_projection as projection
from embodi.sim.benchmark_geometry_projection import (
    ALPHA_GRID,
    EXP38_EVALUATOR_SHA256,
    FIRST_CHUNK_MAX_ABS,
    SEEDS,
    TreatmentDeliveryError,
    _load_or_create_report,
    _new_report,
    _outcome_chain,
    _shard_aggregate,
    _registered_output,
    _validate_outcomes,
    aggregate_registered_shards,
    canonical_rotation_6d,
    classify_causal_support,
    pair_arm_order,
    parse_args,
    project_feasible_command,
    projection_scenarios,
    registered_shard_filename,
    rotation_matrix_from_6d,
    run_projection_episode,
    run_projection_pair,
    scale_canonical_command,
    scale_rotation_shortest_path,
    update_ordered_milestones,
    validate_exp38_summary,
    validate_rotation_matrix,
)
from embodi.sim.tasks import TaskStatus


ROOT = Path(__file__).parents[1]


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def _angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _chunk(translation: float = 1.0, gripper: float = 0.3) -> np.ndarray:
    value = np.zeros((32, 1, 10), dtype=np.float32)
    value[:, 0, 0] = translation
    value[:, 0, 3:9] = canonical_rotation_6d(np.eye(3))
    value[:, 0, 9] = gripper
    return value


def test_registered_slice_uses_fresh_seven_axes_per_cell() -> None:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    selected = projection_scenarios(manifest, "so101", registered=True)
    assert len(selected) == 21
    assert selected[0].scenario_id.endswith("push_to_zone-0007")
    assert selected[-1].scenario_id.endswith("pick_place_bin-0013")
    for task in ("push_to_zone", "lift_object", "pick_place_bin"):
        cell = [scenario for scenario in selected if scenario.task == task]
        assert [scenario.scenario_id[-4:] for scenario in cell] == [
            f"{index:04d}" for index in range(7, 14)
        ]
        axes = [
            name
            for scenario in cell
            for name, factor in scenario.factors.items()
            if factor["bin"].startswith("development:")
        ]
        assert len(axes) == len(set(axes)) == 7


@pytest.mark.parametrize(
    ("angle", "alpha"),
    [(0.0, 0.5), (1e-10, 0.5), (np.pi / 3, 0.25), (np.pi, 0.5), (np.pi, 1.0)],
)
def test_so3_shortest_path_scaling_is_valid_and_has_scaled_angle(
    angle: float, alpha: float
) -> None:
    source = _rotation(np.array([1.0, -2.0, 3.0]), angle)
    scaled = scale_rotation_shortest_path(source, alpha)
    validate_rotation_matrix(scaled)
    assert np.linalg.det(scaled) == pytest.approx(1.0)
    assert _angle(scaled) == pytest.approx(alpha * angle, abs=2e-8)
    encoded = canonical_rotation_6d(scaled)
    assert rotation_matrix_from_6d(encoded) == pytest.approx(scaled, abs=1e-7)


def test_so3_rejects_nonfinite_nonorthonormal_and_reflection_matrices() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_rotation_matrix(np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="orthonormal"):
        validate_rotation_matrix(np.diag([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="positive"):
        validate_rotation_matrix(np.diag([1.0, 1.0, -1.0]))


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

    def __init__(self, chunk: np.ndarray | None = None) -> None:
        self.chunk = _chunk() if chunk is None else chunk
        self.reset_calls = 0
        self.predict_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_canonical_action_chunk(self, _batch):
        self.predict_calls += 1
        return torch.from_numpy(self.chunk[None])


class FeasibilityEnv:
    instruction = "Push the object into the target zone."

    def __init__(
        self,
        *,
        instruction: str | None = None,
        clip: bool = False,
        task: str = "push_to_zone",
    ) -> None:
        if instruction is not None:
            self.instruction = instruction
        self.morphology = SimpleNamespace(id="panda", native_action_dim=10)
        self.task = SimpleNamespace(id=task)
        self.scenario = SimpleNamespace(target_tolerance=0.055)
        self.decoded: list[np.ndarray] = []
        self.decode_iterations: list[int] = []
        self.applied: list[np.ndarray] = []
        self.steps = 0
        self.clip = clip

    def render_top(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def canonical_state(self):
        value = np.zeros((1, 10), dtype=np.float32)
        value[0, 3:9] = canonical_rotation_6d(np.eye(3))
        return value

    def native_action_from_canonical(self, canonical, *, iterations):
        self.decode_iterations.append(iterations)
        native = np.asarray(canonical, dtype=np.float32).reshape(10).copy()
        self.decoded.append(native)
        return native

    def canonical_arm_action(self, native):
        realized = np.asarray(native, dtype=np.float32).reshape(1, 10).copy()
        if realized[0, 0] > 0.5:
            realized[0, 0] += 0.01
        return realized

    def native_to_qpos(self, native):
        return np.asarray(native)

    def qpos_to_native(self, native):
        value = np.asarray(native)
        return value + 0.1 if self.clip else value

    def apply_action(self, native):
        self.applied.append(native)

    def step_control_period(self, frequency):
        assert frequency == 30.0
        self.steps += 1
        return TaskStatus(success=True, terminal=True, stage="success")

    def diagnostic_state(self):
        return {
            "ee_object_distance_m": 0.01,
            "object_target_xy_distance_m": 0.01,
            "object_height_m": 0.02,
            "object_speed": 0.0,
            "gripper_opening_fraction": 1.0,
        }

    def close(self):
        pass


class InfeasibleEnv(FeasibilityEnv):
    def canonical_arm_action(self, native):
        realized = np.asarray(native, dtype=np.float32).reshape(1, 10).copy()
        realized[0, 0] += 0.01
        return realized


def _synthetic_runtime() -> dict:
    return {
        "device": "cpu",
        "python": "3.12.0",
        "torch": "test",
        "numpy": np.__version__,
        "mujoco": "test",
        "cuda_runtime": None,
        "cudnn": None,
        "cuda_device_name": None,
        "mujoco_gl": None,
        "python_hash_seed": None,
        "cublas_workspace_config": None,
        "torch_deterministic_algorithms": False,
        "torch_deterministic_warn_only": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "platform": "test-platform",
        "hostname": "test-host",
    }


def _synthetic_registered_reports(*, treatment_failure: bool) -> tuple[
    list[dict], ScenarioManifest, dict, dict, dict[str, str]
]:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    exp36 = json.loads((ROOT / "reports/benchmark-exp36-summary.json").read_text())
    exp38 = json.loads((ROOT / "reports/benchmark-exp38-summary.json").read_text())
    templates = {
        task: run_projection_pair(
            {arm: FeasibilityEnv(task=task) for arm in ("baseline", "projection")},
            FakePolicy(),
            FakeProcessor(),
            "cpu",
            order=("baseline", "projection"),
            maximum_steps=1,
        )
        for task in projection.TASKS
    }
    treatment_template = run_projection_pair(
        {
            "baseline": FeasibilityEnv(task="push_to_zone"),
            "projection": InfeasibleEnv(task="push_to_zone"),
        },
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("projection", "baseline"),
        maximum_steps=1,
    )
    runtime = _synthetic_runtime()
    runtime_signature = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evaluator_sha256 = file_sha256(Path(projection.__file__))
    reports = []
    source_hashes = {}
    for position, (condition, seed, morphology) in enumerate(projection.REGISTERED_SCHEDULE):
        scenarios = projection_scenarios(manifest, morphology, registered=True)
        outcomes = []
        for ordinal, scenario in enumerate(scenarios):
            use_failure = treatment_failure and position == 0 and ordinal == 0
            pair = copy.deepcopy(
                treatment_template if use_failure else templates[scenario.task]
            )
            pair["arm_order"] = list(pair_arm_order(ordinal, condition, seed, morphology))
            pair.update(
                scenario_id=scenario.scenario_id,
                scenario_seed=scenario.seed,
                morphology=morphology,
                task=scenario.task,
                active_factor_axes=[
                    name
                    for name, value in scenario.factors.items()
                    if value["bin"].startswith("development:")
                ],
            )
            outcomes.append(pair)
        source = morphology if condition == "matched" else "joint"
        identity = {
            "condition": condition,
            "source_condition": source,
            "model_seed": seed,
            "morphology": morphology,
            "core_sha256": exp36["runs"][f"{source}-seed{seed}"]["core_sha256"],
            "config_sha256": projection.CONFIG_SHA256[morphology],
            "data_manifest_sha256": projection.DATA_MANIFEST_SHA256[source],
            "exp36_summary_sha256": projection.EXP36_SUMMARY_SHA256,
            "step": 1200,
            "stage": "core",
        }
        provenance = {
            "git_dirty": False,
            "git_commit": "synthetic-commit",
            "evaluator_sha256": evaluator_sha256,
            "runtime": copy.deepcopy(runtime),
            "runtime_signature_sha256": runtime_signature,
        }
        minute = position * 2
        run = {
            "started_utc": f"2026-01-01T00:{minute:02d}:00Z",
            "finished_utc": f"2026-01-01T00:{minute + 1:02d}:00Z",
            "device": "cpu",
            "lock_identity": projection.LOCK_IDENTITY,
            "lock_path": str(projection.REGISTERED_RUNTIME_DIR / projection.REGISTERED_LOCK_NAME),
            "schedule_identity": projection.SCHEDULE_IDENTITY,
            "schedule_position": position,
        }
        contract = {
            "definition_sha256": projection.DEFINITION_SHA256,
            "manifest_sha256": projection.DEVELOPMENT_MANIFEST_SHA256,
            "exp36_summary_sha256": projection.EXP36_SUMMARY_SHA256,
            "exp38_summary_sha256": projection.EXP38_SUMMARY_SHA256,
            "exp38_evaluator_sha256": projection.EXP38_EVALUATOR_SHA256,
            "split": "development",
            "scenario_indices": list(projection.SCENARIO_INDICES),
            "tasks": list(projection.TASKS),
            "limit_scenarios": None,
            "protocol": projection.PROJECTION_PROTOCOL,
            "final_split_loaded": False,
        }
        report = _new_report(
            registered=True,
            contract=contract,
            identity=identity,
            expected_ids=[scenario.scenario_id for scenario in scenarios],
            provenance=provenance,
            run=run,
        )
        report["outcomes"] = outcomes
        report["outcome_chain_sha256"] = _outcome_chain(outcomes)
        report["treatment_delivery_errors"] = [
            {
                "scenario_id": outcome["scenario_id"],
                "arm": "projection",
                "reason": outcome["arms"]["projection"]["treatment_failure_reason"],
                "details": outcome["arms"]["projection"]["treatment_failure_details"],
            }
            for outcome in outcomes
            if outcome["arms"]["projection"]["treatment_failure"]
        ]
        report["aggregate"] = _shard_aggregate(outcomes)
        report["complete"] = True
        report["status"] = "completed"
        reports.append(report)
        label = f"{condition}/seed{seed}/{morphology}"
        source_hashes[label] = hashlib.sha256(label.encode()).hexdigest()
    return reports, manifest, exp36, exp38, source_hashes


def test_grid_selects_verified_candidate_and_episode_executes_exact_native_once() -> None:
    env = FeasibilityEnv()
    command = _chunk()[0]
    decoded = project_feasible_command(env, command)
    assert decoded.alpha == 0.5
    assert decoded.trials == 9
    assert decoded.errors["translation_m"] == 0.0
    assert env.decode_iterations == [20] * 9
    assert decoded.native is env.decoded[-1]

    episode_env = FeasibilityEnv()
    result = run_projection_episode(
        episode_env,
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        arm="projection",
        initial_chunk=_chunk(),
        maximum_steps=1,
    )
    assert result["alpha_histogram"]["8/16"] == 1
    assert result["decode_trials"] == len(episode_env.decoded) == 9
    assert episode_env.applied[0] is episode_env.decoded[-1]


def test_terminal_mid_chunk_prepares_only_four_and_baseline_decodes_each_once() -> None:
    env = FeasibilityEnv()
    result = run_projection_episode(
        env,
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        arm="baseline",
        initial_chunk=_chunk(),
        maximum_steps=5,
    )
    assert result["commands"] == 1
    assert result["decoded_commands"] == result["decode_trials"] == 4
    assert len(env.decoded) == 4


def test_projection_preserves_normalized_gripper_on_every_grid_candidate() -> None:
    command = _chunk(gripper=0.371)[0]
    for alpha in ALPHA_GRID:
        projected = scale_canonical_command(command, alpha)
        assert projected[0, 9] == command[0, 9]


def test_alpha_zero_infeasibility_is_delivery_failure_without_execution() -> None:
    direct_env = InfeasibleEnv()
    with pytest.raises(TreatmentDeliveryError, match="alpha zero") as captured:
        project_feasible_command(direct_env, _chunk()[0])
    assert captured.value.reason == "alpha_zero_infeasible"
    assert captured.value.trial_count == len(ALPHA_GRID)
    env = InfeasibleEnv()
    result = run_projection_episode(
        env,
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        arm="projection",
        initial_chunk=_chunk(),
        maximum_steps=1,
    )
    assert result["success"] is None
    assert result["treatment_failure"] is True
    assert result["treatment_failure_reason"] == "alpha_zero_infeasible"
    assert result["treatment_failure_details"]["trial_count"] == len(ALPHA_GRID)
    assert len(env.decoded) == len(ALPHA_GRID)
    assert env.applied == []

    pair = run_projection_pair(
        {"baseline": FeasibilityEnv(), "projection": InfeasibleEnv()},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("projection", "baseline"),
        maximum_steps=1,
    )
    assert pair["arms"]["projection"]["success"] is None
    assert pair["arms"]["baseline"]["success"] is True


def test_treatment_failure_preserves_prior_telemetry_and_discards_prepared_chunk() -> None:
    class LaterInfeasibleEnv(FeasibilityEnv):
        def canonical_arm_action(self, native):
            realized = super().canonical_arm_action(native)
            if len(self.decoded) > 54:
                realized[0, 0] += 0.01
            return realized

        def step_control_period(self, frequency):
            assert frequency == 30.0
            self.steps += 1
            return TaskStatus(success=False, terminal=False, stage="push")

    env = LaterInfeasibleEnv()
    result = run_projection_episode(
        env,
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        arm="projection",
        initial_chunk=_chunk(),
        maximum_steps=8,
    )
    assert result["treatment_failure"] is True
    assert result["commands"] == result["steps"] == len(env.applied) == 4
    assert result["decoded_commands"] == 6
    assert result["treatment_failure_details"]["prepared_commands_discarded"] == 2
    assert sum(
        value["count"] for value in result["geometry_by_lead"]["translation_m"].values()
    ) == 4


def test_pair_counterbalance_and_initial_invariants() -> None:
    assert pair_arm_order(0, "joint", 36001, "so101") == ("baseline", "projection")
    assert pair_arm_order(1, "joint", 36001, "so101") == ("projection", "baseline")
    assert pair_arm_order(0, "matched", 36001, "so101") == ("projection", "baseline")
    assert pair_arm_order(0, "joint", 36002, "so101") == ("projection", "baseline")
    assert pair_arm_order(0, "joint", 36001, "panda") == ("projection", "baseline")
    environments = {arm: FeasibilityEnv() for arm in ("baseline", "projection")}
    pair = run_projection_pair(
        environments,
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("projection", "baseline"),
        maximum_steps=1,
    )
    assert pair["initial_input_hash_agreement"] is True
    assert pair["initial_inputs"]["baseline"] == pair["initial_inputs"]["projection"]
    assert pair["first_chunk_max_abs_difference"] <= FIRST_CHUNK_MAX_ABS
    assert pair["arm_order"] == ["projection", "baseline"]

    mismatched = {
        "baseline": FeasibilityEnv(),
        "projection": FeasibilityEnv(instruction="different"),
    }
    with pytest.raises(TreatmentDeliveryError, match="initial policy input"):
        run_projection_pair(
            mismatched,
            FakePolicy(),
            FakeProcessor(),
            "cpu",
            order=("baseline", "projection"),
            maximum_steps=1,
        )
    assert not mismatched["baseline"].decoded
    assert not mismatched["projection"].decoded


def test_milestones_advance_only_through_currently_true_ordered_prefix() -> None:
    env = FeasibilityEnv()
    milestones = {
        name: {"attained": False, "first_step": None}
        for name in ("near_ee_object", "object_in_target_tolerance", "settled")
    }
    diagnostics = env.diagnostic_state()
    diagnostics["ee_object_distance_m"] = 0.1
    update_ordered_milestones(milestones, env, diagnostics, 1)
    assert not any(value["attained"] for value in milestones.values())
    diagnostics["ee_object_distance_m"] = 0.01
    diagnostics["object_target_xy_distance_m"] = 0.2
    update_ordered_milestones(milestones, env, diagnostics, 2)
    assert milestones["near_ee_object"] == {"attained": True, "first_step": 2}
    assert milestones["object_in_target_tolerance"]["attained"] is False
    diagnostics["object_target_xy_distance_m"] = 0.01
    update_ordered_milestones(milestones, env, diagnostics, 3)
    assert milestones["object_in_target_tolerance"]["first_step"] == 3
    assert milestones["settled"]["first_step"] == 3


def test_resume_rejects_contract_and_hash_chain_corruption(tmp_path: Path) -> None:
    provenance = {"git_dirty": True, "git_commit": "smoke", "evaluator_sha256": "hash"}
    expected = _new_report(
        registered=False,
        contract={"contract": "frozen"},
        identity={"condition": "joint", "model_seed": 36001, "morphology": "so101"},
        expected_ids=[],
        provenance=provenance,
    )
    output = tmp_path / "projection.json"
    output.write_text(json.dumps(expected))
    changed = _new_report(
        registered=False,
        contract={"contract": "changed"},
        identity=expected["identity"],
        expected_ids=[],
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="contract, checkpoint, or evaluator"):
        _load_or_create_report(output, changed)
    expected["outcome_chain_sha256"] = "0" * 64
    output.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match="hash chain"):
        _load_or_create_report(output, expected)


def test_validation_reconciles_clipping_success_and_chunk_mutations() -> None:
    pair = run_projection_pair(
        {arm: FeasibilityEnv() for arm in ("baseline", "projection")},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "projection"),
        maximum_steps=1,
    )
    pair.update(
        arm_order=list(pair_arm_order(0, "joint", 36001, "panda")),
        scenario_id="development-panda-push_to_zone-0007",
        scenario_seed=1,
        morphology="panda",
        task="push_to_zone",
        active_factor_axes=["workspace"],
    )
    report = {
        "identity": {"condition": "joint", "model_seed": 36001, "morphology": "panda"},
        "outcomes": [pair],
    }
    _validate_outcomes(report)
    changed = copy.deepcopy(report)
    changed["outcomes"][0]["arms"]["baseline"]["maximum_clipping_error_native"] = 1.0
    with pytest.raises(ValueError, match="clipping counters"):
        _validate_outcomes(changed)
    changed = copy.deepcopy(report)
    changed["outcomes"][0]["arms"]["baseline"]["success"] = None
    with pytest.raises(ValueError, match="counters or identity"):
        _validate_outcomes(changed)
    changed = copy.deepcopy(report)
    changed["outcomes"][0]["arms"]["baseline"]["chunks"] = 2
    with pytest.raises(ValueError, match="chunk count"):
        _validate_outcomes(changed)


def test_registered_runtime_output_defaults_and_lock_are_operational(tmp_path: Path) -> None:
    output = _registered_output("joint", 36001, "so101", None)
    assert output.parent == projection.REGISTERED_RUNTIME_DIR
    assert output.name == "benchmark-exp39-joint-seed36001-so101-projection.json"
    aggregate_args = parse_args(["aggregate"])
    assert aggregate_args.shard_dir == projection.REGISTERED_RUNTIME_DIR
    run_args = parse_args(["run-registered"])
    assert run_args.runtime_dir == projection.REGISTERED_RUNTIME_DIR
    assert "reports/exp39-runtime/" in (ROOT / ".gitignore").read_text()
    lock_path = tmp_path / "runtime" / "registered.lock"
    with projection.registered_run_lock(lock_path):
        with pytest.raises(RuntimeError, match="holds"):
            with projection.registered_run_lock(lock_path):
                pass


def test_shard_continues_after_structured_treatment_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "development.json"
    manifest_path.write_text("{}")
    scenarios = tuple(
        Scenario(
            scenario_id=f"development-panda-push_to_zone-{index:04d}",
            split="development",
            morphology="panda",
            task="push_to_zone",
            seed=index,
            factors={},
        )
        for index in (7, 8)
    )
    manifest = SimpleNamespace(path=manifest_path, scenarios=scenarios)
    definition = SimpleNamespace(sha256="smoke-definition")
    identity = {
        "condition": "joint",
        "model_seed": 36001,
        "morphology": "panda",
    }
    monkeypatch.setattr(projection, "runtime_scenario", lambda factors: factors)
    monkeypatch.setattr(
        projection,
        "resolve_checkpoint",
        lambda *_args, **_kwargs: (tmp_path / "checkpoint", dict(identity)),
    )
    calls = 0

    def env_factory(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return InfeasibleEnv() if calls == 2 else FeasibilityEnv()

    output = tmp_path / "smoke-projection.json"
    report = projection.evaluate_projection_shard(
        definition,
        manifest,
        ROOT / "reports/benchmark-exp36-summary.json",
        ROOT / "reports/benchmark-exp38-summary.json",
        output,
        condition="joint",
        seed=36001,
        morphology="panda",
        device="cpu",
        registered=False,
        env_factory=env_factory,
        policy_loader=lambda *_args: FakePolicy(),
        processor_factory=lambda *_args, **_kwargs: FakeProcessor(),
    )
    assert report["complete"] is True
    assert len(report["outcomes"]) == 2
    assert len(report["treatment_delivery_errors"]) == 1
    assert report["infrastructure_errors"] == []
    assert report["aggregate"]["eligible_pairs"] == 1
    assert report["aggregate"]["delivery_gate"]["treatment_failures"] == 1


def test_frozen_exp38_summary_and_evaluator_hash_validate() -> None:
    summary = validate_exp38_summary(ROOT / "reports/benchmark-exp38-summary.json")
    assert summary["contract"]["evaluator_sha256"] == EXP38_EVALUATOR_SHA256


def test_causal_classification_requires_exact_criteria_and_delivery() -> None:
    success = {36001: 0.10, 36002: 0.05, 36003: 0.0}
    progression = {seed: 0.01 for seed in SEEDS}
    supported = classify_causal_support(success, progression, delivery_gate_passed=True)
    assert supported["supported"] is True
    assert supported["three_seed_mean_paired_success_gain_percentage_points"] == pytest.approx(5.0)
    failed_delivery = classify_causal_support(success, progression, delivery_gate_passed=False)
    assert failed_delivery["supported"] is False
    assert failed_delivery["checks"]["geometry_delivery_gate_passed"] is False
    missing = classify_causal_support(
        {seed: None for seed in SEEDS},
        {seed: None for seed in SEEDS},
        delivery_gate_passed=False,
    )
    assert missing["supported"] is False
    assert missing["three_seed_mean_paired_success_gain_percentage_points"] is None


def test_shard_delivery_gate_fails_clipping_and_registered_aggregate_needs_all_shards() -> None:
    environments = {arm: FeasibilityEnv() for arm in ("baseline", "projection")}
    pair = run_projection_pair(
        environments,
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "projection"),
        maximum_steps=1,
    )
    pair["scenario_id"] = "development-panda-push_to_zone-0007"
    pair["task"] = "push_to_zone"
    pair["morphology"] = "panda"
    pair["arms"]["baseline"]["clipped_commands"] = 1
    pair["arms"]["baseline"]["command_clipping_rate"] = 1.0
    aggregate = _shard_aggregate([pair])
    assert aggregate["delivery_gate"]["arm_clipping_at_most_one_percent"]["baseline"] is False
    assert aggregate["delivery_gate"]["passed"] is False
    with pytest.raises(ValueError, match="all 12 shards"):
        aggregate_registered_shards(
            [],
            SimpleNamespace(scenarios=()),
            exp36_summary={"experiment": 36, "status": "completed_gate_passed", "fixed_step": 1200},
            exp38_summary=json.loads(
                (ROOT / "reports/benchmark-exp38-summary.json").read_text()
            ),
            source_hashes={
                f"{condition}/seed{seed}/{morphology}": "a" * 64
                for condition, seed, morphology in projection.REGISTERED_SCHEDULE
            },
        )


def test_registered_filename_is_exact() -> None:
    assert (
        registered_shard_filename("matched", 36003, "panda")
        == "benchmark-exp39-matched-seed36003-panda-projection.json"
    )
    assert _outcome_chain([]) == _outcome_chain([])


def test_full_registered_aggregate_handles_treatment_failure_and_source_integrity(
    tmp_path: Path,
) -> None:
    reports, manifest, exp36, exp38, source_hashes = _synthetic_registered_reports(
        treatment_failure=True
    )
    summary = aggregate_registered_shards(
        reports,
        manifest,
        exp36_summary=exp36,
        exp38_summary=exp38,
        source_hashes=source_hashes,
    )
    assert summary["contract"]["pairs"] == 252
    assert summary["source_shard_sha256"] == dict(sorted(source_hashes.items()))
    assert summary["equal_weight_seed_gains"]["joint/seed36001"]["eligible_pairs"] == 41
    assert summary["delivery_gates_by_shard"]["joint/seed36001/so101"][
        "treatment_failures"
    ] == 1
    assert summary["causal_support_by_condition"]["joint"]["supported"] is False
    assert summary["equal_weight_seed_gains"]["matched/seed36001"][
        "paired_success_gain_percentage_points"
    ] == 0.0

    missing_hash = dict(source_hashes)
    missing_hash.pop(next(iter(missing_hash)))
    with pytest.raises(ValueError, match="hashes for all 12"):
        aggregate_registered_shards(
            reports,
            manifest,
            exp36_summary=exp36,
            exp38_summary=exp38,
            source_hashes=missing_hash,
        )

    runtime_changed = copy.deepcopy(reports)
    runtime_changed[-1]["provenance"]["runtime"]["torch_num_threads"] = 2
    runtime_changed[-1]["provenance"]["runtime_signature_sha256"] = hashlib.sha256(
        json.dumps(
            runtime_changed[-1]["provenance"]["runtime"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="runtime signature"):
        aggregate_registered_shards(
            runtime_changed,
            manifest,
            exp36_summary=exp36,
            exp38_summary=exp38,
            source_hashes=source_hashes,
        )

    expected = copy.deepcopy(reports[0])
    report_path = tmp_path / "completed.json"
    report_path.write_text(json.dumps(reports[0]))
    assert _load_or_create_report(report_path, expected)["complete"] is True
    changed = copy.deepcopy(reports[0])
    changed["status"] = "completed_delivery_failed"
    report_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="status or aggregate"):
        _load_or_create_report(report_path, expected)
