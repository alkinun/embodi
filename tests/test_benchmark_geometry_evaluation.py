import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodi import EmbodiConfig
from embodi.sim.benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from embodi.sim.benchmark_geometry_evaluation import (
    CONDITIONS,
    CONFIG_SHA256,
    DATA_MANIFEST_SHA256,
    EXP36_SUMMARY_SHA256,
    MORPHOLOGIES,
    SEEDS,
    TASKS,
    _clipping,
    _append_outcome,
    _load_or_create_report,
    _new_report,
    _outcome_chain,
    _validate_outcomes,
    aggregate_registered_shards,
    evaluate_geometry_admission,
    evaluate_policy_shard,
    geometry_policy_batch,
    load_development_contract,
    predict_geometry_chunk,
    resolve_checkpoint,
    run_geometry_policy_episode,
    registered_shard_filename,
    summarize_geometry_admission,
)
from embodi.sim.tasks import TaskStatus


ROOT = Path(__file__).parents[1]


def test_development_contract_rejects_nonregistered_hash(tmp_path: Path) -> None:
    definition = ROOT / "benchmarks" / "sim-v1" / "definition.json"
    source = ROOT / "benchmarks" / "sim-v1" / "manifests" / "development.json"
    _definition, manifest = load_development_contract(definition, source, registered=True)
    assert len(manifest.scenarios) == 400
    changed = tmp_path / "development.json"
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest hash"):
        load_development_contract(definition, changed, registered=False)


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, images, instructions, state, **kwargs):
        self.calls.append((images, instructions, state, kwargs))
        return {"pixels": images[0][0], **kwargs}


class FakePolicy:
    config = SimpleNamespace(
        image_do_rescale=False,
        backbone_name="fake",
        backbone_revision="fake-revision",
    )

    def __init__(self) -> None:
        self.predict_calls = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_canonical_action_chunk(self, batch):
        self.predict_calls += 1
        assert "canonical_state" in batch
        assert "canonical_part_mask" in batch
        return torch.zeros((1, 32, 1, 10), dtype=torch.float32)

    def predict_action_chunk(self, _batch):
        raise AssertionError("geometry evaluation must not call the native predictor")


class BatchEnv:
    instruction = "test instruction"

    def render_top(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def canonical_state(self):
        return np.zeros((1, 10), dtype=np.float32)


def test_geometry_batch_uses_state_none_part_mask_and_canonical_predictor() -> None:
    env = BatchEnv()
    policy = FakePolicy()
    processor = FakeProcessor()
    batch = geometry_policy_batch(env, policy, processor, "cpu")
    chunk = predict_geometry_chunk(env, policy, processor, "cpu")
    assert batch["canonical_state"].shape == (1, 1, 10)
    assert batch["canonical_part_mask"].tolist() == [[True]]
    assert all(call[2] is None for call in processor.calls)
    assert policy.predict_calls == 1
    assert chunk.shape == (32, 1, 10)


class EpisodeEnv(BatchEnv):
    lifted = False
    morphology = SimpleNamespace(native_action_dim=10)

    def __init__(self, terminal_step: int) -> None:
        self.terminal_step = terminal_step
        self.steps = 0
        self.status_calls = 0
        self.decode_anchor_steps = []

    def native_action_from_canonical(self, canonical):
        self.decode_anchor_steps.append(self.steps)
        return np.asarray(canonical[0], dtype=np.float32)

    def native_to_qpos(self, native):
        return np.asarray(native)

    def qpos_to_native(self, qpos):
        return np.asarray(qpos)

    def apply_action(self, _native):
        pass

    def step_control_period(self, frequency):
        assert frequency == 30.0
        self.steps += 1
        self.status_calls += 1
        terminal = self.steps == self.terminal_step
        return TaskStatus(success=terminal, terminal=terminal, stage="success" if terminal else "run")

    def task_status(self):
        raise AssertionError("the returned step status must not be recomputed")

    def diagnostic_state(self):
        return {"height": float(self.steps)}


def test_policy_episode_obeys_horizon_and_returned_status() -> None:
    env = EpisodeEnv(terminal_step=17)
    policy = FakePolicy()
    result = run_geometry_policy_episode(
        env,
        policy,
        FakeProcessor(),
        "cpu",
        maximum_steps=100,
        execution_horizon=16,
    )
    assert result["success"] is True
    assert result["steps"] == env.status_calls == 17
    assert result["chunks"] == policy.predict_calls == 2
    assert result["final_stage"] == "success"
    assert policy.reset_calls == 1
    assert env.decode_anchor_steps == [0] * 16 + [16] * 16


def test_policy_nonfinite_chunk_is_valid_zero_step_failure() -> None:
    class NonfinitePolicy(FakePolicy):
        def predict_canonical_action_chunk(self, batch):
            return torch.full((1, 32, 1, 10), float("nan"))

    result = run_geometry_policy_episode(
        EpisodeEnv(terminal_step=10),
        NonfinitePolicy(),
        FakeProcessor(),
        "cpu",
    )
    assert result["steps"] == result["commands"] == 0
    assert result["failure_reason"] == "policy_nonfinite_canonical_action"
    _validate_outcomes({"mode": "policy_shard", "outcomes": [{"scenario_id": "x", **result}]})


def test_resume_rejects_contract_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    first = _new_report(
        mode="policy_shard",
        registered=False,
        contract={"manifest": "first"},
        identity={"checkpoint": "same"},
        expected_ids=["scenario"],
    )
    output.write_text(json.dumps(first))
    changed = _new_report(
        mode="policy_shard",
        registered=False,
        contract={"manifest": "changed"},
        identity={"checkpoint": "same"},
        expected_ids=["scenario"],
    )
    with pytest.raises(ValueError, match="contract or checkpoint"):
        _load_or_create_report(output, changed)


def test_resume_rejects_edited_outcome_chain(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report = _new_report(
        mode="policy_shard",
        registered=False,
        contract={"manifest": "same"},
        identity={"checkpoint": "same"},
        expected_ids=["scenario"],
    )
    _append_outcome(report, {"scenario_id": "scenario", "success": False})
    report["outcomes"][0]["success"] = True
    output.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="hash chain"):
        _load_or_create_report(output, report)


def test_geometry_admission_enforces_physical_unit_gates() -> None:
    definition = SimpleNamespace(
        values={
            "admission_gates": {
                "maximum_geometry_success_drop": 0.05,
                "maximum_command_clipping_rate": 0.01,
                "maximum_fk_ik_translation_p95_m": 0.001,
                "maximum_fk_ik_rotation_p95_deg": 1.0,
                "maximum_gripper_round_trip_error_native": 1.0,
            }
        }
    )
    frozen = {"cells": {"so101/lift_object": {"success_rate": 1.0}}}
    outcome = {
        "morphology": "so101",
        "task": "lift_object",
        "success": True,
        "translation_errors_m": [0.0005],
        "rotation_errors_deg": [0.5],
        "gripper_errors_native": [0.5],
        "commands": 100,
        "clipped_commands": 1,
    }
    aggregate = summarize_geometry_admission([outcome], definition, frozen)
    assert aggregate["passed"] is True
    outcome["translation_errors_m"] = [0.002]
    aggregate = summarize_geometry_admission([outcome], definition, frozen)
    assert aggregate["checks"]["fk_ik_translation_p95"] is False
    assert aggregate["passed"] is False


def test_admission_orchestrator_persists_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embodi.sim.benchmark_geometry_evaluation as evaluation

    manifest_path = tmp_path / "development.json"
    manifest_path.write_text("{}")
    scenario = Scenario(
        scenario_id="development-so101-lift_object-0000",
        split="development",
        morphology="so101",
        task="lift_object",
        seed=1,
        factors={},
    )
    manifest = SimpleNamespace(path=manifest_path, scenarios=(scenario,))
    definition = SimpleNamespace(
        sha256="definition",
        values={
            "admission_gates": {
                "maximum_geometry_success_drop": 0.05,
                "maximum_command_clipping_rate": 0.01,
                "maximum_fk_ik_translation_p95_m": 0.001,
                "maximum_fk_ik_rotation_p95_deg": 1.0,
                "maximum_gripper_round_trip_error_native": 1.0,
            }
        },
    )
    frozen = {"cells": {"so101/lift_object": {"success_rate": 1.0}}}
    outcome = {
        "success": True,
        "steps": 1,
        "final_stage": "success",
        "lifted": True,
        "translation_errors_m": [0.0],
        "rotation_errors_deg": [0.0],
        "gripper_errors_native": [0.0],
        "clipped_commands": 0,
        "commands": 1,
        "maximum_clipping_error_native": 0.0,
        "diagnostic_extrema": {},
    }
    closed = []
    monkeypatch.setattr(
        evaluation,
        "runtime_provenance",
        lambda **_kwargs: {"git_dirty": True, "git_commit": "smoke"},
    )
    monkeypatch.setattr(evaluation, "runtime_scenario", lambda factors: factors)
    monkeypatch.setattr(
        evaluation,
        "run_geometry_admission_episode",
        lambda env, **_kwargs: dict(outcome),
    )
    output = tmp_path / "admission.json"
    report = evaluate_geometry_admission(
        definition,
        manifest,
        frozen,
        output,
        registered=False,
        env_factory=lambda *_args, **_kwargs: SimpleNamespace(close=lambda: closed.append(True)),
    )
    assert report["complete"] is True
    assert report["aggregate"]["passed"] is True
    assert report["outcome_chain_sha256"] == _outcome_chain(report["outcomes"])
    assert closed == [True]
    resumed = evaluate_geometry_admission(
        definition,
        manifest,
        frozen,
        output,
        registered=False,
        env_factory=lambda *_args, **_kwargs: pytest.fail("complete report must not rerun"),
    )
    assert resumed == report


def test_policy_shard_orchestrator_persists_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embodi.sim.benchmark_geometry_evaluation as evaluation

    manifest_path = tmp_path / "development.json"
    manifest_path.write_text("{}")
    scenario = Scenario(
        scenario_id="development-so101-push_to_zone-0000",
        split="development",
        morphology="so101",
        task="push_to_zone",
        seed=1,
        factors={},
    )
    manifest = SimpleNamespace(path=manifest_path, scenarios=(scenario,))
    definition = SimpleNamespace(sha256="definition")
    admission_path = tmp_path / "admission.json"
    admission_path.write_text("{}")
    admission = {
        "aggregate": {"passed": True},
        "provenance": {"git_commit": "smoke"},
    }
    identity = {"condition": "joint", "model_seed": 36001, "morphology": "so101"}
    monkeypatch.setattr(
        evaluation,
        "runtime_provenance",
        lambda **_kwargs: {"git_dirty": True, "git_commit": "smoke"},
    )
    monkeypatch.setattr(evaluation, "runtime_scenario", lambda factors: factors)
    monkeypatch.setattr(evaluation, "validate_passing_admission", lambda *_args, **_kwargs: admission)
    monkeypatch.setattr(
        evaluation,
        "resolve_checkpoint",
        lambda *_args, **_kwargs: (tmp_path / "checkpoint", identity),
    )
    closed = []

    def env_factory(*_args, **_kwargs):
        env = EpisodeEnv(terminal_step=1)
        env.close = lambda: closed.append(True)
        return env

    output = tmp_path / "shard.json"
    report = evaluate_policy_shard(
        definition,
        manifest,
        admission_path,
        tmp_path / "summary.json",
        output,
        condition="joint",
        seed=36001,
        morphology="so101",
        device="cpu",
        registered=False,
        env_factory=env_factory,
        policy_loader=lambda *_args: FakePolicy(),
        processor_factory=lambda *_args, **_kwargs: FakeProcessor(),
    )
    assert report["complete"] is True
    assert report["aggregate"]["four_task_macro_success"] == pytest.approx(0.25)
    assert closed == [True]
    resumed = evaluate_policy_shard(
        definition,
        manifest,
        admission_path,
        tmp_path / "summary.json",
        output,
        condition="joint",
        seed=36001,
        morphology="so101",
        device="cpu",
        registered=False,
        env_factory=lambda *_args, **_kwargs: pytest.fail("complete report must not rerun"),
        policy_loader=lambda *_args: pytest.fail("complete report must not reload"),
    )
    assert resumed == report


def test_clipping_ignores_float32_round_trip_floor() -> None:
    class RoundTripEnv:
        @staticmethod
        def native_to_qpos(native):
            return native

        @staticmethod
        def qpos_to_native(native):
            return native + np.array([1.5e-5, 3e-5], dtype=np.float32)

    clipped, maximum = _clipping(RoundTripEnv(), np.zeros(2, dtype=np.float32))
    assert clipped is True
    assert maximum == pytest.approx(3e-5)

    RoundTripEnv.qpos_to_native = staticmethod(
        lambda native: native + np.array([1.5e-5, 1.5e-5], dtype=np.float32)
    )
    assert _clipping(RoundTripEnv(), np.zeros(2, dtype=np.float32))[0] is False


def test_checkpoint_resolver_validates_registered_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embodi.sim.benchmark_geometry_evaluation as evaluation

    run_root = tmp_path / "outputs" / "run"
    checkpoint = run_root / "final" / "so101"
    checkpoint.mkdir(parents=True)
    EmbodiConfig.load(ROOT / "configs" / "so101-exp0-top.json").save(
        checkpoint / "config.json"
    )
    checkpoint.joinpath("core.pt").write_bytes(b"core")
    checkpoint.joinpath("embodiment.pt").write_bytes(b"embodiment")
    checkpoint.joinpath("trainer.pt").write_bytes(b"trainer")
    run_root.joinpath("final", "data_manifest.json").write_text(
        json.dumps({"experiment": 36, "condition": "so101"})
    )
    core_hash = evaluation.file_sha256(checkpoint / "core.pt")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "experiment": 36,
                "status": "completed_gate_passed",
                "fixed_step": 1200,
                "runs": {
                    "so101-seed36001": {
                        "model_seed": 36001,
                        "output": "outputs/run",
                        "core_sha256": core_hash,
                    }
                },
            }
        )
    )
    monkeypatch.setattr(evaluation, "EXP36_SUMMARY_SHA256", evaluation.file_sha256(summary))
    monkeypatch.setitem(evaluation.CONFIG_SHA256, "so101", evaluation.file_sha256(checkpoint / "config.json"))
    monkeypatch.setitem(
        evaluation.DATA_MANIFEST_SHA256,
        "so101",
        evaluation.file_sha256(run_root / "final" / "data_manifest.json"),
    )
    monkeypatch.setattr(
        evaluation.torch,
        "load",
        lambda *_args, **_kwargs: {"step": 1200, "stage": "core"},
    )
    resolved, identity = resolve_checkpoint(
        summary,
        "so101",
        36001,
        "so101",
        root=tmp_path,
    )
    assert resolved == checkpoint
    assert identity["core_sha256"] == core_hash
    assert identity["step"] == 1200


def fake_manifest() -> SimpleNamespace:
    scenarios = []
    for morphology in MORPHOLOGIES:
        for task in TASKS:
            for index in range(50):
                scenarios.append(
                    Scenario(
                        scenario_id=f"development-{morphology}-{task}-{index:04d}",
                        split="development",
                        morphology=morphology,
                        task=task,
                        seed=index,
                        factors={},
                    )
                )
    return SimpleNamespace(scenarios=tuple(scenarios))


def fake_shards(manifest: SimpleNamespace, admission_sha256: str) -> list[dict]:
    reports = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            for morphology in MORPHOLOGIES:
                scenarios = [value for value in manifest.scenarios if value.morphology == morphology]
                if condition == "joint":
                    successes_per_cell = 38
                elif condition == morphology:
                    successes_per_cell = 40
                else:
                    successes_per_cell = 10
                task_counts = {task: 0 for task in TASKS}
                outcomes = []
                for scenario in scenarios:
                    success = task_counts[scenario.task] < successes_per_cell
                    task_counts[scenario.task] += 1
                    outcomes.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "task": scenario.task,
                            "success": success,
                            "lifted": success,
                            "steps": 10,
                            "failure_reason": None if success else "maximum_episode_steps",
                            "final_stage": "success" if success else "approach",
                            "commands": 10,
                            "clipped_commands": 0,
                            "command_clipping_rate": 0.0,
                            "maximum_clipping_error_native": 0.0,
                        }
                    )
                reports.append(
                    {
                        "format_version": 1,
                        "experiment": 37,
                        "mode": "policy_shard",
                        "registered": True,
                        "status": "completed",
                        "complete": True,
                        "contract": {
                            "definition_sha256": (
                                "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
                            ),
                            "manifest_sha256": (
                                "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
                            ),
                            "admission_report_sha256": admission_sha256,
                            "admission_evaluator_git_commit": "test-commit",
                            "limit_per_cell": None,
                        },
                        "identity": {
                            "condition": condition,
                            "model_seed": seed,
                            "morphology": morphology,
                            "core_sha256": f"{condition}-{seed}",
                            "config_sha256": CONFIG_SHA256[morphology],
                            "data_manifest_sha256": DATA_MANIFEST_SHA256[condition],
                            "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
                            "step": 1200,
                            "stage": "core",
                        },
                        "provenance": {
                            "git_dirty": False,
                            "git_commit": "test-commit",
                            "evaluator_sha256": evaluation_file_sha256(),
                        },
                        "expected_outcome_ids": [value.scenario_id for value in scenarios],
                        "outcomes": outcomes,
                        "outcome_chain_sha256": _outcome_chain(outcomes),
                        "aggregate": evaluation_aggregate(outcomes),
                    }
                )
    return reports


def evaluation_aggregate(outcomes: list[dict]) -> dict:
    from embodi.sim.benchmark_geometry_evaluation import _shard_aggregate

    return _shard_aggregate(outcomes)


def evaluation_file_sha256() -> str:
    import embodi.sim.benchmark_geometry_evaluation as evaluation

    return evaluation.file_sha256(Path(evaluation.__file__))


def fake_exp36_summary() -> dict:
    return {
        "experiment": 36,
        "status": "completed_gate_passed",
        "fixed_step": 1200,
        "runs": {
            f"{condition}-seed{seed}": {"core_sha256": f"{condition}-{seed}"}
            for condition in CONDITIONS
            for seed in SEEDS
        },
    }


def test_aggregation_recomputes_noninferiority_gate_and_requires_all_shards() -> None:
    manifest = fake_manifest()
    reports = fake_shards(manifest, "admission")
    summary = aggregate_registered_shards(
        reports,
        manifest,
        admission_sha256="admission",
        admission_git_commit="test-commit",
        exp36_summary=fake_exp36_summary(),
    )
    overall = summary["three_seed_comparisons"]["overall"]
    assert overall["matched_specialist_mean"] == pytest.approx(0.8)
    assert overall["joint_mean"] == pytest.approx(0.76)
    assert overall["opposite_specialist_mean"] == pytest.approx(0.2)
    assert overall["joint_minus_matched_percentage_points"] == pytest.approx(-4.0)
    assert summary["gates"]["overall_seed_passes"] == 3
    assert summary["gates"]["passed"] is True
    assert summary["opposite_specialists_are_descriptive_only"] is True
    with pytest.raises(ValueError, match="all 18 shards"):
        aggregate_registered_shards(
            reports[:-1],
            manifest,
            admission_sha256="admission",
            admission_git_commit="test-commit",
            exp36_summary=fake_exp36_summary(),
        )
    reports[0]["identity"].pop("core_sha256")
    with pytest.raises(ValueError, match="wrong contract"):
        aggregate_registered_shards(
            reports,
            manifest,
            admission_sha256="admission",
            admission_git_commit="test-commit",
            exp36_summary=fake_exp36_summary(),
        )


def test_aggregation_rejects_all_failure_noninferiority() -> None:
    manifest = fake_manifest()
    reports = fake_shards(manifest, "admission")
    for report in reports:
        for outcome in report["outcomes"]:
            outcome["success"] = False
            outcome["lifted"] = False
            outcome["failure_reason"] = "maximum_episode_steps"
            outcome["final_stage"] = "approach"
        report["outcome_chain_sha256"] = _outcome_chain(report["outcomes"])
        report["aggregate"] = evaluation_aggregate(report["outcomes"])
    summary = aggregate_registered_shards(
        reports,
        manifest,
        admission_sha256="admission",
        admission_git_commit="test-commit",
        exp36_summary=fake_exp36_summary(),
    )
    assert summary["gates"]["overall_eight_cell_three_seed_mean"] is True
    assert summary["gates"]["matched_overall_assay_sensitive"] is False
    assert summary["gates"]["joint_overall_absolute"] is False
    assert summary["gates"]["passed"] is False


def test_registered_exp37_reports_recompute_failed_gate() -> None:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks" / "sim-v1" / "definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks" / "sim-v1" / "manifests" / "development.json",
        definition,
        require_complete_counts=True,
    )
    admission_path = ROOT / "reports" / "benchmark-exp37-geometry-admission.json"
    reports = [
        json.loads(
            (
                ROOT
                / "reports"
                / registered_shard_filename(condition, seed, morphology)
            ).read_text()
        )
        for condition in CONDITIONS
        for seed in SEEDS
        for morphology in MORPHOLOGIES
    ]
    exp36_summary = json.loads((ROOT / "reports" / "benchmark-exp36-summary.json").read_text())
    recomputed = aggregate_registered_shards(
        reports,
        manifest,
        admission_sha256=file_sha256(admission_path),
        admission_git_commit="bdd34fc51822c3353485d1d2e2547bfc5cca2c5e",
        exp36_summary=exp36_summary,
    )
    recorded = json.loads((ROOT / "reports" / "benchmark-exp37-summary.json").read_text())
    assert recomputed == recorded
    assert recorded["status"] == "completed_gate_failed"
    assert recorded["three_seed_comparisons"]["overall"]["joint_mean"] == 0.06
    assert recorded["three_seed_comparisons"]["overall"]["matched_specialist_mean"] == pytest.approx(
        0.13083333333333333
    )
    assert recorded["gates"]["overall_seed_passes"] == 1
    assert recorded["gates"]["passed"] is False
