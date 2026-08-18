from contextlib import contextmanager
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodi.sim.benchmark import BenchmarkDefinition, ScenarioManifest
import embodi.sim.benchmark_gripper_mechanism as mechanism


ROOT = Path(__file__).parents[1]
IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _sources() -> dict[str, Path]:
    return {
        "exp36_summary": ROOT / "reports/benchmark-exp36-summary.json",
        "exp42_summary": ROOT / "reports/benchmark-exp42-summary.json",
        "exp43_training_summary": ROOT / "reports/benchmark-exp43-training-summary.json",
        "exp42_so101_shard": ROOT / "reports/benchmark-exp42-so101-phase-action-quality.json",
        "exp42_panda_shard": ROOT / "reports/benchmark-exp42-panda-phase-action-quality.json",
    }


def _manifest() -> ScenarioManifest:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    return ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )


class FakeStatus:
    def __init__(self, done: bool) -> None:
        self.terminal = done
        self.success = done


class FakeEnv:
    def __init__(self, task: str, morphology: str) -> None:
        self.task = task
        self.morphology = SimpleNamespace(
            id=morphology, native_action_dim=6 if morphology == "so101" else 8
        )
        self.success = False
        self.applied: list[np.ndarray] = []
        self.native = np.r_[np.zeros(self.morphology.native_action_dim - 1), 0.5].astype(
            np.float32
        )
        self.canonical = np.r_[np.zeros(3), IDENTITY_6D, 0.5].reshape(1, 10).astype(
            np.float32
        )

    def render_top(self) -> np.ndarray:
        return np.full((480, 640, 3), len(self.applied), dtype=np.uint8)

    def canonical_state(self) -> np.ndarray:
        return self.canonical.copy()

    def state(self) -> np.ndarray:
        return self.native.copy()

    def canonical_arm_action(self, native: np.ndarray) -> np.ndarray:
        value = np.asarray(native, dtype=np.float32)
        return np.r_[value[:3], IDENTITY_6D, value[-1]].reshape(1, 10).astype(np.float32)

    def native_action_from_canonical(
        self, canonical: np.ndarray, *, iterations: int
    ) -> np.ndarray:
        assert iterations == 20
        value = np.asarray(canonical, dtype=np.float32).reshape(10)
        return np.r_[
            value[:3], np.zeros(self.morphology.native_action_dim - 4), value[-1]
        ].astype(np.float32)

    @staticmethod
    def native_to_qpos(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)

    @staticmethod
    def qpos_to_native(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)

    def apply_action(self, value: np.ndarray) -> None:
        self.applied.append(np.asarray(value).copy())
        self.native = np.asarray(value, dtype=np.float32).copy()
        self.canonical[0, 0] = len(self.applied) / 100

    def step_control_period(self, frequency: float) -> FakeStatus:
        assert frequency == 30.0
        return FakeStatus(self.success)

    def close(self) -> None:
        pass


class FakeExpert:
    frequency = 30.0

    def __init__(self, env: FakeEnv) -> None:
        self.env = env
        self.phases = ("open", *mechanism.PHASES[env.task], "done")
        self.index = 0
        self.phase = "open"
        self.phase_tick = 0
        self.total_tick = 0
        self.action_calls = 0

    @property
    def terminal(self) -> bool:
        return self.phase == "done"

    def action(self) -> np.ndarray:
        self.action_calls += 1
        self.index += 1
        self.phase = self.phases[self.index]
        self.phase_tick = 1
        self.total_tick += 1
        if self.phase == "done":
            self.env.success = True
            return self.env.state()
        return np.r_[
            np.full(3, self.action_calls / 100),
            np.zeros(self.env.morphology.native_action_dim - 4),
            0.5,
        ].astype(np.float32)


class FakeProcessor:
    def __call__(self, _images, instructions, _state, **kwargs):
        key = next(key for key, value in mechanism.INSTRUCTIONS.items() if value == instructions[0])
        token = tuple(mechanism.INSTRUCTIONS).index(key) + 1
        return {
            "input_ids": torch.tensor([[token, token + 10]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 2), dtype=torch.int64),
            **kwargs,
        }


class FakePolicy:
    config = SimpleNamespace(
        image_do_rescale=False,
        backbone_name="fake-backbone",
        backbone_revision="fake-revision",
    )

    def __init__(self, value: float) -> None:
        self.value = value
        self.reset_calls = 0
        self.predict_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_canonical_action_chunk(self, _batch):
        self.predict_calls += 1
        output = torch.zeros((1, 32, 1, 10), dtype=torch.float32)
        output[..., 0] = self.value
        output[..., 3:9] = torch.from_numpy(IDENTITY_6D)
        output[..., 9] = self.value
        return output


def test_frozen_constants_scenarios_and_all_six_balanced_permutations() -> None:
    assert mechanism.SCENARIO_INDICES == tuple(range(35, 42))
    assert mechanism.EXPECTED_EPISODES == 42
    assert mechanism.EXPECTED_PHASE_UNITS == 182
    assert mechanism.EXPECTED_ENDPOINTS == 364
    assert mechanism.EXPECTED_SCORED_CHUNKS == 3276
    assert mechanism.EXPECTED_REPEAT_CHUNKS == 3276
    assert mechanism.EXPECTED_INFERENCES == 6552
    assert mechanism.EXPECTED_CONTRAST_OBSERVATIONS == 1092
    for morphology in mechanism.MORPHOLOGIES:
        scenarios = mechanism.gripper_mechanism_scenarios(
            _manifest(), morphology, registered=True
        )
        assert len(scenarios) == 21
        assert scenarios[0].scenario_id.endswith("0035")
        assert scenarios[-1].scenario_id.endswith("0041")
        assert sum(len(mechanism.PHASES[value.task]) for value in scenarios) == 91
        for seed in mechanism.SEEDS:
            orders = [mechanism.condition_order(index, seed, morphology) for index in range(182)]
            assert set(orders) == set(mechanism.CONDITION_PERMUTATIONS)
            assert max(orders.count(order) for order in set(orders)) - min(
                orders.count(order) for order in set(orders)
            ) <= 1


def test_added_gripper_metrics_are_raw_signed_strict_and_clipped() -> None:
    target = np.r_[np.zeros(3), IDENTITY_6D, 0.25]
    prediction = target.copy()
    prediction[9] = 1.5
    metrics = mechanism.action_metrics(prediction, target)
    assert metrics["raw_gripper_absolute_error"] == 1.25
    assert metrics["signed_gripper_bias"] == 1.25
    assert metrics["effective_gripper_error"] == 0.75
    assert metrics["gripper_saturation_rate"] == 1.0
    for boundary in (0.0, 1.0):
        prediction[9] = boundary
        assert mechanism.action_metrics(prediction, target)["gripper_saturation_rate"] == 0.0
    prediction[9] = -0.1
    assert mechanism.action_metrics(prediction, target)["signed_gripper_bias"] == pytest.approx(
        -0.35
    )


def _record(
    value: float,
    condition: str,
    *,
    seed: int = 36001,
    morphology: str = "so101",
    task: str = "push_to_zone",
    scenario: str = "a",
    phase: str = "precontact",
    endpoint: str = "first",
) -> dict:
    return {
        "seed": seed,
        "condition": condition,
        "morphology": morphology,
        "task": task,
        "scenario_id": scenario,
        "phase": phase,
        "endpoint": endpoint,
        "metrics": {
            **{name: value for name in mechanism.METRICS},
            "channel_losses": [value] * 10,
            "degenerate_rotation": False,
        },
    }


def test_three_condition_hierarchy_contrasts_and_exact_decomposition() -> None:
    records = []
    for condition, offset in (("full", 0.0), ("half", 1.0), ("joint", 3.0)):
        records.extend(
            [
                _record(offset, condition, endpoint="first"),
                _record(offset + 2, condition, endpoint="last"),
                _record(offset + 8, condition, scenario="b", endpoint="first"),
                _record(offset + 8, condition, scenario="b", endpoint="last"),
            ]
        )
    metrics = mechanism.aggregate_metrics(records)
    cell = metrics["overall"]["effective_gripper_error"]
    assert (cell["full"], cell["half"], cell["joint"]) == (4.5, 5.5, 7.5)
    assert cell["contrasts"]["J-F"]["difference"] == 3
    assert cell["contrasts"]["H-F"]["difference"] == 1
    assert cell["contrasts"]["J-H"]["difference"] == 2
    assert cell["contrasts"]["J-F"]["difference"] == (
        cell["contrasts"]["H-F"]["difference"]
        + cell["contrasts"]["J-H"]["difference"]
    )
    assert cell["contrasts"]["J-F"]["difference_decimal"] == "3.0"
    assert metrics["by_endpoint"]["first"]["effective_gripper_error"]["full"] == 4
    assert metrics["by_channel"]["9"]["contrasts"]["J-H"]["difference"] == 2


def _classification_metrics(full: float, half: float, joint: float) -> dict:
    def cell() -> dict:
        values = {"full": full, "half": half, "joint": joint}
        decimals = {name: Decimal(str(value)) for name, value in values.items()}
        values["contrasts"] = {
            name: {
                "difference": float(decimals[numerator] - decimals[denominator]),
                "difference_decimal": str(decimals[numerator] - decimals[denominator]),
                "ratio": float(decimals[numerator] / decimals[denominator]),
                "ratio_decimal": str(decimals[numerator] / decimals[denominator]),
            }
            for name, (numerator, denominator) in mechanism.CONTRASTS.items()
        }
        return {"effective_gripper_error": values}

    return {
        "overall": cell(),
        "by_seed": {str(seed): cell() for seed in mechanism.SEEDS},
        "by_endpoint": {endpoint: cell() for endpoint in mechanism.ENDPOINTS},
        "by_task": {task: cell() for task in mechanism.TASKS},
        "by_morphology": {morphology: cell() for morphology in mechanism.MORPHOLOGIES},
    }


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.1, 0.111, 0.123), "both-supported-broad"),
        ((0.1, 0.105, 0.12), "shared-training-dominant-broad"),
        ((0.1, 0.115, 0.12), "exposure-dominant-broad"),
        ((0.1, 0.106, 0.112), "neither-materially-supported"),
    ],
)
def test_all_four_mechanism_classes(values: tuple[float, ...], expected: str) -> None:
    result = mechanism.classify_mechanism(_classification_metrics(*values), delivery=True)
    assert result["classification"] == expected
    assert result["inferential_claim"] is False


@pytest.mark.parametrize("failure", ["delivery", "ratio", "difference", "seeds"])
def test_premise_every_materiality_boundary(failure: str) -> None:
    metrics = _classification_metrics(0.1, 0.111, 0.123)
    delivery = failure != "delivery"
    contrast = metrics["overall"]["effective_gripper_error"]["contrasts"]["J-F"]
    if failure == "ratio":
        contrast["ratio_decimal"] = str(np.nextafter(1.10, 0.0))
    elif failure == "difference":
        contrast["difference_decimal"] = str(np.nextafter(0.01, 0.0))
    elif failure == "seeds":
        for seed in (36001, 36002):
            metrics["by_seed"][str(seed)]["effective_gripper_error"]["contrasts"]["J-F"][
                "ratio_decimal"
            ] = "1.09"
    assert mechanism.classify_mechanism(metrics, delivery=delivery)["classification"] == (
        "premise-not-confirmed"
    )
    exact = _classification_metrics(0.1, 0.11, 0.12)
    assert mechanism.classify_mechanism(exact, delivery=True)["premise"]["supported"] is True


def test_exact_decimal_boundaries_pass_every_materiality_gate() -> None:
    exposure = mechanism.classify_mechanism(
        _classification_metrics(0.10, 0.11, 0.13), delivery=True
    )
    assert exposure["premise"]["supported"] is True
    assert exposure["exposure"]["supported"] is True

    shared = mechanism.classify_mechanism(
        _classification_metrics(0.10, 0.10, 0.11), delivery=True
    )
    assert shared["premise"]["supported"] is True
    assert shared["shared_training"]["supported"] is True

    metrics = mechanism.aggregate_metrics(
        [
            _record(value, condition, seed=seed)
            for seed in mechanism.SEEDS
            for condition, value in (("full", 0.10), ("half", 0.105), ("joint", 0.11))
        ]
    )
    premise = metrics["overall"]["effective_gripper_error"]["contrasts"]["J-F"]
    assert premise["difference"] == 0.01
    assert premise["ratio"] == 1.1
    assert premise["difference_decimal"] == "0.01"
    assert premise["ratio_decimal"] == "1.1"
    assert mechanism.classify_mechanism(metrics, delivery=True)["premise"]["supported"] is True


def test_exact_decomposition_survives_adversarial_float_scale() -> None:
    records = [
        _record(value, condition)
        for condition, value in (
            ("full", 1e16),
            ("half", 1e16 + 2),
            ("joint", 1e16 + 4),
        )
    ]
    cell = mechanism.aggregate_metrics(records)["overall"]["effective_gripper_error"]
    assert Decimal(cell["contrasts"]["J-F"]["difference_decimal"]) == (
        Decimal(cell["contrasts"]["H-F"]["difference_decimal"])
        + Decimal(cell["contrasts"]["J-H"]["difference_decimal"])
    )


def test_broad_requires_every_supported_contrast_cell_and_negative_transfer_is_descriptive() -> None:
    metrics = _classification_metrics(0.1, 0.111, 0.123)
    metrics["by_task"]["lift_object"]["effective_gripper_error"]["contrasts"]["H-F"][
        "difference_decimal"
    ] = "-1e-9"
    assert mechanism.classify_mechanism(metrics, delivery=True)["classification"] == (
        "both-supported"
    )
    transfer = mechanism.classify_mechanism(
        _classification_metrics(0.1, 0.13, 0.12), delivery=True
    )
    assert transfer["negative_J_minus_H_describes_positive_transfer"] is True


def test_frozen_source_and_real_half_checkpoint_hash_validation(tmp_path: Path) -> None:
    summaries = mechanism._source_contract(_sources())
    assert summaries["exp36_summary"]["experiment"] == 36
    local_checkpoint = ROOT / "outputs/benchmark-exp43-half-so101-seed36001/final/so101"
    if not local_checkpoint.is_dir():
        pytest.skip("local Experiment 043 outputs are unavailable")
    checkpoint, identity = mechanism.resolve_experiment43_checkpoint(
        _sources(), "half", 36001, "so101", root=ROOT
    )
    assert checkpoint == ROOT / "outputs/benchmark-exp43-half-so101-seed36001/final/so101"
    assert mechanism.file_sha256(checkpoint / "core.pt") == identity["core_sha256"]
    changed = tmp_path / "training.json"
    changed.write_bytes(_sources()["exp43_training_summary"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="training summary hash"):
        mechanism.resolve_experiment43_checkpoint(
            {**_sources(), "exp43_training_summary": changed},
            "half",
            36001,
            "so101",
            root=ROOT,
        )


def test_full_checkpoint_requires_frozen_exp42_auxiliary_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mechanism,
        "resolve_exp36_checkpoint",
        lambda *_args, **_kwargs: (
            Path("checkpoint"),
            {
                "embodiment_sha256": "0" * 64,
                "trainer_sha256": mechanism.EXP42_TRAINER_SHA256,
            },
        ),
    )
    with pytest.raises(ValueError, match="frozen Experiment 042 identity"):
        mechanism.resolve_experiment43_checkpoint(
            _sources(), "full", 36001, "so101", root=ROOT
        )


def _smoke_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    def resolve(_sources_value, condition, seed, morphology, *, root):
        assert root == Path("fake-root")
        path = Path("fake") / condition / str(seed) / morphology
        identity = {
            "condition": condition,
            "source_condition": morphology if condition != "joint" else "joint",
            "model_seed": seed,
            "morphology": morphology,
            "checkpoint_path": str(path),
            "core_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "embodiment_sha256": "3" * 64,
            "trainer_sha256": "4" * 64,
            "data_manifest_sha256": "5" * 64,
            "source_summary_sha256": "6" * 64,
            "step": 1200,
            "stage": "core",
        }
        return path, identity

    monkeypatch.setattr(mechanism, "resolve_experiment43_checkpoint", resolve)
    monkeypatch.setattr(
        mechanism.exp42,
        "camera_frame_to_tensor",
        lambda _frame, *, processor_rescales: torch.zeros((1,), dtype=torch.float32),
    )
    monkeypatch.setattr(
        mechanism,
        "_runtime_provenance",
        lambda _device: {
            "git_dirty": True,
            "git_commit": "smoke",
            "evaluator_sha256": "smoke",
            "imported_exp42_evaluator_sha256": mechanism.EXP42_EVALUATOR_SHA256,
            "expert_sha256": mechanism.EXPERT_SHA256,
            "runtime": {"device": "cpu"},
            "runtime_signature_sha256": "smoke",
        },
    )
    policies = []

    def load(path: Path, _device: str):
        value = {"full": 0.1, "half": 0.2, "joint": 0.3}[path.parts[1]]
        policy = FakePolicy(value)
        policies.append(policy)
        return policy

    output = tmp_path / "smoke-anywhere.json"
    report = mechanism.evaluate_gripper_mechanism_shard(
        BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json"),
        _manifest(),
        _sources(),
        output,
        morphology="so101",
        device="cpu",
        registered=False,
        limit_scenarios=1,
        root=Path("fake-root"),
        env_factory=lambda morphology, task, _scenario, **_kwargs: FakeEnv(task, morphology),
        expert_factory=FakeExpert,
        policy_loader=load,
        processor_factory=lambda *_args, **_kwargs: FakeProcessor(),
    )
    assert len(policies) == 9
    assert all(policy.reset_calls == policy.predict_calls == 8 for policy in policies)
    assert json.loads(output.read_text()) == report
    return report


def test_one_scenario_smoke_report_and_independent_report_mutation_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _smoke_report(monkeypatch, tmp_path)
    assert report["expected_anchor_ids"] == [
        "development-so101-push_to_zone-0035/precontact/first",
        "development-so101-push_to_zone-0035/precontact/last",
        "development-so101-push_to_zone-0035/push/first",
        "development-so101-push_to_zone-0035/push/last",
    ]
    assert report["aggregate"]["scored_chunks"] == 36
    assert report["aggregate"]["repeat_chunks"] == 36
    assert report["aggregate"]["policy_inferences"] == 72
    for anchor in report["anchors"]:
        for seed in mechanism.SEEDS:
            assert tuple(anchor["seeds"][str(seed)]) == mechanism.condition_order(
                anchor["ordinal"], seed, "so101"
            )

    changed = json.loads(json.dumps(report))
    changed["anchors"][0]["seeds"]["36001"]["full"]["queries"][0]["output_chunk"][0][
        0
    ][0] += 1
    changed["outcome_chain_sha256"] = mechanism._outcome_chain(changed["anchors"])
    with pytest.raises(ValueError, match="tensor hashes"):
        mechanism.validate_shard_report(
            changed,
            _manifest(),
            _sources(),
            require_registered=False,
            artifact_root=Path("fake-root"),
        )

    changed = json.loads(json.dumps(report))
    changed["identity"]["checkpoints"]["36001"]["half"]["core_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint identity"):
        mechanism.validate_shard_report(
            changed,
            _manifest(),
            _sources(),
            require_registered=False,
            artifact_root=Path("fake-root"),
        )

    changed = json.loads(json.dumps(report))
    changed["anchors"][0]["pre_call_total_tick"] = float(
        changed["anchors"][0]["pre_call_total_tick"]
    )
    changed["outcome_chain_sha256"] = mechanism._outcome_chain(changed["anchors"])
    with pytest.raises(ValueError, match="endpoint anchor identity"):
        mechanism.validate_shard_report(
            changed,
            _manifest(),
            _sources(),
            require_registered=False,
            artifact_root=Path("fake-root"),
        )


def test_registered_paths_lock_atomic_writes_and_source_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert mechanism.registered_shard_filename("panda") == (
        "benchmark-exp43-panda-gripper-mechanism.json"
    )
    with pytest.raises(ValueError, match="registered shard output"):
        mechanism._registered_output("panda", Path("custom.json"))
    lock = tmp_path / ".lock"
    monkeypatch.setattr(mechanism, "REGISTERED_LOCK_PATH", lock)
    with mechanism.registered_run_lock(lock):
        with pytest.raises(RuntimeError, match="holds"):
            with mechanism.registered_run_lock(lock):
                pass

    final = tmp_path / "final.json"
    mechanism._write_final(final, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        mechanism._write_final(final, {"value": 2})

    staged = tmp_path / "staged"
    published = tmp_path / "published"
    staged.mkdir()
    published.mkdir()
    paths = {}
    hashes = {}
    for morphology in mechanism.MORPHOLOGIES:
        path = staged / mechanism.registered_shard_filename(morphology)
        path.write_bytes(f"{morphology}\n".encode())
        paths[morphology] = path
        hashes[morphology] = hashlib.sha256(path.read_bytes()).hexdigest()
    mechanism._promote_registered_sources(paths, hashes, destination_dir=published)
    for morphology in mechanism.MORPHOLOGIES:
        assert (
            published / mechanism.registered_shard_filename(morphology)
        ).read_bytes() == paths[morphology].read_bytes()
    paths["panda"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after registered aggregation"):
        mechanism._promote_registered_sources(paths, hashes, destination_dir=published)


def test_registered_aggregation_classification_common_runtime_and_summary_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mechanism, "validate_shard_report", lambda *_args, **_kwargs: None)
    anchor_count = len(mechanism.TASKS) * len(mechanism.ENDPOINTS) * len(
        mechanism.MORPHOLOGIES
    )
    observations = anchor_count * len(mechanism.SEEDS)
    scored = observations * len(mechanism.CONDITIONS)
    monkeypatch.setattr(mechanism, "EXPECTED_SCORED_CHUNKS", scored)
    monkeypatch.setattr(mechanism, "EXPECTED_CONTRAST_OBSERVATIONS", observations)
    run_id = "a" * 32
    runtime = {"device": "cpu"}
    signature = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    paths = {}
    for position, morphology in enumerate(mechanism.MORPHOLOGIES):
        anchors = []
        for task in mechanism.TASKS:
            for endpoint in mechanism.ENDPOINTS:
                seeds = {}
                for seed in mechanism.SEEDS:
                    seeds[str(seed)] = {
                        condition: {
                            "seed": seed,
                            "condition": condition,
                            "metrics": _record(
                                {"full": 0.1, "half": 0.111, "joint": 0.123}[
                                    condition
                                ],
                                condition,
                            )["metrics"],
                        }
                        for condition in mechanism.CONDITIONS
                    }
                anchors.append(
                    {
                        "morphology": morphology,
                        "task": task,
                        "scenario_id": f"{morphology}/{task}",
                        "phase": mechanism.PHASES[task][0],
                        "endpoint": endpoint,
                        "seeds": seeds,
                    }
                )
        report = {
            "identity": {"morphology": morphology},
            "anchors": anchors,
            "provenance": {
                "evaluator_sha256": mechanism.file_sha256(Path(mechanism.__file__)),
                "git_commit": "b" * 40,
                "runtime_signature_sha256": signature,
                "runtime": runtime,
            },
            "run": {
                "schedule_position": position,
                "orchestrator_run_id": run_id,
                "started_utc": f"2026-01-01T00:00:0{position * 2}Z",
                "finished_utc": f"2026-01-01T00:00:0{position * 2 + 1}Z",
            },
            "aggregate": {"delivery_gate": {"passed": True}},
        }
        path = tmp_path / mechanism.registered_shard_filename(morphology)
        path.write_text(json.dumps(report, indent=2) + "\n")
        paths[morphology] = path
    summary = mechanism.aggregate_registered_shards(
        paths, _manifest(), _sources(), artifact_root=ROOT
    )
    assert summary["contract"]["scored_chunks"] == scored
    assert summary["contract"]["contrast_observations"] == {
        contrast: observations for contrast in mechanism.CONTRASTS
    }
    assert summary["mechanism_classification"]["classification"] == (
        "both-supported-broad"
    )
    mechanism.validate_summary_report(
        summary, paths, _manifest(), _sources(), artifact_root=ROOT
    )

    panda = paths["panda"]
    original = panda.read_bytes()
    changed = json.loads(original)
    changed["provenance"]["git_commit"] = "c" * 40
    panda.write_text(json.dumps(changed, indent=2) + "\n")
    try:
        with pytest.raises(ValueError, match="common clean commit/runtime"):
            mechanism.aggregate_registered_shards(
                paths, _manifest(), _sources(), artifact_root=ROOT
            )
    finally:
        panda.write_bytes(original)


def test_nonresumable_orchestration_race_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(mechanism, "REGISTERED_RUNTIME_DIR", runtime)
    monkeypatch.setattr(mechanism, "REGISTERED_LOCK_PATH", runtime / ".lock")

    @contextmanager
    def race_lock(_path):
        (runtime / mechanism.registered_shard_filename("so101")).write_text("race")
        yield

    monkeypatch.setattr(mechanism, "registered_run_lock", race_lock)
    with pytest.raises(FileExistsError, match="appeared before schedule"):
        mechanism._run_all_registered(SimpleNamespace(), None, None)


def test_cli_registration_and_smoke_path_guards() -> None:
    smoke = mechanism.parse_args(
        [
            "shard",
            "--morphology",
            "so101",
            "--smoke",
            "--output",
            "/tmp/arbitrary-exp43-smoke.json",
            "--limit-scenarios",
            "1",
        ]
    )
    assert smoke.smoke is True and smoke.limit_scenarios == 1
    assert mechanism.parse_args(["aggregate"]).output == mechanism.SUMMARY_PATH
    assert "embodi-diagnose-exp43-gripper" in (ROOT / "pyproject.toml").read_text()
    assert "reports/exp43-runtime/" in (ROOT / ".gitignore").read_text().splitlines()


def test_registered_summary_hash_counts_decomposition_and_classification() -> None:
    path = ROOT / "reports/benchmark-exp43-summary.json"
    report = json.loads(path.read_text())
    assert mechanism.file_sha256(path) == (
        "acdf15900cdb708e8f9b96a5c12f2d70be18f951906b92ade8d2a874b69d15a4"
    )
    assert report["delivery_gate_passed"] is True
    assert report["contract"]["endpoint_records"] == 364
    primary = report["metrics"]["overall"]["effective_gripper_error"]
    exact = {
        contrast: Decimal(values["difference_decimal"])
        for contrast, values in primary["contrasts"].items()
    }
    assert exact["J-F"] == exact["H-F"] + exact["J-H"]
    classification = report["mechanism_classification"]
    assert classification["premise"]["supported"] is True
    assert classification["exposure"]["supported"] is False
    assert classification["shared_training"]["supported"] is False
    assert classification["classification"] == "neither-materially-supported"


def test_registered_source_archives_reconstruct_validated_reports() -> None:
    archives = {
        "so101": (
            "72f5767a8ff96dfa6d13237ff47e269af0f3848e7d45ebb9a59d924f313feec0",
            "7b2ce863b96ecd873f28cc80a86d8279ec8e58dee3db346d5d82593aee998bb8",
        ),
        "panda": (
            "4dee6a5b629e112b9867d2766e3b03c0a0c0121568691d5522b0836b0b83db06",
            "ec3d3534227b86c4bebfa20ceae264f66d86fd1dbaf32cfd3a78818dd20b05ec",
        ),
    }
    for morphology, (archive_sha256, source_sha256) in archives.items():
        path = ROOT / f"reports/benchmark-exp43-{morphology}-gripper-mechanism.json.gz"
        assert mechanism.file_sha256(path) == archive_sha256
        digest = hashlib.sha256()
        with gzip.open(path, "rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        assert digest.hexdigest() == source_sha256
