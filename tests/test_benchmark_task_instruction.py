import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodi.sim.benchmark import BenchmarkDefinition, ScenarioManifest
import embodi.sim.benchmark_task_instruction as task_instruction


ROOT = Path(__file__).parents[1]
IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _scenario(
    task: str = "pick_place_bin", morphology: str = "panda"
) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=f"development-{morphology}-{task}-0021",
        seed=41,
        morphology=morphology,
        task=task,
        factors={},
    )


class FakeStatus:
    def __init__(self, terminal: bool, success: bool) -> None:
        self.terminal = terminal
        self.success = success


class FakeAnchorEnv:
    def __init__(self, task: str = "pick_place_bin", morphology: str = "panda") -> None:
        self.task = task
        self.morphology = SimpleNamespace(
            id=morphology, native_action_dim=6 if morphology == "so101" else 8
        )
        self.success = False
        self.applied: list[np.ndarray] = []
        self.state_value = (
            np.r_[np.zeros(3), IDENTITY_6D, 0.5].reshape(1, 10).astype(np.float32)
        )

    def render_top(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def canonical_state(self) -> np.ndarray:
        return self.state_value.copy()

    def canonical_arm_action(self, native: np.ndarray) -> np.ndarray:
        native = np.asarray(native, dtype=np.float32)
        value = np.r_[native[:3], IDENTITY_6D, native[-1]].reshape(1, 10)
        return value.astype(np.float32)

    def native_action_from_canonical(
        self, canonical: np.ndarray, *, iterations: int
    ) -> np.ndarray:
        assert iterations == 20
        value = np.asarray(canonical, dtype=np.float32).reshape(10)
        return np.r_[
            value[:3],
            9.0,
            np.zeros(self.morphology.native_action_dim - 5),
            value[-1],
        ].astype(np.float32)

    @staticmethod
    def native_to_qpos(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)

    @staticmethod
    def qpos_to_native(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)

    def apply_action(self, value: np.ndarray) -> None:
        self.applied.append(value)
        self.state_value[0, 0] = len(self.applied)

    def step_control_period(self, frequency: float) -> FakeStatus:
        assert frequency == 30.0
        return FakeStatus(self.success, self.success)

    def close(self) -> None:
        pass


class FakeExpert:
    frequency = 30.0

    def __init__(self, env: FakeAnchorEnv) -> None:
        self.env = env
        self.phases = ("open", *task_instruction.PHASES[env.task], "done")
        self.index = 0
        self.phase = self.phases[0]
        self.phase_tick = 0
        self.action_calls = 0

    @property
    def terminal(self) -> bool:
        return self.phase == "done"

    def action(self) -> np.ndarray:
        self.action_calls += 1
        self.index += 1
        self.phase = self.phases[self.index]
        self.phase_tick = 1
        if self.terminal:
            self.env.success = True
        return np.r_[
            np.full(3, self.action_calls / 100),
            np.zeros(self.env.morphology.native_action_dim - 4),
            0.5,
        ].astype(np.float32)


class FakeProcessor:
    def __call__(self, _images, instructions, _state, **kwargs):
        key = next(
            key
            for key, value in task_instruction.INSTRUCTIONS.items()
            if value == instructions[0]
        )
        token = tuple(task_instruction.INSTRUCTIONS).index(key) + 1
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

    def __init__(self, repeat_delta: float = 0.0) -> None:
        self.reset_calls = 0
        self.predict_calls = 0
        self.repeat_delta = repeat_delta

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_canonical_action_chunk(self, batch):
        self.predict_calls += 1
        token = int(batch["input_ids"][0, 0])
        output = torch.zeros((1, 32, 1, 10), dtype=torch.float32)
        output[..., :3] = token / 100
        output[..., 3:9] = torch.from_numpy(IDENTITY_6D)
        output[..., 9] = 0.5
        if self.predict_calls == 4:
            output[0, 0, 0, 0] += self.repeat_delta
        return output


def _anchor(task: str = "push_to_zone", ordinal: int = 0) -> task_instruction.Anchor:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = np.r_[np.zeros(3), IDENTITY_6D, 0.5].reshape(1, 10).astype(np.float32)
    target = np.r_[np.zeros(3), IDENTITY_6D, 0.5].reshape(1, 10).astype(np.float32)
    return task_instruction.Anchor(
        ordinal,
        f"development-panda-{task}-0021",
        41,
        "panda",
        task,
        task_instruction.PHASES[task][0],
        frame,
        state,
        target,
        {
            "ordinal": ordinal,
            "scenario_id": f"development-panda-{task}-0021",
            "scenario_seed": 41,
            "morphology": "panda",
            "task": task,
            "phase": task_instruction.PHASES[task][0],
            "phase_tick": 1,
            "image_sha256": task_instruction.canonical_array_sha256(frame),
            "state_sha256": task_instruction.canonical_array_sha256(state),
        },
    )


def _frozen_source_paths() -> dict[str, Path]:
    return {
        "exp36_summary": ROOT / "reports/benchmark-exp36-summary.json",
        "exp38_summary": ROOT / "reports/benchmark-exp38-summary.json",
        "exp39_summary": ROOT / "reports/benchmark-exp39-summary.json",
        "exp40_summary": ROOT / "reports/benchmark-exp40-summary.json",
    }


def _metric_record(
    value: float,
    *,
    scenario: str,
    morphology: str = "so101",
    task: str = "push_to_zone",
    phase: str = "precontact",
) -> dict:
    metrics = {
        "correct": {
            "loss": value,
            "translation_norm_error_m": value,
            "rotation_geodesic_error_deg": value,
            "effective_gripper_error": value,
            "degenerate_rotation_count": 0,
        },
        "wrong": {
            "loss": value + 1,
            "translation_norm_error_m": value + 1,
            "rotation_geodesic_error_deg": value + 1,
            "effective_gripper_error": value + 1,
            "degenerate_rotation_count": 0,
            "degenerate_rotation_rate": 0.0,
        },
        "correct_instruction_top_one_rate": value,
        "output_separation": value,
    }
    return {
        "morphology": morphology,
        "task": task,
        "phase": phase,
        "scenario_id": scenario,
        "metrics": metrics,
    }


def test_registered_slice_phase_counts_permutations_condition_order_and_filenames() -> (
    None
):
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    scenarios = task_instruction.task_instruction_scenarios(
        manifest, "so101", registered=True
    )
    assert len(scenarios) == 21
    assert scenarios[0].scenario_id.endswith("push_to_zone-0021")
    assert scenarios[-1].scenario_id.endswith("pick_place_bin-0027")
    assert sum(len(task_instruction.PHASES[value.task]) for value in scenarios) == 91
    assert [task_instruction.instruction_order(index) for index in range(6)] == list(
        task_instruction.INSTRUCTION_PERMUTATIONS
    )
    assert task_instruction.condition_order(36001, "so101") == ("joint", "matched")
    assert task_instruction.condition_order(36001, "panda") == ("matched", "joint")
    assert task_instruction.registered_shard_filename(36003, "panda") == (
        "benchmark-exp41-seed36003-panda-task-instruction.json"
    )
    assert task_instruction.parse_args(["aggregate"]).shard_dir == Path(
        "reports/exp41-runtime"
    )


def test_expert_capture_uses_exact_loop_entry_phases_and_original_actions() -> None:
    env = FakeAnchorEnv()
    expert = FakeExpert(env)
    anchors, episode = task_instruction.capture_expert_episode(
        env,
        _scenario(),
        starting_ordinal=0,
        expert_factory=lambda _env: expert,
    )
    assert [value.phase for value in anchors] == list(
        task_instruction.PHASES["pick_place_bin"]
    )
    assert [value.ordinal for value in anchors] == list(range(7))
    assert expert.action_calls == len(env.applied) == 8
    assert all(applied.shape == (8,) for applied in env.applied)
    assert all(applied[3] == 0 for applied in env.applied)
    assert all(value.evidence["decoded_native"][3] == 9 for value in anchors)
    pre_action_state = (
        np.r_[np.array([1.0, 0.0, 0.0]), IDENTITY_6D, 0.5]
        .reshape(1, 10)
        .astype(np.float32)
    )
    assert anchors[0].evidence[
        "state_sha256"
    ] == task_instruction.canonical_array_sha256(pre_action_state)
    assert episode["success"] is True
    assert episode["final_phase"] == "done"
    assert anchors[0].evidence["round_trip"]["translation_error_m"] == 0


def test_smoke_evaluator_production_path_writes_and_validates_complete_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    exp36 = json.loads(_frozen_source_paths()["exp36_summary"].read_text())
    resolved = []
    loaded = []
    processor_calls = []

    def checkpoint_resolver(summary_path, condition, seed, morphology, *, root):
        assert summary_path == _frozen_source_paths()["exp36_summary"]
        assert root == Path("fake-root")
        resolved.append((condition, seed, morphology))
        identity = {
            "condition": condition,
            "model_seed": seed,
            "morphology": morphology,
            "checkpoint_path": f"fake/{condition}/{seed}/{morphology}",
            "core_sha256": exp36["runs"][f"{condition}-seed{seed}"]["core_sha256"],
            "config_sha256": task_instruction.CONFIG_SHA256[morphology],
            "embodiment_sha256": "3" * 64,
            "trainer_sha256": "4" * 64,
            "data_manifest_sha256": task_instruction.DATA_MANIFEST_SHA256[condition],
            "exp36_summary_sha256": task_instruction.EXP36_SUMMARY_SHA256,
            "step": 1200,
            "stage": "core",
        }
        return Path(identity["checkpoint_path"]), identity

    def policy_loader(path: Path, device: str):
        assert device == "cpu"
        loaded.append(path)
        return FakePolicy()

    def processor_factory(name, *, revision, do_rescale):
        processor_calls.append((name, revision, do_rescale))
        return FakeProcessor()

    monkeypatch.setattr(task_instruction, "resolve_checkpoint", checkpoint_resolver)
    monkeypatch.setattr(
        task_instruction,
        "_runtime_provenance",
        lambda device: {
            "git_dirty": True,
            "git_commit": "smoke",
            "evaluator_sha256": "smoke",
            "runtime": {"device": device},
            "runtime_signature_sha256": "smoke",
        },
    )

    def env_factory(_morphology, task, _scenario, **kwargs):
        assert kwargs == {"width": 640, "height": 480, "render": True}
        return FakeAnchorEnv(task, _morphology)

    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        output = Path(directory) / "exp41-smoke.json"
        report = task_instruction.evaluate_task_instruction_shard(
            definition,
            manifest,
            _frozen_source_paths(),
            output,
            seed=36001,
            morphology="so101",
            device="cpu",
            registered=False,
            limit_scenarios=1,
            root=Path("fake-root"),
            env_factory=env_factory,
            expert_factory=FakeExpert,
            policy_loader=policy_loader,
            processor_factory=processor_factory,
        )
        written = json.loads(output.read_text())
        assert written == report
        task_instruction.validate_shard_report(
            written, manifest, exp36, require_registered=False
        )
    assert resolved == [("joint", 36001, "so101"), ("so101", 36001, "so101")]
    assert [path.parts[1] for path in loaded] == ["joint", "so101"]
    assert processor_calls == [("fake-backbone", "fake-revision", False)]
    assert report["run"]["condition_order"] == ["joint", "matched"]
    assert report["expected_anchor_ids"] == [
        "development-so101-push_to_zone-0021/precontact",
        "development-so101-push_to_zone-0021/push",
    ]
    assert [anchor["phase"] for anchor in report["anchors"]] == [
        "precontact",
        "push",
    ]
    assert all(
        tuple(anchor["conditions"]) == ("joint", "matched")
        for anchor in report["anchors"]
    )
    assert all(
        query["output_sha256"]
        == task_instruction.canonical_array_sha256(
            np.asarray(query["output_chunk"], dtype=np.float32)[None]
        )
        for anchor in report["anchors"]
        for condition in task_instruction.CONDITIONS
        for query in anchor["conditions"][condition]["queries"]
    )
    assert report["aggregate"]["anchors"] == 2
    assert report["aggregate"]["predictions"] == 16
    assert report["aggregate"]["delivery_gate"]["passed"] is True
    assert report["complete"] is True
    assert report["status"] == "completed_smoke_unregistered"
    assert report["run"]["finished_utc"] is not None
    assert report["outcome_chain_sha256"] == task_instruction._outcome_chain(
        report["anchors"]
    )


def test_evaluator_rejects_frozen_source_hash_mismatch() -> None:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
        root = Path(directory)
        altered = root / "benchmark-exp40-summary.json"
        altered.write_text(_frozen_source_paths()["exp40_summary"].read_text() + "\n")
        sources = {**_frozen_source_paths(), "exp40_summary": altered}
        with pytest.raises(ValueError, match="exp40 summary hash"):
            task_instruction.evaluate_task_instruction_shard(
                definition,
                manifest,
                sources,
                root / "smoke.json",
                seed=36001,
                morphology="so101",
                device="cpu",
                registered=False,
                limit_scenarios=1,
            )


def test_policy_queries_reset_hash_tokenizer_and_score_first_correct() -> None:
    policy = FakePolicy(repeat_delta=5e-6)
    result = task_instruction.evaluate_anchor_condition(
        _anchor(ordinal=4), "joint", policy, FakeProcessor(), "cpu"
    )
    assert result["instruction_order"] == ["pick", "push", "lift"]
    assert policy.reset_calls == 4
    assert len({query["input_ids_sha256"] for query in result["queries"][:3]}) == 3
    assert all(
        query["image_sha256"] == result["queries"][0]["image_sha256"]
        for query in result["queries"]
    )
    assert result["repeat_correct_max_abs_difference"] == pytest.approx(5e-6, abs=1e-8)
    queries = result["queries"]
    first = {query["instruction_key"]: query for query in queries[:3]}
    predictions = {
        key: np.asarray(query["output_chunk"], dtype=np.float32)[0, 0]
        for key, query in first.items()
    }
    assert result["metrics"] == task_instruction.score_instruction_predictions(
        "push_to_zone", _anchor().expert_target, predictions
    )
    repeat_prediction = np.asarray(queries[3]["output_chunk"], dtype=np.float32)[0, 0]
    assert not np.array_equal(predictions["push"], repeat_prediction)
    assert queries[0]["input_ids_sha256"] == task_instruction.canonical_array_sha256(
        np.array([[3, 13]], dtype=np.int64)
    )
    assert queries[0]["output_sha256"] == task_instruction.canonical_array_sha256(
        np.asarray(queries[0]["output_chunk"], dtype=np.float32)[None]
    )


def test_policy_query_guards_fail_closed() -> None:
    class DriftProcessor(FakeProcessor):
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, images, instructions, state, **kwargs):
            self.calls += 1
            batch = super().__call__(images, instructions, state, **kwargs)
            if self.calls == 4:
                batch["attention_mask"][0, 0] = 0
            return batch

    with pytest.raises(RuntimeError, match="repeat-correct policy input differs"):
        task_instruction.evaluate_anchor_condition(
            _anchor(), "joint", FakePolicy(), DriftProcessor(), "cpu"
        )

    class SameTokenProcessor(FakeProcessor):
        def __call__(self, images, instructions, state, **kwargs):
            batch = super().__call__(images, instructions, state, **kwargs)
            batch["input_ids"] = torch.tensor([[1, 2]], dtype=torch.int64)
            return batch

    with pytest.raises(RuntimeError, match="distinct tokenizer"):
        task_instruction.evaluate_anchor_condition(
            _anchor(), "joint", FakePolicy(), SameTokenProcessor(), "cpu"
        )

    class WrongShapePolicy(FakePolicy):
        def predict_canonical_action_chunk(self, batch):
            return torch.zeros((1, 31, 1, 10), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"shape \[1,32,1,10\]"):
        task_instruction.evaluate_anchor_condition(
            _anchor(), "joint", WrongShapePolicy(), FakeProcessor(), "cpu"
        )


def test_losses_ties_physical_errors_and_degenerate_rotation() -> None:
    expert = np.r_[np.zeros(3), IDENTITY_6D, 0.5]
    predictions = {key: expert.copy() for key in task_instruction.INSTRUCTIONS}
    score = task_instruction.score_instruction_predictions(
        "push_to_zone", expert, predictions
    )
    assert score["correct"]["loss"] == score["wrong"]["loss"] == 0
    assert score["relative_benefit"] is None
    assert score["correct_instruction_top_one_rate"] == pytest.approx(1 / 3)
    assert score["output_separation"] == 0

    predictions["push"] = expert.copy()
    predictions["push"][:3] = [0.25, 0.5, 0.75]
    predictions["push"][9] = 2
    physical = task_instruction.action_metrics(predictions["push"], expert)
    assert physical["loss"] == pytest.approx((1 + 4 + 9 + 9) / 10)
    assert physical["translation_norm_error_m"] == pytest.approx(np.sqrt(0.875))
    assert physical["effective_gripper_error"] == 0.5

    degenerate = expert.copy()
    degenerate[3:9] = 0
    error, was_degenerate = task_instruction.rotation_geodesic_error(
        degenerate[3:9], expert[3:9]
    )
    assert error == 180
    assert was_degenerate is True


def test_unequal_wrong_arms_use_independent_means_and_fractional_ties() -> None:
    expert = np.r_[np.zeros(3), IDENTITY_6D, 0.5].astype(np.float32)
    correct = _action_for_loss(1.0)
    wrong_two = _action_for_loss(2.0)
    wrong_four = _action_for_loss(4.0)
    frozen_scales = np.array([0.25, 0.25, 0.25, 1, 1, 1, 1, 1, 1, 0.5])
    independently_computed_losses = [
        float(np.mean(((action.astype(np.float64) - expert) / frozen_scales) ** 2))
        for action in (correct, wrong_two, wrong_four)
    ]
    assert independently_computed_losses == pytest.approx([1.0, 2.0, 4.0])
    score = task_instruction.score_instruction_predictions(
        "push_to_zone",
        expert,
        {"push": correct, "lift": wrong_two, "pick": wrong_four},
    )
    assert score["correct"]["loss"] == pytest.approx(1.0)
    assert score["wrong"]["loss"] == pytest.approx(3.0)
    assert score["margin"] == pytest.approx(2.0)
    assert score["relative_benefit"] == pytest.approx(2 / 3)
    expected_wrong_translation = (np.sqrt(20) * 0.25 + np.sqrt(40) * 0.25) / 2
    assert score["wrong"]["translation_norm_error_m"] == pytest.approx(
        expected_wrong_translation
    )
    assert score["wrong"]["rotation_geodesic_error_deg"] == 0.0
    assert score["wrong"]["effective_gripper_error"] == 0.0
    expected_separation = (
        (np.sqrt(10) - np.sqrt(20)) ** 2 / 10 + (np.sqrt(10) - np.sqrt(40)) ** 2 / 10
    ) / 2
    assert score["output_separation"] == pytest.approx(expected_separation)
    assert score["correct_instruction_top_one_rate"] == 1.0

    tied = task_instruction.score_instruction_predictions(
        "push_to_zone",
        expert,
        {"push": correct, "lift": correct.copy(), "pick": wrong_four},
    )
    assert tied["correct_instruction_top_one_rate"] == 0.5


def test_hierarchy_weights_phases_then_scenarios_not_anchor_population() -> None:
    records = [
        _metric_record(0, scenario="a", phase="precontact"),
        _metric_record(2, scenario="a", phase="push"),
        _metric_record(9, scenario="b", phase="precontact"),
    ]
    metrics = task_instruction.hierarchical_metrics(records)
    assert metrics["correct_loss"] == 5  # scenario a mean 1, scenario b mean 9
    assert metrics["wrong_loss"] == 6
    assert metrics["margin"] == 1
    assert metrics["relative_benefit"] == pytest.approx(1 / 6)


def test_delivery_gates_use_default_percentile_and_fail_closed() -> None:
    anchors = []
    for index in range(100):
        native = np.zeros(8, dtype=np.float32)
        target = np.r_[np.zeros(3), IDENTITY_6D, 0.5].reshape(1, 10).astype(np.float32)
        realized = target.copy()
        realized[0, 0] = index / 100_000
        anchors.append(
            {
                "expert_native": native.tolist(),
                "decoded_native": native.tolist(),
                "realizable_native": native.astype(np.float64).tolist(),
                "expert_target": target.tolist(),
                "realized_canonical": realized.tolist(),
            }
        )
    gate = task_instruction.delivery_gate(anchors)
    assert gate["measurements"]["translation_p95_m"] == pytest.approx(
        np.percentile([index / 100_000 for index in range(100)], 95)
    )
    assert gate["passed"] is True
    anchors[0]["realizable_native"][0] = 1e-4
    assert task_instruction.delivery_gate(anchors)["passed"] is True
    anchors[1]["realizable_native"][0] = 1e-4
    assert task_instruction.delivery_gate(anchors)["passed"] is False
    clipped, error = task_instruction._clipping_from_arrays(
        np.zeros(1), np.array([2e-5])
    )
    assert clipped is False
    assert error == 2e-5


def _delivery_boundary_anchor(
    *, translation: float = 0.0, rotation_degrees: float = 0.0, gripper: float = 0.0
) -> dict:
    requested = np.zeros(8, dtype=np.float32)
    requested[-1] = gripper
    decoded = np.zeros(8, dtype=np.float32)
    target = np.r_[np.zeros(3), IDENTITY_6D, 0.0].reshape(1, 10).astype(np.float32)
    angle = np.deg2rad(rotation_degrees)
    rotation_6d = np.array(
        [np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0],
        dtype=np.float32,
    )
    realized = target.copy()
    realized[0, 0] = translation
    realized[0, 3:9] = rotation_6d
    return {
        "expert_native": requested.tolist(),
        "decoded_native": decoded.tolist(),
        "realizable_native": decoded.astype(np.float64).tolist(),
        "expert_target": target.tolist(),
        "realized_canonical": realized.tolist(),
    }


def test_delivery_physical_threshold_boundaries_and_empty_failure() -> None:
    exact = {
        "clipping_rate": 0.01,
        "translation_p95_m": 0.001,
        "rotation_p95_deg": 1.0,
        "maximum_gripper_error_native": 1.0,
    }
    assert all(task_instruction._delivery_checks(exact).values())
    for name, check, above in (
        ("translation_p95_m", "translation_p95_at_most_1mm", 0.0010001),
        ("rotation_p95_deg", "rotation_p95_at_most_1deg", 1.0001),
        (
            "maximum_gripper_error_native",
            "maximum_gripper_error_at_most_one_native_unit",
            1.0001,
        ),
    ):
        assert task_instruction._delivery_checks({**exact, name: above})[check] is False

    translation_at_most = float(np.nextafter(np.float32(0.001), np.float32(0.0)))
    translation_above = float(np.nextafter(np.float32(0.001), np.float32(np.inf)))
    assert (
        task_instruction.delivery_gate(
            [_delivery_boundary_anchor(translation=translation_at_most)]
        )["passed"]
        is True
    )
    assert (
        task_instruction.delivery_gate(
            [_delivery_boundary_anchor(translation=translation_above)]
        )["checks"]["translation_p95_at_most_1mm"]
        is False
    )

    assert (
        task_instruction.delivery_gate(
            [_delivery_boundary_anchor(rotation_degrees=1.0)]
        )["passed"]
        is True
    )
    assert (
        task_instruction.delivery_gate(
            [_delivery_boundary_anchor(rotation_degrees=1.001)]
        )["checks"]["rotation_p95_at_most_1deg"]
        is False
    )

    assert (
        task_instruction.delivery_gate([_delivery_boundary_anchor(gripper=1.0)])[
            "passed"
        ]
        is True
    )
    assert (
        task_instruction.delivery_gate([_delivery_boundary_anchor(gripper=1.001)])[
            "checks"
        ]["maximum_gripper_error_at_most_one_native_unit"]
        is False
    )
    empty = task_instruction.delivery_gate([])
    assert empty["passed"] is False
    assert all(value is False for value in empty["checks"].values())


def _seed_metrics(
    *, benefit: float | None, margin: float, correct: float
) -> dict[int, dict]:
    return {
        seed: {
            "relative_benefit": benefit,
            "margin": margin,
            "correct_loss": correct,
        }
        for seed in task_instruction.SEEDS
    }


def test_support_joint_deficit_and_null_behavior() -> None:
    task_margins = {task: 0.1 for task in task_instruction.TASKS}
    matched_seed = _seed_metrics(benefit=0.25, margin=0.2, correct=1.0)
    support = task_instruction.classify_support(
        matched_seed, task_margins, delivery=True
    )
    assert support["supported"] is True
    joint_seed = _seed_metrics(benefit=0.10, margin=0.1, correct=1.2)
    deficit = task_instruction.classify_joint_deficit(
        {
            "joint": {"correct_loss": 1.2, "relative_benefit": 0.10},
            "matched": {"correct_loss": 1.0, "relative_benefit": 0.25},
        },
        {"joint": joint_seed, "matched": matched_seed},
        support,
    )
    assert deficit["supported"] is True

    null_seed = _seed_metrics(benefit=None, margin=0, correct=0)
    null_support = task_instruction.classify_support(
        null_seed, task_margins, delivery=True
    )
    assert null_support["supported"] is False
    null_deficit = task_instruction.classify_joint_deficit(
        {
            "joint": {"correct_loss": 0, "relative_benefit": None},
            "matched": {"correct_loss": 0, "relative_benefit": None},
        },
        {"joint": null_seed, "matched": null_seed},
        support,
    )
    assert null_deficit["joint_to_matched_correct_loss_ratio"] is None
    assert null_deficit["matched_minus_joint_relative_benefit"] is None
    assert null_deficit["supported"] is False

    partly_undefined = copy.deepcopy(matched_seed)
    partly_undefined[36003]["correct_loss"] = 0.0
    partly_undefined[36003]["relative_benefit"] = None
    undefined = task_instruction.classify_joint_deficit(
        {
            "joint": {"correct_loss": 1.2, "relative_benefit": 0.10},
            "matched": {"correct_loss": 1.0, "relative_benefit": 0.25},
        },
        {"joint": joint_seed, "matched": partly_undefined},
        support,
    )
    assert undefined["checks"]["all_required_seed_values_defined"] is False
    assert undefined["supported"] is False


def test_classification_threshold_boundaries_are_inclusive() -> None:
    task_margins = {task: 0.0 for task in task_instruction.TASKS}
    exact_support = task_instruction.classify_support(
        _seed_metrics(benefit=0.10, margin=0.0, correct=1.0),
        task_margins,
        delivery=True,
    )
    assert exact_support["supported"] is True
    matched = _seed_metrics(benefit=0.20, margin=0.0, correct=1.0)
    support = task_instruction.classify_support(matched, task_margins, delivery=True)
    assert support["supported"] is True
    joint = _seed_metrics(benefit=0.10, margin=0.0, correct=1.1)
    deficit = task_instruction.classify_joint_deficit(
        {
            "joint": {"correct_loss": 1.1, "relative_benefit": 0.10},
            "matched": {"correct_loss": 1.0, "relative_benefit": 0.20},
        },
        {"joint": joint, "matched": matched},
        support,
    )
    assert deficit["supported"] is True


def test_anchor_integrity_rejects_metric_and_hash_mutations() -> None:
    anchor = _anchor()
    env = FakeAnchorEnv(task="push_to_zone")
    requested = np.r_[np.zeros(7), 0.5].astype(np.float32)
    target = env.canonical_arm_action(requested)
    decoded = env.native_action_from_canonical(target, iterations=20)
    realizable = decoded.astype(np.float64)
    realized = env.canonical_arm_action(decoded)
    evidence = anchor.evidence
    evidence.update(
        expert_native=requested.tolist(),
        expert_native_sha256=task_instruction.canonical_array_sha256(requested),
        expert_target=target.tolist(),
        expert_target_sha256=task_instruction.canonical_array_sha256(target),
        decoded_native=decoded.tolist(),
        decoded_native_sha256=task_instruction.canonical_array_sha256(decoded),
        realizable_native=realizable.tolist(),
        realizable_native_sha256=task_instruction.canonical_array_sha256(realizable),
        realized_canonical=realized.tolist(),
        realized_canonical_sha256=task_instruction.canonical_array_sha256(realized),
        round_trip={
            "clipped": False,
            "clipping_error_native": 0.0,
            "translation_error_m": 0.0,
            "rotation_error_deg": 0.0,
            "gripper_error_native": 0.0,
        },
    )
    results = {
        condition: task_instruction.evaluate_anchor_condition(
            anchor, condition, FakePolicy(), FakeProcessor(), "cpu"
        )
        for condition in task_instruction.CONDITIONS
    }
    record = {**evidence, "conditions": results}
    report = {
        "identity": {"morphology": "panda", "model_seed": 36001},
        "run": {"condition_order": ["joint", "matched"]},
    }
    task_instruction._validate_anchor_record(record, report, 0)
    changed = copy.deepcopy(record)
    changed["conditions"]["joint"]["metrics"]["margin"] += 1
    with pytest.raises(ValueError, match="metrics do not recompute"):
        task_instruction._validate_anchor_record(changed, report, 0)
    changed = copy.deepcopy(record)
    changed["conditions"]["matched"]["queries"][0]["image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input/output hashes"):
        task_instruction._validate_anchor_record(changed, report, 0)
    changed = copy.deepcopy(record)
    changed["conditions"]["joint"]["queries"][0]["output_chunk"][0][0][0] += 1
    with pytest.raises(ValueError, match="input/output hashes"):
        task_instruction._validate_anchor_record(changed, report, 0)
    changed = copy.deepcopy(record)
    changed["conditions"]["joint"]["queries"][3]["input_ids"][0][0] += 1
    with pytest.raises(ValueError, match="input/output hashes|repeat-correct"):
        task_instruction._validate_anchor_record(changed, report, 0)
    changed = copy.deepcopy(record)
    changed["round_trip"]["clipped"] = True
    with pytest.raises(ValueError, match="round-trip evidence"):
        task_instruction._validate_anchor_record(changed, report, 0)
    changed = copy.deepcopy(record)
    changed["realizable_native"][0] = 1e-4
    changed["realizable_native_sha256"] = task_instruction.canonical_array_sha256(
        np.asarray(changed["realizable_native"], dtype=np.float64)
    )
    with pytest.raises(ValueError, match="round-trip evidence"):
        task_instruction._validate_anchor_record(changed, report, 0)


def test_outcome_chain_canonical_hash_and_immutable_writes(tmp_path: Path) -> None:
    contiguous = np.arange(12, dtype=np.float32).reshape(3, 4)
    transposed = contiguous.T
    assert task_instruction.canonical_array_sha256(
        transposed
    ) == task_instruction.canonical_array_sha256(np.ascontiguousarray(transposed))
    outcomes = [{"a": 1}, {"b": np.array([2], dtype=np.int64)}]
    assert (
        task_instruction._outcome_chain(outcomes)
        != task_instruction.EMPTY_OUTCOME_CHAIN_SHA256
    )
    path = tmp_path / "report.json"
    task_instruction._write_final(path, {"value": 1})
    task_instruction._write_final(path, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        task_instruction._write_final(path, {"value": 2})


def test_registered_output_and_cli_guards() -> None:
    expected = Path(
        "reports/exp41-runtime/benchmark-exp41-seed36001-so101-task-instruction.json"
    )
    assert task_instruction._registered_output(36001, "so101", None) == expected
    with pytest.raises(ValueError, match="registered shard output"):
        task_instruction._registered_output(36001, "so101", Path("custom.json"))
    smoke = task_instruction.parse_args(
        [
            "shard",
            "--seed",
            "36001",
            "--morphology",
            "so101",
            "--smoke",
            "--limit-scenarios",
            "1",
            "--output",
            "smoke.json",
        ]
    )
    assert smoke.smoke is True
    assert smoke.limit_scenarios == 1


def test_registered_lock_contention() -> None:
    with task_instruction.registered_run_lock():
        with pytest.raises(RuntimeError, match="holds"):
            with task_instruction.registered_run_lock():
                pass


def test_run_all_registered_uses_one_lock_and_exact_six_shard_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    calls = []

    @contextmanager
    def fake_lock():
        events.append("enter")
        yield
        events.append("exit")

    def fake_output(seed, morphology, output):
        assert output is None
        return Path("/dev/shm") / task_instruction.registered_shard_filename(
            seed, morphology
        )

    def fake_evaluate(definition, manifest, sources, output, **kwargs):
        assert definition == "definition"
        assert manifest == "manifest"
        assert set(sources) == {
            "exp36_summary",
            "exp38_summary",
            "exp39_summary",
            "exp40_summary",
        }
        calls.append((kwargs["seed"], kwargs["morphology"], output.name))
        return {"seed": kwargs["seed"], "morphology": kwargs["morphology"]}

    monkeypatch.setattr(task_instruction, "registered_run_lock", fake_lock)
    monkeypatch.setattr(task_instruction, "_registered_output", fake_output)
    monkeypatch.setattr(
        task_instruction, "evaluate_task_instruction_shard", fake_evaluate
    )
    args = SimpleNamespace(
        exp36_summary=Path("36.json"),
        exp38_summary=Path("38.json"),
        exp39_summary=Path("39.json"),
        exp40_summary=Path("40.json"),
        device="cpu",
        artifact_root=Path("artifacts"),
    )
    reports = task_instruction._run_all_registered(args, "definition", "manifest")
    expected = [
        (seed, morphology, task_instruction.registered_shard_filename(seed, morphology))
        for seed, morphology in task_instruction.REGISTERED_SCHEDULE
    ]
    assert calls == expected
    assert [(value["seed"], value["morphology"]) for value in reports] == [
        (seed, morphology) for seed, morphology, _filename in expected
    ]
    assert events == ["enter", "exit"]


def test_run_all_registered_refuses_any_existing_shard_before_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = False

    @contextmanager
    def forbidden_lock():
        nonlocal entered
        entered = True
        yield

    monkeypatch.setattr(
        task_instruction,
        "_registered_output",
        lambda _seed, _morphology, _output: Path("/dev/null"),
    )
    monkeypatch.setattr(task_instruction, "registered_run_lock", forbidden_lock)
    args = SimpleNamespace()
    with pytest.raises(FileExistsError, match="non-resumable"):
        task_instruction._run_all_registered(args, "definition", "manifest")
    assert entered is False


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            SimpleNamespace(
                command="shard",
                smoke=True,
                output=None,
                limit_scenarios=1,
            ),
            "explicit custom --output",
        ),
        (
            SimpleNamespace(
                command="shard",
                smoke=False,
                output=None,
                limit_scenarios=1,
            ),
            "only available in smoke mode",
        ),
        (
            SimpleNamespace(command="aggregate", shard_dir=Path("custom")),
            "shard directory must be",
        ),
    ],
)
def test_main_rejects_cli_contract_violations(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, message: str
) -> None:
    monkeypatch.setattr(task_instruction, "parse_args", lambda: args)
    with pytest.raises(ValueError, match=message):
        task_instruction.main()


def test_instruction_hashes_and_imported_evaluator_are_frozen() -> None:
    assert {
        key: hashlib.sha256(value.encode()).hexdigest()
        for key, value in task_instruction.INSTRUCTIONS.items()
    } == task_instruction.INSTRUCTION_SHA256
    assert task_instruction.file_sha256(Path(task_instruction.exp40.__file__)) == (
        task_instruction.EXP40_EVALUATOR_SHA256
    )
    assert (
        json.loads((ROOT / "reports/benchmark-exp40-summary.json").read_text())[
            "experiment"
        ]
        == 40
    )


def test_literal_preregistration_constants() -> None:
    assert task_instruction.DEFINITION_SHA256 == (
        "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
    )
    assert task_instruction.DEVELOPMENT_MANIFEST_SHA256 == (
        "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
    )
    assert task_instruction.EXP36_SUMMARY_SHA256 == (
        "76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f"
    )
    assert task_instruction.EXP38_SUMMARY_SHA256 == (
        "e0e7eaf7ec25600b611cd66780a6dc3d0ca8c374a4f871b15a801716703307bb"
    )
    assert task_instruction.EXP39_SUMMARY_SHA256 == (
        "87bd3d30f350e3cda16a8ac91dfeeec5249b46ace97e42de860362d7bef5a8a8"
    )
    assert task_instruction.EXP40_SUMMARY_SHA256 == (
        "05018ce882e01a5e30d00672ce84db007bdf60992c629ee86485ee7dab55efea"
    )
    assert task_instruction.EXP40_EVALUATOR_SHA256 == (
        "d9c73baff704b5c0e56fd46127f3be07488a1401fa7add285a096478d893f3dd"
    )
    assert task_instruction.PREREGISTRATION_GIT_COMMIT == (
        "281dfb7720ebb569b8dd486dbd8caf31c6927c23"
    )
    assert task_instruction.INSTRUCTION_SHA256 == {
        "push": "ef3724e245f1e3424e91fd211030b7593747d03e0be0c191cddd686533a50a6b",
        "lift": "c89dbd7f6818e34f3d37c485393d1a67464be96efca3e54d06daac4331828e9a",
        "pick": "f345e35bfcc1697c3beb961531cbf9b7dcf0c14d3b96cf84c0b64aa7b55d5f0c",
    }
    assert task_instruction.INSTRUCTION_PERMUTATIONS == (
        ("push", "lift", "pick"),
        ("push", "pick", "lift"),
        ("lift", "push", "pick"),
        ("lift", "pick", "push"),
        ("pick", "push", "lift"),
        ("pick", "lift", "push"),
    )
    assert task_instruction.PHASES == {
        "push_to_zone": ("precontact", "push"),
        "lift_object": ("precontact", "contact", "close", "lift"),
        "pick_place_bin": (
            "precontact",
            "contact",
            "close",
            "lift",
            "transit",
            "release",
            "retreat",
        ),
    }
    np.testing.assert_array_equal(
        task_instruction.ACTION_SCALES,
        [0.25, 0.25, 0.25, 1, 1, 1, 1, 1, 1, 0.5],
    )
    assert task_instruction.LOCK_IDENTITY == "exp41-seed-major-morphology-major-v1"
    assert task_instruction.CLIP_TOLERANCE == 2e-5


def _action_for_loss(loss: float) -> np.ndarray:
    action = np.r_[np.zeros(3), IDENTITY_6D, 0.5].astype(np.float32)
    action[0] = np.float32(np.sqrt(loss * 10) * 0.25)
    return action


def _synthetic_query(key: str, action: np.ndarray, *, repeat: bool = False) -> dict:
    token = tuple(task_instruction.INSTRUCTIONS).index(key) + 1
    input_ids = np.array([[token, token + 10]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    chunk = np.repeat(action.reshape(1, 1, 10), 32, axis=0).astype(np.float32)
    query = {
        "instruction_key": key,
        "instruction_sha256": task_instruction.INSTRUCTION_SHA256[key],
        "input_ids": input_ids.tolist(),
        "input_ids_sha256": task_instruction.canonical_array_sha256(input_ids),
        "input_ids_dtype": input_ids.dtype.str,
        "input_ids_shape": list(input_ids.shape),
        "attention_mask": attention_mask.tolist(),
        "attention_mask_sha256": task_instruction.canonical_array_sha256(
            attention_mask
        ),
        "attention_mask_dtype": attention_mask.dtype.str,
        "attention_mask_shape": list(attention_mask.shape),
        "image_sha256": "1" * 64,
        "state_sha256": "2" * 64,
        "output_chunk": chunk.tolist(),
        "output_sha256": task_instruction.canonical_array_sha256(chunk[None]),
        "output_dtype": np.dtype(np.float32).str,
        "output_shape": [1, 32, 1, 10],
    }
    if repeat:
        query["repeat_correct"] = True
    return query


def _synthetic_condition(
    task: str,
    ordinal: int,
    condition: str,
    correct_loss: float,
) -> dict:
    benefit = 0.25 if condition == "matched" else 0.12
    correct_key = task_instruction.TASK_KEYS[task]
    actions = {
        key: _action_for_loss(
            correct_loss if key == correct_key else correct_loss / (1 - benefit)
        )
        for key in task_instruction.INSTRUCTIONS
    }
    order = task_instruction.instruction_order(ordinal)
    queries = [_synthetic_query(key, actions[key]) for key in order]
    queries.append(_synthetic_query(correct_key, actions[correct_key], repeat=True))
    target = np.r_[np.zeros(3), IDENTITY_6D, 0.5].astype(np.float32)
    return {
        "condition": condition,
        "instruction_order": list(order),
        "queries": queries,
        "repeat_correct_max_abs_difference": 0.0,
        "metrics": task_instruction.score_instruction_predictions(
            task, target, actions
        ),
    }


def _synthetic_runtime() -> dict:
    runtime = {name: None for name in task_instruction.RUNTIME_FIELDS}
    runtime.update(
        device="cpu",
        python="3.12-test",
        torch="test",
        numpy=np.__version__,
        mujoco="test",
        platform="test",
        hostname="test-host",
    )
    return runtime


def _synthetic_checkpoints(exp36: dict, seed: int, morphology: str) -> dict[str, dict]:
    checkpoints = {}
    for condition in task_instruction.CONDITIONS:
        source = "joint" if condition == "joint" else morphology
        checkpoints[condition] = {
            "condition": condition,
            "source_condition": source,
            "model_seed": seed,
            "morphology": morphology,
            "checkpoint_path": f"synthetic/{source}/{seed}/{morphology}",
            "core_sha256": exp36["runs"][f"{source}-seed{seed}"]["core_sha256"],
            "config_sha256": task_instruction.CONFIG_SHA256[morphology],
            "embodiment_sha256": "3" * 64,
            "trainer_sha256": "4" * 64,
            "data_manifest_sha256": task_instruction.DATA_MANIFEST_SHA256[source],
            "exp36_summary_sha256": task_instruction.EXP36_SUMMARY_SHA256,
            "step": 1200,
            "stage": "core",
        }
    return checkpoints


def _synthetic_report(
    manifest: ScenarioManifest,
    exp36: dict,
    seed: int,
    morphology: str,
) -> dict:
    position = task_instruction.REGISTERED_SCHEDULE.index((seed, morphology))
    scenarios = task_instruction.task_instruction_scenarios(
        manifest, morphology, registered=True
    )
    anchors = []
    episodes = []
    ordinal = 0
    condition_order = task_instruction.condition_order(seed, morphology)
    for task_index, task in enumerate(task_instruction.TASKS):
        task_scenarios = [scenario for scenario in scenarios if scenario.task == task]
        for scenario_index, scenario in enumerate(task_scenarios):
            episodes.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "task": task,
                    "steps": 20,
                    "success": True,
                    "terminal": True,
                    "final_phase": "done",
                    "scored_phases": list(task_instruction.PHASES[task]),
                    "anchor_count": len(task_instruction.PHASES[task]),
                }
            )
            for phase_index, phase in enumerate(task_instruction.PHASES[task]):
                matched_loss = (
                    0.10
                    + task_instruction.SEEDS.index(seed) * 0.03
                    + task_instruction.MORPHOLOGIES.index(morphology) * 0.05
                    + task_index * 0.07
                    + scenario_index * 0.01
                    + phase_index * 0.002
                )
                native = np.zeros(6 if morphology == "so101" else 8, dtype=np.float32)
                native[-1] = 0.5
                realizable = native.astype(np.float64)
                target = (
                    np.r_[np.zeros(3), IDENTITY_6D, 0.5]
                    .reshape(1, 10)
                    .astype(np.float32)
                )
                conditions = {}
                for condition in condition_order:
                    loss = matched_loss * (1.2 if condition == "joint" else 1.0)
                    conditions[condition] = _synthetic_condition(
                        task, ordinal, condition, loss
                    )
                anchors.append(
                    {
                        "ordinal": ordinal,
                        "scenario_id": scenario.scenario_id,
                        "scenario_seed": scenario.seed,
                        "morphology": morphology,
                        "task": task,
                        "phase": phase,
                        "phase_tick": 1,
                        "image_sha256": "1" * 64,
                        "state_sha256": "2" * 64,
                        "expert_native": native.tolist(),
                        "expert_native_sha256": task_instruction.canonical_array_sha256(
                            native
                        ),
                        "expert_target": target.tolist(),
                        "expert_target_sha256": task_instruction.canonical_array_sha256(
                            target
                        ),
                        "decoded_native": native.tolist(),
                        "decoded_native_sha256": task_instruction.canonical_array_sha256(
                            native
                        ),
                        "realizable_native": realizable.tolist(),
                        "realizable_native_sha256": task_instruction.canonical_array_sha256(
                            realizable
                        ),
                        "realized_canonical": target.tolist(),
                        "realized_canonical_sha256": task_instruction.canonical_array_sha256(
                            target
                        ),
                        "round_trip": {
                            "clipped": False,
                            "clipping_error_native": 0.0,
                            "translation_error_m": 0.0,
                            "rotation_error_deg": 0.0,
                            "gripper_error_native": 0.0,
                        },
                        "conditions": conditions,
                    }
                )
                ordinal += 1
    runtime = _synthetic_runtime()
    runtime_signature = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "format_version": 1,
        "experiment": 41,
        "mode": "task_instruction_shard",
        "registered": True,
        "status": "completed",
        "complete": True,
        "contract": task_instruction._contract(True, None),
        "identity": {
            "model_seed": seed,
            "morphology": morphology,
            "checkpoints": _synthetic_checkpoints(exp36, seed, morphology),
        },
        "expected_anchor_ids": [
            f"{scenario.scenario_id}/{phase}"
            for scenario in scenarios
            for phase in task_instruction.PHASES[scenario.task]
        ],
        "provenance": {
            "git_dirty": False,
            "git_commit": "synthetic-exp41-commit",
            "evaluator_sha256": task_instruction.file_sha256(
                Path(task_instruction.__file__)
            ),
            "runtime": runtime,
            "runtime_signature_sha256": runtime_signature,
        },
        "run": {
            "started_utc": f"2026-01-01T00:{position * 2:02d}:00Z",
            "finished_utc": f"2026-01-01T00:{position * 2 + 1:02d}:00Z",
            "device": "cpu",
            "lock_identity": task_instruction.LOCK_IDENTITY,
            "lock_path": str(task_instruction.REGISTERED_LOCK_PATH),
            "schedule_identity": task_instruction.SCHEDULE_IDENTITY,
            "schedule_position": position,
            "condition_order": list(condition_order),
        },
        "episodes": episodes,
        "anchors": anchors,
        "outcome_chain_sha256": task_instruction._outcome_chain(anchors),
        "infrastructure_errors": [],
    }
    report["aggregate"] = task_instruction._shard_aggregate(anchors, report)
    return report


def _synthetic_sources(
    tmp_path: Path,
) -> tuple[dict[str, Path], ScenarioManifest, dict]:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    exp36 = json.loads((ROOT / "reports/benchmark-exp36-summary.json").read_text())
    paths = {}
    for seed, morphology in task_instruction.REGISTERED_SCHEDULE:
        label = f"seed{seed}/{morphology}"
        path = tmp_path / task_instruction.registered_shard_filename(seed, morphology)
        path.write_text(
            json.dumps(_synthetic_report(manifest, exp36, seed, morphology))
        )
        paths[label] = path
    return paths, manifest, exp36


@pytest.fixture(scope="module")
def synthetic_bundle(tmp_path_factory):
    return _synthetic_sources(tmp_path_factory.mktemp("exp41-synthetic"))


def _independent_seed_correct_mean(
    paths: dict[str, Path], seed: int, condition: str
) -> float:
    morphology_means = []
    for morphology in task_instruction.MORPHOLOGIES:
        report = json.loads(paths[f"seed{seed}/{morphology}"].read_text())
        task_means = []
        for task in task_instruction.TASKS:
            scenario_means = []
            scenario_ids = {
                anchor["scenario_id"]
                for anchor in report["anchors"]
                if anchor["task"] == task
            }
            for scenario_id in scenario_ids:
                phase_losses = [
                    anchor["conditions"][condition]["metrics"]["correct"]["loss"]
                    for anchor in report["anchors"]
                    if anchor["scenario_id"] == scenario_id
                ]
                scenario_means.append(sum(phase_losses) / len(phase_losses))
            task_means.append(sum(scenario_means) / len(scenario_means))
        morphology_means.append(sum(task_means) / len(task_means))
    return sum(morphology_means) / len(morphology_means)


def test_six_synthetic_shard_aggregate_hierarchy_sources_and_classification(
    synthetic_bundle,
) -> None:
    paths, manifest, exp36 = synthetic_bundle
    summary = task_instruction.aggregate_registered_shards(
        paths, manifest, exp36_summary=exp36
    )
    expected = _independent_seed_correct_mean(paths, 36001, "matched")
    actual = summary["metrics_by_condition_seed"]["matched/seed36001"]["correct_loss"]
    assert actual == pytest.approx(expected)
    flat = np.mean(
        [
            anchor["conditions"]["matched"]["metrics"]["correct"]["loss"]
            for path in paths.values()
            if "seed36001" in path.name
            for anchor in json.loads(path.read_text())["anchors"]
        ]
    )
    assert actual != pytest.approx(flat)
    assert summary["metrics_by_condition"]["joint"]["correct_loss"] == pytest.approx(
        1.2 * summary["metrics_by_condition"]["matched"]["correct_loss"],
        rel=2e-6,
    )
    assert summary["delivery_gate_passed"] is True
    assert summary["support_by_condition"]["matched"]["supported"] is True
    assert summary["support_by_condition"]["joint"]["supported"] is True
    assert summary["joint_specific_prompt_effect_deficit"]["supported"] is True
    assert summary["source_shard_sha256"] == {
        key: task_instruction.file_sha256(path) for key, path in sorted(paths.items())
    }
    assert len(summary["metrics_by_condition_seed_morphology_task_phase"]) == 156


def test_aggregate_rejects_source_key_swap_and_duplicate_identity(
    synthetic_bundle,
) -> None:
    paths, manifest, exp36 = synthetic_bundle
    missing = dict(paths)
    missing.pop("seed36003/panda")
    with pytest.raises(ValueError, match="exact paths"):
        task_instruction.aggregate_registered_shards(
            missing, manifest, exp36_summary=exp36
        )
    swapped = dict(paths)
    swapped["seed36001/so101"], swapped["seed36001/panda"] = (
        swapped["seed36001/panda"],
        swapped["seed36001/so101"],
    )
    with pytest.raises(ValueError, match="misattributed"):
        task_instruction.aggregate_registered_shards(
            swapped, manifest, exp36_summary=exp36
        )
    path = paths["seed36001/panda"]
    original = path.read_text()
    try:
        report = json.loads(original)
        report["identity"]["model_seed"] = 36002
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="misattributed|duplicate"):
            task_instruction.aggregate_registered_shards(
                paths, manifest, exp36_summary=exp36
            )
    finally:
        path.write_text(original)


def test_aggregate_rejects_commit_runtime_overlap_and_failed_delivery(
    synthetic_bundle,
) -> None:
    paths, manifest, exp36 = synthetic_bundle
    path = paths["seed36001/panda"]
    original = path.read_text()

    report = json.loads(original)
    report["provenance"]["git_commit"] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="commit/runtime/lock/sequential"):
        task_instruction.aggregate_registered_shards(
            paths, manifest, exp36_summary=exp36
        )

    report = json.loads(original)
    report["provenance"]["runtime"]["hostname"] = "different"
    report["provenance"]["runtime_signature_sha256"] = hashlib.sha256(
        json.dumps(
            report["provenance"]["runtime"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="commit/runtime/lock/sequential"):
        task_instruction.aggregate_registered_shards(
            paths, manifest, exp36_summary=exp36
        )

    report = json.loads(original)
    report["run"]["started_utc"] = "2026-01-01T00:00:30Z"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="commit/runtime/lock/sequential"):
        task_instruction.aggregate_registered_shards(
            paths, manifest, exp36_summary=exp36
        )

    report = json.loads(original)
    for anchor in report["anchors"][:2]:
        realizable = np.asarray(anchor["realizable_native"], dtype=np.float64)
        realizable[0] = 1e-4
        anchor["realizable_native"] = realizable.tolist()
        anchor["realizable_native_sha256"] = task_instruction.canonical_array_sha256(
            realizable
        )
        anchor["round_trip"]["clipped"] = True
        anchor["round_trip"]["clipping_error_native"] = 1e-4
    report["outcome_chain_sha256"] = task_instruction._outcome_chain(report["anchors"])
    report["aggregate"] = task_instruction._shard_aggregate(report["anchors"], report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="provenance or delivery"):
        task_instruction.aggregate_registered_shards(
            paths, manifest, exp36_summary=exp36
        )
    path.write_text(original)


@pytest.mark.parametrize(
    "mutation",
    [
        "dirty",
        "timestamp",
        "final_split",
        "expected_order",
        "outcome_chain",
        "aggregate",
    ],
)
def test_full_report_validation_rejects_contract_mutations(
    synthetic_bundle, mutation: str
) -> None:
    paths, manifest, exp36 = synthetic_bundle
    report = json.loads(paths["seed36001/so101"].read_text())
    if mutation == "dirty":
        report["provenance"]["git_dirty"] = True
    elif mutation == "timestamp":
        report["run"]["finished_utc"] = "2025-12-31T23:59:00Z"
    elif mutation == "final_split":
        report["contract"]["final_split_loaded"] = True
    elif mutation == "expected_order":
        report["expected_anchor_ids"] = list(reversed(report["expected_anchor_ids"]))
    elif mutation == "outcome_chain":
        report["outcome_chain_sha256"] = "0" * 64
    else:
        report["aggregate"]["anchors"] += 1
    with pytest.raises(ValueError):
        task_instruction.validate_shard_report(
            report, manifest, exp36, require_registered=True
        )


def test_nonresumable_evaluator_rejects_existing_output(synthetic_bundle) -> None:
    paths, _manifest, _exp36 = synthetic_bundle
    output = paths["seed36001/so101"]
    with pytest.raises(FileExistsError, match="non-resumable"):
        task_instruction.evaluate_task_instruction_shard(
            None,
            None,
            {},
            output,
            seed=36001,
            morphology="so101",
            device="cpu",
            registered=False,
        )
