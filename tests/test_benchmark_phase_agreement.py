from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodi.sim.benchmark import BenchmarkDefinition, ScenarioManifest
import embodi.sim.benchmark_phase_agreement as phase_agreement


ROOT = Path(__file__).parents[1]
IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _scenario(task: str = "push_to_zone", morphology: str = "panda") -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=f"development-{morphology}-{task}-0028",
        seed=42,
        morphology=morphology,
        task=task,
        factors={},
    )


class FakeStatus:
    def __init__(self, terminal: bool, success: bool) -> None:
        self.terminal = terminal
        self.success = success


class FakeEnv:
    def __init__(self, task: str = "push_to_zone", morphology: str = "panda") -> None:
        self.task = task
        self.morphology = SimpleNamespace(
            id=morphology, native_action_dim=6 if morphology == "so101" else 8
        )
        self.success = False
        self.closed = False
        self.applied: list[np.ndarray] = []
        self.native_state = np.r_[
            np.zeros(self.morphology.native_action_dim - 1), 0.5
        ].astype(np.float32)
        self.state_value = np.r_[np.zeros(3), IDENTITY_6D, 0.5].reshape(1, 10).astype(
            np.float32
        )

    def render_top(self) -> np.ndarray:
        return np.full((480, 640, 3), len(self.applied), dtype=np.uint8)

    def canonical_state(self) -> np.ndarray:
        return self.state_value.copy()

    def state(self) -> np.ndarray:
        return self.native_state.copy()

    def canonical_arm_action(self, native: np.ndarray) -> np.ndarray:
        value = np.asarray(native, dtype=np.float32)
        return np.r_[value[:3], IDENTITY_6D, value[-1]].reshape(1, 10).astype(np.float32)

    def native_action_from_canonical(
        self, canonical: np.ndarray, *, iterations: int
    ) -> np.ndarray:
        assert iterations == 20
        value = np.asarray(canonical, dtype=np.float32).reshape(10)
        return np.r_[
            value[:3],
            np.zeros(self.morphology.native_action_dim - 4),
            value[-1],
        ].astype(np.float32)

    @staticmethod
    def native_to_qpos(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)

    @staticmethod
    def qpos_to_native(value: np.ndarray) -> np.ndarray:
        return np.asarray(value)

    def apply_action(self, value: np.ndarray) -> None:
        self.applied.append(np.asarray(value).copy())
        self.native_state = np.asarray(value, dtype=np.float32).copy()
        self.state_value[0, 0] = len(self.applied) / 100

    def step_control_period(self, frequency: float) -> FakeStatus:
        assert frequency == 30.0
        return FakeStatus(self.success, self.success)

    def close(self) -> None:
        self.closed = True


class FakeExpert:
    frequency = 30.0
    durations: dict[str, int] = {}

    def __init__(self, env: FakeEnv) -> None:
        self.env = env
        self.phases = ("open", *phase_agreement.PHASES[env.task], "done")
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
        current = self.phase
        if current == "open":
            self.index = 1
            self.phase = self.phases[self.index]
            self.phase_tick = 1
        else:
            duration = self.durations.get(current, 1)
            if self.phase_tick >= duration:
                self.index += 1
                self.phase = self.phases[self.index]
                self.phase_tick = 1
            else:
                self.phase_tick += 1
        if self.phase == "done":
            self.env.success = True
            value = self.env.state()
        else:
            value = np.r_[
                np.full(3, self.action_calls / 100),
                np.zeros(self.env.morphology.native_action_dim - 4),
                0.5,
            ].astype(np.float32)
        self.total_tick += 1
        return value


class MutatingExpert(FakeExpert):
    def action(self) -> np.ndarray:
        value = super().action()
        if self.action_calls > 1:
            self.env.state_value[0, 1] += 1
        return value


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _images, instructions, _state, **kwargs):
        self.calls += 1
        key = next(
            key
            for key, value in phase_agreement.INSTRUCTIONS.items()
            if value == instructions[0]
        )
        token = tuple(phase_agreement.INSTRUCTIONS).index(key) + 1
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

    def __init__(self, value: float = 0.0, repeat_delta: float = 0.0) -> None:
        self.value = value
        self.repeat_delta = repeat_delta
        self.reset_calls = 0
        self.predict_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_canonical_action_chunk(self, _batch):
        self.predict_calls += 1
        output = torch.zeros((1, 32, 1, 10), dtype=torch.float32)
        output[..., 0] = self.value
        output[..., 3:9] = torch.from_numpy(IDENTITY_6D)
        output[..., 9] = 0.5
        if self.predict_calls % 2 == 0:
            output[0, 0, 0, 0] += self.repeat_delta
        return output


def _anchor(endpoint: str = "first", ordinal: int = 0) -> phase_agreement.Anchor:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = np.r_[np.zeros(3), IDENTITY_6D, 0.5].reshape(1, 10).astype(np.float32)
    target = state.copy()
    evidence = {
        "ordinal": ordinal,
        "scenario_id": "development-panda-push_to_zone-0028",
        "scenario_seed": 42,
        "morphology": "panda",
        "task": "push_to_zone",
        "phase": "precontact",
        "endpoint": endpoint,
        "phase_tick": 1,
        "image_sha256": phase_agreement.canonical_array_sha256(frame),
        "state_sha256": phase_agreement.canonical_array_sha256(state),
        "expert_target_sha256": phase_agreement.canonical_array_sha256(target),
    }
    return phase_agreement.Anchor(
        ordinal,
        evidence["scenario_id"],
        42,
        "panda",
        "push_to_zone",
        "precontact",
        endpoint,
        1,
        frame,
        state,
        target,
        evidence,
    )


def _frozen_sources() -> dict[str, Path]:
    return {
        "exp36_summary": ROOT / "reports/benchmark-exp36-summary.json",
        "exp41_summary": ROOT / "reports/benchmark-exp41-summary.json",
    }


def _manifest() -> ScenarioManifest:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    return ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )


def test_literal_frozen_constants_slice_order_counts_and_factor_coverage() -> None:
    assert phase_agreement.DEFINITION_SHA256 == (
        "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
    )
    assert phase_agreement.DEVELOPMENT_MANIFEST_SHA256 == (
        "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
    )
    assert phase_agreement.EXP36_SUMMARY_SHA256 == (
        "76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f"
    )
    assert phase_agreement.EXP41_SUMMARY_SHA256 == (
        "d8195516eb6c84d90da31e74459b33779aab3382000650df8e9b941f35cd217f"
    )
    assert phase_agreement.EXP41_EVALUATOR_SHA256 == (
        "2ca9ae5b48782b4b378882accf886dd9b65bd1c7ed3905bb20dfc3c859f9d62c"
    )
    assert phase_agreement.EXPERT_SHA256 == (
        "7fb869a10756864530889f809a8305ee1758bdc51eaeeeda0ff2f451f3b7f8e9"
    )
    assert phase_agreement.PREREGISTRATION_GIT_COMMIT == (
        "38a868b6dee792a220f8fd4c6a786eb00f0c1da0"
    )
    assert phase_agreement.PHASES == {
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
    assert tuple(phase_agreement.ACTION_SCALES) == (
        0.25,
        0.25,
        0.25,
        1,
        1,
        1,
        1,
        1,
        1,
        0.5,
    )
    for morphology in phase_agreement.MORPHOLOGIES:
        scenarios = phase_agreement.phase_agreement_scenarios(
            _manifest(), morphology, registered=True
        )
        assert len(scenarios) == 21
        assert scenarios[0].scenario_id == (
            f"development-{morphology}-push_to_zone-0028"
        )
        assert scenarios[-1].scenario_id == (
            f"development-{morphology}-pick_place_bin-0034"
        )
        assert sum(len(phase_agreement.PHASES[value.task]) for value in scenarios) == 91
        assert (
            sum(len(phase_agreement.PHASES[value.task]) * 2 for value in scenarios)
            == 182
        )
    assert phase_agreement.condition_order(0, 36001, "so101") == ("joint", "matched")
    assert phase_agreement.condition_order(0, 36001, "panda") == ("matched", "joint")
    assert phase_agreement.condition_order(1, 36002, "so101") == ("joint", "matched")


def test_one_action_duplicates_and_multi_action_captures_first_and_last() -> None:
    one_env = FakeEnv()
    anchors, episode = phase_agreement.capture_expert_episode(
        one_env, _scenario(), starting_ordinal=0, expert_factory=FakeExpert
    )
    assert [(value.phase, value.endpoint) for value in anchors] == [
        ("precontact", "first"),
        ("precontact", "last"),
        ("push", "first"),
        ("push", "last"),
    ]
    assert [value.phase_tick for value in anchors] == [1, 1, 1, 1]
    assert anchors[0].evidence["expert_native"] == anchors[1].evidence["expert_native"]
    assert anchors[-1].evidence["expert_native"] == anchors[-1].evidence[
        "pre_call_native_state"
    ]
    assert episode["phase_endpoint_ticks"] == {"precontact": [1, 1], "push": [1, 1]}
    np.testing.assert_array_equal(
        one_env.applied[-1], np.asarray(anchors[-1].evidence["expert_native"], dtype=np.float32)
    )

    class MultiActionExpert(FakeExpert):
        durations = {"precontact": 3, "push": 2}

    multi_env = FakeEnv()
    anchors, _ = phase_agreement.capture_expert_episode(
        multi_env, _scenario(), starting_ordinal=7, expert_factory=MultiActionExpert
    )
    assert [value.ordinal for value in anchors] == [7, 8, 9, 10]
    assert [value.phase_tick for value in anchors] == [1, 3, 1, 2]
    assert anchors[0].evidence["expert_native"] != anchors[1].evidence["expert_native"]
    assert len(multi_env.applied) == 6
    for applied, expected_call in zip(multi_env.applied, range(1, 7), strict=True):
        expected = 5 / 100 if expected_call == 6 else expected_call / 100
        assert applied[0] == pytest.approx(expected)


def test_capture_checks_state_immutability_before_conversion() -> None:
    with pytest.raises(ValueError, match="changed canonical state"):
        phase_agreement.capture_expert_episode(
            FakeEnv(), _scenario(), starting_ordinal=0, expert_factory=MutatingExpert
        )


def test_round_trip_evidence_uses_float64_recomputation() -> None:
    class RoundingEnv(FakeEnv):
        def native_action_from_canonical(
            self, canonical: np.ndarray, *, iterations: int
        ) -> np.ndarray:
            value = super().native_action_from_canonical(canonical, iterations=iterations)
            value[0] = np.nextafter(value[0], np.float32(np.inf))
            value[-1] = np.nextafter(value[-1], np.float32(np.inf))
            return value

    env = RoundingEnv()
    scenario = _scenario()
    frame = env.render_top()
    state = env.canonical_state()
    native_state = env.state()
    requested = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0, 0.4], dtype=np.float32)
    target, evidence = phase_agreement._capture_call_evidence(
        env,
        scenario,
        "precontact",
        1,
        1,
        1,
        frame,
        state,
        native_state,
        requested,
        "push",
        1,
        2,
    )
    decoded = np.asarray(evidence["decoded_native"], dtype=np.float32)
    realized = np.asarray(evidence["realized_canonical"], dtype=np.float32)
    assert evidence["round_trip"]["translation_error_m"] == float(
        np.linalg.norm(
            np.asarray(target[0, :3], dtype=np.float64)
            - np.asarray(realized[0, :3], dtype=np.float64)
        )
    )
    assert evidence["round_trip"]["gripper_error_native"] == float(
        abs(np.float64(requested[-1]) - np.float64(decoded[-1]))
    )


def test_capture_rejects_timeout_failed_missing_and_reordered_runtime_phases() -> None:
    class TimeoutExpert(FakeExpert):
        durations = {"precontact": 100}

    with pytest.raises(RuntimeError, match="runtime phase sequence"):
        phase_agreement.capture_expert_episode(
            FakeEnv(),
            _scenario(),
            starting_ordinal=0,
            expert_factory=TimeoutExpert,
            maximum_steps=2,
        )

    class FailedExpert(FakeExpert):
        @property
        def terminal(self) -> bool:
            return self.phase == "failed"

        def action(self) -> np.ndarray:
            self.phase = "failed"
            self.phase_tick = 1
            self.total_tick += 1
            return self.env.state()

    with pytest.raises(RuntimeError, match="failed"):
        phase_agreement.capture_expert_episode(
            FakeEnv(), _scenario(), starting_ordinal=0, expert_factory=FailedExpert
        )

    class MissingExpert(FakeExpert):
        def __init__(self, env: FakeEnv) -> None:
            super().__init__(env)
            self.phases = ("open", "push", "done")

    with pytest.raises(RuntimeError, match="runtime phase sequence"):
        phase_agreement.capture_expert_episode(
            FakeEnv(), _scenario(), starting_ordinal=0, expert_factory=MissingExpert
        )

    class ReorderedExpert(FakeExpert):
        def __init__(self, env: FakeEnv) -> None:
            super().__init__(env)
            self.phases = ("open", "push", "precontact", "done")

    with pytest.raises(RuntimeError, match="runtime phase sequence"):
        phase_agreement.capture_expert_episode(
            FakeEnv(), _scenario(), starting_ordinal=0, expert_factory=ReorderedExpert
        )


@pytest.mark.parametrize("malformed", ["image", "state", "native_state", "action"])
def test_capture_rejects_malformed_arrays_and_closes_environment(malformed: str) -> None:
    class MalformedEnv(FakeEnv):
        def render_top(self) -> np.ndarray:
            if malformed == "image":
                return np.zeros((1, 1, 3), dtype=np.uint8)
            return super().render_top()

        def canonical_state(self) -> np.ndarray:
            if malformed == "state":
                return np.zeros((9,), dtype=np.float32)
            return super().canonical_state()

        def state(self) -> np.ndarray:
            if malformed == "native_state":
                return np.zeros((2,), dtype=np.float32)
            return super().state()

    class MalformedActionExpert(FakeExpert):
        def action(self) -> np.ndarray:
            value = super().action()
            return np.zeros((2,), dtype=np.float32) if malformed == "action" else value

    created: list[MalformedEnv] = []

    def factory(morphology, task, _scenario_value, **_kwargs):
        env = MalformedEnv(task, morphology)
        created.append(env)
        return env

    with pytest.raises(ValueError, match="invalid|must be"):
        phase_agreement.capture_expert_anchors(
            [
                phase_agreement.phase_agreement_scenarios(
                    _manifest(), "panda", registered=True
                )[0]
            ],
            "panda",
            env_factory=factory,
            expert_factory=MalformedActionExpert,
        )
    assert created[0].closed is True


def test_policy_immediate_repeat_order_tokenizer_output_and_first_scoring() -> None:
    policy = FakePolicy(value=0.1, repeat_delta=5e-6)
    processor = FakeProcessor()
    result = phase_agreement.evaluate_anchor_condition(
        _anchor(), 36001, "joint", policy, processor, "cpu"
    )
    assert policy.reset_calls == policy.predict_calls == processor.calls == 2
    assert result["queries"][1]["repeat"] is True
    assert result["repeat_max_abs_difference"] == pytest.approx(5e-6, abs=1e-8)
    first = np.asarray(result["queries"][0]["output_chunk"], dtype=np.float32)
    repeat = np.asarray(result["queries"][1]["output_chunk"], dtype=np.float32)
    assert not np.array_equal(first, repeat)
    assert result["metrics"] == phase_agreement.action_metrics(
        first[0, 0], _anchor().expert_target
    )
    assert result["queries"][0]["input_ids_sha256"] == (
        phase_agreement.canonical_array_sha256(np.array([[1, 11]], dtype=np.int64))
    )
    assert result["queries"][0]["output_shape"] == [32, 1, 10]
    chunk = np.asarray(result["queries"][0]["output_chunk"], dtype=np.float32)
    assert result["queries"][0]["output_sha256"] == (
        phase_agreement.canonical_array_sha256(chunk)
    )

    changed_evidence = _anchor()
    changed_evidence.evidence["image_sha256"] = "f" * 64
    changed_evidence.evidence["state_sha256"] = "e" * 64
    actual = phase_agreement.evaluate_anchor_condition(
        changed_evidence, 36001, "joint", FakePolicy(), FakeProcessor(), "cpu"
    )
    assert actual["queries"][0]["image_sha256"] == phase_agreement.canonical_array_sha256(
        changed_evidence.frame
    )
    assert actual["queries"][0]["state_sha256"] == phase_agreement.canonical_array_sha256(
        changed_evidence.state
    )


def test_policy_guards_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="determinism"):
        phase_agreement.evaluate_anchor_condition(
            _anchor(), 36001, "joint", FakePolicy(repeat_delta=2e-5), FakeProcessor(), "cpu"
        )

    class WrongShapePolicy(FakePolicy):
        def predict_canonical_action_chunk(self, _batch):
            return torch.zeros((1, 31, 1, 10), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"shape \[1,32,1,10\]"):
        phase_agreement.evaluate_anchor_condition(
            _anchor(), 36001, "joint", WrongShapePolicy(), FakeProcessor(), "cpu"
        )

    class NonfinitePolicy(FakePolicy):
        def predict_canonical_action_chunk(self, _batch):
            output = super().predict_canonical_action_chunk(_batch)
            output[0, 0, 0, 0] = torch.nan
            return output

    with pytest.raises(ValueError, match="finite"):
        phase_agreement.evaluate_anchor_condition(
            _anchor(), 36001, "joint", NonfinitePolicy(), FakeProcessor(), "cpu"
        )

    class DriftProcessor(FakeProcessor):
        def __call__(self, images, instructions, state, **kwargs):
            value = super().__call__(images, instructions, state, **kwargs)
            if self.calls == 2:
                value["attention_mask"][0, 0] = 0
            return value

    with pytest.raises(RuntimeError, match="input differs"):
        phase_agreement.evaluate_anchor_condition(
            _anchor(), 36001, "joint", FakePolicy(), DriftProcessor(), "cpu"
        )


def test_cross_arm_tokenizer_drift_is_rejected_even_when_each_repeat_is_stable() -> None:
    class CrossArmDriftProcessor(FakeProcessor):
        def __call__(self, images, instructions, state, **kwargs):
            value = super().__call__(images, instructions, state, **kwargs)
            if self.calls > 2:
                value["input_ids"] += 1
            return value

    processor = CrossArmDriftProcessor()
    anchor = _anchor()
    results = {}
    for seed in phase_agreement.SEEDS:
        results[str(seed)] = {
            condition: phase_agreement.evaluate_anchor_condition(
                anchor, seed, condition, FakePolicy(), processor, "cpu"
            )
            for condition in phase_agreement.CONDITIONS
        }
    with pytest.raises(RuntimeError, match="across seed-condition arms"):
        phase_agreement._require_stable_anchor_inputs(results)


def test_normalized_physical_channel_scoring_clipping_and_degeneration() -> None:
    expert = np.r_[np.zeros(3), IDENTITY_6D, 0.5]
    prediction = expert.copy()
    prediction[:3] = [0.25, 0.5, 0.75]
    prediction[9] = 2
    metrics = phase_agreement.action_metrics(prediction, expert)
    assert metrics["channel_losses"] == pytest.approx([1, 4, 9, 0, 0, 0, 0, 0, 0, 9])
    assert metrics["normalized_loss"] == pytest.approx(2.3)
    assert metrics["translation_norm_error_m"] == pytest.approx(np.sqrt(0.875))
    assert metrics["effective_gripper_error"] == 0.5
    degenerate = prediction.copy()
    degenerate[3:9] = 0
    metrics = phase_agreement.action_metrics(degenerate, expert)
    assert metrics["degenerate_rotation"] is True
    assert metrics["rotation_geodesic_error_deg"] == 180


def _metric_record(
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
            "normalized_loss": value,
            "translation_norm_error_m": value,
            "rotation_geodesic_error_deg": value,
            "effective_gripper_error": value,
            "degenerate_rotation": False,
            "degenerate_rotation_rate": 0.0,
            "channel_losses": [value] * 10,
        },
    }


def test_hierarchy_weights_endpoint_phase_scenario_and_distinguishes_joint_matched() -> None:
    records = []
    for condition, offset in (("joint", 1.0), ("matched", 0.0)):
        records.extend(
            [
                _metric_record(0 + offset, condition, endpoint="first"),
                _metric_record(2 + offset, condition, endpoint="last"),
                _metric_record(9 + offset, condition, scenario="b", endpoint="first"),
                _metric_record(9 + offset, condition, scenario="b", endpoint="last"),
            ]
        )
    metrics = phase_agreement.aggregate_metrics(records)
    assert metrics["overall"]["normalized_loss"]["matched"] == 5
    assert metrics["overall"]["normalized_loss"]["joint"] == 6
    assert metrics["overall"]["normalized_loss"]["difference"] == 1
    assert metrics["by_endpoint"]["first"]["normalized_loss"]["matched"] == 4.5
    assert metrics["by_endpoint"]["last"]["normalized_loss"]["matched"] == 5.5
    assert metrics["by_channel"]["0"]["joint"] == 6


def test_unequal_full_hierarchy_has_exact_independent_means_and_task_phase_cells() -> None:
    records = []
    matched_values: dict[tuple[int, str, str, str, str], float] = {}
    for seed_index, seed in enumerate(phase_agreement.SEEDS):
        for morphology_index, morphology in enumerate(phase_agreement.MORPHOLOGIES):
            for task_index, task in enumerate(phase_agreement.TASKS):
                for phase_index, phase in enumerate(phase_agreement.PHASES[task]):
                    for endpoint_index, endpoint in enumerate(phase_agreement.ENDPOINTS):
                        matched = (
                            1
                            + seed_index * 100
                            + morphology_index * 20
                            + task_index * 5
                            + phase_index
                            + endpoint_index * 0.25
                        )
                        matched_values[(seed, morphology, task, phase, endpoint)] = matched
                        records.append(
                            _metric_record(
                                matched,
                                "matched",
                                seed=seed,
                                morphology=morphology,
                                task=task,
                                scenario=f"{morphology}/{task}",
                                phase=phase,
                                endpoint=endpoint,
                            )
                        )
                        records.append(
                            _metric_record(
                                matched * 1.5,
                                "joint",
                                seed=seed,
                                morphology=morphology,
                                task=task,
                                scenario=f"{morphology}/{task}",
                                phase=phase,
                                endpoint=endpoint,
                            )
                        )
    independent_seed_means = []
    for seed in phase_agreement.SEEDS:
        morphology_means = []
        for morphology in phase_agreement.MORPHOLOGIES:
            task_means = []
            for task in phase_agreement.TASKS:
                phase_means = []
                for phase in phase_agreement.PHASES[task]:
                    phase_means.append(
                        np.mean(
                            [
                                matched_values[(seed, morphology, task, phase, endpoint)]
                                for endpoint in phase_agreement.ENDPOINTS
                            ]
                        )
                    )
                task_means.append(np.mean(phase_means))
            morphology_means.append(np.mean(task_means))
        independent_seed_means.append(np.mean(morphology_means))
    expected_matched = float(np.mean(independent_seed_means))
    flat_matched = float(np.mean(list(matched_values.values())))
    metrics = phase_agreement.aggregate_metrics(records)
    overall = metrics["overall"]["normalized_loss"]
    assert overall["matched"] == pytest.approx(expected_matched)
    assert overall["matched"] != pytest.approx(flat_matched)
    assert overall["joint"] == pytest.approx(expected_matched * 1.5)
    assert overall["ratio"] == pytest.approx(1.5)
    phase_key = "pick_place_bin/retreat"
    expected_phase = np.mean(
        [
            matched_values[(seed, morphology, "pick_place_bin", "retreat", endpoint)]
            for seed in phase_agreement.SEEDS
            for morphology in phase_agreement.MORPHOLOGIES
            for endpoint in phase_agreement.ENDPOINTS
        ]
    )
    assert metrics["by_task_phase"][phase_key]["normalized_loss"][
        "matched"
    ] == pytest.approx(expected_phase)
    assert metrics["by_task_phase"][phase_key]["normalized_loss"][
        "ratio"
    ] == pytest.approx(1.5)


def _delivery_anchor(count: int = 8) -> dict:
    native = np.zeros(count, dtype=np.float32)
    target = np.r_[np.zeros(3), IDENTITY_6D, 0.5].reshape(1, 10).astype(np.float32)
    return {
        "expert_native": native.tolist(),
        "decoded_native": native.tolist(),
        "realizable_native": native.astype(np.float64).tolist(),
        "expert_target": target.tolist(),
        "realized_canonical": target.tolist(),
    }


def test_delivery_thresholds_percentile_clipping_and_expert_degeneration() -> None:
    anchors = [_delivery_anchor() for _ in range(100)]
    for index, anchor in enumerate(anchors):
        anchor["realized_canonical"][0][0] = index / 100_000
    gate = phase_agreement.delivery_gate(anchors)
    assert gate["measurements"]["translation_p95_m"] == pytest.approx(
        np.percentile([index / 100_000 for index in range(100)], 95)
    )
    assert gate["passed"] is True
    anchors[0]["realizable_native"][0] = 1e-4
    assert phase_agreement.delivery_gate(anchors)["passed"] is True
    anchors[1]["realizable_native"][0] = 1e-4
    assert phase_agreement.delivery_gate(anchors)["passed"] is False
    exact = _delivery_anchor()
    exact["expert_native"][-1] = 1.0
    assert phase_agreement.delivery_gate([exact])["passed"] is True
    exact["expert_native"][-1] = 1.0001
    assert phase_agreement.delivery_gate([exact])["passed"] is False
    degenerate = _delivery_anchor()
    degenerate["expert_target"][0][3:9] = [0] * 6
    assert phase_agreement.delivery_gate([degenerate])["passed"] is False
    assert phase_agreement.delivery_gate([])["passed"] is False
    clipped, clipping_error = phase_agreement._clipping_from_arrays(
        np.zeros(1), np.array([2e-5])
    )
    assert clipped is False and clipping_error == 2e-5

    translation = _delivery_anchor()
    translation["realized_canonical"][0][0] = float(
        np.nextafter(np.float32(0.001), np.float32(0.0))
    )
    assert phase_agreement.delivery_gate([translation])["passed"] is True
    translation["realized_canonical"][0][0] = float(
        np.nextafter(np.float32(0.001), np.float32(np.inf))
    )
    assert (
        phase_agreement.delivery_gate([translation])["checks"][
            "translation_p95_at_most_1mm"
        ]
        is False
    )

    rotation = _delivery_anchor()
    angle = np.deg2rad(1.0)
    rotation["realized_canonical"][0][3:9] = [
        np.cos(angle),
        np.sin(angle),
        0,
        -np.sin(angle),
        np.cos(angle),
        0,
    ]
    assert phase_agreement.delivery_gate([rotation])["passed"] is True
    angle = np.deg2rad(1.001)
    rotation["realized_canonical"][0][3:9] = [
        np.cos(angle),
        np.sin(angle),
        0,
        -np.sin(angle),
        np.cos(angle),
        0,
    ]
    assert (
        phase_agreement.delivery_gate([rotation])["checks"][
            "rotation_p95_at_most_1deg"
        ]
        is False
    )


def _classification_metrics() -> dict:
    def cell(ratio: float = 1.10, difference: float = 0.01) -> dict:
        return {
            "normalized_loss": {"ratio": ratio, "difference": difference},
            "effective_gripper_error": {"ratio": ratio, "difference": difference},
        }

    return {
        "overall": cell(),
        "by_seed": {str(seed): cell() for seed in phase_agreement.SEEDS},
        "by_endpoint": {endpoint: cell() for endpoint in phase_agreement.ENDPOINTS},
        "by_task": {task: cell() for task in phase_agreement.TASKS},
        "by_morphology": {
            morphology: cell() for morphology in phase_agreement.MORPHOLOGIES
        },
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "delivery",
        "overall",
        "seed",
        "first",
        "last",
        "task",
        "morphology",
        "gripper_ratio",
        "gripper_absolute",
        "gripper_seed",
        "null",
    ],
)
def test_broad_classification_every_gate_boundary_and_null(mutation: str) -> None:
    metrics = _classification_metrics()
    delivery = True
    if mutation == "delivery":
        delivery = False
    elif mutation == "overall":
        metrics["overall"]["normalized_loss"]["ratio"] = 1.099
    elif mutation == "seed":
        for seed in (36001, 36002):
            metrics["by_seed"][str(seed)]["normalized_loss"]["ratio"] = 1.099
    elif mutation in phase_agreement.ENDPOINTS:
        metrics["by_endpoint"][mutation]["normalized_loss"]["ratio"] = 1.099
    elif mutation == "task":
        metrics["by_task"]["lift_object"]["normalized_loss"]["difference"] = -1e-9
    elif mutation == "morphology":
        metrics["by_morphology"]["panda"]["normalized_loss"]["difference"] = -1e-9
    elif mutation == "gripper_ratio":
        metrics["overall"]["effective_gripper_error"]["ratio"] = 1.099
    elif mutation == "gripper_absolute":
        metrics["overall"]["effective_gripper_error"]["difference"] = 0.0099
    elif mutation == "gripper_seed":
        for seed in (36001, 36002):
            metrics["by_seed"][str(seed)]["effective_gripper_error"]["ratio"] = 1.099
    else:
        metrics["overall"]["normalized_loss"]["ratio"] = None
    result = phase_agreement.classify_broad_deficit(metrics, delivery=delivery)
    assert result["confirmed"] is False

    exact = phase_agreement.classify_broad_deficit(
        _classification_metrics(), delivery=True
    )
    assert exact["confirmed"] is True
    assert exact["classification"] == "broad_correct_prompt_expert_action_agreement_deficit"


def test_classification_exactly_two_seeds_zero_denominators_and_zero_differences() -> None:
    metrics = _classification_metrics()
    metrics["by_seed"]["36003"]["normalized_loss"]["ratio"] = 1.0
    metrics["by_seed"]["36003"]["effective_gripper_error"]["ratio"] = 1.0
    for task in phase_agreement.TASKS:
        metrics["by_task"][task]["normalized_loss"]["difference"] = 0.0
    for morphology in phase_agreement.MORPHOLOGIES:
        metrics["by_morphology"][morphology]["normalized_loss"]["difference"] = 0.0
    assert phase_agreement.classify_broad_deficit(metrics, delivery=True)[
        "confirmed"
    ] is True

    metrics["overall"]["normalized_loss"] = {
        "joint": 1.0,
        "matched": 0.0,
        "difference": 1.0,
        "ratio": phase_agreement._nullable_ratio(1.0, 0.0),
    }
    metrics["overall"]["effective_gripper_error"] = {
        "joint": 1.0,
        "matched": 0.0,
        "difference": 1.0,
        "ratio": phase_agreement._nullable_ratio(1.0, 0.0),
    }
    result = phase_agreement.classify_broad_deficit(metrics, delivery=True)
    assert result["overall_normalized_ratio"] is None
    assert result["overall_effective_gripper_ratio"] is None
    assert result["confirmed"] is False


def test_repeat_exact_tolerance_and_seed36003_orders() -> None:
    result = phase_agreement.evaluate_anchor_condition(
        _anchor(),
        36003,
        "joint",
        FakePolicy(repeat_delta=phase_agreement.REPEAT_MAX_ABS),
        FakeProcessor(),
        "cpu",
    )
    assert result["repeat_max_abs_difference"] == pytest.approx(
        phase_agreement.REPEAT_MAX_ABS, rel=1e-6
    )
    assert phase_agreement.condition_order(0, 36003, "so101") == ("joint", "matched")
    assert phase_agreement.condition_order(0, 36003, "panda") == ("matched", "joint")
    assert phase_agreement.condition_order(181, 36003, "panda") == ("joint", "matched")


def _fake_checkpoint_identity(exp36: dict, seed: int, morphology: str, source: str) -> dict:
    return {
        "model_seed": seed,
        "morphology": morphology,
        "checkpoint_path": f"fake/{source}/{seed}/{morphology}",
        "core_sha256": exp36["runs"][f"{source}-seed{seed}"]["core_sha256"],
        "config_sha256": phase_agreement.CONFIG_SHA256[morphology],
        "embodiment_sha256": "3" * 64,
        "trainer_sha256": "4" * 64,
        "data_manifest_sha256": phase_agreement.DATA_MANIFEST_SHA256[source],
        "exp36_summary_sha256": phase_agreement.EXP36_SUMMARY_SHA256,
        "step": 1200,
        "stage": "core",
    }


def _make_smoke_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = _manifest()
    exp36 = json.loads(_frozen_sources()["exp36_summary"].read_text())

    def resolve(_summary, source, seed, morphology, *, root):
        assert root == Path("fake-root")
        identity = _fake_checkpoint_identity(exp36, seed, morphology, source)
        return Path(identity["checkpoint_path"]), identity

    runtime = {"device": "cpu"}
    monkeypatch.setattr(phase_agreement, "resolve_checkpoint", resolve)
    monkeypatch.setattr(
        phase_agreement,
        "_runtime_provenance",
        lambda _device: {
            "git_dirty": True,
            "git_commit": "smoke",
            "evaluator_sha256": "smoke",
            "runtime": runtime,
            "runtime_signature_sha256": "smoke",
            "imported_exp41_evaluator_sha256": phase_agreement.EXP41_EVALUATOR_SHA256,
            "expert_sha256": phase_agreement.EXPERT_SHA256,
        },
    )

    def env_factory(morphology, task, _scenario_value, **kwargs):
        assert kwargs == {"width": 640, "height": 480, "render": True}
        return FakeEnv(task, morphology)

    policies = []

    def load_policy(path: Path, device: str):
        assert device == "cpu"
        value = 0.02 if "joint" in path.parts else 0.01
        policy = FakePolicy(value=value)
        policies.append(policy)
        return policy

    output = tmp_path / "exp42-smoke.json"
    report = phase_agreement.evaluate_phase_agreement_shard(
        definition,
        manifest,
        _frozen_sources(),
        output,
        morphology="so101",
        device="cpu",
        registered=False,
        limit_scenarios=1,
        root=Path("fake-root"),
        env_factory=env_factory,
        expert_factory=FakeExpert,
        policy_loader=load_policy,
        processor_factory=lambda *_args, **_kwargs: FakeProcessor(),
    )
    assert len(policies) == 6
    assert all(policy.predict_calls == 8 for policy in policies)
    assert json.loads(output.read_text()) == report
    phase_agreement.validate_shard_report(
        report, manifest, exp36, require_registered=False
    )
    return report


def test_complete_smoke_report_generation_validation_and_arm_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _make_smoke_report(monkeypatch, tmp_path)
    assert report["expected_anchor_ids"] == [
        "development-so101-push_to_zone-0028/precontact/first",
        "development-so101-push_to_zone-0028/precontact/last",
        "development-so101-push_to_zone-0028/push/first",
        "development-so101-push_to_zone-0028/push/last",
    ]
    assert report["aggregate"]["endpoint_anchors"] == 4
    assert report["aggregate"]["scored_chunks"] == 24
    assert report["aggregate"]["repeat_chunks"] == 24
    assert report["aggregate"]["policy_inferences"] == 48
    assert report["episodes"][0]["observed_phases"] == [
        "open",
        "precontact",
        "push",
        "done",
    ]
    for anchor in report["anchors"]:
        assert anchor["pre_call_phase"] == anchor["phase"]
        assert anchor["pre_call_phase_tick"] == anchor["phase_tick"]
        assert anchor["post_action_total_tick"] == anchor["pre_call_total_tick"] + 1
        assert anchor["post_action_state_sha256"] == anchor["state_sha256"]
        for seed in phase_agreement.SEEDS:
            assert tuple(anchor["seeds"][str(seed)]) == phase_agreement.condition_order(
                anchor["ordinal"], seed, "so101"
            )
    assert "frame" not in report["anchors"][0]
    assert "state" not in report["anchors"][0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("output", "tensor hashes"),
        ("metric", "metrics do not recompute"),
        ("tokenizer", "tensor hashes"),
        ("cross_arm", "cross-arm tokenizer"),
        ("round_trip", "round-trip evidence"),
        ("duplicate", "one-action endpoint duplicate"),
        ("tick", "endpoint pair ordering"),
        ("post_timing", "did not cause"),
        ("terminal_noop", "native no-op"),
        ("task", "scenario metadata"),
        ("chain", "contract or completion"),
        ("aggregate", "aggregate does not recompute"),
        ("final_split", "contract or completion"),
        ("checkpoint", "checkpoint identity"),
        ("schedule", "contract or completion"),
    ],
)
def test_complete_report_validation_rejects_mutations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str, message: str
) -> None:
    report = _make_smoke_report(monkeypatch, tmp_path)
    changed = json.loads(json.dumps(report))
    first = changed["anchors"][0]
    if mutation == "output":
        first["seeds"]["36001"]["joint"]["queries"][0]["output_chunk"][0][0][0] += 1
    elif mutation == "metric":
        first["seeds"]["36001"]["joint"]["metrics"]["normalized_loss"] += 1
    elif mutation == "tokenizer":
        first["seeds"]["36001"]["joint"]["queries"][1]["input_ids"][0][0] += 1
    elif mutation == "cross_arm":
        queries = first["seeds"]["36003"]["matched"]["queries"]
        for query in queries:
            input_ids = np.asarray(query["input_ids"], dtype=np.int64) + 1
            query["input_ids"] = input_ids.tolist()
            query["input_ids_sha256"] = phase_agreement.canonical_array_sha256(
                input_ids
            )
    elif mutation == "round_trip":
        first["round_trip"]["clipped"] = True
    elif mutation == "duplicate":
        duplicate = changed["anchors"][1]
        duplicate["image_sha256"] = "a" * 64
        for seed in phase_agreement.SEEDS:
            for condition in phase_agreement.CONDITIONS:
                for query in duplicate["seeds"][str(seed)][condition]["queries"]:
                    query["image_sha256"] = "a" * 64
    elif mutation == "tick":
        first["phase_tick"] = 2
        first["pre_call_phase_tick"] = 2
        changed["episodes"][0]["phase_endpoint_ticks"]["precontact"][0] = 2
    elif mutation == "post_timing":
        changed["anchors"][1]["post_action_phase"] = "precontact"
        changed["anchors"][1]["post_action_phase_tick"] = 2
    elif mutation == "terminal_noop":
        terminal = changed["anchors"][-1]
        terminal["pre_call_native_state"][0] += 1
        terminal["pre_call_native_state_sha256"] = phase_agreement.canonical_array_sha256(
            np.asarray(terminal["pre_call_native_state"], dtype=np.float32)
        )
    elif mutation == "task":
        first["task"] = "lift_object"
    elif mutation == "chain":
        changed["outcome_chain_sha256"] = "0" * 64
    elif mutation == "aggregate":
        changed["aggregate"]["endpoint_anchors"] += 1
    elif mutation == "final_split":
        changed["contract"]["final_split_loaded"] = True
    elif mutation == "checkpoint":
        changed["identity"]["checkpoints"]["36001"]["joint"]["core_sha256"] = "0" * 64
    else:
        changed["run"]["seed_order"] = list(reversed(phase_agreement.SEEDS))
    if mutation not in {"chain", "aggregate", "final_split", "checkpoint", "schedule"}:
        changed["outcome_chain_sha256"] = phase_agreement._outcome_chain(
            changed["anchors"]
        )
        changed["aggregate"] = phase_agreement._shard_aggregate(
            changed["anchors"], changed
        )
    with pytest.raises(ValueError, match=message):
        phase_agreement.validate_shard_report(
            changed,
            _manifest(),
            json.loads(_frozen_sources()["exp36_summary"].read_text()),
            require_registered=False,
        )


def test_frozen_source_contracts_and_hash_mutation(tmp_path: Path) -> None:
    summaries = phase_agreement._source_contract(_frozen_sources())
    assert summaries["exp36_summary"]["experiment"] == 36
    assert summaries["exp41_summary"]["experiment"] == 41
    altered = tmp_path / "exp41.json"
    altered.write_bytes(_frozen_sources()["exp41_summary"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="exp41 summary hash"):
        phase_agreement._source_contract(
            {**_frozen_sources(), "exp41_summary": altered}
        )
    assert phase_agreement.file_sha256(Path(phase_agreement.exp41.__file__)) == (
        "2ca9ae5b48782b4b378882accf886dd9b65bd1c7ed3905bb20dfc3c859f9d62c"
    )


def test_immutable_final_and_exact_byte_writes(tmp_path: Path) -> None:
    final = tmp_path / "final.json"
    phase_agreement._write_final(final, {"value": 1})
    phase_agreement._write_final(final, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        phase_agreement._write_final(final, {"value": 2})
    raw = tmp_path / "source.json"
    phase_agreement._write_exact_bytes(raw, b"exact\n")
    phase_agreement._write_exact_bytes(raw, b"exact\n")
    assert raw.read_bytes() == b"exact\n"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        phase_agreement._write_exact_bytes(raw, b"different\n")


def test_registered_filenames_cli_and_lock_guards() -> None:
    assert phase_agreement.registered_shard_filename("so101") == (
        "benchmark-exp42-so101-phase-action-quality.json"
    )
    assert phase_agreement._registered_output("panda", None) == Path(
        "reports/exp42-runtime/benchmark-exp42-panda-phase-action-quality.json"
    )
    with pytest.raises(ValueError, match="registered shard output"):
        phase_agreement._registered_output("panda", Path("custom.json"))
    smoke = phase_agreement.parse_args(
        [
            "shard",
            "--morphology",
            "so101",
            "--smoke",
            "--output",
            "smoke.json",
            "--limit-scenarios",
            "1",
        ]
    )
    assert smoke.smoke is True and smoke.limit_scenarios == 1
    assert phase_agreement.parse_args(["aggregate"]).output == Path(
        "reports/benchmark-exp42-summary.json"
    )
    with phase_agreement.registered_run_lock():
        with pytest.raises(RuntimeError, match="holds"):
            with phase_agreement.registered_run_lock():
                pass


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            SimpleNamespace(command="shard", smoke=False, output=Path("x")),
            "only run through run-registered",
        ),
        (
            SimpleNamespace(
                command="shard",
                smoke=True,
                output=Path("reports/exp42-runtime/x.json"),
            ),
            "may not use",
        ),
        (
            SimpleNamespace(
                command="aggregate",
                shard_dir=Path("custom"),
                output=phase_agreement.SUMMARY_PATH,
            ),
            "shard directory",
        ),
        (
            SimpleNamespace(
                command="aggregate",
                shard_dir=phase_agreement.REGISTERED_RUNTIME_DIR,
                output=Path("custom"),
            ),
            "aggregate output",
        ),
    ],
)
def test_main_cli_guards(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, message: str
) -> None:
    monkeypatch.setattr(phase_agreement, "parse_args", lambda: args)
    with pytest.raises(ValueError, match=message):
        phase_agreement.main()


def test_registered_two_shard_orchestrator_validation_aggregate_and_main_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    publication_dir = tmp_path / "published"
    publication_dir.mkdir()
    summary_path = publication_dir / "benchmark-exp42-summary.json"
    lock_path = runtime_dir / ".registered.lock"
    monkeypatch.setattr(phase_agreement, "REGISTERED_RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(phase_agreement, "REGISTERED_LOCK_PATH", lock_path)
    monkeypatch.setattr(phase_agreement, "SUMMARY_PATH", summary_path)
    monkeypatch.setattr(
        phase_agreement,
        "camera_frame_to_tensor",
        lambda _frame, *, processor_rescales: torch.zeros(
            (1,), dtype=torch.float32
        ),
    )
    runtime = {name: None for name in phase_agreement.RUNTIME_FIELDS}
    runtime["device"] = "cpu"
    runtime_signature = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(
        phase_agreement,
        "_runtime_provenance",
        lambda _device: {
            "git_dirty": False,
            "git_commit": "1" * 40,
            "evaluator_sha256": phase_agreement.file_sha256(
                Path(phase_agreement.__file__)
            ),
            "runtime": runtime,
            "runtime_signature_sha256": runtime_signature,
            "imported_exp41_evaluator_sha256": phase_agreement.EXP41_EVALUATOR_SHA256,
            "expert_sha256": phase_agreement.EXPERT_SHA256,
        },
    )
    real_evaluate = phase_agreement.evaluate_phase_agreement_shard

    def evaluate(*args, **kwargs):
        return real_evaluate(
            *args,
            **kwargs,
            env_factory=lambda morphology, task, _scenario_value, **_options: FakeEnv(
                task, morphology
            ),
            expert_factory=FakeExpert,
            policy_loader=lambda path, _device: FakePolicy(
                value=0.02 if "joint" in str(path) else 0.01
            ),
            processor_factory=lambda *_args, **_kwargs: FakeProcessor(),
        )

    monkeypatch.setattr(phase_agreement, "evaluate_phase_agreement_shard", evaluate)
    args = SimpleNamespace(
        exp36_summary=_frozen_sources()["exp36_summary"],
        exp41_summary=_frozen_sources()["exp41_summary"],
        device="cpu",
        artifact_root=ROOT,
    )
    reports = phase_agreement._run_all_registered(
        args, BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json"), _manifest()
    )
    assert [report["identity"]["morphology"] for report in reports] == [
        "so101",
        "panda",
    ]
    run_ids = {report["run"]["orchestrator_run_id"] for report in reports}
    assert len(run_ids) == 1
    assert phase_agreement._valid_run_id(next(iter(run_ids)))
    for report in reports:
        for seed in phase_agreement.SEEDS:
            first_counts = {
                condition: sum(
                    next(iter(anchor["seeds"][str(seed)])) == condition
                    for anchor in report["anchors"]
                )
                for condition in phase_agreement.CONDITIONS
            }
            assert first_counts == {"joint": 91, "matched": 91}
        phase_agreement.validate_shard_report(
            report,
            _manifest(),
            json.loads(_frozen_sources()["exp36_summary"].read_text()),
            require_registered=True,
            artifact_root=ROOT,
        )
        for seed in phase_agreement.SEEDS:
            for condition in phase_agreement.CONDITIONS:
                identity = report["identity"]["checkpoints"][str(seed)][condition]
                checkpoint = Path(identity["checkpoint_path"])
                assert phase_agreement.file_sha256(checkpoint / "embodiment.pt") == identity[
                    "embodiment_sha256"
                ]
                assert phase_agreement.file_sha256(checkpoint / "trainer.pt") == identity[
                    "trainer_sha256"
                ]

    source_paths = {
        morphology: runtime_dir / phase_agreement.registered_shard_filename(morphology)
        for morphology in phase_agreement.MORPHOLOGIES
    }
    exp36 = json.loads(_frozen_sources()["exp36_summary"].read_text())
    summary = phase_agreement.aggregate_registered_shards(
        source_paths, _manifest(), exp36_summary=exp36, artifact_root=ROOT
    )
    assert summary["contract"]["orchestrator_run_id"] == next(iter(run_ids))
    assert summary["source_shard_sha256"] == {
        morphology: hashlib.sha256(path.read_bytes()).hexdigest()
        for morphology, path in source_paths.items()
    }
    panda_path = source_paths["panda"]
    original_panda_bytes = panda_path.read_bytes()
    so101_report = json.loads(source_paths["so101"].read_bytes())
    for mutation in (
        "orchestrator_run_id",
        "git_commit",
        "runtime",
        "overlapping_interval",
    ):
        changed = json.loads(original_panda_bytes)
        if mutation == "orchestrator_run_id":
            changed["run"]["orchestrator_run_id"] = "f" * 32
        elif mutation == "git_commit":
            changed["provenance"]["git_commit"] = "2" * 40
        elif mutation == "runtime":
            changed["provenance"]["runtime"]["hostname"] = "different-valid-host"
            changed["provenance"]["runtime_signature_sha256"] = hashlib.sha256(
                json.dumps(
                    changed["provenance"]["runtime"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        else:
            changed["run"]["started_utc"] = so101_report["run"]["started_utc"]
        panda_path.write_text(json.dumps(changed, indent=2) + "\n")
        try:
            with pytest.raises(
                ValueError, match="common commit/runtime or nonoverlap"
            ):
                phase_agreement.aggregate_registered_shards(
                    source_paths,
                    _manifest(),
                    exp36_summary=exp36,
                    artifact_root=ROOT,
                )
        finally:
            panda_path.write_bytes(original_panda_bytes)
    swapped = {"so101": source_paths["panda"], "panda": source_paths["so101"]}
    with pytest.raises(ValueError, match="misattributed"):
        phase_agreement.aggregate_registered_shards(
            swapped, _manifest(), exp36_summary=exp36, artifact_root=ROOT
        )

    with pytest.raises(FileExistsError, match="delete both"):
        phase_agreement._run_all_registered(
            args,
            BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json"),
            _manifest(),
        )

    events = []
    real_promote = phase_agreement._promote_registered_sources
    real_write = phase_agreement._write_final

    def promote(paths, hashes):
        events.append("promote")
        real_promote(paths, hashes, destination_dir=publication_dir)

    def write(path, report):
        assert all(
            (publication_dir / phase_agreement.registered_shard_filename(morphology)).exists()
            for morphology in phase_agreement.MORPHOLOGIES
        )
        events.append("summary")
        real_write(path, report)

    monkeypatch.setattr(phase_agreement, "_promote_registered_sources", promote)
    monkeypatch.setattr(phase_agreement, "_write_final", write)
    monkeypatch.setattr(
        phase_agreement,
        "parse_args",
        lambda: SimpleNamespace(
            command="aggregate",
            shard_dir=runtime_dir,
            output=summary_path,
            artifact_root=ROOT,
            definition=ROOT / "benchmarks/sim-v1/definition.json",
            manifest=ROOT / "benchmarks/sim-v1/manifests/development.json",
            exp36_summary=_frozen_sources()["exp36_summary"],
            exp41_summary=_frozen_sources()["exp41_summary"],
        ),
    )
    phase_agreement.main()
    assert events == ["promote", "summary"]
    published_summary = json.loads(summary_path.read_text())
    for morphology in phase_agreement.MORPHOLOGIES:
        promoted = publication_dir / phase_agreement.registered_shard_filename(morphology)
        assert promoted.read_bytes() == source_paths[morphology].read_bytes()
        assert hashlib.sha256(promoted.read_bytes()).hexdigest() == published_summary[
            "source_shard_sha256"
        ][morphology]


def test_registered_direct_evaluation_race_and_promotion_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="orchestrator context"):
        phase_agreement.evaluate_phase_agreement_shard(
            None,
            None,
            {},
            tmp_path / "x.json",
            morphology="so101",
            device="cpu",
            registered=True,
        )
    with pytest.raises(RuntimeError, match="active orchestrator schedule"):
        phase_agreement.evaluate_phase_agreement_shard(
            None,
            None,
            {},
            tmp_path / "forged.json",
            morphology="so101",
            device="cpu",
            registered=True,
            orchestrator=phase_agreement._OrchestratorContext("a" * 32),
        )

    race_dir = tmp_path / "race"
    race_dir.mkdir()
    monkeypatch.setattr(phase_agreement, "REGISTERED_RUNTIME_DIR", race_dir)
    monkeypatch.setattr(phase_agreement, "REGISTERED_LOCK_PATH", race_dir / ".registered.lock")

    @contextmanager
    def race_lock(_path):
        (race_dir / phase_agreement.registered_shard_filename("so101")).write_text("race")
        yield

    monkeypatch.setattr(phase_agreement, "registered_run_lock", race_lock)
    args = SimpleNamespace()
    with pytest.raises(FileExistsError, match="appeared before schedule"):
        phase_agreement._run_all_registered(args, None, None)

    staged = tmp_path / "staged"
    promoted = tmp_path / "promoted"
    staged.mkdir()
    promoted.mkdir()
    paths = {}
    hashes = {}
    for morphology in phase_agreement.MORPHOLOGIES:
        path = staged / phase_agreement.registered_shard_filename(morphology)
        path.write_bytes(f"{morphology}\n".encode())
        paths[morphology] = path
        hashes[morphology] = hashlib.sha256(path.read_bytes()).hexdigest()
    conflict = promoted / phase_agreement.registered_shard_filename("panda")
    conflict.write_bytes(b"conflict")
    with pytest.raises(FileExistsError, match="conflicting promoted source"):
        phase_agreement._promote_registered_sources(
            paths, hashes, destination_dir=promoted
        )
    assert not (promoted / phase_agreement.registered_shard_filename("so101")).exists()
    conflict.unlink()
    paths["panda"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after registered aggregation"):
        phase_agreement._promote_registered_sources(
            paths, hashes, destination_dir=promoted
        )
