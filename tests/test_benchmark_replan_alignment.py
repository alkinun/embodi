import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodi.sim.benchmark import BenchmarkDefinition, ScenarioManifest, file_sha256
import embodi.sim.benchmark_replan_alignment as alignment
from embodi.sim.benchmark_replan_alignment import (
    ALIGNMENT_WEIGHTS,
    AlignmentTreatmentError,
    _load_or_create_report,
    _new_report,
    _outcome_chain,
    _registered_output,
    _shard_aggregate,
    aggregate_registered_shards,
    alignment_scenarios,
    alignment_transform,
    capture_treatment_onset,
    compose_absolute_targets,
    pair_arm_order,
    parse_args,
    registered_shard_filename,
    run_alignment_pair,
    validate_exp39_summary,
)
from embodi.sim.tasks import TaskStatus


ROOT = Path(__file__).parents[1]


def _rotation_z(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rotation_x(angle: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )


def _rotation_y(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )


def _chunk() -> np.ndarray:
    chunk = np.zeros((32, 1, 10), dtype=np.float32)
    for lead in range(32):
        chunk[lead, 0, 0] = 0.01 * lead
        chunk[lead, 0, 3:9] = alignment.exp39.canonical_rotation_6d(
            _rotation_z(np.deg2rad(5.0 * lead))
        )
        chunk[lead, 0, 9] = 0.2 + 0.15 * lead
    return chunk


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
        self.predict_calls = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_canonical_action_chunk(self, _batch):
        self.predict_calls += 1
        return torch.from_numpy(self.chunk[None])


class FakeData:
    def __init__(self) -> None:
        self.qpos = np.zeros(10, dtype=np.float64)
        self.qvel = np.zeros(10, dtype=np.float64)
        self.act = np.zeros(2, dtype=np.float64)
        self.ctrl = np.zeros(10, dtype=np.float64)
        self.qacc_warmstart = np.zeros(10, dtype=np.float64)
        self.time = 0.0


class FakeEnv:
    instruction = "Push the object into the target zone."

    def __init__(
        self,
        *,
        task: str = "push_to_zone",
        terminal_step: int | None = None,
        onset_offset: float = 0.0,
    ) -> None:
        self.morphology = SimpleNamespace(id="panda", native_action_dim=10)
        self.task = SimpleNamespace(id=task)
        self.scenario = SimpleNamespace(target_tolerance=0.055)
        self.data = FakeData()
        self.control_time = 0.0
        self.dwell = 0
        self.lifted = False
        self.success = False
        self.steps = 0
        self.terminal_step = terminal_step
        self.onset_offset = onset_offset
        self.applied: list[np.ndarray] = []
        self.decoded: list[np.ndarray] = []

    def render_top(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def canonical_state(self):
        state = np.zeros((1, 10), dtype=np.float32)
        state[0, 3:9] = alignment.exp39.canonical_rotation_6d(np.eye(3))
        return state

    def native_action_from_canonical(self, canonical, *, iterations):
        assert iterations == 20
        native = np.asarray(canonical, dtype=np.float32).reshape(10).copy()
        self.decoded.append(native)
        return native

    def canonical_arm_action(self, native):
        return np.asarray(native, dtype=np.float32).reshape(1, 10)

    @staticmethod
    def native_to_qpos(native):
        return np.asarray(native)

    @staticmethod
    def qpos_to_native(native):
        return np.asarray(native)

    def apply_action(self, native):
        self.applied.append(native)
        self.data.ctrl[:] = native

    def step_control_period(self, frequency):
        assert frequency == 30.0
        self.steps += 1
        self.control_time += 1 / frequency
        self.data.time = self.control_time
        self.data.qpos[:] = self.data.ctrl
        if self.steps == 4:
            self.data.qpos[0] += self.onset_offset
        terminal = self.terminal_step is not None and self.steps >= self.terminal_step
        self.success = terminal
        return TaskStatus(
            success=terminal,
            terminal=terminal,
            stage="success" if terminal else "push",
        )

    def diagnostic_state(self):
        return {
            "ee_object_distance_m": 0.1,
            "object_target_xy_distance_m": 0.2,
            "object_height_m": 0.02,
            "object_speed": 0.1,
            "gripper_opening_fraction": 1.0,
        }

    def close(self):
        pass


def _rehash_snapshot(snapshot: dict) -> None:
    body = {
        name: snapshot[name] for name in ("schema", "arrays", "scalars", "milestones")
    }
    snapshot["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validation_report() -> dict:
    pair = run_alignment_pair(
        {arm: FakeEnv() for arm in alignment.ARMS},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=8,
    )
    return {
        "identity": {
            "condition": "joint",
            "model_seed": 36001,
            "morphology": "so101",
        },
        "outcomes": [
            {
                **pair,
                "scenario_id": "development-so101-push_to_zone-0014",
                "task": "push_to_zone",
                "morphology": "so101",
            }
        ],
    }


def test_fresh_scenario_slice_covers_all_axes() -> None:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    selected = alignment_scenarios(manifest, "so101", registered=True)
    assert len(selected) == 21
    assert selected[0].scenario_id.endswith("push_to_zone-0014")
    assert selected[-1].scenario_id.endswith("pick_place_bin-0020")
    for task in alignment.TASKS:
        cell = [scenario for scenario in selected if scenario.task == task]
        axes = [
            name
            for scenario in cell
            for name, value in scenario.factors.items()
            if value["bin"].startswith("development:")
        ]
        assert len(axes) == len(set(axes)) == 7


def test_shared_source_chunk_is_decoded_once_and_exact_natives_execute_in_both_arms() -> None:
    baseline = FakeEnv(terminal_step=4)
    treated = FakeEnv(terminal_step=4)
    source_chunk = _chunk()
    repeated_chunk = source_chunk.copy()
    repeated_chunk[0, 0, 0] += 1e-6

    class RepeatPolicy(FakePolicy):
        def predict_canonical_action_chunk(self, _batch):
            self.predict_calls += 1
            value = source_chunk if self.predict_calls == 1 else repeated_chunk
            return torch.from_numpy(value[None])

    pair = run_alignment_pair(
        {"baseline": baseline, "alignment": treated},
        RepeatPolicy(),
        FakeProcessor(),
        "cpu",
        order=("alignment", "baseline"),
        maximum_steps=4,
    )
    assert pair["initial_policy_input_source_arm"] == "alignment"
    assert len(treated.decoded) == 4
    assert baseline.decoded == []
    assert pair["source_chunk_sha256"] == pair["first_chunks"]["alignment"]
    assert pair["source_chunk_sha256"] != pair["first_chunks"]["baseline"]
    assert treated.applied[0][0] == source_chunk[0, 0, 0]
    assert len(baseline.applied) == len(treated.applied) == 4
    assert all(first is second for first, second in zip(baseline.applied, treated.applied, strict=True))


def test_complete_treatment_onset_snapshot_matches_and_mismatch_aborts() -> None:
    pair = run_alignment_pair(
        {arm: FakeEnv(terminal_step=4) for arm in alignment.ARMS},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=4,
    )
    snapshot = pair["treatment_onset_snapshot"]
    assert pair["treatment_onset_equal"] is True
    assert snapshot["schema"] == alignment.SNAPSHOT_SCHEMA
    assert set(snapshot["arrays"]) == set(alignment.SNAPSHOT_ARRAYS)
    alignment._validate_snapshot(snapshot, "push_to_zone", 4)
    with pytest.raises(RuntimeError, match="treatment-onset"):
        run_alignment_pair(
            {"baseline": FakeEnv(), "alignment": FakeEnv(onset_offset=1e-9)},
            FakePolicy(),
            FakeProcessor(),
            "cpu",
            order=("baseline", "alignment"),
            maximum_steps=8,
        )


def test_terminal_common_prefix_is_valid_no_op_pair() -> None:
    pair = run_alignment_pair(
        {arm: FakeEnv(terminal_step=2) for arm in alignment.ARMS},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("alignment", "baseline"),
        maximum_steps=500,
    )
    assert pair["terminal_before_treatment"] is True
    assert pair["common_prefix_commands"] == 2
    assert pair["arms"]["baseline"]["success"] is True
    assert pair["arms"]["alignment"]["success"] is True
    assert pair["arms"]["alignment"]["correction_opportunities"] == 0
    assert pair["treatment_onset_policy"]["applicable"] is False
    assert all(
        value is None
        for name, value in pair["treatment_onset_policy"].items()
        if name != "applicable"
    )
    alignment._validate_onset_policy_evidence(pair)
    for arm in alignment.ARMS:
        outcome = pair["arms"][arm]
        assert outcome["decoded_commands"] == outcome["prepared_commands"] == 4
        assert outcome["commands"] == 2
        assert outcome["executed_commands_by_lead"] == {
            "0": 1,
            "1": 1,
            "2": 0,
            "3": 0,
        }
        assert sum(
            value["count"]
            for value in outcome["metrics_by_lead"]["reconstruction_translation_m"].values()
        ) == 2
        assert outcome["clipping_rate"] == outcome["clipped_commands"] / 2
    assert _shard_aggregate([{**pair, "task": "push_to_zone"}])[
        "no_op_common_prefix_pairs"
    ] == 1


@pytest.mark.parametrize(
    ("terminal_step", "post_counts"),
    [
        (6, {"0": 1, "1": 1, "2": 0, "3": 0}),
        (7, {"0": 1, "1": 1, "2": 1, "3": 0}),
        (8, {"0": 1, "1": 1, "2": 1, "3": 1}),
    ],
)
def test_post_prefix_terminal_mid_chunk_records_executed_leads_only(
    terminal_step: int, post_counts: dict[str, int]
) -> None:
    pair = run_alignment_pair(
        {arm: FakeEnv(terminal_step=terminal_step) for arm in alignment.ARMS},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=500,
    )
    for arm in alignment.ARMS:
        outcome = pair["arms"][arm]
        assert outcome["prepared_commands"] == outcome["decoded_commands"] == 8
        assert outcome["post_prefix_executed_commands_by_lead"] == post_counts
        assert outcome["commands"] == terminal_step
        for lead, expected in post_counts.items():
            assert outcome["metrics_by_lead"]["pre_translation_m"][lead]["count"] == expected
        assert outcome["clipping_rate"] == outcome["clipped_commands"] / outcome["commands"]
        alignment._validate_arm(outcome, arm, "push_to_zone")


def test_validation_rejects_shuffled_executed_leads() -> None:
    pair = run_alignment_pair(
        {arm: FakeEnv(terminal_step=6) for arm in alignment.ARMS},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=8,
    )
    result = copy.deepcopy(pair["arms"]["alignment"])
    result["executed_commands_by_lead"]["0"] -= 1
    result["executed_commands_by_lead"]["3"] += 1
    with pytest.raises(ValueError, match="assigned to invalid leads"):
        alignment._validate_arm(result, "alignment", "push_to_zone")


def test_shared_first_treatment_replan_uses_one_onset_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    original_predict = alignment.exp39.predict_from_input

    def tracked(snapshot, policy, processor, device):
        observed.append(snapshot)
        return original_predict(snapshot, policy, processor, device)

    monkeypatch.setattr(alignment.exp39, "predict_from_input", tracked)
    pair = run_alignment_pair(
        {arm: FakeEnv() for arm in alignment.ARMS},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("alignment", "baseline"),
        maximum_steps=8,
    )
    evidence = pair["treatment_onset_policy"]
    assert evidence["applicable"] is True
    assert evidence["source_arm"] == "alignment"
    assert observed[2] is observed[3]
    assert evidence["anchor_hashes"] == evidence["actual_inputs"]["alignment"]
    assert evidence["source_raw_chunk_sha256"] == evidence["chunks"]["alignment"]
    assert set(evidence["consumed_source_raw_chunk_sha256"].values()) == {
        evidence["source_raw_chunk_sha256"]
    }
    alignment._validate_onset_policy_evidence(pair)


def test_onset_prediction_failure_is_generic_infrastructure_error() -> None:
    class FailingPolicy(FakePolicy):
        def predict_canonical_action_chunk(self, batch):
            if self.predict_calls == 2:
                raise RuntimeError("onset prediction failure")
            return super().predict_canonical_action_chunk(batch)

    with pytest.raises(RuntimeError, match="onset prediction failure"):
        run_alignment_pair(
            {arm: FakeEnv() for arm in alignment.ARMS},
            FailingPolicy(),
            FakeProcessor(),
            "cpu",
            order=("baseline", "alignment"),
            maximum_steps=8,
        )


def test_alignment_transform_math_weights_effective_gripper_and_tail() -> None:
    anchor = np.r_[np.zeros(3), alignment.exp39.canonical_rotation_6d(np.eye(3)), 0.0]
    chunk = np.zeros((32, 1, 10), dtype=np.float32)
    chunk[:, 0, 3:9] = alignment.exp39.canonical_rotation_6d(np.eye(3))
    chunk[:, 0, 9] = -1.0
    previous = compose_absolute_targets(anchor, chunk)
    previous["translation_m"][4] = [1.0, 0.0, 0.0]
    previous["rotation"][4] = _rotation_z(np.pi / 2)
    previous["gripper_raw"][4] = 2.0
    previous["gripper_effective"][4] = 1.0
    original = chunk.copy()
    transformed, intended, details = alignment_transform(anchor, previous, chunk)
    assert ALIGNMENT_WEIGHTS == (1.0, 0.75, 0.5, 0.25)
    assert intended["translation_m"][:4, 0] == pytest.approx(ALIGNMENT_WEIGHTS)
    assert [alignment._rotation_error_degrees(np.eye(3), value) for value in intended["rotation"][:4]] == pytest.approx(
        [90.0, 67.5, 45.0, 22.5], abs=1e-4
    )
    assert intended["gripper_effective"][:4] == pytest.approx([1.0, 0.75, 0.5, 0.25])
    assert np.array_equal(transformed[4:], original[4:])
    assert details["tail_unchanged"] is True
    assert details["lead_zero_errors"]["translation_m"] <= 1e-6
    assert details["lead_zero_errors"]["rotation_deg"] <= 1e-4
    assert details["lead_zero_errors"]["gripper_effective"] <= 1e-6


def test_alignment_transform_numerical_gate_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = np.r_[np.zeros(3), alignment.exp39.canonical_rotation_6d(np.eye(3)), 0.0]
    chunk = np.zeros((32, 1, 10), dtype=np.float32)
    chunk[:, 0, 3:9] = alignment.exp39.canonical_rotation_6d(np.eye(3))
    previous = compose_absolute_targets(anchor, chunk)
    previous["rotation"][4] = _rotation_z(np.pi / 2)
    identity_6d = alignment.exp39.canonical_rotation_6d(np.eye(3))
    monkeypatch.setattr(
        alignment.exp39,
        "canonical_rotation_6d",
        lambda _rotation: identity_6d.copy(),
    )
    with pytest.raises(AlignmentTreatmentError, match="transform threshold exceeded"):
        alignment_transform(anchor, previous, chunk)


def test_nonidentity_anchor_noncommuting_rotation_reencoding_order() -> None:
    anchor_rotation = _rotation_x(np.deg2rad(35)) @ _rotation_z(np.deg2rad(20))
    anchor = np.r_[
        np.array([0.3, -0.2, 0.4]),
        alignment.exp39.canonical_rotation_6d(anchor_rotation),
        0.5,
    ]
    chunk = np.zeros((32, 1, 10), dtype=np.float32)
    new_relative = _rotation_z(np.deg2rad(40)) @ _rotation_y(np.deg2rad(25))
    chunk[:, 0, 3:9] = alignment.exp39.canonical_rotation_6d(new_relative)
    previous = compose_absolute_targets(anchor, chunk)
    old_absolute = _rotation_y(np.deg2rad(55)) @ _rotation_x(np.deg2rad(-30))
    previous["rotation"][4] = old_absolute
    transformed, intended, _details = alignment_transform(anchor, previous, chunk)
    encoded_relative = alignment.exp39.rotation_matrix_from_6d(transformed[0, 0, 3:9])
    assert encoded_relative == pytest.approx(anchor_rotation.T @ old_absolute, abs=1e-6)
    assert intended["rotation"][0] == pytest.approx(old_absolute, abs=1e-6)
    assert not np.allclose(encoded_relative, old_absolute @ anchor_rotation.T, atol=1e-3)


def test_realized_metrics_and_baseline_remain_separate_from_transform() -> None:
    baseline = FakeEnv()
    treated = FakeEnv()
    pair = run_alignment_pair(
        {"baseline": baseline, "alignment": treated},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=8,
    )
    assert baseline.applied[4][0] == pytest.approx(_chunk()[0, 0, 0])
    assert treated.applied[4][0] == pytest.approx(_chunk()[4, 0, 0])
    alignment_arm = pair["arms"]["alignment"]
    assert alignment_arm["correction_opportunities"] == 1
    assert alignment_arm["metrics_by_lead"]["pre_translation_m"]["0"]["maximum"] > 0
    assert alignment_arm["metrics_by_lead"]["intended_translation_m"]["0"]["maximum"] <= 1e-6
    assert alignment_arm["metrics_by_lead"]["realized_translation_m"]["0"]["maximum"] <= 1e-6
    assert pair["arms"]["baseline"]["correction_opportunities"] == 0


def test_float32_nonidentity_baseline_correction_metrics_are_exact_zero() -> None:
    chunk = _chunk()
    chunk[:, 0, 3:9] = alignment.exp39.canonical_rotation_6d(
        _rotation_x(0.371) @ _rotation_z(-0.219)
    )
    pair = run_alignment_pair(
        {arm: FakeEnv() for arm in alignment.ARMS},
        FakePolicy(chunk),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=8,
    )
    metrics = pair["arms"]["baseline"]["metrics_by_lead"]
    for name in (
        "correction_translation_m",
        "correction_rotation_deg",
        "correction_gripper_effective",
    ):
        assert all(value["maximum"] == 0.0 for value in metrics[name].values())
    alignment._validate_arm(pair["arms"]["baseline"], "baseline", "push_to_zone")
    assert pair["arms"]["baseline"]["baseline_chunk_equality_checks"] == 1
    assert pair["arms"]["baseline"]["baseline_chunk_equality_passes"] == 1
    assert pair["treatment_onset_policy"]["applicable"] is True
    assert pair["treatment_onset_policy"]["source_arm"] == "baseline"
    assert set(pair["treatment_onset_policy"]["consumed_source_raw_chunk_sha256"].values()) == {
        pair["treatment_onset_policy"]["source_raw_chunk_sha256"]
    }


def test_raw_gripper_reconstruction_and_clipping_metrics_are_recorded_by_lead() -> None:
    chunk = _chunk()
    chunk[0, 0, 9] = 1.5
    pair = run_alignment_pair(
        {arm: FakeEnv(terminal_step=4) for arm in alignment.ARMS},
        FakePolicy(chunk),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=4,
    )
    metrics = pair["arms"]["baseline"]["metrics_by_lead"]
    assert metrics["raw_gripper_out_of_range"]["0"]["maximum"] == pytest.approx(0.5)
    assert metrics["reconstruction_translation_m"]["0"]["count"] == 1
    assert metrics["clipping_error_native"]["0"]["count"] == 1


def test_alignment_transform_failure_is_structured_and_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise AlignmentTreatmentError(
            "transform_threshold_exceeded", details={"test": True}
        )

    monkeypatch.setattr(alignment, "alignment_transform", fail)
    pair = run_alignment_pair(
        {arm: FakeEnv() for arm in alignment.ARMS},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("alignment", "baseline"),
        maximum_steps=8,
    )
    assert pair["arms"]["alignment"]["success"] is None
    assert pair["arms"]["alignment"]["treatment_failure"] is True
    assert pair["arms"]["baseline"]["success"] is False
    shard = _shard_aggregate([{**pair, "task": "push_to_zone"}])
    assert shard["eligible_pairs"] == 0
    assert shard["delivery_gate"]["passed"] is False


def test_alignment_decode_failure_after_valid_transform_is_structured() -> None:
    class FailingAlignmentEnv(FakeEnv):
        def native_action_from_canonical(self, canonical, *, iterations):
            raise ValueError("alignment IK failure")

    pair = run_alignment_pair(
        {"baseline": FakeEnv(), "alignment": FailingAlignmentEnv()},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=8,
    )
    treated = pair["arms"]["alignment"]
    assert treated["treatment_failure"] is True
    assert treated["correction_opportunities"] == 1
    assert treated["tail_unchanged_checks"] == 1
    assert treated["delivered_corrections"] == 0
    alignment._validate_arm(treated, "alignment", "push_to_zone")


def _decode_failure_arm() -> dict:
    class FailingAlignmentEnv(FakeEnv):
        def native_action_from_canonical(self, canonical, *, iterations):
            if len(self.decoded) == 2:
                raise ValueError("alignment IK failure")
            return super().native_action_from_canonical(canonical, iterations=iterations)

    pair = run_alignment_pair(
        {"baseline": FakeEnv(), "alignment": FailingAlignmentEnv()},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=8,
    )
    return pair["arms"]["alignment"]


def _realized_failure_arm() -> dict:
    class FailingAlignmentEnv(FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.realized_calls = 0

        def canonical_arm_action(self, native):
            self.realized_calls += 1
            if self.realized_calls == 7:
                raise ValueError("alignment FK failure")
            return super().canonical_arm_action(native)

    pair = run_alignment_pair(
        {"baseline": FakeEnv(), "alignment": FailingAlignmentEnv()},
        FakePolicy(),
        FakeProcessor(),
        "cpu",
        order=("baseline", "alignment"),
        maximum_steps=8,
    )
    return pair["arms"]["alignment"]


def _transform_failure_arm() -> dict:
    arm = copy.deepcopy(_decode_failure_arm())
    arm["treatment_failure_reason"] = "invalid_policy_chunk"
    arm["treatment_failure_details"].update(
        phase="transform", lead=None, reason="invalid_policy_chunk"
    )
    arm["decoded_commands"] = arm["commands"]
    arm["prepared_commands"] = arm["commands"]
    return arm


def test_structured_failure_rejects_impossible_chunk_index() -> None:
    arm = _decode_failure_arm()
    alignment._validate_arm(arm, "alignment", "push_to_zone")
    arm["treatment_failure_details"]["chunk_index"] += 1
    with pytest.raises(ValueError, match="structured alignment treatment failure"):
        alignment._validate_arm(arm, "alignment", "push_to_zone")


def test_structured_failure_rejects_unregistered_phase_reasons() -> None:
    for arm in (_transform_failure_arm(), _decode_failure_arm(), _realized_failure_arm()):
        alignment._validate_arm(arm, "alignment", "push_to_zone")
        arm["treatment_failure_reason"] = "unregistered_failure"
        arm["treatment_failure_details"]["reason"] = "unregistered_failure"
        with pytest.raises(ValueError, match="structured alignment treatment failure"):
            alignment._validate_arm(arm, "alignment", "push_to_zone")


def test_structured_failure_rejects_phase_impossible_counters() -> None:
    transform = _transform_failure_arm()
    decode = _decode_failure_arm()
    realized = _realized_failure_arm()
    for arm in (transform, decode, realized):
        alignment._validate_arm(arm, "alignment", "push_to_zone")
    transform["decoded_commands"] += 1
    decode["prepared_commands"] += 1
    realized["prepared_commands"] += 1
    for arm in (transform, decode, realized):
        with pytest.raises(ValueError, match="treatment failure counters"):
            alignment._validate_arm(arm, "alignment", "push_to_zone")


def test_report_rejects_lost_onset_milestone() -> None:
    report = _validation_report()
    outcome = report["outcomes"][0]
    name = next(iter(outcome["treatment_onset_snapshot"]["milestones"]))
    for snapshot in (
        outcome["treatment_onset_snapshot"],
        *(outcome["arms"][arm]["treatment_onset_snapshot"] for arm in alignment.ARMS),
    ):
        snapshot["milestones"][name] = {"attained": True, "first_step": 4}
        _rehash_snapshot(snapshot)
    with pytest.raises(ValueError, match="onset milestone is not preserved"):
        alignment._validate_outcomes(report)


def test_report_rejects_backdated_post_onset_milestone() -> None:
    report = _validation_report()
    arm = report["outcomes"][0]["arms"]["alignment"]
    name = next(iter(arm["milestones"]))
    arm["milestones"][name] = {"attained": True, "first_step": 4}
    arm["ordered_progression"] = 1 / len(arm["milestones"])
    with pytest.raises(ValueError, match="post-onset milestone is backdated"):
        alignment._validate_outcomes(report)


def test_report_rejects_onset_milestone_beyond_actual_prefix() -> None:
    report = _validation_report()
    outcome = report["outcomes"][0]
    name = next(iter(outcome["treatment_onset_snapshot"]["milestones"]))
    for snapshot in (
        outcome["treatment_onset_snapshot"],
        *(outcome["arms"][arm]["treatment_onset_snapshot"] for arm in alignment.ARMS),
    ):
        snapshot["milestones"][name] = {"attained": True, "first_step": 5}
        _rehash_snapshot(snapshot)
    with pytest.raises(ValueError, match="treatment-onset milestone value"):
        alignment._validate_outcomes(report)


def test_report_rejects_baseline_first_audit_not_linked_to_onset_source() -> None:
    report = _validation_report()
    baseline = report["outcomes"][0]["arms"]["baseline"]
    record = baseline["baseline_chunk_audit_records"][0]
    record["raw_sha256"] = record["final_sha256"] = "0" * 64
    digest = bytes.fromhex(alignment.EMPTY_OUTCOME_CHAIN_SHA256)
    baseline["baseline_chunk_hash_chain_sha256"] = hashlib.sha256(
        digest + json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="baseline first replan audit"):
        alignment._validate_outcomes(report)


def test_resume_hash_provenance_and_snapshot_mutations(tmp_path: Path) -> None:
    provenance = {"git_dirty": True, "git_commit": "smoke", "evaluator_sha256": "hash"}
    run = {
        "started_utc": "2026-01-01T00:00:00Z",
        "finished_utc": None,
        "device": "cpu",
        "lock_identity": None,
        "lock_path": None,
        "schedule_identity": "smoke-unregistered",
        "schedule_position": None,
    }
    expected = _new_report(
        registered=False,
        contract={"protocol": "frozen"},
        identity={"condition": "joint", "model_seed": 36001, "morphology": "panda"},
        expected_ids=[],
        provenance=provenance,
        run=run,
    )
    output = tmp_path / "resume.json"
    output.write_text(json.dumps(expected))
    changed = copy.deepcopy(expected)
    changed["provenance"] = {**provenance, "evaluator_sha256": "changed"}
    with pytest.raises(ValueError, match="provenance"):
        _load_or_create_report(output, changed)
    expected["outcome_chain_sha256"] = "0" * 64
    output.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match="hash chain"):
        _load_or_create_report(output, expected)
    milestones = {
        name: {"attained": False, "first_step": None}
        for name in alignment.exp39._milestone_names("push_to_zone")
    }
    snapshot = capture_treatment_onset(FakeEnv(), milestones)
    milestones["near_ee_object"] = {"attained": True, "first_step": 1}
    assert snapshot["milestones"]["near_ee_object"]["attained"] is False
    snapshot["scalars"]["dwell"] = 1
    with pytest.raises(ValueError, match="hash"):
        alignment._validate_snapshot(snapshot, "push_to_zone", 4)


def test_runtime_lock_default_staging_filenames_and_frozen_summary(tmp_path: Path) -> None:
    assert validate_exp39_summary(ROOT / "reports/benchmark-exp39-summary.json")["experiment"] == 39
    output = _registered_output("matched", 36003, "panda", None)
    assert output.parent == Path("reports/exp40-runtime")
    assert output.name == "benchmark-exp40-matched-seed36003-panda-alignment.json"
    assert registered_shard_filename("joint", 36001, "so101") == (
        "benchmark-exp40-joint-seed36001-so101-alignment.json"
    )
    assert parse_args(["aggregate"]).shard_dir == Path("reports/exp40-runtime")
    assert "reports/exp40-runtime/" in (ROOT / ".gitignore").read_text()
    lock = tmp_path / "runtime" / "lock"
    with alignment.registered_run_lock(lock):
        with pytest.raises(RuntimeError, match="holds"):
            with alignment.registered_run_lock(lock):
                pass


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


def _successful_synthetic_reports() -> tuple[list[dict], ScenarioManifest, dict, dict, dict[str, str]]:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    manifest = ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )
    exp36 = json.loads((ROOT / "reports/benchmark-exp36-summary.json").read_text())
    exp39 = json.loads((ROOT / "reports/benchmark-exp39-summary.json").read_text())
    templates = {
        (task, source): run_alignment_pair(
            {arm: FakeEnv(task=task) for arm in alignment.ARMS},
            FakePolicy(),
            FakeProcessor(),
            "cpu",
            order=(source, "alignment" if source == "baseline" else "baseline"),
            maximum_steps=8,
        )
        for task in alignment.TASKS
        for source in alignment.ARMS
    }
    runtime = _synthetic_runtime()
    runtime_signature = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evaluator_sha256 = file_sha256(Path(alignment.__file__))
    reports = []
    source_hashes = {}
    for position, (condition, seed, morphology) in enumerate(alignment.REGISTERED_SCHEDULE):
        scenarios = alignment_scenarios(manifest, morphology, registered=True)
        outcomes = []
        for ordinal, scenario in enumerate(scenarios):
            order = pair_arm_order(ordinal, condition, seed, morphology)
            outcome = copy.deepcopy(templates[(scenario.task, order[0])])
            treated = outcome["arms"]["alignment"]
            for milestone in treated["milestones"].values():
                milestone.update(attained=True, first_step=5)
            treated["ordered_progression"] = 1.0
            if ordinal < 3:
                treated["success"] = True
                treated["final_stage"] = "success"
                treated["failure_reason"] = None
            outcome.update(
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
            outcomes.append(outcome)
        source_condition = morphology if condition == "matched" else "joint"
        identity = {
            "condition": condition,
            "source_condition": source_condition,
            "model_seed": seed,
            "morphology": morphology,
            "core_sha256": exp36["runs"][f"{source_condition}-seed{seed}"]["core_sha256"],
            "config_sha256": alignment.CONFIG_SHA256[morphology],
            "data_manifest_sha256": alignment.DATA_MANIFEST_SHA256[source_condition],
            "exp36_summary_sha256": alignment.EXP36_SUMMARY_SHA256,
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
            "lock_identity": alignment.LOCK_IDENTITY,
            "lock_path": str(alignment.REGISTERED_RUNTIME_DIR / alignment.REGISTERED_LOCK_NAME),
            "schedule_identity": alignment.SCHEDULE_IDENTITY,
            "schedule_position": position,
        }
        contract = {
            "definition_sha256": alignment.DEFINITION_SHA256,
            "manifest_sha256": alignment.DEVELOPMENT_MANIFEST_SHA256,
            "exp36_summary_sha256": alignment.EXP36_SUMMARY_SHA256,
            "exp39_summary_sha256": alignment.EXP39_SUMMARY_SHA256,
            "exp39_evaluator_sha256": alignment.EXP39_EVALUATOR_SHA256,
            "split": "development",
            "scenario_indices": list(alignment.SCENARIO_INDICES),
            "tasks": list(alignment.TASKS),
            "limit_scenarios": None,
            "protocol": alignment.ALIGNMENT_PROTOCOL,
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
        report["aggregate"] = _shard_aggregate(outcomes)
        report["complete"] = True
        report["status"] = "completed"
        reports.append(report)
        label = f"{condition}/seed{seed}/{morphology}"
        source_hashes[label] = hashlib.sha256(label.encode()).hexdigest()
    return reports, manifest, exp36, exp39, source_hashes


def test_full_synthetic_aggregate_support_corroboration_and_delivery_failure() -> None:
    reports, manifest, exp36, exp39, source_hashes = _successful_synthetic_reports()
    summary = aggregate_registered_shards(
        reports,
        manifest,
        exp36_summary=exp36,
        exp39_summary=exp39,
        source_hashes=source_hashes,
    )
    assert summary["contract"]["pairs"] == 252
    assert summary["source_shard_sha256"] == dict(sorted(source_hashes.items()))
    assert summary["realized_continuity_corroboration"]["joint"]["corroborated"] is True
    assert summary["causal_support_by_condition"]["joint"]["supported"] is True
    assert summary["causal_support_by_condition"]["matched"]["supported"] is True

    failed = copy.deepcopy(reports)
    outcome = failed[0]["outcomes"][0]
    treated = outcome["arms"]["alignment"]
    treated.update(
        success=None,
        failure_reason=None,
        chunks=3,
        correction_opportunities=treated["correction_opportunities"] + 1,
        treatment_failure=True,
        treatment_failure_reason="transform_threshold_exceeded",
        treatment_failure_details={
            "phase": "transform",
            "chunk_index": 2,
            "lead": None,
            "reason": "transform_threshold_exceeded",
            "details": {"test": True},
        },
    )
    failed[0]["treatment_delivery_errors"] = [
        {
            "scenario_id": outcome["scenario_id"],
            "arm": "alignment",
            "reason": "transform_threshold_exceeded",
            "details": treated["treatment_failure_details"],
        }
    ]
    failed[0]["outcome_chain_sha256"] = _outcome_chain(failed[0]["outcomes"])
    failed[0]["aggregate"] = _shard_aggregate(failed[0]["outcomes"])
    failed_summary = aggregate_registered_shards(
        failed,
        manifest,
        exp36_summary=exp36,
        exp39_summary=exp39,
        source_hashes=source_hashes,
    )
    assert failed_summary["delivery_gates_by_shard"]["joint/seed36001/so101"][
        "passed"
    ] is False
    assert failed_summary["causal_support_by_condition"]["joint"]["supported"] is False
