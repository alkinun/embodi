import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from embodi import EmbodiConfig
from embodi.joint_train import (
    balanced_cells,
    backward_scaled,
    batch_sample_index,
    canonical_only_batch,
    cell_loader_seed,
    cell_schedule,
    deduplicated_trainable_parameters,
    episode_indices_by_task,
    file_sha256,
    lead_zero_gripper_metrics,
    optimization_loss_denominator,
    save_policies,
    validate_registered_protocol,
    validate_joint_configs,
)


def test_joint_schedule_balances_cells_at_matched_exposure() -> None:
    specialist = cell_schedule("so101", 30)
    joint = cell_schedule("joint", 30)
    assert len(specialist) == len(joint) == 30
    assert all(specialist.count(cell) == 10 for cell in balanced_cells("so101"))
    assert all(joint.count(cell) == 5 for cell in balanced_cells("joint"))
    half_specialist = cell_schedule("so101", 15)
    assert all(half_specialist.count(cell) == 5 for cell in balanced_cells("so101"))
    assert optimization_loss_denominator("exp36", len(specialist)) == 30
    assert optimization_loss_denominator("exp43-half-specialist", len(half_specialist)) == 30
    with pytest.raises(ValueError, match="divisible"):
        cell_schedule("joint", 31)


def test_registered_denominator_controls_accumulated_gradient_coefficient() -> None:
    half_parameter = torch.tensor(1.0, requires_grad=True)
    for _ in range(15):
        backward_scaled(2 * half_parameter, 30)
    assert half_parameter.grad is not None
    assert half_parameter.grad.item() == pytest.approx(1.0)

    full_parameter = torch.tensor(1.0, requires_grad=True)
    for _ in range(30):
        backward_scaled(2 * full_parameter, 30)
    assert full_parameter.grad is not None
    assert full_parameter.grad.item() == pytest.approx(2.0)


def test_cell_loader_seeds_do_not_depend_on_condition() -> None:
    assert cell_loader_seed(36101, "so101", "push_to_zone") == 36101
    assert cell_loader_seed(36101, "panda", "push_to_zone") == 36104


def test_batch_sample_index_requires_one_dataset_index() -> None:
    assert batch_sample_index({"index": torch.tensor([17])}) == 17
    with pytest.raises(ValueError, match="one dataset index"):
        batch_sample_index({"index": torch.tensor([1, 2])})


def test_lead_zero_gripper_metrics_report_raw_effective_and_saturation() -> None:
    metrics = lead_zero_gripper_metrics(
        torch.tensor([-0.2, 0.8, 1.4]),
        torch.tensor([0.1, 0.5, 1.0]),
    )
    assert metrics["normalized_channel9_loss"] == pytest.approx(
        ((-0.3 / 0.5) ** 2 + (0.3 / 0.5) ** 2 + (0.4 / 0.5) ** 2) / 3
    )
    assert metrics["raw_absolute_error"] == pytest.approx(1 / 3)
    assert metrics["signed_bias"] == pytest.approx(0.4 / 3)
    assert metrics["effective_error"] == pytest.approx(0.4 / 3)
    assert metrics["saturation_rate"] == pytest.approx(2 / 3)


def test_deduplicated_parameters_include_shared_module_once() -> None:
    shared = nn.Linear(3, 4)
    first = nn.Sequential(shared, nn.Linear(4, 2))
    second = nn.Sequential(shared, nn.Linear(4, 1))
    parameters = deduplicated_trainable_parameters(first, second)
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert sum(parameter is shared.weight for parameter in parameters) == 1


def test_canonical_only_batch_removes_native_inputs() -> None:
    raw = {
        "observation.state": torch.ones(1, 6),
        "action": torch.ones(1, 32, 6),
        "action_is_pad": torch.zeros(1, 32, dtype=torch.bool),
        "canonical_action": torch.ones(1, 32, 1, 10),
    }
    value = canonical_only_batch(raw)
    assert value["observation.state"] is None
    assert value["action"] is None
    assert "action_is_pad" not in value
    assert raw["observation.state"] is not None


def test_panda_and_so101_configs_share_core_behavior() -> None:
    root = Path(__file__).parents[1]
    configs = {
        "so101": EmbodiConfig.load(root / "configs" / "so101-exp0-top.json"),
        "panda": EmbodiConfig.load(root / "configs" / "panda-exp0-top.json"),
    }
    validate_joint_configs(configs)
    assert configs["so101"].core_signature() == configs["panda"].core_signature()
    assert configs["so101"].state_dim == 6
    assert configs["panda"].state_dim == 8
    args = SimpleNamespace(
        smoke_test=False,
        protocol="exp36",
        condition="so101",
        steps=1200,
        presentations_per_step=30,
        lr=1e-4,
        warmup_steps=120,
        workers=0,
        deterministic=True,
        log_every=20,
        model_seed=36001,
        loader_seed=36101,
    )
    validate_registered_protocol(args, configs)
    args.steps = 1199
    with pytest.raises(ValueError, match="steps"):
        validate_registered_protocol(args, configs)

    args.steps = 1200
    args.protocol = "exp43-half-specialist"
    args.presentations_per_step = 15
    validate_registered_protocol(args, configs)
    args.condition = "joint"
    with pytest.raises(ValueError, match="specialist"):
        validate_registered_protocol(args, configs)
    args.condition = "so101"
    args.presentations_per_step = 30
    with pytest.raises(ValueError, match="presentations-per-step"):
        validate_registered_protocol(args, configs)


def test_frozen_validation_frames_are_balanced_and_hash_bound() -> None:
    root = Path(__file__).parents[1]
    path = root / "reports" / "benchmark-exp36-validation-frames.json"
    values = json.loads(path.read_text())
    assert file_sha256(path) == "32681217e8d6f4142ce754b18b2a3a69527edea9abce3506fb55e7d25c28cc1a"
    assert values["samples_per_cell"] == 256
    assert set(values["cells"]) == {
        f"{morphology}/{task}"
        for morphology in ("so101", "panda")
        for task in ("push_to_zone", "lift_object", "pick_place_bin")
    }
    assert all(len(indices) == len(set(indices)) == 256 for indices in values["cells"].values())


def test_exp36_report_recomputes_registered_gate() -> None:
    root = Path(__file__).parents[1]
    report = json.loads((root / "reports" / "benchmark-exp36-summary.json").read_text())
    matched = []
    opposite = []
    joint = []
    for seed in report["model_seeds"]:
        so101 = report["runs"][f"so101-seed{seed}"]["validation"]
        panda = report["runs"][f"panda-seed{seed}"]["validation"]
        joint_metrics = report["runs"][f"joint-seed{seed}"]["validation"]
        matched.append((so101["macro/so101"] + panda["macro/panda"]) / 2)
        opposite.append((so101["macro/panda"] + panda["macro/so101"]) / 2)
        joint.append(joint_metrics["macro/all"])
        gate = report["paired_gate"]["seeds"][str(seed)]
        assert gate["matched_specialist_ensemble"] == pytest.approx(matched[-1])
        assert gate["opposite_morphology_specialist_transfer"] == pytest.approx(opposite[-1])
        assert gate["joint"] == pytest.approx(joint[-1])
        relative_regression = joint[-1] / matched[-1] - 1
        transfer_reduction = 1 - joint[-1] / opposite[-1]
        assert gate["joint_relative_to_matched_specialists"] == pytest.approx(relative_regression)
        assert gate["joint_reduction_from_opposite_transfer"] == pytest.approx(transfer_reduction)
        passed = relative_regression <= 0.1 and transfer_reduction >= 0.1
        assert gate["passed_both"] is passed

    def mean(values):
        return sum(values) / len(values)

    aggregates = report["aggregates"]
    assert aggregates["matched_specialist_ensemble_mean"] == pytest.approx(mean(matched))
    assert aggregates["opposite_morphology_specialist_transfer_mean"] == pytest.approx(mean(opposite))
    assert aggregates["joint_macro_all_mean"] == pytest.approx(mean(joint))
    mean_margin_passed = mean(joint) <= 1.1 * mean(matched)
    mean_transfer_passed = mean(joint) <= 0.9 * mean(opposite)
    gate = report["paired_gate"]
    assert gate["mean_joint_within_specialist_margin"] is mean_margin_passed
    assert gate["mean_transfer_improvement_met"] is mean_transfer_passed
    assert gate["paired_seeds_passing_both"] == sum(
        value["passed_both"] for value in gate["seeds"].values()
    )
    assert gate["passed"] is (
        mean_margin_passed
        and mean_transfer_passed
        and gate["paired_seeds_passing_both"] >= gate["minimum_paired_seeds_passing_both"]
    )


def test_exp36_report_matches_local_source_metrics() -> None:
    root = Path(__file__).parents[1]
    report = json.loads((root / "reports" / "benchmark-exp36-summary.json").read_text())
    paths = [root / run["output"] for run in report["runs"].values()]
    if not all(path.is_dir() for path in paths):
        pytest.skip("local experiment 036 outputs are unavailable")

    for run, output in zip(report["runs"].values(), paths, strict=True):
        metrics_path = output / "metrics.jsonl"
        manifest_path = output / "run_manifest.json"
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        final_train = next(value for value in records if value["step"] == 1200 and value["split"] == "train")
        validation = next(value for value in records if value["step"] == 1200 and value["split"] == "validation")
        assert file_sha256(metrics_path) == run["metrics_sha256"]
        assert file_sha256(manifest_path) == run["run_manifest_sha256"]
        assert final_train["metrics"]["regression_loss"] == run["final_train_regression_loss"]
        assert final_train["metrics"]["grad_norm"] == run["final_gradient_norm"]
        assert validation["metrics"] == run["validation"]
        assert (output / "final" / "so101" / "core.pt").stat().st_ino == (
            output / "final" / "panda" / "core.pt"
        ).stat().st_ino


def test_episode_mapping_groups_all_three_tasks(tmp_path: Path) -> None:
    metadata = tmp_path / "so101" / "meta"
    metadata.mkdir(parents=True)
    metadata.joinpath("benchmark.json").write_text(
        json.dumps(
            {
                "morphology": "so101",
                "episode_scenarios": [
                    {"episode_index": 0, "scenario_id": "train-so101-push_to_zone-0000"},
                    {"episode_index": 1, "scenario_id": "train-so101-lift_object-0000"},
                    {"episode_index": 2, "scenario_id": "train-so101-pick_place_bin-0000"},
                ],
            }
        )
    )
    assert episode_indices_by_task(tmp_path, "so101") == {
        "push_to_zone": [0],
        "lift_object": [1],
        "pick_place_bin": [2],
    }


def test_joint_checkpoint_hard_links_shared_core(tmp_path: Path) -> None:
    class FakePolicy:
        def save_checkpoint(self, directory, *_args):
            directory.mkdir(parents=True)
            (directory / "core.pt").write_bytes(b"shared")
            (directory / "embodiment.pt").write_bytes(b"specific")

    save_policies(
        tmp_path,
        {"so101": FakePolicy(), "panda": FakePolicy()},
        optimizer=None,
        scheduler=None,
        step=1,
        data_report={"format_version": 1},
    )
    so101 = tmp_path / "so101" / "core.pt"
    panda = tmp_path / "panda" / "core.pt"
    assert so101.read_bytes() == panda.read_bytes() == b"shared"
    assert so101.stat().st_ino == panda.stat().st_ino
