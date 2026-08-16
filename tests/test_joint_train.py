import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from embodi import EmbodiConfig
from embodi.joint_train import (
    balanced_cells,
    canonical_only_batch,
    cell_loader_seed,
    cell_schedule,
    deduplicated_trainable_parameters,
    episode_indices_by_task,
    file_sha256,
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
    with pytest.raises(ValueError, match="divisible"):
        cell_schedule("joint", 31)


def test_cell_loader_seeds_do_not_depend_on_condition() -> None:
    assert cell_loader_seed(36101, "so101", "push_to_zone") == 36101
    assert cell_loader_seed(36101, "panda", "push_to_zone") == 36104


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
