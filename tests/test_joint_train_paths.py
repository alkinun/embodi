from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from embodi import EmbodiConfig
from embodi.joint_train import (
    CONDITIONS,
    MORPHOLOGIES,
    TASKS,
    balanced_cells,
    cell_schedule,
    deduplicated_trainable_parameters,
    episode_indices_by_task,
    generation_manifest_sha256,
    positive_int,
    train_joint,
    validate_joint_configs,
    validate_registered_protocol,
)


def _configs() -> dict[str, EmbodiConfig]:
    root = Path(__file__).parents[1] / "configs"
    return {
        "so101": EmbodiConfig.load(root / "so101-exp0-top.json"),
        "panda": EmbodiConfig.load(root / "panda-exp0-top.json"),
    }


def test_positive_int_accepts_positive_values_and_rejects_nonpositive() -> None:
    assert positive_int("7") == 7
    for value in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError, match="must be positive"):
            positive_int(value)


def test_generation_manifest_hash_requires_manifest_file(tmp_path: Path) -> None:
    manifest = tmp_path / "benchmark-dataset.json"
    manifest.write_text("manifest")
    assert generation_manifest_sha256(tmp_path) == hashlib.sha256(b"manifest").hexdigest()

    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="benchmark dataset is missing"):
        generation_manifest_sha256(tmp_path)


def _write_episode_metadata(root: Path, morphology: str, scenario_ids: list[str]) -> None:
    metadata = root / morphology / "meta"
    metadata.mkdir(parents=True)
    (metadata / "benchmark.json").write_text(
        json.dumps(
            {
                "morphology": morphology,
                "episode_scenarios": [
                    {"episode_index": index, "scenario_id": scenario_id}
                    for index, scenario_id in enumerate(scenario_ids)
                ],
            }
        )
    )


def test_episode_indices_by_task_groups_scenarios(tmp_path: Path) -> None:
    _write_episode_metadata(
        tmp_path,
        "so101",
        [
            "train-so101-push_to_zone-0000",
            "train-so101-lift_object-0000",
            "train-so101-pick_place_bin-0000",
        ],
    )
    assert episode_indices_by_task(tmp_path, "so101") == {
        "push_to_zone": [0],
        "lift_object": [1],
        "pick_place_bin": [2],
    }


@pytest.mark.parametrize(
    ("scenario_ids", "message"),
    [
        (["train-panda-push_to_zone-0000"], "morphology does not match"),
        (["train-so101-push_to_zone-lift_object-0000"], "cannot identify benchmark task"),
        (["train-so101-push_to_zone-0000", "train-so101-lift_object-0000"], "every benchmark task"),
    ],
)
def test_episode_indices_by_task_rejects_invalid_metadata(
    tmp_path: Path,
    scenario_ids: list[str],
    message: str,
) -> None:
    _write_episode_metadata(tmp_path, "so101", scenario_ids)
    if "morphology" in message:
        metadata_path = tmp_path / "so101" / "meta" / "benchmark.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["morphology"] = "panda"
        metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=message):
        episode_indices_by_task(tmp_path, "so101")


def test_balanced_cells_supports_specialist_and_joint_conditions() -> None:
    assert set(CONDITIONS) == {"so101", "panda", "joint"}
    assert balanced_cells("panda") == tuple(("panda", task) for task in TASKS)
    assert balanced_cells("joint") == tuple(
        (morphology, task) for morphology in MORPHOLOGIES for task in TASKS
    )

    with pytest.raises(ValueError, match="unknown joint-core condition"):
        balanced_cells("unknown")


@pytest.mark.parametrize("presentations", [0, -1, 4])
def test_cell_schedule_requires_positive_cell_divisible_presentations(presentations: int) -> None:
    with pytest.raises(ValueError, match="positive and divisible"):
        cell_schedule("so101", presentations)


def test_deduplicated_trainable_parameters_omit_frozen_parameters() -> None:
    trainable = nn.Parameter(torch.ones(2))
    frozen = nn.Parameter(torch.ones(2), requires_grad=False)
    first = nn.ParameterList([trainable, frozen])
    second = nn.ParameterList([trainable])

    parameters = deduplicated_trainable_parameters(first, second)

    assert parameters == [trainable]


@pytest.mark.parametrize("mutation", ["missing", "behavior", "so101_width", "panda_width"])
def test_validate_joint_configs_rejects_incompatible_configurations(mutation: str) -> None:
    configs = _configs()
    if mutation == "missing":
        configs.pop("panda")
    elif mutation == "behavior":
        configs["panda"] = replace(configs["panda"], image_do_rescale=True)
    elif mutation == "so101_width":
        configs["so101"] = replace(
            configs["so101"],
            state_dim=7,
            action_dim=7,
            action_group_state_dims=(7,),
            action_group_native_dims=(7,),
        )
    else:
        configs["panda"] = replace(
            configs["panda"],
            state_dim=7,
            action_dim=7,
            action_group_state_dims=(7,),
            action_group_native_dims=(7,),
        )

    with pytest.raises(ValueError, match="joint|native width"):
        validate_joint_configs(configs)


def test_registered_protocol_is_bypassed_for_smoke_tests() -> None:
    args = SimpleNamespace(
        smoke_test=True,
        steps=1,
        presentations_per_step=1,
        lr=0.5,
        warmup_steps=0,
        workers=4,
        deterministic=False,
        log_every=1,
        model_seed=1,
        loader_seed=2,
    )

    validate_registered_protocol(args, _configs())


@pytest.mark.parametrize(
    ("steps", "presentations_per_step", "warmup_steps"),
    [(0, 1, 0), (1, 0, 0), (1, 1, -1)],
)
def test_train_joint_rejects_invalid_early_arguments(
    steps: int,
    presentations_per_step: int,
    warmup_steps: int,
) -> None:
    args = SimpleNamespace(
        steps=steps,
        presentations_per_step=presentations_per_step,
        warmup_steps=warmup_steps,
    )

    with pytest.raises(ValueError, match="steps and presentations"):
        train_joint(args)
