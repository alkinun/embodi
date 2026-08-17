from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import SequentialSampler, Subset

from embodi import EmbodiConfig
from embodi import joint_train
from embodi.joint_train import (
    MORPHOLOGIES,
    TASKS,
    evaluate_cells,
    load_benchmark_metadata,
    make_policies,
    make_train_loaders,
    make_validation_loaders,
    prepare_validation_manifest,
    runtime_provenance,
    validate_validation_manifest,
)


def _configs() -> dict[str, EmbodiConfig]:
    root = Path(__file__).parents[1] / "configs"
    return {
        "so101": EmbodiConfig.load(root / "so101-exp0-top.json"),
        "panda": EmbodiConfig.load(root / "panda-exp0-top.json"),
    }


class _FakeDataset:
    instances: list["_FakeDataset"] = []

    def __init__(self, repo_id: str, *, root: Path, **kwargs) -> None:
        self.repo_id = repo_id
        self.root = root
        self.kwargs = kwargs
        self.instances.append(self)

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


class _FakeMetadata:
    instances: list["_FakeMetadata"] = []

    def __init__(self, repo_id: str, *, root: Path) -> None:
        self.repo_id = repo_id
        self.root = root
        self.fps = 30
        self.episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 4},
        ]
        self.instances.append(self)


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDataset.instances.clear()
    _FakeMetadata.instances.clear()
    monkeypatch.setattr(joint_train, "_lerobot_imports", lambda: (_FakeDataset, _FakeMetadata))


def test_load_benchmark_metadata_validates_and_returns_both_lineages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lerobot(monkeypatch)
    validations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        joint_train,
        "validate_lerobot_training_dataset",
        lambda metadata, config, stage, benchmark_split: validations.append(
            (metadata.repo_id, benchmark_split)
        ),
    )
    monkeypatch.setattr(
        joint_train,
        "validate_benchmark_dataset_lineage",
        lambda metadata, split: {"repo_id": metadata.repo_id, "split": split},
    )
    roots = {morphology: tmp_path / morphology for morphology in MORPHOLOGIES}
    repo_ids = {morphology: f"embodi/{morphology}" for morphology in MORPHOLOGIES}

    metadata, lineage = load_benchmark_metadata(roots, repo_ids, _configs(), "validation")

    assert set(metadata) == set(MORPHOLOGIES)
    assert lineage == {
        morphology: {"repo_id": repo_ids[morphology], "split": "validation"}
        for morphology in MORPHOLOGIES
    }
    assert validations == [(repo_ids[morphology], "validation") for morphology in MORPHOLOGIES]


def test_load_benchmark_metadata_requires_verified_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lerobot(monkeypatch)
    monkeypatch.setattr(joint_train, "validate_lerobot_training_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(joint_train, "validate_benchmark_dataset_lineage", lambda *args: None)
    roots = {morphology: tmp_path / morphology for morphology in MORPHOLOGIES}
    repo_ids = {morphology: f"embodi/{morphology}" for morphology in MORPHOLOGIES}

    with pytest.raises(ValueError, match="requires verified benchmark lineage"):
        load_benchmark_metadata(roots, repo_ids, _configs(), "train")


def test_make_train_loaders_deduplicates_cells_and_binds_episode_offsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lerobot(monkeypatch)
    monkeypatch.setattr(
        joint_train,
        "episode_indices_by_task",
        lambda root, morphology: {
            "push_to_zone": [1, 2],
            "lift_object": [3],
            "pick_place_bin": [4],
        },
    )
    configs = _configs()
    configs = {
        morphology: replace(config, chunk_size=4, n_action_steps=4)
        for morphology, config in configs.items()
    }
    metadata = {
        morphology: SimpleNamespace(fps=20)
        for morphology in MORPHOLOGIES
    }
    roots = {morphology: tmp_path / morphology for morphology in MORPHOLOGIES}
    repo_ids = {morphology: f"embodi/{morphology}" for morphology in MORPHOLOGIES}
    cells = (
        ("so101", "push_to_zone"),
        ("so101", "push_to_zone"),
        ("panda", "lift_object"),
    )

    loaders = make_train_loaders(
        roots,
        repo_ids,
        metadata,
        configs,
        cells,
        loader_seed=36101,
        workers=0,
    )

    assert set(loaders) == {
        ("so101", "push_to_zone"),
        ("panda", "lift_object"),
    }
    assert len(_FakeDataset.instances) == 2
    so101_dataset = loaders[("so101", "push_to_zone")].dataset
    assert isinstance(so101_dataset, _FakeDataset)
    assert so101_dataset.kwargs == {
        "episodes": [1, 2],
        "delta_timestamps": {
            "canonical_action": [0.0, 0.05, 0.1, 0.15],
            "canonical_state": [0.0, 0.05, 0.1, 0.15],
        },
    }


def _validation_manifest(roots: dict[str, Path], repo_ids: dict[str, str]) -> dict:
    return {
        "format_version": 1,
        "benchmark_id": "embodi-sim-v1",
        "seed": 11,
        "samples_per_cell": 2,
        "sources": {
            morphology: {
                "root": str(roots[morphology]),
                "repo_id": repo_ids[morphology],
                "generation_manifest_sha256": f"hash-{roots[morphology].name}",
            }
            for morphology in MORPHOLOGIES
        },
        "cells": {
            f"{morphology}/{task}": [0, 1]
            for morphology in MORPHOLOGIES
            for task in TASKS
        },
    }


def test_prepare_validation_manifest_is_deterministic_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lerobot(monkeypatch)
    monkeypatch.setattr(
        joint_train,
        "episode_indices_by_task",
        lambda root, morphology: {task: [0] for task in TASKS},
    )
    monkeypatch.setattr(
        joint_train,
        "generation_manifest_sha256",
        lambda root: f"hash-{root.name}",
    )
    roots = {morphology: tmp_path / f"{morphology}-root" for morphology in MORPHOLOGIES}
    repo_ids = {morphology: f"embodi/{morphology}" for morphology in MORPHOLOGIES}
    output = tmp_path / "nested" / "validation.json"

    first = prepare_validation_manifest(
        roots,
        repo_ids,
        output,
        samples_per_cell=2,
        seed=11,
    )
    second = prepare_validation_manifest(
        roots,
        repo_ids,
        output,
        samples_per_cell=2,
        seed=11,
    )

    assert first == second
    assert output.is_file()
    assert all(len(indices) == 2 and indices == sorted(indices) for indices in first["cells"].values())
    with pytest.raises(FileExistsError, match="refusing to replace different"):
        prepare_validation_manifest(
            roots,
            repo_ids,
            output,
            samples_per_cell=1,
            seed=11,
        )


def test_validate_validation_manifest_checks_sources_cells_and_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lerobot(monkeypatch)
    monkeypatch.setattr(
        joint_train,
        "episode_indices_by_task",
        lambda root, morphology: {task: [0] for task in TASKS},
    )
    monkeypatch.setattr(
        joint_train,
        "generation_manifest_sha256",
        lambda root: f"hash-{root.name}",
    )
    roots = {morphology: tmp_path / f"{morphology}-root" for morphology in MORPHOLOGIES}
    repo_ids = {morphology: f"embodi/{morphology}" for morphology in MORPHOLOGIES}
    path = tmp_path / "validation.json"
    value = _validation_manifest(roots, repo_ids)
    path.write_text(json.dumps(value))

    assert validate_validation_manifest(path, roots, repo_ids, registered=False) == value

    duplicate = json.loads(path.read_text())
    duplicate["cells"]["so101/push_to_zone"] = [0, 0]
    path.write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match="unique nonnegative"):
        validate_validation_manifest(path, roots, repo_ids, registered=False)

    outside = json.loads(path.read_text())
    outside["cells"]["so101/push_to_zone"] = [0, 99]
    path.write_text(json.dumps(outside))
    with pytest.raises(ValueError, match="outside its registered task cell"):
        validate_validation_manifest(path, roots, repo_ids, registered=False)


def test_make_validation_loaders_use_nonshuffled_task_subsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lerobot(monkeypatch)
    configs = _configs()
    metadata = {morphology: SimpleNamespace(fps=30) for morphology in MORPHOLOGIES}
    roots = {morphology: tmp_path / morphology for morphology in MORPHOLOGIES}
    repo_ids = {morphology: f"embodi/{morphology}" for morphology in MORPHOLOGIES}
    manifest = {
        "cells": {
            f"{morphology}/{task}": [0, 2]
            for morphology in MORPHOLOGIES
            for task in TASKS
        }
    }

    loaders = make_validation_loaders(
        roots,
        repo_ids,
        metadata,
        configs,
        manifest,
        workers=0,
    )

    assert len(loaders) == len(MORPHOLOGIES) * len(TASKS)
    for loader in loaders.values():
        assert isinstance(loader.dataset, Subset)
        assert loader.dataset.indices == [0, 2]
        assert isinstance(loader.sampler, SequentialSampler)


class _FakeEvaluationPolicy:
    def __init__(self, objective: str, value: float) -> None:
        self.config = SimpleNamespace(core_objective=objective, camera_names=("top",))
        self.value = value
        self.evaluated = False

    def eval(self) -> "_FakeEvaluationPolicy":
        self.evaluated = True
        return self

    def __call__(self, _batch):
        metric_name = "regression_loss" if self.config.core_objective == "regression" else "flow_loss"
        return None, {metric_name: self.value}


def test_evaluate_cells_aggregates_task_and_morphology_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(joint_train, "prepare_model_batch", lambda raw, *args: raw)
    policies = {
        "so101": _FakeEvaluationPolicy("regression", 1.0),
        "panda": _FakeEvaluationPolicy("flow", 3.0),
    }
    loaders = {
        (morphology, task): [{"canonical_action": 0}]
        for morphology in MORPHOLOGIES
        for task in TASKS
    }

    results = evaluate_cells(policies, loaders, processor=None, device=torch.device("cpu"))

    assert all(policy.evaluated for policy in policies.values())
    assert results["macro/so101"] == 1.0
    assert results["macro/panda"] == 3.0
    assert results["macro/all"] == 2.0
    assert all(results[f"{morphology}/{task}"] == (1.0 if morphology == "so101" else 3.0) for morphology, task in loaders)


def test_evaluate_cells_rejects_empty_validation_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(joint_train, "prepare_model_batch", lambda raw, *args: raw)
    policies = {
        "so101": _FakeEvaluationPolicy("regression", 1.0),
        "panda": _FakeEvaluationPolicy("flow", 3.0),
    }
    loaders = {
        (morphology, task): [{"canonical_action": 0}]
        for morphology in MORPHOLOGIES
        for task in TASKS
    }
    loaders["so101", "push_to_zone"] = []

    with pytest.raises(RuntimeError, match="validation cell so101/push_to_zone"):
        evaluate_cells(policies, loaders, processor=None, device=torch.device("cpu"))


def test_runtime_provenance_records_clean_and_dirty_worktrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {"output": ""}

    def fake_run(command, **_kwargs):
        if command[1] == "rev-parse":
            return SimpleNamespace(stdout="commit-sha\n")
        return SimpleNamespace(stdout=status["output"])

    monkeypatch.setattr(joint_train.subprocess, "run", fake_run)
    monkeypatch.setattr(joint_train, "version", lambda package: f"{package}-version")
    monkeypatch.setattr(joint_train, "file_sha256", lambda path: "trainer-sha")

    clean = runtime_provenance(require_clean=True)
    assert clean["git_commit"] == "commit-sha"
    assert clean["git_dirty"] is False
    assert clean["lerobot"] == "lerobot-version"
    assert clean["trainer_sha256"] == "trainer-sha"

    status["output"] = " M experiment.py\n"
    dirty = runtime_provenance(require_clean=False)
    assert dirty["git_dirty"] is True
    with pytest.raises(RuntimeError, match="clean git worktree"):
        runtime_provenance(require_clean=True)


def test_make_policies_shares_core_and_freezes_embodiment_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePolicy:
        instances: list["FakePolicy"] = []

        def __init__(self, config: EmbodiConfig, core=None) -> None:
            self.config = config
            self.core = core if core is not None else object()
            self.state_adapter = nn.Linear(1, 1)
            self.action_decoder = nn.Linear(1, 1)
            self.stage = None
            self.instances.append(self)

        def to(self, device):
            del device
            return self

        def configure_stage(self, stage: str) -> None:
            self.stage = stage

    monkeypatch.setattr(joint_train, "EmbodiPolicy", FakePolicy)

    policies = make_policies(_configs(), torch.device("cpu"))

    assert policies["so101"].core is policies["panda"].core
    assert all(policy.stage == "core" for policy in policies.values())
    assert all(
        not parameter.requires_grad
        for policy in policies.values()
        for module in (policy.state_adapter, policy.action_decoder)
        for parameter in module.parameters()
    )
