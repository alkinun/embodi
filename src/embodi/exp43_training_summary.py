from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .processing import EmbodiProcessor
from .joint_train import (
    MORPHOLOGIES,
    TASKS,
    TRAIN_GENERATION_SHA256,
    VALIDATION_FRAMES_SHA256,
    VALIDATION_GENERATION_SHA256,
    evaluate_cells,
    file_sha256,
    load_benchmark_metadata,
    make_policies,
    make_validation_loaders,
    validate_validation_manifest,
)
from .record_so101 import atomic_write_json
from .token_config import EmbodiConfig


EXP36_SUMMARY_SHA256 = "76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f"
EXP42_SUMMARY_SHA256 = "104c16ca2552eb0d2ba7efc65c9228eeecd488d67d765f49d70931aa0d15abf8"
TRAINER_SHA256 = "e6dbf985ea4b79c7cdf5d2bc0bf6c5fa17186f82f8e8297d5454fbb501cba9a3"
DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
TRAIN_MANIFEST_SHA256 = "0466044de94c55adf53d20de54a5b44f3f64490303055fcde83ed51aa3bc1154"
VALIDATION_MANIFEST_SHA256 = "09d38b323b90b7a7d5e4c78dab2016c1443767cdceb4f1811ab3b8ee3b55a4ae"
DEVELOPMENT_MANIFEST_SHA256 = "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
EXP42_EVALUATOR_SHA256 = "cc4fb718a3f6341dbaaea0026aca193fd86d5b84412ea7061204ed8f4ce24b7d"
EXPERT_SHA256 = "7fb869a10756864530889f809a8305ee1758bdc51eaeeeda0ff2f451f3b7f8e9"
CONFIG_SHA256 = {
    "so101": "a906d494a8f7955de536098cf873efcdeb6566de84f245bc248dea03104dda65",
    "panda": "2a292fb0153d54efd913e4ac77a0d944d7ca7cb006d613375d9c6070fcd4d4a2",
}
SEEDS = (36001, 36002, 36003)
STEPS = 1200
PRESENTATIONS_PER_STEP = 15
LOSS_DENOMINATOR = 30
SAMPLES_PER_CELL = 6000
PROTOCOL = "exp43-half-specialist"
SUMMARY_PATH = Path("reports/benchmark-exp43-training-summary.json")
RUNTIME = {
    "python": "3.12.13",
    "torch": "2.11.0+cu130",
    "cuda": "13.0",
    "cudnn": 91900,
    "lerobot": "0.6.1",
    "transformers": "5.1.0",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def sequence_sha256(indices: Sequence[int], *, seed: int, count: int) -> str:
    if count <= 0 or count > len(indices):
        raise ValueError("sequence count must be positive and no larger than the dataset")
    dataset = torch.tensor(tuple(int(value) for value in indices), dtype=torch.int64)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    digest = hashlib.sha256()
    iterator = iter(loader)
    for _ in range(count):
        digest.update(f"{int(next(iterator).item())}\n".encode())
    return digest.hexdigest()


def reconstruct_expected_sequences(root: Path) -> dict[str, dict[str, str]]:
    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("Experiment 043 reconstruction requires LeRobot") from error
    train_root = root / "datasets" / "embodi-sim-v1-train-v2"
    cell_indices = {}
    for morphology in MORPHOLOGIES:
        repo_id = f"embodi/sim-v1-train-v2-{morphology}"
        metadata = LeRobotDatasetMetadata(repo_id, root=train_root / morphology)
        episode_map = _load_json(train_root / morphology / "meta" / "benchmark.json")
        scenarios = episode_map.get("episode_scenarios", [])
        for task in TASKS:
            episodes = [
                int(item["episode_index"])
                for item in scenarios
                if f"-{task}-" in str(item.get("scenario_id"))
            ]
            if not episodes:
                raise ValueError("independent reconstruction found an empty task cell")
            offsets = [step / metadata.fps for step in range(32)]
            dataset = LeRobotDataset(
                repo_id,
                root=train_root / morphology,
                episodes=episodes,
                delta_timestamps={"canonical_action": offsets, "canonical_state": offsets},
            )
            cell_indices[morphology, task] = tuple(
                int(value) for value in dataset.hf_dataset["index"]
            )
    return {
        str(seed): {
            f"{morphology}/{task}": sequence_sha256(
                cell_indices[(morphology, task)],
                seed=(
                    seed
                    + 100
                    + MORPHOLOGIES.index(morphology) * len(TASKS)
                    + TASKS.index(task)
                ),
                count=SAMPLES_PER_CELL,
            )
            for morphology in MORPHOLOGIES
            for task in TASKS
        }
        for seed in SEEDS
    }


def _metric_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("metrics records must be JSON objects")
    return records


def _finite_metric_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in value.values()
    )


def expected_validation_keys() -> set[str]:
    cells = {f"{morphology}/{task}" for morphology in MORPHOLOGIES for task in TASKS}
    gripper = {
        f"gripper/{morphology}/{task}/{metric}"
        for morphology in MORPHOLOGIES
        for task in TASKS
        for metric in (
            "normalized_channel9_loss",
            "raw_absolute_error",
            "signed_bias",
            "effective_error",
            "saturation_rate",
        )
    }
    return cells | gripper | {"macro/so101", "macro/panda", "macro/all"}


def validate_metrics_records(records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [record for record in records if record.get("split") == "train"]
    validation = [record for record in records if record.get("split") == "validation"]
    required_train = {
        "regression_loss",
        "optimization_loss",
        "grad_norm",
        "expert_grad_norm",
        "learning_rate",
        "gradient_clip_count",
        "gradient_clip_rate",
    }
    if (
        len(records) != 61
        or len(train) != 60
        or [record.get("step") for record in train] != list(range(20, STEPS + 1, 20))
        or len(validation) != 1
        or validation[0].get("step") != STEPS
        or any(not _finite_metric_mapping(record.get("metrics")) for record in records)
    ):
        raise ValueError("Experiment 043 metrics schedule or values are invalid")
    prior_clips = 0
    for record in train:
        step = int(record["step"])
        metrics = record["metrics"]
        progress = (step - 120) / (STEPS - 120)
        expected_lr = (
            1e-4 * (step + 1) / 120
            if step < 120
            else 1e-4 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        )
        clips = metrics["gradient_clip_count"]
        if (
            set(metrics) != required_train
            or any(
                metrics[name] < 0
                for name in (
                    "regression_loss",
                    "optimization_loss",
                    "grad_norm",
                    "expert_grad_norm",
                )
            )
            or not math.isclose(
                metrics["optimization_loss"],
                metrics["regression_loss"] / 2,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                metrics["learning_rate"], expected_lr, rel_tol=1e-12, abs_tol=1e-15
            )
            or int(clips) != clips
            or not prior_clips <= clips <= step
            or not math.isclose(
                metrics["gradient_clip_rate"], clips / step, rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ValueError("Experiment 043 training metric record is invalid")
        prior_clips = int(clips)
    final_train = train[-1]["metrics"]
    validation_metrics = validation[0]["metrics"]
    if set(validation_metrics) != expected_validation_keys():
        raise ValueError("Experiment 043 validation metrics are incomplete")
    for name, value in validation_metrics.items():
        if name.endswith("signed_bias"):
            continue
        if value < 0 or (
            (name.endswith("effective_error") or name.endswith("saturation_rate"))
            and value > 1
        ):
            raise ValueError("Experiment 043 validation metric is outside its domain")
    for morphology in MORPHOLOGIES:
        expected = _mean(
            [validation_metrics[f"{morphology}/{task}"] for task in TASKS]
        )
        if not math.isclose(
            validation_metrics[f"macro/{morphology}"],
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Experiment 043 morphology macro does not recompute")
    expected_all = _mean(
        [validation_metrics[f"macro/{morphology}"] for morphology in MORPHOLOGIES]
    )
    if not math.isclose(
        validation_metrics["macro/all"], expected_all, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Experiment 043 global macro does not recompute")
    return final_train, validation_metrics


def _validate_data_manifest(
    value: dict[str, Any], condition: str, runtime: dict[str, Any]
) -> None:
    schedule = value.get("schedule", {})
    expected_cells = [f"{condition}/{task}" for task in TASKS for _ in range(5)]
    expected_train = {
        morphology: {
            "manifest_sha256": TRAIN_GENERATION_SHA256,
            "definition_sha256": DEFINITION_SHA256,
            "scenario_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "split": "train",
            "morphology": morphology,
        }
        for morphology in MORPHOLOGIES
    }
    expected_validation = {
        morphology: {
            "manifest_sha256": VALIDATION_GENERATION_SHA256,
            "definition_sha256": DEFINITION_SHA256,
            "scenario_manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "split": "validation",
            "morphology": morphology,
        }
        for morphology in MORPHOLOGIES
    }
    if (
        value.get("format_version") != 1
        or value.get("experiment") != 43
        or value.get("protocol") != PROTOCOL
        or value.get("condition") != condition
        or value.get("train_generation_manifest_sha256") != TRAIN_GENERATION_SHA256
        or value.get("validation_generation_manifest_sha256") != VALIDATION_GENERATION_SHA256
        or value.get("validation_frame_manifest_sha256") != VALIDATION_FRAMES_SHA256
        or value.get("train") != expected_train
        or value.get("validation") != expected_validation
        or value.get("runtime") != runtime
        or schedule.get("presentations_per_step") != PRESENTATIONS_PER_STEP
        or schedule.get("optimization_loss_denominator") != LOSS_DENOMINATOR
        or schedule.get("per_sample_gradient_coefficient") != 1 / LOSS_DENOMINATOR
        or schedule.get("cells") != expected_cells
    ):
        raise ValueError("Experiment 043 data manifest contract is invalid")


def _finite_tensors(value: Any) -> bool:
    if torch.is_tensor(value):
        return not (value.is_floating_point() or value.is_complex()) or bool(
            torch.isfinite(value).all()
        )
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(item) for item in value)
    return True


def _tensor_schema(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(name): _tensor_schema(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_schema(item) for item in value]
    return type(value).__name__


def _validate_trainer_payload(value: dict[str, Any], *, expect_optimizer: bool) -> None:
    if not expect_optimizer:
        if value != {"format_version": 3, "step": STEPS, "stage": "core"}:
            raise ValueError("secondary wrapper trainer payload is invalid")
        return
    optimizer = value.get("optimizer", {})
    scheduler = value.get("scheduler", {})
    groups = optimizer.get("param_groups", [])
    states = optimizer.get("state", {})
    if (
        set(value) != {"format_version", "step", "stage", "optimizer", "scheduler"}
        or value.get("format_version") != 3
        or value.get("step") != STEPS
        or value.get("stage") != "core"
        or len(groups) != 1
        or groups[0].get("lr") != 0.0
        or groups[0].get("initial_lr") != 1e-4
        or groups[0].get("betas") != (0.9, 0.95)
        or groups[0].get("weight_decay") != 1e-10
        or groups[0].get("amsgrad") is not False
        or groups[0].get("maximize") is not False
        or groups[0].get("decoupled_weight_decay") is not True
        or len(groups[0].get("params", [])) != 381
        or len(states) != 375
        or any(set(state) != {"step", "exp_avg", "exp_avg_sq"} for state in states.values())
        or scheduler.get("base_lrs") != [1e-4]
        or scheduler.get("last_epoch") != STEPS
        or scheduler.get("_step_count") != STEPS + 1
        or scheduler.get("_last_lr") != [0.0]
        or not _finite_tensors(value)
    ):
        raise ValueError("Experiment 043 trainer payload is incomplete or invalid")


def _validate_run(
    root: Path,
    condition: str,
    seed: int,
    expected_sequences: dict[str, dict[str, str]],
    configs: dict[str, EmbodiConfig],
    validation_loaders: dict[tuple[str, str], DataLoader],
    processor: EmbodiProcessor,
    device: torch.device,
) -> dict[str, Any]:
    relative_output = Path(f"outputs/benchmark-exp43-half-{condition}-seed{seed}")
    output = root / relative_output
    required = (
        "data_manifest.json",
        "run_manifest.json",
        "metrics.jsonl",
        "sample_sequence.json",
    )
    if any(not (output / name).is_file() for name in required):
        raise FileNotFoundError(f"Experiment 043 run is incomplete: {output}")
    run_manifest = _load_json(output / "run_manifest.json")
    arguments = run_manifest.get("arguments", {})
    runtime = run_manifest.get("runtime", {})
    expected_arguments = {
        "protocol": PROTOCOL,
        "condition": condition,
        "train_root": "datasets/embodi-sim-v1-train-v2",
        "validation_root": "datasets/embodi-sim-v1-validation-v2",
        "train_repo_prefix": "embodi/sim-v1-train-v2",
        "validation_repo_prefix": "embodi/sim-v1-validation-v2",
        "so101_config": "configs/so101-exp0-top.json",
        "panda_config": "configs/panda-exp0-top.json",
        "validation_manifest": "reports/benchmark-exp36-validation-frames.json",
        "prepare_validation_manifest": False,
        "validation_samples_per_cell": 256,
        "validation_frame_seed": 36000,
        "steps": STEPS,
        "presentations_per_step": PRESENTATIONS_PER_STEP,
        "lr": 1e-4,
        "warmup_steps": 120,
        "model_seed": seed,
        "loader_seed": seed + 100,
        "workers": 0,
        "log_every": 20,
        "deterministic": True,
        "smoke_test": False,
        "no_progress": True,
        "output_dir": str(relative_output),
    }
    if (
        run_manifest.get("format_version") != 1
        or run_manifest.get("experiment") != 43
        or run_manifest.get("protocol") != PROTOCOL
        or any(arguments.get(name) != value for name, value in expected_arguments.items())
        or runtime.get("git_commit") != "dd86c02a249c088b384eb9733a75aeff119f3e34"
        or runtime.get("git_dirty") is not False
        or runtime.get("trainer_sha256") != TRAINER_SHA256
        or any(runtime.get(name) != value for name, value in RUNTIME.items())
        or set(arguments) != set(expected_arguments)
    ):
        raise ValueError("Experiment 043 run manifest contract is invalid")
    data_path = output / "data_manifest.json"
    data_manifest = _load_json(data_path)
    _validate_data_manifest(data_manifest, condition, runtime)
    final_data_path = output / "final" / "data_manifest.json"
    if final_data_path.read_bytes() != data_path.read_bytes():
        raise ValueError("root and final data manifests differ")
    sequence_path = output / "sample_sequence.json"
    sequence = _load_json(sequence_path)
    expected_cells = {f"{condition}/{task}" for task in TASKS}
    if (
        sequence.get("format_version") != 1
        or sequence.get("experiment") != 43
        or sequence.get("protocol") != PROTOCOL
        or sequence.get("condition") != condition
        or sequence.get("model_seed") != seed
        or sequence.get("loader_seed") != seed + 100
        or sequence.get("steps") != STEPS
        or set(sequence.get("cells", {})) != expected_cells
    ):
        raise ValueError("Experiment 043 sample-sequence identity is invalid")
    for cell, evidence in sequence["cells"].items():
        if (
            evidence.get("samples") != SAMPLES_PER_CELL
            or evidence.get("dataset_index_sequence_sha256")
            != expected_sequences[str(seed)][cell]
        ):
            raise ValueError("Experiment 043 sample sequence differs from reconstruction")
    metrics_path = output / "metrics.jsonl"
    final_train, validation = validate_metrics_records(_metric_records(metrics_path))
    checkpoints = {}
    core_paths = []
    for morphology in MORPHOLOGIES:
        checkpoint = output / "final" / morphology
        paths = {name: checkpoint / name for name in ("config.json", "core.pt", "embodiment.pt", "trainer.pt")}
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Experiment 043 checkpoint is incomplete: {checkpoint}")
        if file_sha256(paths["config.json"]) != CONFIG_SHA256[morphology]:
            raise ValueError("Experiment 043 checkpoint config differs from registration")
        core_payload = torch.load(
            paths["core.pt"], map_location="cpu", weights_only=True, mmap=True
        )
        embodiment_payload = torch.load(
            paths["embodiment.pt"], map_location="cpu", weights_only=True, mmap=True
        )
        trainer = torch.load(paths["trainer.pt"], map_location="cpu", weights_only=True, mmap=True)
        reference_root = (
            root
            / f"outputs/benchmark-exp36-{condition}-seed{seed}"
            / "final"
            / morphology
        )
        reference_core = torch.load(
            reference_root / "core.pt", map_location="cpu", weights_only=True, mmap=True
        )
        if (
            set(core_payload) != {"format_version", "signature", "stores_frozen_adapter", "core"}
            or core_payload.get("format_version") != 3
            or core_payload.get("signature") != configs[morphology].core_signature()
            or core_payload.get("stores_frozen_adapter") is not True
            or _tensor_schema(core_payload.get("core"))
            != _tensor_schema(reference_core.get("core"))
            or not _finite_tensors(core_payload)
            or set(embodiment_payload)
            != {
                "format_version",
                "signature",
                "state_adapter",
                "action_decoder",
                "state_mean",
                "state_std",
                "action_mean",
                "action_std",
            }
            or embodiment_payload.get("format_version") != 3
            or embodiment_payload.get("signature")
            != configs[morphology].embodiment_signature()
            or not _finite_tensors(embodiment_payload)
        ):
            raise ValueError("Experiment 043 checkpoint payload schema or tensors are invalid")
        _validate_trainer_payload(trainer, expect_optimizer=morphology == "so101")
        core_paths.append(paths["core.pt"])
        checkpoints[morphology] = {
            "path": str(checkpoint.relative_to(root)),
            **{f"{name.removesuffix('.json').removesuffix('.pt')}_sha256": file_sha256(path) for name, path in paths.items()},
        }
    core_stats = [path.lstat() for path in core_paths]
    if (
        any(path.is_symlink() or not stat.S_ISREG(value.st_mode) for path, value in zip(core_paths, core_stats, strict=True))
        or (core_stats[0].st_dev, core_stats[0].st_ino)
        != (core_stats[1].st_dev, core_stats[1].st_ino)
        or core_stats[0].st_nlink < 2
        or checkpoints["so101"]["core_sha256"] != checkpoints["panda"]["core_sha256"]
    ):
        raise ValueError("Experiment 043 wrapper cores are not hard-linked and byte-identical")
    policies = make_policies(configs, device)
    for morphology in MORPHOLOGIES:
        checkpoint = output / "final" / morphology
        policies[morphology].load_core_checkpoint(checkpoint)
        policies[morphology].load_embodiment_checkpoint(checkpoint)
    recomputed = evaluate_cells(
        policies,
        validation_loaders,
        processor,
        device,
        detailed_gripper=True,
    )
    if set(recomputed) != set(validation) or any(
        not math.isclose(recomputed[name], validation[name], rel_tol=1e-9, abs_tol=1e-10)
        for name in validation
    ):
        raise ValueError("Experiment 043 checkpoint validation does not reproduce")
    del policies
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "condition": condition,
        "model_seed": seed,
        "loader_seed": seed + 100,
        "output": str(relative_output),
        "metrics_sha256": file_sha256(metrics_path),
        "run_manifest_sha256": file_sha256(output / "run_manifest.json"),
        "data_manifest_sha256": file_sha256(data_path),
        "sample_sequence_sha256": file_sha256(sequence_path),
        "sample_sequences": sequence["cells"],
        "checkpoints": checkpoints,
        "shared_core_hard_link_verified": True,
        "final_train": final_train,
        "validation": validation,
        "runtime": runtime,
    }


def _provenance(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if dirty:
        raise RuntimeError("registered Experiment 043 summary requires a clean worktree")
    return {
        "git_commit": commit,
        "git_dirty": False,
        "validator_sha256": file_sha256(Path(__file__)),
    }


def _validation_context(root: Path) -> tuple[
    dict[str, EmbodiConfig],
    dict[tuple[str, str], DataLoader],
    EmbodiProcessor,
]:
    configs = {
        morphology: EmbodiConfig.load(root / "configs" / f"{morphology}-exp0-top.json")
        for morphology in MORPHOLOGIES
    }
    validation_root = root / "datasets" / "embodi-sim-v1-validation-v2"
    roots = {morphology: validation_root for morphology in MORPHOLOGIES}
    repo_ids = {
        morphology: f"embodi/sim-v1-validation-v2-{morphology}"
        for morphology in MORPHOLOGIES
    }
    metadata, _lineage = load_benchmark_metadata(
        roots, repo_ids, configs, "validation"
    )
    manifest = validate_validation_manifest(
        root / "reports" / "benchmark-exp36-validation-frames.json",
        roots,
        repo_ids,
        registered=True,
    )
    loaders = make_validation_loaders(
        roots,
        repo_ids,
        metadata,
        configs,
        manifest,
        workers=0,
    )
    processor = EmbodiProcessor.from_pretrained(
        configs["so101"].backbone_name,
        revision=configs["so101"].backbone_revision,
        do_rescale=configs["so101"].image_do_rescale,
    )
    return configs, loaders, processor


def build_training_summary(root: Path, *, device_name: str) -> dict[str, Any]:
    exp36_path = root / "reports" / "benchmark-exp36-summary.json"
    trainer_path = root / "src" / "embodi" / "joint_train.py"
    source_hashes = {
        "definition": (root / "benchmarks/sim-v1/definition.json", DEFINITION_SHA256),
        "development_manifest": (
            root / "benchmarks/sim-v1/manifests/development.json",
            DEVELOPMENT_MANIFEST_SHA256,
        ),
        "validation_frames": (
            root / "reports/benchmark-exp36-validation-frames.json",
            VALIDATION_FRAMES_SHA256,
        ),
        "exp36_summary": (exp36_path, EXP36_SUMMARY_SHA256),
        "exp42_summary": (
            root / "reports/benchmark-exp42-summary.json",
            EXP42_SUMMARY_SHA256,
        ),
        "trainer": (trainer_path, TRAINER_SHA256),
        "exp42_evaluator": (
            root / "src/embodi/sim/benchmark_phase_agreement.py",
            EXP42_EVALUATOR_SHA256,
        ),
        "expert": (
            root / "src/embodi/sim/benchmark_expert.py",
            EXPERT_SHA256,
        ),
    }
    for name, (path, expected) in source_hashes.items():
        if file_sha256(path) != expected:
            raise ValueError(f"Experiment 043 frozen {name} hash differs from registration")
    exp36 = _load_json(exp36_path)
    expected_sequences = reconstruct_expected_sequences(root)
    device = torch.device(device_name)
    configs, validation_loaders, processor = _validation_context(root)
    runs = {
        f"{condition}-seed{seed}": _validate_run(
            root,
            condition,
            seed,
            expected_sequences,
            configs,
            validation_loaders,
            processor,
            device,
        )
        for seed in SEEDS
        for condition in MORPHOLOGIES
    }
    runtime_values = {
        json.dumps(run["runtime"], sort_keys=True, separators=(",", ":"))
        for run in runs.values()
    }
    if len(runtime_values) != 1:
        raise ValueError("Experiment 043 registered run runtimes differ")
    by_seed = {}
    for seed in SEEDS:
        full = _mean(
            [
                exp36["runs"][f"{morphology}-seed{seed}"]["validation"][f"macro/{morphology}"]
                for morphology in MORPHOLOGIES
            ]
        )
        half = _mean(
            [
                runs[f"{morphology}-seed{seed}"]["validation"][f"macro/{morphology}"]
                for morphology in MORPHOLOGIES
            ]
        )
        joint = exp36["runs"][f"joint-seed{seed}"]["validation"]["macro/all"]
        by_seed[str(seed)] = {
            "full_matched": full,
            "half_matched": half,
            "joint": joint,
            "half_minus_full": half - full,
            "joint_minus_half": joint - half,
        }
    aggregate = {
        name: _mean([values[name] for values in by_seed.values()])
        for name in (
            "full_matched",
            "half_matched",
            "joint",
            "half_minus_full",
            "joint_minus_half",
        )
    }
    return {
        "format_version": 1,
        "experiment": 43,
        "registered": True,
        "status": "completed_training_controls",
        "scope": "fixed-step training delivery and offline validation; mechanism classification pending fresh development assay",
        "contract": {
            "protocol": PROTOCOL,
            "definition_sha256": DEFINITION_SHA256,
            "development_manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "trainer_sha256": TRAINER_SHA256,
            "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
            "exp42_summary_sha256": EXP42_SUMMARY_SHA256,
            "exp42_evaluator_sha256": EXP42_EVALUATOR_SHA256,
            "expert_sha256": EXPERT_SHA256,
            "train_generation_manifest_sha256": TRAIN_GENERATION_SHA256,
            "validation_generation_manifest_sha256": VALIDATION_GENERATION_SHA256,
            "validation_frame_manifest_sha256": VALIDATION_FRAMES_SHA256,
            "model_seeds": list(SEEDS),
            "loader_seeds": [seed + 100 for seed in SEEDS],
            "fixed_step": STEPS,
            "presentations_per_step": PRESENTATIONS_PER_STEP,
            "optimization_loss_denominator": LOSS_DENOMINATOR,
            "presentations_per_cell": SAMPLES_PER_CELL,
            "new_runs": 6,
            "final_split_loaded": False,
        },
        "provenance": _provenance(root),
        "expected_joint_target_sequence_sha256": expected_sequences,
        "runs": runs,
        "offline_validation": {"by_seed": by_seed, "aggregate": aggregate},
        "delivery_gate_passed": True,
        "mechanism_classification": None,
        "final_split_loaded": False,
        "execution_notes": [
            "The first SO-101 seed 36001 invocation was rejected before model or output creation because --presentations-per-step 15 was omitted; the corrected registered invocation then completed normally."
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="validate and summarize Experiment 043 training")
    parser.add_argument("--artifact-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    root = args.artifact_root.resolve()
    expected_output = (root / SUMMARY_PATH).resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if output != expected_output:
        raise ValueError(f"registered Experiment 043 summary output must be {SUMMARY_PATH}")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing Experiment 043 summary: {output}")
    os.chdir(root)
    summary = build_training_summary(root, device_name=args.device)
    atomic_write_json(output, summary)
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
