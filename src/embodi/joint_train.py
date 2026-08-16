from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .processing import EmbodiProcessor
from .record_so101 import atomic_write_json
from .token_config import EmbodiConfig
from .token_policy import EmbodiPolicy
from .train import (
    append_metrics,
    configure_deterministic_training,
    gradient_norm,
    prepare_model_batch,
    validate_benchmark_dataset_lineage,
    validate_lerobot_training_dataset,
    write_or_validate_data_manifest,
)


MORPHOLOGIES = ("so101", "panda")
TASKS = ("push_to_zone", "lift_object", "pick_place_bin")
CONDITIONS = ("so101", "panda", "joint")
TRAIN_GENERATION_SHA256 = "9d858858c217c2b1200507a503148c99562a06f0beecea054a2b9cc184b840af"
VALIDATION_GENERATION_SHA256 = "2735a044d11db30172ae6039f09dfe06adc49462f3b47d5e7b99cb312c354f45"
VALIDATION_FRAMES_SHA256 = "32681217e8d6f4142ce754b18b2a3a69527edea9abce3506fb55e7d25c28cc1a"
REGISTERED_MODEL_SEEDS = (36001, 36002, 36003)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def generation_manifest_sha256(root: Path) -> str:
    path = root / "benchmark-dataset.json"
    if not path.is_file():
        raise FileNotFoundError(f"benchmark dataset is missing {path}")
    return file_sha256(path)


def episode_indices_by_task(root: Path, morphology: str) -> dict[str, list[int]]:
    metadata_path = root / morphology / "meta" / "benchmark.json"
    values = json.loads(metadata_path.read_text())
    if values.get("morphology") != morphology:
        raise ValueError("benchmark episode mapping morphology does not match")
    grouped = {task: [] for task in TASKS}
    for item in values.get("episode_scenarios", []):
        scenario_id = str(item["scenario_id"])
        matches = [task for task in TASKS if f"-{task}-" in scenario_id]
        if len(matches) != 1:
            raise ValueError(f"cannot identify benchmark task for {scenario_id}")
        grouped[matches[0]].append(int(item["episode_index"]))
    if any(not indices for indices in grouped.values()):
        raise ValueError("every benchmark task must contain episodes")
    return grouped


def balanced_cells(condition: str) -> tuple[tuple[str, str], ...]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown joint-core condition: {condition}")
    morphologies = MORPHOLOGIES if condition == "joint" else (condition,)
    return tuple((morphology, task) for morphology in morphologies for task in TASKS)


def cell_schedule(condition: str, presentations_per_step: int) -> tuple[tuple[str, str], ...]:
    cells = balanced_cells(condition)
    if presentations_per_step <= 0 or presentations_per_step % len(cells):
        raise ValueError("presentations_per_step must be positive and divisible by cell count")
    repeats = presentations_per_step // len(cells)
    return tuple(cell for cell in cells for _ in range(repeats))


def deduplicated_trainable_parameters(*modules: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def cell_loader_seed(loader_seed: int, morphology: str, task: str) -> int:
    return loader_seed + MORPHOLOGIES.index(morphology) * len(TASKS) + TASKS.index(task)


def _core_behavior(config: EmbodiConfig) -> dict[str, Any]:
    return {
        "core_signature": config.core_signature(),
        "camera_names": config.camera_names,
        "image_do_rescale": config.image_do_rescale,
        "core_objective": config.core_objective,
        "flow_first_step_weight": config.flow_first_step_weight,
        "canonical_channel_loss_weights": config.canonical_channel_loss_weights,
        "freeze_backbone": config.freeze_backbone,
    }


def validate_joint_configs(configs: dict[str, EmbodiConfig]) -> None:
    if set(configs) != set(MORPHOLOGIES):
        raise ValueError("joint configs must contain SO-101 and Panda")
    if _core_behavior(configs["so101"]) != _core_behavior(configs["panda"]):
        raise ValueError("joint embodiment configs must have identical core behavior")
    if configs["so101"].state_dim != 6 or configs["so101"].action_dim != 6:
        raise ValueError("SO-101 benchmark config must use native width 6")
    if configs["panda"].state_dim != 8 or configs["panda"].action_dim != 8:
        raise ValueError("Panda benchmark config must use native width 8")


def validate_registered_protocol(args: argparse.Namespace, configs: dict[str, EmbodiConfig]) -> None:
    if args.smoke_test:
        return
    expected = {
        "steps": 1200,
        "presentations_per_step": 30,
        "lr": 1e-4,
        "warmup_steps": 120,
        "workers": 0,
        "deterministic": True,
        "log_every": 20,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"experiment 036 requires --{name.replace('_', '-')}={value}")
    if args.model_seed not in REGISTERED_MODEL_SEEDS or args.loader_seed != args.model_seed + 100:
        raise ValueError("experiment 036 requires a registered paired model/loader seed")
    for config in configs.values():
        if (
            config.backbone_name != "LiquidAI/LFM2.5-VL-450M"
            or config.backbone_revision != "fc6221ca597f3315e4f82fc2df606783267b34ba"
            or config.camera_names != ("top",)
            or config.image_do_rescale is not False
            or config.chunk_size != 32
            or config.n_action_steps != 32
            or config.expert_width != 512
            or config.expert_layers != 12
            or config.expert_heads != 8
            or config.expert_ff_multiplier != 4
            or config.dropout != 0.0
            or config.lora_rank != 16
            or config.lora_alpha != 16
            or config.lora_dropout != 0.0
            or config.core_objective != "regression"
            or config.flow_first_step_weight != 10.0
            or config.canonical_channel_loss_weights != (1.0,) * 10
            or config.freeze_vision_encoder is not True
            or config.freeze_backbone is not False
        ):
            raise ValueError("experiment 036 config differs from the registered core architecture")


def make_policies(
    configs: dict[str, EmbodiConfig],
    device: torch.device,
) -> dict[str, EmbodiPolicy]:
    validate_joint_configs(configs)
    so101 = EmbodiPolicy(configs["so101"]).to(device)
    panda = EmbodiPolicy(configs["panda"], core=so101.core).to(device)
    policies = {"so101": so101, "panda": panda}
    for policy in policies.values():
        policy.configure_stage("core")
        for parameter in policy.state_adapter.parameters():
            parameter.requires_grad_(False)
        for parameter in policy.action_decoder.parameters():
            parameter.requires_grad_(False)
    return policies


def canonical_only_batch(raw_batch: dict[str, Any]) -> dict[str, Any]:
    value = dict(raw_batch)
    value["observation.state"] = None
    value["action"] = None
    value.pop("action_is_pad", None)
    return value


def _next_batch(
    key: tuple[str, str],
    loaders: dict[tuple[str, str], DataLoader],
    iterators: dict[tuple[str, str], Any],
):
    try:
        return next(iterators[key])
    except StopIteration:
        iterators[key] = iter(loaders[key])
        return next(iterators[key])


def _lerobot_imports():
    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("joint training requires the LeRobot training dependencies") from error
    return LeRobotDataset, LeRobotDatasetMetadata


def load_benchmark_metadata(
    roots: dict[str, Path],
    repo_ids: dict[str, str],
    configs: dict[str, EmbodiConfig],
    split: str,
):
    _dataset_type, metadata_type = _lerobot_imports()
    metadata = {}
    lineage = {}
    for morphology in MORPHOLOGIES:
        value = metadata_type(repo_ids[morphology], root=roots[morphology] / morphology)
        validate_lerobot_training_dataset(
            value,
            configs[morphology],
            "core",
            benchmark_split=split,
        )
        lineage[morphology] = validate_benchmark_dataset_lineage(value, split)
        if lineage[morphology] is None:
            raise ValueError("joint training requires verified benchmark lineage")
        metadata[morphology] = value
    return metadata, lineage


def make_train_loaders(
    roots: dict[str, Path],
    repo_ids: dict[str, str],
    metadata: dict,
    configs: dict[str, EmbodiConfig],
    cells: tuple[tuple[str, str], ...],
    *,
    loader_seed: int,
    workers: int,
) -> dict[tuple[str, str], DataLoader]:
    dataset_type, _metadata_type = _lerobot_imports()
    loaders = {}
    unique_cells = tuple(dict.fromkeys(cells))
    for morphology, task in unique_cells:
        episodes = episode_indices_by_task(roots[morphology], morphology)[task]
        offsets = [step / metadata[morphology].fps for step in range(configs[morphology].chunk_size)]
        dataset = dataset_type(
            repo_ids[morphology],
            root=roots[morphology] / morphology,
            episodes=episodes,
            delta_timestamps={"canonical_action": offsets, "canonical_state": offsets},
        )
        loaders[(morphology, task)] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=workers,
            generator=torch.Generator().manual_seed(cell_loader_seed(loader_seed, morphology, task)),
        )
    return loaders


def prepare_validation_manifest(
    roots: dict[str, Path],
    repo_ids: dict[str, str],
    output: Path,
    *,
    samples_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    _dataset_type, metadata_type = _lerobot_imports()
    cells = {}
    for morphology_index, morphology in enumerate(MORPHOLOGIES):
        metadata = metadata_type(repo_ids[morphology], root=roots[morphology] / morphology)
        episode_groups = episode_indices_by_task(roots[morphology], morphology)
        for task_index, task in enumerate(TASKS):
            candidates = []
            for episode_index in episode_groups[task]:
                episode = metadata.episodes[episode_index]
                candidates.extend(
                    range(
                        int(episode["dataset_from_index"]),
                        int(episode["dataset_to_index"]),
                    )
                )
            if len(candidates) < samples_per_cell:
                raise ValueError(f"not enough validation frames for {morphology}/{task}")
            random_generator = random.Random(seed + 100 * morphology_index + task_index)
            cells[f"{morphology}/{task}"] = sorted(
                random_generator.sample(candidates, samples_per_cell)
            )
    value = {
        "format_version": 1,
        "benchmark_id": "embodi-sim-v1",
        "seed": seed,
        "samples_per_cell": samples_per_cell,
        "sources": {
            morphology: {
                "root": str(roots[morphology]),
                "repo_id": repo_ids[morphology],
                "generation_manifest_sha256": generation_manifest_sha256(roots[morphology]),
            }
            for morphology in MORPHOLOGIES
        },
        "cells": cells,
    }
    text = json.dumps(value, indent=2) + "\n"
    if output.exists() and output.read_text() != text:
        raise FileExistsError(f"refusing to replace different validation manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return value


def validate_validation_manifest(
    path: Path,
    roots: dict[str, Path],
    repo_ids: dict[str, str],
    *,
    registered: bool = True,
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("format_version") != 1 or value.get("benchmark_id") != "embodi-sim-v1":
        raise ValueError("joint validation manifest format does not match")
    if registered:
        if path.resolve() != Path("reports/benchmark-exp36-validation-frames.json").resolve():
            raise ValueError("experiment 036 requires the registered validation-frame manifest path")
        if file_sha256(path) != VALIDATION_FRAMES_SHA256:
            raise ValueError("joint validation frame manifest hash does not match registration")
    expected_cells = {f"{morphology}/{task}" for morphology in MORPHOLOGIES for task in TASKS}
    if set(value.get("cells", {})) != expected_cells:
        raise ValueError("joint validation manifest cells do not match the benchmark")
    count = value.get("samples_per_cell")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("joint validation samples_per_cell must be positive")
    for morphology in MORPHOLOGIES:
        source = value.get("sources", {}).get(morphology, {})
        if (
            source.get("repo_id") != repo_ids[morphology]
            or source.get("generation_manifest_sha256")
            != generation_manifest_sha256(roots[morphology])
        ):
            raise ValueError("joint validation source lineage does not match")
        if registered and source.get("generation_manifest_sha256") != VALIDATION_GENERATION_SHA256:
            raise ValueError("joint validation dataset hash does not match registration")
        _dataset_type, metadata_type = _lerobot_imports()
        metadata = metadata_type(repo_ids[morphology], root=roots[morphology] / morphology)
        episodes = episode_indices_by_task(roots[morphology], morphology)
        for task in TASKS:
            allowed = set()
            for episode_index in episodes[task]:
                episode = metadata.episodes[episode_index]
                allowed.update(
                    range(
                        int(episode["dataset_from_index"]),
                        int(episode["dataset_to_index"]),
                    )
                )
            if not set(value["cells"][f"{morphology}/{task}"]).issubset(allowed):
                raise ValueError("joint validation frame lies outside its registered task cell")
    for indices in value["cells"].values():
        if len(indices) != count or len(set(indices)) != count or any(index < 0 for index in indices):
            raise ValueError("joint validation frame indices must be unique nonnegative values")
    return value


def runtime_provenance(require_clean: bool) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if require_clean and dirty:
        raise RuntimeError("registered experiment 036 must run from a clean git worktree")
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "lerobot": version("lerobot"),
        "transformers": version("transformers"),
        "trainer_sha256": file_sha256(Path(__file__)),
    }


def make_validation_loaders(
    roots: dict[str, Path],
    repo_ids: dict[str, str],
    metadata: dict,
    configs: dict[str, EmbodiConfig],
    manifest: dict[str, Any],
    *,
    workers: int,
) -> dict[tuple[str, str], DataLoader]:
    dataset_type, _metadata_type = _lerobot_imports()
    loaders = {}
    for morphology in MORPHOLOGIES:
        offsets = [step / metadata[morphology].fps for step in range(configs[morphology].chunk_size)]
        dataset = dataset_type(
            repo_ids[morphology],
            root=roots[morphology] / morphology,
            delta_timestamps={"canonical_action": offsets, "canonical_state": offsets},
        )
        for task in TASKS:
            subset = Subset(dataset, manifest["cells"][f"{morphology}/{task}"])
            loaders[(morphology, task)] = DataLoader(
                subset,
                batch_size=1,
                shuffle=False,
                num_workers=workers,
            )
    return loaders


@torch.no_grad()
def evaluate_cells(
    policies: dict[str, EmbodiPolicy],
    loaders: dict[tuple[str, str], DataLoader],
    processor: EmbodiProcessor,
    device: torch.device,
) -> dict[str, float]:
    for policy in policies.values():
        policy.eval()
    results = {}
    for (morphology, task), loader in loaders.items():
        total = 0.0
        count = 0
        for raw_batch in loader:
            config = policies[morphology].config
            model_batch = prepare_model_batch(
                canonical_only_batch(raw_batch),
                processor,
                device,
                config.camera_names,
                config,
            )
            _loss, metrics = policies[morphology](model_batch)
            metric_name = (
                "regression_loss"
                if config.core_objective == "regression"
                else "flow_loss"
            )
            total += metrics[metric_name]
            count += 1
        if count == 0:
            raise RuntimeError(f"validation cell {morphology}/{task} produced no batches")
        results[f"{morphology}/{task}"] = total / count
    for morphology in MORPHOLOGIES:
        results[f"macro/{morphology}"] = sum(
            results[f"{morphology}/{task}"] for task in TASKS
        ) / len(TASKS)
    results["macro/all"] = sum(results[f"macro/{value}"] for value in MORPHOLOGIES) / 2
    return results


def save_policies(
    directory: Path,
    policies: dict[str, EmbodiPolicy],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    data_report: dict[str, Any],
) -> None:
    shared_core_path = None
    for index, morphology in enumerate(MORPHOLOGIES):
        policy_directory = directory / morphology
        policies[morphology].save_checkpoint(
            policy_directory,
            optimizer if index == 0 else None,
            scheduler if index == 0 else None,
            step,
        )
        core_path = policy_directory / "core.pt"
        if shared_core_path is None:
            shared_core_path = core_path
        else:
            core_path.unlink()
            os.link(shared_core_path, core_path)
    atomic_write_json(directory / "data_manifest.json", data_report)


def train_joint(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.presentations_per_step <= 0 or args.warmup_steps < 0:
        raise ValueError("steps and presentations must be positive; warmup must be nonnegative")
    configs = {
        "so101": EmbodiConfig.load(args.so101_config),
        "panda": EmbodiConfig.load(args.panda_config),
    }
    validate_joint_configs(configs)
    validate_registered_protocol(args, configs)
    provenance = runtime_provenance(require_clean=not args.smoke_test)
    train_roots = {morphology: args.train_root for morphology in MORPHOLOGIES}
    validation_roots = {morphology: args.validation_root for morphology in MORPHOLOGIES}
    train_repo_ids = {
        morphology: f"{args.train_repo_prefix}-{morphology}" for morphology in MORPHOLOGIES
    }
    validation_repo_ids = {
        morphology: f"{args.validation_repo_prefix}-{morphology}" for morphology in MORPHOLOGIES
    }
    train_metadata, train_lineage = load_benchmark_metadata(
        train_roots,
        train_repo_ids,
        configs,
        "train",
    )
    validation_metadata, validation_lineage = load_benchmark_metadata(
        validation_roots,
        validation_repo_ids,
        configs,
        "validation",
    )
    validation_manifest = validate_validation_manifest(
        args.validation_manifest,
        validation_roots,
        validation_repo_ids,
        registered=not args.smoke_test,
    )
    if not args.smoke_test:
        if generation_manifest_sha256(args.train_root) != TRAIN_GENERATION_SHA256:
            raise ValueError("experiment 036 train dataset hash does not match registration")
        if generation_manifest_sha256(args.validation_root) != VALIDATION_GENERATION_SHA256:
            raise ValueError("experiment 036 validation dataset hash does not match registration")
    schedule = cell_schedule(args.condition, args.presentations_per_step)

    configure_deterministic_training(args.deterministic)
    random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = make_policies(configs, device)
    for morphology in MORPHOLOGIES:
        policies[morphology].set_normalization_stats(train_metadata[morphology].stats)
    processor = EmbodiProcessor.from_pretrained(
        configs["so101"].backbone_name,
        revision=configs["so101"].backbone_revision,
        do_rescale=configs["so101"].image_do_rescale,
    )
    train_loaders = make_train_loaders(
        train_roots,
        train_repo_ids,
        train_metadata,
        configs,
        schedule,
        loader_seed=args.loader_seed,
        workers=args.workers,
    )
    validation_loaders = make_validation_loaders(
        validation_roots,
        validation_repo_ids,
        validation_metadata,
        configs,
        validation_manifest,
        workers=args.workers,
    )

    parameters = deduplicated_trainable_parameters(policies["so101"].core)
    if not parameters:
        raise RuntimeError("joint core has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-10)

    def multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    data_report = {
        "format_version": 1,
        "experiment": "36-smoke" if args.smoke_test else 36,
        "condition": args.condition,
        "train": train_lineage,
        "validation": validation_lineage,
        "train_generation_manifest_sha256": generation_manifest_sha256(args.train_root),
        "validation_generation_manifest_sha256": generation_manifest_sha256(args.validation_root),
        "validation_frame_manifest_sha256": file_sha256(args.validation_manifest),
        "schedule": {
            "presentations_per_step": args.presentations_per_step,
            "cells": [f"{morphology}/{task}" for morphology, task in schedule],
        },
        "runtime": provenance,
    }
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"experiment output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.jsonl"
    start_step = 0
    write_or_validate_data_manifest(output_dir, data_report)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "format_version": 1,
            "experiment": 36,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "configs": {key: asdict(value) for key, value in configs.items()},
            "device": str(device),
            "runtime": provenance,
        },
    )

    iterators = {key: iter(loader) for key, loader in train_loaders.items()}
    for policy in policies.values():
        policy.train()
    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        total=args.steps,
        initial=start_step,
        desc=f"experiment-036-{args.condition}",
        unit="step",
        disable=args.no_progress,
    )
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        metric_name = (
            "regression_loss"
            if configs["so101"].core_objective == "regression"
            else "flow_loss"
        )
        metrics = {metric_name: 0.0}
        for cell in schedule:
            raw_batch = _next_batch(cell, train_loaders, iterators)
            morphology = cell[0]
            config = configs[morphology]
            model_batch = prepare_model_batch(
                canonical_only_batch(raw_batch),
                processor,
                device,
                config.camera_names,
                config,
            )
            loss, batch_metrics = policies[morphology](model_batch)
            (loss / len(schedule)).backward()
            metrics[metric_name] += batch_metrics[metric_name] / len(schedule)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0).item()
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps:
            metrics["grad_norm"] = grad_norm
            metrics["expert_grad_norm"] = gradient_norm(policies["so101"].expert.parameters())
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            append_metrics(metrics_path, step, "train", metrics)
            progress.set_postfix(loss=f"{metrics[metric_name]:.4f}")
        if step == args.steps:
            validation_metrics = evaluate_cells(
                policies,
                validation_loaders,
                processor,
                device,
            )
            append_metrics(metrics_path, step, "validation", validation_metrics)
            progress.write(
                f"step={step} validation_macro_all={validation_metrics['macro/all']:.6f}"
            )
    save_policies(output_dir / "final", policies, optimizer, scheduler, args.steps, data_report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train experiment 036 shared canonical core")
    parser.add_argument("--condition", choices=CONDITIONS, default="joint")
    parser.add_argument("--train-root", type=Path, default=Path("datasets/embodi-sim-v1-train-v2"))
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("datasets/embodi-sim-v1-validation-v2"),
    )
    parser.add_argument("--train-repo-prefix", default="embodi/sim-v1-train-v2")
    parser.add_argument("--validation-repo-prefix", default="embodi/sim-v1-validation-v2")
    parser.add_argument("--so101-config", type=Path, default=Path("configs/so101-exp0-top.json"))
    parser.add_argument("--panda-config", type=Path, default=Path("configs/panda-exp0-top.json"))
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("reports/benchmark-exp36-validation-frames.json"),
    )
    parser.add_argument("--prepare-validation-manifest", action="store_true")
    parser.add_argument("--validation-samples-per-cell", type=int, default=256)
    parser.add_argument("--validation-frame-seed", type=int, default=36000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark-exp36-joint"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--presentations-per-step", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--model-seed", type=int, default=36001)
    parser.add_argument("--loader-seed", type=int, default=36101)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-every", type=positive_int, default=20)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_validation_manifest:
        roots = {morphology: args.validation_root for morphology in MORPHOLOGIES}
        repo_ids = {
            morphology: f"{args.validation_repo_prefix}-{morphology}"
            for morphology in MORPHOLOGIES
        }
        value = prepare_validation_manifest(
            roots,
            repo_ids,
            args.validation_manifest,
            samples_per_cell=args.validation_samples_per_cell,
            seed=args.validation_frame_seed,
        )
        print(
            json.dumps(
                {
                    "path": str(args.validation_manifest),
                    "sha256": file_sha256(args.validation_manifest),
                    "cells": len(value["cells"]),
                    "samples_per_cell": value["samples_per_cell"],
                },
                indent=2,
            )
        )
        return
    train_joint(args)


if __name__ == "__main__":
    main()
