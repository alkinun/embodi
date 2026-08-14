from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from .token_config import (
    CANONICAL_BLOCK_WIDTH,
    EmbodiConfig,
    part_name_features,
)
from .token_canonical import reanchor_canonical_actions
from .token_policy import EmbodiPolicy
from .processing import EmbodiProcessor, split_lerobot_batch
from .xperience import (
    CachedXperienceDataset,
    CachedXperienceEpisode,
    MultiEpisodeCachedXperienceDataset,
    select_fixed_episode_xperience_split,
    select_multi_episode_xperience_split,
    select_xperience_split,
)


class _SmokeBackbone(nn.Module):
    hidden_size = 32

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(state_dim, self.hidden_size)

    def forward(self, state: torch.Tensor, **_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.projection(state)
        return features, torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)


def smoke_test() -> None:
    config = EmbodiConfig(expert_width=32, expert_layers=2, expert_heads=4, denoising_steps=2)
    policy = EmbodiPolicy(config, backbone=_SmokeBackbone(10))
    batch = {
        "observation.state": torch.randn(2, config.state_dim),
        "action": torch.randn(2, config.chunk_size, config.action_dim),
        "canonical_state": torch.randn(2, config.part_count, CANONICAL_BLOCK_WIDTH),
        "canonical_action": torch.randn(2, config.chunk_size, config.part_count, CANONICAL_BLOCK_WIDTH),
    }
    loss, metrics = policy(batch)
    loss.backward()
    actions = policy.predict_action_chunk({"observation.state": batch["observation.state"]})
    print(f"loss={metrics['flow_loss']:.4f} chunk_shape={tuple(actions.shape)}")


def split_episode_indices(total_episodes: int, validation_episodes: int) -> tuple[list[int], list[int]]:
    if validation_episodes <= 0 or validation_episodes >= total_episodes:
        raise ValueError("validation_episodes must be between 1 and total_episodes - 1")
    boundary = total_episodes - validation_episodes
    return list(range(boundary)), list(range(boundary, total_episodes))


def resolve_training_seeds(
    seed: int, model_seed: int | None, loader_seed: int | None
) -> tuple[int, int]:
    return (
        seed if model_seed is None else model_seed,
        seed if loader_seed is None else loader_seed,
    )


def prepare_model_batch(
    raw_batch: dict,
    processor: EmbodiProcessor,
    device: torch.device,
    camera_names: tuple[str, ...],
    config: EmbodiConfig,
    process_images: bool = True,
) -> dict:
    if process_images:
        images, tasks = split_lerobot_batch(raw_batch, camera_names)
    else:
        images, tasks = [], raw_batch.get("task", [])
    canonical_actions = raw_batch.get("canonical_action")
    canonical_state = raw_batch.get("canonical_state")
    batch_size = canonical_actions.shape[0] if canonical_actions is not None else len(tasks)
    part_count = canonical_actions.shape[2] if canonical_actions is not None else config.part_count
    if part_count != config.part_count:
        raise ValueError("homogeneous training batches must match the configured part layout")
    part_kind = torch.tensor([part.kind_id for part in config.parts], dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    name_features = torch.tensor(
        [part_name_features(part.name) for part in config.parts], dtype=torch.float32
    ).unsqueeze(0).expand(batch_size, -1, -1)
    channel_mask = torch.tensor(
        [part.channel_mask for part in config.parts], dtype=torch.bool
    ).unsqueeze(0).expand(batch_size, -1, -1)
    part_mask = raw_batch.get("canonical_part_mask")
    if part_mask is None:
        part_mask = torch.ones(batch_size, part_count, dtype=torch.bool)
    elif part_mask.ndim == 1 and part_count == 1:
        part_mask = part_mask.unsqueeze(-1)
    part_mask = part_mask.bool()
    pad_masks = [
        raw_batch[key]
        for key in ("canonical_action_is_pad", "canonical_state_is_pad", "action_is_pad")
        if key in raw_batch
    ]
    if any(not torch.equal(pad_masks[0], mask) for mask in pad_masks[1:]):
        raise ValueError("canonical state/action and native action padding masks disagree")
    action_is_pad = pad_masks[0] if pad_masks else None
    if canonical_actions is not None and canonical_state is not None and canonical_state.ndim == 4:
        canonical_actions, canonical_state = reanchor_canonical_actions(
            canonical_actions, canonical_state, part_kind, channel_mask, part_mask
        )
    values = {
        "observation.state": raw_batch.get("observation.state"),
        "action": raw_batch.get("action"),
        "canonical_action": canonical_actions,
        "canonical_state": canonical_state,
        "canonical_part_mask": part_mask,
        "canonical_part_kind": part_kind,
        "canonical_part_name_features": name_features,
        "canonical_channel_mask": channel_mask,
        "action_group_mask": raw_batch.get("action_group_mask"),
        "action_is_pad": action_is_pad,
    }
    if process_images:
        model_batch = processor(
            images,
            tasks,
            values.pop("observation.state"),
            actions=values.pop("action"),
            canonical_actions=values.pop("canonical_action"),
            **values,
        )
    else:
        model_batch = {key: value for key, value in values.items() if value is not None}
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in model_batch.items()
    }


def gradient_norm(parameters) -> float:
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared = parameter.grad.detach().float().square().sum()
        total = squared if total is None else total + squared
    return 0.0 if total is None else total.sqrt().item()


@torch.no_grad()
def evaluate(
    policy: EmbodiPolicy,
    loader: DataLoader,
    processor: EmbodiProcessor,
    device: torch.device,
    max_batches: int,
    seed: int,
) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals: dict[str, float] = {}
    batches = 0
    for raw_batch in loader:
        model_batch = prepare_model_batch(
            raw_batch,
            processor,
            device,
            policy.config.camera_names,
            policy.config,
            process_images=policy.stage != "decoder",
        )
        canonical_actions = policy._canonical_action_targets(model_batch)
        noise = torch.randn(
            canonical_actions.shape,
            device=device,
            dtype=canonical_actions.dtype,
            generator=generator,
        )
        time = torch.rand(canonical_actions.shape[0], device=device, dtype=torch.float32, generator=generator)
        _, metrics = policy(model_batch, noise=noise, time=time)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1
        if batches >= max_batches:
            break
    policy.train(was_training)
    if batches == 0:
        raise RuntimeError("validation loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    for name in ("steps", "batch_size", "accumulation_steps", "eval_every", "validation_batches"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    config = EmbodiConfig.load(args.config) if args.config else EmbodiConfig()
    config = replace(
        config,
        camera_names=tuple(args.cameras) if args.cameras else config.camera_names,
        image_do_rescale=False if args.correct_image_rescale else config.image_do_rescale,
    )
    if args.resume and any((args.init_from, args.init_core, args.init_embodiment)):
        raise ValueError("--resume cannot be combined with initialization checkpoints")
    model_seed, loader_seed = resolve_training_seeds(
        args.seed, args.model_seed, args.loader_seed
    )
    print(f"training seeds: data={args.seed} model={model_seed} loader={loader_seed}")
    random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = EmbodiPolicy(config).to(device)
    policy.configure_stage(args.stage)
    processor = EmbodiProcessor.from_pretrained(
        config.backbone_name,
        revision=config.backbone_revision,
        do_rescale=config.image_do_rescale,
    )
    data_report: dict = {}
    training_evaluation_dataset = None
    if args.dataset_format == "xperience":
        if args.stage != "core":
            raise ValueError("Xperience experiment 0 supports --stage core only")
        if config.chunk_size != 32 or config.n_action_steps != 32:
            raise ValueError("Xperience experiment 0 requires a 32-step horizon")
        if config.camera_names != ("stereo_left",):
            raise ValueError("Xperience experiment 0 requires camera_names=[stereo_left]")
        if len(config.parts) != 1 or config.parts[0].name != "primary_effector" or config.parts[0].kind != "pose_scalar":
            raise ValueError("Xperience experiment 0 requires primary_effector / pose_scalar")
        if args.dataset_root is None:
            raise ValueError("Xperience training requires --dataset-root")
        source_manifest_path = args.dataset_root / "xperience_manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError("Xperience cache is missing xperience_manifest.json")
        source_manifest = json.loads(source_manifest_path.read_text())
        episode_paths = [entry["path"] for entry in source_manifest.get("episodes", [])]
        fixed_train_paths = args.xperience_train_episode_path
        fixed_validation_paths = args.xperience_validation_episode_path
        if bool(fixed_train_paths) != bool(fixed_validation_paths):
            raise ValueError("fixed Xperience splitting requires both train and validation episode paths")
        if args.xperience_episode_path and fixed_train_paths:
            raise ValueError("--xperience-episode-path cannot be combined with fixed episode paths")
        if fixed_train_paths:
            selected_paths = list(dict.fromkeys(fixed_train_paths + fixed_validation_paths))
        else:
            selected_paths = [args.xperience_episode_path] if args.xperience_episode_path else episode_paths
        if not selected_paths:
            raise ValueError("Xperience manifest contains no episodes")
        episodes = [
            CachedXperienceEpisode(
                args.dataset_root / episode_path,
                closed_ratio=args.xperience_closed_ratio,
                open_ratio=args.xperience_open_ratio,
            )
            for episode_path in selected_paths
        ]
        if fixed_train_paths:
            path_indices = {path: index for index, path in enumerate(selected_paths)}
            split = select_fixed_episode_xperience_split(
                episodes,
                train_episode_indices=[path_indices[path] for path in fixed_train_paths],
                validation_episode_indices=[path_indices[path] for path in fixed_validation_paths],
                train_clips=args.xperience_train_clips,
                validation_clips=args.xperience_validation_clips,
                seed=args.seed,
            )
            train_dataset = MultiEpisodeCachedXperienceDataset(episodes, split.train)
            validation_dataset = MultiEpisodeCachedXperienceDataset(episodes, split.validation)
            training_evaluation_dataset = MultiEpisodeCachedXperienceDataset(episodes, split.train)
            train_records = [
                {"episode_path": selected_paths[index], **asdict(anchor)} for index, anchor in split.train
            ]
            validation_records = [
                {"episode_path": selected_paths[index], **asdict(anchor)} for index, anchor in split.validation
            ]
            rejection_counts = split.rejection_counts
            train_episode_paths = [selected_paths[index] for index in split.train_episode_indices]
            validation_episode_paths = [selected_paths[index] for index in split.validation_episode_indices]
        elif len(episodes) == 1:
            split = select_xperience_split(
                episodes[0],
                train_clips=args.xperience_train_clips,
                validation_clips=args.xperience_validation_clips,
                seed=args.seed,
            )
            train_dataset = CachedXperienceDataset(episodes[0], split.train)
            validation_dataset = CachedXperienceDataset(episodes[0], split.validation)
            training_evaluation_dataset = CachedXperienceDataset(episodes[0], split.train)
            train_records = [{"episode_path": selected_paths[0], **asdict(anchor)} for anchor in split.train]
            validation_records = [
                {"episode_path": selected_paths[0], **asdict(anchor)} for anchor in split.validation
            ]
            rejection_counts = split.rejection_counts
            train_episode_paths = validation_episode_paths = selected_paths
        else:
            split = select_multi_episode_xperience_split(
                episodes,
                train_clips=args.xperience_train_clips,
                validation_clips=args.xperience_validation_clips,
                seed=args.seed,
            )
            train_dataset = MultiEpisodeCachedXperienceDataset(episodes, split.train)
            validation_dataset = MultiEpisodeCachedXperienceDataset(episodes, split.validation)
            training_evaluation_dataset = MultiEpisodeCachedXperienceDataset(episodes, split.train)
            train_records = [
                {"episode_path": selected_paths[index], **asdict(anchor)} for index, anchor in split.train
            ]
            validation_records = [
                {"episode_path": selected_paths[index], **asdict(anchor)} for index, anchor in split.validation
            ]
            rejection_counts = split.rejection_counts
            train_episode_paths = [selected_paths[index] for index in split.train_episode_indices]
            validation_episode_paths = [selected_paths[index] for index in split.validation_episode_indices]
        data_report = {
            "source": {
                "repo_id": source_manifest.get("repo_id"),
                "revision": source_manifest.get("revision"),
                "episode_paths": selected_paths,
                "source_fps": [episode.source_fps for episode in episodes],
            },
            "target_fps": 30,
            "horizon": 32,
            "seed": args.seed,
            "opening_calibration": {
                "closed_ratio": args.xperience_closed_ratio,
                "open_ratio": args.xperience_open_ratio,
            },
            "train_episode_paths": train_episode_paths,
            "validation_episode_paths": validation_episode_paths,
            "rejection_counts": rejection_counts,
            "train": train_records,
            "validation": validation_records,
        }
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data_manifest.json").write_text(json.dumps(data_report, indent=2) + "\n")
        train_units = len(split.train)
        validation_units = len(split.validation)
    else:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        except ImportError as error:
            raise RuntimeError("dataset training dependencies are missing; run `uv sync --all-extras`") from error
        if args.dataset_root is not None and not (args.dataset_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"no local LeRobot dataset found at {args.dataset_root}; run `uv run embodi-generate` first"
            )
        metadata = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
        features = metadata.features
        if "canonical_action" not in features:
            raise ValueError("dataset is missing required canonical_action labels")
        if "canonical_state" not in features:
            raise ValueError("dataset is missing canonical_state required to anchor action chunks")
        if "canonical_part_mask" not in features:
            raise ValueError("dataset is missing canonical_part_mask")
        expected_shape = (config.part_count, CANONICAL_BLOCK_WIDTH)
        for feature_name in ("canonical_action", "canonical_state"):
            if tuple(features[feature_name]["shape"]) != expected_shape:
                raise ValueError(f"{feature_name} must have frame shape {expected_shape}")
        schema_path = Path(metadata.root) / "meta" / "embodi.json"
        if not schema_path.is_file():
            raise FileNotFoundError("dataset is missing meta/embodi.json part descriptors")
        schema = json.loads(schema_path.read_text())
        configured_parts = json.loads(json.dumps([asdict(part) for part in config.parts]))
        if schema.get("format_version") != 3 or schema.get("parts") != configured_parts:
            raise ValueError("dataset part descriptors do not match the model config")
        if args.stage in ("decoder", "predicted_decoder", "refine") and "action" not in features:
            raise ValueError(f"{args.stage} stage requires native action labels")
        offsets = [step / metadata.fps for step in range(config.chunk_size)]
        delta_timestamps = {"canonical_action": offsets, "canonical_state": offsets}
        if "action" in features:
            delta_timestamps["action"] = offsets
        train_episodes, validation_episodes = split_episode_indices(
            metadata.total_episodes, args.validation_episodes
        )
        train_dataset = LeRobotDataset(
            args.dataset, root=args.dataset_root, episodes=train_episodes, delta_timestamps=delta_timestamps
        )
        validation_dataset = LeRobotDataset(
            args.dataset, root=args.dataset_root, episodes=validation_episodes, delta_timestamps=delta_timestamps
        )
        policy.set_normalization_stats(metadata.stats)
        train_units = len(train_episodes)
        validation_units = len(validation_episodes)
    loader_generator = torch.Generator().manual_seed(loader_seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    training_evaluation_loader = (
        DataLoader(training_evaluation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        if training_evaluation_dataset is not None
        else None
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-10)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("wandb is missing; run `uv sync --all-extras`") from error
        train_config = {
            key: value
            for key, value in vars(args).items()
            if key not in {"smoke_test"}
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or config.model_name,
            mode=args.wandb_mode,
            config={"model": asdict(config), "training": train_config},
        )
        wandb_run.summary["parameters/total"] = sum(parameter.numel() for parameter in policy.parameters())
        wandb_run.summary["parameters/trainable"] = sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        )
        wandb_run.summary["data/train_units"] = train_units
        wandb_run.summary["data/validation_units"] = validation_units

    def learning_rate_multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    start_step = 0
    if args.resume:
        start_step = policy.load_checkpoint(args.resume, optimizer, scheduler)
        if start_step >= args.steps:
            raise ValueError("resume checkpoint step must be lower than --steps")
    elif args.init_from:
        source_step = policy.load_checkpoint(args.init_from, allow_camera_mismatch=True)
        print(f"initialized model from {args.init_from} at source_step={source_step}")
    else:
        if args.init_core:
            components = tuple(args.init_core_component or ("backbone", "expert"))
            policy.load_core_checkpoint(args.init_core, components=components)
            print(f"initialized shared core components={components} from {args.init_core}")
        if args.init_embodiment:
            policy.load_embodiment_checkpoint(args.init_embodiment)
            print(f"initialized embodiment from {args.init_embodiment}")

    validation_metrics = evaluate(
        policy,
        validation_loader,
        processor,
        device,
        args.validation_batches,
        args.validation_seed,
    )
    print(f"step={start_step} " + " ".join(f"validation_{name}={value:.6f}" for name, value in validation_metrics.items()))
    if training_evaluation_loader is not None:
        training_metrics = evaluate(
            policy, training_evaluation_loader, processor, device, len(training_evaluation_loader), args.validation_seed
        )
        print(f"step={start_step} " + " ".join(f"training_{name}={value:.6f}" for name, value in training_metrics.items()))
    if wandb_run is not None:
        wandb_run.log({f"validation/{name}": value for name, value in validation_metrics.items()}, step=start_step)

    from tqdm.auto import tqdm

    iterator = iter(loader)
    policy.train()
    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        total=args.steps,
        initial=start_step,
        desc=config.model_name,
        unit="step",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        accumulated_metrics: dict[str, float] = {}
        for _ in range(args.accumulation_steps):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            model_batch = prepare_model_batch(
                raw_batch,
                processor,
                device,
                policy.config.camera_names,
                policy.config,
                process_images=policy.stage != "decoder",
            )
            loss, metrics = policy(model_batch)
            (loss / args.accumulation_steps).backward()
            for name, value in metrics.items():
                accumulated_metrics[name] = accumulated_metrics.get(name, 0.0) + value / args.accumulation_steps
        should_log = step % args.log_every == 0
        component_norms = {}
        if should_log:
            component_norms = {
                "grad_norm/expert": gradient_norm(policy.expert.parameters()),
                "grad_norm/action_decoder": gradient_norm(policy.action_decoder.parameters()),
                "grad_norm/state_adapter": gradient_norm(policy.state_adapter.parameters()),
                "grad_norm/state_projection": gradient_norm(policy.backbone.state_projection.parameters()),
                "grad_norm/lora": gradient_norm(
                    parameter
                    for name, parameter in policy.backbone.named_parameters()
                    if "lora_" in name
                ),
            }
        total_grad_norm = torch.nn.utils.clip_grad_norm_(policy.get_optim_params(), 10.0).item()
        optimizer.step()
        scheduler.step()
        if should_log:
            progress.set_postfix(loss=f"{sum(accumulated_metrics.values()):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if args.no_progress:
                print(f"step={step} " + " ".join(f"{name}={value:.6f}" for name, value in accumulated_metrics.items()))
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{name}": value for name, value in accumulated_metrics.items()},
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "grad_norm/total": total_grad_norm,
                        **component_norms,
                    },
                    step=step,
                )
        if step % args.eval_every == 0:
            validation_metrics = evaluate(
                policy,
                validation_loader,
                processor,
                device,
                args.validation_batches,
                args.validation_seed,
            )
            progress.write(f"step={step} " + " ".join(f"validation_{name}={value:.6f}" for name, value in validation_metrics.items()))
            if training_evaluation_loader is not None:
                training_metrics = evaluate(
                    policy,
                    training_evaluation_loader,
                    processor,
                    device,
                    len(training_evaluation_loader),
                    args.validation_seed,
                )
                progress.write(f"step={step} " + " ".join(f"training_{name}={value:.6f}" for name, value in training_metrics.items()))
            if wandb_run is not None:
                wandb_run.log({f"validation/{name}": value for name, value in validation_metrics.items()}, step=step)
        if step % args.save_every == 0:
            checkpoint_dir = Path(args.output_dir) / f"step-{step:07d}"
            policy.save_checkpoint(checkpoint_dir, optimizer, scheduler, step)
            if wandb_run is not None:
                wandb_run.summary["checkpoint/latest"] = str(checkpoint_dir)
    if args.steps % args.eval_every:
        validation_metrics = evaluate(
            policy,
            validation_loader,
            processor,
            device,
            args.validation_batches,
            args.validation_seed,
        )
        print(f"step={args.steps} " + " ".join(f"validation_{name}={value:.6f}" for name, value in validation_metrics.items()))
        if wandb_run is not None:
            wandb_run.log({f"validation/{name}": value for name, value in validation_metrics.items()}, step=args.steps)
    final_dir = Path(args.output_dir) / "final"
    policy.save_checkpoint(final_dir, optimizer, scheduler, args.steps)
    if wandb_run is not None:
        wandb_run.summary["checkpoint/final"] = str(final_dir)
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    model_name = EmbodiConfig().model_name
    parser = argparse.ArgumentParser(description="train an embodi model")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset", default="embodi/sim-so101-pickplace")
    parser.add_argument("--dataset-format", choices=("lerobot", "xperience"), default="lerobot")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cameras", nargs="+", choices=("top", "wrist"))
    parser.add_argument("--correct-image-rescale", action="store_true")
    parser.add_argument("--output-dir", default=f"outputs/{model_name}")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="default seed and data-selection seed")
    parser.add_argument("--model-seed", type=int, help="override model initialization seed")
    parser.add_argument("--loader-seed", type=int, help="override training DataLoader seed")
    parser.add_argument("--xperience-episode-path")
    parser.add_argument("--xperience-train-episode-path", action="append", default=[])
    parser.add_argument("--xperience-validation-episode-path", action="append", default=[])
    parser.add_argument("--xperience-train-clips", type=int, default=100)
    parser.add_argument("--xperience-validation-clips", type=int, default=20)
    parser.add_argument("--xperience-closed-ratio", type=float, default=0.18)
    parser.add_argument("--xperience-open-ratio", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--init-core", type=Path)
    parser.add_argument("--init-core-component", action="append", choices=("backbone", "expert"))
    parser.add_argument("--init-embodiment", type=Path)
    parser.add_argument(
        "--stage", choices=("core", "decoder", "predicted_decoder", "refine"), default="refine"
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
