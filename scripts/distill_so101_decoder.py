#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from embodi.checkpoints import load_inference_policy
from embodi.processing import EmbodiProcessor
from embodi.sim.environment import SO101PickPlaceEnv, TASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill SO-101 deterministic IK into a frozen-core decoder.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--embodiment-from", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43000)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--rollout-policy", choices=("teacher", "learned", "mixed"), default="teacher")
    parser.add_argument("--teacher-action-probability", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def model_batch(policy, processor, env: SO101PickPlaceEnv, device: torch.device) -> dict:
    camera_frames = {"top": env.render_top, "wrist": env.render_wrist}
    images = []
    for camera_name in policy.config.camera_names:
        frame = torch.from_numpy(camera_frames[camera_name]().copy()).permute(2, 0, 1)
        images.append(frame if policy.config.image_do_rescale else frame.float() / 255.0)
    state = torch.from_numpy(env.state()).unsqueeze(0)
    canonical_state = torch.from_numpy(env.canonical_state()).unsqueeze(0)
    batch = processor(
        [images],
        [TASK],
        state,
        canonical_state=canonical_state,
        canonical_part_mask=torch.ones(1, canonical_state.shape[1], dtype=torch.bool),
    )
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def collect(policy, processor, env: SO101PickPlaceEnv, args, device: torch.device) -> tuple[dict, list[dict]]:
    states = []
    canonical_chunks = []
    native_chunks = []
    episode_indices = []
    outcomes = []
    rollout_random = random.Random(args.seed)
    for episode_index in range(args.episodes):
        if episode_index:
            env.reset()
        steps = 0
        chunks = 0
        while steps < args.max_steps and not env.success:
            batch = model_batch(policy, processor, env, device)
            canonical = policy.predict_canonical_action_chunk(batch)[0].float().cpu().numpy()
            native = np.stack([env.native_action_from_canonical(command) for command in canonical])
            if args.rollout_policy == "teacher":
                executed = native
            else:
                learned = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                use_teacher = (
                    args.rollout_policy == "mixed"
                    and rollout_random.random() < args.teacher_action_probability
                )
                executed = native if use_teacher else learned
            states.append(env.state())
            canonical_chunks.append(canonical)
            native_chunks.append(native)
            episode_indices.append(episode_index)
            chunks += 1
            for action in executed[: args.execution_horizon]:
                env.apply_action(action)
                env.step_control_period(30.0)
                steps += 1
                if env.success or steps >= args.max_steps:
                    break
        outcomes.append(
            {
                "episode": episode_index,
                "success": bool(env.success),
                "lifted": bool(env.lifted),
                "steps": steps,
                "chunks": chunks,
            }
        )
        print(f"collect episode={episode_index} success={env.success} lifted={env.lifted} chunks={chunks}")
    data = {
        "state": torch.from_numpy(np.stack(states)).float(),
        "canonical": torch.from_numpy(np.stack(canonical_chunks)).float(),
        "action": torch.from_numpy(np.stack(native_chunks)).float(),
        "episode": torch.tensor(episode_indices),
    }
    return data, outcomes


def decoder_loss(policy, data: dict, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    state = data["state"][indices].to(device)
    canonical = data["canonical"][indices].to(device)
    normalized = (canonical - policy.canonical_mean) / policy.canonical_action_scale
    batch = {
        "observation.state": state,
        "action": data["action"][indices].to(device),
        "action_group_mask": torch.ones(indices.numel(), 1, dtype=torch.bool, device=device),
    }
    return policy._decoder_loss(normalized, batch, "mean")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_provenance(args: argparse.Namespace) -> dict:
    embodiment = args.embodiment_from or args.checkpoint
    return {
        "source_core_sha256": file_hash(args.checkpoint / "core.pt"),
        "source_embodiment_sha256": file_hash(embodiment / "embodiment.pt"),
        "collection_seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "execution_horizon": args.execution_horizon,
        "rollout_policy": args.rollout_policy,
        "teacher_action_probability": args.teacher_action_probability,
    }


def main() -> None:
    args = parse_args()
    if min(args.episodes, args.validation_episodes, args.max_steps, args.execution_horizon, args.steps, args.batch_size) <= 0:
        raise ValueError("episode, rollout, and training counts must be positive")
    if args.validation_episodes >= args.episodes:
        raise ValueError("validation episodes must leave at least one training episode")
    if not 0.0 <= args.teacher_action_probability <= 1.0:
        raise ValueError("teacher action probability must be in [0, 1]")
    os.environ.setdefault("MUJOCO_GL", "egl")
    training_seed = args.seed if args.training_seed is None else args.training_seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(training_seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    policy = load_inference_policy(args.checkpoint, device)
    if args.embodiment_from is not None:
        policy.load_embodiment_checkpoint(args.embodiment_from)
    if policy.config.format_version != 3:
        raise ValueError("decoder distillation requires a format-v3 checkpoint")
    provenance = dataset_provenance(args)
    if args.dataset_cache is not None and args.dataset_cache.is_file():
        cached = torch.load(args.dataset_cache, map_location="cpu", weights_only=True)
        if cached.get("format_version") != 1 or cached.get("provenance") != provenance:
            raise ValueError("distillation dataset cache provenance does not match this run")
        data = cached["data"]
        outcomes = cached["outcomes"]
        print(f"loaded dataset cache={args.dataset_cache} samples={len(data['episode'])}")
    else:
        processor = EmbodiProcessor.from_pretrained(
            policy.config.backbone_name,
            revision=policy.config.backbone_revision,
            do_rescale=policy.config.image_do_rescale,
        )
        env = SO101PickPlaceEnv(seed=args.seed)
        try:
            data, outcomes = collect(policy, processor, env, args, device)
        finally:
            env.close()
        if args.dataset_cache is not None:
            args.dataset_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"format_version": 1, "provenance": provenance, "data": data, "outcomes": outcomes},
                args.dataset_cache,
            )
            print(f"saved dataset cache={args.dataset_cache} samples={len(data['episode'])}")
    if args.collect_only:
        if args.dataset_cache is None:
            raise ValueError("--collect-only requires --dataset-cache")
        return

    policy.configure_stage("predicted_decoder")
    policy.config.native_action_delta_limits = ()
    optimizer = torch.optim.AdamW(policy.action_decoder.parameters(), lr=args.lr, weight_decay=1e-10)
    validation_start = args.episodes - args.validation_episodes
    train_indices = torch.where(data["episode"] < validation_start)[0]
    validation_indices = torch.where(data["episode"] >= validation_start)[0]
    generator = torch.Generator().manual_seed(training_seed)
    metrics = []
    policy.train()
    for step in range(1, args.steps + 1):
        selected = train_indices[
            torch.randint(train_indices.numel(), (args.batch_size,), generator=generator)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = decoder_loss(policy, data, selected, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.action_decoder.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            policy.eval()
            with torch.no_grad():
                validation_loss = decoder_loss(policy, data, validation_indices, device).item()
            policy.train()
            entry = {"step": step, "train_loss": loss.item(), "validation_loss": validation_loss}
            metrics.append(entry)
            print(" ".join(f"{name}={value}" for name, value in entry.items()))

    final_dir = args.output_dir / "final"
    policy.save_checkpoint(final_dir, step=args.steps)
    source_core_hash = file_hash(args.checkpoint / "core.pt")
    output_core_hash = file_hash(final_dir / "core.pt")
    if source_core_hash != output_core_hash:
        raise RuntimeError("decoder distillation modified the frozen core checkpoint")
    report = {
        "source_checkpoint": str(args.checkpoint),
        "embodiment_from": str(args.embodiment_from) if args.embodiment_from else None,
        "seed": args.seed,
        "training_seed": training_seed,
        "dataset_cache": str(args.dataset_cache) if args.dataset_cache else None,
        "episodes": args.episodes,
        "validation_episodes": args.validation_episodes,
        "execution_horizon": args.execution_horizon,
        "rollout_policy": args.rollout_policy,
        "teacher_action_probability": args.teacher_action_probability,
        "samples": len(data["episode"]),
        "training_samples": train_indices.numel(),
        "validation_samples": validation_indices.numel(),
        "teacher_successes": sum(outcome["success"] for outcome in outcomes),
        "teacher_lifts": sum(outcome["lifted"] for outcome in outcomes),
        "outcomes": outcomes,
        "metrics": metrics,
        "source_core_sha256": source_core_hash,
        "output_core_sha256": output_core_hash,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "distillation_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
