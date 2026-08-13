#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from embodi.canonical import rotation_6d_to_matrix
from embodi.checkpoints import load_inference_policy
from embodi.processing import EmbodiProcessor
from embodi.train import prepare_model_batch


@torch.no_grad()
def integrate(policy, batch: dict, noise: torch.Tensor, *, steps: int, method: str) -> torch.Tensor:
    context, context_mask, descriptors, _ = policy.encode(batch)
    expert = policy.expert
    actions = noise.to(expert.action_projection.weight.dtype)
    step_size = 1.0 / steps
    common = {
        "context": context,
        "part_kind": descriptors["part_kind"],
        "part_name_features": descriptors["part_name_features"],
        "channel_mask": descriptors["channel_mask"],
        "part_mask": descriptors["part_mask"],
        "context_mask": context_mask,
        "action_is_pad": batch.get("action_is_pad"),
    }
    for step in range(steps):
        time = torch.full((actions.shape[0],), step * step_size, device=actions.device)
        velocity = expert(actions, time, **common)
        if method == "euler":
            actions = actions + step_size * velocity
        elif method == "heun":
            proposal = actions + step_size * velocity
            next_time = torch.full(
                (actions.shape[0],), (step + 1) * step_size, device=actions.device
            )
            next_velocity = expert(proposal, next_time, **common)
            actions = actions + 0.5 * step_size * (velocity + next_velocity)
        else:
            raise ValueError(f"unknown integration method: {method}")
    return actions * policy.canonical_action_scale + policy.canonical_mean


def trajectory_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    translation = torch.linalg.vector_norm(predicted[..., :3] - target[..., :3], dim=-1) * 100
    predicted_rotation = rotation_6d_to_matrix(predicted[..., 3:9])
    target_rotation = rotation_6d_to_matrix(target[..., 3:9])
    relative = predicted_rotation.transpose(-1, -2) @ target_rotation
    angle = torch.rad2deg(
        torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    )
    scalar = (predicted[..., 9] - target[..., 9]).abs()
    return {
        "translation_cm_mean": translation.mean().item(),
        "translation_cm_first": translation[:, 0].mean().item(),
        "translation_cm_endpoint": translation[:, -1].mean().item(),
        "rotation_deg_mean": angle.mean().item(),
        "rotation_deg_first": angle[:, 0].mean().item(),
        "rotation_deg_endpoint": angle[:, -1].mean().item(),
        "scalar_mae_mean": scalar.mean().item(),
        "scalar_mae_first": scalar[:, 0].mean().item(),
        "scalar_mae_endpoint": scalar[:, -1].mean().item(),
        "scalar_out_of_range_fraction": ((predicted[..., 9] < 0) | (predicted[..., 9] > 1)).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate integrated canonical samples in physical units.")
    parser.add_argument("checkpoint", type=Path, nargs="+")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    metadata = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    validation_ids = list(range(metadata.total_episodes - args.validation_episodes, metadata.total_episodes))
    results = []
    for checkpoint in args.checkpoint:
        policy = load_inference_policy(checkpoint, device)
        processor = EmbodiProcessor.from_pretrained(
            policy.config.backbone_name,
            revision=policy.config.backbone_revision,
            do_rescale=policy.config.image_do_rescale,
        )
        offsets = [step / metadata.fps for step in range(policy.config.chunk_size)]
        dataset = LeRobotDataset(
            args.dataset,
            root=args.dataset_root,
            episodes=validation_ids,
            delta_timestamps={"canonical_action": offsets, "canonical_state": offsets},
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        accumulators: dict[str, list[dict[str, float]]] = {}
        generator = torch.Generator(device=device).manual_seed(args.seed)
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            batch = prepare_model_batch(
                raw_batch,
                processor,
                device,
                policy.config.camera_names,
                policy.config,
            )
            target = policy._canonical_action_targets(batch)
            noise = torch.randn(target.shape, device=device, generator=generator)
            for method, steps in (("euler", 10), ("euler", 20), ("euler", 50), ("heun", 10), ("heun", 20)):
                name = f"{method}_{steps}"
                predicted = integrate(policy, batch, noise, steps=steps, method=method)
                accumulators.setdefault(name, []).append(trajectory_metrics(predicted, target))
        integrations = {
            name: {
                metric: float(np.mean([batch_metrics[metric] for batch_metrics in values]))
                for metric in values[0]
            }
            for name, values in accumulators.items()
        }
        results.append({"checkpoint": str(checkpoint), "batches": args.batches, "integrations": integrations})
    payload = {"seed": args.seed, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
