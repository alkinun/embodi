#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from embodi.canonical import rotation_6d_to_matrix
from embodi.checkpoints import load_inference_policy
from embodi.processing import EmbodiProcessor
from embodi.sim.environment import JOINT_NAMES, SO101PickPlaceEnv
from embodi.train import prepare_model_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose baseline/generalist SO-101 trajectory errors.")
    parser.add_argument("checkpoint", type=Path, nargs="+")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rotation_error_degrees(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_matrix = rotation_6d_to_matrix(predicted)
    target_matrix = rotation_6d_to_matrix(target)
    relative = predicted_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "p90": np.quantile(values, 0.9, axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    if min(args.validation_episodes, args.samples, args.execution_horizon) <= 0:
        raise ValueError("validation, sample, and horizon counts must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    metadata = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    validation_ids = list(range(metadata.total_episodes - args.validation_episodes, metadata.total_episodes))
    env = SO101PickPlaceEnv(width=64, height=48, seed=0)
    results = []
    try:
        for checkpoint in args.checkpoint:
            policy = load_inference_policy(checkpoint, device)
            horizon = min(args.execution_horizon, policy.config.chunk_size)
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
                delta_timestamps={
                    "canonical_action": offsets,
                    "canonical_state": offsets,
                    "action": offsets,
                },
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            translation = []
            rotation = []
            opening = []
            native_predicted = []
            native_target = []
            phase_values: dict[str, list[np.ndarray]] = {"open": [], "closed": [], "transition": []}
            with torch.no_grad():
                for sample_index, raw in enumerate(loader):
                    if sample_index >= args.samples:
                        break
                    batch = prepare_model_batch(
                        raw, processor, device, policy.config.camera_names, policy.config
                    )
                    predicted = policy.predict_canonical_action_chunk(batch)[0, :horizon, 0].float().cpu()
                    target = policy._canonical_action_targets(batch)[0, :horizon, 0].float().cpu()
                    translation.append(torch.linalg.vector_norm(predicted[:, :3] - target[:, :3], dim=-1).numpy() * 100)
                    rotation.append(rotation_error_degrees(predicted[:, 3:9], target[:, 3:9]).numpy())
                    opening.append((predicted[:, 9] - target[:, 9]).abs().numpy())

                    state = batch["observation.state"][0].float().cpu().numpy()
                    native = batch["action"][0, :horizon].float().cpu().numpy()
                    env.data.qpos[env.qpos_addresses] = env.dataset_to_sim(state)
                    env.mujoco.mj_forward(env.model, env.data)
                    predicted_ik = np.stack(
                        [env.native_action_from_canonical(command.numpy()) for command in predicted]
                    )
                    target_ik = np.stack(
                        [env.native_action_from_canonical(command.numpy()) for command in target]
                    )
                    predicted_error = np.abs(predicted_ik - native)
                    target_error = np.abs(target_ik - native)
                    native_predicted.append(predicted_error)
                    native_target.append(target_error)
                    opening_target = float(target[0, 9])
                    phase = "open" if opening_target > 0.8 else "closed" if opening_target < 0.2 else "transition"
                    phase_values[phase].append(predicted_error[0])

            translation_array = np.stack(translation)
            rotation_array = np.stack(rotation)
            opening_array = np.stack(opening)
            native_predicted_array = np.stack(native_predicted)
            native_target_array = np.stack(native_target)
            results.append(
                {
                    "checkpoint": str(checkpoint),
                    "samples": len(translation),
                    "execution_horizon": horizon,
                    "translation_error_cm_by_step": summarize(translation_array),
                    "rotation_error_deg_by_step": summarize(rotation_array),
                    "opening_absolute_error_by_step": summarize(opening_array),
                    "predicted_canonical_ik_native_absolute_error_by_step_and_joint": summarize(native_predicted_array),
                    "target_canonical_ik_native_absolute_error_by_step_and_joint": summarize(native_target_array),
                    "first_step_native_mae_by_opening_phase": {
                        phase: {
                            "samples": len(values),
                            "joint_mae": np.mean(values, axis=0).tolist(),
                        }
                        for phase, values in phase_values.items()
                        if values
                    },
                }
            )
    finally:
        env.close()
    payload = {"joint_names": JOINT_NAMES, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
