#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from embodi.checkpoints import load_inference_policy


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def decode(policy, state: torch.Tensor, canonical: torch.Tensor, device: torch.device) -> torch.Tensor:
    state = state.to(device)
    canonical = canonical.to(device)
    normalized = (canonical - policy.canonical_mean) / policy.canonical_action_scale
    normalized_state = (state - policy.state_mean) / policy.state_std
    group_mask = torch.ones(state.shape[0], 1, dtype=torch.bool, device=device)
    decoded = policy.action_decoder(normalized, normalized_state, group_mask)
    if policy.config.decoder_residual:
        normalized_current = (state - policy.action_mean) / policy.action_std
        decoded = decoded + normalized_current.unsqueeze(1)
    return (decoded * policy.action_std + policy.action_mean).cpu()


def decoder_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    action_std: torch.Tensor,
    episodes: torch.Tensor,
    outcomes: list[dict],
) -> dict:
    normalized_error = (prediction - target) / action_std.view(1, 1, -1)
    horizon_rmse = normalized_error.square().mean(dim=(0, 2)).sqrt()
    first_error = prediction[:, 0] - target[:, 0]
    first_normalized = normalized_error[:, 0]
    target_delta = target[:, 0] - state
    predicted_delta = prediction[:, 0] - state
    moving = target_delta.abs() > 1e-3
    direction_agreement = ((target_delta.sign() == predicted_delta.sign()) & moving).sum() / moving.sum()
    episode_metrics = []
    for episode in episodes.unique(sorted=True):
        selected = episodes == episode
        episode_metrics.append(
            {
                "episode": int(episode),
                "teacher_success": bool(outcomes[int(episode)]["success"]),
                "samples": int(selected.sum()),
                "first_step_normalized_rmse": float(
                    first_normalized[selected].square().mean().sqrt()
                ),
                "all_step_normalized_rmse": float(
                    normalized_error[selected].square().mean().sqrt()
                ),
            }
        )
    return {
        "first_step_normalized_rmse": float(first_normalized.square().mean().sqrt()),
        "all_step_normalized_rmse": float(normalized_error.square().mean().sqrt()),
        "step_16_normalized_rmse": float(horizon_rmse[15]),
        "step_32_normalized_rmse": float(horizon_rmse[31]),
        "horizon_normalized_rmse": horizon_rmse.tolist(),
        "first_step_direction_agreement": float(direction_agreement),
        "first_step_per_joint": {
            name: {
                "mae_native": float(first_error[:, index].abs().mean()),
                "rmse_native": float(first_error[:, index].square().mean().sqrt()),
                "normalized_rmse": float(first_normalized[:, index].square().mean().sqrt()),
            }
            for index, name in enumerate(JOINT_NAMES)
        },
        "episodes": episode_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze held-out native decoder fidelity.")
    parser.add_argument("cache", type=Path)
    parser.add_argument("checkpoint", type=Path, nargs="+")
    parser.add_argument("--validation-start", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.validation_start < 0 or args.batch_size <= 0:
        raise ValueError("validation start must be nonnegative and batch size positive")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache.get("format_version") != 1:
        raise ValueError("unsupported decoder cache format")
    data = cache["data"]
    selected = data["episode"] >= args.validation_start
    if not selected.any():
        raise ValueError("validation split is empty")
    state = data["state"][selected]
    canonical = data["canonical"][selected]
    target = data["action"][selected]
    episodes = data["episode"][selected]
    source_core_hash = cache["provenance"]["source_core_sha256"]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    reports = []
    predictions = []
    action_scales = []
    for checkpoint in args.checkpoint:
        if file_hash(checkpoint / "core.pt") != source_core_hash:
            raise ValueError(f"decoder checkpoint core does not match cache: {checkpoint}")
        policy = load_inference_policy(checkpoint, device)
        chunks = []
        for start in range(0, len(state), args.batch_size):
            chunks.append(
                decode(
                    policy,
                    state[start : start + args.batch_size],
                    canonical[start : start + args.batch_size],
                    device,
                )
            )
        prediction = torch.cat(chunks)
        predictions.append(prediction)
        action_scales.append(policy.action_std.cpu().clone())
        reports.append(
            {
                "checkpoint": str(checkpoint),
                "config_sha256": file_hash(checkpoint / "config.json"),
                "embodiment_sha256": file_hash(checkpoint / "embodiment.pt"),
                **decoder_metrics(
                    prediction,
                    target,
                    state,
                    policy.action_std.cpu(),
                    episodes,
                    cache["outcomes"],
                ),
            }
        )
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    stacked = torch.stack(predictions)
    if any(not torch.equal(action_scales[0], scale) for scale in action_scales[1:]):
        raise ValueError("decoder checkpoints use different native action normalization")
    disagreement = (stacked / action_scales[0].view(1, 1, 1, -1)).std(dim=0, correction=0)
    first_mean = sum(report["first_step_normalized_rmse"] for report in reports) / len(reports)
    step_16_mean = sum(report["step_16_normalized_rmse"] for report in reports) / len(reports)
    diagnosis = (
        "local_approximation"
        if first_mean > 0.15
        else "horizon_degradation"
        if step_16_mean >= 2.0 * first_mean
        else "on_policy_distribution_shift"
    )
    payload = {
        "format_version": 1,
        "cache": str(args.cache),
        "cache_sha256": file_hash(args.cache),
        "source_core_sha256": source_core_hash,
        "validation_start_episode": args.validation_start,
        "samples": len(state),
        "decoders": reports,
        "mean_first_step_normalized_rmse": first_mean,
        "mean_step_16_normalized_rmse": step_16_mean,
        "cross_seed_first_step_normalized_disagreement": float(
            disagreement[:, 0].square().mean().sqrt()
        ),
        "diagnosis": diagnosis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
