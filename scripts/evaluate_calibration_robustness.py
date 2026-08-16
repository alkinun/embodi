#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from embodi.checkpoints import load_inference_policy
from embodi.processing import EmbodiProcessor, camera_frame_to_tensor
from embodi.sim.environment import SO101PickPlaceEnv, TASK


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_batch(policy, processor, actual_env, model_env, offset, device):
    native_state = actual_env.state()
    estimated_state = native_state.copy()
    estimated_state[:5] += offset
    model_env.data.qpos[model_env.qpos_addresses] = model_env.dataset_to_sim(estimated_state)
    model_env.mujoco.mj_forward(model_env.model, model_env.data)
    canonical_state = torch.from_numpy(model_env.canonical_state()).unsqueeze(0)
    camera_frames = {"top": actual_env.render_top, "wrist": actual_env.render_wrist}
    images = [
        camera_frame_to_tensor(
            camera_frames[name]().copy(),
            processor_rescales=policy.config.image_do_rescale,
        )
        for name in policy.config.camera_names
    ]
    batch = processor(
        [images],
        [TASK],
        torch.from_numpy(native_state).unsqueeze(0),
        canonical_state=canonical_state,
        canonical_part_mask=torch.ones(1, canonical_state.shape[1], dtype=torch.bool),
    )
    return (
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()},
        estimated_state,
    )


@torch.no_grad()
def evaluate_offset(policy, processor, offset, args, device) -> dict:
    actual_env = SO101PickPlaceEnv(seed=args.seed)
    model_env = SO101PickPlaceEnv(width=64, height=48, seed=0, render=False)
    outcomes = []
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            if episode:
                actual_env.reset()
            policy.reset()
            steps = 0
            while steps < args.max_steps and not actual_env.success:
                batch, estimated_state = model_batch(
                    policy, processor, actual_env, model_env, offset, device
                )
                noise = torch.zeros(
                    1,
                    policy.config.chunk_size,
                    policy.config.part_count,
                    10,
                    device=device,
                )
                canonical = policy.predict_canonical_action_chunk(batch, noise=noise)[0].cpu().numpy()
                model_env.data.qpos[model_env.qpos_addresses] = model_env.dataset_to_sim(
                    estimated_state
                )
                model_env.mujoco.mj_forward(model_env.model, model_env.data)
                native = np.stack(
                    [model_env.native_action_from_canonical(command) for command in canonical]
                )
                native[:, :5] -= offset
                for action in native[: args.execution_horizon]:
                    actual_env.apply_action(action)
                    actual_env.step_control_period(30.0)
                    steps += 1
                    if actual_env.success or steps >= args.max_steps:
                        break
            outcomes.append(
                {
                    "episode": episode,
                    "success": bool(actual_env.success),
                    "lifted": bool(actual_env.lifted),
                    "steps": steps,
                }
            )
            print(
                f"offset={offset.tolist()} episode={episode} "
                f"success={actual_env.success} lifted={actual_env.lifted} steps={steps}"
            )
    finally:
        model_env.close()
        actual_env.close()
    return {
        "offset_degrees": offset.tolist(),
        "successes": sum(outcome["success"] for outcome in outcomes),
        "lifts": sum(outcome["lifted"] for outcome in outcomes),
        "mean_steps": float(np.mean([outcome["steps"] for outcome in outcomes])),
        "elapsed_seconds": time.perf_counter() - started,
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed SO101 joint calibration errors.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0, 5.0))
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33000)
    parser.add_argument("--offset-seed", type=int, default=33001)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.draws, args.episodes, args.max_steps, args.execution_horizon) <= 0:
        raise ValueError("draw, episode, step, and horizon counts must be positive")
    if not args.magnitudes or any(value < 0 for value in args.magnitudes):
        raise ValueError("calibration magnitudes must be nonnegative")
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    policy = load_inference_policy(args.checkpoint, device)
    processor = EmbodiProcessor.from_pretrained(
        policy.config.backbone_name,
        revision=policy.config.backbone_revision,
        do_rescale=policy.config.image_do_rescale,
    )
    random = np.random.default_rng(args.offset_seed)
    results = []
    for magnitude in args.magnitudes:
        offsets = (
            [np.zeros(5, dtype=np.float32)]
            if magnitude == 0
            else [
                random.uniform(-magnitude, magnitude, size=5).astype(np.float32)
                for _ in range(args.draws)
            ]
        )
        for draw, offset in enumerate(offsets):
            result = evaluate_offset(policy, processor, offset, args, device)
            result.update(magnitude_degrees=magnitude, draw=draw)
            results.append(result)
    payload = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "config_sha256": file_hash(args.checkpoint / "config.json"),
        "core_sha256": file_hash(args.checkpoint / "core.pt"),
        "embodiment_sha256": file_hash(args.checkpoint / "embodiment.pt"),
        "episodes": args.episodes,
        "seed": args.seed,
        "offset_seed": args.offset_seed,
        "max_steps": args.max_steps,
        "execution_horizon": args.execution_horizon,
        "magnitudes_degrees": args.magnitudes,
        "draws": args.draws,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2))
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
