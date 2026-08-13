#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from embodi.checkpoints import load_inference_policy
from embodi.processing import EmbodiProcessor
from embodi.sim.environment import SO101PickPlaceEnv, TASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic SO-101 simulator rollouts.")
    parser.add_argument("checkpoint", type=Path, nargs="+")
    parser.add_argument("--embodiment-from", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--execution-horizon", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic-decoder", action="store_true")
    parser.add_argument("--clip-deterministic-actions", action="store_true")
    parser.add_argument("--disable-action-limits", action="store_true")
    parser.add_argument("--noise-mode", choices=("random", "zero"), default="random")
    parser.add_argument("--noise-refresh", choices=("chunk", "episode"), default="chunk")
    parser.add_argument("--cube-x-range", type=float, nargs=2, default=(0.28, 0.32))
    parser.add_argument("--cube-y-range", type=float, nargs=2, default=(-0.025, 0.025))
    parser.add_argument("--output", type=Path, default=Path("reports/so101-evaluation.json"))
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


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    execution_horizon: int,
    device: torch.device,
    deterministic_decoder: bool,
    clip_deterministic_actions: bool,
    disable_action_limits: bool,
    embodiment_from: Path | None,
    noise_mode: str,
    noise_refresh: str,
    cube_x_range: tuple[float, float],
    cube_y_range: tuple[float, float],
) -> dict:
    policy = load_inference_policy(checkpoint, device)
    if embodiment_from is not None:
        policy.load_embodiment_checkpoint(embodiment_from)
    if disable_action_limits:
        policy.config.native_action_delta_limits = ()
    processor = EmbodiProcessor.from_pretrained(
        policy.config.backbone_name,
        revision=policy.config.backbone_revision,
        do_rescale=policy.config.image_do_rescale,
    )
    env = SO101PickPlaceEnv(seed=seed, cube_x_range=cube_x_range, cube_y_range=cube_y_range)
    outcomes = []
    started = time.perf_counter()
    try:
        for episode_index in range(episodes):
            if episode_index:
                env.reset()
            policy.reset()
            initial_cube_position = env.data.xpos[env.cube_body_id].copy()
            steps = 0
            chunks = 0
            while steps < max_steps and not env.success:
                batch = model_batch(policy, processor, env, device)
                noise_index = chunks if noise_refresh == "chunk" else 0
                generator = torch.Generator(device=device).manual_seed(
                    seed + episode_index * 10_000 + noise_index
                )
                noise_shape = (1, policy.config.chunk_size, policy.config.part_count, 10)
                noise = (
                    torch.zeros(noise_shape, device=device)
                    if noise_mode == "zero"
                    else torch.randn(*noise_shape, device=device, generator=generator)
                )
                if deterministic_decoder:
                    canonical_chunk = (
                        policy.predict_canonical_action_chunk(batch, noise=noise)[0]
                        .float()
                        .cpu()
                        .numpy()
                    )
                    chunk = np.stack(
                        [env.native_action_from_canonical(command) for command in canonical_chunk]
                    )
                    if clip_deterministic_actions and policy.config.native_action_delta_limits:
                        current = env.state()
                        limits = np.asarray(policy.config.native_action_delta_limits, dtype=np.float32)
                        chunk = np.clip(chunk, current - limits, current + limits)
                else:
                    chunk = policy.predict_action_chunk(batch, noise=noise)[0].float().cpu().numpy()
                chunks += 1
                for action in chunk[:execution_horizon]:
                    env.apply_action(action)
                    env.step_control_period(30.0)
                    steps += 1
                    if env.success or steps >= max_steps:
                        break
            outcome = {
                "episode": episode_index,
                "initial_cube_xy": initial_cube_position[:2].tolist(),
                "success": bool(env.success),
                "lifted": bool(env.lifted),
                "steps": steps,
                "chunks": chunks,
            }
            outcomes.append(outcome)
            print(
                f"checkpoint={checkpoint} episode={episode_index} success={env.success} "
                f"lifted={env.lifted} steps={steps}"
            )
    finally:
        env.close()
    successes = sum(outcome["success"] for outcome in outcomes)
    lifts = sum(outcome["lifted"] for outcome in outcomes)
    return {
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "seed": seed,
        "successes": successes,
        "success_rate": successes / episodes,
        "lifts": lifts,
        "lift_rate": lifts / episodes,
        "mean_steps": float(np.mean([outcome["steps"] for outcome in outcomes])),
        "elapsed_seconds": time.perf_counter() - started,
        "outcomes": outcomes,
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.execution_horizon <= 0:
        raise ValueError("episode, step, and horizon counts must be positive")
    cube_x_range = tuple(args.cube_x_range)
    cube_y_range = tuple(args.cube_y_range)
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    results = []
    for checkpoint in args.checkpoint:
        results.append(
            evaluate_checkpoint(
                checkpoint,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                execution_horizon=args.execution_horizon,
                device=device,
                deterministic_decoder=args.deterministic_decoder,
                clip_deterministic_actions=args.clip_deterministic_actions,
                disable_action_limits=args.disable_action_limits,
                embodiment_from=args.embodiment_from,
                noise_mode=args.noise_mode,
                noise_refresh=args.noise_refresh,
                cube_x_range=cube_x_range,
                cube_y_range=cube_y_range,
            )
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "frequency_hz": 30,
        "max_steps": args.max_steps,
        "execution_horizon": args.execution_horizon,
        "deterministic_decoder": args.deterministic_decoder,
        "clip_deterministic_actions": args.clip_deterministic_actions,
        "disable_action_limits": args.disable_action_limits,
        "embodiment_from": str(args.embodiment_from) if args.embodiment_from else None,
        "noise_mode": args.noise_mode,
        "noise_refresh": args.noise_refresh,
        "cube_x_range": cube_x_range,
        "cube_y_range": cube_y_range,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
