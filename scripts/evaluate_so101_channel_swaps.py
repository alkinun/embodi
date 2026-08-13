#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from embodi.checkpoints import load_inference_policy
from embodi.processing import EmbodiProcessor
from embodi.sim.environment import SO101PickPlaceEnv
from evaluate_so101 import model_batch


SWAPS = {
    "generalist": (),
    "baseline": (slice(0, 10),),
    "baseline_translation": (slice(0, 3),),
    "baseline_rotation": (slice(3, 9),),
    "baseline_opening": (slice(9, 10),),
    "baseline_pose": (slice(0, 9),),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate causal baseline/generalist canonical channel swaps.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("generalist", type=Path)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=49000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    policies = {
        "baseline": load_inference_policy(args.baseline, device),
        "generalist": load_inference_policy(args.generalist, device),
    }
    if policies["baseline"].config != policies["generalist"].config:
        raise ValueError("channel-swap policies must have identical configs")
    processor = EmbodiProcessor.from_pretrained(
        policies["baseline"].config.backbone_name,
        revision=policies["baseline"].config.backbone_revision,
        do_rescale=policies["baseline"].config.image_do_rescale,
    )
    results = []
    for name, slices in SWAPS.items():
        env = SO101PickPlaceEnv(seed=args.seed)
        outcomes = []
        try:
            for episode_index in range(args.episodes):
                if episode_index:
                    env.reset()
                steps = 0
                while steps < args.max_steps and not env.success:
                    batch = model_batch(policies["generalist"], processor, env, device)
                    generalist = policies["generalist"].predict_canonical_action_chunk(batch)[0].float().cpu().numpy()
                    if slices:
                        baseline = policies["baseline"].predict_canonical_action_chunk(batch)[0].float().cpu().numpy()
                        for channel_slice in slices:
                            generalist[..., channel_slice] = baseline[..., channel_slice]
                    native = np.stack([env.native_action_from_canonical(command) for command in generalist])
                    for action in native[: args.execution_horizon]:
                        env.apply_action(action)
                        env.step_control_period(30.0)
                        steps += 1
                        if env.success or steps >= args.max_steps:
                            break
                outcomes.append({"success": bool(env.success), "lifted": bool(env.lifted), "steps": steps})
            result = {
                "name": name,
                "episodes": args.episodes,
                "successes": sum(outcome["success"] for outcome in outcomes),
                "lifts": sum(outcome["lifted"] for outcome in outcomes),
                "mean_steps": float(np.mean([outcome["steps"] for outcome in outcomes])),
                "outcomes": outcomes,
            }
            results.append(result)
            print(f"name={name} successes={result['successes']} lifts={result['lifts']}")
        finally:
            env.close()
        gc.collect()
    payload = {"seed": args.seed, "execution_horizon": args.execution_horizon, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
