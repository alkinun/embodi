#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from embodi.sim.environment import SO101PickPlaceEnv


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure local continuity of cached SO101 IK labels.")
    parser.add_argument("cache", type=Path)
    parser.add_argument("--validation-start", type=int, required=True)
    parser.add_argument("--translation-mm", type=float, default=1.0)
    parser.add_argument("--jump-degrees", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.validation_start < 0 or args.translation_mm <= 0 or args.jump_degrees <= 0:
        raise ValueError("validation start and continuity thresholds must be positive")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache.get("format_version") != 1:
        raise ValueError("unsupported decoder cache format")
    data = cache["data"]
    selected = data["episode"] >= args.validation_start
    states = data["state"][selected].numpy()
    commands = data["canonical"][selected, 0, 0].numpy()
    targets = data["action"][selected, 0].numpy()
    episodes = data["episode"][selected].numpy()
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = SO101PickPlaceEnv(width=64, height=48, render=False)
    perturbation = args.translation_mm / 1000.0
    reconstruction_errors = []
    maximum_changes = []
    records = []
    try:
        for index, (state, command, target, episode) in enumerate(
            zip(states, commands, targets, episodes, strict=True)
        ):
            env.data.qpos[env.qpos_addresses] = env.dataset_to_sim(state)
            env.mujoco.mj_forward(env.model, env.data)
            reconstructed = env.native_action_from_canonical(command)
            reconstruction_error = float(np.max(np.abs(reconstructed - target)))
            changes = []
            for axis in range(3):
                for direction in (-1.0, 1.0):
                    candidate = command.copy()
                    candidate[axis] += direction * perturbation
                    perturbed = env.native_action_from_canonical(candidate)
                    changes.append(float(np.max(np.abs(perturbed[:5] - reconstructed[:5]))))
            maximum_change = max(changes)
            reconstruction_errors.append(reconstruction_error)
            maximum_changes.append(maximum_change)
            records.append(
                {
                    "sample": index,
                    "episode": int(episode),
                    "reconstruction_max_error": reconstruction_error,
                    "maximum_body_change_degrees": maximum_change,
                }
            )
    finally:
        env.close()
    jumps = np.asarray(maximum_changes) > args.jump_degrees
    payload = {
        "format_version": 1,
        "cache": str(args.cache),
        "cache_sha256": file_hash(args.cache),
        "validation_start_episode": args.validation_start,
        "samples": len(states),
        "translation_perturbation_mm": args.translation_mm,
        "jump_threshold_degrees": args.jump_degrees,
        "maximum_label_reconstruction_error": max(reconstruction_errors),
        "reconstruction_errors_over_0_1": int(
            (np.asarray(reconstruction_errors) > 0.1).sum()
        ),
        "reconstruction_errors_over_0_5": int(
            (np.asarray(reconstruction_errors) > 0.5).sum()
        ),
        "median_maximum_body_change_degrees": float(np.median(maximum_changes)),
        "p95_maximum_body_change_degrees": float(np.quantile(maximum_changes, 0.95)),
        "maximum_body_change_degrees": max(maximum_changes),
        "jump_samples": int(jumps.sum()),
        "jump_fraction": float(jumps.mean()),
        "translation_diagnosis": "translation_discontinuity"
        if jumps.mean() > 0.05
        else "translation_locally_stable",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
