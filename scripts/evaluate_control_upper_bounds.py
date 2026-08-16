#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from embodi.sim.environment import SO101PickPlaceEnv
from embodi.sim.expert import Phase, PrivilegedPickPlaceExpert
from embodi.token_config import CANONICAL_ACTION_SCALE, CANONICAL_MEAN, EmbodiConfig
from embodi.token_decoder import PartTokenActionDecoder


def load_decoder(checkpoint: Path, device: torch.device):
    config = EmbodiConfig.load(checkpoint / "config.json")
    decoder = PartTokenActionDecoder(
        config.parts,
        config.action_group_part_names,
        config.action_group_state_dims,
        config.action_group_native_dims,
        config.inactive_action_modes,
        hidden_width=config.decoder_hidden_width,
        layers=config.decoder_layers,
    ).to(device)
    payload = torch.load(checkpoint / "embodiment.pt", map_location=device, weights_only=True)
    decoder.load_state_dict(payload["action_decoder"])
    decoder.eval()
    return decoder, {
        key: payload[key].to(device)
        for key in ("state_mean", "state_std", "action_mean", "action_std")
    }, config


@torch.no_grad()
def decode_action(
    decoder,
    stats: dict[str, torch.Tensor],
    config: EmbodiConfig,
    state: np.ndarray,
    canonical: np.ndarray,
) -> np.ndarray:
    device = stats["state_mean"].device
    normalized_canonical = (
        torch.from_numpy(canonical).to(device).view(1, 1, 1, 10)
        - torch.tensor(CANONICAL_MEAN, device=device)
    ) / torch.tensor(CANONICAL_ACTION_SCALE, device=device)
    normalized_state = (
        torch.from_numpy(state).to(device).view(1, -1) - stats["state_mean"]
    ) / stats["state_std"]
    normalized_action = decoder(normalized_canonical, normalized_state)
    if config.decoder_residual:
        current_in_action_units = (
            torch.from_numpy(state).to(device).view(1, 1, -1) - stats["action_mean"]
        ) / stats["action_std"]
        normalized_action = normalized_action + current_in_action_units
    action = normalized_action * stats["action_std"] + stats["action_mean"]
    if config.native_action_delta_limits:
        limits = np.asarray(config.native_action_delta_limits, dtype=np.float32)
        action_numpy = action[0, 0].float().cpu().numpy()
        return np.clip(action_numpy, state - limits, state + limits)
    return action[0, 0].float().cpu().numpy()


def run_episode(seed: int, decoder_bundle=None, deterministic_decoder: bool = False, max_steps: int = 750) -> dict:
    env = SO101PickPlaceEnv(seed=seed)
    expert = PrivilegedPickPlaceExpert(env)
    errors = []
    phase_errors: dict[str, list[float]] = {}
    try:
        for _step in range(max_steps):
            state = env.state()
            oracle_action = expert.action()
            if decoder_bundle is None and not deterministic_decoder:
                executed = oracle_action
            else:
                canonical = env.canonical_arm_action(oracle_action)
                executed = (
                    env.native_action_from_canonical(canonical)
                    if deterministic_decoder
                    else decode_action(*decoder_bundle, state, canonical)
                )
                error = float(np.sqrt(np.mean(np.square(executed - oracle_action))))
                errors.append(error)
                phase_errors.setdefault(expert.phase.value, []).append(error)
            env.apply_action(executed)
            env.step_control_period(30.0)
            if expert.terminal:
                break
        return {
            "success": expert.phase == Phase.DONE,
            "lifted": bool(env.lifted),
            "terminal_phase": expert.phase.value,
            "steps": _step + 1,
            "native_rmse": float(np.mean(errors)) if errors else 0.0,
            "phase_native_rmse": {
                phase: float(np.mean(values)) for phase, values in phase_errors.items()
            },
        }
    finally:
        env.close()


def summarize(name: str, outcomes: list[dict]) -> dict:
    return {
        "name": name,
        "episodes": len(outcomes),
        "successes": sum(outcome["success"] for outcome in outcomes),
        "success_rate": float(np.mean([outcome["success"] for outcome in outcomes])),
        "lifts": sum(outcome["lifted"] for outcome in outcomes),
        "lift_rate": float(np.mean([outcome["lifted"] for outcome in outcomes])),
        "mean_native_rmse": float(np.mean([outcome["native_rmse"] for outcome in outcomes])),
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure native-oracle and exact-canonical decoder upper bounds.")
    parser.add_argument("checkpoint", type=Path, nargs="*")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("reports/control-upper-bounds.json"))
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    results = []
    oracle = [run_episode(args.seed + index) for index in range(args.episodes)]
    results.append(summarize("native_oracle", oracle))
    deterministic = [
        run_episode(args.seed + index, deterministic_decoder=True)
        for index in range(args.episodes)
    ]
    results.append(summarize("deterministic_canonical_decoder", deterministic))
    for checkpoint in args.checkpoint:
        decoder_bundle = load_decoder(checkpoint, device)
        outcomes = [
            run_episode(args.seed + index, decoder_bundle=decoder_bundle)
            for index in range(args.episodes)
        ]
        results.append(summarize(str(checkpoint), outcomes))
    payload = {"seed": args.seed, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
