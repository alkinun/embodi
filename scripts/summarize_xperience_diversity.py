#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


CONDITIONS = ("one-session", "five-session")
STEPS = (500, 750, 1000)


def metric_at(path: Path, step: int, split: str) -> float:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    matches = [
        record["metrics"]["flow_loss"]
        for record in records
        if record["step"] == step and record["split"] == split
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {split} metric at step {step}: {path}")
    return float(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Experiment 034 diversity results.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(81001, 81002, 81003))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for condition in CONDITIONS:
        condition_results = []
        for seed in args.seeds:
            run = args.outputs_root / f"xperience-exp34-{condition}-seed{seed}"
            step750_path = args.reports_root / f"xperience-exp34-{condition}-seed{seed}-step750.json"
            final_path = args.reports_root / f"xperience-exp34-{condition}-seed{seed}-final.json"
            step750 = json.loads(step750_path.read_text())
            final = json.loads(final_path.read_text())
            checkpoints = {
                "500": metric_at(run / "metrics.jsonl", 500, "validation"),
                "750": float(step750["aggregate"]["flow_loss"]),
                "1000": float(final["aggregate"]["flow_loss"]),
            }
            best_step = min(checkpoints, key=checkpoints.get)
            condition_results.append(
                {
                    "seed": seed,
                    "validation_flow_loss": checkpoints,
                    "best_fixed_step": int(best_step),
                    "best_fixed_loss": checkpoints[best_step],
                    "final_training_flow_loss": metric_at(
                        run / "metrics.jsonl", 1000, "training_evaluation"
                    ),
                    "final_by_episode": final["by_episode"],
                    "final_core_sha256": final["core_sha256"],
                    "data_manifest_sha256": final["data_manifest_sha256"],
                }
            )
        results[condition] = condition_results

    one_final = [item["validation_flow_loss"]["1000"] for item in results["one-session"]]
    five_final = [item["validation_flow_loss"]["1000"] for item in results["five-session"]]
    paired_improvements = [
        (one - five) / one for one, five in zip(one_final, five_final, strict=True)
    ]
    validation_episodes = sorted(results["one-session"][0]["final_by_episode"])
    by_episode_means = {
        episode: {
            condition: mean(
                item["final_by_episode"][episode]["flow_loss"] for item in results[condition]
            )
            for condition in CONDITIONS
        }
        for episode in validation_episodes
    }
    one_mean = mean(one_final)
    five_mean = mean(five_final)
    payload = {
        "format_version": 1,
        "seeds": args.seeds,
        "fixed_steps": STEPS,
        "results": results,
        "primary": {
            "one_session_mean_final_loss": one_mean,
            "five_session_mean_final_loss": five_mean,
            "relative_mean_reduction": (one_mean - five_mean) / one_mean,
            "paired_relative_reductions": paired_improvements,
            "paired_seeds_improved": sum(value > 0 for value in paired_improvements),
            "gate_passed": (
                sum(value > 0 for value in paired_improvements) >= 2
                and (one_mean - five_mean) / one_mean >= 0.05
            ),
        },
        "final_by_episode_means": by_episode_means,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["primary"], indent=2))


if __name__ == "__main__":
    main()
