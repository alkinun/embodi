from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .benchmark import BenchmarkDefinition, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv, runtime_scenario
from .benchmark_expert import run_benchmark_expert_episode


def evaluate_privileged_admission(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    *,
    limit_per_cell: int | None = None,
) -> dict:
    attempted: Counter[tuple[str, str]] = Counter()
    successes: Counter[tuple[str, str]] = Counter()
    failures = []
    for scenario in manifest.scenarios:
        cell = (scenario.morphology, scenario.task)
        if limit_per_cell is not None and attempted[cell] >= limit_per_cell:
            continue
        attempted[cell] += 1
        env = BenchmarkManipulationEnv(
            scenario.morphology,
            scenario.task,
            runtime_scenario(scenario.factors),
            render=False,
        )
        try:
            success, phase = run_benchmark_expert_episode(
                env,
                max_steps=definition.values["control"]["maximum_episode_steps"],
            )
            if success:
                successes[cell] += 1
            else:
                failures.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "morphology": scenario.morphology,
                        "task": scenario.task,
                        "phase": phase.value,
                        "diagnostics": env.diagnostic_state(),
                    }
                )
        finally:
            env.close()
    minimum = definition.values["admission_gates"]["minimum_privileged_expert_success_rate"]
    cells = {}
    for cell in sorted(attempted):
        count = attempted[cell]
        rate = successes[cell] / count
        cells[f"{cell[0]}/{cell[1]}"] = {
            "attempted": count,
            "successes": successes[cell],
            "success_rate": rate,
            "gate_passed": rate >= minimum,
        }
    return {
        "format_version": 1,
        "benchmark_id": definition.values["benchmark_id"],
        "definition_sha256": definition.sha256,
        "manifest_path": str(manifest.path),
        "manifest_sha256": file_sha256(manifest.path),
        "minimum_success_rate": minimum,
        "limit_per_cell": limit_per_cell,
        "cells": cells,
        "gate_passed": bool(cells) and all(value["gate_passed"] for value in cells.values()),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="run privileged benchmark admission")
    parser.add_argument("definition", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit-per-cell", type=int)
    args = parser.parse_args()
    if args.limit_per_cell is not None and args.limit_per_cell <= 0:
        raise ValueError("--limit-per-cell must be positive")
    definition = BenchmarkDefinition.load(args.definition)
    manifest = ScenarioManifest.load(args.manifest, definition, require_complete_counts=True)
    report = evaluate_privileged_admission(
        definition,
        manifest,
        limit_per_cell=args.limit_per_cell,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != text:
        raise FileExistsError(f"refusing to replace different admission report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text, end="")
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
