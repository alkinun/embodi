from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkDefinition, ScenarioManifest
from .benchmark_environment import BenchmarkScenario, default_scenario


SPLIT_SEEDS = {
    "train": 41000,
    "validation": 42000,
    "development": 43000,
    "final": 44000,
}


def _signed_offset(random_generator: random.Random, lower: float, upper: float) -> float:
    return random_generator.choice((-1.0, 1.0)) * random_generator.uniform(lower, upper)


def _scenario_factors(
    morphology: str,
    task: str,
    split: str,
    index: int,
    random_generator: random.Random,
    factor_ids: tuple[str, ...],
) -> dict[str, Any]:
    baseline = default_scenario(morphology, task)
    object_xy = list(baseline.object_xy)
    target_xy = list(baseline.target_xy)
    half_extents = list(baseline.object_half_extents)
    mass = baseline.object_mass
    friction = baseline.friction
    tolerance = baseline.target_tolerance
    distractors: list[list[float]] = []
    visual_domain = "default"
    table_rgba = [0.09, 0.10, 0.12, 1.0]
    light_rgb = [0.95, 0.95, 0.95]
    axis = factor_ids[index % len(factor_ids)]
    severity = {"train": 0, "validation": 0, "development": 1, "final": 2}[split]

    if axis == "workspace":
        bounds = ((0.001, 0.008), (0.009, 0.018), (0.019, 0.03))[severity]
        object_xy[0] -= random_generator.uniform(*bounds)
        object_xy[1] += _signed_offset(random_generator, *bounds)
        target_xy[0] += _signed_offset(random_generator, bounds[0] / 2, bounds[1] / 2)
        target_xy[1] += _signed_offset(random_generator, bounds[0] / 2, bounds[1] / 2)
    elif axis == "object_geometry":
        ranges = ((0.97, 1.03), (0.93, 0.96), (1.04, 1.08))
        half_extents = [
            value * random_generator.uniform(*ranges[severity]) for value in half_extents
        ]
    elif axis == "mass":
        ranges = ((0.9, 1.1), (0.75, 0.88), (1.12, 1.35))
        mass *= random_generator.uniform(*ranges[severity])
    elif axis == "friction":
        ranges = ((1.1, 1.3), (0.9, 1.05), (1.35, 1.55))
        friction = random_generator.uniform(*ranges[severity])
    elif axis == "target_tolerance":
        ranges = ((0.052, 0.058), (0.050, 0.052), (0.049, 0.050))
        tolerance = random_generator.uniform(*ranges[severity])
    elif axis == "distractors":
        count = (1, 2, 3)[severity]
        for distractor_index in range(count):
            side = -1 if distractor_index % 2 else 1
            distractors.append(
                [
                    object_xy[0] + side * (0.08 + 0.025 * distractor_index),
                    object_xy[1] + side * 0.10,
                ]
            )
            distractors[-1][0] += random_generator.uniform(-0.01, 0.01)
            distractors[-1][1] += random_generator.uniform(-0.01, 0.01)
    elif axis == "visual_domain":
        visual_domain = ("train", "development", "final")[severity]
        if severity == 0:
            table_rgba[:3] = [random_generator.uniform(0.07, 0.15) for _ in range(3)]
            light_rgb = [random_generator.uniform(0.88, 1.0) for _ in range(3)]
        elif severity == 1:
            table_rgba[:3] = [random_generator.uniform(0.45, 0.62) for _ in range(3)]
            light_rgb = [random_generator.uniform(0.75, 1.0) for _ in range(3)]
        else:
            table_rgba[:3] = [
                random_generator.uniform(0.24, 0.34),
                random_generator.uniform(0.14, 0.23),
                random_generator.uniform(0.07, 0.15),
            ]
            light_rgb = [
                random_generator.uniform(0.9, 1.0),
                random_generator.uniform(0.68, 0.86),
                random_generator.uniform(0.48, 0.7),
            ]
    else:
        raise ValueError(f"unsupported factor axis: {axis}")

    factor_bins = {
        factor_id: f"{split}:{factor_id}" if factor_id == axis else "baseline"
        for factor_id in factor_ids
    }
    factors = {
        "workspace": {"bin": factor_bins["workspace"], "object_xy": object_xy, "target_xy": target_xy},
        "object_geometry": {"bin": factor_bins["object_geometry"], "half_extents": half_extents},
        "mass": {"bin": factor_bins["mass"], "kg": mass},
        "friction": {"bin": factor_bins["friction"], "coefficient": friction},
        "target_tolerance": {"bin": factor_bins["target_tolerance"], "meters": tolerance},
        "distractors": {"bin": factor_bins["distractors"], "positions": distractors},
        "visual_domain": {
            "bin": factor_bins["visual_domain"],
            "id": visual_domain,
            "table_rgba": table_rgba,
            "light_rgb": light_rgb,
        },
    }
    BenchmarkScenario(
        object_xy=tuple(object_xy),
        target_xy=tuple(target_xy),
        object_half_extents=tuple(half_extents),
        object_mass=mass,
        friction=friction,
        target_tolerance=tolerance,
        distractor_xy=tuple(tuple(value) for value in distractors),
        visual_domain=visual_domain,
        table_rgba=tuple(table_rgba),
        light_rgb=tuple(light_rgb),
    )
    return factors


def generate_manifest_values(
    definition: BenchmarkDefinition,
    split: str,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    split_definition = definition.split(split)
    seed = SPLIT_SEEDS[split] if seed is None else seed
    random_generator = random.Random(seed)
    scenarios = []
    for morphology in definition.morphology_ids:
        for task in definition.tasks_for_split(split):
            for index in range(split_definition["scenarios_per_cell"]):
                scenarios.append(
                    {
                        "scenario_id": f"{split}-{morphology}-{task}-{index:04d}",
                        "split": split,
                        "morphology": morphology,
                        "task": task,
                        "seed": random_generator.randrange(2**31),
                        "factors": _scenario_factors(
                            morphology,
                            task,
                            split,
                            index,
                            random_generator,
                            definition.factor_ids,
                        ),
                    }
                )
    return {
        "format_version": 1,
        "benchmark_id": definition.values["benchmark_id"],
        "definition_sha256": definition.sha256,
        "split_seed": seed,
        "scenarios": scenarios,
    }


def write_manifest(path: str | Path, values: dict[str, Any]) -> None:
    path = Path(path)
    text = json.dumps(values, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"refusing to replace different scenario manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="prepare a deterministic benchmark scenario manifest")
    parser.add_argument("definition", type=Path)
    parser.add_argument("split", choices=tuple(SPLIT_SEEDS))
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    definition = BenchmarkDefinition.load(args.definition)
    values = generate_manifest_values(definition, args.split, seed=args.seed)
    write_manifest(args.output, values)
    manifest = ScenarioManifest.load(args.output, definition, require_complete_counts=True)
    print(
        json.dumps(
            {
                "path": str(manifest.path),
                "split": args.split,
                "split_seed": values["split_seed"],
                "scenarios": len(manifest.scenarios),
                "definition_sha256": definition.sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
