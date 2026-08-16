from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1


def _unique_ids(values: list[dict[str, Any]], field: str) -> tuple[str, ...]:
    ids = tuple(value.get("id") for value in values)
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{field} must contain non-empty string ids")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field} ids must be unique")
    return ids


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class BenchmarkDefinition:
    path: Path
    values: dict[str, Any]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkDefinition:
        path = Path(path)
        values = json.loads(path.read_text())
        definition = cls(path=path, values=values, sha256=file_sha256(path))
        definition.validate()
        return definition

    @property
    def morphology_ids(self) -> tuple[str, ...]:
        return tuple(value["id"] for value in self.values["morphologies"])

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(value["id"] for value in self.values["tasks"])

    @property
    def split_ids(self) -> tuple[str, ...]:
        return tuple(value["id"] for value in self.values["splits"])

    @property
    def factor_ids(self) -> tuple[str, ...]:
        return tuple(value["id"] for value in self.values["factor_axes"])

    def split(self, split_id: str) -> dict[str, Any]:
        try:
            return next(value for value in self.values["splits"] if value["id"] == split_id)
        except StopIteration as error:
            raise ValueError(f"unknown benchmark split: {split_id}") from error

    def tasks_for_split(self, split_id: str) -> tuple[str, ...]:
        scope = self.split(split_id)["task_scope"]
        if scope == "all":
            return self.task_ids
        return tuple(value["id"] for value in self.values["tasks"] if value["pretraining"])

    def validate(self) -> None:
        if self.values.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"benchmark format_version must be {FORMAT_VERSION}")
        if not isinstance(self.values.get("benchmark_id"), str) or not self.values["benchmark_id"]:
            raise ValueError("benchmark_id must be a non-empty string")
        morphology_ids = _unique_ids(self.values.get("morphologies", []), "morphologies")
        task_ids = _unique_ids(self.values.get("tasks", []), "tasks")
        split_ids = _unique_ids(self.values.get("splits", []), "splits")
        factor_ids = _unique_ids(self.values.get("factor_axes", []), "factor_axes")
        track_ids = _unique_ids(self.values.get("control_tracks", []), "control_tracks")
        if len(morphology_ids) < 2:
            raise ValueError("cross-embodiment benchmark requires at least two morphologies")
        if len(task_ids) < 2 or not any(not value.get("pretraining") for value in self.values["tasks"]):
            raise ValueError("benchmark requires multiple tasks and at least one held-out task")
        if not factor_ids:
            raise ValueError("benchmark requires explicit factor axes")
        if set(track_ids) != {"geometry", "learned_state", "full_learned"}:
            raise ValueError("benchmark must define the three required control tracks")
        if "development" not in split_ids or "final" not in split_ids:
            raise ValueError("benchmark requires development and final splits")
        for split in self.values["splits"]:
            if split.get("task_scope") not in {"pretraining", "all"}:
                raise ValueError("split task_scope must be pretraining or all")
            if not isinstance(split.get("scenarios_per_cell"), int) or split["scenarios_per_cell"] <= 0:
                raise ValueError("scenarios_per_cell must be a positive integer")
        final = self.split("final")
        if final.get("allows_model_selection"):
            raise ValueError("final split cannot allow model selection")
        budgets = self.values.get("adaptation_budgets", [])
        if (
            not budgets
            or any(not isinstance(value, int) or value <= 0 for value in budgets)
            or budgets != sorted(set(budgets))
        ):
            raise ValueError("adaptation_budgets must be unique increasing positive integers")
        gates = self.values.get("admission_gates", {})
        for name in (
            "minimum_privileged_expert_success_rate",
            "maximum_geometry_success_drop",
            "maximum_command_clipping_rate",
        ):
            value = gates.get(name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    morphology: str
    task: str
    seed: int
    factors: dict[str, Any]


@dataclass(frozen=True)
class ScenarioManifest:
    path: Path
    definition_sha256: str
    scenarios: tuple[Scenario, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        definition: BenchmarkDefinition,
        *,
        require_complete_counts: bool = False,
    ) -> ScenarioManifest:
        path = Path(path)
        values = json.loads(path.read_text())
        if values.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"scenario manifest format_version must be {FORMAT_VERSION}")
        if values.get("benchmark_id") != definition.values["benchmark_id"]:
            raise ValueError("scenario manifest benchmark_id does not match definition")
        if values.get("definition_sha256") != definition.sha256:
            raise ValueError("scenario manifest definition_sha256 does not match definition")
        scenarios = tuple(Scenario(**value) for value in values.get("scenarios", []))
        manifest = cls(path=path, definition_sha256=values["definition_sha256"], scenarios=scenarios)
        manifest.validate(definition, require_complete_counts=require_complete_counts)
        return manifest

    def validate(self, definition: BenchmarkDefinition, *, require_complete_counts: bool) -> None:
        if not self.scenarios:
            raise ValueError("scenario manifest must not be empty")
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if any(not value for value in scenario_ids) or len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_id values must be unique non-empty strings")
        factor_ids = set(definition.factor_ids)
        counts: Counter[tuple[str, str, str]] = Counter()
        for scenario in self.scenarios:
            if scenario.split not in definition.split_ids:
                raise ValueError(f"unknown split in scenario {scenario.scenario_id}: {scenario.split}")
            if scenario.morphology not in definition.morphology_ids:
                raise ValueError(
                    f"unknown morphology in scenario {scenario.scenario_id}: {scenario.morphology}"
                )
            if scenario.task not in definition.tasks_for_split(scenario.split):
                raise ValueError(f"task {scenario.task} is not admitted to split {scenario.split}")
            if set(scenario.factors) != factor_ids:
                raise ValueError(
                    f"scenario {scenario.scenario_id} factors must exactly match the definition"
                )
            counts[(scenario.split, scenario.morphology, scenario.task)] += 1
        if not require_complete_counts:
            return
        present_splits = {scenario.split for scenario in self.scenarios}
        for split_id in present_splits:
            expected = definition.split(split_id)["scenarios_per_cell"]
            for morphology in definition.morphology_ids:
                for task in definition.tasks_for_split(split_id):
                    actual = counts[(split_id, morphology, task)]
                    if actual != expected:
                        raise ValueError(
                            f"{split_id}/{morphology}/{task} has {actual} scenarios; expected {expected}"
                        )


def validate_disjoint_manifests(manifests: list[ScenarioManifest]) -> None:
    scenario_ids: set[str] = set()
    physical_cases: dict[str, str] = {}
    for manifest in manifests:
        for scenario in manifest.scenarios:
            if scenario.scenario_id in scenario_ids:
                raise ValueError(f"scenario_id occurs in multiple manifests: {scenario.scenario_id}")
            scenario_ids.add(scenario.scenario_id)
            physical_case = json.dumps(
                {
                    "morphology": scenario.morphology,
                    "task": scenario.task,
                    "factors": {
                        factor_id: {
                            key: value
                            for key, value in factor.items()
                            if key != "bin"
                        }
                        for factor_id, factor in scenario.factors.items()
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if physical_case in physical_cases:
                raise ValueError(
                    "physical scenario is duplicated: "
                    f"{physical_cases[physical_case]} and {scenario.scenario_id}"
                )
            physical_cases[physical_case] = scenario.scenario_id


def main() -> None:
    parser = argparse.ArgumentParser(description="validate an immutable Embodi simulation benchmark")
    parser.add_argument("definition", type=Path)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--require-complete-counts", action="store_true")
    args = parser.parse_args()
    definition = BenchmarkDefinition.load(args.definition)
    manifests = [
        ScenarioManifest.load(
            path,
            definition,
            require_complete_counts=args.require_complete_counts,
        )
        for path in args.manifest
    ]
    validate_disjoint_manifests(manifests)
    print(
        json.dumps(
            {
                "benchmark_id": definition.values["benchmark_id"],
                "definition_sha256": definition.sha256,
                "morphologies": definition.morphology_ids,
                "tasks": definition.task_ids,
                "manifests": [
                    {"path": str(manifest.path), "scenarios": len(manifest.scenarios)}
                    for manifest in manifests
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
