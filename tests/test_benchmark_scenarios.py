import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodi.sim import benchmark_admission
from embodi.sim.benchmark import (
    BenchmarkDefinition,
    Scenario,
    ScenarioManifest,
    validate_disjoint_manifests,
)
from embodi.sim.benchmark_environment import runtime_scenario
from embodi.sim.benchmark_scenarios import generate_manifest_values, write_manifest


DEFINITION_PATH = Path(__file__).parents[1] / "benchmarks" / "sim-v1" / "definition.json"


def _load_generated_manifest(tmp_path: Path, split: str) -> ScenarioManifest:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    path = tmp_path / f"{split}.json"
    write_manifest(path, generate_manifest_values(definition, split))
    return ScenarioManifest.load(path, definition, require_complete_counts=True)


def test_generation_is_deterministic_complete_and_runtime_decodable(tmp_path: Path) -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    first = generate_manifest_values(definition, "development")
    second = generate_manifest_values(definition, "development")
    assert first == second

    manifest = _load_generated_manifest(tmp_path, "development")
    assert len(manifest.scenarios) == 400
    for scenario in manifest.scenarios:
        decoded = runtime_scenario(scenario.factors)
        assert decoded.object_xy == tuple(scenario.factors["workspace"]["object_xy"])
        active_bins = [
            factor["bin"] for factor in scenario.factors.values() if factor["bin"] != "baseline"
        ]
        assert len(active_bins) == 1
        assert active_bins[0].startswith("development:")


def test_default_manifests_are_physically_disjoint(tmp_path: Path) -> None:
    manifests = [
        _load_generated_manifest(tmp_path, split)
        for split in ("train", "validation", "development", "final")
    ]
    validate_disjoint_manifests(manifests)


def test_disjoint_validation_ignores_factor_bin_labels() -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    factors = generate_manifest_values(definition, "development")["scenarios"][0]["factors"]
    renamed_factors = deepcopy(factors)
    for factor in renamed_factors.values():
        factor["bin"] = "renamed"
    manifests = [
        ScenarioManifest(
            path=Path(f"{index}.json"),
            definition_sha256=definition.sha256,
            scenarios=(
                Scenario(
                    scenario_id=f"scenario-{index}",
                    split="development",
                    morphology="so101",
                    task="push_to_zone",
                    seed=index,
                    factors=value,
                ),
            ),
        )
        for index, value in enumerate((factors, renamed_factors))
    ]
    with pytest.raises(ValueError, match="physical scenario is duplicated"):
        validate_disjoint_manifests(manifests)


def test_runtime_decoder_rejects_malformed_factor_shapes() -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    factors = generate_manifest_values(definition, "development")["scenarios"][0]["factors"]
    factors["workspace"]["object_xy"] = [0.1]
    with pytest.raises(ValueError, match="runtime schema"):
        runtime_scenario(factors)


def test_manifest_writer_refuses_to_replace_different_values(tmp_path: Path) -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    path = tmp_path / "manifest.json"
    values = generate_manifest_values(definition, "development")
    write_manifest(path, values)
    changed = deepcopy(values)
    changed["split_seed"] += 1
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_manifest(path, changed)
    assert json.loads(path.read_text()) == values


def test_privileged_admission_reports_failures_and_uses_frozen_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    generated = generate_manifest_values(definition, "development")["scenarios"][:2]
    manifest_path = tmp_path / "development.json"
    manifest_path.write_text("{}")
    manifest = ScenarioManifest(
        path=manifest_path,
        definition_sha256=definition.sha256,
        scenarios=tuple(Scenario(**value) for value in generated),
    )
    environments = []
    maximum_steps = []

    class FakeEnvironment:
        def __init__(self, morphology, task, scenario, *, render):
            self.morphology = morphology
            self.task = task
            self.scenario = scenario
            self.closed = False
            environments.append(self)

        def diagnostic_state(self):
            return {"object_height_m": 0.02}

        def close(self):
            self.closed = True

    def fake_episode(env, max_steps):
        maximum_steps.append(max_steps)
        success = len(maximum_steps) == 1
        return success, SimpleNamespace(value="done" if success else "failed")

    monkeypatch.setattr(benchmark_admission, "BenchmarkManipulationEnv", FakeEnvironment)
    monkeypatch.setattr(benchmark_admission, "run_benchmark_expert_episode", fake_episode)
    report = benchmark_admission.evaluate_privileged_admission(definition, manifest)

    assert report["gate_passed"] is False
    assert report["cells"]["so101/push_to_zone"]["success_rate"] == 0.5
    assert report["failures"][0]["scenario_id"] == generated[1]["scenario_id"]
    assert maximum_steps == [500, 500]
    assert all(env.closed for env in environments)
