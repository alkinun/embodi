import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from embodi.sim import benchmark_admission
from embodi.sim.benchmark import Scenario


def _definition() -> SimpleNamespace:
    return SimpleNamespace(
        values={
            "benchmark_id": "embodi-sim-v1",
            "control": {"maximum_episode_steps": 500},
            "admission_gates": {"minimum_privileged_expert_success_rate": 0.5},
        },
        sha256="definition-sha",
    )


def test_privileged_admission_counts_cells_failures_and_closes_environments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("manifest")
    manifest = SimpleNamespace(
        path=manifest_path,
        scenarios=(
            Scenario("push-success", "train", "so101", "push_to_zone", 1, {"x": 1}),
            Scenario("push-failure", "train", "so101", "push_to_zone", 2, {"x": 2}),
            Scenario("lift-failure", "train", "so101", "lift_object", 3, {"x": 3}),
        ),
    )
    environments = []
    runtime_factors = []

    class FakeEnvironment:
        def __init__(self, morphology, task, scenario, *, render):
            self.morphology = morphology
            self.task = task
            self.scenario = scenario
            self.render = render
            self.closed = False
            environments.append(self)

        def diagnostic_state(self):
            return {"task": self.task}

        def close(self):
            self.closed = True

    monkeypatch.setattr(benchmark_admission, "BenchmarkManipulationEnv", FakeEnvironment)
    monkeypatch.setattr(benchmark_admission, "runtime_scenario", lambda factors: runtime_factors.append(factors) or factors)
    monkeypatch.setattr(benchmark_admission, "file_sha256", lambda path: "manifest-sha")
    monkeypatch.setattr(
        benchmark_admission,
        "run_benchmark_expert_episode",
        lambda env, max_steps: (env.task == "push_to_zone", SimpleNamespace(value="done")),
    )

    report = benchmark_admission.evaluate_privileged_admission(_definition(), manifest)

    assert report["gate_passed"] is False
    assert report["cells"] == {
        "so101/lift_object": {
            "attempted": 1,
            "successes": 0,
            "success_rate": 0.0,
            "gate_passed": False,
        },
        "so101/push_to_zone": {
            "attempted": 2,
            "successes": 2,
            "success_rate": 1.0,
            "gate_passed": True,
        },
    }
    assert [failure["scenario_id"] for failure in report["failures"]] == ["lift-failure"]
    assert runtime_factors == [{"x": 1}, {"x": 2}, {"x": 3}]
    assert all(environment.closed for environment in environments)


def test_privileged_admission_with_no_scenarios_does_not_pass_gate(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("manifest")
    report = benchmark_admission.evaluate_privileged_admission(
        _definition(),
        SimpleNamespace(path=manifest_path, scenarios=()),
    )

    assert report["cells"] == {}
    assert report["gate_passed"] is False


@pytest.mark.parametrize("gate_passed", [True, False])
def test_admission_main_writes_report_and_exits_on_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_passed: bool,
) -> None:
    definition = _definition()
    manifest = SimpleNamespace(path=tmp_path / "manifest.json", scenarios=())
    report = {"gate_passed": gate_passed, "cells": {}, "failures": []}
    monkeypatch.setattr(benchmark_admission.BenchmarkDefinition, "load", lambda path: definition)
    monkeypatch.setattr(benchmark_admission.ScenarioManifest, "load", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(benchmark_admission, "evaluate_privileged_admission", lambda *args, **kwargs: report)
    output = tmp_path / f"admission-{gate_passed}.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodi-admit-benchmark", "definition.json", "manifest.json", str(output)],
    )

    if gate_passed:
        benchmark_admission.main()
    else:
        with pytest.raises(SystemExit, match="1"):
            benchmark_admission.main()

    assert json.loads(output.read_text()) == report


def test_admission_main_rejects_nonpositive_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["embodi-admit-benchmark", "definition.json", "manifest.json", "output.json", "--limit-per-cell", "0"],
    )

    with pytest.raises(ValueError, match="limit-per-cell must be positive"):
        benchmark_admission.main()
