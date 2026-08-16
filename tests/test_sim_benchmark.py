import json
from copy import deepcopy
from pathlib import Path

import pytest

from embodi.sim.benchmark import BenchmarkDefinition, ScenarioManifest
from embodi.sim.tasks import TASKS, task_by_id


DEFINITION_PATH = Path(__file__).parents[1] / "benchmarks" / "sim-v1" / "definition.json"


def test_committed_benchmark_definition_is_valid() -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    expected_hash = (DEFINITION_PATH.parent / "definition.sha256").read_text().split()[0]
    assert definition.sha256 == expected_hash
    assert definition.morphology_ids == ("so101", "panda")
    assert definition.tasks_for_split("train") == (
        "push_to_zone",
        "lift_object",
        "pick_place_bin",
    )
    assert definition.tasks_for_split("final")[-1] == "stack_object"
    assert definition.split("final")["allows_model_selection"] is False


def test_task_contracts_match_frozen_benchmark() -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    frozen = {value["id"]: value for value in definition.values["tasks"]}
    assert set(TASKS) == set(frozen)
    for task_id, task in TASKS.items():
        expected = frozen[task_id]
        assert task_by_id(task_id) is task
        assert task.instruction == expected["instruction"]
        assert task.pretraining == expected["pretraining"]
        assert task.success_predicate == expected["success"]["predicate"]
        assert task.dwell_steps == expected["success"]["dwell_steps"]
        assert task.requires_release == expected["success"]["requires_release"]


def test_definition_rejects_model_selection_on_final_split(tmp_path: Path) -> None:
    values = json.loads(DEFINITION_PATH.read_text())
    next(split for split in values["splits"] if split["id"] == "final")[
        "allows_model_selection"
    ] = True
    path = tmp_path / "definition.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="final split cannot"):
        BenchmarkDefinition.load(path)


def test_scenario_manifest_binds_definition_and_factor_contract(tmp_path: Path) -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    factors = {factor: {"bin": "development-0"} for factor in definition.factor_ids}
    values = {
        "format_version": 1,
        "benchmark_id": "embodi-sim-v1",
        "definition_sha256": definition.sha256,
        "scenarios": [
            {
                "scenario_id": "development-so101-push-000",
                "split": "development",
                "morphology": "so101",
                "task": "push_to_zone",
                "seed": 1,
                "factors": factors,
            }
        ],
    }
    path = tmp_path / "development.json"
    path.write_text(json.dumps(values))
    manifest = ScenarioManifest.load(path, definition)
    assert len(manifest.scenarios) == 1

    wrong_hash = deepcopy(values)
    wrong_hash["definition_sha256"] = "0" * 64
    path.write_text(json.dumps(wrong_hash))
    with pytest.raises(ValueError, match="definition_sha256"):
        ScenarioManifest.load(path, definition)


def test_scenario_manifest_rejects_held_out_task_in_training(tmp_path: Path) -> None:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    values = {
        "format_version": 1,
        "benchmark_id": "embodi-sim-v1",
        "definition_sha256": definition.sha256,
        "scenarios": [
            {
                "scenario_id": "train-so101-stack-000",
                "split": "train",
                "morphology": "so101",
                "task": "stack_object",
                "seed": 1,
                "factors": {factor: {"bin": "train-0"} for factor in definition.factor_ids},
            }
        ],
    }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="not admitted"):
        ScenarioManifest.load(path, definition)
