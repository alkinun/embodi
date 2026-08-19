import json
from pathlib import Path

import pytest

from embodi.sim.benchmark import BenchmarkDefinition, file_sha256, ScenarioManifest
import embodi.sim.benchmark_gripper_support as support


ROOT = Path(__file__).parents[1]


def _manifest() -> ScenarioManifest:
    definition = BenchmarkDefinition.load(ROOT / "benchmarks/sim-v1/definition.json")
    return ScenarioManifest.load(
        ROOT / "benchmarks/sim-v1/manifests/development.json",
        definition,
        require_complete_counts=True,
    )


@pytest.mark.parametrize("morphology", support.MORPHOLOGIES)
def test_support_scenarios_freeze_final_complete_factor_cycle(morphology: str) -> None:
    scenarios = support.support_scenarios(_manifest(), morphology)
    assert len(scenarios) == 21
    assert [scenario.scenario_id for scenario in scenarios] == [
        f"development-{morphology}-{task}-{index:04d}"
        for task in support.TASKS
        for index in support.SCENARIO_INDICES
    ]


def test_support_scenarios_reject_nonregistered_morphology() -> None:
    with pytest.raises(ValueError, match="morphology"):
        support.support_scenarios(_manifest(), "other")


def test_support_counts_do_not_assume_partition_marginals() -> None:
    records = [
        {
            "morphology": "so101",
            "task": "push_to_zone",
            "scenario_id": "scenario-0",
            "phase": "push",
            "partition": "open_steady",
        },
        {
            "morphology": "so101",
            "task": "push_to_zone",
            "scenario_id": "scenario-0",
            "phase": "push",
            "partition": "terminal_noop",
        },
    ]
    counts = support._support_counts(records)
    assert counts["endpoint_anchors"] == 2
    assert counts["scenario_phase_units"] == 1
    assert counts["by_partition_endpoint_anchors"]["open_steady"] == 1
    assert counts["by_partition_endpoint_anchors"]["terminal_noop"] == 1
    assert counts["by_partition_scenario_phase_units"]["open_steady"] == 1
    assert counts["by_partition_scenario_phase_units"]["terminal_noop"] == 1


def test_dependency_registration_and_cli_contract() -> None:
    support._dependency_contract()
    assert support.registered_evaluator_sha256() == file_sha256(
        Path(support.__file__)
    )
    args = support.parse_args([])
    assert args.output == support.SUMMARY_PATH
    assert args.manifest.name == "development.json"


def test_registered_support_report_hash_and_counts() -> None:
    path = ROOT / "reports/benchmark-exp45-support.json"
    assert file_sha256(path) == (
        "34c3252feab6b7153a17cf14baa8708fccfbb4551ea6fc71d1dc65b73b85cb58"
    )
    report = json.loads(path.read_text())
    assert report["contract"]["checkpoint_files_opened"] == 0
    assert report["contract"]["policy_inferences_run"] == 0
    assert report["support"]["endpoint_anchors"] == 364
    assert report["support"]["scenario_phase_units"] == 182
    assert report["support"]["by_partition_endpoint_anchors"] == {
        "open_steady": 154,
        "open_openward": 28,
        "closed_steady": 70,
        "closed_closeward": 56,
        "terminal_noop": 56,
    }
