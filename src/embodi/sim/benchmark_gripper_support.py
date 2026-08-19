from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv, prepare_benchmark_robot_asset
from .benchmark_expert import PrivilegedBenchmarkExpert
from .benchmark_geometry_evaluation import load_development_contract
from .morphology import morphology_by_id
from . import benchmark_gripper_localization as exp44
from . import benchmark_gripper_mechanism as exp43
from . import benchmark_phase_agreement as exp42


DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
DEVELOPMENT_MANIFEST_SHA256 = (
    "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
)
EXP42_EVALUATOR_SHA256 = "cc4fb718a3f6341dbaaea0026aca193fd86d5b84412ea7061204ed8f4ce24b7d"
EXP43_EVALUATOR_SHA256 = "a1056895ab4926aa1d5822ff6ce1bab6d743fc779b3fb1599da26040356e9794"
EXP44_EVALUATOR_SHA256 = "7d40ef70485090ae595a497b4687883baaee26e9349c1899d864416bd18d8c70"
EXPERT_SHA256 = "7fb869a10756864530889f809a8305ee1758bdc51eaeeeda0ff2f451f3b7f8e9"
ASSETS_SHA256 = "cd1516bbb12a9a4fa13e94b24982683bf0d624b056dcd998ceaa24d63488d81d"
ENVIRONMENT_SHA256 = "e3039b41b3e3cb1856688522f6219e8a6b5b797a73ef0bbed985cfc6b646ef52"
MORPHOLOGY_SHA256 = "b8344f1ef1fc5294dd6a61017761973c6294b7a85477ce391fc3142ad043b632"
BASE_ENVIRONMENT_SHA256 = "66efff951ee618edd389b4a93c0f6a4ed59fbe5775e162f1940735efc4d622d5"
TASKS_SHA256 = "1101bf431f371b500e16bd735732636a43ff5a4aaa41a75eaf90cb719dd7139c"

MORPHOLOGIES = exp43.MORPHOLOGIES
TASKS = exp43.TASKS
PHASES = exp43.PHASES
ENDPOINTS = exp43.ENDPOINTS
SCENARIO_INDICES = tuple(range(42, 49))
PARTITIONS = exp44.PARTITIONS
EXPECTED_EPISODES = 42
EXPECTED_PHASE_UNITS = 182
EXPECTED_ENDPOINTS = 364
SUMMARY_PATH = Path("reports/benchmark-exp45-support.json")
LOCK_PATH = Path("reports/exp45-runtime/.support.lock")
REGISTRATION_PATH = Path("experiments/045-fresh-slice-gripper-localization.md")


def registered_evaluator_sha256(path: Path = REGISTRATION_PATH) -> str:
    match = re.search(
        r"Experiment 045 support evaluator SHA-256 `([0-9a-f]{64})`", path.read_text()
    )
    if match is None:
        raise ValueError("Experiment 045 registration does not freeze the support evaluator")
    return match.group(1)


def _dependency_contract() -> None:
    expected = {
        Path(exp42.__file__): EXP42_EVALUATOR_SHA256,
        Path(exp43.__file__): EXP43_EVALUATOR_SHA256,
        Path(exp44.__file__): EXP44_EVALUATOR_SHA256,
        Path(exp42.__file__).with_name("benchmark_expert.py"): EXPERT_SHA256,
        Path(exp42.__file__).with_name("assets.py"): ASSETS_SHA256,
        Path(exp42.__file__).with_name("benchmark_environment.py"): ENVIRONMENT_SHA256,
        Path(exp42.__file__).with_name("morphology.py"): MORPHOLOGY_SHA256,
        Path(exp42.__file__).with_name("environment.py"): BASE_ENVIRONMENT_SHA256,
        Path(exp42.__file__).with_name("tasks.py"): TASKS_SHA256,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("Experiment 045 imported support dependency hash mismatch")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _asset_inventory(morphology_id: str) -> list[dict[str, Any]]:
    morphology = morphology_by_id(morphology_id)
    root, robot_filename = prepare_benchmark_robot_asset(morphology)
    paths = [root / morphology.model_filename, root / robot_filename, root / "LICENSE"]
    paths.extend(path for path in sorted((root / "assets").rglob("*")) if path.is_file())
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]


def _runtime_provenance(evaluator_sha256: str) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    runtime = {
        "device": "cpu-expert-only",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mujoco": version("mujoco"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "runtime": runtime,
        "runtime_signature_sha256": hashlib.sha256(
            json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "evaluator_sha256": evaluator_sha256,
        "imported_exp42_evaluator_sha256": EXP42_EVALUATOR_SHA256,
        "imported_exp43_evaluator_sha256": EXP43_EVALUATOR_SHA256,
        "imported_exp44_evaluator_sha256": EXP44_EVALUATOR_SHA256,
        "expert_sha256": EXPERT_SHA256,
    }


def support_scenarios(manifest: ScenarioManifest, morphology: str) -> tuple[Scenario, ...]:
    if morphology not in MORPHOLOGIES:
        raise ValueError("Experiment 045 morphology is invalid")
    selected = tuple(
        scenario
        for scenario in manifest.scenarios
        if scenario.morphology == morphology
        and scenario.task in TASKS
        and int(scenario.scenario_id.rsplit("-", 1)[-1]) in SCENARIO_INDICES
    )
    expected = [
        f"development-{morphology}-{task}-{index:04d}"
        for task in TASKS
        for index in SCENARIO_INDICES
    ]
    if [scenario.scenario_id for scenario in selected] != expected:
        raise ValueError("Experiment 045 requires exact development indices 0042 through 0048")
    for task in TASKS:
        axes = [
            name
            for scenario in selected
            if scenario.task == task
            for name, value in scenario.factors.items()
            if str(value.get("bin", "")).startswith("development:")
        ]
        if len(axes) != 7 or len(set(axes)) != 7:
            raise ValueError("each Experiment 045 cell must cover all seven factor axes")
    return selected


def _array(record: Mapping[str, Any], name: str, dtype: Any, shape: tuple[int, ...]) -> NDArray[Any]:
    value = np.asarray(record.get(name), dtype=dtype)
    if (
        value.shape != shape
        or not np.isfinite(value).all()
        or record.get(f"{name}_sha256") != exp42.canonical_array_sha256(value)
    ):
        raise ValueError(f"Experiment 045 support {name} array is invalid")
    return value


def _support_record(anchor: exp42.Anchor) -> dict[str, Any]:
    record = dict(anchor.evidence)
    record["partition"] = exp44.partition_label(record, anchor.morphology)
    return record


def _validate_records(
    records: Sequence[dict[str, Any]], scenarios: Sequence[Scenario], morphology: str
) -> None:
    expected_ids = [
        f"{scenario.scenario_id}/{phase}/{endpoint}"
        for scenario in scenarios
        for phase in PHASES[scenario.task]
        for endpoint in ENDPOINTS
    ]
    actual_ids = [
        f"{record.get('scenario_id')}/{record.get('phase')}/{record.get('endpoint')}"
        for record in records
    ]
    if actual_ids != expected_ids or [record.get("ordinal") for record in records] != list(
        range(len(records))
    ):
        raise ValueError("Experiment 045 support endpoint ordering is invalid")
    native_shape = (6,) if morphology == "so101" else (8,)
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    for record in records:
        if (
            record.get("morphology") != morphology
            or record.get("scenario_id") not in scenarios_by_id
            or record.get("scenario_seed") != scenarios_by_id[record["scenario_id"]].seed
            or record.get("task") != scenarios_by_id[record["scenario_id"]].task
            or record.get("phase") not in PHASES[record["task"]]
            or record.get("endpoint") not in ENDPOINTS
            or record.get("partition") not in PARTITIONS
            or type(record.get("phase_tick")) is not int
            or record["phase_tick"] < 1
            or record.get("pre_call_phase") != record.get("phase")
            or record.get("pre_call_phase_tick") != record.get("phase_tick")
            or type(record.get("pre_call_total_tick")) is not int
            or record["pre_call_total_tick"] < 0
            or type(record.get("post_action_phase_tick")) is not int
            or type(record.get("post_action_total_tick")) is not int
            or record["post_action_total_tick"] != record["pre_call_total_tick"] + 1
            or record.get("post_action_state_sha256") != record.get("state_sha256")
            or record.get("expert_call_index") != record.get("pre_call_total_tick")
            or not _valid_sha256(record.get("image_sha256"))
            or not _valid_sha256(record.get("state_sha256"))
        ):
            raise ValueError("Experiment 045 support endpoint identity is invalid")
        current = _array(record, "pre_call_native_state", np.float32, native_shape)
        requested = _array(record, "expert_native", np.float32, native_shape)
        target = _array(record, "expert_target", np.float32, (1, 10))
        decoded = _array(record, "decoded_native", np.float32, native_shape)
        realizable = _array(record, "realizable_native", np.float64, native_shape)
        realized = _array(record, "realized_canonical", np.float32, (1, 10))
        clipped, clipping_error = exp42._clipping_from_arrays(decoded, realizable)
        rotation, degenerate = exp42.rotation_geodesic_error(target[0, 3:9], realized[0, 3:9])
        round_trip = record.get("round_trip", {})
        if (
            degenerate
            or round_trip.get("clipped") is not clipped
            or round_trip.get("clipping_error_native") != clipping_error
            or round_trip.get("translation_error_m")
            != float(np.linalg.norm(target[0, :3].astype(np.float64) - realized[0, :3]))
            or round_trip.get("rotation_error_deg") != rotation
            or round_trip.get("gripper_error_native")
            != float(abs(requested[-1].astype(np.float64) - decoded[-1]))
        ):
            raise ValueError("Experiment 045 support round-trip evidence is invalid")
        phase_order = list(PHASES[record["task"]])
        phase_index = phase_order.index(record["phase"])
        expected_post = (
            phase_order[phase_index + 1] if phase_index + 1 < len(phase_order) else "done"
        )
        transitioned = record.get("post_action_phase") != record["phase"]
        if record["endpoint"] == "last" and (
            not transitioned
            or record.get("post_action_phase") != expected_post
            or record.get("post_action_phase_tick") != 1
        ):
            raise ValueError("Experiment 045 last endpoint transition is invalid")
        if record["endpoint"] == "first" and transitioned and (
            record.get("post_action_phase") != expected_post
            or record.get("post_action_phase_tick") != 1
        ):
            raise ValueError("Experiment 045 first endpoint transition is invalid")
        if record["endpoint"] == "first" and not transitioned and (
            record.get("post_action_phase_tick") != record["phase_tick"] + 1
        ):
            raise ValueError("Experiment 045 nontransition phase tick is invalid")
        if record.get("post_action_phase") == "done" and not np.array_equal(
            requested, current
        ):
            raise ValueError("Experiment 045 terminal target is not a native no-op")
        if exp44.partition_label(record, morphology) != record["partition"]:
            raise ValueError("Experiment 045 support partition does not recompute")
    exp42._validate_endpoint_pairs(records)


def _support_counts(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    partition_counts = Counter(str(record["partition"]) for record in records)
    phase_units = {
        (record["morphology"], record["scenario_id"], record["phase"])
        for record in records
    }
    return {
        "endpoint_anchors": len(records),
        "scenario_phase_units": len(phase_units),
        "by_partition_endpoint_anchors": {
            partition: partition_counts.get(partition, 0) for partition in PARTITIONS
        },
        "by_partition_scenario_phase_units": {
            partition: len(
                {
                    (record["morphology"], record["scenario_id"], record["phase"])
                    for record in records
                    if record["partition"] == partition
                }
            )
            for partition in PARTITIONS
        },
        "by_morphology_task_partition": {
            f"{morphology}/{task}/{partition}": sum(
                record["morphology"] == morphology
                and record["task"] == task
                and record["partition"] == partition
                for record in records
            )
            for morphology in MORPHOLOGIES
            for task in TASKS
            for partition in PARTITIONS
        },
    }


def validate_support_report(report: Mapping[str, Any], manifest: ScenarioManifest) -> None:
    _dependency_contract()
    records = report.get("records", [])
    episodes = report.get("episodes", [])
    contract = report.get("contract", {})
    provenance = report.get("provenance", {})
    if (
        report.get("format_version") != 1
        or report.get("experiment") != 45
        or report.get("mode") != "expert_only_support_freeze"
        or report.get("registered") is not True
        or report.get("status") != "completed_support_freeze"
        or report.get("complete") is not True
        or contract.get("definition_sha256") != DEFINITION_SHA256
        or contract.get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
        or contract.get("scenario_indices") != list(SCENARIO_INDICES)
        or contract.get("policy_inferences_run") != 0
        or contract.get("checkpoint_files_opened") != 0
        or contract.get("training_runs") != 0
        or contract.get("final_split_loaded") is not False
        or report.get("policy_outcomes_available") is not False
        or report.get("final_split_loaded") is not False
        or provenance.get("git_dirty") is not False
        or not isinstance(provenance.get("git_commit"), str)
        or len(provenance["git_commit"]) != 40
        or bool(set(provenance["git_commit"]) - set("0123456789abcdef"))
        or provenance.get("evaluator_sha256") != file_sha256(Path(__file__))
        or provenance.get("imported_exp42_evaluator_sha256") != EXP42_EVALUATOR_SHA256
        or provenance.get("imported_exp43_evaluator_sha256") != EXP43_EVALUATOR_SHA256
        or provenance.get("imported_exp44_evaluator_sha256") != EXP44_EVALUATOR_SHA256
        or provenance.get("expert_sha256") != EXPERT_SHA256
        or not isinstance(provenance.get("runtime"), dict)
        or provenance.get("runtime_signature_sha256")
        != hashlib.sha256(
            json.dumps(
                provenance.get("runtime"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        or len(records) != EXPECTED_ENDPOINTS
        or len(episodes) != EXPECTED_EPISODES
        or any(
            episode.get("success") is not True
            or episode.get("terminal") is not True
            or episode.get("final_phase") != "done"
            for episode in episodes
        )
    ):
        raise ValueError("Experiment 045 support report contract is invalid")
    expected_scenarios = [
        scenario
        for morphology in MORPHOLOGIES
        for scenario in support_scenarios(manifest, morphology)
    ]
    expected_ids = [scenario.scenario_id for scenario in expected_scenarios]
    if [episode.get("scenario_id") for episode in episodes] != expected_ids:
        raise ValueError("Experiment 045 support episode order is invalid")
    for episode, scenario in zip(episodes, expected_scenarios, strict=True):
        matching = [record for record in records if record.get("scenario_id") == scenario.scenario_id]
        ticks = {
            phase: [record["phase_tick"] for record in matching if record["phase"] == phase]
            for phase in PHASES[scenario.task]
        }
        if (
            episode.get("task") != scenario.task
            or episode.get("observed_phases") != ["open", *PHASES[scenario.task], "done"]
            or episode.get("scored_phases") != list(PHASES[scenario.task])
            or episode.get("endpoint_count") != len(PHASES[scenario.task]) * 2
            or episode.get("phase_endpoint_ticks") != ticks
        ):
            raise ValueError("Experiment 045 support episode evidence is invalid")
    if [record.get("scenario_id") for record in records] != [
        scenario.scenario_id
        for scenario in expected_scenarios
        for _phase in PHASES[scenario.task]
        for _endpoint in ENDPOINTS
    ]:
        raise ValueError("Experiment 045 global support record order is invalid")
    for morphology in MORPHOLOGIES:
        morphology_records = [record for record in records if record.get("morphology") == morphology]
        _validate_records(morphology_records, support_scenarios(manifest, morphology), morphology)
    gate = exp42.delivery_gate(records)
    expected_support = _support_counts(records)
    if (
        gate["passed"] is not True
        or report.get("support") != expected_support
        or report.get("asset_inventory")
        != {
            morphology: _asset_inventory(morphology)
            for morphology in MORPHOLOGIES
        }
        or expected_support["scenario_phase_units"] != EXPECTED_PHASE_UNITS
        or report.get("outcome_chain_sha256") != exp42._outcome_chain(list(records))
    ):
        raise ValueError("Experiment 045 support evidence does not recompute")


@contextmanager
def support_lock(path: Path = LOCK_PATH) -> Iterator[None]:
    if path != LOCK_PATH:
        raise ValueError("Experiment 045 support lock path is frozen")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Experiment 045 support capture holds the lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def capture_support(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    *,
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
    expert_factory: Callable[[Any], Any] = PrivilegedBenchmarkExpert,
) -> dict[str, Any]:
    _dependency_contract()
    evaluator_sha256 = file_sha256(Path(__file__))
    if (
        evaluator_sha256 != registered_evaluator_sha256()
        or definition.sha256 != DEFINITION_SHA256
        or manifest.definition_sha256 != DEFINITION_SHA256
        or file_sha256(manifest.path) != DEVELOPMENT_MANIFEST_SHA256
    ):
        raise ValueError("Experiment 045 support registration identity is invalid")
    provenance = _runtime_provenance(evaluator_sha256)
    if provenance.get("git_dirty") is not False:
        raise RuntimeError("Experiment 045 support capture requires a clean worktree")
    records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for morphology in MORPHOLOGIES:
        anchors, morphology_episodes = exp42.capture_expert_anchors(
            support_scenarios(manifest, morphology),
            morphology,
            env_factory=env_factory,
            expert_factory=expert_factory,
        )
        records.extend(_support_record(anchor) for anchor in anchors)
        episodes.extend(morphology_episodes)
    report = {
        "format_version": 1,
        "experiment": 45,
        "mode": "expert_only_support_freeze",
        "registered": True,
        "status": "completed_support_freeze",
        "complete": True,
        "contract": {
            "definition_sha256": DEFINITION_SHA256,
            "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "scenario_indices": list(SCENARIO_INDICES),
            "tasks": list(TASKS),
            "morphologies": list(MORPHOLOGIES),
            "policy_inferences_run": 0,
            "checkpoint_files_opened": 0,
            "training_runs": 0,
            "final_split_loaded": False,
        },
        "provenance": provenance,
        "asset_inventory": {
            morphology: _asset_inventory(morphology)
            for morphology in MORPHOLOGIES
        },
        "episodes": episodes,
        "records": records,
        "support": _support_counts(records),
        "outcome_chain_sha256": exp42._outcome_chain(records),
        "policy_outcomes_available": False,
        "final_split_loaded": False,
    }
    validate_support_report(report, manifest)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="freeze Experiment 045 expert-only support")
    parser.add_argument(
        "--definition", type=Path, default=Path("benchmarks/sim-v1/definition.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/sim-v1/manifests/development.json"),
    )
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output != SUMMARY_PATH:
        raise ValueError(f"registered Experiment 045 support output must be {SUMMARY_PATH}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable support report: {args.output}")
    definition, manifest = load_development_contract(
        args.definition, args.manifest, registered=True
    )
    with support_lock():
        report = capture_support(definition, manifest)
        exp42._write_final(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
