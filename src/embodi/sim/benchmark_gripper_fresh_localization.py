from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import fcntl
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid
import zlib

import numpy as np
import torch

from ..processing import EmbodiProcessor
from . import benchmark_gripper_localization as exp44
from . import benchmark_gripper_mechanism as exp43
from . import benchmark_gripper_support as support
from . import benchmark_phase_agreement as exp42
from .benchmark import BenchmarkDefinition, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv
from .benchmark_expert import PrivilegedBenchmarkExpert
from .benchmark_geometry_evaluation import load_development_contract, load_geometry_policy


SUPPORT_REPORT_SHA256 = "34c3252feab6b7153a17cf14baa8708fccfbb4551ea6fc71d1dc65b73b85cb58"
SUPPORT_EVALUATOR_SHA256 = "4cca61781c45eaa2f3b4778ffe48b87c81630166d2b6d9e2e0b2e0eeda45cb14"
EXP42_EVALUATOR_SHA256 = support.EXP42_EVALUATOR_SHA256
EXP43_EVALUATOR_SHA256 = support.EXP43_EVALUATOR_SHA256
EXP44_EVALUATOR_SHA256 = support.EXP44_EVALUATOR_SHA256
EXP41_EVALUATOR_SHA256 = exp42.EXP41_EVALUATOR_SHA256
EXPERT_SHA256 = support.EXPERT_SHA256
ASSETS_SHA256 = support.ASSETS_SHA256
ENVIRONMENT_SHA256 = support.ENVIRONMENT_SHA256
MORPHOLOGY_SHA256 = support.MORPHOLOGY_SHA256
BASE_ENVIRONMENT_SHA256 = support.BASE_ENVIRONMENT_SHA256
TASKS_SHA256 = support.TASKS_SHA256

DEFINITION_SHA256 = support.DEFINITION_SHA256
DEVELOPMENT_MANIFEST_SHA256 = support.DEVELOPMENT_MANIFEST_SHA256
MORPHOLOGIES = exp43.MORPHOLOGIES
TASKS = exp43.TASKS
PHASES = exp43.PHASES
ENDPOINTS = exp43.ENDPOINTS
SEEDS = exp43.SEEDS
CONDITIONS = exp43.CONDITIONS
CONTRASTS = exp43.CONTRASTS
PARTITIONS = support.PARTITIONS
SCENARIO_INDICES = support.SCENARIO_INDICES

EXPECTED_EPISODES = exp43.EXPECTED_EPISODES
EXPECTED_PHASE_UNITS = exp43.EXPECTED_PHASE_UNITS
EXPECTED_ENDPOINTS = exp43.EXPECTED_ENDPOINTS
EXPECTED_SCORED_CHUNKS = exp43.EXPECTED_SCORED_CHUNKS
EXPECTED_REPEAT_CHUNKS = exp43.EXPECTED_REPEAT_CHUNKS
EXPECTED_INFERENCES = exp43.EXPECTED_INFERENCES
EXPECTED_CONTRAST_OBSERVATIONS = exp43.EXPECTED_CONTRAST_OBSERVATIONS

REGISTERED_RUNTIME_DIR = Path("reports/exp45-runtime")
REGISTERED_LOCK_PATH = REGISTERED_RUNTIME_DIR / ".registered.lock"
LOCK_IDENTITY = "exp45-support-bound-fresh-gripper-v1"
SCHEDULE_IDENTITY = LOCK_IDENTITY
SUPPORT_PATH = Path("reports/benchmark-exp45-support.json")
SUMMARY_PATH = Path("reports/benchmark-exp45-summary.json")
REGISTRATION_PATH = Path("experiments/045-fresh-slice-gripper-localization.md")

SOURCE_FILE_SHA256 = {
    "src/embodi/__init__.py": "2821fbed5245a41c071e3c2e46753e994ca1fa1d8ec1fed76b12e3e8f972d961",
    "src/embodi/action_decoder.py": "6fc20a4de484a5d562fe4ab683e2904c9e5982ca870ddf28039e5b5bc515587e",
    "src/embodi/backbone.py": "466f38f01977dffdddab84a7281ce548cbb73e270f7f9f5d865260b40d3551bf",
    "src/embodi/canonical.py": "6852f39df91501a9bc9d1949b504739994138e86b1416b970274a5a26f929c2b",
    "src/embodi/checkpoints.py": "9df04304b1f482ce81614d2402eb1f0c075798120ab40c962d45708bc5ab0098",
    "src/embodi/config.py": "444cd4af5b7245afc166945708672437b3450565576dc706d3a83b2a9cdeab72",
    "src/embodi/joint_train.py": "e6dbf985ea4b79c7cdf5d2bc0bf6c5fa17186f82f8e8297d5454fbb501cba9a3",
    "src/embodi/processing.py": "9d089a8f715e3e03dec9570fd6a452d9dff15ea38db999d8f72e9ce5a2cb03dc",
    "src/embodi/record_so101.py": "60583054a3be5b0c767ccd73da6fd36ba46d0b29fd2746a8ab6572d5b5397aa1",
    "src/embodi/token_canonical.py": "2f943e87f1eb5bc35f0ba0ca0be519c5bf2ad467970b15bc1476901ead20e885",
    "src/embodi/token_config.py": "9bbc860995c841984f5ff203b35b002aa10714d9761871e295451aeaf920b0cf",
    "src/embodi/token_decoder.py": "cb8016882448be6670c317155b8de29f834e3c4f3592e969ee10f2cffac3ddac",
    "src/embodi/token_expert.py": "5526789f8ac06856c96a5448f5d117fd3dda930e0aa4af4ba3b15c251b782891",
    "src/embodi/token_policy.py": "0606ac1a74e2a2e0bb849c1892ce82f6f6862cdb68c23f4b87f6412d707fb8ea",
    "src/embodi/token_state_adapter.py": "d63ad6dcf6b8f7c5324aa5a3a3fc106f0759ffe59b95a05da88391639282291b",
    "src/embodi/train.py": "148c8cf73dfb6add873f477fdfab9496c8f96532eb4b94db3dda46d0ff2a150b",
    "src/embodi/xperience.py": "62efaf7168ca1daf372a02e680f5cca3997bc1c27ea8ebf1272d33071e13caf4",
    "src/embodi/sim/__init__.py": "dc00b102bc4925f1e063218322af093fe7833cd8192094ff2975af43efcf2456",
    "src/embodi/sim/assets.py": ASSETS_SHA256,
    "src/embodi/sim/benchmark.py": "5a72dd92046833eed7d6e820a6eb26b932a09dbd44d8a9f79a1ab18698b5b5ef",
    "src/embodi/sim/benchmark_environment.py": ENVIRONMENT_SHA256,
    "src/embodi/sim/benchmark_expert.py": EXPERT_SHA256,
    "src/embodi/sim/benchmark_geometry_diagnostics.py": "773dd4c7d136b75b2194f7c04ea7e4b1a56c898357740fb1fc03b22c20fef870",
    "src/embodi/sim/benchmark_geometry_evaluation.py": "3e8754887ba54034e2e3e2fb7db0eadc95302a33ca59d2fde3cf37133f3f2253",
    "src/embodi/sim/benchmark_geometry_projection.py": "7617983a71166a9fa261ecb894e2ee6b02bde6d044c8a2b3edd27d17736c791c",
    "src/embodi/sim/benchmark_gripper_localization.py": EXP44_EVALUATOR_SHA256,
    "src/embodi/sim/benchmark_gripper_mechanism.py": EXP43_EVALUATOR_SHA256,
    "src/embodi/sim/benchmark_gripper_support.py": SUPPORT_EVALUATOR_SHA256,
    "src/embodi/sim/benchmark_phase_agreement.py": EXP42_EVALUATOR_SHA256,
    "src/embodi/sim/benchmark_replan_alignment.py": "d9c73baff704b5c0e56fd46127f3be07488a1401fa7add285a096478d893f3dd",
    "src/embodi/sim/benchmark_task_instruction.py": EXP41_EVALUATOR_SHA256,
    "src/embodi/sim/environment.py": BASE_ENVIRONMENT_SHA256,
    "src/embodi/sim/morphology.py": MORPHOLOGY_SHA256,
    "src/embodi/sim/tasks.py": TASKS_SHA256,
}
SOURCE_ROOT = Path(__file__).resolve().parents[3]
PROVENANCE_FIELDS = {
    "git_commit",
    "git_dirty",
    "python",
    "torch",
    "cuda",
    "cudnn",
    "lerobot",
    "transformers",
    "trainer_sha256",
    "runtime",
    "runtime_signature_sha256",
    "evaluator_sha256",
    "imported_exp41_evaluator_sha256",
    "imported_exp42_evaluator_sha256",
    "imported_exp43_evaluator_sha256",
    "imported_exp44_evaluator_sha256",
    "imported_support_evaluator_sha256",
    "expert_sha256",
    "support_report_sha256",
    "source_file_sha256",
}
RUN_FIELDS = {
    "started_utc",
    "finished_utc",
    "device",
    "lock_identity",
    "lock_path",
    "schedule_identity",
    "schedule_position",
    "orchestrator_run_id",
    "seed_order",
    "condition_order_formula",
}


def registered_evaluator_sha256(path: Path = REGISTRATION_PATH) -> str:
    match = re.search(
        r"Experiment 045 policy evaluator SHA-256 `([0-9a-f]{64})`", path.read_text()
    )
    if match is None:
        raise ValueError("Experiment 045 registration does not freeze the policy evaluator")
    return match.group(1)


def _dependency_contract() -> None:
    _validate_source_file_contract()


def _validate_source_file_contract() -> None:
    if any(
        not (SOURCE_ROOT / relative).is_file()
        or file_sha256(SOURCE_ROOT / relative) != digest
        for relative, digest in SOURCE_FILE_SHA256.items()
    ):
        raise ValueError("Experiment 045 outcome-affecting source closure hash mismatch")


def load_support_report(path: Path, manifest: ScenarioManifest) -> dict[str, Any]:
    if path != SUPPORT_PATH or not path.is_file() or file_sha256(path) != SUPPORT_REPORT_SHA256:
        raise ValueError("Experiment 045 support report path or SHA-256 mismatch")
    report = json.loads(path.read_bytes())
    support.validate_support_report(report, manifest)
    if report.get("provenance", {}).get("evaluator_sha256") != SUPPORT_EVALUATOR_SHA256:
        raise ValueError("Experiment 045 support evaluator identity mismatch")
    return report


def _fresh_record(anchor: exp42.Anchor) -> dict[str, Any]:
    value = dict(anchor.evidence)
    value["partition"] = exp44.partition_label(value, anchor.morphology)
    return value


def _structural_record(record: Mapping[str, Any]) -> dict[str, Any]:
    # The fresh rendered frame may differ from the support run. Nothing else may.
    return {key: value for key, value in record.items() if key != "image_sha256"}


def _without_policy_results(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in record.items() if key != "seeds"} for record in records]


def validate_recaptured_support(
    records: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    support_report: Mapping[str, Any],
    manifest: ScenarioManifest,
    morphology: str,
) -> None:
    scenarios = support.support_scenarios(manifest, morphology)
    support._validate_records(records, scenarios, morphology)
    expected_records = [
        record
        for record in support_report.get("records", [])
        if record.get("morphology") == morphology
    ]
    expected_episodes = [
        episode
        for episode in support_report.get("episodes", [])
        if str(episode.get("scenario_id", "")).startswith(f"development-{morphology}-")
    ]
    if (
        len(records) != EXPECTED_ENDPOINTS // 2
        or len(episodes) != EXPECTED_EPISODES // 2
        or len(expected_records) != len(records)
        or len(expected_episodes) != len(episodes)
        or list(episodes) != expected_episodes
        or any(
            _structural_record(actual) != _structural_record(expected)
            for actual, expected in zip(records, expected_records, strict=True)
        )
    ):
        raise ValueError("Experiment 045 recaptured structural support mismatch")


def recapture_support(
    manifest: ScenarioManifest,
    support_report: Mapping[str, Any],
    morphology: str,
    *,
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
    expert_factory: Callable[[Any], Any] = PrivilegedBenchmarkExpert,
) -> tuple[list[exp42.Anchor], list[dict[str, Any]]]:
    anchors, episodes = exp42.capture_expert_anchors(
        support.support_scenarios(manifest, morphology),
        morphology,
        env_factory=env_factory,
        expert_factory=expert_factory,
    )
    validate_recaptured_support(
        [_fresh_record(anchor) for anchor in anchors],
        episodes,
        support_report,
        manifest,
        morphology,
    )
    return anchors, episodes


@dataclass(frozen=True)
class RegisteredSupportCapture:
    report: dict[str, Any]
    anchors: dict[str, tuple[exp42.Anchor, ...]]
    episodes: dict[str, tuple[dict[str, Any], ...]]


def preflight_registered_support(
    manifest: ScenarioManifest,
    support_path: Path,
    *,
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
    expert_factory: Callable[[Any], Any] = PrivilegedBenchmarkExpert,
) -> RegisteredSupportCapture:
    report = load_support_report(support_path, manifest)
    anchors: dict[str, tuple[exp42.Anchor, ...]] = {}
    episodes: dict[str, tuple[dict[str, Any], ...]] = {}
    for morphology in MORPHOLOGIES:
        morphology_anchors, morphology_episodes = recapture_support(
            manifest,
            report,
            morphology,
            env_factory=env_factory,
            expert_factory=expert_factory,
        )
        anchors[morphology] = tuple(morphology_anchors)
        episodes[morphology] = tuple(morphology_episodes)
    return RegisteredSupportCapture(report=report, anchors=anchors, episodes=episodes)


def _source_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    return exp43._source_contract(paths)


def resolve_checkpoint(
    source_paths: Mapping[str, Path],
    condition: str,
    seed: int,
    morphology: str,
    *,
    root: Path,
) -> tuple[Path, dict[str, Any]]:
    return exp43.resolve_experiment43_checkpoint(
        source_paths, condition, seed, morphology, root=root
    )


def condition_order(anchor_ordinal: int, seed: int, morphology: str) -> tuple[str, ...]:
    return exp43.condition_order(anchor_ordinal, seed, morphology)


evaluate_anchor_condition = exp43.evaluate_anchor_condition


def _checkpoint_identities(
    source_paths: Mapping[str, Path], morphology: str, root: Path
) -> tuple[dict[int, dict[str, Path]], dict[str, dict[str, dict[str, Any]]]]:
    paths: dict[int, dict[str, Path]] = {}
    identities: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        paths[seed] = {}
        identities[str(seed)] = {}
        for condition in CONDITIONS:
            path, identity = resolve_checkpoint(
                source_paths, condition, seed, morphology, root=root
            )
            paths[seed][condition] = path
            identities[str(seed)][condition] = identity
    return paths, identities


def _contract(registered: bool) -> dict[str, Any]:
    return {
        "definition_sha256": DEFINITION_SHA256,
        "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
        "support_report_sha256": SUPPORT_REPORT_SHA256,
        "support_evaluator_sha256": SUPPORT_EVALUATOR_SHA256,
        "exp41_evaluator_sha256": EXP41_EVALUATOR_SHA256,
        "exp42_evaluator_sha256": EXP42_EVALUATOR_SHA256,
        "exp43_evaluator_sha256": EXP43_EVALUATOR_SHA256,
        "exp44_evaluator_sha256": EXP44_EVALUATOR_SHA256,
        "expert_sha256": EXPERT_SHA256,
        "simulation_dependency_sha256": {
            "assets": ASSETS_SHA256,
            "benchmark_environment": ENVIRONMENT_SHA256,
            "morphology": MORPHOLOGY_SHA256,
            "environment": BASE_ENVIRONMENT_SHA256,
            "tasks": TASKS_SHA256,
        },
        "source_file_sha256": dict(SOURCE_FILE_SHA256),
        "exp36_summary_sha256": exp43.EXP36_SUMMARY_SHA256,
        "exp42_summary_sha256": exp43.EXP42_SUMMARY_SHA256,
        "exp43_training_summary_sha256": exp43.EXP43_TRAINING_SUMMARY_SHA256,
        "scenario_indices": list(SCENARIO_INDICES),
        "split": "development",
        "tasks": list(TASKS),
        "morphologies": list(MORPHOLOGIES),
        "conditions": list(CONDITIONS),
        "model_seeds": list(SEEDS),
        "registered": registered,
        "training_runs": 0,
        "final_split_loaded": False,
    }


def _runtime_provenance(device: str) -> dict[str, Any]:
    value = exp43._runtime_provenance(device)
    value.update(
        {
            "evaluator_sha256": file_sha256(Path(__file__)),
            "imported_exp41_evaluator_sha256": EXP41_EVALUATOR_SHA256,
            "imported_exp43_evaluator_sha256": EXP43_EVALUATOR_SHA256,
            "imported_exp44_evaluator_sha256": EXP44_EVALUATOR_SHA256,
            "imported_support_evaluator_sha256": SUPPORT_EVALUATOR_SHA256,
            "support_report_sha256": SUPPORT_REPORT_SHA256,
            "source_file_sha256": dict(SOURCE_FILE_SHA256),
        }
    )
    return value


def _expected_committed_source_sha256() -> dict[str, str]:
    return {
        "src/embodi/sim/benchmark_gripper_fresh_localization.py": registered_evaluator_sha256(),
        **SOURCE_FILE_SHA256,
    }


def _current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("Experiment 045 cannot resolve the current Git commit") from error


def _committed_source_sha256(commit: str) -> dict[str, str]:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
        )
        return {
            path: hashlib.sha256(
                subprocess.run(
                    ["git", "show", f"{commit}:{path}"],
                    check=True,
                    capture_output=True,
                ).stdout
            ).hexdigest()
            for path in _expected_committed_source_sha256()
        }
    except subprocess.CalledProcessError as error:
        raise ValueError("Experiment 045 committed source closure is missing") from error


def _validate_live_provenance(
    persisted: Mapping[str, Any],
    device: str,
    *,
    expected_current_provenance: Mapping[str, Any] | None = None,
    expected_commit: str | None = None,
    committed_source_sha256: Mapping[str, str] | None = None,
) -> None:
    if (expected_commit is None) != (committed_source_sha256 is None):
        raise ValueError("synthetic provenance injection requires commit and blob identities")
    current = (
        _runtime_provenance(device)
        if expected_current_provenance is None
        else dict(expected_current_provenance)
    )
    actual_commit = _current_git_commit() if expected_commit is None else expected_commit
    observed_sources = (
        _committed_source_sha256(actual_commit)
        if committed_source_sha256 is None
        else dict(committed_source_sha256)
    )
    claimed_commit = persisted.get("git_commit")
    if (
        persisted != current
        or current.get("git_dirty") is not False
        or claimed_commit != actual_commit
        or not isinstance(actual_commit, str)
        or len(actual_commit) != 40
        or bool(set(actual_commit) - set("0123456789abcdef"))
        or observed_sources != _expected_committed_source_sha256()
    ):
        raise ValueError("Experiment 045 live runtime, package, or committed source provenance differs")


def _flatten_condition_records(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "condition": condition,
            "morphology": anchor["morphology"],
            "task": anchor["task"],
            "scenario_id": anchor["scenario_id"],
            "phase": anchor["phase"],
            "endpoint": anchor["endpoint"],
            "partition": anchor["partition"],
            **anchor["seeds"][str(seed)][condition],
        }
        for report in reports
        for anchor in report["anchors"]
        for seed in SEEDS
        for condition in CONDITIONS
    ]


def aggregate_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = exp43.aggregate_metrics(records)
    result["by_phase"] = exp43._subgroups(records, ("phase",))
    for group in result["by_phase"].values():
        cells = [group[name] for name in exp43.METRICS]
        cells.extend(group["channels"].values())
        for cell in cells:
            exact = {
                name: Decimal(cell["contrasts"][name]["difference_decimal"])
                for name in CONTRASTS
            }
            if exact["J-F"] != exact["H-F"] + exact["J-H"]:
                raise ValueError("Experiment 045 phase contrast decomposition is inexact")
    return result


def _shard_aggregate(anchors: Sequence[dict[str, Any]], report: Mapping[str, Any]) -> dict[str, Any]:
    records = _flatten_condition_records([report])
    gate = exp42.delivery_gate(anchors)
    integrity = {
        "support_bound_before_checkpoint_load": report.get("support_preflight_passed") is True,
        "identical_fresh_image_across_nine_arms": all(
            query.get("image_sha256") == anchor.get("image_sha256")
            for anchor in anchors
            for seed in SEEDS
            for condition in CONDITIONS
            for query in anchor["seeds"][str(seed)][condition]["queries"]
        ),
        "deterministic_immediate_repeats": all(
            anchor["seeds"][str(seed)][condition]["repeat_max_abs_difference"]
            <= exp43.REPEAT_MAX_ABS
            for anchor in anchors
            for seed in SEEDS
            for condition in CONDITIONS
        ),
        "final_split_not_loaded": report.get("contract", {}).get("final_split_loaded") is False,
    }
    gate["integrity_checks"] = integrity
    gate["passed"] = gate["passed"] and all(integrity.values())
    count = len(anchors) * len(SEEDS) * len(CONDITIONS)
    return {
        "episodes": len(report["episodes"]),
        "phase_units": len(anchors) // 2,
        "endpoint_anchors": len(anchors),
        "scored_chunks": count,
        "repeat_chunks": count,
        "policy_inferences": count * 2,
        "contrast_observations": {
            contrast: len(anchors) * len(SEEDS) for contrast in CONTRASTS
        },
        "metrics": aggregate_metrics(records),
        "delivery_gate": gate,
    }


def _validate_checkpoints(
    report: Mapping[str, Any], source_paths: Mapping[str, Path], *, root: Path
) -> None:
    morphology = report["identity"]["morphology"]
    stored = report["identity"].get("checkpoints", {})
    if tuple(stored) != tuple(str(seed) for seed in SEEDS):
        raise ValueError("Experiment 045 checkpoint seed identities are incomplete")
    for seed in SEEDS:
        if set(stored[str(seed)]) != set(CONDITIONS):
            raise ValueError("Experiment 045 checkpoint conditions are incomplete")
        for condition in CONDITIONS:
            _, expected = resolve_checkpoint(
                source_paths, condition, seed, morphology, root=root
            )
            if stored[str(seed)][condition] != expected:
                raise ValueError("Experiment 045 checkpoint identity or files differ")


def _validate_persisted_identity(report: Mapping[str, Any], morphology: str) -> None:
    identity = report.get("identity", {})
    run = report.get("run", {})
    provenance = report.get("provenance", {})
    runtime = provenance.get("runtime")
    if (
        set(identity) != {"morphology", "model_seeds", "checkpoints"}
        or identity.get("morphology") != morphology
        or identity.get("model_seeds") != list(SEEDS)
        or set(run) != RUN_FIELDS
        or run.get("seed_order") != list(SEEDS)
        or run.get("condition_order_formula") != exp43.PROTOCOL["condition_order_formula"]
        or run.get("device") != (runtime.get("device") if isinstance(runtime, dict) else None)
        or set(provenance) != PROVENANCE_FIELDS
        or provenance.get("git_dirty") is not False
        or not isinstance(provenance.get("git_commit"), str)
        or len(provenance["git_commit"]) != 40
        or bool(set(provenance["git_commit"]) - set("0123456789abcdef"))
        or provenance.get("evaluator_sha256") != file_sha256(Path(__file__))
        or provenance.get("imported_exp41_evaluator_sha256") != EXP41_EVALUATOR_SHA256
        or provenance.get("imported_exp42_evaluator_sha256") != EXP42_EVALUATOR_SHA256
        or provenance.get("imported_exp43_evaluator_sha256") != EXP43_EVALUATOR_SHA256
        or provenance.get("imported_exp44_evaluator_sha256") != EXP44_EVALUATOR_SHA256
        or provenance.get("imported_support_evaluator_sha256") != SUPPORT_EVALUATOR_SHA256
        or provenance.get("expert_sha256") != EXPERT_SHA256
        or provenance.get("support_report_sha256") != SUPPORT_REPORT_SHA256
        or provenance.get("source_file_sha256") != SOURCE_FILE_SHA256
        or provenance.get("trainer_sha256") != SOURCE_FILE_SHA256["src/embodi/joint_train.py"]
        or not isinstance(runtime, dict)
        or set(runtime) != exp43.RUNTIME_FIELDS
        or provenance.get("python") != runtime.get("python")
        or provenance.get("torch") != runtime.get("torch")
        or provenance.get("cuda") != runtime.get("cuda_runtime")
        or provenance.get("cudnn") != runtime.get("cudnn")
        or provenance.get("runtime_signature_sha256")
        != hashlib.sha256(
            json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("Experiment 045 persisted identity or provenance is invalid")


def validate_shard_report(
    report: dict[str, Any],
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    support_report: Mapping[str, Any],
    *,
    artifact_root: Path = Path("."),
    expected_current_provenance: Mapping[str, Any] | None = None,
    expected_commit: str | None = None,
    committed_source_sha256: Mapping[str, str] | None = None,
) -> None:
    _dependency_contract()
    support.validate_support_report(support_report, manifest)
    if support_report.get("provenance", {}).get("evaluator_sha256") != SUPPORT_EVALUATOR_SHA256:
        raise ValueError("Experiment 045 shard support evaluator identity mismatch")
    morphology = report.get("identity", {}).get("morphology")
    scenarios = support.support_scenarios(manifest, morphology)
    expected_ids = [
        f"{scenario.scenario_id}/{phase}/{endpoint}"
        for scenario in scenarios
        for phase in PHASES[scenario.task]
        for endpoint in ENDPOINTS
    ]
    anchors = report.get("anchors", [])
    run = report.get("run", {})
    provenance = report.get("provenance", {})
    _validate_persisted_identity(report, morphology)
    _validate_live_provenance(
        provenance,
        str(run.get("device")),
        expected_current_provenance=expected_current_provenance,
        expected_commit=expected_commit,
        committed_source_sha256=committed_source_sha256,
    )
    _source_contract(source_paths)
    if (
        report.get("format_version") != 1
        or report.get("experiment") != 45
        or report.get("mode") != "support_bound_fresh_gripper_morphology_shard"
        or report.get("registered") is not True
        or report.get("status") != "completed"
        or report.get("complete") is not True
        or report.get("support_preflight_passed") is not True
        or report.get("contract") != _contract(True)
        or report.get("expected_anchor_ids") != expected_ids
        or [
            f"{anchor.get('scenario_id')}/{anchor.get('phase')}/{anchor.get('endpoint')}"
            for anchor in anchors
        ]
        != expected_ids
        or report.get("outcome_chain_sha256") != exp42._outcome_chain(anchors)
        or report.get("infrastructure_errors") != []
        or run.get("orchestrator_run_id") is None
        or run.get("finished_utc") is None
    ):
        raise ValueError("Experiment 045 shard contract or completion state is invalid")
    validate_recaptured_support(
        _without_policy_results(anchors),
        report.get("episodes", []),
        support_report,
        manifest,
        morphology,
    )
    for ordinal, anchor in enumerate(anchors):
        if anchor.get("ordinal") != ordinal or tuple(anchor.get("seeds", {})) != tuple(
            str(seed) for seed in SEEDS
        ):
            raise ValueError("Experiment 045 anchor or seed ordering is invalid")
        for seed in SEEDS:
            results = anchor["seeds"][str(seed)]
            if tuple(results) != condition_order(ordinal, seed, morphology):
                raise ValueError("Experiment 045 condition permutation is invalid")
            for condition in CONDITIONS:
                result = results[condition]
                queries = result.get("queries", [])
                if (
                    result.get("seed") != seed
                    or result.get("condition") != condition
                    or len(queries) != 2
                ):
                    raise ValueError("Experiment 045 condition result is invalid")
                first = exp43._validate_query(queries[0], anchor, repeat=False)
                repeat = exp43._validate_query(queries[1], anchor, repeat=True)
                difference = float(np.max(np.abs(first - repeat)))
                if (
                    result.get("repeat_max_abs_difference") != difference
                    or difference > exp43.REPEAT_MAX_ABS
                    or result.get("metrics")
                    != exp43.action_metrics(first[0, 0], anchor["expert_target"])
                ):
                    raise ValueError("Experiment 045 output metrics or repeat do not recompute")
        try:
            exp43._require_stable_anchor_inputs(anchor["seeds"])
        except RuntimeError as error:
            raise ValueError("Experiment 045 fresh policy inputs differ across nine arms") from error
    _validate_checkpoints(report, source_paths, root=artifact_root)
    aggregate = _shard_aggregate(anchors, report)
    if report.get("aggregate") != aggregate:
        raise ValueError("Experiment 045 shard aggregate does not independently recompute")
    runtime = provenance.get("runtime")
    if (
        len(anchors) != EXPECTED_ENDPOINTS // 2
        or aggregate["phase_units"] != EXPECTED_PHASE_UNITS // 2
        or aggregate["scored_chunks"] != EXPECTED_SCORED_CHUNKS // 2
        or aggregate["repeat_chunks"] != EXPECTED_REPEAT_CHUNKS // 2
        or aggregate["policy_inferences"] != EXPECTED_INFERENCES // 2
        or any(
            aggregate["contrast_observations"][name]
            != EXPECTED_CONTRAST_OBSERVATIONS // 2
            for name in CONTRASTS
        )
        or not exp43._valid_run_id(run.get("orchestrator_run_id"))
        or run.get("lock_identity") != LOCK_IDENTITY
        or run.get("lock_path") != str(REGISTERED_LOCK_PATH)
        or run.get("schedule_identity") != SCHEDULE_IDENTITY
        or run.get("schedule_position") != MORPHOLOGIES.index(morphology)
        or not isinstance(runtime, dict)
        or provenance.get("runtime_signature_sha256")
        != hashlib.sha256(
            json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        or exp43._parse_utc(run["finished_utc"]) < exp43._parse_utc(run["started_utc"])
        or not aggregate["delivery_gate"]["passed"]
    ):
        raise ValueError("registered Experiment 045 shard provenance or delivery is invalid")


@dataclass
class _OrchestratorContext:
    run_id: str
    next_position: int = 0
    active: bool = True

    def require(self, morphology: str) -> None:
        if (
            not self.active
            or self is not _ACTIVE_ORCHESTRATOR
            or not exp43._valid_run_id(self.run_id)
            or self.next_position >= len(MORPHOLOGIES)
            or MORPHOLOGIES[self.next_position] != morphology
        ):
            raise RuntimeError("registered Experiment 045 requires the active shard schedule")

    def complete(self, morphology: str) -> None:
        self.require(morphology)
        self.next_position += 1


_ACTIVE_ORCHESTRATOR: _OrchestratorContext | None = None


@contextmanager
def registered_run_lock(path: Path = REGISTERED_LOCK_PATH) -> Iterator[None]:
    if path != REGISTERED_LOCK_PATH:
        raise ValueError(f"registered Experiment 045 lock path must be {REGISTERED_LOCK_PATH}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another registered Experiment 045 orchestrator holds the lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def registered_shard_filename(morphology: str) -> str:
    if morphology not in MORPHOLOGIES:
        raise ValueError("registered shard filename requires a registered morphology")
    return f"benchmark-exp45-{morphology}-fresh-gripper.json"


def registered_archive_filename(morphology: str) -> str:
    return f"{registered_shard_filename(morphology)}.gz"


def _registered_output(morphology: str, output: Path | None) -> Path:
    expected = REGISTERED_RUNTIME_DIR / registered_shard_filename(morphology)
    if output is not None and output != expected:
        raise ValueError(f"registered Experiment 045 shard output must be {expected}")
    return expected


def evaluate_fresh_gripper_shard(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    support_capture: RegisteredSupportCapture,
    provenance: dict[str, Any],
    output: Path,
    *,
    morphology: str,
    device: str,
    orchestrator: _OrchestratorContext,
    root: Path = Path("."),
    policy_loader: Callable[[Path, str], Any] = load_geometry_policy,
    processor_factory: Callable[..., Any] = EmbodiProcessor.from_pretrained,
) -> dict[str, Any]:
    if (
        definition.sha256 != DEFINITION_SHA256
        or manifest.definition_sha256 != DEFINITION_SHA256
        or file_sha256(manifest.path) != DEVELOPMENT_MANIFEST_SHA256
    ):
        raise ValueError("Experiment 045 benchmark inputs differ from registration")
    orchestrator.require(morphology)
    if output != _registered_output(morphology, None):
        raise ValueError("registered Experiment 045 output path is frozen")
    if output.exists():
        raise FileExistsError(f"Experiment 045 shards are non-resumable; discard {output} first")
    evaluator_sha256 = file_sha256(Path(__file__))
    if evaluator_sha256 != registered_evaluator_sha256() or provenance["git_dirty"]:
        raise RuntimeError("registered Experiment 045 requires a clean frozen evaluator commit")
    started = exp43._utc_now()

    if set(support_capture.anchors) != set(MORPHOLOGIES) or set(
        support_capture.episodes
    ) != set(MORPHOLOGIES):
        raise ValueError("Experiment 045 requires a complete two-morphology support cache")
    support_report = support_capture.report
    support.validate_support_report(support_report, manifest)
    for cached_morphology in MORPHOLOGIES:
        validate_recaptured_support(
            [_fresh_record(anchor) for anchor in support_capture.anchors[cached_morphology]],
            support_capture.episodes[cached_morphology],
            support_report,
            manifest,
            cached_morphology,
        )
    # Source reports, checkpoints, processors, and policies are forbidden above this line.
    _validate_live_provenance(provenance, device)
    _validate_source_file_contract()
    _source_contract(source_paths)
    fresh_anchors = support_capture.anchors[morphology]
    episodes = support_capture.episodes[morphology]
    checkpoints, identities = _checkpoint_identities(source_paths, morphology, root)
    policies = {
        seed: {
            condition: policy_loader(checkpoints[seed][condition], device)
            for condition in CONDITIONS
        }
        for seed in SEEDS
    }
    first_policy = policies[SEEDS[0]][CONDITIONS[0]]
    processor_contract = (
        first_policy.config.backbone_name,
        first_policy.config.backbone_revision,
        first_policy.config.image_do_rescale,
    )
    if any(
        (
            policies[seed][condition].config.backbone_name,
            policies[seed][condition].config.backbone_revision,
            policies[seed][condition].config.image_do_rescale,
        )
        != processor_contract
        for seed in SEEDS
        for condition in CONDITIONS
    ):
        raise ValueError("all Experiment 045 arms must share the processor contract")
    processor = processor_factory(
        processor_contract[0],
        revision=processor_contract[1],
        do_rescale=processor_contract[2],
    )
    records: list[dict[str, Any]] = []
    for anchor in fresh_anchors:
        seed_results = {
            str(seed): {
                condition: evaluate_anchor_condition(
                    anchor,
                    seed,
                    condition,
                    policies[seed][condition],
                    processor,
                    device,
                )
                for condition in condition_order(anchor.ordinal, seed, morphology)
            }
            for seed in SEEDS
        }
        exp43._require_stable_anchor_inputs(seed_results)
        records.append({**_fresh_record(anchor), "seeds": seed_results})
    report: dict[str, Any] = {
        "format_version": 1,
        "experiment": 45,
        "mode": "support_bound_fresh_gripper_morphology_shard",
        "registered": True,
        "status": "completed",
        "complete": True,
        "support_preflight_passed": True,
        "contract": _contract(True),
        "identity": {
            "morphology": morphology,
            "model_seeds": list(SEEDS),
            "checkpoints": identities,
        },
        "expected_anchor_ids": [
            f"{scenario.scenario_id}/{phase}/{endpoint}"
            for scenario in support.support_scenarios(manifest, morphology)
            for phase in PHASES[scenario.task]
            for endpoint in ENDPOINTS
        ],
        "provenance": provenance,
        "run": {
            "started_utc": started,
            "finished_utc": exp43._utc_now(),
            "device": device,
            "lock_identity": LOCK_IDENTITY,
            "lock_path": str(REGISTERED_LOCK_PATH),
            "schedule_identity": SCHEDULE_IDENTITY,
            "schedule_position": MORPHOLOGIES.index(morphology),
            "orchestrator_run_id": orchestrator.run_id,
            "seed_order": list(SEEDS),
            "condition_order_formula": exp43.PROTOCOL["condition_order_formula"],
        },
        "episodes": list(episodes),
        "anchors": records,
        "outcome_chain_sha256": exp42._outcome_chain(records),
        "infrastructure_errors": [],
    }
    report["aggregate"] = _shard_aggregate(records, report)
    validate_shard_report(
        report, manifest, source_paths, support_report, artifact_root=root
    )
    exp43._write_final(output, report)
    orchestrator.complete(morphology)
    return report


def _metric_contrast(view: Mapping[str, Any], contrast: str = "J-F") -> Mapping[str, Any]:
    return view["effective_gripper_error"]["contrasts"][contrast]


def _material_gate(metrics: Mapping[str, Any], contrast: str = "J-F") -> dict[str, Any]:
    overall = _metric_contrast(metrics["overall"], contrast)
    difference = Decimal(overall["difference_decimal"])
    ratio = Decimal(overall["ratio_decimal"]) if overall["ratio_decimal"] is not None else None
    seed_ratios = {
        seed: _metric_contrast(metrics["by_seed"][str(seed)], contrast)["ratio_decimal"]
        for seed in SEEDS
    }
    checks = {
        "difference_at_least_0_01": difference >= Decimal("0.01"),
        "ratio_at_least_1_10": ratio is not None and ratio >= Decimal("1.10"),
        "at_least_two_seed_ratios_at_least_1_10": sum(
            value is not None and Decimal(value) >= Decimal("1.10")
            for value in seed_ratios.values()
        )
        >= 2,
    }
    return {
        "supported": all(checks.values()),
        "checks": checks,
        "difference": float(difference),
        "ratio": float(ratio) if ratio is not None else None,
        "seed_ratios": {
            str(seed): float(Decimal(value)) if value is not None else None
            for seed, value in seed_ratios.items()
        },
    }


def _difference_of_differences(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    left_value = Decimal(_metric_contrast(left)["difference_decimal"])
    right_value = Decimal(_metric_contrast(right)["difference_decimal"])
    difference = left_value - right_value
    return {
        "left_difference": float(left_value),
        "right_difference": float(right_value),
        "difference": float(difference),
        "difference_decimal": str(difference),
    }


def _records_by(records: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return dict(grouped)


def localization_contrast(
    left_records: Sequence[dict[str, Any]],
    right_records: Sequence[dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
    expected_left_ids: set[tuple[str, str, str, str, str]],
    expected_right_ids: set[tuple[str, str, str, str, str]],
    expected_tasks: set[str],
) -> dict[str, Any]:
    def validate_side(
        records: Sequence[dict[str, Any]], expected_ids: set[tuple[str, str, str, str, str]]
    ) -> dict[tuple[str, str], set[str]]:
        actual: dict[tuple[int, str], set[tuple[str, str, str, str, str]]] = defaultdict(set)
        counts: dict[tuple[int, str, tuple[str, str, str, str, str]], set[str]] = defaultdict(set)
        scenarios: dict[tuple[str, str], set[str]] = defaultdict(set)
        for record in records:
            identity = tuple(
                str(record[key])
                for key in ("morphology", "task", "scenario_id", "phase", "endpoint")
            )
            actual[(record["seed"], record["condition"])].add(identity)
            counts[(record["seed"], record["condition"], identity)].add(record["condition"])
            scenarios[(record["morphology"], record["task"])].add(record["scenario_id"])
        if (
            set(actual) != {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
            or any(value != expected_ids for value in actual.values())
            or len(records) != len(expected_ids) * len(SEEDS) * len(CONDITIONS)
            or len(counts) != len(expected_ids) * len(SEEDS) * len(CONDITIONS)
            or {task for _morphology, task in scenarios} != expected_tasks
            or set(scenarios) != {
                (morphology, task) for morphology in MORPHOLOGIES for task in expected_tasks
            }
            or any(len(values) != 7 for values in scenarios.values())
        ):
            raise ValueError("Experiment 045 localization side lacks exact support-manifest records")
        return scenarios

    left_scenarios = validate_side(left_records, expected_left_ids)
    right_scenarios = validate_side(right_records, expected_right_ids)
    if left_scenarios != right_scenarios:
        raise ValueError("Experiment 045 localization sides lack exact common scenario support")
    left = aggregate_metrics(left_records)
    right = aggregate_metrics(right_records)
    differences: dict[str, Any] = {}
    expected_groups = {
        "by_seed": {str(seed) for seed in SEEDS},
        "by_morphology": set(MORPHOLOGIES),
        "by_task": expected_tasks,
        "by_endpoint": set(ENDPOINTS),
    }
    for group, expected in expected_groups.items():
        if set(left[group]) != expected or set(right[group]) != expected:
            raise ValueError("Experiment 045 localization subgroup support differs")
        differences[group] = {
            key: _difference_of_differences(left[group][key], right[group][key])
            for key in sorted(expected)
        }
    return {
        "left": left_label,
        "right": right_label,
        "left_metrics": left,
        "right_metrics": right,
        "overall": _difference_of_differences(left["overall"], right["overall"]),
        **differences,
    }


def _nonnegative_jf(metrics: Mapping[str, Any]) -> bool:
    return all(
        Decimal(_metric_contrast(cell)["difference_decimal"]) >= 0
        for group in ("by_morphology", "by_task", "by_endpoint")
        for cell in metrics[group].values()
    )


def _localization_gate(name: str, contrast: Mapping[str, Any]) -> dict[str, Any]:
    difference = Decimal(contrast["overall"]["difference_decimal"])
    direction = 1 if difference >= 0 else -1
    winner_key = "left_metrics" if direction > 0 else "right_metrics"
    winner = contrast["left"] if direction > 0 else contrast["right"]

    def strict_sign(value: Mapping[str, Any]) -> bool:
        return Decimal(value["difference_decimal"]) * direction > 0

    material = _material_gate(contrast[winner_key])
    checks = {
        "winning_composite_material": material["supported"],
        "absolute_interaction_at_least_0_01": abs(difference) >= Decimal("0.01"),
        "at_least_two_seed_interactions_strict_same_sign": sum(
            strict_sign(value) for value in contrast["by_seed"].values()
        )
        >= 2,
        "both_morphology_interactions_strict_same_sign": all(
            strict_sign(value) for value in contrast["by_morphology"].values()
        ),
    }
    localized = all(checks.values())
    broad = localized and _nonnegative_jf(contrast[winner_key]) and name != "opening_transition"
    classification = (
        f"pick-place-specific-{winner}-localization"
        if name == "opening_transition"
        else f"{winner}-localized-{'broad' if broad else 'narrow'}"
    )
    return {
        "localized": localized,
        "broad": broad,
        "classification": classification if localized else None,
        "winner": winner,
        "winning_composite_materiality": material,
        "checks": checks,
    }


def classify_localization(
    partitions: Mapping[str, Any], localization: Mapping[str, Any], *, delivery: bool
) -> dict[str, Any]:
    if not delivery:
        return {"classification": "invalid_delivery", "delivery_gate_passed": False}
    partition_gates = {
        name: _material_gate(partitions[name]["metrics"]) for name in PARTITIONS
    }
    localization_gates = {
        name: _localization_gate(name, contrast) for name, contrast in localization.items()
    }
    labels = [
        gate["classification"] for gate in localization_gates.values() if gate["localized"]
    ]
    if not any(gate["supported"] for gate in partition_gates.values()):
        classification = "not_materially_localized"
    elif len(labels) > 1:
        classification = "multiple-localizations"
    elif labels:
        classification = labels[0]
    else:
        classification = "distributed_material_deficit"
    return {
        "classification": classification,
        "delivery_gate_passed": True,
        "partition_materiality": partition_gates,
        "localization_gates": localization_gates,
        "inferential_claim": False,
        "causal_mechanism_claim": False,
    }


def _support_identity(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(
        str(record[key])
        for key in ("morphology", "task", "scenario_id", "phase", "endpoint")
    )


def _selected_support_ids(
    support_report: Mapping[str, Any], *, partitions: set[str], tasks: set[str] | None = None
) -> set[tuple[str, str, str, str, str]]:
    return {
        _support_identity(record)
        for record in support_report["records"]
        if record["partition"] in partitions and (tasks is None or record["task"] in tasks)
    }


def deterministic_gzip_bytes(raw: bytes) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=9, mtime=0) as archive:
        archive.write(raw)
    return stream.getvalue()


def validated_archive_bytes(
    stored: bytes, *, compressed_sha256: str, decompressed_sha256: str
) -> bytes:
    if hashlib.sha256(stored).hexdigest() != compressed_sha256:
        raise ValueError("Experiment 045 compressed source hash mismatch")
    try:
        raw = gzip.decompress(stored)
    except (gzip.BadGzipFile, EOFError, zlib.error) as error:
        raise ValueError("Experiment 045 source archive is invalid") from error
    if hashlib.sha256(raw).hexdigest() != decompressed_sha256:
        raise ValueError("Experiment 045 decompressed source hash mismatch")
    return raw


def _common_shard_identity(
    reports: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, str]:
    if set(reports) != set(MORPHOLOGIES):
        raise ValueError("Experiment 045 common identity requires both morphology shards")
    provenances = [reports[morphology]["provenance"] for morphology in MORPHOLOGIES]
    runs = [reports[morphology]["run"] for morphology in MORPHOLOGIES]
    commits = {provenance["git_commit"] for provenance in provenances}
    run_ids = {run["orchestrator_run_id"] for run in runs}
    signatures = {provenance["runtime_signature_sha256"] for provenance in provenances}
    intervals = [
        (exp43._parse_utc(run["started_utc"]), exp43._parse_utc(run["finished_utc"]))
        for run in runs
    ]
    if (
        len(commits) != 1
        or len(run_ids) != 1
        or not exp43._valid_run_id(next(iter(run_ids), None))
        or len(signatures) != 1
        or any(provenance != provenances[0] for provenance in provenances[1:])
        or intervals[0][1] > intervals[1][0]
        or any(
            run["schedule_position"] != position for position, run in enumerate(runs)
        )
    ):
        raise ValueError("Experiment 045 shards violate common clean commit/runtime or schedule")
    return next(iter(commits)), next(iter(run_ids)), next(iter(signatures))


def aggregate_registered_shards(
    shard_paths: Mapping[str, Path],
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    support_path: Path,
    *,
    artifact_root: Path = Path("."),
) -> dict[str, Any]:
    _dependency_contract()
    if file_sha256(Path(__file__)) != registered_evaluator_sha256():
        raise ValueError("Experiment 045 evaluator hash does not match registration")
    support_report = load_support_report(support_path, manifest)
    _source_contract(source_paths)
    if set(shard_paths) != set(MORPHOLOGIES):
        raise ValueError("Experiment 045 aggregation requires exact SO-101 and Panda shards")
    reports: dict[str, dict[str, Any]] = {}
    raw_values: dict[str, bytes] = {}
    for morphology in MORPHOLOGIES:
        path = shard_paths[morphology]
        if path.name != registered_shard_filename(morphology):
            raise ValueError("Experiment 045 source shard path is misattributed")
        raw_values[morphology] = path.read_bytes()
        reports[morphology] = json.loads(raw_values[morphology])
        validate_shard_report(
            reports[morphology],
            manifest,
            source_paths,
            support_report,
            artifact_root=artifact_root,
        )
    commit, run_id, signature = _common_shard_identity(reports)
    records = _flatten_condition_records([reports[morphology] for morphology in MORPHOLOGIES])
    partitions = {
        partition: {
            "support_endpoint_anchors": support_report["support"][
                "by_partition_endpoint_anchors"
            ][partition],
            "condition_records": sum(record["partition"] == partition for record in records),
            "metrics": aggregate_metrics(
                [record for record in records if record["partition"] == partition]
            ),
        }
        for partition in PARTITIONS
    }

    def selected(*, partitions: set[str], tasks: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            record
            for record in records
            if record["partition"] in partitions and (tasks is None or record["task"] in tasks)
        ]

    manipulation = {"lift_object", "pick_place_bin"}

    def contrast(
        left_partitions: set[str],
        right_partitions: set[str],
        left_label: str,
        right_label: str,
        tasks: set[str],
    ) -> dict[str, Any]:
        return localization_contrast(
            selected(partitions=left_partitions, tasks=tasks),
            selected(partitions=right_partitions, tasks=tasks),
            left_label=left_label,
            right_label=right_label,
            expected_left_ids=_selected_support_ids(
                support_report, partitions=left_partitions, tasks=tasks
            ),
            expected_right_ids=_selected_support_ids(
                support_report, partitions=right_partitions, tasks=tasks
            ),
            expected_tasks=tasks,
        )

    localization = {
        "target_state": contrast(
            {"closed_steady", "closed_closeward"},
            {"open_steady", "open_openward"},
            "closed_target",
            "open_target",
            manipulation,
        ),
        "opening_transition": contrast(
            {"open_openward"},
            {"open_steady"},
            "openward",
            "open_steady",
            {"pick_place_bin"},
        ),
        "closing_transition": contrast(
            {"closed_closeward"},
            {"closed_steady"},
            "closeward",
            "closed_steady",
            manipulation,
        ),
        "terminal_behavior": contrast(
            {"terminal_noop"},
            set(PARTITIONS) - {"terminal_noop"},
            "terminal_noop",
            "commanded",
            set(TASKS),
        ),
    }
    raw_hashes = {
        morphology: hashlib.sha256(raw_values[morphology]).hexdigest()
        for morphology in MORPHOLOGIES
    }
    archives = {
        morphology: deterministic_gzip_bytes(raw_values[morphology])
        for morphology in MORPHOLOGIES
    }
    archive_hashes = {
        morphology: hashlib.sha256(archives[morphology]).hexdigest()
        for morphology in MORPHOLOGIES
    }
    delivery_checks = {
        "both_shards_independently_validated": True,
        "support_report_and_assets_revalidated": True,
        "common_clean_commit_runtime_and_run_id": True,
        "exact_42_expert_episodes": sum(
            report["aggregate"]["episodes"] for report in reports.values()
        )
        == EXPECTED_EPISODES,
        "exact_182_scenario_phase_units": sum(
            report["aggregate"]["phase_units"] for report in reports.values()
        )
        == EXPECTED_PHASE_UNITS,
        "exact_364_endpoint_anchors": sum(
            report["aggregate"]["endpoint_anchors"] for report in reports.values()
        )
        == EXPECTED_ENDPOINTS,
        "exact_3276_scored_chunks": len(records) == EXPECTED_SCORED_CHUNKS,
        "exact_3276_repeat_chunks": sum(
            report["aggregate"]["repeat_chunks"] for report in reports.values()
        )
        == EXPECTED_REPEAT_CHUNKS,
        "exact_6552_policy_inferences": sum(
            report["aggregate"]["policy_inferences"] for report in reports.values()
        )
        == EXPECTED_INFERENCES,
        "exact_1092_each_contrast": all(
            sum(
                report["aggregate"]["contrast_observations"][name]
                for report in reports.values()
            )
            == EXPECTED_CONTRAST_OBSERVATIONS
            for name in CONTRASTS
        ),
        "partition_records_exhaustive": sum(
            cell["condition_records"] for cell in partitions.values()
        )
        == EXPECTED_SCORED_CHUNKS,
        "training_runs_is_zero": True,
        "final_split_not_loaded": True,
    }
    delivery = all(delivery_checks.values())
    summary = {
        "format_version": 1,
        "experiment": 45,
        "registered": True,
        "status": "completed_fresh_gripper_localization",
        "scope": (
            "development-only descriptive expert-state assay; no inferential, population, "
            "causal, closed-loop, admission, advancement, retraining, or final-split claim"
        ),
        "contract": {
            **_contract(True),
            "evaluator_sha256": file_sha256(Path(__file__)),
            "evaluator_git_commit": commit,
            "orchestrator_run_id": run_id,
            "runtime_signature_sha256": signature,
            "shards": 2,
            "expert_episodes": EXPECTED_EPISODES,
            "scenario_phase_units": EXPECTED_PHASE_UNITS,
            "endpoint_records": EXPECTED_ENDPOINTS,
            "scored_chunks": EXPECTED_SCORED_CHUNKS,
            "repeat_chunks": EXPECTED_REPEAT_CHUNKS,
            "policy_inferences": EXPECTED_INFERENCES,
            "contrast_observations": {
                name: EXPECTED_CONTRAST_OBSERVATIONS for name in CONTRASTS
            },
            "hierarchy": [
                "endpoint",
                "phase",
                "scenario",
                "task",
                "morphology",
                "seed",
            ],
        },
        "source_shard_sha256": raw_hashes,
        "source_archive_sha256": archive_hashes,
        "source_archive_filenames": {
            morphology: registered_archive_filename(morphology)
            for morphology in MORPHOLOGIES
        },
        "delivery_checks": delivery_checks,
        "delivery_gate_passed": delivery,
        "metrics": aggregate_metrics(records),
        "partitions": partitions,
        "localization_contrasts": localization,
        "localization_classification": classify_localization(
            partitions, localization, delivery=delivery
        ),
        "inferential_claim": False,
        "causal_mechanism_claim": False,
        "closed_loop_success_claim": False,
        "admission_gate": None,
        "advancement_gate": None,
        "retraining_decision": None,
        "final_split_loaded": False,
    }
    return summary


def validate_summary_report(
    summary: Mapping[str, Any],
    shard_paths: Mapping[str, Path],
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    support_path: Path,
    *,
    artifact_root: Path = Path("."),
) -> None:
    expected = aggregate_registered_shards(
        shard_paths,
        manifest,
        source_paths,
        support_path,
        artifact_root=artifact_root,
    )
    if summary != expected:
        raise ValueError("Experiment 045 summary does not independently recompute")


def _promote_registered_archives(
    source_paths: Mapping[str, Path],
    raw_sha256: Mapping[str, str],
    archive_sha256: Mapping[str, str],
    *,
    destination_dir: Path = Path("reports"),
) -> None:
    if not (
        set(source_paths) == set(raw_sha256) == set(archive_sha256) == set(MORPHOLOGIES)
    ):
        raise ValueError("Experiment 045 archive promotion requires both morphology shards")
    raw = {morphology: source_paths[morphology].read_bytes() for morphology in MORPHOLOGIES}
    archives = {
        morphology: deterministic_gzip_bytes(raw[morphology]) for morphology in MORPHOLOGIES
    }
    if any(
        hashlib.sha256(raw[morphology]).hexdigest() != raw_sha256[morphology]
        or hashlib.sha256(archives[morphology]).hexdigest() != archive_sha256[morphology]
        or validated_archive_bytes(
            archives[morphology],
            compressed_sha256=archive_sha256[morphology],
            decompressed_sha256=raw_sha256[morphology],
        )
        != raw[morphology]
        for morphology in MORPHOLOGIES
    ):
        raise ValueError("Experiment 045 staged source bytes changed after aggregation")
    destinations = {
        morphology: destination_dir / registered_archive_filename(morphology)
        for morphology in MORPHOLOGIES
    }
    for morphology, destination in destinations.items():
        if destination.exists() and destination.read_bytes() != archives[morphology]:
            raise FileExistsError(f"refusing conflicting promoted source archive: {destination}")
    for morphology in MORPHOLOGIES:
        exp43._write_exact_bytes(destinations[morphology], archives[morphology])


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "exp36_summary": args.exp36_summary,
        "exp42_summary": args.exp42_summary,
        "exp43_training_summary": args.exp43_training_summary,
        "exp42_so101_shard": args.exp42_so101_shard,
        "exp42_panda_shard": args.exp42_panda_shard,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="run the frozen Experiment 045 support-bound F/H/J evaluator"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--definition", type=Path, default=Path("benchmarks/sim-v1/definition.json")
    )
    common.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/sim-v1/manifests/development.json"),
    )
    common.add_argument("--support", type=Path, default=SUPPORT_PATH)
    common.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    common.add_argument(
        "--exp42-summary", type=Path, default=Path("reports/benchmark-exp42-summary.json")
    )
    common.add_argument(
        "--exp43-training-summary",
        type=Path,
        default=Path("reports/benchmark-exp43-training-summary.json"),
    )
    common.add_argument(
        "--exp42-so101-shard",
        type=Path,
        default=Path("reports/benchmark-exp42-so101-phase-action-quality.json"),
    )
    common.add_argument(
        "--exp42-panda-shard",
        type=Path,
        default=Path("reports/benchmark-exp42-panda-phase-action-quality.json"),
    )
    run = subparsers.add_parser("run-registered", parents=[common])
    run.add_argument("--artifact-root", type=Path, default=Path("."))
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--artifact-root", type=Path, default=Path("."))
    aggregate.add_argument("--shard-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    aggregate.add_argument("--output", type=Path, default=SUMMARY_PATH)
    return parser.parse_args(argv)


def _run_all_registered(
    args: argparse.Namespace,
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
) -> list[dict[str, Any]]:
    global _ACTIVE_ORCHESTRATOR
    outputs = [_registered_output(morphology, None) for morphology in MORPHOLOGIES]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "Experiment 045 is non-resumable; delete both staged shards before restart"
        )
    reports: list[dict[str, Any]] = []
    with registered_run_lock():
        if any(path.exists() for path in outputs):
            raise FileExistsError(
                "Experiment 045 staged shard appeared before schedule; delete both before restart"
            )
        orchestrator = _OrchestratorContext(uuid.uuid4().hex)
        if _ACTIVE_ORCHESTRATOR is not None:
            raise RuntimeError("registered Experiment 045 orchestrator is already active")
        _ACTIVE_ORCHESTRATOR = orchestrator
        try:
            evaluator_sha256 = file_sha256(Path(__file__))
            provenance = _runtime_provenance(args.device)
            if evaluator_sha256 != registered_evaluator_sha256() or provenance["git_dirty"]:
                raise RuntimeError(
                    "registered Experiment 045 requires a clean frozen evaluator commit"
                )
            support_capture = preflight_registered_support(manifest, args.support)
            source_paths = _source_paths(args)
            # The full source closure and frozen reports are touched only after both recaptures pass.
            _validate_live_provenance(provenance, args.device)
            _validate_source_file_contract()
            _source_contract(source_paths)
            for morphology, output in zip(MORPHOLOGIES, outputs, strict=True):
                reports.append(
                    evaluate_fresh_gripper_shard(
                        definition,
                        manifest,
                        source_paths,
                        support_capture,
                        provenance,
                        output,
                        morphology=morphology,
                        device=args.device,
                        orchestrator=orchestrator,
                        root=args.artifact_root,
                    )
                )
            if orchestrator.next_position != len(MORPHOLOGIES):
                raise RuntimeError("registered Experiment 045 orchestrator did not complete")
        finally:
            orchestrator.active = False
            _ACTIVE_ORCHESTRATOR = None
    return reports


def main() -> None:
    args = parse_args()
    if args.support != SUPPORT_PATH:
        raise ValueError(f"registered Experiment 045 support path must be {SUPPORT_PATH}")
    if args.command == "aggregate" and (
        args.shard_dir != REGISTERED_RUNTIME_DIR or args.output != SUMMARY_PATH
    ):
        raise ValueError("registered Experiment 045 aggregate paths are frozen")
    definition, manifest = load_development_contract(
        args.definition, args.manifest, registered=True
    )
    if (
        definition.sha256 != DEFINITION_SHA256
        or file_sha256(manifest.path) != DEVELOPMENT_MANIFEST_SHA256
    ):
        raise ValueError("Experiment 045 benchmark contract does not match registration")
    if args.command == "run-registered":
        reports = _run_all_registered(args, definition, manifest)
        print(json.dumps({"completed_shards": len(reports), "status": "completed"}, indent=2))
        return
    shard_paths = {
        morphology: args.shard_dir / registered_shard_filename(morphology)
        for morphology in MORPHOLOGIES
    }
    with registered_run_lock():
        summary = aggregate_registered_shards(
            shard_paths,
            manifest,
            _source_paths(args),
            args.support,
            artifact_root=args.artifact_root,
        )
        validate_summary_report(
            summary,
            shard_paths,
            manifest,
            _source_paths(args),
            args.support,
            artifact_root=args.artifact_root,
        )
        exp43._preflight_final(args.output, summary)
        _promote_registered_archives(
            shard_paths,
            summary["source_shard_sha256"],
            summary["source_archive_sha256"],
        )
        exp43._write_final(args.output, summary)
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
