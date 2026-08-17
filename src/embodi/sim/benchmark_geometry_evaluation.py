from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch

from ..joint_train import runtime_provenance
from ..processing import EmbodiProcessor, camera_frame_to_tensor
from ..record_so101 import atomic_write_json
from ..token_config import EmbodiConfig
from ..token_policy import EmbodiPolicy
from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv, runtime_scenario
from .benchmark_expert import PrivilegedBenchmarkExpert


DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
DEVELOPMENT_MANIFEST_SHA256 = (
    "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
)
EXP36_SUMMARY_SHA256 = "76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f"
DEVELOPMENT_ADMISSION_SHA256 = (
    "a43bb38eb2eb53e82a84128879de856ed55587caabad64e3ea9fc729897ed019"
)
CONDITIONS = ("so101", "panda", "joint")
MORPHOLOGIES = ("so101", "panda")
TASKS = ("push_to_zone", "lift_object", "pick_place_bin", "stack_object")
SEEDS = (36001, 36002, 36003)
CLIP_TOLERANCE = 2e-5
CONFIG_SHA256 = {
    "so101": "a906d494a8f7955de536098cf873efcdeb6566de84f245bc248dea03104dda65",
    "panda": "2a292fb0153d54efd913e4ac77a0d944d7ca7cb006d613375d9c6070fcd4d4a2",
}
DATA_MANIFEST_SHA256 = {
    "so101": "ca35b9a2536218b8b2ff55b5916284ed623310063bc3ada9d6141d8b308fc9a1",
    "panda": "6408528589825a18b9262ba8e8fe4cb67081d8f82164178a8bff20697f8be005",
    "joint": "a6129550b3a673e712f1a9bec0253911150c7d2815df579421173f4072f5b668",
}
EMPTY_OUTCOME_CHAIN_SHA256 = hashlib.sha256(b"").hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _write_progress(path: Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, _jsonable(report))


def _write_final(path: Path, report: dict[str, Any]) -> None:
    value = _jsonable(report)
    text = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"refusing to replace different report: {path}")
        return
    atomic_write_json(path, value)


def _outcome_chain(outcomes: list[dict[str, Any]]) -> str:
    digest = bytes.fromhex(EMPTY_OUTCOME_CHAIN_SHA256)
    for outcome in outcomes:
        encoded = json.dumps(_jsonable(outcome), sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(digest + encoded).digest()
    return digest.hex()


def _append_outcome(report: dict[str, Any], outcome: dict[str, Any]) -> None:
    report["outcomes"].append(outcome)
    report["outcome_chain_sha256"] = _outcome_chain(report["outcomes"])


def _validate_outcome_chain(report: dict[str, Any]) -> None:
    if report.get("outcome_chain_sha256") != _outcome_chain(report.get("outcomes", [])):
        raise ValueError("resume report outcome hash chain does not match")


def _validate_outcomes(report: dict[str, Any]) -> None:
    mode = report.get("mode")
    for outcome in report.get("outcomes", []):
        if not isinstance(outcome.get("scenario_id"), str) or not isinstance(
            outcome.get("success"), bool
        ):
            raise ValueError("report outcomes require scenario ids and boolean success")
        steps = outcome.get("steps")
        commands = outcome.get("commands")
        clipped = outcome.get("clipped_commands")
        minimum_steps = 1 if mode == "geometry_admission" else 0
        if (
            not isinstance(steps, int)
            or not minimum_steps <= steps <= 500
            or not isinstance(commands, int)
            or commands != steps
            or not isinstance(clipped, int)
            or not 0 <= clipped <= commands
        ):
            raise ValueError("report outcome step and clipping counters are invalid")
        if mode == "policy_shard":
            expected_rate = clipped / commands if commands else 0.0
            if outcome.get("command_clipping_rate") != expected_rate:
                raise ValueError("policy outcome clipping rate does not recompute")
            if outcome["success"] and outcome.get("failure_reason") is not None:
                raise ValueError("successful policy outcomes cannot contain a failure reason")
            if not outcome["success"] and not isinstance(outcome.get("failure_reason"), str):
                raise ValueError("failed policy outcomes require a failure reason")
        elif mode == "geometry_admission":
            for name in (
                "translation_errors_m",
                "rotation_errors_deg",
                "gripper_errors_native",
            ):
                values = outcome.get(name)
                if not isinstance(values, list) or len(values) != commands or not np.isfinite(values).all():
                    raise ValueError("geometry admission error vectors must match command counts")


def load_development_contract(
    definition_path: Path,
    manifest_path: Path,
    *,
    registered: bool,
) -> tuple[BenchmarkDefinition, ScenarioManifest]:
    definition = BenchmarkDefinition.load(definition_path)
    if definition.sha256 != DEFINITION_SHA256:
        raise ValueError("experiment 037 benchmark definition hash does not match registration")
    if file_sha256(manifest_path) != DEVELOPMENT_MANIFEST_SHA256:
        raise ValueError("experiment 037 development manifest hash does not match registration")
    manifest = ScenarioManifest.load(
        manifest_path,
        definition,
        require_complete_counts=registered,
    )
    if any(scenario.split != "development" for scenario in manifest.scenarios):
        raise ValueError("experiment 037 may only load the development split")
    if registered:
        counts = Counter((scenario.morphology, scenario.task) for scenario in manifest.scenarios)
        expected_cells = {(morphology, task) for morphology in MORPHOLOGIES for task in TASKS}
        if len(manifest.scenarios) != 400 or set(counts) != expected_cells:
            raise ValueError("registered experiment 037 requires all 400 development scenarios")
        if any(counts[cell] != 50 for cell in expected_cells):
            raise ValueError("registered experiment 037 requires exactly 50 scenarios per cell")
    return definition, manifest


def select_scenarios(
    manifest: ScenarioManifest,
    *,
    morphology: str | None = None,
    limit_per_cell: int | None = None,
) -> tuple[Scenario, ...]:
    selected = []
    counts: Counter[tuple[str, str]] = Counter()
    for scenario in manifest.scenarios:
        if morphology is not None and scenario.morphology != morphology:
            continue
        cell = (scenario.morphology, scenario.task)
        if limit_per_cell is not None and counts[cell] >= limit_per_cell:
            continue
        counts[cell] += 1
        selected.append(scenario)
    return tuple(selected)


def load_frozen_admission(path: Path) -> dict[str, Any]:
    if file_sha256(path) != DEVELOPMENT_ADMISSION_SHA256:
        raise ValueError("frozen development-admission report hash does not match registration")
    report = json.loads(path.read_text())
    if (
        report.get("benchmark_id") != "embodi-sim-v1"
        or report.get("definition_sha256") != DEFINITION_SHA256
        or report.get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
        or report.get("limit_per_cell") is not None
        or report.get("gate_passed") is not True
    ):
        raise ValueError("frozen development-admission report is not complete and passing")
    expected = {f"{morphology}/{task}" for morphology in MORPHOLOGIES for task in TASKS}
    if set(report.get("cells", {})) != expected or any(
        cell.get("attempted") != 50 for cell in report["cells"].values()
    ):
        raise ValueError("frozen development-admission report does not contain all registered cells")
    return report


def _rotation_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    first = np.asarray(rotation_6d[:3], dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.asarray(rotation_6d[3:6], dtype=np.float64)
    second -= first * np.dot(first, second)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=1)


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = _rotation_matrix(first).T @ _rotation_matrix(second)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _clipping(env: Any, requested: np.ndarray) -> tuple[bool, float]:
    realizable = env.qpos_to_native(env.native_to_qpos(requested))
    error = np.abs(np.asarray(requested, dtype=np.float64) - realizable)
    return bool(np.any(error > CLIP_TOLERANCE)), float(error.max(initial=0.0))


def _new_report(
    *,
    mode: str,
    registered: bool,
    contract: dict[str, Any],
    identity: dict[str, Any],
    expected_ids: list[str],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": 37,
        "mode": mode,
        "registered": registered,
        "status": "in_progress" if registered else "smoke_unregistered_in_progress",
        "contract": contract,
        "identity": identity,
        "expected_outcome_ids": expected_ids,
        "provenance": provenance,
        "complete": False,
        "outcomes": [],
        "outcome_chain_sha256": EMPTY_OUTCOME_CHAIN_SHA256,
        "infrastructure_errors": [],
    }


def _load_or_create_report(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return expected
    report = json.loads(path.read_text())
    immutable = (
        "format_version",
        "experiment",
        "mode",
        "registered",
        "contract",
        "identity",
        "expected_outcome_ids",
        "provenance",
    )
    if any(report.get(key) != expected.get(key) for key in immutable):
        raise ValueError("resume report contract or checkpoint identity does not match")
    outcome_ids = [outcome.get("scenario_id") for outcome in report.get("outcomes", [])]
    if outcome_ids != expected["expected_outcome_ids"][: len(outcome_ids)]:
        raise ValueError("resume report outcomes are not an ordered prefix of the contract")
    if len(outcome_ids) != len(set(outcome_ids)):
        raise ValueError("resume report contains duplicate outcome ids")
    _validate_outcome_chain(report)
    _validate_outcomes(report)
    if report.get("complete") and (
        outcome_ids != expected["expected_outcome_ids"] or report.get("infrastructure_errors")
    ):
        raise ValueError("completed resume report is missing outcomes or contains infrastructure errors")
    return report


def _diagnostic_extrema_update(extrema: dict[str, dict[str, float]], values: dict[str, Any]) -> None:
    for name, raw_value in values.items():
        value = float(raw_value)
        if not np.isfinite(value):
            continue
        if name not in extrema:
            extrema[name] = {"minimum": value, "maximum": value}
        else:
            extrema[name]["minimum"] = min(extrema[name]["minimum"], value)
            extrema[name]["maximum"] = max(extrema[name]["maximum"], value)


def _active_factor_axes(scenario: Scenario) -> list[str]:
    return [
        name
        for name, value in scenario.factors.items()
        if str(value.get("bin", "")).startswith("development:")
    ]


def run_geometry_admission_episode(env: Any, *, maximum_steps: int) -> dict[str, Any]:
    expert = PrivilegedBenchmarkExpert(env)
    translations: list[float] = []
    rotations: list[float] = []
    grippers: list[float] = []
    clipping_errors: list[float] = []
    clipped = 0
    status = None
    extrema: dict[str, dict[str, float]] = {}
    for _ in range(maximum_steps):
        requested = np.asarray(expert.action(), dtype=np.float32)
        if not np.isfinite(requested).all():
            raise FloatingPointError("privileged expert returned a nonfinite native action")
        canonical = np.asarray(env.canonical_arm_action(requested), dtype=np.float32)
        decoded = np.asarray(env.native_action_from_canonical(canonical), dtype=np.float32)
        if not np.isfinite(canonical).all() or not np.isfinite(decoded).all():
            raise FloatingPointError("geometry round trip returned a nonfinite action")
        realized_canonical = np.asarray(env.canonical_arm_action(decoded), dtype=np.float32)
        translations.append(float(np.linalg.norm(canonical[0, :3] - realized_canonical[0, :3])))
        rotations.append(rotation_error_degrees(canonical[0, 3:9], realized_canonical[0, 3:9]))
        grippers.append(float(abs(requested[-1] - decoded[-1])))
        was_clipped, clipping_error = _clipping(env, decoded)
        clipped += int(was_clipped)
        clipping_errors.append(clipping_error)
        env.apply_action(decoded)
        status = env.step_control_period(expert.frequency)
        _diagnostic_extrema_update(extrema, env.diagnostic_state())
        if status.terminal or expert.terminal:
            break
    steps = len(translations)
    return {
        "success": bool(status is not None and status.success),
        "steps": steps,
        "final_stage": None if status is None else status.stage,
        "lifted": bool(env.lifted),
        "translation_errors_m": translations,
        "rotation_errors_deg": rotations,
        "gripper_errors_native": grippers,
        "clipped_commands": clipped,
        "commands": steps,
        "maximum_clipping_error_native": max(clipping_errors, default=0.0),
        "diagnostic_extrema": extrema,
    }


def summarize_geometry_admission(
    outcomes: list[dict[str, Any]],
    definition: BenchmarkDefinition,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        grouped[f"{outcome['morphology']}/{outcome['task']}"].append(outcome)
    cells = {}
    for cell, values in sorted(grouped.items()):
        successes = sum(bool(value["success"]) for value in values)
        success_rate = successes / len(values)
        baseline = float(frozen["cells"][cell]["success_rate"])
        drop = baseline - success_rate
        cells[cell] = {
            "attempted": len(values),
            "successes": successes,
            "success_rate": success_rate,
            "frozen_privileged_success_rate": baseline,
            "success_drop": drop,
            "gate_passed": drop
            <= definition.values["admission_gates"]["maximum_geometry_success_drop"],
        }
    translation = [
        error for outcome in outcomes for error in outcome.get("translation_errors_m", [])
    ]
    rotation = [error for outcome in outcomes for error in outcome.get("rotation_errors_deg", [])]
    gripper = [error for outcome in outcomes for error in outcome.get("gripper_errors_native", [])]
    commands = sum(int(outcome.get("commands", 0)) for outcome in outcomes)
    clipped = sum(int(outcome.get("clipped_commands", 0)) for outcome in outcomes)
    gates = definition.values["admission_gates"]
    translation_p95 = float(np.percentile(translation, 95)) if translation else float("inf")
    rotation_p95 = float(np.percentile(rotation, 95)) if rotation else float("inf")
    gripper_max = max(gripper, default=float("inf"))
    clipping_rate = clipped / commands if commands else float("inf")
    checks = {
        "geometry_success_drop": bool(cells) and all(value["gate_passed"] for value in cells.values()),
        "command_clipping_rate": clipping_rate <= gates["maximum_command_clipping_rate"],
        "fk_ik_translation_p95": translation_p95 <= gates["maximum_fk_ik_translation_p95_m"],
        "fk_ik_rotation_p95": rotation_p95 <= gates["maximum_fk_ik_rotation_p95_deg"],
        "gripper_round_trip": gripper_max <= gates["maximum_gripper_round_trip_error_native"],
    }
    return {
        "cells": cells,
        "measurements": {
            "commands": commands,
            "clipped_commands": clipped,
            "command_clipping_rate": clipping_rate,
            "translation_p95_m": translation_p95,
            "rotation_p95_deg": rotation_p95,
            "maximum_gripper_error_native": gripper_max,
        },
        "thresholds": {
            "maximum_geometry_success_drop": gates["maximum_geometry_success_drop"],
            "maximum_command_clipping_rate": gates["maximum_command_clipping_rate"],
            "maximum_fk_ik_translation_p95_m": gates["maximum_fk_ik_translation_p95_m"],
            "maximum_fk_ik_rotation_p95_deg": gates["maximum_fk_ik_rotation_p95_deg"],
            "maximum_gripper_round_trip_error_native": gates[
                "maximum_gripper_round_trip_error_native"
            ],
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_geometry_admission(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    frozen: dict[str, Any],
    output: Path,
    *,
    registered: bool,
    limit_per_cell: int | None = None,
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
) -> dict[str, Any]:
    provenance = runtime_provenance(require_clean=False)
    if registered and provenance["git_dirty"]:
        raise RuntimeError("registered experiment 037 must run from a clean git worktree")
    provenance["evaluator_sha256"] = file_sha256(Path(__file__))
    scenarios = select_scenarios(manifest, limit_per_cell=limit_per_cell)
    contract = {
        "definition_sha256": definition.sha256,
        "manifest_sha256": file_sha256(manifest.path),
        "frozen_admission_sha256": DEVELOPMENT_ADMISSION_SHA256,
        "split": "development",
        "limit_per_cell": limit_per_cell,
        "protocol": {
            "frequency_hz": 30,
            "maximum_episode_steps": 500,
            "round_trip": "native->canonical_arm_action->native_action_from_canonical",
        },
    }
    expected = _new_report(
        mode="geometry_admission",
        registered=registered,
        contract=contract,
        identity={},
        expected_ids=[scenario.scenario_id for scenario in scenarios],
        provenance=provenance,
    )
    report = _load_or_create_report(output, expected)
    if report.get("complete"):
        aggregate = summarize_geometry_admission(report["outcomes"], definition, frozen)
        if aggregate != report.get("aggregate"):
            raise ValueError("completed geometry admission aggregate does not recompute")
        return report
    completed = len(report["outcomes"])
    for scenario in scenarios[completed:]:
        env = None
        try:
            env = env_factory(
                scenario.morphology,
                scenario.task,
                runtime_scenario(scenario.factors),
                render=False,
            )
            outcome = run_geometry_admission_episode(env, maximum_steps=500)
            outcome.update(
                scenario_id=scenario.scenario_id,
                scenario_seed=scenario.seed,
                morphology=scenario.morphology,
                task=scenario.task,
                active_factor_axes=_active_factor_axes(scenario),
            )
            report["infrastructure_errors"] = [
                value
                for value in report["infrastructure_errors"]
                if value.get("scenario_id") != scenario.scenario_id
            ]
            _append_outcome(report, outcome)
            _write_progress(output, report)
        except BaseException as error:
            report["infrastructure_errors"].append(
                {"scenario_id": scenario.scenario_id, "error": repr(error)}
            )
            _write_progress(output, report)
            raise
        finally:
            if env is not None:
                active_error = sys.exception()
                try:
                    env.close()
                except BaseException as close_error:
                    report["infrastructure_errors"].append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "operation": "close",
                            "error": repr(close_error),
                        }
                    )
                    _write_progress(output, report)
                    if active_error is None:
                        raise
                    active_error.add_note(f"environment close failed: {close_error!r}")
    if report["infrastructure_errors"]:
        raise RuntimeError("geometry admission cannot complete with unresolved infrastructure errors")
    report["aggregate"] = summarize_geometry_admission(report["outcomes"], definition, frozen)
    report["complete"] = True
    report["status"] = (
        "completed_gate_passed"
        if report["aggregate"]["passed"]
        else "completed_gate_failed"
    )
    if not registered:
        report["status"] += "_smoke_unregistered"
    _write_progress(output, report)
    return report


def validate_passing_admission(
    path: Path,
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    *,
    registered: bool = True,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    expected_ids = [scenario.scenario_id for scenario in manifest.scenarios]
    if not registered:
        if (
            report.get("experiment") != 37
            or report.get("mode") != "geometry_admission"
            or report.get("complete") is not True
            or report.get("aggregate", {}).get("passed") is not True
            or report.get("contract", {}).get("definition_sha256") != definition.sha256
            or report.get("contract", {}).get("manifest_sha256") != file_sha256(manifest.path)
        ):
            raise ValueError("smoke policy evaluation requires a complete passing admission report")
        _validate_outcome_chain(report)
        _validate_outcomes(report)
        return report
    if (
        report.get("experiment") != 37
        or report.get("mode") != "geometry_admission"
        or report.get("registered") is not True
        or report.get("complete") is not True
        or report.get("status") != "completed_gate_passed"
        or report.get("expected_outcome_ids") != expected_ids
        or [value.get("scenario_id") for value in report.get("outcomes", [])] != expected_ids
        or report.get("provenance", {}).get("git_dirty") is not False
        or report.get("provenance", {}).get("evaluator_sha256") != file_sha256(Path(__file__))
        or report.get("infrastructure_errors")
        or (
            expected_git_commit is not None
            and report.get("provenance", {}).get("git_commit") != expected_git_commit
        )
    ):
        raise ValueError("registered policy evaluation requires a complete geometry admission report")
    contract = report.get("contract", {})
    _validate_outcome_chain(report)
    _validate_outcomes(report)
    if (
        contract.get("definition_sha256") != definition.sha256
        or contract.get("manifest_sha256") != file_sha256(manifest.path)
        or contract.get("limit_per_cell") is not None
    ):
        raise ValueError("geometry admission report contract does not match experiment 037")
    frozen = load_frozen_admission(definition.path.parent / "reports" / "development-admission.json")
    aggregate = summarize_geometry_admission(report["outcomes"], definition, frozen)
    if aggregate != report.get("aggregate") or not aggregate["passed"]:
        raise ValueError("geometry admission report aggregates or gates do not recompute")
    return report


def resolve_checkpoint(
    summary_path: Path,
    condition: str,
    seed: int,
    morphology: str,
    *,
    root: Path = Path("."),
) -> tuple[Path, dict[str, Any]]:
    if condition not in CONDITIONS or seed not in SEEDS or morphology not in MORPHOLOGIES:
        raise ValueError("checkpoint condition, seed, or morphology is not registered")
    if file_sha256(summary_path) != EXP36_SUMMARY_SHA256:
        raise ValueError("experiment 036 summary hash does not match registration")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("experiment") != 36
        or summary.get("status") != "completed_gate_passed"
        or summary.get("fixed_step") != 1200
    ):
        raise ValueError("experiment 036 summary is not the admitted fixed-step report")
    run = summary.get("runs", {}).get(f"{condition}-seed{seed}")
    if run is None or run.get("model_seed") != seed:
        raise ValueError("experiment 036 summary does not contain the requested run")
    run_root = root / run["output"]
    checkpoint = run_root / "final" / morphology
    required = ("config.json", "core.pt", "embodiment.pt", "trainer.pt")
    if any(not (checkpoint / name).is_file() for name in required):
        raise FileNotFoundError(f"requested morphology checkpoint is incomplete: {checkpoint}")
    core_sha256 = file_sha256(checkpoint / "core.pt")
    if core_sha256 != run.get("core_sha256"):
        raise ValueError("checkpoint core hash does not match the experiment 036 summary")
    trainer = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=True, mmap=True)
    if trainer.get("step") != 1200 or trainer.get("stage") != "core":
        raise ValueError("checkpoint must be the fixed step-1200 core-stage artifact")
    data_manifest_path = run_root / "final" / "data_manifest.json"
    if file_sha256(data_manifest_path) != DATA_MANIFEST_SHA256[condition]:
        raise ValueError("checkpoint final data manifest hash does not match registration")
    data_manifest = json.loads(data_manifest_path.read_text())
    if data_manifest.get("experiment") != 36 or data_manifest.get("condition") != condition:
        raise ValueError("checkpoint final data manifest experiment or condition does not match")
    config_path = checkpoint / "config.json"
    if file_sha256(config_path) != CONFIG_SHA256[morphology]:
        raise ValueError("checkpoint config hash does not match registration")
    config = json.loads(config_path.read_text())
    if (
        config.get("format_version") != 3
        or config.get("backbone_name") != "LiquidAI/LFM2.5-VL-450M"
        or config.get("backbone_revision")
        != "fc6221ca597f3315e4f82fc2df606783267b34ba"
        or config.get("camera_names") != ["top"]
        or config.get("image_do_rescale") is not False
        or config.get("chunk_size") != 32
        or config.get("n_action_steps") != 32
        or config.get("expert_width") != 512
        or config.get("expert_layers") != 12
        or config.get("expert_heads") != 8
        or config.get("expert_ff_multiplier") != 4
        or config.get("dropout") != 0.0
        or config.get("lora_rank") != 16
        or config.get("lora_alpha") != 16
        or config.get("lora_dropout") != 0.0
        or config.get("core_objective") != "regression"
        or config.get("flow_first_step_weight") != 10.0
        or config.get("canonical_channel_loss_weights") != [1.0] * 10
        or config.get("freeze_vision_encoder") is not True
        or config.get("freeze_backbone") is not False
        or len(config.get("parts", [])) != 1
        or config["parts"][0].get("name") != "primary_effector"
        or config["parts"][0].get("kind") != "pose_scalar"
        or config.get("state_dim") != (6 if morphology == "so101" else 8)
        or config.get("action_dim") != (6 if morphology == "so101" else 8)
    ):
        raise ValueError("checkpoint does not implement the registered top/regression/32 geometry core")
    identity = {
        "condition": condition,
        "model_seed": seed,
        "morphology": morphology,
        "checkpoint_path": str(checkpoint),
        "core_sha256": core_sha256,
        "config_sha256": file_sha256(config_path),
        "embodiment_sha256": file_sha256(checkpoint / "embodiment.pt"),
        "trainer_sha256": file_sha256(checkpoint / "trainer.pt"),
        "data_manifest_sha256": file_sha256(data_manifest_path),
        "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
        "step": 1200,
        "stage": "core",
    }
    return checkpoint, identity


def load_geometry_policy(checkpoint: Path, device: str) -> EmbodiPolicy:
    config = EmbodiConfig.load(checkpoint / "config.json")
    policy = EmbodiPolicy(config).to(device)
    policy.load_core_checkpoint(checkpoint)
    policy.eval()
    policy.reset()
    return policy


def geometry_policy_batch(env: Any, policy: Any, processor: Any, device: str) -> dict[str, Any]:
    frame = env.render_top()
    image = camera_frame_to_tensor(
        frame,
        processor_rescales=policy.config.image_do_rescale,
    )
    canonical_state = torch.from_numpy(np.asarray(env.canonical_state(), dtype=np.float32)).unsqueeze(0)
    batch = processor(
        [[image]],
        [env.instruction],
        None,
        canonical_state=canonical_state,
        canonical_part_mask=torch.ones((1, canonical_state.shape[1]), dtype=torch.bool),
    )
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def predict_geometry_chunk(
    env: Any,
    policy: Any,
    processor: Any,
    device: str,
) -> np.ndarray:
    batch = geometry_policy_batch(env, policy, processor, device)
    value = policy.predict_canonical_action_chunk(batch)
    chunk = value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if chunk.shape != (1, 32, 1, 10):
        raise ValueError("canonical policy output must have shape [1, 32, 1, 10]")
    return chunk[0]


def run_geometry_policy_episode(
    env: Any,
    policy: Any,
    processor: Any,
    device: str,
    *,
    maximum_steps: int = 500,
    execution_horizon: int = 16,
) -> dict[str, Any]:
    policy.reset()
    steps = 0
    chunks = 0
    commands = 0
    clipped = 0
    maximum_clipping_error = 0.0
    status = None
    failure_reason = None
    extrema: dict[str, dict[str, float]] = {}
    while steps < maximum_steps and failure_reason is None:
        chunk = predict_geometry_chunk(env, policy, processor, device)
        chunks += 1
        if not np.isfinite(chunk).all():
            failure_reason = "policy_nonfinite_canonical_action"
            break
        native_chunk = []
        remaining = min(execution_horizon, maximum_steps - steps)
        for canonical in chunk[:remaining]:
            try:
                native = np.asarray(env.native_action_from_canonical(canonical), dtype=np.float32)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                failure_reason = "ik_invalid_action"
                break
            if native.shape != (env.morphology.native_action_dim,):
                failure_reason = "ik_invalid_action"
                break
            if not np.isfinite(native).all():
                failure_reason = "ik_nonfinite_native_action"
                break
            native_chunk.append(native)
        if failure_reason is not None:
            break
        for native in native_chunk:
            was_clipped, clipping_error = _clipping(env, native)
            commands += 1
            clipped += int(was_clipped)
            maximum_clipping_error = max(maximum_clipping_error, clipping_error)
            env.apply_action(native)
            status = env.step_control_period(30.0)
            steps += 1
            _diagnostic_extrema_update(extrema, env.diagnostic_state())
            if status.terminal:
                failure_reason = status.failure_reason
                break
        if status is not None and status.terminal:
            break
    success = bool(status is not None and status.success)
    if not success and failure_reason is None:
        failure_reason = "maximum_episode_steps" if steps >= maximum_steps else "task_terminal"
    return {
        "success": success,
        "lifted": bool(env.lifted),
        "steps": steps,
        "chunks": chunks,
        "final_stage": None if status is None else status.stage,
        "failure_reason": failure_reason,
        "commands": commands,
        "clipped_commands": clipped,
        "command_clipping_rate": clipped / commands if commands else 0.0,
        "maximum_clipping_error_native": maximum_clipping_error,
        "diagnostic_extrema": extrema,
    }


def _shard_aggregate(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {}
    for task in TASKS:
        values = [value for value in outcomes if value["task"] == task]
        successes = sum(bool(value["success"]) for value in values)
        cells[task] = {
            "attempted": len(values),
            "successes": successes,
            "success_rate": successes / len(values) if values else 0.0,
        }
    commands = sum(int(value["commands"]) for value in outcomes)
    clipped = sum(int(value["clipped_commands"]) for value in outcomes)
    return {
        "cells": cells,
        "four_task_macro_success": sum(value["success_rate"] for value in cells.values())
        / len(TASKS),
        "lifts": sum(bool(value["lifted"]) for value in outcomes),
        "mean_steps": sum(int(value["steps"]) for value in outcomes) / len(outcomes),
        "failure_reasons": dict(
            Counter(
                "success" if value["success"] else str(value["failure_reason"])
                for value in outcomes
            )
        ),
        "final_stages": dict(Counter(str(value["final_stage"]) for value in outcomes)),
        "commands": commands,
        "clipped_commands": clipped,
        "command_clipping_rate": clipped / commands if commands else 0.0,
        "maximum_clipping_error_native": max(
            (float(value["maximum_clipping_error_native"]) for value in outcomes),
            default=0.0,
        ),
    }


def evaluate_policy_shard(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    admission_path: Path,
    summary_path: Path,
    output: Path,
    *,
    condition: str,
    seed: int,
    morphology: str,
    device: str,
    registered: bool,
    limit_per_cell: int | None = None,
    root: Path = Path("."),
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
    policy_loader: Callable[[Path, str], Any] = load_geometry_policy,
    processor_factory: Callable[..., Any] = EmbodiProcessor.from_pretrained,
) -> dict[str, Any]:
    provenance = runtime_provenance(require_clean=False)
    if registered and provenance["git_dirty"]:
        raise RuntimeError("registered experiment 037 must run from a clean git worktree")
    provenance["evaluator_sha256"] = file_sha256(Path(__file__))
    admission = validate_passing_admission(
        admission_path,
        definition,
        manifest,
        registered=registered,
        expected_git_commit=provenance["git_commit"] if registered else None,
    )
    checkpoint, checkpoint_identity = resolve_checkpoint(
        summary_path, condition, seed, morphology, root=root
    )
    scenarios = select_scenarios(
        manifest,
        morphology=morphology,
        limit_per_cell=limit_per_cell,
    )
    contract = {
        "definition_sha256": definition.sha256,
        "manifest_sha256": file_sha256(manifest.path),
        "admission_report_sha256": file_sha256(admission_path),
        "admission_gate_passed": admission["aggregate"]["passed"],
        "admission_evaluator_git_commit": admission["provenance"]["git_commit"],
        "split": "development",
        "limit_per_cell": limit_per_cell,
        "protocol": {
            "camera": "top",
            "resolution": [640, 480],
            "state_path": "forward_kinematics_canonical",
            "native_state": None,
            "predictor": "predict_canonical_action_chunk",
            "action_path": "native_action_from_canonical",
            "chunk_steps": 32,
            "execution_horizon_steps": 16,
            "frequency_hz": 30,
            "maximum_episode_steps": 500,
            "objective": "regression",
        },
    }
    expected = _new_report(
        mode="policy_shard",
        registered=registered,
        contract=contract,
        identity=checkpoint_identity,
        expected_ids=[scenario.scenario_id for scenario in scenarios],
        provenance=provenance,
    )
    report = _load_or_create_report(output, expected)
    if report.get("complete"):
        aggregate = _shard_aggregate(report["outcomes"])
        if aggregate != report.get("aggregate"):
            raise ValueError("completed policy shard aggregate does not recompute")
        return report
    try:
        policy = policy_loader(checkpoint, device)
        processor = processor_factory(
            policy.config.backbone_name,
            revision=policy.config.backbone_revision,
            do_rescale=policy.config.image_do_rescale,
        )
        report["infrastructure_errors"] = [
            value for value in report["infrastructure_errors"] if value.get("scenario_id") is not None
        ]
    except BaseException as error:
        report["infrastructure_errors"].append(
            {"scenario_id": None, "operation": "policy_setup", "error": repr(error)}
        )
        _write_progress(output, report)
        raise
    completed = len(report["outcomes"])
    for scenario in scenarios[completed:]:
        env = None
        try:
            env = env_factory(
                morphology,
                scenario.task,
                runtime_scenario(scenario.factors),
                width=640,
                height=480,
                render=True,
            )
            outcome = run_geometry_policy_episode(env, policy, processor, device)
            outcome.update(
                scenario_id=scenario.scenario_id,
                scenario_seed=scenario.seed,
                morphology=morphology,
                task=scenario.task,
                active_factor_axes=_active_factor_axes(scenario),
            )
            report["infrastructure_errors"] = [
                value
                for value in report["infrastructure_errors"]
                if value.get("scenario_id") != scenario.scenario_id
            ]
            _append_outcome(report, outcome)
            _write_progress(output, report)
        except BaseException as error:
            report["infrastructure_errors"].append(
                {"scenario_id": scenario.scenario_id, "error": repr(error)}
            )
            _write_progress(output, report)
            raise
        finally:
            if env is not None:
                active_error = sys.exception()
                try:
                    env.close()
                except BaseException as close_error:
                    report["infrastructure_errors"].append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "operation": "close",
                            "error": repr(close_error),
                        }
                    )
                    _write_progress(output, report)
                    if active_error is None:
                        raise
                    active_error.add_note(f"environment close failed: {close_error!r}")
    if report["infrastructure_errors"]:
        raise RuntimeError("policy shard cannot complete with unresolved infrastructure errors")
    report["aggregate"] = _shard_aggregate(report["outcomes"])
    report["complete"] = True
    report["status"] = "completed" if registered else "completed_smoke_unregistered"
    _write_progress(output, report)
    return report


def registered_shard_filename(condition: str, seed: int, morphology: str) -> str:
    return f"benchmark-exp37-{condition}-seed{seed}-{morphology}-geometry.json"


def _expected_scenario_ids(manifest: ScenarioManifest, morphology: str) -> list[str]:
    return [
        scenario.scenario_id
        for scenario in manifest.scenarios
        if scenario.morphology == morphology
    ]


def aggregate_registered_shards(
    reports: list[dict[str, Any]],
    manifest: ScenarioManifest,
    *,
    admission_sha256: str,
    admission_git_commit: str,
    exp36_summary: dict[str, Any],
) -> dict[str, Any]:
    if (
        exp36_summary.get("experiment") != 36
        or exp36_summary.get("status") != "completed_gate_passed"
        or exp36_summary.get("fixed_step") != 1200
    ):
        raise ValueError("registered aggregation requires the admitted Experiment 036 summary")
    indexed = {}
    for report in reports:
        identity = report.get("identity", {})
        key = (identity.get("condition"), identity.get("model_seed"), identity.get("morphology"))
        if key in indexed:
            raise ValueError(f"duplicate registered shard: {key}")
        indexed[key] = report
    expected_keys = {
        (condition, seed, morphology)
        for condition in CONDITIONS
        for seed in SEEDS
        for morphology in MORPHOLOGIES
    }
    missing = sorted(expected_keys - set(indexed))
    extra = sorted(set(indexed) - expected_keys)
    if missing or extra:
        raise ValueError(f"registered aggregation requires all 18 shards; missing={missing}, extra={extra}")
    cell_rates: dict[tuple[str, int, str, str], float] = {}
    clipping_rates: dict[tuple[str, int, str], float] = {}
    shard_summaries: dict[str, Any] = {}
    sources = []
    evaluator_sha256 = file_sha256(Path(__file__))
    git_commits = set()
    for key in sorted(expected_keys):
        condition, seed, morphology = key
        report = indexed[key]
        expected_ids = _expected_scenario_ids(manifest, morphology)
        outcomes = report.get("outcomes", [])
        identity = report.get("identity", {})
        provenance = report.get("provenance", {})
        expected_run = exp36_summary.get("runs", {}).get(f"{condition}-seed{seed}", {})
        if (
            report.get("experiment") != 37
            or report.get("mode") != "policy_shard"
            or report.get("registered") is not True
            or report.get("complete") is not True
            or report.get("status") != "completed"
            or report.get("expected_outcome_ids") != expected_ids
            or [value.get("scenario_id") for value in outcomes] != expected_ids
            or report.get("contract", {}).get("admission_report_sha256") != admission_sha256
            or report.get("contract", {}).get("admission_evaluator_git_commit")
            != admission_git_commit
            or report.get("contract", {}).get("definition_sha256") != DEFINITION_SHA256
            or report.get("contract", {}).get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
            or report.get("contract", {}).get("limit_per_cell") is not None
            or identity.get("core_sha256") != expected_run.get("core_sha256")
            or identity.get("config_sha256") != CONFIG_SHA256[morphology]
            or identity.get("data_manifest_sha256") != DATA_MANIFEST_SHA256[condition]
            or identity.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
            or identity.get("step") != 1200
            or identity.get("stage") != "core"
            or provenance.get("git_dirty") is not False
            or provenance.get("evaluator_sha256") != evaluator_sha256
            or not isinstance(provenance.get("git_commit"), str)
            or report.get("infrastructure_errors")
        ):
            raise ValueError(f"registered shard is incomplete or has the wrong contract: {key}")
        _validate_outcome_chain(report)
        _validate_outcomes(report)
        git_commits.add(provenance["git_commit"])
        outcome_map = {value["scenario_id"]: value for value in outcomes}
        if len(outcome_map) != 200:
            raise ValueError(f"registered shard outcome ids are not unique: {key}")
        for task in TASKS:
            ids = [
                scenario.scenario_id
                for scenario in manifest.scenarios
                if scenario.morphology == morphology and scenario.task == task
            ]
            successes = sum(bool(outcome_map[scenario_id]["success"]) for scenario_id in ids)
            cell_rates[(condition, seed, morphology, task)] = successes / len(ids)
        shard_aggregate = _shard_aggregate(outcomes)
        if report.get("aggregate") != shard_aggregate:
            raise ValueError(f"registered shard aggregate does not recompute: {key}")
        clipping_rates[key] = shard_aggregate["command_clipping_rate"]
        shard_summaries[f"{condition}/seed{seed}/{morphology}"] = shard_aggregate
        sources.append(
            {
                "condition": condition,
                "model_seed": seed,
                "morphology": morphology,
                "core_sha256": report["identity"].get("core_sha256"),
            }
        )
    if len(git_commits) != 1:
        raise ValueError("registered shards must share one evaluator git commit")
    if git_commits != {admission_git_commit}:
        raise ValueError("geometry admission and policy shards must share one evaluator git commit")

    def morphology_macro(condition: str, seed: int, morphology: str) -> float:
        return sum(cell_rates[(condition, seed, morphology, task)] for task in TASKS) / len(TASKS)

    cells = {
        f"{condition}/seed{seed}/{morphology}/{task}": cell_rates[
            (condition, seed, morphology, task)
        ]
        for condition, seed, morphology, task in sorted(cell_rates)
    }
    per_seed = {}
    for seed in SEEDS:
        matched_by_morphology = {
            morphology: morphology_macro(morphology, seed, morphology)
            for morphology in MORPHOLOGIES
        }
        joint_by_morphology = {
            morphology: morphology_macro("joint", seed, morphology)
            for morphology in MORPHOLOGIES
        }
        opposite_by_morphology = {
            "so101": morphology_macro("panda", seed, "so101"),
            "panda": morphology_macro("so101", seed, "panda"),
        }
        matched = sum(matched_by_morphology.values()) / 2
        joint = sum(joint_by_morphology.values()) / 2
        opposite = sum(opposite_by_morphology.values()) / 2
        per_seed[str(seed)] = {
            "matched_specialist": matched,
            "joint": joint,
            "opposite_specialist": opposite,
            "joint_minus_matched_percentage_points": 100.0 * (joint - matched),
            "overall_noninferiority_passed": joint >= matched - 0.05,
            "morphology_macros": {
                morphology: {
                    "matched_specialist": matched_by_morphology[morphology],
                    "joint": joint_by_morphology[morphology],
                    "opposite_specialist": opposite_by_morphology[morphology],
                }
                for morphology in MORPHOLOGIES
            },
        }

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    mean_comparisons: dict[str, Any] = {}
    for endpoint in ("overall", "held_out_stack", "pretraining_tasks", *MORPHOLOGIES):
        if endpoint == "overall":
            matched_values = [per_seed[str(seed)]["matched_specialist"] for seed in SEEDS]
            joint_values = [per_seed[str(seed)]["joint"] for seed in SEEDS]
            opposite_values = [per_seed[str(seed)]["opposite_specialist"] for seed in SEEDS]
        elif endpoint == "held_out_stack":
            matched_values = [
                mean(
                    [
                        cell_rates[(morphology, seed, morphology, "stack_object")]
                        for morphology in MORPHOLOGIES
                    ]
                )
                for seed in SEEDS
            ]
            joint_values = [
                mean(
                    [cell_rates[("joint", seed, morphology, "stack_object")] for morphology in MORPHOLOGIES]
                )
                for seed in SEEDS
            ]
            opposite_values = [
                mean(
                    [
                        cell_rates[("panda", seed, "so101", "stack_object")],
                        cell_rates[("so101", seed, "panda", "stack_object")],
                    ]
                )
                for seed in SEEDS
            ]
        elif endpoint == "pretraining_tasks":
            pretraining_tasks = TASKS[:-1]
            matched_values = [
                mean(
                    [
                        mean(
                            [
                                cell_rates[(morphology, seed, morphology, task)]
                                for task in pretraining_tasks
                            ]
                        )
                        for morphology in MORPHOLOGIES
                    ]
                )
                for seed in SEEDS
            ]
            joint_values = [
                mean(
                    [
                        cell_rates[("joint", seed, morphology, task)]
                        for morphology in MORPHOLOGIES
                        for task in pretraining_tasks
                    ]
                )
                for seed in SEEDS
            ]
            opposite_values = [
                mean(
                    [
                        cell_rates[("panda", seed, "so101", task)]
                        for task in pretraining_tasks
                    ]
                    + [
                        cell_rates[("so101", seed, "panda", task)]
                        for task in pretraining_tasks
                    ]
                )
                for seed in SEEDS
            ]
        else:
            matched_values = [morphology_macro(endpoint, seed, endpoint) for seed in SEEDS]
            joint_values = [morphology_macro("joint", seed, endpoint) for seed in SEEDS]
            opposite_condition = "panda" if endpoint == "so101" else "so101"
            opposite_values = [morphology_macro(opposite_condition, seed, endpoint) for seed in SEEDS]
        matched_mean = mean(matched_values)
        joint_mean = mean(joint_values)
        mean_comparisons[endpoint] = {
            "matched_specialist_mean": matched_mean,
            "joint_mean": joint_mean,
            "opposite_specialist_mean": mean(opposite_values),
            "joint_minus_matched_percentage_points": 100.0 * (joint_mean - matched_mean),
            "noninferiority_passed": joint_mean >= matched_mean - 0.05,
        }
    seed_passes = sum(value["overall_noninferiority_passed"] for value in per_seed.values())
    gates = {
        "noninferiority_margin_percentage_points": 5.0,
        "minimum_matched_overall_success": 0.1,
        "minimum_joint_overall_success": 0.1,
        "minimum_joint_pretraining_task_success": 0.1,
        "minimum_joint_held_out_stack_success": 0.05,
        "minimum_joint_morphology_success": 0.05,
        "matched_overall_assay_sensitive": mean_comparisons["overall"][
            "matched_specialist_mean"
        ]
        >= 0.1,
        "joint_overall_absolute": mean_comparisons["overall"]["joint_mean"] >= 0.1,
        "joint_pretraining_tasks_absolute": mean_comparisons["pretraining_tasks"]["joint_mean"]
        >= 0.1,
        "joint_held_out_stack_absolute": mean_comparisons["held_out_stack"]["joint_mean"]
        >= 0.05,
        "joint_so101_absolute": mean_comparisons["so101"]["joint_mean"] >= 0.05,
        "joint_panda_absolute": mean_comparisons["panda"]["joint_mean"] >= 0.05,
        "overall_eight_cell_three_seed_mean": mean_comparisons["overall"][
            "noninferiority_passed"
        ],
        "held_out_stack_three_seed_mean": mean_comparisons["held_out_stack"][
            "noninferiority_passed"
        ],
        "so101_four_task_three_seed_mean": mean_comparisons["so101"]["noninferiority_passed"],
        "panda_four_task_three_seed_mean": mean_comparisons["panda"]["noninferiority_passed"],
        "overall_seed_passes": seed_passes,
        "minimum_overall_seed_passes": 2,
        "maximum_policy_command_clipping_rate": 0.01,
        "policy_command_clipping": all(rate <= 0.01 for rate in clipping_rates.values()),
    }
    gates["passed"] = bool(
        gates["overall_eight_cell_three_seed_mean"]
        and gates["matched_overall_assay_sensitive"]
        and gates["joint_overall_absolute"]
        and gates["joint_pretraining_tasks_absolute"]
        and gates["joint_held_out_stack_absolute"]
        and gates["joint_so101_absolute"]
        and gates["joint_panda_absolute"]
        and gates["held_out_stack_three_seed_mean"]
        and gates["so101_four_task_three_seed_mean"]
        and gates["panda_four_task_three_seed_mean"]
        and gates["policy_command_clipping"]
        and seed_passes >= 2
    )
    return {
        "format_version": 1,
        "experiment": 37,
        "registered": True,
        "status": "completed_gate_passed" if gates["passed"] else "completed_gate_failed",
        "scope": (
            "shared-capacity closed-loop geometry control; not a learned wrapper/decoder "
            "or representation-invariance test"
        ),
        "contract": {
            "definition_sha256": DEFINITION_SHA256,
            "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "admission_report_sha256": admission_sha256,
            "admission_evaluator_git_commit": admission_git_commit,
            "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
            "evaluator_sha256": evaluator_sha256,
            "evaluator_git_commit": next(iter(git_commits)),
            "split": "development",
            "shards": 18,
            "scenarios_per_shard": 200,
        },
        "sources": sources,
        "cell_success_rates": cells,
        "shard_summaries": shard_summaries,
        "seed_comparisons": per_seed,
        "three_seed_comparisons": mean_comparisons,
        "gates": gates,
        "opposite_specialists_are_descriptive_only": True,
    }


def _registered_output(condition: str, seed: int, morphology: str, output: Path | None) -> Path:
    expected = Path("reports") / registered_shard_filename(condition, seed, morphology)
    if output is not None and output.name != expected.name:
        raise ValueError(f"registered shard output filename must be {expected.name}")
    return output or expected


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="evaluate experiment 037 geometry control")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--definition", type=Path, default=Path("benchmarks/sim-v1/definition.json")
    )
    common.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/sim-v1/manifests/development.json")
    )

    admission = subparsers.add_parser("admission", parents=[common])
    admission.add_argument(
        "--frozen-admission",
        type=Path,
        default=Path("benchmarks/sim-v1/reports/development-admission.json"),
    )
    admission.add_argument("--output", type=Path)
    admission.add_argument("--smoke", action="store_true")
    admission.add_argument("--limit-per-cell", type=_positive)

    shard = subparsers.add_parser("shard", parents=[common])
    shard.add_argument("--condition", choices=CONDITIONS, required=True)
    shard.add_argument("--seed", choices=SEEDS, type=int, required=True)
    shard.add_argument("--morphology", choices=MORPHOLOGIES, required=True)
    shard.add_argument(
        "--admission",
        type=Path,
        default=Path("reports/benchmark-exp37-geometry-admission.json"),
    )
    shard.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    shard.add_argument("--output", type=Path)
    shard.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("."),
        help="workspace root containing the registered Experiment 036 outputs",
    )
    shard.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    shard.add_argument("--smoke", action="store_true")
    shard.add_argument("--limit-per-cell", type=_positive)

    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument(
        "--admission",
        type=Path,
        default=Path("reports/benchmark-exp37-geometry-admission.json"),
    )
    aggregate.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    aggregate.add_argument("--shard-dir", type=Path, default=Path("reports"))
    aggregate.add_argument(
        "--output", type=Path, default=Path("reports/benchmark-exp37-summary.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registered = not getattr(args, "smoke", False)
    if not registered and args.output is None:
        raise ValueError("smoke mode requires an explicit custom --output")
    if registered and getattr(args, "limit_per_cell", None) is not None:
        raise ValueError("--limit-per-cell is only available in smoke mode")
    definition, manifest = load_development_contract(
        args.definition,
        args.manifest,
        registered=registered,
    )
    if args.command == "admission":
        frozen = load_frozen_admission(args.frozen_admission)
        output = args.output or Path("reports/benchmark-exp37-geometry-admission.json")
        if registered and output.name != "benchmark-exp37-geometry-admission.json":
            raise ValueError(
                "registered admission output filename must be benchmark-exp37-geometry-admission.json"
            )
        report = evaluate_geometry_admission(
            definition,
            manifest,
            frozen,
            output,
            registered=registered,
            limit_per_cell=args.limit_per_cell,
        )
        print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))
        if not report["aggregate"]["passed"]:
            raise SystemExit(1)
        return
    if args.command == "shard":
        output = (
            args.output
            if not registered
            else _registered_output(args.condition, args.seed, args.morphology, args.output)
        )
        report = evaluate_policy_shard(
            definition,
            manifest,
            args.admission,
            args.exp36_summary,
            output,
            condition=args.condition,
            seed=args.seed,
            morphology=args.morphology,
            device=args.device,
            registered=registered,
            limit_per_cell=args.limit_per_cell,
            root=args.artifact_root,
        )
        print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))
        return
    admission = validate_passing_admission(args.admission, definition, manifest)
    admission_sha256 = file_sha256(args.admission)
    if file_sha256(args.exp36_summary) != EXP36_SUMMARY_SHA256:
        raise ValueError("experiment 036 summary hash does not match registration")
    exp36_summary = json.loads(args.exp36_summary.read_text())
    reports = [
        json.loads(
            (
                args.shard_dir / registered_shard_filename(condition, seed, morphology)
            ).read_text()
        )
        for condition in CONDITIONS
        for seed in SEEDS
        for morphology in MORPHOLOGIES
    ]
    summary = aggregate_registered_shards(
        reports,
        manifest,
        admission_sha256=admission_sha256,
        admission_git_commit=admission["provenance"]["git_commit"],
        exp36_summary=exp36_summary,
    )
    if not admission["aggregate"]["passed"]:
        raise ValueError("geometry admission gate must pass before aggregation")
    _write_final(args.output, summary)
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
