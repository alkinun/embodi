from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import numpy as np
import torch

from ..joint_train import runtime_provenance
from ..processing import EmbodiProcessor
from ..record_so101 import atomic_write_json
from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv, runtime_scenario
from .benchmark_expert import PrivilegedBenchmarkExpert
from .benchmark_geometry_evaluation import (
    CONFIG_SHA256,
    DATA_MANIFEST_SHA256,
    EXP36_SUMMARY_SHA256,
    _shard_aggregate as _exp37_shard_aggregate,
    _clipping,
    _rotation_matrix,
    load_development_contract,
    load_geometry_policy,
    predict_geometry_chunk,
    resolve_checkpoint,
    rotation_error_degrees,
)


DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
DEVELOPMENT_MANIFEST_SHA256 = (
    "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
)
EXP37_SUMMARY_SHA256 = "7efde348c16863399167794636d9941360a144b565919bd176f181cca1e15aa4"
EXP37_EVALUATOR_SHA256 = "3e8754887ba54034e2e3e2fb7db0eadc95302a33ca59d2fde3cf37133f3f2253"
CONDITIONS = ("joint", "matched")
MORPHOLOGIES = ("so101", "panda")
TASKS = ("push_to_zone", "lift_object", "pick_place_bin")
SEEDS = (36001, 36002, 36003)
HORIZONS = (1, 4, 16)
SCENARIO_INDICES = tuple(range(7))
EMPTY_OUTCOME_CHAIN_SHA256 = hashlib.sha256(b"").hexdigest()
REFERENCE_THRESHOLDS = {
    "translation_m": 0.001,
    "rotation_deg": 1.0,
    "gripper_native": 1.0,
}
DIAGNOSTIC_PROTOCOL = {
    "camera": "top",
    "resolution": [640, 480],
    "state_path": "forward_kinematics_canonical",
    "native_state": None,
    "predictor": "predict_canonical_action_chunk",
    "chunk_steps": 32,
    "decode_all_commands_before_stepping": True,
    "frequency_hz": 30,
    "maximum_episode_steps": 500,
    "objective": "regression",
    "shadow_expert": "one phase-stateful call before every executed policy action",
    "counterfactual_rollout": False,
    "online_ik_intervention": False,
}
HISTOGRAM_EDGES = {
    "translation_m": (0.0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 7.5e-4, 1e-3, 1.25e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 1.0),
    "rotation_deg": (0.0, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 180.0),
    "gripper_native": (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 5.0, 10.0, 20.0, 35.0),
}


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


def compact_accumulator() -> dict[str, int | float]:
    return {"count": 0, "sum": 0.0, "sum_squared": 0.0, "maximum": 0.0}


def update_compact(accumulator: dict[str, int | float], value: float) -> None:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("diagnostic measurements must be finite and nonnegative")
    accumulator["count"] = int(accumulator["count"]) + 1
    accumulator["sum"] = float(accumulator["sum"]) + value
    accumulator["sum_squared"] = float(accumulator["sum_squared"]) + value * value
    accumulator["maximum"] = max(float(accumulator["maximum"]), value)


def merge_compact(values: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    merged = compact_accumulator()
    for value in values:
        merged["count"] = int(merged["count"]) + int(value["count"])
        merged["sum"] = float(merged["sum"]) + float(value["sum"])
        merged["sum_squared"] = float(merged["sum_squared"]) + float(value["sum_squared"])
        merged["maximum"] = max(float(merged["maximum"]), float(value["maximum"]))
    return merged


def _validate_compact(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"count", "sum", "sum_squared", "maximum"}:
        raise ValueError("compact accumulator schema is invalid")
    count = value["count"]
    numbers = [value[name] for name in ("sum", "sum_squared", "maximum")]
    if not isinstance(count, int) or count < 0 or not np.isfinite(numbers).all() or min(numbers) < 0:
        raise ValueError("compact accumulator values are invalid")
    if count == 0 and numbers != [0.0, 0.0, 0.0]:
        raise ValueError("empty compact accumulators must be zero")
    if count > 0 and (
        float(value["maximum"]) > float(value["sum"]) + 1e-12
        or float(value["sum"]) > count * float(value["maximum"]) + 1e-12
        or float(value["sum_squared"]) + 1e-12
        < float(value["sum"]) * float(value["sum"]) / count
        or float(value["sum_squared"]) > float(value["maximum"]) * float(value["sum"]) + 1e-12
    ):
        raise ValueError("compact accumulator statistics are inconsistent")


def _lead_accumulators(size: int) -> dict[str, dict[str, int | float]]:
    return {str(index): compact_accumulator() for index in range(size)}


def _histograms() -> dict[str, dict[str, Any]]:
    return {
        name: {"upper_edges": list(edges), "counts": [0] * len(edges), "overflow_count": 0}
        for name, edges in HISTOGRAM_EDGES.items()
    }


def _histogram_update(histogram: dict[str, Any], value: float) -> None:
    index = int(np.searchsorted(histogram["upper_edges"], value, side="left"))
    if index == len(histogram["counts"]):
        histogram["overflow_count"] += 1
    else:
        histogram["counts"][index] += 1


def _histogram_p95_upper_bound(histogram: dict[str, Any]) -> float | None:
    total = sum(histogram["counts"]) + histogram["overflow_count"]
    if total == 0:
        return 0.0
    target = int(np.ceil(0.95 * total))
    cumulative = 0
    for edge, count in zip(histogram["upper_edges"], histogram["counts"], strict=True):
        cumulative += count
        if cumulative >= target:
            return float(edge)
    if cumulative + histogram["overflow_count"] >= target:
        return None
    raise ValueError("histogram does not contain its recorded samples")


def _merge_histograms(histograms: Iterable[dict[str, Any]]) -> dict[str, Any]:
    histograms = list(histograms)
    if not histograms:
        raise ValueError("cannot merge an empty histogram collection")
    edges = histograms[0]["upper_edges"]
    if any(histogram["upper_edges"] != edges for histogram in histograms):
        raise ValueError("diagnostic histogram edges do not match")
    return {
        "upper_edges": list(edges),
        "counts": [
            sum(histogram["counts"][index] for histogram in histograms)
            for index in range(len(edges))
        ],
        "overflow_count": sum(histogram["overflow_count"] for histogram in histograms),
    }


def _histogram_summary(histogram: dict[str, Any], threshold: float) -> dict[str, Any]:
    p95 = _histogram_p95_upper_bound(histogram)
    overflow = p95 is None
    return {
        "p95_upper_bound": p95,
        "p95_overflow": overflow,
        "reference_threshold": threshold,
        "p95_exceeds_reference": overflow or p95 > threshold,
    }


def _summarize_metric_group(
    outcomes: Iterable[dict[str, Any]], section: str
) -> dict[str, Any]:
    outcomes = list(outcomes)
    by_lead: dict[str, Any] = {}
    overall: dict[str, Any] = {}
    for name in REFERENCE_THRESHOLDS:
        lead_ids = sorted(
            outcomes[0]["chunk_consistency"][section][name], key=int
        ) if outcomes else []
        by_lead[name] = {
            lead: merge_compact(
                outcome["chunk_consistency"][section][name][lead] for outcome in outcomes
            )
            for lead in lead_ids
        }
        overall[name] = merge_compact(by_lead[name].values())
    return {"by_lead": by_lead, "overall": overall}


def _summarize_policy_reconstruction(outcomes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    outcomes = list(outcomes)
    overall = {}
    by_lead = {}
    for name in REFERENCE_THRESHOLDS:
        histogram = _merge_histograms(
            outcome["policy_reconstruction"]["distribution"][name] for outcome in outcomes
        )
        by_lead[name] = {
            str(lead): merge_compact(
                outcome["policy_reconstruction"]["by_lead"][name][str(lead)]
                for outcome in outcomes
            )
            for lead in range(32)
        }
        overall[name] = {
            **merge_compact(by_lead[name].values()),
            **_histogram_summary(histogram, REFERENCE_THRESHOLDS[name]),
        }
    return {"overall": overall, "by_lead": by_lead}


def _summarize_native_clipping(outcomes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    outcomes = list(outcomes)
    error_by_lead = {
        str(lead): merge_compact(
            outcome["policy_reconstruction"]["clipping_error_native_by_lead"][str(lead)]
            for outcome in outcomes
        )
        for lead in range(32)
    }
    clipped_by_lead = {
        str(lead): merge_compact(
            outcome["policy_reconstruction"]["clipped_by_lead"][str(lead)]
            for outcome in outcomes
        )
        for lead in range(32)
    }
    clipped = merge_compact(clipped_by_lead.values())
    return {
        "clipping_error_native_by_lead": error_by_lead,
        "clipped_by_lead": clipped_by_lead,
        "decoded_commands": clipped["count"],
        "clipped_commands": clipped["sum"],
        "clipping_rate": clipped["sum"] / clipped["count"] if clipped["count"] else 0.0,
        "maximum_clipping_error_native": max(
            (value["maximum"] for value in error_by_lead.values()), default=0.0
        ),
    }


def _metric_group(size: int) -> dict[str, dict[str, dict[str, int | float]]]:
    return {
        name: _lead_accumulators(size)
        for name in ("translation_m", "rotation_deg", "gripper_native")
    }


def diagnostic_scenarios(
    manifest: ScenarioManifest,
    morphology: str,
    *,
    registered: bool,
    limit_scenarios: int | None = None,
) -> tuple[Scenario, ...]:
    selected = tuple(
        scenario
        for scenario in manifest.scenarios
        if scenario.morphology == morphology
        and scenario.task in TASKS
        and int(scenario.scenario_id.rsplit("-", 1)[-1]) in SCENARIO_INDICES
    )
    if registered:
        expected_ids = [
            f"development-{morphology}-{task}-{index:04d}"
            for task in TASKS
            for index in SCENARIO_INDICES
        ]
        if [scenario.scenario_id for scenario in selected] != expected_ids:
            raise ValueError("registered experiment 038 requires exact scenario indices 0000 through 0006")
        for task in TASKS:
            axes = []
            for scenario in selected:
                if scenario.task != task:
                    continue
                active = [
                    name
                    for name, value in scenario.factors.items()
                    if str(value.get("bin", "")).startswith("development:")
                ]
                if len(active) != 1:
                    raise ValueError("each diagnostic scenario must activate exactly one factor axis")
                axes.extend(active)
            if len(axes) != 7 or len(set(axes)) != 7:
                raise ValueError("each diagnostic cell must cover the seven active factor axes")
    if limit_scenarios is not None:
        selected = selected[:limit_scenarios]
    return selected


def outcome_id(scenario_id: str, horizon: int) -> str:
    return f"{scenario_id}@h{horizon}"


def expected_outcome_ids(scenarios: Iterable[Scenario]) -> list[str]:
    return [outcome_id(scenario.scenario_id, horizon) for scenario in scenarios for horizon in HORIZONS]


def compose_absolute_targets(
    anchor: np.ndarray,
    chunk: np.ndarray,
    *,
    gripper_native_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    anchor = np.asarray(anchor, dtype=np.float64)
    chunk = np.asarray(chunk, dtype=np.float64)
    if anchor.shape == (1, 10):
        anchor = anchor[0]
    if chunk.shape == (32, 1, 10):
        chunk = chunk[:, 0]
    if anchor.shape != (10,) or chunk.shape != (32, 10):
        raise ValueError("absolute target composition requires one anchor and a 32-command chunk")
    if not np.isfinite(anchor).all() or not np.isfinite(chunk).all():
        raise ValueError("absolute target composition requires finite values")
    anchor_rotation = _rotation_matrix(anchor[3:9])
    rotations = np.stack([anchor_rotation @ _rotation_matrix(command[3:9]) for command in chunk])
    return {
        "translation_m": anchor[:3] + chunk[:, :3],
        "rotation": rotations,
        "gripper_native": chunk[:, 9].copy() * gripper_native_scale,
    }


def _rotation_matrix_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def accumulate_chunk_consistency(
    adjacent: dict[str, dict[str, dict[str, int | float]]],
    overlap: dict[str, dict[str, dict[str, int | float]]],
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray] | None,
    horizon: int,
) -> None:
    for lead in range(31):
        update_compact(
            adjacent["translation_m"][str(lead)],
            np.linalg.norm(current["translation_m"][lead + 1] - current["translation_m"][lead]),
        )
        update_compact(
            adjacent["rotation_deg"][str(lead)],
            _rotation_matrix_error_degrees(current["rotation"][lead], current["rotation"][lead + 1]),
        )
        update_compact(
            adjacent["gripper_native"][str(lead)],
            abs(current["gripper_native"][lead + 1] - current["gripper_native"][lead]),
        )
    if previous is None:
        return
    for lead in range(32 - horizon):
        old_index = horizon + lead
        update_compact(
            overlap["translation_m"][str(lead)],
            np.linalg.norm(previous["translation_m"][old_index] - current["translation_m"][lead]),
        )
        update_compact(
            overlap["rotation_deg"][str(lead)],
            _rotation_matrix_error_degrees(previous["rotation"][old_index], current["rotation"][lead]),
        )
        update_compact(
            overlap["gripper_native"][str(lead)],
            abs(previous["gripper_native"][old_index] - current["gripper_native"][lead]),
        )


def _milestone_names(task: str) -> tuple[str, ...]:
    if task == "push_to_zone":
        return ("near_ee_object", "object_in_target_tolerance", "settled", "success")
    if task == "lift_object":
        return ("near_ee_object", "gripper_closed", "lifted", "lift_predicate", "success")
    return (
        "near_ee_object",
        "gripper_closed",
        "lifted",
        "lift_predicate",
        "object_in_target_tolerance",
        "settled",
        "gripper_released",
        "success",
    )


def _update_milestones(
    milestones: dict[str, dict[str, int | bool | None]],
    env: Any,
    diagnostics: dict[str, float],
    status: Any,
    step: int,
) -> None:
    release_threshold = 0.4 if env.morphology.id == "so101" else 0.7
    values = {
        "near_ee_object": diagnostics["ee_object_distance_m"] < 0.06,
        "object_in_target_tolerance": (
            diagnostics["object_target_xy_distance_m"] <= env.scenario.target_tolerance
        ),
        "settled": diagnostics["object_speed"] < 0.05,
        "gripper_closed": diagnostics["gripper_opening_fraction"] < release_threshold,
        "gripper_released": diagnostics["gripper_opening_fraction"] >= release_threshold,
        "lifted": diagnostics["object_height_m"] >= 0.07,
        "lift_predicate": (
            diagnostics["object_height_m"] >= 0.08
            and diagnostics["ee_object_distance_m"] < 0.06
            and diagnostics["gripper_opening_fraction"] < release_threshold
        ),
        "success": bool(status.success),
    }
    for name in milestones:
        if values[name] and not milestones[name]["attained"]:
            milestones[name] = {"attained": True, "first_step": step}


def _canonical_error(
    first: np.ndarray,
    second: np.ndarray,
    *,
    gripper_native_scale: float,
) -> tuple[float, float, float]:
    first = np.asarray(first, dtype=np.float64).reshape(-1, 10)[0]
    second = np.asarray(second, dtype=np.float64).reshape(-1, 10)[0]
    return (
        float(np.linalg.norm(first[:3] - second[:3])),
        rotation_error_degrees(first[3:9], second[3:9]),
        float(abs(first[9] - second[9])) * gripper_native_scale,
    )


def run_diagnostic_episode(
    env: Any,
    policy: Any,
    processor: Any,
    device: str,
    *,
    execution_horizon: int,
    maximum_steps: int = 500,
    expert_factory: Callable[[Any], Any] = PrivilegedBenchmarkExpert,
) -> dict[str, Any]:
    if execution_horizon not in HORIZONS:
        raise ValueError("experiment 038 execution horizon must be 1, 4, or 16")
    policy.reset()
    expert = expert_factory(env)
    gripper_native_scale = 35.0 if env.morphology.id == "so101" else 0.08
    reconstruction = _metric_group(32)
    reconstruction_distribution = _histograms()
    reconstruction_clipping = _lead_accumulators(32)
    reconstruction_clipped = _lead_accumulators(32)
    adjacent = _metric_group(31)
    overlap = _metric_group(32 - execution_horizon)
    shadow = {phase: _metric_group(1) for phase in ("initial", "policy_visited")}
    milestones: dict[str, dict[str, int | bool | None]] = {
        name: {"attained": False, "first_step": None} for name in _milestone_names(env.task.id)
    }
    steps = chunks = commands = clipped = 0
    maximum_clipping_error = 0.0
    status = None
    failure_reason = None
    previous_targets = None
    expert_terminal_step = None
    shadow_calls = 0
    while steps < maximum_steps and failure_reason is None:
        anchor = np.asarray(env.canonical_state(), dtype=np.float32).copy()
        chunk = predict_geometry_chunk(env, policy, processor, device)
        chunks += 1
        if not np.isfinite(chunk).all():
            failure_reason = "policy_nonfinite_canonical_action"
            break
        targets = compose_absolute_targets(
            anchor, chunk, gripper_native_scale=gripper_native_scale
        )
        accumulate_chunk_consistency(adjacent, overlap, targets, previous_targets, execution_horizon)
        previous_targets = targets
        native_chunk = []
        # All commands are decoded and reconstructed against this unchanged observation anchor.
        for lead, canonical in enumerate(chunk):
            try:
                native = np.asarray(env.native_action_from_canonical(canonical), dtype=np.float32)
                realized = np.asarray(env.canonical_arm_action(native), dtype=np.float32)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                failure_reason = "ik_invalid_action"
                break
            if native.shape != (env.morphology.native_action_dim,) or not np.isfinite(native).all():
                failure_reason = "ik_nonfinite_native_action"
                break
            translation, rotation, gripper = _canonical_error(
                canonical, realized, gripper_native_scale=gripper_native_scale
            )
            for name, value in zip(
                ("translation_m", "rotation_deg", "gripper_native"),
                (translation, rotation, gripper),
                strict=True,
            ):
                update_compact(reconstruction[name][str(lead)], value)
                _histogram_update(reconstruction_distribution[name], value)
            was_clipped, clipping_error = _clipping(env, native)
            update_compact(reconstruction_clipping[str(lead)], clipping_error)
            update_compact(reconstruction_clipped[str(lead)], float(was_clipped))
            native_chunk.append(native)
        if failure_reason is not None:
            break
        remaining = min(execution_horizon, maximum_steps - steps)
        for lead, native in enumerate(native_chunk[:remaining]):
            expert_was_terminal = bool(expert.terminal)
            expert_native = np.asarray(expert.action(), dtype=np.float32)
            shadow_calls += 1
            if expert_terminal_step is None and not expert_was_terminal and expert.terminal:
                expert_terminal_step = steps + 1
            if lead == 0 and not expert_was_terminal:
                expert_canonical = np.asarray(env.canonical_arm_action(expert_native), dtype=np.float32)
                translation, rotation, gripper = _canonical_error(
                    chunk[0], expert_canonical, gripper_native_scale=gripper_native_scale
                )
                phase = "initial" if steps == 0 else "policy_visited"
                for name, value in zip(
                    ("translation_m", "rotation_deg", "gripper_native"),
                    (translation, rotation, gripper),
                    strict=True,
                ):
                    update_compact(shadow[phase][name]["0"], value)
            was_clipped, clipping_error = _clipping(env, native)
            commands += 1
            clipped += int(was_clipped)
            maximum_clipping_error = max(maximum_clipping_error, clipping_error)
            env.apply_action(native)
            status = env.step_control_period(30.0)
            steps += 1
            diagnostics = env.diagnostic_state()
            _update_milestones(milestones, env, diagnostics, status, steps)
            if status.terminal:
                failure_reason = status.failure_reason
                break
        if status is not None and status.terminal:
            break
    success = bool(status is not None and status.success)
    if not success and failure_reason is None:
        failure_reason = "maximum_episode_steps" if steps >= maximum_steps else "task_terminal"
    return {
        "execution_horizon": execution_horizon,
        "success": success,
        "steps": steps,
        "chunks": chunks,
        "final_stage": None if status is None else status.stage,
        "failure_reason": failure_reason,
        "commands": commands,
        "clipped_commands": clipped,
        "command_clipping_rate": clipped / commands if commands else 0.0,
        "maximum_clipping_error_native": maximum_clipping_error,
        "policy_reconstruction": {
            "by_lead": reconstruction,
            "distribution": reconstruction_distribution,
            "clipping_error_native_by_lead": reconstruction_clipping,
            "clipped_by_lead": reconstruction_clipped,
        },
        "chunk_consistency": {"adjacent": adjacent, "cross_replan_overlap": overlap},
        "shadow_expert": {
            "reference": "phase-stateful reference on the live policy trajectory; not a unique oracle action",
            "calls": shadow_calls,
            "errors": shadow,
            "expert_terminal_step": expert_terminal_step,
        },
        "milestones": milestones,
    }


def _validate_metric_group(group: Any, expected_size: int) -> None:
    if not isinstance(group, dict) or set(group) != {
        "translation_m",
        "rotation_deg",
        "gripper_native",
    }:
        raise ValueError("diagnostic metric group schema is invalid")
    expected_leads = {str(index) for index in range(expected_size)}
    for values in group.values():
        if set(values) != expected_leads:
            raise ValueError("diagnostic lead ids do not match the contract")
        for value in values.values():
            _validate_compact(value)


def _validate_histograms(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(HISTOGRAM_EDGES):
        raise ValueError("diagnostic histogram schema is invalid")
    for name, histogram in value.items():
        if (
            histogram.get("upper_edges") != list(HISTOGRAM_EDGES[name])
            or len(histogram.get("counts", [])) != len(HISTOGRAM_EDGES[name])
            or any(not isinstance(count, int) or count < 0 for count in histogram["counts"])
            or not isinstance(histogram.get("overflow_count"), int)
            or histogram["overflow_count"] < 0
        ):
            raise ValueError("diagnostic histogram values are invalid")


def _validate_outcomes(report: dict[str, Any]) -> None:
    for outcome in report.get("outcomes", []):
        if (
            not isinstance(outcome.get("outcome_id"), str)
            or not isinstance(outcome.get("scenario_id"), str)
            or outcome.get("execution_horizon") not in HORIZONS
            or not isinstance(outcome.get("success"), bool)
        ):
            raise ValueError("diagnostic outcome identity schema is invalid")
        if outcome["outcome_id"] != outcome_id(outcome["scenario_id"], outcome["execution_horizon"]):
            raise ValueError("diagnostic outcome id does not match scenario and horizon")
        steps = outcome.get("steps")
        commands = outcome.get("commands")
        chunks = outcome.get("chunks")
        clipped = outcome.get("clipped_commands")
        if (
            not isinstance(steps, int)
            or not 0 <= steps <= 500
            or commands != steps
            or not isinstance(chunks, int)
            or chunks < 1
            or not isinstance(clipped, int)
            or not 0 <= clipped <= commands
            or outcome.get("command_clipping_rate") != (clipped / commands if commands else 0.0)
        ):
            raise ValueError("diagnostic outcome command counters are invalid")
        reconstruction = outcome.get("policy_reconstruction", {})
        _validate_metric_group(reconstruction.get("by_lead"), 32)
        _validate_histograms(reconstruction.get("distribution"))
        reconstruction_counts = [
            reconstruction["by_lead"]["translation_m"][str(lead)]["count"]
            for lead in range(32)
        ]
        analyzed_chunks = chunks - int(outcome.get("failure_reason") == "policy_nonfinite_canonical_action")
        count_drops = sum(
            first != second
            for first, second in zip(
                reconstruction_counts, reconstruction_counts[1:], strict=False
            )
        )
        ik_failure = outcome.get("failure_reason") in {
            "ik_invalid_action",
            "ik_nonfinite_native_action",
        }
        if (
            (not ik_failure and any(count != analyzed_chunks for count in reconstruction_counts))
            or (
                ik_failure
                and (
                    any(count not in {analyzed_chunks, max(analyzed_chunks - 1, 0)} for count in reconstruction_counts)
                    or count_drops > 1
                    or any(
                        first < second
                        for first, second in zip(
                            reconstruction_counts, reconstruction_counts[1:], strict=False
                        )
                    )
                )
            )
        ):
            raise ValueError("policy reconstruction counts do not match chunk decoding")
        for name in REFERENCE_THRESHOLDS:
            if any(
                reconstruction["by_lead"][name][str(lead)]["count"]
                != reconstruction_counts[lead]
                for lead in range(32)
            ):
                raise ValueError("policy reconstruction component counts do not match")
            histogram = reconstruction["distribution"][name]
            if sum(histogram["counts"]) + histogram["overflow_count"] != sum(
                reconstruction_counts
            ):
                raise ValueError("policy reconstruction histogram count does not match")
        for name in ("clipping_error_native_by_lead", "clipped_by_lead"):
            values = reconstruction.get(name, {})
            if set(values) != {str(index) for index in range(32)}:
                raise ValueError("policy reconstruction clipping leads are invalid")
            for lead, value in values.items():
                _validate_compact(value)
                if value["count"] != reconstruction_counts[int(lead)]:
                    raise ValueError("policy reconstruction clipping counts do not match")
                if name == "clipped_by_lead" and (
                    value["maximum"] > 1 or value["sum_squared"] != value["sum"]
                ):
                    raise ValueError("policy reconstruction clipping indicators are invalid")
        _validate_metric_group(outcome.get("chunk_consistency", {}).get("adjacent"), 31)
        _validate_metric_group(
            outcome.get("chunk_consistency", {}).get("cross_replan_overlap"),
            32 - outcome["execution_horizon"],
        )
        for group, expected_count in (
            (outcome["chunk_consistency"]["adjacent"], analyzed_chunks),
            (
                outcome["chunk_consistency"]["cross_replan_overlap"],
                max(analyzed_chunks - 1, 0),
            ),
        ):
            if any(
                accumulator["count"] != expected_count
                for component in group.values()
                for accumulator in component.values()
            ):
                raise ValueError("chunk consistency counts do not match replans")
        shadow = outcome.get("shadow_expert", {})
        if shadow.get("calls") != commands:
            raise ValueError("shadow expert must be called exactly once per executed policy action")
        for phase in ("initial", "policy_visited"):
            _validate_metric_group(shadow.get("errors", {}).get(phase), 1)
        initial_count = shadow["errors"]["initial"]["translation_m"]["0"]["count"]
        visited_count = shadow["errors"]["policy_visited"]["translation_m"]["0"]["count"]
        if initial_count != int(commands > 0) or visited_count > max(chunks - 1, 0):
            raise ValueError("shadow expert phase counts do not match replans")
        if shadow.get("expert_terminal_step") is None and visited_count != max(chunks - 1, 0):
            raise ValueError("nonterminal shadow expert must cover every replan")
        for phase in ("initial", "policy_visited"):
            if len(
                {
                    shadow["errors"][phase][name]["0"]["count"]
                    for name in REFERENCE_THRESHOLDS
                }
            ) != 1:
                raise ValueError("shadow expert component counts do not match")
        task = outcome.get("task")
        if task not in TASKS or outcome.get("morphology") not in MORPHOLOGIES:
            raise ValueError("diagnostic outcome task or morphology is invalid")
        milestones = outcome.get("milestones")
        if not isinstance(milestones, dict) or set(milestones) != set(_milestone_names(task)):
            raise ValueError("diagnostic milestone schema is invalid")
        for milestone in milestones.values():
            if set(milestone) != {"attained", "first_step"} or not isinstance(
                milestone["attained"], bool
            ):
                raise ValueError("diagnostic milestone value is invalid")
            first_step = milestone["first_step"]
            if milestone["attained"] != (type(first_step) is int) or (
                type(first_step) is int and not 1 <= first_step <= steps
            ):
                raise ValueError("diagnostic milestone step is invalid")
        if (outcome["success"] and outcome.get("failure_reason") is not None) or (
            not outcome["success"] and not isinstance(outcome.get("failure_reason"), str)
        ):
            raise ValueError("diagnostic outcome terminal state is invalid")
        maximum_clipping = outcome.get("maximum_clipping_error_native")
        if not isinstance(maximum_clipping, (int, float)) or not np.isfinite(maximum_clipping) or maximum_clipping < 0:
            raise ValueError("diagnostic outcome maximum clipping error is invalid")
        if outcome["execution_horizon"] == 16 and not isinstance(
            outcome.get("exp37_success_reproduced"), bool
        ):
            raise ValueError("H16 outcomes require an Experiment 037 reproduction flag")
        if outcome["execution_horizon"] == 16 and not isinstance(
            outcome.get("exp37_trajectory_reproduced"), bool
        ):
            raise ValueError("H16 outcomes require an Experiment 037 trajectory reproduction flag")


def _new_report(
    *,
    registered: bool,
    contract: dict[str, Any],
    identity: dict[str, Any],
    expected_ids: list[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": 38,
        "mode": "diagnostic_shard",
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
    report: dict[str, Any] = json.loads(path.read_text())
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
    if any(report.get(name) != expected.get(name) for name in immutable):
        raise ValueError("resume report contract, checkpoint, or evaluator identity does not match")
    ids = [outcome.get("outcome_id") for outcome in report.get("outcomes", [])]
    if ids != expected["expected_outcome_ids"][: len(ids)] or len(ids) != len(set(ids)):
        raise ValueError("resume report outcomes are not a unique ordered contract prefix")
    if report.get("outcome_chain_sha256") != _outcome_chain(report.get("outcomes", [])):
        raise ValueError("resume report outcome hash chain does not match")
    if not isinstance(report.get("infrastructure_errors"), list):
        raise ValueError("resume report infrastructure errors are invalid")
    _validate_outcomes(report)
    if report.get("complete") and (
        ids != expected["expected_outcome_ids"] or report.get("infrastructure_errors")
    ):
        raise ValueError("completed diagnostic report is incomplete or has infrastructure errors")
    if not report.get("complete") and (
        report.get("status") != expected["status"] or "aggregate" in report
    ):
        raise ValueError("incomplete diagnostic report has stale derived state")
    return report


def source_condition(condition: str, morphology: str) -> str:
    if condition == "joint":
        return "joint"
    if condition == "matched":
        return morphology
    raise ValueError("experiment 038 condition must be joint or matched")


def registered_shard_filename(condition: str, seed: int, morphology: str) -> str:
    return f"benchmark-exp38-{condition}-seed{seed}-{morphology}-diagnostic.json"


def exp37_shard_filename(condition: str, seed: int, morphology: str) -> str:
    return f"benchmark-exp37-{source_condition(condition, morphology)}-seed{seed}-{morphology}-geometry.json"


def _exp37_trajectory_signature(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        name: outcome.get(name)
        for name in (
            "success",
            "steps",
            "chunks",
            "final_stage",
            "failure_reason",
            "commands",
            "clipped_commands",
        )
    }


def _valid_exp37_outcome(outcome: dict[str, Any], morphology: str) -> bool:
    steps = outcome.get("steps")
    commands = outcome.get("commands")
    chunks = outcome.get("chunks")
    clipped = outcome.get("clipped_commands")
    success = outcome.get("success")
    return (
        isinstance(success, bool)
        and isinstance(steps, int)
        and 0 <= steps <= 500
        and commands == steps
        and isinstance(chunks, int)
        and chunks >= 1
        and isinstance(clipped, int)
        and 0 <= clipped <= commands
        and outcome.get("morphology") == morphology
        and outcome.get("task") in (*TASKS, "stack_object")
        and (not success or outcome.get("failure_reason") is None)
        and (success or isinstance(outcome.get("failure_reason"), str))
    )


def _validate_exp37_source(
    report: dict[str, Any],
    expected_condition: str,
    seed: int,
    morphology: str,
    expected_core_sha256: str,
) -> None:
    identity = report.get("identity", {})
    expected_ids = [
        f"development-{morphology}-{task}-{index:04d}"
        for task in (*TASKS, "stack_object")
        for index in range(50)
    ]
    outcomes = report.get("outcomes", [])
    outcome_ids = [outcome.get("scenario_id") for outcome in outcomes]
    contract = report.get("contract", {})
    provenance = report.get("provenance", {})
    if (
        report.get("experiment") != 37
        or report.get("mode") != "policy_shard"
        or report.get("registered") is not True
        or report.get("complete") is not True
        or report.get("status") != "completed"
        or report.get("infrastructure_errors")
        or identity.get("condition") != expected_condition
        or identity.get("model_seed") != seed
        or identity.get("morphology") != morphology
        or identity.get("config_sha256") != CONFIG_SHA256[morphology]
        or identity.get("data_manifest_sha256") != DATA_MANIFEST_SHA256[expected_condition]
        or identity.get("core_sha256") != expected_core_sha256
        or identity.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
        or identity.get("step") != 1200
        or identity.get("stage") != "core"
        or contract.get("definition_sha256") != DEFINITION_SHA256
        or contract.get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
        or contract.get("split") != "development"
        or contract.get("limit_per_cell") is not None
        or contract.get("protocol", {}).get("execution_horizon_steps") != 16
        or provenance.get("evaluator_sha256") != EXP37_EVALUATOR_SHA256
        or provenance.get("git_dirty") is not False
        or report.get("expected_outcome_ids") != expected_ids
        or outcome_ids != expected_ids
        or len(outcome_ids) != len(set(outcome_ids))
        or report.get("outcome_chain_sha256") != _outcome_chain(outcomes)
        or any(not _valid_exp37_outcome(outcome, morphology) for outcome in outcomes)
        or report.get("aggregate") != _exp37_shard_aggregate(outcomes)
    ):
        raise ValueError("Experiment 037 source shard is incomplete or has the wrong identity")


def _load_exp37_source(
    path: Path,
    expected_condition: str,
    seed: int,
    morphology: str,
    expected_core_sha256: str,
) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    _validate_exp37_source(
        report, expected_condition, seed, morphology, expected_core_sha256
    )
    return report


def validate_h16_reproduction(
    outcomes: Iterable[dict[str, Any]], source_outcomes: dict[str, dict[str, Any]]
) -> None:
    for outcome in outcomes:
        if outcome["execution_horizon"] != 16:
            continue
        source = source_outcomes.get(outcome["scenario_id"])
        expected = None if source is None else _exp37_trajectory_signature(source)
        if (
            expected is None
            or _exp37_trajectory_signature(outcome) != expected
            or outcome.get("exp37_expected_trajectory") != expected
            or outcome.get("exp37_expected_success") != expected["success"]
            or outcome.get("exp37_success_reproduced") is not True
            or outcome.get("exp37_trajectory_reproduced") is not True
        ):
            raise ValueError("H16 does not exactly reproduce Experiment 037")


def _shard_aggregate(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for task in TASKS:
        for horizon in HORIZONS:
            values = [
                value
                for value in outcomes
                if value["task"] == task and value["execution_horizon"] == horizon
            ]
            cells[f"{task}/h{horizon}"] = {
                "attempted": len(values),
                "successes": sum(value["success"] for value in values),
                "success_rate": (
                    sum(value["success"] for value in values) / len(values) if values else 0.0
                ),
                "milestone_attainment": {
                    name: sum(value["milestones"].get(name, {}).get("attained", False) for value in values)
                    / len(values)
                    for name in _milestone_names(task)
                }
                if values
                else {},
            }
    reconstruction = _summarize_policy_reconstruction(outcomes)
    chunk_consistency = {
        f"h{horizon}": {
            section: _summarize_metric_group(
                [value for value in outcomes if value["execution_horizon"] == horizon], section
            )
            for section in ("adjacent", "cross_replan_overlap")
        }
        for horizon in HORIZONS
    }
    return {
        "cells": cells,
        "episodes": len(outcomes),
        "successes": sum(value["success"] for value in outcomes),
        "h16_reproduced": all(
            value.get("exp37_trajectory_reproduced") is True
            for value in outcomes
            if value["execution_horizon"] == 16
        ),
        "policy_reconstruction": reconstruction["overall"],
        "policy_reconstruction_by_lead": reconstruction["by_lead"],
        "native_clipping": _summarize_native_clipping(outcomes),
        "chunk_consistency": chunk_consistency,
    }


def evaluate_diagnostic_shard(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    exp36_summary_path: Path,
    exp37_summary_path: Path,
    exp37_report_dir: Path,
    output: Path,
    *,
    condition: str,
    seed: int,
    morphology: str,
    device: str,
    registered: bool,
    limit_scenarios: int | None = None,
    root: Path = Path("."),
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
    policy_loader: Callable[[Path, str], Any] = load_geometry_policy,
    processor_factory: Callable[..., Any] = EmbodiProcessor.from_pretrained,
) -> dict[str, Any]:
    provenance = runtime_provenance(require_clean=False)
    if registered and provenance["git_dirty"]:
        raise RuntimeError("registered experiment 038 must run from a clean git worktree")
    provenance["evaluator_sha256"] = file_sha256(Path(__file__))
    if file_sha256(exp37_summary_path) != EXP37_SUMMARY_SHA256:
        raise ValueError("experiment 037 summary hash does not match registration")
    exp37_summary = json.loads(exp37_summary_path.read_text())
    if exp37_summary.get("experiment") != 37 or exp37_summary.get("status") != "completed_gate_failed":
        raise ValueError("experiment 037 summary is not the frozen failed evaluation")
    checkpoint_condition = source_condition(condition, morphology)
    checkpoint, identity = resolve_checkpoint(
        exp36_summary_path, checkpoint_condition, seed, morphology, root=root
    )
    identity["condition"] = condition
    identity["source_condition"] = checkpoint_condition
    scenarios = diagnostic_scenarios(
        manifest,
        morphology,
        registered=registered,
        limit_scenarios=limit_scenarios,
    )
    source_path = exp37_report_dir / exp37_shard_filename(condition, seed, morphology)
    source = _load_exp37_source(
        source_path, checkpoint_condition, seed, morphology, identity["core_sha256"]
    )
    source_map = {value["scenario_id"]: value for value in source["outcomes"]}
    if any(scenario.scenario_id not in source_map for scenario in scenarios):
        raise ValueError("Experiment 037 source shard is missing a diagnostic scenario")
    contract = {
        "definition_sha256": definition.sha256,
        "manifest_sha256": file_sha256(manifest.path),
        "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
        "exp37_summary_sha256": EXP37_SUMMARY_SHA256,
        "exp37_source_report_sha256": file_sha256(source_path),
        "source_condition": checkpoint_condition,
        "split": "development",
        "scenario_indices": list(SCENARIO_INDICES),
        "tasks": list(TASKS),
        "horizons": list(HORIZONS),
        "limit_scenarios": limit_scenarios,
        "protocol": DIAGNOSTIC_PROTOCOL,
    }
    expected = _new_report(
        registered=registered,
        contract=contract,
        identity=identity,
        expected_ids=expected_outcome_ids(scenarios),
        provenance=provenance,
    )
    report = _load_or_create_report(output, expected)
    if report.get("complete"):
        validate_h16_reproduction(report["outcomes"], source_map)
        aggregate = _shard_aggregate(report["outcomes"])
        if aggregate != report.get("aggregate"):
            raise ValueError("completed diagnostic shard aggregate does not recompute")
        return report
    try:
        policy = policy_loader(checkpoint, device)
        processor = processor_factory(
            policy.config.backbone_name,
            revision=policy.config.backbone_revision,
            do_rescale=policy.config.image_do_rescale,
        )
        report["infrastructure_errors"] = [
            error for error in report["infrastructure_errors"] if error.get("outcome_id") is not None
        ]
    except BaseException as error:
        report["infrastructure_errors"].append(
            {"outcome_id": None, "operation": "policy_setup", "error": repr(error)}
        )
        _write_progress(output, report)
        raise
    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}
    for identifier in expected["expected_outcome_ids"][len(report["outcomes"]) :]:
        scenario_id, horizon_text = identifier.rsplit("@h", 1)
        scenario = scenario_map[scenario_id]
        horizon = int(horizon_text)
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
            outcome = run_diagnostic_episode(
                env, policy, processor, device, execution_horizon=horizon
            )
            outcome.update(
                outcome_id=identifier,
                scenario_id=scenario.scenario_id,
                scenario_seed=scenario.seed,
                morphology=morphology,
                task=scenario.task,
                active_factor_axes=[
                    name
                    for name, value in scenario.factors.items()
                    if str(value.get("bin", "")).startswith("development:")
                ],
            )
            if horizon == 16:
                expected_trajectory = _exp37_trajectory_signature(source_map[scenario.scenario_id])
                outcome["exp37_expected_success"] = expected_trajectory["success"]
                outcome["exp37_expected_trajectory"] = expected_trajectory
                outcome["exp37_success_reproduced"] = outcome["success"] == expected_trajectory["success"]
                outcome["exp37_trajectory_reproduced"] = (
                    _exp37_trajectory_signature(outcome) == expected_trajectory
                )
                if not outcome["exp37_trajectory_reproduced"]:
                    raise ValueError(
                        f"H16 outcome does not reproduce Experiment 037: {scenario.scenario_id}"
                    )
            report["infrastructure_errors"] = [
                error
                for error in report["infrastructure_errors"]
                if error.get("outcome_id") != identifier
            ]
            closing_env = env
            env = None
            closing_env.close()
            _append_outcome(report, outcome)
            _write_progress(output, report)
        except BaseException as error:
            report["infrastructure_errors"].append(
                {"outcome_id": identifier, "error": repr(error)}
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
                        {"outcome_id": identifier, "operation": "close", "error": repr(close_error)}
                    )
                    _write_progress(output, report)
                    if active_error is None:
                        raise
                    active_error.add_note(f"environment close failed: {close_error!r}")
    if report["infrastructure_errors"]:
        raise RuntimeError("diagnostic shard cannot complete with infrastructure errors")
    report["aggregate"] = _shard_aggregate(report["outcomes"])
    if not report["aggregate"]["h16_reproduced"]:
        raise ValueError("diagnostic shard failed closed on H16 reproduction")
    report["complete"] = True
    report["status"] = "completed" if registered else "completed_smoke_unregistered"
    _write_progress(output, report)
    return report


def classify_horizon_mechanism(seed_values: dict[int, dict[str, float]]) -> dict[str, Any]:
    gains = {
        seed: values["h16_success"] - values["h1_success"] for seed, values in seed_values.items()
    }
    milestone_gains = {
        seed: values["h16_milestones"] - values["h1_milestones"]
        for seed, values in seed_values.items()
    }
    mean_gain = sum(gains.values()) / len(gains)
    nonnegative_seeds = sum(value >= 0 for value in gains.values())
    mean_milestone_gain = sum(milestone_gains.values()) / len(milestone_gains)
    milestone_improves = mean_milestone_gain > 0
    strong = mean_gain + 1e-12 >= 0.05 and nonnegative_seeds >= 2 and milestone_improves
    any_support = mean_gain > 0 or any(value > 0 for value in milestone_gains.values())
    return {
        "classification": "strong" if strong else ("partial" if any_support else "unsupported"),
        "three_seed_mean_paired_success_gain_percentage_points": 100.0 * mean_gain,
        "nonnegative_direction_seeds": nonnegative_seeds,
        "milestone_progression_improves": milestone_improves,
        "three_seed_mean_task_equal_milestone_gain": mean_milestone_gain,
        "per_seed_success_gain_percentage_points": {
            str(seed): 100.0 * gain for seed, gain in sorted(gains.items())
        },
        "materiality_margin_percentage_points": 5.0,
        "materiality_is_not_model_admission": True,
        "milestone_estimand": "task-equal mean outcome milestone fraction",
    }


def classify_distribution_shift(seed_ratios: dict[int, float | str]) -> dict[str, Any]:
    supported = sum(
        value == "infinite" or (isinstance(value, (int, float)) and value >= 2.0)
        for value in seed_ratios.values()
    )
    strong = supported >= 2
    return {
        "classification": "strong_association" if strong else "descriptive_only",
        "late_over_initial_physical_error_ratio": {
            str(seed): value for seed, value in sorted(seed_ratios.items())
        },
        "seeds_at_least_twofold": supported,
        "estimand": "H16 sample-weighted replan means, component-normalized then averaged",
        "causal_claim": False,
    }


def _mean_accumulator(value: dict[str, Any]) -> float:
    return float(value["sum"]) / int(value["count"]) if value["count"] else 0.0


def _task_equal_milestone_progression(outcomes: list[dict[str, Any]]) -> float:
    task_values = []
    for task in TASKS:
        task_outcomes = [outcome for outcome in outcomes if outcome["task"] == task]
        task_values.append(
            sum(
                sum(int(milestone["attained"]) for milestone in outcome["milestones"].values())
                / len(outcome["milestones"])
                for outcome in task_outcomes
            )
            / len(task_outcomes)
        )
    return sum(task_values) / len(task_values)


def _normalized_shadow_error(outcomes: list[dict[str, Any]], phase: str) -> float:
    components = []
    for name, threshold in REFERENCE_THRESHOLDS.items():
        accumulator = merge_compact(
            outcome["shadow_expert"]["errors"][phase][name]["0"] for outcome in outcomes
        )
        if accumulator["count"]:
            components.append(_mean_accumulator(accumulator) / threshold)
    return sum(components) / len(components) if components else 0.0


def _late_initial_ratio(initial: float, late: float) -> float | str:
    if initial > 0:
        return late / initial
    return "infinite" if late > 0 else "undefined"


def aggregate_registered_shards(
    reports: list[dict[str, Any]],
    manifest: ScenarioManifest,
    *,
    source_reports: dict[tuple[str, int, str], tuple[dict[str, Any], str]],
    exp36_summary: dict[str, Any],
) -> dict[str, Any]:
    if (
        exp36_summary.get("experiment") != 36
        or exp36_summary.get("status") != "completed_gate_passed"
        or exp36_summary.get("fixed_step") != 1200
    ):
        raise ValueError("registered aggregation requires the frozen Experiment 036 summary")
    indexed = {}
    for report in reports:
        identity = report.get("identity", {})
        key = (identity.get("condition"), identity.get("model_seed"), identity.get("morphology"))
        if key in indexed:
            raise ValueError(f"duplicate Experiment 038 shard: {key}")
        indexed[key] = report
    expected_keys = {
        (condition, seed, morphology)
        for condition in CONDITIONS
        for seed in SEEDS
        for morphology in MORPHOLOGIES
    }
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise ValueError(f"registered aggregation requires all 12 shards; missing={missing}, extra={extra}")
    evaluator_sha256 = file_sha256(Path(__file__))
    commits = set()
    source_hashes = []
    all_outcomes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    shard_summaries = {}
    for key in sorted(expected_keys):
        condition, seed, morphology = key
        report = indexed[key]
        scenarios = diagnostic_scenarios(manifest, morphology, registered=True)
        ids = expected_outcome_ids(scenarios)
        outcomes = report.get("outcomes", [])
        contract = report.get("contract", {})
        identity = report.get("identity", {})
        provenance = report.get("provenance", {})
        expected_source_condition = source_condition(condition, morphology)
        expected_run = exp36_summary.get("runs", {}).get(
            f"{expected_source_condition}-seed{seed}", {}
        )
        if (
            report.get("experiment") != 38
            or report.get("mode") != "diagnostic_shard"
            or report.get("registered") is not True
            or report.get("complete") is not True
            or report.get("status") != "completed"
            or report.get("expected_outcome_ids") != ids
            or [value.get("outcome_id") for value in outcomes] != ids
            or contract.get("definition_sha256") != DEFINITION_SHA256
            or contract.get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
            or contract.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
            or contract.get("exp37_summary_sha256") != EXP37_SUMMARY_SHA256
            or contract.get("source_condition") != expected_source_condition
            or contract.get("split") != "development"
            or contract.get("scenario_indices") != list(SCENARIO_INDICES)
            or contract.get("tasks") != list(TASKS)
            or contract.get("horizons") != list(HORIZONS)
            or contract.get("limit_scenarios") is not None
            or contract.get("protocol") != DIAGNOSTIC_PROTOCOL
            or identity.get("source_condition") != expected_source_condition
            or identity.get("config_sha256") != CONFIG_SHA256[morphology]
            or identity.get("data_manifest_sha256") != DATA_MANIFEST_SHA256[expected_source_condition]
            or identity.get("core_sha256") != expected_run.get("core_sha256")
            or provenance.get("git_dirty") is not False
            or provenance.get("evaluator_sha256") != evaluator_sha256
            or not isinstance(provenance.get("git_commit"), str)
            or report.get("infrastructure_errors")
        ):
            raise ValueError(f"registered diagnostic shard has the wrong contract: {key}")
        if report.get("outcome_chain_sha256") != _outcome_chain(outcomes):
            raise ValueError(f"registered diagnostic shard hash chain does not match: {key}")
        _validate_outcomes(report)
        for outcome, scenario in zip(outcomes, (scenario for scenario in scenarios for _ in HORIZONS), strict=True):
            expected_axes = [
                name
                for name, value in scenario.factors.items()
                if str(value.get("bin", "")).startswith("development:")
            ]
            if (
                outcome.get("scenario_id") != scenario.scenario_id
                or outcome.get("scenario_seed") != scenario.seed
                or outcome.get("morphology") != morphology
                or outcome.get("task") != scenario.task
                or outcome.get("active_factor_axes") != expected_axes
            ):
                raise ValueError(f"registered diagnostic outcome metadata is invalid: {key}")
        aggregate = _shard_aggregate(outcomes)
        if report.get("aggregate") != aggregate or not aggregate["h16_reproduced"]:
            raise ValueError(f"registered diagnostic shard aggregate does not recompute: {key}")
        source_report, source_hash = source_reports[key]
        if source_hash != contract.get("exp37_source_report_sha256"):
            raise ValueError(f"Experiment 037 source report hash changed: {key}")
        source = source_report
        try:
            _validate_exp37_source(
                source,
                expected_source_condition,
                seed,
                morphology,
                expected_run["core_sha256"],
            )
        except ValueError as error:
            raise ValueError(f"Experiment 037 source report identity is invalid: {key}") from error
        source_map = {value["scenario_id"]: value for value in source["outcomes"]}
        try:
            validate_h16_reproduction(outcomes, source_map)
        except ValueError as error:
            raise ValueError(f"H16 does not exactly reproduce Experiment 037: {key}") from error
        commits.add(provenance["git_commit"])
        source_hashes.append({"key": f"{condition}/seed{seed}/{morphology}", "sha256": source_hash})
        shard_summaries[f"{condition}/seed{seed}/{morphology}"] = aggregate
        all_outcomes[(condition, seed)].extend(outcomes)
    if len(commits) != 1:
        raise ValueError("registered diagnostic shards must share one clean evaluator commit")

    mechanism = {}
    distribution = {}
    milestone_aggregates = {}
    all_values = [outcome for values in all_outcomes.values() for outcome in values]
    geometry = _summarize_policy_reconstruction(all_values)
    geometry_by_condition = {}
    clipping_by_condition = {}
    geometry_classification = {}
    chunk_consistency = {}
    for condition in CONDITIONS:
        condition_values = [
            outcome for seed in SEEDS for outcome in all_outcomes[(condition, seed)]
        ]
        geometry_by_condition[condition] = _summarize_policy_reconstruction(condition_values)
        clipping_by_condition[condition] = _summarize_native_clipping(condition_values)
        geometry_classification[condition] = (
            "threshold_exceedance_evidence"
            if any(
                value["p95_exceeds_reference"]
                for value in geometry_by_condition[condition]["overall"].values()
            )
            else "descriptive_only"
        )
        seed_values = {}
        ratios = {}
        for seed in SEEDS:
            values = all_outcomes[(condition, seed)]
            by_horizon = {
                horizon: [value for value in values if value["execution_horizon"] == horizon]
                for horizon in HORIZONS
            }
            success = {
                horizon: sum(value["success"] for value in outcomes) / len(outcomes)
                for horizon, outcomes in by_horizon.items()
            }
            progression = {
                horizon: _task_equal_milestone_progression(outcomes)
                for horizon, outcomes in by_horizon.items()
            }
            seed_values[seed] = {
                "h1_success": success[1],
                "h16_success": success[16],
                "h1_milestones": progression[1],
                "h16_milestones": progression[16],
            }
            h16_values = by_horizon[16]
            phase_means = {
                phase: _normalized_shadow_error(h16_values, phase)
                for phase in ("initial", "policy_visited")
            }
            ratios[seed] = _late_initial_ratio(
                phase_means["initial"], phase_means["policy_visited"]
            )
        mechanism[condition] = classify_horizon_mechanism(seed_values)
        distribution[condition] = classify_distribution_shift(ratios)
        for horizon in HORIZONS:
            horizon_values = [
                value
                for seed in SEEDS
                for value in all_outcomes[(condition, seed)]
                if value["execution_horizon"] == horizon
            ]
            chunk_consistency[f"{condition}/h{horizon}"] = {
                section: _summarize_metric_group(horizon_values, section)
                for section in ("adjacent", "cross_replan_overlap")
            }
        for seed in SEEDS:
            values = all_outcomes[(condition, seed)]
            for morphology in MORPHOLOGIES:
                for task in TASKS:
                    for horizon in HORIZONS:
                        cell = [
                            value
                            for value in values
                            if value["morphology"] == morphology
                            and value["task"] == task
                            and value["execution_horizon"] == horizon
                        ]
                        milestone_aggregates[
                            f"{condition}/seed{seed}/{morphology}/{task}/h{horizon}"
                        ] = {
                            name: sum(value["milestones"][name]["attained"] for value in cell)
                            / len(cell)
                            for name in _milestone_names(task)
                        }
    return {
        "format_version": 1,
        "experiment": 38,
        "registered": True,
        "status": "completed_diagnostic",
        "scope": "development-only diagnosis; no training, admission, selection, or advancement gate",
        "contract": {
            "definition_sha256": DEFINITION_SHA256,
            "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
            "exp37_summary_sha256": EXP37_SUMMARY_SHA256,
            "evaluator_sha256": evaluator_sha256,
            "evaluator_git_commit": next(iter(commits)),
            "shards": 12,
            "episodes": 756,
            "final_split_loaded": False,
        },
        "source_report_hashes": source_hashes,
        "shard_summaries": shard_summaries,
        "milestone_attainment": milestone_aggregates,
        "horizon_mechanism": mechanism,
        "distribution_shift_association": distribution,
        "chunk_consistency": chunk_consistency,
        "geometry_interpretation": {
            "reference_thresholds": REFERENCE_THRESHOLDS,
            "policy_reconstruction": geometry["overall"],
            "policy_reconstruction_by_lead": geometry["by_lead"],
            "policy_reconstruction_by_condition": geometry_by_condition,
            "native_clipping_by_condition": clipping_by_condition,
            "classification": geometry_classification["joint"],
            "classification_by_condition": geometry_classification,
            "descriptive_unless_policy_reconstruction_p95_exceeds_reference": True,
            "expert_admission_does_not_cover_policy_actions": True,
            "online_ik_intervention": False,
        },
        "advancement_gate": None,
    }


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _registered_output(condition: str, seed: int, morphology: str, output: Path | None) -> Path:
    filename = registered_shard_filename(condition, seed, morphology)
    if output is not None and output.name != filename:
        raise ValueError(f"registered shard output filename must be {filename}")
    return output or Path("reports") / filename


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run Experiment 038 canonical trajectory diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--definition", type=Path, default=Path("benchmarks/sim-v1/definition.json")
    )
    common.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/sim-v1/manifests/development.json")
    )
    shard = subparsers.add_parser("shard", parents=[common])
    shard.add_argument("--condition", choices=CONDITIONS, required=True)
    shard.add_argument("--seed", choices=SEEDS, type=int, required=True)
    shard.add_argument("--morphology", choices=MORPHOLOGIES, required=True)
    shard.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    shard.add_argument(
        "--exp37-summary", type=Path, default=Path("reports/benchmark-exp37-summary.json")
    )
    shard.add_argument("--exp37-report-dir", type=Path, default=Path("reports"))
    shard.add_argument("--artifact-root", type=Path, default=Path("."))
    shard.add_argument("--output", type=Path)
    shard.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    shard.add_argument("--smoke", action="store_true")
    shard.add_argument("--limit-scenarios", type=_positive)
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--shard-dir", type=Path, default=Path("reports"))
    aggregate.add_argument("--exp37-report-dir", type=Path, default=Path("reports"))
    aggregate.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    aggregate.add_argument(
        "--exp37-summary", type=Path, default=Path("reports/benchmark-exp37-summary.json")
    )
    aggregate.add_argument("--output", type=Path, default=Path("reports/benchmark-exp38-summary.json"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    registered = not getattr(args, "smoke", False)
    if not registered and args.output is None:
        raise ValueError("smoke mode requires an explicit custom --output")
    if registered and getattr(args, "limit_scenarios", None) is not None:
        raise ValueError("--limit-scenarios is only available in smoke mode")
    definition, manifest = load_development_contract(
        args.definition, args.manifest, registered=registered
    )
    if definition.sha256 != DEFINITION_SHA256 or file_sha256(manifest.path) != DEVELOPMENT_MANIFEST_SHA256:
        raise ValueError("Experiment 038 frozen benchmark contract does not match")
    if args.command == "shard":
        output = (
            args.output
            if not registered
            else _registered_output(args.condition, args.seed, args.morphology, args.output)
        )
        report = evaluate_diagnostic_shard(
            definition,
            manifest,
            args.exp36_summary,
            args.exp37_summary,
            args.exp37_report_dir,
            output,
            condition=args.condition,
            seed=args.seed,
            morphology=args.morphology,
            device=args.device,
            registered=registered,
            limit_scenarios=args.limit_scenarios,
            root=args.artifact_root,
        )
        print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))
        return
    reports = [
        json.loads((args.shard_dir / registered_shard_filename(condition, seed, morphology)).read_text())
        for condition in CONDITIONS
        for seed in SEEDS
        for morphology in MORPHOLOGIES
    ]
    if file_sha256(args.exp36_summary) != EXP36_SUMMARY_SHA256:
        raise ValueError("experiment 036 summary hash does not match registration")
    if file_sha256(args.exp37_summary) != EXP37_SUMMARY_SHA256:
        raise ValueError("experiment 037 summary hash does not match registration")
    exp36_summary = json.loads(args.exp36_summary.read_text())
    exp37_summary = json.loads(args.exp37_summary.read_text())
    if exp37_summary.get("experiment") != 37 or exp37_summary.get("status") != "completed_gate_failed":
        raise ValueError("experiment 037 summary is not the frozen failed evaluation")
    source_reports = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for morphology in MORPHOLOGIES:
                key = (condition, seed, morphology)
                path = args.exp37_report_dir / exp37_shard_filename(condition, seed, morphology)
                source_reports[key] = (json.loads(path.read_text()), file_sha256(path))
    summary = aggregate_registered_shards(
        reports,
        manifest,
        source_reports=source_reports,
        exp36_summary=exp36_summary,
    )
    _write_final(args.output, summary)
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
