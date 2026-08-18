from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .benchmark import file_sha256
from . import benchmark_gripper_mechanism as exp43
from . import benchmark_phase_agreement as exp42


EXP42_SUMMARY_SHA256 = "104c16ca2552eb0d2ba7efc65c9228eeecd488d67d765f49d70931aa0d15abf8"
EXP43_SUMMARY_SHA256 = "acdf15900cdb708e8f9b96a5c12f2d70be18f951906b92ade8d2a874b69d15a4"
EXP42_EVALUATOR_SHA256 = "cc4fb718a3f6341dbaaea0026aca193fd86d5b84412ea7061204ed8f4ce24b7d"
EXP43_EVALUATOR_SHA256 = "a1056895ab4926aa1d5822ff6ce1bab6d743fc779b3fb1599da26040356e9794"
SOURCE_SHA256 = {
    "exp42": {
        "so101": "f02815631edb583a5faab4a82926afc650b69d5fc131a60227107bc0abecdd51",
        "panda": "55018696519b930d01e859a19365171a5b70d1fb227777b9bf624daecd0732e5",
    },
    "exp43": {
        "so101": "7b2ce863b96ecd873f28cc80a86d8279ec8e58dee3db346d5d82593aee998bb8",
        "panda": "ec3d3534227b86c4bebfa20ceae264f66d86fd1dbaf32cfd3a78818dd20b05ec",
    },
}
EXP43_ARCHIVE_SHA256 = {
    "so101": "72f5767a8ff96dfa6d13237ff47e269af0f3848e7d45ebb9a59d924f313feec0",
    "panda": "4dee6a5b629e112b9867d2766e3b03c0a0c0121568691d5522b0836b0b83db06",
}

SLICES = ("exp42", "exp43")
MORPHOLOGIES = exp43.MORPHOLOGIES
SEEDS = exp43.SEEDS
CONDITIONS_BY_SLICE = {
    "exp42": ("full", "joint"),
    "exp43": ("full", "half", "joint"),
}
SOURCE_CONDITION = {"matched": "full", "full": "full", "half": "half", "joint": "joint"}
CONTRASTS = {
    "J-F": ("joint", "full"),
    "H-F": ("half", "full"),
    "J-H": ("joint", "half"),
}
PARTITIONS = (
    "open_steady",
    "open_openward",
    "closed_steady",
    "closed_closeward",
    "terminal_noop",
)
EXPECTED_SUPPORT = {
    "open_steady": (154, 84),
    "open_openward": (28, 14),
    "closed_steady": (70, 42),
    "closed_closeward": (56, 28),
    "terminal_noop": (56, 42),
}
PARTITION_TASKS = {
    "open_steady": {"push_to_zone", "lift_object", "pick_place_bin"},
    "open_openward": {"pick_place_bin"},
    "closed_steady": {"lift_object", "pick_place_bin"},
    "closed_closeward": {"lift_object", "pick_place_bin"},
    "terminal_noop": {"push_to_zone", "lift_object", "pick_place_bin"},
}
OPEN_NATIVE = {"so101": np.float32(16.0), "panda": np.float32(0.08)}
METRICS = (
    "effective_gripper_error",
    "raw_gripper_absolute_error",
    "signed_gripper_bias",
    "gripper_saturation_rate",
    "normalized_channel_9_loss",
)
SUMMARY_PATH = Path("reports/benchmark-exp44-summary.json")
REGISTRATION_PATH = Path("experiments/044-gripper-target-direction-localization.md")


def _dependency_contract() -> None:
    if (
        file_sha256(Path(exp42.__file__)) != EXP42_EVALUATOR_SHA256
        or file_sha256(Path(exp43.__file__)) != EXP43_EVALUATOR_SHA256
    ):
        raise ValueError("Experiment 044 imported evaluator hash mismatch")


def registered_evaluator_sha256(path: Path = REGISTRATION_PATH) -> str:
    match = re.search(r"Experiment 044 evaluator SHA-256 `([0-9a-f]{64})`", path.read_text())
    if match is None:
        raise ValueError("Experiment 044 registration does not freeze the evaluator hash")
    return match.group(1)


def partition_label(anchor: Mapping[str, Any], morphology: str) -> str:
    current = np.asarray(anchor.get("pre_call_native_state"), dtype=np.float32)
    target = np.asarray(anchor.get("expert_native"), dtype=np.float32)
    if current.shape != target.shape or current.shape not in {(6,), (8,)}:
        raise ValueError("Experiment 044 native gripper arrays are invalid")
    if anchor.get("post_action_phase") == "done":
        if not np.array_equal(current, target):
            raise ValueError("Experiment 044 terminal target is not an exact native no-op")
        return "terminal_noop"
    target_gripper = target[-1]
    if target_gripper == np.float32(0.0):
        state = "closed"
    elif target_gripper == OPEN_NATIVE[morphology]:
        state = "open"
    else:
        raise ValueError("Experiment 044 encountered an unregistered native gripper setpoint")
    phase = anchor.get("phase")
    if phase == "release":
        if state != "open":
            raise ValueError("Experiment 044 release phase does not request the open setpoint")
        direction = "openward"
    elif phase == "close":
        if state != "closed":
            raise ValueError("Experiment 044 close phase does not request the closed setpoint")
        direction = "closeward"
    else:
        direction = "steady"
    return f"{state}_{direction}"


def _source_bytes(path: Path, source_sha256: str, archive_sha256: str | None) -> bytes:
    stored = path.read_bytes()
    if archive_sha256 is None:
        raw = stored
    else:
        if hashlib.sha256(stored).hexdigest() != archive_sha256:
            raise ValueError(f"archive hash mismatch: {path}")
        raw = gzip.decompress(stored)
    if hashlib.sha256(raw).hexdigest() != source_sha256:
        raise ValueError(f"source report hash mismatch: {path}")
    return raw


def _array(anchor: Mapping[str, Any], name: str, dtype: Any) -> NDArray[Any]:
    value = np.asarray(anchor.get(name), dtype=dtype)
    if not np.isfinite(value).all() or anchor.get(f"{name}_sha256") != exp43.canonical_array_sha256(
        value
    ):
        raise ValueError(f"Experiment 044 source {name} array is invalid")
    return value


def _query_output(query: Mapping[str, Any]) -> NDArray[Any]:
    value = np.asarray(query.get("output_chunk"), dtype=np.float32)
    if (
        value.shape != (32, 1, 10)
        or not np.isfinite(value).all()
        or query.get("output_sha256") != exp43.canonical_array_sha256(value)
    ):
        raise ValueError("Experiment 044 source output chunk is invalid")
    return value


def _extract_report_records(
    path: Path,
    *,
    slice_name: str,
    morphology: str,
    source_sha256: str,
    archive_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(_source_bytes(path, source_sha256, archive_sha256))
    source_experiment = 42 if slice_name == "exp42" else 43
    expected_source_conditions = (
        {"matched", "joint"} if slice_name == "exp42" else {"full", "half", "joint"}
    )
    if (
        report.get("experiment") != source_experiment
        or report.get("registered") is not True
        or report.get("complete") is not True
        or report.get("identity", {}).get("morphology") != morphology
        or report.get("aggregate", {}).get("delivery_gate", {}).get("passed") is not True
        or report.get("contract", {}).get("final_split_loaded") is not False
    ):
        raise ValueError("Experiment 044 source shard contract is invalid")
    anchors = report.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 182:
        raise ValueError("Experiment 044 source shard must contain 182 endpoint anchors")
    records: list[dict[str, Any]] = []
    anchor_ids: set[tuple[str, str, str]] = set()
    for ordinal, anchor in enumerate(anchors):
        if anchor.get("ordinal") != ordinal or anchor.get("morphology") != morphology:
            raise ValueError("Experiment 044 anchor identity is invalid")
        current = _array(anchor, "pre_call_native_state", np.float32)
        native_target = _array(anchor, "expert_native", np.float32)
        expert_target = _array(anchor, "expert_target", np.float32)
        expected_native_shape = (6,) if morphology == "so101" else (8,)
        if (
            current.shape != expected_native_shape
            or native_target.shape != expected_native_shape
            or expert_target.shape != (1, 10)
        ):
            raise ValueError("Experiment 044 source expert array shape is invalid")
        partition = partition_label(anchor, morphology)
        canonical_gripper = float(expert_target[0, 9])
        if partition != "terminal_noop":
            expected_canonical = (
                float(np.float32(native_target[-1] / np.float32(35.0)))
                if morphology == "so101"
                else float(np.float32(native_target[-1] / np.float32(0.08)))
            )
            if canonical_gripper != expected_canonical:
                raise ValueError("Experiment 044 canonical and native gripper targets differ")
        identity = (str(anchor.get("scenario_id")), str(anchor.get("phase")), str(anchor.get("endpoint")))
        if identity in anchor_ids:
            raise ValueError("Experiment 044 source contains duplicate endpoint identity")
        anchor_ids.add(identity)
        seed_results = anchor.get("seeds", {})
        if tuple(seed_results) != tuple(str(seed) for seed in SEEDS):
            raise ValueError("Experiment 044 source checkpoint seeds are incomplete")
        for seed in SEEDS:
            condition_results = seed_results[str(seed)]
            if set(condition_results) != expected_source_conditions:
                raise ValueError("Experiment 044 source conditions are incomplete")
            for source_condition, result in condition_results.items():
                condition = SOURCE_CONDITION[source_condition]
                queries = result.get("queries")
                if not isinstance(queries, list) or len(queries) != 2:
                    raise ValueError("Experiment 044 source immediate repeat is missing")
                first = _query_output(queries[0])
                repeat = _query_output(queries[1])
                repeat_difference = float(np.max(np.abs(first - repeat)))
                prediction = first[0, 0]
                recomputed = exp43.action_metrics(prediction, expert_target)
                stored_metrics = result.get("metrics", {})
                if (
                    queries[0].get("repeat") is True
                    or queries[1].get("repeat") is not True
                    or repeat_difference > exp43.REPEAT_MAX_ABS
                    or result.get("repeat_max_abs_difference") != repeat_difference
                    or stored_metrics.get("effective_gripper_error")
                    != recomputed["effective_gripper_error"]
                    or stored_metrics.get("channel_losses", [None] * 10)[9]
                    != recomputed["channel_losses"][9]
                ):
                    raise ValueError("Experiment 044 source gripper metrics do not recompute")
                records.append(
                    {
                        "slice": slice_name,
                        "seed": seed,
                        "condition": condition,
                        "morphology": morphology,
                        "task": anchor["task"],
                        "scenario_id": anchor["scenario_id"],
                        "phase": anchor["phase"],
                        "endpoint": anchor["endpoint"],
                        "anchor_ordinal": ordinal,
                        "partition": partition,
                        "target_opening": canonical_gripper,
                        "metrics": {
                            "effective_gripper_error": recomputed["effective_gripper_error"],
                            "raw_gripper_absolute_error": recomputed[
                                "raw_gripper_absolute_error"
                            ],
                            "signed_gripper_bias": recomputed["signed_gripper_bias"],
                            "gripper_saturation_rate": recomputed["gripper_saturation_rate"],
                            "normalized_channel_9_loss": recomputed["channel_losses"][9],
                        },
                    }
                )
    return records, {
        "path": str(path),
        "source_sha256": source_sha256,
        "archive_sha256": archive_sha256,
        "anchors": len(anchors),
        "records": len(records),
        "delivery_passed": True,
    }


def _metric_vector(record: Mapping[str, Any]) -> NDArray[Any]:
    values = np.asarray([record["metrics"][name] for name in METRICS], dtype=np.float64)
    if values.shape != (len(METRICS),) or not np.isfinite(values).all():
        raise ValueError("Experiment 044 metric vector is invalid")
    return values


def _decimal_mean(values: Sequence[NDArray[Any]]) -> NDArray[Any]:
    if not values:
        raise ValueError("cannot average an empty Experiment 044 cell")
    denominator = Decimal(len(values))
    return np.asarray(
        [
            float(
                sum((Decimal(str(float(value[index]))) for value in values), Decimal(0))
                / denominator
            )
            for index in range(len(METRICS))
        ],
        dtype=np.float64,
    )


def hierarchical_condition_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Experiment 044 hierarchy requires records")
    dimensions = ("seed", "slice", "morphology", "task", "scenario_id", "phase", "endpoint")
    keyed: dict[tuple[Any, ...], list[NDArray[Any]]] = defaultdict(list)
    for record in records:
        keyed[tuple(record[name] for name in dimensions)].append(_metric_vector(record))
    level = {key: _decimal_mean(values) for key, values in keyed.items()}
    for _ in dimensions:
        grouped: dict[tuple[Any, ...], list[NDArray[Any]]] = defaultdict(list)
        for key, value in level.items():
            grouped[key[:-1]].append(value)
        level = {key: _decimal_mean(values) for key, values in grouped.items()}
    if set(level) != {()}:
        raise RuntimeError("Experiment 044 hierarchy did not collapse to one cell")
    value = level[()]
    return {
        **{name: float(value[index]) for index, name in enumerate(METRICS)},
        "records": len(records),
    }


def condition_contrasts(
    records: Sequence[dict[str, Any]], conditions: Sequence[str]
) -> dict[str, Any]:
    if set(record["condition"] for record in records) != set(conditions):
        raise ValueError("Experiment 044 contrast cell has incomplete conditions")
    cells = {
        condition: hierarchical_condition_metrics(
            [record for record in records if record["condition"] == condition]
        )
        for condition in conditions
    }
    contrasts = {
        name: pair for name, pair in CONTRASTS.items() if set(pair) <= set(conditions)
    }
    result: dict[str, Any] = {}
    for metric in METRICS:
        values = {condition: Decimal(str(cells[condition][metric])) for condition in conditions}
        result[metric] = {
            **{condition: float(values[condition]) for condition in conditions},
            "contrasts": {
                name: {
                    "difference": float(values[numerator] - values[denominator]),
                    "difference_decimal": str(values[numerator] - values[denominator]),
                    "ratio": (
                        float(values[numerator] / values[denominator])
                        if values[denominator] != 0
                        else None
                    ),
                    "ratio_decimal": (
                        str(values[numerator] / values[denominator])
                        if values[denominator] != 0
                        else None
                    ),
                }
                for name, (numerator, denominator) in contrasts.items()
            },
        }
    result["records_per_condition"] = {
        condition: cells[condition]["records"] for condition in conditions
    }
    if {"J-F", "H-F", "J-H"} <= set(contrasts):
        for metric in METRICS:
            metric_contrasts = result[metric]["contrasts"]
            if Decimal(metric_contrasts["J-F"]["difference_decimal"]) != Decimal(
                metric_contrasts["H-F"]["difference_decimal"]
            ) + Decimal(metric_contrasts["J-H"]["difference_decimal"]):
                raise RuntimeError("Experiment 044 three-condition decomposition is inexact")
    return result


def _grouped_contrasts(
    records: Sequence[dict[str, Any]], conditions: Sequence[str], key: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return {
        name: condition_contrasts(values, conditions)
        for name, values in sorted(grouped.items())
    }


def _support(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    anchors = {
        (
            record["slice"],
            record["morphology"],
            record["scenario_id"],
            record["phase"],
            record["endpoint"],
        )
        for record in records
    }
    phase_units = {value[:-1] for value in anchors}
    return {
        "endpoint_anchors": len(anchors),
        "scenario_phase_units": len(phase_units),
        "contrast_observations": len(anchors) * len(SEEDS),
        "by_slice": {
            slice_name: {
                "endpoint_anchors": sum(value[0] == slice_name for value in anchors),
                "scenario_phase_units": sum(value[0] == slice_name for value in phase_units),
            }
            for slice_name in SLICES
        },
    }


def _metric_contrast(cell: Mapping[str, Any], contrast: str) -> Mapping[str, Any]:
    return cell["effective_gripper_error"]["contrasts"][contrast]


def _material_gate(view: Mapping[str, Any], contrast: str = "J-F") -> dict[str, Any]:
    overall = _metric_contrast(view["overall"], contrast)
    seed_ratios = {
        seed: _metric_contrast(cell, contrast)["ratio_decimal"]
        for seed, cell in view["by_seed"].items()
    }
    difference_decimal = Decimal(overall["difference_decimal"])
    ratio_decimal = (
        Decimal(overall["ratio_decimal"]) if overall["ratio_decimal"] is not None else None
    )
    checks = {
        "difference_at_least_0_01": difference_decimal >= Decimal("0.01"),
        "ratio_at_least_1_10": ratio_decimal is not None and ratio_decimal >= Decimal("1.10"),
        "at_least_two_seed_ratios_at_least_1_10": sum(
            ratio is not None and Decimal(ratio) >= Decimal("1.10")
            for ratio in seed_ratios.values()
        )
        >= 2,
    }
    return {
        "supported": all(checks.values()),
        "checks": checks,
        "difference": overall["difference"],
        "ratio": overall["ratio"],
        "seed_ratios": {
            seed: float(Decimal(ratio)) if ratio is not None else None
            for seed, ratio in seed_ratios.items()
        },
    }


def _view(records: Sequence[dict[str, Any]], conditions: Sequence[str]) -> dict[str, Any]:
    return {
        "overall": condition_contrasts(records, conditions),
        "by_seed": _grouped_contrasts(records, conditions, "seed"),
        "by_slice": _grouped_contrasts(records, conditions, "slice"),
        "by_morphology": _grouped_contrasts(records, conditions, "morphology"),
        "by_task": _grouped_contrasts(records, conditions, "task"),
        "by_endpoint": _grouped_contrasts(records, conditions, "endpoint"),
    }


def _difference_of_differences(
    left: Mapping[str, Any], right: Mapping[str, Any], contrast: str = "J-F"
) -> dict[str, Any]:
    left_value = Decimal(_metric_contrast(left, contrast)["difference_decimal"])
    right_value = Decimal(_metric_contrast(right, contrast)["difference_decimal"])
    difference = left_value - right_value
    return {
        "left_difference": float(left_value),
        "right_difference": float(right_value),
        "difference": float(difference),
        "difference_decimal": str(difference),
    }


def localization_contrast(
    left_records: Sequence[dict[str, Any]],
    right_records: Sequence[dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
    expected_tasks: set[str],
) -> dict[str, Any]:
    conditions = ("full", "joint")
    expected_base = {
        (seed, slice_name, morphology, task)
        for seed in SEEDS
        for slice_name in SLICES
        for morphology in MORPHOLOGIES
        for task in expected_tasks
    }
    scenario_support: list[dict[tuple[Any, ...], set[str]]] = []
    for side in (left_records, right_records):
        base = {
            (record["seed"], record["slice"], record["morphology"], record["task"])
            for record in side
        }
        endpoint_conditions: dict[tuple[Any, ...], set[str]] = defaultdict(set)
        scenarios_by_base: dict[tuple[Any, ...], set[str]] = defaultdict(set)
        for record in side:
            base_key = (
                record["seed"],
                record["slice"],
                record["morphology"],
                record["task"],
            )
            scenarios_by_base[base_key].add(str(record["scenario_id"]))
            endpoint_conditions[
                (
                    record["seed"],
                    record["slice"],
                    record["morphology"],
                    record["task"],
                    record["scenario_id"],
                    record["phase"],
                    record["endpoint"],
                )
            ].add(record["condition"])
        if (
            base != expected_base
            or set(record["endpoint"] for record in side) != {"first", "last"}
            or set(scenarios_by_base) != expected_base
            or any(len(scenarios) != 7 for scenarios in scenarios_by_base.values())
            or any(values != set(conditions) for values in endpoint_conditions.values())
        ):
            raise ValueError("Experiment 044 localization side lacks registered common support")
        scenario_support.append(dict(scenarios_by_base))
    if scenario_support[0] != scenario_support[1]:
        raise ValueError("Experiment 044 localization sides use different registered scenarios")
    left = _view(left_records, conditions)
    right = _view(right_records, conditions)
    subgroup_differences: dict[str, Any] = {}
    expected_groups = {
        "by_seed": {str(seed) for seed in SEEDS},
        "by_slice": set(SLICES),
        "by_morphology": set(MORPHOLOGIES),
        "by_task": expected_tasks,
        "by_endpoint": {"first", "last"},
    }
    for key, expected in expected_groups.items():
        common = set(left[key]) & set(right[key])
        if common != set(left[key]) or common != set(right[key]) or common != expected:
            raise ValueError("Experiment 044 localization sides have unequal subgroup support")
        subgroup_differences[key] = {
            name: _difference_of_differences(left[key][name], right[key][name])
            for name in sorted(common)
        }
    return {
        "left": left_label,
        "right": right_label,
        "left_view": left,
        "right_view": right,
        "overall": _difference_of_differences(left["overall"], right["overall"]),
        **subgroup_differences,
    }


def _nonnegative_jf(view: Mapping[str, Any]) -> bool:
    return all(
        Decimal(_metric_contrast(cell, "J-F")["difference_decimal"]) >= 0
        for group in ("by_slice", "by_morphology", "by_task", "by_endpoint")
        for cell in view[group].values()
    )


def _localization_gate(name: str, contrast: Mapping[str, Any]) -> dict[str, Any]:
    difference = Decimal(contrast["overall"]["difference_decimal"])
    direction = 1 if difference >= 0 else -1
    winner_key = "left_view" if direction > 0 else "right_view"
    winner = contrast["left"] if direction > 0 else contrast["right"]
    seed_values = [Decimal(value["difference_decimal"]) for value in contrast["by_seed"].values()]
    slice_values = [
        Decimal(value["difference_decimal"]) for value in contrast["by_slice"].values()
    ]
    morphology_values = [
        Decimal(value["difference_decimal"]) for value in contrast["by_morphology"].values()
    ]
    def same_direction(value: Decimal) -> bool:
        return value * direction > 0

    material = _material_gate(contrast[winner_key])
    checks = {
        "winning_cell_material": material["supported"],
        "absolute_interaction_at_least_0_01": abs(difference) >= Decimal("0.01"),
        "at_least_two_seed_interactions_same_direction": sum(
            same_direction(value) for value in seed_values
        )
        >= 2,
        "both_slice_interactions_same_direction": all(
            same_direction(value) for value in slice_values
        ),
    }
    localized = all(checks.values())
    broad = (
        localized
        and all(same_direction(value) for value in morphology_values)
        and _nonnegative_jf(contrast[winner_key])
        and name != "opening_transition"
    )
    if name == "opening_transition":
        label = f"pick-place-specific-{winner}-localization"
    else:
        label = f"{winner}-localized-{'broad' if broad else 'narrow'}"
    return {
        "localized": localized,
        "broad": broad,
        "classification": label if localized else None,
        "winner": winner,
        "winning_cell_materiality": material,
        "checks": checks,
    }


def classify_localization(
    partitions: Mapping[str, Any], localization: Mapping[str, Any], *, delivery: bool
) -> dict[str, Any]:
    if not delivery:
        return {"classification": "invalid_delivery", "delivery_gate_passed": False}
    partition_gates = {
        name: _material_gate(cell["pooled_jf"]) for name, cell in partitions.items()
    }
    localization_gates = {
        name: _localization_gate(name, contrast) for name, contrast in localization.items()
    }
    labels = [
        gate["classification"]
        for gate in localization_gates.values()
        if gate["localized"]
    ]
    any_material_partition = any(gate["supported"] for gate in partition_gates.values())
    if not any_material_partition:
        classification: Any = "not_materially_localized"
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


def _source_summary(path: Path, expected_sha256: str, experiment: int) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"Experiment {experiment} summary hash mismatch")
    value = json.loads(path.read_text())
    if (
        value.get("experiment") != experiment
        or value.get("registered") is not True
        or value.get("delivery_gate_passed") is not True
        or value.get("final_split_loaded") is not False
        or value.get("contract", {}).get("final_split_loaded") is not False
    ):
        raise ValueError(f"Experiment {experiment} summary delivery is invalid")
    return value


def aggregate_localization(
    exp42_summary_path: Path,
    exp43_summary_path: Path,
    source_paths: Mapping[tuple[str, str], Path],
) -> dict[str, Any]:
    _dependency_contract()
    evaluator_sha256 = file_sha256(Path(__file__))
    if evaluator_sha256 != registered_evaluator_sha256():
        raise ValueError("Experiment 044 evaluator hash does not match registration")
    exp42_summary = _source_summary(exp42_summary_path, EXP42_SUMMARY_SHA256, 42)
    exp43_summary = _source_summary(exp43_summary_path, EXP43_SUMMARY_SHA256, 43)
    all_records: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for slice_name in SLICES:
        sources[slice_name] = {}
        summary = exp42_summary if slice_name == "exp42" else exp43_summary
        for morphology in MORPHOLOGIES:
            if summary.get("source_shard_sha256", {}).get(morphology) != SOURCE_SHA256[
                slice_name
            ][morphology]:
                raise ValueError("Experiment 044 summary-to-source binding is invalid")
            records, source = _extract_report_records(
                source_paths[(slice_name, morphology)],
                slice_name=slice_name,
                morphology=morphology,
                source_sha256=SOURCE_SHA256[slice_name][morphology],
                archive_sha256=(
                    EXP43_ARCHIVE_SHA256[morphology] if slice_name == "exp43" else None
                ),
            )
            all_records.extend(records)
            sources[slice_name][morphology] = source
    pooled = [record for record in all_records if record["condition"] in {"full", "joint"}]
    exp43_records = [record for record in all_records if record["slice"] == "exp43"]
    partitions: dict[str, Any] = {}
    support_checks: dict[str, bool] = {}
    for partition in PARTITIONS:
        pooled_cell = [record for record in pooled if record["partition"] == partition]
        exp43_cell = [record for record in exp43_records if record["partition"] == partition]
        support = _support(pooled_cell)
        expected_anchors, expected_phase_units = EXPECTED_SUPPORT[partition]
        support_checks[partition] = all(
            support["by_slice"][slice_name]
            == {
                "endpoint_anchors": expected_anchors,
                "scenario_phase_units": expected_phase_units,
            }
            for slice_name in SLICES
        ) and {
            record["task"] for record in pooled_cell
        } == PARTITION_TASKS[partition] and all(
            len(
                {
                    (record["scenario_id"], record["phase"])
                    for record in pooled_cell
                    if record["slice"] == slice_name
                    and record["morphology"] == morphology
                    and record["task"] == task
                }
            )
            >= 7
            for slice_name in SLICES
            for morphology in MORPHOLOGIES
            for task in PARTITION_TASKS[partition]
        )
        partitions[partition] = {
            "support": support,
            "pooled_jf": _view(pooled_cell, ("full", "joint")),
            "exp43_decomposition": _view(exp43_cell, ("full", "half", "joint")),
        }

    def selected(
        *, partitions: set[str], tasks: set[str] | None = None
    ) -> list[dict[str, Any]]:
        return [
            record
            for record in pooled
            if record["partition"] in partitions and (tasks is None or record["task"] in tasks)
        ]

    manipulation = {"lift_object", "pick_place_bin"}
    localization = {
        "target_state": localization_contrast(
            selected(
                partitions={"closed_steady", "closed_closeward"}, tasks=manipulation
            ),
            selected(partitions={"open_steady", "open_openward"}, tasks=manipulation),
            left_label="closed_target",
            right_label="open_target",
            expected_tasks=manipulation,
        ),
        "opening_transition": localization_contrast(
            selected(partitions={"open_openward"}, tasks={"pick_place_bin"}),
            selected(partitions={"open_steady"}, tasks={"pick_place_bin"}),
            left_label="openward",
            right_label="open_steady",
            expected_tasks={"pick_place_bin"},
        ),
        "closing_transition": localization_contrast(
            selected(partitions={"closed_closeward"}, tasks=manipulation),
            selected(partitions={"closed_steady"}, tasks=manipulation),
            left_label="closeward",
            right_label="closed_steady",
            expected_tasks=manipulation,
        ),
        "terminal_behavior": localization_contrast(
            selected(partitions={"terminal_noop"}),
            selected(partitions=set(PARTITIONS) - {"terminal_noop"}),
            left_label="terminal_noop",
            right_label="commanded",
            expected_tasks=set(exp43.TASKS),
        ),
    }
    delivery_checks = {
        "source_summaries_passed_registered_delivery": True,
        "all_four_source_hashes_match": True,
        "all_source_shards_passed_registered_delivery": all(
            source["delivery_passed"]
            for slice_sources in sources.values()
            for source in slice_sources.values()
        ),
        "exact_728_endpoint_anchors": _support(pooled)["endpoint_anchors"] == 728,
        "exact_2184_primary_paired_observations": _support(pooled)[
            "contrast_observations"
        ]
        == 2184,
        "exact_5460_condition_records": len(all_records) == 5460,
        "partition_labels_exhaustive": sum(
            partitions[name]["support"]["endpoint_anchors"] for name in PARTITIONS
        )
        == 728,
        "registered_partition_support": all(support_checks.values()),
        "policy_inferences_run_is_zero": True,
        "training_runs_is_zero": True,
        "final_split_not_loaded": True,
    }
    delivery = all(delivery_checks.values())
    return {
        "format_version": 1,
        "experiment": 44,
        "registered": True,
        "status": "completed_gripper_target_direction_localization",
        "scope": (
            "post hoc development-only localization; no inferential, causal, population, "
            "closed-loop, admission, advancement, retraining, or final-split claim"
        ),
        "contract": {
            "evaluator_sha256": evaluator_sha256,
            "source_summary_sha256": {
                "exp42": EXP42_SUMMARY_SHA256,
                "exp43": EXP43_SUMMARY_SHA256,
            },
            "source_shard_sha256": SOURCE_SHA256,
            "source_archive_sha256": {"exp43": EXP43_ARCHIVE_SHA256},
            "slices": list(SLICES),
            "model_seeds": list(SEEDS),
            "partitions": list(PARTITIONS),
            "hierarchy": [
                "endpoint",
                "phase",
                "scenario",
                "task",
                "morphology",
                "slice",
                "seed",
            ],
            "policy_inferences_run": 0,
            "training_runs": 0,
            "final_split_loaded": False,
        },
        "sources": sources,
        "delivery_checks": delivery_checks,
        "delivery_gate_passed": delivery,
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


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"git_commit": commit, "git_dirty": dirty, "evaluator_sha256": file_sha256(Path(__file__))}


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    if path != SUMMARY_PATH:
        raise ValueError(f"registered Experiment 044 output must be {SUMMARY_PATH}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="aggregate frozen Experiment 042/043 gripper localization evidence"
    )
    parser.add_argument(
        "--exp42-summary", type=Path, default=Path("reports/benchmark-exp42-summary.json")
    )
    parser.add_argument(
        "--exp43-summary", type=Path, default=Path("reports/benchmark-exp43-summary.json")
    )
    parser.add_argument(
        "--exp42-so101",
        type=Path,
        default=Path("reports/benchmark-exp42-so101-phase-action-quality.json"),
    )
    parser.add_argument(
        "--exp42-panda",
        type=Path,
        default=Path("reports/benchmark-exp42-panda-phase-action-quality.json"),
    )
    parser.add_argument(
        "--exp43-so101",
        type=Path,
        default=Path("reports/benchmark-exp43-so101-gripper-mechanism.json.gz"),
    )
    parser.add_argument(
        "--exp43-panda",
        type=Path,
        default=Path("reports/benchmark-exp43-panda-gripper-mechanism.json.gz"),
    )
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    provenance = _git_provenance()
    if (
        provenance["git_dirty"]
        or provenance["evaluator_sha256"] != registered_evaluator_sha256()
    ):
        raise RuntimeError("registered Experiment 044 requires a clean evaluator commit")
    report = aggregate_localization(
        args.exp42_summary,
        args.exp43_summary,
        {
            ("exp42", "so101"): args.exp42_so101,
            ("exp42", "panda"): args.exp42_panda,
            ("exp43", "so101"): args.exp43_so101,
            ("exp43", "panda"): args.exp43_panda,
        },
    )
    report["provenance"] = provenance
    _write_report(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
