from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray
import torch

from ..processing import EmbodiProcessor
from . import benchmark_phase_agreement as exp42
from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv
from .benchmark_expert import PrivilegedBenchmarkExpert
from .benchmark_geometry_evaluation import (
    CONFIG_SHA256,
    load_development_contract,
    load_geometry_policy,
    resolve_checkpoint as resolve_exp36_checkpoint,
)


DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
DEVELOPMENT_MANIFEST_SHA256 = (
    "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
)
EXP36_SUMMARY_SHA256 = (
    "76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f"
)
EXP42_SUMMARY_SHA256 = (
    "104c16ca2552eb0d2ba7efc65c9228eeecd488d67d765f49d70931aa0d15abf8"
)
EXP43_TRAINING_SUMMARY_SHA256 = (
    "7bba5697e1a022983ecd619f1c1a5f925a21dce3d491b2f7689109edd8c10712"
)
EXP42_EVALUATOR_SHA256 = (
    "cc4fb718a3f6341dbaaea0026aca193fd86d5b84412ea7061204ed8f4ce24b7d"
)
EXPERT_SHA256 = "7fb869a10756864530889f809a8305ee1758bdc51eaeeeda0ff2f451f3b7f8e9"
EXP42_SOURCE_SHA256 = {
    "so101": "f02815631edb583a5faab4a82926afc650b69d5fc131a60227107bc0abecdd51",
    "panda": "55018696519b930d01e859a19365171a5b70d1fb227777b9bf624daecd0732e5",
}
EXP42_TRAINER_SHA256 = "b9502289a8e4ae71c4814f714e3452eaa7f82f534410b26f2f14fbc5847ca53f"
EXP42_EMBODIMENT_SHA256 = {
    "so101": {
        36001: "19a4f1f83889875f9d5d02e3bd3cf0f14344d5956b991a3634d6549b741a071f",
        36002: "77ae8cc66d5565a795ecb4fcd01c601f86fc7335ef0a53fd817a1879633b1fd9",
        36003: "68daa00b3304ace21fa06c8adbbfffc14a5156595a5ded4b8349da7fe124b671",
    },
    "panda": {
        36001: "d501477a51ed1642542ee6d10f2c5a62b9c89ccc698ec9fd54e6abbd73536ee3",
        36002: "c09979cb39587ca306dc0aa41ae16dac092147fe16bf304d04c7c1114aa82932",
        36003: "340de6edcb6984db7f40eade8f61704c443c45fe2b3f34352a5de177b8334c4e",
    },
}

CONDITIONS = ("full", "half", "joint")
CONDITION_CODES = {"full": "F", "half": "H", "joint": "J"}
CONDITION_PERMUTATIONS = (
    ("full", "half", "joint"),
    ("full", "joint", "half"),
    ("half", "full", "joint"),
    ("half", "joint", "full"),
    ("joint", "full", "half"),
    ("joint", "half", "full"),
)
CONTRASTS = {
    "J-F": ("joint", "full"),
    "H-F": ("half", "full"),
    "J-H": ("joint", "half"),
}
MORPHOLOGIES = exp42.MORPHOLOGIES
TASKS = exp42.TASKS
TASK_KEYS = exp42.TASK_KEYS
SEEDS = exp42.SEEDS
SCENARIO_INDICES = tuple(range(35, 42))
ENDPOINTS = exp42.ENDPOINTS
PHASES = exp42.PHASES
INSTRUCTIONS = exp42.INSTRUCTIONS
INSTRUCTION_SHA256 = exp42.INSTRUCTION_SHA256
ACTION_SCALES = exp42.ACTION_SCALES
REPEAT_MAX_ABS = exp42.REPEAT_MAX_ABS
MAXIMUM_STEPS = exp42.MAXIMUM_STEPS
RUNTIME_FIELDS = exp42.RUNTIME_FIELDS

EXPECTED_EPISODES = 42
EXPECTED_PHASE_UNITS = 182
EXPECTED_ENDPOINTS = 364
EXPECTED_SCORED_CHUNKS = 3276
EXPECTED_REPEAT_CHUNKS = 3276
EXPECTED_INFERENCES = 6552
EXPECTED_CONTRAST_OBSERVATIONS = 1092

REGISTERED_RUNTIME_DIR = Path("reports/exp43-runtime")
REGISTERED_LOCK_PATH = REGISTERED_RUNTIME_DIR / ".registered.lock"
LOCK_IDENTITY = "exp43-so101-panda-three-condition-gripper-v1"
SCHEDULE_IDENTITY = LOCK_IDENTITY
SUMMARY_PATH = Path("reports/benchmark-exp43-summary.json")

METRICS = (
    "normalized_loss",
    "translation_norm_error_m",
    "rotation_geodesic_error_deg",
    "effective_gripper_error",
    "raw_gripper_absolute_error",
    "signed_gripper_bias",
    "gripper_saturation_rate",
    "degenerate_rotation_rate",
)

PROTOCOL = {
    **exp42.PROTOCOL,
    "hierarchy": ["endpoint", "phase", "scenario", "task", "morphology", "seed"],
    "condition_order_formula": (
        "CONDITION_PERMUTATIONS[(anchor_ordinal + seed_ordinal + "
        "morphology_ordinal) % 6]"
    ),
    "conditions": ["F", "H", "J"],
    "primary_metric": "effective_gripper_error",
    "final_split_loaded": False,
}

Anchor = exp42.Anchor
canonical_array_sha256 = exp42.canonical_array_sha256
rotation_geodesic_error = exp42.rotation_geodesic_error
capture_expert_anchors = exp42.capture_expert_anchors
delivery_gate = exp42.delivery_gate


@dataclass
class _OrchestratorContext:
    run_id: str
    next_position: int = 0
    active: bool = True

    def require(self, morphology: str) -> None:
        if (
            not self.active
            or self is not _ACTIVE_ORCHESTRATOR
            or not _valid_run_id(self.run_id)
            or self.next_position >= len(MORPHOLOGIES)
            or MORPHOLOGIES[self.next_position] != morphology
        ):
            raise RuntimeError("registered Experiment 043 requires the active shard schedule")

    def complete(self, morphology: str) -> None:
        self.require(morphology)
        self.next_position += 1


_ACTIVE_ORCHESTRATOR: _OrchestratorContext | None = None


def _valid_run_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and set(value) <= set("0123456789abcdef")
    )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _parse_utc(value: Any) -> datetime:
    return exp42._parse_utc(value)


def _utc_now() -> str:
    return exp42._utc_now()


def _outcome_chain(outcomes: Sequence[dict[str, Any]]) -> str:
    return exp42._outcome_chain(outcomes)


def _write_final(path: Path, report: dict[str, Any]) -> None:
    exp42._write_final(path, report)


def _preflight_final(path: Path, report: dict[str, Any]) -> None:
    exp42._preflight_final(path, report)


def _write_exact_bytes(path: Path, value: bytes) -> None:
    exp42._write_exact_bytes(path, value)


@contextmanager
def registered_run_lock(path: Path = REGISTERED_LOCK_PATH) -> Iterator[None]:
    if path != REGISTERED_LOCK_PATH:
        raise ValueError(f"registered Experiment 043 lock path must be {REGISTERED_LOCK_PATH}")
    with _fcntl_lock(path):
        yield


@contextmanager
def _fcntl_lock(path: Path) -> Iterator[None]:
    # Experiment 042's public lock rejects the Experiment 043 path, so use its
    # already-frozen flock implementation's dependencies through a tiny local lock.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another registered Experiment 043 orchestrator holds {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def gripper_mechanism_scenarios(
    manifest: ScenarioManifest,
    morphology: str,
    *,
    registered: bool,
    limit_scenarios: int | None = None,
) -> tuple[Scenario, ...]:
    if morphology not in MORPHOLOGIES:
        raise ValueError("Experiment 043 morphology is invalid")
    selected = tuple(
        scenario
        for scenario in manifest.scenarios
        if scenario.morphology == morphology
        and scenario.task in TASKS
        and int(scenario.scenario_id.rsplit("-", 1)[-1]) in SCENARIO_INDICES
    )
    if registered:
        expected = [
            f"development-{morphology}-{task}-{index:04d}"
            for task in TASKS
            for index in SCENARIO_INDICES
        ]
        if [scenario.scenario_id for scenario in selected] != expected:
            raise ValueError("registered Experiment 043 requires exact indices 0035 through 0041")
        for task in TASKS:
            axes = [
                name
                for scenario in selected
                if scenario.task == task
                for name, value in scenario.factors.items()
                if str(value.get("bin", "")).startswith("development:")
            ]
            if len(axes) != 7 or len(set(axes)) != 7:
                raise ValueError("each Experiment 043 cell must cover all seven factor axes")
    return selected if limit_scenarios is None else selected[:limit_scenarios]


def condition_order(anchor_ordinal: int, seed: int, morphology: str) -> tuple[str, ...]:
    if type(anchor_ordinal) is not int or anchor_ordinal < 0:
        raise ValueError("anchor ordinal must be a nonnegative integer")
    try:
        seed_ordinal = SEEDS.index(seed)
        morphology_ordinal = MORPHOLOGIES.index(morphology)
    except ValueError as error:
        raise ValueError("condition order requires registered seed and morphology") from error
    return CONDITION_PERMUTATIONS[
        (anchor_ordinal + seed_ordinal + morphology_ordinal) % 6
    ]


def action_metrics(prediction: Any, expert: Any) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(10)
    target = np.asarray(expert, dtype=np.float64).reshape(10)
    base = exp42.action_metrics(predicted, target)
    return {
        **base,
        "raw_gripper_absolute_error": float(abs(predicted[9] - target[9])),
        "signed_gripper_bias": float(predicted[9] - target[9]),
        "gripper_saturation_rate": float(predicted[9] < 0.0 or predicted[9] > 1.0),
    }


def evaluate_anchor_condition(
    anchor: Anchor,
    seed: int,
    condition: str,
    policy: Any,
    processor: Any,
    device: str,
) -> dict[str, Any]:
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("Experiment 043 query requires a registered seed and condition")
    first, first_evidence = exp42._predict_once(anchor, policy, processor, device)
    repeat, repeat_evidence = exp42._predict_once(anchor, policy, processor, device)
    repeat_evidence["repeat"] = True
    stable = tuple(first_evidence)
    if any(
        name != "output_chunk"
        and not name.startswith("output_")
        and first_evidence[name] != repeat_evidence[name]
        for name in stable
    ):
        raise RuntimeError("immediate repeat policy input differs from first query")
    difference = float(np.max(np.abs(first - repeat)))
    if not np.isfinite(difference) or difference > REPEAT_MAX_ABS:
        raise RuntimeError("immediate repeated output exceeds determinism tolerance")
    return {
        "seed": seed,
        "condition": condition,
        "queries": [first_evidence, repeat_evidence],
        "repeat_max_abs_difference": difference,
        "metrics": action_metrics(first[0, 0, 0], anchor.expert_target),
    }


def _require_stable_anchor_inputs(seed_results: Mapping[str, Any]) -> None:
    queries = [
        query
        for seed in SEEDS
        for condition in CONDITIONS
        for query in seed_results[str(seed)][condition]["queries"]
    ]
    fields = (
        "instruction_key",
        "instruction",
        "instruction_sha256",
        "input_ids",
        "input_ids_sha256",
        "input_ids_dtype",
        "input_ids_shape",
        "attention_mask",
        "attention_mask_sha256",
        "attention_mask_dtype",
        "attention_mask_shape",
        "image_sha256",
        "state_sha256",
        "processed_image_sha256",
        "processed_image_dtype",
        "processed_image_shape",
        "passed_state_sha256",
        "passed_state_dtype",
        "passed_state_shape",
        "expert_target_sha256",
    )
    first = queries[0]
    if any(query[name] != first[name] for query in queries[1:] for name in fields):
        raise RuntimeError("anchor policy inputs differ across nine seed-condition arms")


def _metric_vector(record: Mapping[str, Any]) -> NDArray[Any]:
    metrics = record["metrics"]
    channels = np.asarray(metrics["channel_losses"], dtype=np.float64)
    values = np.asarray([metrics[name] for name in METRICS], dtype=np.float64)
    if channels.shape != (10,) or not np.isfinite(channels).all() or not np.isfinite(values).all():
        raise ValueError("Experiment 043 metric records must be finite with ten channels")
    return np.concatenate((values, channels))


def _vector_mean(values: Sequence[NDArray[Any]]) -> NDArray[Any]:
    if not values:
        raise ValueError("cannot average an empty Experiment 043 metric cell")
    width = values[0].shape[0]
    denominator = Decimal(len(values))
    return np.asarray(
        [
            float(
                sum(
                    (Decimal(str(float(value[index]))) for value in values),
                    start=Decimal(0),
                )
                / denominator
            )
            for index in range(width)
        ],
        dtype=np.float64,
    )


def hierarchical_condition_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("hierarchical aggregation requires records")
    dimensions = ("seed", "morphology", "task", "scenario_id", "phase", "endpoint")
    keyed: dict[tuple[Any, ...], list[NDArray[Any]]] = defaultdict(list)
    for record in records:
        keyed[tuple(record[name] for name in dimensions)].append(_metric_vector(record))
    level = {key: _vector_mean(values) for key, values in keyed.items()}
    for _ in dimensions:
        grouped: dict[tuple[Any, ...], list[NDArray[Any]]] = defaultdict(list)
        for key, value in level.items():
            grouped[key[:-1]].append(value)
        level = {key: _vector_mean(values) for key, values in grouped.items()}
    if set(level) != {()}:
        raise RuntimeError("Experiment 043 hierarchy did not collapse to one cell")
    value = level[()]
    return {
        **{name: float(value[index]) for index, name in enumerate(METRICS)},
        "channel_losses": value[len(METRICS) :].tolist(),
        "degenerate_rotation_count": sum(
            bool(record["metrics"]["degenerate_rotation"]) for record in records
        ),
        "records": len(records),
    }


def three_condition_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells = {
        condition: hierarchical_condition_metrics(
            [record for record in records if record["condition"] == condition]
        )
        for condition in CONDITIONS
    }

    def metric_cell(name: str, values: Mapping[str, float]) -> dict[str, Any]:
        decimal_values = {
            condition: Decimal(str(values[condition])) for condition in CONDITIONS
        }
        direct_decimal_differences = {
            contrast: decimal_values[numerator] - decimal_values[denominator]
            for contrast, (numerator, denominator) in CONTRASTS.items()
        }
        return {
            **values,
            "contrasts": {
                contrast: {
                    "difference": float(direct_decimal_differences[contrast]),
                    "difference_decimal": str(direct_decimal_differences[contrast]),
                    "ratio": (
                        float(decimal_values[numerator] / decimal_values[denominator])
                        if decimal_values[denominator] != 0
                        else None
                    ),
                    "ratio_decimal": (
                        str(decimal_values[numerator] / decimal_values[denominator])
                        if decimal_values[denominator] != 0
                        else None
                    ),
                }
                for contrast, (numerator, denominator) in CONTRASTS.items()
            },
        }

    result = {
        name: metric_cell(name, {condition: cells[condition][name] for condition in CONDITIONS})
        for name in METRICS
    }
    result["channels"] = {
        str(index): metric_cell(
            str(index),
            {
                condition: cells[condition]["channel_losses"][index]
                for condition in CONDITIONS
            },
        )
        for index in range(10)
    }
    result["degenerate_rotation_count"] = {
        condition: cells[condition]["degenerate_rotation_count"] for condition in CONDITIONS
    }
    result["records_per_condition"] = {
        condition: cells[condition]["records"] for condition in CONDITIONS
    }
    return result


def _subgroups(
    records: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    return {
        "/".join(str(value) for value in key): three_condition_metrics(values)
        for key, values in sorted(grouped.items())
    }


def aggregate_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    overall = three_condition_metrics(records)
    result = {
        "overall": overall,
        "by_seed": _subgroups(records, ("seed",)),
        "by_task": _subgroups(records, ("task",)),
        "by_morphology": _subgroups(records, ("morphology",)),
        "by_endpoint": _subgroups(records, ("endpoint",)),
        "by_task_phase": _subgroups(records, ("task", "phase")),
        "by_channel": overall["channels"],
    }
    _validate_decomposition(result)
    return result


def _validate_decomposition(metrics: Mapping[str, Any]) -> None:
    groups = [metrics["overall"]]
    groups.extend(
        value
        for name in ("by_seed", "by_task", "by_morphology", "by_endpoint", "by_task_phase")
        for value in metrics[name].values()
    )
    for group in groups:
        cells = [group[name] for name in METRICS]
        cells.extend(group["channels"].values())
        for cell in cells:
            contrasts = cell["contrasts"]
            exact = {
                contrast: Decimal(contrasts[contrast]["difference_decimal"])
                for contrast in CONTRASTS
            }
            if exact["J-F"] != exact["H-F"] + exact["J-H"] or any(
                not math.isclose(
                    contrasts[contrast]["difference"],
                    float(exact[contrast]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                for contrast in CONTRASTS
            ):
                raise ValueError("Experiment 043 contrast decomposition is invalid")


def _reported_decimal(
    metrics: Mapping[str, Any], group: str, key: str, contrast: str, field: str
) -> Decimal | None:
    try:
        cell = metrics[group] if not key else metrics[group][key]
        value = Decimal(
            str(cell["effective_gripper_error"]["contrasts"][contrast][field])
        )
    except (KeyError, TypeError, InvalidOperation):
        return None
    return value if value.is_finite() else None


def classify_mechanism(metrics: Mapping[str, Any], *, delivery: bool) -> dict[str, Any]:
    def materiality(contrast: str) -> dict[str, Any]:
        difference_decimal = _reported_decimal(
            metrics, "overall", "", contrast, "difference_decimal"
        )
        ratio_decimal = _reported_decimal(
            metrics, "overall", "", contrast, "ratio_decimal"
        )
        ratio = float(ratio_decimal) if ratio_decimal is not None else None
        difference = (
            float(difference_decimal) if difference_decimal is not None else None
        )
        seed_ratio_decimals = {
            str(seed): _reported_decimal(
                metrics, "by_seed", str(seed), contrast, "ratio_decimal"
            )
            for seed in SEEDS
        }
        seed_ratios = {
            seed: float(value) if value is not None else None
            for seed, value in seed_ratio_decimals.items()
        }
        checks = {
            "delivery_gate_passed": delivery,
            "overall_ratio_at_least_1_10": ratio_decimal is not None
            and ratio_decimal >= Decimal("1.10"),
            "overall_difference_at_least_0_01": difference_decimal is not None
            and difference_decimal >= Decimal("0.01"),
            "at_least_two_seed_ratios_at_least_1_10": sum(
                value is not None and value >= Decimal("1.10")
                for value in seed_ratio_decimals.values()
            )
            >= 2,
        }
        broad_differences = {
            "by_endpoint": {
                endpoint: _reported_decimal(
                    metrics, "by_endpoint", endpoint, contrast, "difference_decimal"
                )
                for endpoint in ENDPOINTS
            },
            "by_task": {
                task: _reported_decimal(
                    metrics, "by_task", task, contrast, "difference_decimal"
                )
                for task in TASKS
            },
            "by_morphology": {
                morphology: _reported_decimal(
                    metrics,
                    "by_morphology",
                    morphology,
                    contrast,
                    "difference_decimal",
                )
                for morphology in MORPHOLOGIES
            },
        }
        broad = all(
            value is not None and value >= Decimal(0)
            for group in broad_differences.values()
            for value in group.values()
        )
        return {
            "supported": all(checks.values()),
            "checks": checks,
            "overall_ratio": ratio,
            "overall_difference": difference,
            "per_seed_ratio": seed_ratios,
            "broad_nonnegative": broad,
            "broad_differences": {
                group: {
                    key: float(value) if value is not None else None
                    for key, value in values.items()
                }
                for group, values in broad_differences.items()
            },
        }

    premise = materiality("J-F")
    exposure = materiality("H-F")
    shared = materiality("J-H")
    if not premise["supported"]:
        classification = "premise-not-confirmed"
    else:
        supported = (exposure["supported"], shared["supported"])
        classification = {
            (True, False): "exposure-dominant",
            (False, True): "shared-training-dominant",
            (True, True): "both-supported",
            (False, False): "neither-materially-supported",
        }[supported]
        supported_gates = [gate for gate in (exposure, shared) if gate["supported"]]
        if supported_gates and all(gate["broad_nonnegative"] for gate in supported_gates):
            classification += "-broad"
    shared_difference = shared["overall_difference"]
    return {
        "classification": classification,
        "premise": premise,
        "exposure": exposure,
        "shared_training": shared,
        "negative_J_minus_H_describes_positive_transfer": shared_difference is not None
        and shared_difference < 0,
        "inferential_claim": False,
        "population_claim": False,
        "closed_loop_claim": False,
    }


def _source_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    expected = {
        "exp36_summary": EXP36_SUMMARY_SHA256,
        "exp42_summary": EXP42_SUMMARY_SHA256,
        "exp43_training_summary": EXP43_TRAINING_SUMMARY_SHA256,
        "exp42_so101_shard": EXP42_SOURCE_SHA256["so101"],
        "exp42_panda_shard": EXP42_SOURCE_SHA256["panda"],
    }
    if set(paths) != set(expected):
        raise ValueError("Experiment 043 requires exactly the three frozen source summaries")
    for name, digest in expected.items():
        if not paths[name].is_file() or file_sha256(paths[name]) != digest:
            raise ValueError(f"{name.replace('_', ' ')} hash differs from Experiment 043 registration")
    if file_sha256(Path(exp42.__file__)) != EXP42_EVALUATOR_SHA256:
        raise ValueError("imported Experiment 042 evaluator hash differs from registration")
    if file_sha256(Path(exp42.__file__).with_name("benchmark_expert.py")) != EXPERT_SHA256:
        raise ValueError("expert implementation hash differs from registration")
    summaries = {
        name: json.loads(paths[name].read_bytes())
        for name in ("exp36_summary", "exp42_summary", "exp43_training_summary")
    }
    exp36 = summaries["exp36_summary"]
    previous = summaries["exp42_summary"]
    training = summaries["exp43_training_summary"]
    if (
        exp36.get("experiment") != 36
        or exp36.get("status") != "completed_gate_passed"
        or exp36.get("fixed_step") != 1200
        or previous.get("experiment") != 42
        or previous.get("delivery_gate_passed") is not True
        or previous.get("final_split_loaded") is not False
        or training.get("experiment") != 43
        or training.get("status") != "completed_training_controls"
        or training.get("delivery_gate_passed") is not True
        or training.get("final_split_loaded") is not False
        or previous.get("source_shard_sha256") != EXP42_SOURCE_SHA256
    ):
        raise ValueError("Experiment 043 frozen source summary contract is invalid")
    return summaries


def _half_checkpoint(
    training_path: Path, seed: int, morphology: str, *, root: Path
) -> tuple[Path, dict[str, Any]]:
    if file_sha256(training_path) != EXP43_TRAINING_SUMMARY_SHA256:
        raise ValueError("Experiment 043 training summary hash differs from registration")
    summary = json.loads(training_path.read_bytes())
    run = summary.get("runs", {}).get(f"{morphology}-seed{seed}", {})
    expected = run.get("checkpoints", {}).get(morphology, {})
    checkpoint = root / str(expected.get("path", ""))
    data_manifest = root / str(run.get("output", "")) / "final/data_manifest.json"
    files = {
        "config_sha256": checkpoint / "config.json",
        "core_sha256": checkpoint / "core.pt",
        "embodiment_sha256": checkpoint / "embodiment.pt",
        "trainer_sha256": checkpoint / "trainer.pt",
        "data_manifest_sha256": data_manifest,
    }
    expected_hashes = {
        **{name: expected.get(name) for name in files if name != "data_manifest_sha256"},
        "data_manifest_sha256": run.get("data_manifest_sha256"),
    }
    if (
        run.get("model_seed") != seed
        or expected.get("path") != str(Path(run.get("output", "")) / "final" / morphology)
        or expected.get("config_sha256") != CONFIG_SHA256[morphology]
        or any(not path.is_file() for path in files.values())
        or any(file_sha256(path) != expected_hashes[name] for name, path in files.items())
    ):
        raise ValueError("half checkpoint files or training-summary hashes are invalid")
    trainer = torch.load(
        checkpoint / "trainer.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if trainer.get("step") != 1200 or trainer.get("stage") != "core":
        raise ValueError("half checkpoint trainer identity is not fixed step 1200 core")
    return checkpoint, {
        "condition": "half",
        "source_condition": morphology,
        "model_seed": seed,
        "morphology": morphology,
        "checkpoint_path": str(checkpoint),
        **expected_hashes,
        "source_summary_sha256": EXP43_TRAINING_SUMMARY_SHA256,
        "step": 1200,
        "stage": "core",
    }


def resolve_experiment43_checkpoint(
    source_paths: Mapping[str, Path],
    condition: str,
    seed: int,
    morphology: str,
    *,
    root: Path = Path("."),
) -> tuple[Path, dict[str, Any]]:
    if condition not in CONDITIONS or seed not in SEEDS or morphology not in MORPHOLOGIES:
        raise ValueError("Experiment 043 checkpoint identity is not registered")
    if condition == "half":
        return _half_checkpoint(source_paths["exp43_training_summary"], seed, morphology, root=root)
    source = "joint" if condition == "joint" else morphology
    checkpoint, identity = resolve_exp36_checkpoint(
        source_paths["exp36_summary"], source, seed, morphology, root=root
    )
    if (
        identity.get("embodiment_sha256") != EXP42_EMBODIMENT_SHA256[morphology][seed]
        or identity.get("trainer_sha256") != EXP42_TRAINER_SHA256
    ):
        raise ValueError("full or joint checkpoint differs from frozen Experiment 042 identity")
    identity = {
        **identity,
        "condition": condition,
        "source_condition": source,
        "source_summary_sha256": EXP36_SUMMARY_SHA256,
    }
    return checkpoint, identity


def _checkpoint_identities(
    source_paths: Mapping[str, Path], morphology: str, root: Path
) -> tuple[dict[int, dict[str, Path]], dict[str, dict[str, dict[str, Any]]]]:
    paths: dict[int, dict[str, Path]] = {}
    identities: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        paths[seed] = {}
        identities[str(seed)] = {}
        for condition in CONDITIONS:
            path, identity = resolve_experiment43_checkpoint(
                source_paths, condition, seed, morphology, root=root
            )
            paths[seed][condition] = path
            identities[str(seed)][condition] = identity
    return paths, identities


def _contract(registered: bool, limit_scenarios: int | None) -> dict[str, Any]:
    return {
        "definition_sha256": DEFINITION_SHA256,
        "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
        "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
        "exp42_summary_sha256": EXP42_SUMMARY_SHA256,
        "exp43_training_summary_sha256": EXP43_TRAINING_SUMMARY_SHA256,
        "exp42_evaluator_sha256": EXP42_EVALUATOR_SHA256,
        "expert_sha256": EXPERT_SHA256,
        "split": "development",
        "scenario_indices": list(SCENARIO_INDICES),
        "tasks": list(TASKS),
        "conditions": list(CONDITIONS),
        "model_seeds": list(SEEDS),
        "limit_scenarios": None if registered else limit_scenarios,
        "protocol": PROTOCOL,
        "final_split_loaded": False,
    }


def _runtime_provenance(device: str) -> dict[str, Any]:
    value = exp42._runtime_provenance(device)
    value["evaluator_sha256"] = file_sha256(Path(__file__))
    value["imported_exp42_evaluator_sha256"] = file_sha256(Path(exp42.__file__))
    value.pop("imported_exp41_evaluator_sha256", None)
    return value


def _run_metadata(
    morphology: str,
    device: str,
    registered: bool,
    orchestrator_run_id: str | None,
) -> dict[str, Any]:
    return {
        "started_utc": _utc_now(),
        "finished_utc": None,
        "device": device,
        "lock_identity": LOCK_IDENTITY if registered else None,
        "lock_path": str(REGISTERED_LOCK_PATH) if registered else None,
        "schedule_identity": SCHEDULE_IDENTITY if registered else "smoke-unregistered",
        "schedule_position": MORPHOLOGIES.index(morphology) if registered else None,
        "orchestrator_run_id": orchestrator_run_id if registered else None,
        "seed_order": list(SEEDS),
        "condition_order_formula": PROTOCOL["condition_order_formula"],
    }


def _flatten_condition_records(reports: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "condition": condition,
            "morphology": anchor["morphology"],
            "task": anchor["task"],
            "scenario_id": anchor["scenario_id"],
            "phase": anchor["phase"],
            "endpoint": anchor["endpoint"],
            **anchor["seeds"][str(seed)][condition],
        }
        for report in reports
        for anchor in report["anchors"]
        for seed in SEEDS
        for condition in CONDITIONS
    ]


def _shard_aggregate(anchors: Sequence[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    gate = delivery_gate(anchors)
    records = _flatten_condition_records([report])
    integrity = {
        "all_expected_endpoint_anchors": [
            f"{anchor['scenario_id']}/{anchor['phase']}/{anchor['endpoint']}"
            for anchor in anchors
        ]
        == report["expected_anchor_ids"],
        "all_expert_episodes_successful_in_done": all(
            episode["success"] is True
            and episode["terminal"] is True
            and episode["final_phase"] == "done"
            for episode in report["episodes"]
        ),
        "identical_anchor_hashes_across_nine_arms": all(
            query["image_sha256"] == anchor["image_sha256"]
            and query["state_sha256"] == anchor["state_sha256"]
            and query["expert_target_sha256"] == anchor["expert_target_sha256"]
            for anchor in anchors
            for seed in SEEDS
            for condition in CONDITIONS
            for query in anchor["seeds"][str(seed)][condition]["queries"]
        ),
        "exact_correct_instruction_hashes": all(
            query["instruction_sha256"] == INSTRUCTION_SHA256[TASK_KEYS[anchor["task"]]]
            for anchor in anchors
            for seed in SEEDS
            for condition in CONDITIONS
            for query in anchor["seeds"][str(seed)][condition]["queries"]
        ),
        "deterministic_immediate_repeats": all(
            anchor["seeds"][str(seed)][condition]["repeat_max_abs_difference"]
            <= REPEAT_MAX_ABS
            for anchor in anchors
            for seed in SEEDS
            for condition in CONDITIONS
        ),
        "complete_checkpoint_hashes": all(
            all(
                _valid_sha256(
                    report["identity"]["checkpoints"][str(seed)][condition].get(name)
                )
                for name in (
                    "core_sha256",
                    "config_sha256",
                    "embodiment_sha256",
                    "trainer_sha256",
                    "data_manifest_sha256",
                )
            )
            for seed in SEEDS
            for condition in CONDITIONS
        ),
        "final_split_not_loaded": report["contract"].get("final_split_loaded") is False,
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


def _validate_query(
    query: Mapping[str, Any], anchor: Mapping[str, Any], *, repeat: bool
) -> NDArray[Any]:
    return exp42._validate_query(query, anchor, repeat=repeat)


def _validate_anchor_record(
    anchor: dict[str, Any], report: dict[str, Any], ordinal: int
) -> None:
    morphology = report["identity"]["morphology"]
    if (
        anchor.get("ordinal") != ordinal
        or anchor.get("morphology") != morphology
        or anchor.get("task") not in TASKS
        or anchor.get("phase") not in PHASES[anchor["task"]]
        or anchor.get("endpoint") not in ENDPOINTS
        or type(anchor.get("phase_tick")) is not int
        or anchor["phase_tick"] < 1
        or anchor.get("pre_call_phase") != anchor.get("phase")
        or anchor.get("pre_call_phase_tick") != anchor.get("phase_tick")
        or type(anchor.get("pre_call_total_tick")) is not int
        or anchor["pre_call_total_tick"] < 0
        or type(anchor.get("post_action_phase_tick")) is not int
        or type(anchor.get("post_action_total_tick")) is not int
        or anchor["post_action_total_tick"] != anchor["pre_call_total_tick"] + 1
        or anchor.get("post_action_state_sha256") != anchor.get("state_sha256")
        or type(anchor.get("expert_call_index")) is not int
        or anchor["expert_call_index"] < 0
        or anchor["expert_call_index"] != anchor["pre_call_total_tick"]
        or not _valid_sha256(anchor.get("image_sha256"))
        or not _valid_sha256(anchor.get("state_sha256"))
        or tuple(anchor.get("seeds", {})) != tuple(str(seed) for seed in SEEDS)
    ):
        raise ValueError("Experiment 043 endpoint anchor identity is invalid")
    native_shape = (6 if morphology == "so101" else 8,)
    specifications = (
        ("pre_call_native_state", np.float32, native_shape),
        ("expert_native", np.float32, native_shape),
        ("expert_target", np.float32, (1, 10)),
        ("decoded_native", np.float32, native_shape),
        ("realizable_native", np.float64, native_shape),
        ("realized_canonical", np.float32, (1, 10)),
    )
    arrays: dict[str, NDArray[Any]] = {}
    for name, dtype, shape in specifications:
        try:
            value = np.asarray(anchor.get(name), dtype=dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("Experiment 043 persisted expert array is invalid") from error
        if (
            value.shape != shape
            or not np.isfinite(value).all()
            or anchor.get(f"{name}_sha256") != canonical_array_sha256(value)
        ):
            raise ValueError("Experiment 043 persisted expert array hash is invalid")
        arrays[name] = value
    target = np.asarray(arrays["expert_target"], dtype=np.float64)
    realized = np.asarray(arrays["realized_canonical"], dtype=np.float64)
    requested = np.asarray(arrays["expert_native"], dtype=np.float64)
    decoded = np.asarray(arrays["decoded_native"], dtype=np.float64)
    clipped, clipping_error = exp42._clipping_from_arrays(
        arrays["decoded_native"], arrays["realizable_native"]
    )
    rotation, degenerate = rotation_geodesic_error(target[0, 3:9], realized[0, 3:9])
    round_trip = anchor.get("round_trip", {})
    if (
        degenerate
        or round_trip.get("clipped") is not clipped
        or round_trip.get("clipping_error_native") != clipping_error
        or round_trip.get("translation_error_m")
        != float(np.linalg.norm(target[0, :3] - realized[0, :3]))
        or round_trip.get("rotation_error_deg") != rotation
        or round_trip.get("gripper_error_native")
        != float(abs(requested[-1] - decoded[-1]))
    ):
        raise ValueError("Experiment 043 round-trip evidence is invalid")
    phase_order = list(PHASES[anchor["task"]])
    phase_index = phase_order.index(anchor["phase"])
    expected_post = (
        phase_order[phase_index + 1] if phase_index + 1 < len(phase_order) else "done"
    )
    transitioned = anchor.get("post_action_phase") != anchor["phase"]
    if anchor["endpoint"] == "last" and (
        not transitioned
        or anchor.get("post_action_phase") != expected_post
        or anchor.get("post_action_phase_tick") != 1
    ):
        raise ValueError("Experiment 043 last endpoint transition is invalid")
    if anchor["endpoint"] == "first" and transitioned and (
        anchor.get("post_action_phase") != expected_post
        or anchor.get("post_action_phase_tick") != 1
    ):
        raise ValueError("Experiment 043 first endpoint transition is invalid")
    if anchor["endpoint"] == "first" and not transitioned and (
        anchor.get("post_action_phase_tick") != anchor["phase_tick"] + 1
    ):
        raise ValueError("Experiment 043 phase tick transition is invalid")
    if anchor.get("post_action_phase") == "done" and not np.array_equal(
        arrays["expert_native"], arrays["pre_call_native_state"]
    ):
        raise ValueError("Experiment 043 terminal target is not the native no-op")
    for seed in SEEDS:
        results = anchor["seeds"][str(seed)]
        if tuple(results) != condition_order(ordinal, seed, morphology):
            raise ValueError("Experiment 043 condition permutation is invalid")
        for condition in CONDITIONS:
            result = results[condition]
            queries = result.get("queries")
            if (
                result.get("seed") != seed
                or result.get("condition") != condition
                or not isinstance(queries, list)
                or len(queries) != 2
            ):
                raise ValueError("Experiment 043 condition result is invalid")
            first = _validate_query(queries[0], anchor, repeat=False)
            repeat = _validate_query(queries[1], anchor, repeat=True)
            difference = float(np.max(np.abs(first - repeat)))
            if (
                result.get("repeat_max_abs_difference") != difference
                or difference > REPEAT_MAX_ABS
                or result.get("metrics") != action_metrics(first[0, 0], target)
            ):
                raise ValueError("Experiment 043 output metrics or repeat do not recompute")
    try:
        _require_stable_anchor_inputs(anchor["seeds"])
    except RuntimeError as error:
        raise ValueError("Experiment 043 cross-arm policy inputs differ") from error


def _validate_checkpoints(
    report: Mapping[str, Any], source_paths: Mapping[str, Path], *, root: Path
) -> None:
    morphology = report["identity"]["morphology"]
    stored = report["identity"].get("checkpoints", {})
    if tuple(stored) != tuple(str(seed) for seed in SEEDS):
        raise ValueError("Experiment 043 checkpoint seed identities are incomplete")
    for seed in SEEDS:
        if set(stored[str(seed)]) != set(CONDITIONS):
            raise ValueError("Experiment 043 checkpoint conditions are incomplete")
        for condition in CONDITIONS:
            _, expected = resolve_experiment43_checkpoint(
                source_paths, condition, seed, morphology, root=root
            )
            if stored[str(seed)][condition] != expected:
                raise ValueError("Experiment 043 checkpoint identity or files differ")


def validate_shard_report(
    report: dict[str, Any],
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    *,
    require_registered: bool,
    artifact_root: Path = Path("."),
) -> None:
    _source_contract(source_paths)
    if (
        manifest.definition_sha256 != DEFINITION_SHA256
        or file_sha256(manifest.path) != DEVELOPMENT_MANIFEST_SHA256
    ):
        raise ValueError("Experiment 043 development manifest identity is invalid")
    identity = report.get("identity", {})
    morphology = identity.get("morphology")
    contract = report.get("contract", {})
    scenarios = gripper_mechanism_scenarios(
        manifest,
        morphology,
        registered=require_registered,
        limit_scenarios=None if require_registered else contract.get("limit_scenarios"),
    )
    expected_ids = [
        f"{scenario.scenario_id}/{phase}/{endpoint}"
        for scenario in scenarios
        for phase in PHASES[scenario.task]
        for endpoint in ENDPOINTS
    ]
    anchors = report.get("anchors", [])
    episodes = report.get("episodes", [])
    run = report.get("run", {})
    provenance = report.get("provenance", {})
    if (
        report.get("format_version") != 1
        or report.get("experiment") != 43
        or report.get("mode") != "three_condition_gripper_morphology_shard"
        or report.get("registered") is not require_registered
        or report.get("complete") is not True
        or report.get("status")
        != ("completed" if require_registered else "completed_smoke_unregistered")
        or morphology not in MORPHOLOGIES
        or identity.get("model_seeds") != list(SEEDS)
        or contract != _contract(require_registered, contract.get("limit_scenarios"))
        or report.get("expected_anchor_ids") != expected_ids
        or [
            f"{anchor.get('scenario_id')}/{anchor.get('phase')}/{anchor.get('endpoint')}"
            for anchor in anchors
        ]
        != expected_ids
        or len(episodes) != len(scenarios)
        or report.get("outcome_chain_sha256") != _outcome_chain(anchors)
        or report.get("infrastructure_errors") != []
        or run.get("seed_order") != list(SEEDS)
        or run.get("condition_order_formula") != PROTOCOL["condition_order_formula"]
        or (require_registered and not _valid_run_id(run.get("orchestrator_run_id")))
        or (not require_registered and run.get("orchestrator_run_id") is not None)
        or run.get("device") != provenance.get("runtime", {}).get("device")
        or run.get("finished_utc") is None
    ):
        raise ValueError("Experiment 043 shard contract or completion state is invalid")
    for episode, scenario in zip(episodes, scenarios, strict=True):
        if (
            episode.get("scenario_id") != scenario.scenario_id
            or episode.get("task") != scenario.task
            or type(episode.get("steps")) is not int
            or not 1 <= episode["steps"] <= MAXIMUM_STEPS
            or episode.get("success") is not True
            or episode.get("terminal") is not True
            or episode.get("final_phase") != "done"
            or episode.get("observed_phases") != ["open", *PHASES[scenario.task], "done"]
            or episode.get("scored_phases") != list(PHASES[scenario.task])
            or episode.get("endpoint_count") != len(PHASES[scenario.task]) * 2
        ):
            raise ValueError("Experiment 043 expert episode metadata is invalid")
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    for ordinal, anchor in enumerate(anchors):
        anchor_scenario = scenarios_by_id.get(anchor.get("scenario_id"))
        if (
            anchor_scenario is None
            or anchor.get("scenario_seed") != anchor_scenario.seed
            or anchor.get("task") != anchor_scenario.task
        ):
            raise ValueError("Experiment 043 anchor scenario metadata is invalid")
        _validate_anchor_record(anchor, report, ordinal)
    exp42._validate_endpoint_pairs(anchors)
    for episode in episodes:
        matching = [anchor for anchor in anchors if anchor["scenario_id"] == episode["scenario_id"]]
        ticks = {
            phase: [anchor["phase_tick"] for anchor in matching if anchor["phase"] == phase]
            for phase in PHASES[episode["task"]]
        }
        if episode.get("phase_endpoint_ticks") != ticks:
            raise ValueError("Experiment 043 episode endpoint ticks do not recompute")
    _validate_checkpoints(report, source_paths, root=artifact_root)
    aggregate = _shard_aggregate(anchors, report)
    if report.get("aggregate") != aggregate:
        raise ValueError("Experiment 043 shard aggregate does not recompute")
    if _parse_utc(run["finished_utc"]) < _parse_utc(run.get("started_utc")):
        raise ValueError("Experiment 043 shard interval is negative")
    if require_registered:
        runtime = provenance.get("runtime")
        if (
            len(anchors) != EXPECTED_ENDPOINTS // 2
            or aggregate["phase_units"] != EXPECTED_PHASE_UNITS // 2
            or aggregate["scored_chunks"] != EXPECTED_SCORED_CHUNKS // 2
            or aggregate["repeat_chunks"] != EXPECTED_REPEAT_CHUNKS // 2
            or aggregate["policy_inferences"] != EXPECTED_INFERENCES // 2
            or any(
                aggregate["contrast_observations"][contrast]
                != EXPECTED_CONTRAST_OBSERVATIONS // 2
                for contrast in CONTRASTS
            )
            or provenance.get("git_dirty") is not False
            or provenance.get("evaluator_sha256") != file_sha256(Path(__file__))
            or provenance.get("imported_exp42_evaluator_sha256")
            != EXP42_EVALUATOR_SHA256
            or provenance.get("expert_sha256") != EXPERT_SHA256
            or not isinstance(provenance.get("git_commit"), str)
            or not isinstance(runtime, dict)
            or set(runtime) != RUNTIME_FIELDS
            or provenance.get("runtime_signature_sha256")
            != hashlib.sha256(
                json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            or run.get("lock_identity") != LOCK_IDENTITY
            or run.get("lock_path") != str(REGISTERED_LOCK_PATH)
            or run.get("schedule_identity") != SCHEDULE_IDENTITY
            or run.get("schedule_position") != MORPHOLOGIES.index(morphology)
            or not aggregate["delivery_gate"]["passed"]
        ):
            raise ValueError("registered Experiment 043 shard provenance or delivery is invalid")


def evaluate_gripper_mechanism_shard(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    output: Path,
    *,
    morphology: str,
    device: str,
    registered: bool,
    orchestrator: _OrchestratorContext | None = None,
    limit_scenarios: int | None = None,
    root: Path = Path("."),
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
    expert_factory: Callable[[Any], Any] = PrivilegedBenchmarkExpert,
    policy_loader: Callable[[Path, str], Any] = load_geometry_policy,
    processor_factory: Callable[..., Any] = EmbodiProcessor.from_pretrained,
) -> dict[str, Any]:
    if (
        definition.sha256 != DEFINITION_SHA256
        or manifest.definition_sha256 != DEFINITION_SHA256
        or file_sha256(manifest.path) != DEVELOPMENT_MANIFEST_SHA256
    ):
        raise ValueError("Experiment 043 benchmark inputs differ from registration")
    if registered:
        if orchestrator is None:
            raise RuntimeError("registered Experiment 043 evaluation requires orchestrator context")
        orchestrator.require(morphology)
    elif orchestrator is not None:
        raise ValueError("smoke evaluation may not use a registered orchestrator")
    if registered and limit_scenarios is not None:
        raise ValueError("registered Experiment 043 does not permit a scenario limit")
    if registered and output != REGISTERED_RUNTIME_DIR / registered_shard_filename(morphology):
        raise ValueError("registered Experiment 043 output path is frozen")
    if output.exists():
        raise FileExistsError(f"Experiment 043 shards are non-resumable; discard {output} first")
    _source_contract(source_paths)
    provenance = _runtime_provenance(device)
    if registered and provenance["git_dirty"]:
        raise RuntimeError("registered Experiment 043 requires a clean worktree")
    checkpoints, identities = _checkpoint_identities(source_paths, morphology, root)
    scenarios = gripper_mechanism_scenarios(
        manifest,
        morphology,
        registered=registered,
        limit_scenarios=limit_scenarios,
    )
    run = _run_metadata(
        morphology, device, registered, None if orchestrator is None else orchestrator.run_id
    )
    anchors, episodes = capture_expert_anchors(
        scenarios, morphology, env_factory=env_factory, expert_factory=expert_factory
    )
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
        raise ValueError("all Experiment 043 arms must share the processor contract")
    processor = processor_factory(
        processor_contract[0],
        revision=processor_contract[1],
        do_rescale=processor_contract[2],
    )
    records = []
    for anchor in anchors:
        seed_results = {}
        for seed in SEEDS:
            seed_results[str(seed)] = {
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
        _require_stable_anchor_inputs(seed_results)
        records.append({**anchor.evidence, "seeds": seed_results})
    report: dict[str, Any] = {
        "format_version": 1,
        "experiment": 43,
        "mode": "three_condition_gripper_morphology_shard",
        "registered": registered,
        "status": "completed" if registered else "completed_smoke_unregistered",
        "complete": True,
        "contract": _contract(registered, limit_scenarios),
        "identity": {
            "morphology": morphology,
            "model_seeds": list(SEEDS),
            "checkpoints": identities,
        },
        "expected_anchor_ids": [
            f"{scenario.scenario_id}/{phase}/{endpoint}"
            for scenario in scenarios
            for phase in PHASES[scenario.task]
            for endpoint in ENDPOINTS
        ],
        "provenance": provenance,
        "run": run,
        "episodes": episodes,
        "anchors": records,
        "outcome_chain_sha256": _outcome_chain(records),
        "infrastructure_errors": [],
    }
    report["run"]["finished_utc"] = _utc_now()
    report["aggregate"] = _shard_aggregate(records, report)
    validate_shard_report(
        report,
        manifest,
        source_paths,
        require_registered=registered,
        artifact_root=root,
    )
    _write_final(output, report)
    if orchestrator is not None:
        orchestrator.complete(morphology)
    return report


def registered_shard_filename(morphology: str) -> str:
    if morphology not in MORPHOLOGIES:
        raise ValueError("registered shard filename requires a registered morphology")
    return f"benchmark-exp43-{morphology}-gripper-mechanism.json"


def _registered_output(morphology: str, output: Path | None) -> Path:
    expected = REGISTERED_RUNTIME_DIR / registered_shard_filename(morphology)
    if output is not None and output != expected:
        raise ValueError(f"registered shard output must be {expected}")
    return expected


def aggregate_registered_shards(
    shard_paths: Mapping[str, Path],
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    *,
    artifact_root: Path = Path("."),
) -> dict[str, Any]:
    _source_contract(source_paths)
    if set(shard_paths) != set(MORPHOLOGIES):
        raise ValueError("registered aggregation requires exact SO-101 and Panda shards")
    reports: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for morphology in MORPHOLOGIES:
        path = shard_paths[morphology]
        if path.name != registered_shard_filename(morphology):
            raise ValueError(f"source shard path is misattributed: {morphology}")
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get("identity", {}).get("morphology") != morphology:
            raise ValueError(f"source shard identity is misattributed: {morphology}")
        reports[morphology] = report
        source_hashes[morphology] = hashlib.sha256(raw).hexdigest()
    commits: set[str] = set()
    run_ids: set[str] = set()
    signatures: set[str] = set()
    runtimes = []
    intervals = []
    delivery_by_shard = {}
    evaluator_hash = file_sha256(Path(__file__))
    for position, morphology in enumerate(MORPHOLOGIES):
        report = reports[morphology]
        validate_shard_report(
            report,
            manifest,
            source_paths,
            require_registered=True,
            artifact_root=artifact_root,
        )
        provenance, run = report["provenance"], report["run"]
        if provenance["evaluator_sha256"] != evaluator_hash or run["schedule_position"] != position:
            raise ValueError("registered shards violate evaluator or morphology order")
        commits.add(provenance["git_commit"])
        run_ids.add(run["orchestrator_run_id"])
        signatures.add(provenance["runtime_signature_sha256"])
        runtimes.append(provenance["runtime"])
        intervals.append((_parse_utc(run["started_utc"]), _parse_utc(run["finished_utc"])))
        delivery_by_shard[morphology] = report["aggregate"]["delivery_gate"]
    if (
        len(commits) != 1
        or len(run_ids) != 1
        or not _valid_run_id(next(iter(run_ids), None))
        or len(signatures) != 1
        or any(runtime != runtimes[0] for runtime in runtimes)
        or intervals[0][1] > intervals[1][0]
    ):
        raise ValueError("registered shards violate common clean commit/runtime or nonoverlap")
    records = _flatten_condition_records([reports[value] for value in MORPHOLOGIES])
    metrics = aggregate_metrics(records)
    delivery = all(value["passed"] for value in delivery_by_shard.values())
    classification = classify_mechanism(metrics, delivery=delivery)
    summary = {
        "format_version": 1,
        "experiment": 43,
        "registered": True,
        "status": "completed_gripper_mechanism_assay",
        "scope": (
            "development-only descriptive mechanism assay; no inferential, population, "
            "closed-loop, admission, advancement, or final-split claim"
        ),
        "contract": {
            **_contract(True, None),
            "evaluator_sha256": evaluator_hash,
            "evaluator_git_commit": next(iter(commits)),
            "orchestrator_run_id": next(iter(run_ids)),
            "runtime_signature_sha256": next(iter(signatures)),
            "shards": 2,
            "expert_episodes": EXPECTED_EPISODES,
            "scenario_phase_units": EXPECTED_PHASE_UNITS,
            "endpoint_records": EXPECTED_ENDPOINTS,
            "scored_chunks": EXPECTED_SCORED_CHUNKS,
            "repeat_chunks": EXPECTED_REPEAT_CHUNKS,
            "policy_inferences": EXPECTED_INFERENCES,
            "contrast_observations": {
                contrast: EXPECTED_CONTRAST_OBSERVATIONS for contrast in CONTRASTS
            },
        },
        "source_shard_sha256": source_hashes,
        "delivery_gates_by_shard": delivery_by_shard,
        "delivery_gate_passed": delivery,
        "metrics": metrics,
        "mechanism_classification": classification,
        "inferential_claim": False,
        "closed_loop_success_claim": False,
        "admission_gate": None,
        "advancement_gate": None,
        "final_split_loaded": False,
    }
    if (
        len(records) != EXPECTED_SCORED_CHUNKS
        or any(
            sum(
                1
                for record in records
                if record["condition"] == numerator
                and any(
                    other["condition"] == denominator
                    and all(
                        other[key] == record[key]
                        for key in (
                            "seed",
                            "morphology",
                            "task",
                            "scenario_id",
                            "phase",
                            "endpoint",
                        )
                    )
                    for other in records
                )
            )
            != EXPECTED_CONTRAST_OBSERVATIONS
            for numerator, denominator in CONTRASTS.values()
        )
    ):
        raise ValueError("Experiment 043 contrast observation counts do not recompute")
    return summary


def validate_summary_report(
    summary: Mapping[str, Any],
    shard_paths: Mapping[str, Path],
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    *,
    artifact_root: Path = Path("."),
) -> None:
    expected = aggregate_registered_shards(
        shard_paths, manifest, source_paths, artifact_root=artifact_root
    )
    if summary != expected:
        raise ValueError("Experiment 043 summary does not independently recompute")


def _promote_registered_sources(
    source_paths: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
    *,
    destination_dir: Path = Path("reports"),
) -> None:
    if set(source_paths) != set(MORPHOLOGIES) or set(expected_sha256) != set(MORPHOLOGIES):
        raise ValueError("source promotion requires both registered shards and hashes")
    values = {morphology: source_paths[morphology].read_bytes() for morphology in MORPHOLOGIES}
    if any(
        hashlib.sha256(values[morphology]).hexdigest() != expected_sha256[morphology]
        for morphology in MORPHOLOGIES
    ):
        raise ValueError("staged source bytes changed after registered aggregation")
    destinations = {
        morphology: destination_dir / registered_shard_filename(morphology)
        for morphology in MORPHOLOGIES
    }
    conflicts = [
        destination
        for morphology, destination in destinations.items()
        if destination.exists() and destination.read_bytes() != values[morphology]
    ]
    if conflicts:
        raise FileExistsError(f"refusing conflicting promoted source: {conflicts[0]}")
    if any(source_paths[m].read_bytes() != values[m] for m in MORPHOLOGIES):
        raise ValueError("staged source bytes changed during promotion preflight")
    for morphology in MORPHOLOGIES:
        _write_exact_bytes(destinations[morphology], values[morphology])
    if any(source_paths[m].read_bytes() != values[m] for m in MORPHOLOGIES):
        raise ValueError("staged source bytes changed during promotion")


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="run the frozen Experiment 043 three-condition gripper evaluator"
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
    shard = subparsers.add_parser("shard", parents=[common])
    shard.add_argument("--morphology", choices=MORPHOLOGIES, required=True)
    shard.add_argument("--artifact-root", type=Path, default=Path("."))
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    shard.add_argument("--smoke", action="store_true")
    shard.add_argument("--limit-scenarios", type=_positive)
    run = subparsers.add_parser("run-registered", parents=[common])
    run.add_argument("--artifact-root", type=Path, default=Path("."))
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--shard-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    aggregate.add_argument("--output", type=Path, default=SUMMARY_PATH)
    aggregate.add_argument("--artifact-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "exp36_summary": args.exp36_summary,
        "exp42_summary": args.exp42_summary,
        "exp43_training_summary": args.exp43_training_summary,
        "exp42_so101_shard": args.exp42_so101_shard,
        "exp42_panda_shard": args.exp42_panda_shard,
    }


def _run_all_registered(
    args: argparse.Namespace,
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
) -> list[dict[str, Any]]:
    global _ACTIVE_ORCHESTRATOR
    outputs = [_registered_output(morphology, None) for morphology in MORPHOLOGIES]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "Experiment 043 is non-resumable; delete both staged shards before restart"
        )
    reports = []
    with registered_run_lock(REGISTERED_LOCK_PATH):
        if any(path.exists() for path in outputs):
            raise FileExistsError(
                "Experiment 043 staged shard appeared before schedule; delete both before restart"
            )
        orchestrator = _OrchestratorContext(uuid.uuid4().hex)
        if _ACTIVE_ORCHESTRATOR is not None:
            raise RuntimeError("registered Experiment 043 orchestrator is already active")
        _ACTIVE_ORCHESTRATOR = orchestrator
        try:
            for morphology, output in zip(MORPHOLOGIES, outputs, strict=True):
                reports.append(
                    evaluate_gripper_mechanism_shard(
                        definition,
                        manifest,
                        _source_paths(args),
                        output,
                        morphology=morphology,
                        device=args.device,
                        registered=True,
                        orchestrator=orchestrator,
                        root=args.artifact_root,
                    )
                )
            if orchestrator.next_position != len(MORPHOLOGIES):
                raise RuntimeError("registered Experiment 043 orchestrator did not complete")
        finally:
            orchestrator.active = False
            _ACTIVE_ORCHESTRATOR = None
    return reports


def main() -> None:
    args = parse_args()
    if args.command == "shard":
        if args.smoke is not True:
            raise ValueError("registered shards may only run through run-registered")
        if args.output.parent == REGISTERED_RUNTIME_DIR:
            raise ValueError("smoke output may not use the registered runtime directory")
    if args.command == "aggregate":
        if args.shard_dir != REGISTERED_RUNTIME_DIR:
            raise ValueError(
                f"registered aggregation shard directory must be {REGISTERED_RUNTIME_DIR}"
            )
        if args.output != SUMMARY_PATH:
            raise ValueError(f"registered aggregate output must be {SUMMARY_PATH}")
    registered = args.command != "shard"
    definition, manifest = load_development_contract(
        args.definition, args.manifest, registered=registered
    )
    if (
        definition.sha256 != DEFINITION_SHA256
        or file_sha256(manifest.path) != DEVELOPMENT_MANIFEST_SHA256
    ):
        raise ValueError("Experiment 043 benchmark contract does not match registration")
    source_paths = _source_paths(args)
    _source_contract(source_paths)
    if args.command == "shard":
        report = evaluate_gripper_mechanism_shard(
            definition,
            manifest,
            source_paths,
            args.output,
            morphology=args.morphology,
            device=args.device,
            registered=False,
            limit_scenarios=args.limit_scenarios,
            root=args.artifact_root,
        )
        print(json.dumps({"output": str(args.output), "status": report["status"]}, indent=2))
        return
    if args.command == "run-registered":
        reports = _run_all_registered(args, definition, manifest)
        print(json.dumps({"completed_shards": len(reports), "status": "completed"}, indent=2))
        return
    shard_paths = {
        morphology: args.shard_dir / registered_shard_filename(morphology)
        for morphology in MORPHOLOGIES
    }
    with registered_run_lock(REGISTERED_LOCK_PATH):
        summary = aggregate_registered_shards(
            shard_paths, manifest, source_paths, artifact_root=args.artifact_root
        )
        _preflight_final(args.output, summary)
        _promote_registered_sources(shard_paths, summary["source_shard_sha256"])
        _write_final(args.output, summary)
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
