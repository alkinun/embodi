from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Any, Callable, Iterable

import numpy as np
import torch

from ..joint_train import runtime_provenance
from ..processing import EmbodiProcessor
from ..record_so101 import atomic_write_json
from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv, runtime_scenario
from . import benchmark_geometry_diagnostics as diagnostics
from . import benchmark_geometry_projection as exp39
from .benchmark_geometry_evaluation import (
    CONFIG_SHA256,
    DATA_MANIFEST_SHA256,
    EXP36_SUMMARY_SHA256,
    _clipping,
    load_development_contract,
    load_geometry_policy,
    resolve_checkpoint,
)


DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
DEVELOPMENT_MANIFEST_SHA256 = (
    "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
)
EXP39_SUMMARY_SHA256 = "87bd3d30f350e3cda16a8ac91dfeeec5249b46ace97e42de860362d7bef5a8a8"
EXP39_EVALUATOR_SHA256 = "7617983a71166a9fa261ecb894e2ee6b02bde6d044c8a2b3edd27d17736c791c"
CONDITIONS = ("joint", "matched")
MORPHOLOGIES = ("so101", "panda")
TASKS = ("push_to_zone", "lift_object", "pick_place_bin")
SEEDS = (36001, 36002, 36003)
SCENARIO_INDICES = tuple(range(14, 21))
ARMS = ("baseline", "alignment")
EXECUTION_HORIZON = 4
IK_ITERATIONS = 20
ALIGNMENT_WEIGHTS = (1.0, 0.75, 0.5, 0.25)
TRANSFORM_THRESHOLDS = {
    "translation_m": 1e-6,
    "rotation_deg": 1e-4,
    "gripper_effective": 1e-6,
}
REGISTERED_RUNTIME_DIR = Path("reports/exp40-runtime")
REGISTERED_LOCK_NAME = ".registered.lock"
LOCK_IDENTITY = "exp40-linux-flock-exclusive-v1"
SCHEDULE_IDENTITY = "condition-major-seed-major-morphology-v1"
REGISTERED_SCHEDULE = tuple(
    (condition, seed, morphology)
    for condition in CONDITIONS
    for seed in SEEDS
    for morphology in MORPHOLOGIES
)
EMPTY_OUTCOME_CHAIN_SHA256 = hashlib.sha256(b"").hexdigest()
SNAPSHOT_SCHEMA = "embodi-exp40-treatment-onset-v1"
SNAPSHOT_ARRAYS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
RUNTIME_FIELDS = exp39.RUNTIME_FIELDS
ALIGNMENT_PROTOCOL = {
    "camera": "top",
    "resolution": [640, 480],
    "state_path": "forward_kinematics_canonical",
    "native_state": None,
    "predictor": "predict_canonical_action_chunk",
    "chunk_steps": 32,
    "execution_horizon_steps": 4,
    "ik_iterations": 20,
    "frequency_hz": 30,
    "maximum_episode_steps": 500,
    "objective": "regression",
    "expert": None,
    "geometry_feasibility_projection": False,
    "initial_input_equivalence": exp39.PROJECTION_PROTOCOL["initial_input_equivalence"],
    "shared_initial_policy_input": exp39.PROJECTION_PROTOCOL["initial_policy_input"],
    "first_chunk_max_abs_tolerance": exp39.FIRST_CHUNK_MAX_ABS,
    "execute_source_arm_chunk_in_both_arms": True,
    "common_prefix_decode_once_on_source": True,
    "shared_first_treatment_replan": True,
    "later_replans_use_independent_live_inputs": True,
    "telemetry_population": "executed commands only",
    "clipping_denominator": "executed commands",
    "treatment_onset_snapshot_schema": SNAPSHOT_SCHEMA,
    "alignment_weights": list(ALIGNMENT_WEIGHTS),
    "transform_thresholds": TRANSFORM_THRESHOLDS,
    "tail_start": 4,
}


class AlignmentTreatmentError(RuntimeError):
    def __init__(self, reason: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason.replace("_", " "))
        self.reason = reason
        self.details = {} if details is None else details


class _CommandPhaseError(RuntimeError):
    def __init__(
        self,
        phase: str,
        lead: int,
        reason: str,
        error: BaseException,
        completed: int,
    ) -> None:
        super().__init__(reason)
        self.phase = phase
        self.lead = lead
        self.reason = reason
        self.error = error
        self.completed = completed


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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _outcome_chain(outcomes: list[dict[str, Any]]) -> str:
    digest = bytes.fromhex(EMPTY_OUTCOME_CHAIN_SHA256)
    for outcome in outcomes:
        encoded = json.dumps(_jsonable(outcome), sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(digest + encoded).digest()
    return digest.hex()


def _append_outcome(report: dict[str, Any], outcome: dict[str, Any]) -> None:
    report["outcomes"].append(outcome)
    report["outcome_chain_sha256"] = _outcome_chain(report["outcomes"])


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Experiment 040 timestamps must be UTC ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("Experiment 040 timestamp is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("Experiment 040 timestamp must use UTC")
    return parsed


def _runtime_provenance(device: str) -> dict[str, Any]:
    provenance = runtime_provenance(require_clean=False)
    cuda_device_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        cuda_device_name = torch.cuda.get_device_name(0 if index is None else index)
    runtime = {
        "device": device,
        "python": provenance["python"],
        "torch": provenance["torch"],
        "numpy": np.__version__,
        "mujoco": version("mujoco"),
        "cuda_runtime": provenance["cuda"],
        "cudnn": provenance["cudnn"],
        "cuda_device_name": cuda_device_name,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    provenance["runtime"] = runtime
    provenance["runtime_signature_sha256"] = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance["evaluator_sha256"] = file_sha256(Path(__file__))
    return provenance


@contextmanager
def registered_run_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another registered Experiment 040 shard holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run_metadata(
    device: str,
    *,
    registered: bool,
    condition: str,
    seed: int,
    morphology: str,
    lock_path: Path | None,
) -> dict[str, Any]:
    return {
        "started_utc": _utc_now(),
        "finished_utc": None,
        "device": device,
        "lock_identity": LOCK_IDENTITY if registered else None,
        "lock_path": str(lock_path) if registered and lock_path is not None else None,
        "schedule_identity": SCHEDULE_IDENTITY if registered else "smoke-unregistered",
        "schedule_position": (
            REGISTERED_SCHEDULE.index((condition, seed, morphology)) if registered else None
        ),
    }


def validate_exp39_summary(path: Path) -> dict[str, Any]:
    if file_sha256(path) != EXP39_SUMMARY_SHA256:
        raise ValueError("Experiment 039 summary hash does not match registration")
    if file_sha256(Path(exp39.__file__)) != EXP39_EVALUATOR_SHA256:
        raise ValueError("imported Experiment 039 evaluator hash does not match registration")
    summary = json.loads(path.read_text())
    _validate_exp39_summary_contract(summary)
    return summary


def _validate_exp39_summary_contract(summary: dict[str, Any]) -> None:
    contract = summary.get("contract", {})
    if (
        summary.get("experiment") != 39
        or summary.get("registered") is not True
        or summary.get("status") != "completed_causal_diagnostic"
        or contract.get("definition_sha256") != DEFINITION_SHA256
        or contract.get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
        or contract.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
        or contract.get("evaluator_sha256") != EXP39_EVALUATOR_SHA256
        or contract.get("shards") != 12
        or contract.get("pairs") != 252
        or contract.get("episodes") != 504
        or contract.get("final_split_loaded") is not False
    ):
        raise ValueError("Experiment 039 summary is not the frozen completed diagnostic")


def alignment_scenarios(
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
        expected = [
            f"development-{morphology}-{task}-{index:04d}"
            for task in TASKS
            for index in SCENARIO_INDICES
        ]
        if [scenario.scenario_id for scenario in selected] != expected:
            raise ValueError("registered Experiment 040 requires exact indices 0014 through 0020")
        for task in TASKS:
            axes = [
                name
                for scenario in selected
                if scenario.task == task
                for name, value in scenario.factors.items()
                if str(value.get("bin", "")).startswith("development:")
            ]
            if len(axes) != 7 or len(set(axes)) != 7:
                raise ValueError("each Experiment 040 cell must cover all seven factor axes")
    return selected if limit_scenarios is None else selected[:limit_scenarios]


def source_condition(condition: str, morphology: str) -> str:
    return exp39.source_condition(condition, morphology)


def pair_arm_order(
    scenario_ordinal: int, condition: str, seed: int, morphology: str
) -> tuple[str, str]:
    parity = (
        scenario_ordinal
        + CONDITIONS.index(condition)
        + SEEDS.index(seed)
        + MORPHOLOGIES.index(morphology)
    ) % 2
    return ARMS if parity == 0 else tuple(reversed(ARMS))


def _rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def compose_absolute_targets(anchor: np.ndarray, chunk: np.ndarray) -> dict[str, np.ndarray]:
    anchor = np.asarray(anchor, dtype=np.float64).reshape(10)
    chunk = np.asarray(chunk, dtype=np.float64).reshape(-1, 10)
    anchor_rotation = exp39.rotation_matrix_from_6d(anchor[3:9])
    rotations = np.stack(
        [anchor_rotation @ exp39.rotation_matrix_from_6d(command[3:9]) for command in chunk]
    )
    return {
        "translation_m": anchor[:3] + chunk[:, :3],
        "rotation": rotations,
        "gripper_raw": chunk[:, 9].copy(),
        "gripper_effective": np.clip(chunk[:, 9], 0.0, 1.0),
    }


def alignment_transform(
    anchor: np.ndarray,
    previous_targets: dict[str, np.ndarray],
    chunk: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    anchor = np.asarray(anchor, dtype=np.float64).reshape(10)
    original = np.asarray(chunk)
    if original.shape != (32, 1, 10) or not np.isfinite(original).all():
        raise AlignmentTreatmentError("invalid_policy_chunk")
    transformed = original.copy()
    new_targets = compose_absolute_targets(anchor, original)
    old_position = np.asarray(previous_targets["translation_m"], dtype=np.float64)[4]
    old_rotation = np.asarray(previous_targets["rotation"], dtype=np.float64)[4]
    old_gripper = float(np.asarray(previous_targets["gripper_effective"])[4])
    translation_correction = old_position - new_targets["translation_m"][0]
    rotation_correction = old_rotation @ new_targets["rotation"][0].T
    gripper_correction = old_gripper - float(new_targets["gripper_effective"][0])
    anchor_rotation = exp39.rotation_matrix_from_6d(anchor[3:9])
    intended_position = new_targets["translation_m"].copy()
    intended_rotation = new_targets["rotation"].copy()
    intended_gripper = new_targets["gripper_effective"].copy()
    for lead, weight in enumerate(ALIGNMENT_WEIGHTS):
        intended_position[lead] += weight * translation_correction
        intended_rotation[lead] = (
            exp39.scale_rotation_shortest_path(rotation_correction, weight)
            @ intended_rotation[lead]
        )
        intended_gripper[lead] += weight * gripper_correction
        transformed[lead, 0, :3] = intended_position[lead] - anchor[:3]
        transformed[lead, 0, 3:9] = exp39.canonical_rotation_6d(
            anchor_rotation.T @ intended_rotation[lead]
        )
        transformed[lead, 0, 9] = intended_gripper[lead]
    tail_unchanged = np.array_equal(transformed[4:], original[4:])
    if not tail_unchanged:
        raise AlignmentTreatmentError("tail_changed")
    intended = compose_absolute_targets(anchor, transformed)
    lead_zero = {
        "translation_m": float(np.linalg.norm(intended["translation_m"][0] - old_position)),
        "rotation_deg": _rotation_error_degrees(intended["rotation"][0], old_rotation),
        "gripper_effective": float(
            abs(intended["gripper_effective"][0] - old_gripper)
        ),
    }
    if any(lead_zero[name] > threshold for name, threshold in TRANSFORM_THRESHOLDS.items()):
        raise AlignmentTreatmentError("transform_threshold_exceeded", details=lead_zero)
    details = {
        "tail_unchanged": tail_unchanged,
        "lead_zero_errors": lead_zero,
        "translation_correction_m": float(np.linalg.norm(translation_correction)),
        "rotation_correction_deg": _rotation_error_degrees(np.eye(3), rotation_correction),
        "gripper_correction_effective": abs(gripper_correction),
    }
    return transformed, intended, details


def capture_treatment_onset(
    env: Any, milestones: dict[str, dict[str, int | bool | None]]
) -> dict[str, Any]:
    arrays = {}
    for name in SNAPSHOT_ARRAYS:
        value = np.asarray(getattr(env.data, name))
        arrays[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
    scalars = {
        "simulation_time": float(env.data.time),
        "control_time": float(env.control_time),
        "dwell": int(env.dwell),
        "lifted": bool(env.lifted),
        "success": bool(env.success),
    }
    milestone_copy = json.loads(json.dumps(_jsonable(milestones)))
    body = {
        "schema": SNAPSHOT_SCHEMA,
        "arrays": arrays,
        "scalars": scalars,
        "milestones": milestone_copy,
    }
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def _status_record(status: Any) -> dict[str, Any]:
    return {
        "success": bool(status.success),
        "terminal": bool(status.terminal),
        "stage": status.stage,
        "failure_reason": status.failure_reason,
    }


def _validate_snapshot(value: Any, task: str, common_prefix_commands: int) -> None:
    if (
        type(common_prefix_commands) is not int
        or not 1 <= common_prefix_commands <= EXECUTION_HORIZON
    ):
        raise ValueError("treatment-onset common-prefix length is invalid")
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "arrays",
        "scalars",
        "milestones",
        "sha256",
    }:
        raise ValueError("treatment-onset snapshot schema is invalid")
    if value["schema"] != SNAPSHOT_SCHEMA or set(value["arrays"]) != set(SNAPSHOT_ARRAYS):
        raise ValueError("treatment-onset snapshot identity is invalid")
    for array in value["arrays"].values():
        if (
            set(array) != {"dtype", "shape", "sha256"}
            or not isinstance(array["dtype"], str)
            or not isinstance(array["shape"], list)
            or any(type(size) is not int or size < 0 for size in array["shape"])
            or not _valid_sha256(array["sha256"])
        ):
            raise ValueError("treatment-onset array record is invalid")
    scalars = value["scalars"]
    if (
        set(scalars) != {"simulation_time", "control_time", "dwell", "lifted", "success"}
        or not np.isfinite([scalars["simulation_time"], scalars["control_time"]]).all()
        or type(scalars["dwell"]) is not int
        or scalars["dwell"] < 0
        or type(scalars["lifted"]) is not bool
        or type(scalars["success"]) is not bool
    ):
        raise ValueError("treatment-onset scalar record is invalid")
    milestones = value["milestones"]
    if not isinstance(milestones, dict) or tuple(milestones) != exp39._milestone_names(task):
        raise ValueError("treatment-onset milestone schema is invalid")
    attained = []
    for milestone in milestones.values():
        if (
            not isinstance(milestone, dict)
            or set(milestone) != {"attained", "first_step"}
            or type(milestone["attained"]) is not bool
            or milestone["attained"] != (type(milestone["first_step"]) is int)
            or (
                type(milestone["first_step"]) is int
                and not 1 <= milestone["first_step"] <= common_prefix_commands
            )
        ):
            raise ValueError("treatment-onset milestone value is invalid")
        attained.append(milestone["attained"])
    if attained != sorted(attained, reverse=True):
        raise ValueError("treatment-onset milestones must form an ordered prefix")
    body = {
        name: value[name] for name in ("schema", "arrays", "scalars", "milestones")
    }
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if value["sha256"] != expected:
        raise ValueError("treatment-onset snapshot hash does not recompute")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _metric_group() -> dict[str, dict[str, dict[str, int | float]]]:
    names = (
        "raw_gripper_out_of_range",
        "pre_translation_m",
        "pre_rotation_deg",
        "pre_gripper_effective",
        "intended_translation_m",
        "intended_rotation_deg",
        "intended_gripper_effective",
        "realized_translation_m",
        "realized_rotation_deg",
        "realized_gripper_effective",
        "correction_translation_m",
        "correction_rotation_deg",
        "correction_gripper_effective",
        "reconstruction_translation_m",
        "reconstruction_rotation_deg",
        "reconstruction_gripper_native",
        "clipping_error_native",
        "clipped",
    )
    return {
        name: {str(lead): diagnostics.compact_accumulator() for lead in range(EXECUTION_HORIZON)}
        for name in names
    }


def _update_metric(metrics: dict[str, Any], name: str, lead: int, value: float) -> None:
    diagnostics.update_compact(metrics[name][str(lead)], float(value))


def _merge_metric_update(destination: dict[str, Any], source: dict[str, Any]) -> None:
    for name in destination:
        for lead in destination[name]:
            destination[name][lead] = diagnostics.merge_compact(
                (destination[name][lead], source[name][lead])
            )


def _audit_baseline_chunk(context: dict[str, Any], raw_chunk: np.ndarray) -> np.ndarray:
    raw_copy = np.asarray(raw_chunk).copy()
    final_chunk = raw_copy.copy()
    record = {
        "chunk_index": context["chunks"] - 1,
        "raw_sha256": _array_sha256(raw_copy),
        "final_sha256": _array_sha256(final_chunk),
        "equal": bool(np.array_equal(raw_copy, final_chunk)),
    }
    previous = bytes.fromhex(context["baseline_chunk_hash_chain_sha256"])
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    context["baseline_chunk_hash_chain_sha256"] = hashlib.sha256(
        previous + encoded
    ).hexdigest()
    context["baseline_chunk_audit_records"].append(record)
    if not record["equal"]:
        raise RuntimeError("baseline raw chunk changed before decoding")
    return final_chunk


def _gripper_native_scale(env: Any) -> float:
    return 35.0 if env.morphology.id == "so101" else 0.08


def _decode_chunk(
    env: Any, chunk: np.ndarray, count: int, *, structured: bool = False
) -> list[np.ndarray]:
    natives = []
    for lead, command in enumerate(chunk[:count]):
        try:
            native = np.asarray(
                env.native_action_from_canonical(command, iterations=IK_ITERATIONS),
                dtype=np.float32,
            )
            if native.shape != (env.morphology.native_action_dim,) or not np.isfinite(native).all():
                raise ValueError("IK returned an invalid native action")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            if structured:
                raise _CommandPhaseError(
                    "decode", lead, "transformed_decode_failure", error, len(natives)
                ) from error
            raise
        natives.append(native)
    return natives


def _prepare_telemetry(
    env: Any,
    anchor: np.ndarray,
    original: np.ndarray,
    executed: np.ndarray,
    natives: list[np.ndarray],
    *,
    previous_targets: dict[str, np.ndarray] | None,
    intended_targets: dict[str, np.ndarray],
    alignment_applied: bool,
    structured: bool = False,
) -> list[dict[str, Any]]:
    raw_targets = compose_absolute_targets(anchor, original)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(10)
    records = []
    for lead, native in enumerate(natives):
        try:
            metrics = _metric_group()
            realized = np.asarray(env.canonical_arm_action(native), dtype=np.float32)
            if realized.shape != (1, 10) or not np.isfinite(realized).all():
                raise ValueError("FK returned an invalid canonical action")
            realized_targets = compose_absolute_targets(anchor, realized)
            requested = np.asarray(executed[lead], dtype=np.float64).reshape(10)
            realized_command = realized.reshape(10)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            if structured:
                raise _CommandPhaseError(
                    "realized_fk", lead, "realized_fk_preparation_failure", error, len(records)
                ) from error
            raise
        _update_metric(
            metrics,
            "raw_gripper_out_of_range",
            lead,
            abs(float(original[lead, 0, 9]) - float(np.clip(original[lead, 0, 9], 0, 1))),
        )
        _update_metric(
            metrics,
            "reconstruction_translation_m",
            lead,
            np.linalg.norm(requested[:3] - realized_command[:3]),
        )
        _update_metric(
            metrics,
            "reconstruction_rotation_deg",
            lead,
            _rotation_error_degrees(
                exp39.rotation_matrix_from_6d(requested[3:9]),
                exp39.rotation_matrix_from_6d(realized_command[3:9]),
            ),
        )
        _update_metric(
            metrics,
            "reconstruction_gripper_native",
            lead,
            abs(float(requested[9] - realized_command[9])) * _gripper_native_scale(env),
        )
        clipped, clipping_error = _clipping(env, native)
        _update_metric(metrics, "clipping_error_native", lead, clipping_error)
        _update_metric(metrics, "clipped", lead, float(clipped))
        if previous_targets is None:
            records.append(metrics)
            continue
        old_position = previous_targets["translation_m"][4]
        old_rotation = previous_targets["rotation"][4]
        old_gripper = previous_targets["gripper_effective"][4]
        for prefix, targets in (
            ("pre", raw_targets),
            ("intended", intended_targets),
            ("realized", realized_targets),
        ):
            _update_metric(
                metrics,
                f"{prefix}_translation_m",
                lead,
                np.linalg.norm(targets["translation_m"][0 if prefix == "realized" else lead] - old_position),
            )
            _update_metric(
                metrics,
                f"{prefix}_rotation_deg",
                lead,
                _rotation_error_degrees(
                    targets["rotation"][0 if prefix == "realized" else lead], old_rotation
                ),
            )
            _update_metric(
                metrics,
                f"{prefix}_gripper_effective",
                lead,
                abs(
                    float(targets["gripper_effective"][0 if prefix == "realized" else lead])
                    - float(old_gripper)
                ),
            )
        correction = {
            "translation_m": 0.0,
            "rotation_deg": 0.0,
            "gripper_effective": 0.0,
        }
        if alignment_applied:
            correction = {
                "translation_m": float(
                    np.linalg.norm(
                        intended_targets["translation_m"][lead]
                        - raw_targets["translation_m"][lead]
                    )
                ),
                "rotation_deg": _rotation_error_degrees(
                    intended_targets["rotation"][lead], raw_targets["rotation"][lead]
                ),
                "gripper_effective": abs(
                    float(intended_targets["gripper_effective"][lead])
                    - float(raw_targets["gripper_effective"][lead])
                ),
            }
        for name, value in correction.items():
            _update_metric(metrics, f"correction_{name}", lead, value)
        records.append(metrics)
    return records


def _new_episode_context(
    env: Any,
    source_chunk: np.ndarray,
    common_natives: list[np.ndarray],
    *,
    maximum_steps: int,
) -> dict[str, Any]:
    anchor = np.asarray(env.canonical_state(), dtype=np.float32).copy()
    metrics = _metric_group()
    source_targets = compose_absolute_targets(anchor, source_chunk)
    telemetry = _prepare_telemetry(
        env,
        anchor,
        source_chunk,
        source_chunk,
        common_natives,
        previous_targets=None,
        intended_targets=source_targets,
        alignment_applied=False,
    )
    milestones = {
        name: {"attained": False, "first_step": None}
        for name in exp39._milestone_names(env.task.id)
    }
    status = None
    steps = 0
    executed_by_lead = {str(lead): 0 for lead in range(EXECUTION_HORIZON)}
    post_prefix_executed_by_lead = {
        str(lead): 0 for lead in range(EXECUTION_HORIZON)
    }
    for lead, (native, record) in enumerate(zip(common_natives, telemetry, strict=True)):
        env.apply_action(native)
        _merge_metric_update(metrics, record)
        executed_by_lead[str(lead)] += 1
        status = env.step_control_period(30.0)
        steps += 1
        exp39.update_ordered_milestones(milestones, env, env.diagnostic_state(), steps)
        if status.terminal or steps >= maximum_steps:
            break
    if status is None:
        raise RuntimeError("common prefix must execute at least one command")
    onset = capture_treatment_onset(env, milestones)
    return {
        "env": env,
        "source_chunk": source_chunk,
        "previous_targets": source_targets,
        "metrics": metrics,
        "milestones": milestones,
        "status": status,
        "onset_snapshot": onset,
        "onset_status": _status_record(status),
        "steps": steps,
        "commands": steps,
        "decoded_commands": len(common_natives),
        "prepared_commands": len(common_natives),
        "executed_commands_by_lead": executed_by_lead,
        "post_prefix_executed_commands_by_lead": post_prefix_executed_by_lead,
        "chunks": 1,
        "common_prefix_commands": steps,
        "maximum_steps": maximum_steps,
        "correction_opportunities": 0,
        "delivered_corrections": 0,
        "tail_unchanged_checks": 0,
        "tail_unchanged_passes": 0,
        "transform_lead_zero_maxima": {name: 0.0 for name in TRANSFORM_THRESHOLDS},
        "baseline_chunk_audit_records": [],
        "baseline_chunk_hash_chain_sha256": EMPTY_OUTCOME_CHAIN_SHA256,
        "first_treatment_source_chunk_sha256": None,
    }


def _finalize_episode(
    context: dict[str, Any],
    arm: str,
    *,
    treatment_failure: bool = False,
    treatment_failure_reason: str | None = None,
    treatment_failure_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = context["status"]
    success = None if treatment_failure else bool(status.success)
    failure_reason = None
    if not treatment_failure and not success:
        failure_reason = (
            status.failure_reason
            or ("maximum_episode_steps" if context["steps"] >= context["maximum_steps"] else "task_terminal")
        )
    milestones = context["milestones"]
    progression = sum(value["attained"] for value in milestones.values()) / len(milestones)
    clipped = sum(
        int(value["sum"]) for value in context["metrics"]["clipped"].values()
    )
    clipping_maximum = max(
        value["maximum"] for value in context["metrics"]["clipping_error_native"].values()
    )
    return {
        "arm": arm,
        "success": success,
        "steps": context["steps"],
        "commands": context["commands"],
        "decoded_commands": context["decoded_commands"],
        "prepared_commands": context["prepared_commands"],
        "executed_commands_by_lead": context["executed_commands_by_lead"],
        "post_prefix_executed_commands_by_lead": context[
            "post_prefix_executed_commands_by_lead"
        ],
        "chunks": context["chunks"],
        "final_stage": status.stage,
        "failure_reason": failure_reason,
        "common_prefix_commands": context["common_prefix_commands"],
        "treatment_onset_snapshot": context["onset_snapshot"],
        "treatment_onset_status": context["onset_status"],
        "terminal_before_treatment": bool(context["onset_status"]["terminal"]),
        "metrics_by_lead": context["metrics"],
        "clipped_commands": clipped,
        "clipping_rate": (
            clipped / context["commands"] if context["commands"] else 0.0
        ),
        "maximum_clipping_error_native": clipping_maximum,
        "correction_opportunities": context["correction_opportunities"],
        "delivered_corrections": context["delivered_corrections"],
        "tail_unchanged_checks": context["tail_unchanged_checks"],
        "tail_unchanged_passes": context["tail_unchanged_passes"],
        "transform_lead_zero_maxima": context["transform_lead_zero_maxima"],
        "baseline_chunk_equality_checks": len(context["baseline_chunk_audit_records"]),
        "baseline_chunk_equality_passes": sum(
            record["equal"] for record in context["baseline_chunk_audit_records"]
        ),
        "baseline_chunk_audit_records": context["baseline_chunk_audit_records"],
        "baseline_chunk_hash_chain_sha256": context[
            "baseline_chunk_hash_chain_sha256"
        ],
        "first_treatment_source_chunk_sha256": context[
            "first_treatment_source_chunk_sha256"
        ],
        "milestones": milestones,
        "ordered_progression": progression,
        "treatment_failure": treatment_failure,
        "treatment_failure_reason": treatment_failure_reason,
        "treatment_failure_details": treatment_failure_details,
    }


def _continue_episode(
    context: dict[str, Any],
    policy: Any,
    processor: Any,
    device: str,
    *,
    arm: str,
    first_replan_snapshot: exp39.PolicyInput | None = None,
    first_replan_chunk: np.ndarray | None = None,
) -> dict[str, Any]:
    if context["onset_status"]["terminal"]:
        return _finalize_episode(context, arm)
    env = context["env"]
    policy.reset()
    first_replan_pending = first_replan_chunk is not None
    while context["steps"] < context["maximum_steps"]:
        context["chunks"] += 1
        if first_replan_pending:
            if first_replan_snapshot is None:
                raise RuntimeError("shared first treatment replan is missing its snapshot")
            snapshot = first_replan_snapshot
            raw_chunk = np.asarray(first_replan_chunk)
            context["first_treatment_source_chunk_sha256"] = _array_sha256(raw_chunk)
            first_replan_pending = False
        else:
            # Input capture and prediction failures are infrastructure failures in both arms.
            snapshot = exp39.capture_policy_input(env)
            raw_chunk = exp39.predict_from_input(snapshot, policy, processor, device)
        anchor = snapshot.canonical_state.copy()
        if arm == "baseline":
            executed_chunk = _audit_baseline_chunk(context, raw_chunk)
            intended = compose_absolute_targets(anchor, executed_chunk)
        else:
            try:
                context["correction_opportunities"] += 1
                executed_chunk, intended, transform = alignment_transform(
                    anchor, context["previous_targets"], raw_chunk
                )
                context["tail_unchanged_checks"] += 1
                context["tail_unchanged_passes"] += int(transform["tail_unchanged"])
                for name, value in transform["lead_zero_errors"].items():
                    context["transform_lead_zero_maxima"][name] = max(
                        context["transform_lead_zero_maxima"][name], value
                    )
            except AlignmentTreatmentError as error:
                return _finalize_episode(
                    context,
                    arm,
                    treatment_failure=True,
                    treatment_failure_reason=error.reason,
                    treatment_failure_details={
                        "phase": "transform",
                        "chunk_index": context["chunks"] - 1,
                        "lead": None,
                        "reason": error.reason,
                        "details": error.details,
                    },
                )
        count = min(EXECUTION_HORIZON, context["maximum_steps"] - context["steps"])
        try:
            natives = _decode_chunk(
                env, executed_chunk, count, structured=arm == "alignment"
            )
        except _CommandPhaseError as error:
            context["decoded_commands"] += error.completed
            return _finalize_episode(
                context,
                arm,
                treatment_failure=True,
                treatment_failure_reason=error.reason,
                treatment_failure_details={
                    "phase": error.phase,
                    "chunk_index": context["chunks"] - 1,
                    "lead": error.lead,
                    "reason": error.reason,
                    "details": {"error": repr(error.error)},
                },
            )
        context["decoded_commands"] += len(natives)
        try:
            telemetry = _prepare_telemetry(
                env,
                anchor,
                raw_chunk,
                executed_chunk,
                natives,
                previous_targets=context["previous_targets"],
                intended_targets=intended,
                alignment_applied=arm == "alignment",
                structured=arm == "alignment",
            )
        except _CommandPhaseError as error:
            context["prepared_commands"] += error.completed
            return _finalize_episode(
                context,
                arm,
                treatment_failure=True,
                treatment_failure_reason=error.reason,
                treatment_failure_details={
                    "phase": error.phase,
                    "chunk_index": context["chunks"] - 1,
                    "lead": error.lead,
                    "reason": error.reason,
                    "details": {"error": repr(error.error)},
                },
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            # Baseline preparation failures are infrastructure failures.
            raise
        context["prepared_commands"] += len(natives)
        if arm == "alignment":
            context["delivered_corrections"] += 1
        for lead, (native, record) in enumerate(zip(natives, telemetry, strict=True)):
            env.apply_action(native)
            _merge_metric_update(context["metrics"], record)
            context["executed_commands_by_lead"][str(lead)] += 1
            context["post_prefix_executed_commands_by_lead"][str(lead)] += 1
            status = env.step_control_period(30.0)
            context["status"] = status
            context["steps"] += 1
            context["commands"] += 1
            exp39.update_ordered_milestones(
                context["milestones"], env, env.diagnostic_state(), context["steps"]
            )
            if status.terminal:
                break
        context["previous_targets"] = intended
        if context["status"].terminal:
            break
    return _finalize_episode(context, arm)


def run_alignment_pair(
    environments: dict[str, Any],
    policy: Any,
    processor: Any,
    device: str,
    *,
    order: tuple[str, str],
    maximum_steps: int = 500,
) -> dict[str, Any]:
    snapshots = {}
    for arm in order:
        snapshots[arm] = exp39.capture_policy_input(environments[arm])
    input_comparison = exp39.compare_initial_inputs(
        snapshots["baseline"], snapshots["alignment"]
    )
    if not input_comparison["initial_input_equivalent"]:
        raise RuntimeError("paired initial policy inputs are not equivalent")
    source_arm = order[0]
    anchor = snapshots[source_arm]
    chunks = {}
    for arm in order:
        policy.reset()
        chunks[arm] = exp39.predict_from_input(anchor, policy, processor, device)
    chunk_difference = float(np.max(np.abs(chunks["baseline"] - chunks["alignment"])))
    if not np.isfinite(chunk_difference) or chunk_difference > exp39.FIRST_CHUNK_MAX_ABS:
        raise RuntimeError("paired first policy chunks exceed the repeatability tolerance")
    source_chunk = chunks[source_arm]
    prefix_count = min(EXECUTION_HORIZON, maximum_steps)
    common_natives = _decode_chunk(environments[source_arm], source_chunk, prefix_count)
    contexts = {
        arm: _new_episode_context(
            environments[arm], source_chunk, common_natives, maximum_steps=maximum_steps
        )
        for arm in order
    }
    first, second = contexts[order[0]], contexts[order[1]]
    onset_equal = (
        first["onset_snapshot"] == second["onset_snapshot"]
        and first["onset_status"] == second["onset_status"]
        and first["common_prefix_commands"] == second["common_prefix_commands"]
    )
    if not onset_equal:
        raise RuntimeError("paired treatment-onset simulator state or status does not match")
    terminal_before_treatment = bool(first["onset_status"]["terminal"])
    if terminal_before_treatment:
        onset_policy_evidence = {
            "applicable": False,
            "actual_inputs": None,
            "source_arm": None,
            "anchor_hashes": None,
            "input_comparison": None,
            "chunks": None,
            "chunk_max_abs_difference": None,
            "source_raw_chunk_sha256": None,
            "consumed_source_raw_chunk_sha256": None,
        }
        outcomes = {
            arm: _continue_episode(contexts[arm], policy, processor, device, arm=arm)
            for arm in order
        }
    else:
        onset_inputs = {
            arm: exp39.capture_policy_input(environments[arm]) for arm in order
        }
        onset_comparison = exp39.compare_initial_inputs(
            onset_inputs["baseline"], onset_inputs["alignment"]
        )
        if not onset_comparison["initial_input_equivalent"]:
            raise RuntimeError("paired treatment-onset policy inputs are not equivalent")
        onset_source_arm = order[0]
        onset_anchor = onset_inputs[onset_source_arm]
        onset_chunks = {}
        for arm in order:
            policy.reset()
            onset_chunks[arm] = exp39.predict_from_input(
                onset_anchor, policy, processor, device
            )
        onset_chunk_difference = float(
            np.max(np.abs(onset_chunks["baseline"] - onset_chunks["alignment"]))
        )
        if (
            not np.isfinite(onset_chunk_difference)
            or onset_chunk_difference > exp39.FIRST_CHUNK_MAX_ABS
        ):
            raise RuntimeError(
                "paired treatment-onset chunks exceed the repeatability tolerance"
            )
        onset_source_chunk = onset_chunks[onset_source_arm]
        onset_source_hash = _array_sha256(onset_source_chunk)
        outcomes = {
            arm: _continue_episode(
                contexts[arm],
                policy,
                processor,
                device,
                arm=arm,
                first_replan_snapshot=onset_inputs[arm],
                first_replan_chunk=onset_source_chunk,
            )
            for arm in order
        }
        consumed = {outcomes[arm]["first_treatment_source_chunk_sha256"] for arm in ARMS}
        if consumed != {onset_source_hash}:
            raise RuntimeError("arms did not consume the shared treatment-onset source chunk")
        onset_policy_evidence = {
            "applicable": True,
            "actual_inputs": {arm: onset_inputs[arm].hashes for arm in ARMS},
            "source_arm": onset_source_arm,
            "anchor_hashes": onset_anchor.hashes,
            "input_comparison": onset_comparison,
            "chunks": {arm: _array_sha256(onset_chunks[arm]) for arm in ARMS},
            "chunk_max_abs_difference": onset_chunk_difference,
            "source_raw_chunk_sha256": onset_source_hash,
            "consumed_source_raw_chunk_sha256": {
                arm: outcomes[arm]["first_treatment_source_chunk_sha256"]
                for arm in ARMS
            },
        }
    return {
        "arm_order": list(order),
        "initial_inputs": {arm: snapshots[arm].hashes for arm in ARMS},
        "initial_policy_input_source_arm": source_arm,
        "initial_policy_input_anchor_hashes": anchor.hashes,
        **input_comparison,
        "first_chunks": {arm: _array_sha256(chunks[arm]) for arm in ARMS},
        "first_chunk_max_abs_difference": chunk_difference,
        "source_chunk_sha256": _array_sha256(source_chunk),
        "common_prefix_native_sha256": [_array_sha256(native) for native in common_natives],
        "treatment_onset_equal": onset_equal,
        "treatment_onset_snapshot": first["onset_snapshot"],
        "treatment_onset_status": first["onset_status"],
        "common_prefix_commands": first["common_prefix_commands"],
        "terminal_before_treatment": terminal_before_treatment,
        "treatment_onset_policy": onset_policy_evidence,
        "arms": outcomes,
    }


def _validate_compact(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"count", "sum", "sum_squared", "maximum"}:
        raise ValueError("alignment compact accumulator schema is invalid")
    count = value["count"]
    numbers = [value[name] for name in ("sum", "sum_squared", "maximum")]
    if type(count) is not int or count < 0 or not np.isfinite(numbers).all() or min(numbers) < 0:
        raise ValueError("alignment compact accumulator values are invalid")
    if count == 0 and numbers != [0.0, 0.0, 0.0]:
        raise ValueError("empty alignment accumulators must be zero")
    if count > 0 and (
        value["maximum"] > value["sum"] + 1e-12
        or value["sum"] > count * value["maximum"] + 1e-12
        or value["sum_squared"] + 1e-12 < value["sum"] ** 2 / count
        or value["sum_squared"] > value["maximum"] * value["sum"] + 1e-12
    ):
        raise ValueError("alignment compact accumulator statistics are inconsistent")


def _validate_status(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"success", "terminal", "stage", "failure_reason"}
        or type(value["success"]) is not bool
        or type(value["terminal"]) is not bool
        or not isinstance(value["stage"], str)
        or (value["failure_reason"] is not None and not isinstance(value["failure_reason"], str))
    ):
        raise ValueError("treatment-onset status schema is invalid")


def _validate_arm(arm: dict[str, Any], expected_arm: str, task: str) -> None:
    if arm.get("arm") != expected_arm:
        raise ValueError("alignment arm identity is invalid")
    treatment = arm.get("treatment_failure")
    success = arm.get("success")
    if (
        type(treatment) is not bool
        or (treatment and expected_arm != "alignment")
        or (treatment and success is not None)
        or (not treatment and type(success) is not bool)
    ):
        raise ValueError("alignment treatment/task state is invalid")
    steps = arm.get("steps")
    commands = arm.get("commands")
    decoded = arm.get("decoded_commands")
    prepared = arm.get("prepared_commands")
    chunks = arm.get("chunks")
    if (
        type(steps) is not int
        or not 1 <= steps <= 500
        or commands != steps
        or type(decoded) is not int
        or type(prepared) is not int
        or not commands <= prepared <= decoded
        or type(chunks) is not int
        or chunks < 1
        or arm.get("common_prefix_commands") not in range(1, EXECUTION_HORIZON + 1)
    ):
        raise ValueError("alignment arm counters are invalid")
    expected_chunks = (
        commands // EXECUTION_HORIZON + 1
        if treatment
        else max(1, (commands + EXECUTION_HORIZON - 1) // EXECUTION_HORIZON)
    )
    if chunks != expected_chunks:
        raise ValueError("alignment chunk count is inconsistent with execution")
    _validate_snapshot(
        arm.get("treatment_onset_snapshot"), task, arm["common_prefix_commands"]
    )
    _validate_status(arm.get("treatment_onset_status"))
    terminal_prefix = arm["treatment_onset_status"]["terminal"]
    if (
        arm.get("terminal_before_treatment") is not terminal_prefix
        or (terminal_prefix and arm.get("correction_opportunities") != 0)
        or (terminal_prefix and commands != arm["common_prefix_commands"])
    ):
        raise ValueError("terminal common-prefix state is invalid")
    if treatment:
        details = arm.get("treatment_failure_details")
        allowed_reasons = {
            "transform": {
                "invalid_policy_chunk",
                "tail_changed",
                "transform_threshold_exceeded",
            },
            "decode": {"transformed_decode_failure"},
            "realized_fk": {"realized_fk_preparation_failure"},
        }
        if (
            not isinstance(arm.get("treatment_failure_reason"), str)
            or not isinstance(details, dict)
            or set(details) != {"phase", "chunk_index", "lead", "reason", "details"}
            or details["phase"] not in {"transform", "decode", "realized_fk"}
            or type(details["chunk_index"]) is not int
            or details["chunk_index"] != chunks - 1
            or (
                details["phase"] == "transform"
                and details["lead"] is not None
            )
            or (
                details["phase"] in {"decode", "realized_fk"}
                and details["lead"] not in range(EXECUTION_HORIZON)
            )
            or details["reason"] != arm.get("treatment_failure_reason")
            or details["reason"] not in allowed_reasons[details["phase"]]
            or not isinstance(details["details"], dict)
            or arm.get("failure_reason") is not None
        ):
            raise ValueError("structured alignment treatment failure is invalid")
        lead = details["lead"]
        if (
            (
                details["phase"] == "transform"
                and not decoded == prepared == commands
            )
            or (
                details["phase"] == "decode"
                and (decoded != commands + lead or prepared != commands)
            )
            or (
                details["phase"] == "realized_fk"
                and not (
                    prepared == commands + lead
                    and commands + lead < decoded <= commands + EXECUTION_HORIZON
                )
            )
        ):
            raise ValueError("structured alignment treatment failure counters are invalid")
    elif (
        arm.get("treatment_failure_reason") is not None
        or arm.get("treatment_failure_details") is not None
        or (success and arm.get("failure_reason") is not None)
        or (success and arm.get("final_stage") != "success")
        or (not success and not isinstance(arm.get("failure_reason"), str))
        or (not success and not isinstance(arm.get("final_stage"), str))
    ):
        raise ValueError("alignment task outcome is invalid")
    metrics = arm.get("metrics_by_lead")
    if not isinstance(metrics, dict) or set(metrics) != set(_metric_group()):
        raise ValueError("alignment metric names are invalid")
    expected_leads = {str(lead) for lead in range(EXECUTION_HORIZON)}
    for values in metrics.values():
        if set(values) != expected_leads:
            raise ValueError("alignment metric leads are invalid")
        for value in values.values():
            _validate_compact(value)
    executed_by_lead = arm.get("executed_commands_by_lead")
    post_by_lead = arm.get("post_prefix_executed_commands_by_lead")
    if (
        not isinstance(executed_by_lead, dict)
        or set(executed_by_lead) != expected_leads
        or not isinstance(post_by_lead, dict)
        or set(post_by_lead) != expected_leads
        or any(type(value) is not int or value < 0 for value in executed_by_lead.values())
        or any(type(value) is not int or value < 0 for value in post_by_lead.values())
        or sum(executed_by_lead.values()) != commands
        or sum(post_by_lead.values()) != commands - arm["common_prefix_commands"]
        or any(post_by_lead[lead] > executed_by_lead[lead] for lead in expected_leads)
    ):
        raise ValueError("alignment executed command lead counters are invalid")
    expected_executed = {
        str(lead): commands // EXECUTION_HORIZON
        + int(lead < commands % EXECUTION_HORIZON)
        for lead in range(EXECUTION_HORIZON)
    }
    post_commands = commands - arm["common_prefix_commands"]
    expected_post = {
        str(lead): post_commands // EXECUTION_HORIZON
        + int(lead < post_commands % EXECUTION_HORIZON)
        for lead in range(EXECUTION_HORIZON)
    }
    if executed_by_lead != expected_executed or post_by_lead != expected_post:
        raise ValueError("alignment executed commands are assigned to invalid leads")
    executed_metric_names = (
        "raw_gripper_out_of_range",
        "reconstruction_translation_m",
        "reconstruction_rotation_deg",
        "reconstruction_gripper_native",
        "clipping_error_native",
        "clipped",
    )
    if any(
        metrics[name][lead]["count"] != executed_by_lead[lead]
        for name in executed_metric_names
        for lead in expected_leads
    ):
        raise ValueError("alignment executed-command metric counts are invalid")
    overlap_names = (
        "pre_translation_m",
        "pre_rotation_deg",
        "pre_gripper_effective",
        "intended_translation_m",
        "intended_rotation_deg",
        "intended_gripper_effective",
        "realized_translation_m",
        "realized_rotation_deg",
        "realized_gripper_effective",
        "correction_translation_m",
        "correction_rotation_deg",
        "correction_gripper_effective",
    )
    if any(
        metrics[name][lead]["count"] != post_by_lead[lead]
        for name in overlap_names
        for lead in expected_leads
    ):
        raise ValueError("alignment overlap metric counts are invalid")
    clipped = sum(metrics["clipped"][str(lead)]["sum"] for lead in range(EXECUTION_HORIZON))
    clipping_maximum = max(
        metrics["clipping_error_native"][str(lead)]["maximum"]
        for lead in range(EXECUTION_HORIZON)
    )
    if (
        arm.get("clipped_commands") != clipped
        or arm.get("clipping_rate") != clipped / commands
        or arm.get("maximum_clipping_error_native") != clipping_maximum
        or any(
            value["sum_squared"] != value["sum"] or value["maximum"] > 1
            for value in metrics["clipped"].values()
        )
    ):
        raise ValueError("alignment clipping telemetry is inconsistent")
    opportunities = arm.get("correction_opportunities")
    delivered = arm.get("delivered_corrections")
    checks = arm.get("tail_unchanged_checks")
    passes = arm.get("tail_unchanged_passes")
    if any(type(value) is not int or value < 0 for value in (opportunities, delivered, checks, passes)):
        raise ValueError("alignment correction counters are invalid")
    if expected_arm == "baseline":
        records = arm.get("baseline_chunk_audit_records")
        checks_count = arm.get("baseline_chunk_equality_checks")
        pass_count = arm.get("baseline_chunk_equality_passes")
        chain = arm.get("baseline_chunk_hash_chain_sha256")
        if not isinstance(records, list) or checks_count != len(records) or pass_count != sum(
            record.get("equal") is True for record in records
        ):
            raise ValueError("baseline chunk audit counters are invalid")
        digest = bytes.fromhex(EMPTY_OUTCOME_CHAIN_SHA256)
        for index, record in enumerate(records, start=1):
            if (
                not isinstance(record, dict)
                or set(record) != {"chunk_index", "raw_sha256", "final_sha256", "equal"}
                or record["chunk_index"] != index
                or not _valid_sha256(record["raw_sha256"])
                or not _valid_sha256(record["final_sha256"])
                or type(record["equal"]) is not bool
                or record["equal"] is not (record["raw_sha256"] == record["final_sha256"])
            ):
                raise ValueError("baseline chunk audit record is invalid")
            digest = hashlib.sha256(
                digest
                + json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).digest()
        if chain != digest.hex() or checks_count != chunks - 1 or pass_count != checks_count:
            raise ValueError("baseline chunk audit hash chain is invalid")
        if (
            opportunities
            or delivered
            or checks
            or passes
            or any(
                metrics[name][str(lead)]["maximum"] != 0.0
                for name in (
                    "correction_translation_m",
                    "correction_rotation_deg",
                    "correction_gripper_effective",
                )
                for lead in range(EXECUTION_HORIZON)
            )
        ):
            raise ValueError("baseline must remain unmodified")
    elif (
        arm.get("baseline_chunk_equality_checks") != 0
        or arm.get("baseline_chunk_equality_passes") != 0
        or arm.get("baseline_chunk_audit_records") != []
        or arm.get("baseline_chunk_hash_chain_sha256") != EMPTY_OUTCOME_CHAIN_SHA256
        or passes != checks
        or not delivered <= checks <= opportunities
        or (
            not treatment
            and (delivered != opportunities or checks != opportunities)
        )
    ):
        raise ValueError("alignment transform counters are inconsistent")
    maxima = arm.get("transform_lead_zero_maxima")
    if not isinstance(maxima, dict) or set(maxima) != set(TRANSFORM_THRESHOLDS) or any(
        not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
        for value in maxima.values()
    ):
        raise ValueError("alignment transform maxima are invalid")
    if not treatment and any(maxima[name] > threshold for name, threshold in TRANSFORM_THRESHOLDS.items()):
        raise ValueError("alignment transform threshold is exceeded")
    milestones = arm.get("milestones")
    if not isinstance(milestones, dict) or tuple(milestones) != exp39._milestone_names(task):
        raise ValueError("alignment milestone schema is invalid")
    attained = []
    for value in milestones.values():
        if (
            set(value) != {"attained", "first_step"}
            or type(value["attained"]) is not bool
            or value["attained"] != (type(value["first_step"]) is int)
            or (type(value["first_step"]) is int and not 1 <= value["first_step"] <= steps)
        ):
            raise ValueError("alignment milestone value is invalid")
        attained.append(value["attained"])
    if attained != sorted(attained, reverse=True) or arm.get("ordered_progression") != sum(attained) / len(attained):
        raise ValueError("alignment ordered progression is invalid")


def _validate_input_evidence(outcome: dict[str, Any]) -> None:
    inputs = outcome.get("initial_inputs")
    source = outcome.get("initial_policy_input_source_arm")
    order = outcome.get("arm_order")
    anchor = outcome.get("initial_policy_input_anchor_hashes")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != set(ARMS)
        or not isinstance(order, list)
        or len(order) != 2
        or source != order[0]
        or anchor != inputs.get(source)
    ):
        raise ValueError("alignment initial input source evidence is invalid")
    for hashes in (*inputs.values(), anchor):
        if not isinstance(hashes, dict) or set(hashes) != {
            "top_image_sha256",
            "canonical_state_sha256",
            "instruction_sha256",
            "combined_sha256",
        } or any(not _valid_sha256(value) for value in hashes.values()):
            raise ValueError("alignment initial input hashes are invalid")
        components = {
            name: hashes[name]
            for name in ("top_image_sha256", "canonical_state_sha256", "instruction_sha256")
        }
        if hashes["combined_sha256"] != hashlib.sha256(
            json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest():
            raise ValueError("alignment combined input hash does not recompute")
    image_match = inputs["baseline"]["top_image_sha256"] == inputs["alignment"]["top_image_sha256"]
    state_match = inputs["baseline"]["canonical_state_sha256"] == inputs["alignment"]["canonical_state_sha256"]
    instruction_match = inputs["baseline"]["instruction_sha256"] == inputs["alignment"]["instruction_sha256"]
    differing = outcome.get("initial_image_differing_channel_values")
    fraction = outcome.get("initial_image_differing_fraction")
    maximum = outcome.get("initial_image_max_abs_difference_uint8")
    if (
        outcome.get("initial_image_exact_hash_match") is not image_match
        or outcome.get("initial_canonical_state_hash_match") is not state_match
        or outcome.get("initial_instruction_hash_match") is not instruction_match
        or type(differing) is not int
        or not 0 <= differing <= exp39.INITIAL_IMAGE_CHANNEL_VALUES
        or not isinstance(fraction, (int, float))
        or not np.isfinite(fraction)
        or fraction != differing / exp39.INITIAL_IMAGE_CHANNEL_VALUES
        or type(maximum) is not int
        or not 0 <= maximum <= 255
        or image_match is not (differing == 0)
        or ((differing == 0 and maximum != 0) or (differing > 0 and maximum == 0))
        or outcome.get("initial_input_equivalent") is not True
        or not state_match
        or not instruction_match
        or maximum > exp39.INITIAL_IMAGE_MAX_ABS_UINT8
        or fraction > exp39.INITIAL_IMAGE_MAX_DIFFERING_FRACTION
    ):
        raise ValueError("alignment actual input equivalence is invalid")


def _validate_onset_policy_evidence(outcome: dict[str, Any]) -> None:
    evidence = outcome.get("treatment_onset_policy")
    expected_keys = {
        "applicable",
        "actual_inputs",
        "source_arm",
        "anchor_hashes",
        "input_comparison",
        "chunks",
        "chunk_max_abs_difference",
        "source_raw_chunk_sha256",
        "consumed_source_raw_chunk_sha256",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        raise ValueError("treatment-onset policy evidence schema is invalid")
    if outcome["terminal_before_treatment"]:
        if evidence["applicable"] is not False or any(
            evidence[name] is not None for name in expected_keys - {"applicable"}
        ):
            raise ValueError("terminal-prefix onset policy evidence must be an explicit no-op")
        for arm in ARMS:
            if outcome["arms"][arm].get("first_treatment_source_chunk_sha256") is not None:
                raise ValueError("terminal-prefix arms cannot consume a treatment source chunk")
        return
    if evidence["applicable"] is not True:
        raise ValueError("nonterminal pair requires treatment-onset policy evidence")
    comparison = evidence["input_comparison"]
    if not isinstance(comparison, dict):
        raise ValueError("treatment-onset input comparison is invalid")
    flattened = {
        "initial_inputs": evidence["actual_inputs"],
        "initial_policy_input_source_arm": evidence["source_arm"],
        "arm_order": outcome["arm_order"],
        "initial_policy_input_anchor_hashes": evidence["anchor_hashes"],
        **comparison,
    }
    _validate_input_evidence(flattened)
    chunks = evidence["chunks"]
    difference = evidence["chunk_max_abs_difference"]
    consumed = evidence["consumed_source_raw_chunk_sha256"]
    if (
        evidence["source_arm"] != outcome["arm_order"][0]
        or not isinstance(chunks, dict)
        or set(chunks) != set(ARMS)
        or any(not _valid_sha256(value) for value in chunks.values())
        or not isinstance(difference, (int, float))
        or not np.isfinite(difference)
        or not 0 <= difference <= exp39.FIRST_CHUNK_MAX_ABS
        or not _valid_sha256(evidence["source_raw_chunk_sha256"])
        or evidence["source_raw_chunk_sha256"] != chunks[evidence["source_arm"]]
        or not isinstance(consumed, dict)
        or set(consumed) != set(ARMS)
        or set(consumed.values()) != {evidence["source_raw_chunk_sha256"]}
        or any(
            outcome["arms"][arm].get("first_treatment_source_chunk_sha256")
            != evidence["source_raw_chunk_sha256"]
            for arm in ARMS
        )
    ):
        raise ValueError("treatment-onset shared source chunk evidence is invalid")
    baseline_audits = outcome.get("arms", {}).get("baseline", {}).get(
        "baseline_chunk_audit_records"
    )
    if (
        not isinstance(baseline_audits, list)
        or not baseline_audits
        or not isinstance(baseline_audits[0], dict)
        or baseline_audits[0].get("raw_sha256")
        != evidence["source_raw_chunk_sha256"]
    ):
        raise ValueError("baseline first replan audit does not match the onset source chunk")


def _validate_outcomes(report: dict[str, Any]) -> None:
    identity = report.get("identity", {})
    for ordinal, outcome in enumerate(report.get("outcomes", [])):
        task = outcome.get("task")
        expected_order = list(
            pair_arm_order(
                ordinal,
                identity.get("condition"),
                identity.get("model_seed"),
                identity.get("morphology"),
            )
        )
        if (
            not isinstance(outcome.get("scenario_id"), str)
            or task not in TASKS
            or outcome.get("morphology") != identity.get("morphology")
            or outcome.get("arm_order") != expected_order
            or set(outcome.get("arms", {})) != set(ARMS)
            or not _valid_sha256(outcome.get("source_chunk_sha256"))
            or not isinstance(outcome.get("common_prefix_native_sha256"), list)
            or len(outcome["common_prefix_native_sha256"]) != EXECUTION_HORIZON
            or any(not _valid_sha256(value) for value in outcome["common_prefix_native_sha256"])
            or outcome.get("treatment_onset_equal") is not True
            or outcome.get("terminal_before_treatment")
            is not outcome.get("treatment_onset_status", {}).get("terminal")
        ):
            raise ValueError("alignment pair identity or common-prefix evidence is invalid")
        _validate_input_evidence(outcome)
        first_chunks = outcome.get("first_chunks")
        difference = outcome.get("first_chunk_max_abs_difference")
        if (
            not isinstance(first_chunks, dict)
            or set(first_chunks) != set(ARMS)
            or any(not _valid_sha256(value) for value in first_chunks.values())
            or not isinstance(difference, (int, float))
            or not np.isfinite(difference)
            or not 0 <= difference <= exp39.FIRST_CHUNK_MAX_ABS
            or outcome["source_chunk_sha256"]
            != first_chunks[outcome["initial_policy_input_source_arm"]]
        ):
            raise ValueError("alignment shared source chunk evidence is invalid")
        _validate_snapshot(
            outcome.get("treatment_onset_snapshot"),
            task,
            outcome.get("common_prefix_commands"),
        )
        _validate_status(outcome.get("treatment_onset_status"))
        if (
            outcome["treatment_onset_snapshot"]["scalars"]["success"]
            is not outcome["treatment_onset_status"]["success"]
        ):
            raise ValueError("alignment onset snapshot and returned status disagree")
        _validate_onset_policy_evidence(outcome)
        for arm in ARMS:
            _validate_arm(outcome["arms"][arm], arm, task)
            if (
                outcome["arms"][arm]["treatment_onset_snapshot"]
                != outcome["treatment_onset_snapshot"]
                or outcome["arms"][arm]["treatment_onset_status"]
                != outcome["treatment_onset_status"]
                or outcome["arms"][arm]["common_prefix_commands"]
                != outcome["common_prefix_commands"]
            ):
                raise ValueError("alignment arm treatment-onset evidence does not match pair")
            onset_milestones = outcome["treatment_onset_snapshot"]["milestones"]
            final_milestones = outcome["arms"][arm]["milestones"]
            for name, onset in onset_milestones.items():
                final = final_milestones[name]
                if onset["attained"] and final != onset:
                    raise ValueError("onset milestone is not preserved in the final arm")
                if (
                    not onset["attained"]
                    and final["attained"]
                    and final["first_step"] <= outcome["common_prefix_commands"]
                ):
                    raise ValueError("post-onset milestone is backdated into the common prefix")


def _new_report(
    *,
    registered: bool,
    contract: dict[str, Any],
    identity: dict[str, Any],
    expected_ids: list[str],
    provenance: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": 40,
        "mode": "replan_alignment_shard",
        "registered": registered,
        "status": "in_progress" if registered else "smoke_unregistered_in_progress",
        "contract": contract,
        "identity": identity,
        "expected_outcome_ids": expected_ids,
        "provenance": provenance,
        "run": run,
        "complete": False,
        "outcomes": [],
        "outcome_chain_sha256": EMPTY_OUTCOME_CHAIN_SHA256,
        "infrastructure_errors": [],
        "treatment_delivery_errors": [],
    }


def _load_or_create_report(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return expected
    report = json.loads(path.read_text())
    for name in (
        "format_version",
        "experiment",
        "mode",
        "registered",
        "contract",
        "identity",
        "expected_outcome_ids",
        "provenance",
    ):
        if report.get(name) != expected.get(name):
            raise ValueError("resume report contract, checkpoint, or provenance does not match")
    stable_run = ("device", "lock_identity", "lock_path", "schedule_identity", "schedule_position")
    if any(report.get("run", {}).get(name) != expected["run"].get(name) for name in stable_run):
        raise ValueError("resume report runtime schedule does not match")
    started = _parse_utc(report.get("run", {}).get("started_utc"))
    finished_text = report["run"].get("finished_utc")
    if finished_text is not None and _parse_utc(finished_text) < started:
        raise ValueError("resume report interval is negative")
    ids = [outcome.get("scenario_id") for outcome in report.get("outcomes", [])]
    if ids != expected["expected_outcome_ids"][: len(ids)] or len(ids) != len(set(ids)):
        raise ValueError("resume report outcomes are not an ordered unique prefix")
    if report.get("outcome_chain_sha256") != _outcome_chain(report.get("outcomes", [])):
        raise ValueError("resume report outcome hash chain does not match")
    if not isinstance(report.get("infrastructure_errors"), list) or not isinstance(
        report.get("treatment_delivery_errors"), list
    ):
        raise ValueError("resume report error records are invalid")
    _validate_outcomes(report)
    expected_treatment = [
        {
            "scenario_id": outcome["scenario_id"],
            "arm": "alignment",
            "reason": outcome["arms"]["alignment"]["treatment_failure_reason"],
            "details": outcome["arms"]["alignment"]["treatment_failure_details"],
        }
        for outcome in report.get("outcomes", [])
        if outcome["arms"]["alignment"]["treatment_failure"]
    ]
    if report["treatment_delivery_errors"] != expected_treatment:
        raise ValueError("resume report treatment errors do not match outcomes")
    if report.get("complete"):
        if (
            ids != expected["expected_outcome_ids"]
            or report.get("infrastructure_errors")
            or report.get("status")
            != ("completed" if report["registered"] else "completed_smoke_unregistered")
            or report["run"].get("finished_utc") is None
            or report.get("aggregate") != _shard_aggregate(report["outcomes"])
        ):
            raise ValueError("completed resume report is invalid")
    elif report.get("status") != expected["status"] or "aggregate" in report:
        raise ValueError("incomplete resume report has stale derived state")
    return report


def _merge_metrics(outcomes: Iterable[dict[str, Any]], arm: str) -> dict[str, Any]:
    outcomes = list(outcomes)
    return {
        name: {
            str(lead): diagnostics.merge_compact(
                outcome["arms"][arm]["metrics_by_lead"][name][str(lead)]
                for outcome in outcomes
            )
            for lead in range(EXECUTION_HORIZON)
        }
        for name in _metric_group()
    }


def _shard_aggregate(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        outcome for outcome in outcomes if not outcome["arms"]["alignment"]["treatment_failure"]
    ]
    arms = {}
    for arm in ARMS:
        commands = sum(outcome["arms"][arm]["commands"] for outcome in outcomes)
        decoded = sum(outcome["arms"][arm]["decoded_commands"] for outcome in outcomes)
        prepared = sum(outcome["arms"][arm]["prepared_commands"] for outcome in outcomes)
        clipped = sum(outcome["arms"][arm]["clipped_commands"] for outcome in outcomes)
        arms[arm] = {
            "episodes": len(outcomes),
            "eligible_pair_episodes": len(eligible),
            "successes": sum(outcome["arms"][arm]["success"] is True for outcome in eligible),
            "commands": commands,
            "decoded_commands": decoded,
            "prepared_commands": prepared,
            "clipped_commands": clipped,
            "clipping_rate": clipped / commands if commands else 0.0,
            "correction_opportunities": sum(
                outcome["arms"][arm]["correction_opportunities"] for outcome in outcomes
            ),
            "delivered_corrections": sum(
                outcome["arms"][arm]["delivered_corrections"] for outcome in outcomes
            ),
            "metrics_by_lead": _merge_metrics(outcomes, arm),
        }
    tasks = {}
    for task in TASKS:
        values = [outcome for outcome in outcomes if outcome["task"] == task]
        valid = [
            outcome for outcome in values if not outcome["arms"]["alignment"]["treatment_failure"]
        ]
        tasks[task] = {
            arm: {
                "attempted": len(values),
                "eligible_pairs": len(valid),
                "success_rate": (
                    sum(outcome["arms"][arm]["success"] is True for outcome in valid) / len(valid)
                    if valid
                    else None
                ),
                "mean_ordered_progression": (
                    sum(outcome["arms"][arm]["ordered_progression"] for outcome in valid)
                    / len(valid)
                    if valid
                    else None
                ),
                "metrics_by_lead": _merge_metrics(values, arm),
            }
            for arm in ARMS
        }
    treatment_failures = len(outcomes) - len(eligible)
    transform_passed = all(
        outcome["arms"]["alignment"]["tail_unchanged_checks"]
        == outcome["arms"]["alignment"]["tail_unchanged_passes"]
        and all(
            outcome["arms"]["alignment"]["transform_lead_zero_maxima"][name]
            <= threshold
            for name, threshold in TRANSFORM_THRESHOLDS.items()
        )
        for outcome in outcomes
        if not outcome["arms"]["alignment"]["treatment_failure"]
    )
    clipping = {arm: arms[arm]["clipping_rate"] <= 0.01 for arm in ARMS}
    delivery = {
        "treatment_failures": treatment_failures,
        "exact_treatment_onset": all(outcome["treatment_onset_equal"] for outcome in outcomes),
        "transform_integrity_passed": transform_passed,
        "tail_integrity_passed": all(
            outcome["arms"]["alignment"]["tail_unchanged_checks"]
            == outcome["arms"]["alignment"]["tail_unchanged_passes"]
            for outcome in outcomes
        ),
        "arm_clipping_at_most_one_percent": clipping,
    }
    delivery["passed"] = (
        treatment_failures == 0
        and delivery["exact_treatment_onset"]
        and transform_passed
        and delivery["tail_integrity_passed"]
        and all(clipping.values())
    )
    return {
        "pairs": len(outcomes),
        "eligible_pairs": len(eligible),
        "no_op_common_prefix_pairs": sum(
            outcome["terminal_before_treatment"] for outcome in outcomes
        ),
        "arms": arms,
        "tasks": tasks,
        "delivery_gate": delivery,
    }


def _active_factor_axes(scenario: Scenario) -> list[str]:
    return [
        name
        for name, value in scenario.factors.items()
        if str(value.get("bin", "")).startswith("development:")
    ]


def evaluate_alignment_shard(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    exp36_summary_path: Path,
    exp39_summary_path: Path,
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
    lock_path: Path | None = None,
) -> dict[str, Any]:
    if registered and limit_scenarios is not None:
        raise ValueError("registered Experiment 040 does not permit a scenario limit")
    provenance = _runtime_provenance(device)
    if registered and provenance["git_dirty"]:
        raise RuntimeError("registered Experiment 040 requires a clean worktree")
    validate_exp39_summary(exp39_summary_path)
    checkpoint_condition = source_condition(condition, morphology)
    checkpoint, identity = resolve_checkpoint(
        exp36_summary_path, checkpoint_condition, seed, morphology, root=root
    )
    identity["condition"] = condition
    identity["source_condition"] = checkpoint_condition
    scenarios = alignment_scenarios(
        manifest, morphology, registered=registered, limit_scenarios=limit_scenarios
    )
    contract = {
        "definition_sha256": definition.sha256,
        "manifest_sha256": file_sha256(manifest.path),
        "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
        "exp39_summary_sha256": EXP39_SUMMARY_SHA256,
        "exp39_evaluator_sha256": EXP39_EVALUATOR_SHA256,
        "split": "development",
        "scenario_indices": list(SCENARIO_INDICES),
        "tasks": list(TASKS),
        "limit_scenarios": limit_scenarios,
        "protocol": ALIGNMENT_PROTOCOL,
        "final_split_loaded": False,
    }
    expected = _new_report(
        registered=registered,
        contract=contract,
        identity=identity,
        expected_ids=[scenario.scenario_id for scenario in scenarios],
        provenance=provenance,
        run=_run_metadata(
            device,
            registered=registered,
            condition=condition,
            seed=seed,
            morphology=morphology,
            lock_path=lock_path,
        ),
    )
    report = _load_or_create_report(output, expected)
    if report.get("complete"):
        return report
    try:
        policy = policy_loader(checkpoint, device)
        processor = processor_factory(
            policy.config.backbone_name,
            revision=policy.config.backbone_revision,
            do_rescale=policy.config.image_do_rescale,
        )
        report["infrastructure_errors"] = [
            error for error in report["infrastructure_errors"] if error.get("scenario_id") is not None
        ]
    except BaseException as error:
        report["infrastructure_errors"].append(
            {"scenario_id": None, "operation": "policy_setup", "error": repr(error)}
        )
        _write_progress(output, report)
        raise
    completed = len(report["outcomes"])
    for ordinal, scenario in enumerate(scenarios[completed:], start=completed):
        environments: dict[str, Any] = {}
        try:
            report["infrastructure_errors"] = [
                error
                for error in report["infrastructure_errors"]
                if error.get("scenario_id") != scenario.scenario_id
            ]
            for arm in ARMS:
                environments[arm] = env_factory(
                    morphology,
                    scenario.task,
                    runtime_scenario(scenario.factors),
                    width=640,
                    height=480,
                    render=True,
                )
            outcome = run_alignment_pair(
                environments,
                policy,
                processor,
                device,
                order=pair_arm_order(ordinal, condition, seed, morphology),
            )
            outcome.update(
                scenario_id=scenario.scenario_id,
                scenario_seed=scenario.seed,
                morphology=morphology,
                task=scenario.task,
                active_factor_axes=_active_factor_axes(scenario),
            )
            for arm in reversed(ARMS):
                environments[arm].close()
                environments.pop(arm)
            alignment = outcome["arms"]["alignment"]
            if alignment["treatment_failure"]:
                report["treatment_delivery_errors"].append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "arm": "alignment",
                        "reason": alignment["treatment_failure_reason"],
                        "details": alignment["treatment_failure_details"],
                    }
                )
            _append_outcome(report, outcome)
            _validate_outcomes(report)
            _write_progress(output, report)
        except BaseException as error:
            report["infrastructure_errors"].append(
                {"scenario_id": scenario.scenario_id, "error": repr(error)}
            )
            _write_progress(output, report)
            raise
        finally:
            active_error = sys.exception()
            for arm, env in list(environments.items()):
                try:
                    env.close()
                except BaseException as close_error:
                    report["infrastructure_errors"].append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "arm": arm,
                            "operation": "close",
                            "error": repr(close_error),
                        }
                    )
                    _write_progress(output, report)
                    if active_error is None:
                        raise
                    active_error.add_note(f"{arm} close failed: {close_error!r}")
    if report["infrastructure_errors"]:
        raise RuntimeError("alignment shard cannot complete with infrastructure errors")
    _validate_outcomes(report)
    report["aggregate"] = _shard_aggregate(report["outcomes"])
    report["complete"] = True
    report["status"] = "completed" if registered else "completed_smoke_unregistered"
    report["run"]["finished_utc"] = _utc_now()
    _write_progress(output, report)
    return report


def _task_equal_progression(values: list[dict[str, Any]], arm: str) -> float | None:
    means = []
    for task in TASKS:
        cell = [value["arms"][arm]["ordered_progression"] for value in values if value["task"] == task]
        if not cell:
            return None
        means.append(sum(cell) / len(cell))
    return sum(means) / len(means)


def _nullable_difference(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else first - second


def _mean_compact(value: dict[str, Any]) -> float | None:
    return value["sum"] / value["count"] if value["count"] else None


def _merge_condition_metric(
    reports: Iterable[dict[str, Any]], name: str, lead: int
) -> dict[str, Any]:
    return diagnostics.merge_compact(
        outcome["arms"]["alignment"]["metrics_by_lead"][name][str(lead)]
        for report in reports
        for outcome in report["outcomes"]
    )


def aggregate_registered_shards(
    reports: list[dict[str, Any]],
    manifest: ScenarioManifest,
    *,
    exp36_summary: dict[str, Any],
    exp39_summary: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if (
        exp36_summary.get("experiment") != 36
        or exp36_summary.get("status") != "completed_gate_passed"
        or exp36_summary.get("fixed_step") != 1200
    ):
        raise ValueError("aggregation requires the frozen Experiment 036 summary")
    try:
        _validate_exp39_summary_contract(exp39_summary)
    except ValueError as error:
        raise ValueError("aggregation requires the frozen Experiment 039 summary") from error
    indexed = {}
    for report in reports:
        identity = report.get("identity", {})
        key = (identity.get("condition"), identity.get("model_seed"), identity.get("morphology"))
        if key in indexed:
            raise ValueError(f"duplicate Experiment 040 shard: {key}")
        indexed[key] = report
    expected_keys = set(REGISTERED_SCHEDULE)
    if set(indexed) != expected_keys:
        raise ValueError("registered aggregation requires all 12 Experiment 040 shards")
    expected_source_keys = {
        f"{condition}/seed{seed}/{morphology}" for condition, seed, morphology in expected_keys
    }
    if set(source_hashes) != expected_source_keys or any(
        not _valid_sha256(value) for value in source_hashes.values()
    ):
        raise ValueError("registered aggregation requires exact hashes for all source shards")
    evaluator_sha256 = file_sha256(Path(__file__))
    commits = set()
    runtime_signatures = set()
    runtimes = []
    lock_paths = set()
    intervals = []
    by_condition_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    shard_summaries = {}
    delivery_by_shard = {}
    for key in REGISTERED_SCHEDULE:
        condition, seed, morphology = key
        report = indexed[key]
        scenarios = alignment_scenarios(manifest, morphology, registered=True)
        expected_ids = [scenario.scenario_id for scenario in scenarios]
        outcomes = report.get("outcomes", [])
        contract = report.get("contract", {})
        identity = report.get("identity", {})
        provenance = report.get("provenance", {})
        runtime = provenance.get("runtime", {})
        run = report.get("run", {})
        checkpoint_condition = source_condition(condition, morphology)
        expected_run = exp36_summary.get("runs", {}).get(f"{checkpoint_condition}-seed{seed}", {})
        if (
            report.get("experiment") != 40
            or report.get("mode") != "replan_alignment_shard"
            or report.get("registered") is not True
            or report.get("complete") is not True
            or report.get("status") != "completed"
            or report.get("expected_outcome_ids") != expected_ids
            or [outcome.get("scenario_id") for outcome in outcomes] != expected_ids
            or len(outcomes) != 21
            or contract.get("definition_sha256") != DEFINITION_SHA256
            or contract.get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
            or contract.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
            or contract.get("exp39_summary_sha256") != EXP39_SUMMARY_SHA256
            or contract.get("exp39_evaluator_sha256") != EXP39_EVALUATOR_SHA256
            or contract.get("scenario_indices") != list(SCENARIO_INDICES)
            or contract.get("tasks") != list(TASKS)
            or contract.get("limit_scenarios") is not None
            or contract.get("protocol") != ALIGNMENT_PROTOCOL
            or contract.get("final_split_loaded") is not False
            or identity.get("source_condition") != checkpoint_condition
            or identity.get("core_sha256") != expected_run.get("core_sha256")
            or identity.get("config_sha256") != CONFIG_SHA256[morphology]
            or identity.get("data_manifest_sha256") != DATA_MANIFEST_SHA256[checkpoint_condition]
            or identity.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
            or identity.get("step") != 1200
            or identity.get("stage") != "core"
            or provenance.get("git_dirty") is not False
            or provenance.get("evaluator_sha256") != evaluator_sha256
            or not isinstance(provenance.get("git_commit"), str)
            or not isinstance(runtime, dict)
            or set(runtime) != RUNTIME_FIELDS
            or provenance.get("runtime_signature_sha256")
            != hashlib.sha256(
                json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            or run.get("device") != runtime.get("device")
            or run.get("lock_identity") != LOCK_IDENTITY
            or run.get("schedule_identity") != SCHEDULE_IDENTITY
            or run.get("schedule_position") != REGISTERED_SCHEDULE.index(key)
            or not isinstance(run.get("lock_path"), str)
            or run.get("finished_utc") is None
            or report.get("infrastructure_errors")
            or report.get("outcome_chain_sha256") != _outcome_chain(outcomes)
        ):
            raise ValueError(f"registered Experiment 040 shard has wrong contract: {key}")
        _validate_outcomes(report)
        expected_treatment = [
            {
                "scenario_id": outcome["scenario_id"],
                "arm": "alignment",
                "reason": outcome["arms"]["alignment"]["treatment_failure_reason"],
                "details": outcome["arms"]["alignment"]["treatment_failure_details"],
            }
            for outcome in outcomes
            if outcome["arms"]["alignment"]["treatment_failure"]
        ]
        if report.get("treatment_delivery_errors") != expected_treatment:
            raise ValueError(f"registered treatment errors are invalid: {key}")
        for outcome, scenario in zip(outcomes, scenarios, strict=True):
            if (
                outcome.get("scenario_seed") != scenario.seed
                or outcome.get("task") != scenario.task
                or outcome.get("active_factor_axes") != _active_factor_axes(scenario)
            ):
                raise ValueError(f"registered outcome metadata is invalid: {key}")
        aggregate = _shard_aggregate(outcomes)
        if report.get("aggregate") != aggregate:
            raise ValueError(f"registered shard aggregate does not recompute: {key}")
        commits.add(provenance["git_commit"])
        runtime_signatures.add(provenance["runtime_signature_sha256"])
        runtimes.append(runtime)
        lock_paths.add(run["lock_path"])
        started = _parse_utc(run["started_utc"])
        finished = _parse_utc(run["finished_utc"])
        if finished < started:
            raise ValueError("registered shard interval is negative")
        intervals.append((run["schedule_position"], started, finished))
        label = f"{condition}/seed{seed}/{morphology}"
        shard_summaries[label] = aggregate
        delivery_by_shard[label] = aggregate["delivery_gate"]
        by_condition_seed.setdefault((condition, seed), []).extend(outcomes)
    if len(commits) != 1 or len(runtime_signatures) != 1 or len(lock_paths) != 1:
        raise ValueError("registered shards do not share evaluator/runtime/lock identity")
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("registered shard runtime signatures differ")
    intervals.sort()
    if [position for position, *_ in intervals] != list(range(12)) or any(
        first[2] > second[1] for first, second in zip(intervals, intervals[1:], strict=False)
    ):
        raise ValueError("registered shard intervals overlap or violate schedule")

    per_seed = {}
    discordants = {}
    subgroup_task = {}
    subgroup_morphology = {}
    support = {}
    corroboration = {}
    condition_reports = {
        condition: [indexed[key] for key in REGISTERED_SCHEDULE if key[0] == condition]
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        seed_success: dict[int, float | None] = {}
        seed_progression: dict[int, float | None] = {}
        for seed in SEEDS:
            values = by_condition_seed[(condition, seed)]
            eligible = [
                value for value in values if not value["arms"]["alignment"]["treatment_failure"]
            ]
            rates = {
                arm: (
                    sum(value["arms"][arm]["success"] is True for value in eligible) / len(eligible)
                    if eligible
                    else None
                )
                for arm in ARMS
            }
            progression = {arm: _task_equal_progression(eligible, arm) for arm in ARMS}
            seed_success[seed] = _nullable_difference(rates["alignment"], rates["baseline"])
            seed_progression[seed] = _nullable_difference(
                progression["alignment"], progression["baseline"]
            )
            per_seed[f"{condition}/seed{seed}"] = {
                "pairs": len(values),
                "eligible_pairs": len(eligible),
                "baseline_success_rate": rates["baseline"],
                "alignment_success_rate": rates["alignment"],
                "paired_success_gain_percentage_points": (
                    None if seed_success[seed] is None else 100 * seed_success[seed]
                ),
                "baseline_task_equal_progression": progression["baseline"],
                "alignment_task_equal_progression": progression["alignment"],
                "task_equal_progression_gain": seed_progression[seed],
            }
            wins = sum(
                not value["arms"]["baseline"]["success"]
                and value["arms"]["alignment"]["success"]
                for value in eligible
            )
            losses = sum(
                value["arms"]["baseline"]["success"]
                and not value["arms"]["alignment"]["success"]
                for value in eligible
            )
            discordants[f"{condition}/seed{seed}"] = {
                "eligible_pairs": len(eligible),
                "alignment_only_success": wins,
                "baseline_only_success": losses,
                "discordant_pairs": wins + losses,
            }
            for task in TASKS:
                cell = [value for value in eligible if value["task"] == task]
                subgroup_task[f"{condition}/seed{seed}/{task}"] = _subgroup_gain(cell)
            for morphology in MORPHOLOGIES:
                cell = [value for value in eligible if value["morphology"] == morphology]
                subgroup_morphology[f"{condition}/seed{seed}/{morphology}"] = _subgroup_gain(cell)
        pre = {
            component: _mean_compact(
                _merge_condition_metric(condition_reports[condition], f"pre_{component}", 0)
            )
            for component in ("translation_m", "rotation_deg", "gripper_effective")
        }
        realized = {
            component: _mean_compact(
                _merge_condition_metric(condition_reports[condition], f"realized_{component}", 0)
            )
            for component in ("translation_m", "rotation_deg", "gripper_effective")
        }
        checks = {
            "translation_improved": pre["translation_m"] is not None
            and realized["translation_m"] is not None
            and realized["translation_m"] < pre["translation_m"],
            "rotation_improved": pre["rotation_deg"] is not None
            and realized["rotation_deg"] is not None
            and realized["rotation_deg"] < pre["rotation_deg"],
            "effective_gripper_not_increased": pre["gripper_effective"] is not None
            and realized["gripper_effective"] is not None
            and realized["gripper_effective"] <= pre["gripper_effective"],
        }
        corroboration[condition] = {
            "pre_transform_lead_zero_means": pre,
            "realized_post_lead_zero_means": realized,
            "checks": checks,
            "corroborated": all(checks.values()),
        }
        delivery = all(
            delivery_by_shard[f"{condition}/seed{seed}/{morphology}"]["passed"]
            for seed in SEEDS
            for morphology in MORPHOLOGIES
        )
        support[condition] = _classify_support(
            seed_success,
            seed_progression,
            delivery=delivery,
            corroborated=corroboration[condition]["corroborated"],
        )
    return {
        "format_version": 1,
        "experiment": 40,
        "registered": True,
        "status": "completed_causal_diagnostic",
        "scope": "development-only intervention; no training, admission, inferential, or advancement claim",
        "contract": {
            "definition_sha256": DEFINITION_SHA256,
            "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
            "exp39_summary_sha256": EXP39_SUMMARY_SHA256,
            "exp39_evaluator_sha256": EXP39_EVALUATOR_SHA256,
            "evaluator_sha256": evaluator_sha256,
            "evaluator_git_commit": next(iter(commits)),
            "runtime_signature_sha256": next(iter(runtime_signatures)),
            "shards": 12,
            "pairs": 252,
            "episodes": 504,
            "final_split_loaded": False,
        },
        "source_shard_sha256": dict(sorted(source_hashes.items())),
        "shard_summaries": shard_summaries,
        "delivery_gates_by_shard": delivery_by_shard,
        "realized_continuity_corroboration": corroboration,
        "equal_weight_seed_gains": per_seed,
        "paired_discordants": discordants,
        "per_task_gains": subgroup_task,
        "per_morphology_gains": subgroup_morphology,
        "causal_support_by_condition": support,
        "multiplicity_adjusted_inferential_claim": False,
        "advancement_gate": None,
    }


def _subgroup_gain(values: list[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {"eligible_pairs": 0, "success_gain_percentage_points": None, "progression_gain": None}
    return {
        "eligible_pairs": len(values),
        "success_gain_percentage_points": 100
        * (
            sum(value["arms"]["alignment"]["success"] for value in values)
            - sum(value["arms"]["baseline"]["success"] for value in values)
        )
        / len(values),
        "progression_gain": (
            sum(value["arms"]["alignment"]["ordered_progression"] for value in values)
            - sum(value["arms"]["baseline"]["ordered_progression"] for value in values)
        )
        / len(values),
    }


def _classify_support(
    success: dict[int, float | None],
    progression: dict[int, float | None],
    *,
    delivery: bool,
    corroborated: bool,
) -> dict[str, Any]:
    success_mean = (
        sum(float(value) for value in success.values()) / len(SEEDS)
        if all(value is not None for value in success.values())
        else None
    )
    progression_mean = (
        sum(float(value) for value in progression.values()) / len(SEEDS)
        if all(value is not None for value in progression.values())
        else None
    )
    checks = {
        "three_seed_mean_success_gain_at_least_five_points": success_mean is not None
        and success_mean + 1e-12 >= 0.05,
        "at_least_two_nonnegative_seed_success_gains": sum(
            value is not None and value >= 0 for value in success.values()
        )
        >= 2,
        "three_seed_mean_task_equal_progression_gain_positive": progression_mean is not None
        and progression_mean > 0,
        "delivery_gate_passed": delivery,
        "realized_continuity_corroborated": corroborated,
    }
    return {
        "supported": all(checks.values()),
        "classification": "development_diagnostic_causal_support"
        if all(checks.values())
        else "no_causal_support",
        "checks": checks,
        "three_seed_mean_paired_success_gain_percentage_points": (
            None if success_mean is None else 100 * success_mean
        ),
        "three_seed_mean_task_equal_progression_gain": progression_mean,
        "per_seed_success_gain_percentage_points": {
            str(seed): None if success[seed] is None else 100 * success[seed] for seed in SEEDS
        },
        "per_seed_task_equal_progression_gain": {
            str(seed): progression[seed] for seed in SEEDS
        },
        "materiality_margin_percentage_points": 5.0,
        "inferential_claim": False,
    }


def registered_shard_filename(condition: str, seed: int, morphology: str) -> str:
    return f"benchmark-exp40-{condition}-seed{seed}-{morphology}-alignment.json"


def _registered_output(
    condition: str,
    seed: int,
    morphology: str,
    output: Path | None,
    *,
    runtime_dir: Path = REGISTERED_RUNTIME_DIR,
) -> Path:
    filename = registered_shard_filename(condition, seed, morphology)
    if output is not None and output.name != filename:
        raise ValueError(f"registered shard output filename must be {filename}")
    return output or runtime_dir / filename


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run Experiment 040 lead-zero replan alignment")
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
        "--exp39-summary", type=Path, default=Path("reports/benchmark-exp39-summary.json")
    )
    shard.add_argument("--artifact-root", type=Path, default=Path("."))
    shard.add_argument("--runtime-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    shard.add_argument("--output", type=Path)
    shard.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    shard.add_argument("--smoke", action="store_true")
    shard.add_argument("--limit-scenarios", type=_positive)
    run = subparsers.add_parser("run-registered", parents=[common])
    run.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    run.add_argument(
        "--exp39-summary", type=Path, default=Path("reports/benchmark-exp39-summary.json")
    )
    run.add_argument("--artifact-root", type=Path, default=Path("."))
    run.add_argument("--runtime-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--shard-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    aggregate.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    aggregate.add_argument(
        "--exp39-summary", type=Path, default=Path("reports/benchmark-exp39-summary.json")
    )
    aggregate.add_argument("--output", type=Path, default=Path("reports/benchmark-exp40-summary.json"))
    return parser.parse_args(argv)


def _run_all_registered(
    args: argparse.Namespace,
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
) -> list[dict[str, Any]]:
    lock_path = args.runtime_dir / REGISTERED_LOCK_NAME
    reports = []
    with registered_run_lock(lock_path):
        for condition, seed, morphology in REGISTERED_SCHEDULE:
            output = _registered_output(
                condition, seed, morphology, None, runtime_dir=args.runtime_dir
            )
            reports.append(
                evaluate_alignment_shard(
                    definition,
                    manifest,
                    args.exp36_summary,
                    args.exp39_summary,
                    output,
                    condition=condition,
                    seed=seed,
                    morphology=morphology,
                    device=args.device,
                    registered=True,
                    root=args.artifact_root,
                    lock_path=lock_path,
                )
            )
    return reports


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
        raise ValueError("Experiment 040 benchmark contract does not match registration")
    if args.command == "shard":
        output = (
            args.output
            if not registered
            else _registered_output(
                args.condition,
                args.seed,
                args.morphology,
                args.output,
                runtime_dir=args.runtime_dir,
            )
        )
        lock_path = args.runtime_dir / REGISTERED_LOCK_NAME
        if registered:
            with registered_run_lock(lock_path):
                report = evaluate_alignment_shard(
                    definition,
                    manifest,
                    args.exp36_summary,
                    args.exp39_summary,
                    output,
                    condition=args.condition,
                    seed=args.seed,
                    morphology=args.morphology,
                    device=args.device,
                    registered=True,
                    root=args.artifact_root,
                    lock_path=lock_path,
                )
        else:
            report = evaluate_alignment_shard(
                definition,
                manifest,
                args.exp36_summary,
                args.exp39_summary,
                output,
                condition=args.condition,
                seed=args.seed,
                morphology=args.morphology,
                device=args.device,
                registered=False,
                limit_scenarios=args.limit_scenarios,
                root=args.artifact_root,
            )
        print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))
        return
    if args.command == "run-registered":
        reports = _run_all_registered(args, definition, manifest)
        print(
            json.dumps(
                {
                    "runtime_dir": str(args.runtime_dir),
                    "completed_shards": len(reports),
                    "status": "completed",
                },
                indent=2,
            )
        )
        return
    if file_sha256(args.exp36_summary) != EXP36_SUMMARY_SHA256:
        raise ValueError("Experiment 036 summary hash does not match registration")
    exp39_summary = validate_exp39_summary(args.exp39_summary)
    paths = {
        f"{condition}/seed{seed}/{morphology}": args.shard_dir
        / registered_shard_filename(condition, seed, morphology)
        for condition, seed, morphology in REGISTERED_SCHEDULE
    }
    reports = [json.loads(path.read_text()) for path in paths.values()]
    source_hashes = {key: file_sha256(path) for key, path in paths.items()}
    summary = aggregate_registered_shards(
        reports,
        manifest,
        exp36_summary=json.loads(args.exp36_summary.read_text()),
        exp39_summary=exp39_summary,
        source_hashes=source_hashes,
    )
    _write_final(args.output, summary)
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
