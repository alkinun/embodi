from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
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
from ..processing import EmbodiProcessor, camera_frame_to_tensor
from ..record_so101 import atomic_write_json
from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv, runtime_scenario
from . import benchmark_geometry_diagnostics as exp38
from .benchmark_geometry_evaluation import (
    CONFIG_SHA256,
    DATA_MANIFEST_SHA256,
    EXP36_SUMMARY_SHA256,
    _clipping,
    load_development_contract,
    load_geometry_policy,
    resolve_checkpoint,
    rotation_error_degrees,
)


DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
DEVELOPMENT_MANIFEST_SHA256 = (
    "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
)
EXP38_SUMMARY_SHA256 = "e0e7eaf7ec25600b611cd66780a6dc3d0ca8c374a4f871b15a801716703307bb"
EXP38_EVALUATOR_SHA256 = "773dd4c7d136b75b2194f7c04ea7e4b1a56c898357740fb1fc03b22c20fef870"
CONDITIONS = ("joint", "matched")
MORPHOLOGIES = ("so101", "panda")
TASKS = ("push_to_zone", "lift_object", "pick_place_bin")
SEEDS = (36001, 36002, 36003)
SCENARIO_INDICES = tuple(range(7, 14))
ARMS = ("baseline", "projection")
EXECUTION_HORIZON = 4
IK_ITERATIONS = 20
FIRST_CHUNK_MAX_ABS = 1e-5
THRESHOLDS = {"translation_m": 0.001, "rotation_deg": 1.0, "gripper_native": 1.0}
ALPHA_GRID = tuple(index / 16 for index in range(16, -1, -1))
ALPHA_LABELS = tuple("1" if index == 16 else "0" if index == 0 else f"{index}/16" for index in range(16, -1, -1))
EMPTY_OUTCOME_CHAIN_SHA256 = hashlib.sha256(b"").hexdigest()
REGISTERED_RUNTIME_DIR = Path("reports/exp39-runtime")
REGISTERED_LOCK_NAME = ".registered.lock"
LOCK_IDENTITY = "exp39-linux-flock-exclusive-v1"
SCHEDULE_IDENTITY = "condition-major-seed-major-morphology-v1"
REGISTERED_SCHEDULE = tuple(
    (condition, seed, morphology)
    for condition in CONDITIONS
    for seed in SEEDS
    for morphology in MORPHOLOGIES
)
RUNTIME_FIELDS = {
    "device",
    "python",
    "torch",
    "numpy",
    "mujoco",
    "cuda_runtime",
    "cudnn",
    "cuda_device_name",
    "mujoco_gl",
    "python_hash_seed",
    "cublas_workspace_config",
    "torch_deterministic_algorithms",
    "torch_deterministic_warn_only",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "cuda_matmul_allow_tf32",
    "torch_num_threads",
    "torch_num_interop_threads",
    "platform",
    "hostname",
}
PROJECTION_PROTOCOL = {
    "camera": "top",
    "resolution": [640, 480],
    "state_path": "forward_kinematics_canonical",
    "native_state": None,
    "predictor": "predict_canonical_action_chunk",
    "chunk_steps": 32,
    "execution_horizon_steps": 4,
    "decode_next_commands_before_stepping": 4,
    "ik_iterations": 20,
    "frequency_hz": 30,
    "maximum_episode_steps": 500,
    "objective": "regression",
    "expert": None,
    "alpha_grid": list(ALPHA_LABELS),
    "translation_threshold_m": 0.001,
    "rotation_threshold_deg": 1.0,
    "gripper_threshold_native": 1.0,
    "first_chunk_max_abs_tolerance": FIRST_CHUNK_MAX_ABS,
    "pairing": "fresh adjacent counterbalanced baseline/projection episodes",
}


class TreatmentDeliveryError(RuntimeError):
    """The registered projection could not produce an executable command."""

    def __init__(
        self,
        reason: str,
        *,
        trial_count: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason.replace("_", " "))
        self.reason = reason
        self.trial_count = trial_count
        self.details = {} if details is None else details


@dataclass(frozen=True)
class PolicyInput:
    frame: np.ndarray
    canonical_state: np.ndarray
    instruction: str
    hashes: dict[str, str]


@dataclass(frozen=True)
class DecodedCommand:
    canonical: np.ndarray
    native: np.ndarray
    realized: np.ndarray
    alpha: float
    alpha_label: str
    errors: dict[str, float]
    trials: int


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Experiment 039 timestamps must be UTC ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("Experiment 039 timestamps must be valid UTC ISO-8601 strings") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("Experiment 039 timestamps must use UTC")
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


def _run_metadata(
    device: str,
    *,
    registered: bool,
    condition: str,
    seed: int,
    morphology: str,
    lock_path: Path | None,
) -> dict[str, Any]:
    key = (condition, seed, morphology)
    return {
        "started_utc": _utc_now(),
        "finished_utc": None,
        "device": device,
        "lock_identity": LOCK_IDENTITY if registered else None,
        "lock_path": str(lock_path) if registered and lock_path is not None else None,
        "schedule_identity": SCHEDULE_IDENTITY if registered else "smoke-unregistered",
        "schedule_position": REGISTERED_SCHEDULE.index(key) if registered else None,
    }


@contextmanager
def registered_run_lock(lock_path: Path) -> Iterable[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another registered Experiment 039 shard holds {lock_path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def exact_initial_input_hashes(
    frame: np.ndarray, canonical_state: np.ndarray, instruction: str
) -> dict[str, str]:
    frame = np.asarray(frame)
    state = np.asarray(canonical_state, dtype=np.float32)
    if frame.shape != (480, 640, 3) or state.shape != (1, 10) or not np.isfinite(state).all():
        raise ValueError("initial policy input must contain a 640x480 image and finite canonical state")
    if not isinstance(instruction, str):
        raise ValueError("initial policy instruction must be text")
    components = {
        "top_image_sha256": _array_sha256(frame),
        "canonical_state_sha256": _array_sha256(state),
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
    }
    components["combined_sha256"] = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return components


def capture_policy_input(env: Any) -> PolicyInput:
    frame = np.asarray(env.render_top()).copy()
    state = np.asarray(env.canonical_state(), dtype=np.float32).copy()
    instruction = env.instruction
    return PolicyInput(frame, state, instruction, exact_initial_input_hashes(frame, state, instruction))


def predict_from_input(snapshot: PolicyInput, policy: Any, processor: Any, device: str) -> np.ndarray:
    image = camera_frame_to_tensor(
        snapshot.frame,
        processor_rescales=policy.config.image_do_rescale,
    )
    canonical_state = torch.from_numpy(snapshot.canonical_state).unsqueeze(0)
    batch = processor(
        [[image]],
        [snapshot.instruction],
        None,
        canonical_state=canonical_state,
        canonical_part_mask=torch.ones((1, canonical_state.shape[1]), dtype=torch.bool),
    )
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }
    value = policy.predict_canonical_action_chunk(batch)
    chunk = value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if chunk.shape != (1, 32, 1, 10):
        raise ValueError("canonical policy output must have shape [1, 32, 1, 10]")
    chunk = np.asarray(chunk[0], dtype=np.float32)
    if not np.isfinite(chunk).all():
        raise TreatmentDeliveryError("policy returned a nonfinite canonical chunk")
    return chunk


def validate_rotation_matrix(rotation: np.ndarray, *, atol: float = 1e-7) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError("rotation matrix must be orthonormal")
    determinant = float(np.linalg.det(matrix))
    if determinant <= 0.0 or not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError("rotation matrix must have positive unit determinant")
    return matrix


def rotation_matrix_from_6d(rotation_6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation_6d, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("canonical rotation must be a finite width-6 vector")
    first = value[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-12:
        raise ValueError("canonical rotation first axis is degenerate")
    first = first / first_norm
    second = value[3:] - first * float(np.dot(first, value[3:]))
    second_norm = float(np.linalg.norm(second))
    if second_norm <= 1e-12:
        raise ValueError("canonical rotation second axis is degenerate")
    second = second / second_norm
    return validate_rotation_matrix(np.stack((first, second, np.cross(first, second)), axis=1))


def canonical_rotation_6d(rotation: np.ndarray) -> np.ndarray:
    matrix = validate_rotation_matrix(rotation)
    return np.asarray(matrix[:, :2].T.reshape(6), dtype=np.float32)


def _rotation_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = validate_rotation_matrix(rotation)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        first = (index + 1) % 3
        second = (index + 2) % 3
        scale = 2.0 * np.sqrt(max(1.0 + matrix[index, index] - matrix[first, first] - matrix[second, second], 0.0))
        if scale <= 1e-15:
            raise ValueError("rotation could not be converted to a quaternion")
        quaternion = np.empty(4, dtype=np.float64)
        quaternion[0] = (matrix[second, first] - matrix[first, second]) / scale
        quaternion[index + 1] = 0.25 * scale
        quaternion[first + 1] = (matrix[first, index] + matrix[index, first]) / scale
        quaternion[second + 1] = (matrix[second, index] + matrix[index, second]) / scale
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < -1e-15 or (
        abs(quaternion[0]) <= 1e-15
        and tuple(quaternion[1:]) < tuple(np.zeros(3, dtype=np.float64))
    ):
        quaternion = -quaternion
    return quaternion


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return validate_rotation_matrix(matrix, atol=1e-6)


def scale_rotation_shortest_path(rotation: np.ndarray, alpha: float) -> np.ndarray:
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("rotation scale must lie in [0, 1]")
    quaternion = _rotation_quaternion(rotation)
    vector = quaternion[1:]
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1e-15 or alpha == 0.0:
        return np.eye(3)
    angle = 2.0 * np.arctan2(magnitude, max(float(quaternion[0]), 0.0))
    half = 0.5 * alpha * angle
    scaled = np.r_[np.cos(half), vector / magnitude * np.sin(half)]
    return _quaternion_rotation(scaled)


def scale_canonical_command(command: np.ndarray, alpha: float) -> np.ndarray:
    canonical = np.asarray(command, dtype=np.float32)
    if canonical.shape == (1, 10):
        canonical = canonical[0]
    if canonical.shape != (10,) or not np.isfinite(canonical).all():
        raise ValueError("canonical command must be finite and width 10")
    result = canonical.copy()
    result[:3] *= alpha
    result[3:9] = canonical_rotation_6d(
        scale_rotation_shortest_path(rotation_matrix_from_6d(canonical[3:9]), alpha)
    )
    result[9] = canonical[9]
    return result.reshape(1, 10)


def _gripper_native_scale(env: Any) -> float:
    return 35.0 if env.morphology.id == "so101" else 0.08


def _canonical_errors(env: Any, requested: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    first = np.asarray(requested, dtype=np.float64).reshape(10)
    second = np.asarray(realized, dtype=np.float64).reshape(10)
    values = {
        "translation_m": float(np.linalg.norm(first[:3] - second[:3])),
        "rotation_deg": rotation_error_degrees(first[3:9], second[3:9]),
        "gripper_native": float(abs(first[9] - second[9])) * _gripper_native_scale(env),
    }
    if not np.isfinite(list(values.values())).all():
        raise TreatmentDeliveryError("canonical reconstruction produced nonfinite errors")
    return values


def _decode_once(env: Any, canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    native = np.asarray(
        env.native_action_from_canonical(canonical, iterations=IK_ITERATIONS), dtype=np.float32
    )
    if native.shape != (env.morphology.native_action_dim,) or not np.isfinite(native).all():
        raise TreatmentDeliveryError("IK returned an invalid native action")
    realized = np.asarray(env.canonical_arm_action(native), dtype=np.float32)
    if realized.shape != (1, 10) or not np.isfinite(realized).all():
        raise TreatmentDeliveryError("FK returned an invalid canonical action")
    return native, realized, _canonical_errors(env, canonical, realized)


def decode_baseline_command(env: Any, command: np.ndarray) -> DecodedCommand:
    canonical = np.asarray(command, dtype=np.float32).reshape(1, 10).copy()
    rotation_matrix_from_6d(canonical[0, 3:9])
    native, realized, errors = _decode_once(env, canonical)
    return DecodedCommand(canonical, native, realized, 1.0, "1", errors, 1)


def project_feasible_command(env: Any, command: np.ndarray) -> DecodedCommand:
    original = np.asarray(command, dtype=np.float32).reshape(1, 10)
    trials = 0
    for alpha, label in zip(ALPHA_GRID, ALPHA_LABELS, strict=True):
        candidate = scale_canonical_command(original, alpha)
        trials += 1
        try:
            native, realized, errors = _decode_once(env, candidate)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, TreatmentDeliveryError):
            if alpha == 0.0:
                break
            continue
        if all(errors[name] <= threshold for name, threshold in THRESHOLDS.items()):
            return DecodedCommand(candidate, native, realized, alpha, label, errors, trials)
    raise TreatmentDeliveryError(
        "alpha_zero_infeasible",
        trial_count=trials,
        details={"grid_exhausted": True, "last_alpha": "0"},
    )


def projection_scenarios(
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
            raise ValueError("registered Experiment 039 requires exact scenario indices 0007 through 0013")
        for task in TASKS:
            axes = [
                name
                for scenario in selected
                if scenario.task == task
                for name, value in scenario.factors.items()
                if str(value.get("bin", "")).startswith("development:")
            ]
            if len(axes) != 7 or len(set(axes)) != 7:
                raise ValueError("each Experiment 039 cell must cover all seven active factor axes")
    if limit_scenarios is not None:
        selected = selected[:limit_scenarios]
    return selected


def source_condition(condition: str, morphology: str) -> str:
    if condition == "joint":
        return condition
    if condition == "matched" and morphology in MORPHOLOGIES:
        return morphology
    raise ValueError("Experiment 039 condition must be joint or matched")


def pair_arm_order(
    scenario_ordinal: int, condition: str, seed: int, morphology: str
) -> tuple[str, str]:
    try:
        parity = (
            scenario_ordinal
            + CONDITIONS.index(condition)
            + SEEDS.index(seed)
            + MORPHOLOGIES.index(morphology)
        ) % 2
    except ValueError as error:
        raise ValueError("pair counterbalance identity is not registered") from error
    return ARMS if parity == 0 else tuple(reversed(ARMS))


def _milestone_names(task: str) -> tuple[str, ...]:
    if task == "push_to_zone":
        return ("near_ee_object", "object_in_target_tolerance", "settled")
    if task == "lift_object":
        return ("near_ee_object", "gripper_closed", "lifted_0_07m", "lift_predicate")
    if task == "pick_place_bin":
        return (
            "near_ee_object",
            "gripper_closed",
            "lifted_0_07m",
            "lift_predicate",
            "object_in_target_tolerance",
            "settled",
            "gripper_released",
        )
    raise ValueError(f"unsupported Experiment 039 task: {task}")


def _milestone_truth(env: Any, diagnostics: dict[str, float]) -> dict[str, bool]:
    release_threshold = 0.4 if env.morphology.id == "so101" else 0.7
    closed = diagnostics["gripper_opening_fraction"] < release_threshold
    near = diagnostics["ee_object_distance_m"] < 0.06
    return {
        "near_ee_object": near,
        "object_in_target_tolerance": (
            diagnostics["object_target_xy_distance_m"] <= env.scenario.target_tolerance
        ),
        "settled": diagnostics["object_speed"] < 0.05,
        "gripper_closed": closed,
        "gripper_released": not closed,
        "lifted_0_07m": diagnostics["object_height_m"] >= 0.07,
        "lift_predicate": diagnostics["object_height_m"] >= 0.08 and near and closed,
    }


def update_ordered_milestones(
    milestones: dict[str, dict[str, int | bool | None]],
    env: Any,
    diagnostics: dict[str, float],
    step: int,
) -> None:
    truth = _milestone_truth(env, diagnostics)
    for name, value in milestones.items():
        if value["attained"]:
            continue
        if truth[name]:
            milestones[name] = {"attained": True, "first_step": step}
            continue
        break


def _metric_group() -> dict[str, dict[str, dict[str, int | float]]]:
    return {
        name: {str(lead): exp38.compact_accumulator() for lead in range(EXECUTION_HORIZON)}
        for name in (
            "original_translation_m",
            "original_rotation_deg",
            "accepted_translation_m",
            "accepted_rotation_deg",
            "translation_m",
            "rotation_deg",
            "gripper_native",
            "clipping_error_native",
            "clipped",
        )
    }


def _command_geometry(command: np.ndarray) -> tuple[float, float]:
    canonical = np.asarray(command, dtype=np.float64).reshape(10)
    rotation = rotation_matrix_from_6d(canonical[3:9])
    return float(np.linalg.norm(canonical[:3])), rotation_error_degrees(
        canonical_rotation_6d(np.eye(3)), canonical_rotation_6d(rotation)
    )


def _record_command(
    metrics: dict[str, dict[str, dict[str, int | float]]],
    original: np.ndarray,
    decoded: DecodedCommand,
    lead: int,
    clipping_error: float,
    clipped: bool,
) -> None:
    original_translation, original_rotation = _command_geometry(original)
    accepted_translation, accepted_rotation = _command_geometry(decoded.canonical)
    values = {
        "original_translation_m": original_translation,
        "original_rotation_deg": original_rotation,
        "accepted_translation_m": accepted_translation,
        "accepted_rotation_deg": accepted_rotation,
        **decoded.errors,
        "clipping_error_native": clipping_error,
        "clipped": float(clipped),
    }
    for name, value in values.items():
        exp38.update_compact(metrics[name][str(lead)], value)


def run_projection_episode(
    env: Any,
    policy: Any,
    processor: Any,
    device: str,
    *,
    arm: str,
    initial_chunk: np.ndarray,
    maximum_steps: int = 500,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("Experiment 039 arm must be baseline or projection")
    policy.reset()
    metrics = _metric_group()
    alpha_histogram = {label: 0 for label in ALPHA_LABELS}
    alpha_histogram_by_lead = {
        str(lead): {label: 0 for label in ALPHA_LABELS} for lead in range(EXECUTION_HORIZON)
    }
    milestones = {
        name: {"attained": False, "first_step": None} for name in _milestone_names(env.task.id)
    }
    steps = chunks = commands = decoded_commands = clipped_commands = decode_trials = 0
    interventions = alpha_zero = alpha_one = 0
    maximum_clipping_error = 0.0
    status = None
    failure_reason = None
    chunk = np.asarray(initial_chunk, dtype=np.float32)

    def result(
        success: bool | None,
        *,
        treatment_failure: bool = False,
        treatment_failure_reason: str | None = None,
        treatment_failure_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        progression = sum(value["attained"] for value in milestones.values()) / len(milestones)
        return {
            "arm": arm,
            "success": success,
            "steps": steps,
            "chunks": chunks,
            "final_stage": None if status is None else status.stage,
            "failure_reason": failure_reason,
            "commands": commands,
            "decoded_commands": decoded_commands,
            "decode_trials": decode_trials,
            "clipped_commands": clipped_commands,
            "command_clipping_rate": clipped_commands / commands if commands else 0.0,
            "maximum_clipping_error_native": maximum_clipping_error,
            "geometry_by_lead": metrics,
            "alpha_histogram": alpha_histogram,
            "alpha_histogram_by_lead": alpha_histogram_by_lead,
            "intervention_count_by_lead": {
                lead: sum(count for label, count in histogram.items() if label != "1")
                for lead, histogram in alpha_histogram_by_lead.items()
            },
            "alpha_zero_count_by_lead": {
                lead: histogram["0"] for lead, histogram in alpha_histogram_by_lead.items()
            },
            "alpha_one_count_by_lead": {
                lead: histogram["1"] for lead, histogram in alpha_histogram_by_lead.items()
            },
            "intervention_count": interventions,
            "alpha_zero_count": alpha_zero,
            "alpha_one_count": alpha_one,
            "milestones": milestones,
            "ordered_progression": progression,
            "treatment_failure": treatment_failure,
            "treatment_failure_reason": treatment_failure_reason,
            "treatment_failure_details": treatment_failure_details,
        }

    while steps < maximum_steps:
        if chunks:
            chunk = predict_from_input(capture_policy_input(env), policy, processor, device)
        chunks += 1
        remaining = min(EXECUTION_HORIZON, maximum_steps - steps)
        decoded_chunk: list[tuple[int, np.ndarray, DecodedCommand]] = []
        for lead, command in enumerate(chunk[:remaining]):
            if arm == "baseline":
                decoded = decode_baseline_command(env, command)
            else:
                try:
                    decoded = project_feasible_command(env, command)
                except TreatmentDeliveryError as error:
                    if error.reason != "alpha_zero_infeasible":
                        raise
                    decode_trials += error.trial_count
                    return result(
                        None,
                        treatment_failure=True,
                        treatment_failure_reason=error.reason,
                        treatment_failure_details={
                            **error.details,
                            "chunk_index": chunks - 1,
                            "lead": lead,
                            "trial_count": error.trial_count,
                            "prepared_commands_discarded": len(decoded_chunk),
                        },
                    )
            decode_trials += decoded.trials
            decoded_commands += 1
            decoded_chunk.append((lead, command, decoded))
        for lead, original, decoded in decoded_chunk:
            was_clipped, clipping_error = _clipping(env, decoded.native)
            _record_command(metrics, original, decoded, lead, clipping_error, was_clipped)
            commands += 1
            clipped_commands += int(was_clipped)
            maximum_clipping_error = max(maximum_clipping_error, clipping_error)
            alpha_histogram[decoded.alpha_label] += 1
            alpha_histogram_by_lead[str(lead)][decoded.alpha_label] += 1
            interventions += int(decoded.alpha < 1.0)
            alpha_zero += int(decoded.alpha == 0.0)
            alpha_one += int(decoded.alpha == 1.0)
            # The retained array is the exact result of the accepted decode trial.
            env.apply_action(decoded.native)
            status = env.step_control_period(30.0)
            steps += 1
            update_ordered_milestones(milestones, env, env.diagnostic_state(), steps)
            if status.terminal:
                failure_reason = status.failure_reason
                break
        if status is not None and status.terminal:
            break
    success = bool(status is not None and status.success)
    if not success and failure_reason is None:
        failure_reason = "maximum_episode_steps" if steps >= maximum_steps else "task_terminal"
    return result(success)


def _active_factor_axes(scenario: Scenario) -> list[str]:
    return [
        name
        for name, value in scenario.factors.items()
        if str(value.get("bin", "")).startswith("development:")
    ]


def run_projection_pair(
    environments: dict[str, Any],
    policy: Any,
    processor: Any,
    device: str,
    *,
    order: tuple[str, str],
    maximum_steps: int = 500,
) -> dict[str, Any]:
    snapshots = {}
    chunks = {}
    for arm in ARMS:
        policy.reset()
        snapshots[arm] = capture_policy_input(environments[arm])
        chunks[arm] = predict_from_input(snapshots[arm], policy, processor, device)
    input_agreement = snapshots["baseline"].hashes == snapshots["projection"].hashes
    chunk_difference = float(np.max(np.abs(chunks["baseline"] - chunks["projection"])))
    if not input_agreement:
        raise TreatmentDeliveryError("paired initial policy input hashes do not match exactly")
    if not np.isfinite(chunk_difference) or chunk_difference > FIRST_CHUNK_MAX_ABS:
        raise TreatmentDeliveryError("paired first policy chunks exceed the registered tolerance")
    outcomes = {}
    for arm in order:
        outcomes[arm] = run_projection_episode(
            environments[arm],
            policy,
            processor,
            device,
            arm=arm,
            initial_chunk=chunks[arm],
            maximum_steps=maximum_steps,
        )
    return {
        "arm_order": list(order),
        "initial_inputs": {arm: snapshots[arm].hashes for arm in ARMS},
        "initial_input_hash_agreement": input_agreement,
        "first_chunks": {arm: _array_sha256(chunks[arm]) for arm in ARMS},
        "first_chunk_max_abs_difference": chunk_difference,
        "arms": outcomes,
    }


def _validate_compact(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"count", "sum", "sum_squared", "maximum"}:
        raise ValueError("projection compact accumulator schema is invalid")
    count = value["count"]
    numbers = [value[name] for name in ("sum", "sum_squared", "maximum")]
    if not isinstance(count, int) or count < 0 or not np.isfinite(numbers).all() or min(numbers) < 0:
        raise ValueError("projection compact accumulator values are invalid")
    if count == 0 and numbers != [0.0, 0.0, 0.0]:
        raise ValueError("empty projection compact accumulators must be zero")
    if count > 0 and (
        float(value["maximum"]) > float(value["sum"]) + 1e-12
        or float(value["sum"]) > count * float(value["maximum"]) + 1e-12
        or float(value["sum_squared"]) + 1e-12
        < float(value["sum"]) * float(value["sum"]) / count
        or float(value["sum_squared"])
        > float(value["maximum"]) * float(value["sum"]) + 1e-12
    ):
        raise ValueError("projection compact accumulator statistics are inconsistent")


def _validate_arm(arm: dict[str, Any], expected_arm: str, task: str) -> None:
    steps = arm.get("steps")
    commands = arm.get("commands")
    chunks = arm.get("chunks")
    clipped = arm.get("clipped_commands")
    decoded_commands = arm.get("decoded_commands")
    treatment_failure = arm.get("treatment_failure")
    success = arm.get("success")
    if (
        arm.get("arm") != expected_arm
        or not isinstance(treatment_failure, bool)
        or (treatment_failure and expected_arm != "projection")
        or (treatment_failure and success is not None)
        or (not treatment_failure and not isinstance(success, bool))
        or not isinstance(steps, int)
        or not 0 <= steps <= 500
        or commands != steps
        or not isinstance(chunks, int)
        or chunks < 1
        or not isinstance(clipped, int)
        or not 0 <= clipped <= commands
        or not isinstance(decoded_commands, int)
        or not commands <= decoded_commands <= commands + EXECUTION_HORIZON - 1
        or arm.get("command_clipping_rate") != (clipped / commands if commands else 0.0)
    ):
        raise ValueError("projection arm counters or identity are invalid")
    expected_chunks = (
        commands // EXECUTION_HORIZON + 1
        if treatment_failure
        else max(1, (commands + EXECUTION_HORIZON - 1) // EXECUTION_HORIZON)
    )
    if chunks != expected_chunks:
        raise ValueError("projection arm chunk count is inconsistent with execution")
    if treatment_failure:
        details = arm.get("treatment_failure_details")
        if (
            arm.get("failure_reason") is not None
            or arm.get("treatment_failure_reason") != "alpha_zero_infeasible"
            or not isinstance(details, dict)
            or details.get("trial_count") != len(ALPHA_GRID)
            or details.get("lead") not in range(EXECUTION_HORIZON)
            or details.get("chunk_index") != chunks - 1
            or details.get("prepared_commands_discarded") != decoded_commands - commands
        ):
            raise ValueError("projection treatment failure state is invalid")
    elif (
        arm.get("treatment_failure_reason") is not None
        or arm.get("treatment_failure_details") is not None
        or (success and arm.get("failure_reason") is not None)
        or (success and arm.get("final_stage") != "success")
        or (not success and not isinstance(arm.get("failure_reason"), str))
        or (not success and not isinstance(arm.get("final_stage"), str))
    ):
        raise ValueError("projection task outcome state is invalid")
    if expected_arm == "baseline" and arm.get("decode_trials") != decoded_commands:
        raise ValueError("baseline must decode exactly once per prepared command")
    if expected_arm == "projection":
        minimum_trials = decoded_commands + (len(ALPHA_GRID) if treatment_failure else 0)
        maximum_trials = len(ALPHA_GRID) * (decoded_commands + int(treatment_failure))
        if not minimum_trials <= arm.get("decode_trials", -1) <= maximum_trials:
            raise ValueError("projection decode trial count is invalid")
    histogram = arm.get("alpha_histogram", {})
    if set(histogram) != set(ALPHA_LABELS) or any(not isinstance(value, int) or value < 0 for value in histogram.values()):
        raise ValueError("projection alpha histogram is invalid")
    if sum(histogram.values()) != commands:
        raise ValueError("projection alpha histogram count does not match commands")
    if arm.get("alpha_zero_count") != histogram["0"] or arm.get("alpha_one_count") != histogram["1"]:
        raise ValueError("projection alpha counters do not match histogram")
    if arm.get("intervention_count") != commands - histogram["1"]:
        raise ValueError("projection intervention count does not match histogram")
    histogram_by_lead = arm.get("alpha_histogram_by_lead", {})
    expected_leads = {str(lead) for lead in range(EXECUTION_HORIZON)}
    if set(histogram_by_lead) != expected_leads or any(
        set(value) != set(ALPHA_LABELS)
        or any(not isinstance(count, int) or count < 0 for count in value.values())
        for value in histogram_by_lead.values()
    ):
        raise ValueError("projection lead-specific alpha histogram is invalid")
    if any(
        histogram[label]
        != sum(histogram_by_lead[str(lead)][label] for lead in range(EXECUTION_HORIZON))
        for label in ALPHA_LABELS
    ):
        raise ValueError("projection alpha histograms do not agree across leads")
    expected_alpha_counts = {
        "intervention_count_by_lead": {
            lead: sum(count for label, count in values.items() if label != "1")
            for lead, values in histogram_by_lead.items()
        },
        "alpha_zero_count_by_lead": {
            lead: values["0"] for lead, values in histogram_by_lead.items()
        },
        "alpha_one_count_by_lead": {
            lead: values["1"] for lead, values in histogram_by_lead.items()
        },
    }
    if any(arm.get(name) != value for name, value in expected_alpha_counts.items()):
        raise ValueError("projection lead-specific alpha counters do not match histograms")
    group = arm.get("geometry_by_lead", {})
    if set(group) != set(_metric_group()):
        raise ValueError("projection geometry metric names are invalid")
    for values in group.values():
        if set(values) != expected_leads:
            raise ValueError("projection geometry lead ids are invalid")
        for value in values.values():
            _validate_compact(value)
    counts = [group["translation_m"][str(lead)]["count"] for lead in range(EXECUTION_HORIZON)]
    if sum(counts) != commands or any(
        group[name][str(lead)]["count"] != counts[lead]
        for name in group
        for lead in range(EXECUTION_HORIZON)
    ):
        raise ValueError("projection geometry sample counts do not match commands")
    clipping_counts = group["clipped"]
    clipping_errors = group["clipping_error_native"]
    telemetry_clipped = sum(float(value["sum"]) for value in clipping_counts.values())
    telemetry_maximum = max(
        (float(value["maximum"]) for value in clipping_errors.values()), default=0.0
    )
    if (
        telemetry_clipped != clipped
        or any(
            value["sum_squared"] != value["sum"] or value["maximum"] > 1.0
            for value in clipping_counts.values()
        )
        or arm.get("maximum_clipping_error_native") != telemetry_maximum
    ):
        raise ValueError("projection clipping counters do not match compact telemetry")
    milestones = arm.get("milestones")
    if not isinstance(milestones, dict) or tuple(milestones) != _milestone_names(task):
        raise ValueError("ordered milestone schema is invalid")
    attained = []
    for value in milestones.values():
        if set(value) != {"attained", "first_step"} or not isinstance(value["attained"], bool):
            raise ValueError("ordered milestone value is invalid")
        first_step = value["first_step"]
        if value["attained"] != (type(first_step) is int) or (
            type(first_step) is int and not 1 <= first_step <= steps
        ):
            raise ValueError("ordered milestone step is invalid")
        attained.append(value["attained"])
    if attained != sorted(attained, reverse=True):
        raise ValueError("milestones must form an ordered attained prefix")
    if arm.get("ordered_progression") != sum(attained) / len(attained):
        raise ValueError("ordered progression does not recompute")
    if expected_arm == "projection" and any(
        group[name][str(lead)]["maximum"] > THRESHOLDS[name]
        for name in THRESHOLDS
        for lead in range(EXECUTION_HORIZON)
    ):
        raise ValueError("an accepted projected command exceeds a delivery threshold")


def _validate_outcomes(report: dict[str, Any]) -> None:
    identity = report.get("identity", {})
    for ordinal, outcome in enumerate(report.get("outcomes", [])):
        task = outcome.get("task")
        initial_inputs = outcome.get("initial_inputs", {})
        first_chunks = outcome.get("first_chunks", {})
        valid_initial_hashes = isinstance(initial_inputs, dict) and set(initial_inputs) == set(
            ARMS
        ) and all(
            isinstance(value, dict)
            and set(value)
            == {
                "top_image_sha256",
                "canonical_state_sha256",
                "instruction_sha256",
                "combined_sha256",
            }
            and all(
                isinstance(digest, str)
                and len(digest) == 64
                and set(digest) <= set("0123456789abcdef")
                for digest in value.values()
            )
            for value in initial_inputs.values()
        )
        valid_chunk_hashes = isinstance(first_chunks, dict) and set(first_chunks) == set(ARMS) and all(
            isinstance(digest, str)
            and len(digest) == 64
            and set(digest) <= set("0123456789abcdef")
            for digest in first_chunks.values()
        )
        chunk_difference = outcome.get("first_chunk_max_abs_difference")
        if (
            not isinstance(outcome.get("scenario_id"), str)
            or task not in TASKS
            or outcome.get("morphology") != identity.get("morphology")
            or outcome.get("arm_order")
            != list(
                pair_arm_order(
                    ordinal,
                    identity.get("condition"),
                    identity.get("model_seed"),
                    identity.get("morphology"),
                )
            )
            or outcome.get("initial_input_hash_agreement") is not True
            or not valid_initial_hashes
            or initial_inputs.get("baseline") != initial_inputs.get("projection")
            or not valid_chunk_hashes
            or not isinstance(chunk_difference, (int, float))
            or not np.isfinite(chunk_difference)
            or not 0.0 <= chunk_difference <= FIRST_CHUNK_MAX_ABS
            or set(outcome.get("arms", {})) != set(ARMS)
        ):
            raise ValueError("projection pair identity or invariant is invalid")
        for arm in ARMS:
            _validate_arm(outcome["arms"][arm], arm, task)


def _new_report(
    *,
    registered: bool,
    contract: dict[str, Any],
    identity: dict[str, Any],
    expected_ids: list[str],
    provenance: dict[str, Any],
    run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": 39,
        "mode": "projection_shard",
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
    expected_run = expected.get("run")
    report_run = report.get("run")
    stable_run_fields = (
        "device",
        "lock_identity",
        "lock_path",
        "schedule_identity",
        "schedule_position",
    )
    if (expected_run is None) != (report_run is None) or (
        expected_run is not None
        and (
            not isinstance(report_run, dict)
            or any(report_run.get(name) != expected_run.get(name) for name in stable_run_fields)
            or _parse_utc(report_run.get("started_utc")) > datetime.now(timezone.utc)
            or (
                report_run.get("finished_utc") is not None
                and _parse_utc(report_run["finished_utc"])
                < _parse_utc(report_run["started_utc"])
            )
        )
    ):
        raise ValueError("resume report runtime or schedule identity does not match")
    ids = [outcome.get("scenario_id") for outcome in report.get("outcomes", [])]
    if ids != expected["expected_outcome_ids"][: len(ids)] or len(ids) != len(set(ids)):
        raise ValueError("resume report pairs are not a unique ordered contract prefix")
    if report.get("outcome_chain_sha256") != _outcome_chain(report.get("outcomes", [])):
        raise ValueError("resume report outcome hash chain does not match")
    if not isinstance(report.get("infrastructure_errors"), list):
        raise ValueError("resume report infrastructure errors are invalid")
    if not isinstance(report.get("treatment_delivery_errors"), list):
        raise ValueError("resume report treatment-delivery errors are invalid")
    _validate_outcomes(report)
    expected_treatment_errors = [
        {
            "scenario_id": outcome["scenario_id"],
            "arm": "projection",
            "reason": outcome["arms"]["projection"]["treatment_failure_reason"],
            "details": outcome["arms"]["projection"]["treatment_failure_details"],
        }
        for outcome in report.get("outcomes", [])
        if outcome["arms"]["projection"]["treatment_failure"]
    ]
    if report["treatment_delivery_errors"] != expected_treatment_errors:
        raise ValueError("resume report treatment-delivery errors do not match outcomes")
    if report.get("complete") and (
        ids != expected["expected_outcome_ids"] or report.get("infrastructure_errors")
    ):
        raise ValueError("completed projection report is incomplete or has infrastructure errors")
    if report.get("complete") and (
        report.get("status")
        != ("completed" if report.get("registered") else "completed_smoke_unregistered")
        or report_run is None
        or report_run.get("finished_utc") is None
        or report.get("aggregate") != _shard_aggregate(report["outcomes"])
    ):
        raise ValueError("completed projection report status or aggregate is invalid")
    if not report.get("complete") and (report.get("status") != expected["status"] or "aggregate" in report):
        raise ValueError("incomplete projection report has stale derived state")
    return report


def validate_exp38_summary(path: Path) -> dict[str, Any]:
    if file_sha256(path) != EXP38_SUMMARY_SHA256:
        raise ValueError("Experiment 038 summary hash does not match registration")
    if file_sha256(Path(exp38.__file__)) != EXP38_EVALUATOR_SHA256:
        raise ValueError("imported Experiment 038 evaluator hash does not match its summary contract")
    summary = json.loads(path.read_text())
    _validate_exp38_summary_contract(summary)
    return summary


def _validate_exp38_summary_contract(summary: dict[str, Any]) -> None:
    contract = summary.get("contract", {})
    if (
        summary.get("experiment") != 38
        or summary.get("registered") is not True
        or summary.get("status") != "completed_diagnostic"
        or contract.get("definition_sha256") != DEFINITION_SHA256
        or contract.get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
        or contract.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
        or contract.get("evaluator_sha256") != EXP38_EVALUATOR_SHA256
        or contract.get("shards") != 12
        or contract.get("episodes") != 756
        or contract.get("final_split_loaded") is not False
        or set(summary.get("horizon_mechanism", {})) != set(CONDITIONS)
    ):
        raise ValueError("Experiment 038 summary contract is not the frozen diagnostic")


def _merge_metric_groups(outcomes: Iterable[dict[str, Any]], arm: str) -> dict[str, Any]:
    outcomes = list(outcomes)
    return {
        name: {
            str(lead): exp38.merge_compact(
                outcome["arms"][arm]["geometry_by_lead"][name][str(lead)]
                for outcome in outcomes
            )
            for lead in range(EXECUTION_HORIZON)
        }
        for name in _metric_group()
    }


def _shard_aggregate(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_outcomes = [
        outcome for outcome in outcomes if not outcome["arms"]["projection"]["treatment_failure"]
    ]
    arms = {}
    for arm in ARMS:
        commands = sum(outcome["arms"][arm]["commands"] for outcome in outcomes)
        clipped = sum(outcome["arms"][arm]["clipped_commands"] for outcome in outcomes)
        geometry = _merge_metric_groups(outcomes, arm)
        arms[arm] = {
            "episodes": len(outcomes),
            "eligible_pair_episodes": len(eligible_outcomes),
            "successes": sum(
                outcome["arms"][arm]["success"] is True for outcome in eligible_outcomes
            ),
            "commands": commands,
            "decoded_commands": sum(
                outcome["arms"][arm]["decoded_commands"] for outcome in outcomes
            ),
            "clipped_commands": clipped,
            "clipping_rate": clipped / commands if commands else 0.0,
            "geometry_by_lead": geometry,
            "alpha_histogram": {
                label: sum(outcome["arms"][arm]["alpha_histogram"][label] for outcome in outcomes)
                for label in ALPHA_LABELS
            },
            "alpha_histogram_by_lead": {
                str(lead): {
                    label: sum(
                        outcome["arms"][arm]["alpha_histogram_by_lead"][str(lead)][label]
                        for outcome in outcomes
                    )
                    for label in ALPHA_LABELS
                }
                for lead in range(EXECUTION_HORIZON)
            },
            "intervention_count_by_lead": {
                str(lead): sum(
                    outcome["arms"][arm]["intervention_count_by_lead"][str(lead)]
                    for outcome in outcomes
                )
                for lead in range(EXECUTION_HORIZON)
            },
            "alpha_zero_count_by_lead": {
                str(lead): sum(
                    outcome["arms"][arm]["alpha_zero_count_by_lead"][str(lead)]
                    for outcome in outcomes
                )
                for lead in range(EXECUTION_HORIZON)
            },
            "alpha_one_count_by_lead": {
                str(lead): sum(
                    outcome["arms"][arm]["alpha_one_count_by_lead"][str(lead)]
                    for outcome in outcomes
                )
                for lead in range(EXECUTION_HORIZON)
            },
            "intervention_count": sum(
                outcome["arms"][arm]["intervention_count"] for outcome in outcomes
            ),
            "alpha_zero_count": sum(outcome["arms"][arm]["alpha_zero_count"] for outcome in outcomes),
            "alpha_one_count": sum(outcome["arms"][arm]["alpha_one_count"] for outcome in outcomes),
        }
    cells = {}
    for task in TASKS:
        values = [outcome for outcome in outcomes if outcome["task"] == task]
        eligible_values = [
            outcome
            for outcome in values
            if not outcome["arms"]["projection"]["treatment_failure"]
        ]
        cells[task] = {
            arm: {
                "attempted": len(values),
                "eligible_pairs": len(eligible_values),
                "successes": sum(
                    value["arms"][arm]["success"] is True for value in eligible_values
                ),
                "success_rate": (
                    sum(value["arms"][arm]["success"] is True for value in eligible_values)
                    / len(eligible_values)
                    if eligible_values
                    else None
                ),
                "mean_ordered_progression": (
                    sum(
                        value["arms"][arm]["ordered_progression"] for value in eligible_values
                    )
                    / len(eligible_values)
                    if eligible_values
                    else None
                ),
                "geometry_by_lead": _merge_metric_groups(values, arm),
                "alpha_histogram": {
                    label: sum(
                        value["arms"][arm]["alpha_histogram"][label] for value in values
                    )
                    for label in ALPHA_LABELS
                },
                "alpha_histogram_by_lead": {
                    str(lead): {
                        label: sum(
                            value["arms"][arm]["alpha_histogram_by_lead"][str(lead)][label]
                            for value in values
                        )
                        for label in ALPHA_LABELS
                    }
                    for lead in range(EXECUTION_HORIZON)
                },
                "intervention_count_by_lead": {
                    str(lead): sum(
                        value["arms"][arm]["intervention_count_by_lead"][str(lead)]
                        for value in values
                    )
                    for lead in range(EXECUTION_HORIZON)
                },
                "alpha_zero_count_by_lead": {
                    str(lead): sum(
                        value["arms"][arm]["alpha_zero_count_by_lead"][str(lead)]
                        for value in values
                    )
                    for lead in range(EXECUTION_HORIZON)
                },
                "alpha_one_count_by_lead": {
                    str(lead): sum(
                        value["arms"][arm]["alpha_one_count_by_lead"][str(lead)]
                        for value in values
                    )
                    for lead in range(EXECUTION_HORIZON)
                },
                "intervention_count": sum(
                    value["arms"][arm]["intervention_count"] for value in values
                ),
                "alpha_zero_count": sum(
                    value["arms"][arm]["alpha_zero_count"] for value in values
                ),
                "alpha_one_count": sum(
                    value["arms"][arm]["alpha_one_count"] for value in values
                ),
            }
            for arm in ARMS
        }
    projection_geometry_passed = all(
        arms["projection"]["geometry_by_lead"][name][str(lead)]["maximum"] <= threshold
        for name, threshold in THRESHOLDS.items()
        for lead in range(EXECUTION_HORIZON)
    )
    delivery = {
        "treatment_failures": len(outcomes) - len(eligible_outcomes),
        "all_projected_accepted_commands_pass": projection_geometry_passed,
        "arm_clipping_at_most_one_percent": {
            arm: arms[arm]["clipping_rate"] <= 0.01 for arm in ARMS
        },
    }
    delivery["passed"] = (
        delivery["treatment_failures"] == 0
        and projection_geometry_passed
        and all(delivery["arm_clipping_at_most_one_percent"].values())
    )
    return {
        "pairs": len(outcomes),
        "eligible_pairs": len(eligible_outcomes),
        "arms": arms,
        "tasks": cells,
        "delivery_gate": delivery,
    }


def evaluate_projection_shard(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    exp36_summary_path: Path,
    exp38_summary_path: Path,
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
        raise ValueError("registered Experiment 039 does not permit a scenario limit")
    provenance = _runtime_provenance(device)
    if registered and provenance["git_dirty"]:
        raise RuntimeError("registered Experiment 039 must run from a clean git worktree")
    validate_exp38_summary(exp38_summary_path)
    checkpoint_condition = source_condition(condition, morphology)
    checkpoint, identity = resolve_checkpoint(
        exp36_summary_path, checkpoint_condition, seed, morphology, root=root
    )
    identity["condition"] = condition
    identity["source_condition"] = checkpoint_condition
    scenarios = projection_scenarios(
        manifest,
        morphology,
        registered=registered,
        limit_scenarios=limit_scenarios,
    )
    contract = {
        "definition_sha256": definition.sha256,
        "manifest_sha256": file_sha256(manifest.path),
        "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
        "exp38_summary_sha256": EXP38_SUMMARY_SHA256,
        "exp38_evaluator_sha256": EXP38_EVALUATOR_SHA256,
        "split": "development",
        "scenario_indices": list(SCENARIO_INDICES),
        "tasks": list(TASKS),
        "limit_scenarios": limit_scenarios,
        "protocol": PROJECTION_PROTOCOL,
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
        aggregate = _shard_aggregate(report["outcomes"])
        if aggregate != report.get("aggregate"):
            raise ValueError("completed projection shard aggregate does not recompute")
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
    for ordinal, scenario in enumerate(scenarios[len(report["outcomes"]) :], start=len(report["outcomes"])):
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
            outcome = run_projection_pair(
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
            projection = outcome["arms"]["projection"]
            if projection["treatment_failure"]:
                report["treatment_delivery_errors"].append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "arm": "projection",
                        "reason": projection["treatment_failure_reason"],
                        "details": projection["treatment_failure_details"],
                    }
                )
            _append_outcome(report, outcome)
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
                    active_error.add_note(f"{arm} environment close failed: {close_error!r}")
    if report["infrastructure_errors"]:
        raise RuntimeError("projection shard cannot complete with infrastructure errors")
    report["aggregate"] = _shard_aggregate(report["outcomes"])
    report["complete"] = True
    report["status"] = "completed" if registered else "completed_smoke_unregistered"
    report["run"]["finished_utc"] = _utc_now()
    _write_progress(output, report)
    return report


def classify_causal_support(
    seed_success_gains: dict[int, float | None],
    seed_progression_gains: dict[int, float | None],
    *,
    delivery_gate_passed: bool,
) -> dict[str, Any]:
    if set(seed_success_gains) != set(SEEDS) or set(seed_progression_gains) != set(SEEDS):
        raise ValueError("causal classification requires the three registered seeds")
    success_complete = all(value is not None for value in seed_success_gains.values())
    progression_complete = all(value is not None for value in seed_progression_gains.values())
    mean_success = (
        sum(float(value) for value in seed_success_gains.values()) / len(SEEDS)
        if success_complete
        else None
    )
    mean_progression = (
        sum(float(value) for value in seed_progression_gains.values()) / len(SEEDS)
        if progression_complete
        else None
    )
    nonnegative = sum(value is not None and value >= 0.0 for value in seed_success_gains.values())
    checks = {
        "three_seed_mean_success_gain_at_least_five_points": (
            mean_success is not None and mean_success + 1e-12 >= 0.05
        ),
        "at_least_two_nonnegative_seed_success_gains": nonnegative >= 2,
        "three_seed_mean_task_equal_ordered_progression_gain_positive": (
            mean_progression is not None and mean_progression > 0.0
        ),
        "geometry_delivery_gate_passed": delivery_gate_passed,
    }
    return {
        "classification": "development_diagnostic_causal_support" if all(checks.values()) else "no_causal_support",
        "supported": all(checks.values()),
        "checks": checks,
        "three_seed_mean_paired_success_gain_percentage_points": (
            None if mean_success is None else 100.0 * mean_success
        ),
        "nonnegative_direction_seeds": nonnegative,
        "three_seed_mean_task_equal_ordered_progression_gain": mean_progression,
        "per_seed_success_gain_percentage_points": {
            str(seed): (
                None
                if seed_success_gains[seed] is None
                else 100.0 * seed_success_gains[seed]
            )
            for seed in SEEDS
        },
        "per_seed_task_equal_ordered_progression_gain": {
            str(seed): seed_progression_gains[seed] for seed in SEEDS
        },
        "materiality_margin_percentage_points": 5.0,
        "inferential_claim": False,
    }


def registered_shard_filename(condition: str, seed: int, morphology: str) -> str:
    return f"benchmark-exp39-{condition}-seed{seed}-{morphology}-projection.json"


def _expected_scenario_ids(manifest: ScenarioManifest, morphology: str) -> list[str]:
    return [
        scenario.scenario_id
        for scenario in projection_scenarios(manifest, morphology, registered=True)
    ]


def _task_equal_progression(values: list[dict[str, Any]], arm: str) -> float | None:
    task_means = []
    for task in TASKS:
        task_values = [value["arms"][arm]["ordered_progression"] for value in values if value["task"] == task]
        if not task_values:
            return None
        task_means.append(sum(task_values) / len(task_values))
    return sum(task_means) / len(task_means)


def _nullable_difference(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else first - second


def _nullable_mean(values: Iterable[float | None]) -> float | None:
    values = list(values)
    return None if any(value is None for value in values) else sum(float(value) for value in values) / len(values)


def aggregate_registered_shards(
    reports: list[dict[str, Any]],
    manifest: ScenarioManifest,
    *,
    exp36_summary: dict[str, Any],
    exp38_summary: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if (
        exp36_summary.get("experiment") != 36
        or exp36_summary.get("status") != "completed_gate_passed"
        or exp36_summary.get("fixed_step") != 1200
    ):
        raise ValueError("registered aggregation requires the frozen Experiment 036 summary")
    try:
        _validate_exp38_summary_contract(exp38_summary)
    except ValueError as error:
        raise ValueError("registered aggregation requires the frozen Experiment 038 summary") from error
    indexed = {}
    for report in reports:
        identity = report.get("identity", {})
        key = (identity.get("condition"), identity.get("model_seed"), identity.get("morphology"))
        if key in indexed:
            raise ValueError(f"duplicate Experiment 039 shard: {key}")
        indexed[key] = report
    expected_keys = {
        (condition, seed, morphology)
        for condition in CONDITIONS
        for seed in SEEDS
        for morphology in MORPHOLOGIES
    }
    expected_source_keys = {
        f"{condition}/seed{seed}/{morphology}"
        for condition, seed, morphology in expected_keys
    }
    if set(source_hashes) != expected_source_keys or any(
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
        for value in source_hashes.values()
    ):
        raise ValueError("registered aggregation requires exact hashes for all 12 source shards")
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise ValueError(f"registered aggregation requires all 12 shards; missing={missing}, extra={extra}")
    evaluator_sha256 = file_sha256(Path(__file__))
    commits = set()
    runtime_signatures = set()
    runtimes = []
    lock_paths = set()
    intervals: list[tuple[int, datetime, datetime, str]] = []
    all_outcomes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    shard_summaries = {}
    delivery_by_shard = {}
    for key in sorted(expected_keys):
        condition, seed, morphology = key
        report = indexed[key]
        expected_ids = _expected_scenario_ids(manifest, morphology)
        outcomes = report.get("outcomes", [])
        contract = report.get("contract", {})
        identity = report.get("identity", {})
        provenance = report.get("provenance", {})
        run = report.get("run", {})
        checkpoint_condition = source_condition(condition, morphology)
        expected_run = exp36_summary.get("runs", {}).get(f"{checkpoint_condition}-seed{seed}", {})
        runtime = provenance.get("runtime", {})
        if (
            report.get("experiment") != 39
            or report.get("mode") != "projection_shard"
            or report.get("registered") is not True
            or report.get("complete") is not True
            or report.get("status") != "completed"
            or report.get("expected_outcome_ids") != expected_ids
            or [value.get("scenario_id") for value in outcomes] != expected_ids
            or len(outcomes) != 21
            or contract.get("definition_sha256") != DEFINITION_SHA256
            or contract.get("manifest_sha256") != DEVELOPMENT_MANIFEST_SHA256
            or contract.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
            or contract.get("exp38_summary_sha256") != EXP38_SUMMARY_SHA256
            or contract.get("exp38_evaluator_sha256") != EXP38_EVALUATOR_SHA256
            or contract.get("split") != "development"
            or contract.get("scenario_indices") != list(SCENARIO_INDICES)
            or contract.get("tasks") != list(TASKS)
            or contract.get("limit_scenarios") is not None
            or contract.get("protocol") != PROJECTION_PROTOCOL
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
            or not isinstance(runtime.get("device"), str)
            or not isinstance(runtime.get("python"), str)
            or not isinstance(runtime.get("torch"), str)
            or not isinstance(runtime.get("numpy"), str)
            or not isinstance(runtime.get("mujoco"), str)
            or (
                runtime.get("device", "").startswith("cuda")
                and not isinstance(runtime.get("cuda_device_name"), str)
            )
            or any(
                not isinstance(runtime.get(name), bool)
                for name in (
                    "torch_deterministic_algorithms",
                    "torch_deterministic_warn_only",
                    "cudnn_deterministic",
                    "cudnn_benchmark",
                    "cuda_matmul_allow_tf32",
                )
            )
            or provenance.get("runtime_signature_sha256")
            != hashlib.sha256(
                json.dumps(
                    runtime, sort_keys=True, separators=(",", ":")
                ).encode()
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
            raise ValueError(f"registered Experiment 039 shard has the wrong contract: {key}")
        _validate_outcomes(report)
        expected_treatment_errors = [
            {
                "scenario_id": outcome["scenario_id"],
                "arm": "projection",
                "reason": outcome["arms"]["projection"]["treatment_failure_reason"],
                "details": outcome["arms"]["projection"]["treatment_failure_details"],
            }
            for outcome in outcomes
            if outcome["arms"]["projection"]["treatment_failure"]
        ]
        if report.get("treatment_delivery_errors") != expected_treatment_errors:
            raise ValueError(f"registered treatment-delivery records are invalid: {key}")
        for outcome, scenario in zip(
            outcomes,
            projection_scenarios(manifest, morphology, registered=True),
            strict=True,
        ):
            if (
                outcome.get("scenario_seed") != scenario.seed
                or outcome.get("task") != scenario.task
                or outcome.get("active_factor_axes") != _active_factor_axes(scenario)
            ):
                raise ValueError(f"registered projection outcome metadata is invalid: {key}")
        aggregate = _shard_aggregate(outcomes)
        if report.get("aggregate") != aggregate:
            raise ValueError(f"registered projection shard aggregate does not recompute: {key}")
        commits.add(provenance["git_commit"])
        runtime_signatures.add(provenance["runtime_signature_sha256"])
        runtimes.append(provenance["runtime"])
        lock_paths.add(run["lock_path"])
        label = f"{condition}/seed{seed}/{morphology}"
        started = _parse_utc(run["started_utc"])
        finished = _parse_utc(run["finished_utc"])
        if finished < started:
            raise ValueError(f"registered Experiment 039 shard interval is negative: {key}")
        intervals.append(
            (
                run["schedule_position"],
                started,
                finished,
                label,
            )
        )
        shard_summaries[label] = aggregate
        delivery_by_shard[label] = aggregate["delivery_gate"]
        all_outcomes[(condition, seed)].extend(outcomes)
    if len(commits) != 1:
        raise ValueError("registered Experiment 039 shards must share one clean evaluator commit")
    if len(runtime_signatures) != 1 or any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("registered Experiment 039 shards must share one runtime signature")
    if len(lock_paths) != 1:
        raise ValueError("registered Experiment 039 shards must share one advisory lock path")
    intervals.sort()
    if [position for position, *_rest in intervals] != list(range(len(REGISTERED_SCHEDULE))) or any(
        first[2] > second[1] for first, second in zip(intervals, intervals[1:], strict=False)
    ):
        raise ValueError("registered Experiment 039 shard intervals overlap or violate schedule order")

    per_seed = {}
    paired_discordants = {}
    task_gains = {}
    morphology_gains = {}
    causal_support = {}
    for condition in CONDITIONS:
        seed_success_gains = {}
        seed_progression_gains = {}
        for seed in SEEDS:
            values = all_outcomes[(condition, seed)]
            eligible = [
                value for value in values if not value["arms"]["projection"]["treatment_failure"]
            ]
            baseline_success = (
                sum(value["arms"]["baseline"]["success"] is True for value in eligible)
                / len(eligible)
                if eligible
                else None
            )
            projection_success = (
                sum(value["arms"]["projection"]["success"] is True for value in eligible)
                / len(eligible)
                if eligible
                else None
            )
            success_gain = _nullable_difference(projection_success, baseline_success)
            progression = {
                arm: _task_equal_progression(eligible, arm) for arm in ARMS
            }
            progression_gain = _nullable_difference(
                progression["projection"], progression["baseline"]
            )
            seed_success_gains[seed] = success_gain
            seed_progression_gains[seed] = progression_gain
            per_seed[f"{condition}/seed{seed}"] = {
                "pairs": len(values),
                "eligible_pairs": len(eligible),
                "treatment_failed_pairs": len(values) - len(eligible),
                "baseline_success_rate": baseline_success,
                "projection_success_rate": projection_success,
                "paired_success_gain_percentage_points": (
                    None if success_gain is None else 100.0 * success_gain
                ),
                "baseline_task_equal_ordered_progression": progression["baseline"],
                "projection_task_equal_ordered_progression": progression["projection"],
                "task_equal_ordered_progression_gain": progression_gain,
            }
            wins = sum(
                not value["arms"]["baseline"]["success"]
                and value["arms"]["projection"]["success"]
                for value in eligible
            )
            losses = sum(
                value["arms"]["baseline"]["success"]
                and not value["arms"]["projection"]["success"]
                for value in eligible
            )
            paired_discordants[f"{condition}/seed{seed}"] = {
                "eligible_pairs": len(eligible),
                "projection_only_success": wins,
                "baseline_only_success": losses,
                "discordant_pairs": wins + losses,
            }
            for task in TASKS:
                cell = [value for value in values if value["task"] == task]
                eligible_cell = [
                    value
                    for value in cell
                    if not value["arms"]["projection"]["treatment_failure"]
                ]
                success_cell_gain = (
                    (
                        sum(
                            value["arms"]["projection"]["success"] is True
                            for value in eligible_cell
                        )
                        - sum(
                            value["arms"]["baseline"]["success"] is True
                            for value in eligible_cell
                        )
                    )
                    / len(eligible_cell)
                    if eligible_cell
                    else None
                )
                progression_cell_gain = (
                    (
                        sum(
                            value["arms"]["projection"]["ordered_progression"]
                            for value in eligible_cell
                        )
                        - sum(
                            value["arms"]["baseline"]["ordered_progression"]
                            for value in eligible_cell
                        )
                    )
                    / len(eligible_cell)
                    if eligible_cell
                    else None
                )
                task_gains[f"{condition}/seed{seed}/{task}"] = {
                    "pairs": len(cell),
                    "eligible_pairs": len(eligible_cell),
                    "success_gain_percentage_points": (
                        None if success_cell_gain is None else 100.0 * success_cell_gain
                    ),
                    "ordered_progression_gain": progression_cell_gain,
                }
            for morphology in MORPHOLOGIES:
                cell = [value for value in values if value["morphology"] == morphology]
                eligible_cell = [
                    value
                    for value in cell
                    if not value["arms"]["projection"]["treatment_failure"]
                ]
                success_cell_gain = (
                    (
                        sum(
                            value["arms"]["projection"]["success"] is True
                            for value in eligible_cell
                        )
                        - sum(
                            value["arms"]["baseline"]["success"] is True
                            for value in eligible_cell
                        )
                    )
                    / len(eligible_cell)
                    if eligible_cell
                    else None
                )
                progression_cell_gain = (
                    (
                        sum(
                            value["arms"]["projection"]["ordered_progression"]
                            for value in eligible_cell
                        )
                        - sum(
                            value["arms"]["baseline"]["ordered_progression"]
                            for value in eligible_cell
                        )
                    )
                    / len(eligible_cell)
                    if eligible_cell
                    else None
                )
                morphology_gains[f"{condition}/seed{seed}/{morphology}"] = {
                    "pairs": len(cell),
                    "eligible_pairs": len(eligible_cell),
                    "success_gain_percentage_points": (
                        None if success_cell_gain is None else 100.0 * success_cell_gain
                    ),
                    "ordered_progression_gain": progression_cell_gain,
                }
        delivery_passed = all(
            delivery_by_shard[f"{condition}/seed{seed}/{morphology}"]["passed"]
            for seed in SEEDS
            for morphology in MORPHOLOGIES
        )
        causal_support[condition] = classify_causal_support(
            seed_success_gains,
            seed_progression_gains,
            delivery_gate_passed=delivery_passed,
        )
    three_seed_task_gains = {}
    for condition in CONDITIONS:
        for task in TASKS:
            values = [task_gains[f"{condition}/seed{seed}/{task}"] for seed in SEEDS]
            three_seed_task_gains[f"{condition}/{task}"] = {
                "equal_weight_success_gain_percentage_points": _nullable_mean(
                    value["success_gain_percentage_points"] for value in values
                ),
                "equal_weight_ordered_progression_gain": _nullable_mean(
                    value["ordered_progression_gain"] for value in values
                ),
            }
    three_seed_morphology_gains = {}
    for condition in CONDITIONS:
        for morphology in MORPHOLOGIES:
            values = [
                morphology_gains[f"{condition}/seed{seed}/{morphology}"] for seed in SEEDS
            ]
            three_seed_morphology_gains[f"{condition}/{morphology}"] = {
                "equal_weight_success_gain_percentage_points": _nullable_mean(
                    value["success_gain_percentage_points"] for value in values
                ),
                "equal_weight_ordered_progression_gain": _nullable_mean(
                    value["ordered_progression_gain"] for value in values
                ),
            }
    discordants_by_condition = {
        condition: {
            name: sum(
                paired_discordants[f"{condition}/seed{seed}"][name] for seed in SEEDS
            )
            for name in (
                "projection_only_success",
                "baseline_only_success",
                "discordant_pairs",
            )
        }
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        discordants_by_condition[condition]["eligible_pairs"] = sum(
            paired_discordants[f"{condition}/seed{seed}"]["eligible_pairs"] for seed in SEEDS
        )
    return {
        "format_version": 1,
        "experiment": 39,
        "registered": True,
        "status": "completed_causal_diagnostic",
        "scope": "development-only intervention; no training, admission, selection, or advancement gate",
        "contract": {
            "definition_sha256": DEFINITION_SHA256,
            "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
            "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
            "exp38_summary_sha256": EXP38_SUMMARY_SHA256,
            "exp38_evaluator_sha256": EXP38_EVALUATOR_SHA256,
            "evaluator_sha256": evaluator_sha256,
            "evaluator_git_commit": next(iter(commits)),
            "shards": 12,
            "pairs": 252,
            "episodes": 504,
            "final_split_loaded": False,
            "runtime_signature_sha256": next(iter(runtime_signatures)),
        },
        "source_shard_sha256": dict(sorted(source_hashes.items())),
        "shard_summaries": shard_summaries,
        "delivery_gates_by_shard": delivery_by_shard,
        "paired_discordants": paired_discordants,
        "paired_discordants_by_condition": discordants_by_condition,
        "per_task_gains": task_gains,
        "equal_weight_three_seed_task_gains": three_seed_task_gains,
        "per_morphology_gains": morphology_gains,
        "equal_weight_three_seed_morphology_gains": three_seed_morphology_gains,
        "equal_weight_seed_gains": per_seed,
        "causal_support_by_condition": causal_support,
        "multiplicity_adjusted_inferential_claim": False,
        "advancement_gate": None,
    }


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run Experiment 039 feasibility projection")
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
        "--exp38-summary", type=Path, default=Path("reports/benchmark-exp38-summary.json")
    )
    shard.add_argument("--artifact-root", type=Path, default=Path("."))
    shard.add_argument("--runtime-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    shard.add_argument("--output", type=Path)
    shard.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    shard.add_argument("--smoke", action="store_true")
    shard.add_argument("--limit-scenarios", type=_positive)
    registered_run = subparsers.add_parser("run-registered", parents=[common])
    registered_run.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    registered_run.add_argument(
        "--exp38-summary", type=Path, default=Path("reports/benchmark-exp38-summary.json")
    )
    registered_run.add_argument("--artifact-root", type=Path, default=Path("."))
    registered_run.add_argument("--runtime-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    registered_run.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--shard-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    aggregate.add_argument(
        "--exp36-summary", type=Path, default=Path("reports/benchmark-exp36-summary.json")
    )
    aggregate.add_argument(
        "--exp38-summary", type=Path, default=Path("reports/benchmark-exp38-summary.json")
    )
    aggregate.add_argument("--output", type=Path, default=Path("reports/benchmark-exp39-summary.json"))
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
                condition,
                seed,
                morphology,
                None,
                runtime_dir=args.runtime_dir,
            )
            reports.append(
                evaluate_projection_shard(
                    definition,
                    manifest,
                    args.exp36_summary,
                    args.exp38_summary,
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
        raise ValueError("Experiment 039 frozen benchmark contract does not match")
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
                report = evaluate_projection_shard(
                    definition,
                    manifest,
                    args.exp36_summary,
                    args.exp38_summary,
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
            report = evaluate_projection_shard(
                definition,
                manifest,
                args.exp36_summary,
                args.exp38_summary,
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
    exp38_summary = validate_exp38_summary(args.exp38_summary)
    source_paths = {
        f"{condition}/seed{seed}/{morphology}": args.shard_dir
        / registered_shard_filename(condition, seed, morphology)
        for condition, seed, morphology in REGISTERED_SCHEDULE
    }
    reports = [json.loads(path.read_text()) for path in source_paths.values()]
    source_hashes = {key: file_sha256(path) for key, path in source_paths.items()}
    summary = aggregate_registered_shards(
        reports,
        manifest,
        exp36_summary=json.loads(args.exp36_summary.read_text()),
        exp38_summary=exp38_summary,
        source_hashes=source_hashes,
    )
    _write_final(args.output, summary)
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
