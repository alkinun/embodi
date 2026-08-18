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
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from ..joint_train import runtime_provenance
from ..processing import EmbodiProcessor, camera_frame_to_tensor
from ..record_so101 import atomic_write_json
from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv, runtime_scenario
from .benchmark_expert import BenchmarkPhase, PrivilegedBenchmarkExpert
from .benchmark_geometry_evaluation import (
    CONFIG_SHA256,
    DATA_MANIFEST_SHA256,
    EXP36_SUMMARY_SHA256,
    load_development_contract,
    load_geometry_policy,
    resolve_checkpoint,
)
from . import benchmark_geometry_projection as exp39
from . import benchmark_replan_alignment as exp40


DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
DEVELOPMENT_MANIFEST_SHA256 = (
    "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
)
EXP38_SUMMARY_SHA256 = (
    "e0e7eaf7ec25600b611cd66780a6dc3d0ca8c374a4f871b15a801716703307bb"
)
EXP39_SUMMARY_SHA256 = (
    "87bd3d30f350e3cda16a8ac91dfeeec5249b46ace97e42de860362d7bef5a8a8"
)
EXP40_SUMMARY_SHA256 = (
    "05018ce882e01a5e30d00672ce84db007bdf60992c629ee86485ee7dab55efea"
)
EXP40_EVALUATOR_SHA256 = (
    "d9c73baff704b5c0e56fd46127f3be07488a1401fa7add285a096478d893f3dd"
)
PREREGISTRATION_GIT_COMMIT = "281dfb7720ebb569b8dd486dbd8caf31c6927c23"

CONDITIONS = ("joint", "matched")
MORPHOLOGIES = ("so101", "panda")
TASKS = ("push_to_zone", "lift_object", "pick_place_bin")
TASK_KEYS = {"push_to_zone": "push", "lift_object": "lift", "pick_place_bin": "pick"}
KEY_TASKS = {value: key for key, value in TASK_KEYS.items()}
SEEDS = (36001, 36002, 36003)
SCENARIO_INDICES = tuple(range(21, 28))
PHASES = {
    "push_to_zone": ("precontact", "push"),
    "lift_object": ("precontact", "contact", "close", "lift"),
    "pick_place_bin": (
        "precontact",
        "contact",
        "close",
        "lift",
        "transit",
        "release",
        "retreat",
    ),
}
INSTRUCTIONS = {
    "push": "Push the object into the target zone.",
    "lift": "Lift the object and hold it above the table.",
    "pick": "Pick up the object and place it in the bin.",
}
INSTRUCTION_SHA256 = {
    "push": "ef3724e245f1e3424e91fd211030b7593747d03e0be0c191cddd686533a50a6b",
    "lift": "c89dbd7f6818e34f3d37c485393d1a67464be96efca3e54d06daac4331828e9a",
    "pick": "f345e35bfcc1697c3beb961531cbf9b7dcf0c14d3b96cf84c0b64aa7b55d5f0c",
}
INSTRUCTION_PERMUTATIONS = (
    ("push", "lift", "pick"),
    ("push", "pick", "lift"),
    ("lift", "push", "pick"),
    ("lift", "pick", "push"),
    ("pick", "push", "lift"),
    ("pick", "lift", "push"),
)
ACTION_SCALES = np.array((0.25, 0.25, 0.25, 1, 1, 1, 1, 1, 1, 0.5))
REPEAT_MAX_ABS = 1e-5
TIE_ABS_TOLERANCE = 1e-12
CLIP_TOLERANCE = 2e-5
IK_ITERATIONS = 20
MAXIMUM_STEPS = 500
EXPECTED_ANCHORS = 91
EXPECTED_PREDICTIONS = 728
REGISTERED_RUNTIME_DIR = Path("reports/exp41-runtime")
REGISTERED_LOCK_PATH = REGISTERED_RUNTIME_DIR / ".registered.lock"
LOCK_IDENTITY = "exp41-seed-major-morphology-major-v1"
SCHEDULE_IDENTITY = LOCK_IDENTITY
REGISTERED_SCHEDULE = tuple(
    (seed, morphology) for seed in SEEDS for morphology in MORPHOLOGIES
)
EMPTY_OUTCOME_CHAIN_SHA256 = hashlib.sha256(b"").hexdigest()
RUNTIME_FIELDS = exp39.RUNTIME_FIELDS

PROTOCOL = {
    "camera": "top",
    "resolution": [640, 480],
    "state_path": "forward_kinematics_canonical",
    "expert": "PrivilegedBenchmarkExpert",
    "frequency_hz": 30,
    "maximum_episode_steps": MAXIMUM_STEPS,
    "capture": "loop entry at scored phase_tick==1",
    "expert_action_calls_per_step": 1,
    "execute_original_native_expert_action": True,
    "round_trip_before_execution": True,
    "ik_iterations": IK_ITERATIONS,
    "predictor": "predict_canonical_action_chunk",
    "required_output_shape": [1, 32, 1, 10],
    "policy_reset_before_every_prediction": True,
    "instruction_permutations": [list(value) for value in INSTRUCTION_PERMUTATIONS],
    "repeat_correct_after_block": True,
    "repeat_max_abs_tolerance": REPEAT_MAX_ABS,
    "lead": 0,
    "action_scales": ACTION_SCALES.tolist(),
    "hierarchy": ["phase", "scenario", "task", "morphology", "seed"],
    "final_split_loaded": False,
}


@dataclass(frozen=True)
class Anchor:
    ordinal: int
    scenario_id: str
    scenario_seed: int
    morphology: str
    task: str
    phase: str
    frame: NDArray[Any]
    state: NDArray[Any]
    expert_target: NDArray[Any]
    evidence: dict[str, Any]


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


def canonical_array_sha256(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _outcome_chain(outcomes: Sequence[dict[str, Any]]) -> str:
    digest = bytes.fromhex(EMPTY_OUTCOME_CHAIN_SHA256)
    for outcome in outcomes:
        encoded = json.dumps(
            _jsonable(outcome), sort_keys=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(digest + encoded).digest()
    return digest.hex()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Experiment 041 timestamps must be UTC ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("Experiment 041 timestamp is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("Experiment 041 timestamp must use UTC")
    return parsed


def _write_final(path: Path, report: dict[str, Any]) -> None:
    value = _jsonable(report)
    text = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"refusing to replace different report: {path}")
        return
    atomic_write_json(path, value)


@contextmanager
def registered_run_lock(path: Path = REGISTERED_LOCK_PATH) -> Iterator[None]:
    if path != REGISTERED_LOCK_PATH:
        raise ValueError(
            f"registered Experiment 041 lock path must be {REGISTERED_LOCK_PATH}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another registered Experiment 041 shard holds {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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


def task_instruction_scenarios(
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
            raise ValueError(
                "registered Experiment 041 requires exact indices 0021 through 0027"
            )
        for task in TASKS:
            axes = [
                name
                for scenario in selected
                if scenario.task == task
                for name, value in scenario.factors.items()
                if str(value.get("bin", "")).startswith("development:")
            ]
            if len(axes) != 7 or len(set(axes)) != 7:
                raise ValueError(
                    "each Experiment 041 cell must cover all seven factor axes"
                )
    return selected if limit_scenarios is None else selected[:limit_scenarios]


def instruction_order(anchor_ordinal: int) -> tuple[str, str, str]:
    if type(anchor_ordinal) is not int or anchor_ordinal < 0:
        raise ValueError("anchor ordinal must be a nonnegative integer")
    return INSTRUCTION_PERMUTATIONS[anchor_ordinal % len(INSTRUCTION_PERMUTATIONS)]


def condition_order(seed: int, morphology: str) -> tuple[str, str]:
    try:
        ordinal = REGISTERED_SCHEDULE.index((seed, morphology))
    except ValueError as error:
        raise ValueError("condition order requires a registered shard") from error
    return ("joint", "matched") if ordinal % 2 == 0 else ("matched", "joint")


def _rotation_matrix(rotation_6d: Any) -> NDArray[Any] | None:
    value = np.asarray(rotation_6d, dtype=np.float64).reshape(6)
    first = value[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-12:
        return None
    first = first / first_norm
    second = value[3:] - first * float(np.dot(first, value[3:]))
    second_norm = float(np.linalg.norm(second))
    if second_norm <= 1e-12:
        return None
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=1)


def rotation_geodesic_error(first: Any, second: Any) -> tuple[float, bool]:
    first_rotation = _rotation_matrix(first)
    second_rotation = _rotation_matrix(second)
    if first_rotation is None or second_rotation is None:
        return 180.0, True
    relative = first_rotation.T @ second_rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine))), False


def action_metrics(prediction: Any, expert: Any) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(10)
    target = np.asarray(expert, dtype=np.float64).reshape(10)
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("action metrics require finite canonical actions")
    rotation, degenerate = rotation_geodesic_error(predicted[3:9], target[3:9])
    return {
        "loss": float(np.mean(((predicted - target) / ACTION_SCALES) ** 2)),
        "translation_norm_error_m": float(np.linalg.norm(predicted[:3] - target[:3])),
        "rotation_geodesic_error_deg": rotation,
        "effective_gripper_error": float(
            abs(np.clip(predicted[9], 0.0, 1.0) - np.clip(target[9], 0.0, 1.0))
        ),
        "degenerate_rotation": degenerate,
    }


def score_instruction_predictions(
    task: str, expert: Any, predictions: Mapping[str, Any]
) -> dict[str, Any]:
    if task not in TASK_KEYS or set(predictions) != set(INSTRUCTIONS):
        raise ValueError(
            "scoring requires one prediction for each registered instruction"
        )
    correct_key = TASK_KEYS[task]
    wrong_keys = [key for key in INSTRUCTIONS if key != correct_key]
    arms = {key: action_metrics(predictions[key], expert) for key in INSTRUCTIONS}
    correct = arms[correct_key]
    wrong = {
        name: float(np.mean([float(arms[key][name]) for key in wrong_keys]))
        for name in (
            "loss",
            "translation_norm_error_m",
            "rotation_geodesic_error_deg",
            "effective_gripper_error",
        )
    }
    wrong["degenerate_rotation_count"] = sum(
        bool(arms[key]["degenerate_rotation"]) for key in wrong_keys
    )
    wrong["degenerate_rotation_rate"] = wrong["degenerate_rotation_count"] / len(
        wrong_keys
    )
    margin = float(wrong["loss"] - correct["loss"])
    benefit = None if wrong["loss"] == 0 else margin / wrong["loss"]
    minimum = min(float(value["loss"]) for value in arms.values())
    ties = [
        key
        for key, value in arms.items()
        if abs(float(value["loss"]) - minimum) <= TIE_ABS_TOLERANCE
    ]
    separation = float(
        np.mean(
            [
                np.mean(
                    (
                        (
                            np.asarray(predictions[correct_key], dtype=np.float64)
                            - np.asarray(predictions[key], dtype=np.float64)
                        )
                        / ACTION_SCALES
                    )
                    ** 2
                )
                for key in wrong_keys
            ]
        )
    )
    return {
        "correct": {
            **correct,
            "degenerate_rotation_count": int(bool(correct["degenerate_rotation"])),
        },
        "wrong": wrong,
        "margin": margin,
        "relative_benefit": benefit,
        "correct_instruction_top_one_rate": 1.0 / len(ties)
        if correct_key in ties
        else 0.0,
        "output_separation": separation,
    }


def _native_shape(env: Any) -> tuple[int, ...]:
    return (int(env.morphology.native_action_dim),)


def _clipping_from_arrays(decoded: Any, realizable: Any) -> tuple[bool, float]:
    decoded_array = np.asarray(decoded, dtype=np.float64)
    realizable_array = np.asarray(realizable, dtype=np.float64)
    if decoded_array.shape != realizable_array.shape:
        raise ValueError("decoded and realizable native actions must share shape")
    error = np.abs(decoded_array - realizable_array)
    return bool(np.any(error > CLIP_TOLERANCE)), float(error.max(initial=0.0))


def capture_expert_episode(
    env: Any,
    scenario: Scenario,
    *,
    starting_ordinal: int,
    expert_factory: Callable[[Any], Any] = PrivilegedBenchmarkExpert,
    maximum_steps: int = MAXIMUM_STEPS,
) -> tuple[list[Anchor], dict[str, Any]]:
    expected_phases = PHASES[scenario.task]
    expert = expert_factory(env)
    initial_phase = str(
        expert.phase.value if isinstance(expert.phase, BenchmarkPhase) else expert.phase
    )
    if initial_phase != "open" or expert.phase_tick != 0:
        raise RuntimeError("expert episode must begin at open phase_tick zero")
    anchors: list[Anchor] = []
    status = None
    steps = 0
    for _ in range(maximum_steps):
        phase = str(
            expert.phase.value
            if isinstance(expert.phase, BenchmarkPhase)
            else expert.phase
        )
        capture = phase in expected_phases and expert.phase_tick == 1
        frame = state = None
        if capture:
            frame = np.asarray(env.render_top()).copy()
            state = np.asarray(env.canonical_state(), dtype=np.float32).copy()
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise ValueError("expert anchor image must be top 640x480 uint8 RGB")
            if state.shape != (1, 10) or not np.isfinite(state).all():
                raise ValueError("expert anchor state must be finite canonical [1,10]")
        requested = np.asarray(expert.action(), dtype=np.float32)
        if requested.shape != _native_shape(env) or not np.isfinite(requested).all():
            raise ValueError("expert returned an invalid native action")
        if capture:
            assert frame is not None and state is not None
            target = np.asarray(env.canonical_arm_action(requested), dtype=np.float32)
            decoded = np.asarray(
                env.native_action_from_canonical(target, iterations=IK_ITERATIONS),
                dtype=np.float32,
            )
            realized = np.asarray(env.canonical_arm_action(decoded), dtype=np.float32)
            unchanged = np.asarray(env.canonical_state(), dtype=np.float32)
            if (
                target.shape != (1, 10)
                or decoded.shape != _native_shape(env)
                or realized.shape != (1, 10)
                or not np.isfinite(target).all()
                or not np.isfinite(decoded).all()
                or not np.isfinite(realized).all()
                or canonical_array_sha256(unchanged) != canonical_array_sha256(state)
            ):
                raise ValueError(
                    "expert canonical round trip changed state or returned invalid data"
                )
            rotation_error, degenerate = rotation_geodesic_error(
                target[0, 3:9], realized[0, 3:9]
            )
            if degenerate:
                raise ValueError("expert round trip produced a degenerate rotation")
            realizable = np.asarray(
                env.qpos_to_native(env.native_to_qpos(decoded)), dtype=np.float64
            )
            if (
                realizable.shape != _native_shape(env)
                or not np.isfinite(realizable).all()
            ):
                raise ValueError(
                    "expert clipping round trip returned invalid native data"
                )
            clipped, clipping_error = _clipping_from_arrays(decoded, realizable)
            evidence = {
                "ordinal": starting_ordinal + len(anchors),
                "scenario_id": scenario.scenario_id,
                "scenario_seed": scenario.seed,
                "morphology": scenario.morphology,
                "task": scenario.task,
                "phase": phase,
                "phase_tick": 1,
                "image_sha256": canonical_array_sha256(frame),
                "state_sha256": canonical_array_sha256(state),
                "expert_native": requested.tolist(),
                "expert_native_sha256": canonical_array_sha256(requested),
                "expert_target": target.tolist(),
                "expert_target_sha256": canonical_array_sha256(target),
                "decoded_native": decoded.tolist(),
                "decoded_native_sha256": canonical_array_sha256(decoded),
                "realizable_native": realizable.tolist(),
                "realizable_native_sha256": canonical_array_sha256(realizable),
                "realized_canonical": realized.tolist(),
                "realized_canonical_sha256": canonical_array_sha256(realized),
                "round_trip": {
                    "clipped": clipped,
                    "clipping_error_native": clipping_error,
                    "translation_error_m": float(
                        np.linalg.norm(target[0, :3] - realized[0, :3])
                    ),
                    "rotation_error_deg": rotation_error,
                    "gripper_error_native": float(abs(requested[-1] - decoded[-1])),
                },
            }
            anchors.append(
                Anchor(
                    starting_ordinal + len(anchors),
                    scenario.scenario_id,
                    scenario.seed,
                    scenario.morphology,
                    scenario.task,
                    phase,
                    frame,
                    state,
                    target,
                    evidence,
                )
            )
        env.apply_action(requested)
        status = env.step_control_period(float(expert.frequency))
        steps += 1
        if expert.terminal:
            break
    phases = [anchor.phase for anchor in anchors]
    final_phase = str(
        expert.phase.value if isinstance(expert.phase, BenchmarkPhase) else expert.phase
    )
    if (
        phases != list(expected_phases)
        or final_phase != "done"
        or status is None
        or status.terminal is not True
        or status.success is not True
        or env.success is not True
    ):
        raise RuntimeError(
            "expert episode did not complete the exact scored phase sequence in done"
        )
    return anchors, {
        "scenario_id": scenario.scenario_id,
        "task": scenario.task,
        "steps": steps,
        "success": True,
        "terminal": True,
        "final_phase": final_phase,
        "scored_phases": phases,
        "anchor_count": len(anchors),
    }


def capture_expert_anchors(
    scenarios: Sequence[Scenario],
    morphology: str,
    *,
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
    expert_factory: Callable[[Any], Any] = PrivilegedBenchmarkExpert,
) -> tuple[list[Anchor], list[dict[str, Any]]]:
    anchors: list[Anchor] = []
    episodes = []
    for scenario in scenarios:
        env = env_factory(
            morphology,
            scenario.task,
            runtime_scenario(scenario.factors),
            width=640,
            height=480,
            render=True,
        )
        try:
            captured, episode = capture_expert_episode(
                env,
                scenario,
                starting_ordinal=len(anchors),
                expert_factory=expert_factory,
            )
            anchors.extend(captured)
            episodes.append(episode)
        finally:
            env.close()
    expected = sum(len(PHASES[scenario.task]) for scenario in scenarios)
    if len(anchors) != expected or [anchor.ordinal for anchor in anchors] != list(
        range(expected)
    ):
        raise RuntimeError(
            "expert anchors do not match the registered scenario/phase ordering"
        )
    return anchors, episodes


def _policy_batch(
    anchor: Anchor,
    instruction_key: str,
    policy: Any,
    processor: Any,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = camera_frame_to_tensor(
        anchor.frame, processor_rescales=policy.config.image_do_rescale
    )
    canonical_state = torch.from_numpy(anchor.state).unsqueeze(0)
    batch = processor(
        [[image]],
        [INSTRUCTIONS[instruction_key]],
        None,
        canonical_state=canonical_state,
        canonical_part_mask=torch.ones((1, canonical_state.shape[1]), dtype=torch.bool),
    )
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
        raise ValueError(
            "processor batch must contain tensor input_ids and attention_mask"
        )
    input_ids_array = input_ids.detach().cpu().contiguous().numpy()
    attention_mask_array = attention_mask.detach().cpu().contiguous().numpy()
    hashes: dict[str, Any] = {
        "instruction_sha256": hashlib.sha256(
            INSTRUCTIONS[instruction_key].encode()
        ).hexdigest(),
        "input_ids_sha256": canonical_array_sha256(input_ids),
        "input_ids_dtype": input_ids_array.dtype.str,
        "input_ids_shape": list(input_ids_array.shape),
        "input_ids": input_ids_array.tolist(),
        "attention_mask_sha256": canonical_array_sha256(attention_mask),
        "attention_mask_dtype": attention_mask_array.dtype.str,
        "attention_mask_shape": list(attention_mask_array.shape),
        "attention_mask": attention_mask_array.tolist(),
        "image_sha256": anchor.evidence["image_sha256"],
        "state_sha256": anchor.evidence["state_sha256"],
    }
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    return moved, hashes


def _predict_once(
    anchor: Anchor,
    instruction_key: str,
    policy: Any,
    processor: Any,
    device: str,
) -> tuple[NDArray[Any], dict[str, Any]]:
    policy.reset()
    batch, hashes = _policy_batch(anchor, instruction_key, policy, processor, device)
    value = policy.predict_canonical_action_chunk(batch)
    output = (
        value.detach().float().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )
    output = np.asarray(output, dtype=np.float32)
    if output.shape != (1, 32, 1, 10) or not np.isfinite(output).all():
        raise ValueError("policy output must be finite with shape [1,32,1,10]")
    hashes["output_sha256"] = canonical_array_sha256(output)
    hashes["output_dtype"] = output.dtype.str
    hashes["output_shape"] = list(output.shape)
    hashes["output_chunk"] = output[0].tolist()
    return output, hashes


def evaluate_anchor_condition(
    anchor: Anchor,
    condition: str,
    policy: Any,
    processor: Any,
    device: str,
) -> dict[str, Any]:
    order = instruction_order(anchor.ordinal)
    outputs: dict[str, NDArray[Any]] = {}
    queries: list[dict[str, Any]] = []
    for key in order:
        output, hashes = _predict_once(anchor, key, policy, processor, device)
        outputs[key] = output
        queries.append({"instruction_key": key, **hashes})
    correct_key = TASK_KEYS[anchor.task]
    repeated, repeat_hashes = _predict_once(
        anchor, correct_key, policy, processor, device
    )
    difference = float(np.max(np.abs(outputs[correct_key] - repeated)))
    if not np.isfinite(difference) or difference > REPEAT_MAX_ABS:
        raise RuntimeError(
            "repeated correct-instruction output exceeds determinism tolerance"
        )
    queries.append(
        {"instruction_key": correct_key, "repeat_correct": True, **repeat_hashes}
    )
    if {query["instruction_sha256"] for query in queries[:3]} != set(
        INSTRUCTION_SHA256.values()
    ):
        raise RuntimeError(
            "policy queries do not contain the exact registered instructions"
        )
    if len({query["input_ids_sha256"] for query in queries[:3]}) != 3:
        raise RuntimeError(
            "registered instructions did not produce distinct tokenizer input ids"
        )
    first_correct = next(
        query for query in queries[:3] if query["instruction_key"] == correct_key
    )
    repeat_input_fields = (
        "instruction_key",
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
    )
    if any(queries[3][name] != first_correct[name] for name in repeat_input_fields):
        raise RuntimeError(
            "repeat-correct policy input differs from first correct query"
        )
    predictions = {key: outputs[key][0, 0, 0].copy() for key in INSTRUCTIONS}
    return {
        "condition": condition,
        "instruction_order": list(order),
        "queries": queries,
        "repeat_correct_max_abs_difference": difference,
        "metrics": score_instruction_predictions(
            anchor.task, anchor.expert_target, predictions
        ),
    }


def _source_condition(condition: str, morphology: str) -> str:
    if condition == "joint":
        return "joint"
    if condition == "matched" and morphology in MORPHOLOGIES:
        return morphology
    raise ValueError("Experiment 041 condition must be joint or matched")


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric cell")
    return float(sum(float(value) for value in values) / len(values))


METRIC_PATHS = {
    "correct_loss": ("metrics", "correct", "loss"),
    "wrong_loss": ("metrics", "wrong", "loss"),
    "top_one_rate": ("metrics", "correct_instruction_top_one_rate"),
    "correct_translation_norm_error_m": (
        "metrics",
        "correct",
        "translation_norm_error_m",
    ),
    "wrong_translation_norm_error_m": (
        "metrics",
        "wrong",
        "translation_norm_error_m",
    ),
    "correct_rotation_geodesic_error_deg": (
        "metrics",
        "correct",
        "rotation_geodesic_error_deg",
    ),
    "wrong_rotation_geodesic_error_deg": (
        "metrics",
        "wrong",
        "rotation_geodesic_error_deg",
    ),
    "correct_effective_gripper_error": (
        "metrics",
        "correct",
        "effective_gripper_error",
    ),
    "wrong_effective_gripper_error": (
        "metrics",
        "wrong",
        "effective_gripper_error",
    ),
    "output_separation": ("metrics", "output_separation"),
    "correct_degenerate_rotation_rate": (
        "metrics",
        "correct",
        "degenerate_rotation_count",
    ),
    "wrong_degenerate_rotation_rate": (
        "metrics",
        "wrong",
        "degenerate_rotation_rate",
    ),
}


def _path_value(value: Mapping[str, Any], path: Sequence[str]) -> float:
    current: Any = value
    for name in path:
        current = current[name]
    return float(current)


def hierarchical_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("hierarchical aggregation requires records")
    scenario_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for record in records:
        scenario_groups[
            (record["morphology"], record["task"], record["scenario_id"])
        ].append(record)
    scenario_means: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, values in scenario_groups.items():
        scenario_means[key] = {
            name: _mean([_path_value(value, path) for value in values])
            for name, path in METRIC_PATHS.items()
        }
    task_groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (morphology, task, _scenario), scenario_values in scenario_means.items():
        task_groups[(morphology, task)].append(scenario_values)
    task_means = {
        key: {name: _mean([value[name] for value in values]) for name in METRIC_PATHS}
        for key, values in task_groups.items()
    }
    morphology_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (morphology, _task), task_values in task_means.items():
        morphology_groups[morphology].append(task_values)
    morphology_means = {
        key: {name: _mean([value[name] for value in values]) for name in METRIC_PATHS}
        for key, values in morphology_groups.items()
    }
    result: dict[str, Any] = {
        name: _mean([value[name] for value in morphology_means.values()])
        for name in METRIC_PATHS
    }
    result["margin"] = result["wrong_loss"] - result["correct_loss"]
    result["relative_benefit"] = (
        None if result["wrong_loss"] == 0 else result["margin"] / result["wrong_loss"]
    )
    result["correct_degenerate_rotation_count"] = sum(
        int(record["metrics"]["correct"]["degenerate_rotation_count"])
        for record in records
    )
    result["wrong_degenerate_rotation_count"] = sum(
        int(record["metrics"]["wrong"]["degenerate_rotation_count"])
        for record in records
    )
    result["anchors"] = len(records)
    return result


def _subgroup_metrics(
    records: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    return {
        "/".join(str(value) for value in key): hierarchical_metrics(values)
        for key, values in sorted(grouped.items())
    }


def _delivery_checks(measurements: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "decoded_action_clipping_at_most_one_percent": measurements["clipping_rate"]
        is not None
        and measurements["clipping_rate"] <= 0.01,
        "translation_p95_at_most_1mm": measurements["translation_p95_m"] is not None
        and measurements["translation_p95_m"] <= 0.001,
        "rotation_p95_at_most_1deg": measurements["rotation_p95_deg"] is not None
        and measurements["rotation_p95_deg"] <= 1.0,
        "maximum_gripper_error_at_most_one_native_unit": measurements[
            "maximum_gripper_error_native"
        ]
        is not None
        and measurements["maximum_gripper_error_native"] <= 1.0,
    }


def delivery_gate(anchors: Sequence[dict[str, Any]]) -> dict[str, Any]:
    round_trips = []
    for anchor in anchors:
        decoded = np.asarray(anchor["decoded_native"], dtype=np.float32)
        realizable = np.asarray(anchor["realizable_native"], dtype=np.float64)
        target = np.asarray(anchor["expert_target"], dtype=np.float32).reshape(1, 10)
        realized = np.asarray(anchor["realized_canonical"], dtype=np.float32).reshape(
            1, 10
        )
        requested = np.asarray(anchor["expert_native"], dtype=np.float32)
        clipped, clipping_error = _clipping_from_arrays(decoded, realizable)
        rotation_error, _degenerate = rotation_geodesic_error(
            target[0, 3:9], realized[0, 3:9]
        )
        round_trips.append(
            {
                "clipped": clipped,
                "clipping_error_native": clipping_error,
                "translation_error_m": float(
                    np.linalg.norm(target[0, :3] - realized[0, :3])
                ),
                "rotation_error_deg": rotation_error,
                "gripper_error_native": float(abs(requested[-1] - decoded[-1])),
            }
        )
    commands = len(round_trips)
    clipped_commands = sum(bool(value["clipped"]) for value in round_trips)
    measurements = {
        "commands": commands,
        "clipped_commands": clipped_commands,
        "clipping_rate": clipped_commands / commands if commands else None,
        "translation_p95_m": float(
            np.percentile([value["translation_error_m"] for value in round_trips], 95)
        )
        if commands
        else None,
        "rotation_p95_deg": float(
            np.percentile([value["rotation_error_deg"] for value in round_trips], 95)
        )
        if commands
        else None,
        "maximum_gripper_error_native": max(
            (float(value["gripper_error_native"]) for value in round_trips),
            default=None,
        ),
    }
    checks = _delivery_checks(measurements)
    return {
        "measurements": measurements,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _anchor_record(
    anchor: Anchor, results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {**anchor.evidence, "conditions": results}


def _flatten_condition_records(
    reports: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    flattened = []
    for report in reports:
        for anchor in report["anchors"]:
            for condition in CONDITIONS:
                flattened.append(
                    {
                        "condition": condition,
                        "seed": report["identity"]["model_seed"],
                        "morphology": anchor["morphology"],
                        "task": anchor["task"],
                        "phase": anchor["phase"],
                        "scenario_id": anchor["scenario_id"],
                        **anchor["conditions"][condition],
                    }
                )
    return flattened


def _validate_anchor_record(
    anchor: dict[str, Any], report: dict[str, Any], ordinal: int
) -> None:
    identity = report["identity"]
    if (
        anchor.get("ordinal") != ordinal
        or anchor.get("morphology") != identity["morphology"]
        or anchor.get("task") not in TASKS
        or anchor.get("phase") not in PHASES[anchor["task"]]
        or anchor.get("phase_tick") != 1
        or set(anchor.get("conditions", {})) != set(CONDITIONS)
        or tuple(anchor["conditions"]) != tuple(report["run"]["condition_order"])
    ):
        raise ValueError("Experiment 041 anchor identity is invalid")
    arrays = {
        "expert_native": "expert_native_sha256",
        "expert_target": "expert_target_sha256",
        "decoded_native": "decoded_native_sha256",
        "realized_canonical": "realized_canonical_sha256",
    }
    for name, hash_name in arrays.items():
        array = np.asarray(anchor.get(name), dtype=np.float32)
        expected_shape = (
            (6 if identity["morphology"] == "so101" else 8,)
            if "native" in name
            else (1, 10)
        )
        if (
            array.shape != expected_shape
            or not np.isfinite(array).all()
            or anchor.get(hash_name) != canonical_array_sha256(array)
        ):
            raise ValueError("Experiment 041 persisted expert array hash is invalid")
    realizable = np.asarray(anchor.get("realizable_native"), dtype=np.float64)
    native_shape = (6 if identity["morphology"] == "so101" else 8,)
    if (
        realizable.shape != native_shape
        or not np.isfinite(realizable).all()
        or anchor.get("realizable_native_sha256") != canonical_array_sha256(realizable)
    ):
        raise ValueError("Experiment 041 realizable native action hash is invalid")
    target = np.asarray(anchor["expert_target"], dtype=np.float32)
    realized = np.asarray(anchor["realized_canonical"], dtype=np.float32)
    requested = np.asarray(anchor["expert_native"], dtype=np.float32)
    decoded = np.asarray(anchor["decoded_native"], dtype=np.float32)
    clipped, clipping_error = _clipping_from_arrays(decoded, realizable)
    rotation, degenerate = rotation_geodesic_error(target[0, 3:9], realized[0, 3:9])
    round_trip = anchor.get("round_trip", {})
    if (
        degenerate
        or round_trip.get("translation_error_m")
        != float(np.linalg.norm(target[0, :3] - realized[0, :3]))
        or round_trip.get("rotation_error_deg") != rotation
        or round_trip.get("gripper_error_native")
        != float(abs(requested[-1] - decoded[-1]))
        or round_trip.get("clipped") is not clipped
        or round_trip.get("clipping_error_native") != clipping_error
    ):
        raise ValueError("Experiment 041 expert round-trip evidence is invalid")
    if not _valid_sha256(anchor.get("image_sha256")) or not _valid_sha256(
        anchor.get("state_sha256")
    ):
        raise ValueError("Experiment 041 anchor input hashes are invalid")
    for condition in CONDITIONS:
        result = anchor["conditions"][condition]
        order = instruction_order(ordinal)
        queries = result.get("queries")
        if (
            result.get("condition") != condition
            or result.get("instruction_order") != list(order)
            or not isinstance(queries, list)
            or len(queries) != 4
            or [query.get("instruction_key") for query in queries[:3]] != list(order)
            or queries[3].get("instruction_key") != TASK_KEYS[anchor["task"]]
            or queries[3].get("repeat_correct") is not True
            or any("repeat_correct" in query for query in queries[:3])
        ):
            raise ValueError("Experiment 041 instruction query ordering is invalid")
        chunks: list[NDArray[Any]] = []
        for index, query in enumerate(queries):
            key = query["instruction_key"]
            required_hashes = (
                "instruction_sha256",
                "input_ids_sha256",
                "attention_mask_sha256",
                "image_sha256",
                "state_sha256",
                "output_sha256",
            )
            try:
                raw_input_ids = np.asarray(query.get("input_ids"))
                raw_attention_mask = np.asarray(query.get("attention_mask"))
                input_ids = np.asarray(query.get("input_ids"), dtype=np.int64)
                attention_mask = np.asarray(query.get("attention_mask"), dtype=np.int64)
                chunk = np.asarray(query.get("output_chunk"), dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Experiment 041 policy input/output arrays are invalid"
                ) from error
            if (
                any(not _valid_sha256(query.get(name)) for name in required_hashes)
                or query["instruction_sha256"] != INSTRUCTION_SHA256[key]
                or raw_input_ids.dtype.kind not in "iu"
                or raw_attention_mask.dtype.kind not in "iub"
                or input_ids.ndim != 2
                or input_ids.shape[0] != 1
                or attention_mask.shape != input_ids.shape
                or query.get("input_ids_dtype") != np.dtype(np.int64).str
                or query.get("input_ids_shape") != list(input_ids.shape)
                or query.get("input_ids_sha256") != canonical_array_sha256(input_ids)
                or query.get("attention_mask_dtype") != np.dtype(np.int64).str
                or query.get("attention_mask_shape") != list(attention_mask.shape)
                or query.get("attention_mask_sha256")
                != canonical_array_sha256(attention_mask)
                or query.get("output_dtype") != np.dtype(np.float32).str
                or query.get("output_shape") != [1, 32, 1, 10]
                or chunk.shape != (32, 1, 10)
                or not np.isfinite(chunk).all()
                or query.get("output_sha256") != canonical_array_sha256(chunk[None])
                or query["image_sha256"] != anchor["image_sha256"]
                or query["state_sha256"] != anchor["state_sha256"]
                or (index == 3 and "repeat_correct" not in query)
            ):
                raise ValueError(
                    "Experiment 041 policy input/output hashes are invalid"
                )
            chunks.append(chunk)
        if len({query["input_ids_sha256"] for query in queries[:3]}) != 3:
            raise ValueError("Experiment 041 tokenizer input ids are not distinct")
        correct_key = TASK_KEYS[anchor["task"]]
        correct_index = list(order).index(correct_key)
        first_correct = queries[correct_index]
        repeat_input_fields = (
            "instruction_key",
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
        )
        if any(queries[3][name] != first_correct[name] for name in repeat_input_fields):
            raise ValueError("Experiment 041 repeat-correct input evidence differs")
        repeat_difference = result.get("repeat_correct_max_abs_difference")
        predictions = {
            key: chunks[index][0, 0].copy() for index, key in enumerate(order)
        }
        recomputed_difference = float(np.max(np.abs(chunks[correct_index] - chunks[3])))
        if (
            not isinstance(repeat_difference, (int, float))
            or not np.isfinite(repeat_difference)
            or repeat_difference != recomputed_difference
            or recomputed_difference > REPEAT_MAX_ABS
            or result.get("metrics")
            != score_instruction_predictions(anchor["task"], target, predictions)
        ):
            raise ValueError("Experiment 041 policy metrics do not recompute")


def _source_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    expected = {
        "exp36_summary": EXP36_SUMMARY_SHA256,
        "exp38_summary": EXP38_SUMMARY_SHA256,
        "exp39_summary": EXP39_SUMMARY_SHA256,
        "exp40_summary": EXP40_SUMMARY_SHA256,
    }
    for name, digest in expected.items():
        if name not in paths or file_sha256(paths[name]) != digest:
            raise ValueError(
                f"{name.replace('_', ' ')} hash does not match Experiment 041 registration"
            )
    if file_sha256(Path(exp40.__file__)) != EXP40_EVALUATOR_SHA256:
        raise ValueError(
            "imported Experiment 040 evaluator hash does not match registration"
        )
    summaries = {name: json.loads(paths[name].read_text()) for name in expected}
    if (
        summaries["exp36_summary"].get("experiment") != 36
        or summaries["exp36_summary"].get("status") != "completed_gate_passed"
        or summaries["exp36_summary"].get("fixed_step") != 1200
        or summaries["exp38_summary"].get("experiment") != 38
        or summaries["exp39_summary"].get("experiment") != 39
        or summaries["exp40_summary"].get("experiment") != 40
        or any(
            summaries[name].get("registered") is not True
            for name in ("exp38_summary", "exp39_summary", "exp40_summary")
        )
    ):
        raise ValueError("Experiment 041 frozen source summary contract is invalid")
    return summaries


def _contract(registered: bool, limit_scenarios: int | None) -> dict[str, Any]:
    return {
        "definition_sha256": DEFINITION_SHA256,
        "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
        "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
        "exp38_summary_sha256": EXP38_SUMMARY_SHA256,
        "exp39_summary_sha256": EXP39_SUMMARY_SHA256,
        "exp40_summary_sha256": EXP40_SUMMARY_SHA256,
        "exp40_evaluator_sha256": EXP40_EVALUATOR_SHA256,
        "preregistration_git_commit": PREREGISTRATION_GIT_COMMIT,
        "split": "development",
        "scenario_indices": list(SCENARIO_INDICES),
        "tasks": list(TASKS),
        "conditions": list(CONDITIONS),
        "limit_scenarios": None if registered else limit_scenarios,
        "protocol": PROTOCOL,
        "final_split_loaded": False,
    }


def _run_metadata(
    seed: int, morphology: str, device: str, registered: bool
) -> dict[str, Any]:
    return {
        "started_utc": _utc_now(),
        "finished_utc": None,
        "device": device,
        "lock_identity": LOCK_IDENTITY if registered else None,
        "lock_path": str(REGISTERED_LOCK_PATH) if registered else None,
        "schedule_identity": SCHEDULE_IDENTITY if registered else "smoke-unregistered",
        "schedule_position": REGISTERED_SCHEDULE.index((seed, morphology))
        if registered
        else None,
        "condition_order": list(condition_order(seed, morphology)),
    }


def _checkpoint_identities(
    summary_path: Path, seed: int, morphology: str, root: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths = {}
    identities = {}
    for condition in CONDITIONS:
        source = _source_condition(condition, morphology)
        checkpoint, identity = resolve_checkpoint(
            summary_path, source, seed, morphology, root=root
        )
        identity["condition"] = condition
        identity["source_condition"] = source
        paths[condition] = checkpoint
        identities[condition] = identity
    return paths, identities


def _shard_aggregate(
    anchors: Sequence[dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any]:
    gate = delivery_gate(anchors)
    records = _flatten_condition_records([report])
    expected_anchor_ids = report["expected_anchor_ids"]
    integrity_checks = {
        "all_expected_scenarios_and_phase_anchors": len(anchors)
        == len(expected_anchor_ids)
        and [f"{value['scenario_id']}/{value['phase']}" for value in anchors]
        == expected_anchor_ids,
        "all_expert_episodes_successful_in_done": all(
            value["success"] is True
            and value["terminal"] is True
            and value["final_phase"] == "done"
            for value in report["episodes"]
        ),
        "exact_instruction_hashes": all(
            query["instruction_sha256"] == INSTRUCTION_SHA256[query["instruction_key"]]
            for anchor in anchors
            for condition in CONDITIONS
            for query in anchor["conditions"][condition]["queries"]
        ),
        "distinct_tokenizer_input_hashes": all(
            len(
                {
                    query["input_ids_sha256"]
                    for query in anchor["conditions"][condition]["queries"][:3]
                }
            )
            == 3
            for anchor in anchors
            for condition in CONDITIONS
        ),
        "identical_image_and_state_hashes_across_queries_and_models": all(
            query["image_sha256"] == anchor["image_sha256"]
            and query["state_sha256"] == anchor["state_sha256"]
            for anchor in anchors
            for condition in CONDITIONS
            for query in anchor["conditions"][condition]["queries"]
        ),
        "deterministic_policy_outputs": all(
            anchor["conditions"][condition]["repeat_correct_max_abs_difference"]
            <= REPEAT_MAX_ABS
            for anchor in anchors
            for condition in CONDITIONS
        ),
        "complete_checkpoint_hashes": all(
            all(
                _valid_sha256(report["identity"]["checkpoints"][condition].get(name))
                for name in (
                    "core_sha256",
                    "config_sha256",
                    "embodiment_sha256",
                    "trainer_sha256",
                    "data_manifest_sha256",
                )
            )
            for condition in CONDITIONS
        ),
        "complete_source_hashes": all(
            report["contract"].get(name) == digest
            for name, digest in (
                ("definition_sha256", DEFINITION_SHA256),
                ("manifest_sha256", DEVELOPMENT_MANIFEST_SHA256),
                ("exp36_summary_sha256", EXP36_SUMMARY_SHA256),
                ("exp38_summary_sha256", EXP38_SUMMARY_SHA256),
                ("exp39_summary_sha256", EXP39_SUMMARY_SHA256),
                ("exp40_summary_sha256", EXP40_SUMMARY_SHA256),
                ("exp40_evaluator_sha256", EXP40_EVALUATOR_SHA256),
            )
        ),
        "final_split_not_loaded": report["contract"].get("final_split_loaded") is False,
    }
    gate["integrity_checks"] = integrity_checks
    gate["passed"] = gate["passed"] and all(integrity_checks.values())
    return {
        "episodes": len(report["episodes"]),
        "anchors": len(anchors),
        "predictions": len(anchors) * len(CONDITIONS) * 4,
        "condition_metrics": {
            condition: hierarchical_metrics(
                [record for record in records if record["condition"] == condition]
            )
            for condition in CONDITIONS
        },
        "by_task_phase": _subgroup_metrics(records, ("condition", "task", "phase")),
        "delivery_gate": gate,
    }


def validate_shard_report(
    report: dict[str, Any],
    manifest: ScenarioManifest,
    exp36_summary: dict[str, Any],
    *,
    require_registered: bool,
) -> None:
    identity = report.get("identity", {})
    seed = identity.get("model_seed")
    morphology = identity.get("morphology")
    scenarios = task_instruction_scenarios(
        manifest,
        morphology,
        registered=require_registered,
        limit_scenarios=None
        if require_registered
        else report.get("contract", {}).get("limit_scenarios"),
    )
    expected_anchor_ids = [
        f"{scenario.scenario_id}/{phase}"
        for scenario in scenarios
        for phase in PHASES[scenario.task]
    ]
    anchors = report.get("anchors", [])
    run = report.get("run", {})
    provenance = report.get("provenance", {})
    checkpoints = identity.get("checkpoints", {})
    if (
        report.get("format_version") != 1
        or report.get("experiment") != 41
        or report.get("mode") != "task_instruction_shard"
        or report.get("registered") is not require_registered
        or report.get("complete") is not True
        or report.get("status")
        != ("completed" if require_registered else "completed_smoke_unregistered")
        or seed not in SEEDS
        or morphology not in MORPHOLOGIES
        or report.get("contract")
        != _contract(
            require_registered, report.get("contract", {}).get("limit_scenarios")
        )
        or report.get("expected_anchor_ids") != expected_anchor_ids
        or [f"{value.get('scenario_id')}/{value.get('phase')}" for value in anchors]
        != expected_anchor_ids
        or len(report.get("episodes", [])) != len(scenarios)
        or any(
            episode.get("success") is not True
            or episode.get("terminal") is not True
            or episode.get("final_phase") != "done"
            or episode.get("scored_phases") != list(PHASES[episode.get("task")])
            for episode in report.get("episodes", [])
        )
        or set(checkpoints) != set(CONDITIONS)
        or report.get("outcome_chain_sha256") != _outcome_chain(anchors)
        or report.get("infrastructure_errors") != []
        or run.get("condition_order") != list(condition_order(seed, morphology))
        or run.get("device") != provenance.get("runtime", {}).get("device")
        or run.get("finished_utc") is None
    ):
        raise ValueError("Experiment 041 shard contract or completion state is invalid")
    for episode, scenario in zip(report["episodes"], scenarios, strict=True):
        if (
            episode.get("scenario_id") != scenario.scenario_id
            or episode.get("task") != scenario.task
            or type(episode.get("steps")) is not int
            or not 1 <= episode["steps"] <= MAXIMUM_STEPS
            or episode.get("scored_phases") != list(PHASES[scenario.task])
            or episode.get("anchor_count") != len(PHASES[scenario.task])
        ):
            raise ValueError("Experiment 041 expert episode metadata is invalid")
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if any(
        anchor.get("scenario_seed") != scenario_by_id[anchor["scenario_id"]].seed
        for anchor in anchors
    ):
        raise ValueError("Experiment 041 anchor scenario metadata is invalid")
    for condition in CONDITIONS:
        source = _source_condition(condition, morphology)
        expected_run = exp36_summary.get("runs", {}).get(f"{source}-seed{seed}", {})
        checkpoint = checkpoints[condition]
        if (
            checkpoint.get("condition") != condition
            or checkpoint.get("source_condition") != source
            or checkpoint.get("model_seed") != seed
            or checkpoint.get("morphology") != morphology
            or checkpoint.get("core_sha256") != expected_run.get("core_sha256")
            or checkpoint.get("config_sha256") != CONFIG_SHA256[morphology]
            or checkpoint.get("data_manifest_sha256") != DATA_MANIFEST_SHA256[source]
            or checkpoint.get("exp36_summary_sha256") != EXP36_SUMMARY_SHA256
            or checkpoint.get("step") != 1200
            or checkpoint.get("stage") != "core"
            or any(
                not _valid_sha256(checkpoint.get(name))
                for name in (
                    "core_sha256",
                    "config_sha256",
                    "embodiment_sha256",
                    "trainer_sha256",
                    "data_manifest_sha256",
                )
            )
        ):
            raise ValueError("Experiment 041 checkpoint identity is invalid")
    for ordinal, anchor in enumerate(anchors):
        _validate_anchor_record(anchor, report, ordinal)
    aggregate = _shard_aggregate(anchors, report)
    if report.get("aggregate") != aggregate:
        raise ValueError("Experiment 041 shard aggregate does not recompute")
    if require_registered:
        runtime = provenance.get("runtime")
        if (
            len(anchors) != EXPECTED_ANCHORS
            or report["aggregate"]["predictions"] != EXPECTED_PREDICTIONS
            or provenance.get("git_dirty") is not False
            or provenance.get("evaluator_sha256") != file_sha256(Path(__file__))
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
            or run.get("schedule_position")
            != REGISTERED_SCHEDULE.index((seed, morphology))
            or not report["aggregate"]["delivery_gate"]["passed"]
        ):
            raise ValueError(
                "registered Experiment 041 shard provenance or delivery is invalid"
            )
    started = _parse_utc(run.get("started_utc"))
    if _parse_utc(run["finished_utc"]) < started:
        raise ValueError("Experiment 041 shard interval is negative")


def evaluate_task_instruction_shard(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    source_paths: Mapping[str, Path],
    output: Path,
    *,
    seed: int,
    morphology: str,
    device: str,
    registered: bool,
    limit_scenarios: int | None = None,
    root: Path = Path("."),
    env_factory: Callable[..., Any] = BenchmarkManipulationEnv,
    expert_factory: Callable[[Any], Any] = PrivilegedBenchmarkExpert,
    policy_loader: Callable[[Path, str], Any] = load_geometry_policy,
    processor_factory: Callable[..., Any] = EmbodiProcessor.from_pretrained,
) -> dict[str, Any]:
    if registered and limit_scenarios is not None:
        raise ValueError("registered Experiment 041 does not permit a scenario limit")
    if registered and output != Path(
        "reports/exp41-runtime"
    ) / registered_shard_filename(seed, morphology):
        raise ValueError("registered Experiment 041 output path is frozen")
    if output.exists():
        raise FileExistsError(
            f"Experiment 041 shards are non-resumable; discard {output} first"
        )
    summaries = _source_contract(source_paths)
    provenance = _runtime_provenance(device)
    if registered and provenance["git_dirty"]:
        raise RuntimeError("registered Experiment 041 requires a clean worktree")
    checkpoints, checkpoint_identities = _checkpoint_identities(
        source_paths["exp36_summary"], seed, morphology, root
    )
    scenarios = task_instruction_scenarios(
        manifest,
        morphology,
        registered=registered,
        limit_scenarios=limit_scenarios,
    )
    run = _run_metadata(seed, morphology, device, registered)
    anchors, episodes = capture_expert_anchors(
        scenarios,
        morphology,
        env_factory=env_factory,
        expert_factory=expert_factory,
    )
    policies = {
        condition: policy_loader(checkpoints[condition], device)
        for condition in run["condition_order"]
    }
    first_policy = policies[run["condition_order"][0]]
    processor = processor_factory(
        first_policy.config.backbone_name,
        revision=first_policy.config.backbone_revision,
        do_rescale=first_policy.config.image_do_rescale,
    )
    records = []
    for anchor in anchors:
        results = {
            condition: evaluate_anchor_condition(
                anchor, condition, policies[condition], processor, device
            )
            for condition in run["condition_order"]
        }
        records.append(_anchor_record(anchor, results))
    report: dict[str, Any] = {
        "format_version": 1,
        "experiment": 41,
        "mode": "task_instruction_shard",
        "registered": registered,
        "status": "completed" if registered else "completed_smoke_unregistered",
        "complete": True,
        "contract": _contract(registered, limit_scenarios),
        "identity": {
            "model_seed": seed,
            "morphology": morphology,
            "checkpoints": checkpoint_identities,
        },
        "expected_anchor_ids": [
            f"{scenario.scenario_id}/{phase}"
            for scenario in scenarios
            for phase in PHASES[scenario.task]
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
        report, manifest, summaries["exp36_summary"], require_registered=registered
    )
    _write_final(output, report)
    return report


def _condition_seed_metrics(
    records: Sequence[dict[str, Any]], condition: str, seed: int
) -> dict[str, Any]:
    return hierarchical_metrics(
        [
            record
            for record in records
            if record["condition"] == condition and record["seed"] == seed
        ]
    )


def classify_support(
    seed_metrics: Mapping[int, Mapping[str, Any]],
    task_margins: Mapping[str, float | None],
    *,
    delivery: bool,
) -> dict[str, Any]:
    benefits = [seed_metrics[seed].get("relative_benefit") for seed in SEEDS]
    margins = [seed_metrics[seed].get("margin") for seed in SEEDS]
    mean_benefit = (
        _mean([float(value) for value in benefits if value is not None])
        if all(value is not None for value in benefits)
        else None
    )

    def nonnegative_task_margin(task: str) -> bool:
        value = task_margins.get(task)
        return value is not None and float(value) >= 0

    checks = {
        "delivery_gate_passed": delivery,
        "three_seed_mean_relative_benefit_at_least_ten_percent": mean_benefit
        is not None
        and mean_benefit >= 0.10,
        "at_least_two_nonnegative_seed_margins": sum(
            value is not None and value >= 0 for value in margins
        )
        >= 2,
        "every_task_three_seed_mean_margin_nonnegative": all(
            nonnegative_task_margin(task) for task in TASKS
        ),
    }
    supported = all(checks.values())
    return {
        "supported": supported,
        "classification": "counterfactual_instruction_expert_agreement_support"
        if supported
        else "no_counterfactual_instruction_support",
        "checks": checks,
        "three_seed_mean_relative_benefit": mean_benefit,
        "per_seed_margin": {
            str(seed): seed_metrics[seed].get("margin") for seed in SEEDS
        },
        "per_seed_relative_benefit": {
            str(seed): seed_metrics[seed].get("relative_benefit") for seed in SEEDS
        },
        "per_task_three_seed_mean_margin": dict(task_margins),
        "inferential_claim": False,
    }


def _nullable_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _nullable_difference(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    return float(first) - float(second)


def classify_joint_deficit(
    condition_metrics: Mapping[str, Mapping[str, Any]],
    seed_metrics: Mapping[str, Mapping[int, Mapping[str, Any]]],
    matched_support: Mapping[str, Any],
) -> dict[str, Any]:
    ratio = _nullable_ratio(
        condition_metrics["joint"].get("correct_loss"),
        condition_metrics["matched"].get("correct_loss"),
    )
    seed_ratios = {
        seed: _nullable_ratio(
            seed_metrics["joint"][seed].get("correct_loss"),
            seed_metrics["matched"][seed].get("correct_loss"),
        )
        for seed in SEEDS
    }
    interaction = _nullable_difference(
        condition_metrics["matched"].get("relative_benefit"),
        condition_metrics["joint"].get("relative_benefit"),
    )
    seed_interactions = {
        seed: _nullable_difference(
            seed_metrics["matched"][seed].get("relative_benefit"),
            seed_metrics["joint"][seed].get("relative_benefit"),
        )
        for seed in SEEDS
    }
    all_seed_values_defined = all(
        seed_ratios[seed] is not None
        and seed_metrics["joint"][seed].get("relative_benefit") is not None
        and seed_metrics["matched"][seed].get("relative_benefit") is not None
        and seed_interactions[seed] is not None
        for seed in SEEDS
    )
    checks = {
        "matched_condition_supported": matched_support.get("supported") is True,
        "all_required_seed_values_defined": all_seed_values_defined,
        "joint_to_matched_correct_loss_ratio_at_least_1_10": ratio is not None
        and ratio >= 1.10,
        "at_least_two_seed_loss_ratios_at_least_1_10": sum(
            value is not None and value >= 1.10 for value in seed_ratios.values()
        )
        >= 2,
        "condition_benefit_interaction_at_least_ten_points": interaction is not None
        and interaction >= 0.10,
        "at_least_two_seed_benefit_interactions_at_least_ten_points": sum(
            value is not None and value >= 0.10 for value in seed_interactions.values()
        )
        >= 2,
    }
    supported = all(checks.values())
    return {
        "supported": supported,
        "classification": "joint_specific_prompt_effect_deficit"
        if supported
        else "no_joint_specific_prompt_effect_deficit",
        "checks": checks,
        "joint_to_matched_correct_loss_ratio": ratio,
        "per_seed_correct_loss_ratio": {
            str(key): value for key, value in seed_ratios.items()
        },
        "matched_minus_joint_relative_benefit": interaction,
        "per_seed_matched_minus_joint_relative_benefit": {
            str(key): value for key, value in seed_interactions.items()
        },
    }


def aggregate_registered_shards(
    source_paths: Mapping[str, Path],
    manifest: ScenarioManifest,
    *,
    exp36_summary: dict[str, Any],
) -> dict[str, Any]:
    if (
        exp36_summary.get("experiment") != 36
        or exp36_summary.get("status") != "completed_gate_passed"
        or exp36_summary.get("fixed_step") != 1200
        or file_sha256(Path(exp40.__file__)) != EXP40_EVALUATOR_SHA256
    ):
        raise ValueError("registered aggregation requires the frozen source contracts")
    expected_source_keys = {
        f"seed{seed}/{morphology}": (seed, morphology)
        for seed, morphology in REGISTERED_SCHEDULE
    }
    if set(source_paths) != set(expected_source_keys):
        raise ValueError(
            "registered aggregation requires exact paths for all six source shards"
        )
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    source_hashes = {}
    for label, key in expected_source_keys.items():
        path = source_paths[label]
        seed, morphology = key
        expected_filename = registered_shard_filename(seed, morphology)
        if path.name != expected_filename:
            raise ValueError(f"source shard path is misattributed: {label}")
        source_bytes = path.read_bytes()
        report = json.loads(source_bytes)
        identity = report.get("identity", {})
        report_key = (identity.get("model_seed"), identity.get("morphology"))
        if report_key != key:
            raise ValueError(f"source shard identity is misattributed: {label}")
        if report_key in indexed:
            raise ValueError(f"duplicate Experiment 041 shard identity: {report_key}")
        indexed[key] = report
        source_hashes[label] = hashlib.sha256(source_bytes).hexdigest()
    if set(indexed) != set(REGISTERED_SCHEDULE):
        raise ValueError("registered aggregation requires all six shard identities")
    evaluator_hash = file_sha256(Path(__file__))
    commits = set()
    runtimes = []
    signatures = set()
    intervals = []
    lock_paths = set()
    delivery_by_shard = {}
    for position, key in enumerate(REGISTERED_SCHEDULE):
        report = indexed[key]
        validate_shard_report(report, manifest, exp36_summary, require_registered=True)
        provenance = report["provenance"]
        run = report["run"]
        if provenance["evaluator_sha256"] != evaluator_hash:
            raise ValueError(
                "registered shard evaluator hash differs from aggregation evaluator"
            )
        commits.add(provenance["git_commit"])
        signatures.add(provenance["runtime_signature_sha256"])
        runtimes.append(provenance["runtime"])
        lock_paths.add(run["lock_path"])
        intervals.append(
            (position, _parse_utc(run["started_utc"]), _parse_utc(run["finished_utc"]))
        )
        delivery_by_shard[f"seed{key[0]}/{key[1]}"] = report["aggregate"][
            "delivery_gate"
        ]
    if (
        len(commits) != 1
        or len(signatures) != 1
        or len(lock_paths) != 1
        or lock_paths != {str(REGISTERED_LOCK_PATH)}
        or any(runtime != runtimes[0] for runtime in runtimes)
        or any(
            first[2] > second[1]
            for first, second in zip(intervals, intervals[1:], strict=False)
        )
    ):
        raise ValueError(
            "registered shards violate commit/runtime/lock/sequential schedule"
        )
    records = _flatten_condition_records([indexed[key] for key in REGISTERED_SCHEDULE])
    seeds = {
        condition: {
            seed: _condition_seed_metrics(records, condition, seed) for seed in SEEDS
        }
        for condition in CONDITIONS
    }
    conditions = {
        condition: {
            name: (
                None
                if any(seeds[condition][seed].get(name) is None for seed in SEEDS)
                else _mean([float(seeds[condition][seed][name]) for seed in SEEDS])
            )
            for name in (*METRIC_PATHS, "margin", "relative_benefit")
        }
        | {
            "correct_degenerate_rotation_count": sum(
                int(seeds[condition][seed]["correct_degenerate_rotation_count"])
                for seed in SEEDS
            ),
            "wrong_degenerate_rotation_count": sum(
                int(seeds[condition][seed]["wrong_degenerate_rotation_count"])
                for seed in SEEDS
            ),
        }
        for condition in CONDITIONS
    }
    task_margins = {
        condition: {
            task: _mean(
                [
                    hierarchical_metrics(
                        [
                            record
                            for record in records
                            if record["condition"] == condition
                            and record["seed"] == seed
                            and record["task"] == task
                        ]
                    )["margin"]
                    for seed in SEEDS
                ]
            )
            for task in TASKS
        }
        for condition in CONDITIONS
    }
    delivery = all(value["passed"] for value in delivery_by_shard.values())
    support = {
        condition: classify_support(
            seeds[condition], task_margins[condition], delivery=delivery
        )
        for condition in CONDITIONS
    }
    deficit = classify_joint_deficit(conditions, seeds, support["matched"])
    return {
        "format_version": 1,
        "experiment": 41,
        "registered": True,
        "status": "completed_development_diagnostic",
        "scope": "development-only diagnostic; no inferential, closed-loop-success, admission, advancement, or final-split claim",
        "contract": {
            **_contract(True, None),
            "evaluator_sha256": evaluator_hash,
            "evaluator_git_commit": next(iter(commits)),
            "runtime_signature_sha256": next(iter(signatures)),
            "shards": 6,
            "expert_episodes": 126,
            "anchors": 546,
            "policy_inferences": 4368,
        },
        "source_shard_sha256": dict(sorted(source_hashes.items())),
        "delivery_gates_by_shard": delivery_by_shard,
        "delivery_gate_passed": delivery,
        "metrics_by_condition": conditions,
        "metrics_by_condition_seed": {
            f"{condition}/seed{seed}": seeds[condition][seed]
            for condition in CONDITIONS
            for seed in SEEDS
        },
        "metrics_by_condition_seed_morphology": _subgroup_metrics(
            records, ("condition", "seed", "morphology")
        ),
        "metrics_by_condition_seed_task": _subgroup_metrics(
            records, ("condition", "seed", "task")
        ),
        "metrics_by_condition_seed_morphology_task": _subgroup_metrics(
            records, ("condition", "seed", "morphology", "task")
        ),
        "metrics_by_condition_seed_morphology_task_phase": _subgroup_metrics(
            records, ("condition", "seed", "morphology", "task", "phase")
        ),
        "task_margins": task_margins,
        "support_by_condition": support,
        "joint_specific_prompt_effect_deficit": deficit,
        "multiplicity_adjusted_inferential_claim": False,
        "closed_loop_success_claim": False,
        "admission_gate": None,
        "advancement_gate": None,
        "final_split_loaded": False,
    }


def registered_shard_filename(seed: int, morphology: str) -> str:
    if seed not in SEEDS or morphology not in MORPHOLOGIES:
        raise ValueError(
            "registered shard filename requires a registered seed and morphology"
        )
    return f"benchmark-exp41-seed{seed}-{morphology}-task-instruction.json"


def _registered_output(seed: int, morphology: str, output: Path | None) -> Path:
    expected = REGISTERED_RUNTIME_DIR / registered_shard_filename(seed, morphology)
    if output is not None and output != expected:
        raise ValueError(f"registered shard output must be {expected}")
    return expected


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="run frozen Experiment 041 task-instruction evaluator"
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
        "--exp36-summary",
        type=Path,
        default=Path("reports/benchmark-exp36-summary.json"),
    )
    common.add_argument(
        "--exp38-summary",
        type=Path,
        default=Path("reports/benchmark-exp38-summary.json"),
    )
    common.add_argument(
        "--exp39-summary",
        type=Path,
        default=Path("reports/benchmark-exp39-summary.json"),
    )
    common.add_argument(
        "--exp40-summary",
        type=Path,
        default=Path("reports/benchmark-exp40-summary.json"),
    )
    shard = subparsers.add_parser("shard", parents=[common])
    shard.add_argument("--seed", choices=SEEDS, type=int, required=True)
    shard.add_argument("--morphology", choices=MORPHOLOGIES, required=True)
    shard.add_argument("--artifact-root", type=Path, default=Path("."))
    shard.add_argument("--output", type=Path)
    shard.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    shard.add_argument("--smoke", action="store_true")
    shard.add_argument("--limit-scenarios", type=_positive)
    run = subparsers.add_parser("run-registered", parents=[common])
    run.add_argument("--artifact-root", type=Path, default=Path("."))
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    aggregate = subparsers.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--shard-dir", type=Path, default=REGISTERED_RUNTIME_DIR)
    aggregate.add_argument(
        "--output", type=Path, default=Path("reports/benchmark-exp41-summary.json")
    )
    return parser.parse_args(argv)


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "exp36_summary": args.exp36_summary,
        "exp38_summary": args.exp38_summary,
        "exp39_summary": args.exp39_summary,
        "exp40_summary": args.exp40_summary,
    }


def _run_all_registered(
    args: argparse.Namespace,
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
) -> list[dict[str, Any]]:
    outputs = [
        _registered_output(seed, morphology, None)
        for seed, morphology in REGISTERED_SCHEDULE
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"Experiment 041 is non-resumable; discard existing shard before restart: {existing[0]}"
        )
    reports = []
    with registered_run_lock():
        for (seed, morphology), output in zip(
            REGISTERED_SCHEDULE, outputs, strict=True
        ):
            reports.append(
                evaluate_task_instruction_shard(
                    definition,
                    manifest,
                    _source_paths(args),
                    output,
                    seed=seed,
                    morphology=morphology,
                    device=args.device,
                    registered=True,
                    root=args.artifact_root,
                )
            )
    return reports


def main() -> None:
    args = parse_args()
    registered = not getattr(args, "smoke", False)
    if args.command == "shard":
        if not registered and args.output is None:
            raise ValueError("smoke mode requires an explicit custom --output")
        if not registered and args.output.parent == REGISTERED_RUNTIME_DIR:
            raise ValueError(
                "smoke output may not use the registered runtime directory"
            )
        if registered and args.limit_scenarios is not None:
            raise ValueError("--limit-scenarios is only available in smoke mode")
    if args.command == "aggregate" and args.shard_dir != REGISTERED_RUNTIME_DIR:
        raise ValueError(
            f"registered aggregation shard directory must be {REGISTERED_RUNTIME_DIR}"
        )
    definition, manifest = load_development_contract(
        args.definition, args.manifest, registered=registered
    )
    if (
        definition.sha256 != DEFINITION_SHA256
        or file_sha256(manifest.path) != DEVELOPMENT_MANIFEST_SHA256
    ):
        raise ValueError(
            "Experiment 041 benchmark contract does not match registration"
        )
    summaries = _source_contract(_source_paths(args))
    if args.command == "shard":
        output = (
            args.output
            if not registered
            else _registered_output(args.seed, args.morphology, args.output)
        )
        if registered:
            with registered_run_lock():
                report = evaluate_task_instruction_shard(
                    definition,
                    manifest,
                    _source_paths(args),
                    output,
                    seed=args.seed,
                    morphology=args.morphology,
                    device=args.device,
                    registered=True,
                    root=args.artifact_root,
                )
        else:
            report = evaluate_task_instruction_shard(
                definition,
                manifest,
                _source_paths(args),
                output,
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
                {"completed_shards": len(reports), "status": "completed"}, indent=2
            )
        )
        return
    paths = {
        f"seed{seed}/{morphology}": args.shard_dir
        / registered_shard_filename(seed, morphology)
        for seed, morphology in REGISTERED_SCHEDULE
    }
    summary = aggregate_registered_shards(
        paths,
        manifest,
        exp36_summary=summaries["exp36_summary"],
    )
    _write_final(args.output, summary)
    print(
        json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2)
    )


if __name__ == "__main__":
    main()
