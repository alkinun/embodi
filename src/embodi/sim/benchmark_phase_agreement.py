from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray
import torch

from ..processing import EmbodiProcessor, camera_frame_to_tensor
from ..record_so101 import atomic_write_json
from . import benchmark_task_instruction as exp41
from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import BenchmarkManipulationEnv, runtime_scenario
from .benchmark_expert import BenchmarkPhase, PrivilegedBenchmarkExpert
from .benchmark_geometry_evaluation import (
    CONFIG_SHA256,
    DATA_MANIFEST_SHA256,
    load_development_contract,
    load_geometry_policy,
    resolve_checkpoint,
)


DEFINITION_SHA256 = "9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f"
DEVELOPMENT_MANIFEST_SHA256 = (
    "4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576"
)
EXP36_SUMMARY_SHA256 = (
    "76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f"
)
EXP41_SUMMARY_SHA256 = (
    "d8195516eb6c84d90da31e74459b33779aab3382000650df8e9b941f35cd217f"
)
EXP41_EVALUATOR_SHA256 = (
    "2ca9ae5b48782b4b378882accf886dd9b65bd1c7ed3905bb20dfc3c859f9d62c"
)
EXPERT_SHA256 = "7fb869a10756864530889f809a8305ee1758bdc51eaeeeda0ff2f451f3b7f8e9"
PREREGISTRATION_GIT_COMMIT = "38a868b6dee792a220f8fd4c6a786eb00f0c1da0"

CONDITIONS = ("joint", "matched")
MORPHOLOGIES = ("so101", "panda")
TASKS = ("push_to_zone", "lift_object", "pick_place_bin")
TASK_KEYS = {"push_to_zone": "push", "lift_object": "lift", "pick_place_bin": "pick"}
SEEDS = (36001, 36002, 36003)
SCENARIO_INDICES = tuple(range(28, 35))
ENDPOINTS = ("first", "last")
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
INSTRUCTIONS = dict(exp41.INSTRUCTIONS)
INSTRUCTION_SHA256 = dict(exp41.INSTRUCTION_SHA256)
ACTION_SCALES = np.array((0.25, 0.25, 0.25, 1, 1, 1, 1, 1, 1, 0.5))
REPEAT_MAX_ABS = 1e-5
CLIP_TOLERANCE = 2e-5
IK_ITERATIONS = 20
MAXIMUM_STEPS = 500
EXPECTED_PHASE_UNITS = 91
EXPECTED_ENDPOINTS = 182
EXPECTED_SCORED_CHUNKS = 1092
EXPECTED_REPEAT_CHUNKS = 1092
EXPECTED_INFERENCES = 2184
REGISTERED_RUNTIME_DIR = Path("reports/exp42-runtime")
REGISTERED_LOCK_PATH = REGISTERED_RUNTIME_DIR / ".registered.lock"
LOCK_IDENTITY = "exp42-so101-panda-phase-endpoints-v1"
SCHEDULE_IDENTITY = LOCK_IDENTITY
REGISTERED_SCHEDULE = MORPHOLOGIES
SUMMARY_PATH = Path("reports/benchmark-exp42-summary.json")
EMPTY_OUTCOME_CHAIN_SHA256 = hashlib.sha256(b"").hexdigest()
RUNTIME_FIELDS = exp41.RUNTIME_FIELDS

METRICS = (
    "normalized_loss",
    "translation_norm_error_m",
    "rotation_geodesic_error_deg",
    "effective_gripper_error",
    "degenerate_rotation_rate",
)

PROTOCOL = {
    "camera": "top",
    "resolution": [640, 480],
    "state_path": "forward_kinematics_canonical",
    "expert": "PrivilegedBenchmarkExpert",
    "frequency_hz": 30,
    "maximum_episode_steps": MAXIMUM_STEPS,
    "capture": "first loop-entry phase_tick==1 and final pre-call phase record",
    "endpoints": list(ENDPOINTS),
    "expert_action_calls_per_step": 1,
    "execute_original_native_expert_action": True,
    "state_unchanged_before_canonical_conversion": True,
    "ik_iterations": IK_ITERATIONS,
    "predictor": "predict_canonical_action_chunk",
    "required_output_shape": [1, 32, 1, 10],
    "policy_reset_before_every_prediction": True,
    "correct_prompt_only": True,
    "immediate_repeat": True,
    "repeat_max_abs_tolerance": REPEAT_MAX_ABS,
    "lead": 0,
    "action_scales": ACTION_SCALES.tolist(),
    "hierarchy": ["endpoint", "phase", "scenario", "task", "morphology", "seed"],
    "condition_order_formula": "(anchor_ordinal + seed_ordinal + morphology_ordinal) % 2",
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
    endpoint: str
    phase_tick: int
    frame: NDArray[Any]
    state: NDArray[Any]
    expert_target: NDArray[Any]
    evidence: dict[str, Any]


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
            raise RuntimeError("registered Experiment 042 requires the active orchestrator schedule")

    def complete(self, morphology: str) -> None:
        self.require(morphology)
        self.next_position += 1


def _valid_run_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and set(value) <= set("0123456789abcdef")
    )


_ACTIVE_ORCHESTRATOR: _OrchestratorContext | None = None


def canonical_array_sha256(value: Any) -> str:
    return exp41.canonical_array_sha256(value)


def rotation_geodesic_error(first: Any, second: Any) -> tuple[float, bool]:
    return exp41.rotation_geodesic_error(first, second)


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
        raise ValueError("Experiment 042 timestamps must be UTC ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("Experiment 042 timestamp is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("Experiment 042 timestamp must use UTC")
    return parsed


def _write_final(path: Path, report: dict[str, Any]) -> None:
    value = _jsonable(report)
    text = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"refusing to replace different report: {path}")
        return
    atomic_write_json(path, value)


def _preflight_final(path: Path, report: dict[str, Any]) -> None:
    text = json.dumps(_jsonable(report), indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"refusing to replace different report: {path}")


def _write_exact_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise FileExistsError(f"refusing to replace different report: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{hashlib.sha256(value).hexdigest()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary report: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
    temporary.replace(path)


@contextmanager
def registered_run_lock(path: Path = REGISTERED_LOCK_PATH) -> Iterator[None]:
    if path != REGISTERED_LOCK_PATH:
        raise ValueError(f"registered Experiment 042 lock path must be {REGISTERED_LOCK_PATH}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another registered Experiment 042 orchestrator holds {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def phase_agreement_scenarios(
    manifest: ScenarioManifest,
    morphology: str,
    *,
    registered: bool,
    limit_scenarios: int | None = None,
) -> tuple[Scenario, ...]:
    if morphology not in MORPHOLOGIES:
        raise ValueError("Experiment 042 morphology is invalid")
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
            raise ValueError("registered Experiment 042 requires exact indices 0028 through 0034")
        for task in TASKS:
            axes = [
                name
                for scenario in selected
                if scenario.task == task
                for name, value in scenario.factors.items()
                if str(value.get("bin", "")).startswith("development:")
            ]
            if len(axes) != 7 or len(set(axes)) != 7:
                raise ValueError("each Experiment 042 cell must cover all seven factor axes")
    return selected if limit_scenarios is None else selected[:limit_scenarios]


def condition_order(anchor_ordinal: int, seed: int, morphology: str) -> tuple[str, str]:
    if type(anchor_ordinal) is not int or anchor_ordinal < 0:
        raise ValueError("anchor ordinal must be a nonnegative integer")
    try:
        seed_ordinal = SEEDS.index(seed)
        morphology_ordinal = MORPHOLOGIES.index(morphology)
    except ValueError as error:
        raise ValueError("condition order requires registered seed and morphology") from error
    parity = (anchor_ordinal + seed_ordinal + morphology_ordinal) % 2
    return ("joint", "matched") if parity == 0 else ("matched", "joint")


def _phase_name(value: Any) -> str:
    return str(value.value if isinstance(value, BenchmarkPhase) else value)


def _native_shape(env: Any) -> tuple[int, ...]:
    return (int(env.morphology.native_action_dim),)


def _clipping_from_arrays(decoded: Any, realizable: Any) -> tuple[bool, float]:
    decoded_array = np.asarray(decoded, dtype=np.float64)
    realizable_array = np.asarray(realizable, dtype=np.float64)
    if decoded_array.shape != realizable_array.shape:
        raise ValueError("decoded and realizable native actions must share shape")
    error = np.abs(decoded_array - realizable_array)
    return bool(np.any(error > CLIP_TOLERANCE)), float(error.max(initial=0.0))


def _capture_call_evidence(
    env: Any,
    scenario: Scenario,
    phase: str,
    phase_tick: int,
    total_tick: int,
    call_index: int,
    frame: NDArray[Any],
    state: NDArray[Any],
    pre_call_native_state: NDArray[Any],
    requested: NDArray[Any],
    post_phase: str,
    post_phase_tick: int,
    post_total_tick: int,
) -> tuple[NDArray[Any], dict[str, Any]]:
    unchanged = np.asarray(env.canonical_state(), dtype=np.float32)
    post_state_sha256 = canonical_array_sha256(unchanged)
    if (
        unchanged.shape != (1, 10)
        or not np.isfinite(unchanged).all()
        or post_state_sha256 != canonical_array_sha256(state)
    ):
        raise ValueError("expert action changed canonical state before target conversion")
    target = np.asarray(env.canonical_arm_action(requested), dtype=np.float32)
    decoded = np.asarray(
        env.native_action_from_canonical(target, iterations=IK_ITERATIONS),
        dtype=np.float32,
    )
    realized = np.asarray(env.canonical_arm_action(decoded), dtype=np.float32)
    realizable = np.asarray(env.qpos_to_native(env.native_to_qpos(decoded)), dtype=np.float64)
    if (
        target.shape != (1, 10)
        or decoded.shape != _native_shape(env)
        or realizable.shape != _native_shape(env)
        or realized.shape != (1, 10)
        or not all(
            np.isfinite(value).all() for value in (target, decoded, realizable, realized)
        )
    ):
        raise ValueError("expert canonical round trip returned invalid data")
    rotation_error, degenerate = rotation_geodesic_error(
        target[0, 3:9], realized[0, 3:9]
    )
    if degenerate:
        raise ValueError("expert round trip produced a degenerate rotation")
    clipped, clipping_error = _clipping_from_arrays(decoded, realizable)
    target_float = np.asarray(target, dtype=np.float64)
    realized_float = np.asarray(realized, dtype=np.float64)
    requested_float = np.asarray(requested, dtype=np.float64)
    decoded_float = np.asarray(decoded, dtype=np.float64)
    evidence = {
        "scenario_id": scenario.scenario_id,
        "scenario_seed": scenario.seed,
        "morphology": scenario.morphology,
        "task": scenario.task,
        "phase": phase,
        "phase_tick": phase_tick,
        "pre_call_phase": phase,
        "pre_call_phase_tick": phase_tick,
        "pre_call_total_tick": total_tick,
        "post_action_phase": post_phase,
        "post_action_phase_tick": post_phase_tick,
        "post_action_total_tick": post_total_tick,
        "expert_call_index": call_index,
        "image_sha256": canonical_array_sha256(frame),
        "state_sha256": canonical_array_sha256(state),
        "post_action_state_sha256": post_state_sha256,
        "pre_call_native_state": pre_call_native_state.tolist(),
        "pre_call_native_state_sha256": canonical_array_sha256(pre_call_native_state),
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
                np.linalg.norm(target_float[0, :3] - realized_float[0, :3])
            ),
            "rotation_error_deg": rotation_error,
            "gripper_error_native": float(abs(requested_float[-1] - decoded_float[-1])),
        },
    }
    return target, evidence


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
    if (
        _phase_name(expert.phase) != "open"
        or expert.phase_tick != 0
        or getattr(expert, "total_tick", None) != 0
    ):
        raise RuntimeError("expert episode must begin at open phase_tick and total_tick zero")
    endpoints: dict[str, dict[str, Anchor]] = defaultdict(dict)
    observed_phases: list[str] = []
    status = None
    steps = 0
    for call_index in range(maximum_steps):
        phase = _phase_name(expert.phase)
        phase_tick = int(expert.phase_tick)
        total_tick = int(expert.total_tick)
        if not observed_phases or observed_phases[-1] != phase:
            observed_phases.append(phase)
        scored = phase in expected_phases
        frame = state = pre_call_native_state = None
        if scored:
            frame = np.asarray(env.render_top()).copy()
            state = np.asarray(env.canonical_state(), dtype=np.float32).copy()
            pre_call_native_state = np.asarray(env.state(), dtype=np.float32).copy()
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise ValueError("expert anchor image must be top 640x480 uint8 RGB")
            if state.shape != (1, 10) or not np.isfinite(state).all():
                raise ValueError("expert anchor state must be finite canonical [1,10]")
            if (
                pre_call_native_state.shape != _native_shape(env)
                or not np.isfinite(pre_call_native_state).all()
            ):
                raise ValueError("expert pre-call native state is invalid")
        requested = np.asarray(expert.action(), dtype=np.float32)
        if requested.shape != _native_shape(env) or not np.isfinite(requested).all():
            raise ValueError("expert returned an invalid native action")
        post_phase = _phase_name(expert.phase)
        post_phase_tick = int(expert.phase_tick)
        post_total_tick = int(expert.total_tick)
        if post_phase != phase and observed_phases[-1] != post_phase:
            observed_phases.append(post_phase)
        transitioned = post_phase != phase
        if scored and (phase_tick == 1 or transitioned):
            assert frame is not None and state is not None and pre_call_native_state is not None
            target, evidence = _capture_call_evidence(
                env,
                scenario,
                phase,
                phase_tick,
                total_tick,
                call_index,
                frame,
                state,
                pre_call_native_state,
                requested,
                post_phase,
                post_phase_tick,
                post_total_tick,
            )
            if phase_tick == 1:
                endpoints[phase]["first"] = Anchor(
                    -1,
                    scenario.scenario_id,
                    scenario.seed,
                    scenario.morphology,
                    scenario.task,
                    phase,
                    "first",
                    phase_tick,
                    frame,
                    state,
                    target,
                    {**evidence, "endpoint": "first"},
                )
            if transitioned:
                endpoints[phase]["last"] = Anchor(
                    -1,
                    scenario.scenario_id,
                    scenario.seed,
                    scenario.morphology,
                    scenario.task,
                    phase,
                    "last",
                    phase_tick,
                    frame,
                    state,
                    target,
                    {**evidence, "endpoint": "last"},
                )
        env.apply_action(requested)
        status = env.step_control_period(float(expert.frequency))
        steps += 1
        if expert.terminal:
            break
    final_phase = _phase_name(expert.phase)
    observed_scored_phases = [
        value for value in observed_phases if value not in {"open", "done", "failed"}
    ]
    if (
        observed_phases != ["open", *expected_phases, "done"]
        or observed_scored_phases != list(expected_phases)
        or final_phase != "done"
        or status is None
        or status.terminal is not True
        or status.success is not True
        or env.success is not True
    ):
        raise RuntimeError(
            f"expert runtime phase sequence or outcome is invalid: {observed_phases}"
        )
    ordered: list[Anchor] = []
    for phase in expected_phases:
        if set(endpoints[phase]) != set(ENDPOINTS):
            raise RuntimeError(f"expert phase {phase} did not provide both endpoints")
        for endpoint in ENDPOINTS:
            anchor = endpoints[phase][endpoint]
            ordinal = starting_ordinal + len(ordered)
            ordered.append(
                replace(anchor, ordinal=ordinal, evidence={**anchor.evidence, "ordinal": ordinal})
            )
    return ordered, {
        "scenario_id": scenario.scenario_id,
        "task": scenario.task,
        "steps": steps,
        "success": True,
        "terminal": True,
        "final_phase": final_phase,
        "observed_phases": observed_phases,
        "scored_phases": observed_scored_phases,
        "endpoint_count": len(ordered),
        "phase_endpoint_ticks": {
            phase: [endpoints[phase][endpoint].phase_tick for endpoint in ENDPOINTS]
            for phase in expected_phases
        },
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
    expected = sum(len(PHASES[scenario.task]) * 2 for scenario in scenarios)
    if len(anchors) != expected or [anchor.ordinal for anchor in anchors] != list(range(expected)):
        raise RuntimeError("expert anchors do not match scenario/phase/endpoint ordering")
    return anchors, episodes


def action_metrics(prediction: Any, expert: Any) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(10)
    target = np.asarray(expert, dtype=np.float64).reshape(10)
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("action metrics require finite canonical actions")
    rotation, degenerate = rotation_geodesic_error(predicted[3:9], target[3:9])
    channel_losses = ((predicted - target) / ACTION_SCALES) ** 2
    return {
        "normalized_loss": float(np.mean(channel_losses)),
        "channel_losses": channel_losses.tolist(),
        "translation_norm_error_m": float(np.linalg.norm(predicted[:3] - target[:3])),
        "rotation_geodesic_error_deg": rotation,
        "effective_gripper_error": float(
            abs(np.clip(predicted[9], 0.0, 1.0) - np.clip(target[9], 0.0, 1.0))
        ),
        "degenerate_rotation": degenerate,
        "degenerate_rotation_rate": float(degenerate),
    }


def _policy_batch(
    anchor: Anchor, policy: Any, processor: Any, device: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    instruction_key = TASK_KEYS[anchor.task]
    source_image = np.asarray(anchor.frame)
    source_state = np.asarray(anchor.state)
    image = camera_frame_to_tensor(
        source_image, processor_rescales=policy.config.image_do_rescale
    )
    canonical_state = torch.from_numpy(source_state).unsqueeze(0)
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
        raise ValueError("processor batch must contain tensor input_ids and attention_mask")
    input_ids_array = input_ids.detach().cpu().contiguous().numpy()
    attention_mask_array = attention_mask.detach().cpu().contiguous().numpy()
    processed_image = image.detach().cpu().contiguous().numpy()
    passed_state = canonical_state.detach().cpu().contiguous().numpy()
    evidence = {
        "instruction_key": instruction_key,
        "instruction": INSTRUCTIONS[instruction_key],
        "instruction_sha256": hashlib.sha256(INSTRUCTIONS[instruction_key].encode()).hexdigest(),
        "input_ids": input_ids_array.tolist(),
        "input_ids_sha256": canonical_array_sha256(input_ids),
        "input_ids_dtype": input_ids_array.dtype.str,
        "input_ids_shape": list(input_ids_array.shape),
        "attention_mask": attention_mask_array.tolist(),
        "attention_mask_sha256": canonical_array_sha256(attention_mask),
        "attention_mask_dtype": attention_mask_array.dtype.str,
        "attention_mask_shape": list(attention_mask_array.shape),
        "image_sha256": canonical_array_sha256(source_image),
        "state_sha256": canonical_array_sha256(source_state),
        "processed_image_sha256": canonical_array_sha256(processed_image),
        "processed_image_dtype": processed_image.dtype.str,
        "processed_image_shape": list(processed_image.shape),
        "passed_state_sha256": canonical_array_sha256(passed_state),
        "passed_state_dtype": passed_state.dtype.str,
        "passed_state_shape": list(passed_state.shape),
        "expert_target_sha256": anchor.evidence["expert_target_sha256"],
    }
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    return moved, evidence


def _predict_once(
    anchor: Anchor, policy: Any, processor: Any, device: str
) -> tuple[NDArray[Any], dict[str, Any]]:
    policy.reset()
    batch, evidence = _policy_batch(anchor, policy, processor, device)
    value = policy.predict_canonical_action_chunk(batch)
    output = (
        value.detach().float().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )
    output = np.asarray(output, dtype=np.float32)
    if output.shape != (1, 32, 1, 10) or not np.isfinite(output).all():
        raise ValueError("policy output must be finite with shape [1,32,1,10]")
    chunk = np.ascontiguousarray(output[0])
    evidence.update(
        output_chunk=chunk.tolist(),
        output_sha256=canonical_array_sha256(chunk),
        output_dtype=chunk.dtype.str,
        output_shape=list(chunk.shape),
    )
    return output, evidence


def evaluate_anchor_condition(
    anchor: Anchor,
    seed: int,
    condition: str,
    policy: Any,
    processor: Any,
    device: str,
) -> dict[str, Any]:
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("policy query requires a registered seed and condition")
    first, first_evidence = _predict_once(anchor, policy, processor, device)
    repeat, repeat_evidence = _predict_once(anchor, policy, processor, device)
    repeat_evidence["repeat"] = True
    input_fields = (
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
        "expert_target_sha256",
        "processed_image_sha256",
        "processed_image_dtype",
        "processed_image_shape",
        "passed_state_sha256",
        "passed_state_dtype",
        "passed_state_shape",
    )
    if any(first_evidence[name] != repeat_evidence[name] for name in input_fields):
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
    stable_fields = (
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
    if any(query[field] != first[field] for query in queries[1:] for field in stable_fields):
        raise RuntimeError("anchor policy inputs differ across seed-condition arms")


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric cell")
    return float(sum(float(value) for value in values) / len(values))


def _nullable_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if not np.isfinite(numerator_value) or not np.isfinite(denominator_value) or denominator_value == 0:
        return None
    return numerator_value / denominator_value


def _metric_vector(record: Mapping[str, Any]) -> NDArray[Any]:
    metrics = record["metrics"]
    channels = np.asarray(metrics["channel_losses"], dtype=np.float64)
    if channels.shape != (10,) or not np.isfinite(channels).all():
        raise ValueError("metric records require ten finite channel losses")
    values = np.asarray([float(metrics[name]) for name in METRICS], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("metric records require finite registered metrics")
    return np.concatenate((values, channels))


def hierarchical_condition_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("hierarchical aggregation requires records")
    keyed: dict[tuple[Any, ...], list[NDArray[Any]]] = defaultdict(list)
    dimensions = ("seed", "morphology", "task", "scenario_id", "phase", "endpoint")
    for record in records:
        keyed[tuple(record[name] for name in dimensions)].append(_metric_vector(record))
    level: dict[tuple[Any, ...], NDArray[Any]] = {
        key: np.mean(values, axis=0) for key, values in keyed.items()
    }
    # Remove endpoint, phase, scenario, task, morphology, then seed in the frozen order.
    for _ in range(len(dimensions)):
        grouped: dict[tuple[Any, ...], list[NDArray[Any]]] = defaultdict(list)
        for key, value in level.items():
            grouped[key[:-1]].append(value)
        level = {key: np.mean(values, axis=0) for key, values in grouped.items()}
    if set(level) != {()}:
        raise RuntimeError("hierarchical aggregation did not collapse to one cell")
    result = level[()]
    return {
        **{name: float(result[index]) for index, name in enumerate(METRICS)},
        "channel_losses": result[len(METRICS) :].tolist(),
        "degenerate_rotation_count": sum(
            bool(record["metrics"]["degenerate_rotation"]) for record in records
        ),
        "records": len(records),
    }


def paired_hierarchical_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    conditions = {
        condition: hierarchical_condition_metrics(
            [record for record in records if record["condition"] == condition]
        )
        for condition in CONDITIONS
    }
    paired = {
        name: {
            "joint": conditions["joint"][name],
            "matched": conditions["matched"][name],
            "difference": conditions["joint"][name] - conditions["matched"][name],
            "ratio": _nullable_ratio(conditions["joint"][name], conditions["matched"][name]),
        }
        for name in METRICS
    }
    paired["channels"] = {
        str(index): {
            "joint": conditions["joint"]["channel_losses"][index],
            "matched": conditions["matched"]["channel_losses"][index],
            "difference": conditions["joint"]["channel_losses"][index]
            - conditions["matched"]["channel_losses"][index],
            "ratio": _nullable_ratio(
                conditions["joint"]["channel_losses"][index],
                conditions["matched"]["channel_losses"][index],
            ),
        }
        for index in range(10)
    }
    paired["degenerate_rotation_count"] = {
        condition: conditions[condition]["degenerate_rotation_count"]
        for condition in CONDITIONS
    }
    paired["records_per_condition"] = {
        condition: conditions[condition]["records"] for condition in CONDITIONS
    }
    return paired


def _paired_subgroups(
    records: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    return {
        "/".join(str(value) for value in key): paired_hierarchical_metrics(values)
        for key, values in sorted(grouped.items())
    }


def aggregate_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": paired_hierarchical_metrics(records),
        "by_seed": _paired_subgroups(records, ("seed",)),
        "by_task": _paired_subgroups(records, ("task",)),
        "by_morphology": _paired_subgroups(records, ("morphology",)),
        "by_endpoint": _paired_subgroups(records, ("endpoint",)),
        "by_task_phase": _paired_subgroups(records, ("task", "phase")),
        "by_channel": paired_hierarchical_metrics(records)["channels"],
    }


def delivery_gate(anchors: Sequence[dict[str, Any]]) -> dict[str, Any]:
    round_trips = []
    for anchor in anchors:
        decoded = np.asarray(anchor["decoded_native"], dtype=np.float32)
        realizable = np.asarray(anchor["realizable_native"], dtype=np.float64)
        target = np.asarray(anchor["expert_target"], dtype=np.float32).reshape(1, 10)
        realized = np.asarray(anchor["realized_canonical"], dtype=np.float32).reshape(1, 10)
        requested = np.asarray(anchor["expert_native"], dtype=np.float32)
        clipped, clipping_error = _clipping_from_arrays(decoded, realizable)
        rotation_error, degenerate = rotation_geodesic_error(
            target[0, 3:9], realized[0, 3:9]
        )
        round_trips.append(
            {
                "clipped": clipped,
                "clipping_error_native": clipping_error,
                "translation_error_m": float(np.linalg.norm(target[0, :3] - realized[0, :3])),
                "rotation_error_deg": rotation_error,
                "gripper_error_native": float(abs(requested[-1] - decoded[-1])),
                "degenerate_expert_target": degenerate,
            }
        )
    count = len(round_trips)
    clipped_count = sum(bool(value["clipped"]) for value in round_trips)
    measurements = {
        "commands": count,
        "clipped_commands": clipped_count,
        "clipping_rate": clipped_count / count if count else None,
        "translation_p95_m": float(
            np.percentile([value["translation_error_m"] for value in round_trips], 95)
        )
        if count
        else None,
        "rotation_p95_deg": float(
            np.percentile([value["rotation_error_deg"] for value in round_trips], 95)
        )
        if count
        else None,
        "maximum_gripper_error_native": max(
            (float(value["gripper_error_native"]) for value in round_trips), default=None
        ),
        "degenerate_expert_targets": sum(
            bool(value["degenerate_expert_target"]) for value in round_trips
        ),
    }
    def at_most(value: Any, maximum: float) -> bool:
        return isinstance(value, (int, float)) and np.isfinite(value) and value <= maximum

    checks = {
        "decoded_action_clipping_at_most_one_percent": at_most(
            measurements["clipping_rate"], 0.01
        ),
        "translation_p95_at_most_1mm": at_most(
            measurements["translation_p95_m"], 0.001
        ),
        "rotation_p95_at_most_1deg": at_most(
            measurements["rotation_p95_deg"], 1.0
        ),
        "maximum_gripper_error_at_most_one_native_unit": at_most(
            measurements["maximum_gripper_error_native"], 1.0
        ),
        "expert_targets_non_degenerate": measurements["degenerate_expert_targets"] == 0
        and count > 0,
    }
    return {"measurements": measurements, "checks": checks, "passed": all(checks.values())}


def classify_broad_deficit(metrics: Mapping[str, Any], *, delivery: bool) -> dict[str, Any]:
    overall = metrics.get("overall", {})
    seeds = metrics.get("by_seed", {})
    endpoints = metrics.get("by_endpoint", {})
    tasks = metrics.get("by_task", {})
    morphologies = metrics.get("by_morphology", {})

    def ratio(cell: Mapping[str, Any], metric: str) -> float | None:
        value = cell.get(metric, {})
        result = value.get("ratio") if isinstance(value, Mapping) else None
        return float(result) if isinstance(result, (int, float)) and np.isfinite(result) else None

    def difference(cell: Mapping[str, Any], metric: str) -> float | None:
        value = cell.get(metric, {})
        result = value.get("difference") if isinstance(value, Mapping) else None
        return float(result) if isinstance(result, (int, float)) and np.isfinite(result) else None

    overall_ratio = ratio(overall, "normalized_loss")
    seed_ratios = {str(seed): ratio(seeds.get(str(seed), {}), "normalized_loss") for seed in SEEDS}
    endpoint_ratios = {
        endpoint: ratio(endpoints.get(endpoint, {}), "normalized_loss")
        for endpoint in ENDPOINTS
    }
    task_differences = {
        task: difference(tasks.get(task, {}), "normalized_loss") for task in TASKS
    }
    morphology_differences = {
        morphology: difference(morphologies.get(morphology, {}), "normalized_loss")
        for morphology in MORPHOLOGIES
    }
    gripper_ratio = ratio(overall, "effective_gripper_error")
    gripper_difference = difference(overall, "effective_gripper_error")
    gripper_seed_ratios = {
        str(seed): ratio(seeds.get(str(seed), {}), "effective_gripper_error")
        for seed in SEEDS
    }

    def value_at_least(value: float | None, minimum: float) -> bool:
        return value is not None and value >= minimum

    checks = {
        "delivery_gate_passed": delivery,
        "overall_normalized_ratio_at_least_1_10": overall_ratio is not None
        and overall_ratio >= 1.10,
        "at_least_two_seed_normalized_ratios_at_least_1_10": sum(
            value is not None and value >= 1.10 for value in seed_ratios.values()
        )
        >= 2,
        "both_endpoint_ratios_at_least_1_10": all(
            value_at_least(endpoint_ratios[endpoint], 1.10)
            for endpoint in ENDPOINTS
        ),
        "every_task_difference_nonnegative": all(
            value_at_least(task_differences[task], 0)
            for task in TASKS
        ),
        "both_morphology_differences_nonnegative": all(
            value_at_least(morphology_differences[morphology], 0)
            for morphology in MORPHOLOGIES
        ),
        "overall_gripper_ratio_at_least_1_10": gripper_ratio is not None
        and gripper_ratio >= 1.10,
        "overall_gripper_absolute_difference_at_least_0_01": gripper_difference
        is not None
        and gripper_difference >= 0.01,
        "at_least_two_seed_gripper_ratios_at_least_1_10": sum(
            value is not None and value >= 1.10 for value in gripper_seed_ratios.values()
        )
        >= 2,
    }
    confirmed = all(checks.values())
    return {
        "confirmed": confirmed,
        "classification": "broad_correct_prompt_expert_action_agreement_deficit"
        if confirmed
        else "broad_correct_prompt_expert_action_agreement_deficit_not_confirmed",
        "checks": checks,
        "overall_normalized_ratio": overall_ratio,
        "per_seed_normalized_ratio": seed_ratios,
        "per_endpoint_normalized_ratio": endpoint_ratios,
        "per_task_normalized_difference": task_differences,
        "per_morphology_normalized_difference": morphology_differences,
        "overall_effective_gripper_ratio": gripper_ratio,
        "overall_effective_gripper_difference": gripper_difference,
        "per_seed_effective_gripper_ratio": gripper_seed_ratios,
        "inferential_claim": False,
        "physical_component_claim": False,
    }


def _source_contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    expected = {
        "exp36_summary": EXP36_SUMMARY_SHA256,
        "exp41_summary": EXP41_SUMMARY_SHA256,
    }
    for name, digest in expected.items():
        if name not in paths or file_sha256(paths[name]) != digest:
            raise ValueError(
                f"{name.replace('_', ' ')} hash does not match Experiment 042 registration"
            )
    if file_sha256(Path(exp41.__file__)) != EXP41_EVALUATOR_SHA256:
        raise ValueError("imported Experiment 041 evaluator hash does not match registration")
    expert_path = Path(__file__).with_name("benchmark_expert.py")
    if file_sha256(expert_path) != EXPERT_SHA256:
        raise ValueError("expert implementation hash does not match Experiment 042 registration")
    summaries = {name: json.loads(paths[name].read_bytes()) for name in expected}
    if (
        summaries["exp36_summary"].get("experiment") != 36
        or summaries["exp36_summary"].get("status") != "completed_gate_passed"
        or summaries["exp36_summary"].get("fixed_step") != 1200
        or summaries["exp41_summary"].get("experiment") != 41
        or summaries["exp41_summary"].get("registered") is not True
        or summaries["exp41_summary"].get("delivery_gate_passed") is not True
        or summaries["exp41_summary"].get("final_split_loaded") is not False
    ):
        raise ValueError("Experiment 042 frozen source summary contract is invalid")
    return summaries


def _contract(registered: bool, limit_scenarios: int | None) -> dict[str, Any]:
    return {
        "definition_sha256": DEFINITION_SHA256,
        "manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
        "exp36_summary_sha256": EXP36_SUMMARY_SHA256,
        "exp41_summary_sha256": EXP41_SUMMARY_SHA256,
        "exp41_evaluator_sha256": EXP41_EVALUATOR_SHA256,
        "expert_sha256": EXPERT_SHA256,
        "preregistration_git_commit": PREREGISTRATION_GIT_COMMIT,
        "split": "development",
        "scenario_indices": list(SCENARIO_INDICES),
        "tasks": list(TASKS),
        "conditions": list(CONDITIONS),
        "model_seeds": list(SEEDS),
        "limit_scenarios": None if registered else limit_scenarios,
        "protocol": PROTOCOL,
        "final_split_loaded": False,
    }


def _source_condition(condition: str, morphology: str) -> str:
    if condition == "joint":
        return "joint"
    if condition == "matched" and morphology in MORPHOLOGIES:
        return morphology
    raise ValueError("Experiment 042 condition must be joint or matched")


def _checkpoint_identities(
    summary_path: Path, morphology: str, root: Path
) -> tuple[dict[int, dict[str, Path]], dict[str, dict[str, dict[str, Any]]]]:
    paths: dict[int, dict[str, Path]] = {}
    identities: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        paths[seed] = {}
        identities[str(seed)] = {}
        for condition in CONDITIONS:
            source = _source_condition(condition, morphology)
            checkpoint, identity = resolve_checkpoint(
                summary_path, source, seed, morphology, root=root
            )
            identity["condition"] = condition
            identity["source_condition"] = source
            paths[seed][condition] = checkpoint
            identities[str(seed)][condition] = identity
    return paths, identities


def _runtime_provenance(device: str) -> dict[str, Any]:
    provenance = exp41._runtime_provenance(device)
    provenance["evaluator_sha256"] = file_sha256(Path(__file__))
    provenance["imported_exp41_evaluator_sha256"] = file_sha256(Path(exp41.__file__))
    provenance["expert_sha256"] = file_sha256(Path(__file__).with_name("benchmark_expert.py"))
    return provenance


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
    flattened = []
    for report in reports:
        for anchor in report["anchors"]:
            for seed in SEEDS:
                seed_results = anchor["seeds"][str(seed)]
                for condition in CONDITIONS:
                    flattened.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "morphology": anchor["morphology"],
                            "task": anchor["task"],
                            "scenario_id": anchor["scenario_id"],
                            "phase": anchor["phase"],
                            "endpoint": anchor["endpoint"],
                            **seed_results[condition],
                        }
                    )
    return flattened


def _shard_aggregate(anchors: Sequence[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    gate = delivery_gate(anchors)
    records = _flatten_condition_records([report])
    integrity = {
        "all_expected_endpoint_anchors": len(anchors) == len(report["expected_anchor_ids"])
        and [
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
        "identical_anchor_hashes_across_six_arms": all(
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
                _valid_sha256(report["identity"]["checkpoints"][str(seed)][condition].get(name))
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
    return {
        "episodes": len(report["episodes"]),
        "phase_units": len(anchors) // 2,
        "endpoint_anchors": len(anchors),
        "scored_chunks": len(anchors) * len(SEEDS) * len(CONDITIONS),
        "repeat_chunks": len(anchors) * len(SEEDS) * len(CONDITIONS),
        "policy_inferences": len(anchors) * len(SEEDS) * len(CONDITIONS) * 2,
        "metrics": aggregate_metrics(records),
        "delivery_gate": gate,
    }


def _validate_query(
    query: Mapping[str, Any], anchor: Mapping[str, Any], *, repeat: bool
) -> NDArray[Any]:
    key = TASK_KEYS[anchor["task"]]
    try:
        input_dtype = np.dtype(query.get("input_ids_dtype"))
        attention_dtype = np.dtype(query.get("attention_mask_dtype"))
        input_ids = np.asarray(query.get("input_ids"), dtype=input_dtype)
        attention_mask = np.asarray(query.get("attention_mask"), dtype=attention_dtype)
        chunk = np.asarray(query.get("output_chunk"), dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Experiment 042 policy arrays are invalid") from error
    if (
        input_dtype.kind not in "iu"
        or attention_dtype.kind not in "iub"
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or attention_mask.shape != input_ids.shape
        or query.get("input_ids_shape") != list(input_ids.shape)
        or query.get("attention_mask_shape") != list(attention_mask.shape)
        or query.get("input_ids_sha256") != canonical_array_sha256(input_ids)
        or query.get("attention_mask_sha256") != canonical_array_sha256(attention_mask)
        or query.get("instruction_key") != key
        or query.get("instruction") != INSTRUCTIONS[key]
        or query.get("instruction_sha256") != INSTRUCTION_SHA256[key]
        or query.get("image_sha256") != anchor["image_sha256"]
        or query.get("state_sha256") != anchor["state_sha256"]
        or query.get("expert_target_sha256") != anchor["expert_target_sha256"]
        or any(
            not _valid_sha256(query.get(name))
            for name in (
                "processed_image_sha256",
                "passed_state_sha256",
            )
        )
        or not isinstance(query.get("processed_image_dtype"), str)
        or not isinstance(query.get("processed_image_shape"), list)
        or query.get("passed_state_dtype") != np.dtype(np.float32).str
        or query.get("passed_state_shape") != [1, 1, 10]
        or query.get("output_dtype") != np.dtype(np.float32).str
        or query.get("output_shape") != [32, 1, 10]
        or chunk.shape != (32, 1, 10)
        or not np.isfinite(chunk).all()
        or query.get("output_sha256") != canonical_array_sha256(chunk)
        or (query.get("repeat") is True) is not repeat
    ):
        raise ValueError("Experiment 042 policy tensor hashes or contracts are invalid")
    return chunk


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
        or set(anchor.get("seeds", {})) != {str(seed) for seed in SEEDS}
        or tuple(anchor["seeds"]) != tuple(str(seed) for seed in SEEDS)
    ):
        raise ValueError("Experiment 042 endpoint anchor identity is invalid")
    native_shape = (6 if morphology == "so101" else 8,)
    arrays = (
        ("pre_call_native_state", "pre_call_native_state_sha256", np.float32, native_shape),
        ("expert_native", "expert_native_sha256", np.float32, native_shape),
        ("expert_target", "expert_target_sha256", np.float32, (1, 10)),
        ("decoded_native", "decoded_native_sha256", np.float32, native_shape),
        ("realizable_native", "realizable_native_sha256", np.float64, native_shape),
        ("realized_canonical", "realized_canonical_sha256", np.float32, (1, 10)),
    )
    decoded = realizable = target = realized = requested = pre_call_native = None
    for name, hash_name, dtype, shape in arrays:
        try:
            array = np.asarray(anchor.get(name), dtype=dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("Experiment 042 persisted expert array is invalid") from error
        if (
            array.shape != shape
            or not np.isfinite(array).all()
            or anchor.get(hash_name) != canonical_array_sha256(array)
        ):
            raise ValueError("Experiment 042 persisted expert array hash is invalid")
        if name == "decoded_native":
            decoded = array
        elif name == "pre_call_native_state":
            pre_call_native = array
        elif name == "realizable_native":
            realizable = array
        elif name == "expert_target":
            target = array
        elif name == "realized_canonical":
            realized = array
        elif name == "expert_native":
            requested = array
    assert decoded is not None and realizable is not None
    assert target is not None and realized is not None and requested is not None
    assert pre_call_native is not None
    target_float = np.asarray(target, dtype=np.float64)
    realized_float = np.asarray(realized, dtype=np.float64)
    requested_float = np.asarray(requested, dtype=np.float64)
    decoded_float = np.asarray(decoded, dtype=np.float64)
    clipped, clipping_error = _clipping_from_arrays(decoded, realizable)
    rotation, degenerate = rotation_geodesic_error(
        target_float[0, 3:9], realized_float[0, 3:9]
    )
    round_trip = anchor.get("round_trip", {})
    if (
        degenerate
        or round_trip.get("clipped") is not clipped
        or round_trip.get("clipping_error_native") != clipping_error
        or round_trip.get("translation_error_m")
        != float(np.linalg.norm(target_float[0, :3] - realized_float[0, :3]))
        or round_trip.get("rotation_error_deg") != rotation
        or round_trip.get("gripper_error_native")
        != float(abs(requested_float[-1] - decoded_float[-1]))
    ):
        raise ValueError("Experiment 042 round-trip evidence is invalid")
    phase_order = list(PHASES[anchor["task"]])
    phase_index = phase_order.index(anchor["phase"])
    expected_post_phase = (
        phase_order[phase_index + 1] if phase_index + 1 < len(phase_order) else "done"
    )
    transitioned = anchor.get("post_action_phase") != anchor["phase"]
    if anchor["endpoint"] == "last":
        if (
            not transitioned
            or anchor.get("post_action_phase") != expected_post_phase
            or anchor.get("post_action_phase_tick") != 1
        ):
            raise ValueError("Experiment 042 last endpoint did not cause the registered transition")
    elif transitioned and (
        anchor.get("post_action_phase") != expected_post_phase
        or anchor.get("post_action_phase_tick") != 1
    ):
        raise ValueError("Experiment 042 first endpoint transition evidence is invalid")
    elif not transitioned and anchor.get("post_action_phase_tick") != anchor["phase_tick"] + 1:
        raise ValueError("Experiment 042 nontransition phase tick is invalid")
    if anchor.get("post_action_phase") == "done" and not np.array_equal(
        requested, pre_call_native
    ):
        raise ValueError("Experiment 042 terminal transition target is not the native no-op state")
    for seed in SEEDS:
        results = anchor["seeds"][str(seed)]
        order = condition_order(ordinal, seed, morphology)
        if tuple(results) != order or set(results) != set(CONDITIONS):
            raise ValueError("Experiment 042 condition query ordering is invalid")
        for condition in CONDITIONS:
            result = results[condition]
            queries = result.get("queries")
            if (
                result.get("seed") != seed
                or result.get("condition") != condition
                or not isinstance(queries, list)
                or len(queries) != 2
            ):
                raise ValueError("Experiment 042 condition result is invalid")
            first = _validate_query(queries[0], anchor, repeat=False)
            repeat = _validate_query(queries[1], anchor, repeat=True)
            input_fields = (
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
                "expert_target_sha256",
                "processed_image_sha256",
                "processed_image_dtype",
                "processed_image_shape",
                "passed_state_sha256",
                "passed_state_dtype",
                "passed_state_shape",
            )
            difference = float(np.max(np.abs(first - repeat)))
            if (
                any(queries[0][name] != queries[1][name] for name in input_fields)
                or result.get("repeat_max_abs_difference") != difference
                or difference > REPEAT_MAX_ABS
                or result.get("metrics") != action_metrics(first[0, 0], target)
            ):
                raise ValueError("Experiment 042 repeat or policy metrics do not recompute")
    try:
        _require_stable_anchor_inputs(anchor["seeds"])
    except RuntimeError as error:
        raise ValueError("Experiment 042 cross-arm tokenizer inputs differ") from error


def _validate_endpoint_pairs(anchors: Sequence[dict[str, Any]]) -> None:
    duplicate_fields = (
        "scenario_id",
        "scenario_seed",
        "morphology",
        "task",
        "phase",
        "phase_tick",
        "pre_call_phase",
        "pre_call_phase_tick",
        "pre_call_total_tick",
        "post_action_phase",
        "post_action_phase_tick",
        "post_action_total_tick",
        "expert_call_index",
        "image_sha256",
        "state_sha256",
        "post_action_state_sha256",
        "pre_call_native_state",
        "pre_call_native_state_sha256",
        "expert_native",
        "expert_native_sha256",
        "expert_target",
        "expert_target_sha256",
        "decoded_native",
        "decoded_native_sha256",
        "realizable_native",
        "realizable_native_sha256",
        "realized_canonical",
        "realized_canonical_sha256",
        "round_trip",
    )
    for index in range(0, len(anchors), 2):
        first, last = anchors[index : index + 2]
        if (
            first.get("endpoint") != "first"
            or last.get("endpoint") != "last"
            or first.get("scenario_id") != last.get("scenario_id")
            or first.get("phase") != last.get("phase")
            or first.get("phase_tick") != 1
            or last.get("phase_tick", 0) < first.get("phase_tick", 1)
            or last.get("expert_call_index", -1) < first.get("expert_call_index", 0)
        ):
            raise ValueError("Experiment 042 endpoint pair ordering or ticks are invalid")
        same_call = first["expert_call_index"] == last["expert_call_index"]
        if same_call != (first["phase_tick"] == last["phase_tick"]):
            raise ValueError("Experiment 042 endpoint call/tick evidence is inconsistent")
        if same_call and any(first[field] != last[field] for field in duplicate_fields):
            raise ValueError("Experiment 042 one-action endpoint duplicate differs")


def _validate_checkpoints(
    report: Mapping[str, Any],
    exp36_summary: Mapping[str, Any],
    *,
    artifact_root: Path | None,
    require_files: bool,
) -> None:
    morphology = report["identity"]["morphology"]
    checkpoints = report["identity"].get("checkpoints", {})
    if set(checkpoints) != {str(seed) for seed in SEEDS}:
        raise ValueError("Experiment 042 checkpoint seed identities are incomplete")
    for seed in SEEDS:
        if set(checkpoints[str(seed)]) != set(CONDITIONS):
            raise ValueError("Experiment 042 checkpoint conditions are incomplete")
        for condition in CONDITIONS:
            source = _source_condition(condition, morphology)
            checkpoint = checkpoints[str(seed)][condition]
            expected_run = exp36_summary.get("runs", {}).get(f"{source}-seed{seed}", {})
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
                raise ValueError("Experiment 042 checkpoint identity is invalid")
            if require_files:
                if artifact_root is None:
                    raise ValueError("registered checkpoint validation requires artifact root")
                expected_path = artifact_root / expected_run["output"] / "final" / morphology
                checkpoint_path = Path(checkpoint.get("checkpoint_path", ""))
                data_manifest_path = expected_path.parent / "data_manifest.json"
                file_contract = {
                    "core_sha256": expected_path / "core.pt",
                    "config_sha256": expected_path / "config.json",
                    "embodiment_sha256": expected_path / "embodiment.pt",
                    "trainer_sha256": expected_path / "trainer.pt",
                    "data_manifest_sha256": data_manifest_path,
                }
                if (
                    checkpoint_path != expected_path
                    or any(not path.is_file() for path in file_contract.values())
                    or any(
                        file_sha256(path) != checkpoint[name]
                        for name, path in file_contract.items()
                    )
                ):
                    raise ValueError("Experiment 042 checkpoint files do not match identity")
                trainer = torch.load(
                    expected_path / "trainer.pt",
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                if trainer.get("step") != 1200 or trainer.get("stage") != "core":
                    raise ValueError("Experiment 042 trainer file contract is invalid")


def validate_shard_report(
    report: dict[str, Any],
    manifest: ScenarioManifest,
    exp36_summary: dict[str, Any],
    *,
    require_registered: bool,
    artifact_root: Path | None = None,
) -> None:
    identity = report.get("identity", {})
    morphology = identity.get("morphology")
    contract = report.get("contract", {})
    scenarios = phase_agreement_scenarios(
        manifest,
        morphology,
        registered=require_registered,
        limit_scenarios=None if require_registered else contract.get("limit_scenarios"),
    )
    expected_anchor_ids = [
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
        or report.get("experiment") != 42
        or report.get("mode") != "phase_action_quality_morphology_shard"
        or report.get("registered") is not require_registered
        or report.get("complete") is not True
        or report.get("status")
        != ("completed" if require_registered else "completed_smoke_unregistered")
        or morphology not in MORPHOLOGIES
        or identity.get("model_seeds") != list(SEEDS)
        or contract != _contract(require_registered, contract.get("limit_scenarios"))
        or report.get("expected_anchor_ids") != expected_anchor_ids
        or [
            f"{anchor.get('scenario_id')}/{anchor.get('phase')}/{anchor.get('endpoint')}"
            for anchor in anchors
        ]
        != expected_anchor_ids
        or len(episodes) != len(scenarios)
        or report.get("outcome_chain_sha256") != _outcome_chain(anchors)
        or report.get("infrastructure_errors") != []
        or run.get("seed_order") != list(SEEDS)
        or run.get("condition_order_formula") != PROTOCOL["condition_order_formula"]
        or (
            require_registered
            and not _valid_run_id(run.get("orchestrator_run_id"))
        )
        or (not require_registered and run.get("orchestrator_run_id") is not None)
        or run.get("device") != provenance.get("runtime", {}).get("device")
        or run.get("finished_utc") is None
    ):
        raise ValueError("Experiment 042 shard contract or completion state is invalid")
    for episode, scenario in zip(episodes, scenarios, strict=True):
        if (
            episode.get("scenario_id") != scenario.scenario_id
            or episode.get("task") != scenario.task
            or type(episode.get("steps")) is not int
            or not 1 <= episode["steps"] <= MAXIMUM_STEPS
            or episode.get("success") is not True
            or episode.get("terminal") is not True
            or episode.get("final_phase") != "done"
            or episode.get("observed_phases")
            != ["open", *PHASES[scenario.task], "done"]
            or episode.get("scored_phases") != list(PHASES[scenario.task])
            or episode.get("endpoint_count") != len(PHASES[scenario.task]) * 2
            or set(episode.get("phase_endpoint_ticks", {})) != set(PHASES[scenario.task])
        ):
            raise ValueError("Experiment 042 expert episode metadata is invalid")
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    for ordinal, anchor in enumerate(anchors):
        scenario = scenario_by_id[anchor["scenario_id"]]
        if (
            anchor.get("scenario_seed") != scenario.seed
            or anchor.get("task") != scenario.task
        ):
            raise ValueError("Experiment 042 anchor scenario metadata is invalid")
        _validate_anchor_record(anchor, report, ordinal)
    _validate_endpoint_pairs(anchors)
    for episode in episodes:
        matching = [anchor for anchor in anchors if anchor["scenario_id"] == episode["scenario_id"]]
        ticks = {
            phase: [
                anchor["phase_tick"]
                for anchor in matching
                if anchor["phase"] == phase
            ]
            for phase in PHASES[episode["task"]]
        }
        if episode["phase_endpoint_ticks"] != ticks:
            raise ValueError("Experiment 042 episode endpoint ticks do not recompute")
    _validate_checkpoints(
        report,
        exp36_summary,
        artifact_root=artifact_root,
        require_files=require_registered,
    )
    aggregate = _shard_aggregate(anchors, report)
    if report.get("aggregate") != aggregate:
        raise ValueError("Experiment 042 shard aggregate does not recompute")
    started = _parse_utc(run.get("started_utc"))
    if _parse_utc(run["finished_utc"]) < started:
        raise ValueError("Experiment 042 shard interval is negative")
    if require_registered:
        runtime = provenance.get("runtime")
        if (
            len(anchors) != EXPECTED_ENDPOINTS
            or aggregate["phase_units"] != EXPECTED_PHASE_UNITS
            or aggregate["scored_chunks"] != EXPECTED_SCORED_CHUNKS
            or aggregate["repeat_chunks"] != EXPECTED_REPEAT_CHUNKS
            or aggregate["policy_inferences"] != EXPECTED_INFERENCES
            or any(
                sum(
                    next(iter(anchor["seeds"][str(seed)])) == condition
                    for anchor in anchors
                )
                != 91
                for seed in SEEDS
                for condition in CONDITIONS
            )
            or provenance.get("git_dirty") is not False
            or provenance.get("evaluator_sha256") != file_sha256(Path(__file__))
            or provenance.get("imported_exp41_evaluator_sha256") != EXP41_EVALUATOR_SHA256
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
            raise ValueError("registered Experiment 042 shard provenance or delivery is invalid")


def evaluate_phase_agreement_shard(
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
    del definition
    if registered:
        if orchestrator is None:
            raise RuntimeError(
                "registered Experiment 042 evaluation requires orchestrator context"
            )
        orchestrator.require(morphology)
    elif orchestrator is not None:
        raise ValueError("smoke evaluation may not use registered orchestrator context")
    if registered and limit_scenarios is not None:
        raise ValueError("registered Experiment 042 does not permit a scenario limit")
    if registered and output != REGISTERED_RUNTIME_DIR / registered_shard_filename(morphology):
        raise ValueError("registered Experiment 042 output path is frozen")
    if output.exists():
        raise FileExistsError(f"Experiment 042 shards are non-resumable; discard {output} first")
    summaries = _source_contract(source_paths)
    provenance = _runtime_provenance(device)
    if registered and provenance["git_dirty"]:
        raise RuntimeError("registered Experiment 042 requires a clean worktree")
    checkpoints, checkpoint_identities = _checkpoint_identities(
        source_paths["exp36_summary"], morphology, root
    )
    scenarios = phase_agreement_scenarios(
        manifest,
        morphology,
        registered=registered,
        limit_scenarios=limit_scenarios,
    )
    run = _run_metadata(
        morphology,
        device,
        registered,
        None if orchestrator is None else orchestrator.run_id,
    )
    anchors, episodes = capture_expert_anchors(
        scenarios,
        morphology,
        env_factory=env_factory,
        expert_factory=expert_factory,
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
        raise ValueError("all Experiment 042 arms must share the processor contract")
    processor = processor_factory(
        processor_contract[0],
        revision=processor_contract[1],
        do_rescale=processor_contract[2],
    )
    records = []
    for anchor in anchors:
        seed_results = {}
        for seed in SEEDS:
            order = condition_order(anchor.ordinal, seed, morphology)
            seed_results[str(seed)] = {
                condition: evaluate_anchor_condition(
                    anchor,
                    seed,
                    condition,
                    policies[seed][condition],
                    processor,
                    device,
                )
                for condition in order
            }
        _require_stable_anchor_inputs(seed_results)
        records.append({**anchor.evidence, "seeds": seed_results})
    report: dict[str, Any] = {
        "format_version": 1,
        "experiment": 42,
        "mode": "phase_action_quality_morphology_shard",
        "registered": registered,
        "status": "completed" if registered else "completed_smoke_unregistered",
        "complete": True,
        "contract": _contract(registered, limit_scenarios),
        "identity": {
            "morphology": morphology,
            "model_seeds": list(SEEDS),
            "checkpoints": checkpoint_identities,
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
        summaries["exp36_summary"],
        require_registered=registered,
        artifact_root=root if registered else None,
    )
    _write_final(output, report)
    if orchestrator is not None:
        orchestrator.complete(morphology)
    return report


def registered_shard_filename(morphology: str) -> str:
    if morphology not in MORPHOLOGIES:
        raise ValueError("registered shard filename requires a registered morphology")
    return f"benchmark-exp42-{morphology}-phase-action-quality.json"


def _registered_output(morphology: str, output: Path | None) -> Path:
    expected = REGISTERED_RUNTIME_DIR / registered_shard_filename(morphology)
    if output is not None and output != expected:
        raise ValueError(f"registered shard output must be {expected}")
    return expected


def aggregate_registered_shards(
    source_paths: Mapping[str, Path],
    manifest: ScenarioManifest,
    *,
    exp36_summary: dict[str, Any],
    artifact_root: Path = Path("."),
) -> dict[str, Any]:
    expected_keys = set(MORPHOLOGIES)
    if set(source_paths) != expected_keys:
        raise ValueError("registered aggregation requires exact SO-101 and Panda source shards")
    if (
        exp36_summary.get("experiment") != 36
        or exp36_summary.get("status") != "completed_gate_passed"
        or exp36_summary.get("fixed_step") != 1200
        or file_sha256(Path(exp41.__file__)) != EXP41_EVALUATOR_SHA256
        or file_sha256(Path(__file__).with_name("benchmark_expert.py")) != EXPERT_SHA256
    ):
        raise ValueError("registered aggregation requires frozen source contracts")
    reports: dict[str, dict[str, Any]] = {}
    source_hashes = {}
    identities = set()
    for morphology in MORPHOLOGIES:
        path = source_paths[morphology]
        if path.name != registered_shard_filename(morphology):
            raise ValueError(f"source shard path is misattributed: {morphology}")
        raw = path.read_bytes()
        report = json.loads(raw)
        report_morphology = report.get("identity", {}).get("morphology")
        if report_morphology != morphology:
            raise ValueError(f"source shard identity is misattributed: {morphology}")
        if report_morphology in identities:
            raise ValueError(f"duplicate Experiment 042 shard identity: {report_morphology}")
        identities.add(report_morphology)
        reports[morphology] = report
        source_hashes[morphology] = hashlib.sha256(raw).hexdigest()
    commits = set()
    orchestrator_run_ids = set()
    signatures = set()
    runtimes = []
    intervals = []
    evaluator_hash = file_sha256(Path(__file__))
    delivery_by_shard = {}
    for position, morphology in enumerate(MORPHOLOGIES):
        report = reports[morphology]
        validate_shard_report(
            report,
            manifest,
            exp36_summary,
            require_registered=True,
            artifact_root=artifact_root,
        )
        provenance = report["provenance"]
        run = report["run"]
        if provenance["evaluator_sha256"] != evaluator_hash:
            raise ValueError("registered shard evaluator differs from aggregation evaluator")
        commits.add(provenance["git_commit"])
        orchestrator_run_ids.add(run["orchestrator_run_id"])
        signatures.add(provenance["runtime_signature_sha256"])
        runtimes.append(provenance["runtime"])
        intervals.append((_parse_utc(run["started_utc"]), _parse_utc(run["finished_utc"])))
        if run["schedule_position"] != position:
            raise ValueError("registered shards violate morphology schedule")
        delivery_by_shard[morphology] = report["aggregate"]["delivery_gate"]
    if (
        len(commits) != 1
        or len(orchestrator_run_ids) != 1
        or not _valid_run_id(next(iter(orchestrator_run_ids), None))
        or len(signatures) != 1
        or any(runtime != runtimes[0] for runtime in runtimes)
        or intervals[0][1] > intervals[1][0]
    ):
        raise ValueError("registered shards violate common commit/runtime or nonoverlap")
    records = _flatten_condition_records([reports[morphology] for morphology in MORPHOLOGIES])
    metrics = aggregate_metrics(records)
    delivery = all(gate["passed"] for gate in delivery_by_shard.values())
    classification = classify_broad_deficit(metrics, delivery=delivery)
    return {
        "format_version": 1,
        "experiment": 42,
        "registered": True,
        "status": "completed_development_diagnostic",
        "scope": "development-only diagnostic; no inferential, population, closed-loop, admission, advancement, or final-split claim",
        "contract": {
            **_contract(True, None),
            "evaluator_sha256": evaluator_hash,
            "evaluator_git_commit": next(iter(commits)),
            "orchestrator_run_id": next(iter(orchestrator_run_ids)),
            "runtime_signature_sha256": next(iter(signatures)),
            "shards": 2,
            "expert_episodes": 42,
            "scenario_phase_units": 182,
            "endpoint_records": 364,
            "scored_chunks": 2184,
            "repeat_chunks": 2184,
            "policy_inferences": 4368,
            "paired_joint_matched_comparisons": 1092,
        },
        "source_shard_sha256": source_hashes,
        "delivery_gates_by_shard": delivery_by_shard,
        "delivery_gate_passed": delivery,
        "metrics": metrics,
        "broad_correct_prompt_expert_action_agreement_deficit": classification,
        "multiplicity_adjusted_inferential_claim": False,
        "closed_loop_success_claim": False,
        "admission_gate": None,
        "advancement_gate": None,
        "final_split_loaded": False,
    }


def _promote_registered_sources(
    source_paths: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
    *,
    destination_dir: Path = Path("reports"),
) -> None:
    if set(source_paths) != set(MORPHOLOGIES):
        raise ValueError("source promotion requires both registered shards")
    if set(expected_sha256) != set(MORPHOLOGIES):
        raise ValueError("source promotion requires both summary source hashes")
    values = {
        morphology: source_paths[morphology].read_bytes() for morphology in MORPHOLOGIES
    }
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
    if any(source_paths[morphology].read_bytes() != values[morphology] for morphology in MORPHOLOGIES):
        raise ValueError("staged source bytes changed during promotion preflight")
    for morphology in MORPHOLOGIES:
        _write_exact_bytes(destinations[morphology], values[morphology])
    if any(source_paths[morphology].read_bytes() != values[morphology] for morphology in MORPHOLOGIES):
        raise ValueError("staged source bytes changed during promotion")


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="run frozen Experiment 042 phase-action-quality evaluator"
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
        "--exp41-summary", type=Path, default=Path("reports/benchmark-exp41-summary.json")
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
        "exp41_summary": args.exp41_summary,
    }


def _run_all_registered(
    args: argparse.Namespace,
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
) -> list[dict[str, Any]]:
    global _ACTIVE_ORCHESTRATOR
    outputs = [_registered_output(morphology, None) for morphology in MORPHOLOGIES]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Experiment 042 is non-resumable; delete both staged shards before a full restart"
        )
    reports = []
    with registered_run_lock(REGISTERED_LOCK_PATH):
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Experiment 042 staged shard appeared before schedule; delete both before restart"
            )
        orchestrator = _OrchestratorContext(uuid.uuid4().hex)
        if _ACTIVE_ORCHESTRATOR is not None:
            raise RuntimeError("registered Experiment 042 orchestrator is already active")
        _ACTIVE_ORCHESTRATOR = orchestrator
        try:
            for morphology, output in zip(MORPHOLOGIES, outputs, strict=True):
                reports.append(
                    evaluate_phase_agreement_shard(
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
                raise RuntimeError("registered Experiment 042 orchestrator did not complete")
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
        raise ValueError("Experiment 042 benchmark contract does not match registration")
    summaries = _source_contract(_source_paths(args))
    if args.command == "shard":
        report = evaluate_phase_agreement_shard(
            definition,
            manifest,
            _source_paths(args),
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
    source_paths = {
        morphology: args.shard_dir / registered_shard_filename(morphology)
        for morphology in MORPHOLOGIES
    }
    with registered_run_lock(REGISTERED_LOCK_PATH):
        summary = aggregate_registered_shards(
            source_paths,
            manifest,
            exp36_summary=summaries["exp36_summary"],
            artifact_root=args.artifact_root,
        )
        _preflight_final(args.output, summary)
        _promote_registered_sources(
            source_paths, summary["source_shard_sha256"]
        )
        _write_final(args.output, summary)
    print(json.dumps({"output": str(args.output), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
