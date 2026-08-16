from pathlib import Path
from queue import Empty, Full
from types import SimpleNamespace
from typing import Any
import threading

import numpy as np
import pytest
import torch

from embodi.sim import runtime as runtime_module
from embodi.sim.runtime import Observation, SimulationRuntime


class _FakeEvent:
    def __init__(self, set_: bool = False) -> None:
        self.set_calls = int(set_)

    def is_set(self) -> bool:
        return self.set_calls > 0

    def set(self) -> None:
        self.set_calls += 1

    def wait(self, timeout: float | None = None) -> bool:
        return self.is_set()


class _InlineThread:
    def __init__(self, *, target: Any, name: str, daemon: bool) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon
        self.alive = False

    def start(self) -> None:
        self.alive = True
        try:
            self.target()
        finally:
            self.alive = False

    def join(self, timeout: float | None = None) -> None:
        assert not self.alive

    def is_alive(self) -> bool:
        return self.alive


class _LatestQueue:
    def __init__(self, value: Any = None) -> None:
        self.value = value
        self.has_value = value is not None
        self.get_timeouts: list[float | None] = []

    def put_nowait(self, value: Any) -> None:
        if self.has_value:
            raise Full
        self.value = value
        self.has_value = True

    def get(self, timeout: float | None = None) -> Any:
        self.get_timeouts.append(timeout)
        return self.get_nowait()

    def get_nowait(self) -> Any:
        if not self.has_value:
            raise Empty
        self.has_value = False
        return self.value

    def empty(self) -> bool:
        return not self.has_value


class _DelayedQueue:
    """Expose a result only after reset's best-effort queue drain."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.empty_calls = 0
        self.consumed = False

    def empty(self) -> bool:
        self.empty_calls += 1
        return True

    def get_nowait(self) -> Any:
        if self.consumed:
            raise Empty
        self.consumed = True
        return self.value


class _FakeProcessor:
    def __init__(self, batch: dict[str, Any] | None = None) -> None:
        self.batch = batch or {"input_ids": torch.tensor([[1]]), "metadata": "kept"}
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        images: list[list[torch.Tensor]],
        instructions: list[str],
        state: torch.Tensor,
        **canonical: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "images": images,
                "instructions": instructions,
                "state": state,
                **canonical,
            }
        )
        return dict(self.batch)


class _FakePolicy:
    def __init__(
        self,
        runtime: SimulationRuntime,
        *,
        format_version: int | None,
        camera_names: tuple[str, ...] = ("top", "wrist"),
        image_do_rescale: bool = True,
        failure: BaseException | None = None,
    ) -> None:
        config = {
            "backbone_name": "fake-backbone",
            "backbone_revision": "fake-revision",
            "camera_names": camera_names,
            "image_do_rescale": image_do_rescale,
        }
        if format_version is not None:
            config["format_version"] = format_version
        self.config = SimpleNamespace(**config)
        self.runtime = runtime
        self.failure = failure
        self.reset_calls = 0
        self.batches: list[dict[str, Any]] = []
        self.action = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        self.batches.append(batch)
        if self.failure is not None:
            raise self.failure
        self.runtime.stop_event.set()
        return self.action


class _FakeEnvironment:
    def __init__(self, runtime: SimulationRuntime) -> None:
        self.runtime = runtime
        self.data = SimpleNamespace(time=0.0)
        self.episode = 1
        self.lifted = False
        self.success = False
        self.reset_calls = 0
        self.applied: list[np.ndarray] = []
        self.closed = False

    def render_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full((2, 3, 3), 255, dtype=np.uint8),
            np.full((3, 4, 3), 64, dtype=np.uint8),
        )

    def state(self) -> np.ndarray:
        return np.arange(6, dtype=np.float32)

    def canonical_state(self) -> np.ndarray:
        return np.arange(10, dtype=np.float32).reshape(1, 10)

    def reset(self) -> None:
        self.reset_calls += 1
        self.episode += 1
        self.data.time = 0.0

    def apply_action(self, action: np.ndarray) -> None:
        self.applied.append(action)

    def step_control_period(self, frequency: float) -> None:
        self.data.time += 1.0 / frequency
        self.runtime.stop_event.set()

    def close(self) -> None:
        self.closed = True


def _observation(generation: int = 17) -> Observation:
    return Observation(
        generation=generation,
        state=np.arange(6, dtype=np.float32),
        canonical_state=np.arange(10, dtype=np.float32).reshape(1, 10),
        top=np.full((2, 3, 3), 255, dtype=np.uint8),
        wrist=np.full((3, 4, 3), 64, dtype=np.uint8),
    )


def _run_policy_once(
    monkeypatch: pytest.MonkeyPatch,
    *,
    format_version: int | None,
    camera_names: tuple[str, ...] = ("top", "wrist"),
    image_do_rescale: bool = True,
    device: str = "cpu",
    processor_batch: dict[str, Any] | None = None,
) -> tuple[SimulationRuntime, _FakePolicy, _FakeProcessor, list[tuple[Any, ...]]]:
    runtime = SimulationRuntime(Path("fake-checkpoint"), device=device)
    runtime.stop_event = _FakeEvent()  # type: ignore[assignment]
    runtime.policy_ready_event = _FakeEvent()  # type: ignore[assignment]
    runtime.observations = _LatestQueue(_observation())  # type: ignore[assignment]
    runtime.action_chunks = _LatestQueue()  # type: ignore[assignment]
    policy = _FakePolicy(
        runtime,
        format_version=format_version,
        camera_names=camera_names,
        image_do_rescale=image_do_rescale,
    )
    processor = _FakeProcessor(processor_batch)
    load_calls: list[tuple[Any, ...]] = []
    processor_calls: list[tuple[Any, ...]] = []

    def load_policy(checkpoint: Path, policy_device: torch.device) -> _FakePolicy:
        load_calls.append((checkpoint, policy_device))
        return policy

    def load_processor(*args: Any, **kwargs: Any) -> _FakeProcessor:
        processor_calls.append((*args, kwargs))
        return processor

    monkeypatch.setattr(runtime_module, "load_inference_policy", load_policy)
    monkeypatch.setattr(
        runtime_module,
        "EmbodiProcessor",
        SimpleNamespace(from_pretrained=load_processor),
    )

    runtime._policy_loop()

    assert load_calls == [(Path("fake-checkpoint"), runtime.device)]
    assert processor_calls == [
        (
            "fake-backbone",
            {
                "revision": "fake-revision",
                "do_rescale": image_do_rescale,
            },
        )
    ]
    return runtime, policy, processor, load_calls


@pytest.mark.parametrize("format_version", [None, 2, 3], ids=["legacy", "v2", "v3"])
def test_policy_loop_builds_checkpoint_specific_batches(
    monkeypatch: pytest.MonkeyPatch,
    format_version: int | None,
) -> None:
    runtime, policy, processor, _ = _run_policy_once(
        monkeypatch,
        format_version=format_version,
    )

    assert policy.reset_calls == 1
    assert runtime.policy_ready_event.is_set()
    assert processor.calls[0]["instructions"] == [runtime_module.TASK]
    torch.testing.assert_close(
        processor.calls[0]["state"],
        torch.arange(6, dtype=torch.float32).unsqueeze(0),
    )
    if format_version is None:
        assert set(processor.calls[0]) == {"images", "instructions", "state"}
    elif format_version == 2:
        canonical = processor.calls[0]["canonical_state"]
        assert canonical.shape == (1, 4, 10)
        torch.testing.assert_close(canonical[0, 0], torch.arange(10, dtype=torch.float32))
        assert torch.count_nonzero(canonical[0, 1:]) == 0
        assert torch.equal(
            processor.calls[0]["canonical_slot_mask"],
            torch.tensor([[True, False, False, False]]),
        )
    else:
        canonical = processor.calls[0]["canonical_state"]
        assert canonical.shape == (1, 1, 10)
        torch.testing.assert_close(canonical[0, 0], torch.arange(10, dtype=torch.float32))
        assert torch.equal(
            processor.calls[0]["canonical_part_mask"],
            torch.ones((1, 1), dtype=torch.bool),
        )

    generation, chunk, elapsed_ms = runtime.action_chunks.value
    assert generation == 17
    np.testing.assert_array_equal(chunk, policy.action[0].numpy())
    assert elapsed_ms >= 0.0
    assert runtime.stop_event.is_set()


@pytest.mark.parametrize("processor_rescales", [True, False])
def test_policy_loop_preserves_camera_order_and_scales_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    processor_rescales: bool,
) -> None:
    _, _, processor, _ = _run_policy_once(
        monkeypatch,
        format_version=None,
        camera_names=("wrist", "top"),
        image_do_rescale=processor_rescales,
    )

    wrist, top = processor.calls[0]["images"][0]
    assert wrist.shape == (3, 3, 4)
    assert top.shape == (3, 2, 3)
    if processor_rescales:
        assert wrist.dtype == torch.uint8
        assert top.dtype == torch.uint8
        assert torch.all(wrist == 64)
        assert torch.all(top == 255)
    else:
        assert wrist.dtype == torch.float32
        assert top.dtype == torch.float32
        torch.testing.assert_close(wrist, torch.full_like(wrist, 64 / 255))
        torch.testing.assert_close(top, torch.ones_like(top))


def test_policy_loop_moves_tensor_batch_values_to_runtime_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = object()
    runtime, policy, _, _ = _run_policy_once(
        monkeypatch,
        format_version=None,
        device="meta",
        processor_batch={
            "input_ids": torch.tensor([[1, 2]]),
            "observation.state": torch.arange(6).unsqueeze(0),
            "metadata": metadata,
        },
    )

    assert runtime.device.type == "meta"
    assert policy.batches[0]["input_ids"].device.type == "meta"
    assert policy.batches[0]["observation.state"].device.type == "meta"
    assert policy.batches[0]["metadata"] is metadata


def test_simulation_reset_rejects_delayed_stale_policy_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimulationRuntime(Path("fake-checkpoint"), device="cpu", frequency=30)
    runtime.stop_event = _FakeEvent()  # type: ignore[assignment]
    runtime.ready_event = _FakeEvent()  # type: ignore[assignment]
    runtime.observations = _LatestQueue()  # type: ignore[assignment]
    acknowledgement = _FakeEvent()
    reset_request = runtime_module._ResetRequest(acknowledgement, _FakeEvent())  # type: ignore[arg-type]
    runtime.reset_requests = _LatestQueue(reset_request)  # type: ignore[assignment]
    stale_chunk = np.ones((2, 6), dtype=np.float32)
    runtime.action_chunks = _DelayedQueue((0, stale_chunk, 123.0))  # type: ignore[assignment]
    environment = _FakeEnvironment(runtime)
    inference_values: list[float] = []
    monkeypatch.setattr(runtime_module, "SO101PickPlaceEnv", lambda **_: environment)
    monkeypatch.setattr(
        runtime,
        "_publish_frame",
        lambda top, wrist, env, inference_ms: inference_values.append(inference_ms),
    )

    runtime._simulation_loop()

    assert acknowledgement.is_set()
    assert environment.reset_calls == 1
    assert environment.applied == []
    assert environment.closed
    assert runtime.observations.value.generation == 1
    assert inference_values == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("failure_stage", ["load", "predict"])
def test_policy_worker_records_exceptions_and_signals_stop(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    runtime = SimulationRuntime(Path("fake-checkpoint"), device="cpu")
    runtime.stop_event = _FakeEvent()  # type: ignore[assignment]
    runtime.policy_ready_event = _FakeEvent()  # type: ignore[assignment]
    runtime.observations = _LatestQueue(_observation())  # type: ignore[assignment]
    failure = RuntimeError(f"{failure_stage} failed")
    policy = _FakePolicy(runtime, format_version=3, failure=failure)
    processor = _FakeProcessor()

    def load_policy(checkpoint: Path, device: torch.device) -> _FakePolicy:
        if failure_stage == "load":
            raise failure
        return policy

    monkeypatch.setattr(runtime_module, "load_inference_policy", load_policy)
    monkeypatch.setattr(
        runtime_module,
        "EmbodiProcessor",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
    )

    runtime._policy_loop()

    assert runtime.error is failure
    assert runtime.stop_event.is_set()
    assert runtime.policy_ready_event.is_set()


def test_start_propagates_policy_load_exception_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimulationRuntime(Path("fake-checkpoint"), device="cpu")
    failure = RuntimeError("policy failed")

    def simulation_worker() -> None:
        runtime.ready_event.set()

    def load_policy(checkpoint: Path, device: torch.device) -> _FakePolicy:
        raise failure

    monkeypatch.setattr(runtime, "_simulation_loop", simulation_worker)
    monkeypatch.setattr(runtime_module, "load_inference_policy", load_policy)
    monkeypatch.setattr(runtime_module, "threading", SimpleNamespace(Thread=_InlineThread))

    with pytest.raises(RuntimeError, match="^policy failed$") as raised:
        runtime.start(timeout=0.0)

    assert raised.value.__cause__ is failure
    assert runtime.sim_thread is not None and not runtime.sim_thread.is_alive()
    assert runtime.policy_thread is not None and not runtime.policy_thread.is_alive()


def test_start_retains_simulation_startup_error_and_does_not_start_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimulationRuntime(Path("fake-checkpoint"), device="cpu")
    failure = RuntimeError("environment failed")
    policy_started = threading.Event()

    def simulation_worker() -> None:
        runtime.error = failure
        runtime.stop_event.set()
        runtime.ready_event.set()

    monkeypatch.setattr(runtime, "_simulation_loop", simulation_worker)
    monkeypatch.setattr(runtime, "_policy_loop", policy_started.set)
    monkeypatch.setattr(runtime_module, "threading", SimpleNamespace(Thread=_InlineThread))

    with pytest.raises(RuntimeError, match="^simulation startup failed$") as raised:
        runtime.start(timeout=0.0)

    assert raised.value.__cause__ is failure
    assert not policy_started.is_set()
    assert runtime.policy_thread is None
