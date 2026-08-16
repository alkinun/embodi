from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from embodi.sim import runtime as runtime_module
from embodi.sim.runtime import ChunkExecutor, SimulationRuntime


class _FakeEnvironment:
    def __init__(self) -> None:
        self.data = SimpleNamespace(time=0.0)
        self.episode = 1
        self.lifted = False
        self.success = False
        self.applied: list[np.ndarray] = []
        self.closed = False
        self.on_step = lambda: None
        self.close_error: BaseException | None = None

    def render_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        return frame, frame.copy()

    def state(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    def canonical_state(self) -> np.ndarray:
        return np.zeros((1, 10), dtype=np.float32)

    def apply_action(self, action: np.ndarray) -> None:
        self.applied.append(action)

    def step_control_period(self, frequency: float) -> None:
        self.data.time += 1.0 / frequency
        self.on_step()

    def reset(self) -> None:
        self.episode += 1
        self.data.time = 0.0

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def test_chunk_executor_does_not_replace_active_actions() -> None:
    executor = ChunkExecutor(horizon=4, prefetch_steps=1)
    first = np.arange(8)[:, None]
    second = np.arange(100, 108)[:, None]
    executor.mark_requested()
    executor.accept(first)
    assert executor.pop().item() == 0
    executor.mark_requested()
    executor.accept(second)
    assert [executor.pop().item() for _ in range(3)] == [1, 2, 3]
    assert executor.pop().item() == 100


def test_chunk_executor_requests_prefetch_near_horizon_end() -> None:
    executor = ChunkExecutor(horizon=4, prefetch_steps=1)
    executor.accept(np.arange(4)[:, None])
    assert not executor.needs_request()
    for _ in range(3):
        executor.pop()
    assert executor.needs_request()


@pytest.mark.parametrize("horizon", [0, -1])
def test_chunk_executor_rejects_nonpositive_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        ChunkExecutor(horizon)


def test_chunk_executor_rejects_negative_prefetch() -> None:
    with pytest.raises(ValueError, match="prefetch"):
        ChunkExecutor(4, prefetch_steps=-1)


@pytest.mark.parametrize(
    "chunk",
    [
        np.array([], dtype=np.float32),
        np.empty((0, 6), dtype=np.float32),
        np.full((1, 6), np.nan, dtype=np.float32),
    ],
)
def test_chunk_executor_rejects_malformed_chunks(chunk: np.ndarray) -> None:
    with pytest.raises(ValueError, match="action chunks"):
        ChunkExecutor(4).accept(chunk)


def test_chunk_executor_clear_discards_all_pending_work() -> None:
    executor = ChunkExecutor(2)
    executor.accept(np.arange(2)[:, None])
    executor.accept(np.arange(10, 12)[:, None])
    executor.mark_requested()

    executor.clear()

    assert executor.pop() is None
    assert executor.needs_request()


def test_replace_keeps_only_latest_queue_value() -> None:
    queue = runtime_module.Queue(maxsize=1)
    queue.put_nowait("old")

    SimulationRuntime._replace(queue, "new")

    assert queue.get_nowait() == "new"


def test_reset_queue_rejects_excess_requests() -> None:
    runtime = SimulationRuntime(Path("unused"), device="cpu", load_policy=False)
    for _ in range(4):
        with pytest.raises(TimeoutError, match="reset timed out"):
            runtime.request_reset(timeout=0)

    with pytest.raises(RuntimeError, match="reset queue is full"):
        runtime.request_reset(timeout=0)


def test_timed_out_reset_is_not_executed_later(monkeypatch) -> None:
    runtime = SimulationRuntime(Path("unused"), device="cpu", load_policy=False)
    environment = _FakeEnvironment()
    environment.on_step = runtime.stop_event.set
    monkeypatch.setattr(runtime_module, "SO101PickPlaceEnv", lambda **_: environment)
    monkeypatch.setattr(runtime, "_publish_frame", lambda *args: None)
    with pytest.raises(TimeoutError, match="reset timed out"):
        runtime.request_reset(timeout=0)

    runtime._simulation_loop()

    assert environment.episode == 1


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("frequency", 0.0, "frequency"),
        ("frequency", float("nan"), "frequency"),
        ("max_episode_seconds", 0.0, "episode"),
        ("jpeg_quality", 0, "JPEG"),
        ("jpeg_quality", 96, "JPEG"),
        ("execution_horizon", 0, "horizon"),
    ],
)
def test_runtime_rejects_invalid_boundaries(argument: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SimulationRuntime(Path("unused"), load_policy=False, **{argument: value})


def test_simulation_loop_rejects_stale_action_telemetry(monkeypatch) -> None:
    runtime = SimulationRuntime(Path("unused"), device="cpu", load_policy=True)
    environment = _FakeEnvironment()
    environment.on_step = runtime.stop_event.set
    published_inference: list[float] = []
    monkeypatch.setattr(runtime_module, "SO101PickPlaceEnv", lambda **_: environment)
    monkeypatch.setattr(
        runtime,
        "_publish_frame",
        lambda top, wrist, env, inference_ms: published_inference.append(inference_ms),
    )
    runtime.action_chunks.put_nowait((1, np.ones((2, 6), dtype=np.float32), 123.0))

    runtime._simulation_loop()

    assert environment.applied == []
    assert published_inference == [0.0, 0.0]
    assert environment.closed


def test_simulation_failure_stops_workers_and_closes_environment(monkeypatch) -> None:
    runtime = SimulationRuntime(Path("unused"), device="cpu", load_policy=False)
    environment = _FakeEnvironment()
    failure = RuntimeError("control failed")

    def fail() -> None:
        raise failure

    environment.on_step = fail
    monkeypatch.setattr(runtime_module, "SO101PickPlaceEnv", lambda **_: environment)
    monkeypatch.setattr(runtime, "_publish_frame", lambda *args: None)

    runtime._simulation_loop()

    assert runtime.error is failure
    assert runtime.stop_event.is_set()
    assert runtime.ready_event.is_set()
    assert environment.closed


def test_environment_close_failure_is_recorded(monkeypatch) -> None:
    runtime = SimulationRuntime(Path("unused"), device="cpu", load_policy=False)
    environment = _FakeEnvironment()
    failure = RuntimeError("close failed")
    environment.close_error = failure
    environment.on_step = runtime.stop_event.set
    monkeypatch.setattr(runtime_module, "SO101PickPlaceEnv", lambda **_: environment)
    monkeypatch.setattr(runtime, "_publish_frame", lambda *args: None)

    runtime._simulation_loop()

    assert runtime.error is failure
    assert runtime.stop_event.is_set()


def test_start_timeout_stops_simulation_worker(monkeypatch) -> None:
    runtime = SimulationRuntime(Path("unused"), device="cpu", load_policy=False)
    worker_started = threading.Event()

    def wait_for_stop() -> None:
        worker_started.set()
        runtime.stop_event.wait()

    monkeypatch.setattr(runtime, "_simulation_loop", wait_for_stop)

    with pytest.raises(TimeoutError, match="simulation"):
        runtime.start(timeout=0.01)

    assert worker_started.is_set()
    assert runtime.stop_event.is_set()
    assert runtime.sim_thread is not None and not runtime.sim_thread.is_alive()


def test_start_waits_for_policy_initialization(monkeypatch) -> None:
    runtime = SimulationRuntime(Path("unused"), device="cpu", load_policy=True)
    policy_started = threading.Event()
    release_policy = threading.Event()
    errors: list[BaseException] = []

    def simulation_worker() -> None:
        runtime.ready_event.set()
        runtime.stop_event.wait()

    def policy_worker() -> None:
        policy_started.set()
        release_policy.wait()
        runtime.policy_ready_event.set()

    def start() -> None:
        try:
            runtime.start(timeout=1.0)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(runtime, "_simulation_loop", simulation_worker)
    monkeypatch.setattr(runtime, "_policy_loop", policy_worker)
    starter = threading.Thread(target=start)
    starter.start()
    assert policy_started.wait(1.0)
    assert starter.is_alive()

    release_policy.set()
    starter.join(timeout=1.0)
    runtime.stop()

    assert not starter.is_alive()
    assert errors == []
