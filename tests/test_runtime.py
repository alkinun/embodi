import numpy as np

from embodi.sim.runtime import ChunkExecutor


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
