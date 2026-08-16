import asyncio
from collections.abc import Callable

from fastapi import HTTPException
import pytest

from embodi.sim import server


class FakeRuntime:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.events: list[str] = []
        self.state = {"episode": 3, "success": False}
        self.reset_result = {"episode": 4}
        self.reset_error: Exception | None = None
        self.frames: list[tuple[int, bytes | None]] = []

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def status(self):
        self.events.append("status")
        return self.state

    def request_reset(self):
        self.events.append("reset")
        if self.reset_error is not None:
            raise self.reset_error
        return self.reset_result

    def frame(self):
        return self.frames.pop(0)


def route_endpoint(app, path: str, method: str = "GET") -> Callable:
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def test_app_disables_docs_and_exposes_index_and_health() -> None:
    app = server.create_app(FakeRuntime())

    assert app.docs_url is None
    assert app.redoc_url is None
    assert asyncio.run(route_endpoint(app, "/")()) == server.PAGE
    assert asyncio.run(route_endpoint(app, "/healthz")()) == {"ok": True}


def test_lifespan_starts_and_stops_runtime() -> None:
    runtime = FakeRuntime()
    app = server.create_app(runtime)

    async def exercise_lifespan() -> None:
        async with app.router.lifespan_context(app):
            assert runtime.events == ["start"]

    asyncio.run(exercise_lifespan())

    assert runtime.events == ["start", "stop"]


def test_lifespan_stops_runtime_when_request_loop_fails() -> None:
    runtime = FakeRuntime()
    app = server.create_app(runtime)

    async def exercise_lifespan() -> None:
        with pytest.raises(RuntimeError, match="request loop failed"):
            async with app.router.lifespan_context(app):
                raise RuntimeError("request loop failed")

    asyncio.run(exercise_lifespan())

    assert runtime.events == ["start", "stop"]


def test_status_returns_runtime_state() -> None:
    runtime = FakeRuntime()
    endpoint = route_endpoint(server.create_app(runtime), "/api/status")

    assert asyncio.run(endpoint()) is runtime.state
    assert runtime.events == ["status"]


def test_status_translates_runtime_error_to_service_unavailable() -> None:
    runtime = FakeRuntime()
    runtime.error = RuntimeError("simulation failed")
    endpoint = route_endpoint(server.create_app(runtime), "/api/status")

    with pytest.raises(HTTPException) as caught:
        asyncio.run(endpoint())

    assert caught.value.status_code == 503
    assert caught.value.detail == "simulation failed"
    assert runtime.events == ["status"]


def test_reset_runs_request_and_returns_result_without_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime()
    to_thread_calls: list[Callable] = []

    async def fake_to_thread(function):
        to_thread_calls.append(function)
        return function()

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    endpoint = route_endpoint(server.create_app(runtime), "/api/reset", "POST")

    assert asyncio.run(endpoint()) is runtime.reset_result
    assert runtime.events == ["reset"]
    assert to_thread_calls == [runtime.request_reset]


@pytest.mark.parametrize("error", [RuntimeError("broken"), TimeoutError("too slow")])
def test_reset_translates_runtime_failures_to_service_unavailable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    runtime = FakeRuntime()
    runtime.reset_error = error
    to_thread_calls: list[Callable] = []

    async def fake_to_thread(function):
        to_thread_calls.append(function)
        return function()

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    endpoint = route_endpoint(server.create_app(runtime), "/api/reset", "POST")

    with pytest.raises(HTTPException) as caught:
        asyncio.run(endpoint())

    assert caught.value.status_code == 503
    assert caught.value.detail == str(error)
    assert caught.value.__cause__ is error
    assert to_thread_calls == [runtime.request_reset]


def test_stream_emits_only_new_nonempty_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    runtime.frames = [(0, b"jpeg-a"), (0, b"jpeg-a"), (1, None), (2, b"jpeg-b")]
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)
    endpoint = route_endpoint(server.create_app(runtime), "/stream.mjpg")

    async def read_frames():
        response = await endpoint()
        iterator = response.body_iterator
        try:
            first = await anext(iterator)
            second = await anext(iterator)
        finally:
            await iterator.aclose()
        return response, first, second

    response, first, second = asyncio.run(read_frames())

    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"
    assert response.headers["cache-control"] == "no-store, no-cache"
    assert response.headers["pragma"] == "no-cache"
    assert first == (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 6\r\n\r\n"
        b"jpeg-a\r\n"
    )
    assert second == (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 6\r\n\r\n"
        b"jpeg-b\r\n"
    )
    assert sleep_calls == [0.02, 0.02, 0.02]
