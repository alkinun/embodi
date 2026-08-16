import argparse
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from embodi.sim import cli


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["embodi-sim"])

    args = cli.parse_args()

    assert vars(args) == {
        "checkpoint": Path("outputs/det-peak-exp20-center-m3993-l3992/step-0001100"),
        "host": "127.0.0.1",
        "port": 8080,
        "device": "cuda",
        "frequency": 30.0,
        "execution_horizon": 16,
        "episode_seconds": 30.0,
        "asset_dir": None,
        "no_policy": False,
        "smoke_test": False,
    }


def test_parse_args_converts_all_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embodi-sim",
            "--checkpoint",
            "/models/step-1",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--device",
            "cpu",
            "--frequency",
            "15.5",
            "--execution-horizon",
            "8",
            "--episode-seconds",
            "12.25",
            "--asset-dir",
            "/assets",
            "--no-policy",
            "--smoke-test",
        ],
    )

    args = cli.parse_args()

    assert vars(args) == {
        "checkpoint": Path("/models/step-1"),
        "host": "0.0.0.0",
        "port": 9000,
        "device": "cpu",
        "frequency": 15.5,
        "execution_horizon": 8,
        "episode_seconds": 12.25,
        "asset_dir": Path("/assets"),
        "no_policy": True,
        "smoke_test": True,
    }


def smoke_args(**overrides: object) -> argparse.Namespace:
    values = {
        "smoke_test": True,
        "asset_dir": Path("assets"),
        "frequency": 20.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_smoke_test_renders_steps_reports_shapes_and_closes_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[object] = []

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            events.append(("init", kwargs))

        def render_cameras(self):
            events.append("render")
            return SimpleNamespace(shape=(240, 320, 3)), SimpleNamespace(shape=(240, 320, 3))

        def step_control_period(self, frequency: float) -> None:
            events.append(("step", frequency))

        def state(self):
            events.append("state")
            return SimpleNamespace(shape=(6,))

        def close(self) -> None:
            events.append("close")

    environment = ModuleType("embodi.sim.environment")
    environment.SO101PickPlaceEnv = FakeEnvironment
    monkeypatch.setitem(sys.modules, "embodi.sim.environment", environment)
    monkeypatch.setattr(cli, "parse_args", smoke_args)
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    cli.main()

    assert events == [
        ("init", {"width": 320, "height": 240, "asset_dir": Path("assets")}),
        "render",
        ("step", 20.0),
        ("step", 20.0),
        ("step", 20.0),
        "state",
        "close",
    ]
    assert capsys.readouterr().out == (
        "state=(6,) top=(240, 320, 3) wrist=(240, 320, 3)\n"
    )
    assert cli.os.environ["MUJOCO_GL"] == "egl"


def test_smoke_test_closes_environment_when_step_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeEnvironment:
        def __init__(self, **_: object) -> None:
            events.append("init")

        def render_cameras(self):
            events.append("render")
            return object(), object()

        def step_control_period(self, _: float) -> None:
            events.append("step")
            raise RuntimeError("step failed")

        def close(self) -> None:
            events.append("close")

    environment = ModuleType("embodi.sim.environment")
    environment.SO101PickPlaceEnv = FakeEnvironment
    monkeypatch.setitem(sys.modules, "embodi.sim.environment", environment)
    monkeypatch.setattr(cli, "parse_args", smoke_args)

    with pytest.raises(RuntimeError, match="step failed"):
        cli.main()

    assert events == ["init", "render", "step", "close"]


def test_server_mode_builds_runtime_and_runs_single_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        smoke_test=False,
        checkpoint=Path("checkpoint"),
        host="localhost",
        port=8123,
        device="cpu",
        frequency=12.0,
        execution_horizon=5,
        episode_seconds=9.0,
        asset_dir=Path("assets"),
        no_policy=True,
    )
    runtime_calls: list[dict[str, object]] = []
    run_calls: list[tuple[object, dict[str, object]]] = []
    app = object()

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            runtime_calls.append(kwargs)

    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda value, **kwargs: run_calls.append((value, kwargs))
    runtime_module = ModuleType("embodi.sim.runtime")
    runtime_module.SimulationRuntime = FakeRuntime
    server_module = ModuleType("embodi.sim.server")
    server_module.create_app = lambda runtime: app
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setitem(sys.modules, "embodi.sim.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "embodi.sim.server", server_module)
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    cli.main()

    assert runtime_calls == [
        {
            "checkpoint": Path("checkpoint"),
            "device": "cpu",
            "frequency": 12.0,
            "max_episode_seconds": 9.0,
            "asset_dir": Path("assets"),
            "load_policy": False,
            "execution_horizon": 5,
        }
    ]
    assert run_calls == [
        (app, {"host": "localhost", "port": 8123, "workers": 1})
    ]
    assert cli.os.environ["MUJOCO_GL"] == "osmesa"
