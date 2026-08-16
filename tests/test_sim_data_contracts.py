from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest

from embodi.sim import data as data_module
from embodi.sim import expert as expert_module


class FakeMujoco:
    mjtObj = SimpleNamespace(mjOBJ_SITE=0)

    @staticmethod
    def mj_name2id(_model: object, _object_type: int, _name: str) -> int:
        return 0


class FakeExpertEnv:
    def __init__(self) -> None:
        self.mujoco = FakeMujoco()
        self.model = SimpleNamespace(
            jnt_dofadr=np.arange(6),
            jnt_range=np.tile([-np.pi, np.pi], (6, 1)),
            nv=6,
        )
        self.joint_ids = np.arange(6)
        self.qpos_addresses = np.arange(6)
        self.cube_body_id = 0
        self.target_body_id = 1
        self.data = SimpleNamespace(
            site_xmat=np.eye(3).reshape(1, 9),
            site_xpos=np.array([[0.25, 0.0, 0.08]]),
            xpos=np.array([[0.30, 0.0, 0.018], [0.10, -0.18, 0.008]]),
            cvel=np.zeros((2, 6)),
            qpos=np.zeros(6),
        )

    @staticmethod
    def state() -> np.ndarray:
        return np.arange(6, dtype=np.float32)


def use_fake_ik(expert: expert_module.PrivilegedPickPlaceExpert, error: float) -> None:
    def fake_ik(
        _goal: np.ndarray,
        gripper: float,
        max_speed: float = 0.10,
        preserve_tilt: bool = True,
    ) -> np.ndarray:
        del max_speed, preserve_tilt
        expert.last_error = error
        return np.full(6, gripper, dtype=np.float32)

    expert._ik_action = fake_ik  # type: ignore[method-assign]


def test_expert_completes_the_successful_phase_sequence() -> None:
    env = FakeExpertEnv()
    expert = expert_module.PrivilegedPickPlaceExpert(env)  # type: ignore[arg-type]
    use_fake_ik(expert, error=0.001)

    expert.phase_tick = 18
    open_action = expert.action()
    assert expert.phase == expert_module.Phase.PREGRASP
    assert open_action[5] == 16.0

    expected_phases = [expert_module.Phase.DESCEND, expert_module.Phase.CLOSE]
    for expected_phase in expected_phases:
        expert.action()
        assert expert.phase == expected_phase
    np.testing.assert_array_equal(expert.grasp_site, env.data.site_xpos[0])

    expert.phase_tick = 24
    expert.action()
    assert expert.phase == expert_module.Phase.LIFT

    env.data.xpos[env.cube_body_id, 2] = 0.07
    expert.action()
    assert expert.phase == expert_module.Phase.TRANSIT
    expert.action()
    assert expert.phase == expert_module.Phase.RELEASE
    np.testing.assert_array_equal(expert.release_site, env.data.site_xpos[0])

    expert.phase_tick = 30
    expert.action()
    assert expert.phase == expert_module.Phase.RETREAT

    target = env.data.xpos[env.target_body_id]
    env.data.xpos[env.cube_body_id] = target + [0.001, -0.001, 0.04]
    expert.action()
    assert expert.phase == expert_module.Phase.DONE
    assert expert.terminal
    assert expert.status().ticks_in_phase == 1


@pytest.mark.parametrize(
    ("phase", "phase_tick"),
    [
        (expert_module.Phase.PREGRASP, 121),
        (expert_module.Phase.DESCEND, 91),
        (expert_module.Phase.LIFT, 101),
        (expert_module.Phase.TRANSIT, 151),
        (expert_module.Phase.RETREAT, 76),
    ],
)
def test_expert_phase_timeouts_fail(phase: expert_module.Phase, phase_tick: int) -> None:
    env = FakeExpertEnv()
    expert = expert_module.PrivilegedPickPlaceExpert(env)  # type: ignore[arg-type]
    use_fake_ik(expert, error=1.0)
    expert.phase = phase
    expert.phase_tick = phase_tick

    expert.action()

    assert expert.phase == expert_module.Phase.FAILED
    assert expert.terminal
    assert expert.phase_tick == 1


def test_expert_global_timeout_overrides_a_nonterminal_phase() -> None:
    expert = expert_module.PrivilegedPickPlaceExpert(FakeExpertEnv())  # type: ignore[arg-type]
    expert.total_tick = 750

    expert.action()

    assert expert.phase == expert_module.Phase.FAILED
    assert expert.total_tick == 751
    assert expert.phase_tick == 0


class FakeEpisodeEnv:
    def __init__(self) -> None:
        self.applied: list[np.ndarray] = []
        self.steps: list[float] = []
        self.render_calls = 0

    def state(self) -> np.ndarray:
        return np.array([len(self.applied)] * 6, dtype=np.float32)

    def render_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        self.render_calls += 1
        return np.full((2, 2, 3), 7, dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)

    def apply_action(self, action: np.ndarray) -> None:
        self.applied.append(action.copy())

    def step_control_period(self, frequency: float) -> None:
        self.steps.append(frequency)


@pytest.mark.parametrize(
    ("terminal_phase", "capture"),
    [(expert_module.Phase.DONE, True), (expert_module.Phase.FAILED, False)],
)
def test_run_expert_episode_reports_terminal_outcome_and_capture(
    monkeypatch: pytest.MonkeyPatch,
    terminal_phase: expert_module.Phase,
    capture: bool,
) -> None:
    class FakeEpisodeExpert:
        def __init__(self, _env: object) -> None:
            self.phase = expert_module.Phase.OPEN

        @property
        def terminal(self) -> bool:
            return self.phase in (expert_module.Phase.DONE, expert_module.Phase.FAILED)

        def action(self) -> np.ndarray:
            self.phase = terminal_phase
            return np.arange(6, dtype=np.float32)

    monkeypatch.setattr(expert_module, "PrivilegedPickPlaceExpert", FakeEpisodeExpert)
    env = FakeEpisodeEnv()

    success, frames = expert_module.run_expert_episode(env, max_steps=5, capture=capture)  # type: ignore[arg-type]

    assert success is (terminal_phase == expert_module.Phase.DONE)
    assert len(env.applied) == len(env.steps) == 1
    assert env.steps == [30.0]
    assert env.render_calls == int(capture)
    assert len(frames) == int(capture)
    if capture:
        np.testing.assert_array_equal(frames[0]["state"], np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(frames[0]["action"], np.arange(6, dtype=np.float32))
        assert frames[0]["image"].shape == (2, 2, 3)


class FakeProgress:
    close_error: BaseException | None = None

    def __init__(self, iterable: object = (), **_kwargs: object) -> None:
        self.iterable = iterable
        self.postfixes: list[dict[str, object]] = []
        self.updates = 0
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        return iter(self.iterable)

    def set_postfix(self, **values: object) -> None:
        self.postfixes.append(values)

    def update(self) -> None:
        self.updates += 1

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def test_evaluate_oracle_aggregates_successes_and_closes_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter([True, False, True])
    progress_objects: list[FakeProgress] = []

    class FakeEvaluationEnv:
        instance: FakeEvaluationEnv

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.resets = 0
            self.closed = False
            FakeEvaluationEnv.instance = self

        def reset(self) -> None:
            self.resets += 1

        def close(self) -> None:
            self.closed = True

    def fake_tqdm(iterable: object = (), **kwargs: object) -> FakeProgress:
        progress = FakeProgress(iterable, **kwargs)
        progress_objects.append(progress)
        return progress

    monkeypatch.setattr(data_module, "SO101PickPlaceEnv", FakeEvaluationEnv)
    monkeypatch.setattr(data_module, "run_expert_episode", lambda _env: (next(outcomes), []))
    monkeypatch.setattr(data_module, "tqdm", fake_tqdm)

    rate = data_module.evaluate_oracle(3, seed=17, cube_x_range=(0.2, 0.3))

    assert rate == pytest.approx(2 / 3)
    assert FakeEvaluationEnv.instance.kwargs == {
        "width": 64,
        "height": 48,
        "seed": 17,
        "cube_x_range": (0.2, 0.3),
        "cube_y_range": (-0.025, 0.025),
    }
    assert FakeEvaluationEnv.instance.resets == 3
    assert FakeEvaluationEnv.instance.closed
    assert [item["success_rate"] for item in progress_objects[0].postfixes] == [
        "100.0%",
        "50.0%",
        "66.7%",
    ]


def test_evaluate_oracle_closes_environment_on_episode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEvaluationEnv:
        instance: FakeEvaluationEnv

        def __init__(self, **_kwargs: object) -> None:
            self.closed = False
            FakeEvaluationEnv.instance = self

        def reset(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    def fail_episode(_env: object) -> tuple[bool, list[object]]:
        raise RuntimeError("episode failed")

    monkeypatch.setattr(data_module, "SO101PickPlaceEnv", FakeEvaluationEnv)
    monkeypatch.setattr(data_module, "run_expert_episode", fail_episode)
    monkeypatch.setattr(data_module, "tqdm", FakeProgress)

    with pytest.raises(RuntimeError, match="episode failed"):
        data_module.evaluate_oracle(1, seed=0)

    assert FakeEvaluationEnv.instance.closed


class FakeDataset:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, object] = {}
        self.buffer: list[dict[str, object]] = []
        self.all_frames: list[dict[str, object]] = []
        self.saved_episodes: list[list[dict[str, object]]] = []
        self.clear_sizes: list[int] = []
        self.events: list[str] = []
        self.finalized = False
        self.add_error: BaseException | None = None
        self.finalize_error: BaseException | None = None

    def add_frame(self, frame: dict[str, object]) -> None:
        self.buffer.append(frame)
        self.all_frames.append(frame)
        if self.add_error is not None:
            raise self.add_error

    def save_episode(self, *, parallel_encoding: bool) -> None:
        assert not parallel_encoding
        self.saved_episodes.append(self.buffer.copy())
        self.buffer.clear()

    def clear_episode_buffer(self) -> None:
        self.events.append("clear")
        self.clear_sizes.append(len(self.buffer))
        self.buffer.clear()

    def finalize(self) -> None:
        self.events.append("finalize")
        self.finalized = True
        if self.finalize_error is not None:
            raise self.finalize_error


class FakeGenerationEnv:
    instance: FakeGenerationEnv
    close_error: BaseException | None = None
    step_error: BaseException | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.cube_x_range = kwargs["cube_x_range"]
        self.cube_y_range = kwargs["cube_y_range"]
        self.cube_body_id = 0
        self.data = SimpleNamespace(xpos=np.array([[0.0, 0.0, 0.018]]))
        self.reset_ranges: list[tuple[float, float]] = []
        self.render_top_calls = 0
        self.render_wrist_calls = 0
        self.render_both_calls = 0
        self.closed = False
        FakeGenerationEnv.instance = self

    def reset(self) -> None:
        self.reset_ranges.append(self.cube_x_range)
        self.data.xpos[0, :2] = [sum(self.cube_x_range) / 2, self.cube_y_range[0]]

    @staticmethod
    def state() -> np.ndarray:
        return np.arange(6, dtype=np.float64)

    @staticmethod
    def canonical_arm_action(_action: np.ndarray) -> np.ndarray:
        return np.arange(10, dtype=np.float32).reshape(1, 10)

    @staticmethod
    def canonical_state() -> np.ndarray:
        return np.arange(10, dtype=np.float32).reshape(1, 10)

    def render_top(self) -> np.ndarray:
        self.render_top_calls += 1
        return np.full((2, 3, 3), 1, dtype=np.uint8)

    def render_wrist(self) -> np.ndarray:
        self.render_wrist_calls += 1
        return np.full((2, 3, 3), 2, dtype=np.uint8)

    def render_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        self.render_both_calls += 1
        top = np.full((2, 3, 3), 1, dtype=np.uint8)
        wrist = np.full((2, 3, 3), 2, dtype=np.uint8)
        return top, wrist

    @staticmethod
    def apply_action(_action: np.ndarray) -> None:
        pass

    @classmethod
    def step_control_period(cls, _frequency: float) -> None:
        if cls.step_error is not None:
            raise cls.step_error

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def install_fake_lerobot(monkeypatch: pytest.MonkeyPatch, dataset: FakeDataset) -> None:
    class FakeLeRobotDataset:
        @staticmethod
        def create(**kwargs: object) -> FakeDataset:
            dataset.create_kwargs = kwargs
            return dataset

    lerobot = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_module)


def install_generation_fakes(
    monkeypatch: pytest.MonkeyPatch,
    dataset: FakeDataset,
    outcomes: list[expert_module.Phase],
) -> list[FakeProgress]:
    phases = iter(outcomes)
    progress_objects: list[FakeProgress] = []

    class FakeGenerationExpert:
        def __init__(self, _env: object) -> None:
            self.phase = next(phases)

        @property
        def terminal(self) -> bool:
            return True

        @staticmethod
        def action() -> np.ndarray:
            return np.arange(6, dtype=np.float64)

    def fake_tqdm(iterable: object = (), **kwargs: object) -> FakeProgress:
        progress = FakeProgress(iterable, **kwargs)
        progress_objects.append(progress)
        return progress

    install_fake_lerobot(monkeypatch, dataset)
    FakeProgress.close_error = None
    FakeGenerationEnv.close_error = None
    FakeGenerationEnv.step_error = None
    monkeypatch.setattr(data_module, "SO101PickPlaceEnv", FakeGenerationEnv)
    monkeypatch.setattr(data_module, "PrivilegedPickPlaceExpert", FakeGenerationExpert)
    monkeypatch.setattr(data_module, "tqdm", fake_tqdm)
    return progress_objects


@pytest.mark.parametrize(
    ("camera_names", "render_counts"),
    [
        (("top",), (2, 0, 0)),
        (("wrist",), (0, 2, 0)),
        (("top", "wrist"), (0, 0, 2)),
    ],
)
def test_generate_dataset_clears_failures_saves_success_and_routes_cameras(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    camera_names: tuple[str, ...],
    render_counts: tuple[int, int, int],
) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    dataset = FakeDataset()
    progress_objects = install_generation_fakes(
        monkeypatch,
        dataset,
        [expert_module.Phase.FAILED, expert_module.Phase.DONE],
    )

    data_module.generate_dataset(
        root=root,
        repo_id="local/fake",
        episodes=1,
        seed=4,
        max_attempts=2,
        camera_names=camera_names,
        cube_x_range=(0.2, 0.4),
        cube_y_range=(-0.1, 0.1),
    )

    env = FakeGenerationEnv.instance
    assert dataset.clear_sizes == [1]
    assert len(dataset.saved_episodes) == 1
    assert len(dataset.saved_episodes[0]) == 1
    assert dataset.finalized
    assert not dataset.buffer
    assert env.closed
    assert progress_objects[0].closed
    assert progress_objects[0].updates == 1
    assert (env.render_top_calls, env.render_wrist_calls, env.render_both_calls) == render_counts

    frame = dataset.saved_episodes[0][0]
    assert {key for key in frame if key.startswith("observation.images.")} == {
        f"observation.images.{name}" for name in camera_names
    }
    state = frame["observation.state"]
    action = frame["action"]
    assert isinstance(state, np.ndarray) and state.dtype == np.float32
    assert isinstance(action, np.ndarray) and action.dtype == np.float32
    np.testing.assert_array_equal(frame["canonical_part_mask"], [True])
    assert dataset.create_kwargs["use_videos"] is True
    assert set(dataset.create_kwargs["features"]) >= {
        "observation.state",
        "action",
        "canonical_action",
        "canonical_state",
        "canonical_part_mask",
    }

    schema = json.loads((root / "meta" / "embodi.json").read_text())
    report = json.loads((root / "meta" / "generation.json").read_text())
    assert schema["format_version"] == 3
    assert report["attempts"] == 2
    assert report["camera_names"] == list(camera_names)
    np.testing.assert_allclose(report["accepted_initial_cube_xy"], [[0.3, -0.1]])


def test_generate_dataset_clears_partial_buffer_and_preserves_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    dataset = FakeDataset()
    dataset.finalize_error = RuntimeError("finalize failed")
    progress_objects = install_generation_fakes(
        monkeypatch,
        dataset,
        [expert_module.Phase.DONE],
    )
    FakeGenerationEnv.step_error = RuntimeError("control failed")

    with pytest.raises(RuntimeError, match="control failed"):
        data_module.generate_dataset(root, "local/fake", 1, 0, 1, ("top",))

    assert dataset.clear_sizes == [1]
    assert not dataset.buffer
    assert dataset.finalized
    assert dataset.events == ["clear", "finalize"]
    assert FakeGenerationEnv.instance.closed
    assert progress_objects[0].closed
    assert not (root / "meta" / "generation.json").exists()


def test_generate_dataset_clears_when_add_frame_mutates_then_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    dataset = FakeDataset()
    dataset.add_error = RuntimeError("frame write failed after mutation")
    progress_objects = install_generation_fakes(
        monkeypatch,
        dataset,
        [expert_module.Phase.DONE],
    )

    with pytest.raises(RuntimeError, match="frame write failed after mutation"):
        data_module.generate_dataset(root, "local/fake", 1, 0, 1, ("top",))

    assert len(dataset.all_frames) == 1
    assert dataset.clear_sizes == [1]
    assert not dataset.buffer
    assert dataset.events == ["clear", "finalize"]
    assert dataset.finalized
    assert FakeGenerationEnv.instance.closed
    assert progress_objects[0].closed


@pytest.mark.parametrize("cleanup", ["progress", "environment", "dataset"])
def test_generate_dataset_propagates_successful_path_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cleanup: str,
) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    dataset = FakeDataset()
    progress_objects = install_generation_fakes(
        monkeypatch,
        dataset,
        [expert_module.Phase.DONE],
    )
    error = RuntimeError(f"{cleanup} cleanup failed")
    if cleanup == "progress":
        FakeProgress.close_error = error
    elif cleanup == "environment":
        FakeGenerationEnv.close_error = error
    else:
        dataset.finalize_error = error

    with pytest.raises(RuntimeError, match=f"{cleanup} cleanup failed"):
        data_module.generate_dataset(root, "local/fake", 1, 0, 1, ("top",))

    assert len(dataset.saved_episodes) == 1
    assert not dataset.buffer
    assert progress_objects[0].closed
    assert FakeGenerationEnv.instance.closed
    assert dataset.finalized


def test_generate_dataset_finalizes_after_environment_construction_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    dataset = FakeDataset()
    dataset.finalize_error = RuntimeError("finalize failed")
    close_calls = 0

    class FailingEnvironment:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("environment construction failed")

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    install_fake_lerobot(monkeypatch, dataset)
    monkeypatch.setattr(data_module, "SO101PickPlaceEnv", FailingEnvironment)

    with pytest.raises(RuntimeError, match="environment construction failed"):
        data_module.generate_dataset(root, "local/fake", 1, 0, 1, ("top",))

    assert dataset.finalized
    assert close_calls == 0
