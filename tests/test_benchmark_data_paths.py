from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodi.sim import benchmark_data
from embodi.sim.benchmark import Scenario
from embodi.sim.benchmark_data import (
    _contract,
    _features_equal,
    generate_benchmark_dataset,
    _load_or_create_generation_manifest,
    _reconcile_progress,
    benchmark_features,
)
from embodi.sim.morphology import SO101


def test_features_equal_requires_complete_default_and_benchmark_schema() -> None:
    expected = {
        **benchmark_features(SO101, width=64, height=48),
        **benchmark_data.LEROBOT_DEFAULT_FEATURES,
    }
    assert _features_equal(expected, benchmark_features(SO101, width=64, height=48))

    missing_default = deepcopy(expected)
    missing_default.pop("timestamp")
    assert not _features_equal(missing_default, benchmark_features(SO101, width=64, height=48))

    extra = deepcopy(expected)
    extra["unexpected"] = {"dtype": "float32", "shape": (1,), "names": None}
    assert not _features_equal(extra, benchmark_features(SO101, width=64, height=48))

    wrong_image = deepcopy(expected)
    wrong_image["observation.images.top"]["shape"] = (48, 64, 1)
    assert not _features_equal(wrong_image, benchmark_features(SO101, width=64, height=48))

    wrong_default = deepcopy(expected)
    wrong_default["timestamp"]["names"] = ["timestamp"]
    assert not _features_equal(wrong_default, benchmark_features(SO101, width=64, height=48))


def _contract_inputs(tmp_path: Path, *, split: str = "train", frequency_hz: int = 30):
    definition_path = tmp_path / "definition.json"
    manifest_path = tmp_path / "manifest.json"
    definition_path.write_text("definition")
    manifest_path.write_text("manifest")
    scenario = Scenario(
        scenario_id="train-so101-push_to_zone-0000",
        split=split,
        morphology="so101",
        task="push_to_zone",
        seed=1,
        factors={},
    )
    definition = SimpleNamespace(
        path=definition_path,
        sha256="definition-sha",
        values={
            "benchmark_id": "embodi-sim-v1",
            "control": {
                "frequency_hz": frequency_hz,
                "camera_names": ["top"],
                "maximum_episode_steps": 500,
            },
        },
    )
    manifest = SimpleNamespace(path=manifest_path, scenarios=(scenario,))
    return definition, manifest, {"so101": (scenario,)}


def test_contract_binds_split_software_and_output_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition, manifest, selected = _contract_inputs(tmp_path)
    monkeypatch.setattr(benchmark_data, "_validate_frozen_manifest_hash", lambda *args: None)
    monkeypatch.setattr(benchmark_data, "_implementation_hashes", lambda: {"benchmark_data.py": "sha"})
    monkeypatch.setattr(benchmark_data, "version", lambda package: "3.3.0")

    contract = _contract(
        definition,
        manifest,
        selected,
        repo_prefix="embodi/train",
        width=64,
        height=48,
        limit_per_cell=1,
        image_writer_threads=2,
    )

    assert contract["split"] == "train"
    assert contract["software"] == {
        "generator": benchmark_data.GENERATOR_FORMAT,
        "lerobot": benchmark_data.LEROBOT_VERSION,
        "mujoco": "3.3.0",
        "implementation_sha256": {"benchmark_data.py": "sha"},
    }
    assert contract["outputs"] == [
        {
            "morphology": "so101",
            "repo_id": "embodi/train-so101",
            "relative_root": "so101",
            "scenario_ids": ["train-so101-push_to_zone-0000"],
        }
    ]


@pytest.mark.parametrize(
    ("split", "frequency_hz", "repo_prefix", "message"),
    [
        ("development", 30, "embodi/train", "only from train or validation"),
        ("train", 60, "embodi/train", "frozen 30 Hz"),
        ("train", 30, "embodi/train-", "must not end with a hyphen"),
    ],
)
def test_contract_rejects_unregistered_generation_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split: str,
    frequency_hz: int,
    repo_prefix: str,
    message: str,
) -> None:
    definition, manifest, selected = _contract_inputs(
        tmp_path,
        split=split,
        frequency_hz=frequency_hz,
    )
    monkeypatch.setattr(benchmark_data, "_validate_frozen_manifest_hash", lambda *args: None)

    with pytest.raises(ValueError, match=message):
        _contract(
            definition,
            manifest,
            selected,
            repo_prefix=repo_prefix,
            width=64,
            height=48,
            limit_per_cell=None,
            image_writer_threads=0,
        )


def test_generation_manifest_rejects_existing_root_without_resume(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    contract = {"outputs": [{"morphology": "so101"}]}

    with pytest.raises(FileExistsError, match="already exists"):
        _load_or_create_generation_manifest(root, contract, resume=False)


def test_reconcile_progress_accepts_matching_completed_episode_and_updates_pending() -> None:
    scenarios = (
        Scenario("scenario-0", "train", "so101", "push_to_zone", 1, {}),
    )
    progress = {"episodes": 0, "frames": 3, "commands": 3, "clipped_commands": 1}
    pending = {
        "scenario_id": "scenario-0",
        "frames": 2,
        "commands": 2,
        "clipped_commands": 0,
    }
    dataset = SimpleNamespace(num_episodes=1, num_frames=5)

    _reconcile_progress(dataset, scenarios, progress, pending)

    assert progress == {"episodes": 1, "frames": 5, "commands": 5, "clipped_commands": 1}


@pytest.mark.parametrize(
    ("num_episodes", "num_frames", "pending", "message"),
    [
        (0, 4, None, "frame count"),
        (2, 5, None, "valid completed scenario prefix"),
        (1, 5, None, "pending frozen scenario"),
        (1, 4, {"scenario_id": "wrong", "frames": 1, "commands": 1, "clipped_commands": 0}, "pending frozen scenario"),
    ],
)
def test_reconcile_progress_rejects_inconsistent_resume_state(
    num_episodes: int,
    num_frames: int,
    pending: dict | None,
    message: str,
) -> None:
    scenarios = (
        Scenario("scenario-0", "train", "so101", "push_to_zone", 1, {}),
    )
    progress = {"episodes": 0, "frames": 3, "commands": 3, "clipped_commands": 0}

    with pytest.raises(ValueError, match=message):
        _reconcile_progress(
            SimpleNamespace(num_episodes=num_episodes, num_frames=num_frames),
            scenarios,
            progress,
            pending,
        )


def test_generate_benchmark_dataset_runs_each_selected_scenario_and_finalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = Scenario(
        scenario_id="train-so101-push_to_zone-0000",
        split="train",
        morphology="so101",
        task="push_to_zone",
        seed=1,
        factors={},
    )
    definition = SimpleNamespace(
        morphology_ids=("so101",),
        values={"benchmark_id": "embodi-sim-v1"},
        sha256="definition-sha",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("manifest")
    manifest = SimpleNamespace(path=manifest_path, scenarios=(scenario,))
    contract = {
        "control": {
            "frequency_hz": 30.0,
            "maximum_episode_steps": 500,
        },
        "outputs": [
            {
                "morphology": "so101",
                "repo_id": "embodi/train-so101",
                "relative_root": "so101",
                "scenario_ids": [scenario.scenario_id],
            }
        ]
    }
    monkeypatch.setattr(benchmark_data.BenchmarkDefinition, "load", lambda path: definition)
    monkeypatch.setattr(benchmark_data.ScenarioManifest, "load", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(benchmark_data, "_contract", lambda *args, **kwargs: contract)
    monkeypatch.setattr(
        benchmark_data,
        "morphology_by_id",
        lambda morphology_id: SimpleNamespace(id=morphology_id),
    )
    calls: dict[str, list] = {"rollout": [], "metadata": [], "validation": []}

    class FakeDataset:
        num_episodes = 0
        num_frames = 0

        def save_episode(self, **kwargs) -> None:
            calls.setdefault("saved", []).append(kwargs)

        def finalize(self) -> None:
            calls.setdefault("finalized", []).append(True)

    dataset = FakeDataset()

    def fake_open_dataset(root, *args, **kwargs):
        root.mkdir(parents=True, exist_ok=True)
        return dataset

    monkeypatch.setattr(benchmark_data, "_open_dataset", fake_open_dataset)
    monkeypatch.setattr(
        benchmark_data,
        "rollout_scenario",
        lambda selected_scenario, **kwargs: (
            calls["rollout"].append(selected_scenario.scenario_id)
            or {"frames": 2, "commands": 2, "clipped_commands": 0}
        ),
    )
    monkeypatch.setattr(
        benchmark_data,
        "_write_dataset_metadata",
        lambda root, definition_value, manifest_value, morphology, scenarios: calls["metadata"].append(
            (root, morphology.id, [value.scenario_id for value in scenarios])
        ),
    )
    monkeypatch.setattr(
        benchmark_data,
        "_validate_lerobot_metadata",
        lambda root, repo_id, morphology, scenarios, **kwargs: calls["validation"].append(
            (root, repo_id, morphology.id, len(scenarios), kwargs["frames"])
        ),
    )
    monkeypatch.setattr(
        benchmark_data,
        "_finalize_generation_manifest",
        lambda root, value, definition_value: {"complete": True, "root": root},
    )

    root = tmp_path / "generated"
    result = generate_benchmark_dataset(
        tmp_path / "definition.json",
        tmp_path / "manifest.json",
        root,
        "embodi/train",
        width=64,
        height=48,
        limit_per_cell=1,
        image_writer_threads=0,
    )

    assert result["complete"] is True
    assert calls["rollout"] == [scenario.scenario_id]
    assert calls["metadata"] == [(root / "so101", "so101", [scenario.scenario_id])]
    assert calls["validation"] == [(root / "so101", "embodi/train-so101", "so101", 1, 2)]
    assert calls["saved"] == [{"parallel_encoding": False}]
    assert calls["finalized"] == [True]


@pytest.mark.parametrize(
    ("width", "height", "image_writer_threads"),
    [(0, 48, 0), (64, 0, 0), (64, 48, -1)],
)
def test_generate_benchmark_dataset_rejects_invalid_writer_contract(
    tmp_path: Path,
    width: int,
    height: int,
    image_writer_threads: int,
) -> None:
    with pytest.raises(ValueError, match="dimensions|writer threads"):
        generate_benchmark_dataset(
            tmp_path / "definition.json",
            tmp_path / "manifest.json",
            tmp_path / "generated",
            "embodi/train",
            width=width,
            height=height,
            image_writer_threads=image_writer_threads,
        )
