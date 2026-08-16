import json
from pathlib import Path

import numpy as np
import pytest

from embodi.sim import benchmark_data
from embodi.sim.benchmark import BenchmarkDefinition, ScenarioManifest
from embodi.sim.benchmark_data import (
    benchmark_features,
    rollout_scenario,
    select_scenarios,
    verify_benchmark_dataset,
)
from embodi.sim.morphology import PANDA, SO101


BENCHMARK_DIRECTORY = Path(__file__).parents[1] / "benchmarks" / "sim-v1"
DEFINITION_PATH = BENCHMARK_DIRECTORY / "definition.json"
TRAIN_MANIFEST_PATH = BENCHMARK_DIRECTORY / "manifests" / "train.json"
DEVELOPMENT_MANIFEST_PATH = BENCHMARK_DIRECTORY / "manifests" / "development.json"
pytestmark = pytest.mark.simulation


def _benchmark() -> tuple[BenchmarkDefinition, ScenarioManifest]:
    definition = BenchmarkDefinition.load(DEFINITION_PATH)
    manifest = ScenarioManifest.load(TRAIN_MANIFEST_PATH, definition, require_complete_counts=True)
    return definition, manifest


def test_benchmark_features_keep_native_morphologies_separate() -> None:
    so101 = benchmark_features(SO101, width=64, height=48)
    panda = benchmark_features(PANDA, width=64, height=48)
    assert so101["observation.state"]["shape"] == (6,)
    assert so101["action"]["names"] == list(SO101.native_action_names)
    assert panda["observation.state"]["shape"] == (8,)
    assert panda["action"]["names"] == list(PANDA.native_action_names)
    assert so101["canonical_action"] == panda["canonical_action"]


def test_scenario_selection_is_manifest_ordered_and_cell_stratified() -> None:
    definition, manifest = _benchmark()
    selected = select_scenarios(manifest, definition.morphology_ids, limit_per_cell=2)
    assert {key: len(value) for key, value in selected.items()} == {"so101": 6, "panda": 6}
    for morphology, scenarios in selected.items():
        assert [scenario.task for scenario in scenarios] == [
            "push_to_zone",
            "push_to_zone",
            "lift_object",
            "lift_object",
            "pick_place_bin",
            "pick_place_bin",
        ]
        assert all(scenario.morphology == morphology for scenario in scenarios)


def test_generation_manifest_resume_requires_exact_incomplete_contract(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    contract = {
        "outputs": [
            {"morphology": "so101"},
            {"morphology": "panda"},
        ]
    }
    created = benchmark_data._load_or_create_generation_manifest(root, contract, resume=False)
    assert created["complete"] is False
    assert benchmark_data._load_or_create_generation_manifest(root, contract, resume=True) == created
    with pytest.raises(ValueError, match="contract does not match"):
        benchmark_data._load_or_create_generation_manifest(
            root,
            {"outputs": [{"morphology": "so101"}]},
            resume=True,
        )
    created["complete"] = True
    (root / benchmark_data.GENERATION_MANIFEST_NAME).write_text(json.dumps(created))
    with pytest.raises(ValueError, match="immutable"):
        benchmark_data._load_or_create_generation_manifest(root, contract, resume=True)


def test_generation_rejects_development_evaluation_scenarios(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only from train or validation"):
        benchmark_data.generate_benchmark_dataset(
            DEFINITION_PATH,
            DEVELOPMENT_MANIFEST_PATH,
            tmp_path / "must-not-exist",
            "embodi/invalid-development-data",
            width=64,
            height=48,
            limit_per_cell=1,
            image_writer_threads=0,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_generation_rejects_modified_train_manifest(tmp_path: Path) -> None:
    values = json.loads(TRAIN_MANIFEST_PATH.read_text())
    values["scenarios"][0]["factors"]["mass"]["kg"] += 0.001
    modified_manifest = tmp_path / "train.json"
    modified_manifest.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="frozen split checksum"):
        benchmark_data.generate_benchmark_dataset(
            DEFINITION_PATH,
            modified_manifest,
            tmp_path / "must-not-exist",
            "embodi/modified-train-data",
            width=64,
            height=48,
            limit_per_cell=1,
            image_writer_threads=0,
        )


@pytest.mark.parametrize("morphology", ["so101", "panda"])
def test_manifest_scenario_rollout_emits_training_frame_contract(morphology: str) -> None:
    definition, manifest = _benchmark()
    scenario = next(value for value in manifest.scenarios if value.morphology == morphology)

    class FrameSink:
        def __init__(self):
            self.frames = 0
            self.cleared = False

        def add_frame(self, frame):
            assert frame["observation.images.top"].shape == (48, 64, 3)
            assert frame["observation.images.top"].dtype == np.uint8
            assert frame["observation.state"].shape == (
                6 if morphology == "so101" else 8,
            )
            assert frame["action"].shape == frame["observation.state"].shape
            assert frame["canonical_action"].shape == (1, 10)
            assert frame["canonical_state"].shape == (1, 10)
            assert frame["canonical_part_mask"].dtype == np.bool_
            assert frame["task"] == "Push the object into the target zone."
            self.frames += 1

        def clear_episode_buffer(self):
            self.cleared = True

    sink = FrameSink()
    result = rollout_scenario(
        scenario,
        dataset=sink,
        width=64,
        height=48,
        maximum_episode_steps=definition.values["control"]["maximum_episode_steps"],
    )
    assert result["frames"] == sink.frames
    assert result["commands"] == sink.frames
    assert result["clipped_commands"] == 0
    assert sink.cleared is False


def test_failed_frozen_scenario_is_cleared_and_not_substituted(monkeypatch) -> None:
    _definition, manifest = _benchmark()
    scenario = manifest.scenarios[0]

    class FailedEnvironment:
        instruction = "failed"
        success = False

        def __init__(self, *_args, **_kwargs):
            self.morphology = SO101

        def state(self):
            return np.zeros(6, dtype=np.float32)

        def canonical_arm_action(self, _action):
            return np.zeros((1, 10), dtype=np.float32)

        def canonical_state(self):
            return np.zeros((1, 10), dtype=np.float32)

        def qpos_to_native(self, value):
            return value

        def native_to_qpos(self, value):
            return value

        def render_top(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def apply_action(self, _action):
            pass

        def step_control_period(self, _frequency):
            pass

        def close(self):
            pass

    class FailedExpert:
        phase = benchmark_data.BenchmarkPhase.FAILED
        terminal = True

        def __init__(self, _env, frequency):
            self.frequency = frequency

        def action(self):
            return np.zeros(6, dtype=np.float32)

    class Sink:
        cleared = False

        def add_frame(self, _frame):
            pass

        def clear_episode_buffer(self):
            self.cleared = True

    sink = Sink()
    monkeypatch.setattr(benchmark_data, "PrivilegedBenchmarkExpert", FailedExpert)
    with pytest.raises(RuntimeError, match=scenario.scenario_id):
        rollout_scenario(
            scenario,
            dataset=sink,
            width=8,
            height=8,
            environment_factory=FailedEnvironment,
        )
    assert sink.cleared is True


def test_resume_reconciles_one_saved_episode_from_pending_identity() -> None:
    _definition, manifest = _benchmark()
    scenarios = (manifest.scenarios[0],)
    dataset = type("Dataset", (), {"num_episodes": 1, "num_frames": 17})()
    progress = {"episodes": 0, "frames": 0, "commands": 0, "clipped_commands": 0}
    benchmark_data._reconcile_progress(
        dataset,
        scenarios,
        progress,
        {
            "scenario_id": scenarios[0].scenario_id,
            "frames": 17,
            "commands": 17,
            "clipped_commands": 2,
        },
    )
    assert progress == {"episodes": 1, "frames": 17, "commands": 17, "clipped_commands": 2}


def test_resume_rejects_saved_episode_without_pending_scenario_identity() -> None:
    _definition, manifest = _benchmark()
    dataset = type("Dataset", (), {"num_episodes": 1, "num_frames": 17})()
    progress = {"episodes": 0, "frames": 0, "commands": 0, "clipped_commands": 0}
    with pytest.raises(ValueError, match="pending frozen scenario"):
        benchmark_data._reconcile_progress(dataset, (manifest.scenarios[0],), progress, None)


def test_real_lerobot_schema_can_be_resumed(tmp_path: Path) -> None:
    first = benchmark_data._open_dataset(
        tmp_path / "so101",
        "embodi/test-resume-so101",
        SO101,
        width=64,
        height=48,
        resume=False,
        image_writer_threads=0,
    )
    first.finalize()
    resumed = benchmark_data._open_dataset(
        tmp_path / "so101",
        "embodi/test-resume-so101",
        SO101,
        width=64,
        height=48,
        resume=True,
        image_writer_threads=0,
    )
    resumed.finalize()


def test_completed_dataset_rejects_non_lerobot_and_payload_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition, manifest = _benchmark()
    selected = select_scenarios(manifest, definition.morphology_ids, limit_per_cell=1)
    contract = benchmark_data._contract(
        definition,
        manifest,
        selected,
        repo_prefix="embodi/test-benchmark-data",
        width=64,
        height=48,
        limit_per_cell=1,
        image_writer_threads=0,
    )
    progress = {}
    outputs = []
    for morphology_id in definition.morphology_ids:
        morphology = benchmark_data.morphology_by_id(morphology_id)
        contract_output = next(
            output for output in contract["outputs"] if output["morphology"] == morphology_id
        )
        dataset_root = tmp_path / morphology_id
        dataset_root.mkdir()
        (dataset_root / "payload.bin").write_bytes(morphology_id.encode())
        counts = {"episodes": 3, "frames": 10, "commands": 10, "clipped_commands": 0}
        progress[morphology_id] = counts
        outputs.append(
            {
                **contract_output,
                **counts,
                "features_sha256": benchmark_data._json_hash(
                    benchmark_features(morphology, width=64, height=48)
                ),
                "files": benchmark_data.dataset_inventory(dataset_root, excluded_paths=set()),
                "assets": {
                    "revision": morphology.asset_revision,
                    "directory": morphology.menagerie_directory,
                    "files": benchmark_data.benchmark_asset_inventory(morphology),
                },
            }
        )
    value = {
        "format_version": 1,
        "generator": benchmark_data.GENERATOR_FORMAT,
        "complete": True,
        "contract": contract,
        "progress": progress,
        "pending": {morphology_id: None for morphology_id in definition.morphology_ids},
        "lerobot_version": benchmark_data.LEROBOT_VERSION,
        "outputs": outputs,
        "command_clipping_rate": 0.0,
    }
    (tmp_path / benchmark_data.GENERATION_MANIFEST_NAME).write_text(json.dumps(value))
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_benchmark_dataset(tmp_path, DEFINITION_PATH, TRAIN_MANIFEST_PATH)
    monkeypatch.setattr(benchmark_data, "_validate_lerobot_metadata", lambda *_args, **_kwargs: None)
    assert verify_benchmark_dataset(tmp_path, DEFINITION_PATH, TRAIN_MANIFEST_PATH) == value
    (tmp_path / "so101" / "payload.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload differs"):
        verify_benchmark_dataset(tmp_path, DEFINITION_PATH, TRAIN_MANIFEST_PATH)
