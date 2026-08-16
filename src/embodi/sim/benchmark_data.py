from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm

from embodi.record_so101 import atomic_write_json, dataset_inventory, sha256_file

from .benchmark import BenchmarkDefinition, Scenario, ScenarioManifest, file_sha256
from .benchmark_environment import (
    BenchmarkManipulationEnv,
    prepare_benchmark_robot_asset,
    runtime_scenario,
)
from .benchmark_expert import BenchmarkPhase, PrivilegedBenchmarkExpert
from .morphology import MorphologySpec, morphology_by_id
from .tasks import task_by_id


GENERATOR_FORMAT = "embodi-sim-benchmark-dataset-v1"
GENERATION_MANIFEST_NAME = "benchmark-dataset.json"
LEROBOT_VERSION = "0.6.1"
SCHEMA = {
    "format_version": 3,
    "producer": "embodi.sim.benchmark_data",
    "parts": [
        {
            "name": "primary_effector",
            "kind": "pose_scalar",
            "channel_mask": [True] * 10,
        }
    ],
}
LEROBOT_DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def benchmark_features(
    morphology: MorphologySpec,
    *,
    width: int,
    height: int,
) -> dict[str, dict[str, Any]]:
    return {
        "observation.images.top": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (morphology.native_state_dim,),
            "names": list(morphology.native_state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (morphology.native_action_dim,),
            "names": list(morphology.native_action_names),
        },
        "canonical_action": {"dtype": "float32", "shape": (1, 10)},
        "canonical_state": {"dtype": "float32", "shape": (1, 10)},
        "canonical_part_mask": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["primary_effector"],
        },
    }


def select_scenarios(
    manifest: ScenarioManifest,
    morphology_ids: tuple[str, ...],
    limit_per_cell: int | None,
) -> dict[str, tuple[Scenario, ...]]:
    if limit_per_cell is not None and limit_per_cell <= 0:
        raise ValueError("limit_per_cell must be positive")
    selected: dict[str, list[Scenario]] = {morphology_id: [] for morphology_id in morphology_ids}
    counts: Counter[tuple[str, str]] = Counter()
    for scenario in manifest.scenarios:
        if scenario.morphology not in selected:
            continue
        cell = (scenario.morphology, scenario.task)
        if limit_per_cell is not None and counts[cell] >= limit_per_cell:
            continue
        selected[scenario.morphology].append(scenario)
        counts[cell] += 1
    if any(not scenarios for scenarios in selected.values()):
        raise ValueError("scenario selection must contain every benchmark morphology")
    return {key: tuple(value) for key, value in selected.items()}


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    filenames = (
        "assets.py",
        "benchmark_data.py",
        "benchmark_environment.py",
        "benchmark_expert.py",
        "environment.py",
        "morphology.py",
        "tasks.py",
    )
    return {filename: sha256_file(directory / filename) for filename in filenames}


def benchmark_asset_inventory(morphology: MorphologySpec) -> list[dict[str, Any]]:
    root, robot_filename = prepare_benchmark_robot_asset(morphology)
    paths = [root / morphology.model_filename, root / robot_filename, root / "LICENSE"]
    paths.extend(path for path in sorted((root / "assets").rglob("*")) if path.is_file())
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _validate_frozen_manifest_hash(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    split: str,
) -> None:
    checksum_path = definition.path.parent / "manifests.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError("benchmark generation requires frozen manifest checksums")
    checksums = {
        relative_path: digest
        for digest, relative_path in (
            line.split() for line in checksum_path.read_text().splitlines()
        )
    }
    expected = checksums.get(f"manifests/{split}.json")
    if expected is None or file_sha256(manifest.path) != expected:
        raise ValueError("demonstration manifest does not match the frozen split checksum")


def _contract(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    selected: dict[str, tuple[Scenario, ...]],
    *,
    repo_prefix: str,
    width: int,
    height: int,
    limit_per_cell: int | None,
    image_writer_threads: int,
) -> dict[str, Any]:
    split_ids = {scenario.split for scenario in manifest.scenarios}
    if len(split_ids) != 1:
        raise ValueError("demonstration manifests must contain exactly one split")
    split = split_ids.pop()
    if split not in {"train", "validation"}:
        raise ValueError("demonstrations may be generated only from train or validation scenarios")
    _validate_frozen_manifest_hash(definition, manifest, split)
    if not repo_prefix or repo_prefix.endswith("-"):
        raise ValueError("repo_prefix must be non-empty and must not end with a hyphen")
    control = definition.values["control"]
    if control["frequency_hz"] != 30 or control["camera_names"] != ["top"]:
        raise ValueError("benchmark generation requires the frozen 30 Hz top-camera contract")
    return {
        "benchmark_id": definition.values["benchmark_id"],
        "definition_path": str(definition.path),
        "definition_sha256": definition.sha256,
        "scenario_manifest_path": str(manifest.path),
        "scenario_manifest_sha256": file_sha256(manifest.path),
        "split": split,
        "scenario_seed_usage": "provenance_only_runtime_is_deterministic",
        "software": {
            "generator": GENERATOR_FORMAT,
            "lerobot": LEROBOT_VERSION,
            "mujoco": version("mujoco"),
            "implementation_sha256": _implementation_hashes(),
        },
        "control": {
            "frequency_hz": control["frequency_hz"],
            "maximum_episode_steps": control["maximum_episode_steps"],
            "camera_names": ["top"],
            "image_width": width,
            "image_height": height,
        },
        "encoding": {"image_writer_threads": image_writer_threads},
        "limit_per_cell": limit_per_cell,
        "outputs": [
            {
                "morphology": morphology_id,
                "repo_id": f"{repo_prefix}-{morphology_id}",
                "relative_root": morphology_id,
                "scenario_ids": [scenario.scenario_id for scenario in scenarios],
            }
            for morphology_id, scenarios in selected.items()
        ],
    }


def _new_generation_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "generator": GENERATOR_FORMAT,
        "complete": False,
        "contract": contract,
        "progress": {
            output["morphology"]: {
                "episodes": 0,
                "frames": 0,
                "commands": 0,
                "clipped_commands": 0,
            }
            for output in contract["outputs"]
        },
        "pending": {output["morphology"]: None for output in contract["outputs"]},
    }


def _load_or_create_generation_manifest(
    root: Path,
    contract: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    path = root / GENERATION_MANIFEST_NAME
    if not resume:
        if root.exists():
            raise FileExistsError(f"benchmark dataset root already exists: {root}")
        root.mkdir(parents=True)
        value = _new_generation_manifest(contract)
        atomic_write_json(path, value)
        return value
    if not root.is_dir() or not path.is_file():
        raise FileNotFoundError("resume requires an existing benchmark generation manifest")
    value = json.loads(path.read_text())
    if value.get("format_version") != 1 or value.get("generator") != GENERATOR_FORMAT:
        raise ValueError("benchmark generation manifest format does not match")
    if value.get("complete") is not False:
        raise ValueError("completed benchmark datasets are immutable and cannot be resumed")
    if value.get("contract") != contract:
        raise ValueError("resume contract does not match the incomplete benchmark dataset")
    expected_progress = set(output["morphology"] for output in contract["outputs"])
    if (
        set(value.get("progress", {})) != expected_progress
        or set(value.get("pending", {})) != expected_progress
    ):
        raise ValueError("resume progress does not match benchmark morphologies")
    return value


def _features_equal(left: dict, right: dict) -> bool:
    if set(left) != set(right) | set(LEROBOT_DEFAULT_FEATURES):
        return False
    for name, expected in right.items():
        actual = left.get(name)
        if actual is None:
            return False
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if key == "shape":
                if tuple(actual_value or ()) != tuple(expected_value):
                    return False
            elif key == "info":
                if any(actual_value.get(item) != value for item, value in expected_value.items()):
                    return False
            elif actual_value != expected_value:
                return False
    for name, expected in LEROBOT_DEFAULT_FEATURES.items():
        actual = left[name]
        if (
            actual.get("dtype") != expected["dtype"]
            or tuple(actual.get("shape", ())) != expected["shape"]
            or actual.get("names") is not None
        ):
            return False
    return True


def _write_or_validate_schema(root: Path) -> None:
    path = root / "meta" / "embodi.json"
    text = json.dumps(SCHEMA, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("existing Embodi canonical schema does not match benchmark generation")
    path.write_text(text)


def _command_is_clipped(env: BenchmarkManipulationEnv, action: np.ndarray) -> bool:
    restored = env.qpos_to_native(env.native_to_qpos(action))
    return not np.allclose(action, restored, atol=1e-5, rtol=0.0)


def rollout_scenario(
    scenario: Scenario,
    *,
    dataset=None,
    width: int = 640,
    height: int = 480,
    maximum_episode_steps: int = 500,
    frequency_hz: float = 30.0,
    environment_factory: Callable[..., BenchmarkManipulationEnv] = BenchmarkManipulationEnv,
) -> dict[str, int]:
    env = environment_factory(
        scenario.morphology,
        scenario.task,
        runtime_scenario(scenario.factors),
        width=width,
        height=height,
        render=dataset is not None,
    )
    frames = 0
    clipped_commands = 0
    try:
        expert = PrivilegedBenchmarkExpert(env, frequency=frequency_hz)
        for _ in range(maximum_episode_steps):
            state = env.state().astype(np.float32, copy=False)
            action = expert.action().astype(np.float32, copy=False)
            canonical_action = env.canonical_arm_action(action).astype(np.float32, copy=False)
            canonical_state = env.canonical_state().astype(np.float32, copy=False)
            clipped_commands += int(_command_is_clipped(env, action))
            if dataset is not None:
                image = env.render_top()
                if image.shape != (height, width, 3) or image.dtype != np.uint8:
                    raise RuntimeError("benchmark renderer produced an invalid top-camera frame")
                dataset.add_frame(
                    {
                        "observation.images.top": image,
                        "observation.state": state,
                        "action": action,
                        "canonical_action": canonical_action,
                        "canonical_state": canonical_state,
                        "canonical_part_mask": np.array([True]),
                        "task": env.instruction,
                    }
                )
            frames += 1
            env.apply_action(action)
            env.step_control_period(frequency_hz)
            if expert.terminal:
                break
        if expert.phase != BenchmarkPhase.DONE or not env.success:
            raise RuntimeError(
                f"privileged expert failed frozen scenario {scenario.scenario_id} "
                f"in phase {expert.phase.value}"
            )
        return {"frames": frames, "commands": frames, "clipped_commands": clipped_commands}
    except BaseException:
        if dataset is not None:
            dataset.clear_episode_buffer()
        raise
    finally:
        env.close()


def _dataset_class():
    if version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(f"benchmark generation requires lerobot=={LEROBOT_VERSION}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def _validate_lerobot_metadata(
    root: Path,
    repo_id: str,
    morphology: MorphologySpec,
    scenarios: tuple[Scenario, ...],
    *,
    frames: int,
    width: int,
    height: int,
    benchmark_metadata: dict[str, Any],
) -> None:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError("generated dataset is missing local LeRobot metadata")
    _validate_payload_files(root)
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    if (
        metadata.repo_id != repo_id
        or metadata.robot_type != f"{morphology.id}_sim"
        or metadata.fps != 30
        or metadata.total_episodes != len(scenarios)
        or metadata.total_frames != frames
    ):
        raise ValueError("generated LeRobot metadata identity or counts do not match")
    expected_features = benchmark_features(morphology, width=width, height=height)
    if not _features_equal(metadata.features, expected_features):
        raise ValueError("generated LeRobot feature schema does not match")
    episodes = metadata.episodes
    if list(episodes["episode_index"]) != list(range(len(scenarios))):
        raise ValueError("generated LeRobot episode indices are not contiguous")
    if sum(int(length) for length in episodes["length"]) != frames:
        raise ValueError("generated LeRobot episode lengths do not match frame count")
    expected_tasks = [[task_by_id(scenario.task).instruction] for scenario in scenarios]
    if list(episodes["tasks"]) != expected_tasks:
        raise ValueError("generated LeRobot episode tasks do not match frozen scenarios")
    if json.loads((root / "meta" / "benchmark.json").read_text()) != benchmark_metadata:
        raise ValueError("generated LeRobot episode/scenario mapping does not match")


def _validate_payload_files(root: Path) -> None:
    import av
    import pyarrow.parquet as parquet

    parquet_paths = sorted(root.glob("data/**/*.parquet")) + sorted(
        root.glob("meta/episodes/**/*.parquet")
    )
    video_paths = sorted(root.glob("videos/**/*.mp4"))
    if not parquet_paths or not video_paths:
        raise FileNotFoundError("generated dataset is missing Parquet or video payloads")
    for path in parquet_paths:
        if parquet.ParquetFile(path).scan_contents() <= 0:
            raise ValueError(f"generated dataset contains empty Parquet payload: {path}")
    for path in video_paths:
        with av.open(str(path)) as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise ValueError(f"generated dataset video must contain one stream: {path}")
            stream = streams[0]
            if next(container.decode(stream), None) is None:
                raise ValueError(f"generated dataset video contains no decodable frame: {path}")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            packet_count = sum(packet.size > 0 for packet in container.demux(stream))
            if packet_count == 0:
                raise ValueError(f"generated dataset video contains no packets: {path}")


def _open_dataset(
    root: Path,
    repo_id: str,
    morphology: MorphologySpec,
    *,
    width: int,
    height: int,
    resume: bool,
    image_writer_threads: int,
):
    dataset_type = _dataset_class()
    features = benchmark_features(morphology, width=width, height=height)
    if root.exists():
        if not resume:
            raise FileExistsError(f"morphology dataset already exists: {root}")
        dataset = dataset_type.resume(
            repo_id=repo_id,
            root=root,
            image_writer_threads=image_writer_threads,
        )
        if dataset.repo_id != repo_id or dataset.fps != 30:
            raise ValueError("existing morphology dataset identity does not match resume contract")
        if dataset.meta.robot_type != f"{morphology.id}_sim":
            raise ValueError("existing morphology dataset robot type does not match resume contract")
        if not _features_equal(dataset.features, features):
            raise ValueError("existing morphology dataset features do not match resume contract")
    else:
        dataset = dataset_type.create(
            repo_id=repo_id,
            root=root,
            fps=30,
            robot_type=f"{morphology.id}_sim",
            features=features,
            use_videos=True,
            image_writer_threads=image_writer_threads,
            metadata_buffer_size=1,
        )
    _write_or_validate_schema(root)
    return dataset


def _reconcile_progress(
    dataset,
    scenarios: tuple[Scenario, ...],
    progress: dict[str, int],
    pending: dict[str, Any] | None,
) -> None:
    recorded_episodes = progress["episodes"]
    if dataset.num_episodes == recorded_episodes:
        if dataset.num_frames != progress["frames"]:
            raise ValueError("resume frame count does not match generation progress")
        return
    if dataset.num_episodes != recorded_episodes + 1 or recorded_episodes >= len(scenarios):
        raise ValueError("resume episode count is not a valid completed scenario prefix")
    expected_scenario = scenarios[recorded_episodes]
    if (
        pending is None
        or pending.get("scenario_id") != expected_scenario.scenario_id
        or dataset.num_frames - progress["frames"] != pending.get("frames")
    ):
        raise ValueError("saved episode does not match the pending frozen scenario")
    progress["episodes"] += 1
    progress["frames"] += pending["frames"]
    progress["commands"] += pending["commands"]
    progress["clipped_commands"] += pending["clipped_commands"]


def _dataset_metadata_value(
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    morphology: MorphologySpec,
    scenarios: tuple[Scenario, ...],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "benchmark_id": definition.values["benchmark_id"],
        "definition_sha256": definition.sha256,
        "scenario_manifest_sha256": file_sha256(manifest.path),
        "split": scenarios[0].split,
        "morphology": morphology.id,
        "episode_scenarios": [
            {"episode_index": index, "scenario_id": scenario.scenario_id}
            for index, scenario in enumerate(scenarios)
        ],
    }


def _write_dataset_metadata(
    root: Path,
    definition: BenchmarkDefinition,
    manifest: ScenarioManifest,
    morphology: MorphologySpec,
    scenarios: tuple[Scenario, ...],
) -> None:
    atomic_write_json(
        root / "meta" / "benchmark.json",
        _dataset_metadata_value(definition, manifest, morphology, scenarios),
    )


def _finalize_generation_manifest(
    root: Path,
    value: dict[str, Any],
    definition: BenchmarkDefinition,
) -> dict[str, Any]:
    outputs = []
    total_commands = 0
    total_clipped = 0
    for output in value["contract"]["outputs"]:
        morphology = morphology_by_id(output["morphology"])
        progress = value["progress"][morphology.id]
        dataset_root = root / output["relative_root"]
        total_commands += progress["commands"]
        total_clipped += progress["clipped_commands"]
        outputs.append(
            {
                **output,
                **progress,
                "features_sha256": _json_hash(
                    benchmark_features(
                        morphology,
                        width=value["contract"]["control"]["image_width"],
                        height=value["contract"]["control"]["image_height"],
                    )
                ),
                "files": dataset_inventory(dataset_root, excluded_paths=set()),
                "assets": {
                    "revision": morphology.asset_revision,
                    "directory": morphology.menagerie_directory,
                    "files": benchmark_asset_inventory(morphology),
                },
            }
        )
    clipping_rate = total_clipped / total_commands if total_commands else 0.0
    maximum_clipping = definition.values["admission_gates"]["maximum_command_clipping_rate"]
    if clipping_rate > maximum_clipping:
        raise RuntimeError(
            f"benchmark command clipping rate {clipping_rate:.3%} exceeds {maximum_clipping:.3%}"
        )
    final = {
        **value,
        "complete": True,
        "lerobot_version": LEROBOT_VERSION,
        "outputs": outputs,
        "command_clipping_rate": clipping_rate,
    }
    atomic_write_json(root / GENERATION_MANIFEST_NAME, final)
    return final


def verify_benchmark_dataset(
    root: Path,
    definition_path: Path,
    scenario_manifest_path: Path,
) -> dict[str, Any]:
    definition = BenchmarkDefinition.load(definition_path)
    manifest = ScenarioManifest.load(
        scenario_manifest_path,
        definition,
        require_complete_counts=True,
    )
    path = root / GENERATION_MANIFEST_NAME
    value = json.loads(path.read_text())
    if (
        value.get("format_version") != 1
        or value.get("generator") != GENERATOR_FORMAT
        or value.get("complete") is not True
    ):
        raise ValueError("benchmark dataset manifest is not complete")
    contract = value.get("contract", {})
    split_ids = {scenario.split for scenario in manifest.scenarios}
    if contract.get("benchmark_id") != definition.values["benchmark_id"]:
        raise ValueError("benchmark dataset id does not match")
    if contract.get("definition_sha256") != definition.sha256:
        raise ValueError("benchmark dataset definition hash does not match")
    if contract.get("scenario_manifest_sha256") != file_sha256(manifest.path):
        raise ValueError("benchmark dataset scenario manifest hash does not match")
    if len(split_ids) != 1 or contract.get("split") != split_ids.pop():
        raise ValueError("benchmark dataset split does not match")
    _validate_frozen_manifest_hash(definition, manifest, contract["split"])
    expected_software = {
        "generator": GENERATOR_FORMAT,
        "lerobot": LEROBOT_VERSION,
        "mujoco": version("mujoco"),
        "implementation_sha256": _implementation_hashes(),
    }
    if contract.get("software") != expected_software or value.get("lerobot_version") != LEROBOT_VERSION:
        raise ValueError("benchmark dataset software contract does not match")
    if (
        set(value.get("progress", {})) != set(definition.morphology_ids)
        or set(value.get("pending", {})) != set(definition.morphology_ids)
    ):
        raise ValueError("benchmark dataset progress does not cover all morphologies")
    if any(value.get("pending", {}).values()):
        raise ValueError("completed benchmark dataset contains a pending episode")
    selected = select_scenarios(
        manifest,
        definition.morphology_ids,
        contract.get("limit_per_cell"),
    )
    output_ids = [output.get("morphology") for output in value.get("outputs", [])]
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != set(definition.morphology_ids):
        raise ValueError("benchmark dataset outputs do not cover all morphologies exactly once")
    for output in value.get("outputs", []):
        morphology = morphology_by_id(output["morphology"])
        contract_output = next(
            item for item in contract.get("outputs", []) if item.get("morphology") == morphology.id
        )
        if any(output.get(key) != contract_output.get(key) for key in contract_output):
            raise ValueError("benchmark dataset output identity differs from its contract")
        relative_root = Path(output["relative_root"])
        if relative_root.name != output["relative_root"]:
            raise ValueError("benchmark dataset output root must be one relative directory")
        scenarios = output.get("scenario_ids", [])
        expected = [scenario.scenario_id for scenario in selected[morphology.id]]
        if scenarios != expected or output.get("episodes") != len(scenarios):
            raise ValueError("benchmark dataset episode mapping is not a manifest-ordered prefix")
        if value["progress"][morphology.id] != {
            key: output[key]
            for key in ("episodes", "frames", "commands", "clipped_commands")
        }:
            raise ValueError("benchmark dataset final counts differ from generation progress")
        if output.get("frames") != output.get("commands") or output["frames"] <= 0:
            raise ValueError("benchmark dataset frame and command counts are invalid")
        expected_features_hash = _json_hash(
            benchmark_features(
                morphology,
                width=contract["control"]["image_width"],
                height=contract["control"]["image_height"],
            )
        )
        if output.get("features_sha256") != expected_features_hash:
            raise ValueError("benchmark dataset feature schema hash does not match")
        dataset_root = root / relative_root
        _validate_lerobot_metadata(
            dataset_root,
            output["repo_id"],
            morphology,
            selected[morphology.id],
            frames=output["frames"],
            width=contract["control"]["image_width"],
            height=contract["control"]["image_height"],
            benchmark_metadata=_dataset_metadata_value(
                definition,
                manifest,
                morphology,
                selected[morphology.id],
            ),
        )
        if output.get("files") != dataset_inventory(dataset_root, excluded_paths=set()):
            raise ValueError("benchmark dataset payload differs from its immutable inventory")
        assets = output.get("assets", {})
        if (
            assets.get("revision") != morphology.asset_revision
            or assets.get("directory") != morphology.menagerie_directory
        ):
            raise ValueError("benchmark dataset asset revision does not match")
        if assets.get("files") != benchmark_asset_inventory(morphology):
            raise ValueError("benchmark simulation assets differ from their immutable inventory")
    total_commands = sum(output["commands"] for output in value["outputs"])
    total_clipped = sum(output["clipped_commands"] for output in value["outputs"])
    clipping_rate = total_clipped / total_commands if total_commands else 0.0
    if value.get("command_clipping_rate") != clipping_rate:
        raise ValueError("benchmark dataset command clipping rate does not match counts")
    if clipping_rate > definition.values["admission_gates"]["maximum_command_clipping_rate"]:
        raise ValueError("benchmark dataset command clipping exceeds the admission gate")
    return value


def generate_benchmark_dataset(
    definition_path: Path,
    scenario_manifest_path: Path,
    root: Path,
    repo_prefix: str,
    *,
    width: int = 640,
    height: int = 480,
    limit_per_cell: int | None = None,
    resume: bool = False,
    image_writer_threads: int = 4,
) -> dict[str, Any]:
    if width <= 0 or height <= 0 or image_writer_threads < 0:
        raise ValueError("image dimensions must be positive and writer threads must be non-negative")
    definition = BenchmarkDefinition.load(definition_path)
    manifest = ScenarioManifest.load(
        scenario_manifest_path,
        definition,
        require_complete_counts=True,
    )
    selected = select_scenarios(manifest, definition.morphology_ids, limit_per_cell)
    contract = _contract(
        definition,
        manifest,
        selected,
        repo_prefix=repo_prefix,
        width=width,
        height=height,
        limit_per_cell=limit_per_cell,
        image_writer_threads=image_writer_threads,
    )
    value = _load_or_create_generation_manifest(root, contract, resume=resume)
    control = contract["control"]
    manifest_path = root / GENERATION_MANIFEST_NAME
    for output in contract["outputs"]:
        morphology = morphology_by_id(output["morphology"])
        scenarios = selected[morphology.id]
        progress = value["progress"][morphology.id]
        pending = value["pending"][morphology.id]
        dataset_root = root / output["relative_root"]
        dataset = _open_dataset(
            dataset_root,
            output["repo_id"],
            morphology,
            width=width,
            height=height,
            resume=dataset_root.exists(),
            image_writer_threads=image_writer_threads,
        )
        try:
            _reconcile_progress(
                dataset,
                scenarios,
                progress,
                pending,
            )
            value["pending"][morphology.id] = None
            atomic_write_json(manifest_path, value)
            remaining = scenarios[progress["episodes"] :]
            progress_bar = tqdm(remaining, desc=f"{morphology.id} demonstrations", unit="episode")
            for scenario in progress_bar:
                result = rollout_scenario(
                    scenario,
                    dataset=dataset,
                    width=width,
                    height=height,
                    maximum_episode_steps=control["maximum_episode_steps"],
                    frequency_hz=control["frequency_hz"],
                )
                value["pending"][morphology.id] = {
                    "scenario_id": scenario.scenario_id,
                    **result,
                }
                atomic_write_json(manifest_path, value)
                dataset.save_episode(parallel_encoding=False)
                progress["episodes"] += 1
                progress["frames"] += result["frames"]
                progress["commands"] += result["commands"]
                progress["clipped_commands"] += result["clipped_commands"]
                value["pending"][morphology.id] = None
                atomic_write_json(manifest_path, value)
                progress_bar.set_postfix(frames=progress["frames"])
        finally:
            dataset.finalize()
        _write_dataset_metadata(dataset_root, definition, manifest, morphology, scenarios)
        _validate_lerobot_metadata(
            dataset_root,
            output["repo_id"],
            morphology,
            scenarios,
            frames=progress["frames"],
            width=width,
            height=height,
            benchmark_metadata=_dataset_metadata_value(
                definition,
                manifest,
                morphology,
                scenarios,
            ),
        )
    return _finalize_generation_manifest(root, value, definition)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate frozen benchmark demonstrations")
    parser.add_argument("definition", type=Path)
    parser.add_argument("scenario_manifest", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-prefix", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--limit-per-cell", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    report = generate_benchmark_dataset(
        args.definition,
        args.scenario_manifest,
        args.root,
        args.repo_prefix,
        width=args.width,
        height=args.height,
        limit_per_cell=args.limit_per_cell,
        resume=args.resume,
        image_writer_threads=args.image_writer_threads,
    )
    print(
        json.dumps(
            {
                "root": str(args.root),
                "split": report["contract"]["split"],
                "outputs": [
                    {
                        "morphology": output["morphology"],
                        "episodes": output["episodes"],
                        "frames": output["frames"],
                    }
                    for output in report["outputs"]
                ],
                "command_clipping_rate": report["command_clipping_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
