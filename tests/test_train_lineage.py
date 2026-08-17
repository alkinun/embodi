from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodi import EmbodiConfig
from embodi.record_so101 import dataset_inventory
from embodi.train import (
    load_json_object,
    prepare_model_batch,
    validate_benchmark_dataset_lineage,
    validate_benchmark_training_dataset,
    validate_lerobot_training_dataset,
    write_or_validate_data_manifest,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _features(config: EmbodiConfig) -> dict:
    return {
        f"observation.images.{config.camera_names[0]}": {
            "dtype": "video",
            "shape": (48, 64, 3),
        },
        "observation.state": {"dtype": "float32", "shape": (config.state_dim,)},
        "action": {"dtype": "float32", "shape": (config.action_dim,)},
        "canonical_action": {
            "dtype": "float32",
            "shape": (config.part_count, 10),
        },
        "canonical_state": {
            "dtype": "float32",
            "shape": (config.part_count, 10),
        },
        "canonical_part_mask": {
            "dtype": "bool",
            "shape": (config.part_count,),
            "names": [part.name for part in config.parts],
        },
    }


def _metadata(root: Path, config: EmbodiConfig, *, robot_type: str = "so101_sim"):
    return SimpleNamespace(
        root=root,
        repo_id="embodi/minimal",
        robot_type=robot_type,
        fps=30,
        total_episodes=2,
        total_frames=3,
        features=_features(config),
        stats={
            "observation.state": {
                "mean": [0.0] * config.state_dim,
                "std": [1.0] * config.state_dim,
            },
            "action": {
                "mean": [0.0] * config.action_dim,
                "std": [1.0] * config.action_dim,
            },
        },
    )


def _write_schema(root: Path, config: EmbodiConfig, **extra) -> dict:
    (root / "meta").mkdir(parents=True)
    schema = {
        "format_version": 3,
        "parts": [asdict(part) for part in config.parts],
        **extra,
    }
    _write_json(root / "meta" / "embodi.json", schema)
    return json.loads(json.dumps(schema))


def _benchmark_dataset(tmp_path: Path):
    root = tmp_path / "train"
    (root / "meta").mkdir(parents=True)
    metadata = SimpleNamespace(
        root=root,
        repo_id="embodi/benchmark-so101",
        total_episodes=2,
        total_frames=3,
    )
    benchmark = {
        "format_version": 1,
        "split": "train",
        "benchmark_id": "pick-place-v1",
        "definition_sha256": "definition-hash",
        "scenario_manifest_sha256": "scenario-hash",
        "morphology": "so101",
        "episode_scenarios": [
            {"episode_index": 0, "scenario_id": "scenario-a"},
            {"episode_index": 1, "scenario_id": "scenario-b"},
        ],
    }
    _write_json(root / "meta" / "benchmark.json", benchmark)
    (root / "data.bin").write_bytes(b"three immutable frames")
    contract_output = {
        "morphology": "so101",
        "relative_root": root.name,
        "repo_id": metadata.repo_id,
        "scenario_ids": ["scenario-a", "scenario-b"],
    }
    output = {
        **contract_output,
        "morphology": "so101",
        "episodes": metadata.total_episodes,
        "frames": metadata.total_frames,
        "commands": 3,
        "clipped_commands": 0,
        "files": dataset_inventory(root, excluded_paths=set()),
    }
    manifest = {
        "format_version": 1,
        "generator": "embodi-sim-benchmark-dataset-v1",
        "complete": True,
        "pending": {"so101": None},
        "contract": {
            "split": "train",
            "benchmark_id": benchmark["benchmark_id"],
            "definition_sha256": benchmark["definition_sha256"],
            "scenario_manifest_sha256": benchmark["scenario_manifest_sha256"],
            "outputs": [contract_output],
        },
        "progress": {
            "so101": {
                key: output[key]
                for key in ("episodes", "frames", "commands", "clipped_commands")
            }
        },
        "outputs": [output],
    }
    manifest_path = tmp_path / "benchmark-dataset.json"
    _write_json(manifest_path, manifest)
    return metadata, benchmark, manifest, manifest_path


def _physical_dataset(tmp_path: Path):
    config = EmbodiConfig(camera_names=("top",))
    root = tmp_path / "physical"
    metadata = _metadata(root, config, robot_type="so_follower")
    recording = {"repo_id": "embodi/raw", "complete": True}
    calibration = {"physical_validation_complete": True, "fixture": "bench-a"}
    provenance = {
        "source_repo_id": "embodi/raw",
        "output_repo_id": metadata.repo_id,
        "recording": recording,
        "calibration": calibration,
    }
    provenance_sha256 = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_schema = _write_schema(
        root,
        config,
        producer="embodi.so101_physical_conversion",
        conversion_provenance_sha256=provenance_sha256,
    )
    (root / "data.bin").write_bytes(b"converted physical observations")
    features_json = json.dumps(metadata.features, sort_keys=True, separators=(",", ":"))
    report = {
        "format_version": 1,
        "complete": True,
        **provenance,
        "fps": metadata.fps,
        "episodes": metadata.total_episodes,
        "frames": metadata.total_frames,
        "conversion_provenance_sha256": provenance_sha256,
        "output": {
            "repo_id": metadata.repo_id,
            "fps": metadata.fps,
            "episodes": metadata.total_episodes,
            "frames": metadata.total_frames,
            "features_sha256": hashlib.sha256(features_json.encode()).hexdigest(),
        },
        "files": dataset_inventory(root, excluded_paths={"meta/conversion.json"}),
    }
    report_path = root / "meta" / "conversion.json"
    _write_json(report_path, report)
    return metadata, config, expected_schema, report, report_path


def _set_nested(value: dict, path: tuple[str | int, ...], replacement) -> None:
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


@pytest.mark.parametrize(
    "document",
    [
        '{"repo_id": "first", "repo_id": "second"}',
        '{"source": {"revision": "one", "revision": "two"}}',
    ],
)
def test_json_loader_rejects_duplicate_fields_at_any_depth(tmp_path, document) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(document)

    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_json_object(path)


def test_minimal_benchmark_lineage_is_admitted(tmp_path) -> None:
    metadata, _, _, manifest_path = _benchmark_dataset(tmp_path)

    lineage = validate_benchmark_training_dataset(metadata)

    assert lineage == {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "definition_sha256": "definition-hash",
        "scenario_manifest_sha256": "scenario-hash",
        "split": "train",
        "morphology": "so101",
    }


@pytest.mark.parametrize(
    ("document_name", "field_path", "replacement"),
    [
        ("manifest", ("generator",), "another-generator"),
        ("manifest", ("complete",), False),
        ("manifest", ("contract", "definition_sha256"), "other-definition"),
        ("manifest", ("contract", "outputs", 0, "morphology"), "panda"),
        ("manifest", ("outputs", 0, "frames"), 4),
        ("manifest", ("progress", "so101", "commands"), 2),
        ("benchmark", ("episode_scenarios", 1, "scenario_id"), "scenario-a"),
    ],
)
def test_benchmark_lineage_rejects_one_field_mutations(
    tmp_path,
    document_name,
    field_path,
    replacement,
) -> None:
    metadata, benchmark, manifest, manifest_path = _benchmark_dataset(tmp_path)
    document = deepcopy(benchmark if document_name == "benchmark" else manifest)
    _set_nested(document, field_path, replacement)
    path = (
        metadata.root / "meta" / "benchmark.json"
        if document_name == "benchmark"
        else manifest_path
    )
    _write_json(path, document)

    with pytest.raises(ValueError, match="integrity validation failed"):
        validate_benchmark_training_dataset(metadata)


def test_minimal_physical_conversion_lineage_is_admitted(tmp_path) -> None:
    metadata, config, expected_schema, _, _ = _physical_dataset(tmp_path)

    schema = validate_lerobot_training_dataset(metadata, config, "decoder")

    assert schema == expected_schema


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("format_version",), 2),
        (("complete",), False),
        (("output", "frames"), 4),
        (("output", "features_sha256"), "unrelated-features"),
        (("conversion_provenance_sha256",), "unrelated-provenance"),
        (("calibration", "physical_validation_complete"), False),
        (("files",), []),
    ],
)
def test_physical_lineage_rejects_one_field_mutations(
    tmp_path, field_path, replacement
) -> None:
    metadata, config, _, report, report_path = _physical_dataset(tmp_path)
    mutated = deepcopy(report)
    _set_nested(mutated, field_path, replacement)
    _write_json(report_path, mutated)

    with pytest.raises(ValueError, match="conversion integrity validation failed"):
        validate_lerobot_training_dataset(metadata, config, "decoder")


def test_existing_data_manifest_is_immutable(tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    report = {
        "format_version": 1,
        "repo_id": "embodi/training-data",
        "metadata_sha256": {"info.json": "abc"},
    }
    write_or_validate_data_manifest(output_dir, report)
    path = output_dir / "data_manifest.json"
    original = path.read_bytes()

    write_or_validate_data_manifest(output_dir, deepcopy(report))
    assert path.read_bytes() == original

    changed = deepcopy(report)
    changed["metadata_sha256"]["info.json"] = "def"
    with pytest.raises(ValueError, match="output directory dataset lineage"):
        write_or_validate_data_manifest(output_dir, changed)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("attribute", "replacement", "message"),
    [
        ("fps", 0, "FPS must be finite and positive"),
        ("fps", -1, "FPS must be finite and positive"),
        ("fps", float("inf"), "FPS must be finite and positive"),
        ("fps", float("nan"), "FPS must be finite and positive"),
        ("total_episodes", 0, "must contain frames and episodes"),
        ("total_frames", 0, "must contain frames and episodes"),
    ],
)
def test_lerobot_admission_rejects_invalid_numeric_boundaries(
    tmp_path,
    attribute,
    replacement,
    message,
) -> None:
    config = EmbodiConfig(camera_names=("top",))
    metadata = _metadata(tmp_path, config)
    _write_schema(tmp_path, config)
    setattr(metadata, attribute, replacement)

    with pytest.raises(ValueError, match=message):
        validate_lerobot_training_dataset(metadata, config, "decoder")


def test_lerobot_admission_stage_boundaries_for_native_actions(tmp_path) -> None:
    config = EmbodiConfig(camera_names=("top",))
    metadata = _metadata(tmp_path, config)
    _write_schema(tmp_path, config)
    del metadata.features["action"]
    del metadata.stats["action"]

    validate_lerobot_training_dataset(metadata, config, "core")
    for stage in ("decoder", "predicted_decoder", "refine"):
        with pytest.raises(ValueError, match=f"{stage} stage requires"):
            validate_lerobot_training_dataset(metadata, config, stage)


def test_lerobot_admission_accepts_zero_variance_but_rejects_negative_std(
    tmp_path,
) -> None:
    config = EmbodiConfig(camera_names=("top",))
    metadata = _metadata(tmp_path, config)
    _write_schema(tmp_path, config)
    metadata.stats["action"]["std"][0] = 0.0

    validate_lerobot_training_dataset(metadata, config, "decoder")

    metadata.stats["action"]["std"][0] = -torch.finfo(torch.float64).eps
    with pytest.raises(ValueError, match="finite and nonnegative"):
        validate_lerobot_training_dataset(metadata, config, "decoder")


@pytest.mark.parametrize(
    ("feature_name", "stage", "message"),
    [
        ("canonical_action", "core", "missing required canonical_action labels"),
        ("canonical_state", "core", "missing required canonical_state labels"),
        ("canonical_part_mask", "core", "canonical_part_mask schema"),
        ("observation.state", "core", "observation.state schema"),
        ("action", "decoder", "decoder stage requires matching native action labels"),
    ],
)
def test_lerobot_admission_rejects_missing_required_features(
    tmp_path,
    feature_name: str,
    stage: str,
    message: str,
) -> None:
    config = EmbodiConfig(camera_names=("top",))
    metadata = _metadata(tmp_path, config)
    _write_schema(tmp_path, config)
    metadata.features.pop(feature_name)

    with pytest.raises(ValueError, match=message):
        validate_lerobot_training_dataset(metadata, config, stage)


@pytest.mark.parametrize(
    ("feature_name", "replacement", "message"),
    [
        (
            "canonical_action",
            {"dtype": "uint8", "shape": (1, 10)},
            "canonical_action must be float32",
        ),
        (
            "canonical_state",
            {"dtype": "float32", "shape": (2, 10)},
            "canonical_state must be float32",
        ),
        (
            "canonical_part_mask",
            {"dtype": "bool", "shape": (1,), "names": ["wrong"]},
            "canonical_part_mask schema",
        ),
        (
            "observation.state",
            {"dtype": "float32", "shape": (5,)},
            "observation.state schema",
        ),
        (
            "action",
            {"dtype": "float32", "shape": (5,)},
            "decoder stage requires matching native action labels",
        ),
        (
            "observation.images.top",
            {"dtype": "float32", "shape": (48, 64, 3)},
            "configured camera 'top'",
        ),
        (
            "observation.images.top",
            {"dtype": "video", "shape": (48, 64)},
            "configured camera 'top'",
        ),
    ],
)
def test_lerobot_admission_rejects_malformed_feature_schema(
    tmp_path,
    feature_name: str,
    replacement: dict,
    message: str,
) -> None:
    config = EmbodiConfig(camera_names=("top",))
    metadata = _metadata(tmp_path, config)
    _write_schema(tmp_path, config)
    metadata.features[feature_name] = replacement

    with pytest.raises(ValueError, match=message):
        validate_lerobot_training_dataset(metadata, config, "decoder")


@pytest.mark.parametrize("mutation", ["missing", "non_numeric", "wrong_shape"])
def test_lerobot_admission_rejects_malformed_normalization_stats(tmp_path, mutation: str) -> None:
    config = EmbodiConfig(camera_names=("top",))
    metadata = _metadata(tmp_path, config)
    _write_schema(tmp_path, config)
    if mutation == "missing":
        metadata.stats.pop("observation.state")
    elif mutation == "non_numeric":
        metadata.stats["observation.state"]["mean"] = ["bad"]
    else:
        metadata.stats["observation.state"]["std"] = [1.0]

    with pytest.raises(ValueError, match="normalization stats"):
        validate_lerobot_training_dataset(metadata, config, "decoder")


@pytest.mark.parametrize("mutation", ["missing", "format", "parts"])
def test_lerobot_admission_rejects_missing_or_mismatched_part_schema(tmp_path, mutation: str) -> None:
    config = EmbodiConfig(camera_names=("top",))
    metadata = _metadata(tmp_path, config)
    schema_path = tmp_path / "meta" / "embodi.json"
    _write_schema(tmp_path, config)
    if mutation == "missing":
        schema_path.unlink()
    else:
        schema = json.loads(schema_path.read_text())
        schema["format_version"] = 2 if mutation == "format" else 3
        if mutation == "parts":
            schema["parts"] = []
        _write_json(schema_path, schema)

    expected = "missing meta/embodi.json" if mutation == "missing" else "part descriptors"
    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        validate_lerobot_training_dataset(metadata, config, "decoder")


def test_benchmark_lineage_returns_none_for_an_ordinary_dataset(tmp_path) -> None:
    root = tmp_path / "ordinary"
    (root / "meta").mkdir(parents=True)

    assert validate_benchmark_dataset_lineage(SimpleNamespace(root=root), "train") is None


def test_benchmark_lineage_rejects_validation_role_and_missing_generation_manifest(tmp_path) -> None:
    metadata, _, _, manifest_path = _benchmark_dataset(tmp_path)

    with pytest.raises(ValueError, match="split does not match"):
        validate_benchmark_dataset_lineage(metadata, "validation")

    manifest_path.unlink()
    with pytest.raises(FileNotFoundError, match="root generation manifest"):
        validate_benchmark_dataset_lineage(metadata, "train")


@pytest.mark.parametrize("output_key", ["outputs", "contract"])
def test_benchmark_lineage_rejects_ambiguous_morphology_mapping(tmp_path, output_key: str) -> None:
    metadata, _, manifest, manifest_path = _benchmark_dataset(tmp_path)
    if output_key == "outputs":
        manifest[output_key].append(deepcopy(manifest[output_key][0]))
    else:
        manifest[output_key]["outputs"].append(
            deepcopy(manifest[output_key]["outputs"][0])
        )
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="morphology mapping is ambiguous"):
        validate_benchmark_dataset_lineage(metadata, "train")


def test_benchmark_lineage_detects_outer_manifest_without_benchmark_metadata(tmp_path) -> None:
    root = tmp_path / "train"
    (root / "meta").mkdir(parents=True)
    _write_json(
        tmp_path / "benchmark-dataset.json",
        {"outputs": [{"relative_root": "train"}]},
    )

    with pytest.raises(FileNotFoundError, match="meta/benchmark.json lineage"):
        validate_benchmark_dataset_lineage(SimpleNamespace(root=root), "train")


class _UnusedProcessor:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError(
            "processor must not be called when image processing is disabled"
        )


def test_prepare_model_batch_builds_descriptors_and_normalizes_mask() -> None:
    config = EmbodiConfig(camera_names=("top",), chunk_size=4, n_action_steps=4)
    padding = torch.tensor([[False, False, True, True], [False, True, True, True]])
    raw_batch = {
        "canonical_action": torch.randn(2, 4, 1, 10),
        "canonical_state": torch.randn(2, 1, 10),
        "canonical_part_mask": torch.tensor([True, False]),
        "canonical_action_is_pad": padding,
        "canonical_state_is_pad": padding.clone(),
        "action_is_pad": padding.clone(),
    }

    batch = prepare_model_batch(
        raw_batch,
        _UnusedProcessor(),
        torch.device("cpu"),
        config.camera_names,
        config,
        process_images=False,
    )

    assert batch["canonical_action"].shape == (2, 4, 1, 10)
    assert batch["canonical_state"].shape == (2, 1, 10)
    assert batch["canonical_part_mask"].tolist() == [[True], [False]]
    assert batch["canonical_part_kind"].shape == (2, 1)
    assert batch["canonical_part_name_features"].shape == (2, 1, 32)
    assert batch["canonical_channel_mask"].shape == (2, 1, 10)
    assert batch["canonical_channel_mask"].all()
    torch.testing.assert_close(batch["action_is_pad"], padding)


def test_prepare_model_batch_defaults_part_mask_and_rejects_wrong_part_layout() -> None:
    config = EmbodiConfig(camera_names=("top",), chunk_size=4, n_action_steps=4)
    raw_batch = {
        "canonical_action": torch.zeros(2, 4, 1, 10),
        "canonical_state": torch.zeros(2, 1, 10),
    }
    batch = prepare_model_batch(
        raw_batch,
        _UnusedProcessor(),
        torch.device("cpu"),
        config.camera_names,
        config,
        process_images=False,
    )
    assert batch["canonical_part_mask"].tolist() == [[True], [True]]

    raw_batch["canonical_action"] = torch.zeros(2, 4, 2, 10)
    with pytest.raises(ValueError, match="configured part layout"):
        prepare_model_batch(
            raw_batch,
            _UnusedProcessor(),
            torch.device("cpu"),
            config.camera_names,
            config,
            process_images=False,
        )


def test_prepare_model_batch_rejects_native_padding_disagreement() -> None:
    config = EmbodiConfig(camera_names=("top",), chunk_size=4, n_action_steps=4)
    raw_batch = {
        "canonical_action": torch.zeros(1, 4, 1, 10),
        "canonical_state": torch.zeros(1, 1, 10),
        "canonical_action_is_pad": torch.tensor([[False, False, True, True]]),
        "canonical_state_is_pad": torch.tensor([[False, False, True, True]]),
        "action_is_pad": torch.tensor([[False, True, True, True]]),
    }

    with pytest.raises(ValueError, match="padding masks disagree"):
        prepare_model_batch(
            raw_batch,
            _UnusedProcessor(),
            torch.device("cpu"),
            config.camera_names,
            config,
            process_images=False,
        )
