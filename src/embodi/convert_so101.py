from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np
import torch

from .physical.so101 import SO101CanonicalCalibration, SO101Canonicalizer, resolve_so101_order
from .record_so101 import (
    LEROBOT_VERSION,
    MANIFEST_NAME,
    RECORDING_MAX_RELATIVE_TARGET,
    RECORDER_FORMAT,
    dataset_inventory,
    sha256_file,
)


DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
CANONICAL_FEATURES = {"canonical_action", "canonical_state", "canonical_part_mask"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a physical LeRobot SO101 recording to Embodi canonical labels."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--lerobot-calibration", type=Path, required=True)
    return parser.parse_args()


def _writer_value(value, feature: dict):
    if feature["dtype"] in {"video", "image"}:
        if not torch.is_tensor(value) or value.ndim != 3 or value.shape[0] not in (1, 3, 4):
            raise ValueError("decoded camera frames must be CHW tensors")
        return value.permute(1, 2, 0).cpu().numpy()
    if torch.is_tensor(value):
        return value.cpu().numpy()
    return np.asarray(value)


def verify_recording_manifest(
    source_root: Path,
    source_repo_id: str,
    metadata,
    calibration_sha256: str,
) -> dict:
    manifest_path = source_root / "meta" / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("source dataset was not recorded with embodi-record-so101")
    recording_manifest = json.loads(manifest_path.read_text())
    if (
        recording_manifest.get("format_version") != 1
        or recording_manifest.get("recorder") != RECORDER_FORMAT
        or recording_manifest.get("complete") is not True
        or recording_manifest.get("hardware_model") != "so101_follower"
        or recording_manifest.get("stored_action") != "robot.send_action return value"
        or recording_manifest.get("lerobot_version") != LEROBOT_VERSION
        or recording_manifest.get("lerobot_calibration_sha256") != calibration_sha256
    ):
        raise ValueError("source recording provenance or calibration does not match")
    max_relative_target = recording_manifest.get("max_relative_target")
    if (
        isinstance(max_relative_target, bool)
        or not isinstance(max_relative_target, (int, float))
        or not np.isfinite(max_relative_target)
        or max_relative_target != RECORDING_MAX_RELATIVE_TARGET
    ):
        raise ValueError("source recording has an invalid max_relative_target")
    dataset = recording_manifest.get("dataset")
    features_json = json.dumps(metadata.features, sort_keys=True, separators=(",", ":"))
    expected_dataset = {
        "repo_id": source_repo_id,
        "fps": metadata.fps,
        "episodes": metadata.total_episodes,
        "frames": metadata.total_frames,
        "features_sha256": hashlib.sha256(features_json.encode()).hexdigest(),
    }
    if dataset != expected_dataset or metadata.total_episodes <= 0 or metadata.total_frames <= 0:
        raise ValueError("source recording identity or counts do not match its manifest")
    if recording_manifest.get("files") != dataset_inventory(source_root):
        raise ValueError("source recording files do not match its immutable manifest")
    return recording_manifest


def rename_noreplace(source: Path, destination: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise RuntimeError("atomic no-clobber dataset publication requires renameat2") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"output dataset already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def convert_dataset(
    source_root: Path,
    source_repo_id: str,
    output_root: Path,
    output_repo_id: str,
    calibration_path: Path,
    lerobot_calibration_path: Path,
) -> dict:
    source_resolved = source_root.resolve()
    output_resolved = output_root.resolve()
    if (
        source_resolved == output_resolved
        or source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise ValueError("source and output dataset roots must be separate, non-nested directories")
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.with_name(f".{output_root.name}.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"another conversion holds the output lock: {lock_path}") from error
    temporary_root = None
    output = None
    finalized = False
    frames = 0
    episodes = 0
    active_episode = None
    try:
        os.close(lock_fd)
        lock_fd = None
        temporary_root = output_root.with_name(f".{output_root.name}.{uuid4().hex}.tmp")
        if output_root.exists():
            raise FileExistsError(f"output dataset already exists: {output_root}")
        metadata = LeRobotDatasetMetadata(source_repo_id, root=source_root)
        if metadata.fps != 30:
            raise ValueError(f"physical SO101 conversion requires 30 FPS, got {metadata.fps}")
        for key in ("observation.state", "action"):
            if key not in metadata.features or tuple(metadata.features[key].get("shape", ())) != (6,):
                raise ValueError(f"source dataset requires six-dimensional {key}")
        camera_keys = sorted(
            key for key in metadata.features if key.startswith("observation.images.")
        )
        if camera_keys != ["observation.images.top"]:
            raise ValueError("physical SO101 conversion requires exactly one top camera")
        camera_feature = metadata.features["observation.images.top"]
        if camera_feature.get("dtype") != "video" or tuple(camera_feature.get("shape", ())) != (
            480,
            640,
            3,
        ):
            raise ValueError("physical SO101 top camera must be 640x480 RGB video")
        state_names = tuple(metadata.features["observation.state"].get("names") or ())
        action_names = tuple(metadata.features["action"].get("names") or ())
        resolve_so101_order(state_names)
        resolve_so101_order(action_names)
        calibration, canonical_calibration_sha256 = SO101CanonicalCalibration.load_snapshot(
            calibration_path
        )
        calibration.require_physical_validation()
        calibration_sha256 = calibration.verify_source_calibration(lerobot_calibration_path)
        calibration_provenance = calibration.provenance(
            calibration_path, file_sha256=canonical_calibration_sha256
        )
        recording_manifest = verify_recording_manifest(
            source_root, source_repo_id, metadata, calibration_sha256
        )
        source_features = {
            key: deepcopy(value)
            for key, value in metadata.features.items()
            if key not in DEFAULT_FEATURES | CANONICAL_FEATURES
        }
        source_features.update(
            canonical_action={"dtype": "float32", "shape": (1, 10)},
            canonical_state={"dtype": "float32", "shape": (1, 10)},
            canonical_part_mask={"dtype": "bool", "shape": (1,), "names": ["primary_effector"]},
        )
        source = LeRobotDataset(source_repo_id, root=source_root, return_uint8=True)
        source.reader.load_and_activate()
        source_frame_count = len(source.hf_dataset)
        source_episode_indices = {
            int(index) for index in source.hf_dataset.unique("episode_index")
        }
        if source_frame_count != metadata.total_frames or source_episode_indices != set(
            range(metadata.total_episodes)
        ):
            raise ValueError("source parquet rows or episode indexes do not match metadata")
        output = LeRobotDataset.create(
            repo_id=output_repo_id,
            root=temporary_root,
            fps=metadata.fps,
            robot_type=metadata.robot_type,
            features=source_features,
            use_videos=any(feature["dtype"] == "video" for feature in source_features.values()),
            image_writer_threads=4,
        )
        conversion_provenance = {
            "source_repo_id": source_repo_id,
            "output_repo_id": output_repo_id,
            "recording": recording_manifest,
            "calibration": calibration_provenance,
        }
        conversion_provenance_sha256 = hashlib.sha256(
            json.dumps(conversion_provenance, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        schema = {
            "format_version": 3,
            "producer": "embodi.so101_physical_conversion",
            "conversion_provenance_sha256": conversion_provenance_sha256,
            "parts": [
                {
                    "name": "primary_effector",
                    "kind": "pose_scalar",
                    "channel_mask": [True] * 10,
                }
            ],
            "canonical_convention": calibration_provenance,
            "source": {
                "repo_id": source_repo_id,
                "root": str(source_root),
                "fps": metadata.fps,
            },
        }
        (temporary_root / "meta" / "embodi.json").write_text(json.dumps(schema, indent=2) + "\n")
        with SO101Canonicalizer(calibration) as canonicalizer:
            for index in range(source_frame_count):
                item = source[index]
                episode = int(item["episode_index"])
                if active_episode is not None and episode != active_episode:
                    output.save_episode(parallel_encoding=False)
                    episodes += 1
                active_episode = episode
                canonical_state, canonical_action = canonicalizer.labels(
                    item["observation.state"],
                    item["action"],
                    state_names,
                    action_names,
                )
                frame = {
                    key: _writer_value(item[key], metadata.features[key])
                    for key in source_features
                    if key not in CANONICAL_FEATURES
                }
                frame.update(
                    canonical_state=canonical_state,
                    canonical_action=canonical_action,
                    canonical_part_mask=np.array([True]),
                    task=item["task"],
                )
                output.add_frame(frame)
                frames += 1
            if active_episode is not None:
                output.save_episode(parallel_encoding=False)
                episodes += 1
        output.finalize()
        finalized = True
        report = {
            "format_version": 1,
            "complete": True,
            "source_repo_id": source_repo_id,
            "output_repo_id": output_repo_id,
            "frames": frames,
            "episodes": episodes,
            "fps": metadata.fps,
            "recording": recording_manifest,
            "calibration": calibration_provenance,
            "conversion_provenance_sha256": conversion_provenance_sha256,
            "output": {
                "repo_id": output_repo_id,
                "fps": metadata.fps,
                "episodes": episodes,
                "frames": frames,
                "features_sha256": hashlib.sha256(
                    json.dumps(output.features, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            },
            "files": dataset_inventory(
                temporary_root, excluded_paths={"meta/conversion.json"}
            ),
        }
        (temporary_root / "meta" / "conversion.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if sha256_file(calibration_path) != canonical_calibration_sha256:
            raise RuntimeError("canonical calibration changed during conversion")
        if calibration.verify_source_calibration(lerobot_calibration_path) != calibration_sha256:
            raise RuntimeError("LeRobot calibration changed during conversion")
        if (
            verify_recording_manifest(source_root, source_repo_id, metadata, calibration_sha256)
            != recording_manifest
        ):
            raise RuntimeError("source recording manifest changed during conversion")
        rename_noreplace(temporary_root, output_root)
    except BaseException as error:
        if output is not None and not finalized:
            try:
                output.finalize()
            except Exception as cleanup_error:
                error.add_note(f"output finalization cleanup failed: {cleanup_error!r}")
        if temporary_root is not None and temporary_root.exists():
            try:
                shutil.rmtree(temporary_root)
            except Exception as cleanup_error:
                error.add_note(f"temporary dataset cleanup failed: {cleanup_error!r}")
        raise
    finally:
        active_error = sys.exception()
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError as cleanup_error:
                if active_error is None:
                    raise
                active_error.add_note(f"conversion lock descriptor cleanup failed: {cleanup_error!r}")
        try:
            lock_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if active_error is None:
                raise
            active_error.add_note(f"conversion lock-file cleanup failed: {cleanup_error!r}")
    return report


def main() -> None:
    args = parse_args()
    report = convert_dataset(
        args.source_root,
        args.source_repo_id,
        args.output_root,
        args.output_repo_id,
        args.calibration,
        args.lerobot_calibration,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
