from __future__ import annotations

import argparse
from contextlib import suppress
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .physical.so101 import SO101CanonicalCalibration, SO101Canonicalizer, resolve_so101_order
from .record_so101 import RECORDER_FORMAT


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


def convert_dataset(
    source_root: Path,
    source_repo_id: str,
    output_root: Path,
    output_repo_id: str,
    calibration_path: Path,
    lerobot_calibration_path: Path,
) -> dict:
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source and output dataset roots must differ")
    if output_root.exists():
        raise FileExistsError(f"output dataset already exists: {output_root}")
    temporary_root = output_root.with_name(f".{output_root.name}.tmp")
    if temporary_root.exists():
        raise FileExistsError(f"temporary output dataset already exists: {temporary_root}")
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    metadata = LeRobotDatasetMetadata(source_repo_id, root=source_root)
    if metadata.fps != 30:
        raise ValueError(f"physical SO101 conversion requires 30 FPS, got {metadata.fps}")
    for key in ("observation.state", "action"):
        if key not in metadata.features or tuple(metadata.features[key].get("shape", ())) != (6,):
            raise ValueError(f"source dataset requires six-dimensional {key}")
    state_names = tuple(metadata.features["observation.state"].get("names") or ())
    action_names = tuple(metadata.features["action"].get("names") or ())
    resolve_so101_order(state_names)
    resolve_so101_order(action_names)

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
    source = LeRobotDataset(
        source_repo_id,
        root=source_root,
        return_uint8=True,
    )
    calibration = SO101CanonicalCalibration.load(calibration_path)
    calibration_sha256 = calibration.verify_source_calibration(lerobot_calibration_path)
    manifest_path = source_root / "meta" / "embodi-recording.json"
    if not manifest_path.is_file():
        raise ValueError("source dataset was not recorded with embodi-record-so101")
    recording_manifest = json.loads(manifest_path.read_text())
    if (
        recording_manifest.get("format_version") != 1
        or recording_manifest.get("recorder") != RECORDER_FORMAT
        or recording_manifest.get("hardware_model") != "so101_follower"
        or recording_manifest.get("stored_action") != "robot.send_action return value"
        or recording_manifest.get("lerobot_calibration_sha256") != calibration_sha256
    ):
        raise ValueError("source recording provenance or calibration does not match")
    max_relative_target = recording_manifest.get("max_relative_target")
    if (
        isinstance(max_relative_target, bool)
        or not isinstance(max_relative_target, (int, float))
        or not np.isfinite(max_relative_target)
        or max_relative_target <= 0
    ):
        raise ValueError("source recording has an invalid max_relative_target")
    output = None
    finalized = False
    frames = 0
    episodes = 0
    active_episode = None
    try:
        output = LeRobotDataset.create(
            repo_id=output_repo_id,
            root=temporary_root,
            fps=metadata.fps,
            robot_type=metadata.robot_type,
            features=source_features,
            use_videos=any(feature["dtype"] == "video" for feature in source_features.values()),
            image_writer_threads=4,
        )
        schema = {
            "format_version": 3,
            "parts": [
                {
                    "name": "primary_effector",
                    "kind": "pose_scalar",
                    "channel_mask": [True] * 10,
                }
            ],
            "canonical_convention": calibration.provenance(calibration_path),
            "source": {
                "repo_id": source_repo_id,
                "root": str(source_root),
                "fps": metadata.fps,
            },
        }
        (temporary_root / "meta" / "embodi.json").write_text(json.dumps(schema, indent=2) + "\n")
        with SO101Canonicalizer(calibration) as canonicalizer:
            for index in range(len(source)):
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
            "source_repo_id": source_repo_id,
            "output_repo_id": output_repo_id,
            "frames": frames,
            "episodes": episodes,
            "fps": metadata.fps,
            "recording": recording_manifest,
            "calibration": calibration.provenance(calibration_path),
        }
        (temporary_root / "meta" / "conversion.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        os.replace(temporary_root, output_root)
    except BaseException:
        if output is not None and not finalized:
            with suppress(Exception):
                output.finalize()
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
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
