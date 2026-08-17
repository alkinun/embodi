import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from embodi.xperience import (
    INDEX_MCP_INDEX,
    INDEX_TIP_INDEX,
    PINKY_MCP_INDEX,
    THUMB_TIP_INDEX,
    XPERIENCE_FPS,
    XPERIENCE_HORIZON,
    CachedXperienceEpisode,
)


def _write_episode(root: Path) -> None:
    root.mkdir()
    (root / "stereo_left.mp4").write_bytes(b"synthetic video placeholder")
    count = XPERIENCE_HORIZON + 1
    timestamps = np.rint(np.arange(count) * 1e9 / XPERIENCE_FPS).astype(np.int64)
    root_pose = np.zeros((count, 7), dtype=np.float32)
    root_pose[:, 0] = 1.0
    slam_quaternion = root_pose[:, :4].copy()
    slam_translation = np.zeros((count, 3), dtype=np.float32)
    hand_joints = np.zeros((count, 21, 3), dtype=np.float32)
    hand_joints[:, 0, 0] = np.arange(count, dtype=np.float32) * 0.001
    hand_joints[:, INDEX_MCP_INDEX, 0] = 0.05
    hand_joints[:, PINKY_MCP_INDEX, 0] = 0.0
    hand_joints[:, THUMB_TIP_INDEX, 0] = 0.0
    hand_joints[:, INDEX_TIP_INDEX, 0] = 0.02
    hand_rotation = np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
    body_keypoints = np.zeros((count, 22, 3), dtype=np.float32)
    body_keypoints[:, 21] = hand_joints[:, 0]
    calibration = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    caption = json.dumps(
        {
            "config": {"Main Task": "Pick up the cube"},
            "segments": [
                {
                    "segment_id": 4,
                    "start_frame": "frame_0",
                    "end_frame": "frame_32",
                    "Sub Task": "move the hand",
                }
            ],
        }
    ).encode()

    with h5py.File(root / "annotation.hdf5", "w") as file:
        file.create_dataset("video/device_timestamp", data=timestamps)
        file.create_dataset("full_body_mocap/Ts_world_root", data=root_pose)
        file.create_dataset("slam/quat_wxyz", data=slam_quaternion)
        file.create_dataset("slam/trans_xyz", data=slam_translation)
        file.create_dataset("hand_mocap/right_joints_3d", data=hand_joints)
        file.create_dataset("hand_mocap/right_mano_hand_global_orient", data=hand_rotation)
        file.create_dataset("full_body_mocap/keypoints", data=body_keypoints)
        file.create_dataset("calibration/cam01/T_c0_b", data=calibration)
        file.create_dataset("caption", data=caption)


def test_cached_episode_loads_hdf5_labels_and_builds_canonical_sample(tmp_path: Path) -> None:
    episode_root = tmp_path / "episode-0"
    _write_episode(episode_root)

    episode = CachedXperienceEpisode(episode_root)

    assert episode.video_path == episode_root / "stereo_left.mp4"
    assert episode.main_task == "Pick up the cube"
    assert episode.segments[0]["id"] == 4
    assert episode.segments[0]["prompt"] == "Task: Pick up the cube.\nCurrent subtask: move the hand."
    assert episode.source_fps == pytest.approx(XPERIENCE_FPS, rel=1e-5)
    expected_opening = (0.02 / 0.05 - 0.18) / (1.0 - 0.18)
    assert float(episode.opening[0]) == pytest.approx(expected_opening)

    sample = episode.sample_at(0)

    assert sample.canonical_state.shape == (1, 10)
    assert sample.canonical_action.shape == (XPERIENCE_HORIZON, 1, 10)
    np.testing.assert_allclose(
        sample.canonical_action[:, 0, 0].numpy(),
        np.arange(1, XPERIENCE_HORIZON + 1, dtype=np.float32) * 0.001,
        atol=1e-5,
    )
    assert float(sample.canonical_state[0, 9]) == pytest.approx(expected_opening)
    anchors, rejection_counts = episode.valid_anchors()
    assert [anchor.frame_index for anchor in anchors] == [0]
    assert anchors[0].segment_id == 4
    assert rejection_counts == {"caption_horizon": XPERIENCE_HORIZON}


def test_cached_episode_rejects_invalid_ratio_and_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="annotation.hdf5 or stereo_left.mp4"):
        CachedXperienceEpisode(tmp_path / "missing")

    episode_root = tmp_path / "episode-0"
    _write_episode(episode_root)
    with pytest.raises(ValueError, match="open_ratio must be greater"):
        CachedXperienceEpisode(episode_root, closed_ratio=0.5, open_ratio=0.5)
