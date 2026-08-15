import hashlib
import json

import numpy as np
import pytest

from embodi.convert_so101 import convert_dataset
from embodi.physical.preflight import verify_torque_disabled
from embodi.physical.safety import SO101SafetyLimits, limit_action, observation_vector
from embodi.physical.so101 import (
    SO101CanonicalCalibration,
    SO101_JOINT_NAMES,
    SO101_POSITION_NAMES,
    resolve_so101_order,
)
from embodi.record_so101 import RECORDER_FORMAT, selected_robot_type, store_sent_action


def write_recording_manifest(root, calibration_path) -> None:
    (root / "meta" / "embodi-recording.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "recorder": RECORDER_FORMAT,
                "hardware_model": "so101_follower",
                "stored_action": "robot.send_action return value",
                "max_relative_target": 5.0,
                "lerobot_calibration_sha256": hashlib.sha256(
                    calibration_path.read_bytes()
                ).hexdigest(),
            }
        )
    )


def test_so101_order_accepts_physical_names_and_reorders() -> None:
    names = list(reversed(SO101_POSITION_NAMES))
    order = resolve_so101_order(names)
    assert tuple(names[index] for index in order) == SO101_POSITION_NAMES
    with pytest.raises(ValueError, match="mismatch"):
        resolve_so101_order(names[:-1])


def test_canonical_calibration_applies_sign_offset_and_gripper_direction() -> None:
    calibration = SO101CanonicalCalibration(
        joint_signs=(-1, 1, 1, 1, 1),
        joint_offsets_degrees=(10, 0, 0, 0, 0),
        gripper_closed_native=100,
        gripper_open_native=20,
    )
    native = np.array([5, 2, 3, 4, 5, 60], dtype=np.float32)
    model = calibration.to_model_state(native, SO101_POSITION_NAMES)
    np.testing.assert_allclose(model, [5, 2, 3, 4, 5, 17.5])


def test_physical_observation_and_action_limits_fail_closed() -> None:
    observation = {name: 0.0 for name in SO101_POSITION_NAMES}
    current = observation_vector(observation)
    requested = np.array([20, -20, 1, 1, 1, 50], dtype=np.float32)
    bounded, clipped = limit_action(requested, current, current, SO101SafetyLimits())
    np.testing.assert_allclose(bounded, [0.5, -0.5, 0.5, 0.5, 0.5, 1.0])
    assert set(clipped) == set(SO101_POSITION_NAMES)
    observation["gripper.pos"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        observation_vector(observation)
    with pytest.raises(ValueError, match="left the startup-centered session envelope"):
        limit_action(current, current + np.array([10, 0, 0, 0, 0, 0]), current)
    with pytest.raises(ValueError, match="finite and positive"):
        SO101SafetyLimits(body_delta_degrees=float("nan"))


def test_torque_readback_and_sent_action_recording_fail_closed() -> None:
    class Bus:
        def __init__(self, values):
            self.values = values

        def sync_read(self, *_args, **_kwargs):
            return self.values

    verify_torque_disabled(Bus({name: 0 for name in SO101_JOINT_NAMES}))
    with pytest.raises(RuntimeError, match="all six"):
        verify_torque_disabled(Bus({name: 0 for name in SO101_JOINT_NAMES[:-1]}))
    with pytest.raises(RuntimeError, match="torque disabled"):
        verify_torque_disabled(Bus({name: int(name == "gripper") for name in SO101_JOINT_NAMES}))


def test_sent_action_replaces_requested_action() -> None:
    requested = {name: 10.0 for name in SO101_POSITION_NAMES}

    def send(_robot, _action):
        return {name: 5.0 for name in SO101_POSITION_NAMES}

    sent = store_sent_action(send, object(), requested)
    assert requested == sent == {name: 5.0 for name in SO101_POSITION_NAMES}


def test_last_robot_type_argument_controls_hardware_identity() -> None:
    assert selected_robot_type(["--robot.type=so101_follower"]) == "so101_follower"
    assert (
        selected_robot_type(
            ["--robot.type=so101_follower", "--robot.type", "so100_follower"]
        )
        == "so100_follower"
    )


def test_physical_lerobot_dataset_conversion(tmp_path) -> None:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    calibration_path = tmp_path / "calibration.json"
    lerobot_calibration_path = tmp_path / "lerobot-calibration.json"
    lerobot_calibration_path.write_text("{}\n")
    SO101CanonicalCalibration(
        source_calibration_sha256=hashlib.sha256(lerobot_calibration_path.read_bytes()).hexdigest()
    ).save(calibration_path)
    vector_feature = {
        "dtype": "float32",
        "shape": (6,),
        "names": list(SO101_POSITION_NAMES),
    }
    source = LeRobotDataset.create(
        repo_id="embodi/test-physical-source",
        root=source_root,
        fps=30,
        robot_type="so_follower",
        features={"observation.state": vector_feature, "action": vector_feature},
        use_videos=False,
    )
    for episode in range(2):
        for frame in range(2):
            state = np.array([0, -20, 30, 20, 0, frame * 20], dtype=np.float32)
            action = state.copy()
            action[0] += 1
            source.add_frame(
                {"observation.state": state, "action": action, "task": f"episode {episode}"}
            )
        source.save_episode(parallel_encoding=False)
    source.finalize()
    with pytest.raises(ValueError, match="not recorded with embodi-record-so101"):
        convert_dataset(
            source_root,
            "embodi/test-physical-source",
            output_root,
            "embodi/test-physical-output",
            calibration_path,
            lerobot_calibration_path,
        )
    write_recording_manifest(source_root, lerobot_calibration_path)

    report = convert_dataset(
        source_root,
        "embodi/test-physical-source",
        output_root,
        "embodi/test-physical-output",
        calibration_path,
        lerobot_calibration_path,
    )
    assert report["frames"] == 4
    assert report["episodes"] == 2
    metadata = LeRobotDatasetMetadata("embodi/test-physical-output", root=output_root)
    assert tuple(metadata.features["canonical_state"]["shape"]) == (1, 10)
    assert tuple(metadata.features["canonical_action"]["shape"]) == (1, 10)
    schema = json.loads((output_root / "meta" / "embodi.json").read_text())
    assert schema["canonical_convention"]["base_frame"] == "baseframe"
    converted = LeRobotDataset("embodi/test-physical-output", root=output_root)
    assert converted[0]["canonical_state"].shape == (1, 10)
    assert converted[0]["canonical_action"].shape == (1, 10)


def test_failed_conversion_removes_temporary_dataset(tmp_path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    lerobot_calibration_path = tmp_path / "lerobot-calibration.json"
    lerobot_calibration_path.write_text("{}\n")
    calibration_path = tmp_path / "calibration.json"
    SO101CanonicalCalibration(
        source_calibration_sha256=hashlib.sha256(lerobot_calibration_path.read_bytes()).hexdigest()
    ).save(calibration_path)
    vector_feature = {
        "dtype": "float32",
        "shape": (6,),
        "names": list(SO101_POSITION_NAMES),
    }
    source = LeRobotDataset.create(
        repo_id="embodi/test-invalid-source",
        root=source_root,
        fps=30,
        robot_type="so_follower",
        features={"observation.state": vector_feature, "action": vector_feature},
        use_videos=False,
    )
    state = np.zeros(6, dtype=np.float32)
    action = state.copy()
    action[0] = 200
    source.add_frame({"observation.state": state, "action": action, "task": "invalid"})
    source.save_episode(parallel_encoding=False)
    source.finalize()
    write_recording_manifest(source_root, lerobot_calibration_path)
    with pytest.raises(ValueError, match="model limits"):
        convert_dataset(
            source_root,
            "embodi/test-invalid-source",
            output_root,
            "embodi/test-invalid-output",
            calibration_path,
            lerobot_calibration_path,
        )
    assert not output_root.exists()
    assert not output_root.with_name(f".{output_root.name}.tmp").exists()
