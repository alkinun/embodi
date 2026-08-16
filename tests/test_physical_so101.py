import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodi import EmbodiConfig
from embodi.convert_so101 import convert_dataset, rename_noreplace, verify_recording_manifest
from embodi.physical.calibration import initialize_calibration
from embodi.physical.preflight import (
    disconnect_torque_off,
    observe_hardware,
    verify_torque_disabled,
)
from embodi.physical.safety import SO101SafetyLimits, limit_action, observation_vector
from embodi.physical.so101 import (
    SO101CanonicalCalibration,
    SO101_JOINT_NAMES,
    SO101_POSITION_NAMES,
    resolve_so101_order,
)
from embodi.record_so101 import (
    cleanup_recording_robot,
    dataset_inventory,
    reject_interactive_calibration,
    selected_option,
    selected_robot_type,
    store_sent_action,
    validate_recording_arguments,
    verify_lerobot_contract,
    write_recording_manifest,
)
from embodi.train import validate_lerobot_training_dataset


def write_test_recording_manifest(root, repo_id, calibration_path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)
    robot = SimpleNamespace(
        calibration_fpath=calibration_path,
        config=SimpleNamespace(max_relative_target=5.0),
    )
    write_recording_manifest(
        dataset,
        robot,
        hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    )


def camera_test_frame(index: int) -> np.ndarray:
    base = index * 10
    frame = np.empty((480, 640, 3), dtype=np.uint8)
    frame[:240, :320] = [200 + base, 20 + base, 20 + base]
    frame[:240, 320:] = [20 + base, 180 + base, 20 + base]
    frame[240:, :320] = [20 + base, 20 + base, 160 + base]
    frame[240:, 320:] = [120 + base, 120 + base, 20 + base]
    return frame


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


def test_physical_calibration_requires_complete_measured_fields(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    calibration = SO101CanonicalCalibration(
        source_calibration_sha256="0" * 64,
        physical_validation_complete=True,
    )
    calibration.save(path)
    values = json.loads(path.read_text())
    for field in values:
        incomplete = dict(values)
        incomplete.pop(field)
        path.write_text(json.dumps(incomplete))
        with pytest.raises(ValueError, match="missing SO101 calibration fields"):
            SO101CanonicalCalibration.load(path)
    with pytest.raises(ValueError, match="physical validation"):
        SO101CanonicalCalibration().require_physical_validation()
    with pytest.raises(ValueError, match="chosen from -1 and 1"):
        SO101CanonicalCalibration(joint_signs=(True, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="unsupported"):
        SO101CanonicalCalibration(format_version=True)
    with pytest.raises(ValueError, match="finite values"):
        SO101CanonicalCalibration(joint_offsets_degrees=(False, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="endpoints must be finite"):
        SO101CanonicalCalibration(gripper_closed_native=False)
    with pytest.raises(ValueError, match="duplicate SO101 calibration field"):
        SO101CanonicalCalibration.from_json(
            '{"format_version": 1, "format_version": 1}'
        )


def test_calibration_initializer_hashes_source_and_refuses_overwrite(tmp_path) -> None:
    source = tmp_path / "lerobot.json"
    output = tmp_path / "canonical.json"
    source.write_text('{"calibration": true}\n')
    initialize_calibration(source, output)
    calibration = SO101CanonicalCalibration.load(output)
    assert calibration.source_calibration_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert calibration.physical_validation_complete is False
    with pytest.raises(FileExistsError):
        initialize_calibration(source, output)


def test_gripper_calibration_rejects_saturation_beyond_tolerance() -> None:
    calibration = SO101CanonicalCalibration(
        gripper_closed_native=20,
        gripper_open_native=80,
    )
    inside = np.zeros(6, dtype=np.float32)
    inside[5] = 19
    assert calibration.to_model_state(inside, SO101_POSITION_NAMES)[5] == 0
    outside = inside.copy()
    outside[5] = 18.9
    with pytest.raises(ValueError, match="outside calibrated endpoints"):
        calibration.to_model_state(outside, SO101_POSITION_NAMES)
    with pytest.raises(ValueError, match="at least 5"):
        SO101CanonicalCalibration(gripper_closed_native=20, gripper_open_native=24)
    with pytest.raises(ValueError, match="mismatch"):
        calibration.to_model_state(np.zeros(6), SO101_JOINT_NAMES)


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
    with pytest.raises(RuntimeError, match="torque disabled"):
        verify_torque_disabled(Bus({name: 0.0 for name in SO101_JOINT_NAMES}))


def test_torque_off_disconnect_falls_back_to_closing_bus() -> None:
    calls = []

    class Bus:
        is_connected = True

        def disable_torque(self, **_kwargs):
            calls.append("disable")

        def sync_read(self, *_args, **_kwargs):
            return {name: 0 for name in SO101_JOINT_NAMES}

        def disconnect(self, *, disable_torque):
            calls.append(("disconnect", disable_torque))
            if disable_torque:
                raise RuntimeError("write failed")
            self.is_connected = False

    with pytest.raises(RuntimeError, match="cleanup failure"):
        disconnect_torque_off(Bus())
    assert calls == ["disable", ("disconnect", True), ("disconnect", False)]


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
    assert (
        selected_option(
            ["--dataset.push_to_hub=true", "--dataset.push_to_hub", "false"],
            "--dataset.push_to_hub",
        )
        == "false"
    )
    validate_recording_arguments(
        ["--robot.type=so101_follower", "--dataset.push_to_hub=false"]
    )
    with pytest.raises(ValueError, match="push_to_hub=false"):
        validate_recording_arguments(["--robot.type=so101_follower"])
    with pytest.raises(ValueError, match="does not permit --resume"):
        validate_recording_arguments(
            [
                "--robot.type=so101_follower",
                "--dataset.push_to_hub=false",
                "--resume=true",
            ]
        )


def test_pinned_lerobot_recording_contract() -> None:
    verify_lerobot_contract()


def test_interactive_calibration_is_rejected_with_torque_off_cleanup() -> None:
    calls = []

    class Bus:
        is_connected = True

        def disable_torque(self, **kwargs):
            calls.append(("disable", kwargs))

        def sync_read(self, *_args, **_kwargs):
            calls.append(("torque", {}))
            return {name: 0 for name in SO101_JOINT_NAMES}

        def disconnect(self, **kwargs):
            calls.append(("disconnect", kwargs))
            self.is_connected = False

    with pytest.raises(RuntimeError, match="rerun calibration and preflight"):
        reject_interactive_calibration(SimpleNamespace(bus=Bus()))
    assert calls == [
        ("disable", {"num_retry": 5}),
        ("torque", {}),
        ("disconnect", {"disable_torque": True}),
    ]


def test_recording_cleanup_disables_torque_before_camera_disconnect() -> None:
    calls = []

    class Camera:
        is_connected = True

        def disconnect(self):
            calls.append("camera")
            self.is_connected = False

    class Bus:
        is_connected = True

        def disable_torque(self, **_kwargs):
            calls.append("disable")

        def sync_read(self, *_args, **_kwargs):
            calls.append("torque")
            return {name: 0 for name in SO101_JOINT_NAMES}

        def disconnect(self, **_kwargs):
            calls.append("bus")
            self.is_connected = False

    cleanup_recording_robot(
        SimpleNamespace(cameras={"top": Camera()}, bus=Bus())
    )
    assert calls == ["disable", "torque", "bus", "camera"]


def test_recording_manifest_detects_file_and_identity_changes(tmp_path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data.bin").write_bytes(b"recorded")
    calibration_path = tmp_path / "lerobot-calibration.json"
    calibration_path.write_text("{}\n")
    features = {"observation.state": {"dtype": "float32", "shape": (6,)}}
    dataset = SimpleNamespace(
        root=root,
        repo_id="embodi/source",
        fps=30,
        num_episodes=1,
        num_frames=2,
        features=features,
    )
    robot = SimpleNamespace(
        calibration_fpath=calibration_path,
        config=SimpleNamespace(max_relative_target=5.0),
    )
    calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    write_recording_manifest(dataset, robot, calibration_sha256)
    metadata = SimpleNamespace(
        fps=30,
        total_episodes=1,
        total_frames=2,
        features=features,
    )
    verify_recording_manifest(root, "embodi/source", metadata, calibration_sha256)
    with pytest.raises(ValueError, match="identity or counts"):
        verify_recording_manifest(root, "embodi/other", metadata, calibration_sha256)
    (root / "data.bin").write_bytes(b"modified")
    with pytest.raises(ValueError, match="immutable manifest"):
        verify_recording_manifest(root, "embodi/source", metadata, calibration_sha256)
    assert dataset_inventory(root)[0]["sha256"] == hashlib.sha256(b"modified").hexdigest()


def test_preflight_observes_with_torque_off_and_always_disconnects(tmp_path, monkeypatch) -> None:
    calibration_path = tmp_path / "lerobot-calibration.json"
    calibration_path.write_text("{}\n")
    calls = []

    class Bus:
        is_connected = False
        is_calibrated = True

        def connect(self):
            calls.append("connect")
            self.is_connected = True

        def disable_torque(self, **_kwargs):
            calls.append("disable")

        def sync_read(self, name, **_kwargs):
            calls.append(name)
            if name == "Torque_Enable":
                return {joint: 0 for joint in SO101_JOINT_NAMES}
            return {joint: 0.0 for joint in SO101_JOINT_NAMES}

        def disconnect(self, **_kwargs):
            calls.append("disconnect")
            self.is_connected = False

    class Robot:
        def __init__(self, config):
            self.config = config
            self.bus = Bus()
            self.calibration = {"present": True}
            self.calibration_fpath = calibration_path

    class Config:
        num_read_retries = 2

        def __init__(self, **_kwargs):
            pass

    import lerobot.robots.so_follower as follower_module

    monkeypatch.setattr(follower_module, "SO101Follower", Robot)
    monkeypatch.setattr(follower_module, "SO101FollowerConfig", Config)
    calibration = SO101CanonicalCalibration(
        source_calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        physical_validation_complete=True,
    )
    report = observe_hardware("port", "robot", 2, calibration)
    assert report["reads"] == 2
    assert report["native_ranges"]["shoulder_pan.pos"] == {"min": 0.0, "max": 0.0}
    assert report["model_ranges"]["gripper.pos"] == {"min": 0.0, "max": 0.0}
    assert calls[0:3] == ["connect", "disable", "Torque_Enable"]
    assert calls[-4:] == ["Torque_Enable", "disable", "Torque_Enable", "disconnect"]
    raw_calibration = SO101CanonicalCalibration(
        source_calibration_sha256=calibration.source_calibration_sha256,
        physical_validation_complete=False,
    )
    raw_report = observe_hardware("port", "robot", 1, raw_calibration, validate_canonical=False)
    assert raw_report["canonical_validation"] is False
    assert raw_report["model_ranges"] is None


def test_recorder_main_stores_sent_action_and_restores_patches(tmp_path, monkeypatch) -> None:
    import embodi.record_so101 as recording
    from lerobot.scripts import lerobot_record
    import lerobot.robots.so_follower.so_follower as follower_module

    calibration_path = tmp_path / "lerobot-calibration.json"
    calibration_path.write_text("{}\n")
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data.parquet").write_bytes(b"complete")
    requested = {name: 10.0 for name in SO101_POSITION_NAMES}

    class FakeSOFollower:
        def __init__(self):
            self.config = SimpleNamespace(
                max_relative_target=5.0,
                disable_torque_on_disconnect=True,
            )
            self.calibration = {"present": True}
            self.calibration_fpath = calibration_path
            self.cameras = {}
            self.bus = SimpleNamespace(is_connected=False)

        def send_action(self, _action):
            return {name: 5.0 for name in SO101_POSITION_NAMES}

        def calibrate(self):
            raise AssertionError("calibration must not run")

    robot = FakeSOFollower()
    dataset = SimpleNamespace(
        root=dataset_root,
        repo_id="embodi/recorded",
        fps=30,
        num_episodes=1,
        num_frames=1,
        features={"observation.state": {"dtype": "float32", "shape": (6,)}},
    )
    original_send = FakeSOFollower.send_action
    original_calibrate = FakeSOFollower.calibrate

    monkeypatch.setattr(follower_module, "SOFollower", FakeSOFollower)
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda _config: robot)
    monkeypatch.setattr(lerobot_record, "register_third_party_plugins", lambda: None)

    def fake_record():
        made_robot = lerobot_record.make_robot_from_config(None)
        made_robot.send_action(requested)
        return dataset

    monkeypatch.setattr(lerobot_record, "record", fake_record)
    monkeypatch.setattr(
        recording.sys,
        "argv",
        [
            "embodi-record-so101",
            "--robot.type=so101_follower",
            "--dataset.push_to_hub=false",
        ],
    )
    recording.main()
    assert requested == {name: 5.0 for name in SO101_POSITION_NAMES}
    assert FakeSOFollower.send_action is original_send
    assert FakeSOFollower.calibrate is original_calibrate
    manifest = json.loads((dataset_root / "meta" / "embodi-recording.json").read_text())
    assert manifest["complete"] is True
    assert manifest["dataset"]["repo_id"] == "embodi/recorded"
    assert manifest["files"] == dataset_inventory(dataset_root)


def test_conversion_rejects_nested_roots_and_no_clobber_publish(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="non-nested"):
        convert_dataset(
            source,
            "embodi/source",
            source / "output",
            "embodi/output",
            tmp_path / "canonical.json",
            tmp_path / "lerobot.json",
        )
    temporary = tmp_path / "temporary"
    destination = tmp_path / "destination"
    temporary.mkdir()
    destination.mkdir()
    (destination / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        rename_noreplace(temporary, destination)
    assert temporary.is_dir()
    assert (destination / "keep").read_text() == "existing"


def test_physical_lerobot_dataset_conversion(tmp_path) -> None:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    calibration_path = tmp_path / "calibration.json"
    lerobot_calibration_path = tmp_path / "lerobot-calibration.json"
    lerobot_calibration_path.write_text("{}\n")
    SO101CanonicalCalibration(
        source_calibration_sha256=hashlib.sha256(lerobot_calibration_path.read_bytes()).hexdigest(),
        physical_validation_complete=True,
    ).save(calibration_path)
    vector_feature = {
        "dtype": "float32",
        "shape": (6,),
        "names": list(SO101_POSITION_NAMES),
    }
    video_feature = {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    }
    source = LeRobotDataset.create(
        repo_id="embodi/test-physical-source",
        root=source_root,
        fps=30,
        robot_type="so_follower",
        features={
            "observation.images.top": video_feature,
            "observation.state": vector_feature,
            "action": vector_feature,
        },
        use_videos=True,
    )
    for episode in range(2):
        for frame in range(2):
            state = np.array([0, -20, 30, 20, 0, frame * 20], dtype=np.float32)
            action = state.copy()
            action[0] += 1
            source.add_frame(
                {
                    "observation.images.top": camera_test_frame(episode * 2 + frame),
                    "observation.state": state,
                    "action": action,
                    "task": f"episode {episode}",
                }
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
    write_test_recording_manifest(
        source_root, "embodi/test-physical-source", lerobot_calibration_path
    )

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
    validate_lerobot_training_dataset(
        metadata,
        EmbodiConfig(camera_names=("top",)),
        "refine",
    )
    assert tuple(metadata.features["canonical_state"]["shape"]) == (1, 10)
    assert tuple(metadata.features["canonical_action"]["shape"]) == (1, 10)
    schema = json.loads((output_root / "meta" / "embodi.json").read_text())
    assert schema["canonical_convention"]["base_frame"] == "baseframe"
    converted = LeRobotDataset(
        "embodi/test-physical-output", root=output_root, return_uint8=True
    )
    assert converted[0]["canonical_state"].shape == (1, 10)
    assert converted[0]["canonical_action"].shape == (1, 10)
    assert converted[0]["observation.images.top"].shape == (3, 480, 640)
    assert converted[0]["observation.images.top"].dtype == torch.uint8
    decoded = converted[0]["observation.images.top"].to(torch.int16)
    top_left = decoded[:, 100, 100]
    top_right = decoded[:, 100, 500]
    bottom_left = decoded[:, 380, 100]
    assert top_left[0] > top_left[1] + 60 and top_left[0] > top_left[2] + 60
    assert top_right[1] > top_right[0] + 60 and top_right[1] > top_right[2] + 60
    assert bottom_left[2] > bottom_left[0] + 60 and bottom_left[2] > bottom_left[1] + 60
    frame_means = [
        float(converted[index]["observation.images.top"].float().mean())
        for index in range(4)
    ]
    assert frame_means[0] < frame_means[1] < frame_means[2] < frame_means[3]
    conversion_path = output_root / "meta" / "conversion.json"
    conversion_report = json.loads(conversion_path.read_text())
    conversion_report["output"]["frames"] += 1
    conversion_path.write_text(json.dumps(conversion_report))
    with pytest.raises(ValueError, match="conversion integrity"):
        validate_lerobot_training_dataset(
            metadata,
            EmbodiConfig(camera_names=("top",)),
            "refine",
        )
    conversion_report["output"]["frames"] -= 1
    conversion_path.write_text(json.dumps(conversion_report))
    data_file = next((output_root / "data").rglob("*.parquet"))
    data_file.write_bytes(data_file.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="conversion integrity"):
        validate_lerobot_training_dataset(
            metadata,
            EmbodiConfig(camera_names=("top",)),
            "refine",
        )


def test_failed_conversion_removes_temporary_dataset(tmp_path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    lerobot_calibration_path = tmp_path / "lerobot-calibration.json"
    lerobot_calibration_path.write_text("{}\n")
    calibration_path = tmp_path / "calibration.json"
    SO101CanonicalCalibration(
        source_calibration_sha256=hashlib.sha256(lerobot_calibration_path.read_bytes()).hexdigest(),
        physical_validation_complete=True,
    ).save(calibration_path)
    vector_feature = {
        "dtype": "float32",
        "shape": (6,),
        "names": list(SO101_POSITION_NAMES),
    }
    video_feature = {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    }
    source = LeRobotDataset.create(
        repo_id="embodi/test-invalid-source",
        root=source_root,
        fps=30,
        robot_type="so_follower",
        features={
            "observation.images.top": video_feature,
            "observation.state": vector_feature,
            "action": vector_feature,
        },
        use_videos=True,
    )
    state = np.zeros(6, dtype=np.float32)
    action = state.copy()
    action[0] = 200
    source.add_frame(
        {
            "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.state": state,
            "action": action,
            "task": "invalid",
        }
    )
    source.save_episode(parallel_encoding=False)
    source.finalize()
    write_test_recording_manifest(
        source_root, "embodi/test-invalid-source", lerobot_calibration_path
    )
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
    assert not list(tmp_path.glob(f".{output_root.name}.*.tmp"))
    assert not output_root.with_name(f".{output_root.name}.lock").exists()
