from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
import sys


RECORDER_FORMAT = "embodi-so101-sent-action-v1"


def selected_robot_type(arguments: list[str]) -> str | None:
    selected = None
    for index, argument in enumerate(arguments):
        if argument.startswith("--robot.type="):
            selected = argument.split("=", 1)[1]
        elif argument == "--robot.type" and index + 1 < len(arguments):
            selected = arguments[index + 1]
    return selected


def store_sent_action(send_action, robot, action):
    sent = send_action(robot, action)
    if set(sent) != set(action):
        raise RuntimeError("sent SO101 action keys differ from the requested action keys")
    action.clear()
    action.update(sent)
    return sent


def write_recording_manifest(dataset, robot) -> None:
    calibration_path = robot.calibration_fpath
    manifest = {
        "format_version": 1,
        "recorder": RECORDER_FORMAT,
        "hardware_model": "so101_follower",
        "stored_action": "robot.send_action return value",
        "max_relative_target": float(robot.config.max_relative_target),
        "lerobot_calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
    }
    (dataset.root / "meta" / "embodi-recording.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    if "--help" not in sys.argv and "-h" not in sys.argv:
        if selected_robot_type(sys.argv[1:]) != "so101_follower":
            raise ValueError("embodi-record-so101 requires an explicit --robot.type=so101_follower")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.scripts import lerobot_record

    original_make_robot = lerobot_record.make_robot_from_config
    original_send_action = SOFollower.send_action
    original_push_to_hub = LeRobotDataset.push_to_hub
    original_resume = LeRobotDataset.__dict__["resume"]
    captured: dict = {}

    def make_robot(config):
        robot = original_make_robot(config)
        if not isinstance(robot, SOFollower):
            raise ValueError("embodi-record-so101 requires an SO follower robot")
        target = robot.config.max_relative_target
        if isinstance(target, bool) or not isinstance(target, Real) or not math.isfinite(target) or target <= 0:
            raise ValueError("embodi-record-so101 requires a finite positive --robot.max_relative_target")
        if not robot.calibration:
            raise FileNotFoundError("no saved LeRobot follower calibration; run lerobot-calibrate first")
        captured["robot"] = robot
        return robot

    def send_and_store(self, action):
        # LeRobot 0.6.1 records this same dictionary after send_action returns.
        return store_sent_action(original_send_action, self, action)

    def push_with_manifest(dataset, *args, **kwargs):
        if "robot" not in captured:
            raise RuntimeError("SO101 recording provenance is unavailable")
        write_recording_manifest(dataset, captured["robot"])
        return original_push_to_hub(dataset, *args, **kwargs)

    def reject_resume(_cls, *_args, **_kwargs):
        raise ValueError("embodi-record-so101 does not permit --resume; record a new immutable dataset")

    lerobot_record.make_robot_from_config = make_robot
    SOFollower.send_action = send_and_store
    LeRobotDataset.push_to_hub = push_with_manifest
    LeRobotDataset.resume = classmethod(reject_resume)
    try:
        lerobot_record.register_third_party_plugins()
        dataset = lerobot_record.record()
    finally:
        LeRobotDataset.resume = original_resume
        LeRobotDataset.push_to_hub = original_push_to_hub
        SOFollower.send_action = original_send_action
        lerobot_record.make_robot_from_config = original_make_robot

    if dataset is None or "robot" not in captured:
        raise RuntimeError("SO101 recording did not produce a dataset")
    write_recording_manifest(dataset, captured["robot"])


if __name__ == "__main__":
    main()
