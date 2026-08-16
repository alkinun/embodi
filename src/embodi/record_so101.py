from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
from pathlib import Path
import sys
import tempfile


RECORDER_FORMAT = "embodi-so101-sent-action-v1"
LEROBOT_VERSION = "0.6.1"
MANIFEST_NAME = "embodi-recording.json"
RECORDING_MAX_RELATIVE_TARGET = 5.0


def selected_robot_type(arguments: list[str]) -> str | None:
    selected = None
    for index, argument in enumerate(arguments):
        if argument.startswith("--robot.type="):
            selected = argument.split("=", 1)[1]
        elif argument == "--robot.type" and index + 1 < len(arguments):
            selected = arguments[index + 1]
    return selected


def selected_option(arguments: list[str], name: str) -> str | None:
    selected = None
    prefix = f"{name}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            selected = argument.split("=", 1)[1]
        elif argument == name and index + 1 < len(arguments):
            selected = arguments[index + 1]
    return selected


def validate_recording_arguments(arguments: list[str]) -> None:
    if selected_robot_type(arguments) != "so101_follower":
        raise ValueError("embodi-record-so101 requires an explicit --robot.type=so101_follower")
    if selected_option(arguments, "--dataset.push_to_hub") != "false":
        raise ValueError("embodi-record-so101 requires --dataset.push_to_hub=false")
    if selected_option(arguments, "--resume") not in (None, "false"):
        raise ValueError("embodi-record-so101 does not permit --resume")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_inventory(root: Path, excluded_paths: set[str] | None = None) -> list[dict]:
    excluded = {f"meta/{MANIFEST_NAME}"} if excluded_paths is None else excluded_paths
    inventory = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and relative.as_posix() not in excluded:
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return inventory


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
        temporary_path.replace(path)
    finally:
        active_error = sys.exception()
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if active_error is None:
                    raise
                active_error.add_note(f"manifest temporary-file cleanup failed: {cleanup_error!r}")


def store_sent_action(send_action, robot, action):
    sent = send_action(robot, action)
    if set(sent) != set(action):
        raise RuntimeError("sent SO101 action keys differ from the requested action keys")
    action.clear()
    action.update(sent)
    return sent


def write_recording_manifest(dataset, robot, calibration_sha256: str) -> None:
    calibration_path = robot.calibration_fpath
    if sha256_file(calibration_path) != calibration_sha256:
        raise RuntimeError("LeRobot calibration file changed during recording")
    if dataset.num_episodes <= 0 or dataset.num_frames <= 0:
        raise RuntimeError("SO101 recording produced no complete episodes")
    features_json = json.dumps(dataset.features, sort_keys=True, separators=(",", ":"))
    manifest = {
        "format_version": 1,
        "recorder": RECORDER_FORMAT,
        "complete": True,
        "hardware_model": "so101_follower",
        "stored_action": "robot.send_action return value",
        "max_relative_target": float(robot.config.max_relative_target),
        "lerobot_version": LEROBOT_VERSION,
        "lerobot_calibration_sha256": calibration_sha256,
        "dataset": {
            "repo_id": dataset.repo_id,
            "fps": dataset.fps,
            "episodes": dataset.num_episodes,
            "frames": dataset.num_frames,
            "features_sha256": hashlib.sha256(features_json.encode()).hexdigest(),
        },
        "files": dataset_inventory(dataset.root),
    }
    atomic_write_json(dataset.root / "meta" / MANIFEST_NAME, manifest)


def verify_lerobot_contract() -> None:
    if version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(f"embodi-record-so101 requires lerobot=={LEROBOT_VERSION}")
    from lerobot.processor import make_default_processors

    teleop_processor, robot_processor, _ = make_default_processors()
    action = {"shoulder_pan.pos": 0.0}
    observation = {}
    teleop_action = teleop_processor((action, observation))
    robot_action = robot_processor((teleop_action, observation))
    if teleop_action is not action or robot_action is not action:
        raise RuntimeError("LeRobot default action processors no longer preserve action identity")


def reject_interactive_calibration(robot) -> None:
    from .physical.preflight import disconnect_torque_off

    try:
        disconnect_torque_off(robot.bus)
    except BaseException as cleanup_error:
        raise RuntimeError(
            "calibration mismatch cleanup failed; motor torque state is unknown"
        ) from cleanup_error
    raise RuntimeError(
        "motor calibration does not match the saved file; stop and rerun calibration and preflight"
    )


def cleanup_recording_robot(robot) -> None:
    cleanup_errors = []
    bus = getattr(robot, "bus", None)
    if bus is not None and bus.is_connected:
        from .physical.preflight import disconnect_torque_off

        try:
            disconnect_torque_off(bus)
        except BaseException as error:
            cleanup_errors.append(error)
    for camera in getattr(robot, "cameras", {}).values():
        if camera.is_connected:
            try:
                camera.disconnect()
            except BaseException as error:
                cleanup_errors.append(error)
    if cleanup_errors:
        raise BaseExceptionGroup("SO101 recording hardware cleanup failed", cleanup_errors)


def main() -> None:
    if "--help" not in sys.argv and "-h" not in sys.argv:
        arguments = sys.argv[1:]
        validate_recording_arguments(arguments)
        verify_lerobot_contract()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.scripts import lerobot_record

    original_make_robot = lerobot_record.make_robot_from_config
    original_send_action = SOFollower.send_action
    original_calibrate = SOFollower.calibrate
    original_resume = LeRobotDataset.__dict__["resume"]
    captured: dict = {}

    def make_robot(config):
        robot = original_make_robot(config)
        if not isinstance(robot, SOFollower):
            raise ValueError("embodi-record-so101 requires an SO follower robot")
        target = robot.config.max_relative_target
        if (
            isinstance(target, bool)
            or not isinstance(target, Real)
            or not math.isfinite(target)
            or float(target) != RECORDING_MAX_RELATIVE_TARGET
        ):
            raise ValueError(
                f"embodi-record-so101 requires --robot.max_relative_target={RECORDING_MAX_RELATIVE_TARGET:g}"
            )
        if not robot.calibration:
            raise FileNotFoundError("no saved LeRobot follower calibration; run lerobot-calibrate first")
        if robot.config.disable_torque_on_disconnect is not True:
            raise ValueError("embodi-record-so101 requires --robot.disable_torque_on_disconnect=true")
        captured["robot"] = robot
        captured["calibration_sha256"] = sha256_file(robot.calibration_fpath)
        return robot

    def send_and_store(self, action):
        # LeRobot 0.6.1 records this same dictionary after send_action returns.
        return store_sent_action(original_send_action, self, action)

    def reject_resume(_cls, *_args, **_kwargs):
        raise ValueError("embodi-record-so101 does not permit --resume; record a new immutable dataset")

    def reject_calibration(self):
        reject_interactive_calibration(self)

    lerobot_record.make_robot_from_config = make_robot
    SOFollower.send_action = send_and_store
    SOFollower.calibrate = reject_calibration
    LeRobotDataset.resume = classmethod(reject_resume)
    try:
        lerobot_record.register_third_party_plugins()
        dataset = lerobot_record.record()
    finally:
        active_error = sys.exception()
        cleanup_error = None
        try:
            if "robot" in captured:
                cleanup_recording_robot(captured["robot"])
        except BaseException as error:
            cleanup_error = error
        finally:
            LeRobotDataset.resume = original_resume
            SOFollower.calibrate = original_calibrate
            SOFollower.send_action = original_send_action
            lerobot_record.make_robot_from_config = original_make_robot
        if cleanup_error is not None:
            if active_error is None:
                raise cleanup_error
            active_error.add_note(f"recording hardware cleanup failed: {cleanup_error!r}")

    if dataset is None or "robot" not in captured:
        raise RuntimeError("SO101 recording did not produce a dataset")
    write_recording_manifest(dataset, captured["robot"], captured["calibration_sha256"])


if __name__ == "__main__":
    main()
