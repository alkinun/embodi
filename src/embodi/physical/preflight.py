from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .safety import observation_vector
from .so101 import SO101CanonicalCalibration, SO101_JOINT_NAMES, SO101_POSITION_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torque-off SO101 dependency and bus preflight.")
    parser.add_argument("--canonical-calibration", type=Path)
    parser.add_argument("--connect-hardware", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--robot-id")
    parser.add_argument("--reads", type=int, default=100)
    parser.add_argument("--raw-only", action="store_true")
    return parser.parse_args()


def dependency_report() -> dict:
    report = {}
    checks = {
        "serial": "serial",
        "deepdiff": "deepdiff",
        "feetech_sdk": "scservo_sdk",
    }
    for name, module in checks.items():
        try:
            __import__(module)
        except ImportError:
            report[name] = False
        else:
            report[name] = True
    return report


def verify_torque_disabled(bus) -> None:
    torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=5)
    if set(torque) != set(SO101_JOINT_NAMES) or any(
        type(value) is not int or value != 0 for value in torque.values()
    ):
        raise RuntimeError("all six SO101 motors must report torque disabled")


def disconnect_torque_off(bus) -> None:
    cleanup_error = None
    try:
        bus.disable_torque(num_retry=5)
        verify_torque_disabled(bus)
    except BaseException as error:
        cleanup_error = error
    try:
        bus.disconnect(disable_torque=True)
    except BaseException as error:
        cleanup_error = cleanup_error or error
        if bus.is_connected:
            try:
                bus.disconnect(disable_torque=False)
            except BaseException as fallback_error:
                cleanup_error = cleanup_error or fallback_error
    if cleanup_error is not None:
        raise RuntimeError("SO101 torque-off disconnect encountered a cleanup failure") from cleanup_error


def observe_hardware(
    port: str,
    robot_id: str,
    reads: int,
    canonical_calibration: SO101CanonicalCalibration,
    validate_canonical: bool = True,
) -> dict:
    if reads <= 0:
        raise ValueError("--reads must be positive")
    if validate_canonical:
        canonical_calibration.require_physical_validation()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras={}))
    if not robot.calibration:
        raise FileNotFoundError(
            f"no LeRobot follower calibration found for robot id {robot_id!r}; run lerobot-calibrate first"
        )
    calibration_sha256 = canonical_calibration.verify_source_calibration(robot.calibration_fpath)
    samples = []
    model_samples = []
    started = time.perf_counter()
    try:
        robot.bus.connect()
        robot.bus.disable_torque(num_retry=5)
        verify_torque_disabled(robot.bus)
        if canonical_calibration.verify_source_calibration(robot.calibration_fpath) != calibration_sha256:
            raise RuntimeError("LeRobot calibration file changed during preflight")
        if not robot.bus.is_calibrated:
            raise RuntimeError("motor calibration does not match the saved LeRobot calibration")
        for _ in range(reads):
            positions = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
            observation = {f"{name}.pos": value for name, value in positions.items()}
            sample = observation_vector(observation)
            if validate_canonical:
                model_samples.append(
                    canonical_calibration.to_model_state(sample, tuple(observation))
                )
            samples.append(sample)
        verify_torque_disabled(robot.bus)
        if canonical_calibration.verify_source_calibration(robot.calibration_fpath) != calibration_sha256:
            raise RuntimeError("LeRobot calibration file changed during preflight")
    finally:
        if robot.bus.is_connected:
            disconnect_torque_off(robot.bus)
    if canonical_calibration.verify_source_calibration(robot.calibration_fpath) != calibration_sha256:
        raise RuntimeError("LeRobot calibration file changed during preflight")
    elapsed = time.perf_counter() - started
    native = np.stack(samples)
    native_ranges = {
        name: {"min": float(native[:, index].min()), "max": float(native[:, index].max())}
        for index, name in enumerate(SO101_POSITION_NAMES)
    }
    model_ranges = None
    if model_samples:
        model = np.stack(model_samples)
        model_ranges = {
            name: {"min": float(model[:, index].min()), "max": float(model[:, index].max())}
            for index, name in enumerate(SO101_POSITION_NAMES)
        }
    return {
        "reads": reads,
        "elapsed_seconds": elapsed,
        "mean_read_ms": elapsed * 1000.0 / reads,
        "first_state": samples[0].tolist(),
        "last_state": samples[-1].tolist(),
        "torque_enabled": False,
        "canonical_validation": validate_canonical,
        "lerobot_calibration_sha256": calibration_sha256,
        "native_ranges": native_ranges,
        "model_ranges": model_ranges,
    }


def main() -> None:
    args = parse_args()
    dependencies = dependency_report()
    payload: dict = {"dependencies": dependencies, "hardware_connected": False}
    if args.canonical_calibration:
        calibration, canonical_calibration_sha256 = SO101CanonicalCalibration.load_snapshot(
            args.canonical_calibration
        )
        payload["canonical_calibration"] = calibration.provenance(
            args.canonical_calibration, file_sha256=canonical_calibration_sha256
        )
    if args.connect_hardware:
        if not args.port or not args.robot_id:
            raise ValueError("--connect-hardware requires --port and --robot-id")
        if not all(dependencies.values()):
            raise RuntimeError("SO101 hardware dependencies are missing; run uv sync --extra hardware")
        if not args.canonical_calibration:
            raise ValueError("--connect-hardware requires --canonical-calibration")
        payload["hardware"] = observe_hardware(
            args.port,
            args.robot_id,
            args.reads,
            calibration,
            validate_canonical=not args.raw_only,
        )
        payload["hardware_connected"] = True
        if (
            hashlib.sha256(args.canonical_calibration.read_bytes()).hexdigest()
            != canonical_calibration_sha256
        ):
            raise RuntimeError("canonical calibration file changed during preflight")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
