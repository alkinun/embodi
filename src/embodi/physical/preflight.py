from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .safety import observation_vector
from .so101 import SO101CanonicalCalibration, SO101_JOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torque-off SO101 dependency and bus preflight.")
    parser.add_argument("--canonical-calibration", type=Path)
    parser.add_argument("--connect-hardware", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--robot-id")
    parser.add_argument("--reads", type=int, default=100)
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
    if set(torque) != set(SO101_JOINT_NAMES) or any(int(value) != 0 for value in torque.values()):
        raise RuntimeError("all six SO101 motors must report torque disabled")


def observe_hardware(
    port: str,
    robot_id: str,
    reads: int,
    canonical_calibration: SO101CanonicalCalibration,
) -> dict:
    if reads <= 0:
        raise ValueError("--reads must be positive")
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras={}))
    if not robot.calibration:
        raise FileNotFoundError(
            f"no LeRobot follower calibration found for robot id {robot_id!r}; run lerobot-calibrate first"
        )
    canonical_calibration.verify_source_calibration(robot.calibration_fpath)
    samples = []
    started = time.perf_counter()
    try:
        robot.bus.connect()
        robot.bus.disable_torque(num_retry=5)
        verify_torque_disabled(robot.bus)
        if not robot.bus.is_calibrated:
            raise RuntimeError("motor calibration does not match the saved LeRobot calibration")
        for _ in range(reads):
            positions = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
            observation = {f"{name}.pos": value for name, value in positions.items()}
            sample = observation_vector(observation)
            canonical_calibration.to_model_state(sample, tuple(observation))
            samples.append(sample)
        verify_torque_disabled(robot.bus)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)
    elapsed = time.perf_counter() - started
    return {
        "reads": reads,
        "elapsed_seconds": elapsed,
        "mean_read_ms": elapsed * 1000.0 / reads,
        "first_state": samples[0].tolist(),
        "last_state": samples[-1].tolist(),
        "torque_enabled": False,
    }


def main() -> None:
    args = parse_args()
    dependencies = dependency_report()
    payload: dict = {"dependencies": dependencies, "hardware_connected": False}
    if args.canonical_calibration:
        calibration = SO101CanonicalCalibration.load(args.canonical_calibration)
        payload["canonical_calibration"] = calibration.provenance(args.canonical_calibration)
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
        )
        payload["hardware_connected"] = True
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
