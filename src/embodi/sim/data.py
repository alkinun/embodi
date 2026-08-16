from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
from tqdm.auto import tqdm

from .environment import JOINT_NAMES, SO101PickPlaceEnv, TASK
from .expert import Phase, PrivilegedPickPlaceExpert, run_expert_episode


CubeXBin = tuple[float, float, int]


def build_cube_x_schedule(
    episodes: int,
    cube_x_range: tuple[float, float],
    cube_x_bins: tuple[CubeXBin, ...] | None,
    stratified_tail_episodes: int,
    seed: int,
) -> tuple[tuple[float, float], ...]:
    if not cube_x_bins:
        if stratified_tail_episodes:
            raise ValueError("a stratified tail requires --cube-x-bin")
        return (tuple(cube_x_range),) * episodes
    if stratified_tail_episodes < 0 or stratified_tail_episodes >= episodes:
        raise ValueError("stratified tail episodes must be between 0 and episodes - 1")
    if any(
        count <= 0 or not np.isfinite((lower, upper)).all() or lower > upper
        for lower, upper, count in cube_x_bins
    ):
        raise ValueError("cube x bins require finite ascending ranges and positive counts")
    if sum(count for _lower, _upper, count in cube_x_bins) != episodes:
        raise ValueError("cube x bin counts must sum to episodes")

    exact_tail_counts = [count * stratified_tail_episodes / episodes for *_range, count in cube_x_bins]
    tail_counts = [int(value) for value in exact_tail_counts]
    remaining = stratified_tail_episodes - sum(tail_counts)
    remainder_order = sorted(
        range(len(cube_x_bins)),
        key=lambda index: (exact_tail_counts[index] - tail_counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        tail_counts[index] += 1

    training_schedule = []
    tail_schedule = []
    for (lower, upper, count), tail_count in zip(cube_x_bins, tail_counts, strict=True):
        cube_range = (lower, upper)
        training_schedule.extend([cube_range] * (count - tail_count))
        tail_schedule.extend([cube_range] * tail_count)
    schedule_random = random.Random(seed)
    schedule_random.shuffle(training_schedule)
    schedule_random.shuffle(tail_schedule)
    return tuple(training_schedule + tail_schedule)


def evaluate_oracle(
    episodes: int,
    seed: int,
    cube_x_range: tuple[float, float] = (0.28, 0.32),
    cube_y_range: tuple[float, float] = (-0.025, 0.025),
) -> float:
    env = SO101PickPlaceEnv(
        width=64,
        height=48,
        seed=seed,
        cube_x_range=cube_x_range,
        cube_y_range=cube_y_range,
    )
    successes = 0
    try:
        progress = tqdm(range(episodes), desc="oracle evaluation", unit="episode")
        for index in progress:
            env.reset()
            success, _frames = run_expert_episode(env)
            successes += int(success)
            progress.set_postfix(success_rate=f"{successes / (index + 1):.1%}")
    finally:
        env.close()
    return successes / episodes


def generate_dataset(
    root: Path,
    repo_id: str,
    episodes: int,
    seed: int,
    max_attempts: int,
    camera_names: tuple[str, ...],
    cube_x_range: tuple[float, float] = (0.28, 0.32),
    cube_y_range: tuple[float, float] = (-0.025, 0.025),
    cube_x_bins: tuple[CubeXBin, ...] | None = None,
    stratified_tail_episodes: int = 0,
) -> None:
    cube_x_schedule = build_cube_x_schedule(
        episodes,
        cube_x_range,
        cube_x_bins,
        stratified_tail_episodes,
        seed,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    camera_features = {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }
        for name in camera_names
    }
    features = {
        **camera_features,
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(JOINT_NAMES),
        },
        "canonical_action": {
            "dtype": "float32",
            "shape": (1, 10),
        },
        "canonical_state": {
            "dtype": "float32",
            "shape": (1, 10),
        },
        "canonical_part_mask": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["primary_effector"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=30,
        robot_type="so101_sim",
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )
    env = None
    progress = None
    episode_buffered = False
    primary_failed = False
    successes = 0
    attempts = 0
    accepted_initial_cube_xy = []
    try:
        schema_path = root / "meta" / "embodi.json"
        schema_path.write_text(
            json.dumps(
                {
                    "format_version": 3,
                    "parts": [
                        {
                            "name": "primary_effector",
                            "kind": "pose_scalar",
                            "channel_mask": [True] * 10,
                        }
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        env = SO101PickPlaceEnv(
            seed=seed,
            cube_x_range=cube_x_schedule[0],
            cube_y_range=cube_y_range,
        )
        progress = tqdm(total=episodes, desc="synthetic demonstrations", unit="episode")
        while successes < episodes and attempts < max_attempts:
            attempts += 1
            env.cube_x_range = cube_x_schedule[successes]
            env.reset()
            initial_cube_xy = env.data.xpos[env.cube_body_id, :2].tolist()
            expert = PrivilegedPickPlaceExpert(env)
            for _ in range(750):
                state = env.state()
                action = expert.action()
                canonical_action = env.canonical_arm_action(action)
                canonical_state = env.canonical_state()
                if camera_names == ("top",):
                    camera_frames = {"top": env.render_top()}
                elif camera_names == ("wrist",):
                    camera_frames = {"wrist": env.render_wrist()}
                else:
                    top, wrist = env.render_cameras()
                    camera_frames = {"top": top, "wrist": wrist}
                episode_buffered = True
                dataset.add_frame(
                    {
                        **{
                            f"observation.images.{name}": camera_frames[name]
                            for name in camera_names
                        },
                        "observation.state": state.astype(np.float32, copy=False),
                        "action": action.astype(np.float32, copy=False),
                        "canonical_action": canonical_action,
                        "canonical_state": canonical_state,
                        "canonical_part_mask": np.array([True]),
                        "task": TASK,
                    }
                )
                env.apply_action(action)
                env.step_control_period(30.0)
                if expert.terminal:
                    break
            if expert.phase == Phase.DONE:
                dataset.save_episode(parallel_encoding=False)
                episode_buffered = False
                accepted_initial_cube_xy.append(initial_cube_xy)
                successes += 1
                progress.update()
            else:
                dataset.clear_episode_buffer()
                episode_buffered = False
            progress.set_postfix(attempts=attempts, success_rate=f"{successes / attempts:.1%}")
        if successes < episodes:
            raise RuntimeError(f"generated {successes}/{episodes} episodes after {attempts} attempts")
    except BaseException:
        primary_failed = True
        if episode_buffered:
            try:
                dataset.clear_episode_buffer()
            except BaseException:
                pass
        raise
    finally:
        cleanup_error = None
        if progress is not None:
            try:
                progress.close()
            except BaseException as error:
                cleanup_error = error
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        try:
            dataset.finalize()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None and not primary_failed:
            raise cleanup_error
    generation_report = {
        "seed": seed,
        "episodes": episodes,
        "attempts": attempts,
        "camera_names": list(camera_names),
        "cube_x_range": [
            min(cube_range[0] for cube_range in cube_x_schedule),
            max(cube_range[1] for cube_range in cube_x_schedule),
        ],
        "cube_y_range": list(cube_y_range),
        "accepted_initial_cube_xy": accepted_initial_cube_xy,
    }
    if cube_x_bins:
        generation_report["cube_x_bins"] = [
            {"range": [lower, upper], "episodes": count}
            for lower, upper, count in cube_x_bins
        ]
        generation_report["stratified_tail_episodes"] = stratified_tail_episodes
    generation_path = root / "meta" / "generation.json"
    generation_path.write_text(json.dumps(generation_report, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate privileged synthetic pick-place demonstrations")
    parser.add_argument("--root", type=Path, default=Path("datasets/embodi-sim-pickplace"))
    parser.add_argument("--repo-id", default="embodi/sim-so101-pickplace")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", choices=("top", "wrist"), default=["top"])
    parser.add_argument("--cube-x-range", type=float, nargs=2, default=(0.28, 0.32))
    parser.add_argument(
        "--cube-x-bin",
        type=float,
        nargs=3,
        action="append",
        metavar=("MIN", "MAX", "EPISODES"),
        help="repeat for an exact stratified cube-x distribution",
    )
    parser.add_argument("--stratified-tail-episodes", type=int, default=0)
    parser.add_argument("--cube-y-range", type=float, nargs=2, default=(-0.025, 0.025))
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    cube_x_range = tuple(args.cube_x_range)
    cube_y_range = tuple(args.cube_y_range)
    if args.evaluate_only:
        rate = evaluate_oracle(args.episodes, args.seed, cube_x_range, cube_y_range)
        print(f"oracle_success_rate={rate:.1%} episodes={args.episodes}")
        return
    max_attempts = args.max_attempts or max(args.episodes * 2, args.episodes + 10)
    camera_names = tuple(args.cameras)
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("--cameras must contain unique camera names")
    cube_x_bins = None
    if args.cube_x_bin:
        if any(not count.is_integer() for _lower, _upper, count in args.cube_x_bin):
            raise ValueError("cube x bin episode counts must be integers")
        cube_x_bins = tuple(
            (lower, upper, int(count)) for lower, upper, count in args.cube_x_bin
        )
    generate_dataset(
        args.root,
        args.repo_id,
        args.episodes,
        args.seed,
        max_attempts,
        camera_names,
        cube_x_range,
        cube_y_range,
        cube_x_bins,
        args.stratified_tail_episodes,
    )
    print(f"dataset_root={args.root} repo_id={args.repo_id} episodes={args.episodes}")


if __name__ == "__main__":
    main()
