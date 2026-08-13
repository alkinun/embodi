from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .environment import JOINT_NAMES, SO101PickPlaceEnv, TASK
from .expert import Phase, PrivilegedPickPlaceExpert, run_expert_episode


def evaluate_oracle(episodes: int, seed: int) -> float:
    env = SO101PickPlaceEnv(width=64, height=48, seed=seed)
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
) -> None:
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
    env = SO101PickPlaceEnv(seed=seed)
    successes = 0
    attempts = 0
    progress = tqdm(total=episodes, desc="synthetic demonstrations", unit="episode")
    try:
        while successes < episodes and attempts < max_attempts:
            attempts += 1
            env.reset()
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
                successes += 1
                progress.update()
            else:
                dataset.clear_episode_buffer()
            progress.set_postfix(attempts=attempts, success_rate=f"{successes / attempts:.1%}")
        if successes < episodes:
            raise RuntimeError(f"generated {successes}/{episodes} episodes after {attempts} attempts")
    finally:
        progress.close()
        env.close()
        dataset.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate privileged synthetic pick-place demonstrations")
    parser.add_argument("--root", type=Path, default=Path("datasets/embodi-sim-pickplace"))
    parser.add_argument("--repo-id", default="embodi/sim-so101-pickplace")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", choices=("top", "wrist"), default=["top"])
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.evaluate_only:
        rate = evaluate_oracle(args.episodes, args.seed)
        print(f"oracle_success_rate={rate:.1%} episodes={args.episodes}")
        return
    max_attempts = args.max_attempts or max(args.episodes * 2, args.episodes + 10)
    camera_names = tuple(args.cameras)
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("--cameras must contain unique camera names")
    generate_dataset(args.root, args.repo_id, args.episodes, args.seed, max_attempts, camera_names)
    print(f"dataset_root={args.root} repo_id={args.repo_id} episodes={args.episodes}")


if __name__ == "__main__":
    main()
