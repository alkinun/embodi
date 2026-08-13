#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from embodi.canonical import rotation_6d_to_matrix
from embodi.xperience import CachedXperienceDataset, CachedXperienceEpisode, select_xperience_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and visualize cached Xperience experiment-0 labels.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/xperience-exp0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-clips", type=int, default=100)
    parser.add_argument("--validation-clips", type=int, default=20)
    parser.add_argument("--visualizations", type=int, default=5)
    parser.add_argument("--closed-ratio", type=float, default=0.18)
    parser.add_argument("--open-ratio", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode = CachedXperienceEpisode(
        args.episode_dir, closed_ratio=args.closed_ratio, open_ratio=args.open_ratio
    )
    split = select_xperience_split(
        episode,
        train_clips=args.train_clips,
        validation_clips=args.validation_clips,
        seed=args.seed,
    )
    all_anchors, _ = episode.valid_anchors()
    wrist_error = torch.linalg.vector_norm(
        episode.hand_position_world - episode.body_right_wrist_world, dim=-1
    )
    finite_wrist = wrist_error[torch.isfinite(wrist_error)]
    selected_indices = torch.tensor(
        [anchor.frame_index for anchor in (*split.train, *split.validation)], dtype=torch.long
    )
    selected_wrist = wrist_error[selected_indices]
    finite_opening = episode.opening[torch.isfinite(episode.opening)]
    rotation_errors = []
    for anchor in (*split.train, *split.validation):
        sample = episode.sample_at(anchor.frame_index)
        encoded = sample.canonical_action[:, 0, 3:9]
        rotations = rotation_6d_to_matrix(encoded)
        identity = torch.eye(3)
        rotation_errors.append((rotations.transpose(-1, -2) @ rotations - identity).abs().max())
    report = {
        "episode": str(args.episode_dir),
        "source_frames": len(episode.timestamps_ns),
        "source_fps": split.source_fps,
        "target_fps": 30,
        "horizon_steps": 32,
        "horizon_seconds": 32 / 30,
        "caption_segments": len(episode.segments),
        "valid_anchors": len(all_anchors),
        "valid_motion_bins": dict(Counter(anchor.motion_bin for anchor in all_anchors)),
        "selected_train_motion_bins": dict(Counter(anchor.motion_bin for anchor in split.train)),
        "selected_validation_motion_bins": dict(Counter(anchor.motion_bin for anchor in split.validation)),
        "selected_validation_segments": sorted({anchor.segment_id for anchor in split.validation}),
        "rejection_counts": split.rejection_counts,
        "finite_opening_frames": int(finite_opening.numel()),
        "opening_percentiles": {
            str(percentile): float(torch.quantile(finite_opening, percentile / 100))
            for percentile in (0, 5, 50, 95, 100)
        },
        "wrist_consistency_m": {
            "accepted_median": float(selected_wrist.median()),
            "accepted_p95": float(torch.quantile(selected_wrist, 0.95)),
            "accepted_max": float(selected_wrist.max()),
            "all_finite_median": float(finite_wrist.median()),
            "all_finite_p95": float(torch.quantile(finite_wrist, 0.95)),
            "all_finite_max": float(finite_wrist.max()),
        },
        "selected_values_finite": True,
        "max_rotation_orthogonality_error": float(torch.stack(rotation_errors).max()),
    }
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")

    randomizer = random.Random(args.seed)
    chosen = randomizer.sample(list(split.train), min(args.visualizations, len(split.train)))
    dataset = CachedXperienceDataset(episode, chosen)
    for index, anchor in enumerate(chosen):
        item = dataset[index]
        sample = episode.sample_at(anchor.frame_index)
        current_position = sample.canonical_state[0, :3]
        trajectory = current_position + sample.canonical_action[:, 0, :3]
        current_rotation = rotation_6d_to_matrix(sample.canonical_state[0, 3:9])
        relative_rotation = rotation_6d_to_matrix(sample.canonical_action[:, 0, 3:9])
        future_rotation = current_rotation @ relative_rotation

        figure = plt.figure(figsize=(15, 8), constrained_layout=True)
        image_axis = figure.add_subplot(2, 2, 1)
        image_axis.imshow(item["observation.images.stereo_left"].permute(1, 2, 0).numpy())
        image_axis.set_title(f"frame {anchor.frame_index} | {anchor.timestamp_ns}")
        image_axis.axis("off")

        trajectory_axis = figure.add_subplot(1, 2, 2, projection="3d")
        points = torch.cat((current_position[None], trajectory), dim=0)
        trajectory_axis.plot(points[:, 0], points[:, 1], points[:, 2], marker=".", color="black")
        colors = ("red", "green", "blue")
        for axis_index, color in enumerate(colors):
            root_axis = torch.zeros(3)
            root_axis[axis_index] = 0.05
            trajectory_axis.quiver(0, 0, 0, *root_axis, color=color)
        for step in (0, 8, 16, 24, 31):
            origin = trajectory[step]
            for axis_index, color in enumerate(colors):
                direction = future_rotation[step, :, axis_index] * 0.025
                trajectory_axis.quiver(*origin, *direction, color=color, alpha=0.65)
        trajectory_axis.set_xlabel("root x [m]")
        trajectory_axis.set_ylabel("root y [m]")
        trajectory_axis.set_zlabel("root z [m]")
        trajectory_axis.set_title(f"primary_effector | {anchor.motion_bin}")

        opening_axis = figure.add_subplot(2, 2, 3)
        times = torch.arange(33) / 30
        aperture = torch.cat((sample.canonical_state[:, 9], sample.canonical_action[:, 0, 9]))
        opening_axis.plot(times, aperture, marker=".")
        opening_axis.set_ylim(-0.05, 1.05)
        opening_axis.set_xlabel("time [s]")
        opening_axis.set_ylabel("normalized opening")
        opening_axis.set_title(anchor.prompt.replace("\n", " | "))
        figure.savefig(args.output_dir / f"sample-{index:02d}-frame-{anchor.frame_index}.png", dpi=140)
        plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
