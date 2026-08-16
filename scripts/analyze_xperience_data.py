#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main_task(prompt: str) -> str:
    first_line = prompt.splitlines()[0]
    if first_line.startswith("Task: "):
        return first_line.removeprefix("Task: ").removesuffix(".")
    return first_line


def temporal_metrics(records: list[dict[str, Any]], horizon_seconds: float) -> dict[str, Any]:
    timestamps_by_episode: dict[str, list[float]] = defaultdict(list)
    for record in records:
        timestamps_by_episode[record["episode_path"]].append(record["timestamp_ns"] / 1e9)

    union_seconds = 0.0
    nonoverlapping_windows = 0
    per_episode = []
    for episode_path, timestamps in sorted(timestamps_by_episode.items()):
        ordered = sorted(timestamps)
        merged: list[list[float]] = []
        for start in ordered:
            end = start + horizon_seconds
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        episode_union = sum(end - start for start, end in merged)
        episode_nonoverlapping = 0
        next_allowed = float("-inf")
        for start in ordered:
            if start >= next_allowed:
                episode_nonoverlapping += 1
                next_allowed = start + horizon_seconds
        union_seconds += episode_union
        nonoverlapping_windows += episode_nonoverlapping
        per_episode.append(
            {
                "episode_path": episode_path,
                "clips": len(ordered),
                "union_labeled_seconds": episode_union,
                "nonoverlapping_window_capacity": episode_nonoverlapping,
            }
        )

    labeled_seconds = len(records) * horizon_seconds
    return {
        "labeled_window_seconds": labeled_seconds,
        "union_labeled_seconds": union_seconds,
        "window_overlap_factor": labeled_seconds / union_seconds if union_seconds else 0.0,
        "nonoverlapping_window_capacity": nonoverlapping_windows,
        "clips_per_nonoverlapping_window": (
            len(records) / nonoverlapping_windows if nonoverlapping_windows else 0.0
        ),
        "episodes": per_episode,
    }


def summarize(records: list[dict[str, Any]], horizon_seconds: float) -> dict[str, Any]:
    episode_paths = {record["episode_path"] for record in records}
    tasks = {main_task(record["prompt"]) for record in records}
    prompts = {record["prompt"] for record in records}
    segments = {(record["episode_path"], record["segment_id"]) for record in records}
    return {
        "clips": len(records),
        "sessions": len({path.split("/", 1)[0] for path in episode_paths}),
        "episodes": len(episode_paths),
        "main_tasks": len(tasks),
        "prompts": len(prompts),
        "caption_segments": len(segments),
        "motion_bins": dict(sorted(Counter(record["motion_bin"] for record in records).items())),
        "task_names": sorted(tasks),
        "temporal": temporal_metrics(records, horizon_seconds),
    }


def analyze_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    target_fps = float(manifest["target_fps"])
    horizon = int(manifest["horizon"])
    if target_fps <= 0 or horizon <= 0:
        raise ValueError("target_fps and horizon must be positive")
    train = manifest.get("train", [])
    validation = manifest.get("validation", [])
    train_episodes = set(manifest.get("train_episode_paths", []))
    validation_episodes = set(manifest.get("validation_episode_paths", []))
    if not train or not validation:
        raise ValueError("manifest must contain nonempty train and validation records")
    if train_episodes & validation_episodes:
        raise ValueError("train and validation episodes overlap")
    train_tasks = {main_task(record["prompt"]) for record in train}
    validation_tasks = {main_task(record["prompt"]) for record in validation}
    train_prompts = {record["prompt"] for record in train}
    validation_prompts = {record["prompt"] for record in validation}
    horizon_seconds = horizon / target_fps
    return {
        "format_version": 1,
        "manifest": str(path),
        "manifest_sha256": file_hash(path),
        "source": manifest.get("source", {}),
        "target_fps": target_fps,
        "horizon": horizon,
        "horizon_seconds": horizon_seconds,
        "episode_disjoint": True,
        "train_validation_overlap": {
            "main_tasks": sorted(train_tasks & validation_tasks),
            "prompts": sorted(train_prompts & validation_prompts),
        },
        "train": summarize(train, horizon_seconds),
        "validation": summarize(validation, horizon_seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure semantic and temporal diversity in an Xperience data manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
