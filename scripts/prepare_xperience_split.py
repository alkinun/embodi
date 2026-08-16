#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from embodi.xperience import (
    CachedXperienceEpisode,
    select_fixed_episode_xperience_split,
    select_pooled_fixed_episode_xperience_split,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an immutable fixed Xperience split.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-episode-path", action="append", required=True)
    parser.add_argument("--validation-episode-path", action="append", required=True)
    parser.add_argument("--train-clips", type=int, required=True)
    parser.add_argument("--validation-clips", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--closed-ratio", type=float, default=0.18)
    parser.add_argument("--open-ratio", type=float, default=1.0)
    parser.add_argument(
        "--pooled-training",
        action="store_true",
        help="balance motion globally instead of assigning equal per-episode clip quotas",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_manifest_path = args.dataset_root / "xperience_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if not source_manifest.get("complete"):
        raise ValueError("Xperience cache manifest is incomplete")
    available_paths = {entry["path"] for entry in source_manifest.get("episodes", [])}
    train_paths = list(dict.fromkeys(args.train_episode_path))
    validation_paths = list(dict.fromkeys(args.validation_episode_path))
    if set(train_paths) & set(validation_paths):
        raise ValueError("train and validation episodes overlap")
    selected_paths = train_paths + validation_paths
    missing = set(selected_paths) - available_paths
    if missing:
        raise ValueError(f"episodes are absent from the cache manifest: {sorted(missing)}")

    episodes = [
        CachedXperienceEpisode(
            args.dataset_root / path,
            closed_ratio=args.closed_ratio,
            open_ratio=args.open_ratio,
        )
        for path in selected_paths
    ]
    selector = (
        select_pooled_fixed_episode_xperience_split
        if args.pooled_training
        else select_fixed_episode_xperience_split
    )
    split = selector(
        episodes,
        train_episode_indices=range(len(train_paths)),
        validation_episode_indices=range(len(train_paths), len(selected_paths)),
        train_clips=args.train_clips,
        validation_clips=args.validation_clips,
        seed=args.seed,
    )
    payload = {
        "format_version": 1,
        "dataset_format": "xperience",
        "source": {
            "repo_id": source_manifest.get("repo_id"),
            "revision": source_manifest.get("revision"),
            "cache_manifest_sha256": file_hash(source_manifest_path),
            "episode_paths": selected_paths,
            "source_fps": [episode.source_fps for episode in episodes],
        },
        "target_fps": 30,
        "horizon": 32,
        "seed": args.seed,
        "opening_calibration": {
            "closed_ratio": args.closed_ratio,
            "open_ratio": args.open_ratio,
        },
        "selection_strategy": "pooled_motion_balanced" if args.pooled_training else "equal_episode_quota",
        "train_episode_paths": train_paths,
        "validation_episode_paths": validation_paths,
        "rejection_counts": split.rejection_counts,
        "train": [
            {"episode_path": selected_paths[index], **asdict(anchor)}
            for index, anchor in split.train
        ],
        "validation": [
            {"episode_path": selected_paths[index], **asdict(anchor)}
            for index, anchor in split.validation
        ],
    }
    serialized = json.dumps(payload, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != serialized:
        raise FileExistsError(f"refusing to replace a different split: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_hash(args.output),
                "train_clips": len(split.train),
                "validation_clips": len(split.validation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
