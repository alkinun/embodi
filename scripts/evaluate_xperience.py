#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from embodi.checkpoints import load_inference_policy
from embodi.processing import EmbodiProcessor
from embodi.train import evaluate, xperience_manifest_samples
from embodi.xperience import CachedXperienceEpisode, MultiEpisodeCachedXperienceDataset


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_records(policy, processor, episodes, path_indices, records, args, device):
    samples = xperience_manifest_samples(records, path_indices)
    dataset = MultiEpisodeCachedXperienceDataset(episodes, samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    return evaluate(policy, loader, processor, device, len(loader), args.validation_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a human-motion checkpoint by session.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-seed", type=int, default=81000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.checkpoint / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_format") != "xperience":
        raise ValueError("checkpoint does not contain an Xperience data manifest")
    source = manifest.get("source", {})
    selected_paths = list(source.get("episode_paths", []))
    if not selected_paths:
        raise ValueError("checkpoint data manifest contains no episodes")
    opening = manifest["opening_calibration"]
    episodes = [
        CachedXperienceEpisode(
            args.dataset_root / path,
            closed_ratio=opening["closed_ratio"],
            open_ratio=opening["open_ratio"],
        )
        for path in selected_paths
    ]
    path_indices = {path: index for index, path in enumerate(selected_paths)}
    validation = manifest.get("validation", [])
    validation_paths = set(manifest.get("validation_episode_paths", []))
    represented_paths = {record["episode_path"] for record in validation}
    if not validation or not validation_paths:
        raise ValueError("checkpoint data manifest contains no validation split")
    if represented_paths - validation_paths:
        raise ValueError("validation records violate the episode partition")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    policy = load_inference_policy(args.checkpoint, device)
    policy.configure_stage("core")
    processor = EmbodiProcessor.from_pretrained(
        policy.config.backbone_name,
        revision=policy.config.backbone_revision,
        do_rescale=policy.config.image_do_rescale,
    )
    aggregate = evaluate_records(
        policy, processor, episodes, path_indices, validation, args, device
    )
    by_episode = {}
    for episode_path in sorted(represented_paths):
        records = [record for record in validation if record["episode_path"] == episode_path]
        by_episode[episode_path] = evaluate_records(
            policy, processor, episodes, path_indices, records, args, device
        )
    report = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "config_sha256": file_hash(args.checkpoint / "config.json"),
        "core_sha256": file_hash(args.checkpoint / "core.pt"),
        "data_manifest_sha256": file_hash(manifest_path),
        "validation_seed": args.validation_seed,
        "validation_clips": len(validation),
        "aggregate": aggregate,
        "by_episode": by_episode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
