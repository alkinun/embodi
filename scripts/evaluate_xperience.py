#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from embodi.checkpoints import load_inference_policy
from embodi.processing import EmbodiProcessor
from embodi.train import prepare_model_batch, xperience_manifest_samples
from embodi.xperience import CachedXperienceEpisode, MultiEpisodeCachedXperienceDataset


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_records(policy, processor, episodes, path_indices, records, args, device):
    samples = xperience_manifest_samples(records, path_indices)
    dataset = MultiEpisodeCachedXperienceDataset(episodes, samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    generator = torch.Generator(device=device).manual_seed(args.validation_seed)
    aggregate_totals = {}
    episode_totals = {}
    episode_counts = {}
    for record, raw_batch in zip(records, loader, strict=True):
        model_batch = prepare_model_batch(
            raw_batch,
            processor,
            device,
            policy.config.camera_names,
            policy.config,
        )
        canonical_actions = policy._canonical_action_targets(model_batch)
        noise = torch.randn(
            canonical_actions.shape,
            device=device,
            dtype=canonical_actions.dtype,
            generator=generator,
        )
        time = torch.rand(
            canonical_actions.shape[0],
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        _, metrics = policy(model_batch, noise=noise, time=time)
        episode_path = record["episode_path"]
        totals = episode_totals.setdefault(episode_path, {})
        episode_counts[episode_path] = episode_counts.get(episode_path, 0) + 1
        for name, value in metrics.items():
            aggregate_totals[name] = aggregate_totals.get(name, 0.0) + value
            totals[name] = totals.get(name, 0.0) + value
    count = len(records)
    aggregate = {name: value / count for name, value in aggregate_totals.items()}
    by_episode = {
        episode_path: {
            name: value / episode_counts[episode_path] for name, value in totals.items()
        }
        for episode_path, totals in sorted(episode_totals.items())
    }
    return aggregate, by_episode


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

    run_manifest_path = args.checkpoint.parent / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError("checkpoint parent is missing run_manifest.json")
    run_manifest = json.loads(run_manifest_path.read_text())
    model_seed = int(run_manifest["resolved_seeds"]["model"])
    random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    policy = load_inference_policy(args.checkpoint, device)
    policy.configure_stage("core")
    processor = EmbodiProcessor.from_pretrained(
        policy.config.backbone_name,
        revision=policy.config.backbone_revision,
        do_rescale=policy.config.image_do_rescale,
    )
    aggregate, by_episode = evaluate_records(
        policy, processor, episodes, path_indices, validation, args, device
    )
    report = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "config_sha256": file_hash(args.checkpoint / "config.json"),
        "core_sha256": file_hash(args.checkpoint / "core.pt"),
        "data_manifest_sha256": file_hash(manifest_path),
        "model_seed": model_seed,
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
