#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile, RepoFolder


DEFAULT_REPO = "ropedia-ai/xperience-10m"
REQUIRED_FILES = ("annotation.hdf5", "stereo_left.mp4")


@dataclass(frozen=True)
class EpisodeFiles:
    path: str
    files: tuple[RepoFile, ...]

    @property
    def size(self) -> int:
        return sum(file.size or 0 for file in self.files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache a bounded, reproducible Xperience-10M episode subset from Hugging Face."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", help="Branch, tag, or commit. The resolved commit is recorded.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/xperience-10m"))
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to select.")
    parser.add_argument("--episode-path", action="append", default=[], help="Explicit <session>/epN path; repeatable.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic repository episode selection seed.")
    parser.add_argument(
        "--max-download-gb",
        type=float,
        default=10.0,
        help="Abort before downloading if selected files exceed this many decimal GB.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve and size files without downloading.")
    return parser.parse_args()


def episode_files(api: HfApi, repo_id: str, revision: str, episode_path: str) -> EpisodeFiles:
    entries = api.list_repo_tree(
        repo_id,
        path_in_repo=episode_path,
        repo_type="dataset",
        revision=revision,
        recursive=False,
        expand=True,
    )
    by_name = {
        Path(entry.path).name: entry
        for entry in entries
        if isinstance(entry, RepoFile) and Path(entry.path).name in REQUIRED_FILES
    }
    missing = sorted(set(REQUIRED_FILES) - set(by_name))
    if missing:
        raise ValueError(f"episode {episode_path!r} is missing required files: {missing}")
    return EpisodeFiles(episode_path, tuple(by_name[name] for name in REQUIRED_FILES))


def select_episode_paths(
    api: HfApi,
    repo_id: str,
    revision: str,
    count: int,
    seed: int,
) -> list[str]:
    randomizer = random.Random(seed)
    sessions = [
        entry.path
        for entry in api.list_repo_tree(
            repo_id, repo_type="dataset", revision=revision, recursive=False, expand=False
        )
        if isinstance(entry, RepoFolder)
    ]
    randomizer.shuffle(sessions)
    episode_groups: list[list[str]] = []
    for session in sessions:
        episodes = [
            entry.path
            for entry in api.list_repo_tree(
                repo_id,
                path_in_repo=session,
                repo_type="dataset",
                revision=revision,
                recursive=False,
                expand=False,
            )
            if isinstance(entry, RepoFolder) and Path(entry.path).name.startswith("ep")
        ]
        randomizer.shuffle(episodes)
        if episodes:
            episode_groups.append(episodes)
            if len(episode_groups) == count:
                return [group[0] for group in episode_groups]
    selected: list[str] = []
    depth = 0
    while len(selected) < count:
        added = False
        for episodes in episode_groups:
            if depth < len(episodes):
                selected.append(episodes[depth])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        depth += 1
    raise RuntimeError(f"repository contains only {len(selected)} discoverable episodes; requested {count}")


def file_record(file: RepoFile) -> dict[str, object]:
    return {
        "path": file.path,
        "size": file.size,
        "blob_id": file.blob_id,
        "sha256": file.lfs.sha256 if file.lfs is not None else None,
    }


def write_manifest(
    path: Path,
    *,
    repo_id: str,
    revision: str,
    seed: int,
    episodes: Iterable[EpisodeFiles],
    complete: bool,
) -> None:
    episode_list = list(episodes)
    payload = {
        "format_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "seed": seed,
        "required_files": list(REQUIRED_FILES),
        "complete": complete,
        "total_bytes": sum(episode.size for episode in episode_list),
        "episodes": [
            {"path": episode.path, "files": [file_record(file) for file in episode.files]}
            for episode in episode_list
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_download_gb <= 0:
        raise ValueError("--max-download-gb must be positive")
    if args.episode_path and args.episodes != 1 and args.episodes != len(args.episode_path):
        raise ValueError("with --episode-path, omit --episodes or set it to the number of explicit paths")

    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision)
    revision = info.sha
    paths = list(dict.fromkeys(args.episode_path))
    if not paths:
        paths = select_episode_paths(api, args.repo_id, revision, args.episodes, args.seed)
    selected = [episode_files(api, args.repo_id, revision, path) for path in paths]
    total_bytes = sum(episode.size for episode in selected)
    limit_bytes = int(args.max_download_gb * 1_000_000_000)
    print(f"repo={args.repo_id} revision={revision}")
    for episode in selected:
        print(f"episode={episode.path} size_gb={episode.size / 1_000_000_000:.3f}")
    print(f"episodes={len(selected)} total_gb={total_bytes / 1_000_000_000:.3f}")
    if total_bytes > limit_bytes:
        raise RuntimeError(
            f"selected files require {total_bytes / 1_000_000_000:.3f} GB, "
            f"exceeding --max-download-gb={args.max_download_gb:g}"
        )
    if args.dry_run:
        return

    manifest_path = args.output_dir / "xperience_manifest.json"
    write_manifest(
        manifest_path,
        repo_id=args.repo_id,
        revision=revision,
        seed=args.seed,
        episodes=selected,
        complete=False,
    )
    for episode in selected:
        for file in episode.files:
            local_path = hf_hub_download(
                args.repo_id,
                file.path,
                repo_type="dataset",
                revision=revision,
                local_dir=args.output_dir,
            )
            print(f"cached={local_path}")
    write_manifest(
        manifest_path,
        repo_id=args.repo_id,
        revision=revision,
        seed=args.seed,
        episodes=selected,
        complete=True,
    )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
