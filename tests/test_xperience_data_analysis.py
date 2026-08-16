import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_xperience_data.py"
SPEC = importlib.util.spec_from_file_location("analyze_xperience_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(episode: str, timestamp_ns: int, task: str, segment: int = 0) -> dict:
    return {
        "episode_path": episode,
        "frame_index": 0,
        "timestamp_ns": timestamp_ns,
        "segment_id": segment,
        "prompt": f"Task: {task}.\nCurrent subtask: move.",
        "motion_bin": "meaningful",
    }


def test_manifest_analysis_reports_task_and_temporal_diversity(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {"repo_id": "source", "revision": "abc"},
                "target_fps": 2,
                "horizon": 2,
                "train_episode_paths": ["session-a/ep1", "session-b/ep1"],
                "validation_episode_paths": ["session-c/ep1"],
                "train": [
                    record("session-a/ep1", 0, "task a"),
                    record("session-a/ep1", 500_000_000, "task a"),
                    record("session-b/ep1", 0, "task b"),
                ],
                "validation": [record("session-c/ep1", 0, "task c")],
            }
        )
    )
    report = MODULE.analyze_manifest(manifest)
    assert report["episode_disjoint"]
    assert report["train"]["sessions"] == 2
    assert report["train"]["main_tasks"] == 2
    assert report["train"]["temporal"]["labeled_window_seconds"] == 3
    assert report["train"]["temporal"]["union_labeled_seconds"] == 2.5
    assert report["train"]["temporal"]["nonoverlapping_window_capacity"] == 2
    assert report["train_validation_overlap"] == {"main_tasks": [], "prompts": []}


def test_manifest_analysis_rejects_episode_leakage(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target_fps": 30,
                "horizon": 32,
                "train_episode_paths": ["session/ep1"],
                "validation_episode_paths": ["session/ep1"],
                "train": [record("session/ep1", 0, "task")],
                "validation": [record("session/ep1", 0, "task")],
            }
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        MODULE.analyze_manifest(manifest)
