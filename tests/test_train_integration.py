import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from embodi import EmbodiConfig
from embodi import train as train_module


class _FakeBackbone(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_projection = nn.Linear(state_dim, 4)


class _FakePolicy(nn.Module):
    instances: list["_FakePolicy"] = []

    def __init__(self, config: EmbodiConfig) -> None:
        super().__init__()
        self.config = config
        self.stage = "refine"
        self.backbone = _FakeBackbone(config.state_dim)
        self.expert = nn.Linear(1, 1)
        self.action_decoder = nn.Linear(1, 1)
        self.state_adapter = nn.Linear(1, 1)
        self.trainable = nn.Parameter(torch.tensor(0.5))
        self.stats = None
        self.saved_steps: list[int] = []
        self.instances.append(self)

    def configure_stage(self, stage: str) -> None:
        self.stage = stage

    def get_optim_params(self) -> list[nn.Parameter]:
        return [self.trainable]

    def set_normalization_stats(self, stats) -> None:
        self.stats = stats

    def _canonical_action_targets(self, batch: dict[str, object]) -> torch.Tensor:
        value = batch["canonical_action"]
        assert isinstance(value, torch.Tensor)
        return value

    def forward(self, batch, *, noise=None, time=None):
        del batch, noise, time
        return self.trainable.square(), {"flow_loss": float(self.trainable.detach())}

    def save_checkpoint(self, directory: Path, optimizer=None, scheduler=None, step: int = 0) -> None:
        del optimizer, scheduler
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "core.pt").write_bytes(b"core")
        (directory / "embodiment.pt").write_bytes(b"embodiment")
        (directory / "trainer.pt").write_bytes(str(step).encode())
        self.saved_steps.append(step)


class _FakeProcessor:
    instances: list["_FakeProcessor"] = []

    def __init__(self) -> None:
        self.calls = 0
        self.instances.append(self)

    @classmethod
    def from_pretrained(cls, model_name: str, revision: str | None, do_rescale: bool):
        del model_name, revision, do_rescale
        return cls()

    def __call__(self, images, instructions, state, actions=None, canonical_actions=None, **values):
        del images, instructions
        self.calls += 1
        return {
            "observation.state": state,
            "action": actions,
            "canonical_action": canonical_actions,
            **values,
        }


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch, config: EmbodiConfig, root: Path):
    class FakeDataset(Dataset):
        instances: list["FakeDataset"] = []

        def __init__(self, repo_id: str, *, root: Path, episodes, delta_timestamps) -> None:
            self.repo_id = repo_id
            self.root = root
            self.episodes = episodes
            self.delta_timestamps = delta_timestamps
            self.instances.append(self)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            return {
                "observation.images.top": torch.zeros(3, 2, 2),
                "observation.state": torch.zeros(config.state_dim),
                "action": torch.zeros(config.chunk_size, config.action_dim),
                "canonical_action": torch.zeros(config.chunk_size, config.part_count, 10),
                "canonical_state": torch.zeros(config.part_count, 10),
                "canonical_part_mask": torch.ones(config.part_count, dtype=torch.bool),
                "task": "test task",
            }

    class FakeMetadata:
        def __init__(self, repo_id: str, *, root: Path) -> None:
            self.repo_id = repo_id
            self.root = root
            self.robot_type = "so101_sim"
            self.fps = 30
            self.total_episodes = 2
            self.total_frames = 2
            self.features = {"action": {"dtype": "float32"}}
            self.stats = {
                "observation.state": {
                    "mean": [0.0] * config.state_dim,
                    "std": [1.0] * config.state_dim,
                },
                "action": {
                    "mean": [0.0] * config.action_dim,
                    "std": [1.0] * config.action_dim,
                },
            }

    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = FakeDataset
    metadata_module = ModuleType("lerobot.datasets.dataset_metadata")
    metadata_module.LeRobotDatasetMetadata = FakeMetadata
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_module)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.dataset_metadata", metadata_module)
    monkeypatch.setattr(train_module, "EmbodiPolicy", _FakePolicy)
    monkeypatch.setattr(train_module, "EmbodiProcessor", _FakeProcessor)
    monkeypatch.setattr(train_module, "validate_lerobot_training_dataset", lambda *args: {"format_version": 3})
    monkeypatch.setattr(train_module, "validate_benchmark_training_dataset", lambda metadata: None)
    return FakeDataset


def test_train_runs_tiny_lerobot_loop_and_persists_manifests_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EmbodiConfig(
        camera_names=("top",),
        chunk_size=2,
        n_action_steps=1,
        expert_width=32,
        expert_layers=2,
        expert_heads=4,
    )
    config_path = tmp_path / "config.json"
    config.save(config_path)
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")
    fake_dataset = _install_fake_lerobot(monkeypatch, config, dataset_root)

    args = train_module.argparse.Namespace(
        steps=1,
        batch_size=1,
        accumulation_steps=1,
        eval_every=1,
        validation_batches=1,
        checkpoint_step=[],
        config=config_path,
        cameras=None,
        dataset_format="lerobot",
        validation_dataset=None,
        validation_dataset_root=None,
        resume=None,
        init_from=None,
        init_core=None,
        init_embodiment=None,
        deterministic=False,
        seed=7,
        model_seed=None,
        loader_seed=None,
        validation_seed=11,
        output_dir=tmp_path / "run",
        dataset="embodi/fake",
        dataset_root=dataset_root,
        stage="core",
        validation_episodes=1,
        workers=0,
        lr=1e-3,
        warmup_steps=0,
        log_every=1,
        save_every=1,
        compact_intermediate_checkpoints=False,
        no_progress=True,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_mode="disabled",
        init_core_component=None,
    )

    train_module.train(args)

    records = [
        json.loads(line)
        for line in (args.output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["split"] for record in records] == ["validation", "train", "validation"]
    assert (args.output_dir / "run_manifest.json").is_file()
    assert (args.output_dir / "data_manifest.json").is_file()
    assert (args.output_dir / "step-0000001" / "data_manifest.json").is_file()
    assert (args.output_dir / "final" / "data_manifest.json").is_file()
    assert len(fake_dataset.instances) == 2
    assert [dataset.episodes for dataset in fake_dataset.instances] == [[0], [1]]
    assert _FakeProcessor.instances[-1].calls >= 3
    assert _FakePolicy.instances[-1].saved_steps == [1, 1]
