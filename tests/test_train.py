from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from embodi import EmbodiConfig, EmbodiPolicy
from embodi.train import (
    _SmokeBackbone,
    append_metrics,
    configure_deterministic_training,
    gradient_norm,
    prepare_model_batch,
    resolve_training_seeds,
    should_save_checkpoint,
    split_episode_indices,
    validate_lerobot_training_dataset,
    write_checkpoint_data_manifest,
    write_or_validate_data_manifest,
    xperience_manifest_samples,
)


def test_episode_split_holds_out_last_episodes() -> None:
    train, validation = split_episode_indices(50, 5)
    assert train == list(range(45))
    assert validation == list(range(45, 50))
    assert set(train).isdisjoint(validation)


def test_episode_split_rejects_empty_partition() -> None:
    for validation_episodes in (0, 50):
        try:
            split_episode_indices(50, validation_episodes)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid validation split was accepted")


def test_training_seed_overrides_are_independent() -> None:
    assert resolve_training_seeds(7, None, None) == (7, 7)
    assert resolve_training_seeds(7, 11, None) == (11, 7)
    assert resolve_training_seeds(7, None, 13) == (7, 13)
    assert resolve_training_seeds(7, 11, 13) == (11, 13)


def test_prepared_xperience_records_restore_typed_anchors() -> None:
    samples = xperience_manifest_samples(
        [
            {
                "episode_path": "session/ep1",
                "frame_index": 3,
                "timestamp_ns": 4,
                "segment_id": 5,
                "prompt": "move",
                "motion_bin": "small",
            }
        ],
        {"session/ep1": 0},
    )
    assert samples[0][0] == 0
    assert samples[0][1].frame_index == 3
    with pytest.raises(ValueError, match="unknown episode"):
        xperience_manifest_samples(
            [{**asdict(samples[0][1]), "episode_path": "other/ep1"}],
            {"session/ep1": 0},
        )


def test_metrics_are_persisted_as_json_lines(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_metrics(path, 10, "validation", {"flow_loss": 0.25})
    append_metrics(path, 20, "train", {"flow_loss": 0.1})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"metrics": {"flow_loss": 0.25}, "split": "validation", "step": 10},
        {"metrics": {"flow_loss": 0.1}, "split": "train", "step": 20},
    ]


def test_checkpoint_schedule_combines_interval_and_explicit_steps() -> None:
    assert should_save_checkpoint(500, 500, [750])
    assert should_save_checkpoint(750, 500, [750])
    assert not should_save_checkpoint(800, 500, [750])


def test_deterministic_training_disabled_preserves_global_setting() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    configure_deterministic_training(False)
    assert torch.are_deterministic_algorithms_enabled() is previous


def test_gradient_norm_uses_only_available_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    first.grad = torch.tensor([3.0, 4.0])
    assert gradient_norm([first, second]) == 5.0


def test_training_stages_freeze_expected_components() -> None:
    config = EmbodiConfig(chunk_size=8, n_action_steps=4, expert_width=32, expert_layers=2, expert_heads=4)
    policy = EmbodiPolicy(config, backbone=_SmokeBackbone(10))
    policy.configure_stage("decoder")
    assert not any(parameter.requires_grad for parameter in policy.core.parameters())
    assert any(parameter.requires_grad for parameter in policy.action_decoder.parameters())


def test_frozen_backbone_pretraining_trains_only_expert_core() -> None:
    config = EmbodiConfig(
        chunk_size=8,
        n_action_steps=4,
        expert_width=32,
        expert_layers=2,
        expert_heads=4,
        freeze_backbone=True,
    )
    policy = EmbodiPolicy(config, backbone=_SmokeBackbone(10))
    policy.configure_stage("core")
    assert not any(parameter.requires_grad for parameter in policy.backbone.parameters())
    assert any(parameter.requires_grad for parameter in policy.expert.parameters())
    assert not any(parameter.requires_grad for parameter in policy.action_decoder.parameters())
    policy.configure_stage("core")
    assert any(parameter.requires_grad for parameter in policy.expert.parameters())
    assert not any(parameter.requires_grad for parameter in policy.action_decoder.parameters())
    policy.configure_stage("predicted_decoder")
    assert not any(parameter.requires_grad for parameter in policy.core.parameters())
    assert not any(parameter.requires_grad for parameter in policy.state_adapter.parameters())
    assert any(parameter.requires_grad for parameter in policy.action_decoder.parameters())


class FakeProcessor:
    def __call__(self, images, instructions, state, **values):
        aliases = {"canonical_actions": "canonical_action", "actions": "action"}
        return {
            "observation.state": state,
            **{
                aliases.get(key, key): value
                for key, value in values.items()
                if value is not None
            },
        }


class _FrozenAdapterBackbone(_SmokeBackbone):
    def __init__(self, state_dim: int) -> None:
        super().__init__(state_dim)
        self.state_projection = torch.nn.Linear(state_dim, self.hidden_size)


def test_checkpoint_persists_frozen_local_backbone_adapter(tmp_path) -> None:
    config = EmbodiConfig(
        chunk_size=8,
        n_action_steps=4,
        expert_width=32,
        expert_layers=2,
        expert_heads=4,
        freeze_backbone=True,
    )
    policy = EmbodiPolicy(config, backbone=_FrozenAdapterBackbone(10))
    policy.configure_stage("core")
    policy.save_checkpoint(tmp_path)
    payload = torch.load(tmp_path / "core.pt", map_location="cpu", weights_only=True)
    assert payload["stores_frozen_adapter"] is True
    assert "backbone.state_projection.weight" in payload["core"]
    restored = EmbodiPolicy(config, backbone=_FrozenAdapterBackbone(10))
    restored.load_core_checkpoint(tmp_path)
    torch.testing.assert_close(
        restored.backbone.state_projection.weight,
        policy.backbone.state_projection.weight,
    )


def test_prepare_batch_reanchors_part_tokens_and_transports_descriptors() -> None:
    config = EmbodiConfig(chunk_size=8, n_action_steps=4, expert_width=32, expert_layers=2, expert_heads=4)
    actions = torch.zeros(2, 8, 1, 10)
    states = torch.zeros_like(actions)
    actions[..., 3] = states[..., 3] = 1
    actions[..., 7] = states[..., 7] = 1
    actions[..., 9] = states[..., 9] = 0.5
    padding = torch.zeros(2, 8, dtype=torch.bool)
    raw = {
        "observation.images.top": torch.randn(2, 3, 4, 4),
        "observation.state": torch.randn(2, 6),
        "task": ["move", "move"],
        "canonical_action": actions,
        "canonical_state": states,
        "canonical_part_mask": torch.ones(2),
        "canonical_action_is_pad": padding,
        "canonical_state_is_pad": padding.clone(),
    }
    batch = prepare_model_batch(raw, FakeProcessor(), torch.device("cpu"), ("top",), config)
    assert batch["canonical_action"].shape == (2, 8, 1, 10)
    assert batch["canonical_state"].shape == (2, 1, 10)
    assert batch["canonical_part_kind"].shape == (2, 1)
    assert batch["canonical_part_name_features"].shape == (2, 1, 32)


def test_prepare_batch_rejects_mismatched_supervision_padding() -> None:
    config = EmbodiConfig(chunk_size=8, n_action_steps=4, expert_width=32, expert_layers=2, expert_heads=4)
    actions = torch.zeros(1, 8, 1, 10)
    states = torch.zeros_like(actions)
    raw = {
        "observation.images.top": torch.randn(1, 3, 4, 4),
        "observation.state": torch.randn(1, 6),
        "task": ["move"],
        "canonical_action": actions,
        "canonical_state": states,
        "canonical_part_mask": torch.ones(1),
        "canonical_action_is_pad": torch.zeros(1, 8, dtype=torch.bool),
        "canonical_state_is_pad": torch.ones(1, 8, dtype=torch.bool),
    }
    try:
        prepare_model_batch(raw, FakeProcessor(), torch.device("cpu"), ("top",), config)
    except ValueError as error:
        assert "padding masks disagree" in str(error)
    else:
        raise AssertionError("mismatched padding masks were accepted")


def test_lerobot_training_admission_validates_schema_cameras_and_stats(tmp_path) -> None:
    config = EmbodiConfig(camera_names=("top",))
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "embodi.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "parts": [asdict(part) for part in config.parts],
            }
        )
    )
    features = {
        "observation.images.top": {"dtype": "video", "shape": (480, 640, 3)},
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
        "canonical_action": {"dtype": "float32", "shape": (1, 10)},
        "canonical_state": {"dtype": "float32", "shape": (1, 10)},
        "canonical_part_mask": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["primary_effector"],
        },
    }
    metadata = SimpleNamespace(
        fps=30,
        total_episodes=2,
        total_frames=4,
        features=features,
        stats={
            "observation.state": {"mean": [0.0] * 6, "std": [1.0] * 6},
            "action": {"mean": [0.0] * 6, "std": [1.0] * 6},
        },
        root=tmp_path,
        robot_type="so101_sim",
        repo_id="embodi/sim",
    )
    validate_lerobot_training_dataset(metadata, config, "refine")
    bad_camera = dict(features)
    bad_camera["observation.images.top"] = {"dtype": "video", "shape": (480, 640, 1)}
    metadata.features = bad_camera
    with pytest.raises(ValueError, match="HWC RGB"):
        validate_lerobot_training_dataset(metadata, config, "refine")
    metadata.features = features
    metadata.stats["action"]["std"][0] = float("nan")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        validate_lerobot_training_dataset(metadata, config, "refine")


def test_resume_preserves_and_validates_dataset_lineage(tmp_path) -> None:
    report = {"format_version": 1, "repo_id": "embodi/data"}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_checkpoint_data_manifest(checkpoint, report)
    output = tmp_path / "output"
    output.mkdir()
    write_or_validate_data_manifest(output, dict(report), resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="dataset lineage"):
        write_or_validate_data_manifest(
            output,
            {"format_version": 1, "repo_id": "embodi/other"},
            resume_checkpoint=checkpoint,
        )
    unverified = tmp_path / "unverified"
    unverified.mkdir()
    with pytest.raises(ValueError, match="missing verified dataset lineage"):
        write_or_validate_data_manifest(
            tmp_path / "physical-output",
            {"robot_type": "so_follower"},
            resume_checkpoint=unverified,
        )
