import torch

from embodi import EmbodiConfig, EmbodiPolicy
from embodi.train import (
    _SmokeBackbone,
    configure_deterministic_training,
    gradient_norm,
    prepare_model_batch,
    resolve_training_seeds,
    split_episode_indices,
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
