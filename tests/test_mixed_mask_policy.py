import pytest
import torch
from torch import nn

from embodi import EmbodiConfig, EmbodiPolicy


class RecordingBackbone(nn.Module):
    hidden_size = 24

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(10, self.hidden_size)
        self.batch_sizes: list[int] = []

    def forward(self, state, state_mask=None, **model_inputs):
        self.batch_sizes.append(state.shape[0])
        return self.projection(state), state_mask


def make_policy(core_objective: str = "regression") -> EmbodiPolicy:
    config = EmbodiConfig(
        chunk_size=4,
        n_action_steps=2,
        expert_width=16,
        expert_layers=1,
        expert_heads=4,
        denoising_steps=1,
        decoder_input_noise_std=0.0,
        core_objective=core_objective,
    )
    return EmbodiPolicy(config, backbone=RecordingBackbone())


def mixed_batch(policy: EmbodiPolicy) -> dict[str, torch.Tensor]:
    batch_size = 2
    return {
        "observation.state": torch.randn(batch_size, policy.config.state_dim),
        "action": torch.randn(
            batch_size, policy.config.chunk_size, policy.config.action_dim
        ),
        "canonical_action": torch.randn(
            batch_size,
            policy.config.chunk_size,
            policy.config.part_count,
            10,
        ),
        "canonical_part_mask": torch.tensor([[False], [True]]),
        "action_group_mask": torch.tensor([[False], [True]]),
    }


def active_row(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value[1:2] for name, value in batch.items()}


def test_predict_action_chunk_skips_inactive_rows_and_returns_safe_actions() -> None:
    policy = make_policy().eval()
    batch = mixed_batch(policy)

    mixed = policy.predict_action_chunk(batch)
    active = policy.predict_action_chunk(active_row(batch))

    expected_safe = batch["observation.state"][0].expand(policy.config.chunk_size, -1)
    torch.testing.assert_close(mixed[0], expected_safe)
    torch.testing.assert_close(mixed[1], active[0])
    assert policy.backbone.batch_sizes == [1, 1]


@pytest.mark.parametrize("core_objective", ["flow", "regression"])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_training_skips_inactive_rows_and_assigns_them_zero_loss(
    core_objective: str, reduction: str
) -> None:
    policy = make_policy(core_objective)
    batch = mixed_batch(policy)
    noise = torch.randn_like(batch["canonical_action"])
    time = torch.tensor([0.25, 0.75])

    mixed_loss, _ = policy(batch, noise=noise, time=time, reduction=reduction)
    active_loss, _ = policy(
        active_row(batch), noise=noise[1:2], time=time[1:2], reduction=reduction
    )

    if reduction == "none":
        torch.testing.assert_close(mixed_loss[0], torch.zeros_like(mixed_loss[0]))
        torch.testing.assert_close(mixed_loss[1], active_loss[0])
    else:
        torch.testing.assert_close(mixed_loss, active_loss)
    assert policy.backbone.batch_sizes == [1, 1]


def test_all_inactive_behavior_is_preserved() -> None:
    policy = make_policy()
    batch = mixed_batch(policy)
    inactive = {name: value[:1] for name, value in batch.items()}

    actions = policy.predict_action_chunk(inactive)
    expected_safe = inactive["observation.state"].unsqueeze(1).expand_as(actions)
    torch.testing.assert_close(actions, expected_safe)
    assert policy.backbone.batch_sizes == []

    with pytest.raises(ValueError, match="at least one active part"):
        policy(inactive, stage="core")


def test_mixed_batch_still_validates_state_and_action_masks() -> None:
    policy = make_policy()
    batch = mixed_batch(policy)
    batch["action_group_mask"] = torch.ones_like(batch["action_group_mask"])

    with pytest.raises(ValueError, match="descriptors disagree"):
        policy(batch, stage="core")


def test_decoder_stage_validates_inactive_canonical_row_against_active_group() -> None:
    policy = make_policy()
    batch = mixed_batch(policy)
    batch["canonical_state"] = torch.randn(
        2, policy.config.part_count, 10
    )
    batch["action_group_mask"] = torch.ones_like(batch["action_group_mask"])

    with pytest.raises(ValueError, match="native group and canonical part masks disagree"):
        policy(batch, stage="decoder")
