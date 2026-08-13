import torch
from torch import nn

from embodi.config import EmbodiConfig
from embodi.policy import EmbodiPolicy


class FakeBackbone(nn.Module):
    hidden_size = 24

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(10, 24)

    def forward(self, state, state_mask=None, **model_inputs):
        context = self.projection(state)
        return context, state_mask


def test_frozen_v2_policy_still_runs_and_saves_split_checkpoint(tmp_path) -> None:
    config = EmbodiConfig(
        chunk_size=8,
        n_action_steps=4,
        expert_width=32,
        expert_layers=2,
        expert_heads=4,
        denoising_steps=2,
    )
    policy = EmbodiPolicy(config, backbone=FakeBackbone())
    batch = {
        "observation.state": torch.randn(2, 6),
        "action": torch.randn(2, 8, 6),
        "canonical_action": torch.randn(2, 8, 25),
        "canonical_slot_mask": torch.tensor([[True, False, False, False]]).expand(2, -1),
    }
    loss, _metrics = policy(batch)
    loss.backward()
    assert policy.predict_action_chunk({"observation.state": torch.randn(2, 6)}).shape == (2, 8, 6)
    policy.save_checkpoint(tmp_path)
    assert (tmp_path / "core.pt").is_file()
    assert (tmp_path / "embodiment.pt").is_file()
