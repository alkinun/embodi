import torch

from embodi.token_canonical import reanchor_canonical_actions
from embodi.token_config import PART_KIND_IDS


def test_part_token_actions_reanchor_each_pose_part() -> None:
    actions = torch.zeros(1, 2, 2, 10)
    states = torch.zeros_like(actions)
    for tensor in (actions, states):
        tensor[..., 3] = 1
        tensor[..., 7] = 1
        tensor[..., 9] = 0.5
    states[0, 1, 0, 0] = 0.1
    states[0, 1, 1, 1] = 0.2
    actions[0, 0, 0, 0] = 0.01
    actions[0, 1, 0, 0] = 0.02
    actions[0, 1, 1, 1] = 0.03
    kinds = torch.tensor([[PART_KIND_IDS["pose_scalar"], PART_KIND_IDS["pose"]]])
    channels = torch.ones(1, 2, 10, dtype=torch.bool)
    channels[:, 1, 9] = False
    parts = torch.ones(1, 2, dtype=torch.bool)
    anchored, initial = reanchor_canonical_actions(actions, states, kinds, channels, parts)
    torch.testing.assert_close(anchored[0, :, 0, 0], torch.tensor([0.01, 0.12]))
    torch.testing.assert_close(anchored[0, :, 1, 1], torch.tensor([0.0, 0.23]))
    assert initial.shape == (1, 2, 10)
