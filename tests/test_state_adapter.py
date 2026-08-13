import torch

from embodi import PartDescriptor
from embodi.token_state_adapter import EmbodimentStateAdapter


def test_state_adapter_has_fixed_core_interface() -> None:
    parts = (PartDescriptor("arm.0", "pose_scalar"),)
    six_joint = EmbodimentStateAdapter(parts, ("arm.0",), (6,))
    seven_joint = EmbodimentStateAdapter(parts, ("arm.0",), (7,))
    first, first_mask = six_joint(torch.randn(2, 6))
    second, second_mask = seven_joint(torch.randn(2, 7))
    assert first.shape == second.shape == (2, 1, 10)
    assert first_mask.tolist() == second_mask.tolist() == [[True]] * 2
