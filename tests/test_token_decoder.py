import torch

from embodi import PartDescriptor
from embodi.token_decoder import PartTokenActionDecoder


def test_decoder_routes_independent_named_part_blocks() -> None:
    parts = (PartDescriptor("left", "pose_scalar"), PartDescriptor("right", "pose"))
    decoder = PartTokenActionDecoder(
        parts,
        ("left", "right"),
        (6, 7),
        (6, 7),
        ("hold", "hold"),
    )
    output = decoder(torch.randn(2, 8, 2, 10), torch.randn(2, 13))
    assert output.shape == (2, 8, 13)
    assert decoder.decoders[0].network[-1].out_features == 6
    assert decoder.decoders[1].network[-1].out_features == 7


def test_decoder_hold_is_native_state() -> None:
    part = (PartDescriptor("tool", "scalar"),)
    decoder = PartTokenActionDecoder(part, ("tool",), (2,), (2,), ("hold",))
    state = torch.randn(3, 2)
    torch.testing.assert_close(decoder.safe_native_action(state), state)
