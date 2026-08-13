import torch

from embodi.action_decoder import MorphologyActionDecoder


def make_decoder() -> MorphologyActionDecoder:
    return MorphologyActionDecoder(("arm.0",), ("arm",), (6,), (6,), ("hold",))


def test_decoder_reads_assigned_fixed_slot() -> None:
    decoder = make_decoder()
    canonical = torch.randn(2, 8, 25)
    state = torch.randn(2, 6)
    assert decoder(canonical, state).shape == (2, 8, 6)


def test_decoder_validates_mask_and_holds_state() -> None:
    decoder = make_decoder()
    state = torch.randn(2, 6)
    torch.testing.assert_close(decoder.safe_native_action(state), state)
    try:
        decoder(torch.randn(2, 8, 25), state, torch.ones(2, 2))
    except ValueError as error:
        assert "action_group_mask" in str(error)
    else:
        raise AssertionError("invalid mask was accepted")


def test_each_arm_slot_can_have_an_independent_decoder_shape() -> None:
    decoder = MorphologyActionDecoder(
        ("arm.0", "arm.1"),
        ("arm", "arm"),
        (6, 7),
        (6, 7),
        ("hold", "hold"),
    )
    assert decoder.decoders["arm_0"].network[-1].out_features == 6
    assert decoder.decoders["arm_1"].network[-1].out_features == 7
