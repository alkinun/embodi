import pytest
import torch
from torch import Tensor, nn

from embodi.action_decoder import MorphologyActionDecoder
from embodi.config import CANONICAL_SLOTS, EmbodiConfig as SlotConfig
from embodi.state_adapter import EmbodimentStateAdapter as SlotStateAdapter
from embodi.token_config import EmbodiConfig as TokenConfig
from embodi.token_config import PartDescriptor
from embodi.token_decoder import PartTokenActionDecoder
from embodi.token_state_adapter import EmbodimentStateAdapter as TokenStateAdapter


class _CanonicalStateProbe(nn.Module):
    def __init__(self, channel: int = 0) -> None:
        super().__init__()
        self.channel = channel

    def forward(self, canonical: Tensor, state: Tensor) -> Tensor:
        return canonical[..., self.channel : self.channel + 1] + state[..., :1]


class _CanonicalSumProbe(nn.Module):
    def forward(self, canonical: Tensor, state: Tensor) -> Tensor:
        del state
        return canonical.sum(dim=-1, keepdim=True)


def _slot_safety_config() -> SlotConfig:
    return SlotConfig(
        state_dim=3,
        action_dim=5,
        action_group_slots=("head", "base"),
        action_group_types=("head", "base"),
        action_group_state_dims=(2, 1),
        action_group_native_dims=(2, 3),
        inactive_action_modes=("hold", "zero"),
    )


def _token_safety_config() -> TokenConfig:
    return TokenConfig(
        parts=(
            PartDescriptor("left", "pose_scalar"),
            PartDescriptor("right", "pose_scalar"),
        ),
        state_dim=3,
        action_dim=5,
        action_group_part_names=("right", "left"),
        action_group_state_dims=(2, 1),
        action_group_native_dims=(2, 3),
        inactive_action_modes=("hold", "zero"),
    )


def _safety_decoder(architecture: str) -> MorphologyActionDecoder | PartTokenActionDecoder:
    if architecture == "slot":
        config = _slot_safety_config()
        return MorphologyActionDecoder(
            config.action_group_slots,
            config.action_group_types,
            config.action_group_state_dims,
            config.action_group_native_dims,
            config.inactive_action_modes,
        )
    config = _token_safety_config()
    return PartTokenActionDecoder(
        config.parts,
        config.action_group_part_names,
        config.action_group_state_dims,
        config.action_group_native_dims,
        config.inactive_action_modes,
    )


def _set_scalar_projections(adapter: SlotStateAdapter | TokenStateAdapter) -> None:
    with torch.no_grad():
        for projection in adapter.projections.values() if isinstance(adapter.projections, nn.ModuleDict) else adapter.projections:
            projection.weight.fill_(1.0)
            projection.bias.zero_()


def test_part_decoder_routes_reordered_groups_by_part_name() -> None:
    config = TokenConfig(
        parts=(PartDescriptor("left"), PartDescriptor("right")),
        state_dim=2,
        action_dim=2,
        action_group_part_names=("right", "left"),
        action_group_state_dims=(1, 1),
        action_group_native_dims=(1, 1),
        inactive_action_modes=("hold", "hold"),
    )
    decoder = PartTokenActionDecoder(
        config.parts,
        config.action_group_part_names,
        config.action_group_state_dims,
        config.action_group_native_dims,
        config.inactive_action_modes,
    )
    decoder.decoders[0] = _CanonicalStateProbe()
    decoder.decoders[1] = _CanonicalStateProbe()
    canonical = torch.zeros(1, 1, 2, 10)
    canonical[0, 0, 0, 0] = 10.0
    canonical[0, 0, 1, 0] = 20.0

    output = decoder(canonical, torch.tensor([[100.0, 200.0]]))

    torch.testing.assert_close(output, torch.tensor([[[120.0, 210.0]]]))


def test_slot_decoder_routes_reordered_groups_by_slot_name() -> None:
    config = SlotConfig(
        state_dim=2,
        action_dim=2,
        action_group_slots=("head", "base"),
        action_group_types=("head", "base"),
        action_group_state_dims=(1, 1),
        action_group_native_dims=(1, 1),
        inactive_action_modes=("hold", "hold"),
    )
    decoder = MorphologyActionDecoder(
        config.action_group_slots,
        config.action_group_types,
        config.action_group_state_dims,
        config.action_group_native_dims,
        config.inactive_action_modes,
    )
    decoder.decoders["head"] = _CanonicalStateProbe()
    decoder.decoders["base"] = _CanonicalStateProbe()
    canonical = torch.zeros(1, 1, config.canonical_action_dim)
    canonical[..., decoder.canonical_slices[0].start] = 30.0
    canonical[..., decoder.canonical_slices[1].start] = 20.0

    output = decoder(canonical, torch.tensor([[100.0, 200.0]]))

    torch.testing.assert_close(output, torch.tensor([[[130.0, 220.0]]]))


def test_part_decoder_ignores_disabled_channels_and_zeroes_disabled_group() -> None:
    only_x = (True,) + (False,) * 9
    part = PartDescriptor("tool", "pose", only_x)
    decoder = PartTokenActionDecoder((part,), ("tool",), (1,), (1,), ("zero",))
    decoder.decoders[0] = _CanonicalSumProbe()
    canonical = torch.full((2, 1, 1, 10), 1000.0)
    canonical[:, :, :, 0] = 2.0

    output = decoder(canonical, torch.zeros(2, 1), torch.tensor([True, False]))

    torch.testing.assert_close(output, torch.tensor([[[2.0]], [[0.0]]]))


@pytest.mark.parametrize("architecture", ["slot", "token"])
def test_native_dim_mask_expands_each_group_in_native_order(architecture: str) -> None:
    decoder = _safety_decoder(architecture)
    group_mask = torch.tensor([[True, False], [False, True]])

    mask = decoder.native_dim_mask(group_mask, batch_size=2, device=torch.device("cpu"))

    assert mask.dtype == torch.bool
    assert mask.tolist() == [
        [True, True, False, False, False],
        [False, False, True, True, True],
    ]
    assert decoder.native_dim_mask(None, 2, torch.device("cpu")).all()


@pytest.mark.parametrize("architecture", ["slot", "token"])
def test_hold_and_zero_safety_cover_all_group_mask_combinations(architecture: str) -> None:
    decoder = _safety_decoder(architecture)
    state = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    group_mask = torch.tensor(
        [
            [True, True],
            [True, False],
            [False, True],
            [False, False],
        ]
    )
    decoded = torch.tensor([20.0, 21.0, 30.0, 31.0, 32.0]).expand(4, 5)
    native_mask = decoder.native_dim_mask(group_mask, 4, state.device)
    safe = decoder.safe_native_action(state)

    selected = torch.where(native_mask, decoded, safe)

    torch.testing.assert_close(
        safe,
        torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0],
                [4.0, 5.0, 0.0, 0.0, 0.0],
                [7.0, 8.0, 0.0, 0.0, 0.0],
                [10.0, 11.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    torch.testing.assert_close(
        selected,
        torch.tensor(
            [
                [20.0, 21.0, 30.0, 31.0, 32.0],
                [20.0, 21.0, 0.0, 0.0, 0.0],
                [7.0, 8.0, 30.0, 31.0, 32.0],
                [10.0, 11.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_token_adapter_routes_masks_by_name_and_applies_channel_mask() -> None:
    only_x = (True,) + (False,) * 9
    parts = (
        PartDescriptor("left", "scalar"),
        PartDescriptor("right", "pose", only_x),
    )
    adapter = TokenStateAdapter(parts, ("right", "left"), (1, 1))
    _set_scalar_projections(adapter)
    state = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    group_mask = torch.tensor([[False, True], [True, False]])

    canonical, part_mask = adapter(state, group_mask)

    assert part_mask.tolist() == [[True, False], [False, True]]
    assert adapter.part_mask(None, 2, state.device).all()
    assert canonical[0, 0].tolist() == [0.0] * 9 + [3.0]
    assert canonical[0, 1].tolist() == [0.0] * 10
    assert canonical[1, 0].tolist() == [0.0] * 10
    assert canonical[1, 1].tolist() == [5.0] + [0.0] * 9


def test_slot_adapter_routes_masks_by_slot_name() -> None:
    adapter = SlotStateAdapter(("head", "base"), (1, 1))
    _set_scalar_projections(adapter)
    state = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    group_mask = torch.tensor([[True, False], [False, True]])

    canonical, slot_mask = adapter(state, group_mask)

    assert slot_mask.tolist() == [
        [False, False, False, True],
        [False, False, True, False],
    ]
    assert adapter.slot_mask(None, 2, state.device).all(dim=0).tolist() == [False, False, True, True]
    torch.testing.assert_close(canonical[0, CANONICAL_SLOTS.index("head")], torch.full((10,), 2.0))
    torch.testing.assert_close(canonical[1, CANONICAL_SLOTS.index("base")], torch.full((10,), 7.0))
    assert torch.count_nonzero(canonical[0, CANONICAL_SLOTS.index("base")]) == 0
    assert torch.count_nonzero(canonical[1, CANONICAL_SLOTS.index("head")]) == 0


@pytest.mark.parametrize("architecture", ["slot", "token"])
def test_single_group_batch_vector_mask_is_supported(architecture: str) -> None:
    batch_mask = torch.tensor([True, False])
    if architecture == "slot":
        decoder = MorphologyActionDecoder(("head",), ("head",), (1,), (2,), ("zero",))
        adapter: SlotStateAdapter | TokenStateAdapter = SlotStateAdapter(("head",), (1,))
        adapter_mask = adapter.slot_mask(batch_mask, 2, torch.device("cpu"))
        expected_adapter_mask = [[False, False, False, True], [False, False, False, False]]
    else:
        parts = (PartDescriptor("tool", "scalar"),)
        decoder = PartTokenActionDecoder(parts, ("tool",), (1,), (2,), ("zero",))
        adapter = TokenStateAdapter(parts, ("tool",), (1,))
        adapter_mask = adapter.part_mask(batch_mask, 2, torch.device("cpu"))
        expected_adapter_mask = [[True], [False]]

    assert decoder.native_dim_mask(batch_mask, 2, torch.device("cpu")).tolist() == [
        [True, True],
        [False, False],
    ]
    assert adapter_mask.tolist() == expected_adapter_mask


def test_part_decoder_rejects_malformed_action_state_and_masks() -> None:
    decoder = _safety_decoder("token")
    canonical = torch.zeros(2, 3, 2, 10)
    state = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="canonical actions"):
        decoder(torch.zeros(2, 3, 2, 9), state)
    with pytest.raises(ValueError, match="state must have shape"):
        decoder(canonical, torch.zeros(3, 3))
    with pytest.raises(ValueError, match="dimensional native state"):
        decoder(canonical, torch.zeros(2, 4))
    with pytest.raises(ValueError, match="action_group_mask"):
        decoder(canonical, state, torch.ones(2))
    with pytest.raises(ValueError, match="action_group_mask"):
        decoder.native_dim_mask(torch.ones(2, 2, 1), 2, state.device)
    with pytest.raises(ValueError, match="native state"):
        decoder.safe_native_action(torch.zeros(2, 1, 3))


def test_slot_decoder_rejects_malformed_action_state_and_masks() -> None:
    decoder = _safety_decoder("slot")
    canonical = torch.zeros(2, 3, 25)
    state = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="canonical actions"):
        decoder(torch.zeros(2, 3, 24), state)
    with pytest.raises(ValueError, match="decoder state"):
        decoder(canonical, torch.zeros(2, 4))
    with pytest.raises(ValueError, match="action_group_mask"):
        decoder(canonical, state, torch.ones(2))
    with pytest.raises(ValueError, match="action_group_mask"):
        decoder.native_dim_mask(torch.ones(2, 2, 1), 2, state.device)
    with pytest.raises(ValueError, match="native state"):
        decoder.safe_native_action(torch.zeros(2, 1, 3))


def test_slot_decoder_rejects_mismatched_state_batch_with_value_error() -> None:
    decoder = _safety_decoder("slot")

    with pytest.raises(ValueError, match="state must have shape"):
        decoder(torch.zeros(2, 3, 25), torch.zeros(3, 3))


@pytest.mark.parametrize("architecture", ["slot", "token"])
def test_state_adapter_rejects_malformed_state_and_mask(architecture: str) -> None:
    if architecture == "slot":
        adapter: SlotStateAdapter | TokenStateAdapter = SlotStateAdapter(("head", "base"), (1, 1))
    else:
        parts = (PartDescriptor("left", "scalar"), PartDescriptor("right", "scalar"))
        adapter = TokenStateAdapter(parts, ("right", "left"), (1, 1))

    with pytest.raises(ValueError, match="native state"):
        adapter(torch.zeros(2, 1, 2))
    with pytest.raises(ValueError, match="native state"):
        adapter(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="action_group_mask"):
        adapter(torch.zeros(2, 2), torch.ones(2))
