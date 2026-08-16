from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import CANONICAL_ACTION_DIMS, canonical_slot_slices


def _slices(widths: tuple[int, ...]) -> list[slice]:
    offset = 0
    result = []
    for width in widths:
        result.append(slice(offset, offset + width))
        offset += width
    return result


class TinyActionDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_width: int = 128, layers: int = 2) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        modules: list[nn.Module] = [nn.Linear(input_dim, hidden_width), nn.GELU()]
        for _ in range(layers - 2):
            modules.extend((nn.Linear(hidden_width, hidden_width), nn.GELU()))
        modules.append(nn.Linear(hidden_width, output_dim))
        self.network = nn.Sequential(*modules)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, canonical_action: Tensor, state: Tensor) -> Tensor:
        return self.network(torch.cat((canonical_action, state), dim=-1))


class MorphologyActionDecoder(nn.Module):
    def __init__(
        self,
        group_slots: tuple[str, ...],
        group_types: tuple[str, ...],
        group_state_dims: tuple[int, ...],
        group_native_dims: tuple[int, ...],
        inactive_modes: tuple[str, ...] | None = None,
        hidden_width: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        count = len(group_types)
        if any(len(values) != count for values in (group_slots, group_state_dims, group_native_dims)):
            raise ValueError("action group settings must have matching counts")
        self.group_slots = tuple(group_slots)
        self.group_types = tuple(group_types)
        self.group_state_dims = tuple(group_state_dims)
        self.group_native_dims = tuple(group_native_dims)
        self.inactive_modes = tuple(inactive_modes or ("hold",) * count)
        self.state_slices = _slices(self.group_state_dims)
        self.native_slices = _slices(self.group_native_dims)
        global_slices = canonical_slot_slices()
        self.canonical_slices = [global_slices[slot] for slot in self.group_slots]

        self.decoders = nn.ModuleDict()
        for slot, group_type, state_dim, native_dim in zip(
            self.group_slots,
            self.group_types,
            self.group_state_dims,
            self.group_native_dims,
            strict=True,
        ):
            self.decoders[slot.replace(".", "_")] = TinyActionDecoder(
                CANONICAL_ACTION_DIMS[group_type] + state_dim,
                native_dim,
                hidden_width=hidden_width,
                layers=layers,
            )

    @property
    def native_dim(self) -> int:
        return sum(self.group_native_dims)

    def _validate_group_mask(self, group_mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if group_mask is None:
            return torch.ones(batch_size, len(self.group_types), device=device, dtype=torch.bool)
        if group_mask.ndim == 1 and len(self.group_types) == 1:
            group_mask = group_mask.unsqueeze(-1)
        if group_mask.shape != (batch_size, len(self.group_types)):
            raise ValueError(f"expected action_group_mask [B, {len(self.group_types)}]")
        return group_mask.to(device=device, dtype=torch.bool)

    def native_dim_mask(self, group_mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        group_mask = self._validate_group_mask(group_mask, batch_size, device)
        return torch.cat(
            [group_mask[:, index : index + 1].expand(batch_size, width) for index, width in enumerate(self.group_native_dims)],
            dim=-1,
        )

    def native_to_canonical(self, native_actions: Tensor, canonical_dim: int) -> Tensor:
        """Legacy-only packing; meaningful cross-robot training must provide canonical_action."""
        if native_actions.ndim != 3 or native_actions.shape[-1] != self.native_dim:
            raise ValueError(f"expected native actions [B, T, {self.native_dim}]")
        canonical = native_actions.new_zeros(*native_actions.shape[:2], canonical_dim)
        for native_slice, canonical_slice, native_dim, group_type in zip(
            self.native_slices, self.canonical_slices, self.group_native_dims, self.group_types, strict=True
        ):
            width = min(native_dim, CANONICAL_ACTION_DIMS[group_type])
            canonical[..., canonical_slice.start : canonical_slice.start + width] = native_actions[
                ..., native_slice.start : native_slice.start + width
            ]
        return canonical

    def safe_native_action(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != sum(self.group_state_dims):
            raise ValueError(f"expected native state [B, {sum(self.group_state_dims)}]")
        safe = state.new_zeros(state.shape[0], self.native_dim)
        for mode, state_slice, native_slice in zip(
            self.inactive_modes, self.state_slices, self.native_slices, strict=True
        ):
            if mode == "hold":
                safe[..., native_slice] = state[..., state_slice]
        return safe

    def forward(self, canonical_actions: Tensor, state: Tensor, group_mask: Tensor | None = None) -> Tensor:
        expected_dim = max(action_slice.stop for action_slice in canonical_slot_slices().values())
        if canonical_actions.ndim != 3 or canonical_actions.shape[-1] != expected_dim:
            raise ValueError(f"expected canonical actions [B, T, {expected_dim}]")
        batch_size, chunk_size, _ = canonical_actions.shape
        if state.ndim == 2:
            if state.shape[0] != batch_size:
                raise ValueError("state must have shape [B, state_dim] or [B, T, state_dim]")
            state = state.unsqueeze(1).expand(batch_size, chunk_size, state.shape[-1])
        if state.ndim != 3 or state.shape[:2] != (batch_size, chunk_size):
            raise ValueError("state must have shape [B, state_dim] or [B, T, state_dim]")
        if state.shape[-1] != sum(self.group_state_dims):
            raise ValueError(f"expected {sum(self.group_state_dims)}-dimensional decoder state")
        group_mask = self._validate_group_mask(group_mask, batch_size, canonical_actions.device)
        native_actions = canonical_actions.new_zeros(batch_size, chunk_size, self.native_dim)
        for index in range(len(self.group_types)):
            decoder_key = self.group_slots[index].replace(".", "_")
            decoded = self.decoders[decoder_key](
                canonical_actions[..., self.canonical_slices[index]], state[..., self.state_slices[index]]
            )
            decoded = decoded * group_mask[:, index].view(batch_size, 1, 1).to(decoded.dtype)
            native_actions[..., self.native_slices[index]] = decoded
        return native_actions
