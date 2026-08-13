from __future__ import annotations

import torch
from torch import Tensor, nn

from .action_decoder import TinyActionDecoder
from .token_config import CANONICAL_BLOCK_WIDTH, PartDescriptor


def _slices(widths: tuple[int, ...]) -> list[slice]:
    result = []
    offset = 0
    for width in widths:
        result.append(slice(offset, offset + width))
        offset += width
    return result


class PartTokenActionDecoder(nn.Module):
    def __init__(
        self,
        parts: tuple[PartDescriptor, ...],
        group_part_names: tuple[str, ...],
        group_state_dims: tuple[int, ...],
        group_native_dims: tuple[int, ...],
        inactive_modes: tuple[str, ...],
        hidden_width: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.parts = tuple(parts)
        self.group_part_names = tuple(group_part_names)
        self.group_state_dims = tuple(group_state_dims)
        self.group_native_dims = tuple(group_native_dims)
        self.inactive_modes = tuple(inactive_modes)
        part_indices = {part.name: index for index, part in enumerate(self.parts)}
        self.group_part_indices = tuple(part_indices[name] for name in self.group_part_names)
        self.state_slices = _slices(self.group_state_dims)
        self.native_slices = _slices(self.group_native_dims)
        self.decoders = nn.ModuleList(
            TinyActionDecoder(
                CANONICAL_BLOCK_WIDTH + state_width,
                native_width,
                hidden_width=hidden_width,
                layers=layers,
            )
            for state_width, native_width in zip(
                self.group_state_dims, self.group_native_dims, strict=True
            )
        )
        self.register_buffer(
            "channel_mask",
            torch.tensor([part.channel_mask for part in self.parts], dtype=torch.bool),
            persistent=False,
        )

    @property
    def native_dim(self) -> int:
        return sum(self.group_native_dims)

    def validate_group_mask(self, group_mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if group_mask is None:
            return torch.ones(batch_size, len(self.group_part_names), device=device, dtype=torch.bool)
        if group_mask.ndim == 1 and len(self.group_part_names) == 1:
            group_mask = group_mask.unsqueeze(-1)
        if group_mask.shape != (batch_size, len(self.group_part_names)):
            raise ValueError(f"expected action_group_mask [B, {len(self.group_part_names)}]")
        return group_mask.to(device=device, dtype=torch.bool)

    def native_dim_mask(self, group_mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        group_mask = self.validate_group_mask(group_mask, batch_size, device)
        return torch.cat(
            [group_mask[:, index : index + 1].expand(-1, width) for index, width in enumerate(self.group_native_dims)],
            dim=-1,
        )

    def safe_native_action(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != sum(self.group_state_dims):
            raise ValueError(f"expected native state [B, {sum(self.group_state_dims)}]")
        safe = state.new_zeros(state.shape[0], self.native_dim)
        for mode, state_slice, native_slice in zip(
            self.inactive_modes, self.state_slices, self.native_slices, strict=True
        ):
            if mode == "hold":
                safe[:, native_slice] = state[:, state_slice]
        return safe

    def forward(self, canonical_actions: Tensor, state: Tensor, group_mask: Tensor | None = None) -> Tensor:
        if canonical_actions.ndim != 4 or canonical_actions.shape[2:] != (
            len(self.parts), CANONICAL_BLOCK_WIDTH
        ):
            raise ValueError(f"expected canonical actions [B, T, {len(self.parts)}, {CANONICAL_BLOCK_WIDTH}]")
        batch_size, chunk_size = canonical_actions.shape[:2]
        if state.ndim == 2:
            state = state.unsqueeze(1).expand(-1, chunk_size, -1)
        if state.ndim != 3 or state.shape[:2] != (batch_size, chunk_size):
            raise ValueError("state must have shape [B, native_state_dim] or [B, T, native_state_dim]")
        if state.shape[-1] != sum(self.group_state_dims):
            raise ValueError(f"expected {sum(self.group_state_dims)}-dimensional native state")
        group_mask = self.validate_group_mask(group_mask, batch_size, canonical_actions.device)
        native = canonical_actions.new_zeros(batch_size, chunk_size, self.native_dim)
        for index, part_index in enumerate(self.group_part_indices):
            block = torch.where(
                self.channel_mask[part_index].view(1, 1, -1),
                canonical_actions[:, :, part_index],
                0.0,
            )
            decoded = self.decoders[index](block, state[:, :, self.state_slices[index]])
            decoded = decoded * group_mask[:, index].view(batch_size, 1, 1).to(decoded.dtype)
            native[:, :, self.native_slices[index]] = decoded
        return native
