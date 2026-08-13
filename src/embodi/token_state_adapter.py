from __future__ import annotations

import torch
from torch import Tensor, nn

from .token_config import CANONICAL_BLOCK_WIDTH, PartDescriptor


class EmbodimentStateAdapter(nn.Module):
    def __init__(
        self,
        parts: tuple[PartDescriptor, ...],
        group_part_names: tuple[str, ...],
        group_state_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.parts = tuple(parts)
        self.group_part_names = tuple(group_part_names)
        self.group_state_dims = tuple(group_state_dims)
        part_indices = {part.name: index for index, part in enumerate(self.parts)}
        self.group_part_indices = tuple(part_indices[name] for name in self.group_part_names)
        self.projections = nn.ModuleList(
            nn.Linear(width, CANONICAL_BLOCK_WIDTH) for width in self.group_state_dims
        )
        self.register_buffer(
            "channel_mask",
            torch.tensor([part.channel_mask for part in self.parts], dtype=torch.bool),
            persistent=False,
        )

    def part_mask(self, group_mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if group_mask is None:
            group_mask = torch.ones(batch_size, len(self.group_part_names), device=device, dtype=torch.bool)
        elif group_mask.ndim == 1 and len(self.group_part_names) == 1:
            group_mask = group_mask.unsqueeze(-1)
        if group_mask.shape != (batch_size, len(self.group_part_names)):
            raise ValueError(f"expected action_group_mask [B, {len(self.group_part_names)}]")
        result = torch.zeros(batch_size, len(self.parts), device=device, dtype=torch.bool)
        for group_index, part_index in enumerate(self.group_part_indices):
            result[:, part_index] = group_mask[:, group_index].to(device=device, dtype=torch.bool)
        return result

    def forward(self, normalized_state: Tensor, group_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if normalized_state.ndim != 2 or normalized_state.shape[-1] != sum(self.group_state_dims):
            raise ValueError(f"expected native state [B, {sum(self.group_state_dims)}]")
        batch_size = normalized_state.shape[0]
        part_mask = self.part_mask(group_mask, batch_size, normalized_state.device)
        canonical = normalized_state.new_zeros(batch_size, len(self.parts), CANONICAL_BLOCK_WIDTH)
        offset = 0
        for group_index, (width, part_index) in enumerate(
            zip(self.group_state_dims, self.group_part_indices, strict=True)
        ):
            canonical[:, part_index] = self.projections[group_index](
                normalized_state[:, offset : offset + width]
            )
            offset += width
        valid = part_mask.unsqueeze(-1) & self.channel_mask.unsqueeze(0)
        return torch.where(valid, canonical, 0.0), part_mask
