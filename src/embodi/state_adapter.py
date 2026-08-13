from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import CANONICAL_SLOTS, CANONICAL_STATE_WIDTH


class EmbodimentStateAdapter(nn.Module):
    def __init__(self, group_slots: tuple[str, ...], group_state_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.group_slots = tuple(group_slots)
        self.group_state_dims = tuple(group_state_dims)
        self.projections = nn.ModuleDict(
            {
                slot.replace(".", "_"): nn.Linear(width, CANONICAL_STATE_WIDTH)
                for slot, width in zip(self.group_slots, self.group_state_dims, strict=True)
            }
        )

    def slot_mask(self, group_mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if group_mask is None:
            group_mask = torch.ones(batch_size, len(self.group_slots), device=device, dtype=torch.bool)
        elif group_mask.ndim == 1 and len(self.group_slots) == 1:
            group_mask = group_mask.unsqueeze(-1)
        if group_mask.shape != (batch_size, len(self.group_slots)):
            raise ValueError(f"expected action_group_mask [B, {len(self.group_slots)}]")
        result = torch.zeros(batch_size, len(CANONICAL_SLOTS), device=device, dtype=torch.bool)
        for group_index, slot in enumerate(self.group_slots):
            result[:, CANONICAL_SLOTS.index(slot)] = group_mask[:, group_index].to(device=device, dtype=torch.bool)
        return result

    def forward(self, normalized_state: Tensor, group_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if normalized_state.ndim != 2 or normalized_state.shape[-1] != sum(self.group_state_dims):
            raise ValueError(f"expected native state [B, {sum(self.group_state_dims)}]")
        batch_size = normalized_state.shape[0]
        slot_mask = self.slot_mask(group_mask, batch_size, normalized_state.device)
        canonical = normalized_state.new_zeros(batch_size, len(CANONICAL_SLOTS), CANONICAL_STATE_WIDTH)
        offset = 0
        for slot, width in zip(self.group_slots, self.group_state_dims, strict=True):
            projected = self.projections[slot.replace(".", "_")](normalized_state[:, offset : offset + width])
            canonical[:, CANONICAL_SLOTS.index(slot)] = projected
            offset += width
        canonical = canonical * slot_mask.unsqueeze(-1).to(canonical.dtype)
        return canonical, slot_mask
