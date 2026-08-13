from __future__ import annotations

import torch
from torch import Tensor

from .canonical import matrix_to_rotation_6d, rotation_6d_to_matrix
from .token_config import CANONICAL_BLOCK_WIDTH, CANONICAL_MEAN, PART_KIND_IDS


def reanchor_canonical_actions(
    actions: Tensor,
    states: Tensor,
    part_kind: Tensor,
    channel_mask: Tensor,
    part_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    if actions.ndim != 4 or actions.shape[-1] != CANONICAL_BLOCK_WIDTH:
        raise ValueError("canonical actions must have shape [B, T, P, 10]")
    if states.shape != actions.shape:
        raise ValueError("chunked canonical state must match canonical action shape")
    batch_size, _time_steps, part_count, _ = actions.shape
    if part_kind.shape != (batch_size, part_count):
        raise ValueError("part_kind must have shape [B, P]")
    if channel_mask.shape != (batch_size, part_count, CANONICAL_BLOCK_WIDTH):
        raise ValueError("channel_mask must have shape [B, P, 10]")
    if part_mask.shape != (batch_size, part_count):
        raise ValueError("part_mask must have shape [B, P]")
    part_mask = part_mask.to(device=actions.device, dtype=torch.bool)
    channel_mask = channel_mask.to(device=actions.device, dtype=torch.bool)
    neutral = actions.new_tensor(CANONICAL_MEAN).view(1, 1, 1, -1)
    anchored = neutral.expand_as(actions).clone()
    initial_state = actions.new_tensor(CANONICAL_MEAN).view(1, 1, -1).expand(batch_size, part_count, -1).clone()
    initial_state = torch.where(
        (part_mask.unsqueeze(-1) & channel_mask),
        states[:, 0],
        initial_state,
    )
    pose_kind = (part_kind == PART_KIND_IDS["pose"]) | (part_kind == PART_KIND_IDS["pose_scalar"])
    valid_parts = part_mask & pose_kind
    target_position = states[..., :3] + actions[..., :3]
    anchor_position = states[:, :1, :, :3]
    translated = target_position - anchor_position
    translation_valid = valid_parts[:, None, :, None] & channel_mask[:, None, :, :3]
    anchored[..., :3] = torch.where(translation_valid, translated, anchored[..., :3])

    rotation_enabled = channel_mask[..., 3:9].all(dim=-1) & valid_parts
    current_rotation = rotation_6d_to_matrix(states[..., 3:9])
    local_rotation = rotation_6d_to_matrix(actions[..., 3:9])
    target_rotation = current_rotation @ local_rotation
    anchor_rotation = rotation_6d_to_matrix(states[:, :1, :, 3:9])
    relative = matrix_to_rotation_6d(anchor_rotation.transpose(-1, -2) @ target_rotation)
    anchored[..., 3:9] = torch.where(
        rotation_enabled[:, None, :, None], relative, anchored[..., 3:9]
    )
    scalar_valid = part_mask[:, None, :] & channel_mask[:, None, :, 9]
    anchored[..., 9] = torch.where(scalar_valid, actions[..., 9], anchored[..., 9])
    return anchored, initial_state
