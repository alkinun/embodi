from __future__ import annotations

import torch
from torch import Tensor


def rotation_6d_to_matrix(rotation: Tensor) -> Tensor:
    if rotation.shape[-1] != 6:
        raise ValueError("rotation must have six components")
    first = torch.nn.functional.normalize(rotation[..., :3], dim=-1, eps=1e-6)
    second = rotation[..., 3:]
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1, eps=1e-6)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rotation_6d(rotation: Tensor) -> Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must have shape [..., 3, 3]")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def reanchor_canonical_actions(actions: Tensor, states: Tensor) -> tuple[Tensor, Tensor]:
    """Convert frame-local action deltas into targets relative to the chunk's first state."""
    if actions.ndim != 3 or actions.shape[-1] != 25:
        raise ValueError("canonical actions must have shape [B, T, 25]")
    if states.ndim != 4 or states.shape[:2] != actions.shape[:2] or states.shape[2:] != (4, 10):
        raise ValueError("chunked canonical state must have shape [B, T, 4, 10]")
    anchored = actions.clone()
    for slot, action_offset in ((0, 0), (1, 10)):
        current_position = states[:, :, slot, :3]
        current_rotation = rotation_6d_to_matrix(states[:, :, slot, 3:9])
        local_rotation = rotation_6d_to_matrix(actions[:, :, action_offset + 3 : action_offset + 9])
        target_position = current_position + actions[:, :, action_offset : action_offset + 3]
        target_rotation = current_rotation @ local_rotation
        anchor_position = states[:, :1, slot, :3]
        anchor_rotation = rotation_6d_to_matrix(states[:, :1, slot, 3:9])
        anchored[:, :, action_offset : action_offset + 3] = target_position - anchor_position
        anchored[:, :, action_offset + 3 : action_offset + 9] = matrix_to_rotation_6d(
            anchor_rotation.transpose(-1, -2) @ target_rotation
        )
    anchored[:, :, 20:23] = states[:, :, 2, :3] + actions[:, :, 20:23] - states[:, :1, 2, :3]
    anchored[:, :, 23:25] = states[:, :, 3, :2] + actions[:, :, 23:25] - states[:, :1, 3, :2]
    return anchored, states[:, 0]
