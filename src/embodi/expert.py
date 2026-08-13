from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sinusoidal_time_embedding(time: Tensor, width: int) -> Tensor:
    if width < 4:
        raise ValueError("time embedding width must be at least 4")
    half = width // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = time.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < width:
        embedding = F.pad(embedding, (0, width - embedding.shape[-1]))
    return embedding


class ExpertBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * ff_multiplier, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        actions: Tensor,
        context: Tensor,
        context_mask: Tensor | None,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        normalized = self.self_norm(actions)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=None if action_is_pad is None else action_is_pad.bool(),
            need_weights=False,
        )
        actions = actions + self.dropout(attended)

        attended, _ = self.cross_attention(
            self.cross_norm(actions),
            context,
            context,
            key_padding_mask=None if context_mask is None else ~context_mask.bool(),
            need_weights=False,
        )
        actions = actions + self.dropout(attended)
        return actions + self.dropout(self.feed_forward(self.ff_norm(actions)))


class FlowMatchingActionExpert(nn.Module):
    def __init__(
        self,
        action_dim: int,
        context_width: int,
        chunk_size: int = 32,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
        mask_width: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.action_projection = nn.Linear(action_dim, width)
        self.context_projection = nn.Linear(context_width, width)
        self.mask_projection = nn.Linear(mask_width, width, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(1, chunk_size, width))
        self.time_mlp = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width))
        self.blocks = nn.ModuleList(
            ExpertBlock(width, heads, ff_multiplier, dropout) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, action_dim)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_actions: Tensor,
        time: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
        action_dim_mask: Tensor | None = None,
        inactive_value: Tensor | None = None,
        slot_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"expected noisy actions [B, {self.chunk_size}, {self.action_dim}], "
                f"got {tuple(noisy_actions.shape)}"
            )
        if time.ndim != 1 or time.shape[0] != noisy_actions.shape[0]:
            raise ValueError("time must have shape [B]")
        if action_dim_mask is not None:
            inactive = torch.zeros(self.action_dim, device=noisy_actions.device, dtype=noisy_actions.dtype)
            if inactive_value is not None:
                inactive = inactive_value.to(device=noisy_actions.device, dtype=noisy_actions.dtype)
            noisy_actions = torch.where(action_dim_mask.unsqueeze(1), noisy_actions, inactive.view(1, 1, -1))
        actions = self.action_projection(noisy_actions) + self.position_embedding
        if slot_mask is not None:
            actions = actions + self.mask_projection(slot_mask.to(actions.dtype)).unsqueeze(1)
        actions = actions + self.time_mlp(sinusoidal_time_embedding(time, actions.shape[-1])).unsqueeze(1)
        context = self.context_projection(context.to(self.context_projection.weight.dtype))
        for block in self.blocks:
            actions = block(actions, context, context_mask, action_is_pad)
        velocity = self.output_projection(self.output_norm(actions))
        if action_dim_mask is not None:
            velocity = velocity * action_dim_mask.unsqueeze(1).to(velocity.dtype)
        return velocity

    def flow_loss(
        self,
        actions: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        action_dim_mask: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
        inactive_value: Tensor | None = None,
        slot_mask: Tensor | None = None,
        return_predicted_actions: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch_size = actions.shape[0]
        noise = torch.randn_like(actions) if noise is None else noise
        time = torch.rand(batch_size, device=actions.device, dtype=torch.float32) if time is None else time
        interpolation = time.to(actions.dtype).view(batch_size, 1, 1)
        noisy_actions = (1.0 - interpolation) * noise + interpolation * actions
        predicted_velocity = self(
            noisy_actions,
            time,
            context,
            context_mask,
            action_dim_mask,
            inactive_value,
            slot_mask,
            action_is_pad,
        )
        loss = (predicted_velocity - (actions - noise)).square()
        if action_dim_mask is not None:
            loss_dim_mask = action_dim_mask.unsqueeze(1) if action_dim_mask.ndim == 2 else action_dim_mask
            if loss_dim_mask.ndim != 3 or loss_dim_mask.shape[-1] != self.action_dim:
                raise ValueError(f"expected action_dim_mask [B, D] or [B, T, {self.action_dim}]")
            dim_mask = loss_dim_mask.to(device=loss.device, dtype=loss.dtype)
            loss = (loss * dim_mask).sum(dim=-1) / dim_mask.sum(dim=-1).clamp_min(1)
        else:
            loss = loss.mean(dim=-1)
        if action_is_pad is not None:
            valid = ~action_is_pad.bool()
            loss = (loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
        else:
            loss = loss.mean(dim=-1)
        predicted_actions = noisy_actions + (1.0 - interpolation) * predicted_velocity
        if action_dim_mask is not None and inactive_value is not None:
            prediction_mask = action_dim_mask.unsqueeze(1) if action_dim_mask.ndim == 2 else action_dim_mask
            predicted_actions = torch.where(
                prediction_mask,
                predicted_actions,
                inactive_value.to(predicted_actions).view(1, 1, -1),
            )
        if reduction == "none":
            reduced = loss
            return (reduced, predicted_actions) if return_predicted_actions else reduced
        if reduction != "mean":
            raise ValueError(f"unsupported reduction: {reduction}")
        reduced = loss.mean()
        return (reduced, predicted_actions) if return_predicted_actions else reduced

    @torch.no_grad()
    def sample(
        self,
        context: Tensor,
        context_mask: Tensor | None = None,
        steps: int = 10,
        noise: Tensor | None = None,
        action_dim_mask: Tensor | None = None,
        inactive_value: Tensor | None = None,
        slot_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = context.shape[0]
        if steps <= 0:
            raise ValueError("steps must be positive")
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.chunk_size,
                self.action_dim,
                device=context.device,
                dtype=self.action_projection.weight.dtype,
            )
        actions = noise.to(self.action_projection.weight.dtype)
        if action_dim_mask is not None and inactive_value is not None:
            actions = torch.where(
                action_dim_mask.unsqueeze(1), actions, inactive_value.to(actions).view(1, 1, -1)
            )
        step_size = 1.0 / steps
        for step in range(steps):
            time = torch.full((batch_size,), step * step_size, device=context.device, dtype=torch.float32)
            actions = actions + step_size * self(
                actions, time, context, context_mask, action_dim_mask, inactive_value, slot_mask
            )
            if action_dim_mask is not None and inactive_value is not None:
                actions = torch.where(
                    action_dim_mask.unsqueeze(1), actions, inactive_value.to(actions).view(1, 1, -1)
                )
        return actions
