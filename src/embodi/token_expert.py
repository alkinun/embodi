from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .token_config import CANONICAL_BLOCK_WIDTH, PART_KIND_IDS, PART_NAME_FEATURE_WIDTH


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
    return F.pad(embedding, (0, width - embedding.shape[-1])) if embedding.shape[-1] < width else embedding


class TokenExpertBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_multiplier: int, dropout: float, mode: str) -> None:
        super().__init__()
        self.mode = mode
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

    @staticmethod
    def _safe_padding_mask(mask: Tensor) -> Tensor:
        mask = mask.clone()
        mask[mask.all(dim=-1)] = False
        return mask

    def forward(
        self,
        tokens: Tensor,
        context: Tensor,
        context_mask: Tensor,
        part_mask: Tensor,
        action_is_pad: Tensor,
    ) -> Tensor:
        batch_size, time_steps, part_count, width = tokens.shape
        normalized = self.self_norm(tokens)
        if self.mode == "temporal":
            queries = normalized.permute(0, 2, 1, 3).reshape(batch_size * part_count, time_steps, width)
            key_padding = action_is_pad[:, None, :].expand(-1, part_count, -1)
            key_padding = key_padding | ~part_mask[:, :, None]
            attended, _ = self.self_attention(
                queries,
                queries,
                queries,
                key_padding_mask=self._safe_padding_mask(key_padding.reshape(batch_size * part_count, time_steps)),
                need_weights=False,
            )
            attended = attended.reshape(batch_size, part_count, time_steps, width).permute(0, 2, 1, 3)
        else:
            queries = normalized.reshape(batch_size * time_steps, part_count, width)
            key_padding = ~part_mask[:, None, :].expand(-1, time_steps, -1)
            key_padding = key_padding | action_is_pad[:, :, None]
            attended, _ = self.self_attention(
                queries,
                queries,
                queries,
                key_padding_mask=self._safe_padding_mask(key_padding.reshape(batch_size * time_steps, part_count)),
                need_weights=False,
            )
            attended = attended.reshape(batch_size, time_steps, part_count, width)
        token_valid = part_mask[:, None, :] & ~action_is_pad[:, :, None]
        tokens = torch.where(token_valid.unsqueeze(-1), tokens + self.dropout(attended), 0.0)

        queries = self.cross_norm(tokens).reshape(batch_size, time_steps * part_count, width)
        attended, _ = self.cross_attention(
            queries,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        attended = attended.reshape(batch_size, time_steps, part_count, width)
        tokens = torch.where(token_valid.unsqueeze(-1), tokens + self.dropout(attended), 0.0)
        tokens = tokens + self.dropout(self.feed_forward(self.ff_norm(tokens)))
        return torch.where(token_valid.unsqueeze(-1), tokens, 0.0)


class PartTokenActionExpert(nn.Module):
    def __init__(
        self,
        context_width: int,
        chunk_size: int = 32,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_projection = nn.Linear(CANONICAL_BLOCK_WIDTH, width)
        self.context_projection = nn.Linear(context_width, width)
        self.kind_embedding = nn.Embedding(len(PART_KIND_IDS) + 1, width, padding_idx=0)
        self.name_projection = nn.Linear(PART_NAME_FEATURE_WIDTH, width, bias=False)
        self.channel_projection = nn.Linear(CANONICAL_BLOCK_WIDTH, width, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(1, chunk_size, 1, width))
        self.time_mlp = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width))
        self.blocks = nn.ModuleList(
            TokenExpertBlock(width, heads, ff_multiplier, dropout, "temporal" if index % 2 == 0 else "part")
            for index in range(layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, CANONICAL_BLOCK_WIDTH)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _validate(
        self,
        actions: Tensor,
        context: Tensor,
        context_mask: Tensor | None,
        part_kind: Tensor,
        part_name_features: Tensor,
        channel_mask: Tensor,
        part_mask: Tensor,
        action_is_pad: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if actions.ndim != 4 or actions.shape[1] != self.chunk_size or actions.shape[-1] != CANONICAL_BLOCK_WIDTH:
            raise ValueError(f"expected canonical actions [B, {self.chunk_size}, P, {CANONICAL_BLOCK_WIDTH}]")
        batch_size, time_steps, part_count, _ = actions.shape
        if part_kind.shape != (batch_size, part_count):
            raise ValueError("part_kind must have shape [B, P]")
        if part_name_features.shape != (batch_size, part_count, PART_NAME_FEATURE_WIDTH):
            raise ValueError(f"part_name_features must have shape [B, P, {PART_NAME_FEATURE_WIDTH}]")
        if channel_mask.shape != (batch_size, part_count, CANONICAL_BLOCK_WIDTH):
            raise ValueError(f"channel_mask must have shape [B, P, {CANONICAL_BLOCK_WIDTH}]")
        if part_mask.shape != (batch_size, part_count):
            raise ValueError("part_mask must have shape [B, P]")
        if part_kind.min() < 0 or part_kind.max() > len(PART_KIND_IDS):
            raise ValueError("part_kind contains an unknown kind id")
        part_mask = part_mask.to(device=actions.device, dtype=torch.bool)
        channel_mask = channel_mask.to(device=actions.device, dtype=torch.bool)
        if (~part_mask).all(dim=-1).any():
            raise ValueError("every sample requires at least one active part")
        active_channels = channel_mask & part_mask.unsqueeze(-1)
        if (active_channels.sum(dim=-1) == 0).logical_and(part_mask).any():
            raise ValueError("every active part requires at least one active channel")
        rotation_count = active_channels[..., 3:9].sum(dim=-1)
        if ((rotation_count != 0) & (rotation_count != 6)).any():
            raise ValueError("rotation6d channels must be enabled as a complete group")
        if action_is_pad is None:
            action_is_pad = torch.zeros(batch_size, time_steps, device=actions.device, dtype=torch.bool)
        if action_is_pad.shape != (batch_size, time_steps):
            raise ValueError("action_is_pad must have shape [B, T]")
        action_is_pad = action_is_pad.to(device=actions.device, dtype=torch.bool)
        if action_is_pad.all(dim=-1).any():
            raise ValueError("every sample requires at least one non-padded timestep")
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        if context_mask.shape != context.shape[:2] or (~context_mask.bool()).all(dim=-1).any():
            raise ValueError("every sample requires at least one valid context token")
        return channel_mask, part_mask, action_is_pad

    @staticmethod
    def _valid_channels(channel_mask: Tensor, part_mask: Tensor, action_is_pad: Tensor) -> Tensor:
        return (
            channel_mask[:, None, :, :]
            & part_mask[:, None, :, None]
            & ~action_is_pad[:, :, None, None]
        )

    def forward(
        self,
        noisy_actions: Tensor,
        time: Tensor,
        context: Tensor,
        *,
        part_kind: Tensor,
        part_name_features: Tensor,
        channel_mask: Tensor,
        part_mask: Tensor,
        context_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        channel_mask, part_mask, action_is_pad = self._validate(
            noisy_actions, context, context_mask, part_kind, part_name_features,
            channel_mask, part_mask, action_is_pad,
        )
        if time.ndim != 1 or time.shape[0] != noisy_actions.shape[0]:
            raise ValueError("time must have shape [B]")
        context_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool) if context_mask is None else context_mask.bool()
        valid = self._valid_channels(channel_mask, part_mask, action_is_pad)
        noisy_actions = torch.where(valid, noisy_actions, 0.0)
        tokens = self.action_projection(noisy_actions) + self.position_embedding
        tokens = tokens + self.kind_embedding(part_kind).unsqueeze(1)
        tokens = tokens + self.name_projection(part_name_features).unsqueeze(1)
        tokens = tokens + self.channel_projection(channel_mask.to(tokens.dtype)).unsqueeze(1)
        tokens = tokens + self.time_mlp(sinusoidal_time_embedding(time, tokens.shape[-1])).view(
            noisy_actions.shape[0], 1, 1, -1
        )
        token_valid = part_mask[:, None, :] & ~action_is_pad[:, :, None]
        tokens = torch.where(token_valid.unsqueeze(-1), tokens, 0.0)
        context = self.context_projection(context.to(self.context_projection.weight.dtype))
        for block in self.blocks:
            tokens = block(tokens, context, context_mask, part_mask, action_is_pad)
        velocity = self.output_projection(self.output_norm(tokens))
        return torch.where(valid, velocity, 0.0)

    def flow_loss(
        self,
        actions: Tensor,
        context: Tensor,
        *,
        part_kind: Tensor,
        part_name_features: Tensor,
        channel_mask: Tensor,
        part_mask: Tensor,
        context_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
        return_clean_estimate: bool = False,
        first_step_weight: float = 1.0,
        channel_weights: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        channel_mask, part_mask, action_is_pad = self._validate(
            actions, context, context_mask, part_kind, part_name_features,
            channel_mask, part_mask, action_is_pad,
        )
        valid = self._valid_channels(channel_mask, part_mask, action_is_pad)
        actions = torch.where(valid, actions, 0.0)
        noise = torch.randn_like(actions) if noise is None else noise
        if noise.shape != actions.shape:
            raise ValueError("noise must match canonical action shape")
        noise = torch.where(valid, noise, 0.0)
        batch_size = actions.shape[0]
        time = torch.rand(batch_size, device=actions.device, dtype=torch.float32) if time is None else time
        interpolation = time.to(actions.dtype).view(batch_size, 1, 1, 1)
        noisy = (1.0 - interpolation) * noise + interpolation * actions
        velocity = self(
            noisy,
            time,
            context,
            part_kind=part_kind,
            part_name_features=part_name_features,
            channel_mask=channel_mask,
            part_mask=part_mask,
            context_mask=context_mask,
            action_is_pad=action_is_pad,
        )
        squared = (velocity - (actions - noise)).square()
        weights = torch.ones_like(squared)
        weights[:, 0] = first_step_weight
        if channel_weights is not None:
            if channel_weights.shape != (actions.shape[-1],):
                raise ValueError("channel_weights must match canonical action width")
            weights = weights * channel_weights.view(1, 1, 1, -1)
        weights = weights * valid
        per_sample = (squared * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1)
        clean = torch.where(valid, noisy + (1.0 - interpolation) * velocity, 0.0)
        if reduction == "none":
            loss = per_sample
        elif reduction == "mean":
            loss = per_sample.mean()
        else:
            raise ValueError(f"unsupported reduction: {reduction}")
        return (loss, clean) if return_clean_estimate else loss

    def regression_loss(
        self,
        actions: Tensor,
        context: Tensor,
        *,
        part_kind: Tensor,
        part_name_features: Tensor,
        channel_mask: Tensor,
        part_mask: Tensor,
        context_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        reduction: str = "mean",
        first_step_weight: float = 1.0,
        return_prediction: bool = False,
        channel_weights: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        channel_mask, part_mask, action_is_pad = self._validate(
            actions, context, context_mask, part_kind, part_name_features,
            channel_mask, part_mask, action_is_pad,
        )
        valid = self._valid_channels(channel_mask, part_mask, action_is_pad)
        prediction = self(
            torch.zeros_like(actions),
            torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32),
            context,
            part_kind=part_kind,
            part_name_features=part_name_features,
            channel_mask=channel_mask,
            part_mask=part_mask,
            context_mask=context_mask,
            action_is_pad=action_is_pad,
        )
        squared = (prediction - actions).square()
        weights = torch.ones_like(squared)
        weights[:, 0] = first_step_weight
        if channel_weights is not None:
            if channel_weights.shape != (actions.shape[-1],):
                raise ValueError("channel_weights must match canonical action width")
            weights = weights * channel_weights.view(1, 1, 1, -1)
        weights = weights * valid
        per_sample = (squared * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1)
        if reduction == "none":
            loss = per_sample
        elif reduction == "mean":
            loss = per_sample.mean()
        else:
            raise ValueError(f"unsupported reduction: {reduction}")
        return (loss, prediction) if return_prediction else loss

    @torch.no_grad()
    def predict(
        self,
        context: Tensor,
        *,
        part_kind: Tensor,
        part_name_features: Tensor,
        channel_mask: Tensor,
        part_mask: Tensor,
        context_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        batch_size, part_count = part_mask.shape
        actions = torch.zeros(
            batch_size,
            self.chunk_size,
            part_count,
            CANONICAL_BLOCK_WIDTH,
            device=context.device,
            dtype=self.action_projection.weight.dtype,
        )
        return self(
            actions,
            torch.zeros(batch_size, device=context.device, dtype=torch.float32),
            context,
            part_kind=part_kind,
            part_name_features=part_name_features,
            channel_mask=channel_mask,
            part_mask=part_mask,
            context_mask=context_mask,
            action_is_pad=action_is_pad,
        )

    @torch.no_grad()
    def sample(
        self,
        context: Tensor,
        *,
        part_kind: Tensor,
        part_name_features: Tensor,
        channel_mask: Tensor,
        part_mask: Tensor,
        context_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        steps: int = 10,
        noise: Tensor | None = None,
    ) -> Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        batch_size, part_count = part_mask.shape
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.chunk_size,
                part_count,
                CANONICAL_BLOCK_WIDTH,
                device=context.device,
                dtype=self.action_projection.weight.dtype,
            )
        channel_mask, part_mask, action_is_pad = self._validate(
            noise, context, context_mask, part_kind, part_name_features,
            channel_mask, part_mask, action_is_pad,
        )
        valid = self._valid_channels(channel_mask, part_mask, action_is_pad)
        actions = torch.where(valid, noise.to(self.action_projection.weight.dtype), 0.0)
        step_size = 1.0 / steps
        for step in range(steps):
            time = torch.full((batch_size,), step * step_size, device=context.device, dtype=torch.float32)
            actions = actions + step_size * self(
                actions,
                time,
                context,
                part_kind=part_kind,
                part_name_features=part_name_features,
                channel_mask=channel_mask,
                part_mask=part_mask,
                context_mask=context_mask,
                action_is_pad=action_is_pad,
            )
            actions = torch.where(valid, actions, 0.0)
        return actions
