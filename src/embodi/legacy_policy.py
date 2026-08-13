from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
import json

import torch
from torch import Tensor, nn

from .action_decoder import TinyActionDecoder
from .backbone import LFMBackboneAdapter
from .config import CANONICAL_ACTION_DIMS
from .expert import FlowMatchingActionExpert


@dataclass
class LegacyConfig:
    model_name: str = "embodi00-so100-pickplace"
    backbone_name: str = "LiquidAI/LFM2.5-VL-450M"
    backbone_revision: str = "fc6221ca597f3315e4f82fc2df606783267b34ba"
    camera_names: tuple[str, ...] = ("top", "wrist")
    image_do_rescale: bool = True
    state_dim: int = 6
    action_dim: int = 6
    action_group_types: tuple[str, ...] | None = None
    action_group_state_dims: tuple[int, ...] | None = None
    action_group_native_dims: tuple[int, ...] | None = None
    chunk_size: int = 32
    n_action_steps: int = 32
    expert_width: int = 512
    expert_layers: int = 12
    expert_heads: int = 8
    expert_ff_multiplier: int = 4
    dropout: float = 0.0
    denoising_steps: int = 10
    decoder_hidden_width: int = 128
    decoder_layers: int = 2
    decoder_loss_weight: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    freeze_vision_encoder: bool = True

    def __post_init__(self) -> None:
        self.camera_names = tuple(self.camera_names)
        for name in ("action_group_types", "action_group_state_dims", "action_group_native_dims"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, tuple(value))
        has_groups = self.action_group_types is not None
        if has_groups != (self.action_group_state_dims is not None) or has_groups != (
            self.action_group_native_dims is not None
        ):
            raise ValueError("legacy action group fields must be present together")

    @property
    def has_action_decoder(self) -> bool:
        return self.action_group_types is not None

    @property
    def expert_action_dim(self) -> int:
        if self.action_group_types is None:
            return self.action_dim
        return sum(CANONICAL_ACTION_DIMS[kind] for kind in self.action_group_types)

    @classmethod
    def load(cls, path: str | Path) -> LegacyConfig:
        values = json.loads(Path(path).read_text())
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unsupported legacy config fields: {unknown}")
        return cls(**values)


def _slices(widths: tuple[int, ...]) -> list[slice]:
    result = []
    offset = 0
    for width in widths:
        result.append(slice(offset, offset + width))
        offset += width
    return result


class LegacyActionDecoder(nn.Module):
    def __init__(self, config: LegacyConfig) -> None:
        super().__init__()
        if not config.has_action_decoder:
            raise ValueError("legacy config has no action decoder")
        self.group_types = config.action_group_types
        self.state_slices = _slices(config.action_group_state_dims)
        self.native_slices = _slices(config.action_group_native_dims)
        canonical_widths = tuple(CANONICAL_ACTION_DIMS[kind] for kind in self.group_types)
        self.canonical_slices = _slices(canonical_widths)
        self.decoders = nn.ModuleDict()
        for kind, state_dim, native_dim in zip(
            self.group_types,
            config.action_group_state_dims,
            config.action_group_native_dims,
            strict=True,
        ):
            if kind in self.decoders:
                continue
            self.decoders[kind] = TinyActionDecoder(
                CANONICAL_ACTION_DIMS[kind] + state_dim,
                native_dim,
                hidden_width=config.decoder_hidden_width,
                layers=config.decoder_layers,
            )

    def forward(self, canonical_actions: Tensor, state: Tensor) -> Tensor:
        batch_size, chunk_size, _ = canonical_actions.shape
        state = state.unsqueeze(1).expand(batch_size, chunk_size, state.shape[-1])
        native_dim = max(action_slice.stop for action_slice in self.native_slices)
        native = canonical_actions.new_zeros(batch_size, chunk_size, native_dim)
        for index, kind in enumerate(self.group_types):
            native[..., self.native_slices[index]] = self.decoders[kind](
                canonical_actions[..., self.canonical_slices[index]],
                state[..., self.state_slices[index]],
            )
        return native


class LegacyEmbodiPolicy(nn.Module):
    """Inference-only adapter for pre-format-v2 direct and padded-canonical checkpoints."""

    def __init__(self, config: LegacyConfig, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone or LFMBackboneAdapter.from_pretrained(
            config.backbone_name,
            state_dim=config.state_dim,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            freeze_vision=config.freeze_vision_encoder,
            revision=config.backbone_revision,
        )
        context_width = getattr(self.backbone, "hidden_size", None)
        if context_width is None:
            raise ValueError("backbone must expose hidden_size")
        self.expert = FlowMatchingActionExpert(
            action_dim=config.expert_action_dim,
            context_width=context_width,
            chunk_size=config.chunk_size,
            width=config.expert_width,
            layers=config.expert_layers,
            heads=config.expert_heads,
            ff_multiplier=config.expert_ff_multiplier,
            dropout=config.dropout,
            mask_width=1,
        )
        self.action_decoder = LegacyActionDecoder(config) if config.has_action_decoder else None
        self.register_buffer("state_mean", torch.zeros(config.state_dim))
        self.register_buffer("state_std", torch.ones(config.state_dim))
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))
        if config.has_action_decoder:
            self.register_buffer("canonical_action_mean", torch.zeros(config.expert_action_dim))
            self.register_buffer("canonical_action_std", torch.ones(config.expert_action_dim))
        self.reset()

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def _latest_state(self, batch: dict[str, Any]) -> Tensor:
        state = batch["observation.state"]
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(f"expected legacy state [B, {self.config.state_dim}]")
        return state

    def encode(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        state = (self._latest_state(batch) - self.state_mean) / self.state_std
        excluded = {
            "observation.state",
            "canonical_state",
            "canonical_slot_mask",
            "action",
            "canonical_action",
            "action_group_mask",
            "action_is_pad",
            "task",
        }
        model_inputs = {key: value for key, value in batch.items() if key not in excluded}
        return self.backbone(state, **model_inputs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        self.eval()
        context, context_mask = self.encode(batch)
        sampled = self.expert.sample(
            context,
            context_mask,
            steps=self.config.denoising_steps,
            noise=noise,
        )
        if self.action_decoder is None:
            return sampled * self.action_std + self.action_mean
        state = (self._latest_state(batch) - self.state_mean) / self.state_std
        native = self.action_decoder(sampled, state)
        return native * self.action_std + self.action_mean

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, noise=noise)
            self._action_queue.extend(chunk[:, : self.config.n_action_steps].transpose(0, 1))
        return self._action_queue.popleft()

    def load_checkpoint(self, directory: str | Path) -> int:
        payload = torch.load(Path(directory) / "checkpoint.pt", map_location="cpu", weights_only=True)
        model_state = payload.get("model")
        if not isinstance(model_state, dict):
            raise ValueError("legacy checkpoint does not contain a model state")
        incompatible = self.load_state_dict(model_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected legacy checkpoint keys: {incompatible.unexpected_keys}")
        allowed_missing = {"expert.mask_projection.weight"}
        missing_project_weights = [
            name
            for name in incompatible.missing_keys
            if name in allowed_missing or not name.startswith("backbone.model.")
        ]
        if set(missing_project_weights) != allowed_missing:
            raise RuntimeError(f"legacy checkpoint is missing required keys: {missing_project_weights}")
        return int(payload.get("step", 0))

    @classmethod
    def from_checkpoint(cls, directory: str | Path, device: torch.device | str) -> LegacyEmbodiPolicy:
        config = LegacyConfig.load(Path(directory) / "config.json")
        policy = cls(config).to(device)
        policy.load_checkpoint(directory)
        policy.eval()
        policy.reset()
        return policy
