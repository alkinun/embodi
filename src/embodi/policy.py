from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Literal
import json

import torch
from torch import Tensor, nn

from .action_decoder import MorphologyActionDecoder
from .backbone import LFMBackboneAdapter
from .config import (
    CANONICAL_ACTION_DIMS,
    CANONICAL_SLOTS,
    CANONICAL_SLOT_TYPES,
    CANONICAL_STATE_WIDTH,
    EmbodiConfig,
    canonical_action_normalization,
    canonical_neutral_action,
    canonical_state_normalization,
)
from .expert import FlowMatchingActionExpert
from .state_adapter import EmbodimentStateAdapter

TrainingStage = Literal["core", "decoder", "refine"]


class EmbodiCore(nn.Module):
    def __init__(self, config: EmbodiConfig, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.backbone = backbone or LFMBackboneAdapter.from_pretrained(
            config.backbone_name,
            state_dim=CANONICAL_STATE_WIDTH,
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
            action_dim=config.canonical_action_dim,
            context_width=context_width,
            chunk_size=config.chunk_size,
            width=config.expert_width,
            layers=config.expert_layers,
            heads=config.expert_heads,
            ff_multiplier=config.expert_ff_multiplier,
            dropout=config.dropout,
            mask_width=config.canonical_slot_count,
        )


class EmbodiPolicy(nn.Module):
    def __init__(
        self,
        config: EmbodiConfig,
        backbone: nn.Module | None = None,
        state_adapter: EmbodimentStateAdapter | None = None,
        action_decoder: MorphologyActionDecoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.core = EmbodiCore(config, backbone)
        self.state_adapter = state_adapter or EmbodimentStateAdapter(
            config.action_group_slots, config.action_group_state_dims
        )
        self.action_decoder = action_decoder or MorphologyActionDecoder(
            config.action_group_slots,
            config.action_group_types,
            config.action_group_state_dims,
            config.action_group_native_dims,
            config.inactive_action_modes,
            hidden_width=config.decoder_hidden_width,
            layers=config.decoder_layers,
        )
        self.register_buffer("state_mean", torch.zeros(config.state_dim))
        self.register_buffer("state_std", torch.ones(config.state_dim))
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))
        canonical_action_mean, canonical_action_std = canonical_action_normalization()
        canonical_state_mean, canonical_state_std = canonical_state_normalization()
        self.register_buffer("canonical_action_mean", torch.tensor(canonical_action_mean))
        self.register_buffer("canonical_action_std", torch.tensor(canonical_action_std))
        self.register_buffer(
            "canonical_state_mean",
            torch.tensor(canonical_state_mean),
        )
        self.register_buffer(
            "canonical_state_std",
            torch.tensor(canonical_state_std),
        )
        self.register_buffer("canonical_neutral", torch.tensor(canonical_neutral_action()))
        self._core_trainable = {
            name for name, parameter in self.core.named_parameters() if parameter.requires_grad
        }
        self.stage: TrainingStage = "refine"
        self.reset()

    @property
    def backbone(self) -> nn.Module:
        return self.core.backbone

    @property
    def expert(self) -> FlowMatchingActionExpert:
        return self.core.expert

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def configure_stage(self, stage: TrainingStage) -> None:
        if stage not in ("core", "decoder", "refine"):
            raise ValueError(f"unknown training stage: {stage}")
        self.stage = stage
        for name, parameter in self.core.named_parameters():
            parameter.requires_grad_(stage != "decoder" and name in self._core_trainable)
        for parameter in self.state_adapter.parameters():
            parameter.requires_grad_(True)
        for parameter in self.action_decoder.parameters():
            parameter.requires_grad_(stage != "core")

    def get_optim_params(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @staticmethod
    def _copy_stats(target_mean: Tensor, target_std: Tensor, stats: dict[str, Any], name: str) -> None:
        if name not in stats or "mean" not in stats[name] or "std" not in stats[name]:
            raise KeyError(f"normalization stats for {name} require mean and std")
        mean = torch.as_tensor(stats[name]["mean"], device=target_mean.device, dtype=target_mean.dtype)
        std = torch.as_tensor(stats[name]["std"], device=target_std.device, dtype=target_std.dtype)
        if mean.shape != target_mean.shape or std.shape != target_std.shape:
            raise ValueError(f"normalization stats for {name} have the wrong shape")
        target_mean.copy_(mean)
        target_std.copy_(std.clamp_min(1e-6))

    def set_normalization_stats(self, stats: dict[str, Any], require_canonical: bool = True) -> None:
        self._copy_stats(self.state_mean, self.state_std, stats, "observation.state")
        if "action" in stats:
            self._copy_stats(self.action_mean, self.action_std, stats, "action")
        if require_canonical and "canonical_action" not in stats:
            raise KeyError("format_version=2 training requires canonical_action statistics")

    def _latest_state(self, batch: dict[str, Any]) -> Tensor:
        state = batch["observation.state"]
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(f"expected native state [B, {self.config.state_dim}]")
        return state

    def _slot_mask(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> Tensor:
        explicit = batch.get("canonical_slot_mask")
        if explicit is not None:
            if explicit.shape != (batch_size, self.config.canonical_slot_count):
                raise ValueError(
                    f"expected canonical_slot_mask [B, {self.config.canonical_slot_count}]"
                )
            return explicit.to(device=device, dtype=torch.bool)
        return self.state_adapter.slot_mask(batch.get("action_group_mask"), batch_size, device)

    def _canonical_dim_mask(self, slot_mask: Tensor) -> Tensor:
        pieces = []
        for index, kind in enumerate(CANONICAL_SLOT_TYPES):
            pieces.append(slot_mask[:, index : index + 1].expand(-1, CANONICAL_ACTION_DIMS[kind]))
        return torch.cat(pieces, dim=-1)

    def _group_mask(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> Tensor:
        explicit = batch.get("action_group_mask")
        if explicit is not None:
            return self.action_decoder._validate_group_mask(explicit, batch_size, device)
        slot_mask = self._slot_mask(batch, batch_size, device)
        indices = [CANONICAL_SLOTS.index(slot) for slot in self.config.action_group_slots]
        return slot_mask[:, indices]

    def _canonical_state(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor | None]:
        native_state = self._latest_state(batch)
        normalized_native = (native_state - self.state_mean) / self.state_std
        predicted, slot_mask = self.state_adapter(normalized_native, batch.get("action_group_mask"))
        slot_mask = self._slot_mask(batch, native_state.shape[0], native_state.device)
        supplied = batch.get("canonical_state")
        if supplied is None:
            return predicted, slot_mask, None
        if supplied.ndim == 2 and supplied.shape[-1] == self.config.canonical_action_dim:
            unpacked = supplied.new_zeros(
                supplied.shape[0], self.config.canonical_slot_count, CANONICAL_STATE_WIDTH
            )
            offset = 0
            for index, kind in enumerate(CANONICAL_SLOT_TYPES):
                width = CANONICAL_ACTION_DIMS[kind]
                unpacked[:, index, :width] = supplied[:, offset : offset + width]
                offset += width
            supplied = unpacked
        if supplied.shape != predicted.shape:
            raise ValueError(
                f"expected canonical_state [B, {self.config.canonical_slot_count}, {CANONICAL_STATE_WIDTH}]"
            )
        normalized = (supplied - self.canonical_state_mean) / self.canonical_state_std
        normalized = normalized * slot_mask.unsqueeze(-1).to(normalized.dtype)
        adapter_loss = ((predicted - normalized.detach()).square() * slot_mask.unsqueeze(-1)).sum()
        adapter_loss = adapter_loss / slot_mask.sum().clamp_min(1) / CANONICAL_STATE_WIDTH
        return normalized, slot_mask, adapter_loss

    def encode(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        canonical_state, slot_mask, adapter_loss = self._canonical_state(batch)
        excluded = {
            "observation.state", "canonical_state", "action", "canonical_action",
            "canonical_slot_mask", "action_group_mask", "action_is_pad", "canonical_action_is_pad", "task",
        }
        model_inputs = {key: value for key, value in batch.items() if key not in excluded}
        context, context_mask = self.backbone(canonical_state, state_mask=slot_mask, **model_inputs)
        return context, context_mask, slot_mask, adapter_loss

    def _canonical_action_targets(self, batch: dict[str, Any]) -> Tensor:
        if "canonical_action" not in batch:
            raise KeyError("canonical_action is required; native joint actions are not canonical supervision")
        actions = batch["canonical_action"]
        expected = (self.config.chunk_size, self.config.canonical_action_dim)
        if actions.shape[1:] != expected:
            raise ValueError(f"expected canonical actions [B, {expected[0]}, {expected[1]}]")
        return actions

    @staticmethod
    def _masked_mse(
        predicted: Tensor,
        target: Tensor,
        time_mask: Tensor | None,
        dim_mask: Tensor,
        reduction: str,
    ) -> Tensor:
        loss = ((predicted - target).square() * dim_mask.unsqueeze(1).to(predicted.dtype)).sum(-1)
        loss = loss / dim_mask.sum(-1).unsqueeze(1).clamp_min(1)
        if time_mask is not None:
            valid = ~time_mask.bool()
            loss = (loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        else:
            loss = loss.mean(-1)
        if reduction == "none":
            return loss
        if reduction != "mean":
            raise ValueError(f"unsupported reduction: {reduction}")
        return loss.mean()

    def _decoder_loss(
        self,
        normalized_canonical: Tensor,
        batch: dict[str, Any],
        reduction: str,
    ) -> Tensor:
        if "action" not in batch:
            raise KeyError("decoder training requires native action")
        native_state = self._latest_state(batch)
        actions = batch["action"]
        if actions.shape[1:] != (self.config.chunk_size, self.config.action_dim):
            raise ValueError(f"expected native actions [B, {self.config.chunk_size}, {self.config.action_dim}]")
        normalized_state = (native_state - self.state_mean) / self.state_std
        normalized_target = (actions - self.action_mean) / self.action_std
        predicted = self.action_decoder(
            normalized_canonical,
            normalized_state,
            self._group_mask(batch, actions.shape[0], actions.device),
        )
        dim_mask = self.action_decoder.native_dim_mask(
            self._group_mask(batch, actions.shape[0], actions.device), actions.shape[0], actions.device
        )
        return self._masked_mse(predicted, normalized_target, batch.get("action_is_pad"), dim_mask, reduction)

    def forward(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
        stage: TrainingStage | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        stage = stage or self.stage
        canonical = self._canonical_action_targets(batch)
        normalized = (canonical - self.canonical_action_mean) / self.canonical_action_std
        slot_mask = self._slot_mask(batch, canonical.shape[0], canonical.device)
        dim_mask = self._canonical_dim_mask(slot_mask)
        inactive = (self.canonical_neutral - self.canonical_action_mean) / self.canonical_action_std
        metrics: dict[str, float] = {}
        total: Tensor | None = None

        if stage in ("core", "refine"):
            context, context_mask, encoded_slot_mask, adapter_loss = self.encode(batch)
            if not torch.equal(slot_mask, encoded_slot_mask):
                raise ValueError("canonical slot and state masks disagree")
            result = self.expert.flow_loss(
                normalized,
                context,
                context_mask,
                batch.get("action_is_pad"),
                dim_mask,
                noise=noise,
                time=time,
                reduction=reduction,
                inactive_value=inactive,
                slot_mask=slot_mask,
                return_predicted_actions=stage == "refine",
            )
            if stage == "refine":
                flow_loss, decoder_input = result
            else:
                flow_loss = result
                decoder_input = None
            total = flow_loss
            metrics["flow_loss"] = flow_loss.mean().detach().item()
            if adapter_loss is not None:
                total = total + self.config.state_adapter_loss_weight * adapter_loss
                metrics["state_adapter_loss"] = adapter_loss.detach().item()
        else:
            _state, _mask, adapter_loss = self._canonical_state(batch)
            decoder_input = normalized.detach()
            if self.config.decoder_input_noise_std:
                decoder_noise = torch.randn_like(decoder_input) if noise is None else noise
                decoder_input = decoder_input + decoder_noise * self.config.decoder_input_noise_std
                decoder_input = torch.where(dim_mask.unsqueeze(1), decoder_input, inactive.view(1, 1, -1))
            if adapter_loss is not None:
                total = self.config.state_adapter_loss_weight * adapter_loss
                metrics["state_adapter_loss"] = adapter_loss.detach().item()

        if stage in ("decoder", "refine"):
            decoder_loss = self._decoder_loss(decoder_input, batch, reduction)
            weighted = self.config.decoder_loss_weight * decoder_loss
            total = weighted if total is None else total + weighted
            metrics["decoder_loss"] = decoder_loss.mean().detach().item()
        if total is None:
            raise RuntimeError("training stage produced no loss")
        return total, metrics

    @torch.no_grad()
    def _sample_normalized(self, batch: dict[str, Any], noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        self.eval()
        context, context_mask, slot_mask, _adapter_loss = self.encode(batch)
        dim_mask = self._canonical_dim_mask(slot_mask)
        inactive = (self.canonical_neutral - self.canonical_action_mean) / self.canonical_action_std
        actions = self.expert.sample(
            context,
            context_mask,
            steps=self.config.denoising_steps,
            noise=noise,
            action_dim_mask=dim_mask,
            inactive_value=inactive,
            slot_mask=slot_mask,
        )
        return actions, slot_mask

    @torch.no_grad()
    def predict_canonical_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        normalized, slot_mask = self._sample_normalized(batch, noise)
        actions = normalized * self.canonical_action_std + self.canonical_action_mean
        return torch.where(
            self._canonical_dim_mask(slot_mask).unsqueeze(1),
            actions,
            self.canonical_neutral.view(1, 1, -1),
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        normalized_canonical, _slot_mask = self._sample_normalized(batch, noise)
        native_state = self._latest_state(batch)
        normalized_state = (native_state - self.state_mean) / self.state_std
        normalized = self.action_decoder(
            normalized_canonical,
            normalized_state,
            self._group_mask(batch, native_state.shape[0], native_state.device),
        )
        actions = normalized * self.action_std + self.action_mean
        group_mask = self._group_mask(batch, native_state.shape[0], native_state.device)
        native_mask = self.action_decoder.native_dim_mask(
            group_mask, native_state.shape[0], native_state.device
        )
        safe = self.action_decoder.safe_native_action(native_state).unsqueeze(1)
        return torch.where(native_mask.unsqueeze(1), actions, safe)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, noise=noise)
            self._action_queue.extend(chunk[:, : self.config.n_action_steps].transpose(0, 1))
        return self._action_queue.popleft()

    def save_checkpoint(
        self,
        directory: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        step: int = 0,
    ) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / "config.json")
        core_state = {
            name: value
            for name, value in self.core.state_dict().items()
            if name in self._core_trainable
        }
        torch.save(
            {
                "format_version": 2,
                "signature": self.config.core_signature(),
                "core": core_state,
                "canonical_action_mean": self.canonical_action_mean,
                "canonical_action_std": self.canonical_action_std,
                "canonical_state_mean": self.canonical_state_mean,
                "canonical_state_std": self.canonical_state_std,
            },
            directory / "core.pt",
        )
        torch.save(
            {
                "format_version": 2,
                "signature": self.config.embodiment_signature(),
                "state_adapter": self.state_adapter.state_dict(),
                "action_decoder": self.action_decoder.state_dict(),
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
            },
            directory / "embodiment.pt",
        )
        trainer: dict[str, Any] = {"format_version": 2, "step": step, "stage": self.stage}
        if optimizer is not None:
            trainer["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            trainer["scheduler"] = scheduler.state_dict()
        torch.save(trainer, directory / "trainer.pt")

    def load_core_checkpoint(self, directory: str | Path) -> None:
        directory = Path(directory)
        if not (directory / "core.pt").is_file() and (directory / "checkpoint.pt").is_file():
            raise ValueError(
                "legacy checkpoint has no separable canonical core; native-space experts require explicit conversion"
            )
        payload = torch.load(directory / "core.pt", map_location="cpu", weights_only=True)
        if payload.get("format_version") != 2 or payload.get("signature") != self.config.core_signature():
            raise ValueError("core checkpoint schema or architecture does not match")
        missing = self._core_trainable - payload["core"].keys()
        if missing:
            raise RuntimeError(f"core checkpoint is missing trainable keys: {sorted(missing)}")
        incompatible = self.core.load_state_dict(payload["core"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected core checkpoint keys: {incompatible.unexpected_keys}")
        for name in (
            "canonical_action_mean", "canonical_action_std", "canonical_state_mean", "canonical_state_std"
        ):
            getattr(self, name).copy_(payload[name])

    def load_embodiment_checkpoint(self, directory: str | Path) -> None:
        directory = Path(directory)
        if not (directory / "embodiment.pt").is_file() and (directory / "checkpoint.pt").is_file():
            raise ValueError("legacy checkpoint has no independently loadable embodiment component")
        payload = torch.load(directory / "embodiment.pt", map_location="cpu", weights_only=True)
        if payload.get("format_version") != 2 or payload.get("signature") != self.config.embodiment_signature():
            raise ValueError("embodiment checkpoint schema or architecture does not match")
        self.state_adapter.load_state_dict(payload["state_adapter"])
        self.action_decoder.load_state_dict(payload["action_decoder"])
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            getattr(self, name).copy_(payload[name])

    def load_checkpoint(
        self,
        directory: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        allow_camera_mismatch: bool = False,
    ) -> int:
        directory = Path(directory)
        saved = EmbodiConfig.load(directory / "config.json")
        current = self.config
        if saved != current:
            saved.camera_names = current.camera_names
            if not allow_camera_mismatch or saved != current:
                raise ValueError("checkpoint model config does not match the current config")
        self.load_core_checkpoint(directory)
        self.load_embodiment_checkpoint(directory)
        trainer = torch.load(directory / "trainer.pt", map_location="cpu", weights_only=True)
        if optimizer is not None:
            if trainer.get("stage") != self.stage:
                raise ValueError("cannot resume an optimizer under a different training stage")
            optimizer.load_state_dict(trainer["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(trainer["scheduler"])
        return int(trainer.get("step", 0))

    @classmethod
    def from_checkpoint(cls, directory: str | Path, device: torch.device | str) -> EmbodiPolicy:
        config_path = Path(directory) / "config.json"
        config_values = json.loads(config_path.read_text())
        if config_values.get("format_version") != 2:
            from .legacy_policy import LegacyEmbodiPolicy

            return LegacyEmbodiPolicy.from_checkpoint(directory, device)
        config = EmbodiConfig.load(config_path)
        policy = cls(config).to(device)
        policy.load_checkpoint(directory)
        policy.eval()
        policy.reset()
        return policy
