from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Literal
import json

import torch
from torch import Tensor, nn

from .backbone import LFMBackboneAdapter
from .token_config import (
    CANONICAL_ACTION_SCALE,
    CANONICAL_BLOCK_WIDTH,
    CANONICAL_MEAN,
    CANONICAL_STATE_SCALE,
    PART_NAME_FEATURE_WIDTH,
    EmbodiConfig,
    part_name_features,
)
from .token_decoder import PartTokenActionDecoder
from .token_expert import PartTokenActionExpert
from .token_state_adapter import EmbodimentStateAdapter

TrainingStage = Literal["core", "decoder", "predicted_decoder", "refine"]


class EmbodiCore(nn.Module):
    def __init__(self, config: EmbodiConfig, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.backbone = backbone or LFMBackboneAdapter.from_pretrained(
            config.backbone_name,
            state_dim=CANONICAL_BLOCK_WIDTH,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            freeze_vision=config.freeze_vision_encoder,
            condition_parts=True,
            revision=config.backbone_revision,
        )
        context_width = getattr(self.backbone, "hidden_size", None)
        if context_width is None:
            raise ValueError("backbone must expose hidden_size")
        self.expert = PartTokenActionExpert(
            context_width=context_width,
            chunk_size=config.chunk_size,
            width=config.expert_width,
            layers=config.expert_layers,
            heads=config.expert_heads,
            ff_multiplier=config.expert_ff_multiplier,
            dropout=config.dropout,
        )


class EmbodiPolicy(nn.Module):
    def __init__(
        self,
        config: EmbodiConfig,
        backbone: nn.Module | None = None,
        state_adapter: EmbodimentStateAdapter | None = None,
        action_decoder: PartTokenActionDecoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.core = EmbodiCore(config, backbone)
        self.state_adapter = state_adapter or EmbodimentStateAdapter(
            config.parts, config.action_group_part_names, config.action_group_state_dims
        )
        self.action_decoder = action_decoder or PartTokenActionDecoder(
            config.parts,
            config.action_group_part_names,
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
        self.register_buffer("canonical_mean", torch.tensor(CANONICAL_MEAN))
        self.register_buffer("canonical_action_scale", torch.tensor(CANONICAL_ACTION_SCALE))
        self.register_buffer("canonical_state_scale", torch.tensor(CANONICAL_STATE_SCALE))
        self.register_buffer(
            "configured_part_kind",
            torch.tensor([part.kind_id for part in config.parts], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "configured_part_name_features",
            torch.tensor([part_name_features(part.name) for part in config.parts]),
            persistent=False,
        )
        self.register_buffer(
            "configured_channel_mask",
            torch.tensor([part.channel_mask for part in config.parts], dtype=torch.bool),
            persistent=False,
        )
        if config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self._core_trainable = {
            name for name, parameter in self.core.named_parameters() if parameter.requires_grad
        }
        self.stage: TrainingStage = "refine"
        self.reset()

    @property
    def backbone(self) -> nn.Module:
        return self.core.backbone

    @property
    def expert(self) -> PartTokenActionExpert:
        return self.core.expert

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def configure_stage(self, stage: TrainingStage) -> None:
        if stage not in ("core", "decoder", "predicted_decoder", "refine"):
            raise ValueError(f"unknown training stage: {stage}")
        self.stage = stage
        for name, parameter in self.core.named_parameters():
            parameter.requires_grad_(stage not in ("decoder", "predicted_decoder") and name in self._core_trainable)
        for parameter in self.state_adapter.parameters():
            parameter.requires_grad_(stage != "predicted_decoder")
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

    def set_normalization_stats(self, stats: dict[str, Any]) -> None:
        self._copy_stats(self.state_mean, self.state_std, stats, "observation.state")
        if "action" in stats:
            self._copy_stats(self.action_mean, self.action_std, stats, "action")

    def _latest_state(self, batch: dict[str, Any], required: bool = True) -> Tensor | None:
        state = batch.get("observation.state")
        if state is None:
            if required:
                raise KeyError("batch requires observation.state for this stage")
            return None
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(f"expected native state [B, {self.config.state_dim}]")
        return state

    def _descriptors(
        self,
        batch: dict[str, Any],
        batch_size: int,
        part_count: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        explicit_kind = batch.get("canonical_part_kind")
        explicit_names = batch.get("canonical_part_name_features")
        explicit_channels = batch.get("canonical_channel_mask")
        explicit_parts = batch.get("canonical_part_mask")
        has_explicit = any(value is not None for value in (explicit_kind, explicit_names, explicit_channels))
        if has_explicit:
            if explicit_kind is None or explicit_names is None or explicit_channels is None:
                raise ValueError("dynamic part descriptors require kind, name features, and channel mask")
            part_kind = explicit_kind.to(device=device, dtype=torch.long)
            name_features = explicit_names.to(device=device, dtype=torch.float32)
            channel_mask = explicit_channels.to(device=device, dtype=torch.bool)
        else:
            if part_count != self.config.part_count:
                raise ValueError("part count differs from config but dynamic descriptors were not supplied")
            part_kind = self.configured_part_kind.to(device).unsqueeze(0).expand(batch_size, -1)
            name_features = self.configured_part_name_features.to(device).unsqueeze(0).expand(batch_size, -1, -1)
            channel_mask = self.configured_channel_mask.to(device).unsqueeze(0).expand(batch_size, -1, -1)
        expected_kind = (batch_size, part_count)
        if part_kind.shape != expected_kind:
            raise ValueError("canonical_part_kind must have shape [B, P]")
        if name_features.shape != (batch_size, part_count, PART_NAME_FEATURE_WIDTH):
            raise ValueError("canonical_part_name_features must have shape [B, P, name_width]")
        if channel_mask.shape != (batch_size, part_count, CANONICAL_BLOCK_WIDTH):
            raise ValueError("canonical_channel_mask must have shape [B, P, 10]")
        if explicit_parts is None:
            if part_count == self.config.part_count and not has_explicit:
                part_mask = self.state_adapter.part_mask(
                    batch.get("action_group_mask"), batch_size, device
                )
            else:
                part_mask = part_kind != 0
        else:
            part_mask = explicit_parts.to(device=device, dtype=torch.bool)
        if part_mask.shape != (batch_size, part_count):
            raise ValueError("canonical_part_mask must have shape [B, P]")
        part_kind = torch.where(part_mask, part_kind, 0)
        channel_mask = channel_mask & part_mask.unsqueeze(-1)
        return {
            "part_kind": part_kind,
            "part_name_features": name_features,
            "channel_mask": channel_mask,
            "part_mask": part_mask,
        }

    def _canonical_state(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Tensor], Tensor | None]:
        supplied = batch.get("canonical_state")
        native_state = self._latest_state(batch, required=supplied is None)
        if supplied is not None:
            if supplied.ndim != 3 or supplied.shape[-1] != CANONICAL_BLOCK_WIDTH:
                raise ValueError("canonical_state must have shape [B, P, 10]")
            batch_size, part_count = supplied.shape[:2]
            descriptors = self._descriptors(batch, batch_size, part_count, supplied.device)
            normalized = (supplied - self.canonical_mean) / self.canonical_state_scale
            valid = descriptors["part_mask"].unsqueeze(-1) & descriptors["channel_mask"]
            normalized = torch.where(valid, normalized, 0.0)
            adapter_loss = None
            if native_state is not None and part_count == self.config.part_count:
                predicted, predicted_mask = self.state_adapter(
                    (native_state - self.state_mean) / self.state_std,
                    batch.get("action_group_mask"),
                )
                if not torch.equal(predicted_mask, descriptors["part_mask"]):
                    raise ValueError("native group and canonical part masks disagree")
                squared = (predicted - normalized.detach()).square()
                adapter_loss = (squared * valid).sum() / valid.sum().clamp_min(1)
            return normalized, descriptors, adapter_loss
        predicted, part_mask = self.state_adapter(
            (native_state - self.state_mean) / self.state_std,
            batch.get("action_group_mask"),
        )
        synthetic_batch = dict(batch)
        synthetic_batch["canonical_part_mask"] = part_mask
        descriptors = self._descriptors(
            synthetic_batch, predicted.shape[0], predicted.shape[1], predicted.device
        )
        return predicted, descriptors, None

    def encode(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor | None]:
        state, descriptors, adapter_loss = self._canonical_state(batch)
        excluded = {
            "observation.state", "canonical_state", "action", "canonical_action",
            "canonical_part_kind", "canonical_part_name_features", "canonical_channel_mask",
            "canonical_part_mask", "action_group_mask", "action_is_pad",
            "canonical_action_is_pad", "canonical_state_is_pad", "task",
        }
        model_inputs = {key: value for key, value in batch.items() if key not in excluded}
        context, context_mask = self.backbone(
            state,
            state_mask=descriptors["part_mask"],
            part_kind=descriptors["part_kind"],
            part_name_features=descriptors["part_name_features"],
            channel_mask=descriptors["channel_mask"],
            **model_inputs,
        )
        return context, context_mask, descriptors, adapter_loss

    def _canonical_action_targets(self, batch: dict[str, Any]) -> Tensor:
        actions = batch.get("canonical_action")
        if actions is None:
            raise KeyError("canonical_action is required")
        if actions.ndim != 4 or actions.shape[1] != self.config.chunk_size or actions.shape[-1] != CANONICAL_BLOCK_WIDTH:
            raise ValueError(f"expected canonical actions [B, {self.config.chunk_size}, P, 10]")
        return actions

    def _group_mask(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> Tensor:
        explicit = batch.get("action_group_mask")
        if explicit is not None:
            return self.action_decoder.validate_group_mask(explicit, batch_size, device)
        part_mask = batch.get("canonical_part_mask")
        if part_mask is None:
            return self.action_decoder.validate_group_mask(None, batch_size, device)
        if part_mask.shape != (batch_size, self.config.part_count):
            raise ValueError("decoder stages require the configured embodiment part layout")
        part_indices = {part.name: index for index, part in enumerate(self.config.parts)}
        indices = [part_indices[name] for name in self.config.action_group_part_names]
        return part_mask[:, indices].to(device=device, dtype=torch.bool)

    @staticmethod
    def _masked_native_mse(
        predicted: Tensor,
        target: Tensor,
        time_mask: Tensor | None,
        dim_mask: Tensor,
        reduction: str,
        first_step_weight: float = 1.0,
    ) -> Tensor:
        squared = (predicted - target).square()
        valid = dim_mask.unsqueeze(1).expand_as(squared)
        if time_mask is not None:
            valid = valid & ~time_mask.bool().unsqueeze(-1)
        weights = torch.ones_like(squared)
        weights[:, 0] = first_step_weight
        weights = weights * valid
        per_sample = (squared * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1)
        if reduction == "none":
            return per_sample
        if reduction != "mean":
            raise ValueError(f"unsupported reduction: {reduction}")
        return per_sample.mean()

    def _decoder_loss(self, normalized_canonical: Tensor, batch: dict[str, Any], reduction: str) -> Tensor:
        actions = batch.get("action")
        native_state = self._latest_state(batch)
        if actions is None:
            raise KeyError("decoder training requires native action")
        if normalized_canonical.shape[2] != self.config.part_count:
            raise ValueError("decoder training requires the configured embodiment part layout")
        if actions.shape[1:] != (self.config.chunk_size, self.config.action_dim):
            raise ValueError(f"expected native actions [B, {self.config.chunk_size}, {self.config.action_dim}]")
        normalized_state = (native_state - self.state_mean) / self.state_std
        normalized_target = (actions - self.action_mean) / self.action_std
        group_mask = self._group_mask(batch, actions.shape[0], actions.device)
        predicted = self.action_decoder(normalized_canonical, normalized_state, group_mask)
        if self.config.decoder_residual:
            if self.config.state_dim != self.config.action_dim:
                raise ValueError("residual decoding requires matching native state and action dimensions")
            normalized_current = (native_state - self.action_mean) / self.action_std
            predicted = predicted + normalized_current.unsqueeze(1)
        dim_mask = self.action_decoder.native_dim_mask(group_mask, actions.shape[0], actions.device)
        return self._masked_native_mse(
            predicted,
            normalized_target,
            batch.get("action_is_pad"),
            dim_mask,
            reduction,
            self.config.decoder_first_step_weight,
        )

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
        descriptors = self._descriptors(
            batch, canonical.shape[0], canonical.shape[2], canonical.device
        )
        normalized = (canonical - self.canonical_mean) / self.canonical_action_scale
        valid_channels = descriptors["part_mask"].unsqueeze(-1) & descriptors["channel_mask"]
        normalized = torch.where(valid_channels.unsqueeze(1), normalized, 0.0)
        metrics: dict[str, float] = {}
        total: Tensor | None = None

        if stage in ("core", "predicted_decoder", "refine"):
            context, context_mask, encoded_descriptors, adapter_loss = self.encode(batch)
            for name in ("part_kind", "channel_mask", "part_mask"):
                if not torch.equal(descriptors[name], encoded_descriptors[name]):
                    raise ValueError("canonical action and state descriptors disagree")
            common = {
                "part_kind": descriptors["part_kind"],
                "part_name_features": descriptors["part_name_features"],
                "channel_mask": descriptors["channel_mask"],
                "part_mask": descriptors["part_mask"],
                "context_mask": context_mask,
                "action_is_pad": batch.get("action_is_pad"),
                "reduction": reduction,
                "first_step_weight": self.config.flow_first_step_weight,
                "channel_weights": normalized.new_tensor(self.config.canonical_channel_loss_weights),
            }
            if self.config.core_objective == "regression":
                result = self.expert.regression_loss(
                    normalized,
                    context,
                    return_prediction=stage in ("predicted_decoder", "refine"),
                    **common,
                )
            else:
                result = self.expert.flow_loss(
                    normalized,
                    context,
                    noise=noise,
                    time=time,
                    return_clean_estimate=stage in ("predicted_decoder", "refine"),
                    **common,
                )
            if stage in ("predicted_decoder", "refine"):
                flow_loss, decoder_input = result
            else:
                flow_loss = result
                decoder_input = None
            if stage == "predicted_decoder":
                decoder_input = decoder_input.detach()
            else:
                total = flow_loss
                metric_name = "regression_loss" if self.config.core_objective == "regression" else "flow_loss"
                metrics[metric_name] = flow_loss.mean().detach().item()
                if adapter_loss is not None:
                    total = total + self.config.state_adapter_loss_weight * adapter_loss
                    metrics["state_adapter_loss"] = adapter_loss.detach().item()
        else:
            _state, _descriptors, adapter_loss = self._canonical_state(batch)
            decoder_input = normalized.detach()
            if self.config.decoder_input_noise_std:
                decoder_noise = torch.randn_like(decoder_input) if noise is None else noise
                decoder_input = decoder_input + decoder_noise * self.config.decoder_input_noise_std
                decoder_input = torch.where(valid_channels.unsqueeze(1), decoder_input, 0.0)
            if adapter_loss is not None:
                total = self.config.state_adapter_loss_weight * adapter_loss
                metrics["state_adapter_loss"] = adapter_loss.detach().item()

        if stage in ("decoder", "predicted_decoder", "refine"):
            decoder_loss = self._decoder_loss(decoder_input, batch, reduction)
            weighted = self.config.decoder_loss_weight * decoder_loss
            total = weighted if total is None else total + weighted
            metrics["decoder_loss"] = decoder_loss.mean().detach().item()
        if total is None:
            raise RuntimeError("training stage produced no loss")
        return total, metrics

    @torch.no_grad()
    def _sample_normalized(
        self, batch: dict[str, Any], noise: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.eval()
        context, context_mask, descriptors, _adapter_loss = self.encode(batch)
        common = {
            "part_kind": descriptors["part_kind"],
            "part_name_features": descriptors["part_name_features"],
            "channel_mask": descriptors["channel_mask"],
            "part_mask": descriptors["part_mask"],
            "context_mask": context_mask,
        }
        if self.config.core_objective == "regression":
            actions = self.expert.predict(context, **common)
        else:
            actions = self.expert.sample(
                context,
                steps=self.config.denoising_steps,
                noise=noise,
                **common,
            )
        return actions, descriptors

    @torch.no_grad()
    def predict_canonical_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        normalized, descriptors = self._sample_normalized(batch, noise)
        physical = normalized * self.canonical_action_scale + self.canonical_mean
        valid = descriptors["part_mask"].unsqueeze(-1) & descriptors["channel_mask"]
        return torch.where(valid.unsqueeze(1), physical, self.canonical_mean.view(1, 1, 1, -1))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor:
        native_state = self._latest_state(batch)
        group_mask = self._group_mask(batch, native_state.shape[0], native_state.device)
        if not group_mask.any():
            safe = self.action_decoder.safe_native_action(native_state).unsqueeze(1)
            return safe.expand(-1, self.config.chunk_size, -1).clone()
        normalized, descriptors = self._sample_normalized(batch, noise)
        if normalized.shape[2] != self.config.part_count:
            raise ValueError("native decoding requires the configured embodiment part layout")
        normalized_state = (native_state - self.state_mean) / self.state_std
        decoded = self.action_decoder(normalized, normalized_state, group_mask)
        if self.config.decoder_residual:
            if self.config.state_dim != self.config.action_dim:
                raise ValueError("residual decoding requires matching native state and action dimensions")
            normalized_current = (native_state - self.action_mean) / self.action_std
            decoded = decoded + normalized_current.unsqueeze(1)
        actions = decoded * self.action_std + self.action_mean
        if self.config.native_action_delta_limits:
            limits = actions.new_tensor(self.config.native_action_delta_limits).view(1, 1, -1)
            current = native_state.unsqueeze(1)
            actions = torch.maximum(torch.minimum(actions, current + limits), current - limits)
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
            name: value for name, value in self.core.state_dict().items() if name in self._core_trainable
        }
        torch.save(
            {
                "format_version": 3,
                "signature": self.config.core_signature(),
                "core": core_state,
            },
            directory / "core.pt",
        )
        torch.save(
            {
                "format_version": 3,
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
        trainer: dict[str, Any] = {"format_version": 3, "step": step, "stage": self.stage}
        if optimizer is not None:
            trainer["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            trainer["scheduler"] = scheduler.state_dict()
        torch.save(trainer, directory / "trainer.pt")

    def load_core_checkpoint(
        self, directory: str | Path, components: tuple[str, ...] = ("backbone", "expert")
    ) -> None:
        if not components or any(component not in {"backbone", "expert"} for component in components):
            raise ValueError("core components must contain backbone and/or expert")
        payload = torch.load(Path(directory) / "core.pt", map_location="cpu", weights_only=True)
        if payload.get("format_version") != 3 or payload.get("signature") != self.config.core_signature():
            raise ValueError("token core checkpoint schema or architecture does not match")
        selected = {
            name: value
            for name, value in payload["core"].items()
            if name.split(".", 1)[0] in components
        }
        expected = {
            name for name in self._core_trainable if name.split(".", 1)[0] in components
        }
        missing = expected - selected.keys()
        if missing:
            raise RuntimeError(f"token core checkpoint is missing trainable keys: {sorted(missing)}")
        incompatible = self.core.load_state_dict(selected, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected token core keys: {incompatible.unexpected_keys}")

    def load_embodiment_checkpoint(self, directory: str | Path) -> None:
        payload = torch.load(Path(directory) / "embodiment.pt", map_location="cpu", weights_only=True)
        if payload.get("format_version") != 3 or payload.get("signature") != self.config.embodiment_signature():
            raise ValueError("token embodiment checkpoint schema or architecture does not match")
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
                raise ValueError("token checkpoint model config does not match")
        self.load_core_checkpoint(directory)
        self.load_embodiment_checkpoint(directory)
        trainer = torch.load(directory / "trainer.pt", map_location="cpu", weights_only=True)
        if optimizer is not None:
            if trainer.get("stage") != self.stage:
                raise ValueError("cannot resume an optimizer under a different training stage")
            if "optimizer" not in trainer:
                raise ValueError("checkpoint does not contain optimizer state")
            optimizer.load_state_dict(trainer["optimizer"])
        if scheduler is not None:
            if "scheduler" not in trainer:
                raise ValueError("checkpoint does not contain scheduler state")
            scheduler.load_state_dict(trainer["scheduler"])
        return int(trainer.get("step", 0))

    @classmethod
    def from_checkpoint(cls, directory: str | Path, device: torch.device | str) -> EmbodiPolicy:
        config_path = Path(directory) / "config.json"
        if json.loads(config_path.read_text()).get("format_version") != 3:
            from .checkpoints import load_inference_policy

            return load_inference_policy(directory, device)
        config = EmbodiConfig.load(config_path)
        policy = cls(config).to(device)
        policy.load_checkpoint(directory)
        policy.eval()
        policy.reset()
        return policy
