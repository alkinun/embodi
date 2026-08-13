from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class LFMBackboneAdapter(nn.Module):
    """LFM2-VL adapter that exposes the output after its final GQA layer."""

    def __init__(
        self,
        model: nn.Module,
        state_dim: int,
        freeze_vision: bool = True,
        condition_parts: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.freeze_vision = freeze_vision
        text_config = model.config.text_config
        self.hidden_size = int(text_config.hidden_size)
        layer_types = list(text_config.layer_types)
        attention_layers = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
        if not attention_layers:
            raise ValueError("LFM text config contains no full_attention layer")
        self.tap_layer = attention_layers[-1]
        self.state_projection = nn.Linear(state_dim, self.hidden_size)
        self.condition_parts = condition_parts
        if condition_parts:
            from .token_config import PART_KIND_IDS, PART_NAME_FEATURE_WIDTH

            self.state_kind_embedding = nn.Embedding(len(PART_KIND_IDS) + 1, self.hidden_size, padding_idx=0)
            self.state_name_projection = nn.Linear(PART_NAME_FEATURE_WIDTH, self.hidden_size, bias=False)
            self.state_channel_projection = nn.Linear(state_dim, self.hidden_size, bias=False)
        if freeze_vision:
            self._freeze_vision_parameters()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        state_dim: int,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        freeze_vision: bool = True,
        condition_parts: bool = False,
        **model_kwargs: Any,
    ) -> LFMBackboneAdapter:
        from transformers import AutoModelForImageTextToText
        from transformers.utils.versions import require_version

        require_version(
            "transformers>=5.1,<5.2",
            "embodi pins the lfm2-vl hidden-state and inputs_embeds contract to transformers 5.1.x.",
        )
        model_kwargs.setdefault("dtype", "auto")
        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        if lora_rank:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=(
                    r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
                    r"|feed_forward\.(w1|w2|w3))$"
                ),
                bias="none",
            )
            model = get_peft_model(model, lora_config)
        return cls(
            model,
            state_dim=state_dim,
            freeze_vision=freeze_vision,
            condition_parts=condition_parts,
        )

    def _freeze_vision_parameters(self) -> None:
        found = False
        for name, parameter in self.model.named_parameters():
            if "vision" in name:
                parameter.requires_grad_(False)
                found = True
        if not found:
            raise RuntimeError("could not identify the LFM vision encoder by parameter name")

    def train(self, mode: bool = True) -> LFMBackboneAdapter:
        super().train(mode)
        if self.freeze_vision:
            for name, module in self.model.named_modules():
                if "vision" in name:
                    module.eval()
        return self

    def forward(
        self,
        state: Tensor,
        state_mask: Tensor | None = None,
        part_kind: Tensor | None = None,
        part_name_features: Tensor | None = None,
        channel_mask: Tensor | None = None,
        **model_inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        input_ids = model_inputs.pop("input_ids", None)
        if input_ids is None:
            raise KeyError("processor output must include input_ids")
        attention_mask = model_inputs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeddings = self.model.get_input_embeddings()(input_ids)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError("canonical state must have shape [B, slots, state_width]")
        state_tokens = self.state_projection(state).to(embeddings.dtype)
        if self.condition_parts:
            if part_kind is None or part_name_features is None or channel_mask is None:
                raise ValueError("part-conditioned backbone requires kind, name, and channel descriptors")
            if part_kind.shape != state.shape[:2] or channel_mask.shape != state.shape:
                raise ValueError("part descriptor shapes must match canonical state")
            if part_name_features.shape[:2] != state.shape[:2]:
                raise ValueError("part name features must have shape [B, parts, name_width]")
            state_tokens = state_tokens + self.state_kind_embedding(part_kind)
            state_tokens = state_tokens + self.state_name_projection(part_name_features)
            state_tokens = state_tokens + self.state_channel_projection(channel_mask.to(state.dtype))
            state_tokens = state_tokens.to(embeddings.dtype)
        embeddings = torch.cat((embeddings, state_tokens), dim=1)
        if state_mask is None:
            state_mask = torch.ones(state.shape[:2], device=state.device, dtype=torch.bool)
        if state_mask.shape != state.shape[:2]:
            raise ValueError("state_mask must have shape [B, slots]")
        attention_mask = torch.cat((attention_mask, state_mask.to(attention_mask.dtype)), dim=1)
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
            **model_inputs,
        )
        hidden_states = outputs.hidden_states
        expected_count = len(self.model.config.text_config.layer_types) + 1
        if hidden_states is None or len(hidden_states) != expected_count:
            raise RuntimeError(
                f"unexpected LFM hidden-state contract: expected {expected_count} tensors, "
                f"got {0 if hidden_states is None else len(hidden_states)}"
            )
        features = hidden_states[self.tap_layer + 1]
        if features.shape[-1] != self.hidden_size or features.shape[:2] != attention_mask.shape:
            raise RuntimeError("unexpected tapped LFM feature shape")
        return features, attention_mask.bool()
