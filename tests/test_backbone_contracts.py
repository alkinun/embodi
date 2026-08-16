import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch
from torch import Tensor, nn

from embodi.backbone import LFMBackboneAdapter
from embodi.token_config import PART_NAME_FEATURE_WIDTH


class FakeLFM(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 4,
        layer_types: tuple[str, ...] = ("conv", "full_attention", "conv"),
        include_vision: bool = True,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size, layer_types=layer_types)
        )
        self.embedding = nn.Embedding(16, hidden_size)
        if include_vision:
            self.vision_tower = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
        self.language_model = nn.Linear(hidden_size, hidden_size)
        self.hidden_states: tuple[Tensor, ...] | None = None
        self.received: dict[str, object] = {}

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, **kwargs: object) -> SimpleNamespace:
        self.received = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            **kwargs,
        }
        hidden_states = self.hidden_states
        if hidden_states is None:
            hidden_states = tuple(
                inputs_embeds + index
                for index in range(len(self.config.text_config.layer_types) + 1)
            )
        return SimpleNamespace(hidden_states=hidden_states)


def _inputs(batch_size: int = 2) -> dict[str, Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3]]).expand(batch_size, -1),
        "attention_mask": torch.tensor([[1, 1, 0]]).expand(batch_size, -1),
    }


def _install_loading_fakes(
    monkeypatch: pytest.MonkeyPatch,
    auto_model: type,
    require_version: object,
    lora_config: object,
    get_peft_model: object,
) -> None:
    transformers_module = ModuleType("transformers")
    transformers_utils = ModuleType("transformers.utils")
    transformers_versions = ModuleType("transformers.utils.versions")
    transformers_module.AutoModelForImageTextToText = auto_model
    transformers_versions.require_version = require_version
    transformers_module.utils = transformers_utils
    transformers_utils.versions = transformers_versions
    peft_module = ModuleType("peft")
    peft_module.LoraConfig = lora_config
    peft_module.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.versions", transformers_versions)
    monkeypatch.setitem(sys.modules, "peft", peft_module)


def test_part_conditioned_forwarding_combines_all_descriptors() -> None:
    model = FakeLFM(hidden_size=3, layer_types=("conv", "full_attention"))
    adapter = LFMBackboneAdapter(model, state_dim=2, condition_parts=True)
    with torch.no_grad():
        model.embedding.weight.zero_()
        adapter.state_projection.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        adapter.state_projection.bias.zero_()
        adapter.state_kind_embedding.weight.zero_()
        adapter.state_kind_embedding.weight[1] = torch.tensor([10.0, 20.0, 30.0])
        adapter.state_name_projection.weight.zero_()
        adapter.state_name_projection.weight[0, 0] = 1.0
        adapter.state_name_projection.weight[1, 1] = 1.0
        adapter.state_name_projection.weight[2, :2] = 1.0
        adapter.state_channel_projection.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )

    name_features = torch.zeros(1, 1, PART_NAME_FEATURE_WIDTH)
    name_features[0, 0, :2] = torch.tensor([1.0, 2.0])
    pixel_values = torch.randn(1, 3, 2, 2)
    features, mask = adapter(
        torch.tensor([[2.0, 3.0]]),
        part_kind=torch.tensor([[1]]),
        part_name_features=name_features,
        channel_mask=torch.tensor([[[True, False]]]),
        input_ids=torch.tensor([[1, 2]]),
        pixel_values=pixel_values,
    )

    received_embeddings = model.received["inputs_embeds"]
    assert isinstance(received_embeddings, Tensor)
    torch.testing.assert_close(received_embeddings[0, -1], torch.tensor([14.0, 25.0, 39.0]))
    torch.testing.assert_close(features, received_embeddings + 2)
    assert mask.tolist() == [[True, True, True]]
    assert model.received["attention_mask"].tolist() == [[1, 1, 1]]
    assert model.received["pixel_values"] is pixel_values
    assert model.received["output_hidden_states"] is True
    assert model.received["use_cache"] is False
    assert model.received["logits_to_keep"] == 1
    assert model.received["return_dict"] is True


def test_forward_accepts_single_state_token_and_builds_default_masks() -> None:
    model = FakeLFM()
    adapter = LFMBackboneAdapter(model, state_dim=2)

    features, mask = adapter(torch.ones(2, 2), input_ids=_inputs()["input_ids"])

    assert features.shape == (2, 4, 4)
    assert mask.dtype == torch.bool
    assert mask.all()
    assert model.received["attention_mask"].dtype == torch.long


def test_initialization_rejects_config_without_full_attention() -> None:
    with pytest.raises(ValueError, match="no full_attention layer"):
        LFMBackboneAdapter(FakeLFM(layer_types=("conv", "conv")), state_dim=2)


def test_forward_requires_input_ids() -> None:
    adapter = LFMBackboneAdapter(FakeLFM(), state_dim=2)

    with pytest.raises(KeyError, match="processor output must include input_ids"):
        adapter(torch.ones(2, 2))


@pytest.mark.parametrize("shape", [(2,), (2, 1, 1, 2)])
def test_forward_rejects_noncanonical_state_rank(shape: tuple[int, ...]) -> None:
    adapter = LFMBackboneAdapter(FakeLFM(), state_dim=2)

    with pytest.raises(ValueError, match=r"canonical state must have shape \[B, slots, state_width\]"):
        adapter(torch.ones(shape), **_inputs())


def test_forward_rejects_mismatched_state_mask() -> None:
    adapter = LFMBackboneAdapter(FakeLFM(), state_dim=2)

    with pytest.raises(ValueError, match=r"state_mask must have shape \[B, slots\]"):
        adapter(torch.ones(2, 2, 2), state_mask=torch.ones(2, 1), **_inputs())


@pytest.mark.parametrize("missing", ["part_kind", "part_name_features", "channel_mask"])
def test_part_conditioning_requires_every_descriptor(missing: str) -> None:
    adapter = LFMBackboneAdapter(FakeLFM(), state_dim=2, condition_parts=True)
    descriptors: dict[str, Tensor | None] = {
        "part_kind": torch.ones(2, 2, dtype=torch.long),
        "part_name_features": torch.ones(2, 2, PART_NAME_FEATURE_WIDTH),
        "channel_mask": torch.ones(2, 2, 2, dtype=torch.bool),
    }
    descriptors[missing] = None

    with pytest.raises(ValueError, match="requires kind, name, and channel descriptors"):
        adapter(torch.ones(2, 2, 2), **descriptors, **_inputs())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"part_kind": torch.ones(2, 1, dtype=torch.long)}, "descriptor shapes"),
        ({"channel_mask": torch.ones(2, 1, 2)}, "descriptor shapes"),
        (
            {"part_name_features": torch.ones(2, 1, PART_NAME_FEATURE_WIDTH)},
            "part name features must have shape",
        ),
    ],
    ids=["kind", "channel", "name"],
)
def test_part_conditioning_rejects_descriptor_shape_mismatch(
    updates: dict[str, Tensor], message: str
) -> None:
    adapter = LFMBackboneAdapter(FakeLFM(), state_dim=2, condition_parts=True)
    descriptors = {
        "part_kind": torch.ones(2, 2, dtype=torch.long),
        "part_name_features": torch.ones(2, 2, PART_NAME_FEATURE_WIDTH),
        "channel_mask": torch.ones(2, 2, 2),
    }
    descriptors.update(updates)

    with pytest.raises(ValueError, match=message):
        adapter(torch.ones(2, 2, 2), **descriptors, **_inputs())


@pytest.mark.parametrize("hidden_states", [None, (torch.zeros(2, 5, 4),) * 3])
def test_forward_rejects_missing_or_incomplete_hidden_states(
    hidden_states: tuple[Tensor, ...] | None,
) -> None:
    model = FakeLFM()
    adapter = LFMBackboneAdapter(model, state_dim=2)
    model.forward = lambda **kwargs: SimpleNamespace(hidden_states=hidden_states)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="expected 4 tensors"):
        adapter(torch.ones(2, 2, 2), **_inputs())


@pytest.mark.parametrize("bad_shape", [(2, 4, 4), (2, 5, 3), (1, 5, 4)])
def test_forward_rejects_incompatible_tapped_feature_shape(
    bad_shape: tuple[int, int, int],
) -> None:
    model = FakeLFM()
    adapter = LFMBackboneAdapter(model, state_dim=2)
    hidden_states = [torch.zeros(2, 5, 4) for _ in range(4)]
    hidden_states[2] = torch.zeros(bad_shape)
    model.hidden_states = tuple(hidden_states)

    with pytest.raises(RuntimeError, match="unexpected tapped LFM feature shape"):
        adapter(torch.ones(2, 2, 2), **_inputs())


def test_frozen_vision_parameters_and_train_mode_remain_frozen() -> None:
    model = FakeLFM()
    adapter = LFMBackboneAdapter(model, state_dim=2)

    assert not any(parameter.requires_grad for parameter in model.vision_tower.parameters())
    assert all(parameter.requires_grad for parameter in model.language_model.parameters())
    assert adapter.train() is adapter
    assert adapter.training
    assert model.language_model.training
    assert not model.vision_tower.training
    adapter.eval()
    assert not adapter.training
    assert not model.language_model.training
    assert not model.vision_tower.training


def test_unfrozen_vision_follows_adapter_train_mode() -> None:
    model = FakeLFM()
    adapter = LFMBackboneAdapter(model, state_dim=2, freeze_vision=False)

    assert all(parameter.requires_grad for parameter in model.vision_tower.parameters())
    adapter.eval()
    assert not model.vision_tower.training
    adapter.train()
    assert model.vision_tower.training


def test_freezing_requires_an_identifiable_vision_parameter() -> None:
    with pytest.raises(RuntimeError, match="could not identify the LFM vision encoder"):
        LFMBackboneAdapter(FakeLFM(include_vision=False), state_dim=2)


def test_from_pretrained_wires_transformers_and_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeLFM()
    calls: dict[str, object] = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> FakeLFM:
            calls["load"] = (model_name, kwargs)
            return model

    class FakeLoraConfig:
        def __init__(self, **kwargs: object) -> None:
            calls["lora_config"] = kwargs

    def fake_get_peft_model(base: nn.Module, config: FakeLoraConfig) -> nn.Module:
        calls["get_peft_model"] = (base, config)
        return base

    _install_loading_fakes(
        monkeypatch,
        FakeAutoModel,
        lambda requirement, message: calls.setdefault("version", (requirement, message)),
        FakeLoraConfig,
        fake_get_peft_model,
    )

    adapter = LFMBackboneAdapter.from_pretrained(
        "local/fake-lfm",
        state_dim=2,
        lora_rank=8,
        lora_alpha=12,
        lora_dropout=0.25,
        freeze_vision=False,
        condition_parts=True,
        revision="test-revision",
    )

    assert calls["load"] == (
        "local/fake-lfm",
        {"revision": "test-revision", "dtype": "auto"},
    )
    requirement, message = calls["version"]
    assert requirement == "transformers>=5.1,<5.2"
    assert "hidden-state and inputs_embeds contract" in message
    assert calls["lora_config"] == {
        "r": 8,
        "lora_alpha": 12,
        "lora_dropout": 0.25,
        "target_modules": (
            r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
            r"|feed_forward\.(w1|w2|w3))$"
        ),
        "bias": "none",
    }
    assert calls["get_peft_model"][0] is model
    assert adapter.model is model
    assert adapter.condition_parts


def test_from_pretrained_zero_rank_skips_lora_and_preserves_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeLFM()
    calls: dict[str, object] = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> FakeLFM:
            calls["load"] = (model_name, kwargs)
            return model

    def unexpected_lora(*args: object, **kwargs: object) -> None:
        raise AssertionError("LoRA must not be configured for rank zero")

    _install_loading_fakes(
        monkeypatch,
        FakeAutoModel,
        lambda *args: None,
        unexpected_lora,
        unexpected_lora,
    )

    adapter = LFMBackboneAdapter.from_pretrained(
        "local/fake-lfm",
        state_dim=2,
        lora_rank=0,
        freeze_vision=False,
        dtype=torch.float32,
    )

    assert calls["load"] == ("local/fake-lfm", {"dtype": torch.float32})
    assert adapter.model is model
