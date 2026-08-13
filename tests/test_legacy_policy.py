import torch
from torch import nn
import json

from embodi import load_inference_policy
from embodi.policy import EmbodiPolicy as FixedV2Policy
from embodi.token_policy import EmbodiPolicy as TokenV3Policy
from embodi.legacy_policy import LegacyConfig, LegacyEmbodiPolicy


class FakeLegacyBackbone(nn.Module):
    hidden_size = 16

    def __init__(self, state_dim: int = 6) -> None:
        super().__init__()
        self.projection = nn.Linear(state_dim, 16)

    def forward(self, state, **model_inputs):
        context = self.projection(state).unsqueeze(1)
        return context, torch.ones(context.shape[:2], dtype=torch.bool)


def make_config(with_decoder: bool) -> LegacyConfig:
    values = dict(
        chunk_size=8,
        n_action_steps=4,
        expert_width=32,
        expert_layers=2,
        expert_heads=4,
        denoising_steps=2,
    )
    if with_decoder:
        values.update(
            action_group_types=("arm",),
            action_group_state_dims=(6,),
            action_group_native_dims=(6,),
        )
    return LegacyConfig(**values)


def test_legacy_direct_expert_inference() -> None:
    policy = LegacyEmbodiPolicy(make_config(False), backbone=FakeLegacyBackbone())
    chunk = policy.predict_action_chunk({"observation.state": torch.randn(2, 6)})
    assert chunk.shape == (2, 8, 6)


def test_legacy_padded_canonical_decoder_inference() -> None:
    policy = LegacyEmbodiPolicy(make_config(True), backbone=FakeLegacyBackbone())
    assert policy.expert.action_dim == 10
    chunk = policy.predict_action_chunk({"observation.state": torch.randn(2, 6)})
    assert chunk.shape == (2, 8, 6)


def test_legacy_checkpoint_allows_only_new_unused_mask_weight_to_be_missing(tmp_path) -> None:
    source = LegacyEmbodiPolicy(make_config(True), backbone=FakeLegacyBackbone())
    state = source.state_dict()
    state.pop("expert.mask_projection.weight")
    torch.save({"model": state, "step": 12}, tmp_path / "checkpoint.pt")
    restored = LegacyEmbodiPolicy(make_config(True), backbone=FakeLegacyBackbone())
    assert restored.load_checkpoint(tmp_path) == 12
    torch.testing.assert_close(restored.expert.action_projection.weight, source.expert.action_projection.weight)


def test_public_checkpoint_loader_dispatches_legacy_format(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_name": "legacy"}))
    sentinel = object()
    monkeypatch.setattr(
        LegacyEmbodiPolicy,
        "from_checkpoint",
        classmethod(lambda cls, directory, device: sentinel),
    )
    (tmp_path / "checkpoint.pt").touch()
    assert load_inference_policy(tmp_path, "cpu") is sentinel


def test_inference_dispatcher_is_explicit_about_versions(tmp_path, monkeypatch) -> None:
    for filename in ("core.pt", "embodiment.pt"):
        (tmp_path / filename).touch()
    sentinel_v2 = object()
    sentinel_v3 = object()
    monkeypatch.setattr(FixedV2Policy, "from_checkpoint", classmethod(lambda cls, path, device: sentinel_v2))
    monkeypatch.setattr(TokenV3Policy, "from_checkpoint", classmethod(lambda cls, path, device: sentinel_v3))
    (tmp_path / "config.json").write_text(json.dumps({"format_version": 2}))
    assert load_inference_policy(tmp_path, "cpu") is sentinel_v2
    (tmp_path / "config.json").write_text(json.dumps({"format_version": 3}))
    assert load_inference_policy(tmp_path, "cpu") is sentinel_v3
    (tmp_path / "config.json").write_text(json.dumps({"format_version": 99}))
    try:
        load_inference_policy(tmp_path, "cpu")
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unknown explicit checkpoint version was accepted")
