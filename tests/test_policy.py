import torch
from torch import nn

from embodi import EmbodiConfig, EmbodiPolicy, PartDescriptor


class FakeBackbone(nn.Module):
    hidden_size = 24

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(10, 24)
        self.calls = 0

    def forward(self, state, state_mask=None, **model_inputs):
        self.calls += 1
        context = self.projection(state)
        return context, state_mask


def make_policy(
    n_action_steps: int = 4,
    camera_names: tuple[str, ...] = ("top", "wrist"),
    **config_values,
) -> EmbodiPolicy:
    config = EmbodiConfig(
        camera_names=camera_names,
        chunk_size=8,
        n_action_steps=n_action_steps,
        expert_width=32,
        expert_layers=2,
        expert_heads=4,
        denoising_steps=2,
        **config_values,
    )
    return EmbodiPolicy(config, backbone=FakeBackbone())


def training_batch(policy: EmbodiPolicy, batch_size: int = 2) -> dict:
    return {
        "observation.state": torch.randn(batch_size, policy.config.state_dim),
        "action": torch.randn(batch_size, 8, policy.config.action_dim),
        "canonical_action": torch.randn(batch_size, 8, policy.config.part_count, 10),
        "canonical_part_mask": torch.ones(batch_size, policy.config.part_count, dtype=torch.bool),
    }


def test_policy_training_contract() -> None:
    policy = make_policy()
    loss, metrics = policy(training_batch(policy))
    assert loss.ndim == 0
    assert set(metrics) == {"flow_loss", "decoder_loss"}
    assert policy.backbone.calls == 1


def test_policy_rejects_native_actions_as_canonical_supervision() -> None:
    policy = make_policy()
    try:
        policy({"observation.state": torch.randn(2, 6), "action": torch.randn(2, 8, 6)})
    except KeyError as error:
        assert "canonical_action" in str(error)
    else:
        raise AssertionError("native actions were accepted as canonical targets")


def test_core_and_decoder_stages_run_only_their_required_paths() -> None:
    policy = make_policy()
    batch = training_batch(policy)
    policy.configure_stage("core")
    core_loss, core_metrics = policy(batch)
    assert set(core_metrics) == {"flow_loss"}
    core_loss.backward()
    assert all(parameter.grad is None for parameter in policy.action_decoder.parameters())

    policy.zero_grad(set_to_none=True)
    calls = policy.backbone.calls
    policy.configure_stage("decoder")
    decoder_loss, decoder_metrics = policy(batch)
    assert set(decoder_metrics) == {"decoder_loss"}
    decoder_loss.backward()
    assert policy.backbone.calls == calls
    assert all(parameter.grad is None for parameter in policy.core.parameters())


def test_predicted_decoder_stage_uses_frozen_core_predictions() -> None:
    policy = make_policy()
    batch = training_batch(policy)
    policy.configure_stage("predicted_decoder")
    loss, metrics = policy(batch)
    assert set(metrics) == {"decoder_loss"}
    loss.backward()
    assert policy.backbone.calls == 1
    assert all(parameter.grad is None for parameter in policy.core.parameters())
    assert all(parameter.grad is None for parameter in policy.state_adapter.parameters())
    assert any(parameter.grad is not None for parameter in policy.action_decoder.parameters())


def test_backbone_runs_once_for_all_denoising_steps() -> None:
    policy = make_policy()
    chunk = policy.predict_action_chunk({"observation.state": torch.randn(2, 6)})
    assert chunk.shape == (2, 8, 6)
    assert policy.backbone.calls == 1


def test_policy_returns_part_token_canonical_chunk() -> None:
    policy = make_policy()
    chunk = policy.predict_canonical_action_chunk({"observation.state": torch.randn(2, 6)})
    assert chunk.shape == (2, 8, 1, 10)


def test_action_queue_reuses_one_chunk_then_replans() -> None:
    policy = make_policy(n_action_steps=3)
    batch = {"observation.state": torch.randn(1, 6)}
    for _ in range(3):
        assert policy.select_action(batch).shape == (1, 6)
    assert policy.backbone.calls == 1
    policy.select_action(batch)
    assert policy.backbone.calls == 2


def test_policy_normalizes_and_unnormalizes_actions() -> None:
    policy = make_policy()
    policy.set_normalization_stats(
        {
            "observation.state": {"mean": torch.ones(6), "std": torch.full((6,), 2.0)},
            "action": {"mean": torch.full((6,), 10.0), "std": torch.full((6,), 3.0)},
        }
    )
    noise = torch.zeros(1, 8, 1, 10)
    actions = policy.predict_action_chunk({"observation.state": torch.ones(1, 6)}, noise=noise)
    torch.testing.assert_close(actions, torch.full_like(actions, 10.0))


def test_residual_decoder_zero_output_holds_current_state() -> None:
    policy = make_policy(decoder_residual=True)
    policy.set_normalization_stats(
        {
            "observation.state": {"mean": torch.ones(6), "std": torch.full((6,), 2.0)},
            "action": {"mean": torch.full((6,), 10.0), "std": torch.full((6,), 3.0)},
        }
    )
    state = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    noise = torch.zeros(1, 8, 1, 10)
    actions = policy.predict_action_chunk({"observation.state": state}, noise=noise)
    torch.testing.assert_close(actions, state.unsqueeze(1).expand_as(actions))


def test_first_step_weight_changes_masked_decoder_loss() -> None:
    predicted = torch.zeros(1, 2, 1)
    target = torch.tensor([[[1.0], [0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    unweighted = EmbodiPolicy._masked_native_mse(predicted, target, None, mask, "mean")
    weighted = EmbodiPolicy._masked_native_mse(
        predicted, target, None, mask, "mean", first_step_weight=10.0
    )
    assert weighted > unweighted


def test_regression_channel_weights_change_loss() -> None:
    policy = make_policy(core_objective="regression")
    batch = training_batch(policy)
    policy.config.canonical_channel_loss_weights = (10.0,) + (1.0,) * 9
    weighted, _ = policy(batch, stage="core")
    policy.config.canonical_channel_loss_weights = (1.0,) * 10
    unweighted, _ = policy(batch, stage="core")
    assert weighted != unweighted


def test_native_action_delta_limits_bound_residual_decoder() -> None:
    policy = make_policy(
        decoder_residual=True,
        native_action_delta_limits=(0.5,) * 6,
    )
    state = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy.action_decoder.decoders[0].network[-1].bias.fill_(100.0)
    actions = policy.predict_action_chunk(
        {"observation.state": state}, noise=torch.zeros(1, 8, 1, 10)
    )
    torch.testing.assert_close(actions, state.unsqueeze(1).expand_as(actions) + 0.5)


def test_masked_group_holds_current_native_state() -> None:
    policy = make_policy()
    state = torch.randn(1, 6)
    actions = policy.predict_action_chunk(
        {"observation.state": state, "action_group_mask": torch.tensor([[False]])}
    )
    torch.testing.assert_close(actions, state.unsqueeze(1).expand_as(actions))


def test_checkpoint_restores_split_training_state(tmp_path) -> None:
    policy = make_policy()
    policy.configure_stage("refine")
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - 0.1 * step)
    loss, _ = policy(training_batch(policy))
    loss.backward()
    optimizer.step()
    scheduler.step()
    policy.save_checkpoint(tmp_path, optimizer, scheduler, step=7)
    assert (tmp_path / "core.pt").is_file()
    assert (tmp_path / "embodiment.pt").is_file()

    restored = make_policy()
    restored.configure_stage("refine")
    restored_optimizer = torch.optim.AdamW(restored.get_optim_params(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0 - 0.1 * step)
    step = restored.load_checkpoint(tmp_path, restored_optimizer, restored_scheduler)
    assert step == 7
    torch.testing.assert_close(restored.expert.action_projection.weight, policy.expert.action_projection.weight)
    torch.testing.assert_close(
        restored.action_decoder.decoders[0].network[-1].weight,
        policy.action_decoder.decoders[0].network[-1].weight,
    )


def test_core_load_ignores_native_embodiment_dimensions(tmp_path) -> None:
    source = make_policy()
    source.save_checkpoint(tmp_path)
    target = make_policy(
        parts=(PartDescriptor("left", "pose_scalar"), PartDescriptor("right", "pose")),
        state_dim=13,
        action_dim=13,
        action_group_part_names=("left", "right"),
        action_group_state_dims=(6, 7),
        action_group_native_dims=(6, 7),
        inactive_action_modes=("hold", "hold"),
    )
    target.load_core_checkpoint(tmp_path)


def test_core_load_can_select_expert_component(tmp_path) -> None:
    source = make_policy()
    source.save_checkpoint(tmp_path)
    target = make_policy()
    original_backbone = {
        name: value.clone() for name, value in target.backbone.state_dict().items()
    }
    target.load_core_checkpoint(tmp_path, components=("expert",))
    torch.testing.assert_close(
        target.expert.action_projection.weight, source.expert.action_projection.weight
    )
    for name, value in target.backbone.state_dict().items():
        torch.testing.assert_close(value, original_backbone[name])
    torch.testing.assert_close(target.expert.action_projection.weight, source.expert.action_projection.weight)


def test_checkpoint_allows_camera_only_warm_start(tmp_path) -> None:
    source = make_policy(camera_names=("top", "wrist"))
    source.save_checkpoint(tmp_path, step=10)
    top_only = make_policy(camera_names=("top",))
    assert top_only.load_checkpoint(tmp_path, allow_camera_mismatch=True) == 10
