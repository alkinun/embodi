import torch

from embodi.expert import FlowMatchingActionExpert


def make_expert() -> FlowMatchingActionExpert:
    return FlowMatchingActionExpert(
        action_dim=6,
        context_width=16,
        chunk_size=8,
        width=32,
        layers=2,
        heads=4,
        mask_width=2,
    )


def test_flow_loss_backpropagates() -> None:
    expert = make_expert()
    actions = torch.randn(3, 8, 6)
    context = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    loss = expert.flow_loss(actions, context, mask)
    loss.backward()
    assert loss.ndim == 0
    assert expert.action_projection.weight.grad is not None


def test_sampling_shape_and_deterministic_noise() -> None:
    expert = make_expert().eval()
    context = torch.randn(2, 5, 16)
    noise = torch.randn(2, 8, 6)
    first = expert.sample(context, steps=3, noise=noise)
    second = expert.sample(context, steps=3, noise=noise)
    assert first.shape == (2, 8, 6)
    torch.testing.assert_close(first, second)


def test_padding_is_excluded_from_per_sample_loss() -> None:
    expert = make_expert()
    actions = torch.randn(2, 8, 6)
    context = torch.randn(2, 3, 16)
    padding = torch.zeros(2, 8, dtype=torch.bool)
    padding[:, -2:] = True
    loss = expert.flow_loss(
        actions,
        context,
        action_is_pad=padding,
        noise=torch.zeros_like(actions),
        time=torch.full((2,), 0.5),
        reduction="none",
    )
    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()


def test_inactive_noise_cannot_change_active_predictions() -> None:
    expert = make_expert().eval()
    torch.nn.init.normal_(expert.output_projection.weight)
    context = torch.randn(1, 3, 16)
    first = torch.randn(1, 8, 6)
    second = first.clone()
    second[..., 3:] = torch.randn_like(second[..., 3:]) * 100
    dim_mask = torch.tensor([[True, True, True, False, False, False]])
    slot_mask = torch.tensor([[True, False]])
    time = torch.tensor([0.5])
    output_a = expert(first, time, context, action_dim_mask=dim_mask, slot_mask=slot_mask)
    output_b = expert(second, time, context, action_dim_mask=dim_mask, slot_mask=slot_mask)
    torch.testing.assert_close(output_a[..., :3], output_b[..., :3])


def test_sampling_keeps_inactive_channels_at_neutral() -> None:
    expert = make_expert().eval()
    context = torch.randn(1, 3, 16)
    dim_mask = torch.tensor([[True, True, True, False, False, False]])
    neutral = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    output = expert.sample(
        context,
        steps=2,
        action_dim_mask=dim_mask,
        inactive_value=neutral,
        slot_mask=torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(output[..., 3:], neutral[3:].view(1, 1, 3).expand(1, 8, 3))
