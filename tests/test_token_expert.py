import torch

from embodi.token_config import PART_NAME_FEATURE_WIDTH
from embodi.token_expert import PartTokenActionExpert


def make_expert() -> PartTokenActionExpert:
    return PartTokenActionExpert(
        context_width=16,
        chunk_size=8,
        width=32,
        layers=2,
        heads=4,
    )


def descriptors(batch_size: int, part_count: int):
    return {
        "part_kind": torch.ones(batch_size, part_count, dtype=torch.long),
        "part_name_features": torch.randn(batch_size, part_count, PART_NAME_FEATURE_WIDTH),
        "channel_mask": torch.ones(batch_size, part_count, 10, dtype=torch.bool),
        "part_mask": torch.ones(batch_size, part_count, dtype=torch.bool),
    }


def test_one_expert_accepts_different_part_counts() -> None:
    expert = make_expert().eval()
    context = torch.randn(2, 3, 16)
    assert expert.sample(context, **descriptors(2, 1), steps=2).shape == (2, 8, 1, 10)
    assert expert.sample(context, **descriptors(2, 5), steps=2).shape == (2, 8, 5, 10)
    assert expert.action_projection.in_features == expert.output_projection.out_features == 10


def test_part_permutation_permutes_outputs() -> None:
    expert = make_expert().eval()
    torch.nn.init.normal_(expert.output_projection.weight)
    actions = torch.randn(1, 8, 3, 10)
    context = torch.randn(1, 4, 16)
    description = descriptors(1, 3)
    time = torch.tensor([0.4])
    first = expert(actions, time, context, **description)
    permutation = torch.tensor([2, 0, 1])
    permuted_description = {
        name: value[:, permutation] for name, value in description.items()
    }
    second = expert(actions[:, :, permutation], time, context, **permuted_description)
    torch.testing.assert_close(second, first[:, :, permutation], atol=1e-5, rtol=1e-5)


def test_inactive_part_noise_cannot_affect_active_part() -> None:
    expert = make_expert().eval()
    torch.nn.init.normal_(expert.output_projection.weight)
    context = torch.randn(1, 3, 16)
    description = descriptors(1, 2)
    description["part_mask"][:, 1] = False
    description["channel_mask"][:, 1] = False
    first = torch.randn(1, 8, 2, 10)
    second = first.clone()
    second[:, :, 1] = torch.randn_like(second[:, :, 1]) * 100
    time = torch.tensor([0.5])
    output_a = expert(first, time, context, **description)
    output_b = expert(second, time, context, **description)
    torch.testing.assert_close(output_a[:, :, 0], output_b[:, :, 0])
    torch.testing.assert_close(output_a[:, :, 1], torch.zeros_like(output_a[:, :, 1]))


def test_inactive_channel_and_padded_time_values_cannot_leak() -> None:
    expert = make_expert().eval()
    torch.nn.init.normal_(expert.output_projection.weight)
    context = torch.randn(1, 3, 16)
    description = descriptors(1, 1)
    description["channel_mask"][..., 9] = False
    padding = torch.zeros(1, 8, dtype=torch.bool)
    padding[:, -2:] = True
    first = torch.randn(1, 8, 1, 10)
    second = first.clone()
    second[..., 9] = torch.randn_like(second[..., 9]) * 100
    second[:, -2:] = torch.randn_like(second[:, -2:]) * 100
    time = torch.tensor([0.3])
    output_a = expert(first, time, context, **description, action_is_pad=padding)
    output_b = expert(second, time, context, **description, action_is_pad=padding)
    torch.testing.assert_close(output_a[:, :-2, :, :9], output_b[:, :-2, :, :9])


def test_sampling_keeps_masked_channels_zero() -> None:
    expert = make_expert().eval()
    context = torch.randn(1, 3, 16)
    description = descriptors(1, 1)
    description["channel_mask"][..., 9] = False
    output = expert.sample(context, **description, steps=3)
    torch.testing.assert_close(output[..., 9], torch.zeros_like(output[..., 9]))


def test_expert_rejects_fully_masked_samples() -> None:
    expert = make_expert()
    context = torch.randn(1, 3, 16)
    description = descriptors(1, 1)
    description["part_mask"][:] = False
    try:
        expert.sample(context, **description)
    except ValueError as error:
        assert "active part" in str(error)
    else:
        raise AssertionError("fully masked sample was accepted")


def test_expert_rejects_all_padded_time() -> None:
    expert = make_expert()
    try:
        expert.sample(
            torch.randn(1, 3, 16),
            **descriptors(1, 1),
            action_is_pad=torch.ones(1, 8, dtype=torch.bool),
        )
    except ValueError as error:
        assert "non-padded" in str(error)
    else:
        raise AssertionError("all-padded trajectory was accepted")


def test_clean_estimate_follows_flow_equation() -> None:
    expert = make_expert()
    actions = torch.randn(2, 8, 2, 10)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.25, 0.75])
    context = torch.randn(2, 3, 16)
    loss, clean = expert.flow_loss(
        actions,
        context,
        **descriptors(2, 2),
        noise=noise,
        time=time,
        return_clean_estimate=True,
    )
    interpolation = time.view(2, 1, 1, 1)
    expected = (1 - interpolation) * noise + interpolation * actions
    assert loss.ndim == 0
    torch.testing.assert_close(clean, expected)


def test_first_step_weight_changes_flow_loss() -> None:
    expert = make_expert()
    actions = torch.zeros(1, 8, 1, 10)
    actions[:, 0] = 1.0
    context = torch.zeros(1, 2, 16)
    description = descriptors(1, 1)
    values = dict(noise=torch.zeros_like(actions), time=torch.ones(1))
    unweighted = expert.flow_loss(actions, context, **description, **values)
    weighted = expert.flow_loss(
        actions, context, **description, **values, first_step_weight=10.0
    )
    assert weighted > unweighted


def test_regression_prediction_and_loss_contract() -> None:
    expert = make_expert()
    actions = torch.randn(2, 8, 1, 10)
    context = torch.randn(2, 3, 16)
    description = descriptors(2, 1)
    loss, prediction = expert.regression_loss(
        actions, context, **description, return_prediction=True
    )
    assert loss.ndim == 0
    assert prediction.shape == actions.shape
    torch.testing.assert_close(expert.predict(context, **description), prediction)
