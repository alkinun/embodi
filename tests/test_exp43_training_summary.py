import hashlib
import math

import pytest
import torch

from embodi.exp43_training_summary import (
    STEPS,
    _finite_tensors,
    expected_validation_keys,
    sequence_sha256,
    validate_metrics_records,
)


def test_sequence_sha256_reconstructs_deterministic_loader_order() -> None:
    indices = list(range(100))
    first = sequence_sha256(indices, seed=17, count=50)
    assert first == "964eb6be120af10b7e5742656ae9c1025680845b75aeb6e8616f38520b3e38e1"
    assert first == sequence_sha256(indices, seed=17, count=50)
    assert first != sequence_sha256(indices, seed=18, count=50)
    assert first != hashlib.sha256("".join(f"{value}\n" for value in indices[:50]).encode()).hexdigest()
    with pytest.raises(ValueError, match="count"):
        sequence_sha256(indices, seed=17, count=101)


def test_validate_metrics_records_requires_fixed_schedule_and_gripper_cells() -> None:
    train = []
    for step in range(20, STEPS + 1, 20):
        progress = (step - 120) / (STEPS - 120)
        learning_rate = (
            1e-4 * (step + 1) / 120
            if step < 120
            else 1e-4 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        )
        train.append(
            {
                "step": step,
                "split": "train",
                "metrics": {
                    "regression_loss": 2.0,
                    "optimization_loss": 1.0,
                    "grad_norm": 0.5,
                    "expert_grad_norm": 0.5,
                    "learning_rate": learning_rate,
                    "gradient_clip_count": 0,
                    "gradient_clip_rate": 0.0,
                },
            }
        )
    validation_metrics = {key: 0.1 for key in expected_validation_keys()}
    final_train, validation = validate_metrics_records(
        [
            *train,
            {"step": STEPS, "split": "validation", "metrics": validation_metrics},
        ]
    )
    assert final_train == train[-1]["metrics"]
    assert validation == validation_metrics

    invalid = [*train, {"step": STEPS, "split": "validation", "metrics": {}}]
    with pytest.raises(ValueError, match="invalid"):
        validate_metrics_records(invalid)

    poisoned = [
        *train,
        {
            "step": STEPS,
            "split": "validation",
            "metrics": {**validation_metrics, "macro/all": 0.2},
        },
    ]
    with pytest.raises(ValueError, match="global macro"):
        validate_metrics_records(poisoned)

    unknown_split = [
        *train,
        {"step": STEPS, "split": "other", "metrics": validation_metrics},
    ]
    with pytest.raises(ValueError, match="schedule"):
        validate_metrics_records(unknown_split)


def test_finite_tensor_validation_rejects_nested_nonfinite_state() -> None:
    assert _finite_tensors({"state": [torch.tensor([1.0]), {"step": torch.tensor(2)}]})
    assert not _finite_tensors({"state": {"exp_avg": torch.tensor([float("nan")])}})
