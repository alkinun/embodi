from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodi import EmbodiConfig
from embodi import train as train_module
from embodi.train import (
    evaluate,
    gradient_norm,
    load_json_object,
    train,
    write_or_validate_data_manifest,
    xperience_manifest_samples,
)


def test_load_json_object_rejects_non_object_documents(tmp_path: Path) -> None:
    path = tmp_path / "values.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_json_object(path)


def test_xperience_manifest_samples_coerces_anchor_fields_and_rejects_empty() -> None:
    samples = xperience_manifest_samples(
        [
            {
                "episode_path": "episode-0",
                "frame_index": "4",
                "timestamp_ns": "500",
                "segment_id": "2",
                "prompt": 7,
                "motion_bin": 8,
            }
        ],
        {"episode-0": 3},
    )

    episode_index, anchor = samples[0]
    assert episode_index == 3
    assert anchor.frame_index == 4
    assert anchor.timestamp_ns == 500
    assert anchor.segment_id == 2
    assert anchor.prompt == "7"
    assert anchor.motion_bin == "8"

    with pytest.raises(ValueError, match="must not be empty"):
        xperience_manifest_samples([], {"episode-0": 3})


def test_gradient_norm_returns_zero_when_no_parameter_has_a_gradient() -> None:
    parameters = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1))]

    assert gradient_norm(parameters) == 0.0


def test_configure_deterministic_training_enables_required_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    with monkeypatch.context() as context:
        context.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        context.setattr(
            train_module.torch,
            "use_deterministic_algorithms",
            lambda enabled: calls.append(enabled),
        )
        context.setattr(train_module.torch.backends.cudnn, "benchmark", True)
        context.setattr(train_module.torch.backends.cudnn, "deterministic", False)

        train_module.configure_deterministic_training(True)

        assert calls == [True]
        assert train_module.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert train_module.torch.backends.cudnn.benchmark is False
        assert train_module.torch.backends.cudnn.deterministic is True


def test_legacy_resume_without_lineage_emits_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    write_or_validate_data_manifest(
        output,
        {"format_version": 1, "repo_id": "embodi/data"},
        resume_checkpoint=checkpoint,
    )

    assert "legacy resume checkpoint has no dataset lineage manifest" in capsys.readouterr().out
    assert (output / "data_manifest.json").is_file()


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, images, instructions, state, actions=None, canonical_actions=None, **values):
        self.calls += 1
        return {
            "observation.state": state,
            "action": actions,
            "canonical_action": canonical_actions,
            **values,
        }


class _FakeEvaluationPolicy:
    def __init__(self, config: EmbodiConfig, stage: str) -> None:
        self.config = config
        self.stage = stage
        self.training = True
        self.calls: list[dict] = []

    def eval(self) -> "_FakeEvaluationPolicy":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "_FakeEvaluationPolicy":
        self.training = mode
        return self

    def _canonical_action_targets(self, batch: dict[str, object]) -> torch.Tensor:
        value = batch["canonical_action"]
        assert isinstance(value, torch.Tensor)
        return value

    def __call__(self, batch: dict[str, object], *, noise: torch.Tensor, time: torch.Tensor):
        assert self.training is False
        self.calls.append({"batch": batch, "noise": noise, "time": time})
        return torch.zeros(()), {"loss": float(len(self.calls))}


def _evaluation_batch(config: EmbodiConfig, value: float, *, with_images: bool) -> dict[str, object]:
    batch: dict[str, object] = {
        "observation.state": torch.zeros(1, config.state_dim),
        "canonical_action": torch.full(
            (1, config.chunk_size, config.part_count, 10),
            value,
        ),
        "canonical_state": torch.zeros(1, config.part_count, 10),
    }
    if with_images:
        batch["observation.images.top"] = torch.zeros(1, 3, 2, 2)
        batch["task"] = ["test task"]
    return batch


@pytest.mark.parametrize(
    ("stage", "with_images"),
    [("core", True), ("decoder", False)],
)
def test_evaluate_averages_metrics_restores_mode_and_honors_batch_limit(
    stage: str,
    with_images: bool,
) -> None:
    config = EmbodiConfig(camera_names=("top",), chunk_size=4, n_action_steps=2)
    policy = _FakeEvaluationPolicy(config, stage)
    processor = _FakeProcessor()
    loader = [
        _evaluation_batch(config, 0.0, with_images=with_images),
        _evaluation_batch(config, 1.0, with_images=with_images),
        _evaluation_batch(config, 2.0, with_images=with_images),
    ]

    metrics = evaluate(policy, loader, processor, torch.device("cpu"), max_batches=2, seed=17)

    assert metrics == {"loss": 1.5}
    assert len(policy.calls) == 2
    assert policy.training is True
    assert processor.calls == (2 if with_images else 0)


def test_evaluate_rejects_empty_validation_loader_and_restores_mode() -> None:
    config = EmbodiConfig(camera_names=("top",), chunk_size=4, n_action_steps=2)
    policy = _FakeEvaluationPolicy(config, "decoder")

    with pytest.raises(RuntimeError, match="validation loader produced no batches"):
        evaluate(policy, [], _FakeProcessor(), torch.device("cpu"), max_batches=2, seed=17)

    assert policy.training is True


def _train_args(tmp_path: Path, **updates: object) -> SimpleNamespace:
    values = {
        "steps": 2,
        "batch_size": 1,
        "accumulation_steps": 1,
        "eval_every": 1,
        "validation_batches": 1,
        "checkpoint_step": [],
        "config": None,
        "cameras": None,
        "dataset_format": "lerobot",
        "validation_dataset": None,
        "validation_dataset_root": None,
        "resume": False,
        "init_from": None,
        "init_core": None,
        "init_embodiment": None,
        "deterministic": False,
        "seed": 1,
        "model_seed": None,
        "loader_seed": None,
        "validation_seed": 1,
        "output_dir": tmp_path / "output",
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("batch_size", 0),
        ("accumulation_steps", 0),
        ("eval_every", 0),
        ("validation_batches", 0),
    ],
)
def test_train_rejects_nonpositive_preflight_arguments(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    args = _train_args(tmp_path, **{field: value})

    with pytest.raises(ValueError, match=f"--{field.replace('_', '-')}"):
        train(args)

    assert not args.output_dir.exists()


@pytest.mark.parametrize("checkpoint_step", [[0], [3]])
def test_train_rejects_checkpoint_steps_outside_training_range(
    tmp_path: Path,
    checkpoint_step: list[int],
) -> None:
    args = _train_args(tmp_path, checkpoint_step=checkpoint_step)

    with pytest.raises(ValueError, match="checkpoint-step"):
        train(args)


def test_train_rejects_mismatched_validation_dataset_arguments(tmp_path: Path) -> None:
    args = _train_args(tmp_path, validation_dataset=Path("validation"))

    with pytest.raises(ValueError, match="provided together"):
        train(args)


def test_train_rejects_xperience_without_processor_rescaling(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    EmbodiConfig(image_do_rescale=False).save(config_path)
    args = _train_args(tmp_path, config=config_path, dataset_format="xperience")

    with pytest.raises(ValueError, match="requires image_do_rescale=true"):
        train(args)


def test_train_rejects_explicit_validation_for_xperience(tmp_path: Path) -> None:
    args = _train_args(
        tmp_path,
        dataset_format="xperience",
        validation_dataset=Path("validation"),
        validation_dataset_root=tmp_path / "validation",
    )

    with pytest.raises(ValueError, match="cannot be used with Xperience"):
        train(args)


def test_train_rejects_resume_with_initialization_checkpoint(tmp_path: Path) -> None:
    args = _train_args(tmp_path, resume=True, init_core=Path("core"))

    with pytest.raises(ValueError, match="cannot be combined"):
        train(args)
