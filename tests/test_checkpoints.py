import json
import sys
from types import ModuleType

import pytest
import torch

from embodi.checkpoints import interpolate_checkpoint_value
from embodi.checkpoints import interpolate_checkpoints
from embodi.checkpoints import load_inference_policy


def test_checkpoint_value_interpolation_preserves_nested_schema() -> None:
    integer = torch.tensor([3], dtype=torch.int64)
    first = {
        "weight": torch.tensor([0.0, 2.0]),
        "metadata": ([integer], ("shared",)),
    }
    second = {
        "weight": torch.tensor([2.0, 6.0]),
        "metadata": ([integer.clone()], ("shared",)),
    }

    result = interpolate_checkpoint_value(first, second, 0.25)

    torch.testing.assert_close(result["weight"], torch.tensor([0.5, 3.0]))
    assert isinstance(result["metadata"], tuple)
    assert isinstance(result["metadata"][0], list)
    assert result["metadata"][1] == ("shared",)
    assert result["metadata"][0][0].data_ptr() != integer.data_ptr()


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        (torch.zeros(2), torch.zeros(3), "tensor schemas"),
        (torch.zeros(1), torch.zeros(1, dtype=torch.float64), "tensor schemas"),
        (torch.tensor([1]), torch.tensor([2]), "non-floating"),
        ({"left": 1}, {"right": 1}, "mapping schemas"),
        ([1], [1, 2], "sequence schemas"),
        ("first", "second", "non-tensor checkpoint values"),
    ],
    ids=["shape", "dtype", "integer-value", "mapping", "sequence", "metadata"],
)
def test_checkpoint_value_interpolation_rejects_schema_mismatches(first, second, message) -> None:
    with pytest.raises(ValueError, match=message):
        interpolate_checkpoint_value(first, second, 0.5)


def _write_checkpoint(directory, *, value: float, step: int, stage: str = "core", model: str = "same") -> None:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"format_version": 3, "model_name": model}))
    torch.save({"weight": torch.tensor([value]), "count": torch.tensor([4])}, directory / "core.pt")
    torch.save({"decoder": [torch.tensor([value + 1.0])]}, directory / "embodiment.pt")
    torch.save({"step": step, "stage": stage}, directory / "trainer.pt")


def test_interpolate_checkpoints_writes_compatible_payload_and_provenance(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "output"
    _write_checkpoint(first, value=0.0, step=100)
    _write_checkpoint(second, value=2.0, step=200)

    interpolate_checkpoints(first, second, output, 0.25)

    core = torch.load(output / "core.pt", map_location="cpu", weights_only=True)
    embodiment = torch.load(output / "embodiment.pt", map_location="cpu", weights_only=True)
    trainer = torch.load(output / "trainer.pt", map_location="cpu", weights_only=True)
    torch.testing.assert_close(core["weight"], torch.tensor([0.5]))
    torch.testing.assert_close(core["count"], torch.tensor([4]))
    torch.testing.assert_close(embodiment["decoder"][0], torch.tensor([1.5]))
    assert trainer == {
        "format_version": 3,
        "step": 125,
        "stage": "core",
        "interpolation": {"first": str(first), "second": str(second), "alpha": 0.25},
    }


@pytest.mark.parametrize("alpha", [-0.01, 1.01, float("nan")], ids=["below", "above", "nan"])
def test_interpolate_checkpoints_rejects_invalid_alpha(tmp_path, alpha) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        interpolate_checkpoints(tmp_path / "first", tmp_path / "second", tmp_path / "output", alpha)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [("config", "model configs"), ("stage", "training stages")],
)
def test_interpolate_checkpoints_rejects_incompatible_checkpoints(tmp_path, mismatch, message) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "output"
    _write_checkpoint(first, value=0.0, step=10)
    _write_checkpoint(
        second,
        value=1.0,
        step=20,
        model="different" if mismatch == "config" else "same",
        stage="refine" if mismatch == "stage" else "core",
    )

    with pytest.raises(ValueError, match=message):
        interpolate_checkpoints(first, second, output, 0.5)
    assert not output.exists()


@pytest.mark.parametrize(
    ("version", "module_name", "class_name", "artifacts"),
    [
        (None, "embodi.legacy_policy", "LegacyEmbodiPolicy", ("checkpoint.pt",)),
        (2, "embodi.policy", "EmbodiPolicy", ("core.pt", "embodiment.pt")),
        (3, "embodi.token_policy", "EmbodiPolicy", ("core.pt", "embodiment.pt")),
    ],
    ids=["historical", "v2", "v3"],
)
def test_inference_loader_dispatches_supported_checkpoint_formats(
    tmp_path, monkeypatch, version, module_name, class_name, artifacts
) -> None:
    values = {"model_name": "test"}
    if version is not None:
        values["format_version"] = version
    (tmp_path / "config.json").write_text(json.dumps(values))
    for artifact in artifacts:
        (tmp_path / artifact).touch()

    sentinel = object()

    class StubPolicy:
        @classmethod
        def from_checkpoint(cls, directory, device):
            assert directory == tmp_path
            assert device == "cpu"
            return sentinel

    module = ModuleType(module_name)
    setattr(module, class_name, StubPolicy)
    monkeypatch.setitem(sys.modules, module_name, module)

    assert load_inference_policy(tmp_path, "cpu") is sentinel


@pytest.mark.parametrize(
    ("version", "artifacts", "message"),
    [
        (None, (), "missing checkpoint.pt"),
        (2, ("core.pt",), "format-v2 checkpoint requires"),
        (3, ("embodiment.pt",), "format-v3 checkpoint requires"),
        (99, (), "unsupported explicit checkpoint"),
    ],
    ids=["historical", "v2", "v3", "unknown-version"],
)
def test_inference_loader_rejects_missing_or_unknown_formats(tmp_path, version, artifacts, message) -> None:
    values = {"model_name": "test"}
    if version is not None:
        values["format_version"] = version
    (tmp_path / "config.json").write_text(json.dumps(values))
    for artifact in artifacts:
        (tmp_path / artifact).touch()

    with pytest.raises(ValueError, match=message):
        load_inference_policy(tmp_path, "cpu")
