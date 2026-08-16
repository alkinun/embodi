from __future__ import annotations

from pathlib import Path
from typing import Protocol, Any
import json

import torch
from torch import Tensor


class InferencePolicy(Protocol):
    config: Any

    def reset(self) -> None: ...

    def predict_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor: ...


def interpolate_checkpoint_value(first: Any, second: Any, alpha: float) -> Any:
    if torch.is_tensor(first) and torch.is_tensor(second):
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError("checkpoint tensor schemas do not match")
        if first.is_floating_point() or first.is_complex():
            return torch.lerp(first, second, alpha)
        if not torch.equal(first, second):
            raise ValueError("non-floating checkpoint tensors do not match")
        return first.clone()
    if isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            raise ValueError("checkpoint mapping schemas do not match")
        return {key: interpolate_checkpoint_value(first[key], second[key], alpha) for key in first}
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        if len(first) != len(second):
            raise ValueError("checkpoint sequence schemas do not match")
        values = [interpolate_checkpoint_value(a, b, alpha) for a, b in zip(first, second, strict=True)]
        return type(first)(values)
    if first != second:
        raise ValueError("non-tensor checkpoint values do not match")
    return first


def interpolate_checkpoints(
    first_directory: str | Path,
    second_directory: str | Path,
    output_directory: str | Path,
    alpha: float,
) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    first_directory = Path(first_directory)
    second_directory = Path(second_directory)
    output_directory = Path(output_directory)
    first_config = json.loads((first_directory / "config.json").read_text())
    second_config = json.loads((second_directory / "config.json").read_text())
    if first_config != second_config:
        raise ValueError("checkpoint model configs do not match")
    first_trainer = torch.load(first_directory / "trainer.pt", map_location="cpu", weights_only=True)
    second_trainer = torch.load(second_directory / "trainer.pt", map_location="cpu", weights_only=True)
    if first_trainer.get("stage") != second_trainer.get("stage"):
        raise ValueError("checkpoint training stages do not match")
    interpolated = {}
    for filename in ("core.pt", "embodiment.pt"):
        first = torch.load(first_directory / filename, map_location="cpu", weights_only=True)
        second = torch.load(second_directory / filename, map_location="cpu", weights_only=True)
        interpolated[filename] = interpolate_checkpoint_value(first, second, alpha)
    step = round((1.0 - alpha) * int(first_trainer.get("step", 0)) + alpha * int(second_trainer.get("step", 0)))
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "config.json").write_text(json.dumps(first_config, indent=2) + "\n")
    for filename, value in interpolated.items():
        torch.save(value, output_directory / filename)
    torch.save(
        {
            "format_version": 3,
            "step": step,
            "stage": first_trainer.get("stage"),
            "interpolation": {
                "first": str(first_directory),
                "second": str(second_directory),
                "alpha": alpha,
            },
        },
        output_directory / "trainer.pt",
    )


def load_inference_policy(directory: str | Path, device: str):
    directory = Path(directory)
    values = json.loads((directory / "config.json").read_text())
    version = values.get("format_version")
    if version is None:
        if not (directory / "checkpoint.pt").is_file():
            raise ValueError("historical checkpoint is missing checkpoint.pt")
        from .legacy_policy import LegacyEmbodiPolicy

        return LegacyEmbodiPolicy.from_checkpoint(directory, device)
    if version == 2:
        if not (directory / "core.pt").is_file() or not (directory / "embodiment.pt").is_file():
            raise ValueError("format-v2 checkpoint requires core.pt and embodiment.pt")
        from .policy import EmbodiPolicy as FixedV2Policy

        return FixedV2Policy.from_checkpoint(directory, device)
    if version == 3:
        if not (directory / "core.pt").is_file() or not (directory / "embodiment.pt").is_file():
            raise ValueError("format-v3 checkpoint requires core.pt and embodiment.pt")
        from .token_policy import EmbodiPolicy

        return EmbodiPolicy.from_checkpoint(directory, device)
    raise ValueError(f"unsupported explicit checkpoint format_version: {version}")
