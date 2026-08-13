from __future__ import annotations

from pathlib import Path
from typing import Protocol, Any
import json

from torch import Tensor


class InferencePolicy(Protocol):
    config: Any

    def reset(self) -> None: ...

    def predict_action_chunk(self, batch: dict[str, Any], noise: Tensor | None = None) -> Tensor: ...


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
