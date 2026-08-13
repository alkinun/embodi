from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json


CANONICAL_ACTION_DIMS = {"arm": 10, "base": 3, "head": 2}
CANONICAL_SLOTS = ("arm.0", "arm.1", "base", "head")
CANONICAL_SLOT_TYPES = ("arm", "arm", "base", "head")
CANONICAL_STATE_WIDTH = 10


@dataclass
class EmbodiConfig:
    format_version: int = 2
    model_name: str = "embodi02-canonical-core"
    backbone_name: str = "LiquidAI/LFM2.5-VL-450M"
    backbone_revision: str = "fc6221ca597f3315e4f82fc2df606783267b34ba"
    camera_names: tuple[str, ...] = ("top", "wrist")
    image_do_rescale: bool = True
    state_dim: int = 6
    action_dim: int = 6
    action_group_slots: tuple[str, ...] = ("arm.0",)
    action_group_types: tuple[str, ...] = ("arm",)
    action_group_state_dims: tuple[int, ...] = (6,)
    action_group_native_dims: tuple[int, ...] = (6,)
    inactive_action_modes: tuple[str, ...] = ("hold",)
    chunk_size: int = 32
    n_action_steps: int = 32
    expert_width: int = 512
    expert_layers: int = 12
    expert_heads: int = 8
    expert_ff_multiplier: int = 4
    dropout: float = 0.0
    denoising_steps: int = 10
    decoder_hidden_width: int = 128
    decoder_layers: int = 2
    decoder_loss_weight: float = 1.0
    decoder_input_noise_std: float = 0.05
    state_adapter_loss_weight: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    freeze_vision_encoder: bool = True

    def __post_init__(self) -> None:
        for name in (
            "camera_names",
            "action_group_slots",
            "action_group_types",
            "action_group_state_dims",
            "action_group_native_dims",
            "inactive_action_modes",
        ):
            setattr(self, name, tuple(getattr(self, name)))
        if self.format_version != 2:
            raise ValueError(f"unsupported model format version: {self.format_version}")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must contain unique camera names")
        group_count = len(self.action_group_types)
        if not group_count:
            raise ValueError("action groups must not be empty")
        for name in (
            "action_group_slots",
            "action_group_state_dims",
            "action_group_native_dims",
            "inactive_action_modes",
        ):
            if len(getattr(self, name)) != group_count:
                raise ValueError(f"{name} must match action_group_types")
        if len(set(self.action_group_slots)) != group_count:
            raise ValueError("action_group_slots must be unique")
        for slot, group_type in zip(self.action_group_slots, self.action_group_types, strict=True):
            if slot not in CANONICAL_SLOTS:
                raise ValueError(f"unknown canonical slot: {slot}")
            expected_type = CANONICAL_SLOT_TYPES[CANONICAL_SLOTS.index(slot)]
            if group_type != expected_type:
                raise ValueError(f"slot {slot} requires group type {expected_type}")
        unknown_modes = sorted(set(self.inactive_action_modes) - {"hold", "zero"})
        if unknown_modes:
            raise ValueError(f"unknown inactive action modes: {unknown_modes}")
        positive = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "expert_width": self.expert_width,
            "expert_layers": self.expert_layers,
            "expert_heads": self.expert_heads,
            "expert_ff_multiplier": self.expert_ff_multiplier,
            "denoising_steps": self.denoising_steps,
            "decoder_hidden_width": self.decoder_hidden_width,
            "decoder_layers": self.decoder_layers,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"dimensions and horizons must be positive: {positive}")
        if any(value <= 0 for value in self.action_group_state_dims + self.action_group_native_dims):
            raise ValueError("action group dimensions must be positive")
        if sum(self.action_group_state_dims) != self.state_dim:
            raise ValueError("action_group_state_dims must sum to state_dim")
        if sum(self.action_group_native_dims) != self.action_dim:
            raise ValueError("action_group_native_dims must sum to action_dim")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.expert_width % self.expert_heads:
            raise ValueError("expert_width must be divisible by expert_heads")
        if self.decoder_layers < 2:
            raise ValueError("decoder_layers must be at least 2")
        if min(self.decoder_loss_weight, self.decoder_input_noise_std, self.state_adapter_loss_weight) < 0:
            raise ValueError("loss weights and decoder noise cannot be negative")
        if self.lora_rank < 0:
            raise ValueError("lora_rank cannot be negative")
        for mode, state_dim, action_dim in zip(
            self.inactive_action_modes,
            self.action_group_state_dims,
            self.action_group_native_dims,
            strict=True,
        ):
            if mode == "hold" and state_dim != action_dim:
                raise ValueError("hold mode requires matching group state and action dimensions")

    @property
    def canonical_action_dim(self) -> int:
        return sum(CANONICAL_ACTION_DIMS[kind] for kind in CANONICAL_SLOT_TYPES)

    @property
    def canonical_slot_count(self) -> int:
        return len(CANONICAL_SLOTS)

    def core_signature(self) -> dict:
        names = (
            "format_version", "backbone_name", "backbone_revision", "chunk_size", "expert_width",
            "expert_layers", "expert_heads", "expert_ff_multiplier", "dropout", "lora_rank",
            "lora_alpha", "lora_dropout", "freeze_vision_encoder",
        )
        return {name: getattr(self, name) for name in names}

    def embodiment_signature(self) -> dict:
        names = (
            "format_version", "state_dim", "action_dim", "action_group_slots", "action_group_types",
            "action_group_state_dims", "action_group_native_dims", "inactive_action_modes",
            "decoder_hidden_width", "decoder_layers",
        )
        return {name: getattr(self, name) for name in names}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> EmbodiConfig:
        values = json.loads(Path(path).read_text())
        if values.get("format_version") != 2:
            raise ValueError(
                "legacy checkpoint uses the native-action architecture and cannot be loaded as a canonical core; "
                "train a format_version=2 core or explicitly export compatible weights"
            )
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown config fields: {unknown}")
        return cls(**values)


def canonical_slot_slices() -> dict[str, slice]:
    result = {}
    offset = 0
    for name, kind in zip(CANONICAL_SLOTS, CANONICAL_SLOT_TYPES, strict=True):
        width = CANONICAL_ACTION_DIMS[kind]
        result[name] = slice(offset, offset + width)
        offset += width
    return result


def canonical_neutral_action() -> tuple[float, ...]:
    values: list[float] = []
    for kind in CANONICAL_SLOT_TYPES:
        if kind == "arm":
            values.extend((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        else:
            values.extend((0.0,) * CANONICAL_ACTION_DIMS[kind])
    return tuple(values)


def canonical_action_normalization() -> tuple[tuple[float, ...], tuple[float, ...]]:
    means: list[float] = []
    scales: list[float] = []
    for kind in CANONICAL_SLOT_TYPES:
        if kind == "arm":
            means.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))
            scales.extend((0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5))
        elif kind == "base":
            means.extend((0.0, 0.0, 0.0))
            scales.extend((0.5, 0.5, 3.141592653589793))
        else:
            means.extend((0.0, 0.0))
            scales.extend((3.141592653589793, 3.141592653589793))
    return tuple(means), tuple(scales)


def canonical_state_normalization() -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    means = []
    scales = []
    for kind in CANONICAL_SLOT_TYPES:
        if kind == "arm":
            means.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))
            scales.append((0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5))
        elif kind == "base":
            means.append((0.0,) * CANONICAL_STATE_WIDTH)
            scales.append((1.0, 1.0, 3.141592653589793) + (1.0,) * 7)
        else:
            means.append((0.0,) * CANONICAL_STATE_WIDTH)
            scales.append((3.141592653589793, 3.141592653589793) + (1.0,) * 8)
    return tuple(means), tuple(scales)
