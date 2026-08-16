from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import blake2b
from pathlib import Path
import json


CANONICAL_BLOCK_WIDTH = 10
PART_NAME_FEATURE_WIDTH = 32
PART_KIND_IDS = {"pose": 1, "pose_scalar": 2, "scalar": 3}
PART_KIND_VERSION = 1
CANONICAL_MEAN = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5)
CANONICAL_ACTION_SCALE = (0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5)
CANONICAL_STATE_SCALE = (0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5)


def default_channel_mask(kind: str) -> tuple[bool, ...]:
    if kind == "pose":
        return (True,) * 9 + (False,)
    if kind == "pose_scalar":
        return (True,) * 10
    if kind == "scalar":
        return (False,) * 9 + (True,)
    raise ValueError(f"unknown canonical part kind: {kind}")


@dataclass(frozen=True)
class PartDescriptor:
    name: str
    kind: str = "pose_scalar"
    channel_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("part name must not be empty")
        if self.kind not in PART_KIND_IDS:
            raise ValueError(f"unknown canonical part kind: {self.kind}")
        mask = default_channel_mask(self.kind) if self.channel_mask is None else tuple(self.channel_mask)
        if len(mask) != CANONICAL_BLOCK_WIDTH:
            raise ValueError(f"part channel mask must have {CANONICAL_BLOCK_WIDTH} values")
        mask = tuple(bool(value) for value in mask)
        if not any(mask):
            raise ValueError("part channel mask must activate at least one channel")
        rotation_count = sum(mask[3:9])
        if rotation_count not in (0, 6):
            raise ValueError("rotation6d channels must be enabled or disabled as a complete group")
        if self.kind == "scalar" and mask != default_channel_mask("scalar"):
            raise ValueError("scalar parts may only activate the scalar channel")
        if self.kind == "pose" and mask[9]:
            raise ValueError("pose parts cannot activate the scalar channel")
        if self.kind == "pose_scalar" and not mask[9]:
            raise ValueError("pose_scalar parts must activate the scalar channel")
        object.__setattr__(self, "channel_mask", mask)

    @property
    def kind_id(self) -> int:
        return PART_KIND_IDS[self.kind]


def part_name_features(name: str) -> tuple[float, ...]:
    digest = blake2b(name.encode("utf-8"), digest_size=PART_NAME_FEATURE_WIDTH).digest()
    return tuple((value - 127.5) / 127.5 for value in digest)


@dataclass
class EmbodiConfig:
    format_version: int = 3
    model_name: str = "embodi03-part-token-core"
    backbone_name: str = "LiquidAI/LFM2.5-VL-450M"
    backbone_revision: str = "fc6221ca597f3315e4f82fc2df606783267b34ba"
    camera_names: tuple[str, ...] = ("top", "wrist")
    image_do_rescale: bool = True
    parts: tuple[PartDescriptor, ...] = (PartDescriptor("primary_effector", "pose_scalar"),)
    state_dim: int = 6
    action_dim: int = 6
    action_group_part_names: tuple[str, ...] = ("primary_effector",)
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
    decoder_residual: bool = False
    decoder_first_step_weight: float = 1.0
    flow_first_step_weight: float = 1.0
    canonical_channel_loss_weights: tuple[float, ...] = (1.0,) * CANONICAL_BLOCK_WIDTH
    core_objective: str = "flow"
    native_action_delta_limits: tuple[float, ...] = ()
    state_adapter_loss_weight: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    freeze_vision_encoder: bool = True
    freeze_backbone: bool = False

    def __post_init__(self) -> None:
        self.camera_names = tuple(self.camera_names)
        self.parts = tuple(
            part if isinstance(part, PartDescriptor) else PartDescriptor(**part)
            for part in self.parts
        )
        for name in (
            "action_group_part_names",
            "action_group_state_dims",
            "action_group_native_dims",
            "inactive_action_modes",
            "native_action_delta_limits",
            "canonical_channel_loss_weights",
        ):
            setattr(self, name, tuple(getattr(self, name)))
        if self.format_version != 3:
            raise ValueError(f"unsupported token model format version: {self.format_version}")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must contain unique camera names")
        if not self.parts or len({part.name for part in self.parts}) != len(self.parts):
            raise ValueError("parts must contain unique names")
        group_count = len(self.action_group_part_names)
        if not group_count:
            raise ValueError("action groups must not be empty")
        for name in ("action_group_state_dims", "action_group_native_dims", "inactive_action_modes"):
            if len(getattr(self, name)) != group_count:
                raise ValueError(f"{name} must match action_group_part_names")
        part_names = {part.name for part in self.parts}
        if set(self.action_group_part_names) != part_names or group_count != len(self.parts):
            raise ValueError("every configured part must have exactly one native action group")
        if any(value <= 0 for value in self.action_group_state_dims + self.action_group_native_dims):
            raise ValueError("native group dimensions must be positive")
        if sum(self.action_group_state_dims) != self.state_dim:
            raise ValueError("action_group_state_dims must sum to state_dim")
        if sum(self.action_group_native_dims) != self.action_dim:
            raise ValueError("action_group_native_dims must sum to action_dim")
        if any(mode not in {"hold", "zero"} for mode in self.inactive_action_modes):
            raise ValueError("inactive action modes must be hold or zero")
        for mode, state_width, action_width in zip(
            self.inactive_action_modes,
            self.action_group_state_dims,
            self.action_group_native_dims,
            strict=True,
        ):
            if mode == "hold" and state_width != action_width:
                raise ValueError("hold mode requires matching native state and action widths")
        positive = (
            self.state_dim, self.action_dim, self.chunk_size, self.n_action_steps,
            self.expert_width, self.expert_layers, self.expert_heads, self.expert_ff_multiplier,
            self.denoising_steps, self.decoder_hidden_width, self.decoder_layers,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("dimensions, layers, and horizons must be positive")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.expert_width % self.expert_heads:
            raise ValueError("expert_width must be divisible by expert_heads")
        if self.decoder_layers < 2:
            raise ValueError("decoder_layers must be at least 2")
        if min(self.decoder_loss_weight, self.decoder_input_noise_std, self.state_adapter_loss_weight) < 0:
            raise ValueError("loss weights and decoder noise cannot be negative")
        if self.decoder_first_step_weight <= 0:
            raise ValueError("decoder_first_step_weight must be positive")
        if self.flow_first_step_weight <= 0:
            raise ValueError("flow_first_step_weight must be positive")
        if (
            len(self.canonical_channel_loss_weights) != CANONICAL_BLOCK_WIDTH
            or any(value <= 0 for value in self.canonical_channel_loss_weights)
        ):
            raise ValueError(f"canonical_channel_loss_weights must contain {CANONICAL_BLOCK_WIDTH} positive values")
        if self.core_objective not in {"flow", "regression"}:
            raise ValueError("core_objective must be flow or regression")
        if self.native_action_delta_limits and (
            len(self.native_action_delta_limits) != self.action_dim
            or any(value <= 0 for value in self.native_action_delta_limits)
        ):
            raise ValueError("native_action_delta_limits must be empty or contain one positive value per action")
        if self.lora_rank < 0:
            raise ValueError("lora_rank cannot be negative")

    @property
    def part_count(self) -> int:
        return len(self.parts)

    def core_signature(self) -> dict[str, object]:
        names = (
            "format_version", "backbone_name", "backbone_revision", "chunk_size", "expert_width",
            "expert_layers", "expert_heads", "expert_ff_multiplier", "dropout", "lora_rank",
            "lora_alpha", "lora_dropout", "freeze_vision_encoder",
        )
        signature: dict[str, object] = {name: getattr(self, name) for name in names}
        signature.update(
            canonical_block_width=CANONICAL_BLOCK_WIDTH,
            part_kind_version=PART_KIND_VERSION,
            part_name_feature_width=PART_NAME_FEATURE_WIDTH,
        )
        return signature

    def embodiment_signature(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "parts": [asdict(part) for part in self.parts],
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_group_part_names": self.action_group_part_names,
            "action_group_state_dims": self.action_group_state_dims,
            "action_group_native_dims": self.action_group_native_dims,
            "inactive_action_modes": self.inactive_action_modes,
            "decoder_hidden_width": self.decoder_hidden_width,
            "decoder_layers": self.decoder_layers,
            "decoder_residual": self.decoder_residual,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> EmbodiConfig:
        values = json.loads(Path(path).read_text())
        if values.get("format_version") != 3:
            raise ValueError("token training requires a format_version=3 checkpoint")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown token config fields: {unknown}")
        return cls(**values)
