from dataclasses import asdict
import json

import pytest

from embodi.config import EmbodiConfig as FixedConfig
from embodi.token_config import EmbodiConfig as TokenConfig
from embodi.token_config import PartDescriptor


@pytest.mark.parametrize("config_type", [FixedConfig, TokenConfig], ids=["v2", "v3"])
def test_config_json_round_trip_preserves_typed_sequences(tmp_path, config_type) -> None:
    config = config_type()
    path = tmp_path / "config.json"

    config.save(path)
    loaded = config_type.load(path)

    assert loaded == config
    assert isinstance(loaded.camera_names, tuple)
    if isinstance(loaded, TokenConfig):
        assert isinstance(loaded.parts, tuple)
        assert isinstance(loaded.parts[0], PartDescriptor)


@pytest.mark.parametrize(
    ("config_type", "updates", "message"),
    [
        (FixedConfig, {"format_version": 1}, "legacy checkpoint"),
        (FixedConfig, {"future_option": True}, "unknown config fields"),
        (TokenConfig, {"format_version": 2}, "format_version=3"),
        (TokenConfig, {"future_option": True}, "unknown token config fields"),
    ],
    ids=["v2-version", "v2-unknown-field", "v3-version", "v3-unknown-field"],
)
def test_config_load_rejects_incompatible_json(tmp_path, config_type, updates, message) -> None:
    values = asdict(config_type())
    values.update(updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))

    with pytest.raises(ValueError, match=message):
        config_type.load(path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"camera_names": ("top", "top")}, "unique camera"),
        ({"action_group_slots": ("head",)}, "requires group type head"),
        ({"inactive_action_modes": ("drop",)}, "unknown inactive"),
        ({"chunk_size": 1, "n_action_steps": 2}, "cannot exceed"),
        ({"expert_width": 30, "expert_heads": 8}, "divisible"),
        (
            {
                "state_dim": 5,
                "action_group_state_dims": (5,),
                "action_group_native_dims": (6,),
            },
            "hold mode requires matching",
        ),
    ],
    ids=["cameras", "slot-type", "mode", "horizon", "attention-width", "hold-width"],
)
def test_v2_config_rejects_incompatible_shapes_and_semantics(updates, message) -> None:
    with pytest.raises(ValueError, match=message):
        FixedConfig(**updates)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"parts": (PartDescriptor("other"),)}, "every configured part"),
        ({"inactive_action_modes": ("drop",)}, "hold or zero"),
        ({"chunk_size": 1, "n_action_steps": 2}, "cannot exceed"),
        ({"core_objective": "classification"}, "flow or regression"),
        ({"canonical_channel_loss_weights": (1.0,) * 9}, "10 positive values"),
        ({"native_action_delta_limits": (1.0,) * 5}, "one positive value per action"),
        (
            {
                "state_dim": 5,
                "action_group_state_dims": (5,),
                "action_group_native_dims": (6,),
            },
            "hold mode requires matching",
        ),
    ],
    ids=["part-mapping", "mode", "horizon", "objective", "channel-weights", "delta-limits", "hold-width"],
)
def test_v3_config_rejects_incompatible_shapes_and_semantics(updates, message) -> None:
    with pytest.raises(ValueError, match=message):
        TokenConfig(**updates)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": " "}, "name must not be empty"),
        ({"name": "tool", "kind": "joint"}, "unknown canonical part kind"),
        ({"name": "tool", "kind": "scalar", "channel_mask": (True,) * 10}, "scalar parts"),
        (
            {"name": "tool", "kind": "pose_scalar", "channel_mask": (True,) * 9 + (False,)},
            "must activate the scalar channel",
        ),
    ],
    ids=["blank-name", "kind", "scalar-mask", "pose-scalar-mask"],
)
def test_part_descriptor_rejects_ambiguous_schemas(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PartDescriptor(**kwargs)
