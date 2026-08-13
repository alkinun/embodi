from embodi import EmbodiConfig, PartDescriptor


def test_core_signature_does_not_depend_on_robot_parts() -> None:
    one = EmbodiConfig()
    two = EmbodiConfig(
        parts=(PartDescriptor("left", "pose_scalar"), PartDescriptor("right", "pose")),
        state_dim=13,
        action_dim=13,
        action_group_part_names=("left", "right"),
        action_group_state_dims=(6, 7),
        action_group_native_dims=(6, 7),
        inactive_action_modes=("hold", "hold"),
    )
    assert one.core_signature() == two.core_signature()
    assert one.embodiment_signature() != two.embodiment_signature()


def test_rotation_channels_cannot_be_partially_masked() -> None:
    mask = [True] * 10
    mask[4] = False
    try:
        PartDescriptor("broken", "pose_scalar", tuple(mask))
    except ValueError as error:
        assert "rotation6d" in str(error)
    else:
        raise AssertionError("partial rotation representation was accepted")
