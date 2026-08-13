import pytest
import torch
import json

from embodi.canonical import rotation_6d_to_matrix
from embodi.xperience import (
    _caption_segments,
    XperienceAnchor,
    camera_hand_to_world,
    convert_right_hand_trajectory,
    interpolate_rotations,
    normalized_hand_opening,
    quaternion_wxyz_to_matrix,
    select_multi_episode_xperience_split,
    select_fixed_episode_xperience_split,
    validate_right_hand_sample,
)


class FakeEpisode:
    def __init__(self, episode_index: int, count: int) -> None:
        self.anchors = [
            XperienceAnchor(
                frame_index=index,
                timestamp_ns=index,
                segment_id=index // 10,
                prompt=f"episode {episode_index}",
                motion_bin=("meaningful", "small", "stationary")[index % 3],
            )
            for index in range(count)
        ]

    def valid_anchors(self):
        return self.anchors, {"rejected": 1}


def test_multi_episode_split_is_episode_disjoint() -> None:
    episodes = [FakeEpisode(index, count) for index, count in enumerate((60, 60, 0, 60))]
    split = select_multi_episode_xperience_split(
        episodes, train_clips=100, validation_clips=30, seed=7
    )
    train_episodes = {index for index, _ in split.train}
    validation_episodes = {index for index, _ in split.validation}
    assert len(split.train) == 100
    assert len(split.validation) == 30
    assert train_episodes.isdisjoint(validation_episodes)
    assert 2 not in train_episodes | validation_episodes
    assert split.rejection_counts == {"rejected": 4}


def test_multi_episode_training_budgets_are_nested_with_fixed_validation() -> None:
    episodes = [FakeEpisode(index, 600) for index in range(4)]
    small = select_multi_episode_xperience_split(
        episodes, train_clips=100, validation_clips=100, seed=11
    )
    large = select_multi_episode_xperience_split(
        episodes, train_clips=1000, validation_clips=100, seed=11
    )
    assert set(small.train).issubset(large.train)
    assert small.validation == large.validation
    assert small.train_episode_indices == large.train_episode_indices
    assert small.validation_episode_indices == large.validation_episode_indices


def test_fixed_episode_split_balances_training_episodes() -> None:
    episodes = [FakeEpisode(index, 600) for index in range(4)]
    split = select_fixed_episode_xperience_split(
        episodes,
        train_episode_indices=(0, 1, 2),
        validation_episode_indices=(3,),
        train_clips=1000,
        validation_clips=100,
        seed=13,
    )
    counts = {index: sum(sample_index == index for sample_index, _ in split.train) for index in range(3)}
    assert counts == {0: 334, 1: 333, 2: 333}
    assert {index for index, _ in split.validation} == {3}


def test_caption_terminal_frame_reference_maps_to_last_timestamp() -> None:
    caption = json.dumps(
        {
            "config": {"Main Task": "test task"},
            "segments": [
                {
                    "start_frame": "100",
                    "end_frame": "frame_0000003",
                    "Sub Task": "test subtask",
                }
            ],
        }
    )
    _, segments = _caption_segments(caption, torch.tensor([100, 200, 300]))
    assert segments[0]["start_ns"] == 100
    assert segments[0]["end_ns"] == 300
    assert segments[0]["last_index"] == 2


def _rotation_z(angle: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(*angle.shape, 3, 3, dtype=angle.dtype)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1
    return result


def test_hand_opening_is_palm_normalized_calibrated_and_clamped() -> None:
    thumb = torch.zeros(3, 3)
    index = torch.tensor([[0.01, 0.0, 0.0], [0.04, 0.0, 0.0], [0.08, 0.0, 0.0]])
    opening = normalized_hand_opening(
        thumb, index, torch.full((3,), 0.1), closed_ratio=0.2, open_ratio=0.6
    )
    torch.testing.assert_close(opening, torch.tensor([0.0, 0.5, 1.0]))


def test_converter_uses_frozen_initial_root_and_initial_hand_frames() -> None:
    horizon = 32
    root_rotation = _rotation_z(torch.tensor(torch.pi / 2))
    transform_world_root = torch.eye(4)
    transform_world_root[:3, :3] = root_rotation
    transform_world_root[:3, 3] = torch.tensor([3.0, -2.0, 0.5])

    times = torch.arange(horizon + 1, dtype=torch.float32)
    position_root = torch.stack((0.01 * times, 0.02 * times, torch.full_like(times, 0.3)), dim=-1)
    palm_position_world = transform_world_root[:3, 3] + (root_rotation @ position_root.T).T
    initial_hand_root = _rotation_z(torch.tensor(0.3))
    relative = _rotation_z(0.01 * times)
    palm_rotation_world = root_rotation @ initial_hand_root @ relative
    opening = times / horizon

    sample = convert_right_hand_trajectory(
        transform_world_root, palm_position_world, palm_rotation_world, opening
    )
    validate_right_hand_sample(sample)

    assert sample.canonical_state.shape == (1, 10)
    assert sample.canonical_action.shape == (32, 1, 10)
    torch.testing.assert_close(sample.canonical_state[0, :3], position_root[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sample.canonical_action[:, 0, :3], position_root[1:] - position_root[0])
    decoded = rotation_6d_to_matrix(sample.canonical_action[:, 0, 3:9])
    torch.testing.assert_close(decoded, relative[1:], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sample.canonical_action[:, 0, 9], opening[1:])


def test_root_world_motion_does_not_leak_into_stationary_root_relative_labels() -> None:
    horizon = 32
    transform_world_root = torch.eye(4)
    transform_world_root[:3, 3] = torch.tensor([10.0, -4.0, 2.0])
    palm_position_world = transform_world_root[:3, 3] + torch.tensor([0.2, 0.1, 0.4])
    palm_position_world = palm_position_world.expand(horizon + 1, -1).clone()
    palm_rotation_world = torch.eye(3).expand(horizon + 1, -1, -1).clone()
    opening = torch.full((horizon + 1,), 0.4)

    sample = convert_right_hand_trajectory(
        transform_world_root, palm_position_world, palm_rotation_world, opening
    )
    torch.testing.assert_close(sample.canonical_action[:, 0, :3], torch.zeros(horizon, 3))
    torch.testing.assert_close(
        rotation_6d_to_matrix(sample.canonical_action[:, 0, 3:9]),
        torch.eye(3).expand(horizon, -1, -1),
    )


def test_converter_rejects_incomplete_horizon_and_invalid_opening() -> None:
    with pytest.raises(ValueError, match="33"):
        convert_right_hand_trajectory(
            torch.eye(4), torch.zeros(32, 3), torch.eye(3).expand(32, -1, -1), torch.zeros(32)
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        convert_right_hand_trajectory(
            torch.eye(4),
            torch.zeros(33, 3),
            torch.eye(3).expand(33, -1, -1),
            torch.full((33,), 1.1),
        )


def test_wxyz_quaternion_and_camera_world_convention() -> None:
    angle = torch.tensor(torch.pi / 2)
    quaternion = torch.tensor([torch.cos(angle / 2), 0.0, 0.0, torch.sin(angle / 2)])
    rotation = quaternion_wxyz_to_matrix(quaternion)
    torch.testing.assert_close(rotation @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=1e-6)

    position, orientation = camera_hand_to_world(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.eye(4),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.eye(3),
    )
    torch.testing.assert_close(position, torch.tensor([0.0, 2.0, 3.0]))
    torch.testing.assert_close(orientation, torch.eye(3))


def test_rotation_interpolation_uses_fixed_target_time_weight() -> None:
    left = torch.eye(3).unsqueeze(0)
    right = _rotation_z(torch.tensor([torch.pi / 2]))
    midpoint = interpolate_rotations(left, right, torch.tensor([0.5]))
    expected = _rotation_z(torch.tensor([torch.pi / 4]))
    torch.testing.assert_close(midpoint, expected, atol=1e-5, rtol=1e-5)
