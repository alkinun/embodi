import sys
import types
from pathlib import Path

import pytest
import torch

from embodi.canonical import matrix_to_rotation_6d, rotation_6d_to_matrix
from embodi.xperience import (
    XPERIENCE_FPS,
    XPERIENCE_HORIZON,
    CachedXperienceDataset,
    CachedXperienceEpisode,
    MultiEpisodeCachedXperienceDataset,
    RightHandCanonicalSample,
    XperienceAnchor,
    classify_motion,
    select_xperience_split,
)


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


def _episode(timestamps_ns: torch.Tensor) -> CachedXperienceEpisode:
    episode = CachedXperienceEpisode.__new__(CachedXperienceEpisode)
    episode.timestamps_ns = timestamps_ns.to(torch.int64)
    count = len(timestamps_ns)
    episode.root_pose = torch.zeros(count, 7)
    episode.root_pose[:, 0] = 1
    episode.hand_position_world = torch.zeros(count, 3)
    episode.hand_rotation_world = torch.eye(3).expand(count, -1, -1).clone()
    episode.opening = torch.full((count,), 0.25)
    episode.body_right_wrist_world = episode.hand_position_world.clone()
    episode.segments = []
    episode.video_path = Path("synthetic.mp4")
    return episode


def _target_timestamps() -> torch.Tensor:
    offsets = torch.arange(XPERIENCE_HORIZON + 1, dtype=torch.float64) / XPERIENCE_FPS
    return torch.round(offsets * 1e9).to(torch.int64)


def _sample_with_action(action: torch.Tensor | None = None) -> RightHandCanonicalSample:
    if action is None:
        action = torch.zeros(XPERIENCE_HORIZON, 1, 10)
        action[:, :, 3:9] = matrix_to_rotation_6d(torch.eye(3))
        action[:, :, 9] = 0.25
    state = torch.zeros(1, 10, dtype=action.dtype)
    state[:, 3:9] = matrix_to_rotation_6d(torch.eye(3, dtype=action.dtype))
    state[:, 9] = 0.25
    return RightHandCanonicalSample(state, action, torch.ones(1, dtype=torch.bool))


def _anchor(frame_index: int, segment_id: int, motion_bin: str = "stationary") -> XperienceAnchor:
    return XperienceAnchor(
        frame_index=frame_index,
        timestamp_ns=1_000_000_000 + frame_index,
        segment_id=segment_id,
        prompt=f"segment {segment_id}",
        motion_bin=motion_bin,
    )


def test_sample_at_exact_target_timestamps() -> None:
    timestamps_ns = _target_timestamps()
    episode = _episode(timestamps_ns)
    seconds = timestamps_ns.float() / 1e9
    episode.hand_position_world[:, 0] = 0.3 * seconds
    episode.opening = 0.2 + 0.4 * seconds
    episode.body_right_wrist_world = episode.hand_position_world.clone()

    sample = episode.sample_at(0)

    torch.testing.assert_close(sample.canonical_action[:, 0, 0], 0.3 * seconds[1:])
    torch.testing.assert_close(sample.canonical_action[:, 0, 9], 0.2 + 0.4 * seconds[1:])
    torch.testing.assert_close(sample.canonical_state[0, :3], torch.zeros(3))


def test_sample_at_interpolates_positions_rotations_and_opening() -> None:
    timestamps_ns = torch.arange(55, dtype=torch.int64) * 20_000_000
    episode = _episode(timestamps_ns)
    seconds = timestamps_ns.double() / 1e9
    episode.hand_position_world[:, 0] = seconds.float()
    episode.hand_rotation_world = _rotation_z(0.2 * seconds).float()
    episode.opening = (0.1 + 0.2 * seconds).float()
    episode.body_right_wrist_world = episode.hand_position_world.clone()

    sample = episode.sample_at(0)
    targets = _target_timestamps().double() / 1e9

    torch.testing.assert_close(sample.canonical_action[:, 0, 0], targets[1:].float(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        sample.canonical_action[:, 0, 9], (0.1 + 0.2 * targets[1:]).float(), atol=1e-6, rtol=1e-6
    )
    expected_rotations = _rotation_z(0.2 * targets[1:]).float()
    torch.testing.assert_close(
        rotation_6d_to_matrix(sample.canonical_action[:, 0, 3:9]),
        expected_rotations,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda episode: None, "incomplete future horizon"),
        (
            lambda episode: episode.hand_position_world.__setitem__((1, 0), float("nan")),
            "non-finite synchronized labels",
        ),
        (
            lambda episode: episode.body_right_wrist_world.__setitem__((0, 0), 0.1001),
            "hand/body wrist disagreement exceeds 10 cm",
        ),
    ],
)
def test_sample_at_reports_rejection_reasons(mutate, reason: str) -> None:
    episode = _episode(_target_timestamps()[:-1])
    if reason != "incomplete future horizon":
        episode = _episode(_target_timestamps())
    mutate(episode)

    with pytest.raises(ValueError, match=reason):
        episode.sample_at(0)


def test_sample_at_enforces_interpolation_gap_threshold() -> None:
    accepted = _episode(torch.arange(16, dtype=torch.int64) * 75_000_000)
    accepted.sample_at(0)

    rejected = _episode(torch.arange(16, dtype=torch.int64) * 75_000_001)
    with pytest.raises(ValueError, match="interpolation bracket exceeds 75 ms"):
        rejected.sample_at(0)


def test_sample_at_accepts_wrist_and_rotation_thresholds_but_rejects_above_rotation() -> None:
    timestamps_ns = torch.arange(19, dtype=torch.int64) * 60_000_000
    accepted = _episode(timestamps_ns)
    accepted.body_right_wrist_world[0, 0] = 0.1
    accepted.hand_rotation_world[1:] = _rotation_z(torch.tensor(torch.pi / 2))
    accepted.sample_at(0)

    rejected = _episode(timestamps_ns)
    rejected.hand_rotation_world[1:] = _rotation_z(torch.deg2rad(torch.tensor(90.1)))
    with pytest.raises(ValueError, match="adjacent hand rotations differ by more than 90 degrees"):
        rejected.sample_at(0)


def test_valid_anchors_accounts_for_caption_and_label_rejections(monkeypatch) -> None:
    episode = _episode(torch.arange(4, dtype=torch.int64) * 100_000_000)
    horizon_ns = round(XPERIENCE_HORIZON / XPERIENCE_FPS * 1e9)
    episode.segments = [
        {
            "id": 7,
            "first_index": 0,
            "last_index": 3,
            "end_ns": horizon_ns + 150_000_000,
            "prompt": "Task: synthetic.",
        }
    ]
    sampled: list[int] = []

    def sample_at(_episode, frame_index: int) -> RightHandCanonicalSample:
        sampled.append(frame_index)
        if frame_index == 1:
            raise ValueError("synthetic label rejection")
        return _sample_with_action()

    monkeypatch.setattr(CachedXperienceEpisode, "sample_at", sample_at)

    anchors, rejection_counts = episode.valid_anchors()

    assert sampled == [0, 1]
    assert len(anchors) == 1
    assert anchors[0].frame_index == 0
    assert anchors[0].timestamp_ns == 0
    assert anchors[0].segment_id == 7
    assert anchors[0].prompt == "Task: synthetic."
    assert anchors[0].motion_bin == "stationary"
    assert rejection_counts == {"synthetic label rejection": 1, "caption_horizon": 2}


@pytest.mark.parametrize(
    ("channel", "value", "expected"),
    [
        ("translation", 0.0099, "stationary"),
        ("translation", 0.01, "small"),
        ("translation", 0.05, "meaningful"),
        ("rotation", 4.9, "stationary"),
        ("rotation", 5.0, "small"),
        ("rotation", 20.0, "meaningful"),
        ("aperture", 0.049, "stationary"),
        ("aperture", 0.05, "small"),
        ("aperture", 0.15, "meaningful"),
    ],
)
def test_classify_motion_boundaries(channel: str, value: float, expected: str) -> None:
    action = torch.zeros(XPERIENCE_HORIZON, 1, 10, dtype=torch.float64)
    action[:, :, 3:9] = matrix_to_rotation_6d(torch.eye(3, dtype=torch.float64))
    if channel == "translation":
        action[-1, 0, 0] = value
    elif channel == "rotation":
        angle = torch.deg2rad(torch.tensor(value, dtype=torch.float64))
        action[-1, 0, 3:9] = matrix_to_rotation_6d(_rotation_z(angle))
    else:
        action[-1, 0, 9] = value

    assert classify_motion(action) == expected


def test_single_episode_split_is_deterministic_balanced_and_segment_disjoint(monkeypatch) -> None:
    anchors = [
        _anchor(segment_id * 100 + index, segment_id, ("meaningful", "small", "stationary")[index % 3])
        for segment_id in range(3)
        for index in range(12)
    ]
    episode = _episode(torch.tensor([0, 50_000_000], dtype=torch.int64))

    def valid_anchors(_episode):
        return anchors, {"synthetic": 3}

    monkeypatch.setattr(CachedXperienceEpisode, "valid_anchors", valid_anchors)

    first = select_xperience_split(episode, train_clips=8, validation_clips=4, seed=17)
    second = select_xperience_split(episode, train_clips=8, validation_clips=4, seed=17)

    assert first == second
    assert {anchor.segment_id for anchor in first.train}.isdisjoint(
        anchor.segment_id for anchor in first.validation
    )
    assert [anchor.motion_bin for anchor in first.train].count("meaningful") == 4
    assert [anchor.motion_bin for anchor in first.train].count("small") == 2
    assert [anchor.motion_bin for anchor in first.train].count("stationary") == 2
    assert [anchor.frame_index for anchor in first.train] == sorted(anchor.frame_index for anchor in first.train)
    assert first.rejection_counts == {"synthetic": 3}
    assert first.source_fps == 20.0


def _install_fake_decoder(monkeypatch, calls: list[tuple[Path, list[float]]]) -> None:
    lerobot = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    video_utils = types.ModuleType("lerobot.datasets.video_utils")

    def decode_video_frames(path, timestamps, **_kwargs):
        calls.append((path, timestamps))
        return [torch.full((2, 2, 3), len(calls), dtype=torch.uint8)]

    video_utils.decode_video_frames = decode_video_frames
    datasets.video_utils = video_utils
    lerobot.datasets = datasets
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.video_utils", video_utils)


def test_cached_dataset_indexing_and_frame_cache(monkeypatch) -> None:
    calls: list[tuple[Path, list[float]]] = []
    _install_fake_decoder(monkeypatch, calls)
    episode = _episode(torch.tensor([1_000_000_000, 1_200_000_000]))
    episode._sample_calls = []

    def sample_at(_episode, frame_index: int) -> RightHandCanonicalSample:
        _episode._sample_calls.append(frame_index)
        return _sample_with_action()

    monkeypatch.setattr(CachedXperienceEpisode, "sample_at", sample_at)
    anchors = (
        XperienceAnchor(1, 1_200_000_000, 0, "first", "stationary"),
        XperienceAnchor(1, 1_200_000_000, 0, "last", "stationary"),
    )
    dataset = CachedXperienceDataset(episode, anchors)

    first = dataset[0]
    last = dataset[-1]

    assert len(dataset) == 2
    assert first["task"] == "first"
    assert last["task"] == "last"
    assert first["observation.images.stereo_left"] is last["observation.images.stereo_left"]
    assert episode._sample_calls == [1, 1]
    assert calls == [(Path("synthetic.mp4"), [0.2])]


def test_multi_episode_dataset_uses_per_episode_frame_caches(monkeypatch) -> None:
    calls: list[tuple[Path, list[float]]] = []
    _install_fake_decoder(monkeypatch, calls)
    episodes = [_episode(torch.tensor([base, base + 200_000_000])) for base in (1_000_000_000, 2_000_000_000)]
    for index, episode in enumerate(episodes):
        episode.video_path = Path(f"episode-{index}.mp4")
        episode._sample_calls = []

    def sample_at(_episode, frame_index: int) -> RightHandCanonicalSample:
        _episode._sample_calls.append(frame_index)
        return _sample_with_action()

    monkeypatch.setattr(CachedXperienceEpisode, "sample_at", sample_at)
    samples = (
        (0, XperienceAnchor(1, 1_200_000_000, 0, "zero", "stationary")),
        (1, XperienceAnchor(1, 2_200_000_000, 0, "one", "stationary")),
        (0, XperienceAnchor(1, 1_200_000_000, 0, "zero again", "stationary")),
    )
    dataset = MultiEpisodeCachedXperienceDataset(episodes, samples)

    images = [dataset[index]["observation.images.stereo_left"] for index in range(len(dataset))]

    assert images[0] is images[2]
    assert images[0] is not images[1]
    assert episodes[0]._sample_calls == [1, 1]
    assert episodes[1]._sample_calls == [1]
    assert calls == [
        (Path("episode-0.mp4"), [0.2]),
        (Path("episode-1.mp4"), [0.2]),
    ]
