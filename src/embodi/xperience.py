from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .canonical import matrix_to_rotation_6d, rotation_6d_to_matrix


XPERIENCE_HORIZON = 32
XPERIENCE_FPS = 30
THUMB_TIP_INDEX = 4
INDEX_TIP_INDEX = 8
INDEX_MCP_INDEX = 5
PINKY_MCP_INDEX = 17
RIGHT_BODY_WRIST_INDEX = 21


@dataclass(frozen=True)
class RightHandCanonicalSample:
    canonical_state: Tensor
    canonical_action: Tensor
    canonical_part_mask: Tensor


@dataclass(frozen=True)
class XperienceAnchor:
    frame_index: int
    timestamp_ns: int
    segment_id: int
    prompt: str
    motion_bin: str


@dataclass(frozen=True)
class XperienceSplit:
    train: tuple[XperienceAnchor, ...]
    validation: tuple[XperienceAnchor, ...]
    rejection_counts: dict[str, int]
    source_fps: float


@dataclass(frozen=True)
class MultiEpisodeXperienceSplit:
    train: tuple[tuple[int, XperienceAnchor], ...]
    validation: tuple[tuple[int, XperienceAnchor], ...]
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]
    rejection_counts: dict[str, int]


def normalized_hand_opening(
    thumb_tip: Tensor,
    index_tip: Tensor,
    palm_width: Tensor,
    *,
    closed_ratio: float,
    open_ratio: float,
) -> Tensor:
    """Convert fingertip aperture divided by palm width to a calibrated [0, 1] value."""
    thumb_tip = torch.as_tensor(thumb_tip)
    index_tip = torch.as_tensor(index_tip, device=thumb_tip.device, dtype=thumb_tip.dtype)
    palm_width = torch.as_tensor(palm_width, device=thumb_tip.device, dtype=thumb_tip.dtype)
    if thumb_tip.shape != index_tip.shape or thumb_tip.shape[-1:] != (3,):
        raise ValueError("thumb_tip and index_tip must have matching shape [..., 3]")
    if palm_width.shape not in (thumb_tip.shape[:-1], thumb_tip.shape[:-1] + (1,)):
        raise ValueError("palm_width must have shape [...] or [..., 1]")
    if not open_ratio > closed_ratio:
        raise ValueError("open_ratio must be greater than closed_ratio")
    if not torch.isfinite(thumb_tip).all() or not torch.isfinite(index_tip).all():
        raise ValueError("hand joints must be finite")
    if not torch.isfinite(palm_width).all() or (palm_width <= 0).any():
        raise ValueError("palm_width must be finite and positive")
    width = palm_width.squeeze(-1) if palm_width.ndim == thumb_tip.ndim else palm_width
    ratio = torch.linalg.vector_norm(thumb_tip - index_tip, dim=-1) / width
    return ((ratio - closed_ratio) / (open_ratio - closed_ratio)).clamp(0.0, 1.0)


def quaternion_wxyz_to_matrix(quaternion: Tensor) -> Tensor:
    quaternion = torch.as_tensor(quaternion)
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have shape [..., 4]")
    if not torch.isfinite(quaternion).all():
        raise ValueError("quaternion must be finite")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if (norm < 1e-6).any():
        raise ValueError("quaternion norm must be nonzero")
    w, x, y, z = (quaternion / norm).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def interpolate_rotations(left: Tensor, right: Tensor, weight: Tensor) -> Tensor:
    """Interpolate proper rotation matrices along their shortest geodesic."""
    left = torch.as_tensor(left)
    right = torch.as_tensor(right, device=left.device, dtype=left.dtype)
    weight = torch.as_tensor(weight, device=left.device, dtype=left.dtype)
    if left.shape != right.shape or left.shape[-2:] != (3, 3):
        raise ValueError("left and right rotations must have matching shape [..., 3, 3]")
    if weight.shape != left.shape[:-2]:
        raise ValueError("weight must match rotation batch dimensions")
    relative = left.transpose(-1, -2) @ right
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    if (angle > torch.pi / 2).any():
        raise ValueError("adjacent hand rotations differ by more than 90 degrees")
    skew = relative - relative.transpose(-1, -2)
    sin_angle = torch.sin(angle)
    scale = torch.where(angle.abs() < 1e-5, torch.full_like(angle, 0.5), angle / (2 * sin_angle))
    log_rotation = skew * scale[..., None, None]
    interpolated_relative = torch.matrix_exp(log_rotation * weight[..., None, None])
    return left @ interpolated_relative


def _pose7_to_transform(pose: Tensor) -> Tensor:
    pose = torch.as_tensor(pose)
    if pose.shape[-1] != 7:
        raise ValueError("pose must have shape [..., 7] as quaternion wxyz plus translation")
    transform = torch.zeros(*pose.shape[:-1], 4, 4, device=pose.device, dtype=pose.dtype)
    transform[..., :3, :3] = quaternion_wxyz_to_matrix(pose[..., :4])
    transform[..., :3, 3] = pose[..., 4:]
    transform[..., 3, 3] = 1
    return transform


def camera_hand_to_world(
    slam_quaternion_wxyz: Tensor,
    slam_translation: Tensor,
    transform_camera_body: Tensor,
    hand_position_camera: Tensor,
    hand_rotation_camera: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply the official Xperience world-to-body SLAM and stereo calibration convention."""
    world_to_body = _pose7_to_transform(torch.cat((slam_quaternion_wxyz, slam_translation), dim=-1))
    transform_camera_body = torch.as_tensor(
        transform_camera_body, device=world_to_body.device, dtype=world_to_body.dtype
    )
    world_to_camera = transform_camera_body @ world_to_body
    rotation_camera_world = world_to_camera[..., :3, :3]
    translation_camera_world = world_to_camera[..., :3, 3]
    rotation_world_camera = rotation_camera_world.transpose(-1, -2)
    position_world = rotation_world_camera @ (hand_position_camera - translation_camera_world).unsqueeze(-1)
    rotation_world_hand = rotation_world_camera @ hand_rotation_camera
    return position_world.squeeze(-1), rotation_world_hand


def _caption_frame_time(value: Any, timestamps_ns: Tensor) -> int:
    if isinstance(value, str) and value.startswith("frame_"):
        frame_index = int(value.removeprefix("frame_"))
        if frame_index < 0:
            raise ValueError("caption frame index must be non-negative")
        return int(timestamps_ns[min(frame_index, len(timestamps_ns) - 1)])
    return int(value)


def _caption_segments(raw_caption: bytes | str, timestamps_ns: Tensor) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(raw_caption, bytes):
        raw_caption = raw_caption.decode("utf-8")
    data = json.loads(raw_caption)
    main_task = str(data.get("config", {}).get("Main Task", "")).strip()
    segments: list[dict[str, Any]] = []
    for segment in data.get("segments", []):
        start = _caption_frame_time(segment["start_frame"], timestamps_ns)
        end = _caption_frame_time(segment["end_frame"], timestamps_ns)
        subtask = str(segment.get("Sub Task", "")).strip()
        action = ""
        for candidate in segment.get("Current Action", []):
            action_start = _caption_frame_time(candidate.get("start_frame", start), timestamps_ns)
            action_end = _caption_frame_time(candidate.get("end_frame", end), timestamps_ns)
            if action_start <= start <= action_end:
                action = str(candidate.get("label", "")).strip()
                break
        instruction = subtask or action or main_task
        if not instruction:
            continue
        prompt = f"Task: {main_task}.\nCurrent subtask: {instruction}." if main_task else f"Task: {instruction}."
        first_index = int(torch.searchsorted(timestamps_ns, start, right=False))
        last_index = int(torch.searchsorted(timestamps_ns, end, right=True)) - 1
        if first_index <= last_index:
            segments.append(
                {
                    "id": int(segment.get("segment_id", len(segments))),
                    "start_ns": start,
                    "end_ns": end,
                    "first_index": first_index,
                    "last_index": last_index,
                    "prompt": prompt,
                }
            )
    return main_task, segments


class CachedXperienceEpisode:
    """Required small arrays from one cached episode; large depth arrays remain unopened."""

    def __init__(self, episode_dir: str | Path, *, closed_ratio: float = 0.18, open_ratio: float = 1.0) -> None:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("Xperience loading requires h5py; install the train extras") from error

        if not open_ratio > closed_ratio:
            raise ValueError("open_ratio must be greater than closed_ratio")
        self.episode_dir = Path(episode_dir)
        self.video_path = self.episode_dir / "stereo_left.mp4"
        annotation_path = self.episode_dir / "annotation.hdf5"
        if not annotation_path.is_file() or not self.video_path.is_file():
            raise FileNotFoundError(f"episode is missing annotation.hdf5 or stereo_left.mp4: {self.episode_dir}")
        with h5py.File(annotation_path, "r") as file:
            self.timestamps_ns = torch.tensor(
                [int(value) for value in file["video/device_timestamp"][:]], dtype=torch.int64
            )
            self.root_pose = torch.from_numpy(file["full_body_mocap/Ts_world_root"][:]).float()
            self.slam_quaternion = torch.from_numpy(file["slam/quat_wxyz"][:]).float()
            self.slam_translation = torch.from_numpy(file["slam/trans_xyz"][:]).float()
            self.hand_joints = torch.from_numpy(file["hand_mocap/right_joints_3d"][:]).float()
            self.hand_rotation_camera = torch.from_numpy(
                file["hand_mocap/right_mano_hand_global_orient"][:]
            ).float().reshape(-1, 3, 3)
            self.body_right_wrist_world = torch.from_numpy(
                file["full_body_mocap/keypoints"][:, RIGHT_BODY_WRIST_INDEX]
            ).float()
            self.transform_camera_body = torch.from_numpy(file["calibration/cam01/T_c0_b"][:]).float()
            raw_caption = file["caption"][()]
        if not torch.all(self.timestamps_ns[1:] > self.timestamps_ns[:-1]):
            raise ValueError("video timestamps must be strictly increasing")
        self.main_task, self.segments = _caption_segments(raw_caption, self.timestamps_ns)
        self.closed_ratio = closed_ratio
        self.open_ratio = open_ratio
        self.hand_position_world, self.hand_rotation_world = camera_hand_to_world(
            self.slam_quaternion,
            self.slam_translation,
            self.transform_camera_body,
            self.hand_joints[:, 0],
            self.hand_rotation_camera,
        )
        palm_width = torch.linalg.vector_norm(
            self.hand_joints[:, INDEX_MCP_INDEX] - self.hand_joints[:, PINKY_MCP_INDEX], dim=-1
        )
        aperture = torch.linalg.vector_norm(
            self.hand_joints[:, THUMB_TIP_INDEX] - self.hand_joints[:, INDEX_TIP_INDEX], dim=-1
        )
        self.opening = ((aperture / palm_width - closed_ratio) / (open_ratio - closed_ratio)).clamp(0, 1)
        del self.hand_joints

    @property
    def source_fps(self) -> float:
        intervals = (self.timestamps_ns[1:] - self.timestamps_ns[:-1]).double() / 1e9
        return float(1 / intervals.median())

    def sample_at(self, frame_index: int) -> RightHandCanonicalSample:
        anchor_ns = self.timestamps_ns[frame_index]
        offsets = torch.arange(XPERIENCE_HORIZON + 1, dtype=torch.float64) / XPERIENCE_FPS
        target_ns = anchor_ns + torch.round(offsets * 1e9).to(torch.int64)
        right_indices = torch.searchsorted(self.timestamps_ns, target_ns, right=False)
        if (right_indices >= len(self.timestamps_ns)).any():
            raise ValueError("incomplete future horizon")
        exact = self.timestamps_ns[right_indices] == target_ns
        left_indices = torch.where(exact, right_indices, right_indices - 1)
        if (left_indices < 0).any():
            raise ValueError("target precedes the first source frame")
        left_ns = self.timestamps_ns[left_indices]
        right_ns = self.timestamps_ns[right_indices]
        bracket_ns = right_ns - left_ns
        if (bracket_ns[~exact] > 75_000_000).any():
            raise ValueError("interpolation bracket exceeds 75 ms")
        denominator = torch.where(exact, torch.ones_like(bracket_ns), bracket_ns)
        weight = ((target_ns - left_ns).double() / denominator.double()).float()
        required = (
            self.root_pose[frame_index],
            self.hand_position_world[left_indices],
            self.hand_position_world[right_indices],
            self.hand_rotation_world[left_indices],
            self.hand_rotation_world[right_indices],
            self.opening[left_indices],
            self.opening[right_indices],
        )
        if not all(torch.isfinite(value).all() for value in required):
            raise ValueError("non-finite synchronized labels")
        wrist_error = torch.linalg.vector_norm(
            self.hand_position_world[frame_index] - self.body_right_wrist_world[frame_index]
        )
        if not torch.isfinite(wrist_error) or wrist_error > 0.10:
            raise ValueError("hand/body wrist disagreement exceeds 10 cm")
        position = torch.lerp(
            self.hand_position_world[left_indices], self.hand_position_world[right_indices], weight[:, None]
        )
        rotation = interpolate_rotations(
            self.hand_rotation_world[left_indices], self.hand_rotation_world[right_indices], weight
        )
        opening = torch.lerp(self.opening[left_indices], self.opening[right_indices], weight)
        root_transform = _pose7_to_transform(self.root_pose[frame_index])
        sample = convert_right_hand_trajectory(root_transform, position, rotation, opening)
        validate_right_hand_sample(sample)
        return sample

    def valid_anchors(self) -> tuple[list[XperienceAnchor], dict[str, int]]:
        anchors: list[XperienceAnchor] = []
        rejection_counts: dict[str, int] = {}
        horizon_ns = round(XPERIENCE_HORIZON / XPERIENCE_FPS * 1e9)
        for segment in self.segments:
            for frame_index in range(segment["first_index"], segment["last_index"] + 1):
                timestamp_ns = int(self.timestamps_ns[frame_index])
                if timestamp_ns + horizon_ns > segment["end_ns"]:
                    rejection_counts["caption_horizon"] = rejection_counts.get("caption_horizon", 0) + 1
                    continue
                try:
                    sample = self.sample_at(frame_index)
                except ValueError as error:
                    reason = str(error)
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue
                motion_bin = classify_motion(sample.canonical_action)
                anchors.append(
                    XperienceAnchor(
                        frame_index=frame_index,
                        timestamp_ns=timestamp_ns,
                        segment_id=segment["id"],
                        prompt=segment["prompt"],
                        motion_bin=motion_bin,
                    )
                )
        return anchors, rejection_counts


def classify_motion(canonical_action: Tensor) -> str:
    translation = torch.linalg.vector_norm(canonical_action[:, 0, :3], dim=-1).max().item()
    rotation = rotation_6d_to_matrix(canonical_action[:, 0, 3:9])
    angle = torch.acos(((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).max().item()
    aperture = (canonical_action[:, 0, 9] - canonical_action[0, 0, 9]).abs().max().item()
    if translation >= 0.05 or angle >= torch.deg2rad(torch.tensor(20.0)).item() or aperture >= 0.15:
        return "meaningful"
    if translation >= 0.01 or angle >= torch.deg2rad(torch.tensor(5.0)).item() or aperture >= 0.05:
        return "small"
    return "stationary"


def select_xperience_split(
    episode: CachedXperienceEpisode,
    *,
    train_clips: int,
    validation_clips: int,
    seed: int,
) -> XperienceSplit:
    if train_clips <= 0 or validation_clips <= 0:
        raise ValueError("train and validation clip counts must be positive")
    anchors, rejection_counts = episode.valid_anchors()
    segment_ids = sorted({anchor.segment_id for anchor in anchors})
    if len(segment_ids) < 2:
        raise RuntimeError("at least two valid caption segments are required")
    randomizer = random.Random(seed)
    randomizer.shuffle(segment_ids)
    validation_segments: set[int] = set()
    validation_pool: list[XperienceAnchor] = []
    train_pool: list[XperienceAnchor] = anchors
    for segment_id in segment_ids:
        validation_segments.add(segment_id)
        validation_pool = [anchor for anchor in anchors if anchor.segment_id in validation_segments]
        train_pool = [anchor for anchor in anchors if anchor.segment_id not in validation_segments]
        if len(validation_pool) >= validation_clips and len(train_pool) >= train_clips:
            break
    if len(validation_pool) < validation_clips or len(train_pool) < train_clips:
        raise RuntimeError("not enough disjoint valid anchors for the requested split")
    train = _balanced_sample(train_pool, train_clips, randomizer)
    validation = randomizer.sample(validation_pool, validation_clips)
    return XperienceSplit(
        train=tuple(sorted(train, key=lambda anchor: anchor.frame_index)),
        validation=tuple(sorted(validation, key=lambda anchor: anchor.frame_index)),
        rejection_counts=rejection_counts,
        source_fps=episode.source_fps,
    )


def _balanced_sample(
    anchors: Sequence[XperienceAnchor], count: int, randomizer: random.Random
) -> list[XperienceAnchor]:
    targets = {
        "meaningful": round(count * 0.50),
        "small": round(count * 0.25),
    }
    targets["stationary"] = count - targets["meaningful"] - targets["small"]
    remaining = list(anchors)
    selected: list[XperienceAnchor] = []
    for motion_bin in ("meaningful", "small", "stationary"):
        candidates = [anchor for anchor in remaining if anchor.motion_bin == motion_bin]
        take = min(targets[motion_bin], len(candidates))
        chosen = randomizer.sample(candidates, take)
        selected.extend(chosen)
        chosen_set = set(chosen)
        remaining = [anchor for anchor in remaining if anchor not in chosen_set]
    if len(selected) < count:
        selected.extend(randomizer.sample(remaining, count - len(selected)))
    return selected


def _nested_balanced_sample(
    anchors: Sequence[XperienceAnchor], count: int, randomizer: random.Random
) -> list[XperienceAnchor]:
    if count > len(anchors):
        raise ValueError("sample count exceeds available anchors")
    by_bin = {
        motion_bin: [anchor for anchor in anchors if anchor.motion_bin == motion_bin]
        for motion_bin in ("meaningful", "small", "stationary")
    }
    for candidates in by_bin.values():
        randomizer.shuffle(candidates)
    pattern = ("meaningful", "meaningful", "small", "stationary")
    selected: list[XperienceAnchor] = []
    pattern_index = 0
    while len(selected) < count and any(by_bin.values()):
        motion_bin = pattern[pattern_index % len(pattern)]
        pattern_index += 1
        candidates = by_bin[motion_bin]
        if candidates:
            selected.append(candidates.pop())
        elif pattern_index % len(pattern) == 0:
            remaining = [anchor for values in by_bin.values() for anchor in values]
            randomizer.shuffle(remaining)
            selected.extend(remaining[: count - len(selected)])
            break
    return selected


def select_multi_episode_xperience_split(
    episodes: Sequence[CachedXperienceEpisode],
    *,
    train_clips: int,
    validation_clips: int,
    seed: int,
) -> MultiEpisodeXperienceSplit:
    if len(episodes) < 2:
        raise ValueError("multi-episode splitting requires at least two episodes")
    if train_clips <= 0 or validation_clips <= 0:
        raise ValueError("train and validation clip counts must be positive")
    anchors_by_episode = []
    rejection_counts: dict[str, int] = {}
    for episode in episodes:
        anchors, rejected = episode.valid_anchors()
        anchors_by_episode.append(anchors)
        for name, count in rejected.items():
            rejection_counts[name] = rejection_counts.get(name, 0) + count
    randomizer = random.Random(seed)
    episode_indices = [index for index, anchors in enumerate(anchors_by_episode) if anchors]
    if len(episode_indices) < 2:
        raise RuntimeError("at least two episodes with valid anchors are required")
    randomizer.shuffle(episode_indices)
    episode_indices.sort(key=lambda index: len(anchors_by_episode[index]))
    validation_indices: list[int] = []
    validation_count = 0
    total_count = sum(len(anchors) for anchors in anchors_by_episode)
    for index in episode_indices:
        episode_count = len(anchors_by_episode[index])
        if total_count - validation_count - episode_count < train_clips:
            continue
        validation_indices.append(index)
        validation_count += episode_count
        if validation_count >= validation_clips:
            break
    training_indices = [index for index in episode_indices if index not in validation_indices]
    if validation_count < validation_clips or sum(len(anchors_by_episode[i]) for i in training_indices) < train_clips:
        raise RuntimeError("not enough episode-disjoint anchors for the requested multi-episode split")
    train_pool = [
        (index, anchor)
        for index in training_indices
        for anchor in anchors_by_episode[index]
    ]
    validation_pool = [
        (index, anchor)
        for index in validation_indices
        for anchor in anchors_by_episode[index]
    ]
    balanced_anchors = _nested_balanced_sample(
        [anchor for _, anchor in train_pool], train_clips, random.Random(seed + 1)
    )
    locations = {id(anchor): episode_index for episode_index, anchor in train_pool}
    train = [(locations[id(anchor)], anchor) for anchor in balanced_anchors]
    validation = random.Random(seed + 2).sample(validation_pool, validation_clips)
    sort_key = lambda item: (item[0], item[1].frame_index)
    return MultiEpisodeXperienceSplit(
        train=tuple(sorted(train, key=sort_key)),
        validation=tuple(sorted(validation, key=sort_key)),
        train_episode_indices=tuple(sorted(training_indices)),
        validation_episode_indices=tuple(sorted(validation_indices)),
        rejection_counts=rejection_counts,
    )


def select_fixed_episode_xperience_split(
    episodes: Sequence[CachedXperienceEpisode],
    *,
    train_episode_indices: Sequence[int],
    validation_episode_indices: Sequence[int],
    train_clips: int,
    validation_clips: int,
    seed: int,
) -> MultiEpisodeXperienceSplit:
    train_indices = tuple(dict.fromkeys(train_episode_indices))
    validation_indices = tuple(dict.fromkeys(validation_episode_indices))
    if not train_indices or not validation_indices:
        raise ValueError("fixed splitting requires train and validation episodes")
    if set(train_indices) & set(validation_indices):
        raise ValueError("train and validation episodes must be disjoint")
    if train_clips <= 0 or validation_clips <= 0:
        raise ValueError("train and validation clip counts must be positive")
    if any(index < 0 or index >= len(episodes) for index in train_indices + validation_indices):
        raise IndexError("episode index is out of range")
    anchors_by_episode: dict[int, list[XperienceAnchor]] = {}
    rejection_counts: dict[str, int] = {}
    for index in train_indices + validation_indices:
        anchors, rejected = episodes[index].valid_anchors()
        anchors_by_episode[index] = anchors
        for name, count in rejected.items():
            rejection_counts[name] = rejection_counts.get(name, 0) + count
    base_quota, remainder = divmod(train_clips, len(train_indices))
    train: list[tuple[int, XperienceAnchor]] = []
    for position, index in enumerate(train_indices):
        quota = base_quota + (position < remainder)
        if len(anchors_by_episode[index]) < quota:
            raise RuntimeError(f"episode {index} has fewer than {quota} valid training anchors")
        selected = _nested_balanced_sample(
            anchors_by_episode[index], quota, random.Random(seed + 1000 + index)
        )
        train.extend((index, anchor) for anchor in selected)
    validation_pool = [
        (index, anchor)
        for index in validation_indices
        for anchor in anchors_by_episode[index]
    ]
    if len(validation_pool) < validation_clips:
        raise RuntimeError("not enough valid anchors in fixed validation episodes")
    validation = random.Random(seed + 2).sample(validation_pool, validation_clips)
    sort_key = lambda item: (item[0], item[1].frame_index)
    return MultiEpisodeXperienceSplit(
        train=tuple(sorted(train, key=sort_key)),
        validation=tuple(sorted(validation, key=sort_key)),
        train_episode_indices=tuple(sorted(train_indices)),
        validation_episode_indices=tuple(sorted(validation_indices)),
        rejection_counts=rejection_counts,
    )


class CachedXperienceDataset(Dataset):
    def __init__(self, episode: CachedXperienceEpisode, anchors: Sequence[XperienceAnchor]) -> None:
        self.episode = episode
        self.anchors = tuple(anchors)
        self._image_cache: dict[int, Tensor] = {}

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor = self.anchors[index]
        sample = self.episode.sample_at(anchor.frame_index)
        image = self._image_cache.get(anchor.frame_index)
        if image is None:
            from lerobot.datasets.video_utils import decode_video_frames

            timestamp_s = (anchor.timestamp_ns - int(self.episode.timestamps_ns[0])) / 1e9
            image = decode_video_frames(
                self.episode.video_path,
                [timestamp_s],
                tolerance_s=0.04,
                backend="pyav",
                return_uint8=True,
            )[0]
            self._image_cache[anchor.frame_index] = image
        return {
            "observation.images.stereo_left": image,
            "task": anchor.prompt,
            "canonical_state": sample.canonical_state,
            "canonical_action": sample.canonical_action,
            "canonical_part_mask": sample.canonical_part_mask,
        }


class MultiEpisodeCachedXperienceDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[CachedXperienceEpisode],
        samples: Sequence[tuple[int, XperienceAnchor]],
    ) -> None:
        self.episodes = tuple(episodes)
        self.samples = tuple(samples)
        self._image_caches: list[dict[int, Tensor]] = [{} for _ in episodes]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, anchor = self.samples[index]
        episode = self.episodes[episode_index]
        sample = episode.sample_at(anchor.frame_index)
        image = self._image_caches[episode_index].get(anchor.frame_index)
        if image is None:
            from lerobot.datasets.video_utils import decode_video_frames

            timestamp_s = (anchor.timestamp_ns - int(episode.timestamps_ns[0])) / 1e9
            image = decode_video_frames(
                episode.video_path,
                [timestamp_s],
                tolerance_s=0.04,
                backend="pyav",
                return_uint8=True,
            )[0]
            self._image_caches[episode_index][anchor.frame_index] = image
        return {
            "observation.images.stereo_left": image,
            "task": anchor.prompt,
            "canonical_state": sample.canonical_state,
            "canonical_action": sample.canonical_action,
            "canonical_part_mask": sample.canonical_part_mask,
        }


def convert_right_hand_trajectory(
    transform_world_root: Tensor,
    palm_position_world: Tensor,
    palm_rotation_world: Tensor,
    opening: Tensor,
    *,
    horizon: int = XPERIENCE_HORIZON,
) -> RightHandCanonicalSample:
    """Convert an anchor plus future world-frame hand poses to experiment-0 labels.

    ``transform_world_root`` maps root-frame points into the world frame at the
    anchor time. Pose and opening arrays contain the anchor followed by exactly
    ``horizon`` future samples on the 30 Hz grid.
    """
    transform_world_root = torch.as_tensor(transform_world_root)
    palm_position_world = torch.as_tensor(
        palm_position_world, device=transform_world_root.device, dtype=transform_world_root.dtype
    )
    palm_rotation_world = torch.as_tensor(
        palm_rotation_world, device=transform_world_root.device, dtype=transform_world_root.dtype
    )
    opening = torch.as_tensor(opening, device=transform_world_root.device, dtype=transform_world_root.dtype)
    sample_count = horizon + 1
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if transform_world_root.shape != (4, 4):
        raise ValueError("transform_world_root must have shape [4, 4]")
    if palm_position_world.shape != (sample_count, 3):
        raise ValueError(f"palm_position_world must have shape [{sample_count}, 3]")
    if palm_rotation_world.shape != (sample_count, 3, 3):
        raise ValueError(f"palm_rotation_world must have shape [{sample_count}, 3, 3]")
    if opening.shape != (sample_count,):
        raise ValueError(f"opening must have shape [{sample_count}]")
    if not all(
        torch.isfinite(value).all()
        for value in (transform_world_root, palm_position_world, palm_rotation_world, opening)
    ):
        raise ValueError("root transform, hand poses, and opening must be finite")
    if (opening < 0).any() or (opening > 1).any():
        raise ValueError("opening must remain in [0, 1]")

    root_rotation_world = transform_world_root[:3, :3]
    root_position_world = transform_world_root[:3, 3]
    _validate_rotation(root_rotation_world, "root rotation")
    _validate_rotation(palm_rotation_world, "palm rotations")
    expected_bottom = transform_world_root.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(transform_world_root[3], expected_bottom, atol=1e-5, rtol=1e-5):
        raise ValueError("transform_world_root must be a rigid homogeneous transform")

    world_to_root_rotation = root_rotation_world.transpose(-1, -2)
    position_root = (world_to_root_rotation @ (palm_position_world - root_position_world).T).T
    rotation_root = world_to_root_rotation.unsqueeze(0) @ palm_rotation_world

    canonical_state = transform_world_root.new_empty((1, 10))
    canonical_state[0, :3] = position_root[0]
    canonical_state[0, 3:9] = matrix_to_rotation_6d(rotation_root[0])
    canonical_state[0, 9] = opening[0]

    relative_rotation = rotation_root[0].transpose(-1, -2).unsqueeze(0) @ rotation_root[1:]
    canonical_action = transform_world_root.new_empty((horizon, 1, 10))
    canonical_action[:, 0, :3] = position_root[1:] - position_root[0]
    canonical_action[:, 0, 3:9] = matrix_to_rotation_6d(relative_rotation)
    canonical_action[:, 0, 9] = opening[1:]
    return RightHandCanonicalSample(
        canonical_state=canonical_state,
        canonical_action=canonical_action,
        canonical_part_mask=torch.ones(1, dtype=torch.bool, device=transform_world_root.device),
    )


def validate_right_hand_sample(sample: RightHandCanonicalSample, *, horizon: int = XPERIENCE_HORIZON) -> None:
    if sample.canonical_state.shape != (1, 10):
        raise ValueError("canonical_state must have shape [1, 10]")
    if sample.canonical_action.shape != (horizon, 1, 10):
        raise ValueError(f"canonical_action must have shape [{horizon}, 1, 10]")
    if sample.canonical_part_mask.shape != (1,) or sample.canonical_part_mask.dtype != torch.bool:
        raise ValueError("canonical_part_mask must be boolean with shape [1]")
    if not bool(sample.canonical_part_mask[0]):
        raise ValueError("primary_effector must be active")
    if not torch.isfinite(sample.canonical_state).all() or not torch.isfinite(sample.canonical_action).all():
        raise ValueError("canonical values must be finite")
    scalars = torch.cat((sample.canonical_state[:, 9], sample.canonical_action[:, :, 9].reshape(-1)))
    if (scalars < 0).any() or (scalars > 1).any():
        raise ValueError("canonical opening must remain in [0, 1]")
    rotations = torch.cat(
        (sample.canonical_state[:, 3:9], sample.canonical_action[:, :, 3:9].reshape(-1, 6))
    )
    reconstructed = rotation_6d_to_matrix(rotations)
    identity = torch.eye(3, device=reconstructed.device, dtype=reconstructed.dtype)
    if not torch.allclose(reconstructed.transpose(-1, -2) @ reconstructed, identity, atol=1e-4, rtol=1e-4):
        raise ValueError("canonical rotation6d values do not decode to rotations")


def _validate_rotation(rotation: Tensor, name: str) -> None:
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    gram = rotation.transpose(-1, -2) @ rotation
    determinant = torch.linalg.det(rotation)
    if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4) or not torch.allclose(
        determinant, torch.ones_like(determinant), atol=1e-4, rtol=1e-4
    ):
        raise ValueError(f"{name} must be proper orthonormal rotation matrices")
