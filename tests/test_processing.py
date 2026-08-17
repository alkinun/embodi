import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from embodi.processing import (
    EmbodiProcessor,
    camera_frame_to_tensor,
    split_lerobot_batch,
    validate_image_rescaling,
)


def test_split_lerobot_batch_selects_configured_cameras() -> None:
    top = torch.ones(2, 3, 4, 4)
    wrist = torch.zeros(2, 3, 4, 4)
    images, tasks = split_lerobot_batch(
        {
            "observation.state": torch.zeros(2, 6),
            "observation.images.top": top,
            "observation.images.wrist": wrist,
            "task": ["pick", "place"],
        },
        camera_names=("top",),
    )
    assert tasks == ["pick", "place"]
    assert len(images) == 2
    assert len(images[0]) == 1
    torch.testing.assert_close(images[0][0], top[0])


def test_processor_can_disable_float_image_rescaling() -> None:
    processor = SimpleNamespace(image_processor=SimpleNamespace(do_rescale=True))
    EmbodiProcessor(processor, do_rescale=False)
    assert processor.image_processor.do_rescale is False


def test_processor_builds_conversations_and_transports_supervision() -> None:
    class FakeProcessor:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, conversations, **kwargs):
            self.calls.append((conversations, kwargs))
            return {"input_ids": torch.tensor([[1, 2], [3, 4]])}

    raw_processor = FakeProcessor()
    processor = EmbodiProcessor(raw_processor, do_rescale=False)
    images = [
        [torch.zeros(3, 2, 2)],
        [torch.ones(3, 2, 2)],
    ]
    state = torch.zeros(2, 6)
    actions = torch.zeros(2, 4, 6)
    canonical_actions = torch.zeros(2, 4, 1, 10)
    canonical_state = torch.zeros(2, 1, 10)
    part_mask = torch.ones(2, 1, dtype=torch.bool)
    part_kind = torch.zeros(2, 1, dtype=torch.long)
    name_features = torch.zeros(2, 1, 32)
    channel_mask = torch.ones(2, 1, 10, dtype=torch.bool)
    group_mask = torch.ones(2, 3, dtype=torch.bool)
    action_is_pad = torch.zeros(2, 4, dtype=torch.bool)

    batch = processor(
        images,
        ["pick", "place"],
        state,
        actions=actions,
        canonical_actions=canonical_actions,
        canonical_state=canonical_state,
        canonical_part_mask=part_mask,
        canonical_part_kind=part_kind,
        canonical_part_name_features=name_features,
        canonical_channel_mask=channel_mask,
        action_group_mask=group_mask,
        action_is_pad=action_is_pad,
    )

    conversations, kwargs = raw_processor.calls[0]
    assert conversations[0][0]["content"] == [
        {"type": "image", "image": images[0][0]},
        {"type": "text", "text": "pick"},
    ]
    assert conversations[1][0]["content"][-1] == {"type": "text", "text": "place"}
    assert kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
    }
    assert batch["observation.state"] is state
    assert batch["action"] is actions
    assert batch["canonical_action"] is canonical_actions
    assert batch["canonical_state"] is canonical_state
    assert batch["canonical_part_mask"] is part_mask
    assert batch["canonical_part_kind"] is part_kind
    assert batch["canonical_part_name_features"] is name_features
    assert batch["canonical_channel_mask"] is channel_mask
    assert batch["action_group_mask"] is group_mask
    assert batch["action_is_pad"] is action_is_pad


def test_processor_rejects_mismatched_batch_sizes() -> None:
    processor = EmbodiProcessor(SimpleNamespace(), do_rescale=False)

    with pytest.raises(ValueError, match="same batch size"):
        processor([[torch.zeros(3, 2, 2)]], ["one", "two"], None)

    with pytest.raises(ValueError, match="same batch size"):
        processor([[torch.zeros(3, 2, 2)]], ["one"], torch.zeros(2, 6))


def test_processor_from_pretrained_forwards_model_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name: str, revision: str | None = None):
            calls.append((model_name, revision))
            return SimpleNamespace()

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    processor = EmbodiProcessor.from_pretrained("fake-model", revision="fake-revision", do_rescale=False)

    assert calls == [("fake-model", "fake-revision")]
    assert processor.do_rescale is False


def test_split_batch_does_not_require_native_state_for_core_pretraining() -> None:
    images, tasks = split_lerobot_batch(
        {
            "observation.images.stereo_left": torch.zeros(2, 3, 4, 4, dtype=torch.uint8),
            "task": ["first", "second"],
        },
        camera_names=("stereo_left",),
    )
    assert len(images) == 2
    assert tasks == ["first", "second"]


def test_camera_frame_preserves_uint8_when_processor_rescales() -> None:
    frame = np.full((2, 3, 3), 255, dtype=np.uint8)
    tensor = camera_frame_to_tensor(frame, processor_rescales=True)
    assert tensor.shape == (3, 2, 3)
    assert tensor.dtype == torch.uint8
    assert tensor.max() == 255


def test_camera_frame_scales_once_when_processor_does_not_rescale() -> None:
    frame = np.full((2, 3, 3), 255, dtype=np.uint8)
    tensor = camera_frame_to_tensor(frame, processor_rescales=False)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, torch.ones_like(tensor))


def test_camera_frame_rejects_non_uint8_or_non_rgb() -> None:
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        camera_frame_to_tensor(np.zeros((2, 3, 3), dtype=np.float32), processor_rescales=True)
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        camera_frame_to_tensor(np.zeros((2, 3), dtype=np.uint8), processor_rescales=True)


def test_image_rescaling_contract_accepts_exactly_one_scaling_path() -> None:
    validate_image_rescaling(torch.full((3, 2, 2), 255, dtype=torch.uint8), processor_rescales=True)
    validate_image_rescaling(torch.ones(3, 2, 2), processor_rescales=False)


@pytest.mark.parametrize(
    ("image", "processor_rescales", "message"),
    [
        (torch.ones(3, 2, 2), True, "requires uint8"),
        (torch.ones(3, 2, 2, dtype=torch.uint8), False, "requires floating-point"),
        (torch.full((3, 2, 2), 2.0), False, "must lie in"),
    ],
)
def test_image_rescaling_contract_rejects_double_or_missing_scaling(
    image: torch.Tensor,
    processor_rescales: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_image_rescaling(image, processor_rescales=processor_rescales)
