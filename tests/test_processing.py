from types import SimpleNamespace

import torch

from embodi.processing import EmbodiProcessor, split_lerobot_batch


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
