from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor


def validate_image_rescaling(image: Any, *, processor_rescales: bool) -> None:
    """Require exactly one image rescaling step before backbone normalization."""
    tensor = torch.as_tensor(image)
    if processor_rescales:
        if tensor.dtype != torch.uint8:
            raise ValueError("processor rescaling requires uint8 images in [0, 255]")
        return
    if not tensor.is_floating_point():
        raise ValueError("disabled processor rescaling requires floating-point images in [0, 1]")
    if not torch.isfinite(tensor).all() or tensor.numel() == 0:
        raise ValueError("pre-scaled images must be finite and non-empty")
    if tensor.min() < 0 or tensor.max() > 1:
        raise ValueError("pre-scaled images must lie in [0, 1]")


def camera_frame_to_tensor(frame: Any, *, processor_rescales: bool) -> Tensor:
    """Convert an HWC uint8 camera frame without applying rescaling twice."""
    tensor = torch.as_tensor(frame)
    if tensor.ndim != 3 or tensor.shape[-1] != 3 or tensor.dtype != torch.uint8:
        raise ValueError("camera frames must be HWC uint8 RGB images")
    tensor = tensor.permute(2, 0, 1)
    return tensor if processor_rescales else tensor.float() / 255.0


class EmbodiProcessor:
    def __init__(self, processor: Any, do_rescale: bool = True) -> None:
        self.processor = processor
        self.do_rescale = do_rescale
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "do_rescale"):
            image_processor.do_rescale = do_rescale

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        revision: str | None = None,
        do_rescale: bool = True,
    ) -> EmbodiProcessor:
        from transformers import AutoProcessor

        return cls(AutoProcessor.from_pretrained(model_name, revision=revision), do_rescale=do_rescale)

    def __call__(
        self,
        images: Sequence[Sequence[Any]],
        instructions: Sequence[str],
        state: Tensor | None,
        actions: Tensor | None = None,
        canonical_actions: Tensor | None = None,
        canonical_state: Tensor | None = None,
        canonical_slot_mask: Tensor | None = None,
        canonical_part_mask: Tensor | None = None,
        canonical_part_kind: Tensor | None = None,
        canonical_part_name_features: Tensor | None = None,
        canonical_channel_mask: Tensor | None = None,
        action_group_mask: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Any]:
        if len(images) != len(instructions) or (state is not None and len(images) != state.shape[0]):
            raise ValueError("images, instructions, and state must have the same batch size")
        conversations = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            for image in sample_images:
                validate_image_rescaling(image, processor_rescales=self.do_rescale)
            content = [{"type": "image", "image": image} for image in sample_images]
            content.append({"type": "text", "text": instruction})
            conversations.append([{"role": "user", "content": content}])
        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        if state is not None:
            batch["observation.state"] = state
        if actions is not None:
            batch["action"] = actions
        if canonical_actions is not None:
            batch["canonical_action"] = canonical_actions
        if canonical_state is not None:
            batch["canonical_state"] = canonical_state
        if canonical_slot_mask is not None:
            batch["canonical_slot_mask"] = canonical_slot_mask
        if canonical_part_mask is not None:
            batch["canonical_part_mask"] = canonical_part_mask
        if canonical_part_kind is not None:
            batch["canonical_part_kind"] = canonical_part_kind
        if canonical_part_name_features is not None:
            batch["canonical_part_name_features"] = canonical_part_name_features
        if canonical_channel_mask is not None:
            batch["canonical_channel_mask"] = canonical_channel_mask
        if action_group_mask is not None:
            batch["action_group_mask"] = action_group_mask
        if action_is_pad is not None:
            batch["action_is_pad"] = action_is_pad
        return batch


def split_lerobot_batch(
    batch: dict[str, Any],
    camera_names: Sequence[str] | None = None,
) -> tuple[list[list[Any]], list[str]]:
    if camera_names is None:
        image_keys = sorted(key for key in batch if key.startswith("observation.images."))
    else:
        image_keys = [f"observation.images.{name}" for name in camera_names]
    if not image_keys:
        raise KeyError("batch contains no observation.images.* keys")
    missing = [key for key in image_keys if key not in batch]
    if missing:
        raise KeyError(f"batch is missing configured cameras: {missing}")
    tasks = batch.get("task")
    if tasks is None:
        raise KeyError("batch contains no task instructions")
    batch_size = len(tasks)
    images = [[batch[key][index] for key in image_keys] for index in range(batch_size)]
    return images, list(tasks)
