from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def tensor_to_image(tensor) -> Image.Image:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def create_preview(
    dataset_id: str,
    dataset_root: Path,
    episode: int,
    frame_count: int,
    output: Path,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_id, root=dataset_root)
    if episode < 0 or episode >= dataset.num_episodes:
        raise ValueError(f"episode must be between 0 and {dataset.num_episodes - 1}")
    episode_metadata = dataset.meta.episodes[episode]
    start = int(episode_metadata["dataset_from_index"])
    stop = int(episode_metadata["dataset_to_index"])
    indices = np.linspace(start, stop - 1, min(frame_count, stop - start), dtype=int)
    camera_keys = sorted(key for key in dataset.features if key.startswith("observation.images."))
    if not camera_keys:
        raise RuntimeError("dataset contains no cameras")

    thumbnail_width = 320
    thumbnail_height = 240
    tile_height = thumbnail_height * len(camera_keys) + 28
    columns = min(4, len(indices))
    rows = math.ceil(len(indices) / columns)
    sheet = Image.new("RGB", (columns * thumbnail_width, rows * tile_height), (15, 17, 21))
    draw = ImageDraw.Draw(sheet)
    for tile_index, dataset_index in enumerate(indices):
        item = dataset[int(dataset_index)]
        x = tile_index % columns * thumbnail_width
        y = tile_index // columns * tile_height
        for camera_index, key in enumerate(camera_keys):
            image = tensor_to_image(item[key]).resize((thumbnail_width, thumbnail_height))
            sheet.paste(image, (x, y + camera_index * thumbnail_height))
            draw.text(
                (x + 8, y + camera_index * thumbnail_height + 8),
                key.removeprefix("observation.images."),
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
        frame_index = int(item["frame_index"].item())
        draw.text((x + 8, y + tile_height - 22), f"frame {frame_index}", fill=(210, 218, 230))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    print(f"preview={output} episode={episode} frames={len(indices)} cameras={len(camera_keys)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="export a synthetic dataset episode contact sheet")
    parser.add_argument("--dataset", default="embodi/sim-so101-pickplace-two-camera")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/embodi-sim-pickplace-two-camera"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset-preview.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    create_preview(args.dataset, args.dataset_root, args.episode, args.frames, args.output)


if __name__ == "__main__":
    main()
