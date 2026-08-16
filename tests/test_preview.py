from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from embodi import preview


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.permutation: tuple[int, ...] | None = None

    def detach(self):
        return self

    def cpu(self):
        return self

    def permute(self, *dimensions: int):
        self.permutation = dimensions
        self.array = self.array.transpose(dimensions)
        return self

    def numpy(self) -> np.ndarray:
        return self.array


def install_dataset(monkeypatch: pytest.MonkeyPatch, dataset) -> list[tuple[str, Path]]:
    calls: list[tuple[str, Path]] = []

    def factory(dataset_id: str, *, root: Path):
        calls.append((dataset_id, root))
        return dataset

    lerobot = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = factory
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_module)
    return calls


def test_tensor_to_image_converts_chw_values_to_clipped_uint8_rgb() -> None:
    tensor = FakeTensor(
        np.array(
            [
                [[-0.1, 0.5]],
                [[0.0, 1.0]],
                [[1.1, 0.25]],
            ],
            dtype=np.float32,
        )
    )

    image = preview.tensor_to_image(tensor)

    assert tensor.permutation == (1, 2, 0)
    assert image.mode == "RGB"
    np.testing.assert_array_equal(
        np.asarray(image),
        np.array([[[0, 0, 255], [127, 255, 63]]], dtype=np.uint8),
    )


def test_create_preview_samples_frames_and_builds_contact_sheet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Dataset:
        num_episodes = 1
        meta = SimpleNamespace(
            episodes=[{"dataset_from_index": 10, "dataset_to_index": 16}]
        )
        features = {
            "observation.images.wrist": object(),
            "state": object(),
            "observation.images.top": object(),
        }

        def __init__(self) -> None:
            self.indices: list[int] = []

        def __getitem__(self, index: int):
            self.indices.append(index)
            return {
                "observation.images.top": f"top-{index}",
                "observation.images.wrist": f"wrist-{index}",
                "frame_index": SimpleNamespace(item=lambda: index - 10),
            }

    class Thumbnail:
        def __init__(self, source: str) -> None:
            self.source = source
            self.size = None

        def resize(self, size: tuple[int, int]):
            self.size = size
            return self

    class Sheet:
        def __init__(self) -> None:
            self.pastes: list[tuple[Thumbnail, tuple[int, int]]] = []
            self.saved: tuple[Path, int] | None = None

        def paste(self, image: Thumbnail, position: tuple[int, int]) -> None:
            self.pastes.append((image, position))

        def save(self, output: Path, *, quality: int) -> None:
            self.saved = (output, quality)

    class Draw:
        def __init__(self) -> None:
            self.texts: list[tuple[tuple[int, int], str, dict[str, object]]] = []

        def text(self, position: tuple[int, int], text: str, **kwargs: object) -> None:
            self.texts.append((position, text, kwargs))

    dataset = Dataset()
    dataset_calls = install_dataset(monkeypatch, dataset)
    sheet = Sheet()
    draw = Draw()
    image_new_calls: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []

    def fake_new(mode: str, size: tuple[int, int], color: tuple[int, int, int]):
        image_new_calls.append((mode, size, color))
        return sheet

    monkeypatch.setattr(preview.Image, "new", fake_new)
    monkeypatch.setattr(preview.ImageDraw, "Draw", lambda _: draw)
    monkeypatch.setattr(preview, "tensor_to_image", Thumbnail)
    output = tmp_path / "nested" / "preview.jpg"
    dataset_root = tmp_path / "dataset"

    preview.create_preview("example/data", dataset_root, 0, 3, output)

    assert dataset_calls == [("example/data", dataset_root)]
    assert dataset.indices == [10, 12, 15]
    assert image_new_calls == [("RGB", (960, 508), (15, 17, 21))]
    assert [(image.source, image.size, position) for image, position in sheet.pastes] == [
        ("top-10", (320, 240), (0, 0)),
        ("wrist-10", (320, 240), (0, 240)),
        ("top-12", (320, 240), (320, 0)),
        ("wrist-12", (320, 240), (320, 240)),
        ("top-15", (320, 240), (640, 0)),
        ("wrist-15", (320, 240), (640, 240)),
    ]
    assert [text for _, text, _ in draw.texts] == [
        "top",
        "wrist",
        "frame 0",
        "top",
        "wrist",
        "frame 2",
        "top",
        "wrist",
        "frame 5",
    ]
    assert output.parent.is_dir()
    assert sheet.saved == (output, 90)
    assert capsys.readouterr().out == (
        f"preview={output} episode=0 frames=3 cameras=2\n"
    )


@pytest.mark.parametrize("episode", [-1, 2])
def test_create_preview_rejects_episode_outside_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode: int
) -> None:
    dataset = SimpleNamespace(num_episodes=2)
    install_dataset(monkeypatch, dataset)

    with pytest.raises(ValueError, match="episode must be between 0 and 1"):
        preview.create_preview("data", tmp_path, episode, 1, tmp_path / "out.jpg")


def test_create_preview_rejects_dataset_without_cameras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = SimpleNamespace(
        num_episodes=1,
        meta=SimpleNamespace(
            episodes=[{"dataset_from_index": 0, "dataset_to_index": 1}]
        ),
        features={"observation.state": object()},
    )
    install_dataset(monkeypatch, dataset)

    with pytest.raises(RuntimeError, match="dataset contains no cameras"):
        preview.create_preview("data", tmp_path, 0, 1, tmp_path / "out.jpg")


def test_create_preview_rejects_empty_episode_with_domain_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = SimpleNamespace(
        num_episodes=1,
        meta=SimpleNamespace(
            episodes=[{"dataset_from_index": 4, "dataset_to_index": 4}]
        ),
        features={"observation.images.top": object()},
    )
    install_dataset(monkeypatch, dataset)

    with pytest.raises(ValueError, match="episode contains no frames"):
        preview.create_preview("data", tmp_path, 0, 1, tmp_path / "out.jpg")


def test_parse_args_converts_paths_and_numeric_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embodi-preview",
            "--dataset",
            "org/data",
            "--dataset-root",
            "/data/root",
            "--episode",
            "4",
            "--frames",
            "7",
            "--output",
            "/tmp/sheet.png",
        ],
    )

    args = preview.parse_args()

    assert vars(args) == {
        "dataset": "org/data",
        "dataset_root": Path("/data/root"),
        "episode": 4,
        "frames": 7,
        "output": Path("/tmp/sheet.png"),
    }


def test_main_rejects_nonpositive_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(frames=0)
    monkeypatch.setattr(preview, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="--frames must be positive"):
        preview.main()


def test_main_forwards_parsed_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(
        dataset="org/data",
        dataset_root=Path("root"),
        episode=2,
        frames=5,
        output=Path("preview.jpg"),
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(preview, "parse_args", lambda: args)
    monkeypatch.setattr(preview, "create_preview", lambda *values: calls.append(values))

    preview.main()

    assert calls == [("org/data", Path("root"), 2, 5, Path("preview.jpg"))]
