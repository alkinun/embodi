from io import BytesIO
from pathlib import Path

import pytest

from embodi.sim import assets


def test_mesh_files_returns_sorted_unique_obj_and_stl_paths() -> None:
    model_xml = """
    <mesh name="z" file="z.obj"/>
    <mesh file="nested/part.stl" scale="1 1 1"/>
    <mesh file="z.obj"/>
    <texture file="ignored.obj"/>
    <mesh file="ignored.png"/>
    """

    assert assets._mesh_files(model_xml) == ("nested/part.stl", "z.obj")


def test_download_streams_to_part_file_then_atomically_replaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    requests: list[tuple[str, int]] = []

    def fake_urlopen(url: str, *, timeout: int):
        requests.append((url, timeout))
        return Response(b"asset contents")

    monkeypatch.setattr(assets, "urlopen", fake_urlopen)
    destination = tmp_path / "mesh.obj"

    assets._download("https://example.test/mesh.obj", destination)

    assert requests == [("https://example.test/mesh.obj", 120)]
    assert destination.read_bytes() == b"asset contents"
    assert list(tmp_path.glob("mesh.obj.*.part")) == []


def test_download_removes_part_file_after_transfer_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingResponse:
        def __init__(self) -> None:
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self, _: int) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                return b"partial"
            raise OSError("transfer interrupted")

    monkeypatch.setattr(assets, "urlopen", lambda *_, **__: FailingResponse())
    destination = tmp_path / "mesh.obj"
    destination.write_bytes(b"valid target")

    with pytest.raises(OSError, match="transfer interrupted"):
        assets._download("https://example.test/mesh.obj", destination)

    assert destination.read_bytes() == b"valid target"
    assert list(tmp_path.glob("mesh.obj.*.part")) == []


def test_download_isolates_overlapping_callers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "mesh.obj"
    temporary_paths: list[Path] = []
    requests: list[str] = []
    real_named_temporary_file = assets.NamedTemporaryFile

    def tracking_named_temporary_file(*args: object, **kwargs: object):
        output = real_named_temporary_file(*args, **kwargs)
        temporary_paths.append(Path(output.name))
        return output

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    class OverlappingResponse:
        def __init__(self) -> None:
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self, _: int) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                assets._download("https://example.test/inner", destination)
                return b"outer contents"
            return b""

    def fake_urlopen(url: str, *, timeout: int):
        assert timeout == 120
        requests.append(url)
        if url.endswith("/inner"):
            return Response(b"inner contents")
        return OverlappingResponse()

    monkeypatch.setattr(assets, "NamedTemporaryFile", tracking_named_temporary_file)
    monkeypatch.setattr(assets, "urlopen", fake_urlopen)

    assets._download("https://example.test/outer", destination)

    assert requests == [
        "https://example.test/outer",
        "https://example.test/inner",
    ]
    assert destination.read_bytes() == b"outer contents"
    assert len(temporary_paths) == 2
    assert temporary_paths[0] != temporary_paths[1]
    assert all(path.parent == tmp_path for path in temporary_paths)
    assert all(path.name.startswith("mesh.obj.") for path in temporary_paths)
    assert all(path.suffix == ".part" for path in temporary_paths)
    assert all(not path.exists() for path in temporary_paths)


@pytest.mark.parametrize(
    ("model_directory", "model_filename", "message"),
    [
        ("", "model.xml", "model_directory"),
        ("nested/model", "model.xml", "model_directory"),
        ("model", "", "model_filename"),
        ("model", "nested/model.xml", "model_filename"),
        ("model", "model.txt", "model_filename"),
    ],
)
def test_ensure_menagerie_assets_rejects_unsafe_names(
    tmp_path: Path, model_directory: str, model_filename: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        assets.ensure_menagerie_assets(model_directory, model_filename, tmp_path)


def test_ensure_menagerie_assets_downloads_only_missing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cache"
    downloads: list[tuple[str, Path]] = []

    def fake_download(url: str, destination: Path) -> None:
        downloads.append((url, destination))
        if destination.name == "robot.xml":
            destination.write_text(
                '<mesh file="z.obj"/><mesh file="nested/part.stl"/>'
                '<mesh file="z.obj"/>'
            )
        else:
            destination.write_bytes(b"downloaded")

    monkeypatch.setattr(assets, "_download", fake_download)

    result = assets.ensure_menagerie_assets("example_robot", "robot.xml", root)

    raw_root = (
        f"{assets.RAW_ROOT}/{assets.MENAGERIE_REVISION}/example_robot"
    )
    assert result == root
    assert downloads == [
        (f"{raw_root}/robot.xml", root / "robot.xml"),
        (f"{raw_root}/assets/nested/part.stl", root / "assets/nested/part.stl"),
        (f"{raw_root}/assets/z.obj", root / "assets/z.obj"),
        (f"{raw_root}/LICENSE", root / "LICENSE"),
    ]
    assert (root / "assets" / "nested").is_dir()

    downloads.clear()
    assert assets.ensure_menagerie_assets("example_robot", "robot.xml", root) == root
    assert downloads == []


def test_ensure_menagerie_assets_uses_revision_scoped_default_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destinations: list[Path] = []

    def fake_download(_: str, destination: Path) -> None:
        destinations.append(destination)
        if destination.suffix == ".xml":
            destination.write_text("")
        else:
            destination.write_bytes(b"")

    monkeypatch.setattr(assets.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(assets, "_download", fake_download)

    result = assets.ensure_menagerie_assets("robot", "robot.xml")

    expected = (
        tmp_path
        / ".cache"
        / "embodi"
        / "mujoco_menagerie"
        / assets.MENAGERIE_REVISION
        / "robot"
    )
    assert result == expected
    assert destinations == [expected / "robot.xml", expected / "LICENSE"]


@pytest.mark.parametrize(
    ("wrapper", "directory", "filename"),
    [
        (assets.ensure_so101_assets, "robotstudio_so101", "so101.xml"),
        (assets.ensure_panda_assets, "franka_emika_panda", "panda.xml"),
    ],
)
def test_model_wrappers_delegate_with_expected_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wrapper,
    directory: str,
    filename: str,
) -> None:
    calls: list[tuple[str, str, Path | None]] = []
    monkeypatch.setattr(
        assets,
        "ensure_menagerie_assets",
        lambda *args: calls.append(args) or tmp_path,
    )

    assert wrapper(tmp_path) == tmp_path
    assert calls == [(directory, filename, tmp_path)]
