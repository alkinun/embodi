from __future__ import annotations

import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


MENAGERIE_REVISION = "c1a4eeb85694ae1dffe33ff1797d4e528928a133"
RAW_ROOT = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"


def _download(url: str, destination: Path) -> None:
    temporary_file = NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".part",
        delete=False,
    )
    temporary = Path(temporary_file.name)
    try:
        with temporary_file as output, urlopen(url, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _mesh_files(model_xml: str) -> tuple[str, ...]:
    return tuple(
        sorted(set(re.findall(r'<mesh[^>]+file="([^"]+\.(?:obj|stl))"', model_xml)))
    )


def ensure_menagerie_assets(
    model_directory: str,
    model_filename: str,
    cache_dir: Path | None = None,
) -> Path:
    if not model_directory or Path(model_directory).name != model_directory:
        raise ValueError("model_directory must be one Menagerie directory name")
    if (
        not model_filename
        or Path(model_filename).name != model_filename
        or not model_filename.endswith(".xml")
    ):
        raise ValueError("model_filename must be one XML filename")
    root = cache_dir or (
        Path.home() / ".cache" / "embodi" / "mujoco_menagerie" / MENAGERIE_REVISION / model_directory
    )
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    raw_root = f"{RAW_ROOT}/{MENAGERIE_REVISION}/{model_directory}"
    model_path = root / model_filename
    if not model_path.exists():
        _download(f"{raw_root}/{model_filename}", model_path)
    model_xml = model_path.read_text()
    for filename in _mesh_files(model_xml):
        destination = assets / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            _download(f"{raw_root}/assets/{filename}", destination)
    license_path = root / "LICENSE"
    if not license_path.exists():
        _download(f"{raw_root}/LICENSE", license_path)
    return root


def ensure_so101_assets(cache_dir: Path | None = None) -> Path:
    return ensure_menagerie_assets("robotstudio_so101", "so101.xml", cache_dir)


def ensure_panda_assets(cache_dir: Path | None = None) -> Path:
    return ensure_menagerie_assets("franka_emika_panda", "panda.xml", cache_dir)
