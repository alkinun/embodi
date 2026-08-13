from __future__ import annotations

import os
from pathlib import Path
import re
from urllib.request import urlopen


MENAGERIE_REVISION = "c1a4eeb85694ae1dffe33ff1797d4e528928a133"
RAW_ROOT = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{MENAGERIE_REVISION}/robotstudio_so101"
)


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    os.replace(temporary, destination)


def ensure_so101_assets(cache_dir: Path | None = None) -> Path:
    root = cache_dir or (
        Path.home() / ".cache" / "embodi" / "mujoco_menagerie" / MENAGERIE_REVISION / "robotstudio_so101"
    )
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    model_path = root / "so101.xml"
    if not model_path.exists():
        _download(f"{RAW_ROOT}/so101.xml", model_path)
    model_xml = model_path.read_text()
    mesh_files = sorted(set(re.findall(r'<mesh[^>]+file="([^"]+\.stl)"', model_xml)))
    for filename in mesh_files:
        destination = assets / filename
        if not destination.exists():
            _download(f"{RAW_ROOT}/assets/{filename}", destination)
    license_path = root / "LICENSE"
    if not license_path.exists():
        _download(f"{RAW_ROOT}/LICENSE", license_path)
    return root
