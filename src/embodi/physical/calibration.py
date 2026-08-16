from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from .so101 import SO101CanonicalCalibration


def initialize_calibration(lerobot_calibration: Path, output: Path) -> dict:
    if not lerobot_calibration.is_file():
        raise FileNotFoundError(f"LeRobot calibration does not exist: {lerobot_calibration}")
    calibration = SO101CanonicalCalibration(
        source_calibration_sha256=hashlib.sha256(lerobot_calibration.read_bytes()).hexdigest(),
        physical_validation_complete=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        calibration.save(temporary)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"calibration output already exists: {output}") from error
    finally:
        active_error = sys.exception()
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if active_error is None:
                raise
            active_error.add_note(f"calibration temporary-file cleanup failed: {cleanup_error!r}")
    return calibration.provenance(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize an unvalidated SO101 canonical calibration from LeRobot provenance."
    )
    parser.add_argument("--lerobot-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(initialize_calibration(args.lerobot_calibration, args.output), indent=2))


if __name__ == "__main__":
    main()
