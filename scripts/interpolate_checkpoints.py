#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from embodi.checkpoints import interpolate_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Linearly interpolate two compatible checkpoints.")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    interpolate_checkpoints(args.first, args.second, args.output, args.alpha)
    print(f"interpolated={args.output} alpha={args.alpha}")


if __name__ == "__main__":
    main()
