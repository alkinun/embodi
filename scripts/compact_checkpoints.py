#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove optimizer state from completed inference checkpoints.")
    parser.add_argument("root", type=Path, nargs="+")
    args = parser.parse_args()
    for root in args.root:
        for path in root.rglob("trainer.pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as error:
                print(f"skipped={path} error={error}")
                continue
            compact = {
                key: payload[key]
                for key in ("format_version", "step", "stage")
                if key in payload
            }
            temporary = path.with_suffix(".compact.pt")
            torch.save(compact, temporary)
            temporary.replace(path)
            print(f"compacted={path}")


if __name__ == "__main__":
    main()
