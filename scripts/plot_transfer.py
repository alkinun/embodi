#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot controlled SO-101 transfer curves.")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    steps = np.asarray(data["steps"])
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, color in (("baseline", "#8b8f97"), ("generalist", "#d04a35")):
        values = np.asarray(list(data[name].values()))
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axis.plot(steps, mean, marker="o", linewidth=2.5, color=color, label=name)
        axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)
    axis.axhline(0.2, color="black", linestyle="--", linewidth=1, label="loss 0.20")
    axis.set_xlabel("SO-101 optimization steps")
    axis.set_ylabel("held-out canonical flow loss")
    axis.set_title("Xperience pretraining accelerates canonical SO-101 learning")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
