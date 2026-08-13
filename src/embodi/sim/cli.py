from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run the embodi mujoco evaluation server")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/embodi00-so100-pickplace/final"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--execution-horizon", type=int, default=32)
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.smoke_test:
        from .environment import SO101PickPlaceEnv

        env = SO101PickPlaceEnv(width=320, height=240, asset_dir=args.asset_dir)
        try:
            top, wrist = env.render_cameras()
            for _ in range(3):
                env.step_control_period(args.frequency)
            print(f"state={env.state().shape} top={top.shape} wrist={wrist.shape}")
        finally:
            env.close()
        return

    import uvicorn

    from .runtime import SimulationRuntime
    from .server import create_app

    runtime = SimulationRuntime(
        checkpoint=args.checkpoint,
        device=args.device,
        frequency=args.frequency,
        max_episode_seconds=args.episode_seconds,
        asset_dir=args.asset_dir,
        load_policy=not args.no_policy,
        execution_horizon=args.execution_horizon,
    )
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
