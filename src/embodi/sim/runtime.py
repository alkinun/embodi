from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from io import BytesIO
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..checkpoints import load_inference_policy
from ..processing import EmbodiProcessor, camera_frame_to_tensor
from .environment import SO101PickPlaceEnv, TASK


@dataclass(frozen=True)
class Observation:
    generation: int
    state: np.ndarray
    canonical_state: np.ndarray
    top: np.ndarray
    wrist: np.ndarray


class ChunkExecutor:
    def __init__(self, horizon: int, prefetch_steps: int = 2) -> None:
        if horizon <= 0:
            raise ValueError("execution horizon must be positive")
        self.horizon = horizon
        self.prefetch_steps = min(prefetch_steps, max(horizon - 1, 0))
        self.active: deque[np.ndarray] = deque()
        self.upcoming: deque[np.ndarray] | None = None
        self.request_pending = False

    def clear(self) -> None:
        self.active.clear()
        self.upcoming = None
        self.request_pending = False

    def mark_requested(self) -> None:
        self.request_pending = True

    def accept(self, chunk: np.ndarray) -> None:
        candidate = deque(chunk[: self.horizon])
        self.request_pending = False
        if self.active:
            self.upcoming = candidate
        else:
            self.active = candidate

    def pop(self) -> np.ndarray | None:
        if not self.active and self.upcoming is not None:
            self.active = self.upcoming
            self.upcoming = None
        return self.active.popleft() if self.active else None

    def needs_request(self) -> bool:
        return (
            not self.request_pending
            and self.upcoming is None
            and len(self.active) <= self.prefetch_steps
        )


class SimulationRuntime:
    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        frequency: float = 30.0,
        max_episode_seconds: float = 30.0,
        jpeg_quality: int = 80,
        asset_dir: Path | None = None,
        load_policy: bool = True,
        execution_horizon: int = 32,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.frequency = frequency
        self.max_episode_seconds = max_episode_seconds
        self.jpeg_quality = jpeg_quality
        self.asset_dir = asset_dir
        self.load_policy = load_policy
        self.execution_horizon = execution_horizon
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.reset_requests: Queue[threading.Event] = Queue(maxsize=4)
        self.observations: Queue[Observation] = Queue(maxsize=1)
        self.action_chunks: Queue[tuple[int, np.ndarray, float]] = Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest_jpeg = b""
        self.frame_sequence = 0
        self.status_data: dict[str, Any] = {"ready": False}
        self.error: BaseException | None = None
        self.sim_thread: threading.Thread | None = None
        self.policy_thread: threading.Thread | None = None

    def start(self, timeout: float = 180.0) -> None:
        self.sim_thread = threading.Thread(target=self._simulation_loop, name="embodi-sim", daemon=True)
        self.sim_thread.start()
        if self.load_policy:
            self.policy_thread = threading.Thread(target=self._policy_loop, name="embodi-policy", daemon=True)
            self.policy_thread.start()
        if not self.ready_event.wait(timeout):
            raise TimeoutError("simulation did not become ready")
        if self.error is not None:
            raise RuntimeError("simulation startup failed") from self.error

    def stop(self) -> None:
        self.stop_event.set()
        for thread in (self.policy_thread, self.sim_thread):
            if thread is not None:
                thread.join(timeout=10)

    def request_reset(self, timeout: float = 5.0) -> dict[str, Any]:
        acknowledgement = threading.Event()
        try:
            self.reset_requests.put_nowait(acknowledgement)
        except Full as error:
            raise RuntimeError("reset queue is full") from error
        if not acknowledgement.wait(timeout):
            raise TimeoutError("simulation reset timed out")
        return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.status_data)

    def frame(self) -> tuple[int, bytes]:
        with self.lock:
            return self.frame_sequence, self.latest_jpeg

    @staticmethod
    def _replace(queue: Queue, value: Any) -> None:
        try:
            queue.put_nowait(value)
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass
            queue.put_nowait(value)

    def _publish_frame(self, top: np.ndarray, wrist: np.ndarray, env: SO101PickPlaceEnv, inference_ms: float) -> None:
        top_image = Image.fromarray(top)
        wrist_image = Image.fromarray(wrist).resize((top_image.width // 3, top_image.height // 3))
        top_image.paste(wrist_image, (top_image.width - wrist_image.width - 12, 12))
        draw = ImageDraw.Draw(top_image)
        draw.rectangle((0, 0, 300, 54), fill=(12, 15, 20))
        draw.text((10, 8), f"episode {env.episode}  sim {env.data.time:.1f}s", fill="white")
        state = "success" if env.success else ("lifted" if env.lifted else "running")
        draw.text((10, 30), f"{state}  inference {inference_ms:.0f}ms", fill=(120, 240, 170))
        output = BytesIO()
        top_image.save(output, format="JPEG", quality=self.jpeg_quality)
        with self.lock:
            self.latest_jpeg = output.getvalue()
            self.frame_sequence += 1
            self.status_data = {
                "ready": True,
                "episode": env.episode,
                "sim_time": round(float(env.data.time), 3),
                "lifted": bool(env.lifted),
                "success": bool(env.success),
                "inference_ms": round(inference_ms, 1),
                "frame_sequence": self.frame_sequence,
                "policy": self.load_policy,
            }

    def _simulation_loop(self) -> None:
        env = None
        try:
            env = SO101PickPlaceEnv(asset_dir=self.asset_dir)
            generation = 0
            executor = ChunkExecutor(self.execution_horizon)
            inference_ms = 0.0
            top, wrist = env.render_cameras()
            self._publish_frame(top, wrist, env, inference_ms)
            if self.load_policy:
                self._replace(self.observations, Observation(generation, env.state(), env.canonical_state(), top, wrist))
                executor.mark_requested()
            self.ready_event.set()
            next_tick = time.monotonic()
            while not self.stop_event.is_set():
                reset = False
                while True:
                    try:
                        acknowledgement = self.reset_requests.get_nowait()
                    except Empty:
                        break
                    env.reset()
                    generation += 1
                    executor.clear()
                    while not self.action_chunks.empty():
                        self.action_chunks.get_nowait()
                    top, wrist = env.render_cameras()
                    self._publish_frame(top, wrist, env, inference_ms)
                    if self.load_policy:
                        self._replace(self.observations, Observation(generation, env.state(), env.canonical_state(), top, wrist))
                        executor.mark_requested()
                    acknowledgement.set()
                    reset = True
                try:
                    result_generation, chunk, inference_ms = self.action_chunks.get_nowait()
                    if result_generation == generation:
                        executor.accept(chunk)
                except Empty:
                    pass
                action = executor.pop()
                if action is not None:
                    env.apply_action(action)
                env.step_control_period(self.frequency)
                top, wrist = env.render_cameras()
                self._publish_frame(top, wrist, env, inference_ms)
                if self.load_policy and executor.needs_request():
                    self._replace(self.observations, Observation(generation, env.state(), env.canonical_state(), top, wrist))
                    executor.mark_requested()
                if env.data.time >= self.max_episode_seconds and not reset:
                    env.reset()
                    generation += 1
                    executor.clear()
                    if self.load_policy:
                        top, wrist = env.render_cameras()
                        self._replace(self.observations, Observation(generation, env.state(), env.canonical_state(), top, wrist))
                        executor.mark_requested()
                next_tick += 1.0 / self.frequency
                self.stop_event.wait(max(0.0, next_tick - time.monotonic()))
        except BaseException as error:
            self.error = error
            self.ready_event.set()
        finally:
            if env is not None:
                env.close()

    def _policy_loop(self) -> None:
        try:
            policy = load_inference_policy(self.checkpoint, self.device)
            processor = EmbodiProcessor.from_pretrained(
                policy.config.backbone_name,
                revision=policy.config.backbone_revision,
                do_rescale=policy.config.image_do_rescale,
            )
            while not self.stop_event.is_set():
                try:
                    observation = self.observations.get(timeout=0.2)
                except Empty:
                    continue
                policy.reset()
                state = torch.from_numpy(observation.state).unsqueeze(0)
                camera_images = {"top": observation.top, "wrist": observation.wrist}
                images = [
                    camera_frame_to_tensor(
                        camera_images[name].copy(),
                        processor_rescales=policy.config.image_do_rescale,
                    )
                    for name in policy.config.camera_names
                ]
                if getattr(policy.config, "format_version", None) == 3:
                    canonical_state = torch.from_numpy(observation.canonical_state).unsqueeze(0)
                    batch = processor(
                        [images], [TASK], state,
                        canonical_state=canonical_state,
                        canonical_part_mask=torch.ones(1, canonical_state.shape[1], dtype=torch.bool),
                    )
                elif getattr(policy.config, "format_version", None) == 2:
                    canonical_state = torch.zeros(1, 4, 10)
                    canonical_state[:, 0] = torch.from_numpy(observation.canonical_state[0])
                    batch = processor(
                        [images], [TASK], state,
                        canonical_state=canonical_state,
                        canonical_slot_mask=torch.tensor([[True, False, False, False]]),
                    )
                else:
                    batch = processor([images], [TASK], state)
                batch = {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                started = time.perf_counter()
                chunk = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._replace(self.action_chunks, (observation.generation, chunk, elapsed_ms))
        except BaseException as error:
            self.error = error
            self.stop_event.set()
