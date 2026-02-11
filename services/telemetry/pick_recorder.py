"""Lightweight video recorder — captures frames from all cameras during pick attempts."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import httpx

from shared.config.camera_config import latest_url, CAM_OVERHEAD, CAM_ARM, CAM_SIDE

logger = logging.getLogger("th3cl4w.telemetry.pick_recorder")

RECORDINGS_DIR = Path("data/pick_recordings")
ALL_CAMS = [CAM_OVERHEAD, CAM_ARM, CAM_SIDE]
FRAME_INTERVAL = 0.2  # ~5fps


class PickVideoRecorder:
    """Grabs JPEG frames from camera server at ~5fps during pick attempts."""

    def __init__(self, recording_dir: Path = RECORDINGS_DIR):
        self._recording_dir = recording_dir
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._frame_count = 0
        self._episode_id: str | None = None

    @property
    def recording(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self, episode_id: str) -> None:
        """Begin recording frames for the given episode."""
        if self.recording:
            logger.warning("Already recording, stopping previous")
            self._stop_event.set()
        self._episode_id = episode_id
        self._frame_count = 0
        self._stop_event = asyncio.Event()
        ep_dir = self._recording_dir / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._record_loop(episode_id, ep_dir))
        logger.info("Pick recording started: %s", episode_id)

    async def stop(self) -> int:
        """Stop recording. Returns total frame count."""
        if self._task is None:
            return 0
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=3.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None
        count = self._frame_count
        logger.info("Pick recording stopped: %s (%d frames)", self._episode_id, count)
        return count

    async def _record_loop(self, episode_id: str, ep_dir: Path) -> None:
        async with httpx.AsyncClient(timeout=2.0) as client:
            while not self._stop_event.is_set():
                ts = int(time.time() * 1000)
                for cam_id in ALL_CAMS:
                    try:
                        resp = await client.get(latest_url(cam_id))
                        if resp.status_code == 200:
                            fname = f"cam{cam_id}_{ts}.jpg"
                            (ep_dir / fname).write_bytes(resp.content)
                            self._frame_count += 1
                    except Exception:
                        pass  # lightweight — don't fail the pick
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=FRAME_INTERVAL
                    )
                    break  # stop event was set
                except asyncio.TimeoutError:
                    pass  # continue recording
