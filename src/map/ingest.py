"""Async data ingest layer for the map server.

Polls main server (joint state), camera server (depth frames),
and location server (objects). Feeds data into the scene graph.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class IngestConfig:
    def __init__(
        self,
        main_server_url: str = "http://localhost:8080",
        camera_server_url: str = "http://localhost:8081",
        location_server_url: str = "http://localhost:8082",
        arm_poll_hz: float = 2,
        depth_poll_hz: float = 3,
        location_poll_hz: float = 5,
        camera_id: int = 0,
        use_ws: bool = True,
    ):
        self.main_server_url = main_server_url
        self.camera_server_url = camera_server_url
        self.location_server_url = location_server_url
        self.arm_poll_hz = arm_poll_hz
        self.depth_poll_hz = depth_poll_hz
        self.location_poll_hz = location_poll_hz
        self.camera_id = camera_id
        self.use_ws = use_ws


class IngestStats:
    def __init__(self):
        self.arm_polls: int = 0
        self.arm_errors: int = 0
        self.arm_last: float = 0
        self.depth_polls: int = 0
        self.depth_errors: int = 0
        self.depth_last: float = 0
        self.location_polls: int = 0
        self.location_errors: int = 0
        self.location_last: float = 0
        self.location_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": {
                "polls": self.arm_polls,
                "errors": self.arm_errors,
                "last": self.arm_last,
                "rate_hz": self.arm_polls / max(time.time() - self.arm_last, 1) if self.arm_last else 0,
            },
            "depth": {
                "polls": self.depth_polls,
                "errors": self.depth_errors,
                "last": self.depth_last,
            },
            "location": {
                "polls": self.location_polls,
                "errors": self.location_errors,
                "last": self.location_last,
                "available": self.location_available,
            },
        }


class DataIngest:
    """Async data ingest from external servers."""

    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()
        self.stats = IngestStats()
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Callbacks — set by map_server to wire into scene/arm_model/env_map
        self.on_arm_state = None  # async callable(joints, gripper_mm)
        self.on_depth_frame = None  # async callable(frame_bgr, joints)
        self.on_objects = None  # async callable(objects_list)

        # Cached state
        self._last_joints: List[float] = [0.0] * 7
        self._last_gripper: float = 0.0

    async def start(self) -> None:
        """Start all ingest loops."""
        if self._running:
            return
        self._running = True
        if self.config.use_ws:
            arm_task = asyncio.create_task(self._arm_ws_loop())
        else:
            arm_task = asyncio.create_task(self._arm_loop())
        self._tasks = [
            arm_task,
            asyncio.create_task(self._depth_loop()),
            asyncio.create_task(self._location_loop()),
        ]
        logger.info("Data ingest started (arm via %s)", "WebSocket" if self.config.use_ws else "polling")

    async def stop(self) -> None:
        """Stop all ingest loops."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        logger.info("Data ingest stopped")

    async def _arm_ws_loop(self) -> None:
        """Subscribe to main server /ws/state WebSocket for arm state.

        Falls back to polling if WebSocket connection fails.
        """
        import json
        try:
            import websockets
        except ImportError:
            logger.warning("websockets package not installed, falling back to polling")
            return await self._arm_loop()

        ws_url = self.config.main_server_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url += "/ws/state"
        retry_delay = 1.0

        while self._running:
            try:
                async with websockets.connect(ws_url, close_timeout=2) as ws:
                    logger.info("Connected to main server WebSocket at %s", ws_url)
                    retry_delay = 1.0
                    async for message in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(message)
                            joints = data.get("joints", data.get("joint_angles", [0] * 7))
                            gripper = data.get("gripper_mm", data.get("gripper", 0.0))
                            self._last_joints = joints
                            self._last_gripper = gripper
                            self.stats.arm_polls += 1
                            self.stats.arm_last = time.time()

                            if self.on_arm_state:
                                await self.on_arm_state(joints, gripper)
                        except (json.JSONDecodeError, KeyError) as e:
                            self.stats.arm_errors += 1
                            logger.debug("WS parse error: %s", e)
            except Exception as e:
                self.stats.arm_errors += 1
                if self.stats.arm_errors <= 3 or self.stats.arm_errors % 100 == 0:
                    logger.debug("Arm WS error (retry in %.0fs): %s", retry_delay, e)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 30.0)

    async def _arm_loop(self) -> None:
        """Poll main server for joint state (fallback when WS unavailable)."""
        import httpx

        interval = 1.0 / self.config.arm_poll_hz
        async with httpx.AsyncClient(timeout=2.0) as client:
            while self._running:
                try:
                    resp = await client.get(f"{self.config.main_server_url}/api/state")
                    if resp.status_code == 200:
                        data = resp.json()
                        joints = data.get("joints", data.get("joint_angles", [0] * 7))
                        gripper = data.get("gripper_mm", data.get("gripper", 0.0))
                        self._last_joints = joints
                        self._last_gripper = gripper
                        self.stats.arm_polls += 1
                        self.stats.arm_last = time.time()

                        if self.on_arm_state:
                            await self.on_arm_state(joints, gripper)
                    else:
                        self.stats.arm_errors += 1
                except Exception as e:
                    self.stats.arm_errors += 1
                    if self.stats.arm_errors <= 3 or self.stats.arm_errors % 100 == 0:
                        logger.debug("Arm poll error: %s", e)

                await asyncio.sleep(interval)

    async def _depth_loop(self) -> None:
        """Poll camera server for depth frames."""
        import httpx
        import cv2

        interval = 1.0 / self.config.depth_poll_hz
        # Wait a bit for other services to start
        await asyncio.sleep(2.0)

        async with httpx.AsyncClient(timeout=5.0) as client:
            while self._running:
                try:
                    resp = await client.get(
                        f"{self.config.camera_server_url}/snap/{self.config.camera_id}"
                    )
                    if resp.status_code == 200:
                        img_array = np.frombuffer(resp.content, dtype=np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.stats.depth_polls += 1
                            self.stats.depth_last = time.time()

                            if self.on_depth_frame:
                                await self.on_depth_frame(frame, self._last_joints)
                    else:
                        self.stats.depth_errors += 1
                except Exception as e:
                    self.stats.depth_errors += 1
                    if self.stats.depth_errors <= 3 or self.stats.depth_errors % 100 == 0:
                        logger.debug("Depth poll error: %s", e)

                await asyncio.sleep(interval)

    async def _location_loop(self) -> None:
        """Poll location server for object list.

        Gracefully handles location server not being available (retry/skip).
        """
        import httpx

        interval = 1.0 / self.config.location_poll_hz
        # Give location server extra time to start
        await asyncio.sleep(5.0)
        retry_backoff = 5.0  # Start with 5s between retries when unavailable
        max_backoff = 60.0

        async with httpx.AsyncClient(timeout=3.0) as client:
            while self._running:
                try:
                    resp = await client.get(
                        f"{self.config.location_server_url}/api/objects"
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        objects = data if isinstance(data, list) else data.get("objects", [])
                        self.stats.location_polls += 1
                        self.stats.location_last = time.time()
                        self.stats.location_available = True
                        retry_backoff = 5.0  # Reset backoff on success

                        if self.on_objects:
                            await self.on_objects(objects)
                    else:
                        self.stats.location_errors += 1
                except Exception as e:
                    self.stats.location_errors += 1
                    if not self.stats.location_available:
                        # Location server not yet available — back off
                        if self.stats.location_errors <= 3:
                            logger.info(
                                "Location server not available at %s (will retry)",
                                self.config.location_server_url,
                            )
                        await asyncio.sleep(retry_backoff)
                        retry_backoff = min(retry_backoff * 1.5, max_backoff)
                        continue
                    else:
                        if self.stats.location_errors % 50 == 0:
                            logger.warning("Location poll error: %s", e)

                await asyncio.sleep(interval)
