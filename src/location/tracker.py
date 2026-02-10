"""Main tracking loop — polls cameras, runs detection, updates world model.

Runs as a background asyncio task inside the location server.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np

try:
    import httpx
except ImportError:
    httpx = None

from .detector import UnifiedDetector, DetectionResult
from .world_model import LocationWorldModel
from .reachability import ARM_MAX_REACH_MM

logger = logging.getLogger("th3cl4w.location.tracker")

# Camera server base URL
CAMERA_SERVER = "http://localhost:8081"

# Polling intervals (seconds)
FAST_SCAN_INTERVAL = 1.0       # HSV detection every 1s
DEEP_SCAN_INTERVAL = 30.0      # LLM scan every 30s
VERIFY_INTERVAL = 5.0          # Re-verify known objects every 5s
STALE_SWEEP_INTERVAL = 10.0    # Sweep stale objects every 10s

# Camera IDs
CAMERA_IDS = [0, 1, 2]  # overhead, side, arm

# Rough pixel-to-mm scale for overhead camera (cam 0)
# Assumes ~800mm workspace across 1920px
OVERHEAD_MM_PER_PX = 800.0 / 1920.0
# For side camera height estimation
SIDE_MM_PER_PX = 600.0 / 1080.0


def _pixel_to_position_overhead(
    cx: int, cy: int, frame_w: int = 1920, frame_h: int = 1080
) -> np.ndarray:
    """Convert overhead camera pixel to approximate XY position in mm.

    Very rough: assumes camera centered over arm base, workspace ~800mm across.
    Real systems would use calibrated extrinsics.
    """
    # Center of image ≈ arm base
    x_mm = (cx - frame_w / 2) * OVERHEAD_MM_PER_PX
    y_mm = (cy - frame_h / 2) * OVERHEAD_MM_PER_PX
    return np.array([x_mm, y_mm, 0.0])


def _pixel_to_position_side(
    cx: int, cy: int, frame_w: int = 1920, frame_h: int = 1080
) -> np.ndarray:
    """Convert side camera pixel to approximate XZ position."""
    x_mm = (cx - frame_w / 2) * SIDE_MM_PER_PX
    z_mm = (frame_h - cy) * SIDE_MM_PER_PX  # y pixel → height
    return np.array([x_mm, 0.0, z_mm])


class ObjectTracker:
    """Continuously tracks objects via camera polling and detection.

    Lifecycle:
    1. start() — launches background tasks
    2. Tasks poll cameras, run detection, update world model
    3. stop() — cancels everything
    """

    def __init__(self, world_model: LocationWorldModel):
        self.world_model = world_model
        self.detector = UnifiedDetector()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._camera_status: dict[int, bool] = {i: False for i in CAMERA_IDS}
        self._last_deep_scan = 0.0
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def camera_status(self) -> dict[int, bool]:
        return dict(self._camera_status)

    async def start(self):
        """Start tracking tasks."""
        if self._running:
            return
        self._running = True
        self._http_client = httpx.AsyncClient(timeout=5.0)
        self._tasks = [
            asyncio.create_task(self._fast_scan_loop(), name="fast-scan"),
            asyncio.create_task(self._deep_scan_loop(), name="deep-scan"),
            asyncio.create_task(self._stale_sweep_loop(), name="stale-sweep"),
        ]
        logger.info("Object tracker started")

    async def stop(self):
        """Stop all tracking tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("Object tracker stopped")

    async def trigger_scan(self):
        """Trigger an immediate full scan (all cameras, CV + LLM)."""
        logger.info("Triggered immediate full scan")
        for cam_id in CAMERA_IDS:
            frame, jpeg = await self._grab_frame(cam_id)
            if frame is not None:
                self._process_cv_detections(frame, cam_id)
                if jpeg is not None:
                    await self._process_llm_detections(jpeg, cam_id)
        self.world_model.mark_scan()

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _fast_scan_loop(self):
        """Poll cameras and run fast HSV detection."""
        while self._running:
            try:
                for cam_id in CAMERA_IDS:
                    frame, _ = await self._grab_frame(cam_id)
                    if frame is not None:
                        self._camera_status[cam_id] = True
                        self._process_cv_detections(frame, cam_id)
                    else:
                        self._camera_status[cam_id] = False

                self.world_model.mark_scan()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Fast scan error")

            await asyncio.sleep(FAST_SCAN_INTERVAL)

    async def _deep_scan_loop(self):
        """Periodically run LLM detection for comprehensive object discovery."""
        # Wait a bit before first deep scan
        await asyncio.sleep(5.0)

        while self._running:
            try:
                for cam_id in [0, 1]:  # overhead and side only
                    _, jpeg = await self._grab_frame(cam_id)
                    if jpeg is not None:
                        await self._process_llm_detections(jpeg, cam_id)
                self._last_deep_scan = time.time()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Deep scan error")

            await asyncio.sleep(DEEP_SCAN_INTERVAL)

    async def _stale_sweep_loop(self):
        """Periodically sweep stale objects."""
        while self._running:
            try:
                self.world_model.sweep_stale()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Stale sweep error")
            await asyncio.sleep(STALE_SWEEP_INTERVAL)

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    async def _grab_frame(self, cam_id: int) -> tuple[Optional[np.ndarray], Optional[bytes]]:
        """Grab a frame from the camera server. Returns (decoded_frame, raw_jpeg)."""
        if self._http_client is None:
            return None, None

        try:
            resp = await self._http_client.get(f"{CAMERA_SERVER}/snap/{cam_id}")
            if resp.status_code != 200:
                return None, None

            jpeg_bytes = resp.content
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame, jpeg_bytes

        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Detection processing
    # ------------------------------------------------------------------

    def _process_cv_detections(self, frame: np.ndarray, cam_id: int):
        """Run fast HSV + CV detection and update world model."""
        # Fast HSV
        results = self.detector.detect_fast(frame, cam_id)
        # Also run multi-color CV
        results.extend(self.detector.detect_cv(frame, cam_id))

        # Deduplicate by label (keep highest confidence)
        best: dict[str, DetectionResult] = {}
        for r in results:
            key = r.label
            if key not in best or r.confidence > best[key].confidence:
                best[key] = r

        h, w = frame.shape[:2]
        for det in best.values():
            position = self._detection_to_position(det, cam_id, w, h)
            dims = self._estimate_dimensions(det, cam_id, w, h)
            graspable = min(dims[0], dims[2]) <= 65.0 and min(dims[0], dims[2]) > 3.0

            self.world_model.upsert(
                label=det.label,
                position_mm=position,
                dimensions_mm=dims,
                confidence=det.confidence,
                source=det.source,
                bbox_px=det.bbox,
                camera_id=cam_id,
                graspable=graspable,
            )

    async def _process_llm_detections(self, jpeg_bytes: bytes, cam_id: int):
        """Run LLM detection and update world model."""
        results = await self.detector.detect_llm(jpeg_bytes, cam_id)

        for det in results:
            position = self._detection_to_position(det, cam_id, 1920, 1080)
            dims = self._estimate_dimensions(det, cam_id, 1920, 1080)
            graspable = min(dims[0], dims[2]) <= 65.0 and min(dims[0], dims[2]) > 3.0

            self.world_model.upsert(
                label=det.label,
                position_mm=position,
                dimensions_mm=dims,
                confidence=det.confidence,
                source="llm",
                bbox_px=det.bbox,
                camera_id=cam_id,
                graspable=graspable,
            )

    def _detection_to_position(
        self, det: DetectionResult, cam_id: int, frame_w: int, frame_h: int
    ) -> np.ndarray:
        """Convert a detection centroid to 3D position based on camera."""
        cx, cy = det.centroid_px
        if cam_id == 0:
            # Overhead camera → XY plane
            return _pixel_to_position_overhead(cx, cy, frame_w, frame_h)
        elif cam_id == 1:
            # Side camera → XZ plane
            return _pixel_to_position_side(cx, cy, frame_w, frame_h)
        else:
            # Arm camera — rough forward estimate
            return np.array([250.0, 0.0, 0.0])

    def _estimate_dimensions(
        self, det: DetectionResult, cam_id: int, frame_w: int, frame_h: int
    ) -> np.ndarray:
        """Estimate object dimensions from bounding box."""
        _, _, bw, bh = det.bbox
        if cam_id == 0:
            w_mm = bw * OVERHEAD_MM_PER_PX
            d_mm = bh * OVERHEAD_MM_PER_PX
            h_mm = 50.0  # default height, unknown from overhead
            return np.array([w_mm, h_mm, d_mm])
        elif cam_id == 1:
            w_mm = bw * SIDE_MM_PER_PX
            h_mm = bh * SIDE_MM_PER_PX
            d_mm = w_mm  # assume symmetric
            return np.array([w_mm, h_mm, d_mm])
        else:
            return np.array([30.0, 50.0, 30.0])
