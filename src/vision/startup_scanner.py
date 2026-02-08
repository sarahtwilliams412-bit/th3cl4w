"""
Startup Scanner — Immediate environment assessment when cameras come online.

When the camera server starts, this module:
1. Waits for both cameras to deliver their first valid frames
2. Captures a rapid burst of frames from each camera
3. Feeds every frame through the dimension estimator pipeline
4. Aggressively grades and re-assesses each estimate
5. Builds an initial world model from the converged estimates
6. Reports what's in the area, what's reachable, and where obstacles are

Design goals:
- Operational world model within 2-3 frames per camera (~0.5s at 15fps)
- Aggressive grading so the arm never acts on bad data
- Continuous refinement: keeps scanning in background after initial build
- Thread-safe: runs alongside camera capture and HTTP serving
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol

import numpy as np

from .dimension_estimator import ObjectDimensionEstimator, DimensionEstimate
from .world_model import WorldModel, WorldModelSnapshot, WorldObject
from .scene_analyzer import SceneAnalyzer, SceneDescription

logger = logging.getLogger("th3cl4w.vision.startup_scanner")


class FrameProvider(Protocol):
    """Protocol for anything that can provide raw camera frames."""

    def get_raw_frame(self) -> Optional[np.ndarray]: ...

    @property
    def connected(self) -> bool: ...


class ScanPhase(Enum):
    """Phases of the startup scanning process."""

    WAITING_FOR_CAMERAS = "waiting_for_cameras"
    INITIAL_CAPTURE = "initial_capture"
    DIMENSION_ANALYSIS = "dimension_analysis"
    GRADING = "grading"
    WORLD_MODEL_BUILD = "world_model_build"
    CONTINUOUS_REFINEMENT = "continuous_refinement"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ScanResult:
    """Result of a single scan pass (one pair of frames)."""

    scan_index: int
    estimates: list[DimensionEstimate]
    scene: Optional[SceneDescription]
    elapsed_ms: float
    phase: ScanPhase


@dataclass
class StartupScanReport:
    """Complete report from the startup scanning process."""

    phase: ScanPhase
    scans_completed: int
    total_objects_detected: int
    world_model: Optional[WorldModelSnapshot]
    scan_results: list[ScanResult]
    elapsed_total_ms: float
    message: str

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "scans_completed": self.scans_completed,
            "total_objects_detected": self.total_objects_detected,
            "world_model": self.world_model.to_dict() if self.world_model else None,
            "scan_results": [
                {
                    "scan_index": r.scan_index,
                    "estimates": [e.to_dict() for e in r.estimates],
                    "elapsed_ms": round(r.elapsed_ms, 1),
                    "phase": r.phase.value,
                }
                for r in self.scan_results
            ],
            "elapsed_total_ms": round(self.elapsed_total_ms, 1),
            "message": self.message,
        }


class StartupScanner:
    """Orchestrates immediate environment scanning when cameras start.

    Usage:
        scanner = StartupScanner(cam0_thread, cam1_thread)
        scanner.start()  # non-blocking, runs in background

        # Later:
        report = scanner.get_report()
        model = scanner.get_world_model()

        # Or register a callback:
        scanner.on_model_ready(my_callback)

    The scanner proceeds through phases:
    1. WAITING_FOR_CAMERAS: polls until both cameras have valid frames
    2. INITIAL_CAPTURE: takes a burst of frames rapidly
    3. DIMENSION_ANALYSIS: runs dimension estimation on each frame pair
    4. GRADING: aggressively grades and re-assesses all estimates
    5. WORLD_MODEL_BUILD: constructs the spatial world model
    6. CONTINUOUS_REFINEMENT: keeps scanning at reduced rate to improve model
    """

    # How many initial frame pairs to capture for the first assessment
    INITIAL_BURST_COUNT = 3
    # How long to wait for cameras before giving up (seconds)
    CAMERA_TIMEOUT_S = 10.0
    # Delay between frame captures during burst (seconds)
    BURST_FRAME_DELAY_S = 0.15
    # Delay between refinement scans (seconds)
    REFINEMENT_INTERVAL_S = 2.0
    # Maximum refinement scans before stopping
    MAX_REFINEMENT_SCANS = 10
    # Minimum model confidence to stop refining early
    TARGET_MODEL_CONFIDENCE = 0.6

    def __init__(
        self,
        cam0: Optional[FrameProvider] = None,
        cam1: Optional[FrameProvider] = None,
        estimator: Optional[ObjectDimensionEstimator] = None,
        world_model: Optional[WorldModel] = None,
        scene_analyzer: Optional[SceneAnalyzer] = None,
    ):
        self._cam0 = cam0
        self._cam1 = cam1
        self._estimator = estimator or ObjectDimensionEstimator()
        self._world_model = world_model or WorldModel()
        self._scene_analyzer = scene_analyzer or SceneAnalyzer()

        self._phase = ScanPhase.WAITING_FOR_CAMERAS
        self._scan_results: list[ScanResult] = []
        self._scan_count = 0
        self._start_time: float = 0.0

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._on_model_ready_callbacks: list[Callable[[WorldModelSnapshot], None]] = []
        self._model_ready_fired = False

    def set_cameras(
        self,
        cam0: Optional[FrameProvider] = None,
        cam1: Optional[FrameProvider] = None,
    ):
        """Set or update camera references (can be called before start)."""
        if cam0 is not None:
            self._cam0 = cam0
        if cam1 is not None:
            self._cam1 = cam1

    def on_model_ready(self, callback: Callable[[WorldModelSnapshot], None]):
        """Register a callback for when the initial world model is ready."""
        self._on_model_ready_callbacks.append(callback)

    def start(self):
        """Start the scanning process in a background thread."""
        if self._running:
            logger.warning("Startup scanner already running")
            return
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True, name="startup-scanner")
        self._thread.start()
        logger.info("Startup scanner started")

    def stop(self):
        """Stop the scanner gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Startup scanner stopped")

    @property
    def phase(self) -> ScanPhase:
        with self._lock:
            return self._phase

    @property
    def is_running(self) -> bool:
        return self._running

    def get_world_model(self) -> WorldModel:
        """Get the world model (may be partially built)."""
        return self._world_model

    def get_report(self) -> StartupScanReport:
        """Get the current scan report."""
        with self._lock:
            model_snap = self._world_model.snapshot()
            total_objects = sum(len(r.estimates) for r in self._scan_results)
            elapsed = (time.monotonic() - self._start_time) * 1000 if self._start_time else 0

            return StartupScanReport(
                phase=self._phase,
                scans_completed=self._scan_count,
                total_objects_detected=total_objects,
                world_model=model_snap,
                scan_results=list(self._scan_results),
                elapsed_total_ms=elapsed,
                message=self._phase_message(),
            )

    # ------------------------------------------------------------------
    # Internal scanning loop
    # ------------------------------------------------------------------

    def _run(self):
        """Main scanning loop — runs in background thread."""
        try:
            # Phase 1: Wait for cameras
            self._set_phase(ScanPhase.WAITING_FOR_CAMERAS)
            if not self._wait_for_cameras():
                self._set_phase(ScanPhase.ERROR)
                logger.error("Cameras did not come online within timeout")
                return

            # Phase 2: Initial burst capture + analysis
            self._set_phase(ScanPhase.INITIAL_CAPTURE)
            self._initial_burst()

            # Phase 3: Build initial world model
            self._set_phase(ScanPhase.WORLD_MODEL_BUILD)
            self._build_model()

            # Fire model-ready callbacks
            self._fire_model_ready()

            # Phase 4: Continuous refinement
            self._set_phase(ScanPhase.CONTINUOUS_REFINEMENT)
            self._refinement_loop()

            self._set_phase(ScanPhase.COMPLETE)
            logger.info(
                "Startup scanner complete: %d scans, %d objects",
                self._scan_count,
                len(self._world_model.snapshot().objects),
            )

        except Exception:
            logger.exception("Startup scanner failed")
            self._set_phase(ScanPhase.ERROR)
        finally:
            self._running = False

    def _set_phase(self, phase: ScanPhase):
        with self._lock:
            self._phase = phase
        logger.info("Scanner phase: %s", phase.value)

    def _phase_message(self) -> str:
        messages = {
            ScanPhase.WAITING_FOR_CAMERAS: "Waiting for cameras to connect...",
            ScanPhase.INITIAL_CAPTURE: f"Capturing initial frames ({self._scan_count}/{self.INITIAL_BURST_COUNT})...",
            ScanPhase.DIMENSION_ANALYSIS: "Analyzing object dimensions...",
            ScanPhase.GRADING: "Grading and re-assessing estimates...",
            ScanPhase.WORLD_MODEL_BUILD: "Building spatial world model...",
            ScanPhase.CONTINUOUS_REFINEMENT: f"Refining model (scan {self._scan_count})...",
            ScanPhase.COMPLETE: "Environment assessment complete.",
            ScanPhase.ERROR: "Scanner encountered an error.",
        }
        return messages.get(self._phase, "Unknown phase")

    def _wait_for_cameras(self) -> bool:
        """Wait until at least one camera is connected and delivering frames."""
        deadline = time.monotonic() + self.CAMERA_TIMEOUT_S
        while self._running and time.monotonic() < deadline:
            cam0_ok = self._cam0 is not None and self._cam0.connected
            cam1_ok = self._cam1 is not None and self._cam1.connected

            if cam0_ok or cam1_ok:
                logger.info(
                    "Camera(s) ready: cam0=%s, cam1=%s",
                    "connected" if cam0_ok else "offline",
                    "connected" if cam1_ok else "offline",
                )
                # Give cameras a moment to stabilize (auto-exposure, focus)
                time.sleep(0.3)
                return True

            time.sleep(0.2)

        return False

    def _capture_frame_pair(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture one frame from each available camera."""
        f0 = self._cam0.get_raw_frame() if self._cam0 else None
        f1 = self._cam1.get_raw_frame() if self._cam1 else None
        return f0, f1

    def _initial_burst(self):
        """Capture a rapid burst of frame pairs and analyze each one."""
        for i in range(self.INITIAL_BURST_COUNT):
            if not self._running:
                break

            f0, f1 = self._capture_frame_pair()
            if f0 is None and f1 is None:
                logger.warning("Burst frame %d: no frames available", i)
                time.sleep(self.BURST_FRAME_DELAY_S)
                continue

            result = self._analyze_frame_pair(f0, f1, i)
            with self._lock:
                self._scan_results.append(result)
                self._scan_count += 1

            logger.info(
                "Burst scan %d: %d objects, %.1fms",
                i,
                len(result.estimates),
                result.elapsed_ms,
            )

            if i < self.INITIAL_BURST_COUNT - 1:
                time.sleep(self.BURST_FRAME_DELAY_S)

    def _analyze_frame_pair(
        self,
        cam0_frame: Optional[np.ndarray],
        cam1_frame: Optional[np.ndarray],
        scan_index: int,
    ) -> ScanResult:
        """Run full analysis on a frame pair: detect, estimate dimensions, grade."""
        t0 = time.monotonic()

        # Dimension estimation (includes detection + grading + re-assessment)
        self._set_phase(ScanPhase.DIMENSION_ANALYSIS)
        estimates = self._estimator.estimate_from_frames(cam0_frame, cam1_frame)

        # Scene analysis for spatial relationships
        scene = None
        if cam1_frame is not None:
            scene = self._scene_analyzer.analyze(
                cam1_frame,
                cam0_frame=cam0_frame,
                timestamp=time.monotonic(),
            )

        self._set_phase(ScanPhase.GRADING)

        elapsed_ms = (time.monotonic() - t0) * 1000

        return ScanResult(
            scan_index=scan_index,
            estimates=estimates,
            scene=scene,
            elapsed_ms=elapsed_ms,
            phase=ScanPhase.GRADING,
        )

    def _build_model(self):
        """Build the world model from all accumulated scan results."""
        # Collect positions from scene analysis if available
        positions: dict[str, np.ndarray] = {}
        for result in self._scan_results:
            if result.scene is not None:
                for obj in result.scene.objects:
                    if obj.centroid_3d is not None:
                        positions[obj.label] = np.array(obj.centroid_3d)

        # Feed all estimates into the world model
        all_estimates = self._estimator.get_best_estimates()
        self._world_model.update(all_estimates, positions)

        snap = self._world_model.snapshot()
        logger.info(
            "Initial world model built: %d objects, confidence=%.2f grade=%s, "
            "reachable=%d, obstacles=%d, free=%.0f%%",
            len(snap.objects),
            snap.model_confidence,
            snap.model_grade,
            snap.reachable_targets,
            snap.obstacles_detected,
            snap.free_zone_pct,
        )

    def _fire_model_ready(self):
        """Notify registered callbacks that the initial model is ready."""
        if self._model_ready_fired:
            return
        self._model_ready_fired = True

        snap = self._world_model.snapshot()
        for cb in self._on_model_ready_callbacks:
            try:
                cb(snap)
            except Exception:
                logger.exception("Error in model-ready callback")

    def _refinement_loop(self):
        """Continue scanning at reduced rate to improve model confidence."""
        refinement_count = 0
        while self._running and refinement_count < self.MAX_REFINEMENT_SCANS:
            time.sleep(self.REFINEMENT_INTERVAL_S)
            if not self._running:
                break

            f0, f1 = self._capture_frame_pair()
            if f0 is None and f1 is None:
                continue

            result = self._analyze_frame_pair(f0, f1, self._scan_count)
            with self._lock:
                self._scan_results.append(result)
                self._scan_count += 1

            # Update world model with new estimates
            positions: dict[str, np.ndarray] = {}
            if result.scene is not None:
                for obj in result.scene.objects:
                    if obj.centroid_3d is not None:
                        positions[obj.label] = np.array(obj.centroid_3d)

            best = self._estimator.get_best_estimates()
            self._world_model.update(best, positions)

            refinement_count += 1

            # Check if model confidence is good enough to stop early
            snap = self._world_model.snapshot()
            if snap.model_confidence >= self.TARGET_MODEL_CONFIDENCE:
                logger.info(
                    "Model confidence %.2f >= target %.2f, stopping refinement",
                    snap.model_confidence,
                    self.TARGET_MODEL_CONFIDENCE,
                )
                break

            logger.info(
                "Refinement scan %d: confidence=%.2f grade=%s",
                refinement_count,
                snap.model_confidence,
                snap.model_grade,
            )
