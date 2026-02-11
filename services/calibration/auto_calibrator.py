"""
Auto Calibrator — Automated checkerboard intrinsic calibration for th3cl4w cameras.

Captures frames from the camera HTTP server, detects checkerboard corners,
runs OpenCV calibrateCamera, and saves results to the standard JSON files.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests  # type: ignore[import-untyped]

logger = logging.getLogger("th3cl4w.calibration.auto")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIBRATION_RESULTS_DIR = PROJECT_ROOT / "calibration_results"
DATA_DIR = PROJECT_ROOT / "data"

from shared.config.camera_config import CAMERA_SERVER_URL

# Board sizes to try if primary detection fails
FALLBACK_BOARD_SIZES = [(7, 4), (9, 6), (8, 6), (7, 5)]

CAMERA_LABELS = {0: "overhead", 1: "arm-mounted", 2: "side"}


@dataclass
class CalibrationResult:
    camera_id: int
    rms: float
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    image_size: tuple[int, int]
    fov_h: float
    fov_v: float
    fov_d: float
    board_size: tuple[int, int]
    square_size_mm: float
    num_frames: int
    num_detected: int
    warnings: list[str] = field(default_factory=list)
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "camera": f"cam{self.camera_id}",
            "method": "checkerboard_calibration",
            "board_size": list(self.board_size),
            "square_size_mm": self.square_size_mm,
            "num_frames": self.num_frames,
            "num_detected": self.num_detected,
            "rms_reprojection_px": round(self.rms, 6),
            "camera_matrix": {
                "fx": round(self.fx, 2),
                "fy": round(self.fy, 2),
                "cx": round(self.cx, 2),
                "cy": round(self.cy, 2),
            },
            "distortion": {
                "k1": round(self.k1, 6),
                "k2": round(self.k2, 6),
                "p1": round(self.p1, 6),
                "p2": round(self.p2, 6),
                "k3": round(self.k3, 6),
            },
            "fov_deg": {
                "horizontal": round(self.fov_h, 1),
                "vertical": round(self.fov_v, 1),
                "diagonal": round(self.fov_d, 1),
            },
            "image_size": list(self.image_size),
            "warnings": self.warnings,
            "date": date.today().isoformat(),
        }


@dataclass
class CalibrationProgress:
    """Tracks calibration progress for status reporting."""

    camera_id: int = -1
    state: str = "idle"  # idle, capturing, detecting, calibrating, saving, done, error
    frames_captured: int = 0
    frames_total: int = 0
    corners_found: int = 0
    error_message: str = ""
    result: Optional[CalibrationResult] = None

    def to_dict(self) -> dict:
        d = {
            "camera_id": self.camera_id,
            "state": self.state,
            "frames_captured": self.frames_captured,
            "frames_total": self.frames_total,
            "corners_found": self.corners_found,
            "error_message": self.error_message,
        }
        if self.result:
            d["result"] = self.result.to_dict()
        return d


class AutoCalibrator:
    """Automated checkerboard camera calibration engine."""

    def __init__(
        self,
        board_size: tuple[int, int] = (8, 5),
        square_size_mm: float = 19.0,
        num_frames: int = 10,
        camera_server_url: str = CAMERA_SERVER_URL,
        auto_detect_board: bool = True,
    ):
        self.board_size = board_size
        self.square_size_mm = square_size_mm
        self.num_frames = num_frames
        self.camera_server_url = camera_server_url.rstrip("/")
        self.auto_detect_board = auto_detect_board
        self.progress = CalibrationProgress()

    def capture_frames(
        self, cam_id: int, num_frames: int = 0, interval_s: float = 0.5
    ) -> list[np.ndarray]:
        """Capture frames from camera server via HTTP snapshots."""
        n = num_frames or self.num_frames
        self.progress.camera_id = cam_id
        self.progress.state = "capturing"
        self.progress.frames_total = n
        self.progress.frames_captured = 0

        frames = []
        for i in range(n):
            try:
                url = f"{self.camera_server_url}/snap/{cam_id}"
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                img_array = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    frames.append(frame)
                    self.progress.frames_captured = len(frames)
                    logger.info(f"Captured frame {len(frames)}/{n} from cam{cam_id}")
                else:
                    logger.warning(f"Failed to decode frame {i+1} from cam{cam_id}")
            except Exception as e:
                logger.warning(f"Failed to capture frame {i+1} from cam{cam_id}: {e}")

            if i < n - 1:
                time.sleep(interval_s)

        return frames

    def detect_checkerboard(
        self, frame: np.ndarray, board_size: Optional[tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        """Detect and refine checkerboard corners in a frame.

        Returns refined corners array or None if not found.
        If board_size is None and auto_detect_board is True, tries fallback sizes.
        """
        sizes_to_try = [board_size or self.board_size]
        if board_size is None and self.auto_detect_board:
            sizes_to_try.extend(s for s in FALLBACK_BOARD_SIZES if s != self.board_size)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for sz in sizes_to_try:
            found, corners = cv2.findChessboardCorners(gray, sz, flags)
            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Store which board size was actually detected
                self._last_detected_size = sz
                return corners_refined

        self._last_detected_size = None
        return None

    def _assess_quality(
        self, result: CalibrationResult, all_corners: list, image_size: tuple[int, int]
    ) -> list[str]:
        """Run quality checks and return warnings."""
        warnings = []

        # fx/fy ratio check
        ratio = result.fx / result.fy if result.fy > 0 else 1.0
        if abs(ratio - 1.0) > 0.10:
            warnings.append(
                f"fx/fy ratio is {ratio:.3f} (>10% off) — possible astigmatism or bad board angle"
            )

        # Distortion coefficient magnitude
        if abs(result.k2) > 10:
            warnings.append(
                f"|k2|={abs(result.k2):.1f} is large — possible overfitting, need more diverse board positions"
            )
        if abs(result.k3) > 10:
            warnings.append(
                f"|k3|={abs(result.k3):.1f} is large — possible overfitting, need more diverse board positions"
            )

        # Board coverage check
        w, h = image_size
        for corners in all_corners:
            pts = corners.reshape(-1, 2)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            coverage = ((x_max - x_min) * (y_max - y_min)) / (w * h)
            if coverage < 0.05:
                warnings.append(
                    f"Board covers only {coverage*100:.1f}% of frame — move closer or use more diverse positions"
                )
                break

        # RMS check
        if result.rms > 1.0:
            warnings.append(
                f"RMS reprojection error is {result.rms:.3f}px — consider recalibrating"
            )

        return warnings

    def calibrate(self, frames: list[np.ndarray]) -> CalibrationResult:
        """Run OpenCV calibration on captured frames with detected checkerboards."""
        self.progress.state = "detecting"
        self.progress.corners_found = 0

        obj_points_list = []
        img_points_list = []
        detected_size = self.board_size

        for frame in frames:
            corners = self.detect_checkerboard(frame)
            if corners is not None:
                detected_size = (
                    getattr(self, "_last_detected_size", self.board_size) or self.board_size
                )
                # Build object points for detected board size
                objp = np.zeros((detected_size[0] * detected_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0 : detected_size[0], 0 : detected_size[1]].T.reshape(-1, 2)
                objp *= self.square_size_mm

                obj_points_list.append(objp)
                img_points_list.append(corners)
                self.progress.corners_found += 1

        if len(obj_points_list) < 3:
            self.progress.state = "error"
            self.progress.error_message = (
                f"Only {len(obj_points_list)} frames had detectable checkerboards (need ≥3)"
            )
            raise ValueError(self.progress.error_message)

        h, w = frames[0].shape[:2]
        image_size = (w, h)

        self.progress.state = "calibrating"
        logger.info(f"Calibrating with {len(obj_points_list)} frames, board {detected_size}")

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list, img_points_list, (w, h), None, None
        )

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5]

        # Compute FOV
        fov_h = 2 * math.degrees(math.atan2(w / 2, fx))
        fov_v = 2 * math.degrees(math.atan2(h / 2, fy))
        diag = math.sqrt(w**2 + h**2)
        f_avg = (fx + fy) / 2
        fov_d = 2 * math.degrees(math.atan2(diag / 2, f_avg))

        result = CalibrationResult(
            camera_id=self.progress.camera_id,
            rms=rms,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            k1=k1,
            k2=k2,
            p1=p1,
            p2=p2,
            k3=k3,
            image_size=image_size,
            fov_h=fov_h,
            fov_v=fov_v,
            fov_d=fov_d,
            board_size=detected_size,
            square_size_mm=self.square_size_mm,
            num_frames=len(frames),
            num_detected=len(obj_points_list),
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )

        result.warnings = self._assess_quality(result, img_points_list, image_size)

        return result

    def save_results(self, result: CalibrationResult) -> dict:
        """Save calibration result to JSON files."""
        self.progress.state = "saving"
        cam_id = result.camera_id
        result_dict = result.to_dict()

        # Ensure directories exist
        CALIBRATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Save per-camera file
        cam_file = CALIBRATION_RESULTS_DIR / f"cam{cam_id}_checkerboard_calibration.json"
        cam_file.write_text(json.dumps(result_dict, indent=2))
        logger.info(f"Saved {cam_file}")

        # 2. Update camera_intrinsics.json (both locations)
        for intrinsics_path in [
            CALIBRATION_RESULTS_DIR / "camera_intrinsics.json",
            DATA_DIR / "camera_intrinsics.json",
        ]:
            intrinsics = {}
            if intrinsics_path.exists():
                try:
                    intrinsics = json.loads(intrinsics_path.read_text())
                except json.JSONDecodeError:
                    intrinsics = {}

            cameras = intrinsics.setdefault("cameras", {})
            cam_key = f"cam{cam_id}"
            cam_entry = cameras.get(cam_key, {})

            # Update intrinsic fields
            cam_entry["image_size"] = list(result.image_size)
            cam_entry["camera_matrix"] = result_dict["camera_matrix"]
            cam_entry["distortion_coefficients"] = result_dict["distortion"]
            cam_entry["fov_diagonal_deg"] = result_dict["fov_deg"]["diagonal"]
            cam_entry["fov_horizontal_deg"] = result_dict["fov_deg"]["horizontal"]
            cam_entry["fov_vertical_deg"] = result_dict["fov_deg"]["vertical"]
            cam_entry["calibration_method"] = (
                f"checkerboard_{result.square_size_mm:.0f}mm_{result.num_detected}frames"
            )
            cam_entry["rms_reprojection_px"] = result_dict["rms_reprojection_px"]
            if result.warnings:
                cam_entry["calibration_warnings"] = result.warnings

            cameras[cam_key] = cam_entry
            intrinsics["cameras"] = cameras
            intrinsics_path.write_text(json.dumps(intrinsics, indent=2))
            logger.info(f"Updated {intrinsics_path}")

        self.progress.state = "done"
        self.progress.result = result
        return result_dict

    def calibrate_camera(self, cam_id: int) -> CalibrationResult:
        """Full pipeline: capture → detect → calibrate → save for one camera."""
        self.progress = CalibrationProgress(camera_id=cam_id, state="capturing")
        try:
            frames = self.capture_frames(cam_id, self.num_frames)
            if not frames:
                raise ValueError(f"No frames captured from cam{cam_id}")
            result = self.calibrate(frames)
            self.save_results(result)
            logger.info(
                f"cam{cam_id} calibrated: RMS={result.rms:.4f} fx={result.fx:.1f} fy={result.fy:.1f} "
                f"FOV={result.fov_d:.1f}° ({result.num_detected}/{result.num_frames} frames)"
            )
            return result
        except Exception as e:
            self.progress.state = "error"
            self.progress.error_message = str(e)
            raise

    def calibrate_all(self, camera_ids: list[int] = None) -> dict[int, CalibrationResult]:
        """Calibrate multiple cameras sequentially."""
        ids = camera_ids or [0, 1, 2]
        results = {}
        for cam_id in ids:
            try:
                results[cam_id] = self.calibrate_camera(cam_id)
            except Exception as e:
                logger.error(f"cam{cam_id} calibration failed: {e}")
        return results

    def validate(self, cam_id: int) -> dict:
        """Validate existing calibration by grabbing a new frame and checking corner alignment."""
        # Load existing calibration
        cam_file = CALIBRATION_RESULTS_DIR / f"cam{cam_id}_checkerboard_calibration.json"
        if not cam_file.exists():
            return {"valid": False, "error": f"No calibration file for cam{cam_id}"}

        cal_data = json.loads(cam_file.read_text())
        cm = cal_data["camera_matrix"]
        camera_matrix = np.array(
            [
                [cm["fx"], 0, cm["cx"]],
                [0, cm["fy"], cm["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        d = cal_data["distortion"]
        dist_coeffs = np.array([d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]], dtype=np.float64)

        # Capture a fresh frame
        frames = self.capture_frames(cam_id, num_frames=1, interval_s=0)
        if not frames:
            return {"valid": False, "error": "Could not capture validation frame"}

        frame = frames[0]
        corners = self.detect_checkerboard(frame)
        if corners is None:
            return {"valid": False, "error": "No checkerboard detected in validation frame"}

        # Undistort and re-detect
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        corners_undist = self.detect_checkerboard(undistorted)

        if corners_undist is None:
            return {"valid": False, "error": "Checkerboard not detected after undistortion"}

        # Check that corners in undistorted image are more grid-like
        # Compute reprojection: use solvePnP on undistorted corners
        detected_size = getattr(self, "_last_detected_size", self.board_size) or self.board_size
        objp = np.zeros((detected_size[0] * detected_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : detected_size[0], 0 : detected_size[1]].T.reshape(-1, 2)
        objp *= self.square_size_mm

        _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = np.sqrt(
            ((corners.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2).sum(axis=1)
        ).mean()

        return {
            "valid": True,
            "mean_reprojection_px": round(float(error), 4),
            "corners_detected": len(corners),
            "board_size": list(detected_size),
            "quality": "good" if error < 1.0 else "fair" if error < 3.0 else "poor",
        }
