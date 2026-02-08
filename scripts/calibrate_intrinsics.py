#!/usr/bin/env python3
"""
Camera intrinsic calibration using checkerboard images.

Usage:
  1. Hold a checkerboard in front of each camera
  2. Run: python scripts/calibrate_intrinsics.py --capture --cam 0
     This captures 15 frames with 3-second intervals (move the board between captures)
  3. Repeat for cam 1: python scripts/calibrate_intrinsics.py --capture --cam 1
  4. Run calibration: python scripts/calibrate_intrinsics.py --calibrate
  
Or provide pre-captured images:
  python scripts/calibrate_intrinsics.py --calibrate --cam0-dir /path/to/cam0 --cam1-dir /path/to/cam1

The checkerboard should be:
  - Held at various angles and positions (10+ different poses per camera)
  - Covering different parts of the image frame
  - A standard black/white checkerboard (e.g., 9x6 inner corners)
"""

import argparse
import cv2
import json
import glob
import os
import sys
import time
from pathlib import Path
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "calibration_results"
CAMERA_SERVER = "http://localhost:8081"

BOARD_SIZES = [(9, 6), (8, 6), (7, 5), (7, 4), (6, 4), (8, 5), (10, 7)]
SQUARE_SIZE_MM = 25.0  # doesn't affect intrinsics


def capture_frames(cam_id: int, output_dir: str, num_frames: int = 15, interval: float = 3.0):
    """Capture frames from camera server with delay between each."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Capturing {num_frames} frames from camera {cam_id}")
    print(f"Move the checkerboard between captures! ({interval}s interval)")
    print()

    for i in range(num_frames):
        url = f"{CAMERA_SERVER}/snap/{cam_id}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"  [{i+1}/{num_frames}] Captured {path}")
        except Exception as e:
            print(f"  [{i+1}/{num_frames}] Failed: {e}")

        if i < num_frames - 1:
            print(f"  Move the checkerboard... waiting {interval}s")
            time.sleep(interval)

    print(f"\nDone! {num_frames} frames saved to {output_dir}")


def find_checkerboard(gray):
    """Try multiple board sizes to find checkerboard corners."""
    for size in BOARD_SIZES:
        ret, corners = cv2.findChessboardCorners(
            gray, size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            return size, corners
    return None, None


def calibrate_camera(image_dir: str, cam_name: str):
    """Run OpenCV calibration on checkerboard images."""
    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        print(f"  No images in {image_dir}")
        return None

    print(f"  Processing {len(images)} images...")

    obj_points, img_points = [], []
    board_size_used = None
    img_size = None

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        board_size, corners = find_checkerboard(gray)
        if corners is not None:
            if board_size_used is None:
                board_size_used = board_size
            elif board_size != board_size_used:
                continue

            objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM
            obj_points.append(objp)
            img_points.append(corners)
            print(f"    ✓ {os.path.basename(path)}: board {board_size}")
        else:
            print(f"    ✗ {os.path.basename(path)}: no checkerboard")

    if len(obj_points) < 3:
        print(f"  ⚠ Only {len(obj_points)} valid images (need ≥3)")
        return None

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    print(f"  RMS error: {ret:.4f} px")
    print(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"  dist: {dist.ravel()[:5]}")

    return {
        "fx": round(float(K[0, 0]), 2),
        "fy": round(float(K[1, 1]), 2),
        "cx": round(float(K[0, 2]), 2),
        "cy": round(float(K[1, 2]), 2),
        "dist_coeffs": [round(x, 6) for x in dist.ravel().tolist()[:5]],
        "image_size": list(img_size),
        "rms_error": round(float(ret), 4),
        "num_images_used": len(obj_points),
        "board_size": list(board_size_used),
        "source": "opencv_checkerboard_calibration",
    }


def main():
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration")
    parser.add_argument("--capture", action="store_true", help="Capture frames from camera")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--cam", type=int, choices=[0, 1], help="Camera ID for capture")
    parser.add_argument("--cam0-dir", default="/tmp/intrinsic_cal/cam0")
    parser.add_argument("--cam1-dir", default="/tmp/intrinsic_cal/cam1")
    parser.add_argument("--num-frames", type=int, default=15)
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()

    if args.capture:
        if args.cam is None:
            parser.error("--cam required with --capture")
        out = args.cam0_dir if args.cam == 0 else args.cam1_dir
        capture_frames(args.cam, out, args.num_frames, args.interval)

    if args.calibrate:
        results = {}
        for cam_id, cam_dir in [("cam0", args.cam0_dir), ("cam1", args.cam1_dir)]:
            print(f"\n{'='*50}")
            print(f"Calibrating {cam_id}")
            print(f"{'='*50}")
            if os.path.isdir(cam_dir):
                result = calibrate_camera(cam_dir, cam_id)
                if result:
                    results[cam_id] = result
            else:
                print(f"  Directory not found: {cam_dir}")

        if results:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RESULTS_DIR / "camera_intrinsics.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Saved to {out_path}")
        else:
            print("\n❌ No calibrations succeeded.")

    if not args.capture and not args.calibrate:
        parser.print_help()


if __name__ == "__main__":
    main()
