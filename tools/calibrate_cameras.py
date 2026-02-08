#!/usr/bin/env python3.12
"""
Interactive calibration CLI for two independent cameras.

Captures checkerboard images from each camera separately,
computes intrinsics + distortion, and optionally the extrinsic
camera-to-workspace transform.

Usage:
    python3.12 tools/calibrate_cameras.py --cam0 0 --cam1 2
    python3.12 tools/calibrate_cameras.py --images-dir calibration_images/
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from src.vision.calibration import (
    IndependentCalibrator,
    CameraCalibration,
    DEFAULT_BOARD_SIZE,
    DEFAULT_SQUARE_SIZE_MM,
)


def capture_calibration_images(
    cap: cv2.VideoCapture,
    camera_id: str,
    calibrator: IndependentCalibrator,
    num_images: int = 10,
    save_dir: str = "calibration_images",
) -> int:
    """Interactively capture checkerboard images from a camera.

    Press SPACE to capture, ESC to finish early.
    Returns number of images successfully captured.
    """
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    print(f"\n=== Calibrating {camera_id} ===")
    print(f"Need {num_images} checkerboard images. Press SPACE to capture, ESC to finish.\n")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Try to find corners for preview
        display = frame.copy()
        corners = calibrator.find_corners(frame, refine=False)
        if corners is not None:
            cv2.drawChessboardCorners(
                display, calibrator.board_size, corners, True
            )
            cv2.putText(
                display,
                f"Board found! SPACE to capture ({count}/{num_images})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display,
                f"No board detected ({count}/{num_images})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow(f"Calibrate {camera_id}", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(" ") and corners is not None:
            # Capture with refined corners
            result = calibrator.add_calibration_image(camera_id, frame)
            if result is not None:
                path = os.path.join(save_dir, f"{camera_id}_{count:03d}.jpg")
                cv2.imwrite(path, frame)
                count += 1
                print(f"  Captured {count}/{num_images}: {path}")

    cv2.destroyWindow(f"Calibrate {camera_id}")
    return count


def calibrate_from_directory(
    calibrator: IndependentCalibrator,
    camera_id: str,
    image_dir: str,
) -> int:
    """Load calibration images from a directory."""
    count = 0
    prefix = f"{camera_id}_"
    for fname in sorted(os.listdir(image_dir)):
        if not fname.startswith(prefix):
            continue
        if not fname.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        path = os.path.join(image_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        result = calibrator.add_calibration_image(camera_id, img)
        if result is not None:
            count += 1
            print(f"  Loaded {path} — corners found")
        else:
            print(f"  Loaded {path} — NO corners")
    return count


def main():
    parser = argparse.ArgumentParser(description="Calibrate two independent cameras")
    parser.add_argument("--cam0", type=int, default=None, help="Camera 0 device index")
    parser.add_argument("--cam1", type=int, default=None, help="Camera 1 device index")
    parser.add_argument("--images-dir", type=str, default=None, help="Load from directory instead of live capture")
    parser.add_argument("--board-cols", type=int, default=DEFAULT_BOARD_SIZE[0])
    parser.add_argument("--board-rows", type=int, default=DEFAULT_BOARD_SIZE[1])
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_SIZE_MM)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="calibration")
    args = parser.parse_args()

    board_size = (args.board_cols, args.board_rows)
    calibrator = IndependentCalibrator(
        board_size=board_size, square_size_mm=args.square_mm
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for cam_id in ["cam0", "cam1"]:
        print(f"\n{'='*50}")
        print(f"  Camera: {cam_id}")
        print(f"{'='*50}")

        if args.images_dir:
            n = calibrate_from_directory(calibrator, cam_id, args.images_dir)
        else:
            dev_idx = args.cam0 if cam_id == "cam0" else args.cam1
            if dev_idx is None:
                print(f"  Skipping {cam_id} (no device index)")
                continue
            cap = cv2.VideoCapture(dev_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                print(f"  ERROR: Cannot open camera {dev_idx}")
                continue
            n = capture_calibration_images(
                cap, cam_id, calibrator, num_images=args.num_images
            )
            cap.release()

        print(f"  {n} images collected for {cam_id}")

        if n < 3:
            print(f"  Not enough images to calibrate {cam_id} (need >= 3)")
            continue

        cal = calibrator.calibrate_camera(cam_id)
        if cal is None:
            print(f"  Calibration failed for {cam_id}")
            continue

        print(f"  RMS reprojection error: {cal.reprojection_error:.4f}")
        print(f"  Focal length: fx={cal.fx:.1f}, fy={cal.fy:.1f}")
        print(f"  Principal point: cx={cal.cx:.1f}, cy={cal.cy:.1f}")

        out_path = os.path.join(args.output_dir, f"{cam_id}_calibration.json")
        cal.save(out_path)
        print(f"  Saved to {out_path}")

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
