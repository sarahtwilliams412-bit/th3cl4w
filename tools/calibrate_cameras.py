#!/usr/bin/env python3.12
"""
CLI tool for stereo camera calibration.

Usage:
    python3.12 tools/calibrate_cameras.py --help
    python3.12 tools/calibrate_cameras.py --cam0 0 --cam1 4 --pairs 15
    python3.12 tools/calibrate_cameras.py --load calibration/stereo.npz --verify
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vision.calibration import StereoCalibrator


def capture_calibration_pairs(
    cam0_id: int,
    cam1_id: int,
    num_pairs: int,
    board_size: tuple[int, int],
    width: int = 640,
    height: int = 480,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Interactively capture calibration image pairs.

    Shows live feed; press SPACE to capture, ESC to finish early.
    """
    cap0 = cv2.VideoCapture(cam0_id, cv2.CAP_V4L2)
    cap1 = cv2.VideoCapture(cam1_id, cv2.CAP_V4L2)

    for cap in (cap0, cap1):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap0.isOpened() or not cap1.isOpened():
        print("ERROR: Could not open one or both cameras")
        sys.exit(1)

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    calibrator = StereoCalibrator(board_size=board_size)

    print(f"\nCapturing {num_pairs} calibration pairs.")
    print("Hold checkerboard visible to BOTH cameras.")
    print("SPACE = capture, ESC = finish early\n")

    while len(pairs) < num_pairs:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            continue

        # Preview with corner detection
        vis0 = frame0.copy()
        vis1 = frame1.copy()

        corners0 = calibrator.find_corners(frame0)
        corners1 = calibrator.find_corners(frame1)

        if corners0 is not None:
            cv2.drawChessboardCorners(vis0, board_size, corners0, True)
        if corners1 is not None:
            cv2.drawChessboardCorners(vis1, board_size, corners1, True)

        both_found = corners0 is not None and corners1 is not None
        status = f"Pair {len(pairs)}/{num_pairs}"
        if both_found:
            status += " - READY (press SPACE)"
            color = (0, 255, 0)
        else:
            status += " - searching..."
            color = (0, 0, 255)

        cv2.putText(vis0, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        combined = np.hstack([vis0, vis1])
        cv2.imshow("Stereo Calibration", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32 and both_found:  # SPACE
            pairs.append((frame0.copy(), frame1.copy()))
            print(f"  Captured pair {len(pairs)}/{num_pairs}")
            # Brief flash
            flash = np.ones_like(combined) * 255
            cv2.imshow("Stereo Calibration", flash.astype(np.uint8))
            cv2.waitKey(200)

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Stereo Camera Calibration Tool")
    parser.add_argument("--cam0", type=int, default=0, help="Left camera device index")
    parser.add_argument("--cam1", type=int, default=4, help="Right camera device index")
    parser.add_argument("--pairs", type=int, default=15, help="Number of image pairs to capture")
    parser.add_argument("--board-cols", type=int, default=9, help="Checkerboard inner corners (cols)")
    parser.add_argument("--board-rows", type=int, default=6, help="Checkerboard inner corners (rows)")
    parser.add_argument("--square-size", type=float, default=25.0, help="Square size in mm")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--output", type=str, default="calibration/stereo.npz", help="Output calibration file")
    parser.add_argument("--load", type=str, help="Load existing calibration instead of capturing")
    parser.add_argument("--verify", action="store_true", help="Show rectified live view to verify calibration")
    parser.add_argument("--save-pairs", type=str, help="Directory to save captured image pairs")
    args = parser.parse_args()

    board_size = (args.board_cols, args.board_rows)
    image_size = (args.width, args.height)

    calibrator = StereoCalibrator(
        board_size=board_size,
        square_size=args.square_size,
        image_size=image_size,
    )

    if args.load:
        print(f"Loading calibration from {args.load}")
        calibrator.load(args.load)
        print("Calibration loaded successfully")
    else:
        # Capture pairs
        pairs = capture_calibration_pairs(
            args.cam0, args.cam1, args.pairs, board_size, args.width, args.height
        )

        if len(pairs) < 3:
            print(f"ERROR: Need at least 3 pairs, got {len(pairs)}")
            sys.exit(1)

        # Save raw pairs if requested
        if args.save_pairs:
            pair_dir = Path(args.save_pairs)
            pair_dir.mkdir(parents=True, exist_ok=True)
            for i, (l, r) in enumerate(pairs):
                cv2.imwrite(str(pair_dir / f"left_{i:03d}.png"), l)
                cv2.imwrite(str(pair_dir / f"right_{i:03d}.png"), r)
            print(f"Saved {len(pairs)} pairs to {pair_dir}")

        # Calibrate
        print(f"\nCalibrating with {len(pairs)} pairs...")
        rms = calibrator.calibrate(pairs)
        print(f"Stereo calibration RMS error: {rms:.4f}")

        if rms > 1.0:
            print("WARNING: RMS error > 1.0 — consider recapturing with better images")

        # Save
        output = Path(args.output)
        calibrator.save(output)
        print(f"Calibration saved to {output}")

    # Verify mode
    if args.verify:
        print("\nVerification mode — press ESC to exit")
        cap0 = cv2.VideoCapture(args.cam0, cv2.CAP_V4L2)
        cap1 = cv2.VideoCapture(args.cam1, cv2.CAP_V4L2)
        cap0.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        while True:
            ret0, f0 = cap0.read()
            ret1, f1 = cap1.read()
            if not ret0 or not ret1:
                continue

            rect0, rect1 = calibrator.rectify(f0, f1)

            # Draw horizontal epipolar lines
            combined = np.hstack([rect0, rect1])
            for y in range(0, combined.shape[0], 30):
                cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

            cv2.imshow("Rectified Stereo (ESC to exit)", combined)
            if cv2.waitKey(30) & 0xFF == 27:
                break

        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
