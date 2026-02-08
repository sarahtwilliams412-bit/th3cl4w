#!/usr/bin/env python3.12
"""Quick stereo calibration — auto-detects board size, captures from camera server."""
import sys, time, json, argparse
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import httpx

CAM_SERVER = "http://localhost:8081"
OUTPUT = Path(project_root) / "data" / "stereo_calibration.json"


def grab_frame(cam_idx: int) -> np.ndarray:
    resp = httpx.get(f"{CAM_SERVER}/snap/{cam_idx}", timeout=5.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Camera {cam_idx} snap failed: {resp.status_code}")
    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode camera {cam_idx}")
    return img


def detect_board(gray, board_sizes=None):
    """Try multiple board sizes, return first that works."""
    if board_sizes is None:
        # Common sizes, largest first
        board_sizes = [
            (15, 11), (13, 9), (11, 8), (10, 7), (9, 6),
            (8, 6), (7, 5), (7, 4), (6, 4), (5, 4), (5, 3), (4, 3),
        ]
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    for cols, rows in board_sizes:
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if ret:
            return (cols, rows), corners
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Quick Stereo Calibration")
    parser.add_argument("--pairs", type=int, default=8, help="Number of image pairs (default: 8)")
    parser.add_argument("--square-mm", type=float, default=24.0,
                        help="Square size in mm (default: 24.0 = 15/16 inch)")
    parser.add_argument("--cols", type=int, default=0, help="Board inner corners cols (0=auto)")
    parser.add_argument("--rows", type=int, default=0, help="Board inner corners rows (0=auto)")
    parser.add_argument("--output", type=str, default=str(OUTPUT))
    parser.add_argument("--cam-server", type=str, default=CAM_SERVER)
    args = parser.parse_args()

    output = Path(args.output)

    board_sizes = None
    if args.cols > 0 and args.rows > 0:
        board_sizes = [(args.cols, args.rows)]
        print(f"Using fixed board size: {args.cols}x{args.rows}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Phase 1: Auto-detect board size from a single frame
    print("Detecting board size...", flush=True)
    detected_size = None
    for cam in [0, 1]:
        try:
            frame = grab_frame(cam)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"  Cam {cam}: {gray.shape[1]}x{gray.shape[0]}", flush=True)
            size, corners = detect_board(gray, board_sizes)
            if size:
                detected_size = size
                print(f"  Cam {cam}: detected {size[0]}x{size[1]} board ✓", flush=True)
                break
            else:
                print(f"  Cam {cam}: no board found", flush=True)
        except Exception as e:
            print(f"  Cam {cam}: error: {e}", flush=True)

    if detected_size is None:
        print("\nFAILED: Could not detect checkerboard in either camera.")
        print("Make sure the board is visible and well-lit.")
        sys.exit(1)

    cols, rows = detected_size
    print(f"\nUsing board: {cols}x{rows} inner corners, {args.square_mm}mm squares")

    obj_p = np.zeros((cols * rows, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_mm

    # Phase 2: Capture pairs
    obj_points = []
    img_points_l = []
    img_points_r = []
    img_size = None
    good = 0

    print(f"Capturing {args.pairs} pairs...", flush=True)
    for attempt in range(args.pairs * 3):
        if good >= args.pairs:
            break

        try:
            left = grab_frame(0)
            right = grab_frame(1)
        except Exception as e:
            print(f"  [{attempt+1}] Camera error: {e}", flush=True)
            time.sleep(0.5)
            continue

        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        img_size = gray_l.shape[::-1]

        size_l, corners_l = detect_board(gray_l, [(cols, rows)])
        size_r, corners_r = detect_board(gray_r, [(cols, rows)])

        if size_l and size_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_p)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            good += 1
            print(f"  [{attempt+1}] Pair {good}/{args.pairs} ✓", flush=True)
        else:
            status = f"L={'✓' if size_l else '✗'} R={'✓' if size_r else '✗'}"
            print(f"  [{attempt+1}] {status}", flush=True)

        time.sleep(0.3)

    if good < 3:
        print(f"\nFAILED: Only {good} valid pairs (need ≥3).")
        print("Ensure the board is visible to BOTH cameras simultaneously.")
        sys.exit(1)

    # Phase 3: Calibrate
    print(f"\nCalibrating with {good} pairs at {img_size[0]}x{img_size[1]}...", flush=True)

    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, img_points_l, img_size, None, None)
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, img_points_r, img_size, None, None)
    print(f"  Left RMS: {ret_l:.4f}, Right RMS: {ret_r:.4f}")

    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, D_l, K_r, D_r, img_size,
        criteria=criteria, flags=flags
    )
    print(f"  Stereo RMS: {ret:.4f}")
    print(f"  Baseline: {np.linalg.norm(T):.1f}mm")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T, alpha=0
    )

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    calib = {
        "image_size": list(img_size),
        "board_size": [cols, rows],
        "square_mm": args.square_mm,
        "K_left": K_l.tolist(), "D_left": D_l.tolist(),
        "K_right": K_r.tolist(), "D_right": D_r.tolist(),
        "R": R.tolist(), "T": T.tolist(),
        "E": E.tolist(), "F": F.tolist(),
        "R1": R1.tolist(), "R2": R2.tolist(),
        "P1": P1.tolist(), "P2": P2.tolist(),
        "Q": Q.tolist(),
        "stereo_rms": ret, "left_rms": ret_l, "right_rms": ret_r,
        "baseline_mm": float(np.linalg.norm(T)),
        "pairs_used": good,
        "timestamp": time.time(),
    }
    output.write_text(json.dumps(calib, indent=2))
    print(f"\n✓ Saved to {output}")
    print("Calibration complete!")


if __name__ == "__main__":
    main()
