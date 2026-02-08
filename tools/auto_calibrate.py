#!/usr/bin/env python3.12
"""Auto stereo calibration — captures pairs from camera server, runs calibration."""
import sys, time, json
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import httpx

CAM_SERVER = "http://localhost:8081"
BOARD_COLS = 15  # inner corners
BOARD_ROWS = 11
SQUARE_SIZE_MM = 25.0  # estimate — adjust if known
NUM_PAIRS = 8
OUTPUT = Path(project_root) / "data" / "stereo_calibration.json"


def grab_frame(cam_idx: int) -> np.ndarray:
    resp = httpx.get(f"{CAM_SERVER}/snap/{cam_idx}", timeout=5.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Camera {cam_idx} snap failed: {resp.status_code}")
    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode camera {cam_idx}")
    return img


def main():
    board_size = (BOARD_COLS, BOARD_ROWS)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_p = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE_MM

    obj_points = []
    img_points_l = []
    img_points_r = []
    img_size = None

    print(f"Auto stereo calibration: {BOARD_COLS}x{BOARD_ROWS} board, {SQUARE_SIZE_MM}mm squares")
    print(f"Capturing {NUM_PAIRS} pairs from {CAM_SERVER}...")

    good = 0
    attempts = 0
    max_attempts = NUM_PAIRS * 3

    while good < NUM_PAIRS and attempts < max_attempts:
        attempts += 1
        print(f"\n[{attempts}] Grabbing pair...", end=" ")

        try:
            left = grab_frame(0)
            right = grab_frame(1)
        except Exception as e:
            print(f"SKIP (camera error: {e})")
            time.sleep(1)
            continue

        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        img_size = gray_l.shape[::-1]

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, flags)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, flags)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_p)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            good += 1
            print(f"OK ({good}/{NUM_PAIRS})")
        else:
            print(f"SKIP (board not found: L={ret_l}, R={ret_r})")
            # Try different board sizes
            if attempts == 2:
                print("  Tip: verify board dimensions match actual checkerboard")

        time.sleep(0.5)

    if good < 3:
        # Try alternate board sizes
        for alt_cols, alt_rows in [(9, 6), (7, 5), (13, 9), (11, 8), (10, 7)]:
            print(f"\nRetrying with {alt_cols}x{alt_rows} board...")
            alt_size = (alt_cols, alt_rows)
            alt_obj = np.zeros((alt_cols * alt_rows, 3), np.float32)
            alt_obj[:, :2] = np.mgrid[0:alt_cols, 0:alt_rows].T.reshape(-1, 2) * SQUARE_SIZE_MM

            obj_points = []
            img_points_l = []
            img_points_r = []
            good = 0

            for _ in range(NUM_PAIRS * 2):
                try:
                    left = grab_frame(0)
                    right = grab_frame(1)
                except:
                    time.sleep(0.5)
                    continue

                gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                img_size = gray_l.shape[::-1]

                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, alt_size, flags)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, alt_size, flags)

                if ret_l and ret_r:
                    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                    obj_points.append(alt_obj)
                    img_points_l.append(corners_l)
                    img_points_r.append(corners_r)
                    good += 1
                    print(f"  OK ({good}/{NUM_PAIRS})")
                    if good >= NUM_PAIRS:
                        break
                time.sleep(0.3)

            if good >= 3:
                print(f"Found working board size: {alt_cols}x{alt_rows}")
                break

    if good < 3:
        print(f"\nFAILED: Only {good} valid pairs. Need at least 3.")
        print("Check: is the checkerboard visible to BOTH cameras?")
        sys.exit(1)

    print(f"\nCalibrating with {good} pairs...")

    # Individual camera calibration
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, img_points_l, img_size, None, None)
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, img_points_r, img_size, None, None)
    print(f"Left RMS: {ret_l:.4f}, Right RMS: {ret_r:.4f}")

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, D_l, K_r, D_r, img_size,
        criteria=criteria, flags=flags
    )
    print(f"Stereo RMS: {ret:.4f}")
    print(f"Baseline: {np.linalg.norm(T):.1f}mm")

    # Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T, alpha=0
    )

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    calib = {
        "image_size": list(img_size),
        "K_left": K_l.tolist(),
        "D_left": D_l.tolist(),
        "K_right": K_r.tolist(),
        "D_right": D_r.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "stereo_rms": ret,
        "left_rms": ret_l,
        "right_rms": ret_r,
        "baseline_mm": float(np.linalg.norm(T)),
        "pairs_used": good,
    }
    OUTPUT.write_text(json.dumps(calib, indent=2))
    print(f"\nSaved to {OUTPUT}")
    print("Calibration complete!")


if __name__ == "__main__":
    main()
