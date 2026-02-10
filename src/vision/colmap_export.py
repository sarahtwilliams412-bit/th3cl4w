"""COLMAP export — convert calibration results to COLMAP's format.

CRITICAL CONVENTION NOTE (per Claude Opus 4.6 Thinking unique finding):
  COLMAP expects **world-to-camera** transforms, but our calibration
  pipeline produces **camera-to-world** transforms. Getting this inversion
  wrong silently breaks reconstruction — COLMAP will "run" but produce
  garbage geometry.

  Our pipeline: T_world_cam (camera position/orientation in world)
  COLMAP wants: T_cam_world = inv(T_world_cam)

  Quaternion format: COLMAP uses (qw, qx, qy, qz) ordering.

Exports two files that COLMAP needs for known-pose reconstruction:
  1. cameras.txt — intrinsic parameters per camera
  2. images.txt — per-image extrinsics (world-to-camera quaternion + translation)

With these files, COLMAP can skip feature matching and pose estimation,
running only dense reconstruction (patch_match_stereo + stereo_fusion).

Usage:
    from src.vision.colmap_export import export_colmap_workspace
    export_colmap_workspace(
        image_dir="scan_data/images",
        camera_poses=list_of_T_world_cam,
        camera_matrix=K,
        dist_coeffs=dist,
        output_dir="scan_data/colmap_workspace",
    )
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz).

    Uses the Shepperd method for numerical stability.
    Returns COLMAP's quaternion ordering: (qw, qx, qy, qz).
    """
    # Ensure proper rotation matrix
    assert R.shape == (3, 3), f"Expected 3x3 rotation matrix, got {R.shape}"

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    # Normalize
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Convention: ensure qw >= 0
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz

    return float(qw), float(qx), float(qy), float(qz)


def camera_to_world_to_colmap(
    T_world_cam: np.ndarray,
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """Convert camera-to-world transform to COLMAP's world-to-camera format.

    CRITICAL: This inversion is the most common source of silent failure
    in photogrammetry pipelines. Our pipeline gives T_world_cam (where the
    camera IS in world). COLMAP wants T_cam_world (how to transform world
    points INTO the camera frame).

    Args:
        T_world_cam: 4x4 camera-to-world transform.
            Columns of R are camera axes in world frame.
            Translation is camera position in world frame.

    Returns:
        (quaternion, translation) in COLMAP's convention:
          quaternion: (qw, qx, qy, qz) of world-to-camera rotation
          translation: (3,) world-to-camera translation
    """
    # Invert: T_cam_world = inv(T_world_cam)
    R_wc = T_world_cam[:3, :3]  # camera-to-world rotation
    t_wc = T_world_cam[:3, 3]  # camera position in world

    # World-to-camera rotation: R_cw = R_wc^T
    R_cw = R_wc.T

    # World-to-camera translation: t_cw = -R_wc^T @ t_wc
    t_cw = -R_cw @ t_wc

    quat = rotation_matrix_to_quaternion(R_cw)

    return quat, t_cw


def export_cameras_txt(
    output_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    camera_id: int = 1,
) -> None:
    """Export COLMAP cameras.txt file.

    COLMAP camera models:
      SIMPLE_RADIAL: f, cx, cy, k1
      RADIAL: f, cx, cy, k1, k2
      OPENCV: fx, fy, cx, cy, k1, k2, p1, p2

    We use OPENCV model since we have full intrinsics.

    Args:
        output_path: Path to write cameras.txt.
        camera_matrix: 3x3 intrinsic matrix.
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3].
        image_size: (width, height).
        camera_id: COLMAP camera ID (1-indexed).
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    w, h = image_size

    dist = np.array(dist_coeffs).flatten()
    k1 = float(dist[0]) if len(dist) > 0 else 0.0
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    p1 = float(dist[2]) if len(dist) > 2 else 0.0
    p2 = float(dist[3]) if len(dist) > 3 else 0.0

    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: 1",
        f"{camera_id} OPENCV {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} "
        f"{k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f}",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Exported cameras.txt to %s", output_path)


def export_images_txt(
    output_path: str,
    image_names: list[str],
    camera_poses: list[np.ndarray],
    camera_id: int = 1,
) -> None:
    """Export COLMAP images.txt file with known camera poses.

    Each image gets two lines:
      Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      Line 2: (empty — no 2D point observations when importing known poses)

    CRITICAL: camera_poses must be T_world_cam (camera-to-world).
    This function handles the inversion to COLMAP's world-to-camera convention.

    Args:
        output_path: Path to write images.txt.
        image_names: List of image filenames (relative to image dir).
        camera_poses: List of 4x4 T_world_cam transforms.
        camera_id: COLMAP camera ID these images belong to.
    """
    assert len(image_names) == len(camera_poses), (
        f"Mismatch: {len(image_names)} images vs {len(camera_poses)} poses"
    )

    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(image_names)}",
    ]

    for i, (name, T_world_cam) in enumerate(
        zip(image_names, camera_poses)
    ):
        image_id = i + 1  # COLMAP uses 1-indexed IDs

        # Convert our camera-to-world to COLMAP's world-to-camera
        quat, t_cw = camera_to_world_to_colmap(T_world_cam)
        qw, qx, qy, qz = quat
        tx, ty, tz = float(t_cw[0]), float(t_cw[1]), float(t_cw[2])

        # Line 1: image pose
        lines.append(
            f"{image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
            f"{tx:.8f} {ty:.8f} {tz:.8f} {camera_id} {name}"
        )
        # Line 2: empty (no 2D observations)
        lines.append("")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(
        "Exported images.txt with %d images to %s",
        len(image_names),
        output_path,
    )


def export_points3D_txt(output_path: str) -> None:
    """Export empty points3D.txt (required by COLMAP even if empty)."""
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        "# Number of points: 0",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_colmap_workspace(
    image_dir: str,
    camera_poses: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int] = (1920, 1080),
    output_dir: Optional[str] = None,
    image_names: Optional[list[str]] = None,
) -> str:
    """Export a complete COLMAP workspace for known-pose reconstruction.

    Creates the directory structure COLMAP expects:
      output_dir/
        sparse/0/
          cameras.txt
          images.txt
          points3D.txt
        images/  (symlink or copy)

    After export, run:
      colmap patch_match_stereo --workspace_path output_dir
      colmap stereo_fusion --workspace_path output_dir --output_path fused.ply

    Args:
        image_dir: Directory containing the scan images.
        camera_poses: List of 4x4 T_world_cam transforms per image.
        camera_matrix: 3x3 intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        image_size: (width, height) of images.
        output_dir: COLMAP workspace directory. Defaults to image_dir/../colmap.
        image_names: Optional list of image filenames. If None, lists image_dir.

    Returns:
        Path to the COLMAP workspace directory.
    """
    if output_dir is None:
        output_dir = str(Path(image_dir).parent / "colmap")

    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # Get image names
    if image_names is None:
        img_dir = Path(image_dir)
        image_names = sorted(
            f.name
            for f in img_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

    if len(image_names) != len(camera_poses):
        logger.error(
            "Image count (%d) != pose count (%d)",
            len(image_names),
            len(camera_poses),
        )
        return output_dir

    # Export the three required files
    export_cameras_txt(
        os.path.join(sparse_dir, "cameras.txt"),
        camera_matrix,
        dist_coeffs,
        image_size,
    )
    export_images_txt(
        os.path.join(sparse_dir, "images.txt"),
        image_names,
        camera_poses,
    )
    export_points3D_txt(os.path.join(sparse_dir, "points3D.txt"))

    # Symlink images directory
    images_link = os.path.join(output_dir, "images")
    if not os.path.exists(images_link):
        try:
            os.symlink(os.path.abspath(image_dir), images_link)
        except OSError:
            logger.warning(
                "Could not create symlink for images — copy manually"
            )

    logger.info(
        "COLMAP workspace exported to %s (%d images)",
        output_dir,
        len(image_names),
    )
    logger.info(
        "Run: colmap patch_match_stereo --workspace_path %s", output_dir
    )
    logger.info(
        "Then: colmap stereo_fusion --workspace_path %s "
        "--output_path %s/fused.ply",
        output_dir,
        output_dir,
    )

    return output_dir
