"""Point cloud generation from depth maps + camera poses.

Back-projects depth pixels to 3D using pinhole camera model and known poses.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default camera intrinsics for MX Brio at 640x480
# (approximate — should be replaced with calibrated values)
DEFAULT_INTRINSICS = {
    "fx": 600.0,
    "fy": 600.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480,
}

# Hand-eye calibration: camera frame relative to end-effector
# Camera is ~5cm behind gripper, same orientation (looking forward along EE z-axis)
HAND_EYE_OFFSET = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.05],  # 5cm behind (negative z in EE frame)
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def backproject_depth(
    depth_map: np.ndarray,
    rgb_frame: np.ndarray,
    intrinsics: Optional[dict] = None,
    camera_pose: Optional[np.ndarray] = None,
    max_depth: float = 2.0,
    min_depth: float = 0.05,
    subsample: int = 2,
) -> np.ndarray:
    """Back-project depth map to colored 3D point cloud.

    Parameters
    ----------
    depth_map : HxW float32, metric depth in meters
    rgb_frame : HxWx3 uint8, RGB image
    intrinsics : dict with fx, fy, cx, cy
    camera_pose : 4x4 camera-to-world transform (None = identity)
    max_depth : max depth cutoff in meters
    min_depth : min depth cutoff in meters
    subsample : take every Nth pixel (speed vs density tradeoff)

    Returns
    -------
    points : Nx6 float32 array [x, y, z, r, g, b] in world frame
    """
    if intrinsics is None:
        intrinsics = DEFAULT_INTRINSICS

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    h, w = depth_map.shape[:2]

    # Create pixel grid
    v, u = np.mgrid[0:h:subsample, 0:w:subsample]
    u = u.flatten().astype(np.float32)
    v = v.flatten().astype(np.float32)

    # Sample depth and color
    d = depth_map[v.astype(int), u.astype(int)]

    # Filter by depth range
    valid = (d > min_depth) & (d < max_depth) & np.isfinite(d)
    u, v, d = u[valid], v[valid], d[valid]

    if len(d) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Back-project to camera frame (pinhole model)
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

    # Stack as Nx3
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Transform to world frame
    if camera_pose is not None:
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        pts_world = (R @ pts_cam.T).T + t
    else:
        pts_world = pts_cam

    # Sample colors
    rgb = rgb_frame[v.astype(int), u.astype(int)]  # Nx3 uint8
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)

    # Combine: Nx6 [x,y,z,r,g,b]
    points = np.concatenate([pts_world, rgb], axis=-1).astype(np.float32)
    return points


def compute_camera_pose_from_joints(
    joint_angles_deg: list,
    hand_eye_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute camera world pose from arm joint angles via FK + hand-eye offset.

    Parameters
    ----------
    joint_angles_deg : 6 joint angles in degrees
    hand_eye_offset : 4x4 camera-to-EE transform (None = use default)

    Returns
    -------
    T_cam_world : 4x4 camera pose in world frame
    """
    try:
        from shared.kinematics.kinematics import D1Kinematics
    except ImportError:
        logger.warning("Kinematics module not available, returning identity pose")
        return np.eye(4)

    if hand_eye_offset is None:
        hand_eye_offset = HAND_EYE_OFFSET

    kin = D1Kinematics()
    # Convert to radians - kinematics uses 7 joints but we have 6
    angles_rad = np.radians(joint_angles_deg[:6])
    # Pad to 7 joints if needed (wrist roll = 0)
    if len(angles_rad) < 7:
        angles_rad = np.concatenate([angles_rad, np.zeros(7 - len(angles_rad))])

    T_ee = kin.forward_kinematics(angles_rad)
    T_cam = T_ee @ hand_eye_offset

    return T_cam


def merge_point_clouds(clouds: list[np.ndarray]) -> np.ndarray:
    """Concatenate multiple Nx6 point clouds into one.

    Parameters
    ----------
    clouds : list of Nx6 float32 arrays

    Returns
    -------
    merged : Mx6 float32 array
    """
    valid = [c for c in clouds if c is not None and len(c) > 0]
    if not valid:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(valid, axis=0)


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.005) -> np.ndarray:
    """Voxel grid downsampling of point cloud.

    Uses Open3D if available, otherwise falls back to a simple grid-based approach.

    Parameters
    ----------
    points : Nx6 [x,y,z,r,g,b]
    voxel_size : voxel edge length in meters

    Returns
    -------
    downsampled : Mx6 array
    """
    if len(points) == 0:
        return points

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)

        pcd_down = pcd.voxel_down_sample(voxel_size)

        pts = np.asarray(pcd_down.points, dtype=np.float32)
        colors = (np.asarray(pcd_down.colors) * 255).astype(np.float32)
        return np.concatenate([pts, colors], axis=-1)

    except ImportError:
        logger.info("Open3D not available, using simple grid downsampling")
        # Simple voxel grid: hash each point to a voxel, keep first
        keys = (points[:, :3] / voxel_size).astype(np.int32)
        _, idx = np.unique(
            keys[:, 0] * 1000000 + keys[:, 1] * 1000 + keys[:, 2],
            return_index=True,
        )
        return points[idx]


def save_ply(points: np.ndarray, filepath: str) -> bool:
    """Save Nx6 point cloud as PLY file.

    Parameters
    ----------
    points : Nx6 [x,y,z,r,g,b]
    filepath : output .ply path

    Returns
    -------
    success : bool
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
        o3d.io.write_point_cloud(filepath, pcd)
        logger.info("Saved PLY: %s (%d points)", filepath, len(points))
        return True

    except ImportError:
        # Manual PLY write
        n = len(points)
        header = (
            f"ply\n"
            f"format ascii 1.0\n"
            f"element vertex {n}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property uchar red\n"
            f"property uchar green\n"
            f"property uchar blue\n"
            f"end_header\n"
        )
        with open(filepath, "w") as f:
            f.write(header)
            for p in points:
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} " f"{int(p[3])} {int(p[4])} {int(p[5])}\n"
                )
        logger.info("Saved PLY (manual): %s (%d points)", filepath, n)
        return True

    except Exception as e:
        logger.error("Failed to save PLY: %s", e)
        return False
