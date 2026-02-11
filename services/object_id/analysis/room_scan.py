"""Room scan trajectory generator using Fibonacci sphere sampling.

Generates evenly distributed viewpoints on a hemisphere for photogrammetry
scanning. Uses the Fibonacci (golden angle) spiral to avoid polar clustering
that plagues uniform angular sampling — mathematically optimal for coverage.

Per Gemini 3 Pro: Fibonacci sphere sampling produces the most geometrically
stable reconstructions in COLMAP by ensuring uniform spatial distribution
of viewpoints.

The D1 arm's workspace is a hemisphere (~550mm reach). Scan viewpoints
are generated as:
  - Points on a Fibonacci hemisphere pointing inward at workspace center
  - Multiple radii for baseline diversity (photogrammetry needs ~10-30deg
    baseline between views)
  - Camera always aimed at the workspace center

Output: List of (position, look_at_target) pairs that can be converted
to joint angles via IK.

Usage:
    from services.object_id.analysis.room_scan import generate_scan_trajectory
    poses = generate_scan_trajectory(num_points=50, radii=[0.30, 0.45])
    for pos, target in poses:
        # Convert to joint angles via IK and execute
        pass
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Golden ratio for Fibonacci sampling
PHI = (1 + math.sqrt(5)) / 2

# D1 arm workspace constraints
ARM_REACH_M = 0.550  # 550mm max reach
ARM_BASE_HEIGHT_M = 0.1215  # base to shoulder height


@dataclass
class ScanViewpoint:
    """A single scan viewpoint."""

    position: np.ndarray  # (3,) XYZ in base frame (meters)
    look_at: np.ndarray  # (3,) target point
    azimuth_deg: float  # for logging
    elevation_deg: float  # for logging
    radius_m: float


def fibonacci_hemisphere_points(
    n: int,
    radius: float = 1.0,
    min_elevation_deg: float = -10.0,
    max_elevation_deg: float = 75.0,
) -> list[tuple[np.ndarray, float, float]]:
    """Generate n points on a hemisphere using Fibonacci spiral sampling.

    Points are distributed on a spherical cap defined by the elevation
    range, with the Fibonacci spiral ensuring near-uniform spacing.

    Args:
        n: Number of points to generate.
        radius: Sphere radius in meters.
        min_elevation_deg: Minimum elevation angle (negative = below horizontal).
        max_elevation_deg: Maximum elevation angle.

    Returns:
        List of (position_xyz, azimuth_deg, elevation_deg) tuples.
    """
    # Convert elevation range to cos(theta) range
    # elevation = 90 - theta (theta from pole)
    # cos(theta) = sin(elevation)
    sin_min = math.sin(math.radians(min_elevation_deg))
    sin_max = math.sin(math.radians(max_elevation_deg))

    points = []
    for i in range(n):
        # Fibonacci-distributed z (height) coordinate
        # Map i to [sin_min, sin_max] range
        t = i / (n - 1) if n > 1 else 0.5
        sin_elev = sin_min + t * (sin_max - sin_min)
        elev_rad = math.asin(max(-1, min(1, sin_elev)))
        elev_deg = math.degrees(elev_rad)

        # Golden angle azimuthal distribution
        az_rad = 2 * math.pi * i / PHI
        az_deg = math.degrees(az_rad) % 360

        # Spherical to Cartesian
        cos_elev = math.cos(elev_rad)
        x = radius * cos_elev * math.cos(az_rad)
        y = radius * cos_elev * math.sin(az_rad)
        z = radius * sin_elev

        points.append((np.array([x, y, z]), az_deg, elev_deg))

    return points


def generate_scan_trajectory(
    num_points: int = 50,
    radii: Optional[list[float]] = None,
    workspace_center: Optional[np.ndarray] = None,
    min_elevation_deg: float = -10.0,
    max_elevation_deg: float = 65.0,
    azimuth_range_deg: tuple[float, float] = (-150.0, 150.0),
) -> list[ScanViewpoint]:
    """Generate a room scan trajectory for photogrammetry.

    Uses Fibonacci sphere sampling to produce evenly distributed viewpoints
    on hemispheres at multiple radii around the workspace center.

    Args:
        num_points: Total number of viewpoints per radius shell.
        radii: List of radii in meters. Defaults to [0.30, 0.45].
        workspace_center: XYZ center of workspace in base frame.
            Defaults to (0.2, 0, 0.05) — roughly center of desk area
            in front of the robot.
        min_elevation_deg: Minimum elevation (-10 allows slight downward tilt).
        max_elevation_deg: Maximum elevation (65 avoids extreme overhead).
        azimuth_range_deg: (min, max) azimuth in degrees. The D1 can't
            reach directly behind itself, so we limit to front hemisphere.

    Returns:
        List of ScanViewpoint objects with position and look-at target.
    """
    if radii is None:
        radii = [0.30, 0.45]
    if workspace_center is None:
        workspace_center = np.array([0.20, 0.0, 0.05])

    az_min, az_max = azimuth_range_deg

    all_viewpoints = []

    for radius in radii:
        if radius > ARM_REACH_M:
            logger.warning(
                "Radius %.3fm exceeds arm reach %.3fm — skipping",
                radius,
                ARM_REACH_M,
            )
            continue

        raw_points = fibonacci_hemisphere_points(
            num_points, radius, min_elevation_deg, max_elevation_deg
        )

        for pos, az_deg, elev_deg in raw_points:
            # Filter by azimuth range (arm can't reach behind itself)
            # Normalize azimuth to [-180, 180]
            az_norm = ((az_deg + 180) % 360) - 180
            if az_norm < az_min or az_norm > az_max:
                continue

            # Offset position relative to workspace center
            camera_pos = workspace_center + pos

            # Camera height must be above table surface
            if camera_pos[2] < 0.02:
                continue

            viewpoint = ScanViewpoint(
                position=camera_pos,
                look_at=workspace_center.copy(),
                azimuth_deg=az_norm,
                elevation_deg=elev_deg,
                radius_m=radius,
            )
            all_viewpoints.append(viewpoint)

    # Sort by azimuth then elevation for smooth trajectory
    all_viewpoints.sort(key=lambda v: (v.radius_m, v.azimuth_deg))

    logger.info(
        "Generated %d scan viewpoints across %d radius shells",
        len(all_viewpoints),
        len(radii),
    )

    return all_viewpoints


def generate_tabletop_detail_pass(
    grid_spacing_m: float = 0.05,
    height_m: float = 0.20,
    x_range: tuple[float, float] = (0.05, 0.40),
    y_range: tuple[float, float] = (-0.20, 0.20),
    camera_tilt_deg: float = 30.0,
) -> list[ScanViewpoint]:
    """Generate a tabletop detail pass — raster grid looking down.

    For high-resolution capture of objects on the desk surface.

    Args:
        grid_spacing_m: Distance between adjacent viewpoints.
        height_m: Camera height above table.
        x_range: (min, max) X extent in base frame.
        y_range: (min, max) Y extent in base frame.
        camera_tilt_deg: Camera tilt from vertical (0 = straight down).

    Returns:
        List of ScanViewpoint objects.
    """
    viewpoints = []

    x_start, x_end = x_range
    y_start, y_end = y_range

    x = x_start
    row = 0
    while x <= x_end:
        y = y_start if row % 2 == 0 else y_end
        y_step = grid_spacing_m if row % 2 == 0 else -grid_spacing_m

        while (y_step > 0 and y <= y_end) or (y_step < 0 and y >= y_start):
            pos = np.array([x, y, height_m])
            # Look at a point directly below, offset slightly forward
            tilt_offset = height_m * math.tan(math.radians(camera_tilt_deg))
            look_at = np.array([x + tilt_offset, y, 0.0])

            viewpoints.append(
                ScanViewpoint(
                    position=pos,
                    look_at=look_at,
                    azimuth_deg=0.0,
                    elevation_deg=90.0 - camera_tilt_deg,
                    radius_m=height_m,
                )
            )
            y += y_step

        x += grid_spacing_m
        row += 1

    logger.info(
        "Generated %d tabletop detail viewpoints (%.0fmm grid at %.0fmm height)",
        len(viewpoints),
        grid_spacing_m * 1000,
        height_m * 1000,
    )

    return viewpoints


def viewpoint_to_camera_pose(
    viewpoint: ScanViewpoint,
) -> np.ndarray:
    """Convert a scan viewpoint to a 4x4 camera-to-world transform.

    The camera frame convention is: Z forward (optical axis), X right, Y down.
    The camera looks from viewpoint.position toward viewpoint.look_at.

    Returns:
        4x4 T_world_cam homogeneous transform.
    """
    pos = viewpoint.position
    target = viewpoint.look_at

    # Camera forward = normalized direction from position to target
    forward = target - pos
    forward = forward / np.linalg.norm(forward)

    # Up reference (world Z up)
    up_ref = np.array([0.0, 0.0, 1.0])

    # Camera right = forward x up (right-handed)
    right = np.cross(forward, up_ref)
    if np.linalg.norm(right) < 1e-6:
        # Looking straight up/down — use X as up reference
        up_ref = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up_ref)
    right = right / np.linalg.norm(right)

    # Camera down = right x forward
    down = np.cross(right, forward)
    down = down / np.linalg.norm(down)

    # Build rotation matrix: columns are camera axes in world frame
    # Camera convention: X=right, Y=down, Z=forward
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = forward
    T[:3, 3] = pos

    return T
