"""Action retargeting — cross-embodiment transfer from Franka Panda to Unitree D1.

Converts actions from the NVIDIA PhysicalAI-Robotics-Kitchen-Sim-Demos
dataset (Franka Panda, OSC_POSE Cartesian deltas) to Unitree D1 joint-space
commands that can be processed by ActionDecoder.

The NVIDIA dataset actions are 7D:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
    where (dx..dyaw) are end-effector pose deltas in OSC_POSE space.

The D1 needs:
    [dj0, dj1, dj2, dj3, dj4, dj5, dgripper]
    where (dj0..dj5) are joint angle deltas in degrees.

Approach:
    1. Scale Cartesian deltas by workspace ratio (D1:550mm / Franka:855mm)
    2. Apply scaled deltas to D1's current EE pose (via FK)
    3. Solve IK for new EE target → new joint angles
    4. Compute joint deltas = new_joints - current_joints
    5. Clamp through ActionDecoder safety pipeline
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Workspace scaling
FRANKA_REACH_MM = 855.0
D1_REACH_MM = 550.0
WORKSPACE_SCALE = D1_REACH_MM / FRANKA_REACH_MM  # ~0.643

# Gripper scaling: Franka PandaGripper 0-0.04m, D1 0-65mm
FRANKA_GRIPPER_MAX_M = 0.04
D1_GRIPPER_MAX_MM = 65.0

# Maximum joint delta per step (degrees) — matches ActionDecoder
MAX_JOINT_DELTA_DEG = 10.0

# Joint limits with safety margin (degrees) — matches ActionDecoder
JOINT_LIMITS_DEG = {
    0: (-130.0, 130.0),
    1: (-80.0, 80.0),
    2: (-130.0, 130.0),
    3: (-130.0, 130.0),
    4: (-80.0, 80.0),
    5: (-130.0, 130.0),
}


@dataclass
class RetargetResult:
    """Result of retargeting a single action."""

    joint_deltas_deg: np.ndarray      # (6,) joint deltas in degrees
    gripper_delta_mm: float           # gripper delta in mm
    clamped: bool = False             # True if any delta was clamped
    cartesian_delta_scaled: Optional[np.ndarray] = None  # (6,) scaled Cartesian delta
    ik_converged: bool = True


@dataclass
class RetargetConfig:
    """Configuration for action retargeting."""

    workspace_scale: float = WORKSPACE_SCALE
    max_joint_delta_deg: float = MAX_JOINT_DELTA_DEG
    position_gain: float = 1.0       # Scaling factor for position deltas
    rotation_gain: float = 0.5       # Scaling factor for rotation deltas (more conservative)
    ik_max_iter: int = 100
    ik_damping: float = 0.05


class ActionRetargeter:
    """Retargets Franka Panda OSC_POSE actions to D1 joint-space actions."""

    def __init__(self, config: Optional[RetargetConfig] = None):
        self._config = config or RetargetConfig()
        self._kin = None  # Lazy-load kinematics

    def _ensure_kinematics(self):
        if self._kin is None:
            from shared.kinematics.kinematics import D1Kinematics
            self._kin = D1Kinematics()

    def retarget_cartesian(
        self,
        franka_action: np.ndarray,
        d1_joints_rad: np.ndarray,
    ) -> RetargetResult:
        """Retarget a Franka OSC_POSE action to D1 joint deltas.

        Args:
            franka_action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
                Position deltas in meters, rotation deltas in radians,
                gripper is normalized 0-1 or raw 0-0.04m.
            d1_joints_rad: (7,) current D1 joint angles in radians
                (6 arm joints + tool frame, as expected by D1Kinematics)

        Returns:
            RetargetResult with joint deltas in degrees and gripper delta in mm.
        """
        self._ensure_kinematics()
        cfg = self._config

        # Decompose Franka action
        pos_delta = franka_action[:3]       # (dx, dy, dz) in meters
        rot_delta = franka_action[3:6]      # (droll, dpitch, dyaw) in radians
        gripper_raw = franka_action[6]      # gripper command

        # Scale position deltas by workspace ratio
        pos_delta_scaled = pos_delta * cfg.workspace_scale * cfg.position_gain
        rot_delta_scaled = rot_delta * cfg.rotation_gain

        cartesian_delta = np.concatenate([pos_delta_scaled, rot_delta_scaled])

        # Current EE pose via FK
        T_current = self._kin.forward_kinematics(d1_joints_rad)

        # Apply Cartesian delta to get target EE pose
        T_target = T_current.copy()
        T_target[:3, 3] += pos_delta_scaled

        # Apply rotation delta (small-angle approximation)
        if np.linalg.norm(rot_delta_scaled) > 1e-6:
            from scipy.spatial.transform import Rotation
            R_delta = Rotation.from_rotvec(rot_delta_scaled).as_matrix()
            T_target[:3, :3] = R_delta @ T_current[:3, :3]

        # Solve IK for target pose
        q_new = self._kin.inverse_kinematics(
            T_target,
            q_init=d1_joints_rad,
            max_iter=cfg.ik_max_iter,
            damping=cfg.ik_damping,
        )

        # Compute joint deltas in radians → degrees
        q_deltas_rad = q_new[:6] - d1_joints_rad[:6]
        q_deltas_deg = np.degrees(q_deltas_rad)

        # Check IK convergence
        T_achieved = self._kin.forward_kinematics(q_new)
        pos_err = np.linalg.norm(T_target[:3, 3] - T_achieved[:3, 3])
        ik_converged = pos_err < 0.01  # 1cm threshold

        # Clamp joint deltas
        clamped = False
        for i in range(6):
            if abs(q_deltas_deg[i]) > cfg.max_joint_delta_deg:
                q_deltas_deg[i] = np.sign(q_deltas_deg[i]) * cfg.max_joint_delta_deg
                clamped = True

        # Verify resulting angles stay within limits
        current_deg = np.degrees(d1_joints_rad[:6])
        target_deg = current_deg + q_deltas_deg
        for i in range(6):
            lo, hi = JOINT_LIMITS_DEG[i]
            if target_deg[i] < lo:
                q_deltas_deg[i] = lo - current_deg[i]
                clamped = True
            elif target_deg[i] > hi:
                q_deltas_deg[i] = hi - current_deg[i]
                clamped = True

        # Retarget gripper
        gripper_delta_mm = self._retarget_gripper(gripper_raw)

        return RetargetResult(
            joint_deltas_deg=q_deltas_deg,
            gripper_delta_mm=gripper_delta_mm,
            clamped=clamped,
            cartesian_delta_scaled=cartesian_delta,
            ik_converged=ik_converged,
        )

    def retarget_joint_direct(
        self,
        franka_joint_deltas: np.ndarray,
        gripper_raw: float,
    ) -> RetargetResult:
        """Direct joint-to-joint mapping (fallback for joint-space datasets).

        Maps Franka 7-DOF joint deltas to D1 6-DOF joint deltas by
        discarding the redundant joint (Franka J3, elbow/nullspace).

        Args:
            franka_joint_deltas: (7,) Franka joint deltas in radians
            gripper_raw: Franka gripper command

        Returns:
            RetargetResult with joint deltas in degrees.
        """
        # Map Franka joints to D1 joints:
        # Franka: J1(shoulder), J2(shoulder), J3(elbow), J4(elbow/nullspace),
        #         J5(wrist), J6(wrist), J7(wrist)
        # D1:     J0(base yaw), J1(shoulder pitch), J2(elbow pitch),
        #         J3(forearm roll), J4(wrist pitch), J5(wrist roll)
        # Approximate mapping: Franka J1→D1 J0, J2→J1, J3→J2, J5→J3, J6→J4, J7→J5
        mapping = [0, 1, 2, 4, 5, 6]  # Franka indices → D1 joints (skip J3/nullspace)

        d1_deltas_rad = np.array([franka_joint_deltas[m] for m in mapping])
        d1_deltas_deg = np.degrees(d1_deltas_rad)

        # Scale by workspace ratio for safety
        d1_deltas_deg *= self._config.workspace_scale

        # Clamp
        clamped = False
        for i in range(6):
            if abs(d1_deltas_deg[i]) > self._config.max_joint_delta_deg:
                d1_deltas_deg[i] = np.sign(d1_deltas_deg[i]) * self._config.max_joint_delta_deg
                clamped = True

        gripper_delta_mm = self._retarget_gripper(gripper_raw)

        return RetargetResult(
            joint_deltas_deg=d1_deltas_deg,
            gripper_delta_mm=gripper_delta_mm,
            clamped=clamped,
        )

    def _retarget_gripper(self, gripper_raw: float) -> float:
        """Convert Franka gripper command to D1 gripper delta in mm.

        Franka gripper: typically normalized 0-1 or in meters 0-0.04.
        D1 gripper: 0-65mm.
        """
        # Detect if normalized (0-1) or in meters (0-0.04)
        if gripper_raw > 1.0:
            # Probably already in mm or some other unit
            return float(np.clip(gripper_raw, -D1_GRIPPER_MAX_MM, D1_GRIPPER_MAX_MM))

        if gripper_raw <= FRANKA_GRIPPER_MAX_M:
            # In meters: convert to D1 mm
            normalized = gripper_raw / FRANKA_GRIPPER_MAX_M
            return normalized * D1_GRIPPER_MAX_MM

        # Normalized 0-1
        return gripper_raw * D1_GRIPPER_MAX_MM

    def retarget_to_action_dicts(
        self,
        franka_action: np.ndarray,
        d1_joints_rad: np.ndarray,
        d1_gripper_mm: float,
    ) -> list:
        """Retarget and return action dicts compatible with ActionDecoder.decode().

        Returns a list of action dicts in the format:
            [{"type": "joint", "id": i, "delta": d, "reason": "retargeted"}, ...]
        """
        result = self.retarget_cartesian(franka_action, d1_joints_rad)
        actions = []

        for i in range(6):
            delta = float(result.joint_deltas_deg[i])
            if abs(delta) >= 0.5:  # ActionDecoder threshold
                actions.append({
                    "type": "joint",
                    "id": i,
                    "delta": round(delta, 1),
                    "reason": "retargeted from Franka OSC_POSE",
                })

        if abs(result.gripper_delta_mm) >= 1.0:
            target_mm = np.clip(d1_gripper_mm + result.gripper_delta_mm, 0.0, D1_GRIPPER_MAX_MM)
            actions.append({
                "type": "gripper",
                "position_mm": round(float(target_mm), 1),
                "reason": "retargeted gripper",
            })

        return actions


class LeRobotAdapter:
    """Reads NVIDIA LeRobot datasets and yields retargeted D1 episodes.

    Usage:
        adapter = LeRobotAdapter(Path("datasets/OpenCabinet/lerobot"))
        for episode in adapter.iter_episodes():
            # episode contains retargeted (observation, action) pairs
            for obs_state, action_dicts in episode:
                ...
    """

    def __init__(self, dataset_dir: Path, retarget_config: Optional[RetargetConfig] = None):
        self._dir = Path(dataset_dir)
        self._retargeter = ActionRetargeter(retarget_config)

    def load_info(self) -> dict:
        """Load dataset info.json metadata."""
        info_path = self._dir / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"No info.json at {info_path}")
        with open(info_path) as f:
            return json.load(f)

    def load_tasks(self) -> dict:
        """Load tasks.jsonl → {task_index: language_instruction}."""
        tasks_path = self._dir / "meta" / "tasks.jsonl"
        tasks = {}
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    obj = json.loads(line)
                    tasks[obj["task_index"]] = obj["language_instruction"]
        return tasks

    def iter_episodes_raw(self):
        """Iterate over episodes yielding raw (state, action, task) tuples.

        Yields:
            (episode_idx, task_str, frames) where frames is a list of
            (observation_state, action) numpy arrays.
        """
        import json

        info = self.load_info()
        tasks = self.load_tasks()

        # Find parquet files
        data_dir = self._dir / "data"
        if not data_dir.exists():
            logger.warning("No data directory at %s", data_dir)
            return

        try:
            import pyarrow.parquet as pq

            for chunk_dir in sorted(data_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                    table = pq.read_table(parquet_file)
                    df = table.to_pydict()

                    episode_idx = df["episode_index"][0] if df["episode_index"] else 0
                    task_idx = df["task_index"][0] if df["task_index"] else 0
                    task_str = tasks.get(task_idx, f"task_{task_idx}")

                    frames = []
                    for i in range(len(df["episode_index"])):
                        state = np.array(df["observation.state"][i], dtype=np.float32)
                        action = np.array(df["action"][i], dtype=np.float32)
                        frames.append((state, action))

                    yield episode_idx, task_str, frames

        except ImportError:
            logger.warning("pyarrow not available; trying JSONL fallback")
            for chunk_dir in sorted(data_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                for jsonl_file in sorted(chunk_dir.glob("*.jsonl")):
                    frames = []
                    episode_idx = 0
                    task_idx = 0
                    with open(jsonl_file) as f:
                        for line in f:
                            row = json.loads(line)
                            episode_idx = row["episode_index"]
                            task_idx = row["task_index"]
                            state = np.array(row["observation.state"], dtype=np.float32)
                            action = np.array(row["action"], dtype=np.float32)
                            frames.append((state, action))

                    task_str = tasks.get(task_idx, f"task_{task_idx}")
                    yield episode_idx, task_str, frames

    def iter_episodes_retargeted(self, initial_d1_joints_rad: Optional[np.ndarray] = None):
        """Iterate over episodes with actions retargeted to D1.

        Yields:
            (episode_idx, task_str, retargeted_frames) where each frame is
            (d1_state, retarget_result).
        """
        d1_joints = initial_d1_joints_rad
        if d1_joints is None:
            d1_joints = np.zeros(7)  # Home position

        for episode_idx, task_str, frames in self.iter_episodes_raw():
            retargeted = []
            q_current = d1_joints.copy()

            for state, action in frames:
                if len(action) >= 7:
                    result = self._retargeter.retarget_cartesian(action, q_current)
                    # Update simulated D1 state
                    q_current[:6] += np.radians(result.joint_deltas_deg)
                    retargeted.append((state, result))

            yield episode_idx, task_str, retargeted
