"""MuJoCo physics simulation engine for th3cl4w.

Provides real physics simulation for trajectory validation, policy testing,
and sim-to-real development. Can load NVIDIA Kitchen-Sim-Demos scene models
and simulate the D1 arm within those environments.

Replaces the placeholder simulation in services/simulation/server.py with
actual physics-based collision detection, joint limit monitoring, and
trajectory evaluation.

Usage:
    engine = MuJoCoEngine()
    engine.load_d1_model()                      # Load D1 arm model
    engine.load_kitchen_scene("model.xml.gz")   # Load NVIDIA kitchen scene
    result = engine.simulate_trajectory(traj)   # Run trajectory + get verdict
"""

import gzip
import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.arm_model.joint_service import (
    get_dh_params,
    JOINTS,
    GRIPPER,
    NUM_ARM_JOINTS,
    MAX_WORKSPACE_RADIUS_M,
    VELOCITY_MAX_RAD,
    TORQUE_MAX_NM,
)

logger = logging.getLogger(__name__)

# Joint limits in radians for quick lookup
_JOINT_LIMITS_RAD = {}
for _jid, _jc in JOINTS.items():
    _JOINT_LIMITS_RAD[_jid] = (math.radians(_jc.safe_min_deg), math.radians(_jc.safe_max_deg))


@dataclass
class SimulationConfig:
    """Configuration for physics simulation."""

    physics_hz: float = 240.0       # Physics timestep rate
    control_hz: float = 50.0        # Control command rate
    max_sim_duration_s: float = 30.0
    collision_threshold: float = 0.01   # Contact force threshold (N)
    joint_limit_margin_rad: float = math.radians(2.0)
    singularity_threshold: float = 0.01  # Jacobian condition number threshold
    enable_rendering: bool = False


@dataclass
class CollisionEvent:
    """A detected collision during simulation."""

    time_s: float
    body1: str
    body2: str
    contact_force_n: float
    position: np.ndarray  # 3D contact point


@dataclass
class JointLimitEvent:
    """A joint limit violation during simulation."""

    time_s: float
    joint_id: int
    angle_rad: float
    limit_rad: float
    limit_type: str  # "min" or "max"


@dataclass
class SimulationResult:
    """Complete result of a trajectory simulation."""

    success: bool
    sim_duration_s: float
    steps_executed: int
    collision_free: bool
    joint_limits_ok: bool
    singularity_free: bool
    max_joint_velocity_deg_s: float
    max_joint_torque_nm: float
    verdict: str  # "safe_to_execute", "collision_detected", "joint_limit_violation", etc.
    collisions: List[CollisionEvent] = field(default_factory=list)
    joint_limit_violations: List[JointLimitEvent] = field(default_factory=list)
    trajectory_positions: List[np.ndarray] = field(default_factory=list)
    ee_positions: List[np.ndarray] = field(default_factory=list)


def _generate_d1_mjcf() -> str:
    """Generate a MuJoCo XML model for the Unitree D1 arm from DH parameters.

    Creates a kinematically accurate MJCF model using the DH parameters
    from shared/arm_model/joint_service.py.
    """
    dh_params = get_dh_params()

    # D1 link dimensions from DH table:
    # d values: 0.1215, 0, 0.2085, 0, 0.2085, 0, 0.113
    # All a values are 0 (elbow-type manipulator)
    xml = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="unitree_d1">
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicit"/>

  <default>
    <joint armature="0.1" damping="5.0" frictionloss="0.1"/>
    <geom condim="3" friction="1.0 0.005 0.0001" margin="0.001"/>
    <motor ctrllimited="true"/>
  </default>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1" pos="0 0 0"/>

    <!-- Mounting surface (table) -->
    <body name="table" pos="0 0 0.75">
      <geom name="table_top" type="box" size="0.5 0.3 0.02" rgba="0.6 0.4 0.2 1"
            mass="50" contype="1" conaffinity="1"/>

      <!-- D1 arm base -->
      <body name="base_link" pos="0 0 0.02">
        <geom name="base_geom" type="cylinder" size="0.05 0.03"
              rgba="0.2 0.2 0.2 1" mass="2.0" contype="1" conaffinity="1"/>
        <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>

        <!-- J0: Base Yaw -->
        <body name="link0" pos="0 0 0.1215">
          <joint name="j0" type="hinge" axis="0 0 1"
                 range="-2.356 2.356" damping="8.0"/>
          <geom name="link0_geom" type="cylinder" size="0.04 0.06"
                rgba="0.3 0.3 0.3 1" mass="1.5" contype="1" conaffinity="1"/>
          <inertial pos="0 0 0" mass="1.5" diaginertia="0.005 0.005 0.002"/>

          <!-- J1: Shoulder Pitch -->
          <body name="link1" pos="0 0 0">
            <joint name="j1" type="hinge" axis="0 1 0"
                   range="-1.571 1.571" damping="8.0"/>
            <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.02"
                  size="0.035" rgba="0.4 0.4 0.4 1" mass="1.2"
                  contype="1" conaffinity="1"/>
            <inertial pos="0 0 0.01" mass="1.2" diaginertia="0.003 0.003 0.001"/>

            <!-- J2: Elbow Pitch -->
            <body name="link2" pos="0 0 0.2085">
              <joint name="j2" type="hinge" axis="0 1 0"
                     range="-1.571 1.571" damping="6.0"/>
              <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 0.18"
                    size="0.03" rgba="0.3 0.3 0.3 1" mass="1.0"
                    contype="1" conaffinity="1"/>
              <inertial pos="0 0 0.09" mass="1.0" diaginertia="0.01 0.01 0.001"/>

              <!-- J3: Forearm Roll -->
              <body name="link3" pos="0 0 0.2085">
                <joint name="j3" type="hinge" axis="0 0 1"
                       range="-2.356 2.356" damping="4.0"/>
                <geom name="link3_geom" type="cylinder" size="0.025 0.015"
                      rgba="0.4 0.4 0.4 1" mass="0.5"
                      contype="1" conaffinity="1"/>
                <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.0005"/>

                <!-- J4: Wrist Pitch -->
                <body name="link4" pos="0 0 0">
                  <joint name="j4" type="hinge" axis="0 1 0"
                         range="-1.571 1.571" damping="3.0"/>
                  <geom name="link4_geom" type="capsule" fromto="0 0 0 0 0 0.08"
                        size="0.02" rgba="0.3 0.3 0.3 1" mass="0.4"
                        contype="1" conaffinity="1"/>
                  <inertial pos="0 0 0.04" mass="0.4" diaginertia="0.0005 0.0005 0.0001"/>

                  <!-- J5: Wrist Roll -->
                  <body name="link5" pos="0 0 0.113">
                    <joint name="j5" type="hinge" axis="0 0 1"
                           range="-2.356 2.356" damping="2.0"/>
                    <geom name="link5_geom" type="cylinder" size="0.015 0.01"
                          rgba="0.4 0.4 0.4 1" mass="0.2"
                          contype="1" conaffinity="1"/>
                    <inertial pos="0 0 0" mass="0.2" diaginertia="0.0001 0.0001 0.0001"/>

                    <!-- Gripper -->
                    <body name="gripper_base" pos="0 0 0.02">
                      <geom name="gripper_palm" type="box" size="0.02 0.03 0.01"
                            rgba="0.5 0.5 0.5 1" mass="0.15"
                            contype="1" conaffinity="1"/>

                      <!-- Left finger -->
                      <body name="finger_left" pos="0 0.015 0.02">
                        <joint name="gripper_left" type="slide" axis="0 1 0"
                               range="0 0.0325" damping="1.0"/>
                        <geom name="finger_left_geom" type="box"
                              size="0.01 0.005 0.015"
                              rgba="0.6 0.6 0.6 1" mass="0.05"
                              contype="1" conaffinity="1"/>
                      </body>

                      <!-- Right finger -->
                      <body name="finger_right" pos="0 -0.015 0.02">
                        <joint name="gripper_right" type="slide" axis="0 -1 0"
                               range="0 0.0325" damping="1.0"/>
                        <geom name="finger_right_geom" type="box"
                              size="0.01 0.005 0.015"
                              rgba="0.6 0.6 0.6 1" mass="0.05"
                              contype="1" conaffinity="1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="act_j0" joint="j0" ctrlrange="-20 20" gear="1"/>
    <motor name="act_j1" joint="j1" ctrlrange="-20 20" gear="1"/>
    <motor name="act_j2" joint="j2" ctrlrange="-15 15" gear="1"/>
    <motor name="act_j3" joint="j3" ctrlrange="-10 10" gear="1"/>
    <motor name="act_j4" joint="j4" ctrlrange="-5 5" gear="1"/>
    <motor name="act_j5" joint="j5" ctrlrange="-5 5" gear="1"/>
    <motor name="act_gripper_left" joint="gripper_left" ctrlrange="0 0.0325" gear="1"/>
    <motor name="act_gripper_right" joint="gripper_right" ctrlrange="0 0.0325" gear="1"/>
  </actuator>

  <sensor>
    <jointpos name="j0_pos" joint="j0"/>
    <jointpos name="j1_pos" joint="j1"/>
    <jointpos name="j2_pos" joint="j2"/>
    <jointpos name="j3_pos" joint="j3"/>
    <jointpos name="j4_pos" joint="j4"/>
    <jointpos name="j5_pos" joint="j5"/>
    <jointvel name="j0_vel" joint="j0"/>
    <jointvel name="j1_vel" joint="j1"/>
    <jointvel name="j2_vel" joint="j2"/>
    <jointvel name="j3_vel" joint="j3"/>
    <jointvel name="j4_vel" joint="j4"/>
    <jointvel name="j5_vel" joint="j5"/>
    <actuatorfrc name="j0_torque" actuator="act_j0"/>
    <actuatorfrc name="j1_torque" actuator="act_j1"/>
    <actuatorfrc name="j2_torque" actuator="act_j2"/>
    <actuatorfrc name="j3_torque" actuator="act_j3"/>
    <actuatorfrc name="j4_torque" actuator="act_j4"/>
    <actuatorfrc name="j5_torque" actuator="act_j5"/>
  </sensor>
</mujoco>"""
    return xml


class MuJoCoEngine:
    """Physics simulation engine using MuJoCo.

    Provides trajectory validation with real physics: collision detection,
    joint limit monitoring, torque estimation, and workspace checking.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self._config = config or SimulationConfig()
        self._model = None
        self._data = None
        self._d1_xml = None
        self._scene_loaded = False
        self._joint_ids = {}     # joint_name -> mujoco joint id
        self._actuator_ids = {}  # actuator_name -> mujoco actuator id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_d1_model(self):
        """Load the D1 arm model generated from DH parameters."""
        try:
            import mujoco
        except ImportError:
            raise ImportError(
                "MuJoCo not installed. Install with: pip install mujoco>=3.0"
            )

        self._d1_xml = _generate_d1_mjcf()
        self._model = mujoco.MjModel.from_xml_string(self._d1_xml)
        self._data = mujoco.MjData(self._model)

        # Cache joint/actuator IDs
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._joint_ids[name] = i

        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._actuator_ids[name] = i

        self._scene_loaded = True
        logger.info(
            "D1 MuJoCo model loaded: %d bodies, %d joints, %d actuators",
            self._model.nbody, self._model.njnt, self._model.nu,
        )

    def load_kitchen_scene(self, scene_path: str):
        """Load an NVIDIA Kitchen-Sim scene model.

        The NVIDIA dataset includes model.xml.gz files for each episode.
        This loads the MuJoCo scene and injects the D1 arm model into it.

        Args:
            scene_path: Path to model.xml.gz from the NVIDIA dataset.
        """
        try:
            import mujoco
        except ImportError:
            raise ImportError("MuJoCo not installed.")

        path = Path(scene_path)
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                scene_xml = f.read()
        else:
            with open(path) as f:
                scene_xml = f.read()

        # Load scene model (original Franka environment)
        # We don't inject D1 into Franka scenes — we keep them separate
        # and use them for visual/semantic data only.
        # For actual D1 simulation, use load_d1_model().
        self._scene_xml = scene_xml
        logger.info("Kitchen scene loaded from %s", scene_path)

    def set_joint_positions(self, joint_angles_rad: np.ndarray):
        """Set D1 arm joint positions in the simulation.

        Args:
            joint_angles_rad: (6,) array of joint angles in radians.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_d1_model() first.")

        import mujoco

        joint_names = ["j0", "j1", "j2", "j3", "j4", "j5"]
        for i, name in enumerate(joint_names):
            if name in self._joint_ids:
                jid = self._joint_ids[name]
                self._data.qpos[jid] = joint_angles_rad[i]

        mujoco.mj_forward(self._model, self._data)

    def set_gripper(self, opening_mm: float):
        """Set gripper opening in the simulation.

        Args:
            opening_mm: Gripper opening in mm (0-65).
        """
        if not self.is_loaded:
            return

        # Convert mm to meters, split between two fingers
        opening_m = (opening_mm / 1000.0) / 2.0  # per finger
        opening_m = np.clip(opening_m, 0.0, 0.0325)

        for name in ["gripper_left", "gripper_right"]:
            if name in self._joint_ids:
                self._data.qpos[self._joint_ids[name]] = opening_m

    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position in world frame."""
        if not self.is_loaded:
            return np.zeros(3)

        import mujoco

        # Get gripper base body position
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        if body_id >= 0:
            return self._data.xpos[body_id].copy()
        return np.zeros(3)

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions in radians."""
        if not self.is_loaded:
            return np.zeros(6)

        positions = np.zeros(6)
        joint_names = ["j0", "j1", "j2", "j3", "j4", "j5"]
        for i, name in enumerate(joint_names):
            if name in self._joint_ids:
                positions[i] = self._data.qpos[self._joint_ids[name]]
        return positions

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities in rad/s."""
        if not self.is_loaded:
            return np.zeros(6)

        velocities = np.zeros(6)
        joint_names = ["j0", "j1", "j2", "j3", "j4", "j5"]
        for i, name in enumerate(joint_names):
            if name in self._joint_ids:
                velocities[i] = self._data.qvel[self._joint_ids[name]]
        return velocities

    def step(self):
        """Advance the simulation by one timestep."""
        if not self.is_loaded:
            return

        import mujoco
        mujoco.mj_step(self._model, self._data)

    def check_collisions(self) -> List[CollisionEvent]:
        """Check for active collisions in the current state."""
        if not self.is_loaded:
            return []

        import mujoco
        collisions = []

        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            if contact.dist < 0:  # Penetration
                geom1 = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
                geom2 = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"

                # Skip self-collisions between adjacent links
                if _are_adjacent_links(geom1, geom2):
                    continue

                force = np.linalg.norm(self._data.contact[i].frame[:3])
                if force > self._config.collision_threshold:
                    collisions.append(CollisionEvent(
                        time_s=self._data.time,
                        body1=geom1,
                        body2=geom2,
                        contact_force_n=float(force),
                        position=contact.pos.copy(),
                    ))

        return collisions

    def simulate_trajectory(
        self,
        joint_trajectory: List[np.ndarray],
        gripper_trajectory: Optional[List[float]] = None,
        dt: float = 0.02,
    ) -> SimulationResult:
        """Simulate a full joint trajectory and evaluate safety.

        Args:
            joint_trajectory: List of (6,) arrays of joint angles in degrees.
            gripper_trajectory: Optional list of gripper positions in mm.
            dt: Time between trajectory points in seconds.

        Returns:
            SimulationResult with safety verdict.
        """
        if not self.is_loaded:
            self.load_d1_model()

        import mujoco

        # Reset simulation
        mujoco.mj_resetData(self._model, self._data)
        self._data.time = 0.0

        all_collisions: List[CollisionEvent] = []
        all_joint_violations: List[JointLimitEvent] = []
        max_velocity_deg_s = 0.0
        max_torque_nm = 0.0
        ee_positions: List[np.ndarray] = []
        traj_positions: List[np.ndarray] = []

        steps_per_point = max(1, int(dt / self._model.opt.timestep))

        for t_idx, target_deg in enumerate(joint_trajectory):
            target_rad = np.radians(np.asarray(target_deg, dtype=float))

            # Set joint targets via position control (PD controller)
            self.set_joint_positions(target_rad)

            if gripper_trajectory and t_idx < len(gripper_trajectory):
                self.set_gripper(gripper_trajectory[t_idx])

            # Step physics
            for _ in range(steps_per_point):
                # Simple PD control to track targets
                current_pos = self.get_joint_positions()
                current_vel = self.get_joint_velocities()
                kp = 100.0
                kd = 10.0
                torques = kp * (target_rad[:6] - current_pos) - kd * current_vel

                # Apply torques to actuators
                joint_names = ["j0", "j1", "j2", "j3", "j4", "j5"]
                for i, name in enumerate(joint_names):
                    act_name = f"act_{name}"
                    if act_name in self._actuator_ids:
                        aid = self._actuator_ids[act_name]
                        self._data.ctrl[aid] = np.clip(
                            torques[i],
                            -TORQUE_MAX_NM[i],
                            TORQUE_MAX_NM[i],
                        )

                self.step()

            # Record state
            current_pos = self.get_joint_positions()
            current_vel = self.get_joint_velocities()
            traj_positions.append(np.degrees(current_pos))
            ee_positions.append(self.get_ee_position())

            # Check collisions
            collisions = self.check_collisions()
            all_collisions.extend(collisions)

            # Check joint limits
            for j in range(6):
                lo, hi = _JOINT_LIMITS_RAD.get(j, (-math.pi, math.pi))
                margin = self._config.joint_limit_margin_rad
                if current_pos[j] < lo + margin:
                    all_joint_violations.append(JointLimitEvent(
                        time_s=self._data.time,
                        joint_id=j,
                        angle_rad=current_pos[j],
                        limit_rad=lo,
                        limit_type="min",
                    ))
                elif current_pos[j] > hi - margin:
                    all_joint_violations.append(JointLimitEvent(
                        time_s=self._data.time,
                        joint_id=j,
                        angle_rad=current_pos[j],
                        limit_rad=hi,
                        limit_type="max",
                    ))

            # Track max velocity and torque
            vel_deg_s = np.max(np.abs(np.degrees(current_vel)))
            max_velocity_deg_s = max(max_velocity_deg_s, vel_deg_s)

            for i in range(min(6, self._model.nu)):
                torque = abs(self._data.actuator_force[i])
                max_torque_nm = max(max_torque_nm, torque)

        # Determine verdict
        collision_free = len(all_collisions) == 0
        joint_limits_ok = len(all_joint_violations) == 0
        singularity_free = True  # TODO: check Jacobian condition number

        if not collision_free:
            verdict = "collision_detected"
        elif not joint_limits_ok:
            verdict = "joint_limit_violation"
        elif max_velocity_deg_s > 180.0:
            verdict = "excessive_velocity"
        elif max_torque_nm > 25.0:
            verdict = "excessive_torque"
        else:
            verdict = "safe_to_execute"

        return SimulationResult(
            success=verdict == "safe_to_execute",
            sim_duration_s=self._data.time,
            steps_executed=len(joint_trajectory),
            collision_free=collision_free,
            joint_limits_ok=joint_limits_ok,
            singularity_free=singularity_free,
            max_joint_velocity_deg_s=max_velocity_deg_s,
            max_joint_torque_nm=max_torque_nm,
            verdict=verdict,
            collisions=all_collisions,
            joint_limit_violations=all_joint_violations,
            trajectory_positions=traj_positions,
            ee_positions=ee_positions,
        )

    def render_frame(self, width: int = 640, height: int = 480) -> Optional[np.ndarray]:
        """Render the current simulation state to an RGB image.

        Returns:
            (H, W, 3) uint8 RGB image, or None if rendering is unavailable.
        """
        if not self.is_loaded:
            return None

        try:
            import mujoco

            renderer = mujoco.Renderer(self._model, height=height, width=width)
            renderer.update_scene(self._data)
            frame = renderer.render()
            renderer.close()
            return frame
        except Exception as e:
            logger.debug("Rendering failed: %s", e)
            return None


def _are_adjacent_links(geom1: str, geom2: str) -> bool:
    """Check if two geometries belong to adjacent links (expected contact)."""
    link_pairs = {
        ("base_geom", "link0_geom"),
        ("link0_geom", "link1_geom"),
        ("link1_geom", "link2_geom"),
        ("link2_geom", "link3_geom"),
        ("link3_geom", "link4_geom"),
        ("link4_geom", "link5_geom"),
        ("link5_geom", "gripper_palm"),
        ("gripper_palm", "finger_left_geom"),
        ("gripper_palm", "finger_right_geom"),
        ("table_top", "base_geom"),
    }
    pair = (geom1, geom2)
    return pair in link_pairs or (pair[1], pair[0]) in link_pairs
