// ═══════════════════════════════════════════════════════
//  arm3d.js — Three.js D1 Arm Model Module
//  Accurate kinematic chain using DH parameters
// ═══════════════════════════════════════════════════════
'use strict';

/**
 * D1 DH Parameters (from src/kinematics/kinematics.py)
 * Standard DH convention (Craig):
 *   a     — link length (along x_i)
 *   d     — link offset (along z_{i-1})
 *   alpha — link twist  (about x_i) [radians]
 */
const D1_DH = [
  { a: 0, d: 0.1215, alpha: -Math.PI/2, offset: 0 },  // J0 base yaw
  { a: 0, d: 0,      alpha:  Math.PI/2, offset: 0 },  // J1 shoulder pitch
  { a: 0, d: 0.2085, alpha: -Math.PI/2, offset: 0 },  // J2 elbow pitch
  { a: 0, d: 0,      alpha:  Math.PI/2, offset: 0 },  // J3 forearm roll
  { a: 0, d: 0.2085, alpha: -Math.PI/2, offset: 0 },  // J4 wrist pitch
  { a: 0, d: 0,      alpha:  Math.PI/2, offset: 0 },  // J5 wrist roll
  { a: 0, d: 0.1130, alpha:  0,         offset: 0 },  // J6 (gripper)
];

const JOINT_COLORS = [
  0x00aaff,  // J0 base yaw - blue
  0xff6600,  // J1 shoulder pitch - orange
  0x00cc44,  // J2 elbow pitch - green
  0xcc00ff,  // J3 forearm roll - purple
  0xffcc00,  // J4 wrist pitch - yellow
  0xff0066,  // J5 wrist roll - pink
  0x44ffcc,  // EE/gripper - cyan
];

const JOINT_NAMES = ['Base Yaw', 'Shoulder', 'Elbow', 'Forearm Roll', 'Wrist Pitch', 'Wrist Roll', 'Gripper'];

/**
 * Compute 4x4 DH transform for a single joint
 */
function dhTransform(dh, theta) {
  const t = theta + dh.offset;
  const ct = Math.cos(t), st = Math.sin(t);
  const ca = Math.cos(dh.alpha), sa = Math.sin(dh.alpha);
  // Return column-major for Three.js Matrix4
  return new THREE.Matrix4().set(
    ct,  -st*ca,  st*sa,  dh.a*ct,
    st,   ct*ca, -ct*sa,  dh.a*st,
    0,    sa,     ca,     dh.d,
    0,    0,      0,      1
  );
}

/**
 * Forward kinematics — returns array of 4x4 transforms (base to each joint frame)
 * @param {number[]} jointAnglesRad - 7 joint angles in radians
 * @returns {THREE.Matrix4[]} - 8 transforms (base + 7 joints)
 */
function forwardKinematics(jointAnglesRad) {
  const transforms = [new THREE.Matrix4()]; // identity for base
  let T = new THREE.Matrix4();
  for (let i = 0; i < D1_DH.length; i++) {
    const q = jointAnglesRad[i] || 0;
    T = T.clone().multiply(dhTransform(D1_DH[i], q));
    transforms.push(T.clone());
  }
  return transforms;
}

/**
 * Get joint positions from FK transforms
 */
function getJointPositions(transforms) {
  return transforms.map(T => new THREE.Vector3().setFromMatrixPosition(T));
}

/**
 * D1Arm3D — Three.js 3D arm model
 */
class D1Arm3D {
  constructor(options = {}) {
    this.group = new THREE.Group();
    this.group.name = 'D1Arm';

    this.jointAngles = new Float64Array(7);     // current rendered angles (rad)
    this.targetAngles = new Float64Array(7);     // target from WS
    this.gripperWidth = 0;                       // mm
    this.targetGripperWidth = 0;

    this.smoothing = options.smoothing !== undefined ? options.smoothing : 0.15;
    this.showLabels = options.showLabels || false;
    this.scale = options.scale || 1;
    this.ghostArm = null;

    // Meshes
    this.jointMeshes = [];
    this.linkMeshes = [];
    this.gripperGroup = null;
    this.gripperLeft = null;
    this.gripperRight = null;
    this.transforms = [];

    this._buildGeometry();
  }

  _buildGeometry() {
    // Joint spheres
    for (let i = 0; i <= 7; i++) {
      const radius = i === 0 ? 0.025 : i === 7 ? 0.012 : 0.018;
      const geo = new THREE.SphereGeometry(radius * this.scale, 16, 12);
      const mat = new THREE.MeshPhongMaterial({
        color: JOINT_COLORS[Math.min(i, 6)],
        emissive: JOINT_COLORS[Math.min(i, 6)],
        emissiveIntensity: 0.3,
        shininess: 80,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = `joint_${i}`;
      this.group.add(mesh);
      this.jointMeshes.push(mesh);
    }

    // Link cylinders (drawn between joints dynamically)
    const linkColors = [0x222222, 0x1a1a1a, 0x222222, 0x1a1a1a, 0x222222, 0x1a1a1a, 0x1a1a1a];
    const linkRadii = [0.022, 0.018, 0.018, 0.014, 0.014, 0.012, 0.010];
    for (let i = 0; i < 7; i++) {
      const geo = new THREE.CylinderGeometry(
        linkRadii[i] * this.scale,
        linkRadii[i] * this.scale,
        1, 8 // height=1, will be scaled
      );
      const mat = new THREE.MeshPhongMaterial({
        color: linkColors[i],
        emissive: 0x0a3d20,
        emissiveIntensity: 0.1,
        shininess: 40,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = `link_${i}`;
      this.group.add(mesh);
      this.linkMeshes.push(mesh);
    }

    // Base cylinder
    const baseGeo = new THREE.CylinderGeometry(0.04 * this.scale, 0.05 * this.scale, 0.03 * this.scale, 20);
    const baseMat = new THREE.MeshPhongMaterial({ color: 0x1a1a1a, shininess: 60 });
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.name = 'base_cylinder';
    this.group.add(baseMesh);

    // Green accent ring on base
    const ringGeo = new THREE.TorusGeometry(0.045 * this.scale, 0.003 * this.scale, 8, 24);
    const ringMat = new THREE.MeshPhongMaterial({ color: 0x00cc44, emissive: 0x00cc44, emissiveIntensity: 0.5 });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = Math.PI / 2;
    ringMesh.position.y = 0.015 * this.scale;
    this.group.add(ringMesh);

    // Gripper
    this._buildGripper();

    // Initial pose
    this.updatePose();
  }

  _buildGripper() {
    this.gripperGroup = new THREE.Group();
    this.gripperGroup.name = 'gripper';

    const fingerLen = 0.04 * this.scale;
    const fingerW = 0.006 * this.scale;
    const fingerD = 0.015 * this.scale;

    const fingerGeo = new THREE.BoxGeometry(fingerW, fingerLen, fingerD);
    const fingerMat = new THREE.MeshPhongMaterial({
      color: 0x333333,
      emissive: 0x00cc44,
      emissiveIntensity: 0.15,
      shininess: 60,
    });

    // Finger tip nubs
    const tipGeo = new THREE.BoxGeometry(fingerW * 1.3, 0.008 * this.scale, fingerD * 0.6);
    const tipMat = new THREE.MeshPhongMaterial({ color: 0x555555, shininess: 40 });

    this.gripperLeft = new THREE.Group();
    const leftFinger = new THREE.Mesh(fingerGeo, fingerMat);
    leftFinger.position.y = fingerLen / 2;
    this.gripperLeft.add(leftFinger);
    const leftTip = new THREE.Mesh(tipGeo, tipMat);
    leftTip.position.y = fingerLen;
    leftTip.position.x = fingerW * 0.15;
    this.gripperLeft.add(leftTip);

    this.gripperRight = new THREE.Group();
    const rightFinger = new THREE.Mesh(fingerGeo.clone(), fingerMat.clone());
    rightFinger.position.y = fingerLen / 2;
    this.gripperRight.add(rightFinger);
    const rightTip = new THREE.Mesh(tipGeo.clone(), tipMat.clone());
    rightTip.position.y = fingerLen;
    rightTip.position.x = -fingerW * 0.15;
    this.gripperRight.add(rightTip);

    this.gripperGroup.add(this.gripperLeft);
    this.gripperGroup.add(this.gripperRight);
    this.group.add(this.gripperGroup);
  }

  /**
   * Set target joint angles (degrees) and gripper width (mm)
   */
  setTarget(jointsDeg, gripperMM) {
    for (let i = 0; i < 7 && i < jointsDeg.length; i++) {
      this.targetAngles[i] = jointsDeg[i] * Math.PI / 180;
    }
    if (gripperMM !== undefined) this.targetGripperWidth = gripperMM;
  }

  /**
   * Set joint angles immediately (no interpolation)
   */
  setImmediate(jointsDeg, gripperMM) {
    for (let i = 0; i < 7 && i < jointsDeg.length; i++) {
      this.jointAngles[i] = jointsDeg[i] * Math.PI / 180;
      this.targetAngles[i] = this.jointAngles[i];
    }
    if (gripperMM !== undefined) {
      this.gripperWidth = gripperMM;
      this.targetGripperWidth = gripperMM;
    }
    this.updatePose();
  }

  /**
   * Animate one step — call each frame
   */
  animate(dt) {
    const alpha = 1 - Math.pow(1 - this.smoothing, dt * 60);
    let dirty = false;
    for (let i = 0; i < 7; i++) {
      const diff = this.targetAngles[i] - this.jointAngles[i];
      if (Math.abs(diff) > 0.0001) {
        this.jointAngles[i] += diff * alpha;
        dirty = true;
      }
    }
    const gDiff = this.targetGripperWidth - this.gripperWidth;
    if (Math.abs(gDiff) > 0.01) {
      this.gripperWidth += gDiff * alpha;
      dirty = true;
    }
    if (dirty) this.updatePose();
    return dirty;
  }

  /**
   * Update mesh positions from current joint angles
   */
  updatePose() {
    this.transforms = forwardKinematics(this.jointAngles);
    const positions = getJointPositions(this.transforms);

    // Update joint spheres
    for (let i = 0; i < this.jointMeshes.length && i < positions.length; i++) {
      this.jointMeshes[i].position.copy(positions[i].multiplyScalar(this.scale));
    }

    // Update link cylinders
    for (let i = 0; i < this.linkMeshes.length && i + 1 < positions.length; i++) {
      const p0 = getJointPositions(this.transforms)[i].multiplyScalar(this.scale);
      const p1 = getJointPositions(this.transforms)[i + 1].multiplyScalar(this.scale);
      const mid = p0.clone().add(p1).multiplyScalar(0.5);
      const dir = p1.clone().sub(p0);
      const len = dir.length();

      this.linkMeshes[i].position.copy(mid);
      this.linkMeshes[i].scale.y = len || 0.001;

      if (len > 0.0001) {
        const up = new THREE.Vector3(0, 1, 0);
        const quat = new THREE.Quaternion().setFromUnitVectors(up, dir.normalize());
        this.linkMeshes[i].quaternion.copy(quat);
      }
    }

    // Update gripper
    if (this.gripperGroup && positions.length >= 8) {
      const eePos = getJointPositions(this.transforms)[7].multiplyScalar(this.scale);
      this.gripperGroup.position.copy(eePos);

      // Set gripper orientation from last transform
      const eeT = this.transforms[7];
      const quat = new THREE.Quaternion().setFromRotationMatrix(eeT);
      this.gripperGroup.quaternion.copy(quat);

      // Gripper opening
      const halfOpen = (this.gripperWidth / 1000 / 2) * this.scale;
      this.gripperLeft.position.x = halfOpen;
      this.gripperRight.position.x = -halfOpen;
    }
  }

  /**
   * Get end-effector position (meters)
   */
  getEEPosition() {
    if (this.transforms.length >= 8) {
      return new THREE.Vector3().setFromMatrixPosition(this.transforms[7]);
    }
    return new THREE.Vector3();
  }

  /**
   * Get end-effector pose as { position, quaternion }
   */
  getEEPose() {
    if (this.transforms.length >= 8) {
      const T = this.transforms[7];
      return {
        position: new THREE.Vector3().setFromMatrixPosition(T),
        quaternion: new THREE.Quaternion().setFromRotationMatrix(T),
      };
    }
    return { position: new THREE.Vector3(), quaternion: new THREE.Quaternion() };
  }

  /**
   * Create a ghost (transparent) copy for target preview
   */
  createGhost() {
    const ghost = new D1Arm3D({ scale: this.scale, smoothing: 0.3 });
    ghost.group.traverse(child => {
      if (child.isMesh) {
        child.material = child.material.clone();
        child.material.transparent = true;
        child.material.opacity = 0.25;
        child.material.depthWrite = false;
      }
    });
    ghost.group.name = 'D1Arm_Ghost';
    this.ghostArm = ghost;
    return ghost;
  }

  dispose() {
    this.group.traverse(child => {
      if (child.isMesh) {
        child.geometry.dispose();
        child.material.dispose();
      }
    });
  }
}

/**
 * Create a workspace table mesh
 */
function createWorkTable(options = {}) {
  const w = options.width || 0.6;
  const d = options.depth || 0.4;
  const h = options.height || 0.02;
  const y = options.y || -0.02;
  const group = new THREE.Group();
  group.name = 'WorkTable';

  const topGeo = new THREE.BoxGeometry(w, h, d);
  const topMat = new THREE.MeshPhongMaterial({ color: 0x2a2a2a, shininess: 20 });
  const top = new THREE.Mesh(topGeo, topMat);
  top.position.y = y;
  group.add(top);

  // Grid lines on table
  const gridMat = new THREE.LineBasicMaterial({ color: 0x3a3a3a });
  const step = 0.05;
  for (let x = -w/2; x <= w/2; x += step) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x, y + h/2 + 0.001, -d/2),
      new THREE.Vector3(x, y + h/2 + 0.001, d/2),
    ]);
    group.add(new THREE.Line(geo, gridMat));
  }
  for (let z = -d/2; z <= d/2; z += step) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-w/2, y + h/2 + 0.001, z),
      new THREE.Vector3(w/2, y + h/2 + 0.001, z),
    ]);
    group.add(new THREE.Line(geo, gridMat));
  }

  return group;
}

/**
 * Create coordinate axes helper
 */
function createAxes(size = 0.1) {
  return new THREE.AxesHelper(size);
}

// Export for module or global use
if (typeof window !== 'undefined') {
  window.D1Arm3D = D1Arm3D;
  window.D1_DH = D1_DH;
  window.JOINT_COLORS = JOINT_COLORS;
  window.JOINT_NAMES = JOINT_NAMES;
  window.forwardKinematics = forwardKinematics;
  window.getJointPositions = getJointPositions;
  window.createWorkTable = createWorkTable;
  window.createAxes = createAxes;
}
