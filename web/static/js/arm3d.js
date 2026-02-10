// ═══════════════════════════════════════════════════════
//  arm3d.js — Three.js D1 Arm Model Module
//  Accurate kinematic chain using DH parameters
// ═══════════════════════════════════════════════════════
'use strict';

/**
 * D1 link lengths (meters) — matches the proven geometric FK in index.html
 */
const D1_LINKS = {
  d0: 0.1215,   // base to shoulder
  L1: 0.2085,   // shoulder to elbow
  L2: 0.2085,   // elbow to wrist
  L3: 0.1130,   // wrist to end-effector
};

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
 * 3x3 rotation matrix helpers (row-major arrays of 9)
 */
function _rz(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c,-s,0, s,c,0, 0,0,1];
}
function _ry(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c,0,s, 0,1,0, -s,0,c];
}
function _mul(A, B) {
  const C = new Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      C[r*3+c] = A[r*3]*B[c] + A[r*3+1]*B[3+c] + A[r*3+2]*B[6+c];
  return C;
}
function _apply(R, v) {
  return [
    R[0]*v[0]+R[1]*v[1]+R[2]*v[2],
    R[3]*v[0]+R[4]*v[1]+R[5]*v[2],
    R[6]*v[0]+R[7]*v[1]+R[8]*v[2]
  ];
}
function _vadd(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }

/**
 * Geometric forward kinematics — proven correct, matches index.html 2D viz.
 * Computes in Z-up frame, returns positions converted to Three.js Y-up.
 *
 * @param {number[]} jointAnglesRad - 6 or 7 joint angles in radians
 * @returns {{ positions: THREE.Vector3[], rotations: number[][] }}
 *   positions: [base, shoulder, elbow, wrist, ee] in Y-up
 *   rotations: rotation matrices at each joint (for gripper orientation)
 */
function forwardKinematics(jointAnglesRad) {
  const j = jointAnglesRad.map(v => v || 0);
  const { d0, L1, L2, L3 } = D1_LINKS;

  // All computation in Z-up frame
  const base = [0, 0, 0];
  const shoulder = [0, 0, d0];

  let R = _mul(_rz(j[0]), _ry(j[1]));
  const elbow = _vadd(shoulder, _apply(R, [0, 0, L1]));

  R = _mul(R, _ry(Math.PI/2 + j[2]));
  const wrist = _vadd(elbow, _apply(R, [0, 0, L2]));

  R = _mul(R, _rz(j[3]));
  R = _mul(R, _ry(j[4]));
  const ee = _vadd(wrist, _apply(R, [0, 0, L3]));

  // Convert Z-up to Three.js Y-up: (x, y, z) -> (x, z, -y)
  const toYUp = p => new THREE.Vector3(p[0], p[2], -p[1]);

  const positions = [base, shoulder, elbow, wrist, ee].map(toYUp);

  // Build Matrix4 transforms for orientation (used by gripper)
  // Last rotation R is for the EE frame
  const eeRotYUp = [
    R[0], R[2], -R[1],
    R[6], R[8], -R[7],
    -R[3], -R[5], R[4]
  ];

  const eeTransform = new THREE.Matrix4().set(
    eeRotYUp[0], eeRotYUp[1], eeRotYUp[2], positions[4].x,
    eeRotYUp[3], eeRotYUp[4], eeRotYUp[5], positions[4].y,
    eeRotYUp[6], eeRotYUp[7], eeRotYUp[8], positions[4].z,
    0, 0, 0, 1
  );

  return { positions, eeTransform };
}

/**
 * Get joint positions from FK result (backward compat)
 */
function getJointPositions(fkResult) {
  if (Array.isArray(fkResult)) return fkResult;  // legacy
  return fkResult.positions || [];
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
    this.fkResult = null;                        // last FK computation result
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
    this.fkResult = forwardKinematics(this.jointAngles);
    const positions = this.fkResult.positions;

    // Update joint spheres (we have 8 meshes but only 5 FK positions now)
    // Map: 0=base, 1=shoulder, 2=elbow, 3=wrist, 4=ee
    for (let i = 0; i < this.jointMeshes.length; i++) {
      const pi = Math.min(i, positions.length - 1);
      if (pi < positions.length) {
        this.jointMeshes[i].position.copy(positions[pi].clone().multiplyScalar(this.scale));
      }
      // Hide extra joint meshes beyond our 5 points
      this.jointMeshes[i].visible = (i < positions.length);
    }

    // Update link cylinders (4 links between 5 positions)
    for (let i = 0; i < this.linkMeshes.length; i++) {
      if (i + 1 >= positions.length) {
        this.linkMeshes[i].visible = false;
        continue;
      }
      this.linkMeshes[i].visible = true;
      const p0 = positions[i].clone().multiplyScalar(this.scale);
      const p1 = positions[i + 1].clone().multiplyScalar(this.scale);
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
    if (this.gripperGroup && positions.length >= 5) {
      const eePos = positions[4].clone().multiplyScalar(this.scale);
      this.gripperGroup.position.copy(eePos);

      // Set gripper orientation from FK
      const quat = new THREE.Quaternion().setFromRotationMatrix(this.fkResult.eeTransform);
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
    if (this.fkResult && this.fkResult.positions.length >= 5) {
      return this.fkResult.positions[4].clone();
    }
    return new THREE.Vector3();
  }

  /**
   * Get end-effector pose as { position, quaternion }
   */
  getEEPose() {
    if (this.fkResult && this.fkResult.eeTransform) {
      return {
        position: this.fkResult.positions[4].clone(),
        quaternion: new THREE.Quaternion().setFromRotationMatrix(this.fkResult.eeTransform),
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

/**
 * Camera3D — Visual camera frustum for 3D scene
 */
const CAMERA_COLORS = { 0: 0x00ffff, 1: 0xff4444, 2: 0x44ff44 };

class Camera3D {
  /**
   * @param {object} config - { id, label, position, rotation, fov, perspective }
   *   position/rotation in Z-up arm frame
   */
  constructor(config) {
    this.id = config.id;
    this.label = config.label || `Cam ${config.id}`;
    this.perspective = config.perspective || 'custom';
    this.isArmMounted = this.perspective === 'arm-mounted';
    this.color = CAMERA_COLORS[config.id] || 0xffffff;

    this.group = new THREE.Group();
    this.group.name = `Camera3D_${this.id}`;

    // Store config coords (Z-up)
    this.configPos = config.position || { x: 0, y: 0, z: 0 };
    this.configRot = config.rotation || { rx: 0, ry: 0, rz: 0 };
    this.fov = config.fov || 60;

    this._buildFrustum();
    this._buildLabel();
    this._applyTransform();
  }

  _buildFrustum() {
    const size = 0.03;
    const depth = 0.05;
    const halfFov = (this.fov / 2) * Math.PI / 180;
    const farW = depth * Math.tan(halfFov);
    const farH = farW * 0.75; // 4:3 aspect

    // Frustum wireframe: apex at origin, opening along +Y (Three.js forward for camera)
    const verts = [
      new THREE.Vector3(0, 0, 0),           // apex
      new THREE.Vector3(-farW, depth, -farH), // bottom-left
      new THREE.Vector3( farW, depth, -farH), // bottom-right
      new THREE.Vector3( farW, depth,  farH), // top-right
      new THREE.Vector3(-farW, depth,  farH), // top-left
    ];

    const edges = [
      0,1, 0,2, 0,3, 0,4,  // apex to corners
      1,2, 2,3, 3,4, 4,1,  // far rectangle
    ];

    const geo = new THREE.BufferGeometry();
    const positions = [];
    for (const idx of edges) {
      positions.push(verts[idx].x, verts[idx].y, verts[idx].z);
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

    const mat = new THREE.LineBasicMaterial({ color: this.color, linewidth: 2 });
    this.frustum = new THREE.LineSegments(geo, mat);
    this.group.add(this.frustum);

    // Small body box
    const boxGeo = new THREE.BoxGeometry(size * 0.8, size * 0.4, size * 0.6);
    const boxMat = new THREE.MeshPhongMaterial({
      color: this.color,
      emissive: this.color,
      emissiveIntensity: 0.4,
      transparent: true,
      opacity: 0.6,
    });
    const box = new THREE.Mesh(boxGeo, boxMat);
    box.position.y = -size * 0.2;
    this.group.add(box);
  }

  _buildLabel() {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = `#${this.color.toString(16).padStart(6, '0')}`;
    ctx.font = 'bold 18px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(this.label, 64, 22);

    const tex = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(0.08, 0.02, 1);
    sprite.position.y = -0.03;
    this.group.add(sprite);
  }

  /**
   * Convert Z-up config coords to Three.js Y-up and apply rotation.
   */
  _applyTransform() {
    const { x, y, z } = this.configPos;
    // Z-up to Y-up: (x, y, z) -> (x, z, -y)
    this.group.position.set(x, z, -y);

    // Apply rotations (config is degrees, convert to radians)
    const rx = this.configRot.rx * Math.PI / 180;
    const ry = this.configRot.ry * Math.PI / 180;
    const rz = this.configRot.rz * Math.PI / 180;
    this.group.rotation.set(rx, rz, -ry, 'XYZ');
  }

  /**
   * Update from config object
   */
  updateConfig(config) {
    this.configPos = config.position || this.configPos;
    this.configRot = config.rotation || this.configRot;
    this.fov = config.fov || this.fov;
    this._applyTransform();
  }

  /**
   * For arm-mounted cameras: attach to EE transform.
   * @param {THREE.Matrix4} eeTransform - end-effector transform from FK
   */
  attachToEE(eeTransform) {
    if (!this.isArmMounted) return;
    // Offset position in EE local frame
    const offset = new THREE.Vector3(
      this.configPos.x,
      this.configPos.z,   // Z-up z -> Y-up y
      -this.configPos.y   // Z-up y -> Y-up -z
    );
    const eePos = new THREE.Vector3().setFromMatrixPosition(eeTransform);
    const eeQuat = new THREE.Quaternion().setFromRotationMatrix(eeTransform);

    offset.applyQuaternion(eeQuat);
    this.group.position.copy(eePos).add(offset);

    // Apply EE rotation + local rotation offset
    const localQuat = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(
        this.configRot.rx * Math.PI / 180,
        this.configRot.rz * Math.PI / 180,
        -this.configRot.ry * Math.PI / 180,
        'XYZ'
      )
    );
    this.group.quaternion.copy(eeQuat).multiply(localQuat);
  }

  dispose() {
    this.group.traverse(child => {
      if (child.isMesh || child.isLineSegments) {
        child.geometry.dispose();
        child.material.dispose();
      }
      if (child.isSprite) {
        child.material.map.dispose();
        child.material.dispose();
      }
    });
  }
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
  window.Camera3D = Camera3D;
  window.CAMERA_COLORS = CAMERA_COLORS;
}
