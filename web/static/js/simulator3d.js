// ═══════════════════════════════════════════════════════
//  simulator3d.js — Full 3D D1 Arm Simulator
//  Interactive simulation environment with Three.js
//  Reuses D1Arm3D kinematic model from arm3d.js
// ═══════════════════════════════════════════════════════
'use strict';

/**
 * D1 joint limits (degrees) — matches src/safety/limits.py
 */
const SIM_JOINT_LIMITS = [
  [-135, 135],   // J0 Base Yaw
  [-80, 80],     // J1 Shoulder Pitch
  [-80, 80],     // J2 Elbow Pitch
  [-135, 135],   // J3 Elbow Roll
  [-80, 80],     // J4 Wrist Pitch
  [-135, 135],   // J5 Wrist Roll
];
const SIM_GRIPPER_RANGE = [0, 65]; // mm

/**
 * D1ArmSimulator — Full 3D simulation environment
 *
 * Provides:
 *  - Interactive Three.js scene with workspace, grid, shadows
 *  - Real arm model (from D1Arm3D) for current state
 *  - Ghost arm for previewing movements before execution
 *  - Orbit camera controls
 *  - Workspace boundary visualization (reach envelope)
 *  - End-effector trace path
 *  - Joint limit indicators
 *  - Simulation-only mode: manipulate without sending to hardware
 *  - Execute mode: send simulated pose to physical arm
 */
class D1ArmSimulator {
  constructor(containerEl) {
    this.container = containerEl;
    this.active = false;
    this._disposed = false;

    // Mode: 'simulate' = sim only, 'physical' = send to real arm
    this.mode = 'simulate';

    // Sim state
    this.simJoints = [0, 0, 0, 0, 0, 0];
    this.simGripper = 0;
    this.liveJoints = [0, 0, 0, 0, 0, 0];
    this.liveGripper = 0;

    // Trail
    this.trailPoints = [];
    this.maxTrailPoints = 500;
    this.showTrail = true;

    // Display
    this.showWorkspaceEnvelope = true;
    this.showGrid = true;
    this.showAxes = true;
    this.showGhost = true;

    // Callbacks
    this.onJointsChanged = null;  // called when sim joints change
    this.onExecute = null;        // called when user clicks Execute

    this._initScene();
    this._initArm();
    this._initEnvironment();
    this._initWorkspaceEnvelope();
    this._initTrail();
    this._initLighting();

    this._animId = null;
    this._clock = new THREE.Clock();
  }

  _initScene() {
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 600;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1525);
    this.scene.fog = new THREE.FogExp2(0x0d1525, 0.8);

    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 10);
    this.camera.position.set(0.5, 0.45, 0.5);
    this.camera.lookAt(0, 0.2, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.15, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 2.5;
    this.controls.maxPolarAngle = Math.PI * 0.85;
    this.controls.update();

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.container);
  }

  _initArm() {
    // Main sim arm (full color)
    this.arm = new D1Arm3D({ scale: 1, smoothing: 0.2 });
    this.scene.add(this.arm.group);

    // Ghost arm (transparent, shows live/target state)
    this.ghost = this.arm.createGhost();
    this.scene.add(this.ghost.group);
    this.ghost.group.visible = this.showGhost;
  }

  _initEnvironment() {
    // Ground plane
    const groundGeo = new THREE.PlaneGeometry(2, 2);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x1a1a2e,
      roughness: 0.9,
      metalness: 0.1,
    });
    this.ground = new THREE.Mesh(groundGeo, groundMat);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.y = -0.02;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);

    // Grid
    this.gridHelper = new THREE.GridHelper(1.2, 24, 0x2a4a7c, 0x1a2744);
    this.gridHelper.position.y = -0.019;
    this.scene.add(this.gridHelper);

    // Axes
    this.axesHelper = new THREE.AxesHelper(0.15);
    this.axesHelper.position.y = 0.001;
    this.scene.add(this.axesHelper);

    // Work table
    this.workTable = createWorkTable({
      width: 0.7,
      depth: 0.5,
      height: 0.015,
      y: -0.015,
    });
    this.scene.add(this.workTable);

    // Base pedestal
    const pedGeo = new THREE.CylinderGeometry(0.05, 0.06, 0.04, 24);
    const pedMat = new THREE.MeshStandardMaterial({
      color: 0x2a2a3e,
      roughness: 0.4,
      metalness: 0.6,
    });
    const pedestal = new THREE.Mesh(pedGeo, pedMat);
    pedestal.position.y = -0.01;
    pedestal.castShadow = true;
    this.scene.add(pedestal);
  }

  _initWorkspaceEnvelope() {
    // Workspace sphere (max reach ~0.55m)
    const envGeo = new THREE.SphereGeometry(0.55, 32, 24, 0, Math.PI * 2, 0, Math.PI * 0.6);
    const envMat = new THREE.MeshBasicMaterial({
      color: 0x4a90d9,
      wireframe: true,
      transparent: true,
      opacity: 0.06,
    });
    this.envelope = new THREE.Mesh(envGeo, envMat);
    this.envelope.position.y = 0.0;
    this.envelope.visible = this.showWorkspaceEnvelope;
    this.scene.add(this.envelope);

    // Reach ring at table height
    const ringGeo = new THREE.RingGeometry(0.54, 0.55, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x4a90d9,
      transparent: true,
      opacity: 0.15,
      side: THREE.DoubleSide,
    });
    this.reachRing = new THREE.Mesh(ringGeo, ringMat);
    this.reachRing.rotation.x = -Math.PI / 2;
    this.reachRing.position.y = 0.001;
    this.reachRing.visible = this.showWorkspaceEnvelope;
    this.scene.add(this.reachRing);
  }

  _initTrail() {
    const trailGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(this.maxTrailPoints * 3);
    trailGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    trailGeo.setDrawRange(0, 0);

    const trailMat = new THREE.LineBasicMaterial({
      color: 0x44ffcc,
      transparent: true,
      opacity: 0.6,
    });
    this.trailLine = new THREE.Line(trailGeo, trailMat);
    this.trailLine.visible = this.showTrail;
    this.scene.add(this.trailLine);
  }

  _initLighting() {
    // Ambient
    const ambient = new THREE.AmbientLight(0x404060, 0.6);
    this.scene.add(ambient);

    // Hemisphere
    const hemi = new THREE.HemisphereLight(0x8899cc, 0x2a2a3e, 0.4);
    this.scene.add(hemi);

    // Key light
    const key = new THREE.DirectionalLight(0xffffff, 0.8);
    key.position.set(0.5, 1.0, 0.5);
    key.castShadow = true;
    key.shadow.mapSize.width = 1024;
    key.shadow.mapSize.height = 1024;
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 3;
    key.shadow.camera.left = -1;
    key.shadow.camera.right = 1;
    key.shadow.camera.top = 1;
    key.shadow.camera.bottom = -1;
    this.scene.add(key);

    // Fill light
    const fill = new THREE.DirectionalLight(0x4a90d9, 0.3);
    fill.position.set(-0.5, 0.5, -0.3);
    this.scene.add(fill);

    // Rim light
    const rim = new THREE.PointLight(0xe94560, 0.3, 2);
    rim.position.set(-0.3, 0.8, -0.5);
    this.scene.add(rim);
  }

  // ----------------------------------------------------------
  //  Joint manipulation
  // ----------------------------------------------------------

  /**
   * Set a single simulated joint angle (degrees).
   */
  setSimJoint(index, angleDeg) {
    if (index < 0 || index >= 6) return;
    const [lo, hi] = SIM_JOINT_LIMITS[index];
    this.simJoints[index] = Math.max(lo, Math.min(hi, angleDeg));
    this._updateArm();
  }

  /**
   * Set all simulated joint angles (degrees).
   */
  setSimJoints(anglesDeg) {
    for (let i = 0; i < 6 && i < anglesDeg.length; i++) {
      const [lo, hi] = SIM_JOINT_LIMITS[i];
      this.simJoints[i] = Math.max(lo, Math.min(hi, anglesDeg[i]));
    }
    this._updateArm();
  }

  /**
   * Set simulated gripper position (mm).
   */
  setSimGripper(mm) {
    this.simGripper = Math.max(SIM_GRIPPER_RANGE[0], Math.min(SIM_GRIPPER_RANGE[1], mm));
    this._updateArm();
  }

  /**
   * Update live arm state (from WebSocket data) — shown as ghost.
   */
  setLiveState(jointsDeg, gripperMM) {
    this.liveJoints = jointsDeg.slice(0, 6);
    this.liveGripper = gripperMM;
    // Ghost shows live hardware state
    const ghostAngles = [...this.liveJoints, 0]; // 7th for DH chain
    this.ghost.setTarget(ghostAngles, this.liveGripper);
  }

  /**
   * Sync sim joints FROM current live state.
   */
  syncFromLive() {
    this.simJoints = [...this.liveJoints];
    this.simGripper = this.liveGripper;
    this._updateArm();
    if (this.onJointsChanged) {
      this.onJointsChanged(this.simJoints, this.simGripper);
    }
  }

  /**
   * Reset sim joints to zero (home).
   */
  resetToHome() {
    this.simJoints = [0, 0, 0, 0, 0, 0];
    this.simGripper = 0;
    this._updateArm();
    if (this.onJointsChanged) {
      this.onJointsChanged(this.simJoints, this.simGripper);
    }
  }

  _updateArm() {
    // Main arm shows sim pose — pad to 7 joints for DH chain
    const simAngles7 = [...this.simJoints, 0];
    this.arm.setTarget(simAngles7, this.simGripper);

    // Update trail
    if (this.showTrail) {
      const ee = this.arm.getEEPosition();
      if (ee) this._addTrailPoint(ee);
    }
  }

  _addTrailPoint(pos) {
    this.trailPoints.push(pos.clone());
    if (this.trailPoints.length > this.maxTrailPoints) {
      this.trailPoints.shift();
    }

    const geo = this.trailLine.geometry;
    const posAttr = geo.getAttribute('position');
    for (let i = 0; i < this.trailPoints.length; i++) {
      posAttr.setXYZ(i, this.trailPoints[i].x, this.trailPoints[i].y, this.trailPoints[i].z);
    }
    posAttr.needsUpdate = true;
    geo.setDrawRange(0, this.trailPoints.length);
  }

  clearTrail() {
    this.trailPoints = [];
    this.trailLine.geometry.setDrawRange(0, 0);
  }

  // ----------------------------------------------------------
  //  Display toggles
  // ----------------------------------------------------------

  toggleEnvelope(show) {
    this.showWorkspaceEnvelope = show;
    this.envelope.visible = show;
    this.reachRing.visible = show;
  }

  toggleGrid(show) {
    this.showGrid = show;
    this.gridHelper.visible = show;
  }

  toggleAxes(show) {
    this.showAxes = show;
    this.axesHelper.visible = show;
  }

  toggleGhost(show) {
    this.showGhost = show;
    this.ghost.group.visible = show;
  }

  toggleTrail(show) {
    this.showTrail = show;
    this.trailLine.visible = show;
  }

  // ----------------------------------------------------------
  //  Camera presets
  // ----------------------------------------------------------

  setCameraPreset(preset) {
    const presets = {
      front: { pos: [0, 0.3, 0.8], target: [0, 0.15, 0] },
      side: { pos: [0.8, 0.3, 0], target: [0, 0.15, 0] },
      top: { pos: [0, 0.9, 0.01], target: [0, 0.0, 0] },
      iso: { pos: [0.5, 0.45, 0.5], target: [0, 0.15, 0] },
      close: { pos: [0.2, 0.25, 0.25], target: [0, 0.2, 0] },
    };
    const p = presets[preset];
    if (!p) return;
    this.camera.position.set(...p.pos);
    this.controls.target.set(...p.target);
    this.controls.update();
  }

  // ----------------------------------------------------------
  //  EE info
  // ----------------------------------------------------------

  getEEInfo() {
    const pose = this.arm.getEEPose();
    return {
      x: pose.position.x.toFixed(4),
      y: pose.position.y.toFixed(4),
      z: pose.position.z.toFixed(4),
      reach: Math.sqrt(
        pose.position.x ** 2 + pose.position.y ** 2 + pose.position.z ** 2
      ).toFixed(4),
    };
  }

  // ----------------------------------------------------------
  //  Lifecycle
  // ----------------------------------------------------------

  activate() {
    if (this.active || this._disposed) return;
    this.active = true;
    this._onResize();
    this._animate();
  }

  deactivate() {
    this.active = false;
    if (this._animId != null) {
      cancelAnimationFrame(this._animId);
      this._animId = null;
    }
  }

  _animate() {
    if (!this.active || this._disposed) return;
    this._animId = requestAnimationFrame(() => this._animate());

    const dt = this._clock.getDelta();

    this.arm.animate(dt);
    if (this.showGhost && this.ghost.group.visible) {
      this.ghost.animate(dt);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  dispose() {
    this._disposed = true;
    this.deactivate();
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
    }
    this.arm.dispose();
    this.ghost.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}

// Export for global use
if (typeof window !== 'undefined') {
  window.D1ArmSimulator = D1ArmSimulator;
  window.SIM_JOINT_LIMITS = SIM_JOINT_LIMITS;
  window.SIM_GRIPPER_RANGE = SIM_GRIPPER_RANGE;
}
