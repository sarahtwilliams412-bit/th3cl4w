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

    // Follow mode — slave sim to live telemetry
    this.followLive = false;

    // Callbacks
    this.onJointsChanged = null;  // called when sim joints change
    this.onExecute = null;        // called when user clicks Execute

    // Camera3D objects
    this.cameras3d = {};
    this.showCameras = true;

    this._initScene();
    this._initArm();
    this._initEnvironment();
    this._initWorkspaceEnvelope();
    this._initTrail();
    this._initLighting();
    this._initDetectedObjects();
    this._initCameras();

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
  //  Camera 3D objects
  // ----------------------------------------------------------

  _initCameras() {
    if (typeof Camera3D === 'undefined') return;
    fetch('/api/cameras/orientation')
      .then(r => r.json())
      .then(config => {
        this._createCameras(config);
      })
      .catch(e => console.warn('Failed to load camera orientations:', e));
  }

  _createCameras(config) {
    // Remove existing
    for (const id in this.cameras3d) {
      this.scene.remove(this.cameras3d[id].group);
      this.cameras3d[id].dispose();
    }
    this.cameras3d = {};

    for (const [id, cfg] of Object.entries(config)) {
      const cam = new Camera3D({
        id: parseInt(id),
        label: cfg.label || `Cam ${id}`,
        position: cfg.position || { x: 0, y: 0, z: 0 },
        rotation: cfg.rotation || { rx: 0, ry: 0, rz: 0 },
        fov: cfg.fov || 60,
        perspective: cfg.perspective || 'custom',
      });
      cam.group.visible = this.showCameras;
      this.scene.add(cam.group);
      this.cameras3d[id] = cam;
    }
  }

  /**
   * Update camera configs (called from UI).
   */
  updateCameras(config) {
    this._createCameras(config);
  }

  /**
   * Update a single camera config in real-time.
   */
  updateSingleCamera(id, cfg) {
    const cam = this.cameras3d[String(id)];
    if (cam) {
      cam.updateConfig(cfg);
    }
  }

  toggleCameras(show) {
    this.showCameras = show;
    for (const id in this.cameras3d) {
      this.cameras3d[id].group.visible = show;
    }
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
    // Snap immediately (no smooth interpolation)
    const simAngles7 = [...this.simJoints, 0];
    this.arm.setImmediate(simAngles7, this.simGripper);
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

    // Follow mode: copy live → sim each frame
    if (this.followLive) {
      let changed = false;
      for (let i = 0; i < 6; i++) {
        if (this.simJoints[i] !== this.liveJoints[i]) { changed = true; break; }
      }
      if (this.simGripper !== this.liveGripper) changed = true;
      if (changed) {
        this.simJoints = [...this.liveJoints];
        this.simGripper = this.liveGripper;
        const simAngles7 = [...this.simJoints, 0];
        this.arm.setTarget(simAngles7, this.simGripper);
        if (this.onJointsChanged) {
          this.onJointsChanged(this.simJoints, this.simGripper);
        }
      }
      // Hide ghost in follow mode (main arm IS the live state)
      if (this.ghost.group.visible) this.ghost.group.visible = false;
    } else {
      // Restore ghost visibility based on showGhost setting
      if (this.showGhost && !this.ghost.group.visible) this.ghost.group.visible = true;
    }

    this.arm.animate(dt);
    if (this.ghost.group.visible) {
      this.ghost.animate(dt);
    }

    // Update arm-mounted cameras to follow EE
    if (this.arm.fkResult && this.arm.fkResult.eeTransform) {
      for (const id in this.cameras3d) {
        const cam = this.cameras3d[id];
        if (cam.isArmMounted) {
          cam.attachToEE(this.arm.fkResult.eeTransform);
        }
      }
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

  // ----------------------------------------------------------
  //  Detected Object Visualization
  // ----------------------------------------------------------

  /**
   * Initialize the detected objects layer.
   * Called once during construction (after scene is ready).
   */
  _initDetectedObjects() {
    // Group to hold all object meshes
    this._objectsGroup = new THREE.Group();
    this._objectsGroup.name = 'detected_objects';
    this.scene.add(this._objectsGroup);

    // Currently rendered objects keyed by id
    this._renderedObjects = {};

    // Distance lines from base to objects
    this._distanceLines = {};

    // Labels (sprite-based)
    this._objectLabels = {};

    // Settings
    this.showDetectedObjects = true;
    this.showObjectLabels = true;
    this.showDistanceLines = false;

    // Max reach constant for visual ring matching
    this._maxReachM = 0.55;
  }

  /**
   * Update detected objects in the 3D scene.
   * @param {Object[]} objects — array of detected object dicts from the API
   *   Each has: id, label, x_m, y_m, z_m, width_mm, depth_mm, height_mm,
   *             within_reach, color_hex, distance_mm, confidence
   */
  updateDetectedObjects(objects) {
    if (!this._objectsGroup) this._initDetectedObjects();
    if (!this.showDetectedObjects) {
      this._objectsGroup.visible = false;
      return;
    }
    this._objectsGroup.visible = true;

    const currentIds = new Set();

    for (const obj of objects) {
      currentIds.add(obj.id);

      if (this._renderedObjects[obj.id]) {
        // Update existing object position
        this._updateObjectMesh(obj);
      } else {
        // Create new object mesh
        this._createObjectMesh(obj);
      }
    }

    // Remove objects no longer detected
    for (const id of Object.keys(this._renderedObjects)) {
      if (!currentIds.has(parseInt(id))) {
        this._removeObjectMesh(parseInt(id));
      }
    }
  }

  _createObjectMesh(obj) {
    // Object dimensions in meters
    const w = Math.max(0.01, (obj.width_mm || 40) / 1000);
    const d = Math.max(0.01, (obj.depth_mm || 40) / 1000);
    const h = Math.max(0.01, (obj.height_mm || 50) / 1000);

    // Pick geometry based on detected shape
    const shape = obj.shape || 'box';
    let geo;
    let wireGeo;

    if (shape === 'cylinder') {
      const radius = Math.max(w, d) / 2;
      geo = new THREE.CylinderGeometry(radius, radius, h, 24);
      wireGeo = new THREE.CylinderGeometry(radius + 0.002, radius + 0.002, h + 0.004, 24);
    } else if (shape === 'sphere') {
      const radius = Math.max(w, d, h) / 2;
      geo = new THREE.SphereGeometry(radius, 20, 16);
      wireGeo = new THREE.SphereGeometry(radius + 0.002, 20, 16);
    } else {
      // "box" or "irregular" — use BoxGeometry with actual proportions
      geo = new THREE.BoxGeometry(w, h, d);
      wireGeo = new THREE.BoxGeometry(w + 0.004, h + 0.004, d + 0.004);
    }

    // Parse color
    const color = new THREE.Color(obj.color_hex || '#ff8800');

    // Reachable objects get full opacity + glow; out-of-reach are dimmed
    const mat = new THREE.MeshStandardMaterial({
      color: color,
      roughness: 0.5,
      metalness: 0.2,
      transparent: true,
      opacity: obj.within_reach ? 0.85 : 0.35,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    // Apply rotation from detection (around Y axis in Three.js = overhead rotation)
    if (obj.rotation_deg && shape === 'box') {
      mesh.rotation.y = -(obj.rotation_deg || 0) * Math.PI / 180;
    }

    // Position: x_m and y_m are workspace coords; Three.js uses Y-up
    // Workspace X → Three.js X, Workspace Y → Three.js -Z (depth), height → Three.js Y
    mesh.position.set(
      obj.x_m || 0,
      (h / 2) + 0.001,  // half height above table surface
      -(obj.y_m || 0)    // workspace Y maps to negative Z (forward from arm)
    );

    this._objectsGroup.add(mesh);
    this._renderedObjects[obj.id] = mesh;

    // Wireframe outline for reachable objects
    if (obj.within_reach) {
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x44ff88,
        wireframe: true,
        transparent: true,
        opacity: 0.5,
      });
      const wire = new THREE.Mesh(wireGeo, wireMat);
      wire.position.copy(mesh.position);
      if (mesh.rotation.y) wire.rotation.y = mesh.rotation.y;
      this._objectsGroup.add(wire);
      mesh.userData.wireframe = wire;

      // Pulsing glow ring at base of reachable objects
      const ringGeo = new THREE.RingGeometry(
        Math.max(w, d) * 0.6,
        Math.max(w, d) * 0.8,
        24
      );
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x44ff88,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(mesh.position.x, 0.002, mesh.position.z);
      this._objectsGroup.add(ring);
      mesh.userData.baseRing = ring;
    }

    // Distance line from arm base to object
    if (this.showDistanceLines) {
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0.01, 0),
        mesh.position.clone().setY(0.01),
      ]);
      const lineMat = new THREE.LineBasicMaterial({
        color: obj.within_reach ? 0x44ff88 : 0xff6644,
        transparent: true,
        opacity: 0.3,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      this._objectsGroup.add(line);
      this._distanceLines[obj.id] = line;
    }

    // Label sprite
    if (this.showObjectLabels) {
      const label = this._createLabelSprite(obj);
      if (label) {
        label.position.set(
          mesh.position.x,
          h + 0.02,
          mesh.position.z
        );
        this._objectsGroup.add(label);
        this._objectLabels[obj.id] = label;
      }
    }
  }

  _updateObjectMesh(obj) {
    const mesh = this._renderedObjects[obj.id];
    if (!mesh) return;

    const h = Math.max(0.01, (obj.height_mm || 50) / 1000);

    // Smoothly interpolate position
    const targetX = obj.x_m || 0;
    const targetY = (h / 2) + 0.001;
    const targetZ = -(obj.y_m || 0);

    mesh.position.lerp(new THREE.Vector3(targetX, targetY, targetZ), 0.2);

    // Update opacity based on reachability
    mesh.material.opacity = obj.within_reach ? 0.85 : 0.35;

    // Update wireframe position
    if (mesh.userData.wireframe) {
      mesh.userData.wireframe.position.copy(mesh.position);
    }
    if (mesh.userData.baseRing) {
      mesh.userData.baseRing.position.set(mesh.position.x, 0.002, mesh.position.z);
    }

    // Update label position
    if (this._objectLabels[obj.id]) {
      this._objectLabels[obj.id].position.set(
        mesh.position.x,
        h + 0.02,
        mesh.position.z
      );
    }
  }

  _removeObjectMesh(id) {
    const mesh = this._renderedObjects[id];
    if (mesh) {
      if (mesh.userData.wireframe) {
        this._objectsGroup.remove(mesh.userData.wireframe);
        mesh.userData.wireframe.geometry.dispose();
        mesh.userData.wireframe.material.dispose();
      }
      if (mesh.userData.baseRing) {
        this._objectsGroup.remove(mesh.userData.baseRing);
        mesh.userData.baseRing.geometry.dispose();
        mesh.userData.baseRing.material.dispose();
      }
      this._objectsGroup.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
      delete this._renderedObjects[id];
    }

    const line = this._distanceLines[id];
    if (line) {
      this._objectsGroup.remove(line);
      line.geometry.dispose();
      line.material.dispose();
      delete this._distanceLines[id];
    }

    const label = this._objectLabels[id];
    if (label) {
      this._objectsGroup.remove(label);
      if (label.material.map) label.material.map.dispose();
      label.material.dispose();
      delete this._objectLabels[id];
    }
  }

  _createLabelSprite(obj) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.fillStyle = obj.within_reach ? 'rgba(0,40,0,0.8)' : 'rgba(40,20,0,0.8)';
    ctx.fillRect(0, 0, 256, 64);
    ctx.strokeStyle = obj.within_reach ? '#44ff88' : '#ff8844';
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, 254, 62);

    ctx.fillStyle = obj.within_reach ? '#44ff88' : '#ff8844';
    ctx.font = 'bold 18px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(obj.label.toUpperCase(), 128, 22);

    if (obj.category) {
      ctx.fillStyle = '#8af';
      ctx.font = '12px monospace';
      ctx.fillText(obj.category, 128, 38);
    }

    ctx.fillStyle = '#ccc';
    ctx.font = '12px monospace';
    ctx.fillText(`${obj.distance_mm}mm ${obj.within_reach ? 'REACH' : 'FAR'}`, 128, 54);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(0.12, 0.03, 1);
    return sprite;
  }

  /**
   * Clear all detected objects from the scene.
   */
  clearDetectedObjects() {
    if (!this._objectsGroup) return;
    const ids = Object.keys(this._renderedObjects).map(Number);
    for (const id of ids) {
      this._removeObjectMesh(id);
    }
  }

  /**
   * Toggle detected objects visibility.
   */
  toggleDetectedObjects(show) {
    this.showDetectedObjects = show;
    if (this._objectsGroup) {
      this._objectsGroup.visible = show;
    }
  }

  /**
   * Toggle object labels visibility.
   */
  toggleObjectLabels(show) {
    this.showObjectLabels = show;
    for (const label of Object.values(this._objectLabels)) {
      label.visible = show;
    }
  }

  dispose() {
    this._disposed = true;
    this.deactivate();
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
    }
    this.clearDetectedObjects();
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
