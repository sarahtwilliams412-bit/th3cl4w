// ═══════════════════════════════════════════════════════
//  mini-sim.js — Lightweight mini 3D arm viewer
//  Reuses D1Arm3D from arm3d.js
// ═══════════════════════════════════════════════════════
'use strict';

class MiniArmView {
  constructor(containerEl, options = {}) {
    this.container = containerEl;
    this.running = false;
    this._rafId = null;
    this._lastTime = 0;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1525);

    // Camera — isometric-ish
    const w = containerEl.clientWidth || 400;
    const h = containerEl.clientHeight || 300;
    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.01, 5);
    this.camera.position.set(0.45, 0.45, 0.45);
    this.camera.lookAt(0, 0.2, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerEl.appendChild(this.renderer.domElement);

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(1, 2, 1.5);
    this.scene.add(dir);

    // Arm
    this.arm = new D1Arm3D({ smoothing: 0.15 });
    this.scene.add(this.arm.group);

    // Ghost arm (initially hidden)
    this.ghostArm = null;
    this._ghostVisible = false;

    // ResizeObserver
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(containerEl);
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _ensureGhost() {
    if (this.ghostArm) return;
    this.ghostArm = new D1Arm3D({ smoothing: 0.08 });
    this.ghostArm.group.traverse(child => {
      if (child.material) {
        child.material = child.material.clone();
        child.material.transparent = true;
        child.material.opacity = 0.3;
        child.material.emissiveIntensity = 0.1;
      }
    });
    this.ghostArm.group.visible = false;
    this.scene.add(this.ghostArm.group);
  }

  setJoints(anglesDeg) {
    this.arm.setTarget(anglesDeg, undefined);
  }

  setGripper(mm) {
    this.arm.targetGripperWidth = mm;
  }

  setGhostJoints(anglesDeg) {
    this._ensureGhost();
    this.ghostArm.setImmediate(anglesDeg, undefined);
    this.ghostArm.group.visible = true;
    this._ghostVisible = true;
  }

  clearGhost() {
    if (this.ghostArm) {
      this.ghostArm.group.visible = false;
      this._ghostVisible = false;
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    this._lastTime = performance.now();
    this._loop();
  }

  stop() {
    this.running = false;
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  _loop() {
    if (!this.running) return;
    this._rafId = requestAnimationFrame(() => this._loop());
    const now = performance.now();
    const dt = Math.min((now - this._lastTime) / 1000, 0.1);
    this._lastTime = now;

    // Sync from global armState
    if (typeof armState !== 'undefined' && armState.joints) {
      this.arm.setTarget(armState.joints, armState.gripper || 0);
    }

    this.arm.animate(dt);
    if (this.ghostArm && this._ghostVisible) {
      this.ghostArm.animate(dt);
    }
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.stop();
    this._resizeObserver.disconnect();
    this.arm.dispose();
    if (this.ghostArm) this.ghostArm.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}

// ── Mini-sim manager ──
const _miniSims = {};

function getMiniSim(panelId) {
  return _miniSims[panelId] || null;
}

function activateMiniSim(panelId) {
  const container = document.getElementById('miniSim-' + panelId);
  if (!container) return;
  let sim = _miniSims[panelId];
  if (!sim) {
    sim = new MiniArmView(container);
    _miniSims[panelId] = sim;
  }
  sim.start();
}

function deactivateMiniSim(panelId) {
  const sim = _miniSims[panelId];
  if (sim) sim.stop();
}
