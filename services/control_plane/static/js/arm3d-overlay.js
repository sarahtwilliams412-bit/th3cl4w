// ═══════════════════════════════════════════════════════
//  arm3d-overlay.js — Three.js D1 Arm overlay for
//  Factory 3D and Real World 3D tabs
// ═══════════════════════════════════════════════════════
'use strict';

(function() {

if (typeof D1Arm3D === 'undefined' || typeof THREE === 'undefined') {
  console.warn('[arm3d-overlay] Three.js or arm3d.js not loaded');
  return;
}

// ──────────────────────────────────────────────
// Shared state
// ──────────────────────────────────────────────
let currentJoints = [0,0,0,0,0,0];
let currentGripper = 0;

// Listen for arm state from main WS
const _origWsConnect = window.connectWS;
// We'll poll armState from the global instead
setInterval(() => {
  if (typeof armState !== 'undefined' && armState.joints) {
    currentJoints = armState.joints.slice(0, 6);
    currentGripper = armState.gripper || 0;
  }
}, 100);

// ──────────────────────────────────────────────
// Factory 3D — Three.js overlay
// ──────────────────────────────────────────────
let fScene, fCamera, fRenderer, fControls, fArm, fTable, fAxes;
let fCanvas, fActive = false, fAnimId;
let fOverlayVisible = false;

function fCreateOverlay() {
  const tab = document.getElementById('tabFactory');
  if (!tab || document.getElementById('factory3dOverlay')) return;

  // Create overlay canvas
  fCanvas = document.createElement('canvas');
  fCanvas.id = 'factory3dOverlay';
  fCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;display:none;';
  tab.appendChild(fCanvas);

  // Toggle button
  const btn = document.createElement('button');
  btn.id = 'btnFactory3DToggle';
  btn.className = 'btn';
  btn.style.cssText = 'position:absolute;top:8px;right:8px;z-index:20;font-size:10px;padding:4px 10px;pointer-events:auto;';
  btn.textContent = '🤖 3D ARM';
  btn.title = 'Toggle Three.js arm overlay';
  btn.onclick = () => {
    fOverlayVisible = !fOverlayVisible;
    fCanvas.style.display = fOverlayVisible ? 'block' : 'none';
    fCanvas.style.pointerEvents = fOverlayVisible ? 'auto' : 'none';
    btn.classList.toggle('active', fOverlayVisible);
    if (fOverlayVisible) fResize();
  };
  tab.appendChild(btn);

  // Scene
  fScene = new THREE.Scene();
  fScene.background = null; // transparent

  fCamera = new THREE.PerspectiveCamera(50, 1, 0.01, 10);
  fCamera.position.set(0.6, 0.5, 0.6);

  fRenderer = new THREE.WebGLRenderer({ canvas: fCanvas, alpha: true, antialias: true });
  fRenderer.setClearColor(0x000000, 0);
  fRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  fControls = new THREE.OrbitControls(fCamera, fCanvas);
  fControls.target.set(0, 0.15, 0);
  fControls.enableDamping = true;
  fControls.dampingFactor = 0.1;
  fControls.update();

  // Lights
  fScene.add(new THREE.AmbientLight(0x404060, 0.5));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 2, 1);
  fScene.add(dir);
  fScene.add(new THREE.PointLight(0xffaa33, 0.4, 3));

  // Grid
  fScene.add(new THREE.GridHelper(2, 40, 0x1a2744, 0x111827));

  // Arm
  fArm = new D1Arm3D({ smoothing: 0.15 });
  fScene.add(fArm.group);

  // Table
  fTable = createWorkTable({ width: 0.8, depth: 0.5, y: -0.01 });
  fTable.position.set(0.15, 0, 0);
  fScene.add(fTable);

  // Axes
  fAxes = createAxes(0.1);
  fScene.add(fAxes);

  // Factory-style ambient (dark with warm point lights)
  const warmLight = new THREE.PointLight(0xffcc66, 0.3, 2);
  warmLight.position.set(-0.3, 0.4, -0.3);
  fScene.add(warmLight);
}

function fResize() {
  if (!fCanvas || !fRenderer) return;
  const tab = document.getElementById('tabFactory');
  const w = tab.clientWidth, h = tab.clientHeight;
  fCanvas.width = w;
  fCanvas.height = h;
  fRenderer.setSize(w, h);
  fCamera.aspect = w / h;
  fCamera.updateProjectionMatrix();
}

function fAnimate(time) {
  if (!fActive) return;
  fAnimId = requestAnimationFrame(fAnimate);
  if (!fOverlayVisible) return;

  fArm.setTarget([...currentJoints, 0], currentGripper);
  fArm.animate(0.016);
  fControls.update();
  fRenderer.render(fScene, fCamera);
}

// Hook into factory activate/deactivate
const origFwActivate = window.fw_activate;
const origFwDeactivate = window.fw_deactivate;

window.fw_activate = function() {
  if (origFwActivate) origFwActivate();
  fCreateOverlay();
  fActive = true;
  fResize();
  fAnimId = requestAnimationFrame(fAnimate);
  window.addEventListener('resize', fResize);
};

window.fw_deactivate = function() {
  fActive = false;
  if (fAnimId) cancelAnimationFrame(fAnimId);
  window.removeEventListener('resize', fResize);
  if (origFwDeactivate) origFwDeactivate();
};

// ──────────────────────────────────────────────
// Real World 3D — Three.js overlay
// ──────────────────────────────────────────────
let rScene, rCamera, rRenderer, rControls, rArm;
let rCanvas, rActive = false, rAnimId;
let rOverlayVisible = false;

function rCreateOverlay() {
  const tab = document.getElementById('tabRealworld');
  if (!tab || document.getElementById('realworld3dOverlay')) return;

  rCanvas = document.createElement('canvas');
  rCanvas.id = 'realworld3dOverlay';
  rCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;display:none;';
  tab.appendChild(rCanvas);

  const btn = document.createElement('button');
  btn.id = 'btnRealworld3DToggle';
  btn.className = 'btn';
  btn.style.cssText = 'position:absolute;top:8px;right:8px;z-index:20;font-size:10px;padding:4px 10px;pointer-events:auto;';
  btn.textContent = '🤖 3D ARM';
  btn.title = 'Toggle Three.js arm overlay';
  btn.onclick = () => {
    rOverlayVisible = !rOverlayVisible;
    rCanvas.style.display = rOverlayVisible ? 'block' : 'none';
    rCanvas.style.pointerEvents = rOverlayVisible ? 'auto' : 'none';
    btn.classList.toggle('active', rOverlayVisible);
    if (rOverlayVisible) rResize();
  };
  tab.appendChild(btn);

  rScene = new THREE.Scene();
  rScene.background = null;

  rCamera = new THREE.PerspectiveCamera(50, 1, 0.01, 10);
  rCamera.position.set(0.4, 0.35, 0.5);

  rRenderer = new THREE.WebGLRenderer({ canvas: rCanvas, alpha: true, antialias: true });
  rRenderer.setClearColor(0x000000, 0);
  rRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  rControls = new THREE.OrbitControls(rCamera, rCanvas);
  rControls.target.set(0, 0.15, 0);
  rControls.enableDamping = true;
  rControls.dampingFactor = 0.1;
  rControls.update();

  // Lights — brighter for real-world overlay
  rScene.add(new THREE.AmbientLight(0x606080, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(0.5, 2, 1);
  rScene.add(dir);

  // Arm (slightly more emissive for visibility against voxels)
  rArm = new D1Arm3D({ smoothing: 0.12 });
  // Make joints glow more for overlay visibility
  rArm.jointMeshes.forEach(m => {
    m.material.emissiveIntensity = 0.6;
  });
  rScene.add(rArm.group);

  // Subtle grid
  const grid = new THREE.GridHelper(1, 20, 0x1a2744, 0x0d1117);
  grid.material.transparent = true;
  grid.material.opacity = 0.3;
  rScene.add(grid);

  // Axes
  rScene.add(createAxes(0.08));

  // Try loading camera extrinsics to position camera
  loadCameraExtrinsics();
}

async function loadCameraExtrinsics() {
  try {
    const resp = await fetch('/api/viz/calibration');
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.ok && data.camera_params) {
      const cam = data.camera_params.cam0 || data.camera_params.cam1;
      if (cam) {
        // Position overlay camera to roughly match real camera viewpoint
        // Convert rodrigues to euler (approximate)
        const angle = Math.sqrt(cam.rx*cam.rx + cam.ry*cam.ry + cam.rz*cam.rz);
        if (angle > 0.01) {
          rCamera.position.set(cam.tx || 0.4, cam.tz || 0.35, cam.ty || 0.5);
          rControls.update();
        }
        console.log('[arm3d-overlay] Camera extrinsics loaded for RW3D');
      }
    }
  } catch(e) {}
}

function rResize() {
  if (!rCanvas || !rRenderer) return;
  const tab = document.getElementById('tabRealworld');
  const w = tab.clientWidth, h = tab.clientHeight;
  rCanvas.width = w;
  rCanvas.height = h;
  rRenderer.setSize(w, h);
  rCamera.aspect = w / h;
  rCamera.updateProjectionMatrix();
}

function rAnimate(time) {
  if (!rActive) return;
  rAnimId = requestAnimationFrame(rAnimate);
  if (!rOverlayVisible) return;

  rArm.setTarget([...currentJoints, 0], currentGripper);
  rArm.animate(0.016);
  rControls.update();
  rRenderer.render(rScene, rCamera);
}

const origRwActivate = window.rw_activate;
const origRwDeactivate = window.rw_deactivate;

window.rw_activate = function() {
  if (origRwActivate) origRwActivate();
  rCreateOverlay();
  rActive = true;
  rResize();
  rAnimId = requestAnimationFrame(rAnimate);
  window.addEventListener('resize', rResize);
};

window.rw_deactivate = function() {
  rActive = false;
  if (rAnimId) cancelAnimationFrame(rAnimId);
  window.removeEventListener('resize', rResize);
  if (origRwDeactivate) origRwDeactivate();
};

console.log('[arm3d-overlay] Loaded — toggle 3D arm with 🤖 button in Factory/RealWorld tabs');

})();
