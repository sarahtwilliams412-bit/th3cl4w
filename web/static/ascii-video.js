// ascii-video.js — ASCII Video Tab engine (extracted from index.html)
'use strict';

(function() {
  const ascManagers = [null, null]; // WSManager instances
  let ascPaused = false;
  let ascColorMode = false;
  let ascInvertMode = true;
  let ascFrameCount = 0;
  let ascLastFpsTime = performance.now();

  const ascOutputs = [document.getElementById('asciiOutput0'), document.getElementById('asciiOutput1')];
  const ascDots = [document.getElementById('ascDot0'), document.getElementById('ascDot1')];
  const ascFpsEl = document.getElementById('ascFps');

  function ascSettings(camId) {
    return {
      type: 'settings',
      cam: camId,
      charset: document.getElementById('ascCharset').value,
      width: parseInt(document.getElementById('ascWidth').value) || 320,
      height: parseInt(document.getElementById('ascHeight').value) || 140,
      color: ascColorMode,
      invert: ascInvertMode
    };
  }

  function ascSendSettings(camId) {
    if (ascManagers[camId]) ascManagers[camId].send(ascSettings(camId));
  }

  function ascSendAllSettings() {
    ascSendSettings(0);
    ascSendSettings(1);
  }

  function renderFrame(camId, data) {
    const out = ascOutputs[camId];
    if (data.colors && ascColorMode) {
      out.classList.add('color-mode');
      out.innerHTML = '';
      const frag = document.createDocumentFragment();
      for (let y = 0; y < data.lines.length; y++) {
        const line = data.lines[y], rowC = data.colors[y];
        for (let x = 0; x < line.length; x++) {
          const sp = document.createElement('span');
          const c = rowC[x];
          sp.style.color = `rgb(${c[0]},${c[1]},${c[2]})`;
          sp.textContent = line[x];
          frag.appendChild(sp);
        }
        frag.appendChild(document.createTextNode('\n'));
      }
      out.appendChild(frag);
    } else {
      out.classList.remove('color-mode');
      out.textContent = data.lines.join('\n');
    }
  }

  function connectCam(camId) {
    if (ascManagers[camId] && ascManagers[camId].connected) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/ascii`;
    const mgr = new WSManager(url, { idleTimeout: 0, initialDelay: 2000 }); // no idle timeout for streaming
    ascManagers[camId] = mgr;
    ascDots[camId].className = 'ascii-status-dot';

    mgr.onOpen(() => {
      ascDots[camId].className = 'ascii-status-dot on';
      mgr.send(ascSettings(camId));
    });
    mgr.onClose(() => {
      ascDots[camId].className = 'ascii-status-dot';
    });
    mgr.onError(() => {
      ascDots[camId].className = 'ascii-status-dot err';
    });
    mgr.onMessage((evt) => {
      if (ascPaused) return;
      const data = JSON.parse(evt.data);
      if (data.type === 'frame') {
        renderFrame(camId, data);
        ascFrameCount++;
        const now = performance.now();
        if (now - ascLastFpsTime >= 1000) {
          ascFpsEl.textContent = (ascFrameCount / ((now - ascLastFpsTime) / 1000)).toFixed(1) + ' fps';
          ascFrameCount = 0;
          ascLastFpsTime = now;
        }
      }
    });
    mgr.connect();
  }

  function disconnectCam(camId) {
    if (ascManagers[camId]) { ascManagers[camId].disconnect(); ascManagers[camId] = null; }
    ascDots[camId].className = 'ascii-status-dot';
  }

  window.asciiConnect = function() {
    connectCam(0);
    connectCam(1);
    ascFitFont();
  };

  window.asciiDisconnect = function() {
    disconnectCam(0);
    disconnectCam(1);
    ascFpsEl.textContent = '-- fps';
  };

  function ascFitFont() {
    const vp = document.getElementById('asciiViewport');
    if (!vp) return;
    const w = parseInt(document.getElementById('ascWidth').value) || 320;
    const h = parseInt(document.getElementById('ascHeight').value) || 140;
    const maxW = (vp.clientWidth - 40) / 2;
    const maxH = vp.clientHeight - 40;
    const sizeByW = maxW / (w * 0.62);
    const sizeByH = maxH / (h * 1.15);
    const sz = Math.max(4, Math.min(16, Math.floor(Math.min(sizeByW, sizeByH)))) + 'px';
    ascOutputs[0].style.fontSize = sz;
    ascOutputs[1].style.fontSize = sz;
  }

  document.getElementById('ascCharset').addEventListener('change', ascSendAllSettings);
  document.getElementById('ascWidth').addEventListener('change', () => { ascFitFont(); ascSendAllSettings(); });
  document.getElementById('ascHeight').addEventListener('change', () => { ascFitFont(); ascSendAllSettings(); });

  document.getElementById('ascColor').addEventListener('click', function() {
    ascColorMode = !ascColorMode;
    this.classList.toggle('on', ascColorMode);
    ascSendAllSettings();
  });
  document.getElementById('ascInvert').addEventListener('click', function() {
    ascInvertMode = !ascInvertMode;
    this.classList.toggle('on', ascInvertMode);
    ascSendAllSettings();
  });
  document.getElementById('ascPause').addEventListener('click', function() {
    ascPaused = !ascPaused;
    this.classList.toggle('on', ascPaused);
    this.textContent = ascPaused ? 'RESUME' : 'PAUSE';
  });

  window.addEventListener('resize', ascFitFont);

  // --- FK Overlay for calibration debugging ---
  let fkOverlayEnabled = false;
  const fkOverlayBtn = document.getElementById('ascFkOverlay');
  const FK_MARKER_COLORS = ['#ff0000','#00ff00','#0088ff','#ffaa00','#ff00ff'];
  const FK_MARKER_CHARS = ['●','●','●','●','◆'];

  fkOverlayBtn.addEventListener('click', function() {
    fkOverlayEnabled = !fkOverlayEnabled;
    this.classList.toggle('on', fkOverlayEnabled);
  });

  const origRenderFrame = renderFrame;
  renderFrame = function(camId, data) {
    origRenderFrame(camId, data);
    if (!fkOverlayEnabled) return;

    fetch('/api/state').then(r=>r.json()).then(state=>{
      if(!state||!state.joints) return;
      const positions = fkPositions(state.joints);
      if(!CAM_PARAMS) return;

      const ascW = parseInt(document.getElementById('ascWidth').value)||320;
      const ascH = parseInt(document.getElementById('ascHeight').value)||140;
      const camW = 1920, camH = 1080;

      const out = ascOutputs[camId];
      const overlay = document.createElement('div');
      overlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;';
      out.style.position = 'relative';

      positions.forEach((p3d, idx) => {
        const p2d = projectPoint3D(p3d, CAM_PARAMS);
        if(!p2d) return;
        const charX = Math.round(p2d[0] / camW * ascW);
        const charY = Math.round(p2d[1] / camH * ascH);
        if(charX<0||charX>=ascW||charY<0||charY>=ascH) return;

        const marker = document.createElement('span');
        marker.textContent = FK_MARKER_CHARS[idx];
        marker.style.cssText = `position:absolute;color:${FK_MARKER_COLORS[idx]};font-weight:bold;text-shadow:0 0 3px #000;pointer-events:none;`;
        const fontSize = parseFloat(getComputedStyle(out).fontSize)||8;
        marker.style.left = (charX * fontSize * 0.62) + 'px';
        marker.style.top = (charY * fontSize * 1.15) + 'px';
        overlay.appendChild(marker);
      });

      const oldOverlay = out.querySelector('.fk-overlay');
      if(oldOverlay) oldOverlay.remove();
      overlay.className = 'fk-overlay';
      out.appendChild(overlay);
    }).catch(()=>{});
  };
})();
