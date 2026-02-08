/**
 * Browser Hook for ASCII Frame Streaming
 *
 * Inject into the existing ASCII renderer to send each frame to the
 * Python WebSocket server for visual hull processing.
 *
 * Usage: Include this script after the ASCII renderer is initialized.
 * It hooks into the existing render pipeline and sends each 128×128
 * ASCII buffer over WebSocket to ws://localhost:9100.
 *
 * Message format per frame:
 *   char 0:      Camera ID — 'T' (top-down) or 'P' (profile)
 *   chars 1-8:   Zero-padded millisecond timestamp from performance.now()
 *   chars 9+:    128×128 = 16384 ASCII characters (row-major)
 *   Total: 16393 characters per message
 */

(function () {
  "use strict";

  const WS_URL = "ws://localhost:9100";
  const GRID_SIZE = 128;
  const RECONNECT_DELAY_MS = 2000;

  let ws = null;
  let reconnectTimer = null;

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";

      ws.onopen = function () {
        console.log("[VisualHull] Connected to frame server at " + WS_URL);
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
      };

      ws.onclose = function () {
        console.log("[VisualHull] Disconnected from frame server, reconnecting...");
        scheduleReconnect();
      };

      ws.onerror = function () {
        // Silently handle errors — will reconnect on close
      };
    } catch (e) {
      scheduleReconnect();
    }
  }

  function scheduleReconnect() {
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(function () {
        reconnectTimer = null;
        connect();
      }, RECONNECT_DELAY_MS);
    }
  }

  /**
   * Format a timestamp as an 8-character zero-padded string.
   * @param {number} ms - Millisecond timestamp from performance.now()
   * @returns {string} 8-character zero-padded string
   */
  function formatTimestamp(ms) {
    const ts = Math.floor(ms) % 100000000; // Keep within 8 digits
    return String(ts).padStart(8, "0");
  }

  /**
   * Send an ASCII frame to the WebSocket server.
   * Non-blocking: silently drops frames if socket is not connected.
   *
   * @param {string} cameraId - 'T' for top-down, 'P' for profile
   * @param {string|string[]} asciiBuffer - 128×128 ASCII characters.
   *        Can be a flat string of 16384 chars or an array of 128 strings.
   */
  function sendFrame(cameraId, asciiBuffer) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return; // Silently drop — don't block the render loop
    }

    // Normalize buffer to flat string
    let flat;
    if (Array.isArray(asciiBuffer)) {
      // Array of row strings — join and pad/truncate each to GRID_SIZE
      flat = asciiBuffer
        .slice(0, GRID_SIZE)
        .map(function (row) {
          if (row.length >= GRID_SIZE) return row.substring(0, GRID_SIZE);
          return row + " ".repeat(GRID_SIZE - row.length);
        })
        .join("");
      // Pad if fewer than GRID_SIZE rows
      while (flat.length < GRID_SIZE * GRID_SIZE) {
        flat += " ";
      }
    } else {
      flat = String(asciiBuffer);
      if (flat.length < GRID_SIZE * GRID_SIZE) {
        flat += " ".repeat(GRID_SIZE * GRID_SIZE - flat.length);
      } else if (flat.length > GRID_SIZE * GRID_SIZE) {
        flat = flat.substring(0, GRID_SIZE * GRID_SIZE);
      }
    }

    const timestamp = formatTimestamp(performance.now());
    const message = cameraId + timestamp + flat;

    try {
      ws.send(message);
    } catch (e) {
      // Non-blocking: silently drop on send failure
    }
  }

  // Auto-connect on load
  connect();

  // Expose API globally
  window.VisualHullHook = {
    sendFrame: sendFrame,
    connect: connect,
    isConnected: function () {
      return ws && ws.readyState === WebSocket.OPEN;
    },
  };

  console.log("[VisualHull] Browser hook loaded. Use VisualHullHook.sendFrame('T'|'P', buffer)");
})();
