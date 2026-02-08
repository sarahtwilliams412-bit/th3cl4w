// ws-manager.js — Shared WebSocket connection manager with backoff & idle timeout
'use strict';

class WSManager {
  constructor(url, options = {}) {
    this.url = url;
    this.maxReconnectDelay = options.maxReconnectDelay || 10000;
    this.initialDelay = options.initialDelay || 1000;
    this.idleTimeout = options.idleTimeout || 60000; // auto-disconnect after 60s idle
    this._ws = null;
    this._reconnectDelay = this.initialDelay;
    this._reconnectTimer = null;
    this._idleTimer = null;
    this._messageCallback = null;
    this._openCallback = null;
    this._closeCallback = null;
    this._errorCallback = null;
    this._intentionalClose = false;
    this._state = 'disconnected'; // disconnected | connecting | connected
  }

  get state() { return this._state; }
  get connected() { return this._state === 'connected'; }
  get ws() { return this._ws; }

  connect() {
    if (this._ws && this._ws.readyState <= 1) return;
    this._intentionalClose = false;
    this._state = 'connecting';
    this._ws = new WebSocket(this.url);

    this._ws.onopen = () => {
      this._state = 'connected';
      this._reconnectDelay = this.initialDelay;
      this._resetIdleTimer();
      WSManager._active.add(this);
      if (this._openCallback) this._openCallback();
    };

    this._ws.onclose = () => {
      this._state = 'disconnected';
      this._ws = null;
      WSManager._active.delete(this);
      this._clearIdleTimer();
      if (this._closeCallback) this._closeCallback();
      if (!this._intentionalClose) {
        this._scheduleReconnect();
      }
    };

    this._ws.onerror = () => {
      if (this._errorCallback) this._errorCallback();
    };

    this._ws.onmessage = (evt) => {
      this._resetIdleTimer();
      if (this._messageCallback) this._messageCallback(evt);
    };
  }

  disconnect() {
    this._intentionalClose = true;
    this._clearReconnectTimer();
    this._clearIdleTimer();
    if (this._ws) {
      this._ws.onclose = null;
      this._ws.close();
      this._ws = null;
    }
    this._state = 'disconnected';
    WSManager._active.delete(this);
    if (this._closeCallback) this._closeCallback();
  }

  send(data) {
    if (this._ws && this._ws.readyState === 1) {
      this._ws.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }

  onOpen(cb) { this._openCallback = cb; return this; }
  onMessage(cb) { this._messageCallback = cb; return this; }
  onClose(cb) { this._closeCallback = cb; return this; }
  onError(cb) { this._errorCallback = cb; return this; }

  _scheduleReconnect() {
    this._clearReconnectTimer();
    this._reconnectTimer = setTimeout(() => {
      this._reconnectDelay = Math.min(this._reconnectDelay * 2, this.maxReconnectDelay);
      this.connect();
    }, this._reconnectDelay);
  }

  _clearReconnectTimer() {
    if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
  }

  _resetIdleTimer() {
    this._clearIdleTimer();
    if (this.idleTimeout > 0) {
      this._idleTimer = setTimeout(() => this.disconnect(), this.idleTimeout);
    }
  }

  _clearIdleTimer() {
    if (this._idleTimer) { clearTimeout(this._idleTimer); this._idleTimer = null; }
  }

  // --- Static tracking ---
  static _active = new Set();
  static get activeCount() { return WSManager._active.size; }
  static disconnectAll() { WSManager._active.forEach(m => m.disconnect()); }
}

window.WSManager = WSManager;
