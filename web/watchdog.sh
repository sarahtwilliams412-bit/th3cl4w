#!/bin/bash
# Watchdog for th3cl4w web + camera servers — auto-restarts on crash
cd "$(dirname "$0")/.."

PIDFILE="/tmp/th3cl4w-server.pid"
CAM_PIDFILE="/tmp/th3cl4w-cam.pid"

kill_server() {
    # Use PID file for reliable kills, fall back to port-based kill
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[$(date)] Sending SIGTERM to server (pid $pid)..."
            kill -TERM "$pid"
            # Wait up to 5s for graceful shutdown
            for i in $(seq 1 50); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.1
            done
            # Force kill if still alive
            if kill -0 "$pid" 2>/dev/null; then
                echo "[$(date)] Force killing server (pid $pid)..."
                kill -9 "$pid" 2>/dev/null
            fi
        fi
        rm -f "$PIDFILE"
    fi
    # Also kill anything on port 8080 as fallback
    fuser -k 8080/tcp 2>/dev/null || true
}

kill_camera() {
    if [ -f "$CAM_PIDFILE" ]; then
        pid=$(cat "$CAM_PIDFILE")
        kill -TERM "$pid" 2>/dev/null
        rm -f "$CAM_PIDFILE"
    fi
}

cleanup() {
    echo "[$(date)] Watchdog shutting down..."
    kill_server
    kill_camera
    kill "$CAM_WATCH_PID" 2>/dev/null
    wait
    exit 0
}

trap cleanup EXIT INT TERM

# --- Camera server watchdog (background) ---
(
    while true; do
        echo "[$(date)] Starting camera server..."
        python3.12 web/camera_server.py 2>&1 | tee -a /tmp/th3cl4w-cam.log &
        echo $! > "$CAM_PIDFILE"
        wait $!
        echo "[$(date)] Camera server exited ($?), restarting in 3s..."
        sleep 3
    done
) &
CAM_WATCH_PID=$!

# --- Main web server watchdog ---
while true; do
    kill_server  # Ensure clean state before starting
    echo "[$(date)] Starting server..."
    python3.12 web/server.py --interface eno1 2>&1 | tee -a /tmp/server.log
    echo "[$(date)] Server exited ($?), restarting in 3s..."
    sleep 3
done
