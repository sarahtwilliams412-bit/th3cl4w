#!/bin/bash
# Watchdog for th3cl4w web + camera servers — auto-restarts on crash
cd "$(dirname "$0")/.."

# --- Camera server watchdog (background) ---
(
    while true; do
        echo "[$(date)] Starting camera server..."
        python3.12 web/camera_server.py 2>&1 | tee -a /tmp/th3cl4w-cam.log
        echo "[$(date)] Camera server exited ($?), restarting in 3s..."
        sleep 3
    done
) &
CAM_PID=$!
trap "kill $CAM_PID 2>/dev/null; wait" EXIT

# --- Main web server watchdog ---
while true; do
    echo "[$(date)] Starting server..."
    python3.12 web/server.py --interface eno1 2>&1 | tee -a /tmp/server.log
    echo "[$(date)] Server exited ($?), restarting in 3s..."
    sleep 3
done
