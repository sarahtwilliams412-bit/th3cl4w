#!/bin/bash
# Watchdog for th3cl4w web server — auto-restarts on crash
cd "$(dirname "$0")/.."
while true; do
    echo "[$(date)] Starting server..."
    python3.12 web/server.py --interface eno1 2>&1 | tee -a /tmp/server.log
    echo "[$(date)] Server exited ($?), restarting in 3s..."
    sleep 3
done
