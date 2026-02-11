#!/bin/bash
# Watchdog for th3cl4w web + camera servers — auto-restarts on crash
#
# NOTE: systemd user services are now the preferred way to manage th3cl4w.
# See systemd/ directory and scripts/th3cl4w-ctl.sh for the new approach.
# This script is kept as a fallback only.

# Re-exec under setsid so the watchdog survives exec session cleanup
if [ -z "$WATCHDOG_SETSID" ]; then
    export WATCHDOG_SETSID=1
    exec setsid "$0" "$@"
fi

cd "$(dirname "$0")/.."

# API keys are loaded via python-dotenv from .env — no need to source bashrc

WATCHDOG_LOCK="/tmp/th3cl4w-watchdog.lock"

# Kill ALL processes holding the lock file (previous watchdogs and their children)
if [ -f "$WATCHDOG_LOCK" ]; then
    LOCK_HOLDERS=$(fuser "$WATCHDOG_LOCK" 2>/dev/null | tr -s ' ')
    if [ -n "$LOCK_HOLDERS" ]; then
        echo "[$(date)] Killing lock holders: $LOCK_HOLDERS"
        for pid in $LOCK_HOLDERS; do
            [ "$pid" -eq $$ ] 2>/dev/null && continue
            # Kill entire process group first (gets children), then individual
            kill -- -"$pid" 2>/dev/null
            kill "$pid" 2>/dev/null
        done
        sleep 1
        # Force-kill any survivors
        for pid in $LOCK_HOLDERS; do
            [ "$pid" -eq $$ ] 2>/dev/null && continue
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        done
        sleep 0.5
    fi
fi

# Also kill any old watchdog processes by name (belt and suspenders)
MY_PID=$$
for pid in $(pgrep -f 'watchdog\.sh' 2>/dev/null); do
    [ "$pid" -eq "$MY_PID" ] && continue
    kill -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
done

# Kill ALL old server processes on ports 8080-8084
kill_all_server_ports() {
    for port in 8080 8081 8082 8083 8084; do
        fuser -k "$port/tcp" 2>/dev/null || true
    done
    sleep 0.5
    # Force-kill anything that survived SIGTERM
    for port in 8080 8081 8082 8083 8084; do
        if fuser "$port/tcp" >/dev/null 2>&1; then
            fuser -k -9 "$port/tcp" 2>/dev/null || true
        fi
    done
}
echo "[$(date)] Cleaning up old server processes on ports 8080-8084..."
kill_all_server_ports
sleep 1

# Acquire lock
exec 200>"$WATCHDOG_LOCK"
if ! flock -n 200; then
    echo "[$(date)] Another watchdog is already running (lock held after cleanup!). Exiting."
    exit 1
fi

PIDFILE="/tmp/th3cl4w-server.pid"
CAM_PIDFILE="/tmp/th3cl4w-cam.pid"
LOC_PIDFILE="/tmp/th3cl4w-loc.pid"
ASCII_PIDFILE="/tmp/th3cl4w-ascii.pid"
MAP_PIDFILE="/tmp/th3cl4w-map.pid"

kill_server() {
    # 1) Kill by PID file
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[$(date)] Sending SIGTERM to server (pid $pid)..."
            kill -TERM "$pid"
            for i in $(seq 1 50); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.1
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "[$(date)] Force killing server (pid $pid)..."
                kill -9 "$pid" 2>/dev/null
            fi
        fi
        rm -f "$PIDFILE"
    fi

    # 2) Kill ALL python processes matching our server pattern (catches orphans)
    pkill -9 -f 'python3.*web/server\.py' 2>/dev/null || true

    # 3) Kill anything still holding port 8080
    fuser -k 8080/tcp 2>/dev/null || true

    # 4) Wait a moment and verify port is free
    sleep 0.5
    if fuser 8080/tcp >/dev/null 2>&1; then
        echo "[$(date)] WARNING: Port 8080 STILL in use after cleanup, force killing..."
        fuser -k -9 8080/tcp 2>/dev/null || true
        sleep 0.5
    fi
}

kill_camera() {
    if [ -f "$CAM_PIDFILE" ]; then
        pid=$(cat "$CAM_PIDFILE")
        kill -TERM "$pid" 2>/dev/null
        rm -f "$CAM_PIDFILE"
    fi
    fuser -k 8081/tcp 2>/dev/null || true
}

kill_location() {
    if [ -f "$LOC_PIDFILE" ]; then
        pid=$(cat "$LOC_PIDFILE")
        kill -TERM "$pid" 2>/dev/null
        rm -f "$LOC_PIDFILE"
    fi
    fuser -k 8082/tcp 2>/dev/null || true
}

kill_ascii() {
    if [ -f "$ASCII_PIDFILE" ]; then
        pid=$(cat "$ASCII_PIDFILE")
        kill -TERM "$pid" 2>/dev/null
        rm -f "$ASCII_PIDFILE"
    fi
    fuser -k 8084/tcp 2>/dev/null || true
}

kill_map() {
    if [ -f "$MAP_PIDFILE" ]; then
        pid=$(cat "$MAP_PIDFILE")
        kill -TERM "$pid" 2>/dev/null
        rm -f "$MAP_PIDFILE"
    fi
    fuser -k 8083/tcp 2>/dev/null || true
}

cleanup() {
    echo "[$(date)] Watchdog shutting down..."
    kill_server
    kill_camera
    kill_location
    kill_map
    kill_ascii
    kill "$CAM_WATCH_PID" 2>/dev/null
    kill "$LOC_WATCH_PID" 2>/dev/null
    kill "$MAP_WATCH_PID" 2>/dev/null
    kill "$ASCII_WATCH_PID" 2>/dev/null
    # Final sweep: kill anything still on our ports
    kill_all_server_ports
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

# --- Location server watchdog (background) ---
(
    # Wait for camera server to start first
    sleep 5
    while true; do
        echo "[$(date)] Starting location server..."
        python3.12 web/location_server.py 2>&1 | tee -a /tmp/th3cl4w-loc.log &
        echo $! > "$LOC_PIDFILE"
        wait $!
        echo "[$(date)] Location server exited ($?), restarting in 3s..."
        sleep 3
    done
) &
LOC_WATCH_PID=$!

# --- Map server watchdog (background) ---
(
    # Wait for camera + location servers to start first
    sleep 8
    while true; do
        echo "[$(date)] Starting map server..."
        python3.12 web/map_server.py 2>&1 | tee -a /tmp/th3cl4w-map.log &
        echo $! > "$MAP_PIDFILE"
        wait $!
        echo "[$(date)] Map server exited ($?), restarting in 3s..."
        sleep 3
    done
) &
MAP_WATCH_PID=$!

# --- ASCII video server watchdog (background) ---
(
    # Wait for camera server to start first
    sleep 6
    while true; do
        echo "[$(date)] Starting ASCII video server..."
        python3.12 web/ascii_server.py 2>&1 | tee -a /tmp/th3cl4w-ascii.log &
        echo $! > "$ASCII_PIDFILE"
        wait $!
        echo "[$(date)] ASCII video server exited ($?), restarting in 3s..."
        sleep 3
    done
) &
ASCII_WATCH_PID=$!

# --- Main web server watchdog ---
while true; do
    # ALWAYS clean up before starting — catches orphans from previous watchdog instances
    kill_server
    echo "[$(date)] Starting server..."
    python3.12 web/server.py --interface eno1 2>&1 | tee -a /tmp/server.log
    RC=$?
    echo "[$(date)] Server exited ($RC), restarting in 3s..."
    sleep 3
done
