#!/bin/bash
# ============================================================
# th3cl4w Cold Start — D1 Arm Startup Script
# ============================================================
# Handles everything needed to go from "arm plugged in" to "ready"
#
# Usage: ./scripts/startup.sh [--skip-network] [--skip-test]
#
# What this does (in order):
#   1. Configure network (static IP on eno1 for D1 subnet)
#   2. Verify D1 is reachable
#   3. Kill any stale server/watchdog processes
#   4. Start camera + web servers via watchdog
#   5. Wait for DDS connection
#   6. Power on + enable the arm
#   7. Run basic joint test (small moves to verify feedback)
#   8. Return to home position
#
# Requirements:
#   - D1 arm powered on and ethernet connected to eno1
#   - sudo access (for network config, or pre-configure)

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SKIP_NETWORK=false
SKIP_TEST=false
D1_IP="192.168.123.100"
LOCAL_IP="192.168.123.10"
INTERFACE="eno1"
SERVER_PORT=8080
API="http://localhost:${SERVER_PORT}"
MAX_WAIT=30

for arg in "$@"; do
  case $arg in
    --skip-network) SKIP_NETWORK=true ;;
    --skip-test) SKIP_TEST=true ;;
  esac
done

log() { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
ok()  { echo -e "${GREEN}  ✓${NC} $1"; }
err() { echo -e "${RED}  ✗${NC} $1"; }
warn(){ echo -e "${YELLOW}  ⚠${NC} $1"; }

api_get()  { curl -sf "${API}$1" 2>/dev/null; }
api_post() { curl -sf -X POST "${API}$1" 2>/dev/null; }
api_post_json() { curl -sf -X POST -H 'Content-Type: application/json' -d "$2" "${API}$1" 2>/dev/null; }

# ============================================================
# Step 1: Network Configuration
# ============================================================
step_network() {
  log "Step 1: Network configuration"
  
  if $SKIP_NETWORK; then
    warn "Skipping network setup (--skip-network)"
    return 0
  fi

  # Check if interface exists and is UP
  if ! ip link show "$INTERFACE" >/dev/null 2>&1; then
    err "Interface $INTERFACE not found!"
    err "Is the ethernet cable connected?"
    return 1
  fi

  local state
  state=$(ip link show "$INTERFACE" | grep -o "state [A-Z]*" | awk '{print $2}')
  if [ "$state" != "UP" ]; then
    err "Interface $INTERFACE is $state (expected UP)"
    err "Check ethernet cable connection"
    return 1
  fi
  ok "Interface $INTERFACE is UP"

  # Check if IP is already assigned
  if ip addr show "$INTERFACE" | grep -q "$LOCAL_IP"; then
    ok "IP $LOCAL_IP already configured on $INTERFACE"
  else
    warn "No IP on $INTERFACE — configuring $LOCAL_IP/24"
    
    # Try nmcli first (persists), fall back to ip addr (temporary)
    if nmcli device show "$INTERFACE" >/dev/null 2>&1; then
      # Use NetworkManager for persistence
      nmcli con mod "netplan-${INTERFACE}" ipv4.addresses "${LOCAL_IP}/24" ipv4.method manual 2>/dev/null \
        || nmcli con add type ethernet con-name "d1-arm" ifname "$INTERFACE" \
           ipv4.addresses "${LOCAL_IP}/24" ipv4.method manual 2>/dev/null \
        || {
          warn "nmcli failed, falling back to sudo ip addr (non-persistent)"
          sudo ip addr add "${LOCAL_IP}/24" dev "$INTERFACE" 2>/dev/null || {
            err "Failed to assign IP. Run manually:"
            err "  sudo ip addr add ${LOCAL_IP}/24 dev ${INTERFACE}"
            return 1
          }
        }
      # Bring the connection up
      nmcli con up "netplan-${INTERFACE}" 2>/dev/null \
        || nmcli con up "d1-arm" 2>/dev/null \
        || true
    else
      sudo ip addr add "${LOCAL_IP}/24" dev "$INTERFACE" 2>/dev/null || {
        err "Failed to assign IP. Run manually:"
        err "  sudo ip addr add ${LOCAL_IP}/24 dev ${INTERFACE}"
        return 1
      }
    fi
    
    sleep 1
    if ip addr show "$INTERFACE" | grep -q "$LOCAL_IP"; then
      ok "IP $LOCAL_IP assigned to $INTERFACE"
    else
      err "IP assignment failed"
      return 1
    fi
  fi
}

# ============================================================
# Step 2: Verify D1 Reachable
# ============================================================
step_ping_d1() {
  log "Step 2: Checking D1 arm connectivity"
  
  local attempts=5
  for i in $(seq 1 $attempts); do
    if ping -c 1 -W 1 "$D1_IP" >/dev/null 2>&1; then
      ok "D1 arm reachable at $D1_IP"
      return 0
    fi
    [ $i -lt $attempts ] && sleep 1
  done

  err "D1 arm not reachable at $D1_IP after $attempts attempts"
  err "Check:"
  err "  - Is the arm powered on? (LED should be solid)"
  err "  - Is ethernet connected between NUC and arm?"
  err "  - Try: ping $D1_IP"
  
  # Scan subnet for any device
  warn "Scanning 192.168.123.0/24 for devices..."
  local found=false
  for ip in $(seq 1 254); do
    if ping -c 1 -W 0.3 "192.168.123.$ip" >/dev/null 2>&1; then
      warn "  Found device at 192.168.123.$ip"
      found=true
    fi
  done
  if ! $found; then
    err "  No devices found on subnet"
  fi
  return 1
}

# ============================================================
# Step 3: Kill Stale Processes
# ============================================================
step_cleanup() {
  log "Step 3: Cleaning up stale processes"
  
  local killed=false
  
  # Kill old server instances
  local pids
  pids=$(pgrep -f "server.py.*interface" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill -9 2>/dev/null || true
    ok "Killed stale server processes: $pids"
    killed=true
  fi
  
  # Kill old watchdog
  pids=$(pgrep -f "watchdog.sh" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill -9 2>/dev/null || true
    ok "Killed stale watchdog processes: $pids"
    killed=true
  fi
  
  # Kill old camera server
  pids=$(pgrep -f "camera_server.py" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill -9 2>/dev/null || true
    ok "Killed stale camera processes: $pids"
    killed=true
  fi

  if ! $killed; then
    ok "No stale processes found"
  fi
  
  # Wait for ports to free
  sleep 1
  if ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT} "; then
    warn "Port $SERVER_PORT still in use, waiting..."
    sleep 3
  fi
}

# ============================================================
# Step 4: Start Servers
# ============================================================
step_start_servers() {
  log "Step 4: Starting servers"
  
  # Start watchdog (manages both camera + web server)
  setsid bash web/watchdog.sh >> /tmp/watchdog.log 2>&1 &
  disown
  ok "Watchdog launched (PID: $!)"
  
  # Wait for web server to be ready
  log "  Waiting for web server on port $SERVER_PORT..."
  for i in $(seq 1 $MAX_WAIT); do
    if curl -sf -o /dev/null "http://localhost:${SERVER_PORT}/api/state" 2>/dev/null; then
      ok "Web server ready"
      return 0
    fi
    sleep 1
    printf "."
  done
  echo
  err "Web server did not start within ${MAX_WAIT}s"
  err "Check logs: tail -50 /tmp/watchdog.log"
  return 1
}

# ============================================================
# Step 5: Wait for DDS Connection
# ============================================================
step_wait_dds() {
  log "Step 5: Waiting for DDS connection to arm"
  
  for i in $(seq 1 $MAX_WAIT); do
    local state
    state=$(api_get "/api/state" || echo '{}')
    if echo "$state" | python3.12 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('connected') else 1)" 2>/dev/null; then
      ok "DDS connected to D1 arm"
      return 0
    fi
    sleep 1
    printf "."
  done
  echo
  
  err "DDS connection failed after ${MAX_WAIT}s"
  err "Check: grep -i 'dds\|error' /tmp/watchdog.log | tail -10"
  
  # Show diagnostic info
  local stats
  stats=$(api_get "/api/debug/stats" 2>/dev/null || echo "{}")
  warn "DDS rx rate: $(echo "$stats" | python3.12 -c "import json,sys; print(json.load(sys.stdin).get('rates_per_sec',{}).get('dds_receive','?'))" 2>/dev/null || echo '?')"
  return 1
}

# ============================================================
# Step 6: Power On & Enable
# ============================================================
step_power_on() {
  log "Step 6: Power on & enable arm"
  
  # Power on
  local result
  result=$(api_post "/api/command/power-on" || echo '{}')
  sleep 2
  
  local state
  state=$(api_get "/api/state" || echo '{}')
  local power
  power=$(echo "$state" | python3.12 -c "import json,sys; print(json.load(sys.stdin).get('power', False))" 2>/dev/null)
  
  if [ "$power" = "True" ]; then
    ok "Power ON"
  else
    warn "Power command sent but state shows power=False"
    warn "Retrying power-on..."
    for i in 1 2 3; do
      api_post "/api/command/power-on" >/dev/null
      sleep 2
      state=$(api_get "/api/state" || echo '{}')
      power=$(echo "$state" | python3.12 -c "import json,sys; print(json.load(sys.stdin).get('power', False))" 2>/dev/null)
      if [ "$power" = "True" ]; then
        ok "Power ON (attempt $((i+1)))"
        break
      fi
    done
    if [ "$power" != "True" ]; then
      err "Failed to power on after 4 attempts"
      return 1
    fi
  fi
  
  # Enable motors
  result=$(api_post "/api/command/enable" || echo '{}')
  sleep 2
  
  state=$(api_get "/api/state" || echo '{}')
  local enabled
  enabled=$(echo "$state" | python3.12 -c "import json,sys; print(json.load(sys.stdin).get('enabled', False))" 2>/dev/null)
  
  if [ "$enabled" = "True" ]; then
    ok "Motors ENABLED"
  else
    warn "Enable command sent but state shows enabled=False"
    warn "Retrying enable..."
    api_post "/api/command/enable" >/dev/null
    sleep 2
    state=$(api_get "/api/state" || echo '{}')
    enabled=$(echo "$state" | python3.12 -c "import json,sys; print(json.load(sys.stdin).get('enabled', False))" 2>/dev/null)
    if [ "$enabled" = "True" ]; then
      ok "Motors ENABLED (retry)"
    else
      err "Failed to enable motors"
      return 1
    fi
  fi
}

# ============================================================
# Step 7: Joint Test (verify feedback loop)
# ============================================================
step_joint_test() {
  log "Step 7: Joint feedback test"
  
  if $SKIP_TEST; then
    warn "Skipping joint test (--skip-test)"
    return 0
  fi
  
  # Read current joints
  local state
  state=$(api_get "/api/state")
  local joints
  joints=$(echo "$state" | python3.12 -c "import json,sys; j=json.load(sys.stdin)['joints']; print(' '.join(str(round(v,1)) for v in j))")
  ok "Current joints: [$joints]"
  
  # Test: move J4 (wrist pitch) by +5° then back — safest joint for small test
  local j4_current
  j4_current=$(echo "$joints" | awk '{print $5}')
  local j4_target
  j4_target=$(python3.12 -c "print(round($j4_current + 5, 1))")
  
  log "  Testing J4: ${j4_current}° → ${j4_target}° → ${j4_current}°"
  api_post_json "/api/command/set-joint" "{\"id\": 4, \"angle\": $j4_target}" >/dev/null
  sleep 2
  
  # Check if joint moved
  state=$(api_get "/api/state")
  local j4_actual
  j4_actual=$(echo "$state" | python3.12 -c "import json,sys; print(round(json.load(sys.stdin)['joints'][4], 1))")
  local j4_error
  j4_error=$(python3.12 -c "print(abs($j4_actual - $j4_target))")
  
  if python3.12 -c "exit(0 if $j4_error < 3.0 else 1)"; then
    ok "J4 moved to ${j4_actual}° (target: ${j4_target}°, error: ${j4_error}°)"
  else
    warn "J4 at ${j4_actual}° (target: ${j4_target}°, error: ${j4_error}°)"
    warn "Feedback may be delayed or joint didn't reach target"
  fi
  
  # Move back
  api_post_json "/api/command/set-joint" "{\"id\": 4, \"angle\": $j4_current}" >/dev/null
  sleep 2
  ok "J4 returned to ${j4_current}°"
  
  # Verify DDS stats
  local stats
  stats=$(api_get "/api/debug/stats")
  local rx_rate
  rx_rate=$(echo "$stats" | python3.12 -c "import json,sys; print(json.load(sys.stdin)['rates_per_sec'].get('dds_receive', 0))" 2>/dev/null)
  
  if python3.12 -c "exit(0 if float('$rx_rate') > 0 else 1)" 2>/dev/null; then
    ok "DDS feedback active (rx rate: ${rx_rate}/s)"
  else
    warn "DDS rx rate is 0 — feedback may not be flowing"
  fi
}

# ============================================================
# Step 8: Home Position
# ============================================================
step_home() {
  log "Step 8: Moving to home position"
  
  local result
  result=$(api_post "/api/task/home")
  local task_ok
  task_ok=$(echo "$result" | python3.12 -c "import json,sys; print(json.load(sys.stdin).get('ok', False))" 2>/dev/null)
  
  if [ "$task_ok" = "True" ]; then
    ok "Home task completed"
  else
    warn "Home task response: $result"
    warn "Arm may already be at home or task failed"
  fi
  
  sleep 2
  local state
  state=$(api_get "/api/state")
  local joints
  joints=$(echo "$state" | python3.12 -c "import json,sys; j=json.load(sys.stdin)['joints']; print(' '.join(str(round(v,1)) for v in j))")
  ok "Final joints: [$joints]"
}

# ============================================================
# MAIN
# ============================================================
echo
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  th3cl4w — D1 Arm Cold Start${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo

FAILED=false
for step in step_network step_ping_d1 step_cleanup step_start_servers step_wait_dds step_power_on step_joint_test step_home; do
  if ! $step; then
    err "FAILED at: $step"
    FAILED=true
    break
  fi
  echo
done

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
if $FAILED; then
  echo -e "${RED}  ✗ Startup FAILED — see errors above${NC}"
  echo -e "${YELLOW}  Logs: /tmp/watchdog.log${NC}"
  exit 1
else
  echo -e "${GREEN}  ✓ D1 arm is READY${NC}"
  echo -e "${CYAN}  UI: http://localhost:${SERVER_PORT}${NC}"
  echo -e "${CYAN}  Logs: /tmp/watchdog.log${NC}"
fi
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo
