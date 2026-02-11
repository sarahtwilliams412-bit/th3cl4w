#!/bin/bash
# th3cl4w Service Control Script
# Usage:
#   ./th3cl4w-ctl.sh start all          # Start all services
#   ./th3cl4w-ctl.sh start control_plane # Start specific service
#   ./th3cl4w-ctl.sh stop all            # Stop all services
#   ./th3cl4w-ctl.sh status              # Show service status
#   ./th3cl4w-ctl.sh logs control_plane  # View service logs
#   ./th3cl4w-ctl.sh restart all         # Restart all services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Service definitions: name -> module path -> port
declare -A SERVICES=(
    [gateway]="services.gateway.server:8080"
    [control_plane]="services.control_plane.server:8090"
    [camera]="services.camera.server:8081"
    [world_model]="services.world_model.server:8082"
    [local_model]="services.local_model.server:8083"
    [object_id]="services.object_id.server:8084"
    [kinematics]="services.kinematics_app.server:8085"
    [mapping]="services.mapping.server:8086"
    [positioning]="services.positioning.server:8087"
    [tasker]="services.tasker.server:8088"
    [simulation]="services.simulation.server:8089"
    [telemetry]="services.telemetry.server:8091"
)

# Startup order (dependencies first)
STARTUP_ORDER=(
    control_plane
    camera
    world_model
    local_model
    object_id
    kinematics
    mapping
    positioning
    tasker
    simulation
    telemetry
    gateway
)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

get_module() {
    local svc="$1"
    echo "${SERVICES[$svc]}" | cut -d: -f1
}

get_port() {
    local svc="$1"
    echo "${SERVICES[$svc]}" | cut -d: -f2
}

is_running() {
    local port
    port=$(get_port "$1")
    ss -tlnp 2>/dev/null | grep -q ":${port} " && return 0
    return 1
}

start_service() {
    local svc="$1"
    local module port extra_args

    if ! [[ ${SERVICES[$svc]+_} ]]; then
        log_error "Unknown service: $svc"
        return 1
    fi

    module=$(get_module "$svc")
    port=$(get_port "$svc")

    if is_running "$svc"; then
        log_warn "$svc is already running on port $port"
        return 0
    fi

    extra_args=""
    if [[ "$svc" == "control_plane" && "${SIMULATE:-false}" == "true" ]]; then
        extra_args="--simulate"
    fi

    log_info "Starting $svc on port $port..."
    python3 -m "$module" $extra_args &
    disown

    # Wait briefly for startup
    for i in {1..10}; do
        sleep 0.5
        if is_running "$svc"; then
            log_info "$svc started successfully on port $port"
            return 0
        fi
    done

    log_warn "$svc may still be starting on port $port"
}

stop_service() {
    local svc="$1"
    local port
    port=$(get_port "$svc")

    local pids
    pids=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K\d+' | sort -u)

    if [[ -z "$pids" ]]; then
        log_info "$svc is not running"
        return 0
    fi

    log_info "Stopping $svc (port $port, PIDs: $pids)..."
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 1

    # Force kill if still running
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    log_info "$svc stopped"
}

show_status() {
    echo -e "\n${BLUE}=== th3cl4w Service Status ===${NC}\n"
    printf "%-18s %-8s %-10s\n" "SERVICE" "PORT" "STATUS"
    printf "%-18s %-8s %-10s\n" "-------" "----" "------"

    for svc in "${STARTUP_ORDER[@]}"; do
        local port
        port=$(get_port "$svc")
        if is_running "$svc"; then
            printf "%-18s %-8s ${GREEN}%-10s${NC}\n" "$svc" "$port" "RUNNING"
        else
            printf "%-18s %-8s ${RED}%-10s${NC}\n" "$svc" "$port" "STOPPED"
        fi
    done

    # Check Redis
    if redis-cli ping &>/dev/null; then
        printf "%-18s %-8s ${GREEN}%-10s${NC}\n" "redis" "6379" "RUNNING"
    else
        printf "%-18s %-8s ${RED}%-10s${NC}\n" "redis" "6379" "STOPPED"
    fi
    echo ""
}

show_logs() {
    local svc="$1"
    local log_file="logs/${svc}.log"
    if [[ -f "$log_file" ]]; then
        tail -f "$log_file"
    else
        log_error "No log file found at $log_file"
    fi
}

case "${1:-help}" in
    start)
        target="${2:-all}"
        if [[ "$target" == "all" ]]; then
            # Start Redis first
            if ! redis-cli ping &>/dev/null; then
                log_info "Starting Redis..."
                redis-server --daemonize yes
                sleep 1
            fi
            for svc in "${STARTUP_ORDER[@]}"; do
                start_service "$svc"
            done
        else
            start_service "$target"
        fi
        ;;
    stop)
        target="${2:-all}"
        if [[ "$target" == "all" ]]; then
            for svc in "${STARTUP_ORDER[@]}"; do
                stop_service "$svc"
            done
        else
            stop_service "$target"
        fi
        ;;
    restart)
        target="${2:-all}"
        if [[ "$target" == "all" ]]; then
            for svc in "${STARTUP_ORDER[@]}"; do
                stop_service "$svc"
            done
            sleep 1
            for svc in "${STARTUP_ORDER[@]}"; do
                start_service "$svc"
            done
        else
            stop_service "$target"
            sleep 1
            start_service "$target"
        fi
        ;;
    status)
        show_status
        ;;
    logs)
        if [[ -z "${2:-}" ]]; then
            log_error "Usage: $0 logs <service_name>"
            exit 1
        fi
        show_logs "$2"
        ;;
    help|*)
        echo "th3cl4w Service Control"
        echo ""
        echo "Usage: $0 <command> [service|all]"
        echo ""
        echo "Commands:"
        echo "  start <service|all>   Start service(s)"
        echo "  stop <service|all>    Stop service(s)"
        echo "  restart <service|all> Restart service(s)"
        echo "  status                Show status of all services"
        echo "  logs <service>        Tail service log file"
        echo ""
        echo "Services:"
        for svc in "${STARTUP_ORDER[@]}"; do
            printf "  %-18s port %s\n" "$svc" "$(get_port "$svc")"
        done
        echo ""
        echo "Environment:"
        echo "  SIMULATE=true    Run control_plane in simulation mode"
        ;;
esac
