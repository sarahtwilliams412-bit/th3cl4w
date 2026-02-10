#!/bin/bash
# th3cl4w service management
set -euo pipefail

SERVICES=(th3cl4w-main th3cl4w-camera th3cl4w-location th3cl4w-map th3cl4w-ascii)
TARGET=th3cl4w.target
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_DIR="$HOME/.config/systemd/user"

usage() {
    echo "Usage: $0 {start|stop|restart|status|logs [service]|install|uninstall}"
    echo ""
    echo "Services: ${SERVICES[*]}"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 logs th3cl4w-main"
    echo "  $0 logs th3cl4w-camera -f    # follow logs"
    exit 1
}

case "${1:-}" in
    start)
        systemctl --user start "$TARGET"
        echo "Started all th3cl4w services"
        ;;
    stop)
        systemctl --user stop "$TARGET"
        for svc in "${SERVICES[@]}"; do
            systemctl --user stop "$svc.service" 2>/dev/null || true
        done
        echo "Stopped all th3cl4w services"
        ;;
    restart)
        for svc in "${SERVICES[@]}"; do
            systemctl --user restart "$svc.service" 2>/dev/null || true
        done
        echo "Restarted all th3cl4w services"
        ;;
    status)
        for svc in "${SERVICES[@]}"; do
            state=$(systemctl --user is-active "$svc.service" 2>/dev/null || echo "unknown")
            printf "  %-25s %s\n" "$svc" "$state"
        done
        ;;
    logs)
        svc="${2:-th3cl4w-main}"
        shift 2 2>/dev/null || shift 1
        # Pass remaining args (e.g. -f, -n 100) to journalctl
        journalctl --user -u "$svc.service" "${@:---lines=50}"
        ;;
    install)
        exec "$SCRIPT_DIR/install-services.sh"
        ;;
    uninstall)
        echo "Stopping and disabling th3cl4w services..."
        systemctl --user stop "$TARGET" 2>/dev/null || true
        for svc in "${SERVICES[@]}"; do
            systemctl --user stop "$svc.service" 2>/dev/null || true
            systemctl --user disable "$svc.service" 2>/dev/null || true
        done
        systemctl --user disable "$TARGET" 2>/dev/null || true
        for f in "$SYSTEMD_DIR"/th3cl4w*.service "$SYSTEMD_DIR"/th3cl4w*.target; do
            [ -e "$f" ] && rm -f "$f" && echo "  Removed $(basename "$f")"
        done
        systemctl --user daemon-reload
        echo "Uninstalled."
        ;;
    *)
        usage
        ;;
esac
