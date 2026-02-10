#!/bin/bash
# Install th3cl4w systemd user services
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_DIR="$HOME/.config/systemd/user"

echo "Installing th3cl4w systemd user services..."

mkdir -p "$SYSTEMD_DIR"

# Symlink all service/target files
for f in "$PROJECT_DIR"/systemd/*.service "$PROJECT_DIR"/systemd/*.target; do
    name="$(basename "$f")"
    ln -sf "$f" "$SYSTEMD_DIR/$name"
    echo "  Linked $name"
done

# Reload systemd
systemctl --user daemon-reload
echo "  Reloaded systemd user daemon"

# Enable lingering so services survive logout
loginctl enable-linger "$USER" 2>/dev/null || echo "  Warning: could not enable linger (may need admin)"

# Enable and start
systemctl --user enable th3cl4w.target
systemctl --user start th3cl4w.target
echo "  Started th3cl4w.target (all services)"

echo ""
echo "Done! Check status with: scripts/th3cl4w-ctl.sh status"
