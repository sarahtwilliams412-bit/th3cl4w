#!/bin/bash
# ============================================================
# One-time network setup for D1 arm on eno1
# ============================================================
# Creates a persistent NetworkManager connection profile so
# 192.168.123.10/24 is assigned to eno1 automatically on boot.
#
# Run once: sudo ./scripts/setup-network.sh
# After this, startup.sh --skip-network works.

set -euo pipefail

INTERFACE="eno1"
LOCAL_IP="192.168.123.10"
CON_NAME="d1-arm"

echo "Setting up persistent network for D1 arm..."
echo "  Interface: $INTERFACE"
echo "  IP: ${LOCAL_IP}/24"
echo "  Connection: $CON_NAME"
echo

# Remove existing connection if present
nmcli con delete "$CON_NAME" 2>/dev/null && echo "Removed old '$CON_NAME' profile" || true

# Create new connection - manual IP, no gateway (isolated subnet)
nmcli con add \
  type ethernet \
  con-name "$CON_NAME" \
  ifname "$INTERFACE" \
  ipv4.addresses "${LOCAL_IP}/24" \
  ipv4.method manual \
  ipv6.method link-local \
  connection.autoconnect yes \
  connection.autoconnect-priority 10

echo
echo "Bringing up connection..."
nmcli con up "$CON_NAME"

echo
echo "Verifying..."
ip addr show "$INTERFACE" | grep inet

echo
echo "✓ Done! $INTERFACE will now get ${LOCAL_IP}/24 on every boot."
echo "  To remove: sudo nmcli con delete $CON_NAME"
