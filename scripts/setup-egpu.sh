#!/usr/bin/env bash
# eGPU Compute Setup — XFX RX 580 via Thunderbolt 3
# Installs OpenCL, Vulkan compute, and monitoring tools
set -euo pipefail

echo "🦾 Setting up eGPU compute stack for RX 580..."

# 1. Install Mesa OpenCL (Clover) — works with Polaris/GFX8
echo "[1/5] Installing Mesa OpenCL runtime..."
sudo apt-get install -y mesa-opencl-icd ocl-icd-opencl-dev clinfo

# 2. Install Vulkan compute stack
echo "[2/5] Installing Vulkan compute stack..."
sudo apt-get install -y mesa-vulkan-drivers vulkan-tools libvulkan-dev

# 3. Install GPU monitoring tools
echo "[3/5] Installing GPU monitoring tools..."
sudo apt-get install -y radeontop

# 4. Install Python compute libraries
echo "[4/5] Installing Python GPU compute libraries..."
pip3 install --user pyopencl 2>/dev/null || echo "pyopencl install failed (optional)"

# 5. Add user to render + video groups (needed for GPU access)
echo "[5/5] Setting up permissions..."
sudo usermod -aG render,video "$USER"

echo ""
echo "=== Verifying Setup ==="

# Verify OpenCL
echo "--- OpenCL ---"
clinfo --list 2>/dev/null || echo "clinfo not found"

# Verify Vulkan
echo "--- Vulkan ---"
vulkaninfo --summary 2>/dev/null | grep -A2 'GPU' || echo "vulkaninfo not found"

# Verify GPU access
echo "--- GPU VRAM ---"
echo "$(( $(cat /sys/class/drm/card0/device/mem_info_vram_total) / 1024 / 1024 )) MB"

echo ""
echo "✅ eGPU compute setup complete!"
echo "   GPU: XFX RX 580 (8GB) via Thunderbolt 3 @ 40 Gb/s"
echo "   DRI: /dev/dri/renderD129"
echo "   Use DRI_PRIME=1 to route rendering to eGPU"
echo "   Use MESA_LOADER_DRIVER_OVERRIDE=radeonsi for explicit Mesa driver"
echo ""
echo "⚠️  Log out and back in for group changes to take effect"
