#!/usr/bin/env bash
# One-time setup of a Linux NVIDIA GPU pod (RunPod, Ubuntu 22.04/24.04 template) for rendering the fire video.
# The pod needs the env var NVIDIA_DRIVER_CAPABILITIES=all: headless Chrome runs WebGPU on Vulkan, and the NVIDIA
# Vulkan driver is only mounted into the container with the 'graphics' capability.
# Usage (on the pod, inside the project folder): bash tools/runpod/setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget curl ca-certificates gnupg ffmpeg libvulkan1 vulkan-tools fonts-liberation >/dev/null
if ! command -v node >/dev/null || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 20 ]; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
  apt-get install -y -qq nodejs >/dev/null
fi
if [ ! -x /usr/bin/google-chrome-stable ]; then
  wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  apt-get install -y -qq /tmp/chrome.deb >/dev/null
fi
npm install --omit=dev --no-audit --no-fund >/dev/null
echo "node $(node -v), $(google-chrome-stable --version), $(ffmpeg -version | head -1)"
if vulkaninfo --summary 2>/dev/null | grep -q "deviceName"; then
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" | head -4
else
  echo "WARNING: no Vulkan GPU visible. Restart the pod with NVIDIA_DRIVER_CAPABILITIES=all, or WebGPU falls back to the CPU."
fi
# one-frame WebGPU smoke test (prints the GPU Chrome picked)
node tools/render_video.mjs --w 270 --h 480 --q low --frames 1 --aa 1 --stills 0 --out out/video/smoke.mp4
