// Headless Chrome with WebGPU, for Windows (this machine) and Linux GPU servers (e.g. a RunPod NVIDIA pod).
// CHROME_PATH overrides the browser binary. On Linux, WebGPU runs on Vulkan, which needs the NVIDIA graphics
// libraries inside the container (NVIDIA_DRIVER_CAPABILITIES=all).
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const LINUX_CANDIDATES = ['/usr/bin/google-chrome-stable', '/usr/bin/google-chrome', '/opt/google/chrome/chrome', '/usr/bin/chromium'];

export function chromePath() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  if (process.platform === 'win32') return 'C:/Program Files/Google/Chrome/Application/chrome.exe';
  return LINUX_CANDIDATES.find(p => fs.existsSync(p)) || 'google-chrome';
}

export function chromeArgs() {
  const common = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-sandbox', '--disable-gpu-watchdog', '--disable-features=GpuWatchdog', '--disable-dev-shm-usage'];
  if (process.platform === 'linux') return [...common, '--enable-features=Vulkan', '--use-angle=vulkan', '--use-vulkan=native', '--disable-vulkan-surface', '--enable-gpu'];
  return [...common, '--enable-gpu'];
}

export function launchChrome(extra = {}) {
  return puppeteer.launch({ executablePath: chromePath(), headless: 'new', protocolTimeout: 0, timeout: 300000, args: chromeArgs(), ...extra });
}
