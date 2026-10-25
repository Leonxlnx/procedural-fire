// Smoke test of the interactive viewer in headless Chrome: loads the page, clicks "light it",
// presses a few controls and saves screenshots + console errors. Usage: node tools/ui_test.mjs [q=low]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import { createServer } from './server.mjs';

const q = process.argv[2] || 'low';
const port = 8600 + Math.floor(Math.random() * 90);
const server = await createServer(port);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', protocolTimeout: 120000,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--no-sandbox', '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage();
const errors = [];
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`[${m.type()}] ${m.text()}`); });
page.on('pageerror', e => errors.push('[pageerror] ' + e.message));
await page.setViewport({ width: 1280, height: 720 });
await page.goto(`http://localhost:${port}/index.html?q=${q}`);
fs.mkdirSync('out/ui', { recursive: true });
const wait = (ms) => new Promise(r => setTimeout(r, ms));
await wait(2500);
await page.screenshot({ path: 'out/ui/1_start.png' });
await page.click('#start');
await wait(5000);
await page.screenshot({ path: 'out/ui/2_lit.png' });
await page.click('[data-act="stoke"]');
await wait(700);
await page.screenshot({ path: 'out/ui/3_stoke.png' });
const hud = await page.$eval('#hud', e => e.textContent).catch(() => '');
console.log('hud:', hud);
console.log(errors.filter(e => !e.includes('404')).slice(0, 20).join('\n') || 'no errors');
await browser.close();
server.close();
