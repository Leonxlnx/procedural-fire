// node run.mjs <headless: new|false|shell> <mode> [extra flags...]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import { startServer } from './server.mjs';

const [, , hl = 'new', mode = 'probe', ...extra] = process.argv;
const server = await startServer(8765);
const base = [
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',     // un-quantized timestamp queries
  '--ignore-gpu-blocklist',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows',
  '--no-first-run', '--no-default-browser-check',
];
const args = process.env.NOBASE === '1' ? extra : [...base, ...extra];
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: hl === 'false' ? false : hl === 'shell' ? 'shell' : true,
  args,
  defaultViewport: { width: 800, height: 600 },
  userDataDir: `${process.cwd()}/profile-${hl}`,
  protocolTimeout: 900000,
  pipe: process.env.PIPE === '1',          // CDP over stdio pipes instead of a localhost WebSocket
});
const page = await browser.newPage();
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.error('[page]', m.type(), m.text()); });
page.on('pageerror', e => console.error('[pageerror]', e.message));
let exposedBytes = 0;
await page.exposeFunction('__sendFrame', b64 => { exposedBytes += Buffer.from(b64, 'base64').length; return true; });
// Downloads as a transfer channel: Chrome's browser process writes Blobs straight to disk.
const dlDir = `${process.cwd()}\\dl`;
fs.mkdirSync(dlDir, { recursive: true });
const cdp = await browser.target().createCDPSession();
await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dlDir, eventsEnabled: true });
let dlCompleted = 0; const dlWaiters = [];
cdp.on('Browser.downloadProgress', e => { if (e.state === 'completed') { dlCompleted++; dlWaiters.filter(w => dlCompleted >= w.n).forEach(w => w.r()); } });
await page.exposeFunction('__waitDownloads', n => new Promise(r => { if (dlCompleted >= n) r(true); else dlWaiters.push({ n, r: () => r(true) }); }));
await page.exposeFunction('__resetDownloads', () => { dlCompleted = 0; return true; });
const outdir = process.cwd().replace(/\\/g, '/');
const t0 = Date.now();
await page.goto(`http://127.0.0.1:8765/bench.html?mode=${mode}&plan=${process.env.PLAN || ''}&sizes=${process.env.SIZES || ''}&outdir=${encodeURIComponent(outdir)}`);
await page.waitForFunction('window.__out && window.__out.done', { timeout: 900000, polling: 500 });
const res = await page.evaluate(() => window.__out);
res.headless = hl; res.flags = args; res.seconds = (Date.now() - t0) / 1000; res.exposedBytes = exposedBytes;
const file = `result-${hl}-${mode}${extra.length ? '-' + extra.join('').replace(/[^a-z0-9]+/gi, '_').slice(0, 40) : ''}.json`;
fs.writeFileSync(file, JSON.stringify(res, null, 1));
console.log(JSON.stringify({ file, adapterInfo: res.adapterInfo, errors: res.errors, probes: res.probes, seconds: res.seconds }, null, 1));
await browser.close();
server.close();
