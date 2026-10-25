// A/B test of capture settings at chosen moments of the timeline: fast-forwards the sim (no rendering)
// and renders consecutive frames with different shutter / aperture / render params.
// Usage: node tools/lens_test.mjs --config tests.json
// config: { q, w, h, aa, simDt, out, preroll, sim: {...}, render: {...},
//           tests: [{ frame, label, shutter, apScale, dof, render: {...} }] }
// Tests are rendered in frame order; a test whose frame was already passed renders the next free frame.
import { spawn } from 'node:child_process';
import { launchChrome } from './chrome.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { createServer } from './server.mjs';

const argv = process.argv.slice(2);
const C = JSON.parse(fs.readFileSync(argv[argv.indexOf('--config') + 1], 'utf8'));
const W = C.w || 1080, H = C.h || 1920;
const OUT = path.resolve(C.out || 'out/lens');
fs.mkdirSync(OUT, { recursive: true });

let pending = null;
const port = 8700 + Math.floor(Math.random() * 80);
const server = await createServer(port, (req, res) => {
  if (req.method !== 'POST' || !req.url.startsWith('/frame')) return false;
  const chunks = [];
  req.on('data', c => chunks.push(c));
  req.on('end', () => {
    const buf = Buffer.concat(chunks);
    const file = path.join(OUT, `${pending}.png`);
    const p = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${H}`, '-i', '-', '-frames:v', '1', file]);
    p.on('close', () => res.writeHead(200).end('ok'));
    p.stdin.end(buf);
  });
  return true;
});

const browser = await launchChrome();
try {
  const page = await browser.newPage();
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', e => console.log('[pageerror]', e.message));
  await page.setViewport({ width: 400, height: 400 });
  await page.goto(`http://localhost:${port}/index.html?test&w=${W}&h=${H}&q=${C.q || 'hero'}`);
  await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
  const err = await page.evaluate('window.__error');
  if (err) throw new Error(err);
  const base = C.render || {};
  await page.evaluate(p => FIRE.setRender(p), base);
  if (C.sim) await page.evaluate(p => FIRE.setSim(p), C.sim);
  const t0 = Date.now();
  await page.evaluate(o => FIRE.captureBegin(o), { w: W, h: H, simDt: C.simDt || 1 / 240, aa: C.aa || 2, shutter: 0.5, dof: true, ...(C.preroll ? { preroll: C.preroll } : {}) });
  console.log(`preroll ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  const tests = [...C.tests].sort((a, b) => a.frame - b.frame);
  const baseCap = { shutter: 0.5, apScale: 1, dof: true, aa: C.aa || 2 };
  let i = 0;
  for (const t of tests) {
    while (i < t.frame) await page.evaluate(i => FIRE.captureSkip(i), i++);
    await page.evaluate(p => FIRE.captureSet(p), { ...baseCap, ...Object.fromEntries(['shutter', 'apScale', 'dof', 'aa'].filter(k => k in t).map(k => [k, t[k]])) });
    await page.evaluate(p => FIRE.setRender(p), { ...base, ...(t.render || {}) });
    pending = `${String(i).padStart(4, '0')}_${t.label}`;
    const tf = Date.now();
    const st = await page.evaluate(i => FIRE.captureFrame(i), i++);
    console.log(`${pending} status ${st} ${((Date.now() - tf) / 1000).toFixed(1)}s`);
  }
  if (C.profile) {
    const pr = await page.evaluate(() => FIRE.profile());
    console.log('Q =', pr.Q_kW, 'kW');
    for (const r of pr.rows || []) console.log('  ' + JSON.stringify(r));
  }
  console.log(`done in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
} catch (e) {
  console.log('ERROR', e?.stack || e);
  process.exitCode = 1;
} finally {
  await browser.close().catch(() => {});
  server.close();
}
