// Headless test harness: loads the app in test mode, steps the sim deterministically and saves screenshots.
// Usage: node tools/shot.mjs '<json>'
//   json: { url: "?test&w=960&h=1080&q=high", warm: 240, shots: [{ name, steps, state, sim, render, prof }] }
import fs from 'node:fs';
import path from 'node:path';
import { createServer } from './server.mjs';
import { execFileSync } from 'node:child_process';
import { launchChrome } from './chrome.mjs';
const cfg = JSON.parse(process.argv[2] || '{}');
const outDir = path.resolve(cfg.out || 'out/shots');
fs.mkdirSync(outDir, { recursive: true });
const port = 8790 + Math.floor(Math.random() * 200);
const server = await createServer(port);
const browser = await launchChrome({ protocolTimeout: 600000 });
const page = await browser.newPage();
const logs = [];
page.on('console', m => { const t = `[${m.type()}] ${m.text()}`; logs.push(t); if (m.type() === 'error' || m.type() === 'warning' || cfg.verbose) console.log(t); });
page.on('pageerror', e => console.log('[pageerror]', e.message));
const w = cfg.w || 960, h = cfg.h || 1080;
await page.setViewport({ width: w, height: h });
await page.goto(`http://localhost:${port}/index.html${cfg.url || `?test&w=${w}&h=${h}`}`);
await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
const err = await page.evaluate('window.__error');
if (err) { console.log('INIT ERROR', err); await browser.close(); server.close(); process.exit(1); }

const t0 = Date.now();
if (cfg.warm) {
  if (cfg.warmSim) await page.evaluate(p => FIRE.setSim(p), cfg.warmSim);
  // warm up in chunks to avoid long GPU submissions
  for (let i = 0; i < cfg.warm; i += 30) await page.evaluate(n => FIRE.step(n), Math.min(30, cfg.warm - i));
}
console.log(`warmup ${cfg.warm || 0} steps: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
for (const s of cfg.shots || [{ name: 'shot' }]) {
  if (s.state) await page.evaluate(p => FIRE.set(p), s.state);
  if (s.sim) await page.evaluate(p => FIRE.setSim(p), s.sim);
  if (s.render) await page.evaluate(p => FIRE.setRender(p), s.render);
  if (s.reset) { await page.evaluate(() => FIRE.reset()); }
  if (s.event) await page.evaluate(e => FIRE.event(e), s.event);
  if (s.warm) { const t2 = Date.now(); for (let i = 0; i < s.warm; i += 30) await page.evaluate(n => FIRE.step(n), Math.min(30, s.warm - i)); console.log(`  warm ${s.warm}: ${((Date.now() - t2) / 1000).toFixed(1)}s`); }
  if (s.steps) for (let i = 0; i < s.steps; i += 30) await page.evaluate(n => FIRE.step(n), Math.min(30, s.steps - i));
  const t1 = Date.now();
  if (s.accum || cfg.accum) await page.evaluate(n => FIRE.drawAccum(n), s.accum || cfg.accum);
  else { await page.evaluate(() => FIRE.draw()); await page.evaluate(() => FIRE.draw()); }
  const dt = Date.now() - t1;
  const el = await page.$('#c');
  const file = path.join(outDir, `${s.name || 'shot'}.png`);
  await el.screenshot({ path: file });
  let prof = '';
  if (s.prof) {
    prof = JSON.stringify(await page.evaluate(() => FIRE.profileFrame(6)));
  }
  console.log(`saved ${file} (2 draws ${dt}ms) ${prof}`);
  if (s.metrics || cfg.metrics) {
    try {
      const m = JSON.parse(execFileSync('.venv/Scripts/python.exe', ['tools/fire_metrics.py', file], { encoding: 'utf8' }).split(String.fromCharCode(10))[0]);
      console.log(`  metrics: flame ${m.flame_frac} Rclip ${m.R_clip} RGclip ${m.RG_clip} white ${m.all_clip} dim/bright ${m.dim_bright_ratio} ramp ${m.ramp_err} medianL ${m.image_median_luma} nearblack ${m.near_black}`);
    } catch (e) { console.log('  metrics failed', e.message); }
  }
  if (s.profile) { const pr = await page.evaluate(() => FIRE.profile()); console.log('Q =', pr.Q_kW, 'kW'); for (const r of pr.rows) console.log('  ' + JSON.stringify(r)); }
  if (s.nan) console.log('nanScan', JSON.stringify(await page.evaluate(() => FIRE.nanScan())));
  if (s.stats) for (const f of s.stats) console.log(f, JSON.stringify(await page.evaluate(n => FIRE.stats(n), f)));
}
await browser.close();
server.close();
