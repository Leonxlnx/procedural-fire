#!/usr/bin/env node
// worklet_gc_trace.mjs — records a Chrome trace while audio-test.html plays (headless, autoplay allowed) and reports
// garbage collections and process() timing on the AudioWorklet thread.
//   node tools/audio_analysis/worklet_gc_trace.mjs [--secs 8] [--stress] [--out trace.json]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import { createServer } from '../server.mjs';

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const args = process.argv.slice(2);
const val = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const secs = +val('secs', 8);
let port = 8787, server = null;
for (const p of [8787, 8797, 8798]) {
  try { const r = await fetch(`http://localhost:${p}/audio-test.html`); if (r.ok && (await r.text()).includes('Procedural campfire audio')) { port = p; break; } } catch { /* none */ }
  try { server = await createServer(p); port = p; break; } catch { /* busy */ }
}
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });
const res = {};
try {
  const page = await browser.newPage();
  await page.goto(`http://localhost:${port}/${val('page', 'audio-test.html')}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__audioTest && window.__audioTest.ready, { timeout: 20000 });
  res.chrome = await browser.version();
  await page.click('#start');
  await new Promise((r) => setTimeout(r, 1000 * +val('warmup', 2.5))); // JIT tiering settles within ~10-20 s
  const traceFile = val('out', null) || 'out/audio/review/worklet_trace.json';
  await page.tracing.start({ path: traceFile, categories: ['disabled-by-default-v8.gc', 'v8', 'devtools.timeline', 'webaudio', 'audio', '__metadata'] });
  if (args.includes('--stress')) await page.evaluate(() => { const fa = window.__audioTest.fa; fa.setParam('popRate', 3); setInterval(() => { for (const type of ['pop', 'crackle-burst', 'log-shift']) fa.trigger({ type, strength: 1, pan: Math.random() * 2 - 1 }); }, 700); });
  await new Promise((r) => setTimeout(r, secs * 1000));
  await page.tracing.stop();
  const tr = JSON.parse(fs.readFileSync(traceFile, 'utf8')), ev = tr.traceEvents || tr;
  const names = new Map(); for (const e of ev) if (e.ph === 'M' && e.name === 'thread_name') names.set(e.pid + ':' + e.tid, e.args.name);
  const byThread = new Map();
  for (const e of ev) {
    if (e.ph !== 'X' && e.ph !== 'B') continue;
    const tn = names.get(e.pid + ':' + e.tid) || '?';
    if (!/audio|worklet/i.test(tn)) continue;
    const k = tn + ' | ' + e.name, s = byThread.get(k) || { n: 0, totalMs: 0, maxMs: 0 };
    s.n++; const d = (e.dur || 0) / 1000; s.totalMs += d; s.maxMs = Math.max(s.maxMs, d); byThread.set(k, s);
  }
  res.threads = [...new Set(names.values())].filter((n) => /audio|worklet/i.test(n));
  res.secs = secs;
  res.events = Object.fromEntries([...byThread.entries()].sort((a, b) => b[1].totalMs - a[1].totalMs).slice(0, 25).map(([k, s]) => [k, { n: s.n, perSec: +(s.n / secs).toFixed(1), totalMs: +s.totalMs.toFixed(1), maxMs: +s.maxMs.toFixed(3) }]));
} finally { await browser.close(); if (server) server.close(); }
console.log(JSON.stringify(res, null, 1));
