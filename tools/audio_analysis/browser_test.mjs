// browser_test.mjs — headless Chrome test of the live audio engine (puppeteer-core).
// usage: node tools/audio_analysis/browser_test.mjs [--port 8787] [--headful]
// Checks: worklet loads without errors, real-time output level (AnalyserNode), scripted + random events and their
// audible-time delivery, limiter ceiling under stress, user-gesture handling with the default autoplay policy,
// OfflineAudioContext render, and bit-exact parity of the worklet DSP with the Node render.
import puppeteer from 'puppeteer-core';
import crypto from 'node:crypto';
import { createServer } from '../server.mjs';
import { FireAudioEngine } from '../../src/audio/fire-dsp.js';

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const args = process.argv.slice(2);
let port = +(args[args.indexOf('--port') + 1] || 0) || 8787;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const result = { ok: true, checks: {}, consoleErrors: [], pageErrors: [] };
const check = (name, pass, info) => { result.checks[name] = { pass: !!pass, ...info }; if (!pass) result.ok = false; };

async function ensureServer() {
  for (const p of [port, 8797, 8798]) {
    try { const r = await fetch(`http://localhost:${p}/audio-test.html`); if (r.ok && (await r.text()).includes('Procedural campfire audio')) { port = p; return null; } } catch { /* not running */ }
    try { const s = await createServer(p); port = p; return s; } catch { /* port busy with something else */ }
  }
  throw new Error('no port for the static server');
}

function watch(page, tag) {
  page.on('console', (m) => { if (m.type() === 'error') result.consoleErrors.push(`[${tag}] ${m.text()}`); });
  page.on('pageerror', (e) => result.pageErrors.push(`[${tag}] ${e.message}`));
  page.on('requestfailed', (r) => result.consoleErrors.push(`[${tag}] request failed ${r.url()}`));
}

const server = await ensureServer();
const url = `http://localhost:${port}/audio-test.html`;
const browser = await puppeteer.launch({ executablePath: CHROME, headless: !args.includes('--headful'), args: ['--autoplay-policy=no-user-gesture-required'] });
try {
  // ------------------------------------------------------------ 1. real-time run
  const page = await browser.newPage(); watch(page, 'rt');
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__audioTest && window.__audioTest.ready, { timeout: 20000 });
  await page.click('#start');
  await sleep(2500);
  let m = await page.evaluate(() => window.__audioTest.meter());
  check('realtime_running', m.state === 'running', { state: m.state });
  check('realtime_level', m.rmsDb > -50 && m.rmsDb < -6, { analyserRmsDb: +m.rmsDb.toFixed(1), analyserPeakDb: +m.peakDb.toFixed(1), master: m.master });

  // scripted events via the page buttons
  for (const ev of ['ignite', 'pop', 'crackle-burst', 'log-shift', 'whoosh', 'gust']) { await page.click(`[data-ev="${ev}"]`); await sleep(350); }
  await sleep(1500);
  const scripted = await page.evaluate(() => window.__audioTest.events.filter((e) => e.source === 'scripted').map((e) => e.type));
  check('scripted_events_roundtrip', ['ignite', 'pop', 'crackle-burst', 'log-shift', 'whoosh', 'gust'].every((t) => scripted.includes(t)), { received: scripted });

  // random pops: default subscription delivers random pops at their audible time
  const rnd = await page.evaluate(async () => {
    const T = window.__audioTest, fa = T.fa, got = [];
    const off = fa.onEvent((ev) => got.push({ type: ev.type, source: ev.source, lateMs: performance.now() - ev.time, contextTime: ev.contextTime }));
    fa.setParam('popRate', 4);
    await new Promise((r) => setTimeout(r, 4000));
    fa.setParam('popRate', 0.12); off();
    return got;
  });
  const lates = rnd.map((e) => e.lateMs);
  check('random_pop_events', rnd.length >= 3 && rnd.every((e) => e.type === 'pop' && e.source === 'random') && Math.min(...lates) > -8, { count: rnd.length, deliveryVsAudibleMs: { min: +Math.min(...lates).toFixed(1), max: +Math.max(...lates).toFixed(1) } });

  // stress: hot fire, camera inside the near field, volume 1, burst of events -> limiter must hold -1 dBFS
  const stress = await page.evaluate(async () => {
    const T = window.__audioTest, fa = T.fa;
    document.getElementById('intensity').value = 1.5; document.getElementById('camDist').value = 0.3; document.getElementById('hrr').value = 3;
    fa.setVolume(1);
    let peak = 0; const buf = new Float32Array(T.analyser.fftSize);
    const t0 = performance.now();
    for (const type of ['ignite', 'log-shift', 'pop', 'pop', 'gust', 'whoosh', 'crackle-burst', 'pop']) fa.trigger({ type, strength: 1, pan: Math.random() * 2 - 1 });
    let maxMasterPeak = -999;
    while (performance.now() - t0 < 3000) {
      await new Promise((r) => setTimeout(r, 20));
      T.analyser.getFloatTimeDomainData(buf); for (const v of buf) peak = Math.max(peak, Math.abs(v));
      maxMasterPeak = Math.max(maxMasterPeak, fa.meter.peakDb);
    }
    document.getElementById('intensity').value = 1; document.getElementById('camDist').value = 2.2; document.getElementById('hrr').value = 1; fa.setVolume(0.8);
    return { analyserPeakDb: 20 * Math.log10(peak), masterPeakDb: maxMasterPeak, grDb: fa.meter.grDb };
  });
  check('limiter_ceiling', stress.analyserPeakDb <= -0.99 && stress.masterPeakDb <= -0.99, stress);
  await page.click('#start'); // stop
  await sleep(500);
  check('stop', (await page.evaluate(() => window.__audioTest.fa.state)) === 'suspended', {});

  // ------------------------------------------------------------ 2. OfflineAudioContext: full graph + DSP parity with Node
  const off = await page.evaluate(async () => {
    const { FireAudio } = await import('/src/audio/engine.js');
    const sr = 48000, N = sr * 3;
    // a) full graph (worklet + convolver + master)
    const ctx = new OfflineAudioContext({ numberOfChannels: 2, length: N, sampleRate: sr });
    const fa = await FireAudio.create(ctx, { seed: 7, controls: { intensity: 1, camDist: 2 } });
    const evs = []; fa.onEvent((e) => evs.push(e), { all: true });
    await fa.start();
    fa.trigger({ type: 'pop', strength: 1, pan: 0.3, time: 1.0 });
    await new Promise((r) => setTimeout(r, 50));
    const buf = await ctx.startRendering();
    await new Promise((r) => setTimeout(r, 100));
    let ss = 0, pk = 0, bad = 0;
    for (let c = 0; c < 2; c++) for (const v of buf.getChannelData(c)) { if (!Number.isFinite(v)) bad++; ss += v * v; pk = Math.max(pk, Math.abs(v)); }
    // b) worklet dry output only, for bit-exact comparison with the Node engine
    const ctx2 = new OfflineAudioContext({ numberOfChannels: 2, length: sr, sampleRate: sr });
    await ctx2.audioWorklet.addModule('/src/audio/worklet.js');
    const node = new AudioWorkletNode(ctx2, 'fire-audio', { numberOfInputs: 0, numberOfOutputs: 2, outputChannelCount: [2, 2], processorOptions: { seed: 99, controls: { intensity: 1.1, hrr: 1, camDist: 1.7, camAz: 0.4 }, running: true } });
    node.connect(ctx2.destination, 0);
    const b2 = await ctx2.startRendering();
    const L = b2.getChannelData(0), R = b2.getChannelData(1);
    // same DSP on the page's main thread (same V8): must be bit-identical to the worklet (128-frame quanta vs 333-frame blocks)
    const { FireAudioEngine } = await import('/src/audio/fire-dsp.js');
    const e = new FireAudioEngine({ sampleRate: sr, seed: 99 }); e.setControls({ intensity: 1.1, hrr: 1, camDist: 1.7, camAz: 0.4 });
    const mL = new Float32Array(sr), mR = new Float32Array(sr), x0 = new Float32Array(sr), x1 = new Float32Array(sr);
    for (let i = 0; i < sr; i += 333) { const n = Math.min(333, sr - i); e.render(mL.subarray(i, i + n), mR.subarray(i, i + n), x0.subarray(i, i + n), x1.subarray(i, i + n), n); }
    let same = true; for (let i = 0; i < sr; i++) if (L[i] !== mL[i] || R[i] !== mR[i]) { same = false; break; }
    // c) full graph at 44.1 kHz (device rate of many laptops), scripted pop at 0.5 s
    const c44 = new OfflineAudioContext({ numberOfChannels: 2, length: 44100 * 2, sampleRate: 44100 });
    const f44 = await FireAudio.create(c44, { seed: 3, controls: { intensity: 1.2, camDist: 1.5 } });
    const ev44 = []; f44.onEvent((e) => ev44.push(e), { all: true });
    await f44.start(); f44.trigger({ type: 'pop', strength: 1, pan: -0.3, time: 0.5 });
    await new Promise((r) => setTimeout(r, 50));
    const b44 = await c44.startRendering(); await new Promise((r) => setTimeout(r, 100));
    let s44 = 0, p44 = 0, bad44 = 0;
    for (let c = 0; c < 2; c++) for (const v of b44.getChannelData(c)) { if (!Number.isFinite(v)) bad44++; s44 += v * v; p44 = Math.max(p44, Math.abs(v)); }
    const pop44 = ev44.find((e) => e.type === 'pop' && e.source === 'scripted');
    const r44 = { sampleRate: b44.sampleRate, rmsDb: +(10 * Math.log10(s44 / (2 * b44.length))).toFixed(1), peakDb: +(20 * Math.log10(p44)).toFixed(2), nonFinite: bad44, popContextTime: pop44 ? pop44.contextTime : null };
    return { rmsDb: 10 * Math.log10(ss / (2 * N)), peakDb: 20 * Math.log10(pk), nonFinite: bad, events: evs.map((e) => ({ type: e.type, source: e.source, contextTime: +e.contextTime.toFixed(4) })), workletEqualsMainThread: same, L: Array.from(L), R: Array.from(R), r44 };
  });
  // Node reference for the parity check
  const eng = new FireAudioEngine({ sampleRate: 48000, seed: 99 });
  eng.setControls({ intensity: 1.1, hrr: 1, camDist: 1.7, camAz: 0.4 });
  const L = new Float32Array(48000), R = new Float32Array(48000), s0 = new Float32Array(48000), s1 = new Float32Array(48000);
  for (let i = 0; i < 48000; i += 128) eng.render(L.subarray(i, i + 128), R.subarray(i, i + 128), s0.subarray(i, i + 128), s1.subarray(i, i + 128), 128);
  let maxDiff = 0, nDiff = 0, firstDiff = -1;
  for (let i = 0; i < 48000; i++) {
    const d = Math.max(Math.abs(off.L[i] - L[i]), Math.abs(off.R[i] - R[i]));
    if (d > 0) { nDiff++; if (firstDiff < 0) firstDiff = i; }
    if (d > maxDiff) maxDiff = d;
  }
  const popEv = off.events.find((e) => e.type === 'pop' && e.source === 'scripted');
  check('offline_full_graph', off.nonFinite === 0 && off.rmsDb > -50 && off.peakDb <= -0.99 && popEv && Math.abs(popEv.contextTime - 1.0) < 0.001, { rmsDb: +off.rmsDb.toFixed(1), peakDb: +off.peakDb.toFixed(2), events: off.events });
  check('offline_full_graph_44k', off.r44.nonFinite === 0 && off.r44.rmsDb > -50 && off.r44.peakDb <= -0.99 && off.r44.popContextTime !== null && Math.abs(off.r44.popContextTime - 0.5) < 0.001, off.r44);
  check('worklet_bit_exact_vs_main_thread_js', off.workletEqualsMainThread, {});
  // Node 22 and Chrome ship different V8 versions; transcendental functions may differ by an ulp
  check('worklet_vs_node_render', maxDiff < 1e-6, { maxAbsDiff: maxDiff, maxDiffDbFS: +(20 * Math.log10(maxDiff + 1e-30)).toFixed(1), differingSamples: nDiff, firstDiffSample: firstDiff, nodeVersion: process.versions.v8 });
  await page.close();

  // ------------------------------------------------------------ 3. default autoplay policy: start() needs a gesture
  const b2 = await puppeteer.launch({ executablePath: CHROME, headless: !args.includes('--headful') });
  try {
    const p2 = await b2.newPage(); watch(p2, 'gesture');
    await p2.goto(url, { waitUntil: 'load' });
    await p2.waitForFunction(() => window.__audioTest && window.__audioTest.ready, { timeout: 20000 });
    const before = await p2.evaluate(async () => { const fa = window.__audioTest.fa; const st = await fa.start(); return { state: st, needsGesture: fa.needsGesture }; });
    await p2.mouse.click(5, 5); // any user gesture on the page resumes the context
    await sleep(800);
    const after = await p2.evaluate(() => ({ state: window.__audioTest.fa.state, needsGesture: window.__audioTest.fa.needsGesture }));
    check('gesture_handling', after.state === 'running' && !after.needsGesture, { beforeGesture: before, afterGesture: after });
  } finally { await b2.close(); }
} catch (e) {
  result.ok = false; result.exception = String(e && e.stack || e);
} finally {
  await browser.close();
  if (server) server.close();
}
// "The AudioContext was not allowed to start" is the expected warning before a gesture, not an error
result.consoleErrors = result.consoleErrors.filter((s) => !/not allowed to start/i.test(s));
if (result.consoleErrors.length || result.pageErrors.length) result.ok = false;
console.log(JSON.stringify(result, null, 1));
process.exit(result.ok ? 0 : 1);
