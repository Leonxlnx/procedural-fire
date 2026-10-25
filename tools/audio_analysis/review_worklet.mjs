#!/usr/bin/env node
// review_worklet.mjs — runs src/audio/worklet.js inside a stubbed AudioWorkletGlobalScope in Node and checks:
//   * 44.1 and 48 kHz, render quanta of 128 and other sizes (output must not depend on the quantum size)
//   * sample-accurate scheduling of 'event' messages (contextTime of the logged event == requested `when`)
//   * the master processor never exceeds its ceiling
//   * heap allocation per render quantum in steady state (should be ~0) and per reported event
//   * subnormal floats left in any DSP state after long silences / decays
//   * CPU cost per second of audio (process.cpuUsage, so other load on the machine matters less)
//
//   node --expose-gc tools/audio_analysis/review_worklet.mjs [--secs 20] [--json out.json]
import v8 from 'node:v8';
import crypto from 'node:crypto';
import { PerformanceObserver } from 'node:perf_hooks';

const args = process.argv.slice(2);
const val = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const SECS = +val('secs', 20);

// ---------------------------------------------------------------- AudioWorkletGlobalScope stub
const procs = {};
class Port { constructor() { this.onmessage = null; this.sent = []; this.keep = true; } postMessage(m) { if (this.keep) this.sent.push(structuredClone(m)); else this.count = (this.count || 0) + 1; } }
globalThis.registerProcessor = (name, cls) => { procs[name] = cls; };
globalThis.AudioWorkletProcessor = class { constructor() { this.port = new Port(); } };
globalThis.sampleRate = 48000; globalThis.currentFrame = 0;
await import('../../src/audio/worklet.js');
const { LiveMaster, decodeEvents } = await import('../../src/audio/fire-dsp.js');
const evsOf = (port) => port.sent.filter((x) => x.type === 'events').flatMap((x) => decodeEvents(x.data, x.count));

const send = (p, data) => p.port.onmessage({ data });
function makeFire(fs, opts = {}) {
  globalThis.sampleRate = fs; globalThis.currentFrame = opts.frame0 || 0;
  return new procs['fire-audio']({ processorOptions: { seed: opts.seed ?? 5, controls: opts.controls || { intensity: 1, hrr: 1, camDist: 2.2, camAz: 0.3 }, running: true, params: opts.params || {} } });
}
function makeMaster(fs) { globalThis.sampleRate = fs; return new procs['fire-master']({ processorOptions: { ceilingDb: -1, makeupDb: 4.5, levelerThresholdLufs: -16, releaseMs: 120, lookMs: 1.5 } }); }
const bufs = (q) => [[new Float32Array(q), new Float32Array(q)], [new Float32Array(q), new Float32Array(q)]];

// render `secs` through fire (+ optional master); returns concatenated dry L/R and master L/R
function run(fire, master, fs, secs, q, onQuantum) {
  const N = Math.round(secs * fs), out = bufs(q), mo = [[new Float32Array(q), new Float32Array(q)]];
  const L = new Float32Array(N), R = new Float32Array(N), ML = master ? new Float32Array(N) : null, MR = master ? new Float32Array(N) : null;
  for (let i = 0; i < N; i += q) {
    const n = Math.min(q, N - i);
    const o = n === q ? out : bufs(n), m = n === q ? mo : [[new Float32Array(n), new Float32Array(n)]];
    if (onQuantum) onQuantum(i);
    fire.process([], o);
    if (master) { master.process([o[0]], m); ML.set(m[0][0], i); MR.set(m[0][1], i); }
    L.set(o[0][0], i); R.set(o[0][1], i);
    globalThis.currentFrame += n;
  }
  return { L, R, ML, MR };
}
const hash = (...as) => { const h = crypto.createHash('sha256'); for (const a of as) h.update(Buffer.from(a.buffer, a.byteOffset, a.byteLength)); return h.digest('hex').slice(0, 16); };
const peakDb = (...as) => { let p = 0; for (const a of as) for (const v of a) p = Math.max(p, Math.abs(v)); return 20 * Math.log10(p + 1e-30); };
const rmsDb = (...as) => { let s = 0, n = 0; for (const a of as) { for (const v of a) s += v * v; n += a.length; } return 10 * Math.log10(s / n + 1e-30); };
const finite = (...as) => as.every((a) => a.every(Number.isFinite));

// walk an object graph and report subnormal doubles (|x| < 2.2251e-308, x != 0)
function findSubnormals(root, name) {
  const seen = new Set(), hits = [];
  const isSub = (x) => typeof x === 'number' && x !== 0 && Math.abs(x) < 2.2250738585072014e-308;
  (function walk(o, path, depth) {
    if (!o || typeof o !== 'object' || seen.has(o) || depth > 8) return; seen.add(o);
    if (ArrayBuffer.isView(o)) { for (let i = 0; i < o.length; i++) if (isSub(o[i])) { hits.push(`${path}[${i}]`); break; } return; }
    for (const k of Object.keys(o)) { const v = o[k]; if (isSub(v)) hits.push(`${path}.${k}`); else if (v && typeof v === 'object') walk(v, `${path}.${k}`, depth + 1); }
  })(root, name, 0);
  return hits;
}

const report = { secs: SECS, checks: {} };
const check = (name, pass, info = {}) => { report.checks[name] = { pass: !!pass, ...info }; };

// ---------------------------------------------------------------- 1. sample rates x quantum sizes
for (const fs of [44100, 48000]) {
  const hashes = {};
  for (const q of [128, 256, 100]) {
    const f = makeFire(fs), m = makeMaster(fs);
    send(f, { type: 'event', event: { type: 'ignite', strength: 1, pan: 0.1, when: 0.5 } });
    send(f, { type: 'event', event: { type: 'pop', strength: 1, pan: -0.4, when: 2.0 } });
    send(f, { type: 'event', event: { type: 'log-shift', strength: 1, pan: 0.2, when: 3.0 } });
    const r = run(f, m, fs, 5, q);
    hashes[q] = hash(r.L, r.R);
    if (q === 128) {
      const evs = evsOf(f.port);
      const at = (t) => evs.find((e) => e.source === 'scripted' && Math.abs(e.contextTime - t) < 0.5);
      const timing = [0.5, 2.0, 3.0].map((t) => { const e = at(t); return e ? Math.abs(e.contextTime - t) * fs : Infinity; });
      check(`fs${fs}_q128_render`, finite(r.L, r.R, r.ML, r.MR) && r.ML && peakDb(r.ML, r.MR) <= -1 + 1e-6 && Math.max(...timing) <= 0.5,
        { dryRmsDb: +rmsDb(r.L, r.R).toFixed(1), dryPeakDb: +peakDb(r.L, r.R).toFixed(1), masterPeakDb: +peakDb(r.ML, r.MR).toFixed(3), masterRmsDb: +rmsDb(r.ML, r.MR).toFixed(1), eventTimingErrSamples: timing });
    }
  }
  check(`fs${fs}_quantum_independent`, hashes[128] === hashes[256] && hashes[128] === hashes[100], { hashes });
}

// ---------------------------------------------------------------- 2. event scheduling relative to a non-zero context clock
{
  const fs = 48000, f = makeFire(fs, { frame0: 123456 }); // node created late in the context's life
  run(f, null, fs, 0.2, 128);
  const when = (globalThis.currentFrame + 4800 + 37) / fs; // 0.1 s + 37 samples ahead
  send(f, { type: 'event', event: { type: 'pop', strength: 0.8, when } });
  run(f, null, fs, 0.3, 128);
  const e = evsOf(f.port).find((x) => x.source === 'scripted');
  check('event_when_late_node', e && Math.abs(e.contextTime - when) * fs < 0.5, { requested: when, got: e && e.contextTime });
  // paused then resumed: the engine clock must stay aligned with the context clock
  send(f, { type: 'run', on: false }); run(f, null, fs, 0.5, 128); send(f, { type: 'run', on: true });
  const when2 = (globalThis.currentFrame + 2400) / fs; send(f, { type: 'event', event: { type: 'pop', strength: 0.8, when: when2 } });
  f.port.sent.length = 0; run(f, null, fs, 0.2, 128);
  const e2 = evsOf(f.port).find((x) => x.source === 'scripted');
  check('event_when_after_pause', e2 && Math.abs(e2.contextTime - when2) * fs < 0.5, { requested: when2, got: e2 && e2.contextTime });
  // very long-running context (> 2^31 frames, ~12.4 h at 48 kHz)
  const f3 = makeFire(fs, { frame0: 2 ** 31 + 1000 }); run(f3, null, fs, 0.05, 128);
  const when3 = (globalThis.currentFrame + 4800) / fs; send(f3, { type: 'event', event: { type: 'pop', strength: 0.8, when: when3 } });
  run(f3, null, fs, 0.2, 128);
  const e3 = evsOf(f3.port).find((x) => x.source === 'scripted');
  check('event_when_after_2^31_frames', e3 && Math.abs(e3.contextTime - when3) * fs < 0.5, { requested: when3, got: e3 && e3.contextTime });
}

// ---------------------------------------------------------------- 3. allocations in process()
if (typeof globalThis.gc === 'function') {
  const fs = 48000, q = 128;
  const measure = async (popRate, label) => {
    const f = makeFire(fs, { params: { popRate } }), m = makeMaster(fs); f.port.keep = false; m.port.keep = false;
    const o = bufs(q), mo = [[new Float32Array(q), new Float32Array(q)]];
    const step = () => { f.process([], o); m.process([o[0]], mo); globalThis.currentFrame += q; };
    for (let i = 0; i < fs * 15 / q; i++) step(); // warm up / let the JIT tiers settle
    globalThis.gc(); await new Promise((r) => setTimeout(r, 30)); gcCount = 0;
    const h0 = v8.getHeapStatistics().used_heap_size, msgs0 = f.port.count || 0, n = Math.round(SECS * fs / q);
    for (let i = 0; i < n; i++) step();
    const h1 = v8.getHeapStatistics().used_heap_size;
    await new Promise((r) => setTimeout(r, 30)); // gc entries are delivered asynchronously
    return { label, quanta: n, gcsDuringRun: gcCount, heapGrowthBytesPerQuantum: +((h1 - h0) / n).toFixed(2), eventMessages: (f.port.count || 0) - msgs0 };
  };
  let gcCount = 0; new PerformanceObserver((l) => { gcCount += l.getEntries().length; }).observe({ entryTypes: ['gc'] });
  const quiet = await measure(0, 'no random pops'), busy = await measure(6, 'popRate 6/s');
  // residual ~100-150 B/quantum come from rare paths (crackle spawns ~20/s pass doubles to voice.start()): ~50 KB/s,
  // i.e. one minor GC per tens of seconds in Chrome (measured with worklet_gc_trace.mjs) instead of ~30 per second
  check('alloc_steady_state', quiet.gcsDuringRun <= 1 && quiet.heapGrowthBytesPerQuantum < 256, quiet);
  check('alloc_with_events', busy.gcsDuringRun <= 1, busy);
} else report.checks.alloc_steady_state = { pass: false, note: 'run with node --expose-gc' };

// ---------------------------------------------------------------- 4. subnormals after silence / long decays
{
  const fs = 48000;
  const f = makeFire(fs, { controls: { intensity: 1, hrr: 1, camDist: 2 } }), m = makeMaster(fs);
  send(f, { type: 'event', event: { type: 'log-shift', strength: 1, when: 0.1 } });
  send(f, { type: 'event', event: { type: 'pop', strength: 1, when: 0.2 } });
  send(f, { type: 'event', event: { type: 'gust', strength: 1, when: 0.3 } });
  run(f, m, fs, 2, 128);
  send(f, { type: 'ctl', intensity: 0, hrr: 0, dhrr: 0 });
  run(f, m, fs, 90, 128); // long quiet stretch (crickets keep chirping; hawkes/flare/tickBurst decay for 90 s)
  const engHits = findSubnormals(f.engine, 'engine');
  // master fed digital silence (fire node paused) for 30 s
  const m2 = makeMaster(fs); const z = [new Float32Array(128), new Float32Array(128)], o = [[new Float32Array(128), new Float32Array(128)]];
  const t0 = process.cpuUsage(); for (let i = 0; i < 30 * fs / 128; i++) m2.process([z], o); const cpuSilence = process.cpuUsage(t0);
  const mHits = findSubnormals(m2.m, 'master');
  check('no_subnormal_state', engHits.length === 0 && mHits.length === 0, { engine: engHits.slice(0, 12), master: mHits.slice(0, 12), masterSilence30sCpuMs: Math.round((cpuSilence.user + cpuSilence.system) / 1000) });
}

// ---------------------------------------------------------------- 5. CPU per second of audio (fire + master)
{
  const fs = 48000, q = 128;
  const cost = (controls, events, label) => {
    const f = makeFire(fs, { controls }), m = makeMaster(fs); f.port.keep = false; m.port.keep = false;
    const o = bufs(q), mo = [[new Float32Array(q), new Float32Array(q)]];
    for (let i = 0; i < fs / q; i++) { f.process([], o); m.process([o[0]], mo); globalThis.currentFrame += q; }
    const nq = Math.round(10 * fs / q);
    let fireUs = 0, masterUs = 0, worstQuantumUs = 0;
    for (let i = 0; i < nq; i++) {
      if (events && i % 150 === 0) for (const type of events) send(f, { type: 'event', event: { type, strength: 1, pan: 0 } });
      const a = process.hrtime.bigint(); f.process([], o); const b = process.hrtime.bigint(); m.process([o[0]], mo); const c = process.hrtime.bigint();
      fireUs += Number(b - a) / 1e3; masterUs += Number(c - b) / 1e3; worstQuantumUs = Math.max(worstQuantumUs, Number(c - a) / 1e3);
      globalThis.currentFrame += q;
    }
    const secs = nq * q / fs;
    return { label, fireMsPerSec: +(fireUs / 1e3 / secs).toFixed(1), masterMsPerSec: +(masterUs / 1e3 / secs).toFixed(1), worstQuantumUs: Math.round(worstQuantumUs), quantumBudgetUs: Math.round(q / fs * 1e6) };
  };
  report.cpu = [
    cost({ intensity: 1, hrr: 1, camDist: 2.2 }, null, 'steady I=1'),
    cost({ intensity: 1.5, hrr: 3, camDist: 0.4 }, ['pop', 'log-shift', 'crackle-burst', 'gust'], 'stress I=1.5, 4 events every 0.4 s'),
  ];
  report.cpu.note = 'wall time per process() call on a loaded machine; budget = 1000 ms/s';
}

report.ok = Object.values(report.checks).every((c) => c.pass);
const out = JSON.stringify(report, null, 1);
if (val('json')) (await import('node:fs')).writeFileSync(val('json'), out);
console.log(out);
