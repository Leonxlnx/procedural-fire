#!/usr/bin/env node
// alloc_profile.mjs — sampling heap profile of the worklet render path (fire + master processors, no events),
// to find allocation sites in process(). Prints the top allocating functions (self bytes).
//   node tools/audio_analysis/alloc_profile.mjs [--secs 10] [--pops 0]
import inspector from 'node:inspector';

const args = process.argv.slice(2);
const val = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const procs = {};
globalThis.registerProcessor = (name, cls) => { procs[name] = cls; };
globalThis.AudioWorkletProcessor = class { constructor() { this.port = { onmessage: null, postMessage() {} }; } };
globalThis.sampleRate = 48000; globalThis.currentFrame = 0;
await import('../../src/audio/worklet.js');

const fire = new procs['fire-audio']({ processorOptions: { seed: 5, controls: { intensity: 1, hrr: 1, camDist: 2.2 }, running: true, params: { popRate: +val('pops', 0) } } });
const master = new procs['fire-master']({ processorOptions: { ceilingDb: -1, makeupDb: 4.5 } });
const o = [[new Float32Array(128), new Float32Array(128)], [new Float32Array(128), new Float32Array(128)]], mo = [[new Float32Array(128), new Float32Array(128)]];
const noIn = [], mIn = [o[0]];
const step = () => { fire.process(noIn, o); master.process(mIn, mo); globalThis.currentFrame += 128; };
for (let i = 0; i < 48000 * 3 / 128; i++) step();

const s = new inspector.Session(); s.connect();
const post = (m, p = {}) => new Promise((res, rej) => s.post(m, p, (e, r) => (e ? rej(e) : res(r))));
await post('HeapProfiler.enable');
await post('HeapProfiler.startSampling', { samplingInterval: 64, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
const nq = Math.round(+val('secs', 10) * 48000 / 128);
for (let i = 0; i < nq; i++) step();
const { profile } = await post('HeapProfiler.stopSampling');
const agg = new Map();
(function walk(n, stack) {
  const f = n.callFrame, name = `${f.functionName || '(anon)'} ${f.url.split('/').pop()}:${f.lineNumber + 1}:${f.columnNumber + 1}`;
  const st = [...stack, name];
  if (n.selfSize > 0) { const k = st.slice(-3).join(' <- '); agg.set(k, (agg.get(k) || 0) + n.selfSize); }
  for (const c of n.children) walk(c, st);
})(profile.head, []);
const rows = [...agg.entries()].sort((a, b) => b[1] - a[1]).slice(0, 15);
const total = [...agg.values()].reduce((a, b) => a + b, 0);
console.log(JSON.stringify({ quanta: nq, sampledBytes: total, bytesPerQuantum: +(total / nq).toFixed(1), top: rows.map(([k, v]) => ({ bytes: v, perQuantum: +(v / nq).toFixed(2), site: k })) }, null, 1));
