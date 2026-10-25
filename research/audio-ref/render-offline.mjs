// render-offline.mjs — deterministic offline render of the procedural fire soundtrack (Node >= 18)
// usage: node render-offline.mjs --timeline timeline.example.json --out fire.wav [--events events.json]
//        [--solo roar] [--block 128] [--nomaster] [--stats]
import fs from 'node:fs';
import crypto from 'node:crypto';
import { FireAudioEngine, makeOutdoorIR, masterChain, integratedLoudness, makeTruePeak, Biquad } from './fire-audio-dsp.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => {
  if (x.startsWith('--')) a.push([x.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]); return a;
}, []));
const tl = JSON.parse(fs.readFileSync(args.timeline || new URL('./timeline.example.json', import.meta.url), 'utf8'));
const sr = tl.sampleRate || 48000, N = Math.round(tl.duration * sr), block = +(args.block || 128);

// ---------------------------------------------------------------- render engine (dry + reverb send)
const t0 = Date.now();
const eng = new FireAudioEngine({ sampleRate: sr, seed: tl.seed, params: tl.params || {}, curves: tl.curves || {}, solo: args.solo || null });
for (const ev of tl.events || []) eng.schedule(ev);
const dL = new Float32Array(N), dR = new Float32Array(N), sL = new Float32Array(N), sR = new Float32Array(N);
for (let i = 0; i < N; i += block) {
  const n = Math.min(block, N - i);
  eng.render(dL.subarray(i, i + n), dR.subarray(i, i + n), sL.subarray(i, i + n), sR.subarray(i, i + n), n);
}
const tEngine = Date.now() - t0;

// ---------------------------------------------------------------- FFT convolution reverb
function fft(re, im, inv) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inv ? 2 : -2) * Math.PI / len, half = len >> 1;
    const wr0 = Math.cos(ang), wi0 = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        const t = wr * wr0 - wi * wi0; wi = wr * wi0 + wi * wr0; wr = t;
      }
    }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
// overlap-add FFT convolution (FFT size >= 2*IR length keeps the working set cache-friendly)
function convolve(x, h) {
  let P = 1; while (P < 2 * h.length) P <<= 1;
  const B = P - h.length + 1, y = new Float64Array(x.length + h.length);
  const hr = new Float64Array(P), hi = new Float64Array(P); hr.set(h); fft(hr, hi, false);
  const ar = new Float64Array(P), ai = new Float64Array(P);
  for (let s = 0; s < x.length; s += B) {
    ar.fill(0); ai.fill(0); ar.set(x.subarray(s, Math.min(x.length, s + B)));
    fft(ar, ai, false);
    for (let k = 0; k < P; k++) { const r = ar[k] * hr[k] - ai[k] * hi[k], i = ar[k] * hi[k] + ai[k] * hr[k]; ar[k] = r; ai[k] = i; }
    fft(ar, ai, true);
    const n = Math.min(P, y.length - s); for (let k = 0; k < n; k++) y[s + k] += ar[k];
  }
  return y.subarray(0, x.length);
}
const ir = makeOutdoorIR(sr, tl.seed, tl.reverb || {});
const wL = convolve(sL, ir[0]), wR = convolve(sR, ir[1]);
let L = new Float64Array(N), R = new Float64Array(N);
for (let i = 0; i < N; i++) { L[i] = dL[i] + wL[i] + 0.15 * wR[i]; R[i] = dR[i] + wR[i] + 0.15 * wL[i]; }

// ---------------------------------------------------------------- master (2-pass loudness)
const target = (tl.master && tl.master.targetLufs) || -14, ceilingDb = (tl.master && tl.master.ceilingDbTP) || -1.0;
const preLufs = integratedLoudness([L, R], sr);
let outL = L, outR = R, gainDb = 0;
if (args.gainDb !== undefined) gainDb = +args.gainDb; // fixed gain (e.g. to analyse a soloed layer at the full-mix gain)
else if (!args.nomaster) {
  gainDb = target - preLufs;
  // internal estimator reads ~0.4 dB lower than ffmpeg ebur128 true-peak -> keep 0.6 dB margin
  const mopts = { ceilingDb: ceilingDb - 0.6, compThresholdDb: preLufs - 2, phoneBodyDb: 1.5, airDb: -1.5 };
  for (let pass = 0; pass < 3; pass++) {
    [outL, outR] = masterChain(L, R, sr, { ...mopts, gainDb });
    const got = integratedLoudness([outL, outR], sr);
    if (Math.abs(got - target) < 0.05) break;
    gainDb += target - got;
  }
}
const lufs = integratedLoudness([outL, outR], sr);
// how hard do transients hit the ceiling? (pre-limiter excess over the ceiling after the loudness gain)
const clip = (() => {
  const g = Math.pow(10, gainDb / 20), c = Math.pow(10, (ceilingDb - 0.6) / 20);
  let n = 0, maxEx = 0, events = 0, last = -1e9;
  for (let i = 0; i < N; i++) { const a = Math.max(Math.abs(L[i]), Math.abs(R[i])) * g; if (a > c) { n++; maxEx = Math.max(maxEx, 20 * Math.log10(a / c)); if (i - last > 0.01 * sr) events++; last = i; } }
  return { samplesOverCeilingPct: +(100 * n / N).toFixed(3), clipEventsPerSec: +(events / tl.duration).toFixed(2), maxExcessDb: +maxEx.toFixed(1) };
})();
const tp = makeTruePeak(16);
let tpMax = 0; for (const a of [tp(outL), tp(outR)]) for (let i = 0; i < a.length; i++) if (a[i] > tpMax) tpMax = a[i];
const tpDb = 20 * Math.log10(tpMax);
let spk = 0; for (let i = 0; i < N; i++) spk = Math.max(spk, Math.abs(outL[i]), Math.abs(outR[i]));

// ---------------------------------------------------------------- WAV (24-bit PCM, TPDF dither from seeded LCG)
function writeWav24(path, chans, sr) {
  const n = chans[0].length, nc = chans.length, data = n * nc * 3, buf = Buffer.alloc(44 + data);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + data, 4); buf.write('WAVE', 8); buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(nc, 22); buf.writeUInt32LE(sr, 24);
  buf.writeUInt32LE(sr * nc * 3, 28); buf.writeUInt16LE(nc * 3, 32); buf.writeUInt16LE(24, 34); buf.write('data', 36); buf.writeUInt32LE(data, 40);
  let seed = 12345, o = 44;
  const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++) {
    let v = Math.round(chans[c][i] * 8388607 + (rnd() - rnd()));
    v = Math.max(-8388608, Math.min(8388607, v)); buf.writeIntLE(v, o, 3); o += 3;
  }
  fs.writeFileSync(path, buf);
}
if (args.out) writeWav24(args.out, [outL, outR], sr);
if (args.events) fs.writeFileSync(args.events, JSON.stringify({ sampleRate: sr, fps: tl.fps || 60, events: eng.eventLog.map((e) => ({ ...e, frame: Math.round(e.t * (tl.fps || 60)) })) }, null, 1));

// ---------------------------------------------------------------- report
const hash = crypto.createHash('sha256').update(Buffer.from(new Float32Array(dL).buffer)).update(Buffer.from(new Float32Array(sR).buffer)).digest('hex').slice(0, 16);
const out = { seconds: tl.duration, block, engineMs: tEngine, realtimeFactor: +(tl.duration * 1000 / tEngine).toFixed(1), solo: args.solo || null,
  preMasterLufs: +preLufs.toFixed(2), masterGainDb: +gainDb.toFixed(2), lufs: +lufs.toFixed(2), truePeakDb: +tpDb.toFixed(2), samplePeakDb: +(20 * Math.log10(spk)).toFixed(2),
  events: eng.eventLog.length, pops: eng.eventLog.filter((e) => e.type === 'pop').length, dryHash: hash, clip };
if (args.stats) {
  // octave-band RMS (dB re full scale) of the final mix, and band kurtosis (sparsity) as in McDermott et al.
  const bands = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
  out.octaveDb = {}; out.octaveKurtosis = {};
  for (const fc of bands) {
    const b1 = new Biquad().bp(fc, 1.41, sr), b2 = new Biquad().bp(fc, 1.41, sr);
    let s2 = 0, s4 = 0; for (let i = 0; i < N; i++) { const y = b2.tick(b1.tick(0.5 * (outL[i] + outR[i]))); const q = y * y; s2 += q; s4 += q * q; }
    const m2 = s2 / N; out.octaveDb[fc] = +(10 * Math.log10(m2 + 1e-30)).toFixed(1); out.octaveKurtosis[fc] = +((s4 / N) / (m2 * m2)).toFixed(1);
  }
  let c = 0, ll = 0, rr = 0; for (let i = 0; i < N; i++) { c += outL[i] * outR[i]; ll += outL[i] ** 2; rr += outR[i] ** 2; }
  out.lrCorrelation = +(c / Math.sqrt(ll * rr)).toFixed(3);
  // mono fold-down loss: LUFS(mono L+R/2 duplicated) - LUFS(stereo)
  const mono = new Float64Array(N); for (let i = 0; i < N; i++) mono[i] = 0.5 * (outL[i] + outR[i]);
  out.monoFoldDownDb = +(integratedLoudness([mono, mono], sr) - lufs).toFixed(2);
  // short-term loudness (3 s windows, 1 s hop) to show the dynamic arc of the 20 s piece
  out.shortTerm = []; for (let s = 0; s + 3 * sr <= N; s += sr) out.shortTerm.push(+integratedLoudness([outL.subarray(s, s + 3 * sr), outR.subarray(s, s + 3 * sr)], sr).toFixed(1));
}
console.log(JSON.stringify(out, null, 1));
