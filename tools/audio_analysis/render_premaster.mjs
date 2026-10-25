// render_premaster.mjs - render a timeline WITHOUT mastering (no glue comp, soft clip or
// limiter) as 32-bit float WAV at a fixed gain, so texture statistics can be measured
// on the raw synthesis and compared with the mastered file. Diagnostic only.
//
// usage: node tools/audio_analysis/render_premaster.mjs <timeline.json> <out.wav>
//          [--engine research/audio-ref/fire-audio-dsp.mjs] [--gainDb -6] [--solo layer]
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const argv = process.argv.slice(2);
const pos = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && argv[i - 1].startsWith('--')));
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const [tlPath, outPath] = pos;
if (!tlPath || !outPath) { console.error('usage: render_premaster.mjs <timeline.json> <out.wav> [--engine m.mjs] [--gainDb -6] [--solo layer]'); process.exit(1); }
const enginePath = path.resolve(opt('engine', 'research/audio-ref/fire-audio-dsp.mjs'));
const { FireAudioEngine, makeOutdoorIR } = await import(pathToFileURL(enginePath).href);

const tl = JSON.parse(fs.readFileSync(tlPath, 'utf8'));
const sr = tl.sampleRate || 48000, N = Math.round(tl.duration * sr), block = 128;
const eng = new FireAudioEngine({ sampleRate: sr, seed: tl.seed, params: tl.params || {}, curves: tl.curves || {}, solo: opt('solo', null) });
for (const ev of tl.events || []) eng.schedule(ev);
const dL = new Float32Array(N), dR = new Float32Array(N), sL = new Float32Array(N), sR = new Float32Array(N);
for (let i = 0; i < N; i += block) {
  const n = Math.min(block, N - i);
  eng.render(dL.subarray(i, i + n), dR.subarray(i, i + n), sL.subarray(i, i + n), sR.subarray(i, i + n), n);
}

// direct-form convolution via radix-2 FFT overlap-add (same as render-offline.mjs)
function fft(re, im, inv) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inv ? 2 : -2) * Math.PI / len, half = len >> 1, wr0 = Math.cos(ang), wi0 = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half, xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        const t = wr * wr0 - wi * wi0; wi = wr * wi0 + wi * wr0; wr = t;
      }
    }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
function convolve(x, h) {
  let P = 1; while (P < 2 * h.length) P <<= 1;
  const B = P - h.length + 1, y = new Float64Array(x.length + h.length);
  const hr = new Float64Array(P), hi = new Float64Array(P); hr.set(h); fft(hr, hi, false);
  const ar = new Float64Array(P), ai = new Float64Array(P);
  for (let s = 0; s < x.length; s += B) {
    ar.fill(0); ai.fill(0); ar.set(x.subarray(s, Math.min(x.length, s + B)));
    fft(ar, ai, false);
    for (let k = 0; k < P; k++) { const r = ar[k] * hr[k] - ai[k] * hi[k], q = ar[k] * hi[k] + ai[k] * hr[k]; ar[k] = r; ai[k] = q; }
    fft(ar, ai, true);
    const n = Math.min(P, y.length - s); for (let k = 0; k < n; k++) y[s + k] += ar[k];
  }
  return y.subarray(0, x.length);
}
const ir = makeOutdoorIR(sr, tl.seed, tl.reverb || {});
const wL = convolve(sL, ir[0]), wR = convolve(sR, ir[1]);
const g = 10 ** (+opt('gainDb', -6) / 20);
const buf = Buffer.alloc(44 + N * 8);
buf.write('RIFF', 0); buf.writeUInt32LE(36 + N * 8, 4); buf.write('WAVE', 8); buf.write('fmt ', 12);
buf.writeUInt32LE(16, 16); buf.writeUInt16LE(3, 20); buf.writeUInt16LE(2, 22); buf.writeUInt32LE(sr, 24);
buf.writeUInt32LE(sr * 8, 28); buf.writeUInt16LE(8, 32); buf.writeUInt16LE(32, 34); buf.write('data', 36); buf.writeUInt32LE(N * 8, 40);
let peak = 0;
for (let i = 0, o = 44; i < N; i++, o += 8) {
  const l = g * (dL[i] + wL[i] + 0.15 * wR[i]), r = g * (dR[i] + wR[i] + 0.15 * wL[i]);
  peak = Math.max(peak, Math.abs(l), Math.abs(r));
  buf.writeFloatLE(l, o); buf.writeFloatLE(r, o + 4);
}
fs.writeFileSync(outPath, buf);
console.log(JSON.stringify({ out: outPath, seconds: tl.duration, gainDb: +opt('gainDb', -6), samplePeakDb: +(20 * Math.log10(peak)).toFixed(2) }));
