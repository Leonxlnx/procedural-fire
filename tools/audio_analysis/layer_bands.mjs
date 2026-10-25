#!/usr/bin/env node
// layer_bands.mjs — per-layer octave-band power (dry + reverb, pre-master) for a steady fire, to see which layer
// dominates which band (e.g. which one fills the high-frequency bed between crackles).
//   node tools/audio_analysis/layer_bands.mjs [--dur 6] [--params '{...}'] [--layers ticks,crackles,...]
import { FireAudioEngine, REVERB_DEFAULTS, RNG, deriveSeed } from '../../src/audio/fire-dsp.js';
import { mixWithReverb, fft } from '../../src/audio/master-offline.js';

const a = process.argv.slice(2), val = (k, d) => { const i = a.indexOf('--' + k); return i >= 0 ? a[i + 1] : d; };
const sr = 48000, dur = +val('dur', 6), N = dur * sr, params = val('params') ? JSON.parse(val('params')) : {};
const layers = val('layers', 'all,ticks,crackles,roar,breath,lapping,hiss,whump,pops,embers,crickets').split(',');
const fps = 60, r = new RNG(deriveSeed(7, 'steady-hrr')); let nb1 = 0, nb2 = 0; const hrr = [];
for (let i = 0; i < dur * fps; i++) { const t = i / fps; nb1 += 0.25 * (r.gauss() - nb1); nb2 += 0.05 * (nb1 - nb2); hrr.push(1 + 0.06 * Math.sin(2 * Math.PI * 1.75 * t + 0.4 * Math.sin(2 * Math.PI * 0.21 * t)) + 0.12 * nb2); }
const frames = hrr.map((h, i) => ({ intensity: 1, hrr: h, dhrr: (hrr[Math.min(hrr.length - 1, i + 1)] - hrr[Math.max(0, i - 1)]) * fps / 2, camDist: 2, camAz: 0.35 }));
const OCT = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
function bands([L, R]) {
  const n = 8192, P = new Float64Array(n / 2 + 1);
  for (const x of [L, R]) for (let s = sr; s + n <= N; s += n / 2) { const re = new Float64Array(n), im = new Float64Array(n); for (let i = 0; i < n; i++) re[i] = x[s + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n)); fft(re, im, false); for (let k = 0; k <= n / 2; k++) P[k] += re[k] * re[k] + im[k] * im[k]; }
  return OCT.map((fc) => { let s = 0; for (let k = 0; k <= n / 2; k++) { const f = k * sr / n; if (f >= fc / Math.SQRT2 && f < fc * Math.SQRT2) s += P[k]; } return 10 * Math.log10(s + 1e-30); });
}
const rows = [];
for (const l of layers) {
  const e = new FireAudioEngine({ sampleRate: sr, seed: 7, params, solo: l === 'all' ? null : l });
  e.setTimeline({ fps, frames });
  const b = [0, 1, 2, 3].map(() => new Float32Array(N));
  for (let i = 0; i < N; i += 128) e.render(b[0].subarray(i, i + 128), b[1].subarray(i, i + 128), b[2].subarray(i, i + 128), b[3].subarray(i, i + 128), 128);
  rows.push([l, bands(mixWithReverb(b[0], b[1], b[2], b[3], sr, 7, REVERB_DEFAULTS))]);
}
const ref = rows[0][1];
console.log('octave band power, dB re the full mix in that band (all = absolute dB)');
console.log('layer    ' + OCT.map((f) => String(f).padStart(7)).join(''));
for (const [l, v] of rows) console.log(l.padEnd(9) + v.map((x, i) => (process.argv.includes('--abs') || l === 'all' ? x : x - ref[i]).toFixed(1).padStart(7)).join(''));
