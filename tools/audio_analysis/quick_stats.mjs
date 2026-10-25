#!/usr/bin/env node
// quick_stats.mjs — fast inner-loop check for statistical tuning (no Python start-up): renders a timeline with
// optional parameter overrides, masters it like render_audio.mjs, and prints the transient statistics of
// texture_stats.py (same detector: >1.5 kHz, 0.5 ms frame peaks over a 250 ms running median), 1/3-octave
// LTAS re total, and band-wise L/R correlation. The authoritative numbers still come from texture_stats.py.
//
//   node tools/audio_analysis/quick_stats.mjs [--demo | --timeline tl.json | (default) steady]
//        [--dur 12] [--params '{"crkAmp0":0.02,"mix":{"ticks":-20}}'] [--no-master] [--wav out.wav]
import fs from 'node:fs';
import { Biquad, RNG, deriveSeed, REVERB_DEFAULTS } from '../../src/audio/fire-dsp.js';
import { mixWithReverb, fft, encodeWav24 } from '../../src/audio/master-offline.js';
import { renderTimeline, master, makeDemoTimeline } from '../render_audio.mjs';

const args = process.argv.slice(2);
const has = (k) => args.includes('--' + k);
const val = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const FS = 48000;

function steady(dur, I = 1, dist = 2, seed = 7) {
  const fps = 60, r = new RNG(deriveSeed(seed, 'steady-hrr')); let nb1 = 0, nb2 = 0; const hrr = [];
  for (let i = 0; i < dur * fps; i++) { const t = i / fps; nb1 += 0.25 * (r.gauss() - nb1); nb2 += 0.05 * (nb1 - nb2); hrr.push(Math.max(0, I * (1 + 0.06 * Math.sin(2 * Math.PI * 1.75 * t + 0.4 * Math.sin(2 * Math.PI * 0.21 * t)) + 0.12 * nb2))); }
  const frames = hrr.map((h, i) => ({ intensity: I, hrr: h, dhrr: (hrr[Math.min(hrr.length - 1, i + 1)] - hrr[Math.max(0, i - 1)]) * fps / 2, camDist: dist, camAz: 0.35 + 0.01 * i / fps }));
  return { fps, duration: dur, seed, frames, events: [] };
}

function filt(x, coefs, zeroPhase = false) {
  let y = Float64Array.from(x);
  for (const c of coefs) { const b = new Biquad(); b[c[0]](...c.slice(1)); for (let i = 0; i < y.length; i++) y[i] = b.tick(y[i]); }
  if (zeroPhase) { y.reverse(); for (const c of coefs) { const b = new Biquad(); b[c[0]](...c.slice(1)); for (let i = 0; i < y.length; i++) y[i] = b.tick(y[i]); } y.reverse(); }
  return y;
}
const butter4 = (kind, f) => [[kind, f, 0.5412, FS], [kind, f, 1.3066, FS]];

function median(a) { const s = Float64Array.from(a).sort(); const n = s.length; return n % 2 ? s[n >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]); }

function transients(mono) {
  const x = filt(mono, butter4('hp', 1500), true), hop = 24, nfr = Math.floor(x.length / hop), frs = FS / hop;
  const pk = new Float64Array(nfr), en = new Float64Array(nfr);
  for (let f = 0; f < nfr; f++) { let m = 0, e = 0; for (let k = 0; k < hop; k++) { const v = x[f * hop + k]; const a = Math.abs(v); if (a > m) m = a; e += v * v; } pk[f] = m; en[f] = e; }
  const win = Math.round(0.25 * frs), step = Math.round(0.01 * frs), lo = win >> 1, hi = win - lo;
  const padded = new Float64Array(nfr + win);
  for (let i = 0; i < padded.length; i++) { let j = i - lo; if (j < 0) j = -j; if (j >= nfr) j = 2 * (nfr - 1) - j; padded[i] = pk[Math.max(0, Math.min(nfr - 1, j))]; }
  const nb = Math.floor((padded.length - win) / step) + 1, bgc = new Float64Array(nb);
  for (let k = 0; k < nb; k++) bgc[k] = median(padded.subarray(k * step, k * step + win));
  const ratio = new Float64Array(nfr);
  for (let f = 0; f < nfr; f++) { const p = f / step, i0 = Math.min(nb - 1, Math.floor(p)), i1 = Math.min(nb - 1, i0 + 1), bg = bgc[i0] + (bgc[i1] - bgc[i0]) * (p - i0); ratio[f] = 20 * Math.log10(Math.max(pk[f], 1e-12) / Math.max(bg, 1e-12)); }
  // scipy find_peaks(height=6, distance=10)
  const cand = []; for (let f = 1; f < nfr - 1; f++) if (ratio[f] >= 6 && ratio[f] > ratio[f - 1] && ratio[f] >= ratio[f + 1]) cand.push(f);
  const dist = Math.round(0.005 * frs), keep = new Uint8Array(nfr), order = cand.slice().sort((a, b) => ratio[b] - ratio[a]), taken = [];
  for (const f of order) { let ok = true; for (let d = -dist + 1; d < dist; d++) if (f + d >= 0 && f + d < nfr && keep[f + d]) { ok = false; break; } if (ok) { keep[f] = 1; taken.push(f); } }
  taken.sort((a, b) => a - b);
  const dur = mono.length / FS, h = taken.map((f) => ratio[f]);
  const res = { rate: {} };
  for (const th of [6, 10, 15, 20]) res.rate[th] = +(h.filter((v) => v >= th).length / dur).toFixed(1);
  const mle = (xmin) => { const a = h.map((v) => 10 ** (v / 20)).filter((v) => v >= xmin); return a.length < 10 ? null : +(1 + a.length / a.reduce((s, v) => s + Math.log(v / xmin), 0)).toFixed(2); };
  res.alpha6 = mle(10 ** (6 / 20)); res.alpha10 = mle(10 ** (10 / 20));
  for (const th of [6, 10]) {
    const t = taken.filter((f) => ratio[f] >= th).map((f) => f / frs), ioi = t.slice(1).map((v, i) => v - t[i]);
    const m = ioi.reduce((a, b) => a + b, 0) / ioi.length, sd = Math.sqrt(ioi.reduce((a, b) => a + (b - m) ** 2, 0) / ioi.length);
    let lv = 0; for (let i = 0; i < ioi.length - 1; i++) lv += ((ioi[i] - ioi[i + 1]) / (ioi[i] + ioi[i + 1])) ** 2;
    res['cv' + th] = +(sd / m).toFixed(2); res['lv' + th] = +(3 * lv / (ioi.length - 1)).toFixed(2);
  }
  let e10 = 0, et = 0; for (let f = 0; f < nfr; f++) { et += en[f]; if (ratio[f] >= 10) e10 += en[f]; }
  res.efrac10 = +(e10 / et).toFixed(3);
  { const t = taken.filter((f) => ratio[f] >= 6).map((f) => f / frs), ioi = t.slice(1).map((v, i) => 1000 * (v - t[i])), edges = [5, 10, 20, 50, 100, 1e9];
    res.ioiHist6 = edges.slice(0, -1).map((e, i) => +(ioi.filter((v) => v >= e && v < edges[i + 1]).length / ioi.length).toFixed(3)); }
  return res;
}

function ltas(mono) {
  const n = 8192, w = new Float64Array(n).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n)), P = new Float64Array(n / 2 + 1);
  let cnt = 0;
  for (let s = 0; s + n <= mono.length; s += n / 2) { const re = new Float64Array(n), im = new Float64Array(n); for (let i = 0; i < n; i++) re[i] = mono[s + i] * w[i]; fft(re, im, false); for (let k = 0; k <= n / 2; k++) P[k] += re[k] * re[k] + im[k] * im[k]; cnt++; }
  const tot = P.reduce((a, b) => a + b, 0), out = {};
  for (const fc of [63, 125, 250, 500, 1000, 2000, 4000, 8000]) { let s = 0; for (let k = 0; k <= n / 2; k++) { const f = k * FS / n; if (f >= fc / Math.SQRT2 && f < fc * Math.SQRT2) s += P[k]; } out[fc] = +(10 * Math.log10(s / tot) - 4.77).toFixed(1); } // ~ per 1/3 octave
  return out;
}

function corr(a, b) { let ab = 0, aa = 0, bb = 0; for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; } return +(ab / Math.sqrt(aa * bb)).toFixed(3); }

function readWav(path) { // 16/24-bit PCM stereo/mono
  const b = fs.readFileSync(path); let o = 12, fmt = null, data = null;
  while (o + 8 <= b.length) { const id = b.toString('ascii', o, o + 4), sz = b.readUInt32LE(o + 4); if (id === 'fmt ') fmt = { ch: b.readUInt16LE(o + 10), bits: b.readUInt16LE(o + 22) }; if (id === 'data') data = [o + 8, sz]; o += 8 + sz + (sz & 1); }
  const bps = fmt.bits / 8, n = Math.floor(data[1] / (bps * fmt.ch)), ch = [new Float64Array(n), new Float64Array(n)];
  for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) { const p = data[0] + (i * fmt.ch + Math.min(c, fmt.ch - 1)) * bps; ch[c][i] = bps === 3 ? b.readIntLE(p, 3) / 8388608 : b.readInt16LE(p) / 32768; }
  return ch;
}

const params = val('params') ? JSON.parse(val('params')) : {};
let out, grDb = 0, gain = 0;
if (val('in')) out = readWav(val('in'));
else {
  const tl = has('demo') ? makeDemoTimeline() : val('timeline') ? JSON.parse(fs.readFileSync(val('timeline'), 'utf8')) : steady(+val('dur', 12));
  tl.params = { ...(tl.params || {}), ...params };
  const R = renderTimeline(tl, { seed: val('seed') !== undefined ? +val('seed') : undefined });
  const [L0, R0] = mixWithReverb(R.dL, R.dR, R.sL, R.sR, R.sr, R.seed, REVERB_DEFAULTS);
  out = [L0, R0];
  if (!has('no-master')) {
    const M = master(L0, R0, R.sr, { comp: has('comp'), kneeDb: val('knee') !== undefined ? +val('knee') : undefined, releaseMs: val('rel') !== undefined ? +val('rel') : undefined }); out = M.out; grDb = M.limiterMaxGrDb; gain = M.gainDb;
    // how much >1.5 kHz energy the master's clipper/limiter removes, and how often the gained-up signal exceeds the knee
    const g = 10 ** (gain / 20), hpPre = filt(L0.map((v, i) => 0.5 * g * (v + R0[i])), butter4('hp', 1500)), hpPost = filt(M.out[0].map((v, i) => 0.5 * (v + M.out[1][i])), butter4('hp', 1500));
    let e0 = 0, e1 = 0, over = 0, kn = 10 ** ((-1.4 - 4.5) / 20); for (let i = 0; i < hpPre.length; i++) { e0 += hpPre[i] ** 2; e1 += hpPost[i] ** 2; if (Math.abs(g * L0[i]) > kn) over++; }
    globalThis.clipInfo = { hfLossDb: +(10 * Math.log10(e1 / e0)).toFixed(2), overKneePct: +(100 * over / hpPre.length).toFixed(3) };
  }
  if (val('wav')) fs.writeFileSync(val('wav'), encodeWav24(out, R.sr));
}
const mono = out[0].map((v, i) => 0.5 * (v + out[1][i]));
const lo = (x) => filt(x, butter4('lp', 500)), mid = (x) => filt(filt(x, butter4('hp', 500)), butter4('lp', 3000)), hi = (x) => filt(x, butter4('hp', 3000));
const lstd = (w) => { const n = Math.round(w * FS), lv = []; for (let s = 0; s + n <= mono.length; s += n) { let a = 0; for (let i = s; i < s + n; i++) a += mono[i] * mono[i]; lv.push(10 * Math.log10(a / n + 1e-20)); } const m = lv.reduce((a, b) => a + b, 0) / lv.length; return +Math.sqrt(lv.reduce((a, b) => a + (b - m) ** 2, 0) / lv.length).toFixed(2); };
const rep = {
  levelStd400ms: lstd(0.4), levelStd3s: lstd(3),
  params, masterGainDb: +gain.toFixed(2), limiterMaxGrDb: +grDb.toFixed(2), clip: globalThis.clipInfo || null, ...transients(mono), ltas: ltas(mono),
  stereo: { all: corr(out[0], out[1]), low: corr(lo(out[0]), lo(out[1])), mid: corr(mid(out[0]), mid(out[1])), high: corr(hi(out[0]), hi(out[1])) },
};
console.log(JSON.stringify(rep));
