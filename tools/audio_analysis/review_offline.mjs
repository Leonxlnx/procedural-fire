#!/usr/bin/env node
// review_offline.mjs — adversarial checks of the offline soundtrack path (src/audio/fire-dsp.js, master-offline.js,
// tools/render_audio.mjs). Subcommands (each prints JSON):
//   determinism [--blocks 1,128,333,997,4096] [--secs 20]   engine dry hash of the demo timeline per block size
//   analyze <file.wav>                                      DC, clipping runs, L/R correlation per band, M/S, mono
//                                                           downmix loudness loss, negative-correlation windows
//   voices [--secs 20]                                      voice steals, output left at voice deactivation, queue
//   master [--secs 20]                                      soft-clip / limiter engagement and soft-clip aliasing
//   zipper                                                  spectral lines at multiples of the control rate (fs/32)
//                                                           with steady and with jumpy (per-frame step) controls
import fs from 'node:fs';
import crypto from 'node:crypto';
import { FireAudioEngine, Biquad, CR, REVERB_DEFAULTS } from '../../src/audio/fire-dsp.js';
import { mixWithReverb, integratedLoudness, fft, makeTruePeak, softClipOS } from '../../src/audio/master-offline.js';
import { makeDemoTimeline, renderTimeline } from '../render_audio.mjs';

const args = process.argv.slice(2), cmd = args[0];
const val = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const FS = 48000;
const dbs = (x) => +(10 * Math.log10(x + 1e-30)).toFixed(2);
const out = (o) => console.log(JSON.stringify(o, null, 1));

export function readWav(file) {
  const b = fs.readFileSync(file), dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12, fmt = null, data = null;
  while (o + 8 <= b.length) {
    const id = b.toString('ascii', o, o + 4), sz = dv.getUint32(o + 4, true);
    if (id === 'fmt ') fmt = { ch: dv.getUint16(o + 10, true), sr: dv.getUint32(o + 12, true), bits: dv.getUint16(o + 22, true) };
    if (id === 'data') data = { off: o + 8, size: sz };
    o += 8 + sz + (sz & 1);
  }
  const bps = fmt.bits / 8, n = Math.floor(data.size / (bps * fmt.ch)), ch = Array.from({ length: fmt.ch }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let c = 0; c < fmt.ch; c++) {
    const p = data.off + (i * fmt.ch + c) * bps;
    let v; if (fmt.bits === 24) { v = b[p] | (b[p + 1] << 8) | (b[p + 2] << 16); if (v & 0x800000) v -= 0x1000000; v /= 8388608; } else if (fmt.bits === 16) v = dv.getInt16(p, true) / 32768; else v = dv.getFloat32(p, true);
    ch[c][i] = v;
  }
  return { sr: fmt.sr, ch };
}
function filt(x, specs, fs) { let y = Float64Array.from(x); for (const [t, ...a] of specs) { const q = new Biquad()[t](...a, fs); for (let i = 0; i < y.length; i++) y[i] = q.tick(y[i]); } return y; }
function corr(a, b) { let c = 0, aa = 0, bb = 0; for (let i = 0; i < a.length; i++) { c += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; } return c / Math.sqrt(aa * bb + 1e-300); }

if (cmd === 'determinism') {
  const blocks = val('blocks', '1,128,333,997,4096').split(',').map(Number), secs = +val('secs', 20);
  const tl = makeDemoTimeline(); tl.duration = secs; tl.frames = tl.frames.slice(0, secs * tl.fps);
  const res = {};
  for (const b of blocks) { const r = renderTimeline(tl, { block: b, seed: tl.seed }); res[b] = { dryHash: r.dryHash, pops: r.log.filter((e) => e.source === 'random').map((e) => e.n).join(','), engineSec: +(r.engineMs / 1000).toFixed(1) }; }
  const h = Object.values(res).map((r) => r.dryHash + '|' + r.pops);
  out({ identical: h.every((x) => x === h[0]), res });
} else if (cmd === 'analyze') {
  const { sr, ch: [L, R] } = readWav(args[1]), N = L.length;
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  let pk = 0; for (const c of [L, R]) for (const v of c) pk = Math.max(pk, Math.abs(v));
  let atPk = 0, run = 0, maxRun = 0; for (const c of [L, R]) { run = 0; for (const v of c) { if (Math.abs(v) >= pk * 0.9999) { atPk++; run++; maxRun = Math.max(maxRun, run); } else run = 0; } }
  let dcWin = 0; const W = sr; for (const c of [L, R]) for (let s = 0; s + W <= N; s += W) { let m = 0; for (let i = s; i < s + W; i++) m += c[i]; dcWin = Math.max(dcWin, Math.abs(m / W)); }
  const bands = { '<150': [['lp', 150, 0.7071]], '150-1k': [['hp', 150, 0.7071], ['lp', 1000, 0.7071]], '1k-5k': [['hp', 1000, 0.7071], ['lp', 5000, 0.7071]], '>5k': [['hp', 5000, 0.7071]] };
  const bandCorr = {}; for (const [k, sp] of Object.entries(bands)) bandCorr[k] = +corr(filt(L, sp, sr), filt(R, sp, sr)).toFixed(3);
  let mm = 0, ss = 0; const M = new Float64Array(N); for (let i = 0; i < N; i++) { const m = 0.5 * (L[i] + R[i]), s = 0.5 * (L[i] - R[i]); M[i] = m; mm += m * m; ss += s * s; }
  const lst = integratedLoudness([L, R], sr), lmono = integratedLoudness([M, M], sr);
  // 50 ms windows with clearly negative correlation (anti-phase moments) above -50 dBFS
  const w = Math.round(0.05 * sr); let neg = 0, nw = 0, worst = 1, worstT = 0;
  for (let s = 0; s + w <= N; s += w) { const a = L.subarray(s, s + w), b = R.subarray(s, s + w); let e = 0; for (let i = 0; i < w; i++) e += a[i] * a[i] + b[i] * b[i]; if (e / (2 * w) < 1e-5) continue; nw++; const c = corr(a, b); if (c < -0.2) neg++; if (c < worst) { worst = c; worstT = s / sr; } }
  const lfL = filt(L, bands['<150'], sr), lfR = filt(R, bands['<150'], sr); let lw = 1, lwT = 0; const w2 = Math.round(0.2 * sr);
  for (let s = 0; s + w2 <= N; s += w2) { const c = corr(lfL.subarray(s, s + w2), lfR.subarray(s, s + w2)); if (c < lw) { lw = c; lwT = s / sr; } }
  out({ file: args[1], sr, seconds: +(N / sr).toFixed(3), dc: [mean(L), mean(R)].map((v) => +v.toExponential(2)), maxDc1s: +dcWin.toExponential(2), samplePeakDb: +(20 * Math.log10(pk)).toFixed(3),
    samplesAtPeak: atPk, longestRunAtPeak: maxRun, lrCorrelation: +corr(L, R).toFixed(3), bandCorrelation: bandCorr, sideToMidDb: dbs(ss / mm),
    lufsStereo: +lst.toFixed(2), lufsMonoDownmix: +lmono.toFixed(2), monoLossDb: +(lst - lmono).toFixed(2),
    windows50ms: { n: nw, negCorrBelowMinus0_2: neg, worst: +worst.toFixed(3), worstAt: +worstT.toFixed(2) }, lfCorr200msWorst: { corr: +lw.toFixed(3), at: +lwT.toFixed(2) } });
} else if (cmd === 'voices') {
  const tl = makeDemoTimeline(), secs = +val('secs', 20);
  const eng = new FireAudioEngine({ sampleRate: FS, seed: tl.seed });
  eng.setTimeline({ fps: tl.fps, frames: tl.frames });
  for (const ev of tl.events) eng.schedule(ev);
  // stress section appended after the demo: many hero events at once
  for (let k = 0; k < 3; k++) for (const type of ['ignite', 'log-shift', 'pop', 'pop', 'gust', 'whoosh', 'crackle-burst', 'pop']) eng.schedule({ t: 10.5 + k * 0.05, type, strength: 1, pan: 0 });
  const stats = {};
  for (const [name, pool] of [['crackles', eng.crackles], ['modals', eng.modals], ['noises', eng.noises]]) {
    const S = (stats[name] = { size: pool.v.length, starts: 0, steals: 0, maxActive: 0, worstEndResidueDb: -999, worstStealJumpDb: -999 });
    const g = pool.get.bind(pool);
    pool.get = () => { const v = g(); S.starts++; if (v.active) { S.steals++; v._stolen = true; } return v; };
    for (const v of pool.v) {
      const tick = v.tick.bind(v), start = v.start.bind(v); v._pk = 0; v._last = 0;
      v.start = (...a) => { if (v.active && v._stolen) { v._stealFrom = v._last; } v._pk = 0; return start(...a); };
      v.tick = () => {
        tick(); const y = name === 'crackles' ? 0.5 * (v.yL + v.yR) : v.y, yy = name === 'crackles' ? Math.max(Math.abs(v.yL), Math.abs(v.yR)) : Math.abs(y);
        if (v._stolen) { // jump caused by a steal: last output of the old voice vs the new voice's first sample
          const j = Math.abs((v._stealFrom || 0)); if (j > 0) S.worstStealJumpDb = Math.max(S.worstStealJumpDb, +(20 * Math.log10(j)).toFixed(1)); v._stolen = false;
        }
        v._pk = Math.max(v._pk, yy); v._last = name === 'crackles' ? 0.5 * (v.yL + v.yR) : y;
        if (!v.active && v._pk > 0) S.worstEndResidueDb = Math.max(S.worstEndResidueDb, +(20 * Math.log10(yy / v._pk + 1e-30)).toFixed(1));
        return y;
      };
    }
  }
  const N = secs * FS, b = [0, 1, 2, 3].map(() => new Float32Array(128));
  for (let i = 0; i < N; i += 128) {
    eng.render(b[0], b[1], b[2], b[3], 128);
    for (const [name, pool] of [['crackles', eng.crackles], ['modals', eng.modals], ['noises', eng.noises]]) { let a = 0; for (const v of pool.v) if (v.active) a++; stats[name].maxActive = Math.max(stats[name].maxActive, a); }
  }
  out({ secs, pools: stats, queueDropped: eng.queue.dropped, note: 'endResidue = |output| at deactivation re the voice peak; stealJump = |old voice output| cut off by a steal (dBFS pre-outputGain)' });
} else if (cmd === 'master') {
  const tl = makeDemoTimeline();
  const R = renderTimeline(tl, { seed: tl.seed, solo: val('solo', null) });
  const [L, Rr] = mixWithReverb(R.dL, R.dR, R.sL, R.sR, FS, R.seed, REVERB_DEFAULTS);
  const report = JSON.parse(fs.readFileSync(val('report', 'out/audio/demo_report.json'), 'utf8'));
  const gain = Math.pow(10, report.masterGainDb / 20), ceil = Math.pow(10, (report.limiterCeilingDb ?? -1.4) / 20), kn = ceil * Math.pow(10, -4.5 / 20), top = ceil * 1.12;
  // replicate masterChain's EQ to get the soft-clip input
  const eq = (x) => filt(x, [['hp', 30, 0.5412], ['hp', 30, 1.3066], ['peak', 220, 0.9, 0.5], ['highShelf', 10000, 0.7, -1.5]], FS).map((v) => v * gain);
  const X = [eq(L), eq(Rr)];
  const sc = (a) => { const m = Math.abs(a); return m > kn ? Math.sign(a) * (kn + (top - kn) * Math.tanh((m - kn) / (top - kn))) : a; };
  let above = 0, eSig = 0, eDist = 0, clusters = 0, lastAbove = -1e9, maxOver = 0;
  const hist = { '0-3': 0, '3-6': 0, '6-10': 0, '10-15': 0, '15-20': 0, '>20': 0 };
  for (const x of X) for (let i = 0; i < x.length; i++) {
    const y = sc(x[i]); eSig += x[i] * x[i]; eDist += (y - x[i]) ** 2;
    if (Math.abs(x[i]) > kn) {
      above++; if (i - lastAbove > 0.05 * FS) clusters++; lastAbove = i;
      const o = 20 * Math.log10(Math.abs(x[i]) / kn); maxOver = Math.max(maxOver, o);
      hist[o < 3 ? '0-3' : o < 6 ? '3-6' : o < 10 ? '6-10' : o < 15 ? '10-15' : o < 20 ? '15-20' : '>20']++;
    }
  }
  const overKnee = { maxDb: +maxOver.toFixed(1), histogramDbOverKnee: hist };
  // aliasing of the memoryless soft clip: 1x vs 4x oversampled (windowed-sinc interpolation, clip, ideal decimation)
  const tp = makeTruePeak(32); let tpOver = 0; for (const x of X) { const p = tp(Float64Array.from(x, sc)); for (const v of p) if (v > ceil) tpOver++; }
  function os4Clip(x) {
    const T = 64, P = 4, h = []; for (let ph = 0; ph < P; ph++) { const r = new Float64Array(T); let s = 0; for (let k = 0; k < T; k++) { const t = (k - T / 2 + 1) - ph / P, si = Math.abs(t) < 1e-12 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t), u = (t + T / 2) / T; r[k] = si * (0.42 - 0.5 * Math.cos(2 * Math.PI * u) + 0.08 * Math.cos(4 * Math.PI * u)); s += r[k]; } for (let k = 0; k < T; k++) r[k] /= s; h.push(r); }
    const N = x.length, up = new Float64Array(N * P);
    for (let n = 0; n < N; n++) for (let ph = 0; ph < P; ph++) { if (ph === 0) { up[n * P] = sc(x[n]); continue; } let a = 0; const o = n - T / 2 + 1; for (let k = 0; k < T; k++) { const j = o + k; if (j >= 0 && j < N) a += h[ph][k] * x[j]; } up[n * P + ph] = sc(a); }
    // decimate with a long low-pass (FFT brick wall at 0.45 fs of the base rate)
    let M = 1; while (M < up.length) M <<= 1; const re = new Float64Array(M), im = new Float64Array(M); re.set(up); fft(re, im, false);
    const cut = Math.floor(M * (0.5 / P) * 0.96); for (let k = cut; k <= M - cut; k++) { re[k] = 0; im[k] = 0; } fft(re, im, true);
    const y = new Float64Array(N); for (let n = 0; n < N; n++) y[n] = re[n * P]; return y;
  }
  // only around the clip clusters (whole-file 4x is slow): 0.1 s windows centred on the worst samples
  let ePk = 0, eAlias = 0, eAliasOS = 0, eDistW = 0, nWin = 0;
  for (const x of X) {
    const idx = []; for (let i = 0; i < x.length; i++) if (Math.abs(x[i]) > kn && (idx.length === 0 || i - idx[idx.length - 1] > 0.1 * FS)) idx.push(i);
    for (const c of idx.slice(0, 40)) {
      const s = Math.max(0, c - 2400), e = Math.min(x.length, c + 2400), seg = x.slice(s, e), y1 = Float64Array.from(seg, sc), y4 = os4Clip(seg);
      // compare in the band both can represent (low-pass both at 0.45 fs) so only folded components remain
      const lp = (a) => { let M2 = 1; while (M2 < a.length * 2) M2 <<= 1; const r = new Float64Array(M2), i2 = new Float64Array(M2); r.set(a); fft(r, i2, false); const cut = Math.floor(M2 * 0.45); for (let k = cut; k <= M2 - cut; k++) { r[k] = 0; i2[k] = 0; } fft(r, i2, true); return r.slice(0, a.length); };
      const a1 = lp(y1), a4 = lp(y4), aOS = lp(softClipOS(Float64Array.from(seg), kn, top));
      for (let i = 600; i < seg.length - 600; i++) { ePk += seg[i] * seg[i]; eAlias += (a1[i] - a4[i]) ** 2; eAliasOS += (aOS[i] - a4[i]) ** 2; eDistW += (y1[i] - seg[i]) ** 2; }
      nWin++;
    }
  }
  out({ masterGainDb: report.masterGainDb, kneeDbFS: +(20 * Math.log10(kn)).toFixed(2), samplesAboveKnee: above, clipClusters: clusters, overKnee, softClipDistortionReSignalDb: dbs(eDist / eSig),
    nearClips: { windows: nWin, distortionReLocalSignalDb: dbs(eDistW / ePk), aliasingReLocalSignalDb: dbs(eAlias / ePk), oversampledClipDeviationReLocalSignalDb: dbs(eAliasOS / ePk) }, softClippedSamplesWithTruePeakOverCeiling: tpOver });
} else if (cmd === 'zipper') {
  // pre-master dry mix (no reverb), crickets off; steady controls vs controls that step every video frame
  const mk = (jumpy) => {
    const fps = 60, dur = 12, frames = [];
    for (let i = 0; i < dur * fps; i++) frames.push(jumpy ? { intensity: i % 6 < 3 ? 0.4 : 1.4, hrr: i % 4 < 2 ? 0.5 : 2, dhrr: 0, camDist: i % 10 < 5 ? 0.6 : 4, camAz: (i % 8) * 0.4 } : { intensity: 1, hrr: 1, dhrr: 0, camDist: 2, camAz: 0.3 });
    return { fps, duration: dur, frames, events: [], params: { crickets: 0 } };
  };
  const res = {};
  for (const jumpy of [false, true]) {
    const r = renderTimeline(mk(jumpy), { seed: 3 });
    const x = Float64Array.from(r.dL, (v, i) => v + r.dR[i]);
    const Nf = 16384, win = new Float64Array(Nf).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / Nf)), psd = new Float64Array(Nf / 2);
    let segs = 0; for (let s = FS; s + Nf <= x.length; s += Nf / 2) { const re = new Float64Array(Nf), im = new Float64Array(Nf); for (let i = 0; i < Nf; i++) re[i] = x[s + i] * win[i]; fft(re, im, false); for (let k = 0; k < Nf / 2; k++) psd[k] += re[k] * re[k] + im[k] * im[k]; segs++; }
    const bin = FS / Nf, lines = [];
    for (let h = 1; h * FS / CR < FS / 2 - 500; h++) {
      const k = Math.round(h * FS / CR / bin), nb = []; for (let j = k - 40; j <= k + 40; j++) if (Math.abs(j - k) > 3) nb.push(psd[j]);
      nb.sort((a, b) => a - b); const med = nb[nb.length >> 1]; let pk = 0; for (let j = k - 1; j <= k + 1; j++) pk = Math.max(pk, psd[j]);
      lines.push(+(10 * Math.log10(pk / med)).toFixed(1));
    }
    res[jumpy ? 'jumpyControls' : 'steady'] = { maxLineExcessDb: Math.max(...lines), linesDb: lines.slice(0, 8), segments: segs };
  }
  out({ note: 'peak PSD at h*fs/32 re the local median (a value > ~6 dB would be an audible control-rate tone)', ...res });
} else if (cmd === 'aliastest') {
  // A sine driven 12 dB over the soft-clip knee: odd harmonics above fs/2 fold back to non-harmonic frequencies.
  // Energy at those folded frequencies re the fundamental, for the 1x clip and the 4x residual clip (softClipOS).
  const ceil = Math.pow(10, -1.4 / 20), kn = ceil * Math.pow(10, -4.5 / 20), top = ceil * 1.12, N = FS, res = {};
  for (const f0 of [3100, 7300, 9900]) {
    const x = new Float64Array(N); for (let i = 0; i < N; i++) x[i] = kn * Math.pow(10, 12 / 20) * Math.sin(2 * Math.PI * f0 * i / FS);
    const one = x.map((v) => { const a = Math.abs(v); return a > kn ? Math.sign(v) * (kn + (top - kn) * Math.tanh((a - kn) / (top - kn))) : v; });
    const os = softClipOS(Float64Array.from(x), kn, top);
    const spec = (y) => { const M = 65536, re = new Float64Array(M), im = new Float64Array(M); for (let i = 0; i < N; i++) re[i] = y[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)); fft(re, im, false); return Array.from({ length: M / 2 }, (_, k) => re[k] * re[k] + im[k] * im[k]); };
    const bandE = (P, f) => { const k = Math.round(f / FS * 65536); let s = 0; for (let j = k - 6; j <= k + 6; j++) if (j > 0 && j < P.length) s += P[j]; return s; };
    const measure = (y) => {
      const P = spec(y), e1 = bandE(P, f0); let alias = 0;
      const legit = []; for (let k = 1; k * f0 < FS / 2; k += 2) legit.push(k * f0); // in-band harmonics are not aliases
      const seen = new Set();
      for (let h = 3; h * f0 < 40 * FS; h += 2) {
        const fh = h * f0; if (fh < FS / 2) continue; let fa = fh % FS; if (fa > FS / 2) fa = FS - fa;
        if (legit.some((f) => Math.abs(fa - f) < 60)) continue; const key = Math.round(fa); if (seen.has(key)) continue; seen.add(key);
        alias += bandE(P, fa);
      }
      return +(10 * Math.log10(alias / e1 + 1e-30)).toFixed(1);
    };
    res[f0 + ' Hz'] = { aliasReFundamentalDb_1x: measure(one), aliasReFundamentalDb_os4: measure(os) };
  }
  // live master (worklet DSP): this version vs the pre-review copy in out/audio/review/orig (if present)
  const { LiveMaster } = await import('../../src/audio/fire-dsp.js');
  let Orig = null; try { Orig = (await import('../../out/audio/review/orig/fire-dsp.js')).LiveMaster; } catch { /* no copy */ }
  const live = {};
  for (const [name, LM] of [['new', LiveMaster], ['orig', Orig]]) {
    if (!LM) continue;
    for (const f0 of [3100, 7300, 9900]) {
      const m = new LM(FS, { ceilingDb: -1, makeupDb: 0, levelerThresholdLufs: 40 }), n = 2 * FS, iL = new Float32Array(n), oL = new Float32Array(n), oR = new Float32Array(n);
      const kn2 = Math.pow(10, -1 / 20) * Math.pow(10, -4.5 / 20); for (let i = 0; i < n; i++) iL[i] = kn2 * Math.pow(10, 12 / 20) * Math.sin(2 * Math.PI * f0 * i / FS);
      for (let i = 0; i < n; i += 128) m.process(iL.subarray(i, i + 128), iL.subarray(i, i + 128), oL.subarray(i, i + 128), oR.subarray(i, i + 128), 128);
      const y = Float64Array.from(oL.subarray(FS, 2 * FS)), P = (() => { const M = 65536, re = new Float64Array(M), im = new Float64Array(M); for (let i = 0; i < FS; i++) re[i] = y[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / FS)); fft(re, im, false); return Array.from({ length: M / 2 }, (_, k) => re[k] * re[k] + im[k] * im[k]); })();
      const bandE = (f) => { const k = Math.round(f / FS * 65536); let s = 0; for (let j = k - 6; j <= k + 6; j++) if (j > 0 && j < P.length) s += P[j]; return s; };
      const legit = []; for (let k = 1; k * f0 < FS / 2; k += 2) legit.push(k * f0);
      let alias = 0; const seen = new Set();
      for (let h = 3; h * f0 < 40 * FS; h += 2) { const fh = h * f0; if (fh < FS / 2) continue; let fa = fh % FS; if (fa > FS / 2) fa = FS - fa; if (legit.some((f) => Math.abs(fa - f) < 60)) continue; const key = Math.round(fa); if (seen.has(key)) continue; seen.add(key); alias += bandE(fa); }
      let pk = 0; for (const v of y) pk = Math.max(pk, Math.abs(v));
      (live[f0 + ' Hz'] ||= {})[name] = { aliasReFundamentalDb: +(10 * Math.log10(alias / bandE(f0) + 1e-30)).toFixed(1), peakDb: +(20 * Math.log10(pk)).toFixed(2) };
    }
  }
  out({ note: 'sine at +12 dB over the knee; energy at folded (non-harmonic) frequencies re the fundamental', offline: res, liveMaster: live });
} else {
  console.error('usage: review_offline.mjs determinism|analyze <wav>|voices|master|zipper|aliastest'); process.exit(2);
}
