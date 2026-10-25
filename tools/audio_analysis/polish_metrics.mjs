#!/usr/bin/env node
// polish_metrics.mjs — measures the "phone / picture / loop" targets of a mastered video soundtrack
// (out/audio/polish/log.md):
//   phone    share of power above 250 Hz (unweighted, L+R), band shares, sub (< 100 Hz) swell range
//   hero     1-5 kHz short-time peaks (mono, 5 ms and 20 ms RMS) of hero events (pop strength >= 0.8, log-shift)
//            vs the loudest unscripted crackle (everything outside the windows of scripted events)
//   picture  Pearson r between ln(hrr) per video frame and the bed level per frame (master 100-1000 Hz band, and the
//            roar+breath stem rendered on its own with --stem)
//   start    400 ms momentary loudness from sample 0; loop seam: loudness of the first and the last second
//   loudness integrated LUFS, true peak (4x), limiter gain reduction (from render_audio's --json report)
//
//   node tools/audio_analysis/polish_metrics.mjs <master.wav> <timeline.json> [--report r.json] [--stem] [--json out.json]
import fs from 'node:fs';
import { Biquad } from '../../src/audio/fire-dsp.js';
import { integratedLoudness, truePeakDb, makeTruePeak, fft, kWeightingCoefs } from '../../src/audio/master-offline.js';

function readWav(file) { // PCM 16/24 or float32
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

const args = process.argv.slice(2);
const val = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && ['--report', '--json'].includes(args[i - 1])));
const [wavPath, tlPath] = pos;
if (!wavPath || !tlPath) { console.error('usage: polish_metrics.mjs <master.wav> <timeline.json> [--report r.json] [--stem] [--json out.json]'); process.exit(2); }
const { sr, ch } = readWav(wavPath);
const tl = JSON.parse(fs.readFileSync(tlPath, 'utf8'));
const [L, R] = ch.length > 1 ? ch : [ch[0], ch[0]];
const N = L.length, fps = tl.fps || 60;
const r2 = (v) => +v.toFixed(2);
const db10 = (v) => 10 * Math.log10(v + 1e-30);

function cascade(x, specs) { // specs: [type, f, Q] Butterworth sections
  let y = Float64Array.from(x);
  for (const [t, f, Q] of specs) { const q = new Biquad()[t](f, Q, sr); for (let i = 0; i < y.length; i++) y[i] = q.tick(y[i]); }
  return y;
}
const BW4 = [0.5412, 1.3066];
const band = (x, lo, hi) => cascade(x, [...(lo ? BW4.map((Q) => ['hp', lo, Q]) : []), ...(hi ? BW4.map((Q) => ['lp', hi, Q]) : [])]);
const mono = Float64Array.from(L, (v, i) => 0.5 * (v + R[i]));

// ------------------------------------------------------------------ phone translation
function psd(x) {
  const n = 8192, P = new Float64Array(n / 2 + 1), w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let s = 0; s + n <= x.length; s += n / 2) {
    for (let i = 0; i < n; i++) { re[i] = x[s + i] * w[i]; im[i] = 0; }
    fft(re, im, false);
    for (let k = 0; k <= n / 2; k++) P[k] += re[k] * re[k] + im[k] * im[k];
  }
  return { P, df: sr / n };
}
const pL = psd(L), pR = psd(R), P = pL.P.map((v, k) => v + pR.P[k]), df = pL.df;
const tot = P.reduce((a, b) => a + b, 0);
const share = (lo, hi) => { let s = 0; for (let k = 0; k < P.length; k++) { const f = k * df; if (f >= lo && f < hi) s += P[k]; } return +(100 * s / tot).toFixed(1); };
const oct = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000].map((fc) => [fc, r2(db10(P.reduce((a, v, k) => (k * df >= fc / Math.SQRT2 && k * df < fc * Math.SQRT2 ? a + v : a), 0) / tot))]);
// sub swells: 100 ms RMS of the < 100 Hz band, range p95 - p5 and its peak
const sub = band(mono, 0, 100), w100 = Math.round(0.1 * sr), subLv = [];
for (let s = 0; s + w100 <= N; s += Math.round(0.05 * sr)) { let a = 0; for (let i = s; i < s + w100; i++) a += sub[i] * sub[i]; subLv.push(db10(a / w100)); }
const srt = (a) => Float64Array.from(a).sort();
const pct = (a, p) => { const s = srt(a); return s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))]; };
const phone = {
  pctAbove250: share(250, sr / 2), pctBelow80: share(0, 80), pct80to300: share(80, 300), pctAbove300: share(300, sr / 2),
  octaveDbReTotal: Object.fromEntries(oct),
  subSwellRangeDb: r2(pct(subLv, 0.95) - pct(subLv, 0.05)), subPeakDbFS: r2(Math.max(...subLv)),
};

// ------------------------------------------------------------------ hero events vs unscripted crackles, 1-5 kHz
const ev = (tl.events || []).map((e) => ({ ...e, strength: e.strength ?? 0.7 }));
const isHero = (e) => (e.type === 'pop' && e.strength >= 0.8) || e.type === 'log-shift';
const excl = ev.filter((e) => e.type !== 'gust').map((e) => [e.t - 0.03, e.t + (e.type === 'log-shift' ? 1.6 : e.type === 'crackle-burst' ? 1.0 : 0.5 + 0.4 * e.strength)]);
const b15 = band(mono, 1000, 5000);
const cs = new Float64Array(N + 1); for (let i = 0; i < N; i++) cs[i + 1] = cs[i] + b15[i] * b15[i];
function stp(winMs) { // short-time power, hop 1 ms, window centred on the hop time
  const w = Math.round(winMs * sr / 1000), hop = Math.round(sr / 1000), t = [], v = [];
  for (let s = 0; s + w <= N; s += hop) { t.push((s + w / 2) / sr); v.push(db10((cs[s + w] - cs[s]) / w)); }
  return { t, v };
}
function heroTable(winMs) {
  const { t, v } = stp(winMs), bed = pct(v, 0.5);
  const heroes = ev.filter(isHero).map((e) => {
    let m = -Infinity, at = 0; for (let k = 0; k < t.length; k++) if (t[k] >= e.t - 0.005 && t[k] <= e.t + 0.25 && v[k] > m) { m = v[k]; at = t[k]; }
    return { t: e.t, type: e.type, strength: e.strength, peakReBed: r2(m - bed), at: +at.toFixed(3), peak: m };
  });
  const un = []; for (let k = 0; k < t.length; k++) if (!excl.some(([a, b]) => t[k] >= a && t[k] <= b)) un.push(k);
  const unV = un.map((k) => v[k]);
  // distinct unscripted peaks (local maxima >= 30 ms apart)
  const peaks = []; for (const k of un) if (v[k] >= (v[k - 1] ?? -1e9) && v[k] >= (v[k + 1] ?? -1e9)) peaks.push([v[k], t[k]]);
  peaks.sort((a, b) => b[0] - a[0]); const top = [];
  for (const p of peaks) { if (top.every((q) => Math.abs(q[1] - p[1]) > 0.03)) top.push(p); if (top.length >= 5) break; }
  const unMax = top.length ? top[0][0] : -Infinity;
  for (const h of heroes) { h.marginDb = r2(h.peak - unMax); delete h.peak; }
  return { bedDb: r2(bed), unscriptedMaxReBed: r2(unMax - bed), unscriptedP999ReBed: r2(pct(unV, 0.999) - bed), unscriptedTop: top.map(([p, tt]) => ({ t: +tt.toFixed(3), reBed: r2(p - bed) })), heroes, minMarginDb: r2(Math.min(...heroes.map((h) => h.marginDb))) };
}
const hero = { win5ms: heroTable(5), win20ms: heroTable(20) };

// ------------------------------------------------------------------ picture: ln(hrr) vs bed level per frame
function frameLevels(x) {
  const out = [], hw = Math.round(sr / fps / 2);
  for (let i = 0; i < (tl.frames || []).length; i++) {
    const c = Math.round(i * sr / fps), a = Math.max(0, c - hw), b = Math.min(N, c + hw); if (b <= a) break;
    let s = 0; for (let k = a; k < b; k++) s += x[k] * x[k]; out.push(db10(s / (b - a)));
  }
  return out;
}
function pearson(a, b) {
  const n = Math.min(a.length, b.length); let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
  let c = 0, va = 0, vb = 0; for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; c += x * y; va += x * x; vb += y * y; }
  return c / Math.sqrt(va * vb + 1e-300);
}
const rank = (a) => { const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]), r = new Array(a.length); idx.forEach(([, i], k) => (r[i] = k)); return r; };
const lnH = (tl.frames || []).map((f) => Math.log(Math.max(1e-3, f.hrr)));
function corrSet(lv) {
  const best = [-3, -2, -1, 0, 1, 2, 3].map((lag) => { const a = [], b = []; for (let i = 0; i < lv.length; i++) { const j = i + lag; if (j >= 0 && j < lv.length) { a.push(lnH[i]); b.push(lv[j]); } } return [lag, pearson(a, b)]; }).sort((p, q) => q[1] - p[1])[0];
  return { r: r2(pearson(lnH, lv) * 100) / 100, spearman: r2(pearson(rank(lnH.slice(0, lv.length)), rank(lv)) * 100) / 100, bestLagFrames: best[0], rBestLag: r2(best[1] * 100) / 100, levelStdDb: r2(Math.sqrt(lv.reduce((a, v) => a + (v - lv.reduce((p, q) => p + q, 0) / lv.length) ** 2, 0) / lv.length)) };
}
const picture = { lnHrrStd: r2(Math.sqrt(lnH.reduce((a, v) => a + (v - lnH.reduce((p, q) => p + q, 0) / lnH.length) ** 2, 0) / lnH.length) * 100) / 100, master100to1k: corrSet(frameLevels(band(mono, 100, 1000))) };
if (args.includes('--stem')) {
  const { renderTimeline } = await import('../render_audio.mjs');
  const mix = { ...(tl.params?.mix || {}) }; for (const k of ['ticks', 'crackles', 'lapping', 'hiss', 'whump', 'pops', 'logs', 'embers', 'whoosh', 'crickets']) mix[k] = -300;
  const S = renderTimeline({ ...tl, params: { ...(tl.params || {}), mix } }, { block: 128 });
  const st = Float64Array.from(S.dL, (v, i) => 0.5 * (v + S.dR[i]));
  picture.stemRoarBreath = corrSet(frameLevels(st));
}

// ------------------------------------------------------------------ start / loop seam
const [s1, s2] = kWeightingCoefs(sr), kp = new Float64Array(N + 1);
{
  const sq = new Float64Array(N);
  for (const x of [L, R]) { const f1 = new Biquad().setCoefs(s1.b[0], s1.b[1], s1.b[2], s1.a[0], s1.a[1]), f2 = new Biquad().setCoefs(1, -2, 1, s2.a[0], s2.a[1]); for (let i = 0; i < N; i++) { const y = f2.tick(f1.tick(x[i])); sq[i] += y * y; } }
  for (let i = 0; i < N; i++) kp[i + 1] = kp[i] + sq[i];
}
const lufs = (a, b) => -0.691 + db10((kp[b] - kp[a]) / (b - a));
// bed loudness of a range: K-weighted power of its 50 ms windows below their 70th percentile (pops and crackles gated out)
const bedLufs = (a, b) => { const w = Math.round(0.05 * sr), v = []; for (let s = a; s + w <= b; s += w) v.push((kp[s + w] - kp[s]) / w); v.sort((x, y) => x - y); const k = v.slice(0, Math.max(1, Math.round(0.7 * v.length))); return -0.691 + db10(k.reduce((x, y) => x + y, 0) / k.length); };
const mom = []; const w4 = Math.round(0.4 * sr); for (let s = 0; s + w4 <= N; s += Math.round(0.01 * sr)) mom.push(lufs(s, s + w4));
const S1 = Math.round(sr);
// waveform continuity across the seam: |x[0] - x[N-1]| vs the typical sample-to-sample step (RMS of first differences)
let dsum = 0; for (let i = 1; i < N; i++) dsum += (L[i] - L[i - 1]) ** 2 + (R[i] - R[i - 1]) ** 2;
const stepRms = Math.sqrt(dsum / (2 * (N - 1))), seamStep = Math.sqrt(((L[0] - L[N - 1]) ** 2 + (R[0] - R[N - 1]) ** 2) / 2);
const rms = (x, a, b) => { let s = 0; for (let i = a; i < b; i++) s += x[i] * x[i]; return db10(s / (b - a)); };
const start = {
  firstWindowLufs: r2(mom[0]), minMomentaryFirst3s: r2(Math.min(...mom.slice(0, 260))), minMomentaryAll: r2(Math.min(...mom)),
  // fade checks: first 50 ms vs 50-250 ms (before the 0.3 s hero pop), last 50 ms vs the 200 ms before, first vs last 50 ms
  fadeInDb: r2(rms(mono, 0, Math.round(0.05 * sr)) - rms(mono, Math.round(0.05 * sr), Math.round(0.25 * sr))),
  fadeOutDb: r2(rms(mono, N - Math.round(0.05 * sr), N) - rms(mono, N - Math.round(0.25 * sr), N - Math.round(0.05 * sr))),
  seam50msDb: r2(rms(mono, 0, Math.round(0.05 * sr)) - rms(mono, N - Math.round(0.05 * sr), N)),
  first1sLufs: r2(lufs(0, S1)), last1sLufs: r2(lufs(N - S1, N)), seamDiffDb: r2(lufs(N - S1, N) - lufs(0, S1)),
  seamBedDiffDb: r2(bedLufs(N - S1, N) - bedLufs(0, S1)),
  seamStepReTypical: r2(seamStep / stepRms),
};

// ------------------------------------------------------------------ loudness
const rep = val('report') ? JSON.parse(fs.readFileSync(val('report'), 'utf8')) : null;
const loud = { lufs: r2(integratedLoudness([L, R], sr)), truePeakDbTP: r2(truePeakDb([L, R], makeTruePeak(32))), limiterMaxGrDb: rep?.limiterMaxGrDb ?? null, limiter: rep?.limiter ?? null, ffmpeg: rep?.ffmpeg ?? null };

const res = { file: wavPath, phone, hero, picture, start, loudness: loud };
if (val('json')) fs.writeFileSync(val('json'), JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));
