// master-offline.js — offline (whole-buffer) helpers for the deterministic video soundtrack:
// FFT convolution reverb, BS.1770-4 loudness / EBU 3342 LRA / true peak, mastering chain, 24-bit WAV.
// Dependency-free; used by tools/render_audio.mjs and tools/audio_analysis/*.
import { Biquad, makeOutdoorIR, kWeightingCoefs, osKernel, OS_HALF } from './fire-dsp.js';
export { kWeightingCoefs };

const dB = (x) => Math.pow(10, x / 20);

// ------------------------------------------------------------------ FFT / convolution
export function fft(re, im, inv) {
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
// overlap-add FFT convolution; output length = x.length (tail truncated)
export function convolve(x, h) {
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
// dry + procedural outdoor reverb on the send bus (same IR + 15 % cross-feed as the live ConvolverNode)
export function mixWithReverb(dL, dR, sL, sR, fs, seed, o = {}) {
  const ir = makeOutdoorIR(fs, seed, o), cf = o.crossFeed ?? 0.15, N = dL.length;
  const wL = convolve(sL, ir[0]), wR = convolve(sR, ir[1]);
  const L = new Float64Array(N), R = new Float64Array(N);
  for (let i = 0; i < N; i++) { L[i] = dL[i] + wL[i] + cf * wR[i]; R[i] = dR[i] + wR[i] + cf * wL[i]; }
  return [L, R];
}

// ------------------------------------------------------------------ BS.1770-4 loudness
// cumulative sum of the K-weighted channel-summed power (index i+1 = sum of samples 0..i)
function kPowerCumsum(chans, fs) {
  const [s1, s2] = kWeightingCoefs(fs), N = chans[0].length, cs = new Float64Array(N + 1);
  const sq = new Float64Array(N);
  for (const x of chans) {
    const f1 = new Biquad().setCoefs(s1.b[0], s1.b[1], s1.b[2], s1.a[0], s1.a[1]), f2 = new Biquad().setCoefs(1, -2, 1, s2.a[0], s2.a[1]);
    for (let i = 0; i < N; i++) { const y = f2.tick(f1.tick(x[i])); sq[i] += y * y; }
  }
  for (let i = 0; i < N; i++) cs[i + 1] = cs[i] + sq[i];
  return cs;
}
const lk = (v) => -0.691 + 10 * Math.log10(v);
export function integratedLoudness(chans, fs) {
  const cs = kPowerCumsum(chans, fs), N = chans[0].length, blk = Math.round(0.4 * fs), hop = Math.round(0.1 * fs), z = [];
  for (let s = 0; s + blk <= N; s += hop) z.push((cs[s + blk] - cs[s]) / blk);
  const abs = z.filter((v) => lk(v) > -70);
  if (!abs.length) return -Infinity;
  const relT = lk(abs.reduce((a, b) => a + b, 0) / abs.length) - 10;
  const g = abs.filter((v) => lk(v) > relT);
  return lk(g.reduce((a, b) => a + b, 0) / g.length);
}
// short-term loudness (3 s window) every `hopS` seconds
export function shortTermLoudness(chans, fs, hopS = 0.1, winS = 3) {
  const cs = kPowerCumsum(chans, fs), N = chans[0].length, w = Math.round(winS * fs), h = Math.round(hopS * fs), out = [];
  for (let s = 0; s + w <= N; s += h) out.push(lk((cs[s + w] - cs[s]) / w));
  return out;
}
// EBU Tech 3342 loudness range
export function loudnessRange(chans, fs) {
  const st = shortTermLoudness(chans, fs, 0.1, 3).filter((v) => v > -70);
  if (st.length < 2) return 0;
  const pw = st.map((v) => Math.pow(10, (v + 0.691) / 10));
  const rel = lk(pw.reduce((a, b) => a + b, 0) / pw.length) - 20;
  const g = st.filter((v) => v > rel).sort((a, b) => a - b);
  const pct = (p) => g[Math.min(g.length - 1, Math.max(0, Math.round(p * (g.length - 1))))];
  return pct(0.95) - pct(0.10);
}

// ------------------------------------------------------------------ true peak (4x polyphase, Blackman-windowed sinc)
export function makeTruePeak(tapsPerPhase = 32, P = 4) {
  const h = [];
  for (let ph = 0; ph < P; ph++) {
    const row = new Float64Array(tapsPerPhase); let sum = 0;
    for (let k = 0; k < tapsPerPhase; k++) {
      const x = (k - tapsPerPhase / 2 + 1) - ph / P;
      const sinc = Math.abs(x) < 1e-12 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      const u = (x + tapsPerPhase / 2) / tapsPerPhase; // 0..1 across the window
      const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * u) + 0.08 * Math.cos(4 * Math.PI * u);
      row[k] = sinc * Math.max(0, w); sum += row[k];
    }
    for (let k = 0; k < tapsPerPhase; k++) row[k] /= sum;
    h.push(row);
  }
  const T = tapsPerPhase;
  return function peaks(x) {
    const N = x.length, out = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      let m = Math.abs(x[n]);
      for (let ph = 1; ph < P; ph++) {
        const row = h[ph]; let acc = 0; const o = n - T / 2 + 1;
        const k0 = Math.max(0, -o), k1 = Math.min(T, N - o);
        for (let k = k0; k < k1; k++) acc += row[k] * x[o + k];
        const a = Math.abs(acc); if (a > m) m = a;
      }
      out[n] = m;
    }
    return out;
  };
}
export function truePeakDb(chans, tp = makeTruePeak()) {
  let m = 0; for (const c of chans) { const p = tp(c); for (let i = 0; i < p.length; i++) if (p[i] > m) m = p[i]; }
  return 20 * Math.log10(m + 1e-30);
}

// ------------------------------------------------------------------ 4x oversampled soft clip
// The memoryless soft clip folds its harmonics above fs/2 back into the band (measured: aliasing -24..-31 dB re the
// clipped crackles at 1x). Here only the clip's *residual* is computed at 4x: x is interpolated (windowed sinc), the
// residual clip(x4) - x4 is low-passed below the base Nyquist and decimated, and added to x. Samples whose
// neighbourhood never reaches the knee are left bit-identical; the result is zero-phase (no latency).
// (the live master in fire-dsp.js runs the same algorithm, streaming, with the same kernel)
// The residual is only exact for input band-limited below the kernel's cutoff (0.46 fs): content in 0.46..0.5 fs
// would pass unclipped (measured: limiter gain reduction 9 dB instead of 6 on the demo). bandLimit() first.
const OS = 4, OS_C = OS * OS_HALF;  // +-OS_HALF base samples of support
export function bandLimit(x) { // zero-phase FIR low-pass at 0.46 fs: g[t] = 4 h[C + 4t] (the kernel's phase 0)
  const h = osKernel(), N = x.length, y = new Float64Array(N);
  for (let n = 0; n < N; n++) { let s = 0; for (let t = -OS_HALF; t <= OS_HALF; t++) { const i = n + t; if (i >= 0 && i < N) s += x[i] * 4 * h[OS_C + OS * t]; } y[n] = s; }
  return y;
}
export function softClipOS(x, kn, top) {
  const N = x.length, h = osKernel(), res = new Float64Array(N), gate = kn * 0.5, sc = top - kn;
  let busyUntil = -1;
  for (let j = 0; j < N; j++) { // only near samples within 6 dB of the knee (intersample overs of band-limited audio are far smaller)
    if (Math.abs(x[j]) > gate) busyUntil = j + 2;
    else if (j > busyUntil && !(j + 2 < N && Math.abs(x[j + 1]) > gate) && !(j + 2 < N && Math.abs(x[j + 2]) > gate)) continue;
    for (let p = 0; p < OS; p++) {
      const m = j * OS + p; // 4x time index
      let v = 0; // interpolated x at m: sum_n x[n] * OS * h[m - OS n + OS_C]
      for (let nn = j - OS_HALF; nn <= j + OS_HALF; nn++) { const k = m - OS * nn + OS_C; if (nn >= 0 && nn < N && k >= 0 && k <= 2 * OS_C) v += x[nn] * OS * h[k]; }
      const a = Math.abs(v); if (a <= kn) continue;
      const d = (v < 0 ? -1 : 1) * (kn + sc * Math.tanh((a - kn) / sc)) - v; // clip residual at 4x
      for (let nn = j - OS_HALF; nn <= j + OS_HALF; nn++) { const k = OS * nn - m + OS_C; if (nn >= 0 && nn < N && k >= 0 && k <= 2 * OS_C) res[nn] += d * h[k]; }
    }
  }
  for (let i = 0; i < N; i++) if (res[i] !== 0) x[i] += res[i];
  return x;
}

// ------------------------------------------------------------------ mastering chain
// HP o.hpHz (default 30 Hz, 4th-order Butterworth, 24 dB/oct) -> phone-body peak / air shelf -> glue comp -> gain -> soft clip
// (4x oversampled residual, o.clipOversample = false for the old 1x clip) -> TP limiter.
// o.measureFrom / o.measureTo: sample range the limiter statistics cover (render_audio's loop padding is excluded).
export function masterChain(L, R, fs, o = {}) {
  const N = L.length, hpHz = o.hpHz ?? 30;
  const ch = [L, R].map((x) => {
    const h1 = new Biquad().hp(hpHz, 0.5412, fs), h2 = new Biquad().hp(hpHz, 1.3066, fs);
    const pk = new Biquad().peak(220, 0.9, o.phoneBodyDb ?? 1.5, fs);
    const hs = new Biquad().highShelf(10000, 0.7, o.airDb ?? -1.5, fs);
    const y = new Float64Array(N); for (let i = 0; i < N; i++) y[i] = hs.tick(pk.tick(h2.tick(h1.tick(x[i])))); return y;
  });
  if (o.compThresholdDb !== undefined) { // stereo-linked RMS(10 ms), 1.6:1, knee 8 dB, attack 20 ms, release 250 ms
    const T = o.compThresholdDb, ratio = 1.6, knee = 8;
    const aR = 1 - Math.exp(-1 / (0.01 * fs)), aA = 1 - Math.exp(-1 / (0.02 * fs)), aRel = 1 - Math.exp(-1 / (0.25 * fs));
    let ms = 0, gr = 0;
    for (let i = 0; i < N; i++) {
      const v = 0.5 * (ch[0][i] * ch[0][i] + ch[1][i] * ch[1][i]); ms += aR * (v - ms);
      const lv = 10 * Math.log10(ms + 1e-20), over = lv - T; let target = 0;
      if (over > knee / 2) target = over * (1 - 1 / ratio); else if (over > -knee / 2) target = (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee);
      gr += (target > gr ? aA : aRel) * (target - gr);
      const g = dB(-gr); ch[0][i] *= g; ch[1][i] *= g;
    }
  }
  const gain = dB(o.gainDb || 0);
  for (const y of ch) for (let i = 0; i < N; i++) y[i] *= gain;
  const ceil = dB(o.ceilingDb ?? -1.4), kn = ceil * dB(-(o.kneeDb ?? 4.5)), top = ceil * 1.12;
  if (o.clipOversample === false) for (const y of ch) for (let i = 0; i < N; i++) { // 1x (aliases; kept for comparison)
    const a = Math.abs(y[i]); if (a > kn) y[i] = Math.sign(y[i]) * (kn + (top - kn) * Math.tanh((a - kn) / (top - kn)));
  }
  else for (let c = 0; c < 2; c++) {
    // o.clipPasses > 1: clip the band-limited result again. A transient driven far over the knee leaves an overshoot after
    // one pass (the clip residual's content above 0.46 fs is discarded); without a second pass the limiter has to take it
    // and ducks the whole mix for ~100 ms after every big crackle.
    let y = softClipOS(bandLimit(ch[c]), kn, top);
    for (let k = 1; k < (o.clipPasses || 1); k++) y = softClipOS(y, kn, top);
    ch[c] = y;
  }
  const look = Math.round(0.0015 * fs), tp = o.tp || makeTruePeak();
  const pk0 = tp(ch[0]), pk1 = tp(ch[1]);
  const req = new Float64Array(N); for (let i = 0; i < N; i++) { const m = Math.max(pk0[i], pk1[i]); req[i] = m > ceil ? ceil / m : 1; }
  const smin = new Float64Array(N), dq = new Int32Array(N); let h = 0, t = 0;
  for (let i = 0; i < N; i++) { while (t > h && req[dq[t - 1]] >= req[i]) t--; dq[t++] = i; if (dq[h] <= i - look - 1) h++; smin[i] = req[dq[h]]; }
  const out = [new Float64Array(N), new Float64Array(N)];
  let box = look + 1; const aRel = 1 - Math.exp(-1 / ((o.releaseMs ?? 80) * 1e-3 * fs)); let g = 1, grMax = 0, grAt = 0, n1 = 0;
  const m0 = o.measureFrom ?? 0, m1 = o.measureTo ?? N; // region the gain-reduction statistics cover (loop padding excluded)
  const grBlk = new Float64Array(Math.ceil(N / Math.round(0.05 * fs)) + 1), blk = Math.round(0.05 * fs); // max GR per 50 ms
  for (let i = 0; i < N + look; i++) {
    const si = i < N ? smin[i] : 1;
    box += si - (i - look - 1 >= 0 && i - look - 1 < N ? smin[i - look - 1] : 1);
    g = Math.min(box / (look + 1), g + aRel * (1 - g));
    const j = i - look; // delayed sample, written back at its own index (latency compensated)
    if (j >= 0) {
      out[0][j] = ch[0][j] * g; out[1][j] = ch[1][j] * g;
      if (g < 1 && j >= m0 && j < m1) { const gr = -20 * Math.log10(g); if (gr > grMax) { grMax = gr; grAt = j; } if (gr > 1) n1++; const b = Math.floor((j - m0) / blk); if (gr > grBlk[b]) grBlk[b] = gr; }
    }
  }
  out.limiterMaxGrDb = grMax;
  // where the limiter works: the time of the maximum, the share of time above 1 dB and the five worst 50 ms blocks
  const worst = Array.from(grBlk, (v, b) => [v, b]).filter((p) => p[0] > 0.5).sort((a, b) => b[0] - a[0]).slice(0, 5);
  out.limiterStats = { maxGrDb: +grMax.toFixed(2), atS: +((grAt - m0) / fs).toFixed(3), pctTimeOver1dB: +(100 * n1 / Math.max(1, m1 - m0)).toFixed(2), worst50ms: worst.map(([v, b]) => ({ t: +(b * blk / fs).toFixed(2), grDb: +v.toFixed(2) })) };
  return out;
}

// ------------------------------------------------------------------ 24-bit PCM WAV with seeded TPDF dither
export function encodeWav24(chans, sr, seed = 12345) {
  const n = chans[0].length, nc = chans.length, data = n * nc * 3, buf = new Uint8Array(44 + data), dv = new DataView(buf.buffer);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) buf[o + i] = s.charCodeAt(i); };
  str(0, 'RIFF'); dv.setUint32(4, 36 + data, true); str(8, 'WAVE'); str(12, 'fmt ');
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, nc, true); dv.setUint32(24, sr, true);
  dv.setUint32(28, sr * nc * 3, true); dv.setUint16(32, nc * 3, true); dv.setUint16(34, 24, true); str(36, 'data'); dv.setUint32(40, data, true);
  let s = seed >>> 0, o = 44;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++) {
    let v = Math.round(chans[c][i] * 8388607 + (rnd() - rnd()));
    v = v < -8388608 ? -8388608 : v > 8388607 ? 8388607 : v;
    buf[o] = v & 255; buf[o + 1] = (v >> 8) & 255; buf[o + 2] = (v >> 16) & 255; o += 3;
  }
  return buf;
}
