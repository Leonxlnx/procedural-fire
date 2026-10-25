// fire-audio-dsp.mjs
// Dependency-free, deterministic procedural campfire/bonfire sound engine.
// Runs unchanged inside an AudioWorkletGlobalScope (browser) and in Node.js (offline render).
// No Math.random, no Date, no allocation in the per-sample path (pools are preallocated).
// Output of render() is bit-identical for the same seed + same event list, independent of block size.

export const CR = 32; // control-rate interval (samples). Control work happens when (absSample % CR) == 0.
const TAU = Math.PI * 2;
const dB = (x) => Math.pow(10, x / 20);

// ---------------------------------------------------------------------------------------------
// Seeded RNG (mulberry32) + seed derivation so every layer has an independent stream.
// Changing/adding one layer never reshuffles the randomness of another layer.
// ---------------------------------------------------------------------------------------------
export function deriveSeed(seed, salt) {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  for (let i = 0; i < salt.length; i++) {
    h = Math.imul(h ^ salt.charCodeAt(i), 0x85ebca6b);
    h ^= h >>> 13;
  }
  h = Math.imul(h ^ (h >>> 16), 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

export class RNG {
  constructor(seed) { this.s = seed >>> 0; this._g = 0; this._has = false; }
  next() { // uniform [0,1)
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  bi() { return this.next() * 2 - 1; }                       // uniform [-1,1), variance 1/3
  range(a, b) { return a + (b - a) * this.next(); }
  logRange(a, b) { return a * Math.pow(b / a, this.next()); } // log-uniform
  gauss() {                                                   // N(0,1), Box-Muller with cache
    if (this._has) { this._has = false; return this._g; }
    let u = this.next(); if (u < 1e-12) u = 1e-12;
    const v = this.next(), r = Math.sqrt(-2 * Math.log(u));
    this._g = r * Math.sin(TAU * v); this._has = true;
    return r * Math.cos(TAU * v);
  }
  // truncated Pareto / power law: pdf(x) ~ x^-b on [xmin, xmax], b > 1
  pareto(xmin, xmax, b) {
    const k = b - 1, r = Math.pow(xmin / xmax, k);
    return xmin * Math.pow(1 - this.next() * (1 - r), -1 / k);
  }
}

// ---------------------------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------------------------
// RBJ biquad (W3C Audio EQ Cookbook), transposed direct form II.
export class Biquad {
  constructor() { this.b0 = 1; this.b1 = 0; this.b2 = 0; this.a1 = 0; this.a2 = 0; this.z1 = 0; this.z2 = 0; }
  _set(b0, b1, b2, a0, a1, a2) {
    const i = 1 / a0; this.b0 = b0 * i; this.b1 = b1 * i; this.b2 = b2 * i; this.a1 = a1 * i; this.a2 = a2 * i; return this;
  }
  lp(f, Q, fs) { const w = TAU * f / fs, c = Math.cos(w), al = Math.sin(w) / (2 * Q); return this._set((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al); }
  hp(f, Q, fs) { const w = TAU * f / fs, c = Math.cos(w), al = Math.sin(w) / (2 * Q); return this._set((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al); }
  bp(f, Q, fs) { const w = TAU * f / fs, c = Math.cos(w), al = Math.sin(w) / (2 * Q); return this._set(al, 0, -al, 1 + al, -2 * c, 1 - al); } // 0 dB peak
  peak(f, Q, gdB, fs) {
    const A = Math.pow(10, gdB / 40), w = TAU * f / fs, c = Math.cos(w), al = Math.sin(w) / (2 * Q);
    return this._set(1 + al * A, -2 * c, 1 - al * A, 1 + al / A, -2 * c, 1 - al / A);
  }
  lowShelf(f, S, gdB, fs) {
    const A = Math.pow(10, gdB / 40), w = TAU * f / fs, c = Math.cos(w);
    const al = Math.sin(w) / 2 * Math.sqrt((A + 1 / A) * (1 / S - 1) + 2), sA = 2 * Math.sqrt(A) * al;
    return this._set(A * ((A + 1) - (A - 1) * c + sA), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sA),
      (A + 1) + (A - 1) * c + sA, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sA);
  }
  highShelf(f, S, gdB, fs) {
    const A = Math.pow(10, gdB / 40), w = TAU * f / fs, c = Math.cos(w);
    const al = Math.sin(w) / 2 * Math.sqrt((A + 1 / A) * (1 / S - 1) + 2), sA = 2 * Math.sqrt(A) * al;
    return this._set(A * ((A + 1) + (A - 1) * c + sA), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sA),
      (A + 1) - (A - 1) * c + sA, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sA);
  }
  setCoefs(b0, b1, b2, a1, a2) { this.b0 = b0; this.b1 = b1; this.b2 = b2; this.a1 = a1; this.a2 = a2; return this; }
  reset() { this.z1 = 0; this.z2 = 0; }
  tick(x) { const y = this.b0 * x + this.z1; this.z1 = this.b1 * x - this.a1 * y + this.z2; this.z2 = this.b2 * x - this.a2 * y; return y; }
}

// Simper/Cytomic trapezoidal SVF: stable and artefact-free under fast cutoff modulation.
// bp output peak gain = Q, so k*bp is the unity-peak band-pass.
export class SVF {
  constructor() { this.ic1 = 0; this.ic2 = 0; this.k = 1; this.a1 = 0; this.a2 = 0; this.a3 = 0; this.lp = 0; this.bp = 0; this.hp = 0; }
  set(f, Q, fs) {
    const g = Math.tan(Math.PI * Math.min(f, fs * 0.45) / fs);
    this.k = 1 / Q; this.a1 = 1 / (1 + g * (g + this.k)); this.a2 = g * this.a1; this.a3 = g * this.a2; return this;
  }
  reset() { this.ic1 = 0; this.ic2 = 0; }
  tick(v0) {
    const v3 = v0 - this.ic2;
    const v1 = this.a1 * this.ic1 + this.a2 * v3;
    const v2 = this.ic2 + this.a2 * this.ic1 + this.a3 * v3;
    this.ic1 = 2 * v1 - this.ic1; this.ic2 = 2 * v2 - this.ic2;
    this.lp = v2; this.bp = v1; this.hp = v0 - this.k * v1 - v2;
    return v1;
  }
}

// One-pole: y += a (x - y), a = 1 - exp(-2 pi fc / fs)
export class OnePole {
  constructor() { this.a = 1; this.y = 0; }
  set(fc, fs) { this.a = 1 - Math.exp(-TAU * fc / fs); return this; }
  lp(x) { return (this.y += this.a * (x - this.y)); }
  hp(x) { return x - this.lp(x); }
}
// std of a one-pole LP fed with unit-variance white noise: sqrt(a / (2 - a))
export const onePoleNoiseStd = (a) => Math.sqrt(a / (2 - a));

// Paul Kellet "refined" pink filter (-3 dB/oct, +-0.05 dB above ~10 Hz). Input white, output ~unit-ish.
export class Pink {
  constructor() { this.b0 = 0; this.b1 = 0; this.b2 = 0; this.b3 = 0; this.b4 = 0; this.b5 = 0; this.b6 = 0; }
  tick(w) {
    this.b0 = 0.99886 * this.b0 + w * 0.0555179; this.b1 = 0.99332 * this.b1 + w * 0.0750759;
    this.b2 = 0.969 * this.b2 + w * 0.153852; this.b3 = 0.8665 * this.b3 + w * 0.3104856;
    this.b4 = 0.55 * this.b4 + w * 0.5329522; this.b5 = -0.7616 * this.b5 - w * 0.016898;
    const p = this.b0 + this.b1 + this.b2 + this.b3 + this.b4 + this.b5 + this.b6 + w * 0.5362;
    this.b6 = w * 0.115926; return p * 0.11;
  }
}

// Narrow-band normalised Gaussian modulator running at control rate (quasi-periodic "puffing").
class NBMod {
  constructor(rng, f, Q, fcr) { this.rng = rng; this.svf = new SVF(); this.f = f; this.Q = Q; this.fcr = fcr; this.set(f, Q); this.v = 0; }
  set(f, Q) {
    this.svf.set(f, Q, this.fcr);
    // unity-peak 2nd-order BP: ENBW = (pi/2) f/Q ; variance = ENBW / (fcr/2)
    this.norm = 1 / Math.sqrt((Math.PI / 2) * (f / Q) / (this.fcr / 2));
  }
  tick() { this.svf.tick(this.rng.gauss()); return (this.v = this.svf.k * this.svf.bp * this.norm); }
}
// Slow normalised Gaussian (one-pole LP of white) at control rate.
class SlowMod {
  constructor(rng, fc, fcr) { this.rng = rng; this.op = new OnePole().set(fc, fcr); this.norm = 1 / onePoleNoiseStd(this.op.a); this.v = 0; }
  tick() { return (this.v = this.op.lp(this.rng.gauss()) * this.norm); }
}

// Equal-power pan, p in [-1, 1]
function panGains(p, out) { const th = (Math.max(-1, Math.min(1, p)) + 1) * Math.PI / 4; out[0] = Math.cos(th); out[1] = Math.sin(th); return out; }

// Piecewise curve with smoothstep interpolation: pts = [[t, v], ...]
export function curveAt(pts, t) {
  if (!pts || pts.length === 0) return null;
  if (t <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    if (t <= pts[i][0]) {
      const t0 = pts[i - 1][0], v0 = pts[i - 1][1], t1 = pts[i][0], v1 = pts[i][1];
      const x = (t - t0) / Math.max(1e-9, t1 - t0), s = x * x * (3 - 2 * x);
      return v0 + (v1 - v0) * s;
    }
  }
  return pts[pts.length - 1][1];
}

// ---------------------------------------------------------------------------------------------
// Voices (preallocated pools)
// ---------------------------------------------------------------------------------------------
// Crackle: 1..5 sub-impulses (fracture sequence) of noise with exponential decay,
// band-passed (random centre 1.5-12 kHz) + a raw broadband fraction (keeps cross-band correlation).
class CrackleVoice {
  constructor(fs) { this.fs = fs; this.active = false; this.svf = new SVF(); this.subT = new Int32Array(6); this.subA = new Float64Array(6); this.g = [0, 0]; }
  start(rng, amp, tauMs, fc, Q, raw, pan, send, nSub, subGapMs, lvl = 1) {
    this.active = true; this.t = 0; this.env = 0; this.amp = amp; this.lvl = lvl;
    this.d = Math.exp(-1000 / (tauMs * this.fs));
    this.svf.reset(); this.svf.set(fc, Q, this.fs); this.raw = raw;
    panGains(pan, this.g); this.send = send;
    this.nSub = nSub; this.si = 0; let tt = 0;
    for (let i = 0; i < nSub; i++) {
      this.subT[i] = Math.round(tt); this.subA[i] = amp * (i === 0 ? 1 : rng.range(0.25, 0.9));
      tt += rng.range(0.3, 1.0) * subGapMs * this.fs / 1000 + 1;
    }
    this.rng = rng;
  }
  tick() {
    if (this.si < this.nSub && this.t === this.subT[this.si]) { if (this.subA[this.si] > this.env) this.env = this.subA[this.si]; this.si++; }
    const x = this.rng.bi() * this.env;
    this.svf.tick(x);
    const y = (1 - this.raw) * this.svf.k * this.svf.bp * 2.2 + this.raw * x; // 2.2 ~ compensates BP energy loss
    this.env *= this.d; this.t++;
    if (this.si >= this.nSub && this.env < this.amp * 3e-4) this.active = false;
    return y;
  }
}

// Modal / damped-sinusoid bank (up to 3 modes). Used for pop cavity ring, thumps, knocks, charcoal tinkles.
class ModalVoice {
  constructor(fs) {
    this.fs = fs; this.active = false; this.n = 0; this.c1 = new Float64Array(3); this.c2 = new Float64Array(3);
    this.y1 = new Float64Array(3); this.y2 = new Float64Array(3); this.r = new Float64Array(3); this.g = [0, 0];
  }
  // freqs, taus(s), amps: arrays (length = n). glide: relative pitch change per second (negative = drop)
  start(freqs, taus, amps, n, pan, send) {
    this.active = true; this.n = n; this.t = 0; this.maxTau = 0;
    for (let i = 0; i < n; i++) {
      const th = TAU * Math.min(freqs[i], this.fs * 0.45) / this.fs, r = Math.exp(-1 / (taus[i] * this.fs));
      this.c1[i] = 2 * r * Math.cos(th); this.c2[i] = -r * r; this.r[i] = r;
      this.y2[i] = 0; this.y1[i] = amps[i] * r * Math.sin(th); // s[0]=0, s[1]=A r sin(th)  (phase 0: click-free)
      if (taus[i] > this.maxTau) this.maxTau = taus[i];
    }
    this.len = Math.ceil(this.maxTau * 9.2 * this.fs); // -80 dB
    panGains(pan, this.g); this.send = send;
  }
  tick() {
    let y = 0;
    for (let i = 0; i < this.n; i++) { const s = this.y2[i]; y += s; const nx = this.c1[i] * this.y1[i] + this.c2[i] * this.y2[i]; this.y2[i] = this.y1[i]; this.y1[i] = nx; }
    if (++this.t > this.len) this.active = false;
    return y;
  }
}

// Filtered noise burst with attack/exp-decay envelope and optional "sputter" jitter AM.
// mode: 0 = band-pass (unity peak), 1 = low-pass, 2 = high-pass
class NoiseVoice {
  constructor(fs) { this.fs = fs; this.active = false; this.svf = new SVF(); this.jit = new OnePole(); this.g = [0, 0]; }
  start(rng, amp, f, Q, mode, attackMs, tauMs, jitterDepth, jitterHz, pan, send, sweepTo) {
    this.rng = rng; this.active = true; this.t = 0; this.amp = amp; this.mode = mode;
    this.f0 = f; this.f1 = sweepTo || f; this.Q = Q; this.svf.reset(); this.svf.set(f, Q, this.fs);
    this.att = Math.max(1, Math.round(attackMs * this.fs / 1000)); this.d = Math.exp(-1000 / (tauMs * this.fs));
    this.len = this.att + Math.ceil(tauMs * 9.2 * this.fs / 1000); this.env = 0;
    this.jd = jitterDepth; this.jit.set(jitterHz || 30, this.fs); this.jit.y = 0; this.jn = jitterDepth > 0 ? 1 / onePoleNoiseStd(this.jit.a) : 0;
    panGains(pan, this.g); this.send = send;
  }
  tick() {
    if (this.f1 !== this.f0 && (this.t & 31) === 0) { const x = Math.min(1, this.t / this.len); this.svf.set(this.f0 * Math.pow(this.f1 / this.f0, x), this.Q, this.fs); }
    const e = this.t < this.att ? (this.t / this.att) * this.amp : (this.env = (this.t === this.att ? this.amp : this.env * this.d));
    const n = this.rng.bi();
    this.svf.tick(n);
    let y = this.mode === 0 ? this.svf.k * this.svf.bp : this.mode === 1 ? this.svf.lp : this.svf.hp;
    if (this.jd > 0) { const j = 1 + this.jd * this.jit.lp(this.rng.gauss()) * this.jn; y *= j > 0 ? j : 0; }
    if (++this.t > this.len) this.active = false;
    return y * e * 1.7;
  }
}

class Pool {
  constructor(n, make) { this.v = []; for (let i = 0; i < n; i++) this.v.push(make()); this.rr = 0; }
  get() { // free voice, otherwise steal round-robin (deterministic)
    for (let i = 0; i < this.v.length; i++) if (!this.v[i].active) return this.v[i];
    const v = this.v[this.rr]; this.rr = (this.rr + 1) % this.v.length; return v;
  }
}

// ---------------------------------------------------------------------------------------------
// Default parameters (all levels are pre-master; the master stage normalises to -14 LUFS)
// ---------------------------------------------------------------------------------------------
export const DEFAULTS = {
  intensity: 0.7,       // 0..1.5 (fire heat-release proxy; 1 = healthy campfire)
  wind: 0,              // 0..1 steady wind (gusts are events)
  distance: 2.0,        // listener/camera distance to fire centre (m)
  fireDiameter: 0.7,    // m -> puffing frequency f = 1.5 / sqrt(D)
  width: 0.55,          // stereo width of the fire (pan range +-width)
  crickets: 1,          // 0 = off
  // event rates at intensity 1 (per second)
  tickRate: 260,        // micro ticks per channel
  crackleRate: 16,      // crackle "immigrants" (Hawkes base rate)
  hawkesBranching: 0.45,// mean offspring per crackle (cluster size 1/(1-n))
  hawkesTauMs: 70,
  popRate: 0.12,
  emberRate: 0.15,
  // mix (dB). Continuous layers: 0 dB == -20 LKu (ungated K-weighted level) at intensity 1, distance 2 m.
  // Event layers: 0 dB == size-1 pop / log-shift peaking at about -6 dBFS (pre-master).
  mix: { ticks: -9, crackles: -4, roar: -7, breath: -8, lapping: -15, hiss: -16, pops: 2, logs: -1, embers: -6, whoosh: -9, crickets: -22 },
  send: { ticks: 0.12, crackles: 0.2, roar: 0.08, hiss: 0.1, pops: 0.4, logs: 0.3, embers: 0.35, whoosh: 0.15, crickets: 0.7 },
};

// Internal calibration (linear) so that the mix table above has the meaning documented there.
// Measured with audio-ref/calibrate.mjs; re-run it after changing any layer's DSP.
export const CAL = { ticks: 3.677, crackles: 15.53, roar: 0.1106, breath: 0.4097, lapping: 0.1996, hiss: 0.3187, crickets: 0.8694, whoosh: 0.2219, pops: 1.304, logs: 1.404, embers: 2.391 };

// ---------------------------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------------------------
export class FireAudioEngine {
  constructor({ sampleRate = 48000, seed = 1, params = {}, curves = {}, solo = null } = {}) {
    const fs = (this.fs = sampleRate);
    this.fcr = fs / CR;
    this.p = JSON.parse(JSON.stringify(DEFAULTS));
    for (const k in params) { if (k === 'mix' || k === 'send') Object.assign(this.p[k], params[k]); else this.p[k] = params[k]; }
    this.curves = curves; // { intensity: [[t,v]...], wind: [...], distance: [...] }
    this.solo = solo;     // e.g. 'roar' -> only that layer (for calibration)
    this.n = 0;           // absolute sample counter
    this.queue = [];      // scheduled actions, sorted by .n
    this.eventLog = [];   // every hero event (scripted + random) -> visual sync

    const R = (s) => new RNG(deriveSeed(seed, s));
    this.rTick = R('tick'); this.rCrk = R('crackle'); this.rPop = R('pop'); this.rRoar = R('roar'); this.rHiss = R('hiss');
    this.rMod = R('mod'); this.rLog = R('log'); this.rEmb = R('ember'); this.rCri = R('cricket'); this.rWhoosh = R('whoosh');
    this.rVoiceNoise = R('voicenoise');

    // smoothed controls
    this.I = this._target('intensity', 0); this.windS = this._target('wind', 0); this.dist = this._target('distance', 0);
    this.aSmooth = 1 - Math.exp(-1 / (0.15 * this.fcr));
    this.flare = 0; this.gust = 0; this.gustLag = 0; this.hawkes = 0; this.gusts = [];

    // modulators (control rate)
    this.fPuff = 1.5 / Math.sqrt(this.p.fireDiameter);
    this.mPuff = new NBMod(this.rMod, this.fPuff, 2.5, this.fcr);
    this.mFlut = new NBMod(this.rMod, 7.0, 1.2, this.fcr);
    this.mFlap = new NBMod(this.rMod, 14.0, 2.0, this.fcr);
    this.mSlow = new SlowMod(this.rMod, 0.12, this.fcr);
    this.mHiss = new SlowMod(this.rMod, 1.0, this.fcr);   // Farnell: lop~ 1 -> ^4
    this.mDens = new SlowMod(this.rMod, 0.6, this.fcr);   // tick density wander

    // --- ticks: tiny noise bursts -> 2 static colour filters, independent per channel
    this.tickF = [0, 1].map(() => ({ hp: new Biquad().hp(2500, 0.7, fs), b2: new Biquad().bp(5500, 0.9, fs), env: 0, d: 0, route: 0 }));

    // --- roar: correlated stereo noise -> pink -> 1-pole LP (knee) -> HP 45 Hz -> body/breath peaks -> soft sat
    this.roarCh = [0, 1].map(() => ({ pink: new Pink(), knee: new OnePole().set(250, fs), hp: new Biquad().hp(45, 0.7, fs), body: new Biquad().peak(120, 0.8, 4, fs), breath: new Biquad().peak(420, 1.0, 2, fs), ms: 1e-4 }));
    // --- breath: mid-band (300 Hz - 3 kHz) turbulent flame "whoosh", deeply flutter-modulated, independent L/R
    this.breathCh = [0, 1].map(() => ({ pink: new Pink(), svf: new SVF().set(750, 0.55, fs), hp: new Biquad().hp(220, 0.7, fs) }));
    // --- Farnell lapping: BP 30 Hz Q5 -> drive -> tanh -> HP 25 Hz x2
    this.lapBP = new Biquad().bp(30, 5, fs); this.lapHP1 = new Biquad().hp(25, 0.7, fs); this.lapHP2 = new Biquad().hp(25, 0.7, fs);
    this.lapNorm = 1 / Math.sqrt((Math.PI / 2) * (30 / 5) / (fs / 2)); // unit-RMS for Gaussian input
    // --- hiss: HP 1.2 kHz noise, high shelf -4 dB @ 9 kHz, Farnell 4th-power surges
    this.hissCh = [0, 1].map(() => ({ hp: new Biquad().hp(1200, 0.6, fs), hs: new Biquad().highShelf(9000, 0.8, -4, fs) }));
    // --- whoosh (gust band) : pink -> SVF BP swept 300..1000 Hz
    this.whooshSvf = [new SVF(), new SVF()]; this.whooshPink = [new Pink(), new Pink()];
    // --- distance tone: high shelf whose gain falls with distance
    this.distHS = [new Biquad(), new Biquad()];
    // --- ground reflection (short LP'd delay)
    this.grBuf = [new Float64Array(256), new Float64Array(256)]; this.grIdx = 0; this.grLP = [new OnePole().set(2500, fs), new OnePole().set(2500, fs)];
    this.grD = [Math.round(0.00085 * fs), Math.round(0.00105 * fs)];

    // pools
    this.crackles = new Pool(48, () => new CrackleVoice(fs));
    this.modals = new Pool(40, () => new ModalVoice(fs));
    this.noises = new Pool(16, () => new NoiseVoice(fs));
    this._pg = [0, 0];
    this._fr = new Float64Array(3); this._ta = new Float64Array(3); this._am = new Float64Array(3);

    // crickets (distant, optional)
    this.crickets = [];
    const nc = this.p.crickets ? 3 : 0;
    for (let i = 0; i < nc; i++) {
      const r = this.rCri;
      const pan = r.range(-0.85, 0.85), g = panGains(pan, [0, 0]);
      this.crickets.push({ f: r.range(4100, 4900), period: r.range(0.48, 0.66), sylN: r.next() < 0.5 ? 3 : 4, sylP: r.range(0.036, 0.044), sylD: r.range(0.014, 0.018), next: r.range(0.0, 0.6), phase: 0, amp: dB(r.range(-6, 0)), gL: g[0], gR: g[1], chirpStart: -1, chirpEnd: -1, lp: new OnePole().set(6000, fs) });
    }
    this.mixG = {}; for (const k in this.p.mix) this.mixG[k] = (this.solo && this.solo !== k) ? 0 : dB(this.p.mix[k]) * CAL[k];
    // precomputed control-rate constants
    this.kGustLag = 1 - Math.exp(-1 / (0.4 * this.fcr)); this.kFlare = Math.exp(-1 / (2.0 * this.fcr));
    this.kHawkes = Math.exp(-CR / (this.p.hawkesTauMs * 1e-3 * fs)); this._lastHs = 1e9; this._lastKnee = -1;
    this.tickBurst = 0; this.kTickBurst = Math.exp(-1 / (0.25 * this.fcr));
    // control state is initialised by the first _control() call inside render() (sample 0)
  }

  _target(name, t) { const c = this.curves[name]; const v = c ? curveAt(c, t) : this.p[name]; return v; }
  setParam(name, value) {
    this.p[name] = value;
    if (name === 'fireDiameter') { this.fPuff = 1.5 / Math.sqrt(value); this.mPuff.set(this.fPuff, 2.5); } // Cetegen & Ahmed puffing law
  }

  // Schedule an event at absolute time ev.t (seconds since engine start).
  schedule(ev) {
    const n = Math.max(this.n, Math.round(ev.t * this.fs));
    const item = { n, ev }; let i = this.queue.length;
    while (i > 0 && this.queue[i - 1].n > n) i--;
    this.queue.splice(i, 0, item);
  }

  // ------------------------------------------------------------------- control rate
  _control() {
    const t = this.n / this.fs, a = this.aSmooth, p = this.p;
    this.I += a * (this._target('intensity', t) - this.I);
    this.windS += a * (this._target('wind', t) - this.windS);
    this.dist += a * (this._target('distance', t) - this.dist);
    // gusts (sum of raised-cosine attack / exp release envelopes)
    let g = 0;
    for (let i = 0; i < this.gusts.length; i++) {
      const G = this.gusts[i], x = t - G.t0;
      if (x < 0) continue;
      if (x < G.att) g += G.s * 0.5 * (1 - Math.cos(Math.PI * x / G.att));
      else g += G.s * Math.exp(-(x - G.att) / G.rel);
    }
    this.gust = Math.min(1.5, g + this.windS);
    this.gustLag += this.kGustLag * (this.gust - this.gustLag);
    this.flare *= this.kFlare;
    this.hawkes *= this.kHawkes;
    this.tickBurst *= this.kTickBurst;

    const Ie = Math.max(0, this.I * (1 + this.flare));
    this.Ie = Ie;
    // modulators
    const up = this.mPuff.tick(), uf = this.mFlut.tick(), ufl = this.mFlap.tick(), us = this.mSlow.tick();
    const zh = this.mHiss.tick(), ud = this.mDens.tick();
    const flapDepth = 0.12 + 0.35 * this.gust;
    const sdA = Math.sqrt(0.32 * 0.32 + 0.16 * 0.16 + flapDepth * flapDepth + 0.22 * 0.22);
    let lnAmp = 0.32 * up + 0.16 * uf + flapDepth * ufl + 0.22 * us;
    lnAmp = 2.2 * sdA * Math.tanh(lnAmp / (2.2 * sdA)) - 0.5 * sdA * sdA; // soft-cap swells at ~2.2 sigma
    this.roarAmp = Math.pow(Ie, 1.5) * Math.exp(lnAmp) * (1 + 1.4 * this.gust);
    // breath: deeper, faster modulation (tongue flutter 5-12 Hz) than the roar
    const lnB = 0.25 * up + 0.4 * uf + (0.2 + 0.3 * this.gust) * ufl - 0.5 * (0.0625 + 0.16 + (0.2 + 0.3 * this.gust) ** 2);
    this.breathAmp = Math.pow(Ie, 1.3) * Math.exp(lnB) * (1 + 1.0 * this.gust);
    this.puff = up;
    // brighter when stronger / windier: knee 150..600 Hz
    const knee = Math.max(60, 150 + 220 * Math.min(1.5, Ie) + 260 * this.gust + 40 * up);
    if (Math.abs(knee - this._lastKnee) > 2) { this._lastKnee = knee; this.roarCh[0].knee.set(knee, this.fs); this.roarCh[1].knee.a = this.roarCh[0].knee.a; }
    // Farnell hiss: H * z^4 / 3 (mean H), capped
    // (plus a steady floor: pure z^4 is extremely kurtotic, real fires also have a continuous "singing" gas hiss)
    this.hissAmp = Ie * (0.4 + 0.6 * Math.min(zh * zh * zh * zh, 16) / 3) * (1 + 0.8 * this.gust);
    // event rates (per sample probabilities)
    const dens = Math.exp(0.35 * ud - 0.06) * (1 + 0.25 * up);
    this.pTick = Math.min(0.2, (p.tickRate * Math.pow(Ie, 1.2) * dens * (1 + 0.8 * this.gustLag) + this.tickBurst) / this.fs);
    this.pCrk = Math.min(0.2, (p.crackleRate * Math.pow(Ie, 1.3) * (1 + 0.2 * up) * (1 + 1.5 * this.gustLag) + this.hawkes) / this.fs);
    this.pPop = p.popRate * Math.pow(Ie, 1.5) * (1 + this.gustLag) / this.fs;
    this.pEmb = p.emberRate * Math.min(1, Ie + 0.3) / this.fs;
    // distance: 1/r level re 2 m (extended source -> soften to r^-0.8), HF loss when far
    const r = Math.max(0.4, this.dist);
    this.distGain = Math.pow(2 / r, 0.8);
    const hsDb = Math.round(20 * Math.max(-9, Math.min(4, -3.5 * Math.log2(r / 2)))) / 20; // 0.05 dB steps
    if (this._lastHs !== hsDb) { this._lastHs = hsDb; for (const b of this.distHS) b.highShelf(5000, 0.7, hsDb, this.fs); }
    // whoosh band follows gust
    this.whooshAmp = Math.pow(Math.min(1.5, this.gust), 1.5);
    if (this.whooshAmp > 1e-4) {
      const wf = 280 + 750 * Math.min(1, this.gust);
      this.whooshSvf[0].set(wf, 0.9, this.fs); this.whooshSvf[1].set(wf * 1.12, 0.9, this.fs);
    }
    this.whooshPan = Math.max(-1, Math.min(1, this.gust * 0.6 - 0.3));
  }

  // ------------------------------------------------------------------- spawners
  _spawnCrackle(sizeScale = 1, panCenter = null, panSpread = null) {
    const r = this.rCrk, p = this.p;
    // energy ~ power law (b = 1.6, cf. AE of breaking wood/bamboo b ~ 1.4-1.5) over 20 dB of energy
    // (wider ranges / flatter exponents make the crest factor unmasterable at -14 LUFS)
    const E = r.pareto(1, 100, 1.6) * sizeScale;
    const sz = Math.min(1, Math.log10(E) / 2.0);
    const amp = 0.02 * Math.pow(E, 0.4); // E^0.4 not E^0.5: 17.6 dB peak range (mic/preamp-like compression of extremes)
    const tau = 2.0 * Math.pow(E, 0.15) * r.logRange(0.5, 2.5); // ms; Farnell uses 0-29 ms squared ramps (~same energy duration)
    const fc = r.logRange(1800, 11000) / (1 + 0.6 * sz);
    const Q = r.range(0.8, 2.5);
    const raw = r.range(0.15, 0.45);
    const pc = panCenter === null ? r.range(-p.width, p.width) : panCenter + r.range(-1, 1) * (panSpread || 0.1);
    let nSub = 1; while (nSub < 5 && r.next() < 0.45) nSub++;
    const v = this.crackles.get();
    v.start(this.rVoiceNoise, amp, tau, fc, Q, raw, pc, p.send.crackles, nSub, 1.6 + 2 * sz, this.mixG.crackles);
    // Hawkes self-excitation: bigger crackles excite more offspring
    const nBranch = Math.min(0.85, p.hawkesBranching * (0.6 + 0.8 * sz));
    this.hawkes += nBranch / (p.hawkesTauMs * 1e-3);
  }

  _modal(freqs, taus, amps, n, pan, send) {
    const v = this.modals.get(); v.start(freqs, taus, amps, n, pan, send); return v;
  }

  _firePop(ev) {
    const r = this.rPop, s = ev.size !== undefined ? ev.size : r.pareto(0.25, 1.2, 2.2);
    const pan = ev.pan !== undefined ? ev.pan : r.range(-0.6, 0.6) * this.p.width / 0.55;
    const L = this.mixG.pops, send = this.p.send.pops;
    // 1) rupture click: very short N-wave (pressure pulse then rarefaction) - straight into a crackle voice with tiny tau
    const vc = this.crackles.get();
    vc.start(this.rVoiceNoise, 0.5 * L * Math.pow(s, 0.8), 0.5, 7000, 0.7, 0.85, pan, send, 1, 0);
    // 2) cavity ring (Helmholtz-like): f = c/(2pi) sqrt(A/(V L_neck)) -> 1.2-4.5 kHz, bigger pocket = lower
    const f1 = r.logRange(1200, 4500) / Math.pow(s, 0.4);
    this._fr[0] = f1; this._fr[1] = f1 * r.range(1.4, 2.3); this._fr[2] = f1 * r.range(2.6, 3.8);
    const tau1 = r.range(0.004, 0.012) * Math.sqrt(s);
    this._ta[0] = tau1; this._ta[1] = tau1 * 0.6; this._ta[2] = tau1 * 0.4;
    const a1 = 0.11 * L * s; this._am[0] = a1; this._am[1] = a1 * 0.5; this._am[2] = a1 * 0.25;
    this._modal(this._fr, this._ta, this._am, 3, pan, send);
    // 3) crack noise: broadband-ish noise burst, 2-6 ms
    this.noises.get().start(this.rVoiceNoise, 0.16 * L * s, r.range(2500, 5000) / Math.sqrt(s), 0.7, 0, 0.1, r.range(2, 6) * Math.sqrt(s), 0, 0, pan, send);
    // 4) LF body / thump for big pops (monopole volume pulse)
    if (s > 0.5) {
      this._fr[0] = r.range(90, 200); this._ta[0] = 0.02 * s; this._am[0] = 0.32 * L * s;
      this._modal(this._fr, this._ta, this._am, 1, pan * 0.5, send * 0.5);
    }
    // 4b) spark/resin sizzle: short burst of extra micro-ticks (decays ~0.25 s) for big pops
    if (s > 0.7) this.tickBurst += 700 * s;
    // 5) micro-crackle cluster (fracture propagation) over ~60 ms
    const nMicro = Math.round(3 + 9 * s * r.next());
    let tt = 0;
    for (let i = 0; i < nMicro; i++) { tt += r.range(0.002, 0.012); this.schedule({ t: this.n / this.fs + tt, type: '_microCrackle', pan, s: 0.35 * s }); }
    // 6) steam jet hiss tail ("psssh"), probability grows with size
    if (r.next() < 0.3 + 0.3 * s) {
      this.noises.get().start(this.rVoiceNoise, 0.03 * L * s, r.range(3500, 7000), 0.9, 0, 8, r.range(60, 250) * s, 0.6, 40, pan, send);
    }
    // 7) aftershock: Omori-like burst of crackle rate
    this.hawkes += 35 * s;
    this.eventLog.push({ t: this.n / this.fs, type: 'pop', size: +s.toFixed(3), pan: +pan.toFixed(3) });
  }

  _fireLogShift(ev) {
    const r = this.rLog, s = ev.size !== undefined ? ev.size : r.range(0.4, 1.0);
    const pan = ev.pan !== undefined ? ev.pan : r.range(-0.4, 0.4);
    const L = this.mixG.logs, send = this.p.send.logs, t0 = this.n / this.fs;
    // thump: 55-90 Hz damped sine + LP noise burst
    this._fr[0] = r.range(55, 90); this._ta[0] = 0.06 * s; this._am[0] = 0.3 * L * s;
    this._fr[1] = this._fr[0] * r.range(2.1, 2.9); this._ta[1] = 0.03 * s; this._am[1] = 0.12 * L * s;
    this._modal(this._fr, this._ta, this._am, 2, pan, send * 0.5);
    this.noises.get().start(this.rVoiceNoise, 0.25 * L * s, 180, 0.7, 1, 3, 45, 0, 0, pan, send * 0.5);
    // knock (wood on wood): 350-700 Hz pair
    this._fr[0] = r.range(350, 700); this._fr[1] = this._fr[0] * r.range(2.4, 2.9); this._ta[0] = 0.025; this._ta[1] = 0.015; this._am[0] = 0.08 * L * s; this._am[1] = 0.05 * L * s;
    this._modal(this._fr, this._ta, this._am, 2, pan, send);
    // debris rustle
    this.noises.get().start(this.rVoiceNoise, 0.05 * L * s, 2500, 0.6, 2, 20, 280, 0.8, 25, pan, send);
    // charcoal tinkles: cascade + bounces
    const N = Math.round(6 + 18 * s); let tt = 0.015;
    for (let i = 0; i < N; i++) {
      tt += i < N / 2 ? r.range(0.01, 0.08) : r.range(0.03, 0.18);
      this.schedule({ t: t0 + tt, type: '_tinkle', pan: pan + r.range(-0.25, 0.25), level: L * Math.sqrt(s) * r.range(0.25, 1) * (1 - 0.5 * i / N) });
    }
    // flare-up of the flames and crackle burst
    this.flare += 0.35 * s; this.hawkes += 25 * s;
    this.eventLog.push({ t: t0, type: 'logShift', size: +s.toFixed(3), pan: +pan.toFixed(3) });
  }

  _tinkle(level, pan, send) {
    const r = this.rEmb;
    const f1 = r.logRange(2200, 6500);
    this._fr[0] = f1; this._fr[1] = f1 * r.range(1.55, 2.4); this._fr[2] = f1 * r.range(2.9, 4.6);
    const tau = r.range(0.004, 0.018); this._ta[0] = tau; this._ta[1] = tau * 0.7; this._ta[2] = tau * 0.5;
    const a = 0.035 * level; this._am[0] = a; this._am[1] = a * 0.6; this._am[2] = a * 0.35;
    this._modal(this._fr, this._ta, this._am, 3, pan, send);
  }

  _fire(item) {
    const ev = item.ev;
    switch (ev.type) {
      case 'pop': this._firePop(ev); break;
      case 'logShift': this._fireLogShift(ev); break;
      case 'gust': this.gusts.push({ t0: this.n / this.fs, s: ev.strength || 0.7, att: (ev.dur || 1.5) * 0.35, rel: (ev.dur || 1.5) * 0.4 });
        this.eventLog.push({ t: this.n / this.fs, type: 'gust', strength: ev.strength || 0.7 }); break;
      case 'crackleBurst': { const k = ev.count || 8; for (let i = 0; i < k; i++) this.schedule({ t: this.n / this.fs + i * (ev.spacing || 0.03) * this.rCrk.range(0.3, 1.7), type: '_microCrackle', pan: ev.pan || 0, s: ev.size || 1 }); break; }
      case 'ember': this._tinkle(this.mixG.embers * (ev.level || 1), ev.pan || 0, this.p.send.embers); break;
      case 'flare': this.flare += ev.amount || 0.4; this.eventLog.push({ t: this.n / this.fs, type: 'flare', amount: ev.amount || 0.4 }); break;
      case '_microCrackle': this._spawnCrackle(ev.s || 1, ev.pan, 0.08); break;
      case '_tinkle': this._tinkle(ev.level, ev.pan, this.p.send.logs); break;
    }
  }

  // ------------------------------------------------------------------- audio
  // Writes n frames. dryL/dryR: direct sound. sendL/sendR: reverb send (convolve with makeOutdoorIR()).
  render(dryL, dryR, sendL, sendR, n) {
    const fs = this.fs, M = this.mixG, S = this.p.send, pg = this._pg;
    const rT = this.rTick, rR = this.rRoar, rH = this.rHiss;
    for (let i = 0; i < n; i++) {
      if (this.n % CR === 0) this._control();
      while (this.queue.length && this.queue[0].n <= this.n) this._fire(this.queue.shift());

      // stochastic spawns (Bernoulli per sample == Poisson with time-varying rate)
      if (this.rCrk.next() < this.pCrk) this._spawnCrackle();
      if (this.rPop.next() < this.pPop) this._firePop({});
      if (this.rEmb.next() < this.pEmb) { const pan = this.rEmb.range(-0.5, 0.5); this._tinkle(this.mixG.embers * this.rEmb.range(0.2, 1), pan, S.embers); }

      let L = 0, R = 0, sL = 0, sR = 0;

      // --- ticks (independent L/R streams, static colour filters)
      if (M.ticks > 0) {
        for (let c = 0; c < 2; c++) {
          let x0 = 0, x1 = 0;
          const F = this.tickF[c];
          if (rT.next() < this.pTick) {
            // tick = tiny decaying noise burst (tau 0.08-0.3 ms = 4-14 samples): more energy per peak than a
            // single-sample impulse, so the micro-texture can be loud without hammering the limiter.
            const a = dB(-17 + 3.5 * rT.gauss());
            if (a > F.env) { F.env = a; F.d = Math.exp(-1 / (rT.range(0.08, 0.3) * 1e-3 * this.fs)); F.route = rT.next(); }
          }
          if (F.env > 1e-5) {
            const x = F.env * rT.bi(); F.env *= F.d;
            if (F.route < 0.55) x0 = x; else x1 = x; // 55% broadband (HP 2.5 kHz), 45% coloured (BP 5.5 kHz, Q 0.9)
          }
          const y = (F.hp.tick(x0) + F.b2.tick(x1) * 1.5) * M.ticks;
          if (c === 0) { L += y; sL += y * S.ticks; } else { R += y; sR += y * S.ticks; }
        }
      }

      // --- roar (correlated stereo: mid + 0.45 side) + Farnell lapping
      if (M.roar > 0 || M.lapping > 0) {
        // uniform noise (var 1/3) is fine here: the heavy low-pass filtering makes it Gaussian (CLT)
        const m = rR.bi(), sd = rR.bi() * 0.45, ra = this.roarAmp;
        for (let c = 0; c < 2; c++) {
          const C = this.roarCh[c];
          let x = C.pink.tick(c === 0 ? m + sd : m - sd);
          x = C.knee.lp(x); // pink (-3 dB/oct) x one-pole (-6 dB/oct) => -9 dB/oct above knee: P(f) ~ f^-3 (Chadwick & James alpha = 3)
          x = C.breath.tick(C.body.tick(C.hp.tick(x)));
          // soft saturation at ~3x running RMS: tames LF peaks (which cost headroom but little loudness)
          // and adds low-order harmonics that keep the roar audible on phone speakers
          C.ms += 2e-5 * (x * x - C.ms);
          const cl = 3.0 * Math.sqrt(C.ms + 1e-12);
          x = cl * Math.tanh(x / cl);
          const y = x * ra * M.roar * 6.0;
          if (c === 0) { L += y; sL += y * S.roar; } else { R += y; sR += y * S.roar; }
        }
        // lapping: BP 30 Hz Q5 on noise, normalised, driven into tanh -> odd harmonics 90/150/210 Hz ("flapping")
        const lb = this.lapBP.tick(rR.bi() * 1.7320508) * this.lapNorm;
        const ly = this.lapHP2.tick(this.lapHP1.tick(Math.tanh(1.6 * lb))) * Math.pow(this.Ie, 1.2) * (1 + 1.2 * this.gust) * M.lapping;
        L += ly; R += ly;
      }

      // --- breath (mid-band flame whoosh)
      if (M.breath > 0) {
        const ba = this.breathAmp * M.breath;
        for (let c = 0; c < 2; c++) {
          const C = this.breathCh[c];
          C.svf.tick(C.pink.tick(rR.bi() * 1.7320508));
          const y = C.hp.tick(C.svf.k * C.svf.bp) * ba;
          if (c === 0) { L += y; sL += y * S.roar; } else { R += y; sR += y * S.roar; }
        }
      }

      // --- hiss (outgassing): HP noise x Farnell z^4 surges
      if (M.hiss > 0) {
        const h = this.hissAmp * M.hiss;
        for (let c = 0; c < 2; c++) {
          const C = this.hissCh[c];
          const y = C.hs.tick(C.hp.tick(rH.bi())) * h * 0.35;
          if (c === 0) { L += y; sL += y * S.hiss; } else { R += y; sR += y * S.hiss; }
        }
      }

      // --- whoosh (gust band)
      if (M.whoosh > 0 && this.whooshAmp > 1e-4) {
        const wa = this.whooshAmp * M.whoosh, w0 = this.rWhoosh.bi() * 1.7320508, w1 = this.rWhoosh.bi() * 1.7320508;
        panGains(this.whooshPan, pg);
        const y0 = this.whooshSvf[0].tick(this.whooshPink[0].tick(w0)) * this.whooshSvf[0].k * wa * 2.5;
        const y1 = this.whooshSvf[1].tick(this.whooshPink[1].tick(w1)) * this.whooshSvf[1].k * wa * 2.5;
        L += y0 * pg[0] * 1.2 + y1 * 0.3; R += y1 * pg[1] * 1.2 + y0 * 0.3; sL += y0 * S.whoosh; sR += y1 * S.whoosh;
      }

      // --- voices
      const cv = this.crackles.v;
      for (let k = 0; k < cv.length; k++) {
        const v = cv[k]; if (!v.active) continue;
        const y = v.tick() * v.lvl; L += y * v.g[0]; R += y * v.g[1]; sL += y * v.g[0] * v.send; sR += y * v.g[1] * v.send;
      }
      const mv = this.modals.v;
      for (let k = 0; k < mv.length; k++) {
        const v = mv[k]; if (!v.active) continue;
        const y = v.tick(); L += y * v.g[0]; R += y * v.g[1]; sL += y * v.g[0] * v.send; sR += y * v.g[1] * v.send;
      }
      const nv = this.noises.v;
      for (let k = 0; k < nv.length; k++) {
        const v = nv[k]; if (!v.active) continue;
        const y = v.tick(); L += y * v.g[0]; R += y * v.g[1]; sL += y * v.g[0] * v.send; sR += y * v.g[1] * v.send;
      }

      // --- crickets (distant; mostly reverb)
      if (M.crickets > 0) {
        const t = this.n / fs;
        for (let k = 0; k < this.crickets.length; k++) {
          const c = this.crickets[k];
          if (t >= c.next) {
            c.chirpStart = t; c.chirpEnd = t + c.sylN * c.sylP + 0.01;
            c.next = t + c.period * (1 + 0.06 * this.rCri.gauss());
            if (this.rCri.next() < 0.08) c.chirpStart = -1; // skipped chirp
          }
          if (c.chirpStart < 0 || t > c.chirpEnd) continue;
          const x = t - c.chirpStart, si = Math.floor(x / c.sylP);
          if (si >= c.sylN) continue;
          const xs = x - si * c.sylP;
          // syllable: 3 ms raised-cosine attack, 5 ms release, 1.5% downward glide (file-and-scraper)
          let e = 0;
          if (xs < c.sylD) { const a = 0.003, rl = 0.005; e = xs < a ? 0.5 - 0.5 * Math.cos(Math.PI * xs / a) : xs > c.sylD - rl ? 0.5 + 0.5 * Math.cos(Math.PI * (xs - (c.sylD - rl)) / rl) : 1; }
          const f = c.f * (1 + 0.015 * (1 - xs / c.sylD));
          c.phase += TAU * f / fs; if (c.phase > TAU) c.phase -= TAU;
          const y = c.lp.lp((Math.sin(c.phase) + 0.08 * Math.sin(2 * c.phase)) * e) * c.amp * M.crickets;
          L += y * c.gL * 0.4; R += y * c.gR * 0.4; sL += y * c.gL * S.crickets; sR += y * c.gR * S.crickets;
        }
      }

      // --- distance (level + HF tilt) and ground reflection on the dry path
      const dg = this.distGain;
      L = this.distHS[0].tick(L) * dg; R = this.distHS[1].tick(R) * dg;
      const gi = this.grIdx, gb = this.grBuf;
      gb[0][gi & 255] = L; gb[1][gi & 255] = R;
      L += 0.25 * this.grLP[0].lp(gb[0][(gi - this.grD[0]) & 255]);
      R += 0.25 * this.grLP[1].lp(gb[1][(gi - this.grD[1]) & 255]);
      this.grIdx = gi + 1;

      dryL[i] = L; dryR[i] = R; sendL[i] = sL; sendR[i] = sR;
      this.n++;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Procedural outdoor IR (night clearing near trees): sparse early reflections + velvet-noise tail
// with 3-band frequency-dependent decay. Deterministic. Returns [Float32Array L, Float32Array R].
// ---------------------------------------------------------------------------------------------
export function makeOutdoorIR(fs, seed, o = {}) {
  const len = Math.round((o.length || 1.6) * fs);
  const T60 = o.t60 || [0.55, 0.85, 0.6];          // low (<400 Hz), mid, high (>3 kHz) in seconds
  const out = [new Float32Array(len), new Float32Array(len)];
  for (let ch = 0; ch < 2; ch++) {
    const r = new RNG(deriveSeed(seed, 'ir' + ch));
    const er = new Float64Array(len), tail = [new Float64Array(len), new Float64Array(len), new Float64Array(len)];
    // early reflections: trees/rocks/tent at 9..140 ms, -14 dB at 10 ms falling ~9 dB per doubling of delay
    const nEr = o.nEarly || 10;
    for (let k = 0; k < nEr; k++) {
      const t = r.logRange(0.009, 0.14), i = Math.round(t * fs);
      const g = dB(-14 - 9 * Math.log2(t / 0.01) + 3 * r.gauss()) * (r.next() < 0.5 ? -1 : 1);
      if (i < len) er[i] += g;
    }
    // velvet noise: one +-1 pulse per grid cell; density ramps 600 -> 2500 pulses/s
    let t = 0.012;
    while (t < len / fs) {
      const dens = 600 + 1900 * Math.min(1, t / 0.3), Td = 1 / dens;
      const i = Math.round((t + r.next() * Td) * fs);
      if (i < len) { const s = r.next() < 0.5 ? -1 : 1; for (let b = 0; b < 3; b++) tail[b][i] = s; }
      t += Td;
    }
    // band split + per-band exponential decay + fade-in (build-up) 12..60 ms
    const lpL = new Biquad().lp(400, 0.7, fs), hpH = new Biquad().hp(3000, 0.7, fs), bpM1 = new Biquad().hp(400, 0.7, fs), bpM2 = new Biquad().lp(3000, 0.7, fs);
    const lpE = new OnePole().set(4500, fs);
    const tailGain = dB(o.tailDb !== undefined ? o.tailDb : -20);
    for (let i = 0; i < len; i++) {
      const tt = i / fs;
      const fade = tt < 0.012 ? 0 : tt < 0.06 ? 0.5 - 0.5 * Math.cos(Math.PI * (tt - 0.012) / 0.048) : 1;
      const lo = lpL.tick(tail[0][i]) * Math.exp(-6.9078 * tt / T60[0]);
      const mi = bpM2.tick(bpM1.tick(tail[1][i])) * Math.exp(-6.9078 * tt / T60[1]);
      const hi = hpH.tick(tail[2][i]) * Math.exp(-6.9078 * tt / T60[2]) * dB(-4);
      out[ch][i] = lpE.lp(er[i]) * 3 + (lo + mi + hi) * fade * tailGain;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// ITU-R BS.1770-4 loudness meter (K-weighting via libebur128-style analog prototype -> any fs)
// ---------------------------------------------------------------------------------------------
export function kWeightingCoefs(fs) {
  let f0 = 1681.974450955533, G = 3.999843853973347, Q = 0.7071752369554196;
  let K = Math.tan(Math.PI * f0 / fs);
  const Vh = Math.pow(10, G / 20), Vb = Math.pow(Vh, 0.4996667741545416);
  let a0 = 1 + K / Q + K * K;
  const s1 = { b: [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0], a: [2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0] };
  f0 = 38.13547087602444; Q = 0.5003270373238773; K = Math.tan(Math.PI * f0 / fs);
  a0 = 1 + K / Q + K * K;
  const s2 = { b: [1, -2, 1], a: [2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0] };
  return [s1, s2];
}

export function integratedLoudness(chans, fs) {
  const [s1, s2] = kWeightingCoefs(fs);
  const blk = Math.round(0.4 * fs), hop = Math.round(0.1 * fs), N = chans[0].length;
  const z = [];
  const ms = chans.map((x) => {
    const f1 = new Biquad().setCoefs(s1.b[0], s1.b[1], s1.b[2], s1.a[0], s1.a[1]), f2 = new Biquad().setCoefs(1, -2, 1, s2.a[0], s2.a[1]);
    const sq = new Float64Array(N); for (let i = 0; i < N; i++) { const y = f2.tick(f1.tick(x[i])); sq[i] = y * y; }
    const cs = new Float64Array(N + 1); for (let i = 0; i < N; i++) cs[i + 1] = cs[i] + sq[i];
    return cs;
  });
  for (let s = 0; s + blk <= N; s += hop) { let sum = 0; for (const cs of ms) sum += (cs[s + blk] - cs[s]) / blk; z.push(sum); }
  const lk = (v) => -0.691 + 10 * Math.log10(v);
  const abs = z.filter((v) => lk(v) > -70);
  if (!abs.length) return -Infinity;
  const relT = lk(abs.reduce((a, b) => a + b, 0) / abs.length) - 10;
  const g = abs.filter((v) => lk(v) > relT);
  return lk(g.reduce((a, b) => a + b, 0) / g.length);
}

// 4x oversampled true-peak estimate (polyphase windowed-sinc, 12 taps/phase like BS.1770 Annex 2)
export function makeTruePeak(taps = 12) {
  const P = 4, h = [];
  for (let ph = 0; ph < P; ph++) {
    const row = new Float64Array(taps); let sum = 0;
    for (let k = 0; k < taps; k++) {
      const x = (k - taps / 2 + 1) - ph / P; // fractional position
      const sinc = Math.abs(x) < 1e-12 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      const w = 0.5 + 0.5 * Math.cos(Math.PI * x / (taps / 2 + 0.5)); // Hann-ish window
      row[k] = sinc * w; sum += row[k];
    }
    for (let k = 0; k < taps; k++) row[k] /= sum;
    h.push(row);
  }
  // returns per-sample max |interpolated| looking at the 4 phases around each input sample
  return function peaks(x) {
    const N = x.length, out = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      let m = Math.abs(x[n]);
      for (let ph = 1; ph < P; ph++) {
        const row = h[ph]; let acc = 0;
        for (let k = 0; k < taps; k++) { const idx = n + k - taps / 2 + 1; if (idx >= 0 && idx < N) acc += row[k] * x[idx]; }
        const a = Math.abs(acc); if (a > m) m = a;
      }
      out[n] = m;
    }
    return out;
  };
}

// ---------------------------------------------------------------------------------------------
// Master: HP 22 Hz (4th-order Butterworth) -> tilt EQ -> glue compressor -> gain -> soft clip -> TP limiter
// Offline (whole-buffer) version; a streaming version is a straightforward port.
// ---------------------------------------------------------------------------------------------
export function masterChain(L, R, fs, o = {}) {
  const N = L.length;
  const ch = [L, R].map((x) => {
    const h1 = new Biquad().hp(30, 0.5412, fs), h2 = new Biquad().hp(30, 1.3066, fs); // 4th-order Butterworth
    const pk = new Biquad().peak(220, 0.9, o.phoneBodyDb !== undefined ? o.phoneBodyDb : 1.5, fs);
    const hs = new Biquad().highShelf(10000, 0.7, o.airDb !== undefined ? o.airDb : -1.5, fs);
    const y = new Float64Array(N); for (let i = 0; i < N; i++) y[i] = hs.tick(pk.tick(h2.tick(h1.tick(x[i])))); return y;
  });
  // glue compressor: stereo-linked, RMS 10 ms detector, 1.6:1, soft knee 8 dB, attack 20 ms, release 250 ms
  if (o.compThresholdDb !== undefined) {
    const T = o.compThresholdDb, ratio = 1.6, knee = 8;
    const aR = 1 - Math.exp(-1 / (0.01 * fs)), aA = 1 - Math.exp(-1 / (0.02 * fs)), aRel = 1 - Math.exp(-1 / (0.25 * fs));
    let ms = 0, gr = 0;
    for (let i = 0; i < N; i++) {
      const v = 0.5 * (ch[0][i] * ch[0][i] + ch[1][i] * ch[1][i]); ms += aR * (v - ms);
      const lv = 10 * Math.log10(ms + 1e-20); let over = lv - T, target = 0;
      if (over > knee / 2) target = over * (1 - 1 / ratio); else if (over > -knee / 2) target = (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee);
      gr += (target > gr ? aA : aRel) * (target - gr);
      const g = dB(-gr); ch[0][i] *= g; ch[1][i] *= g;
    }
  }
  const gain = dB(o.gainDb || 0);
  for (const y of ch) for (let i = 0; i < N; i++) y[i] *= gain;
  // soft clip: transparent below 'knee', tanh into the ceiling region (eats sub-ms pop peaks with no release pumping)
  const ceil = dB(o.ceilingDb !== undefined ? o.ceilingDb : -1.2), kn = ceil * dB(-4.5);
  for (const y of ch) for (let i = 0; i < N; i++) {
    const a = Math.abs(y[i]); if (a > kn) { const s = kn + (ceil * 1.12 - kn) * Math.tanh((a - kn) / (ceil * 1.12 - kn)); y[i] = Math.sign(y[i]) * s; }
  }
  // lookahead true-peak limiter: sliding min of required gain over L+1 samples, box-smoothed, then release
  const look = Math.round(0.0015 * fs), tp = makeTruePeak(16);
  const pk0 = tp(ch[0]), pk1 = tp(ch[1]);
  const req = new Float64Array(N); for (let i = 0; i < N; i++) { const m = Math.max(pk0[i], pk1[i]); req[i] = m > ceil ? ceil / m : 1; }
  const smin = new Float64Array(N); const dq = new Int32Array(N); let h = 0, t = 0;
  for (let i = 0; i < N; i++) { while (t > h && req[dq[t - 1]] >= req[i]) t--; dq[t++] = i; if (dq[h] <= i - look - 1) h++; smin[i] = req[dq[h]]; }
  const out = [new Float32Array(N), new Float32Array(N)];
  // box average of the sliding minimum over the same window guarantees g <= req at the delayed sample
  let box = look + 1; const aRel = 1 - Math.exp(-1 / (0.08 * fs)); let g = 1;
  for (let i = 0; i < N; i++) {
    box += smin[i] - (i - look - 1 >= 0 ? smin[i - look - 1] : 1);
    const gb = box / (look + 1);
    g = Math.min(gb, g + aRel * (1 - g));
    const j = i - look; // delayed signal
    const xl = j >= 0 ? ch[0][j] : 0, xr = j >= 0 ? ch[1][j] : 0;
    out[0][i] = xl * g; out[1][i] = xr * g;
  }
  // compensate lookahead delay (shift left) so events stay frame-accurate
  return [out[0].subarray(look), out[1].subarray(look)].map((a) => { const b = new Float32Array(N); b.set(a); return b; });
}
