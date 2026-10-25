// fire-dsp.js — shared deterministic DSP core for the procedural campfire audio.
//
// Runs unchanged inside an AudioWorkletGlobalScope (live) and in Node (offline render for the video).
// Rules followed everywhere in this file:
//   * no Math.random / Date: seeded per-layer RNG streams (changing one layer never reshuffles another);
//   * all control work happens on a global 32-sample grid (absSample % 32 == 0) and every event is
//     scheduled by absolute sample index, so render() output is bit-identical for any block size;
//   * no allocation in render(): voices, the event queue, swell slots and the event log are preallocated.
//
// Controls (per video frame / per update()):
//   intensity 0..1.5  overall fire size (1 = healthy campfire). 0 = no fire (silence except night ambience).
//   hrr       normalised heat-release rate (~1 at steady state). Only its fluctuation relative to its own
//             1.5 s running mean drives the roar AM/brightness, plus a mild absolute level coupling, so the
//             mapping works whether the sim normalises hrr globally or per intensity.
//   dhrr      d(hrr)/dt [1/s]. Combustion noise p ~ dQ/dt (Chadwick & James 2011): drives a band-limited
//             low-frequency "whump" layer and, on fast rises, discrete whump onsets.
//   camDist   camera-to-fire distance [m]: ~1/r level with a near-field clamp, HF shelf (closer = brighter),
//             stereo width of the fire narrows with distance; reverb send is not distance-scaled.
//   camAz     camera azimuth [rad] (src/camera.js orbit(): camera at (sin az, cos az)*d, right = (cos az, 0, -sin az)).
//             Events located at fire-local (x, z) are panned by their angle as seen from the camera.
//
// Event types: 'ignite' | 'pop' | 'crackle-burst' | 'log-shift' | 'whoosh' | 'gust' (+ 'ember', 'flare').
//   { type, strength 0..1, pan -1..1 } where pan is the fire-local lateral position (x = pan * fireRadius,
//   world x axis); the engine projects it through the camera. Optional: x, y, z [m, fire-local],
//   screenPan (-1..1, bypasses the projection), size, dur, count, id.

export const CR = 32; // control-rate interval (samples)
const TAU = Math.PI * 2;
const dB = (x) => Math.pow(10, x / 20);
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const DENORM = 1e-20; // tiny DC offset fed into filters whose input can be exactly zero (avoids denormal slowdowns)
const FTZ = 1e-30;    // decaying control values below this are flushed to 0 (they would end up subnormal)
const q3 = (v) => Math.round(v * 1000) / 1000;

// ---------------------------------------------------------------------------------------------
// Seeded RNG (mulberry32) + per-layer seed derivation
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

// Allocation-free hot loops: V8 boxes a double that is passed to or returned from a call it did not inline into a
// heap number. The per-sample code therefore reads uniforms straight from the RNG's buffer
//   if (r.i === RB) r.refill(); const u = r.buf[r.i++];      // == r.next()
// and drives filters through fields + argument-less run() methods (same arithmetic as tick(), bit-exact).
export const RB = 64; // uniforms generated per refill
export class RNG {
  constructor(seed) { this.s = seed >>> 0; this._g = 0; this._has = false; this.gv = 0; this.buf = new Float64Array(RB); this.i = RB; }
  // the next RB outputs of the generator (mulberry32), in order
  refill() {
    const b = this.buf; let s = this.s;
    for (let k = 0; k < RB; k++) {
      let t = (s = (s + 0x6d2b79f5) >>> 0);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      b[k] = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    this.s = s; this.i = 0;
  }
  next() { if (this.i === RB) this.refill(); return this.buf[this.i++]; }
  bi() { return this.next() * 2 - 1; }                       // uniform [-1,1), variance 1/3
  range(a, b) { return a + (b - a) * this.next(); }
  logRange(a, b) { return a * Math.pow(b / a, this.next()); }
  gauss() { this.gaussRun(); return this.gv; }
  // gauss() into this.gv (Box-Muller pair, second value cached)
  gaussRun() {
    if (this._has) { this._has = false; this.gv = this._g; return; }
    if (this.i === RB) this.refill(); let u = this.buf[this.i++]; if (u < 1e-12) u = 1e-12;
    if (this.i === RB) this.refill(); const v = this.buf[this.i++], r = Math.sqrt(-2 * Math.log(u));
    this._g = r * Math.sin(TAU * v); this._has = true;
    this.gv = r * Math.cos(TAU * v);
  }
  // truncated power law pdf ~ x^-b on [xmin, xmax]
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
  constructor() { this.b0 = 1; this.b1 = 0; this.b2 = 0; this.a1 = 0; this.a2 = 0; this.z1 = 0; this.z2 = 0; this.x = 0; this.y = 0; }
  _set(b0, b1, b2, a0, a1, a2) {
    const i = 1 / a0; this.b0 = b0 * i; this.b1 = b1 * i; this.b2 = b2 * i; this.a1 = a1 * i; this.a2 = a2 * i; return this;
  }
  lp(f, Q, fs) { const w = TAU * f / fs, c = Math.cos(w), al = Math.sin(w) / (2 * Q); return this._set((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al); }
  hp(f, Q, fs) { const w = TAU * f / fs, c = Math.cos(w), al = Math.sin(w) / (2 * Q); return this._set((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al); }
  bp(f, Q, fs) { const w = TAU * f / fs, c = Math.cos(w), al = Math.sin(w) / (2 * Q); return this._set(al, 0, -al, 1 + al, -2 * c, 1 - al); }
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
  run() { const x = this.x, y = this.b0 * x + this.z1; this.z1 = this.b1 * x - this.a1 * y + this.z2; this.z2 = this.b2 * x - this.a2 * y; this.y = y; } // this.y = tick(this.x), see RB
}

// Simper/Cytomic trapezoidal SVF (stable under fast modulation). k*bp = unity-peak band-pass.
export class SVF {
  constructor() { this.ic1 = 0; this.ic2 = 0; this.k = 1; this.a1 = 0; this.a2 = 0; this.a3 = 0; this.lp = 0; this.bp = 0; this.hp = 0; this.x = 0; }
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
  run() { // tick(this.x) without a double argument/return (see RB)
    const v0 = this.x, v3 = v0 - this.ic2;
    const v1 = this.a1 * this.ic1 + this.a2 * v3;
    const v2 = this.ic2 + this.a2 * this.ic1 + this.a3 * v3;
    this.ic1 = 2 * v1 - this.ic1; this.ic2 = 2 * v2 - this.ic2;
    this.lp = v2; this.bp = v1; this.hp = v0 - this.k * v1 - v2;
  }
}

export class OnePole {
  constructor() { this.a = 1; this.y = 0; }
  set(fc, fs) { this.a = 1 - Math.exp(-TAU * fc / fs); return this; }
  lp(x) { return (this.y += this.a * (x - this.y)); }
  hp(x) { return x - this.lp(x); }
}
export const onePoleNoiseStd = (a) => Math.sqrt(a / (2 - a));

// Paul Kellet "refined" pink filter. NOTE: large LF gain; every layer is calibrated by measurement (CAL).
export class Pink {
  constructor() { this.b0 = 0; this.b1 = 0; this.b2 = 0; this.b3 = 0; this.b4 = 0; this.b5 = 0; this.b6 = 0; this.w = 0; this.y = 0; }
  tick(w) { this.w = w; this.run(); return this.y; }
  run() { // this.y = filtered this.w (see RB)
    const w = this.w;
    this.b0 = 0.99886 * this.b0 + w * 0.0555179; this.b1 = 0.99332 * this.b1 + w * 0.0750759;
    this.b2 = 0.969 * this.b2 + w * 0.153852; this.b3 = 0.8665 * this.b3 + w * 0.3104856;
    this.b4 = 0.55 * this.b4 + w * 0.5329522; this.b5 = -0.7616 * this.b5 - w * 0.016898;
    const p = this.b0 + this.b1 + this.b2 + this.b3 + this.b4 + this.b5 + this.b6 + w * 0.5362;
    this.b6 = w * 0.115926; this.y = p * 0.11;
  }
}

// Narrow-band normalised Gaussian modulator at control rate (quasi-periodic puffing / flutter).
class NBMod {
  constructor(rng, f, Q, fcr) { this.rng = rng; this.svf = new SVF(); this.fcr = fcr; this.set(f, Q); this.v = 0; }
  set(f, Q) { this.svf.set(f, Q, this.fcr); this.norm = 1 / Math.sqrt((Math.PI / 2) * (f / Q) / (this.fcr / 2)); }
  tick() { const r = this.rng, f = this.svf; r.gaussRun(); f.x = r.gv; f.run(); this.v = f.k * f.bp * this.norm; } // -> this.v
}
class SlowMod {
  constructor(rng, fc, fcr) { this.rng = rng; this.op = new OnePole().set(fc, fcr); this.norm = 1 / onePoleNoiseStd(this.op.a); this.v = 0; }
  tick() { const r = this.rng, o = this.op; r.gaussRun(); o.y += o.a * (r.gv - o.y); this.v = o.y * this.norm; } // -> this.v
}

// Equal-power pan, p in [-1, 1]; writes into out[0], out[1].
function panGains(p, out) { const th = (clamp(p, -1, 1) + 1) * Math.PI / 4; out[0] = Math.cos(th); out[1] = Math.sin(th); return out; }

// ---------------------------------------------------------------------------------------------
// Voices (preallocated pools)
// ---------------------------------------------------------------------------------------------
// Crackle: 1..8 sub-impulses of decaying noise spread over up to ~30 ms (real crackles show 2-3 peaks
// within 12 dB of the first inside 30 ms), band-passed + a raw broadband fraction + an optional low "body"
// (broadband co-modulated transients are what make it read as fire, not water: McDermott et al. 2009).
// Stereo: L/R share a common noise with correlation `corr` (real crackles: HF L/R xcorr ~0.5-0.65 at
// ~zero lag), so the image is a point source that is still partly diffuse in the highs.
class CrackleVoice {
  constructor(fs) {
    this.fs = fs; this.active = false; this.svfL = new SVF(); this.svfR = new SVF(); this.svfB = new SVF();
    this.subT = new Int32Array(8); this.subA = new Float64Array(8); this.g = new Float64Array(2); this.rng = null;
    this.t = 0; this.env = 0; this.amp = 0; this.lvl = 0; this.d = 0; this.raw = 0; this.send = 0; this.nSub = 0; this.si = 0; // stable hidden class
    this.ca = 1; this.cb = 0; this.body = 0; this.yL = 0; this.yR = 0; this.env2 = 0; this.d2 = 0; this.tail = 0;
  }
  // subGapMs: maximum gap between sub-impulses (gaps are log-uniform in [subGapMin, 1] x subGapMs)
  // tail/tailMs: optional slower second decay stage (real crackles fall ~8 dB in the first ms, then ~3 dB/ms)
  start(rng, amp, tauMs, fc, Q, raw, pan, send, nSub, subGapMs, lvl, corr = 1, body = 0, fBody = 300, tail = 0, tailMs = 4, subGapMin = 0.08) {
    this.active = true; this.t = 0; this.env = 0; this.env2 = 0; this.amp = amp; this.lvl = lvl;
    this.d = Math.exp(-1000 / (tauMs * this.fs)); this.tail = tail; this.d2 = Math.exp(-1000 / (tailMs * this.fs));
    this.svfL.reset(); this.svfL.set(fc, Q, this.fs); this.svfR.reset(); this.svfR.set(fc, Q, this.fs); this.raw = raw;
    this.ca = Math.sqrt(clamp(corr, 0, 1)); this.cb = Math.sqrt(1 - this.ca * this.ca);
    this.body = body; if (body > 0) { this.svfB.reset(); this.svfB.set(fBody, 0.9, this.fs); }
    panGains(pan, this.g); this.send = send;
    nSub = Math.min(8, nSub); this.nSub = nSub; this.si = 0; let tt = 0;
    for (let i = 0; i < nSub; i++) {
      this.subT[i] = Math.round(tt); this.subA[i] = amp * (i === 0 ? 1 : rng.range(0.3, 1.0));
      tt += rng.logRange(subGapMin, 1.0) * subGapMs * this.fs / 1000 + 1;
    }
    this.rng = rng;
  }
  tick() { // -> this.yL, this.yR (no double argument/return: see RB)
    if (this.si < this.nSub && this.t === this.subT[this.si]) { const a = this.subA[this.si]; if (a > this.env) this.env = a; if (a * this.tail > this.env2) this.env2 = a * this.tail; this.si++; }
    const r = this.rng, e = this.env + this.env2;
    if (r.i === RB) r.refill(); const c = (r.buf[r.i++] * 2 - 1) * e;
    if (r.i === RB) r.refill(); const xl = this.ca * c + this.cb * e * (r.buf[r.i++] * 2 - 1);
    if (r.i === RB) r.refill(); const xr = this.ca * c + this.cb * e * (r.buf[r.i++] * 2 - 1);
    const fL = this.svfL, fR = this.svfR; fL.x = xl; fL.run(); fR.x = xr; fR.run();
    const kb = (1 - this.raw) * fL.k * 2.2;
    let yb = 0; if (this.body > 0) { const fB = this.svfB; fB.x = c; fB.run(); yb = this.body * fB.k * fB.bp; }
    this.yL = (kb * fL.bp + this.raw * xl + yb) * this.lvl;
    this.yR = (kb * fR.bp + this.raw * xr + yb) * this.lvl;
    this.env *= this.d; this.env2 *= this.d2; this.t++;
    if (this.si >= this.nSub && this.env + this.env2 < this.amp * 3e-4) this.active = false;
  }
}

// Damped-sinusoid bank (<= 3 modes): pop cavity rings, thumps, knocks, charcoal tinkles. Phase-0 start (click-free).
class ModalVoice {
  constructor(fs) {
    this.fs = fs; this.active = false; this.n = 0; this.c1 = new Float64Array(3); this.c2 = new Float64Array(3);
    this.y1 = new Float64Array(3); this.y2 = new Float64Array(3); this.g = new Float64Array(2);
    this.t = 0; this.len = 0; this.send = 0; this.y = 0;
  }
  start(freqs, taus, amps, n, pan, send) {
    this.active = true; this.n = n; this.t = 0; let maxTau = 0;
    for (let i = 0; i < n; i++) {
      const th = TAU * Math.min(freqs[i], this.fs * 0.45) / this.fs, r = Math.exp(-1 / (taus[i] * this.fs));
      this.c1[i] = 2 * r * Math.cos(th); this.c2[i] = -r * r;
      this.y2[i] = 0; this.y1[i] = amps[i] * r * Math.sin(th);
      if (taus[i] > maxTau) maxTau = taus[i];
    }
    this.len = Math.ceil(maxTau * 9.2 * this.fs); // -80 dB
    panGains(pan, this.g); this.send = send;
  }
  tick() { // -> this.y
    let y = 0;
    for (let i = 0; i < this.n; i++) { const s = this.y2[i]; y += s; const nx = this.c1[i] * this.y1[i] + this.c2[i] * this.y2[i]; this.y2[i] = this.y1[i]; this.y1[i] = nx; }
    if (++this.t > this.len) this.active = false;
    this.y = y;
  }
}

// Filtered noise burst: linear attack, exponential decay, optional log sweep and "sputter" AM.
// mode: 0 = band-pass (unity peak), 1 = low-pass, 2 = high-pass
class NoiseVoice {
  constructor(fs) {
    this.fs = fs; this.active = false; this.svf = new SVF(); this.jit = new OnePole(); this.g = new Float64Array(2); this.rng = null;
    this.t = 0; this.amp = 0; this.mode = 0; this.f0 = 1000; this.f1 = 1000; this.Q = 1; this.att = 1; this.d = 0; this.len = 0; this.env = 0; this.jd = 0; this.jn = 0; this.send = 0; this.y = 0;
  }
  start(rng, amp, f, Q, mode, attackMs, tauMs, jitterDepth, jitterHz, pan, send, sweepTo) {
    this.rng = rng; this.active = true; this.t = 0; this.amp = amp; this.mode = mode;
    this.f0 = f; this.f1 = sweepTo > 0 ? sweepTo : f; this.Q = Q; this.svf.reset(); this.svf.set(f, Q, this.fs);
    this.att = Math.max(1, Math.round(attackMs * this.fs / 1000)); this.d = Math.exp(-1000 / (tauMs * this.fs));
    this.len = this.att + Math.ceil(tauMs * 9.2 * this.fs / 1000); this.env = 0;
    this.jd = jitterDepth; this.jit.set(jitterHz > 0 ? jitterHz : 30, this.fs); this.jit.y = 0; this.jn = jitterDepth > 0 ? 1 / onePoleNoiseStd(this.jit.a) : 0;
    panGains(pan, this.g); this.send = send;
  }
  tick() { // -> this.y (no double argument/return: see RB)
    if (this.f1 !== this.f0 && (this.t & 31) === 0) { const x = Math.min(1, this.t / this.len); this.svf.set(this.f0 * Math.pow(this.f1 / this.f0, x), this.Q, this.fs); }
    const e = this.t < this.att ? (this.t / this.att) * this.amp : (this.env = (this.t === this.att ? this.amp : this.env * this.d));
    const r = this.rng, sv = this.svf;
    if (r.i === RB) r.refill(); sv.x = r.buf[r.i++] * 2 - 1; sv.run();
    let y = this.mode === 0 ? sv.k * sv.bp : this.mode === 1 ? sv.lp : sv.hp;
    if (this.jd > 0) { const J = this.jit; r.gaussRun(); J.y += J.a * (r.gv - J.y); const j = 1 + this.jd * J.y * this.jn; y *= j > 0 ? j : 0; }
    if (++this.t > this.len) this.active = false;
    this.y = y * e * 1.7;
  }
}

class Pool {
  constructor(n, make) { this.v = []; for (let i = 0; i < n; i++) this.v.push(make()); this.rr = 0; }
  get() { // free voice, otherwise steal round-robin (deterministic)
    const v = this.v;
    for (let i = 0; i < v.length; i++) if (!v[i].active) return v[i];
    const s = v[this.rr]; this.rr = (this.rr + 1) % v.length; return s;
  }
}

// ---------------------------------------------------------------------------------------------
// Preallocated event queue (sorted, next-to-fire at the end) and event log ring
// ---------------------------------------------------------------------------------------------
const C_POP = 1, C_LOG = 2, C_GUST = 3, C_BURST = 4, C_IGNITE = 5, C_WHOOSH = 6, C_EMBER = 7, C_FLARE = 8;
const C_MICRO = 20, C_TINKLE = 21, C_MATCHHEAD = 22, C_HAWKES = 23, C_THUMP = 24, C_DUCK = 25;
const TYPE_CODES = {
  pop: C_POP, 'log-shift': C_LOG, logShift: C_LOG, log_shift: C_LOG, logshift: C_LOG, gust: C_GUST,
  'crackle-burst': C_BURST, crackleBurst: C_BURST, crackle_burst: C_BURST, ignite: C_IGNITE, whoosh: C_WHOOSH,
  ember: C_EMBER, flare: C_FLARE,
};
const CODE_NAMES = { [C_POP]: 'pop', [C_LOG]: 'log-shift', [C_GUST]: 'gust', [C_BURST]: 'crackle-burst', [C_IGNITE]: 'ignite', [C_WHOOSH]: 'whoosh', [C_EMBER]: 'ember', [C_FLARE]: 'flare' };
export const EVENT_TYPES = ['ignite', 'pop', 'crackle-burst', 'log-shift', 'whoosh', 'gust', 'ember', 'flare'];
const IGNITE_SURGE = [0.42, 18, 0.7, 21, 1.05, 15]; // (time s, Hawkes rate boost) pairs after an ignite

class EvRec {
  constructor() {
    this.n = 0; this.code = 0; this.s = 0; this.strength = 0; this.hasSize = false;
    this.x = 0; this.y = 0; this.z = 0; this.spMode = 0; this.sp = 0; // spMode 1: fixed stereo pan sp
    this.level = 0; this.dur = 0; this.count = 0; this.spacing = 0; this.src = 0; this.id = -1;
  }
}
class EventQueue {
  constructor(cap) {
    this.cap = cap; this.q = new Array(cap).fill(null); this.len = 0;
    this.free = new Array(cap); for (let i = 0; i < cap; i++) this.free[i] = new EvRec(); this.nFree = cap;
    this.dropped = 0;
  }
  alloc() { if (this.nFree === 0) { this.dropped++; return null; } return this.free[--this.nFree]; }
  insert(r) { // FIFO for equal times: the new record goes below every record firing at or before it
    let i = this.len; const q = this.q;
    while (i > 0 && q[i - 1].n <= r.n) { q[i] = q[i - 1]; i--; }
    q[i] = r; this.len++;
  }
  peekN() { return this.len ? this.q[this.len - 1].n : Infinity; }
  pop() { const r = this.q[--this.len]; this.q[this.len] = null; return r; }
  release(r) { this.free[this.nFree++] = r; }
}

class LogRec {
  constructor() { this.seq = 0; this.n = 0; this.t = 0; this.type = ''; this.code = 0; this.strength = 0; this.size = 0; this.pan = 0; this.x = 0; this.y = 0; this.z = 0; this.stereoPan = 0; this.source = ''; this.random = 0; this.id = -1; }
}
// Packed event record (EventLog.pack / decodeEvents): the worklet posts these numbers in a reused Float64Array.
export const EVF = 14; // seq, n, t, code, strength, size, pan, x, y, z, stereoPan, random(0|1), id, contextTime
export class EventLog {
  constructor(cap = 256) { this.cap = cap; this.r = []; for (let i = 0; i < cap; i++) this.r.push(new LogRec()); this.w = 0; this.packedTo = 0; }
  _next() { const rec = this.r[this.w % this.cap]; rec.seq = this.w++; return rec; }
  // Copy records written since `since` into plain objects (allocates; call outside the audio hot path).
  read(since, out) {
    let s = Math.max(since, this.w - this.cap);
    for (; s < this.w; s++) {
      const r = this.r[s % this.cap];
      out.push({ seq: r.seq, n: r.n, t: r.t, type: r.type, strength: r.strength, size: r.size, pan: r.pan, x: r.x, y: r.y, z: r.z, stereoPan: r.stereoPan, source: r.source, id: r.id });
    }
    return this.w;
  }
  // Allocation-free variant for the audio thread: up to `max` records since `since` into out (EVF numbers each,
  // contextTime left 0). Returns the count; this.packedTo = the next unread sequence number.
  pack(since, out, max) {
    let s = Math.max(since, this.w - this.cap), k = 0;
    for (; s < this.w && k < max; s++, k++) {
      const r = this.r[s % this.cap], o = k * EVF;
      out[o] = r.seq; out[o + 1] = r.n; out[o + 2] = r.t; out[o + 3] = r.code; out[o + 4] = r.strength; out[o + 5] = r.size; out[o + 6] = r.pan;
      out[o + 7] = r.x; out[o + 8] = r.y; out[o + 9] = r.z; out[o + 10] = r.stereoPan; out[o + 11] = r.random; out[o + 12] = r.id; out[o + 13] = 0;
    }
    this.packedTo = s; return k;
  }
}
// main thread: packed records -> event objects { seq, n, t, type, strength, size, pan, x, y, z, stereoPan, source, id, contextTime }
export function decodeEvents(d, count) {
  const out = [];
  for (let k = 0; k < count; k++) {
    const o = k * EVF;
    out.push({ seq: d[o], n: d[o + 1], t: d[o + 2], type: CODE_NAMES[d[o + 3]] || 'unknown', strength: d[o + 4], size: d[o + 5], pan: d[o + 6], x: d[o + 7], y: d[o + 8], z: d[o + 9], stereoPan: d[o + 10], source: d[o + 11] ? 'random' : 'scripted', id: d[o + 12], contextTime: d[o + 13] });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------------------
export const DEFAULTS = {
  // controls (initial values; update()/timeline override)
  intensity: 0.8, hrr: 1, dhrr: 0, camDist: 2.2, camAz: 0, wind: 0,
  // fire geometry / spatial
  fireDiameter: 0.7,   // m -> puffing frequency 1.5/sqrt(D) (Cetegen & Ahmed); keep in sync with the visual flicker
  fireRadius: 0.38,    // m, fire bed radius (src/scene.js logs span about +-0.4 m)
  stereoWidth: 1.2,    // stereo pan = angle / 45 deg * stereoWidth
  refDist: 2.0,        // m: distance at which the mix table is calibrated (gain 1)
  nearField: 0.6,      // m: near-field clamp of the 1/r law (extended source)
  crickets: 1,
  // event rates at intensity 1 (per second)
  tickRate: 150, crackleRate: 10, hawkesBranching: 0.3, hawkesTauMs: 70, popRate: 0.12, emberRate: 0.15,
  // crackle texture (tuned against 6 real campfire recordings: tools/audio_analysis/, out/audio/tuning_log.md)
  crkEmax: 3000,       // energy range of the AE power law (35 dB of amplitude at crkExp 0.5)
  crkB: 1.6,           // P(E) ~ E^-b
  crkExp: 0.5,         // amplitude ~ E^crkExp
  crkAmp0: 0.0045,     // amplitude of the smallest crackle (E = 1)
  crkTauMs: 0.9,       // decay time constant of one sub-impulse (x E^0.12 x [0.5, 2])
  crkFcLo: 1500, crkFcHi: 8000,   // band-pass colour range
  crkRawLo: 0.3, crkRawHi: 0.6,   // raw broadband fraction
  crkSubP: 0.35,       // probability of each further sub-impulse (mean count 1/(1-p))
  crkSubGapMs: 10, crkSubGapMin: 0.08, // sub-impulse gaps log-uniform in [crkSubGapMin, 1] x crkSubGapMs
  crkBody: 1.0,      // low-frequency body (160-800 Hz) of a crackle, relative
  crkTail: 0.25, crkTailMs: 3.5, // second, slower decay stage: relative amplitude and time constant
  crkShape: 2,         // gamma shape of the crackle renewal process (1 = Poisson)
  crkCorr: 0.3,        // L/R correlation of a crackle's noise
  // flame AM (log-normal): quasi-periodic puffing (broad, Q 0.7), tongue flutter, wind flapping, slow swelling
  // (slowDepth at slowHz); hrr AM depth. crkPuff/crkFlut: crackle-rate coupling to puffing/flutter.
  puffDepth: 0.28, puffQ: 0.7, flutDepth: 0.25, crkPuff: 0.4, crkFlut: 0, flapDepth: 0.12, slowDepth: 0.4, slowHz: 0.2, hrrAmDepth: 0.5,
  // hiss (the steady high-frequency floor between crackles): Farnell surge depth, slow activity and flutter AM
  hissSurge: 0.15, hissAct: 0.5, hissFlut: 0.3,
  actHz: 0.35, actDepth: 0.4, actSizeExp: 1.0, actRoar: 0.2, // slow activity: crackle rate x act, sizes x act^exp, roar AM
  roarBodyDb: -2.0,    // roar body peak at 120 Hz (a cut since the phone/social polish: the 63-315 Hz bands were 4-5 dB
                       // above the real recordings)
  roarKnee0: 350,      // roar spectral knee at intensity 0 (+220 Hz x intensity, + wind/flare/heat terms)
  roarSat: 3.0,        // roar soft saturation at roarSat x running RMS (lower = less peaky bed, more mastering headroom)
  roarLpHz: 1200,      // roar low-pass on top of the f^-3 knee: keeps the bed out of the crackle band
  roarHpHz: 70,        // roar high-pass (real fires: little energy below ~50 Hz; was 45 before the phone polish)
  roarHarm: 0.4,       // harmonics of the roar's low band (< 150 Hz, rectified = even products) band-passed to 150-600 Hz,
                       // relative to the low band's RMS: the bed's low swell stays audible on small speakers
  // picture coupling: the bed (roar, breath) follows the light output hrr (normalised to ~1 at the steady fire) as an absolute
  // ln-gain picDepth x ln(hrr), smoothed over picSmoothMs and soft-limited to +-picLim. The internal slow swell and puffing AM
  // fade (to picIntKeep of their depth) and the relative hrr term hands over to it as the picture's own fast flicker (std of
  // ln(hrr / its 1.5 s mean) clipped to +-0.2, ~1.2 s window) rises from picStdLo to picStdHi: a flickering picture drives the bed level, a
  // steady one leaves it to the internal model. picDepth 0 = off.
  picDepth: 2.3, picDown: 0.6, picLim: 0.6, picSmoothMs: 40, picStdLo: 0.05, picStdHi: 0.09, picIntKeep: 0.2, // picDown: depth
                       // below hrr 1 (a dimming picture lowers the bed less than a flare raises it: no holes in the bed)
  distDepth: 1,        // exponent on the distance gain (1 = the ~1/r law below; < 1 = smaller level jumps at camera cuts)
  // hero events (every log-shift, pops with strength >= heroMin): a 1.5-4 kHz crack + ring (mix.hero), the random crackle
  // layer ducked by duckDb for duckPreMs before to duckHoldMs after the event, and a steam "psssh" tail on the log-shift
  heroMin: 0.8, duckDb: -4, duckPreMs: 50, duckHoldMs: 250, logPsssh: 1,
  // soft cap of the largest random crackles: energies above crkCapE grow with exponent crkCapRatio (1 = no cap)
  crkCapE: 200, crkCapRatio: 0.3,
  breathHz: 750,       // breath band centre
  breathLpHz: 1100,    // breath low-pass: keeps the continuous bed out of the crackle band
  breathSide: 0.5,     // breath layer stereo: side/mid ratio (0 = mono, 1 = ~uncorrelated)
  // overall output trim (the statistical tuning lowered the bed and the crackles relative to the mix table; this
  // restores the pre-tuning output level, so the live master and the offline master see the same levels as before)
  outputGainDb: 9.5,
  // sim coupling
  simCoupling: 0.6,    // 0 = internal puffing AM only, 1 = roar AM fully from hrr fluctuation
  whumpThresh: 1.6,    // relative dhrr (1/s) that triggers a discrete whump onset
  whumpRearm: 0.6,
  whumpHpHz: 55,       // high-pass of the continuous whump layer (was 35 before the phone polish)
  // mix (dB, before outputGainDb). Continuous layers: 0 dB == -20 LKu at intensity 1, 2 m. Event layers: size-1
  // event peaks near -6 dBFS. Re-measured after the tuning (CAL below); the values keep the tuned balance.
  // (phone polish: lapping -26 -> -32 and whump -14 -> -17 dB, the sub layers; hero = the hero crack/ring, not calibrated)
  mix: { ticks: -38.17, crackles: -12.13, roar: -16.46, breath: -15.83, lapping: -32, hiss: -31.24, whump: -17, pops: -2.66, logs: -11.17, embers: -10, whoosh: -13.24, crickets: -38, hero: 6.5 },
  send: { ticks: 0.12, crackles: 0.12, roar: 0.08, hiss: 0.1, whump: 0.06, pops: 0.4, logs: 0.4, embers: 0.35, whoosh: 0.15, crickets: 0.7, hero: 0.4 },
};

// Linear calibration so the mix table has the meaning above (tools/audio_analysis/calibrate.mjs).
export const CAL = { ticks: 5.086, crackles: 39.57, roar: 0.1342, breath: 0.7272, lapping: 0.1996, hiss: 0.3145, crickets: 0.8872, whoosh: 0.2646, whump: 0.1481, pops: 1.236, logs: 2.784, embers: 2.391 };

// ---------------------------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------------------------
export class FireAudioEngine {
  constructor({ sampleRate = 48000, seed = 1, params = {}, solo = null } = {}) {
    const fs = (this.fs = sampleRate);
    this.fcr = fs / CR;
    this.seed = seed >>> 0;
    this.p = JSON.parse(JSON.stringify(DEFAULTS));
    for (const k in params) { if (k === 'mix' || k === 'send') Object.assign(this.p[k], params[k]); else this.p[k] = params[k]; }
    this.solo = solo;
    this.n = 0;
    this.queue = new EventQueue(1024);
    this.log = new EventLog(512);
    this.ctl = { intensity: this.p.intensity, hrr: this.p.hrr, dhrr: this.p.dhrr, camDist: this.p.camDist, camAz: this.p.camAz, wind: this.p.wind };
    this.tl = null; // offline frame timeline (setTimeline)

    const R = (s) => new RNG(deriveSeed(this.seed, s));
    this.rTick = R('tick'); this.rCrk = R('crackle'); this.rPop = R('pop'); this.rRoar = R('roar'); this.rHiss = R('hiss');
    this.rMod = R('mod'); this.rLog = R('log'); this.rEmb = R('ember'); this.rCri = R('cricket'); this.rWhoosh = R('whoosh');
    this.rVoiceNoise = R('voicenoise'); this.rIgn = R('ignite'); this.rWhump = R('whump'); this.rPos = R('position'); this.rAct = R('activity');
    this.rHero = R('hero');

    // smoothing coefficients (control rate)
    const k = (tau) => 1 - Math.exp(-1 / (tau * this.fcr));
    this.kI = k(0.15); this.kH = k(0.03); this.kHs = k(1.5); this.kCam = k(0.08); this.kWind = k(0.15);
    this.kGustLag = k(0.4); this.kFlare = Math.exp(-1 / (2.0 * this.fcr));
    this.kHawkes = Math.exp(-CR / (this.p.hawkesTauMs * 1e-3 * fs));
    this.kTickBurst = Math.exp(-1 / (0.25 * this.fcr));
    this._init = false;
    this.I = 0; this.H = 1; this.Hs = 1; this.D = 0; this.dist = 2; this.az = 0; this.windS = 0;
    this.cAz = 1; this.sAz = 0; this.Ie = 0; this.fl = 0; this.gust = 0; this.gustLag = 0;
    this.flare = 0; this.hawkes = 0; this.tickBurst = 0;
    this.whumpArmed = true; this.lastWhumpN = -1e12;
    // picture coupling (see DEFAULTS.picDepth) and the hero-event duck of the crackle layer
    this.kP = k(Math.max(0.001, this.p.picSmoothMs * 1e-3)); this.kPv = k(1.2);
    this.Hp = 1; this.pfm = 0; this.pfv = 0; this.wp = 0;
    this.duckUntil = -1; this.duckG = 1; this.duckT = 1; this.kDuckA = k(0.02); this.kDuckR = k(0.12);
    this.swells = []; for (let i = 0; i < 8; i++) this.swells.push({ active: false, kind: 0, t0: 0, s: 0, att: 1, rel: 1 });

    // modulators (control rate)
    this.fPuff = 1.5 / Math.sqrt(this.p.fireDiameter);
    this.mPuff = new NBMod(this.rMod, this.fPuff, this.p.puffQ, this.fcr);
    this.mFlut = new NBMod(this.rMod, 7.0, 1.2, this.fcr);
    this.mFlap = new NBMod(this.rMod, 14.0, 2.0, this.fcr);
    this.mSlow = new SlowMod(this.rMod, this.p.slowHz, this.fcr); // slow swelling / dying down of the whole fire
    this.mHiss = new SlowMod(this.rMod, 1.0, this.fcr);   // Farnell: lop~ 1 -> ^4
    this.mDens = new SlowMod(this.rMod, 0.6, this.fcr);   // tick density wander
    this.mAct = new SlowMod(this.rAct, this.p.actHz, this.fcr); // slow fire activity (crackle rate/size, roar)
    this.act = 1; this.uAct = 0; this.crkAcc = 0; this.crkThr = 0; this.crkThr = this._gammaThr();

    // ticks: tiny noise bursts -> 2 static colour filters, independent per channel
    this.tickF = [0, 1].map(() => ({ hp: new Biquad().hp(2500, 0.7, fs), b2: new Biquad().bp(5500, 0.9, fs), env: 0, d: 0, route: 0 }));
    // roar: correlated stereo noise -> pink -> 1-pole LP knee (=> f^-3 above knee) -> HP 45 -> body peaks -> soft sat
    this.roarCh = [0, 1].map(() => ({ pink: new Pink(), knee: new OnePole().set(250, fs), hp: new Biquad().hp(this.p.roarHpHz, 0.7, fs), body: new Biquad().peak(120, 0.8, this.p.roarBodyDb, fs), breath: new Biquad().peak(420, 1.0, 2, fs), lp: new Biquad().lp(this.p.roarLpHz, 0.7, fs), ms: 1e-4,
      // harmonics of the low band: LP 150 -> |x| (unit RMS) -> band-pass 150-600 Hz (DEFAULTS.roarHarm)
      hlp: new Biquad().lp(150, 0.7, fs), hhp: new Biquad().hp(150, 0.7, fs), hlp2: new Biquad().lp(600, 0.7, fs), hms: 1e-8 }));
    // breath: mid-band flame whoosh, deeply flutter-modulated, independent L/R
    this.breathCh = [0, 1].map(() => ({ pink: new Pink(), svf: new SVF().set(this.p.breathHz, 0.55, fs), hp: new Biquad().hp(220, 0.7, fs), lp: new Biquad().lp(this.p.breathLpHz, 0.6, fs) }));
    // Farnell lapping
    this.lapBP = new Biquad().bp(30, 5, fs); this.lapHP1 = new Biquad().hp(25, 0.7, fs); this.lapHP2 = new Biquad().hp(25, 0.7, fs);
    this.lapNorm = 1 / Math.sqrt((Math.PI / 2) * (30 / 5) / (fs / 2));
    // hiss
    this.hissCh = [0, 1].map(() => ({ hp: new Biquad().hp(1200, 0.6, fs), hs: new Biquad().highShelf(9000, 0.8, -4, fs) }));
    // whoosh (gust / flame swell band)
    this.whooshSvf = [new SVF(), new SVF()]; this.whooshPink = [new Pink(), new Pink()];
    this.whooshLP = [new Biquad().lp(2200, 0.7, fs), new Biquad().lp(2200, 0.7, fs)]; // keep the swell out of the crackle band
    // whump (combustion noise ~ dQ/dt): pink -> LP 150 -> HP 35 -> LP 300, soft saturation, centre
    this.whPink = new Pink(); this.whLP = new OnePole().set(150, fs); this.whHP = new Biquad().hp(this.p.whumpHpHz, 0.7, fs); this.whLP2 = new Biquad().lp(300, 0.7, fs); this.whMs = 1e-4;
    // distance tone + ground reflection
    this.distHS = [new Biquad(), new Biquad()];
    this.grBuf = [new Float64Array(256), new Float64Array(256)]; this.grIdx = 0; this.grLP = [new OnePole().set(2500, fs), new OnePole().set(2500, fs)];
    this.grD = [Math.round(0.00085 * fs), Math.round(0.00105 * fs)];

    // pools
    this.crackles = new Pool(64, () => new CrackleVoice(fs));
    this.modals = new Pool(48, () => new ModalVoice(fs));
    this.noises = new Pool(24, () => new NoiseVoice(fs));
    this._pg = new Float64Array(2);
    this._fr = new Float64Array(3); this._ta = new Float64Array(3); this._am = new Float64Array(3);

    // crickets: distant, fixed world azimuths; panned relative to the camera azimuth
    this.crickets = [];
    if (this.p.crickets) {
      for (let i = 0; i < 3; i++) {
        const r = this.rCri;
        this.crickets.push({ phi: r.range(-Math.PI, Math.PI), f: r.range(4100, 4900), period: r.range(0.48, 0.66), sylN: r.next() < 0.5 ? 3 : 4, sylP: r.range(0.036, 0.044), sylD: r.range(0.014, 0.018), next: r.range(0.0, 0.6), phase: 0, amp: dB(r.range(-6, 0)), gL: 0.7, gR: 0.7, chirpStart: -1, chirpEnd: -1, lp: new OnePole().set(6000, fs) });
      }
    }
    this._mixGains();
    this._lastHs = 1e9; this._lastKnee = -1;
    this.distGain = 1; this.bedW = 1;
    // control outputs (assigned here so the engine keeps one hidden class)
    this.roarAmp = 0; this.breathAmp = 0; this.hissAmp = 0; this.lapAmp = 0; this.whumpAmp = 0; this.whooshAmp = 0; this.whooshPan = 0; this.puff = 0;
    this.pTick = 0; this.pCrk = 0; this.pPop = 0; this.pEmb = 0; this._px = 0; this._pz = 0;
  }

  _mixGains() {
    this.mixG = {};
    for (const k in this.p.mix) this.mixG[k] = (this.solo && this.solo !== k) ? 0 : dB(this.p.mix[k]) * (CAL[k] ?? 1);
    this.outG = dB(this.p.outputGainDb || 0);
  }

  // ------------------------------------------------------------------- public API
  setParam(name, value) {
    if (name === 'mix' || name === 'send') { Object.assign(this.p[name], value); this._mixGains(); return; }
    this.p[name] = value;
    if (name === 'fireDiameter' || name === 'puffQ') { this.fPuff = 1.5 / Math.sqrt(this.p.fireDiameter); this.mPuff.set(this.fPuff, this.p.puffQ); }
    if (name === 'hawkesTauMs') this.kHawkes = Math.exp(-CR / (value * 1e-3 * this.fs));
    if (name === 'outputGainDb') this._mixGains();
  }
  // Live control targets (smoothed inside the engine at control rate).
  setControls(c) {
    const t = this.ctl;
    if (typeof c.intensity === 'number' && isFinite(c.intensity)) t.intensity = clamp(c.intensity, 0, 2);
    if (typeof c.hrr === 'number' && isFinite(c.hrr)) t.hrr = clamp(c.hrr, 0, 5);
    if (typeof c.dhrr === 'number' && isFinite(c.dhrr)) t.dhrr = clamp(c.dhrr, -50, 50);
    if (typeof c.camDist === 'number' && isFinite(c.camDist)) t.camDist = clamp(c.camDist, 0.05, 1000);
    if (typeof c.camAz === 'number' && isFinite(c.camAz)) t.camAz = c.camAz;
    if (typeof c.wind === 'number' && isFinite(c.wind)) t.wind = clamp(c.wind, 0, 1.5);
  }
  // Offline: per-frame controls { fps, frames: [{intensity, hrr, dhrr, camDist, camAz}] }, linearly interpolated.
  setTimeline({ fps, frames }) {
    const N = frames.length, f = (name, def) => { const a = new Float64Array(Math.max(1, N)); for (let i = 0; i < N; i++) { const v = frames[i][name]; a[i] = typeof v === 'number' && isFinite(v) ? v : (i > 0 ? a[i - 1] : def); } if (!N) a[0] = def; return a; };
    const tl = { fps, N: Math.max(1, N), I: f('intensity', this.p.intensity), H: f('hrr', this.p.hrr), D: f('dhrr', 0), Cd: f('camDist', this.p.camDist), Az: f('camAz', this.p.camAz), W: f('wind', 0) };
    // dhrr missing everywhere -> central difference of hrr
    if (!frames.some((fr) => typeof fr.dhrr === 'number')) for (let i = 0; i < N; i++) tl.D[i] = (tl.H[Math.min(N - 1, i + 1)] - tl.H[Math.max(0, i - 1)]) * fps / ((i > 0 && i < N - 1) ? 2 : 1);
    for (let i = 1; i < N; i++) { let d = tl.Az[i] - tl.Az[i - 1]; d -= TAU * Math.round(d / TAU); tl.Az[i] = tl.Az[i - 1] + d; } // unwrap
    this.tl = tl;
  }
  _tl(arr, t) {
    const tl = this.tl, x = t * tl.fps, i = Math.floor(x);
    if (i <= 0) return x <= 0 ? arr[0] : arr[0] + (arr[Math.min(1, tl.N - 1)] - arr[0]) * x;
    if (i >= tl.N - 1) return arr[tl.N - 1];
    return arr[i] + (arr[i + 1] - arr[i]) * (x - i);
  }

  // Schedule an event at engine time ev.t (seconds since engine sample 0); without t: next sample.
  schedule(ev) {
    const n = typeof ev.t === 'number' && isFinite(ev.t) ? Math.max(this.n, Math.round(ev.t * this.fs)) : this.n;
    return this.scheduleAt(ev, n);
  }
  scheduleAt(ev, n) {
    const code = TYPE_CODES[ev.type]; if (!code) return false;
    const r = this.queue.alloc(); if (!r) return false;
    this._parse(ev, code, r); r.n = Math.max(this.n, Math.floor(n) || 0); r.src = 0; // (n | 0 wrapped after 2^31 samples)
    this.queue.insert(r);
    // hero events duck the random crackle layer from duckPreMs before them to duckHoldMs after them
    if (this.p.duckDb < 0 && ((code === C_POP && r.strength >= this.p.heroMin) || code === C_LOG)) {
      const d = this._q(Math.max(this.n, r.n - Math.round(this.p.duckPreMs * 1e-3 * this.fs)), C_DUCK);
      if (d) { d.dur = (r.n - d.n) / this.fs + this.p.duckHoldMs * 1e-3; this.queue.insert(d); }
    }
    return true;
  }
  _parse(ev, code, r) {
    const num = (v, d) => (typeof v === 'number' && isFinite(v) ? v : d);
    const R = this.p.fireRadius;
    const defStr = code === C_IGNITE ? 1 : code === C_EMBER ? 1 : 0.7;
    const st = clamp(num(ev.strength, defStr), 0, 1.5);
    r.code = code; r.strength = st; r.hasSize = typeof ev.size === 'number';
    r.s = r.hasSize ? ev.size : code === C_POP ? 0.3 + 0.95 * st : code === C_LOG ? 0.35 + 0.65 * st : st;
    if (typeof ev.x === 'number' || typeof ev.z === 'number') { r.x = num(ev.x, 0); r.z = num(ev.z, 0); }
    else { r.x = clamp(num(ev.pan, 0), -1.5, 1.5) * R; r.z = 0; }
    r.y = num(ev.y, 0.15);
    r.spMode = typeof ev.screenPan === 'number' ? 1 : 0; r.sp = num(ev.screenPan, 0);
    r.dur = num(ev.dur, 0); r.count = num(ev.count, 0); r.spacing = num(ev.spacing, 0); r.level = num(ev.level, st);
    r.id = num(ev.id, -1);
  }
  _q(n, code) { const r = this.queue.alloc(); if (!r) return null; r.n = n; r.code = code; r.src = 1; r.spMode = 0; r.id = -1; r.hasSize = true; return r; }

  // ------------------------------------------------------------------- spatial helpers
  // stereo pan of a fire-local point (x, z) [m] as seen from the camera
  _stereo(x, z) {
    const xs = x * this.cAz - z * this.sAz, zc = x * this.sAz + z * this.cAz;
    const th = Math.atan2(xs, Math.max(0.15, this.dist - zc));
    return clamp(th * (4 / Math.PI) * this.p.stereoWidth, -1, 1);
  }
  _recStereo(r) { return r.spMode === 1 ? clamp(r.sp, -1, 1) : this._stereo(r.x, r.z); }
  _randPos(rad) { const a = this.rPos.next() * TAU, rr = rad * Math.sqrt(this.rPos.next()); this._px = rr * Math.cos(a); this._pz = rr * Math.sin(a); }
  _logEvent(type, r, size, stereoPan, src) {
    const L = this.log._next();
    L.n = this.n; L.t = this.n / this.fs; L.type = type; L.code = TYPE_CODES[type] || 0; L.strength = q3(r.strength); L.size = q3(size);
    L.pan = q3(clamp(r.x / this.p.fireRadius, -1.5, 1.5)); L.x = q3(r.x); L.y = q3(r.y); L.z = q3(r.z);
    L.stereoPan = q3(stereoPan); L.source = src ? 'random' : 'scripted'; L.random = src ? 1 : 0; L.id = r.id;
  }

  // ------------------------------------------------------------------- control rate
  // Runs every 32 samples on the audio thread. clamp() is written out as `x < a ? a : x > b ? b : x` here: V8 does not
  // always inline the helper into this large function, and a real call boxes its double argument and result.
  _control() {
    const t = this.n / this.fs, p = this.p;
    let tI, tH, tD, tCd, tAz, tW;
    if (this.tl) { const tl = this.tl; tI = this._tl(tl.I, t); tH = this._tl(tl.H, t); tD = this._tl(tl.D, t); tCd = this._tl(tl.Cd, t); tAz = this._tl(tl.Az, t); tW = this._tl(tl.W, t); }
    else { const c = this.ctl; tI = c.intensity; tH = c.hrr; tD = c.dhrr; tCd = c.camDist; tAz = c.camAz; tW = c.wind; }
    tI = tI < 0 ? 0 : tI > 2 ? 2 : tI; tH = tH < 0 ? 0 : tH > 5 ? 5 : tH; tD = tD < -50 ? -50 : tD > 50 ? 50 : tD;
    tCd = tCd < 0.05 ? 0.05 : tCd > 1000 ? 1000 : tCd; tW = tW < 0 ? 0 : tW > 1.5 ? 1.5 : tW;
    if (!this._init) { this._init = true; this.I = tI; this.H = tH; this.Hs = tH; this.Hp = tH; this.D = tD; this.dist = tCd; this.az = tAz; this.windS = tW; }
    this.I += this.kI * (tI - this.I);
    this.H += this.kH * (tH - this.H);
    this.Hs += this.kHs * (this.H - this.Hs);
    this.D += this.kH * (tD - this.D);
    this.dist += this.kCam * (tCd - this.dist);
    let dAz = tAz - this.az; dAz -= TAU * Math.round(dAz / TAU); this.az += this.kCam * dAz;
    this.cAz = Math.cos(this.az); this.sAz = Math.sin(this.az);
    this.windS += this.kWind * (tW - this.windS);

    // swells: gusts (kind 0, wind) and flame swells (kind 1: ignite / whoosh)
    let g = 0, fl = 0;
    for (let i = 0; i < this.swells.length; i++) {
      const S = this.swells[i]; if (!S.active) continue;
      const x = t - S.t0; if (x < 0) continue;
      let e;
      if (x < S.att) e = 0.5 * (1 - Math.cos(Math.PI * x / S.att));
      else { e = Math.exp(-(x - S.att) / S.rel); if (x - S.att > 8 * S.rel) S.active = false; }
      if (S.kind === 0) g += S.s * e; else fl += S.s * e;
    }
    this.gust = Math.min(1.5, g + this.windS); this.fl = Math.min(1.5, fl);
    this.gustLag += this.kGustLag * (this.gust - this.gustLag);
    this.flare *= this.kFlare; this.hawkes *= this.kHawkes; this.tickBurst *= this.kTickBurst;
    // values decaying towards 0 would otherwise sit in the subnormal range (slow FPU path) for seconds
    if (this.I < FTZ) this.I = 0; if (this.H < FTZ) this.H = 0; if (this.Hs < FTZ) this.Hs = 0; if (this.D > -FTZ && this.D < FTZ) this.D = 0;
    if (this.windS < FTZ) this.windS = 0; if (this.gustLag < FTZ) this.gustLag = 0;
    if (this.flare < FTZ) this.flare = 0; if (this.hawkes < FTZ) this.hawkes = 0; if (this.tickBurst < FTZ) this.tickBurst = 0;

    const Ie = Math.max(0, this.I * (1 + this.flare)) + 0.4 * this.fl;
    this.Ie = Ie;
    // heat-release coupling (convention-independent: fluctuation relative to its own 1.5 s mean)
    const Hs = Math.max(0.05, this.Hs), rel0 = Math.max(0, this.H) / Hs, rel = rel0 < 0.25 ? 0.25 : rel0 > 3 ? 3 : rel0, lnRel = Math.log(rel);
    const lnRelC = lnRel < -1.2 ? -1.2 : lnRel > 1.0 ? 1.0 : lnRel, simC = p.simCoupling;
    const heatLvl = 0.75 + 0.25 * Math.min(2, this.Hs), beta = simC < 0 ? 0 : simC > 1 ? 1 : simC;
    // picture coupling (DEFAULTS.picDepth): absolute ln(hrr), and how strongly the picture flickers by itself
    this.Hp += this.kP * (tH - this.Hp);
    let wp = 0, lnPic = 0;
    if (p.picDepth > 0) {
      // (clipped at +-0.2: an ignition ramp or a single flare must not pass for a flickering picture)
      const fr = lnRel < -0.2 ? -0.2 : lnRel > 0.2 ? 0.2 : lnRel, dv = fr - this.pfm;
      this.pfm += this.kPv * dv; this.pfv += this.kPv * (dv * dv - this.pfv);
      const xw = (Math.sqrt(this.pfv) - p.picStdLo) / Math.max(1e-6, p.picStdHi - p.picStdLo);
      wp = xw < 0 ? 0 : xw > 1 ? 1 : xw;
      const lh = Math.log(this.Hp > 0.05 ? this.Hp : 0.05), PL = p.picLim;
      lnPic = p.picDepth * PL * Math.tanh(lh / PL) * (lh < 0 ? p.picDown : 1);
    }
    this.wp = wp;
    const wi = 1 - (1 - p.picIntKeep) * wp; // depth of the internal slow swell / puffing AM that the picture leaves
    // hero-event duck of the random crackle layer (C_DUCK)
    const duT = this.n < this.duckUntil ? this.duckT : 1;
    this.duckG += (duT < this.duckG ? this.kDuckA : this.kDuckR) * (duT - this.duckG);

    // modulators
    this.mPuff.tick(); this.mFlut.tick(); this.mFlap.tick(); this.mSlow.tick(); this.mHiss.tick(); this.mDens.tick(); this.mAct.tick();
    const up = this.mPuff.v, uf = this.mFlut.v, ufl = this.mFlap.v, us = this.mSlow.v;
    const zh = this.mHiss.v, ud = this.mDens.v, ua = (this.uAct = this.mAct.v);
    this.act = Math.exp(p.actDepth * ua - 0.5 * p.actDepth * p.actDepth);
    const pd = p.puffDepth * wi, fd = p.flutDepth, sd = p.slowDepth * wi, flapDepth = p.flapDepth + 0.35 * this.gust;
    const sdA = Math.sqrt(pd * pd + fd * fd + flapDepth * flapDepth + sd * sd);
    let lnAmp = pd * up + fd * uf + flapDepth * ufl + sd * us;
    lnAmp = 2.2 * sdA * Math.tanh(lnAmp / (2.2 * sdA)) - 0.5 * sdA * sdA; // log-normal AM, unit mean, soft-capped
    lnAmp = (1 - 0.5 * beta) * lnAmp + (1 - wp) * beta * p.hrrAmDepth * lnRelC + lnPic * wp + p.actRoar * ua - 0.5 * p.actRoar * p.actRoar;
    this.roarAmp = Math.pow(Ie, 1.5) * heatLvl * Math.exp(lnAmp) * (1 + 1.4 * this.gust) * (1 + 0.3 * this.fl);
    const bp = 0.85 * pd, bf = 2.0 * fd * (1 - 0.4 * wp), bfl = 1.6 * p.flapDepth + 0.3 * this.gust;
    const lnB = bp * up + bf * uf + bfl * ufl - 0.5 * (bp * bp + bf * bf + bfl * bfl) + (1 - wp) * 0.6 * beta * p.hrrAmDepth * lnRelC + lnPic * wp + 0.8 * sd * us + p.actRoar * ua - 0.5 * (0.64 * sd * sd + p.actRoar * p.actRoar);
    this.breathAmp = Math.pow(Ie, 1.3) * heatLvl * Math.exp(lnB) * (1 + this.gust + 0.6 * this.fl);
    this.puff = up;
    // brightness: knee 150..~700 Hz with size, wind, flame swell and heat release
    const rm1 = rel - 1, relC = rm1 < -0.5 ? -0.5 : rm1 > 1.5 ? 1.5 : rm1;
    const kx = p.roarKnee0 + 220 * Math.min(1.5, Ie) + 260 * this.gust + 200 * this.fl + 40 * up + 160 * relC + 100 * (Math.min(2, this.Hs) - 1);
    const knee = kx < 60 ? 60 : kx > 1200 ? 1200 : kx;
    if (Math.abs(knee - this._lastKnee) > 2) { this._lastKnee = knee; this.roarCh[0].knee.a = this.roarCh[1].knee.a = 1 - Math.exp(-TAU * knee / this.fs); } // == OnePole.set
    this.hissAmp = Ie * ((1 - p.hissSurge) + p.hissSurge * Math.min(zh * zh * zh * zh, 16) / 3) * Math.exp(p.hissAct * ua - 0.5 * p.hissAct * p.hissAct + p.hissFlut * uf - 0.5 * p.hissFlut * p.hissFlut) * (1 + 0.8 * this.gust);
    this.lapAmp = Math.pow(Ie, 1.2) * (1 + 1.2 * this.gust);

    // combustion "whump": p ~ dQ/dt (expansion louder than contraction)
    const dr = this.D / Math.max(0.25, Hs);
    let eW = Math.tanh(Math.max(0, dr) / 2) + 0.35 * Math.tanh(Math.max(0, -dr) / 2);
    if (typeof p.whumpForce === 'number') eW = p.whumpForce; // calibration hook
    this.whumpAmp = eW;
    if (dr < p.whumpRearm) this.whumpArmed = true;
    if (this.whumpArmed && dr > p.whumpThresh && this.n - this.lastWhumpN > 0.6 * this.fs) {
      this.whumpArmed = false;
      this._thump(clamp((dr - 0.8) / 3, 0.2, 1), 0, 0);
    }

    // event rates (per-sample probabilities)
    const dens = Math.exp(0.35 * ud - 0.06) * (1 + 0.25 * up);
    this.pTick = Math.min(0.2, (p.tickRate * Math.pow(Ie, 1.2) * dens * (1 + 0.8 * this.gustLag) + this.tickBurst) / this.fs);
    this.pCrk = Math.min(0.2, (p.crackleRate * this.act * Math.pow(Ie, 1.3) * Math.max(0.2, 1 + p.crkPuff * up + p.crkFlut * uf) * (1 + 0.8 * this.gustLag) * Math.sqrt(heatLvl) + this.hawkes) / this.fs);
    this.pPop = p.popRate * Math.pow(Ie, 1.5) * (1 + this.gustLag) / this.fs;
    this.pEmb = p.emberRate * Math.min(1, 1.5 * Ie) / this.fs;

    // distance: ~1/r with near-field clamp (gain 1 at refDist), HF shelf, apparent width
    const r = this.dist, rN = p.nearField, r0 = p.refDist;
    const dg2 = (r0 * r0 + rN * rN) / (r * r + rN * rN);
    this.distGain = p.distDepth === 1 ? Math.sqrt(dg2) : Math.pow(dg2, 0.5 * p.distDepth);
    const hx = -3.5 * Math.log2(r / r0) - 0.06 * (r - r0), hsDb = Math.round(20 * (hx < -14 ? -14 : hx > 4 ? 4 : hx)) / 20; // 0.05 dB steps
    if (this._lastHs !== hsDb) { this._lastHs = hsDb; this.distHS[0].highShelf(4500, 0.7, hsDb, this.fs); this.distHS[1].highShelf(4500, 0.7, hsDb, this.fs); }
    const R = p.fireRadius;
    const bw = 0.85 * Math.atan(R / Math.max(0.1, r)) / Math.atan(R / r0);
    this.bedW = bw < 0.3 ? 0.3 : bw > 1 ? 1 : bw;

    // whoosh band: wind gust + flame swell
    const wd = this.gust + 1.6 * this.fl;
    this.whooshAmp = Math.pow(Math.min(1.5, wd), 1.5);
    if (this.whooshAmp > 1e-4) {
      const wf = 250 + 750 * Math.min(1, wd), fs = this.fs;
      for (let k = 0; k < 2; k++) { // == whooshSvf[k].set(k ? wf * 1.12 : wf, 0.9, fs), without boxed arguments
        const S = this.whooshSvf[k], g = Math.tan(Math.PI * Math.min(k ? wf * 1.12 : wf, fs * 0.45) / fs);
        S.k = 1 / 0.9; S.a1 = 1 / (1 + g * (g + S.k)); S.a2 = g * S.a1; S.a3 = g * S.a2;
      }
    }
    const wpx = this.gust * 0.6 - 0.3;
    this.whooshPan = (wpx < -1 ? -1 : wpx > 1 ? 1 : wpx) * (this.gust / Math.max(1e-6, wd));

    // crickets: pan by world azimuth relative to the camera
    for (let k = 0; k < this.crickets.length; k++) {
      const c = this.crickets[k], rel2 = c.phi - this.az;
      const sx = 0.85 * Math.sin(rel2), th = ((sx < -1 ? -1 : sx > 1 ? 1 : sx) + 1) * Math.PI / 4; c.gL = Math.cos(th); c.gR = Math.sin(th);
    }
  }

  // ------------------------------------------------------------------- spawners
  // Crackle at fire-local (cx, cz) +- spread [m], or a random position in the fire bed when hasPos is false.
  // spMode 1: the crackle belongs to an event placed at stereo position sp by the caller (screenPan): it is panned at sp plus
  // its offset from the event centre (cx, cz) as seen from the camera, never re-projected as a whole.
  _spawnCrackle(sizeScale, hasPos, cx, cz, spread, spMode = 0, sp = 0) {
    const r = this.rCrk, p = this.p;
    // AE energies ~ power law P(E) ~ E^-b; amplitude ~ E^crkExp -> amplitude pdf exponent 1 + (b-1)/crkExp
    // (real fires: 1.9-2.5 above 10 dB over the local background, spanning > 30 dB)
    const E = r.pareto(1, p.crkEmax, p.crkB) * sizeScale * (hasPos ? 1 : Math.pow(this.act, p.actSizeExp));
    const sz = clamp(Math.log(E) / Math.log(p.crkEmax), 0, 1);
    // random crackles: soft cap of the largest energies (hero events stay on top) and the hero-event duck
    const Ea = !hasPos && p.crkCapRatio < 1 && E > p.crkCapE ? p.crkCapE * Math.pow(E / p.crkCapE, p.crkCapRatio) : E;
    const amp = p.crkAmp0 * Math.pow(Ea, p.crkExp) * (hasPos ? 1 : this.duckG);
    const tau = p.crkTauMs * Math.pow(E, 0.12) * r.logRange(0.5, 2.0);   // real: -20 dB within ~2-3 ms
    const fc = r.logRange(p.crkFcLo, p.crkFcHi) / (1 + 0.35 * sz);       // real excess spectra peak ~4-8 kHz
    const Q = r.range(0.6, 1.6);
    const raw = r.range(p.crkRawLo, p.crkRawHi);
    let x, z;
    if (hasPos) { x = cx + r.range(-1, 1) * spread; z = cz + r.range(-1, 1) * spread; }
    else { this._randPos(p.fireRadius * 0.9); x = this._px; z = this._pz; }
    let nSub = 1; while (nSub < 8 && r.next() < p.crkSubP) nSub++;
    const body = p.crkBody * r.logRange(0.3, 1.5) * (0.5 + sz), fBody = r.logRange(160, 800);
    const pan = spMode === 1 ? clamp(sp + this._stereo(x, z) - this._stereo(cx, cz), -1, 1) : this._stereo(x, z);
    this.crackles.get().start(this.rVoiceNoise, amp, tau, fc, Q, raw, pan, p.send.crackles, nSub, p.crkSubGapMs, this.mixG.crackles,
      p.crkCorr * r.range(0.7, 1.3), body, fBody, p.crkTail, p.crkTailMs * r.logRange(0.6, 1.6), p.crkSubGapMin);
    this.hawkes += Math.min(0.85, p.hawkesBranching * (0.6 + 0.8 * sz)) / (p.hawkesTauMs * 1e-3);
  }
  // unit-mean gamma(k) threshold, k = crkShape (integer 1..4; 1 = exponential = Poisson)
  _gammaThr() { const k = clamp(Math.round(this.p.crkShape), 1, 4); let s = 0; for (let i = 0; i < k; i++) s -= Math.log(1 - this.rCrk.next()); return s / k; }
  _modal(freqs, taus, amps, n, pan, send) { this.modals.get().start(freqs, taus, amps, n, pan, send); }

  // low "whoomph": damped 42-68 Hz mode + LP noise burst (flare-ups, ignition, whoosh)
  _thump(s, pan, extra) {
    const r = this.rWhump, L = this.mixG.whump > 0 ? this.mixG.logs : 0, send = this.p.send.logs * 0.5;
    if (L <= 0 || this.n - this.lastWhumpN < 0.6 * this.fs) return; // one whoomph per flare-up (dhrr onset or event)
    this.lastWhumpN = this.n;
    this._fr[0] = r.range(42, 68); this._ta[0] = 0.09 * (0.6 + 0.4 * s); this._am[0] = 0.11 * L * s;
    this._fr[1] = this._fr[0] * r.range(2.1, 2.6); this._ta[1] = 0.035; this._am[1] = 0.032 * L * s;
    this._modal(this._fr, this._ta, this._am, 2, pan * 0.5, send);
    this.noises.get().start(this.rVoiceNoise, 0.09 * L * s, 240, 0.7, 1, 8, 70 + 30 * extra, 0.3, 18, pan * 0.5, send, 0);
  }

  _firePop(rec) {
    const r = this.rPop, random = rec.src === 1 && rec.code === C_POP && !rec.hasSize;
    let s = rec.s;
    if (random) { s = r.pareto(0.25, 1.2, 2.2); rec.strength = clamp((s - 0.3) / 0.95, 0, 1); this._randPos(this.p.fireRadius * 0.8); rec.x = this._px; rec.z = this._pz; rec.y = r.range(0.05, 0.3); }
    const pan = this._recStereo(rec);
    const L = this.mixG.pops, send = this.p.send.pops;
    // 1) rupture click (sub-ms N-wave-ish burst)
    this.crackles.get().start(this.rVoiceNoise, 0.5 * Math.pow(s, 0.8), 0.5, 7000, 0.7, 0.85, pan, send, 1, 0, L, 0.5);
    // 2) cavity / fragment ring (Helmholtz-like 1.2-4.5 kHz; bigger pocket = lower, longer)
    const f1 = r.logRange(1200, 4500) / Math.pow(s, 0.4);
    this._fr[0] = f1; this._fr[1] = f1 * r.range(1.4, 2.3); this._fr[2] = f1 * r.range(2.6, 3.8);
    const tau1 = r.range(0.004, 0.012) * Math.sqrt(s);
    this._ta[0] = tau1; this._ta[1] = tau1 * 0.6; this._ta[2] = tau1 * 0.4;
    const a1 = 0.11 * L * s; this._am[0] = a1; this._am[1] = a1 * 0.5; this._am[2] = a1 * 0.25;
    this._modal(this._fr, this._ta, this._am, 3, pan, send);
    // 3) crack noise 2-6 ms
    this.noises.get().start(this.rVoiceNoise, 0.16 * L * s, r.range(2500, 5000) / Math.sqrt(s), 0.7, 0, 0.1, r.range(2, 6) * Math.sqrt(s), 0, 0, pan, send, 0);
    // 3b) hero pop: 1.5-4 kHz crack + ring
    if (!random && rec.strength >= this.p.heroMin) this._heroCrack(s, pan, 0);
    // 4) LF body for big pops
    if (s > 0.5) { this._fr[0] = r.range(90, 200); this._ta[0] = 0.02 * s; this._am[0] = 0.2 * L * s; this._modal(this._fr, this._ta, this._am, 1, pan * 0.5, send * 0.5); }
    // 4b) spark / resin sizzle
    if (s > 0.7) this.tickBurst += 700 * s;
    // 5) fracture cascade over ~60 ms
    const nMicro = Math.round(3 + 9 * s * r.next());
    let tt = 0;
    for (let i = 0; i < nMicro; i++) {
      tt += r.range(0.002, 0.012);
      const q = this._q(this.n + Math.round(tt * this.fs), C_MICRO); if (!q) break;
      q.s = 0.35 * s; q.x = rec.x; q.z = rec.z; q.dur = 0.03; q.spMode = rec.spMode; q.sp = rec.sp; this.queue.insert(q);
    }
    // 6) steam jet "psssh"
    if (r.next() < 0.3 + 0.3 * s) this.noises.get().start(this.rVoiceNoise, 0.01 * L * s, r.range(3500, 7000), 0.9, 0, 8, r.range(60, 250) * s, 0.6, 40, pan, send, 0);
    // 7) aftershock (Omori-like)
    this.hawkes += 35 * s;
    this._logEvent('pop', rec, s, pan, random);
  }

  _fireLogShift(rec) {
    const r = this.rLog, s = rec.s, pan = this._recStereo(rec);
    const L = this.mixG.logs, send = this.p.send.logs, n0 = this.n;
    // thump 55-90 Hz + 2nd mode, LP noise
    this._fr[0] = r.range(55, 90); this._ta[0] = 0.06 * s; this._am[0] = 0.12 * L * s;
    this._fr[1] = this._fr[0] * r.range(2.1, 2.9); this._ta[1] = 0.03 * s; this._am[1] = 0.05 * L * s;
    this._modal(this._fr, this._ta, this._am, 2, pan, send * 0.5);
    this.noises.get().start(this.rVoiceNoise, 0.1 * L * s, 180, 0.7, 1, 3, 45, 0, 0, pan, send * 0.5, 0);
    // knock (wood on wood)
    this._fr[0] = r.range(350, 700); this._fr[1] = this._fr[0] * r.range(2.4, 2.9); this._ta[0] = 0.025; this._ta[1] = 0.015; this._am[0] = 0.08 * L * s; this._am[1] = 0.05 * L * s;
    this._modal(this._fr, this._ta, this._am, 2, pan, send);
    // debris rustle
    this.noises.get().start(this.rVoiceNoise, 0.015 * L * s, 2500, 0.6, 2, 20, 280, 0.8, 25, pan, send, 0);
    // charcoal / ember tinkles cascade
    const N = Math.round(4 + 10 * s); let tt = 0.015;
    for (let i = 0; i < N; i++) {
      tt += i < N / 2 ? r.range(0.01, 0.08) : r.range(0.03, 0.18);
      const q = this._q(n0 + Math.round(tt * this.fs), C_TINKLE); if (!q) break;
      q.x = rec.x + r.range(-0.25, 0.25) * this.p.fireRadius; q.z = rec.z + r.range(-0.25, 0.25) * this.p.fireRadius;
      q.level = L * Math.sqrt(s) * r.range(0.25, 1) * (1 - 0.5 * i / N); q.s = send;
      if (rec.spMode === 1) { q.spMode = 1; q.sp = clamp(rec.sp + this._stereo(q.x, q.z) - this._stereo(rec.x, rec.z), -1, 1); }
      this.queue.insert(q);
    }
    // crackle burst over the first ~0.4 s
    const nb = Math.round(4 + 8 * s); tt = 0.02;
    for (let i = 0; i < nb; i++) {
      tt += r.range(0.01, 0.06);
      const q = this._q(n0 + Math.round(tt * this.fs), C_MICRO); if (!q) break;
      q.s = 0.6 + 0.6 * s; q.x = rec.x; q.z = rec.z; q.dur = 0.08; q.spMode = rec.spMode; q.sp = rec.sp; this.queue.insert(q);
    }
    // hero: 1.5-4 kHz crack + ring, and a steam "psssh" tail (attack 25 ms, ~150-300 ms audible, falling 4.2 -> 2.6 kHz)
    this._heroCrack(s, pan, 1);
    if (this.p.logPsssh > 0) this.noises.get().start(this.rVoiceNoise, 0.05 * this.p.logPsssh * L * s, 4200, 0.8, 0, 25, 75, 0.5, 35, pan, this.p.send.hero, 2600);
    this.flare += 0.35 * s; this.hawkes += 25 * s;
    this._logEvent('log-shift', rec, s, pan, false);
  }

  // Hero events (kind 0 = pop, 1 = log-shift): a 1.5-4 kHz ring (3 modes, 40-70 ms) + a ~6 ms band-passed crack, so the event
  // stands out in the band small speakers reproduce. Own RNG stream; silent (and draws nothing) while mix.hero is off.
  _heroCrack(s, pan, kind) {
    const L = this.mixG.hero; if (!(L > 1e-4)) return;
    const r = this.rHero, send = this.p.send.hero, g = L * (s < 1.25 ? s : 1.25);
    const f1 = r.logRange(1600, 2300) * (kind === 1 ? 0.85 : 1);
    this._fr[0] = f1; this._fr[1] = f1 * r.range(1.42, 1.62); this._fr[2] = f1 * r.range(1.95, 2.3);
    const tau = (kind === 1 ? 0.065 : 0.045) * r.range(0.85, 1.2);
    this._ta[0] = tau; this._ta[1] = tau * 0.7; this._ta[2] = tau * 0.45;
    this._am[0] = 0.1 * g; this._am[1] = 0.06 * g; this._am[2] = 0.035 * g;
    this._modal(this._fr, this._ta, this._am, 3, pan, send);
    this.noises.get().start(this.rVoiceNoise, 0.12 * g, r.range(2200, 3400), 1.0, 0, 0.3, r.range(4, 8), 0, 0, pan, send, 0);
  }

  _tinkle(level, pan, send) {
    const r = this.rEmb, f1 = r.logRange(2200, 6500);
    this._fr[0] = f1; this._fr[1] = f1 * r.range(1.55, 2.4); this._fr[2] = f1 * r.range(2.9, 4.6);
    const tau = r.range(0.004, 0.018); this._ta[0] = tau; this._ta[1] = tau * 0.7; this._ta[2] = tau * 0.5;
    const a = 0.035 * level; this._am[0] = a; this._am[1] = a * 0.6; this._am[2] = a * 0.35;
    this._modal(this._fr, this._ta, this._am, 3, pan, send);
  }

  _swell(kind, n0, s, att, rel) {
    let S = null;
    for (let i = 0; i < this.swells.length; i++) if (!this.swells[i].active) { S = this.swells[i]; break; }
    if (!S) { // all busy: replace the oldest
      S = this.swells[0]; for (let i = 1; i < this.swells.length; i++) if (this.swells[i].t0 < S.t0) S = this.swells[i];
    }
    S.active = true; S.kind = kind; S.t0 = n0 / this.fs; S.s = s; S.att = Math.max(0.02, att); S.rel = Math.max(0.02, rel);
  }

  // Ignition: match strike (scrape -> head flare "pfft" + sizzle) -> rising flame whoosh/roar over ~0.8 s -> crackle surge.
  _fireIgnite(rec) {
    const r = this.rIgn, s = clamp(rec.s, 0.1, 1.5), Lp = 0.7 * this.mixG.pops, send = this.p.send.pops, n0 = this.n, fs = this.fs;
    const pan = this._recStereo(rec);
    // 1) scrape: rough friction noise rising in pitch, granular AM, ~70 ms
    this.noises.get().start(this.rVoiceNoise, 0.05 * Lp * s, r.range(2400, 3200), 0.8, 0, 14, 20, 0.95, 260, pan, send * 0.5, r.range(4500, 5500));
    const ng = 5 + Math.round(3 * r.next()); let tt = 0;
    for (let i = 0; i < ng; i++) {
      tt += r.range(0.006, 0.012);
      const q = this._q(n0 + Math.round(tt * fs), C_MICRO); if (!q) break;
      q.s = 0.3; q.x = rec.x; q.z = rec.z; q.dur = 0.01; q.spMode = rec.spMode; q.sp = rec.sp; this.queue.insert(q);
    }
    // 2) match-head flare at +65..90 ms
    let q = this._q(n0 + Math.round(r.range(0.065, 0.09) * fs), C_MATCHHEAD);
    if (q) { q.s = s; q.x = rec.x; q.z = rec.z; q.spMode = rec.spMode; q.sp = rec.sp; this.queue.insert(q); }
    // 3) flame swell: raised-cosine rise over 0.75 s from +0.12 s, release tau 1.1 s (drives roar, breath, whoosh, knee)
    this._swell(1, n0 + Math.round(0.12 * fs), s, 0.75, 1.1);
    // 4) the kindling catching: crackle surges + a low whump
    for (let i = 0; i < IGNITE_SURGE.length; i += 2) { q = this._q(n0 + Math.round(IGNITE_SURGE[i] * fs), C_HAWKES); if (q) { q.level = IGNITE_SURGE[i + 1] * s; this.queue.insert(q); } }
    q = this._q(n0 + Math.round(0.5 * fs), C_THUMP); if (q) { q.s = 0.75 * s; q.x = rec.x; q.z = rec.z; q.spMode = rec.spMode; q.sp = rec.sp; q.level = 1; this.queue.insert(q); }
    this._logEvent('ignite', rec, s, pan, false);
  }
  _fireMatchHead(rec) {
    const s = rec.s, Lp = 0.7 * this.mixG.pops, send = this.p.send.pops, pan = this._recStereo(rec);
    this.crackles.get().start(this.rVoiceNoise, 0.3 * s, 0.35, 6000, 0.7, 0.8, pan, send, 2, 0.6, Lp);          // crack
    this.noises.get().start(this.rVoiceNoise, 0.075 * Lp * s, 1100, 0.7, 2, 2, 140, 0.7, 35, pan, send, 0);      // fizz
    this.noises.get().start(this.rVoiceNoise, 0.06 * Lp * s, 6000, 0.9, 0, 1, 45, 0, 0, pan, send, 0);           // bright flash
    this.noises.get().start(this.rVoiceNoise, 0.09 * Lp * s, 700, 0.7, 1, 4, 60, 0.4, 25, pan, send * 0.5, 0);  // "pff" body
    this.tickBurst += 450 * s;
  }

  _fire(rec) {
    switch (rec.code) {
      case C_POP: this._firePop(rec); break;
      case C_LOG: this._fireLogShift(rec); break;
      case C_GUST: {
        const g = rec.strength, d = rec.dur > 0 ? rec.dur : 1.6 + 1.2 * g;
        this._swell(0, this.n, g, 0.35 * d, 0.4 * d);
        this._logEvent('gust', rec, g, this._recStereo(rec), false); break;
      }
      case C_WHOOSH: {
        const s = rec.strength, d = rec.dur > 0 ? rec.dur : 1.2;
        this._swell(1, this.n, 0.85 * s, 0.3 * d, 0.45 * d);
        const q = this._q(this.n + Math.round(0.12 * d * this.fs), C_THUMP);
        if (q) { q.s = 0.55 * s; q.x = rec.x; q.z = rec.z; q.spMode = rec.spMode; q.sp = rec.sp; q.level = 0; this.queue.insert(q); }
        this.hawkes += 12 * s;
        this._logEvent('whoosh', rec, s, this._recStereo(rec), false); break;
      }
      case C_BURST: {
        const st = rec.strength, k = rec.count > 0 ? Math.round(rec.count) : Math.round(5 + 15 * st), sp = rec.spacing > 0 ? rec.spacing : 0.025;
        const sz = rec.hasSize ? rec.s : 0.8 + st;
        for (let i = 0; i < k; i++) {
          const q = this._q(this.n + Math.round(i * sp * this.rCrk.range(0.3, 1.7) * this.fs), C_MICRO); if (!q) break;
          q.s = sz; q.x = rec.x; q.z = rec.z; q.dur = 0.06; q.spMode = rec.spMode; q.sp = rec.sp; this.queue.insert(q);
        }
        this._logEvent('crackle-burst', rec, sz, this._recStereo(rec), false); break;
      }
      case C_IGNITE: this._fireIgnite(rec); break;
      case C_EMBER: this._tinkle(this.mixG.embers * rec.level, this._recStereo(rec), this.p.send.embers); break;
      case C_FLARE: this.flare += rec.strength * 0.5; this._logEvent('flare', rec, rec.strength, 0, false); break;
      case C_MICRO: this._spawnCrackle(rec.s, true, rec.x, rec.z, rec.dur, rec.spMode, rec.sp); break;
      case C_TINKLE: this._tinkle(rec.level, rec.spMode === 1 ? rec.sp : this._stereo(rec.x, rec.z), rec.s); break;
      case C_DUCK: { const u = this.n + Math.round(rec.dur * this.fs); if (u > this.duckUntil) this.duckUntil = u; this.duckT = dB(this.p.duckDb); break; }
      case C_MATCHHEAD: this._fireMatchHead(rec); break;
      case C_HAWKES: this.hawkes += rec.level; break;
      case C_THUMP: this._thump(rec.s, this._recStereo(rec), rec.level); break;
    }
  }

  // ------------------------------------------------------------------- audio
  // Writes n frames: dryL/dryR = direct sound, sendL/sendR = reverb send (convolve with makeOutdoorIR()).
  // Allocation-free: in this per-sample loop no call passes or returns a double (see RB); filters run through
  // their x/y fields, RNG draws read the generator buffer directly. Rare paths (events, spawns) may allocate.
  render(dryL, dryR, sendL, sendR, n) {
    const fs = this.fs, M = this.mixG, S = this.p.send, Q = this.queue;
    const rT = this.rTick, rR = this.rRoar, rH = this.rHiss, rW = this.rWhump, rP = this.rPop, rE = this.rEmb, rWo = this.rWhoosh;
    const hg = this.p.roarHarm;
    let q = null, P = null, V = null;
    for (let i = 0; i < n; i++) {
      if (this.n % CR === 0) this._control();
      while (Q.len && Q.q[Q.len - 1].n <= this.n) { const rec = Q.pop(); this._fire(rec); Q.release(rec); }

      // random crackles: time-rescaled gamma renewal process (shape crkShape) driven by the Cox/Hawkes intensity.
      // Real crackling is locally more regular than Poisson (interval LV ~0.55) while its rate wanders (CV ~1).
      this.crkAcc += this.pCrk;
      if (this.crkAcc >= this.crkThr) { this.crkAcc = 0; this.crkThr = this._gammaThr(); this._spawnCrackle(1, false, 0, 0, 0); }
      if (rP.i === RB) rP.refill();
      if (rP.buf[rP.i++] < this.pPop) { const qp = this._q(this.n, C_POP); if (qp) { qp.hasSize = false; qp.strength = 0; this._firePop(qp); Q.release(qp); } }
      if (rE.i === RB) rE.refill();
      if (rE.buf[rE.i++] < this.pEmb) { this._randPos(this.p.fireRadius); this._tinkle(M.embers * this.rEmb.range(0.2, 1), this._stereo(this._px, this._pz), S.embers); }

      let bL = 0, bR = 0, sL = 0, sR = 0;
      // --- ticks (independent L/R micro-crackle streams)
      if (M.ticks > 0) {
        for (let c = 0; c < 2; c++) {
          let x0 = DENORM, x1 = DENORM;
          const F = this.tickF[c];
          if (rT.i === RB) rT.refill();
          if (rT.buf[rT.i++] < this.pTick) { // ~300/s: same draws as dB(-17 + 3.5 * gauss()), range(0.08, 0.3), next()
            rT.gaussRun(); const a = Math.pow(10, (-17 + 3.5 * rT.gv) / 20);
            if (a > F.env) {
              F.env = a; if (rT.i === RB) rT.refill(); F.d = Math.exp(-1 / ((0.08 + (0.3 - 0.08) * rT.buf[rT.i++]) * 1e-3 * fs));
              if (rT.i === RB) rT.refill(); F.route = rT.buf[rT.i++];
            }
          }
          if (F.env > 1e-5) { if (rT.i === RB) rT.refill(); const x = F.env * (rT.buf[rT.i++] * 2 - 1); F.env *= F.d; if (F.route < 0.55) x0 += x; else x1 += x; }
          q = F.hp; q.x = x0; q.run(); const y0 = q.y;
          q = F.b2; q.x = x1; q.run();
          const y = (y0 + q.y * 1.5) * M.ticks * this.duckG;
          if (c === 0) { bL += y; sL += y * S.ticks; } else { bR += y; sR += y * S.ticks; }
        }
      }
      // --- roar (mid + 0.45 side) + Farnell lapping
      if (M.roar > 0 || M.lapping > 0) {
        if (rR.i === RB) rR.refill(); const m = rR.buf[rR.i++] * 2 - 1;
        if (rR.i === RB) rR.refill(); const sd = (rR.buf[rR.i++] * 2 - 1) * 0.45, ra = this.roarAmp * M.roar * 6.0;
        for (let c = 0; c < 2; c++) {
          const C = this.roarCh[c];
          P = C.pink; P.w = c === 0 ? m + sd : m - sd; P.run(); let x = P.y;
          q = C.knee; q.y += q.a * (x - q.y); x = q.y; // pink x one-pole => P(f) ~ f^-3 above the knee (Chadwick & James alpha = 3)
          q = C.hp; q.x = x; q.run(); x = q.y; q = C.body; q.x = x; q.run(); x = q.y;
          q = C.breath; q.x = x; q.run(); x = q.y; q = C.lp; q.x = x; q.run(); x = q.y;
          C.ms += 2e-5 * (x * x - C.ms);
          const cl = this.p.roarSat * Math.sqrt(C.ms + 1e-12);
          let yh = 0;
          if (hg > 0) { // harmonics of the low band (DEFAULTS.roarHarm): LP 150 -> |unit-RMS| -> 150-600 Hz, back at the band's RMS
            q = C.hlp; q.x = x; q.run(); const lo = q.y;
            C.hms += 2e-4 * (lo * lo - C.hms);
            const sl = Math.sqrt(C.hms + 1e-20), u = lo / sl;
            q = C.hhp; q.x = (u < 0 ? -u : u) - 0.8; q.run(); const h1 = q.y; q = C.hlp2; q.x = h1; q.run();
            yh = q.y * sl * hg;
          }
          const y = (cl * Math.tanh(x / cl) + yh) * ra;
          if (c === 0) { bL += y; sL += y * S.roar; } else { bR += y; sR += y * S.roar; }
        }
        if (rR.i === RB) rR.refill();
        q = this.lapBP; q.x = (rR.buf[rR.i++] * 2 - 1) * 1.7320508; q.run(); const lb = q.y * this.lapNorm;
        q = this.lapHP1; q.x = Math.tanh(1.6 * lb); q.run(); const l1 = q.y; q = this.lapHP2; q.x = l1; q.run();
        const ly = q.y * this.lapAmp * M.lapping;
        bL += ly; bR += ly;
      }
      // --- breath (mid-band flame whoosh)
      if (M.breath > 0) {
        const ba = this.breathAmp * M.breath, bs = this.p.breathSide, bn = 1.7320508 / Math.sqrt(1 + bs * bs);
        if (rR.i === RB) rR.refill(); const bm = rR.buf[rR.i++] * 2 - 1;
        if (rR.i === RB) rR.refill(); const bsd = (rR.buf[rR.i++] * 2 - 1) * bs;
        for (let c = 0; c < 2; c++) {
          const C = this.breathCh[c];
          P = C.pink; P.w = (c === 0 ? bm + bsd : bm - bsd) * bn; P.run();
          V = C.svf; V.x = P.y; V.run();
          q = C.hp; q.x = V.k * V.bp; q.run(); const h1 = q.y; q = C.lp; q.x = h1; q.run();
          const y = q.y * ba;
          if (c === 0) { bL += y; sL += y * S.roar; } else { bR += y; sR += y * S.roar; }
        }
      }
      // --- hiss (outgassing)
      if (M.hiss > 0) {
        const h = this.hissAmp * M.hiss * 0.35;
        for (let c = 0; c < 2; c++) {
          const C = this.hissCh[c];
          if (rH.i === RB) rH.refill();
          q = C.hp; q.x = rH.buf[rH.i++] * 2 - 1; q.run(); const h1 = q.y; q = C.hs; q.x = h1; q.run();
          const y = q.y * h;
          if (c === 0) { bL += y; sL += y * S.hiss; } else { bR += y; sR += y * S.hiss; }
        }
      }
      // --- whump (combustion noise ~ dQ/dt), centre
      if (M.whump > 0) {
        if (rW.i === RB) rW.refill();
        P = this.whPink; P.w = (rW.buf[rW.i++] * 2 - 1) * 1.7320508; P.run();
        q = this.whLP; q.y += q.a * (P.y - q.y); const w1 = q.y;
        q = this.whHP; q.x = w1; q.run(); const w2 = q.y; q = this.whLP2; q.x = w2; q.run();
        let x = q.y;
        this.whMs += 2e-5 * (x * x - this.whMs);
        x = x / Math.sqrt(this.whMs + 1e-12); // unit RMS carrier
        const y = Math.tanh(1.2 * x * this.whumpAmp) * (1 / 1.2) * M.whump;
        bL += y; bR += y; sL += y * S.whump; sR += y * S.whump;
      }
      // --- whoosh (gust / flame swell band)
      if (M.whoosh > 0 && this.whooshAmp > 1e-4) {
        if (rWo.i === RB) rWo.refill(); const w0 = (rWo.buf[rWo.i++] * 2 - 1) * 1.7320508;
        if (rWo.i === RB) rWo.refill(); const w1 = (rWo.buf[rWo.i++] * 2 - 1) * 1.7320508;
        const wa = this.whooshAmp * M.whoosh * 2.5, wp = this.whooshPan, th = ((wp < -1 ? -1 : wp > 1 ? 1 : wp) + 1) * Math.PI / 4; // == panGains
        const g0 = Math.cos(th), g1 = Math.sin(th);
        P = this.whooshPink[0]; P.w = w0; P.run(); V = this.whooshSvf[0]; V.x = P.y; V.run(); q = this.whooshLP[0]; q.x = V.bp; q.run();
        const y0 = q.y * V.k * wa;
        P = this.whooshPink[1]; P.w = w1; P.run(); V = this.whooshSvf[1]; V.x = P.y; V.run(); q = this.whooshLP[1]; q.x = V.bp; q.run();
        const y1 = q.y * V.k * wa;
        bL += y0 * g0 * 1.2 + y1 * 0.3; bR += y1 * g1 * 1.2 + y0 * 0.3; sL += y0 * S.whoosh; sR += y1 * S.whoosh;
      }
      // --- bed width narrows with distance (fire subtends a smaller angle)
      const mid = 0.5 * (bL + bR), side = 0.5 * (bL - bR) * this.bedW;
      let L = mid + side, R = mid - side;

      // --- voices (already panned by camera-relative position)
      const cv = this.crackles.v;
      for (let k = 0; k < cv.length; k++) {
        const v = cv[k]; if (!v.active) continue;
        v.tick(); const yl = v.yL * v.g[0], yr = v.yR * v.g[1];
        L += yl; R += yr; sL += yl * v.send; sR += yr * v.send;
      }
      const mv = this.modals.v;
      for (let k = 0; k < mv.length; k++) {
        const v = mv[k]; if (!v.active) continue;
        v.tick(); const y = v.y; L += y * v.g[0]; R += y * v.g[1]; sL += y * v.g[0] * v.send; sR += y * v.g[1] * v.send;
      }
      const nv = this.noises.v;
      for (let k = 0; k < nv.length; k++) {
        const v = nv[k]; if (!v.active) continue;
        v.tick(); const y = v.y; L += y * v.g[0]; R += y * v.g[1]; sL += y * v.g[0] * v.send; sR += y * v.g[1] * v.send;
      }

      // --- distance (level + HF shelf) and ground reflection on the dry path
      const dg = this.distGain;
      q = this.distHS[0]; q.x = L + DENORM; q.run(); L = q.y * dg;
      q = this.distHS[1]; q.x = R + DENORM; q.run(); R = q.y * dg;
      const gi = this.grIdx, gb = this.grBuf;
      gb[0][gi & 255] = L; gb[1][gi & 255] = R;
      q = this.grLP[0]; q.y += q.a * (gb[0][(gi - this.grD[0]) & 255] - q.y); L += 0.25 * q.y;
      q = this.grLP[1]; q.y += q.a * (gb[1][(gi - this.grD[1]) & 255] - q.y); R += 0.25 * q.y;
      this.grIdx = (gi + 1) & 0x3fffffff;

      // --- crickets (distant night ambience; not affected by the camera-to-fire distance)
      if (M.crickets > 0) {
        const t = this.n / fs;
        for (let k = 0; k < this.crickets.length; k++) {
          const c = this.crickets[k];
          if (t >= c.next) {
            c.chirpStart = t; c.chirpEnd = t + c.sylN * c.sylP + 0.01;
            c.next = t + c.period * (1 + 0.06 * this.rCri.gauss());
            if (this.rCri.next() < 0.08) c.chirpStart = -1;
          }
          if (c.chirpStart < 0 || t > c.chirpEnd) continue;
          const x = t - c.chirpStart, si = Math.floor(x / c.sylP);
          if (si >= c.sylN) continue;
          const xs = x - si * c.sylP;
          let e = 0;
          if (xs < c.sylD) { const a = 0.003, rl = 0.005; e = xs < a ? 0.5 - 0.5 * Math.cos(Math.PI * xs / a) : xs > c.sylD - rl ? 0.5 + 0.5 * Math.cos(Math.PI * (xs - (c.sylD - rl)) / rl) : 1; }
          const f = c.f * (1 + 0.015 * (1 - xs / c.sylD));
          c.phase += TAU * f / fs; if (c.phase > TAU) c.phase -= TAU;
          q = c.lp; q.y += q.a * ((Math.sin(c.phase) + 0.08 * Math.sin(2 * c.phase)) * e - q.y);
          const y = q.y * c.amp * M.crickets;
          L += y * c.gL * 0.4; R += y * c.gR * 0.4; sL += y * c.gL * S.crickets; sR += y * c.gR * S.crickets;
        }
      }

      const og = this.outG;
      dryL[i] = L * og; dryR[i] = R * og; sendL[i] = sL * og; sendR[i] = sR * og;
      this.n++;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Procedural outdoor IR (night clearing near trees): sparse early reflections + velvet-noise tail with
// 3-band frequency-dependent decay. Deterministic. Returns [Float32Array L, Float32Array R].
// ---------------------------------------------------------------------------------------------
export function makeOutdoorIR(fs, seed, o = {}) {
  const len = Math.round((o.length || 1.6) * fs);
  const T60 = o.t60 || [0.55, 0.85, 0.6];
  const out = [new Float32Array(len), new Float32Array(len)];
  for (let ch = 0; ch < 2; ch++) {
    const r = new RNG(deriveSeed(seed, 'ir' + ch));
    const er = new Float64Array(len), tail = new Float64Array(len);
    const nEr = o.nEarly || 10;
    for (let k = 0; k < nEr; k++) {
      const t = r.logRange(0.009, 0.14), i = Math.round(t * fs);
      const g = dB(-14 - 9 * Math.log2(t / 0.01) + 3 * r.gauss()) * (r.next() < 0.5 ? -1 : 1);
      if (i < len) er[i] += g;
    }
    let t = 0.012;
    while (t < len / fs) {
      const dens = 600 + 1900 * Math.min(1, t / 0.3), Td = 1 / dens;
      const i = Math.round((t + r.next() * Td) * fs);
      if (i < len) tail[i] = r.next() < 0.5 ? -1 : 1;
      t += Td;
    }
    const lpL = new Biquad().lp(400, 0.7, fs), hpH = new Biquad().hp(3000, 0.7, fs), bpM1 = new Biquad().hp(400, 0.7, fs), bpM2 = new Biquad().lp(3000, 0.7, fs);
    const lpE = new OnePole().set(4500, fs);
    const tailGain = dB(o.tailDb !== undefined ? o.tailDb : -24);
    for (let i = 0; i < len; i++) {
      const tt = i / fs;
      const fade = tt < 0.012 ? 0 : tt < 0.06 ? 0.5 - 0.5 * Math.cos(Math.PI * (tt - 0.012) / 0.048) : 1;
      const x = tail[i];
      const lo = lpL.tick(x) * Math.exp(-6.9078 * tt / T60[0]);
      const mi = bpM2.tick(bpM1.tick(x)) * Math.exp(-6.9078 * tt / T60[1]);
      const hi = hpH.tick(x) * Math.exp(-6.9078 * tt / T60[2]) * dB(-4);
      out[ch][i] = lpE.lp(er[i]) * 3 + (lo + mi + hi) * fade * tailGain;
    }
  }
  return out;
}
export const REVERB_DEFAULTS = { length: 1.6, t60: [0.55, 0.85, 0.6], tailDb: -24, nEarly: 10, crossFeed: 0.15 };

// ---------------------------------------------------------------------------------------------
// BS.1770 K-weighting (libebur128-style analog prototype -> any fs)
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

// ---------------------------------------------------------------------------------------------
// 4x oversampled soft clip, residual method (offline: master-offline.js softClipOS; live: LiveMaster). A memoryless
// soft clip at the base rate folds its harmonics above fs/2 back into the band (sine 12 dB over the knee: aliases at
// -13..-38 dB re the fundamental). Only the clip residual clip(x4) - x4 is computed at 4x and low-passed back.
export const OS_HALF = 24; // kernel support in base-rate samples on each side (= extra latency of the live clip: 2x)
let OS_H = null;
export function osKernel() { // h[k], k = 0..8*OS_HALF: low-pass at 0.46 x base fs (Kaiser beta 9), sum = 1 at the 4x rate
  if (OS_H) return OS_H;
  const C = 4 * OS_HALF, n = 2 * C + 1, h = new Float64Array(n), fc = 0.46 / 4, beta = 9;
  const i0 = (x) => { let s = 1, t = 1; for (let k = 1; k < 30; k++) { t *= (x / (2 * k)) ** 2; s += t; } return s; };
  let sum = 0;
  for (let k = 0; k < n; k++) {
    const m = k - C, sinc = m === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * m) / (Math.PI * m), r = m / C;
    h[k] = sinc * i0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / i0(beta); sum += h[k];
  }
  for (let k = 0; k < n; k++) h[k] /= sum;
  return (OS_H = h);
}

// ---------------------------------------------------------------------------------------------
// Streaming live master (used by the 'fire-master' worklet):
//   EQ (HP 30 Hz, +0.5 dB @ 220 Hz, -1.5 dB shelf @ 10 kHz) -> make-up -> slow K-weighted leveler (keeps
//   "closer = louder" but bounded) -> 0.46 fs FIR band-limit -> soft clip (4x oversampled residual) -> 18 kHz
//   band-limit -> lookahead limiter driven by a 4x true-peak estimate (16 taps/phase) -> hard safety clamp.
//   Preallocated, allocation-free; samples never exceed the ceiling. Latency 3 x OS_HALF + 1.5 ms lookahead (~3.1 ms).
// ---------------------------------------------------------------------------------------------
const TPT = 16, TPH = TPT / 2 - 1; // taps per phase; centre offset (taps cover j-7 .. j+8)
const TP_PHASES = (() => {
  const h = [];
  for (let ph = 1; ph < 4; ph++) {
    const f = ph / 4, row = new Float64Array(TPT); let s = 0;
    for (let k = 0; k < TPT; k++) {
      const x = (k - TPH) - f, u = (x + TPT / 2) / TPT;
      const w = 0.42 - 0.5 * Math.cos(TAU * u) + 0.08 * Math.cos(2 * TAU * u);
      row[k] = (Math.sin(Math.PI * x) / (Math.PI * x)) * w; s += row[k];
    }
    for (let k = 0; k < TPT; k++) row[k] /= s;
    h.push(row);
  }
  return h;
})();

export class LiveMaster {
  constructor(fs, o = {}) {
    this.fs = fs;
    this.ceil = dB(o.ceilingDb ?? -1.0);
    this.makeup = dB(o.makeupDb ?? 4.5);
    this.eq = [0, 1].map(() => ({
      h1: new Biquad().hp(30, 0.5412, fs), h2: new Biquad().hp(30, 1.3066, fs),
      // same tone as the offline master (render_audio.mjs: phoneBodyDb 0.5 after the statistical tuning; was 1.5 here)
      pk: new Biquad().peak(220, 0.9, o.phoneBodyDb ?? 0.5, fs), hs: new Biquad().highShelf(10000, 0.7, o.airDb ?? -1.5, fs),
    }));
    // leveler: K-weighted mean square (tau 0.4 s) -> soft-knee downward compression above levT LUFS
    const [s1, s2] = kWeightingCoefs(fs);
    this.kw = [0, 1].map(() => [new Biquad().setCoefs(s1.b[0], s1.b[1], s1.b[2], s1.a[0], s1.a[1]), new Biquad().setCoefs(1, -2, 1, s2.a[0], s2.a[1])]);
    this.levT = o.levelerThresholdLufs ?? -16; this.levR = o.levelerRatio ?? 2.5; this.levK = 8;
    this.kMs = 1 - Math.exp(-1 / (0.4 * fs)); this.lms = 0; this.levDb = 0; this.levG = 1;
    this.kLevA = 1 - Math.exp(-CR / (0.3 * fs)); this.kLevR = 1 - Math.exp(-CR / (1.5 * fs));
    // limiter
    this.look = Math.max(1, Math.round((o.lookMs ?? 1.5) * fs / 1000));
    this.lat = this.look + TPT - TPH - 1; // audio delay = lookahead + detector latency
    let P = 1; while (P < this.lat + 4) P <<= 1; this.P = P; this.M = P - 1;
    this.dL = new Float64Array(P); this.dR = new Float64Array(P);
    this.req = new Float64Array(P).fill(1); this.smin = new Float64Array(P).fill(1); this.dq = new Float64Array(P);
    this.head = 0; this.tail = 0; this.i = 0; this.box = this.look + 1; this.g = 1;
    this.aRel = 1 - Math.exp(-1 / ((o.releaseMs ?? 120) * 1e-3 * fs));
    this.kn = this.ceil * dB(-4.5); this.top = this.ceil * 1.12;
    this.det = this.ceil * dB(-0.2); // detection margin for the estimator's residual error
    // band-limit after the soft clip (4th-order Butterworth): keeps the 16-tap true-peak estimate accurate
    const fl = Math.min(18000, 0.4 * fs);
    this.lpf = [0, 1].map(() => [new Biquad().lp(fl, 0.5412, fs), new Biquad().lp(fl, 1.3066, fs)]);
    this.gainTarget = 1; this.gain = 1; this.kGain = 1 - Math.exp(-1 / (0.03 * fs)); this.makeupTarget = this.makeup;
    // oversampled soft clip state (rings of 128 >= 2 * OS_HALF + 5)
    this.osH = osKernel(); this.osM = 127; this.osGate = this.kn * 0.7; // -3 dB: intersample overs of band-limited audio are smaller
    this.xbL = new Float64Array(128); this.xbR = new Float64Array(128); this.rbL = new Float64Array(128); this.rbR = new Float64Array(128); this.loud = new Uint8Array(128);
    this.blL = new Float64Array(128); this.blR = new Float64Array(128); this.blG = new Float64Array(2 * OS_HALF + 1);
    for (let t = 0; t <= 2 * OS_HALF; t++) this.blG[t] = 4 * this.osH[4 * OS_HALF + 4 * (t - OS_HALF)];
    this.grMin = 1; // min limiter gain since the last meter read
  }
  setMakeupDb(db) { this.makeupTarget = dB(db); } // smoothed (30 ms) like the gain: no click when changed live
  // input-to-output delay in samples (FIR band-limit + oversampled clip + detector + lookahead)
  static latencySamples(fs, lookMs = 1.5) { return 3 * OS_HALF + Math.max(1, Math.round(lookMs * fs / 1000)) + TPT - TPH - 1; }
  // 4 sub-samples around base sample j: interpolate, soft clip, scatter the low-passed residual into rb
  _clipOS(xb, rb, j) {
    const h = this.osH, OM = this.osM, kn = this.kn, sc = this.top - this.kn, C = 4 * OS_HALF;
    for (let p = 0; p < 4; p++) {
      let v = 0;
      for (let t = -OS_HALF; t <= OS_HALF; t++) { const k = p - 4 * t + C; if (k >= 0 && k <= 2 * C) v += xb[(j + t) & OM] * 4 * h[k]; }
      const a = v < 0 ? -v : v; if (a <= kn) continue;
      const d = (v < 0 ? -1 : 1) * (kn + sc * Math.tanh((a - kn) / sc)) - v;
      for (let t = -OS_HALF; t <= OS_HALF; t++) { const k = 4 * t - p + C; if (k >= 0 && k <= 2 * C) rb[(j + t) & OM] += d * h[k]; }
    }
  }
  // Allocation-free (filters run through their x/y fields, see RB).
  process(inL, inR, outL, outR, n) {
    const M = this.M, look = this.look, kn = this.kn, top = this.top, ceil = this.ceil, dL = this.dL, dR = this.dR;
    const E0 = this.eq[0], E1 = this.eq[1], K0 = this.kw[0], K1 = this.kw[1], F0 = this.lpf[0], F1 = this.lpf[1];
    let q = null;
    for (let k = 0; k < n; k++) {
      const i = this.i;
      this.gain += this.kGain * (this.gainTarget - this.gain);
      this.makeup += this.kGain * (this.makeupTarget - this.makeup);
      const gm = this.makeup * this.gain;
      q = E0.h1; q.x = (inL ? inL[k] : 0) + DENORM; q.run(); E0.h2.x = q.y; q = E0.h2; q.run(); E0.pk.x = q.y; q = E0.pk; q.run(); E0.hs.x = q.y; q = E0.hs; q.run();
      let l = q.y * gm;
      q = E1.h1; q.x = (inR ? inR[k] : 0) + DENORM; q.run(); E1.h2.x = q.y; q = E1.h2; q.run(); E1.pk.x = q.y; q = E1.pk; q.run(); E1.hs.x = q.y; q = E1.hs; q.run();
      let r = q.y * gm;
      // leveler (detector before its own gain; gain computer at control rate)
      q = K0[0]; q.x = l; q.run(); K0[1].x = q.y; q = K0[1]; q.run(); const kl = q.y;
      q = K1[0]; q.x = r; q.run(); K1[1].x = q.y; q = K1[1]; q.run(); const kr = q.y;
      this.lms += this.kMs * (kl * kl + kr * kr - this.lms);
      if ((i & (CR - 1)) === 0) {
        const over = -0.691 + 10 * Math.log10(this.lms + 1e-20) - this.levT, s = 1 - 1 / this.levR, kh = this.levK / 2;
        const tgt = over > kh ? s * over : over > -kh ? s * (over + kh) * (over + kh) / (2 * this.levK) : 0;
        this.levDb += (tgt > this.levDb ? this.kLevA : this.kLevR) * (tgt - this.levDb);
        if (this.lms < 1e-30) this.lms = 0; if (this.levDb < 1e-30) this.levDb = 0; // no subnormals after long silence
        this.levG = Math.pow(10, -this.levDb / 20);
      }
      l *= this.levG; r *= this.levG;
      // band-limit to 0.46 fs (linear-phase FIR, OS_HALF samples of latency): the residual clip below is exact only
      // for band-limited input (content above 0.46 fs would pass it unclipped)
      const blL = this.blL, blR = this.blR, bg = this.blG; blL[i & 127] = l; blR[i & 127] = r;
      let sl = 0, sr = 0;
      for (let t = 0; t <= 2 * OS_HALF; t++) { const idx = (i - t) & 127, c = bg[t]; sl += blL[idx] * c; sr += blR[idx] * c; }
      l = sl; r = sr;
      // soft clip into the ceiling region, 4x oversampled residual (osKernel; adds 2 x OS_HALF samples of latency)
      const OM = this.osM, xbL = this.xbL, xbR = this.xbR, rbL = this.rbL, rbR = this.rbR, LF = this.loud, gate = this.osGate;
      xbL[i & OM] = l; xbR[i & OM] = r; LF[i & OM] = (l > gate || l < -gate || r > gate || r < -gate) ? 1 : 0;
      const jc = i - OS_HALF; // interpolation centre: needs x[jc - OS_HALF .. jc + OS_HALF] = x[i - 2 OS_HALF .. i]
      // sub-samples between jc and jc + 1 can only reach the knee near loud samples (a missed one is still caught by the limiter)
      if (LF[(jc - 1) & OM] | LF[jc & OM] | LF[(jc + 1) & OM] | LF[(jc + 2) & OM]) { this._clipOS(xbL, rbL, jc); this._clipOS(xbR, rbR, jc); }
      const oi = (i - 2 * OS_HALF) & OM; // final: no later centre reaches it
      l = xbL[oi] + rbL[oi]; r = xbR[oi] + rbR[oi]; rbL[oi] = 0; rbR[oi] = 0;
      q = F0[0]; q.x = l; q.run(); F0[1].x = q.y; q = F0[1]; q.run(); l = q.y;
      q = F1[0]; q.x = r; q.run(); F1[1].x = q.y; q = F1[1]; q.run(); r = q.y;
      dL[i & M] = l; dR[i & M] = r;
      // true-peak detector for sample j = i - (TPT - TPH - 1) and the interval [j, j+1]
      const j = i - (TPT - TPH - 1);
      let pk = Math.max(Math.abs(dL[j & M]), Math.abs(dR[j & M]));
      for (let ph = 0; ph < 3; ph++) {
        const h = TP_PHASES[ph]; let al = 0, ar = 0;
        for (let t = 0; t < TPT; t++) { const idx = (j - TPH + t) & M; al += h[t] * dL[idx]; ar += h[t] * dR[idx]; }
        al = al < 0 ? -al : al; ar = ar < 0 ? -ar : ar;
        if (al > pk) pk = al; if (ar > pk) pk = ar;
      }
      const rq = pk > this.det ? this.det / pk : 1;
      // lookahead limiter: sliding min of required gain over look+1 pushes, box-averaged (g <= req at the output sample)
      this.req[i & M] = rq;
      while (this.tail > this.head && this.req[this.dq[(this.tail - 1) & M] & M] >= rq) this.tail--;
      this.dq[this.tail & M] = i; this.tail++;
      if (this.dq[this.head & M] <= i - look - 1) this.head++;
      const sm = this.req[this.dq[this.head & M] & M];
      this.box += sm - this.smin[(i - look - 1) & M];
      this.smin[i & M] = sm;
      this.g = Math.min(this.box / (look + 1), this.g + this.aRel * (1 - this.g));
      if (this.g < this.grMin) this.grMin = this.g;
      const o = (i - this.lat) & M; // output sample d = j - look, whose requirement was pushed look samples ago
      let yl = dL[o] * this.g, yr = dR[o] * this.g;
      yl = yl > ceil ? ceil : yl < -ceil ? -ceil : yl; yr = yr > ceil ? ceil : yr < -ceil ? -ceil : yr;
      outL[k] = yl; if (outR) outR[k] = yr;
      this.i = i + 1; // double counter (exact to 2^53); `& M` wraps correctly because P divides 2^32
    }
  }
}
