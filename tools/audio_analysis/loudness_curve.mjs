#!/usr/bin/env node
// loudness_curve.mjs — momentary (400 ms) loudness and band levels of a WAV every --hop s (default 0.2), one line each.
//   node tools/audio_analysis/loudness_curve.mjs <file.wav> [--hop 0.2] [--t0 0] [--t1 end]
import fs from 'node:fs';
import { Biquad, kWeightingCoefs } from '../../src/audio/fire-dsp.js';

const args = process.argv.slice(2), val = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? +args[i + 1] : d; };
const b = fs.readFileSync(args[0]), dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
let o = 12, fmt = null, data = null;
while (o + 8 <= b.length) { const id = b.toString('ascii', o, o + 4), sz = dv.getUint32(o + 4, true); if (id === 'fmt ') fmt = { ch: dv.getUint16(o + 10, true), sr: dv.getUint32(o + 12, true), bits: dv.getUint16(o + 22, true) }; if (id === 'data') data = { off: o + 8, size: sz }; o += 8 + sz + (sz & 1); }
const bps = fmt.bits / 8, N = Math.floor(data.size / (bps * fmt.ch)), sr = fmt.sr, ch = [new Float64Array(N), new Float64Array(N)];
for (let i = 0; i < N; i++) for (let c = 0; c < Math.min(2, fmt.ch); c++) { const p = data.off + (i * fmt.ch + c) * bps; let v; if (fmt.bits === 24) { v = b[p] | (b[p + 1] << 8) | (b[p + 2] << 16); if (v & 0x800000) v -= 0x1000000; v /= 8388608; } else if (fmt.bits === 16) v = dv.getInt16(p, true) / 32768; else v = dv.getFloat32(p, true); ch[c][i] = v; }
if (fmt.ch === 1) ch[1] = ch[0];
const cum = (f) => { const c = new Float64Array(N + 1); for (let i = 0; i < N; i++) c[i + 1] = c[i] + f[i]; return c; };
const [s1, s2] = kWeightingCoefs(sr), kq = new Float64Array(N);
for (const x of ch) { const f1 = new Biquad().setCoefs(s1.b[0], s1.b[1], s1.b[2], s1.a[0], s1.a[1]), f2 = new Biquad().setCoefs(1, -2, 1, s2.a[0], s2.a[1]); for (let i = 0; i < N; i++) { const y = f2.tick(f1.tick(x[i])); kq[i] += y * y; } }
const bandSq = (lo, hi) => { const y = new Float64Array(N); for (const x of ch) { const fs = []; if (lo) fs.push(new Biquad().hp(lo, 0.5412, sr), new Biquad().hp(lo, 1.3066, sr)); if (hi) fs.push(new Biquad().lp(hi, 0.5412, sr), new Biquad().lp(hi, 1.3066, sr)); for (let i = 0; i < N; i++) { let v = x[i]; for (const f of fs) v = f.tick(v); y[i] += 0.5 * v * v; } } return y; };
const K = cum(kq), SUB = cum(bandSq(0, 100)), MID = cum(bandSq(100, 1000)), HI = cum(bandSq(1000, 5000));
const hop = val('hop', 0.2), t0 = val('t0', 0), t1 = val('t1', N / sr), w = Math.round(0.4 * sr);
const d = (c, a, z) => (10 * Math.log10((c[z] - c[a]) / (z - a) + 1e-30)).toFixed(1).padStart(6);
console.log('   t    LUFS(400ms)  sub<100  100-1k  1k-5k  (dB re full scale, window starts at t)');
for (let t = t0; t + 0.4 <= t1 + 1e-9; t += hop) { const a = Math.round(t * sr), z = Math.min(N, a + w); console.log(t.toFixed(2).padStart(6), (-0.691 + 10 * Math.log10((K[z] - K[a]) / (z - a) + 1e-30)).toFixed(1).padStart(10), d(SUB, a, z), d(MID, a, z), d(HI, a, z)); }
