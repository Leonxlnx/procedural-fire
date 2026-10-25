// stems.mjs — per-layer momentary loudness over a time range of a timeline (pre-master, dry + reverb),
// to check the balance of layers without listening.
// usage: node tools/audio_analysis/stems.mjs <timeline.json|--demo> [--t0 0] [--t1 3] [--hop 0.1] [--win 0.4] [--layers a,b,c]
import fs from 'node:fs';
import { FireAudioEngine, REVERB_DEFAULTS } from '../../src/audio/fire-dsp.js';
import { mixWithReverb, kWeightingCoefs } from '../../src/audio/master-offline.js';
import { Biquad } from '../../src/audio/fire-dsp.js';
import { makeDemoTimeline } from '../render_audio.mjs';

const a = process.argv.slice(2), opt = {};
for (let i = 0; i < a.length; i++) if (a[i].startsWith('--')) opt[a[i].slice(2)] = a[i + 1] && !a[i + 1].startsWith('--') ? a[++i] : true; else opt.file = a[i];
const tl = opt.demo || !opt.file ? makeDemoTimeline() : JSON.parse(fs.readFileSync(opt.file, 'utf8'));
const sr = 48000, t0 = +(opt.t0 ?? 0), t1 = +(opt.t1 ?? 3), hop = +(opt.hop ?? 0.1), win = +(opt.win ?? 0.4);
const layers = (opt.layers || 'all,ticks,crackles,roar,breath,lapping,hiss,whump,pops,logs,embers,whoosh,crickets').split(',');
const N = Math.round(t1 * sr);

function render(solo) {
  const e = new FireAudioEngine({ sampleRate: sr, seed: tl.seed ?? 1, params: tl.params || {}, solo: solo === 'all' ? null : solo });
  e.setTimeline({ fps: tl.fps, frames: tl.frames });
  for (const ev of tl.events || []) e.schedule(ev);
  const b = [0, 1, 2, 3].map(() => new Float32Array(N));
  e.render(b[0], b[1], b[2], b[3], N);
  return mixWithReverb(b[0], b[1], b[2], b[3], sr, tl.seed ?? 1, REVERB_DEFAULTS);
}
function momentary([L, R]) {
  const [s1, s2] = kWeightingCoefs(sr), p = new Float64Array(N);
  for (const x of [L, R]) {
    const f1 = new Biquad().setCoefs(...s1.b, ...s1.a), f2 = new Biquad().setCoefs(1, -2, 1, ...s2.a);
    for (let i = 0; i < N; i++) { const y = f2.tick(f1.tick(x[i])); p[i] += y * y; }
  }
  const cs = new Float64Array(N + 1); for (let i = 0; i < N; i++) cs[i + 1] = cs[i] + p[i];
  const out = [], w = Math.round(win * sr);
  for (let t = t0; t + win <= t1 + 1e-9; t += hop) { const s = Math.round(t * sr); out.push(-0.691 + 10 * Math.log10((cs[s + w] - cs[s]) / w + 1e-20)); }
  return out;
}
const rows = [];
for (const l of layers) {
  const [L, R] = render(l); let pk = 0; for (let i = Math.round(t0 * sr); i < N; i++) pk = Math.max(pk, Math.abs(L[i]), Math.abs(R[i]));
  rows.push([l.padEnd(9), 'pk ' + (20 * Math.log10(pk + 1e-12)).toFixed(1).padStart(6), momentary([L, R]).map((v) => (v < -99 ? ' -inf' : v.toFixed(0).padStart(5))).join('')]);
}
const head = [];
for (let t = t0; t + win <= t1 + 1e-9; t += hop) head.push((t + win / 2).toFixed(1).padStart(5));
console.log('layer     peakdBFS  momentary LUFS (window centre, s):');
console.log(' '.repeat(19) + head.join(''));
for (const r of rows) console.log(r.join(' '));
