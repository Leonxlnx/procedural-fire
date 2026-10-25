#!/usr/bin/env node
// make_steady_timeline.mjs — a steady-state timeline (no scripted events) for statistical tuning, so the
// texture can be compared with the real recordings without the demo's cinematic events skewing the stats.
//   node tools/audio_analysis/make_steady_timeline.mjs <out.json> [--dur 30] [--intensity 1] [--dist 2] [--seed 7]
import fs from 'node:fs';
import { RNG, deriveSeed } from '../../src/audio/fire-dsp.js';

const args = process.argv.slice(2);
const out = args.find((a) => !a.startsWith('--')) || 'out/audio/tuning/steady_timeline.json';
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? +args[i + 1] : d; };
const fps = 60, dur = opt('dur', 30), I = opt('intensity', 1), dist = opt('dist', 2), seed = opt('seed', 7);
const r = new RNG(deriveSeed(seed, 'steady-hrr'));
let nb1 = 0, nb2 = 0;
const hrr = [];
for (let i = 0; i < dur * fps; i++) {
  const t = i / fps;
  nb1 += 0.25 * (r.gauss() - nb1); nb2 += 0.05 * (nb1 - nb2);
  hrr.push(Math.max(0, I * (1 + 0.06 * Math.sin(2 * Math.PI * 1.75 * t + 0.4 * Math.sin(2 * Math.PI * 0.21 * t)) + 0.12 * nb2)));
}
const frames = hrr.map((h, i) => ({
  intensity: I, hrr: +h.toFixed(4),
  dhrr: +((hrr[Math.min(hrr.length - 1, i + 1)] - hrr[Math.max(0, i - 1)]) * fps / 2).toFixed(4),
  camDist: dist, camAz: +(0.35 + 0.01 * i / fps).toFixed(4),
}));
fs.writeFileSync(out, JSON.stringify({ fps, duration: dur, seed, notes: 'steady-state tuning timeline, no events', frames, events: [] }));
console.log(out);
