#!/usr/bin/env node
// bitexact.mjs — regression guard for refactors of src/audio/fire-dsp.js that must not change a single output bit.
// Renders a set of scenarios (offline demo timeline, live-style controls + every event type at 44.1 kHz, jumpy
// controls, LiveMaster) and prints/compares SHA-256 hashes.
//   node tools/audio_analysis/bitexact.mjs --save ref.json      (before the change)
//   node tools/audio_analysis/bitexact.mjs --check ref.json     (after; exit 1 on any difference)
import fs from 'node:fs';
import crypto from 'node:crypto';
import { FireAudioEngine, LiveMaster } from '../../src/audio/fire-dsp.js';
import { makeDemoTimeline } from '../render_audio.mjs';

const args = process.argv.slice(2);
const val = (k) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : null; };
const H = (...as) => { const h = crypto.createHash('sha256'); for (const a of as) h.update(Buffer.from(a.buffer, a.byteOffset, a.byteLength)); return h.digest('hex').slice(0, 20); };
function run(e, secs, block, sched) {
  const N = Math.round(secs * e.fs), b = [0, 1, 2, 3].map(() => new Float32Array(N));
  for (let i = 0; i < N; i += block) { const n = Math.min(block, N - i); if (sched) sched(e, i); e.render(b[0].subarray(i, i + n), b[1].subarray(i, i + n), b[2].subarray(i, i + n), b[3].subarray(i, i + n), n); }
  return b;
}
const logHash = (e) => { const l = []; e.log.read(0, l); return crypto.createHash('sha256').update(JSON.stringify(l)).digest('hex').slice(0, 12); };
const res = {};

{ // a) offline demo timeline (the video soundtrack path)
  const tl = makeDemoTimeline(), e = new FireAudioEngine({ sampleRate: 48000, seed: tl.seed });
  e.setTimeline({ fps: tl.fps, frames: tl.frames }); for (const ev of tl.events) e.schedule(ev);
  const b = run(e, 20, 128); res.demo = H(...b) + ':' + logHash(e);
}
{ // b) live-style at 44.1 kHz: every event type, random pops, wind, whoosh; controls updated every 735 samples
  const e = new FireAudioEngine({ sampleRate: 44100, seed: 77, params: { popRate: 2 } });
  e.setControls({ intensity: 1.1, hrr: 1, camDist: 1.4, camAz: 0.6, wind: 0.3 });
  ['ignite', 'pop', 'crackle-burst', 'log-shift', 'whoosh', 'gust', 'ember', 'flare'].forEach((type, k) => e.schedule({ t: 0.2 + 0.7 * k, type, strength: 0.9, pan: -0.5 + 0.15 * k }));
  let ph = 0;
  const b = run(e, 7, 128, (eng, i) => { if (i % 735 < 128) { ph += 0.1; eng.setControls({ intensity: 1.1 + 0.3 * Math.sin(ph), hrr: 1 + 0.4 * Math.sin(3 * ph), dhrr: 1.2 * Math.cos(3 * ph), camDist: 1.4 + 0.5 * Math.sin(0.5 * ph), camAz: 0.6 + 0.2 * ph }); } });
  res.live44k = H(...b) + ':' + logHash(e);
  // d) LiveMaster on that output (makeup 4.5 dB, leveler, soft clip, TP limiter), odd block size
  const m = new LiveMaster(44100, { ceilingDb: -1, makeupDb: 4.5 }), N = b[0].length, oL = new Float32Array(N), oR = new Float32Array(N);
  const g = new Float32Array(N); for (let i = 0; i < N; i++) g[i] = 4; // drive it hard
  for (let i = 0; i < N; i += 100) { const n = Math.min(100, N - i), L = b[0].subarray(i, i + n).map((v, k) => v * g[i + k]), R = b[1].subarray(i, i + n).map((v, k) => v * g[i + k]); m.process(L, R, oL.subarray(i, i + n), oR.subarray(i, i + n), n); }
  res.master44k = H(oL, oR) + ':' + m.levDb.toFixed(12) + ':' + m.grMin.toFixed(12);
}
{ // c) jumpy controls at 48 kHz (intensity steps, camera jumps, dhrr spikes -> whumps), block 997
  const e = new FireAudioEngine({ sampleRate: 48000, seed: 5 });
  let k = 0;
  const b = run(e, 6, 997, (eng) => { k++; eng.setControls({ intensity: k % 3 ? 1.5 : 0.2, hrr: k % 4 < 2 ? 0.4 : 2.5, dhrr: k % 5 === 0 ? 8 : -1, camDist: k % 2 ? 0.4 : 6, camAz: k * 0.7 }); if (k % 40 === 0) eng.schedule({ type: 'whoosh', strength: 1 }); });
  res.jumpy = H(...b) + ':' + logHash(e);
}
{ // e) silence: intensity 0, no crickets, long run
  const e = new FireAudioEngine({ sampleRate: 48000, seed: 9, params: { crickets: 0 } }); e.setControls({ intensity: 0, hrr: 0 });
  res.silence = H(...run(e, 3, 512));
}
console.log(JSON.stringify(res, null, 1));
if (val('save')) fs.writeFileSync(val('save'), JSON.stringify(res, null, 1));
if (val('check')) {
  const ref = JSON.parse(fs.readFileSync(val('check'), 'utf8'));
  const diff = Object.keys(ref).filter((k) => ref[k] !== res[k]);
  console.log(diff.length ? 'DIFFERENT: ' + diff.join(', ') : 'BIT-EXACT: all ' + Object.keys(ref).length + ' scenarios identical');
  process.exit(diff.length ? 1 : 0);
}
