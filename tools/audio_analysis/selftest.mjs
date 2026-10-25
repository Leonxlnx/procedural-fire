// selftest.mjs — quick functional checks of src/audio/fire-dsp.js (no audio device needed).
// usage: node tools/audio_analysis/selftest.mjs
import crypto from 'node:crypto';
import { FireAudioEngine, LiveMaster, EVENT_TYPES } from '../../src/audio/fire-dsp.js';

const sr = 48000;
const out = { ok: true, checks: {} };
const check = (name, pass, info = {}) => { out.checks[name] = { pass: !!pass, ...info }; if (!pass) out.ok = false; };
function run(e, secs, block = 128) {
  const N = Math.round(secs * sr), b = [0, 1, 2, 3].map(() => new Float32Array(N));
  for (let i = 0; i < N; i += block) { const n = Math.min(block, N - i); e.render(b[0].subarray(i, i + n), b[1].subarray(i, i + n), b[2].subarray(i, i + n), b[3].subarray(i, i + n), n); }
  return b;
}
const finite = (bs) => bs.every((a) => a.every(Number.isFinite));
const hash = (bs) => { const h = crypto.createHash('sha256'); for (const a of bs) h.update(Buffer.from(a.buffer)); return h.digest('hex').slice(0, 16); };
const logOf = (e) => { const l = []; e.log.read(0, l); return l; };

// 1) every event type renders, is logged with the right name, at the exact scheduled sample
{
  const e = new FireAudioEngine({ sampleRate: sr, seed: 3 });
  e.setControls({ intensity: 1, camDist: 2, camAz: 0 });
  EVENT_TYPES.forEach((type, k) => e.schedule({ t: 0.25 + 0.4 * k, type, strength: 0.8, pan: 0.3, id: k }));
  e.schedule({ t: 3.5, type: 'log_shift', strength: 1 }); e.schedule({ t: 3.6, type: 'crackleBurst', count: 12 });
  const ok = e.schedule({ t: 1, type: 'nope' }) === false;
  const b = run(e, 4.2);
  const log = logOf(e).filter((x) => x.source === 'scripted');
  const types = log.map((x) => x.type);
  const exact = log.filter((x) => x.id >= 0).every((x) => x.n === Math.round((0.25 + 0.4 * x.id) * sr));
  check('all_event_types', finite(b) && ok && EVENT_TYPES.filter((t) => t !== 'ember').every((t) => types.includes(t)) && exact, { types, unknownRejected: ok, sampleExact: exact });
}
// 2) block-size independence incl. block = 1
{
  const mk = () => { const e = new FireAudioEngine({ sampleRate: sr, seed: 11 }); e.setControls({ intensity: 1.2, hrr: 1.3, dhrr: 2.5, camDist: 1.2, camAz: 0.7 }); e.schedule({ t: 0.3, type: 'ignite' }); e.schedule({ t: 0.9, type: 'pop', strength: 1 }); return e; };
  const hs = [1, 128, 517].map((bl) => hash(run(mk(), 1.5, bl)));
  check('block_size_independent', hs[0] === hs[1] && hs[1] === hs[2], { hashes: hs });
}
// 3) camera azimuth: an event at x = +0.3 m pans right from az = 0 and left from az = pi (camera on the other side)
{
  const pans = [0, Math.PI].map((az) => { const e = new FireAudioEngine({ sampleRate: sr, seed: 5 }); e.setControls({ intensity: 1, camDist: 1.5, camAz: az }); e.schedule({ t: 0.1, type: 'pop', x: 0.3, z: 0 }); run(e, 0.3); return logOf(e).find((x) => x.source === 'scripted').stereoPan; });
  const sp = (() => { const e = new FireAudioEngine({ sampleRate: sr, seed: 5 }); e.setControls({ intensity: 1 }); e.schedule({ t: 0.1, type: 'pop', screenPan: -0.7 }); run(e, 0.3); return logOf(e).find((x) => x.source === 'scripted').stereoPan; })();
  check('azimuth_panning', pans[0] > 0.1 && pans[1] < -0.1 && Math.abs(sp + 0.7) < 1e-9, { az0: pans[0], azPi: pans[1], screenPan: sp });
}
// 4) intensity 0 + no crickets = silence (and no denormal slowdown)
{
  const e = new FireAudioEngine({ sampleRate: sr, seed: 2, params: { crickets: 0 } }); e.setControls({ intensity: 0, hrr: 0 });
  const t0 = performance.now(); const b = run(e, 2); const ms = performance.now() - t0;
  let pk = 0; for (const a of b) for (const v of a) pk = Math.max(pk, Math.abs(v));
  check('silent_at_zero_intensity', pk < 1e-9, { peak: pk, msFor2s: Math.round(ms) });
}
// 5) queue overflow is bounded and reported, never throws
{
  const e = new FireAudioEngine({ sampleRate: sr, seed: 9 }); e.setControls({ intensity: 1 });
  for (let i = 0; i < 1500; i++) e.schedule({ t: 0.5 + i * 1e-4, type: 'crackle-burst', strength: 1 });
  let threw = false; try { run(e, 1); } catch { threw = true; }
  check('queue_overflow_safe', !threw && e.queue.dropped > 0, { dropped: e.queue.dropped });
}
// 6) random pops are logged as random with positions inside the fire bed
{
  const e = new FireAudioEngine({ sampleRate: sr, seed: 21, params: { popRate: 5 } }); e.setControls({ intensity: 1 });
  run(e, 3);
  const pops = logOf(e).filter((x) => x.type === 'pop' && x.source === 'random');
  const inside = pops.every((p) => Math.hypot(p.x, p.z) <= 0.38 + 1e-6 && p.strength >= 0 && p.strength <= 1);
  check('random_pops_logged', pops.length >= 5 && inside, { count: pops.length, example: pops[0] });
}
// 7) dhrr onset triggers a whump (low-frequency energy rises when hrr jumps)
{
  const lf = (dh) => { const e = new FireAudioEngine({ sampleRate: sr, seed: 4, params: { mix: { ticks: -300, crackles: -300, hiss: -300, breath: -300 } } }); e.setControls({ intensity: 1, hrr: 1, dhrr: dh }); const b = run(e, 1.5); let s = 0; for (let i = sr / 2; i < b[0].length; i++) s += b[0][i] * b[0][i]; return 10 * Math.log10(s); };
  const d0 = lf(0), d4 = lf(4);
  check('dhrr_drives_whump', d4 - d0 > 1, { lfEnergyDb_dhrr0: +d0.toFixed(1), lfEnergyDb_dhrr4: +d4.toFixed(1) });
}
// 8) live master never exceeds its ceiling, even when fed +12 dBFS noise
{
  const m = new LiveMaster(sr, { ceilingDb: -1, makeupDb: 4.5 }), N = sr, L = new Float32Array(N), R = new Float32Array(N), oL = new Float32Array(N), oR = new Float32Array(N);
  let s = 1; for (let i = 0; i < N; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; L[i] = 4 * (s / 2 ** 32 - 0.5) * 2; R[i] = -L[i]; }
  for (let i = 0; i < N; i += 128) m.process(L.subarray(i, i + 128), R.subarray(i, i + 128), oL.subarray(i, i + 128), oR.subarray(i, i + 128), 128);
  let pk = 0; for (let i = 0; i < N; i++) pk = Math.max(pk, Math.abs(oL[i]), Math.abs(oR[i]));
  check('live_master_ceiling', 20 * Math.log10(pk) <= -1 + 1e-6, { peakDb: +(20 * Math.log10(pk)).toFixed(3) });
}
console.log(JSON.stringify(out, null, 1));
process.exit(out.ok ? 0 : 1);
