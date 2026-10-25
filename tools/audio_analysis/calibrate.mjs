// calibrate.mjs — re-measures each layer of src/audio/fire-dsp.js and prints the CAL table.
// Continuous layers: mix 0 dB -> -20 LKu (ungated K-weighted level of the dry signal) at intensity 1, camDist = refDist (2 m).
// Pops / logs: mix 0 dB -> median per-event sample peak of -6 dBFS (size 1). Embers: median peak -20 dBFS.
// usage: node tools/audio_analysis/calibrate.mjs
import { FireAudioEngine, CAL } from '../../src/audio/fire-dsp.js';
import { kWeightingCoefs } from '../../src/audio/master-offline.js';
import { Biquad } from '../../src/audio/fire-dsp.js';

const sr = 48000, dur = 20, N = sr * dur;
const OLD = { ...CAL };
for (const k in CAL) CAL[k] = 1;
const zeroMix = Object.fromEntries(Object.keys(OLD).map((k) => [k, 0]));

function run(solo, params = {}, events = [], ctl = {}) {
  const e = new FireAudioEngine({ sampleRate: sr, seed: 7, solo, params: { mix: zeroMix, popRate: 0, emberRate: 0, crickets: 1, outputGainDb: 0, ...params } });
  e.setControls({ intensity: 1, hrr: 1, dhrr: 0, camDist: 2, camAz: 0, wind: 0, ...ctl });
  for (const ev of events) e.schedule(ev);
  const L = new Float32Array(N), R = new Float32Array(N), a = new Float32Array(N), b = new Float32Array(N);
  e.render(L, R, a, b, N); return [L, R];
}
function lku([L, R]) {
  const [s1, s2] = kWeightingCoefs(sr); let acc = 0;
  for (const x of [L, R]) {
    const f1 = new Biquad().setCoefs(...s1.b, ...s1.a), f2 = new Biquad().setCoefs(1, -2, 1, ...s2.a);
    for (let i = 0; i < N; i++) { const y = f2.tick(f1.tick(x[i])); if (i >= sr) acc += y * y; }
  }
  return -0.691 + 10 * Math.log10(acc / (N - sr));
}
function medianPeak([L, R], times, win = 0.25) {
  const p = times.map((t) => { let m = 0; const s = Math.round(t * sr); for (let i = s; i < Math.min(N, s + win * sr); i++) m = Math.max(m, Math.abs(L[i]), Math.abs(R[i])); return m; }).sort((a, b) => a - b);
  return 20 * Math.log10(p[p.length >> 1]);
}
const out = {}, meas = {};
for (const k of ['ticks', 'crackles', 'roar', 'breath', 'lapping', 'hiss', 'crickets']) { meas[k] = lku(run(k)); out[k] = Math.pow(10, (-20 - meas[k]) / 20); }
meas.whoosh = lku(run('whoosh', {}, [], { wind: 1 })); out.whoosh = Math.pow(10, (-20 - meas.whoosh) / 20);
meas.whump = lku(run('whump', { whumpForce: 1 })); out.whump = Math.pow(10, (-20 - meas.whump) / 20);
const popT = []; for (let t = 1; t < dur - 1; t += 1) popT.push(t);
meas.pops = medianPeak(run('pops', {}, popT.map((t) => ({ t, type: 'pop', size: 1, screenPan: 0 }))), popT); out.pops = Math.pow(10, (-6 - meas.pops) / 20);
const logT = []; for (let t = 1; t < dur - 2; t += 2.5) logT.push(t);
meas.logs = medianPeak(run('logs', {}, logT.map((t) => ({ t, type: 'log-shift', size: 1, screenPan: 0 }))), logT, 0.3); out.logs = Math.pow(10, (-6 - meas.logs) / 20);
const embT = []; for (let t = 1; t < dur - 1; t += 0.5) embT.push(t);
meas.embers = medianPeak(run('embers', {}, embT.map((t) => ({ t, type: 'ember', screenPan: 0, level: 1 }))), embT, 0.1); out.embers = Math.pow(10, (-20 - meas.embers) / 20);
for (const k in out) out[k] = +out[k].toPrecision(4);
console.log('measured (mix 0 dB, CAL 1):', JSON.stringify(Object.fromEntries(Object.entries(meas).map(([k, v]) => [k, +v.toFixed(2)]))));
console.log('previous CAL:', JSON.stringify(OLD));
console.log('export const CAL = ' + JSON.stringify(out).replace(/"/g, '').replace(/,/g, ', ').replace(/:/g, ': ') + ';');
