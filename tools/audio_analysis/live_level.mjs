// live_level.mjs — simulates the live graph (engine -> reverb -> LiveMaster) in Node and reports loudness/peaks
// for a few steady scenarios, to choose the live make-up gain.  usage: node tools/audio_analysis/live_level.mjs [makeupDb]
import { FireAudioEngine, LiveMaster, REVERB_DEFAULTS } from '../../src/audio/fire-dsp.js';
import { mixWithReverb, integratedLoudness, truePeakDb } from '../../src/audio/master-offline.js';
const sr = 48000, N = sr * 12, makeup = +(process.argv[2] ?? 4.5);
const scenes = [
  { name: 'default I=1 d=2.2', c: { intensity: 1, camDist: 2.2 } },
  { name: 'small I=0.5 d=2.2', c: { intensity: 0.5, camDist: 2.2 } },
  { name: 'close I=1.2 d=1.0', c: { intensity: 1.2, camDist: 1.0 } },
  { name: 'far I=1 d=6', c: { intensity: 1, camDist: 6 } },
  { name: 'max I=1.5 d=0.4', c: { intensity: 1.5, camDist: 0.4 } },
];
for (const s of scenes) {
  const e = new FireAudioEngine({ sampleRate: sr, seed: 5 });
  e.setControls({ hrr: 1, dhrr: 0, camAz: 0, ...s.c });
  const b = [0, 1, 2, 3].map(() => new Float32Array(N));
  for (let i = 0; i < N; i += 128) e.render(b[0].subarray(i, i + 128), b[1].subarray(i, i + 128), b[2].subarray(i, i + 128), b[3].subarray(i, i + 128), 128);
  const [L, R] = mixWithReverb(b[0], b[1], b[2], b[3], sr, 5, REVERB_DEFAULTS);
  const pre = integratedLoudness([L, R], sr);
  const m = new LiveMaster(sr, { makeupDb: makeup, ceilingDb: -1 });
  const oL = new Float32Array(N), oR = new Float32Array(N);
  for (let i = 0; i < N; i += 128) m.process(L.subarray(i, i + 128), R.subarray(i, i + 128), oL.subarray(i, i + 128), oR.subarray(i, i + 128), 128);
  let pk = 0, gr = 0; for (let i = 0; i < N; i++) pk = Math.max(pk, Math.abs(oL[i]), Math.abs(oR[i]));
  console.log(s.name.padEnd(20), 'pre', pre.toFixed(1), 'LUFS -> live', integratedLoudness([oL, oR], sr).toFixed(1), 'LUFS, sample peak', (20 * Math.log10(pk)).toFixed(2), 'dBFS, TP', truePeakDb([oL, oR]).toFixed(2));
}
