// The final 20 s cinematic in one command: renders the picture (tools/render_video.mjs), builds the soundtrack and
// muxes the deliverable (tools/finish_video.mjs). Runs the same on this Windows machine and on a Linux GPU pod
// (see tools/runpod/README.md).
// Usage: node tools/render_final.mjs [--preview] [--horizontal] [--aa 3] [--preroll 5] [--q hero|cine] [--out out/video/fire]
//   default    1080x1920, hero quality: ~3 h on the local integrated GPU, roughly 15-25 min on an RTX 4090
//   --preview  540x960, high quality, 1 sample per sub-step: a quick end-to-end check of picture + sound
//   --q cine   finer grid (5.4 mm cells) for a big GPU; untested look, compare with hero first
import { spawnSync } from 'node:child_process';

const A = {};
for (let i = 2; i < process.argv.length; i++) {
  if (!process.argv[i].startsWith('--')) continue;
  const v = process.argv[i + 1];
  A[process.argv[i].slice(2)] = v === undefined || v.startsWith('--') ? '1' : v;
}

// Flame look chosen from the A/B tests in out/lens (flame-shape plan: out/review/flame_shape_plan.json):
// anchored pyrolysis sources with dark gaps, a laminar flame base (hot-gas viscosity + age-gated forcing),
// coarser sub-grid forcing, a clamped B-spline prefilter and a partial flame-sheet edge.
export const FINAL = {
  sim: { visc: 2, lamAge: 0.2, anchorFreq: 1, srcFloor: -0.3, anchorGain: 2.7, fuelSrc: 0.15, vortEps: 4, turbAmp: 0.4, turbFreq: 0.7, turbEvolve: 0.28 },
  // lightK 0.6: close-ups showed the charred logs and the ground over-lit (golden char) next to reference photos
  render: { cubic: 1, flameSharp: 1, detailAmt: 0.3, sootEdge: 0.12, lightK: 0.6 },
  shutter: 0.25,   // 90 degrees: phone-like crisp flame tongues
};

const preview = !!A.preview;
const horizontal = !!A.horizontal;          // 16:9 landscape (same vertical field of view per shot)
const out = A.out || `out/video/final${horizontal ? '_16x9' : ''}${preview ? '_preview' : ''}`;
const q = A.q || (preview ? 'high' : 'hero');
const simDt = q === 'cine' ? 1 / 360 : 1 / 240;
let [w, h] = preview ? [540, 960] : [1080, 1920];
if (horizontal) [w, h] = [h, w];
const args = ['tools/render_video.mjs',
  '--w', String(w), '--h', String(h), '--q', q, '--frames', A.frames || '600',
  '--aa', A.aa || (preview ? '1' : '3'), '--shutter', String(FINAL.shutter), '--simDt', String(simDt), '--dof',
  '--stills', preview ? '60' : '30', '--render', JSON.stringify(FINAL.render), '--sim', JSON.stringify(FINAL.sim),
  ...(A.preroll ? ['--preroll', A.preroll] : []), '--out', `${out}_raw.mp4`];
const run = (a) => { console.log('>', 'node', a.join(' ')); const r = spawnSync('node', a, { stdio: 'inherit' }); if (r.status !== 0) process.exit(r.status || 1); };
run(args);
run(['tools/finish_video.mjs', '--video', `${out}_raw.mp4`, '--crf', '17']);
