// Calibrates white balance + tone curve so blackbody soot emission reproduces the measured camera "fire ramp"
// (research/visual_reference.md §2.1, natural family). Usage: node tools/calib_tonemap.mjs
import { blackbodyXYZ, whiteBalance } from '../src/blackbody.js';

const NAT = [[0.07, '270C06'], [0.13, '421908'], [0.19, '5C2708'], [0.25, '6F350C'], [0.31, '844412'], [0.37, '9D5318'],
  [0.43, 'B5611E'], [0.49, 'CC6F27'], [0.55, 'DF7D35'], [0.61, 'EE8C46'], [0.67, 'F99D53'], [0.73, 'FEAF62'],
  [0.79, 'FEC27A'], [0.85, 'FFD595'], [0.91, 'FFE8B2'], [0.97, 'FEFCD2']].map(([l, h]) => [l, [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16))]);

const srgbEnc = v => (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
const luma = c => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
const mv = (m, v) => [0, 1, 2].map(i => m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]);

// AgX (same constants as the shader; GLSL column-major -> rows here are transposed)
const AGX = [[0.842479062253094, 0.0784335999999992, 0.0792237451477643], [0.0423282422610123, 0.878468636469772, 0.0791661274605434], [0.0423756549057051, 0.0784336, 0.879142973793104]];
const AGXI = [[1.19687900512017, -0.0980208811401368, -0.0990297440797205], [-0.0528968517574562, 1.15190312990417, -0.0989611768448433], [-0.0529716355144438, -0.0980434501171241, 1.15107367264116]];
function agx(c, power = 1, sat = 1) {
  let v = mv(AGX, c.map(x => Math.max(x, 1e-10)));
  v = v.map(x => (Math.min(Math.max(Math.log2(x), -12.47393), 4.026069) + 12.47393) / 16.5);
  v = v.map(x => { const x2 = x * x, x4 = x2 * x2; return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232; });
  // ASC CDL-style look (Blender "Punchy": power 1.35, sat 1.4)
  v = v.map(x => Math.pow(Math.max(x, 0), power));
  const l = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
  v = v.map(x => l + sat * (x - l));
  v = mv(AGXI, v);
  return v.map(x => Math.min(Math.max(x, 0), 1) * 255); // already display-encoded
}
// Per-channel filmic with a channel-crosstalk "inset" (path to white), then sRGB.
function filmicInset(c, inset = 0.1, shoulder = 1.0) {
  const m = [[1 - 2 * inset, inset, inset], [inset, 1 - 2 * inset, inset], [inset, inset, 1 - 2 * inset]];
  let v = mv(m, c);
  // Hill ACES fit curve (per channel)
  v = v.map(x => { x *= shoulder; return Math.min(Math.max((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0), 1); });
  return v.map(x => srgbEnc(x) * 255);
}
// Exponential per-channel (film-like) with inset.
function expo(c, inset = 0.08) {
  const m = [[1 - 2 * inset, inset, inset], [inset, 1 - 2 * inset, inset], [inset, inset, 1 - 2 * inset]];
  return mv(m, c).map(x => srgbEnc(1 - Math.exp(-Math.max(x, 0))) * 255);
}

function ramp(T, Twb, tm) {
  const toRGB = whiteBalance(Twb);
  const xyz = blackbodyXYZ(T, true);
  let rgb = toRGB(xyz).map(v => Math.max(v, 0));
  const Y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  rgb = rgb.map(v => v / Y);
  // sweep intensity, collect (luma, color)
  const pts = [];
  for (let e = -8; e <= 6; e += 0.01) pts.push(tm(rgb.map(v => v * Math.pow(2, e))));
  return pts;
}
function score(pts) {
  let err = 0; const rows = [];
  for (const [l, tgt] of NAT) {
    let best = null, bd = 1e9;
    for (const p of pts) { const d = Math.abs(luma(p) - l); if (d < bd) { bd = d; best = p; } }
    const e = Math.hypot(best[0] - tgt[0], best[1] - tgt[1], best[2] - tgt[2]);
    err += e; rows.push([l, tgt, best.map(Math.round), Math.round(e)]);
  }
  return { err: err / NAT.length, rows };
}

const results = [];
for (const T of [1400, 1500, 1600]) {
  for (const Twb of [0, 6500, 5000, 4000, 3500, 3000, 2500]) {
    results.push({ name: `agx T${T} wb${Twb}`, ...score(ramp(T, Twb, agx)) });
    for (const pw of [1.2, 1.35, 1.5]) for (const st of [1.0, 1.2, 1.4]) results.push({ name: `agx p${pw} s${st} T${T} wb${Twb}`, ...score(ramp(T, Twb, c => agx(c, pw, st))) });
    for (const inset of [0, 0.05, 0.1, 0.15]) results.push({ name: `aces+inset${inset} T${T} wb${Twb}`, ...score(ramp(T, Twb, c => filmicInset(c, inset))) });
    for (const inset of [0.03, 0.06, 0.1]) results.push({ name: `expo+inset${inset} T${T} wb${Twb}`, ...score(ramp(T, Twb, c => expo(c, inset))) });
  }
}
results.sort((a, b) => a.err - b.err);
for (const r of results.slice(0, 12)) console.log(r.err.toFixed(1), r.name);
const best = results[0];
console.log('\nbest:', best.name);
for (const [l, t, g, e] of best.rows) console.log(l.toFixed(2), 'target', t.join(','), 'got', g.join(','), 'err', e);
const agxRow = results.find(r => r.name === 'agx T1500 wb0');
console.log('\nagx T1500 no-WB:', agxRow.err.toFixed(1));
for (const [l, t, g, e] of agxRow.rows) console.log(l.toFixed(2), 'target', t.join(','), 'got', g.join(','), 'err', e);

// darkness check: output luma for a grey surface at 1%, 3%, 10% of the level where the flame ramp reaches luma 0.6
for (const [name, tm] of [['agx', c => agx(c)], ['agx punchy', c => agx(c, 1.35, 1.4)], ['agx p1.5', c => agx(c, 1.5, 1.2)], ['aces inset.1', c => filmicInset(c, 0.1)]]) {
  const pts = ramp(1500, 5000, tm); let x06 = 0;
  for (let e = -8; e <= 6; e += 0.01) { const c = tm(blackbodyXYZ(1500, true).map(() => 0)); }
  // find intensity where luma hits 0.6 for the 1500K source
  const toRGB = whiteBalance(5000); let rgb = toRGB(blackbodyXYZ(1500, true)).map(v => Math.max(v, 0)); const Y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]; rgb = rgb.map(v => v / Y);
  let k = 0; for (let e = -8; e <= 6; e += 0.01) { if (luma(tm(rgb.map(v => v * 2 ** e))) >= 0.6) { k = 2 ** e; break; } }
  const g = f => luma(tm([f * k, f * k, f * k])).toFixed(3);
  console.log(name, 'k', k.toFixed(3), 'grey@1%', g(0.01), '@3%', g(0.03), '@10%', g(0.1), '@30%', g(0.3));
}
