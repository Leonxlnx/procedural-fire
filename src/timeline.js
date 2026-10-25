// The 20 s cinematic: shots (camera + lens + exposure), fire intensity curve and scripted events.
// Shared by the page (rendering) and the audio renderer, so picture and sound use the same cue sheet.
// Fire centred at the origin, flames ~1 m tall. One continuous simulation, hard cuts between shots;
// every cut sits on a frame boundary and is followed one frame later by a pop (cuts land on sound).

export const FPS = 30;
export const DURATION = 20;
export const PREROLL = 5.0;   // seconds of simulation before t = 0: frame 0 is already a full, moving fire

// Shots: [t0, t1) with start/end camera, f-number and exposure offset (EV). Cut times are multiples of 1/30 s.
const SHOTS = [
  // S1 hook: close on the flame roots, full burn at frame 0; hero pop at 0.30 s throws sparks past the lens
  { t0: 0.0, t1: 2.4, a: { pos: [0.62, 0.30, 1.05], target: [0, 0.40, 0], fov: 42 }, b: { pos: [0.55, 0.28, 0.93], target: [0, 0.38, 0], fov: 42 }, N: 8.0, ev: -0.3 },
  // S2 crane up the tongues into the rising sparks
  { t0: 2.4, t1: 5.6, a: { pos: [0.45, 0.25, 0.95], target: [0, 0.30, 0], fov: 38 }, b: { pos: [0.40, 1.00, 0.95], target: [0, 1.05, 0], fov: 38 }, N: 7.0, ev: 0.0 },
  // S3 low across the ember bed: coals breathing, blue flamelets, firelit stones
  { t0: 5.6, t1: 8.4, a: { pos: [-0.56, 0.16, -0.05], target: [0, 0.12, 0], fov: 50 }, b: { pos: [-0.48, 0.15, -0.04], target: [0, 0.11, 0], fov: 50 }, N: 8.0, ev: -0.9 },
  // S4 full-fire hero: log shift at 9.0 s, gust across the frame 11.0-12.6 s. Seen through the gap between
  // teepee logs 3 and 4 (azimuth -36 deg): from straight ahead log 3 points at the lens and reads as a post.
  { t0: 8.4, t1: 13.4, a: { pos: [-0.94, 0.60, 1.29], target: [0, 0.40, 0], fov: 46 }, b: { pos: [-0.835, 0.55, 1.149], target: [0, 0.40, 0], fov: 46 }, N: 11.0, ev: 0.0 },
  // S5 scale beat: low angle looking up the flame column
  { t0: 13.4, t1: 16.6, a: { pos: [0.75, 0.10, 0.95], target: [0, 0.68, 0], fov: 50 }, b: { pos: [0.80, 0.12, 0.88], target: [0, 0.72, 0], fov: 50 }, N: 8.0, ev: -0.2 },
  // S6 push-in that lands on the S1 framing (loops back to frame 0)
  { t0: 16.6, t1: 20.0, a: { pos: [0.90, 0.38, 1.55], target: [0, 0.45, 0], fov: 42 }, b: { pos: [0.62, 0.30, 1.05], target: [0, 0.40, 0], fov: 42 }, N: 8.0, ev: -0.3 },
];

// Fire intensity (fuel release) over time; t < 0 is pre-roll.
const INTENSITY = [
  [-PREROLL, 1.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.12], [6.0, 1.12], [8.6, 1.0], [9.0, 1.2], [9.6, 1.05],
  [11.0, 1.08], [12.6, 1.08], [14.0, 1.05], [18.3, 1.05], [18.5, 1.15], [19.2, 1.0], [20.0, 1.0],
];

// Scripted events (audio + sparks + sim bursts). `aim` throws a hero burst towards a direction (the lens).
const SCRIPTED = [
  { t: 0.30, type: 'pop', strength: 1.0, pos: [0.06, 0.14, 0.12], aim: [0.5, 0, 0.86] },
  { t: 2.433, type: 'pop', strength: 0.5, pos: [-0.08, 0.12, 0.10] },
  { t: 4.6, type: 'pop', strength: 0.6, pos: [0.10, 0.16, 0.02] },
  { t: 5.633, type: 'pop', strength: 0.45, pos: [-0.05, 0.10, 0.12] },
  { t: 7.2, type: 'crackle-burst', strength: 0.5, pos: [-0.10, 0.08, 0.06] },
  { t: 8.433, type: 'pop', strength: 0.5, pos: [0.06, 0.12, 0.10] },
  { t: 9.0, type: 'log-shift', strength: 1.0, pos: [0.05, 0.12, 0.0], log: 3 },
  { t: 11.0, type: 'gust', strength: 1.0, pos: [0, 0.3, 0], dur: 1.6 },
  { t: 13.433, type: 'pop', strength: 0.5, pos: [0.08, 0.13, -0.05] },
  { t: 15.2, type: 'pop', strength: 0.85, pos: [-0.06, 0.15, 0.08] },
  { t: 16.633, type: 'pop', strength: 0.5, pos: [0.02, 0.12, 0.12] },
  { t: 18.5, type: 'pop', strength: 1.0, pos: [0.06, 0.14, 0.12], aim: [0.5, 0, 0.86] },
];

// Deterministic background pops (resin/steam pockets): a clustered Poisson process.
// Both the picture (spark bursts) and the soundtrack use these exact times.
function randomPops(seed = 20261025) {
  let st = seed >>> 0;
  const rnd = () => { st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
  const out = [];
  let t = 0.6;
  while (t < DURATION - 0.3) {
    t += -Math.log(1 - rnd()) / 0.8;          // ~0.8 pops/s
    if (t >= DURATION - 0.3) break;
    const strength = Math.min(0.6, 0.12 + 0.45 * Math.pow(rnd(), 2.2));
    const a = rnd() * Math.PI * 2, r = Math.sqrt(rnd()) * 0.2;
    out.push({ t: +t.toFixed(3), type: 'pop', strength: +strength.toFixed(2), pos: [Math.cos(a) * r, 0.07 + rnd() * 0.15, Math.sin(a) * r] });
    if (rnd() < 0.25) {                         // aftershock
      const t2 = t + 0.05 + rnd() * 0.25;
      out.push({ t: +t2.toFixed(3), type: 'pop', strength: +(strength * 0.5).toFixed(2), pos: [Math.cos(a) * r, 0.1, Math.sin(a) * r] });
    }
  }
  // keep clear of the hero events
  return out.filter(e => !SCRIPTED.some(s => Math.abs(s.t - e.t) < 0.35));
}
export const EVENTS = [...SCRIPTED, ...randomPops()].sort((a, b) => a.t - b.t);

const lerp = (a, b, u) => (Array.isArray(a) ? a.map((v, i) => v + (b[i] - v) * u) : a + (b - a) * u);
export function shotAt(t) {
  for (const s of SHOTS) if (t < s.t1) return s;
  return SHOTS[SHOTS.length - 1];
}
export const CUTS = SHOTS.slice(1).map(s => s.t0);

export function cameraAt(t) {
  const s = shotAt(t);
  let u = Math.min(Math.max((t - s.t0) / (s.t1 - s.t0), 0), 1);
  u = u + (u * u * (3 - 2 * u) - u) * 0.3;   // mostly constant speed, slight ease inside the shot
  const pos = lerp(s.a.pos, s.b.pos, u), target = lerp(s.a.target, s.b.target, u), fov = lerp(s.a.fov, s.b.fov, u);
  // gimbal/handheld feel: pink-ish noise, a few mm and ~0.1-0.2 degrees, plus a bump on the log shift
  const n = (f, ph) => Math.sin(t * f + ph) * 0.5 + Math.sin(t * f * 2.13 + ph * 1.7) * 0.3 + Math.sin(t * f * 4.7 + ph * 0.3) * 0.2;
  const d = Math.hypot(...pos.map((v, i) => v - target[i]));
  const jit = [n(1.1, 0.3), n(0.9, 2.1), n(1.3, 4.2)].map(v => v * 0.0022);
  const aim = [n(0.7, 1.7), n(0.8, 3.3), n(0.6, 5.1)].map(v => v * 0.0028 * d);
  const lb = EVENTS.find(e => e.type === 'log-shift');
  const bump = lb && t >= lb.t && t < lb.t + 0.2 ? 0.003 * Math.exp(-(t - lb.t) / 0.05) : 0;
  return {
    pos: pos.map((v, i) => v + jit[i] + (i === 1 ? -bump : 0)),
    target: target.map((v, i) => v + aim[i] + (i === 1 ? -bump * 0.5 : 0)),
    fov,
  };
}

// Lens: full-frame vertical gate (36 mm tall), so focal length f = 18 mm / tan(vfov/2).
export function lensAt(t) {
  const c = cameraAt(t);
  const N = shotAt(t).N;
  const fmm = 18 / Math.tan((c.fov * Math.PI / 180) / 2);
  const focus = Math.hypot(c.target[0] - c.pos[0], c.target[1] - c.pos[1], c.target[2] - c.pos[2]);
  return { focus, aperture: fmm / (2 * N) / 1000, fmm, N };   // aperture radius in metres
}

export function intensityAt(t) {
  const k = INTENSITY;
  if (t <= k[0][0]) return k[0][1];
  for (let i = 0; i < k.length - 1; i++) {
    if (t <= k[i + 1][0]) {
      const u = (t - k[i][0]) / (k[i + 1][0] - k[i][0]);
      const s = u * u * (3 - 2 * u);
      return k[i][1] + (k[i + 1][1] - k[i][1]) * s;
    }
  }
  return k[k.length - 1][1];
}

// Exposure multiplier from the shot's EV offset.
export function exposureAt(t) {
  return Math.pow(2, shotAt(t).ev);
}

// Log transforms over time: the log-shift drops and tilts one teepee log (offsets applied to its end points).
export function logOffsetsAt(t) {
  const e = EVENTS.find(x => x.type === 'log-shift');
  if (!e || t < e.t) return null;
  const u = Math.min((t - e.t) / 0.12, 1);
  const s = u * u * (3 - 2 * u);
  // the foot slides 2 cm outward, the top drops 4 cm (applied along the log's own outward direction)
  return { index: e.log ?? 3, s, footOut: 0.02, topDrop: 0.04, topOut: 0.012 };
}

// Sim parameter overrides at time t (bursts from scripted events, gust wind, slow draft).
export function simAt(t) {
  const p = { intensity: intensityAt(t), burstAmt: 0, burstX: 0, burstY: 0, burstZ: 0, windX: 0, windZ: 0 };
  for (const e of EVENTS) {
    const dt = t - e.t;
    if (dt < 0 || dt > 0.4) continue;
    if (e.type === 'log-shift' || e.type === 'pop' || e.type === 'crackle-burst') {
      const w = (e.type === 'log-shift' ? 1.0 : 0.6) * (e.strength ?? 1) * Math.exp(-dt / (e.type === 'log-shift' ? 0.12 : 0.04));
      if (w > p.burstAmt) { p.burstAmt = w; [p.burstX, p.burstY, p.burstZ] = e.pos; }
    }
  }
  // a slow, wandering draft leans the flames 5-15 degrees
  p.windX = 0.28 * Math.sin(t * 0.23 + 1.0) + 0.14 * Math.sin(t * 0.61 + 0.4);
  p.windZ = 0.18 * Math.sin(t * 0.17 + 2.3);
  // the gust blows across the S4 frame (along x) at ~2.5-3 m/s
  const g = EVENTS.find(e => e.type === 'gust');
  if (g) {
    const dt = t - g.t, dur = g.dur ?? 1.6;
    const w = dt > 0 && dt < dur + 1.2 ? Math.sin(Math.PI * Math.min(dt / (dur + 0.6), 1)) ** 1.5 * g.strength : 0;
    p.windX += 2.8 * w;
  }
  return p;
}
