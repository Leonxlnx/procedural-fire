// Procedural campfire layout (meters, y up, fire centre at origin on the ground).
// Logs are capsules: a, b endpoints, radius r, burn = how much the log is burning (0..1).
// The surface shader derives each log's shape class from the record alone (no extra data, same 8 floats):
//   |b - a| < 0.3 m      -> charred chunk lying in the coals
//   r < 0.05 m           -> split piece (pie-wedge cross-section, bark on the outer arc)
//   otherwise            -> round log with bark
// Stones (8 floats): (c.xyz, yaw), (e.xyz, kind + seed).
//   e = semi-axes of the ellipsoid the shader cuts the rock from (12-18 deep planar facets); the finished rock is
//   ~0.7 e. kind = floor(w) in 0..3 (granite, sandstone, basalt, quartzite), fract(w) = per-stone seed.
//   The shader assumes every stone lies within STONE_RMIN..STONE_RMAX (0.40..0.93 m) and below STONE_YMAX (0.19 m);
//   buildCampfire() checks this.

function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

// Camera positions the cinematic passes through low inside the ring (src/timeline.js shot 3): keep stones away.
const KEEP_CLEAR = [[-0.56, 0.16, -0.05], [-0.48, 0.15, -0.04]];

export function buildCampfire(seed = 7) {
  const R = rng(seed);
  const logs = [];
  const add = (a, b, r, burn) => logs.push({ a, b, r, burn });

  // Two heavy base logs, lying roughly parallel, slightly crossed.
  add([-0.40, 0.055, -0.17], [0.38, 0.065, -0.13], 0.062, 0.8);
  add([-0.36, 0.060, 0.15], [0.41, 0.050, 0.19], 0.058, 0.8);
  // Teepee of split wood leaning into the centre.
  const n = 5;
  for (let i = 0; i < n; i++) {
    const ang = (i / n) * Math.PI * 2 + 0.35 + (R() - 0.5) * 0.4;
    const rb = 0.30 + R() * 0.06;
    const base = [Math.cos(ang) * rb, 0.03, Math.sin(ang) * rb];
    const topR = 0.035 + R() * 0.03;
    const ta = ang + Math.PI + (R() - 0.5) * 0.8;
    const top = [Math.cos(ta) * topR, 0.36 + R() * 0.1, Math.sin(ta) * topR];
    add(base, top, 0.032 + R() * 0.012, 1.0);
  }
  // A charred chunk lying in the coals.
  add([-0.12, 0.035, 0.02], [0.10, 0.03, -0.05], 0.035, 1.0);

  // Ring of stones walked around the circle: rocks 19-29 cm long, 7-14 cm proud of the ground with the widest girth
  // near the ground (centre at ground level), long axis along the ring, a gap where the low camera passes.
  // A separate generator keeps the log layout (and the fire simulation that depends on it) independent.
  const S = rng(seed * 7919 + 13);
  const stones = [];
  const ns = 14;
  const SHRINK = 0.64;                    // finished rock / ellipsoid (src/shaders/scene.js STONE_SHRINK)
  const raw = [];
  for (let i = 0; i < ns; i++) {
    const L = 0.095 + S() * 0.05;        // finished tangential semi-axis
    const u = S();
    raw.push({
      L,
      W: L * (0.6 + S() * 0.25),         // radial
      H: L * (0.55 + S() * 0.25),        // vertical
      kind: u < 0.4 ? 0 : u < 0.6 ? 1 : u < 0.8 ? 2 : 3,
      seed: S(),
    });
  }
  const ring = 0.64;
  const gapC = Math.atan2(KEEP_CLEAR[0][2], KEEP_CLEAR[0][0]);
  const gapW = 0.62;                      // rad kept free around the camera path
  const avail = Math.PI * 2 - gapW;
  const arcs = raw.map(s => 2 * s.L / ring);
  const slack = (avail - arcs.reduce((a, b) => a + b, 0)) / ns;   // mean gap between neighbours
  let phi = gapC + gapW / 2;
  for (let i = 0; i < ns; i++) {
    const s = raw[i];
    const gap = slack * (0.4 + 1.2 * S());
    phi += arcs[i] / 2 + (i === 0 ? 0 : gap * 0.5);
    const rr = ring + (S() - 0.5) * 0.05 + (s.W - 0.08) * 0.3;
    const rot = -phi - Math.PI / 2 + (S() - 0.5) * 0.3;
    const cy = 0.005 + S() * 0.03;
    stones.push({ c: [Math.cos(phi) * rr, cy, Math.sin(phi) * rr], s: [s.L / SHRINK, s.H / SHRINK, s.W / SHRINK], rot, kind: s.kind + s.seed * 0.999 });
    phi += arcs[i] / 2 + gap * 0.5;
  }
  // shader bounds (see header)
  for (const st of stones) {
    const r = Math.hypot(st.c[0], st.c[2]), e = Math.max(st.s[0], st.s[2]);
    if (r - e < 0.40 || r + e > 0.93 || st.c[1] + st.s[1] > 0.19) console.warn('stone outside shader bounds', st);
    for (const k of KEEP_CLEAR) {
      const d = Math.hypot(k[0] - st.c[0], k[1] - st.c[1], k[2] - st.c[2]);
      if (d < Math.max(...st.s) * 0.8) console.warn('stone too close to the camera path', st, d);
    }
  }
  return { logs, stones };
}

// Packs logs into vec4 pairs: (a.xyz, r), (b.xyz, burn).
export function packLogs(logs, max = 16) {
  const f = new Float32Array(max * 8);
  logs.slice(0, max).forEach((l, i) => {
    f.set([...l.a, l.r, ...l.b, l.burn], i * 8);
  });
  return f;
}

// Stones: (c.xyz, rot), (e.xyz, kind + seed).
export function packStones(stones, max = 16) {
  const f = new Float32Array(max * 8);
  stones.slice(0, max).forEach((s, i) => {
    f.set([...s.c, s.rot, ...s.s, s.kind ?? 0], i * 8);
  });
  return f;
}
