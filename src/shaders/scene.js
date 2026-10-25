// SDF scene (charred logs, coal bed, fire-ring stones, ground, pebbles, distant trees) + surface shading pass. WGSL.
// Bindings used by the surface pass: 0 uniforms R (RP struct, see renderer.js RP_FIELDS), 1 logs, 2 stones,
// 3 noise texture, 4 repeat sampler, 5 blackbody LUT (bbLoad/bbSoot), 6 fire lights, 7 color out, 8 depth out,
// 9 sim S texture (x = (T-293)/1000), 10 linear clamp sampler.
//
// Data (see src/scene.js): log = (a.xyz, r), (b.xyz, burn); stone = (c.xyz, yaw), (ellipsoid semi-axes, kind + seed).
// Log shape class comes from the record: length < 0.3 m -> chunk in the coals, r < 0.05 -> split wedge, else round.
// Materials: 1 log, 2 stone, 3 ground (incl. the coal bed's glowing base layer), 4 coal chunk / ember, 5 pebble.
// All geometric micro-detail fades with the pixel footprint (gFoot), so macro shots get mm-scale detail and wide
// shots do not shimmer. R.surfDetail (0..1) scales detail and selects the cheap path for real time.

export const SCENE_SDF = /* wgsl */`
@group(0) @binding(1) var<storage, read> logs: array<vec4f>;
@group(0) @binding(2) var<storage, read> stones: array<vec4f>;
@group(0) @binding(3) var noiseT: texture_3d<f32>;
@group(0) @binding(4) var rep: sampler;
fn nz(q: vec3f) -> vec4f { return textureSampleLevel(noiseT, rep, q, 0.0); }
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }
fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn rotY(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(c * p.x - s * p.z, p.y, s * p.x + c * p.z); }
fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
  let k0 = length(p / r); let k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / max(k1, 1e-6);
}
const TAU = 6.2831853;
// Fast float-only hashes and gradient noise for the scene (integer multiplies are quarter rate on the
// target GPU; the lib's pcg3d/gnoise cost ~4x more per lookup).
fn hq33(p: vec3f) -> vec3f {
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}
fn hqi(c: vec3i) -> vec3f { return hq33(vec3f(c) + vec3f(0.31, 0.57, 0.13)); }
fn gn3(p: vec3f) -> f32 {
  let i = floor(p); let f = p - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n000 = dot(hq33(i) * 2.0 - 1.0, f);
  let n100 = dot(hq33(i + vec3f(1, 0, 0)) * 2.0 - 1.0, f - vec3f(1, 0, 0));
  let n010 = dot(hq33(i + vec3f(0, 1, 0)) * 2.0 - 1.0, f - vec3f(0, 1, 0));
  let n110 = dot(hq33(i + vec3f(1, 1, 0)) * 2.0 - 1.0, f - vec3f(1, 1, 0));
  let n001 = dot(hq33(i + vec3f(0, 0, 1)) * 2.0 - 1.0, f - vec3f(0, 0, 1));
  let n101 = dot(hq33(i + vec3f(1, 0, 1)) * 2.0 - 1.0, f - vec3f(1, 0, 1));
  let n011 = dot(hq33(i + vec3f(0, 1, 1)) * 2.0 - 1.0, f - vec3f(0, 1, 1));
  let n111 = dot(hq33(i + vec3f(1, 1, 1)) * 2.0 - 1.0, f - vec3f(1, 1, 1));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
             mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z) * 1.5;
}
fn fb3(p: vec3f, oct: i32) -> f32 {
  var a = 0.5; var s = 0.0; var q = p;
  for (var i = 0; i < oct; i++) { s += a * gn3(q); q = q * 2.03 + vec3f(17.1, 3.7, 9.2); a *= 0.5; }
  return s;
}
// 1D value noise (~[-1, 1]), two hashes
fn vn1(x: f32, s: f32) -> f32 {
  let i = floor(x); let f = x - i;
  let a = hq33(vec3f(i, s, 7.7)).x; let b = hq33(vec3f(i + 1.0, s, 7.7)).x;
  return (mix(a, b, f * f * (3.0 - 2.0 * f)) * 2.0 - 1.0) * 1.3;
}
// ~normal deviate (sd ~0.5) from one hash value
fn gauss(h: f32) -> f32 { return fract(h * 7.31) + fract(h * 13.71 + 0.3) + fract(h * 3.37 + 0.6) - 1.5; }

var<private> gFoot: f32 = 0.002;   // pixel footprint (m) at the point being evaluated
var<private> gDet: f32 = 1.0;      // global detail level (R.surfDetail)
var<private> gDbg: vec3f = vec3f(0.0);   // debug channel (R.matC = 3)
var<private> gIter: i32 = 0;              // march steps of the primary ray (R.matC = 7)
// 1 while a feature of size w (m) spans a few pixels, fading to 0 once it is sub-pixel (no shimmer).
fn pf(b: u32) -> bool { return (u32(R.matB) & b) != 0u; }   // TEMP perf toggles
fn dfade(w: f32) -> f32 { return (1.0 - smoothstep(0.3 * w, 1.2 * w, gFoot)) * gDet; }

// 2D gradient noise, ~[-1, 1]
fn gnoise2(p: vec2f, s: i32) -> f32 {
  let i = vec2i(floor(p)); let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let fi = vec3f(vec2f(i), f32(s) * 3.7 + 0.5);
  let a = dot(hq33(fi).xy * 2.0 - 1.0, f);
  let b = dot(hq33(fi + vec3f(1.0, 0.0, 0.0)).xy * 2.0 - 1.0, f - vec2f(1.0, 0.0));
  let c = dot(hq33(fi + vec3f(0.0, 1.0, 0.0)).xy * 2.0 - 1.0, f - vec2f(0.0, 1.0));
  let d = dot(hq33(fi + vec3f(1.0, 1.0, 0.0)).xy * 2.0 - 1.0, f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 1.6;
}
fn fbm2(p: vec2f, oct: i32, s: i32) -> f32 {
  var a = 0.5; var r = 0.0; var q = p;
  for (var i = 0; i < oct; i++) { r += a * gnoise2(q, s + i); q = mat2x2f(1.6, 1.2, -1.2, 1.6) * q; a *= 0.5; }
  return r;
}

// ---------------------------------------------------------------- stones (bounds used by the ground)
// Every stone lies inside this annulus / below this height (src/scene.js checks it).
const STONE_RMIN = 0.40;
const STONE_RMAX = 0.93;
const STONE_YMAX = 0.19;
const STONE_SHRINK = 0.64;  // finished rock ~ this x the ellipsoid it is cut from

// ---------------------------------------------------------------- coal bed
const BED_R = 0.325;     // nominal bed radius; lobes run out to ~0.39 m between the teepee feet
const BED_H = 0.05;      // base (fines) layer height at the centre; chunk tops reach ~7.5 cm
const CC = 0.02;         // chunk grid cell
const EMBER_R = 0.47;    // single embers and char bits are scattered in the ash out to here
// Bed radius towards the unit direction d: low harmonics, the 5-fold one puts fingers between the teepee feet.
fn bedRadDir(d: vec2f) -> f32 {
  let x = d.x; let y = d.y; let x2 = x * x; let y2 = y * y;
  let c2 = x2 - y2; let s2 = 2.0 * x * y;
  let c3 = x * (4.0 * x2 - 3.0); let s3 = y * (3.0 - 4.0 * y2);
  let c5 = x * (16.0 * x2 * x2 - 20.0 * x2 + 5.0); let s5 = y * (16.0 * y2 * y2 - 20.0 * y2 + 5.0);
  return BED_R + 0.018 * (0.8 * c2 - 0.6 * s2) + 0.012 * (0.5 * c3 + 0.87 * s3) + 0.03 * (0.19 * c5 - 0.98 * s5);
}
fn bedRho(xz: vec2f) -> f32 { let r = length(xz); return r / bedRadDir(xz / max(r, 1e-5)); }
fn bedBase(rho: f32) -> f32 { let s = sat(1.0 - rho * rho); return BED_H * s * sqrt(s); }

// ---------------------------------------------------------------- ground
fn terrain(xz: vec2f) -> f32 {
  let r = length(xz);
  return gnoise2(xz * 1.3, 3) * 0.008 + gnoise2(xz * 4.1, 4) * 0.003 - 0.012 * (1.0 - smoothstep(0.3, 0.62, r));
}
fn ashEdge(xz: vec2f) -> f32 { return 0.5 + 0.07 * fbm2(xz * 2.4, 2, 41) + 0.025 * gnoise2(xz * 12.0, 44); }
// Ash cover 0..1: everywhere inside 0.3 m, feathered (thinning, flaky) edge around 0.5 m.
fn ashCover(xz: vec2f, r: f32) -> f32 {
  if (r < 0.3) { return 1.0; }
  if (r > 0.72) { return 0.0; }
  return 1.0 - smoothstep(-0.08, 0.04, r - ashEdge(xz) + 0.03 * gnoise2(xz * 40.0, 46));
}
// Height the coal chunks are seated on (terrain + mean ash layer).
fn coalTer(xz: vec2f) -> f32 { return terrain(xz) + 0.004; }
// Stone berm: dirt pushed up ~1.3 cm in a 2-4 cm collar around each stone (distance to the footprint ellipse).
fn stoneBerm(xz: vec2f, r: f32) -> f32 {
  if (r < STONE_RMIN - 0.05 || r > STONE_RMAX + 0.05) { return 0.0; }
  var b = 0.0;
  let ns = i32(R.nStones);
  for (var i = 0; i < ns; i++) {
    let C = stones[i * 2]; let S = stones[i * 2 + 1];
    let dxz = xz - C.xz;
    let ext = max(S.x, S.z) * 0.85 + 0.06;
    if (dot(dxz, dxz) > ext * ext) { continue; }
    let q = rotY(vec3f(dxz.x, 0.0, dxz.y), C.w).xz;
    let a = vec2f(S.x, S.z) * (STONE_SHRINK + 0.02);
    let dd = max((length(q / a) - 1.0) * min(a.x, a.y), 0.0);
    b = max(b, exp(-dd * dd / 0.0007));
  }
  return b;
}
struct GParts { h: f32, ash: f32, clump: f32, berm: f32, crumb: f32, twig: f32, twigT: f32, twigH: f32, rho: f32,
                gran: f32, granH: f32 };
// Granular charcoal fines (4-6 mm granules) on the bed floor between the chunks: (bump 0..1, granule hash).
fn granules(xz: vec2f) -> vec2f {
  let q = xz / 0.0055;
  let i0 = vec2i(floor(q - 0.5));
  var gb = 0.0; var hc = 0.0;
  for (var k = 0; k < 4; k++) {
    let ci = i0 + vec2i(k & 1, k >> 1u);
    let hh = hqi(vec3i(ci, 163));
    let c = vec2f(ci) + 0.25 + 0.5 * hh.xy - q;
    let b = sat(1.0 - dot(c, c) * (2.6 + 1.2 * hh.z)) * (0.5 + 0.5 * fract(hh.z * 5.9));
    if (b > gb) { gb = b; hc = hh.z; }
  }
  return vec2f(gb, hc);
}
// ter: terrain(xz), passed in so the coal field can share it
fn groundParts(xz: vec2f, ter: f32, detail: bool) -> GParts {
  var o = GParts(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0);
  let r = length(xz);
  var h = ter;
  // coal bed base: a mound of charcoal fines the chunks sit in
  if (r < 0.4) {
    o.rho = bedRho(xz); h += bedBase(o.rho);
    if (o.rho < 1.08) {
      let g = granules(xz);
      o.gran = g.x; o.granH = g.y;
      if (detail) { h += g.x * 0.0016 * dfade(0.003) * (1.0 - smoothstep(0.95, 1.08, o.rho)); }
    }
  }
  // ash: patchy (2-10 cm) layer 2-7 mm thick, fluffy 1-3 cm clumps around the bed
  o.ash = ashCover(xz, r);
  if (o.rho < 0.85) { h += 0.004; }   // inside the bed: the fines sit on an even ash layer (= coalTer)
  else if (o.ash > 0.0) {
    let pt = sat(0.5 + 0.9 * fbm2(xz * 14.0, 2, 45));
    h += o.ash * (0.002 + 0.005 * pt);
    let q = xz / 0.03 - 0.5; let i0 = vec2i(floor(q));
    var cl = 0.0;
    for (var k = 0; k < 4; k++) {
      let ci = i0 + vec2i(k & 1, k >> 1u);
      let hh = hqi(vec3i(ci, 151));
      if (hh.z > 0.12 + 0.35 * pt) { continue; }
      let c = (vec2f(ci) + 0.25 + 0.5 * hh.xy) * 0.03;
      let rad = 0.006 + 0.0085 * fract(hh.z * 13.0);
      let dv = xz - c;
      let dd = length(dv) / (rad * (1.0 + 0.3 * sin(atan2(dv.y, dv.x) * 3.0 + hh.x * 6.0)));
      let b = sat(1.0 - dd * dd);
      cl = max(cl, b * b * (0.55 + 0.45 * fract(hh.z * 7.0)));
    }
    o.clump = cl * o.ash * smoothstep(0.85, 1.05, o.rho);
    h += o.clump * 0.0065;
  }
  // stone berms
  if (!pf(16u)) { o.berm = stoneBerm(xz, r); }
  h += o.berm * 0.013;
  if (detail) {
    let hT = h;
    let soil = 1.0 - o.ash;
    // crumbs and clods in the soil (cellular, 2-10 mm, 1-3 mm high)
    let fc = select(dfade(0.004) * sat(soil * 1.2 + o.berm * 0.5), 0.0, pf(32u));
    if (fc > 0.0) {
      let q = xz / 0.0085;
      let i0 = vec2i(floor(q - 0.5));
      var cb = 0.0;
      for (var k = 0; k < 4; k++) {
        let ci = i0 + vec2i(k & 1, k >> 1u);
        let hh = hqi(vec3i(ci, 157));
        let c = vec2f(ci) + 0.1 + 0.8 * hh.xy - q;
        let rr = 0.25 + 0.3 * fract(hh.z * 3.1);
        cb = max(cb, sat(1.0 - dot(c, c) / (rr * rr)) * step(0.62, fract(hh.z * 7.3)) * (0.4 + 0.6 * hh.z));
      }
      let clod = smoothstep(-0.25, 0.45, gnoise2(xz * 18.0, 159));
      o.crumb = cb * (0.2 + 0.8 * clod);
      h += o.crumb * 0.0013 * fc;
    }
    // twigs and sticks lying on the forest floor (charred ones inside the ring)
    let ft = dfade(0.003);
    if (ft > 0.0 && r > 0.42) {
      let q2 = xz * 7.0;
      let i2 = vec2i(floor(q2)); let f2 = q2 - floor(q2);
      var tw = 0.0;
      for (var k = 0; k < 9; k++) {
        let off = vec2i(k % 3 - 1, k / 3 - 1);
        let hh = hqi(vec3i(i2 + off, 53));
        if (hh.z > 0.5) { continue; }
        let dir = normalize(vec2f(fract(hh.z * 17.0), fract(hh.z * 41.0)) - 0.5 + 1e-4);
        let c = vec2f(off) + 0.2 + 0.6 * hh.xy - f2;
        let L = 0.15 + 0.35 * fract(hh.z * 5.3);
        let al = dot(-c, dir);
        let along = clamp(al, -L, L);
        let rt = 0.0012 + 0.003 * fract(hh.z * 11.0);
        let bend = 0.6 * rt * sin(along * 5.0 + hh.x * 6.0);
        let dperp = abs(dot(-c, vec2f(-dir.y, dir.x)) / 7.0 + bend);
        let dl = max(abs(al) - L, 0.0) / 7.0;
        let d = length(vec2f(dperp, dl));
        let b = sat(1.0 - d * d / (rt * rt));
        let th = sqrt(b) * rt * 1.4;
        if (th > tw) { tw = th; o.twigT = (along / L) * 0.5 + 0.5; o.twigH = hh.z; }
      }
      o.twig = tw * ft;
      h = max(h, hT + o.twig * 0.9);
    }
    h += gn3(vec3f(xz * 140.0, 2.0)) * 0.0004 * dfade(0.002);
  }
  o.h = h;
  return o;
}
fn groundH(xz: vec2f, detail: bool) -> f32 { return groundParts(xz, terrain(xz), detail).h; }
// Upper bound of the ground height around radius r (for cheap rejection while marching).
fn groundMax(r: f32) -> f32 { return 0.034 + select(0.0, 0.013, r > STONE_RMIN - 0.06) + bedBase(r / 0.39); }

// ---------------------------------------------------------------- coal chunks
// Chunk in grid cell ci, layer L -> (centre, half-size; 0 = empty). ter: coalTer near the query point.
// Inside the bed: two stacked layers of broken charcoal (log-normal sizes ~0.8-5 cm, smaller towards the rim, upper
// layer ~40% empty) so chunks overlap and tilt with dark undercuts; outside: sparse single embers and char bits.
fn chunkPlace(ci: vec2i, L: i32, ter: f32, rb: f32) -> vec4f {
  let h = hqi(vec3i(ci, 101 + L));
  let cxz = (vec2f(ci) + 0.2 + 0.6 * h.xy) * CC;
  let r = length(cxz);
  let rho = r / rb;
  var s = 0.0; var y = 0.0;
  if (rho < 1.04) {
    s = clamp(0.0108 * exp(0.7 * gauss(h.z)) * (1.0 - 0.35 * rho * rho) * select(1.0, 0.75, L == 1), 0.0035, select(0.0145, 0.0085, L == 1));
    let keep = select(0.98, 0.7, L == 1) * (1.0 - 0.5 * smoothstep(0.75, 1.04, rho));
    if (fract(h.z * 31.17) > keep) { s = 0.0; }
    y = ter + bedBase(rho) + select(0.1 * s, 0.004 + 0.6 * s, L == 1);
  } else if (L == 0 && r < EMBER_R) {
    let dens = 0.035 + 0.13 * (1.0 - smoothstep(0.0, 0.1, rho - 1.0));
    if (fract(h.z * 31.17) < dens) {
      s = 0.0025 + 0.0045 * fract(h.z * 7.7);
      y = ter + 0.25 * s;
    }
  }
  return vec4f(cxz.x, y, cxz.y, s);
}
struct ChunkG { F: mat3x3f, e: vec3f, hc: vec3f };
// Random orientation (grain axis g mostly horizontal, tilted up to ~45 deg) and half extents of a chunk.
fn chunkGeo(ci: vec2i, L: i32, s: f32) -> ChunkG {
  let a = hqi(vec3i(ci, 111 + L));
  let b = hqi(vec3i(ci, 121 + L));
  let yaw = normalize(a.xy - 0.5 + 1e-4);
  let g0 = vec3f(yaw.x, 0.0, yaw.y);
  let up = normalize(vec3f((b.x - 0.5) * 1.6, 1.0, (b.y - 0.5) * 1.6));
  let g = normalize(g0 - up * dot(g0, up));
  let w = cross(g, up);
  let e = s * vec3f(1.0 + 0.6 * a.z, 0.45 + 0.3 * b.z, 0.6 + 0.35 * fract(a.z * 7.3 + b.z));
  return ChunkG(mat3x3f(g, up, w), e, hqi(vec3i(ci, 131 + L)));
}
// Rounded box with three corners knocked off by random fracture planes (a convex charcoal lump).
fn chunkSDF(q: vec3f, e: vec3f, hc: vec3f) -> f32 {
  let rr = 0.28 * min(e.y, e.z);
  let qa = abs(q) - (e - rr);
  var d = length(max(qa, vec3f(0.0))) + min(max(qa.x, max(qa.y, qa.z)), 0.0) - rr;
  // three of the eight corners knocked off (corner directions: no normalisation needed)
  let sup = (e.x + e.y + e.z) * 0.57735;
  for (var k = 0; k < 3; k++) {
    let hk = fract(hc * (1.0 + 2.37 * f32(k)) + f32(k) * 0.31);
    let n = sign(hk - 0.5) * 0.57735;
    d = smax(d, dot(q, n) - sup * (0.55 + 0.3 * hk.x), rr * 0.5);
  }
  return d;
}
// Upper bound of the coal tops (chunk tops are <= their base + 3.1 cm; 1.2 cm margin for terrain and slope).
fn coalMax(r: f32) -> f32 { return select(0.03, 0.012 + bedBase(r / 0.39) + 0.037, r < 0.4); }
fn chunkId(ci: vec2i, L: i32) -> f32 { return f32(L * 4194304 + (ci.x + 1024) + (ci.y + 1024) * 2048); }
fn chunkCell(id: f32) -> vec3i {
  let k = i32(id + 0.5);
  let L = k / 4194304; let m = k - L * 4194304;
  return vec3i(m % 2048 - 1024, m / 2048 - 1024, L);
}
// -> (distance, chunk id) of the nearest coal chunk or ember (distance capped: chunks two cells away may come
// within ~4 mm, so the field never claims more free space than that).
// ter: terrain(p.xz). Lower layer: 3x3 cell search (chunks up to ~5 cm); upper layer: 2x2 (chunks <= 0.7 cells).
fn coalField(p: vec3f, ter0: f32, detail: bool) -> vec2f {
  let r = length(p.xz);
  if (r > EMBER_R + 0.03) { return vec2f(1e3, -1.0); }
  let bnd = p.y - coalMax(r);
  if (bnd > 0.008) { return vec2f(bnd, -1.0); }
  let ter = ter0 + 0.004;
  let rb = bedRadDir(p.xz / max(r, 1e-5));
  // tight bound with the local terrain and bed height (neighbouring bases differ by < 5 mm)
  let bnd2 = p.y - (ter + select(0.012, bedBase(r / rb) + 0.036, r < rb * 1.05));
  if (bnd2 > 0.004) { return vec2f(bnd2, -1.0); }
  let ip = vec2i(floor(p.xz / CC));
  var best = vec2f(0.3 * CC + max(bnd2, 0.0), -1.0);
  let i2 = vec2i(floor(p.xz / CC - 0.5));
  for (var k = 0; k < 13; k++) {
    let L = select(0, 1, k >= 9);
    let kk = (k + 4) % 9;
    let ci = select(ip + vec2i(kk % 3 - 1, kk / 3 - 1), i2 + vec2i((k - 9) & 1, (k - 9) >> 1u), L == 1);
    let pl = chunkPlace(ci, L, ter, rb);
    if (pl.w <= 0.0) { continue; }
    let dc = p - pl.xyz;
    if (length(dc) - 1.95 * pl.w > best.x) { continue; }
    let G = chunkGeo(ci, L, pl.w);
    let d = chunkSDF(dc * G.F, G.e, G.hc);
    if (d < best.x) { best = vec2f(d, chunkId(ci, L)); }
  }
  if (detail && best.y >= 0.0 && best.x < 0.003) {
    best.x += gn3(p * 380.0) * 0.00035 * dfade(0.0015);
  }
  return best;
}

// ---------------------------------------------------------------- pebbles
const PC = 0.034;
// Pebble in cell ci -> (x, z, size (radius, m), hash); size 0 = none. Log-normal sizes 3-30 mm, clustered, dense
// along the stone ring, sparse in the open, none in the ash disc.
fn pebblePlace(ci: vec2i) -> vec4f {
  let h = hqi(vec3i(ci, 141));
  let cxz = (vec2f(ci) + 0.25 + 0.5 * h.xy) * PC;
  let r = length(cxz);
  let cl = hqi(vec3i(vec2i(floor(vec2f(ci) / 3.0)), 143)).x;
  let ringB = exp(-(r - 0.66) * (r - 0.66) / 0.012);
  let dens = (0.05 + 0.4 * ringB) * (0.2 + 1.6 * cl * cl) * smoothstep(0.42, 0.55, r);
  if (h.z > dens) { return vec4f(0.0); }
  let s = clamp(0.0045 * exp(1.1 * gauss(h.z * 3.7)), 0.0015, 0.0135);
  return vec4f(cxz, s, h.z);
}
fn pebbleSDF(p: vec3f, gh: f32, ci: vec2i, pl: vec4f) -> f32 {
  let hh = hqi(vec3i(ci, 147));
  let e = pl.z * vec3f(1.0 + 0.4 * hh.x, 0.8 + 0.25 * hh.y, 0.75 + 0.25 * hh.z);
  let c = vec3f(pl.x, gh - 0.22 * e.y, pl.y);
  var q = rotY(p - c, hh.x * 6.28) / e;
  // lumpy, not a perfect ellipsoid
  q += 0.18 * vec3f(sin(q.y * 2.3 + hh.z * 9.0), sin(q.z * 2.1 + hh.x * 7.0), sin(q.x * 2.5 + hh.y * 8.0));
  return (length(q) - 1.0) * min(e.x, min(e.y, e.z)) * 0.65;
}
// -> (distance, cell id) of the nearest pebble (2x2 cell search: pebbles are smaller than 0.75 cells).
fn pebbleField(p: vec3f, gh: f32) -> vec2f {
  let q = p.xz / PC - 0.5;
  let i0 = vec2i(floor(q));
  var best = vec2f(1e3, -1.0);
  for (var k = 0; k < 4; k++) {
    let ci = i0 + vec2i(k & 1, k >> 1u);
    let pl = pebblePlace(ci);
    if (pl.z <= 0.0) { continue; }
    let d = pebbleSDF(p, gh, ci, pl);
    if (d < best.x) { best = vec2f(d, f32((ci.x + 1024) + (ci.y + 1024) * 2048)); }
  }
  return best;
}

// ---------------------------------------------------------------- logs
// Static fire proximity (0 far .. 1 at the core); the sim adds the live part via gasTemp().
fn heatAt(p: vec3f) -> f32 {
  let r = length(p.xz);
  return 1.0 - smoothstep(0.06, 0.5, r + 0.4 * max(p.y - 0.18, 0.0));
}
fn logClass(A: vec4f, B: vec4f) -> i32 {
  if (length(B.xyz - A.xyz) < 0.3) { return 2; }
  if (A.w < 0.05) { return 1; }
  return 0;
}
struct LogF { u: f32, x: f32, y: f32, len: f32, t: f32 };
// Log-local frame: u along the axis (m), (x, y) across it with a random roll; the axis bows sideways a little and
// split pieces twist slightly along their length.
fn logFrame(p: vec3f, A: vec3f, B: vec3f, i: i32, cls: i32) -> LogF {
  let ba = B - A; let len = length(ba); let ax = ba / len;
  let b1 = normalize(cross(ax, vec3f(0.37, 1.0, 0.21)));
  let b2 = cross(ax, b1);
  let pa = p - A; let u = dot(pa, ax);
  let t = sat(u / len);
  let hb = hqi(vec3i(i, 13, 9));
  let bow = (hb.x - 0.5) * 0.045 * len * 4.0 * t * (1.0 - t);
  let x0 = dot(pa, b1) - bow; let y0 = dot(pa, b2);
  var rv = normalize(hqi(vec3i(i, 13, 5)).xy - 0.5 + 1e-4);
  if (cls == 1) {
    let tw = (hb.z - 0.5) * 0.5 * (t - 0.5);
    let c = cos(tw); let s = sin(tw);
    rv = vec2f(rv.x * c - rv.y * s, rv.x * s + rv.y * c);
  }
  return LogF(u, rv.x * x0 - rv.y * y0, rv.y * x0 + rv.x * y0, len, t);
}
fn sdPie(p: vec2f, c: vec2f, r: f32) -> f32 {
  let q = vec2f(abs(p.x), p.y);
  let l = length(q) - r;
  let m = length(q - c * clamp(dot(q, c), 0.0, r));
  return max(l, m * sign(c.y * q.x - c.x * q.y));
}
const SPLIT_RW = 1.65;    // wedge outer radius / nominal r
const SPLIT_APEX = 0.62;  // pith offset below the log axis (x outer radius)
// Wedge half-angle as (sin, cos), ~36-48 degrees.
fn splitCS(i: i32) -> vec2f { return normalize(vec2f(0.72 + 0.35 * hqi(vec3i(i, 7, 1)).x, 1.0)); }
// Burnt-thin towards the fire centre: identical to the simulation's solid so flames sit on the wood.
fn burnThin(p: vec3f, r: f32, burn: f32) -> f32 { return r * 0.3 * (1.0 - smoothstep(0.0, 0.16, length(p.xz))) * burn; }
// Side surface (no end caps): tapered, lumpy (5-15 cm, stronger on the fire side), knots; torn split faces.
fn logSide(p: vec3f, f: LogF, r0: f32, cls: i32, i: i32, burn: f32) -> f32 {
  let hb = hqi(vec3i(i, 13, 9));
  let r = r0 * (1.0 + (hb.y - 0.5) * 0.16 * (2.0 * f.t - 1.0));
  let heat = heatAt(p) * burn;
  let lq = vec3f(f.u, f.x, f.y);
  let lump = gn3(lq * 11.0 + f32(i) * 3.1) * 0.75 + 0.25 * sin(lq.x * 61.0 + lq.y * 37.0 + f32(i)) * sin(lq.z * 43.0 - lq.x * 23.0);
  if (cls == 1) {
    let Rw = r * SPLIT_RW; let rr = r * 0.12;
    var d = sdPie(vec2f(f.x, f.y + Rw * SPLIT_APEX), splitCS(i), Rw - rr) - rr;
    d += r * (0.035 + 0.1 * heat) * lump;
    // torn, fibrous split faces: relief stretched ~30:1 along the grain
    d += 0.0016 * gn3(vec3f(f.u * 5.0, f.x * 150.0, f.y * 150.0) + f32(i) * 7.0);
    return d;
  }
  var d = length(vec2f(f.x, f.y)) - r * (1.0 + (0.035 + 0.11 * heat) * lump);
  if (cls == 0) {
    for (var k = 0; k < 2; k++) {
      let hk = hqi(vec3i(i, 17, k));
      let kd = normalize(hqi(vec3i(i, 18, k)).xy - 0.5 + 1e-4);
      let kp = vec3f(f.len * (0.12 + 0.76 * hk.x), kd * r);
      let dd = lq - kp;
      d -= 0.0055 * exp(-dot(dd, dd) / (0.0004 + 0.0004 * hk.z));
    }
  }
  return d;
}
fn logBase(p: vec3f, f: LogF, r0: f32, cls: i32, i: i32, burn: f32) -> f32 {
  let d2 = logSide(p, f, r0, cls, i, burn);
  // ends: sawn (crisp edge) when fresh, burnt round when charred
  var e = r0 * 0.35;
  if (min(abs(f.u + e), abs(f.u - f.len - e)) < 0.02) { e += 0.004 * gn3(vec3f(f.x * 30.0, f.y * 30.0, f32(i))); }
  let dc = max(-f.u - e, f.u - f.len - e);
  return smax(d2, dc, r0 * (0.07 + 0.3 * heatAt(p) * burn));
}
// Burn hollows: wood is eaten away where another log crosses or rests on it.
fn logHollows(p: vec3f, i: i32, d0: f32) -> f32 {
  var d = d0;
  let nl = i32(R.nLogs);
  for (var j = 0; j < nl; j++) {
    if (j == i) { continue; }
    let Aj = logs[j * 2]; let Bj = logs[j * 2 + 1];
    let dj = sdCapsule(p, Aj.xyz, Bj.xyz, Aj.w * 1.08 + 0.003);
    if (dj < 0.03) { d = smax(d, -dj, 0.01); }
  }
  return d;
}
// How charred a point on a log is (0 fresh .. 1 deep charcoal). Base logs char over ~75-80% of their length.
fn logChar(p: vec3f, burn: f32, i: i32) -> f32 {
  let h = heatAt(p) + 0.2 * gn3(p * 9.0 + f32(i) * 5.3);
  return burn * smoothstep(0.08, 0.35, h);
}
// Bark coverage: round logs everywhere, split pieces only on the outer arc, chunks none.
fn logBarkness(f: LogF, r: f32, cls: i32) -> f32 {
  if (cls == 2) { return 0.0; }
  if (cls == 0) { return 1.0; }
  let Rw = r * SPLIT_RW;
  return smoothstep(Rw * 0.82, Rw * 0.95, length(vec2f(f.x, f.y + Rw * SPLIT_APEX)));
}
// 1 on the sawn end faces of a log, 0 on its side.
fn logCap(f: LogF, r: f32) -> f32 { return smoothstep(0.12 * r, 0.28 * r, max(-f.u, f.u - f.len)); }
// Perimeter coordinate around the cross-section (0..1, continuous and periodic) and the perimeter length (m).
// Split wedges: arc + two flat split faces meeting at the pith.
fn logPerim(f: LogF, r: f32, cls: i32, i: i32) -> vec2f {
  if (cls != 1) { return vec2f(atan2(f.y, f.x) / TAU + 0.5, TAU * r); }
  let Rw = r * SPLIT_RW;
  let q = vec2f(f.x, f.y + Rw * SPLIT_APEX);
  let cs = splitCS(i);
  let al = atan2(cs.x, cs.y);
  let th = atan2(q.x, q.y);
  let rho = length(q);
  let Pl = 2.0 * Rw * al + 2.0 * Rw;
  let dArc = abs(rho - Rw) + max(abs(th) - al, 0.0) * rho;
  let dFace = abs(rho * sin(abs(th) - al));
  var s = Rw * th;
  if (dFace < dArc) { s = sign(th) * (Rw * al + Rw - clamp(rho * cos(abs(th) - al), 0.0, Rw)); }
  return vec2f(s / Pl + 0.5, Pl);
}
// Char cell size (along the grain, across), m: small cobbles on the teepee pieces, bigger on the base logs.
fn charCell(cls: i32, burn: f32) -> vec2f {
  var cu = select(0.018, 0.0105, cls == 1);
  if (cls == 2) { cu = 0.014; }
  cu *= 0.85 + 0.3 * burn;
  return vec2f(cu, cu * 0.75);
}
// Static exposure used for the crack geometry (wider, deeper cracks where the flames and coals are).
fn charExposure(p: vec3f, burn: f32) -> f32 {
  let bedTop = bedBase(bedRho(p.xz)) + 0.016;
  let nearBed = (1.0 - smoothstep(0.0, 0.05, p.y - bedTop)) * (1.0 - smoothstep(0.34, 0.42, length(p.xz)));
  return max(heatAt(p) * burn, nearBed);
}
// Alligator char: rows of blocks along the grain (u), separated by lengthwise cracks; deep, staggered cross-grain
// cracks split each row into blocks. Periodic around the log (no seam). Rows wander +-0.3 rows over ~10 cm; the
// lengthwise cracks are switched off ~40% of the time by a 1D noise along u (rows merge, cracks end after 2-5
// blocks), some rows split in two for a while; where rows merge the neighbour row's cross cracks run on into the
// block and taper out (T-junctions, no seams).
struct CharC { e: f32, w: f32, dep: f32, dome: f32, id: f32, eAll: f32, eGate: f32, id2: f32 };
// Lengthwise crack between row j-1 and row j: openness 0..1 along u.
fn rowGate(X: f32, j: i32, seed: i32, cu: f32) -> f32 {
  return smoothstep(-0.08, 0.08, vn1(X / (cu * 3.2), f32(j) * 1.37 + f32(seed) * 0.31) + 0.03);
}
// Cross cracks of row jw at X (fy: position across the row, 0..1): (distance to the nearest (m), its hash, block
// hash, block length (m)). Each crack is slanted/bent on its own, so blocks are not rectangles.
fn rowT(X: f32, fy: f32, jw: i32, seed: i32, cu: f32) -> vec4f {
  let hr = hqi(vec3i(jw, seed, 7));
  let sx = 0.7 + 0.65 * hr.y;
  let x = X / cu * sx + hr.x * 5.0;
  let k = i32(floor(x));
  var lo = -1e3; var hi = 1e3; var hcl = 0.0; var hch = 0.0; var hb0 = 0.0;
  let fc = fy - 0.5;
  for (var mm = -1; mm <= 2; mm++) {
    let m = k + mm;
    let hb = hqi(vec3i(m, jw, seed + 3));
    // own position jitter, slant and curvature per crack
    let b = f32(m) + 0.75 * (hb.x - 0.5) + fc * 0.8 * (hb.z - 0.5) + fc * fc * 1.2 * (fract(hb.y * 7.9) - 0.5);
    if (b <= x) { if (b > lo) { lo = b; hcl = hb.y; hb0 = hb.z; } } else { if (b < hi) { hi = b; hch = hb.y; } }
  }
  let dLo = x - lo; let dHi = hi - x;
  return vec4f(min(dLo, dHi) * cu / sx, select(hch, hcl, dLo < dHi), hb0, (hi - lo) * cu / sx);
}
// Some cross cracks only run part of the way across their row (they end inside the block).
fn partCrack(h: f32, fy: f32) -> f32 {
  let k = fract(h * 5.7);
  if (k > 0.35) { return 1.0; }
  return select(smoothstep(0.75, 0.45, fy), smoothstep(0.25, 0.55, fy), k < 0.17);
}
fn crackW(h: f32, wk: f32) -> f32 { return (0.22 + 1.05 * h * h) * wk * 0.001; }
fn crackDep(d: f32, w: f32) -> f32 {
  if (w <= 1e-6) { return 0.0; }
  let x = sat(1.0 - d / w);
  return min(1.8 * w, 0.0055) * x * sqrt(x);
}
fn charCells(X: f32, a01: f32, P: f32, cu: f32, cs: f32, seed: i32, wk: f32) -> CharC {
  let Nf = max(3.0, round(P / cs));
  let Ni = i32(Nf);
  let csN = P / Nf;
  let ph = a01 * TAU; let sd = f32(seed);
  let warp = 0.17 * sin(X * 23.0 + ph + sd * 1.7) + 0.12 * sin(X * 41.0 - 2.0 * ph + sd * 4.1) + 0.07 * sin(X * 97.0 + 3.0 * ph + sd);
  // patches of finer / coarser checking
  let wkp = wk * (0.6 + 0.8 * smoothstep(-0.6, 0.6, sin(X * 9.0 - ph + sd * 2.3) * sin(X * 5.3 + 2.0 * ph + sd)));
  let y = a01 * Nf + warp;
  let j = i32(floor(y)); let fy = y - floor(y);
  let jw = ((j % Ni) + Ni) % Ni;
  let up = fy > 0.5;
  let jnw = (((j + select(-1, 1, up)) % Ni) + Ni) % Ni;
  let dB = min(fy, 1.0 - fy) * csN;
  let oLo = rowGate(X, jw, seed, cu);
  let oHi = rowGate(X, (jw + 1) % Ni, seed, cu);
  let oNear = select(oLo, oHi, up);
  let hs = hqi(vec3i(jw, seed, 9));
  let oSp = smoothstep(-0.08, 0.08, vn1(X / (cu * 2.4), f32(jw) * 2.11 + 7.0) - 0.45);
  let dSp = abs(fy - 0.5) * csN;
  // cross cracks of this row and of the nearer neighbour row (running on where the rows merge)
  let fyN = select(fy + 1.0, fy - 1.0, up);
  let tO = rowT(X, fy, jw, seed, cu);
  let tN = rowT(X, fyN, jnw, seed, cu);
  let wNb = (1.0 - oNear) * (1.0 - smoothstep(0.0, 0.45 * csN, dB));
  let pt1 = partCrack(tO.y, fy);
  let pt2 = wNb * partCrack(tN.y, sat(fyN));
  let wT1 = crackW(tO.y, wkp) * pt1;
  let wT2 = crackW(tN.y, wkp) * pt2;
  let wL0 = (0.1 + 0.3 * hs.x) * wkp * 0.001;
  let wLo = wL0 * oLo; let wHi = (0.1 + 0.3 * hqi(vec3i((jw + 1) % Ni, seed, 9)).x) * wkp * 0.001 * oHi;
  let wSp = wL0 * 0.7 * oSp;
  let dLo = fy * csN; let dHi = (1.0 - fy) * csN;
  var dep = max(max(crackDep(tO.x, wT1), crackDep(tN.x, wT2)), max(max(crackDep(dLo, wLo), crackDep(dHi, wHi)), crackDep(dSp, wSp)));
  // nearest crack lip (distance from the lip, its half-width)
  var eAct = tO.x - wT1; var wAct = wT1; var id2 = tO.y;
  if (wT2 > 1e-5 && tN.x - wT2 < eAct) { eAct = tN.x - wT2; wAct = wT2; id2 = tN.y; }
  if (wLo > 1e-5 && dLo - wLo < eAct) { eAct = dLo - wLo; wAct = wLo; id2 = hs.y; }
  if (wHi > 1e-5 && dHi - wHi < eAct) { eAct = dHi - wHi; wAct = wHi; id2 = hs.z; }
  if (wSp > 1e-5 && dSp - wSp < eAct) { eAct = dSp - wSp; wAct = wSp; id2 = hs.z * 3.1; }
  // block dome: 1 - cen^2 across and along (saturates half a block in, so merged edges stay continuous)
  let dA = min(min(dLo / max(oLo, 1e-3), dHi / max(oHi, 1e-3)), dSp / max(oSp, 1e-3));
  let dU = min(tO.x / max(pt1, 1e-3), tN.x / max(pt2, 1e-3));
  let cA = 1.0 - min(dA / (0.5 * csN), 1.0);
  let cU = 1.0 - min(dU / (0.45 * tO.w), 1.0);
  let dome = (1.0 - cA * cA * cA) * (1.0 - cU * cU * cU);
  let eAll = min(min(dB, tO.x), select(1.0, dSp, oSp > 0.5));
  let eGate = select(1.0, dB, oNear < 0.5);
  let id = fract(tO.z + hs.x * 3.7 + select(0.0, 0.37, oSp > 0.5 && fy > 0.5));
  return CharC(eAct, wAct, dep, dome, id, eAll, eGate, id2);
}
// Relief of the charred side (positive = recessed): domed blocks (height ~8-16% of the row width), random sink
// +-1 mm (faded out where rows merge), V cracks with rounded lips (hairlines ~1 mm deep, gaping cross cracks up to
// ~5.5 mm), ~4% burnt-out pits.
fn charRelief(c: CharC, cs: f32) -> f32 {
  let hd = cs * (0.05 + 0.06 * fract(c.id * 5.31));
  let sink = (fract(c.id * 11.7) - 0.5) * 0.002 * smoothstep(0.0, 0.3 * cs, c.eGate);
  let pit = select(0.0, 0.0045 * smoothstep(0.0, 0.003, c.eAll), fract(c.id * 23.3) < 0.04);
  return c.dep + sink + pit - hd * c.dome;
}
fn logDisp(p: vec3f, f: LogF, A: vec4f, B: vec4f, i: i32, cls: i32) -> vec2f {
  let r = A.w;
  let ch = logChar(p, B.w, i);
  let capF = logCap(f, r);
  var disp = 0.0; var stp = 0.9;
  if (ch > 0.05 && capF < 0.999) {
    let cz = charCell(cls, B.w);
    let fa = dfade(cz.y * 0.35);
    if (fa > 0.0) {
      let pc = logPerim(f, r, cls, i);
      let wk = 0.55 + 1.7 * charExposure(p, B.w);
      let c = charCells(f.u, pc.x, pc.y, cz.x, cz.y, i, wk);
      disp += charRelief(c, cz.y) * smoothstep(0.05, 0.4, ch) * (1.0 - capF) * fa;
      // crack walls are steep: small steps only near cracks
      stp = mix(0.5, 0.9, smoothstep(0.0, 2.0 * c.w + 0.004, c.e + c.w));
    }
  }
  let bk = logBarkness(f, r, cls) * (1.0 - smoothstep(0.15, 0.6, ch)) * (1.0 - capF);
  if (bk > 0.02) {
    let fa = dfade(0.005);
    if (fa > 0.0) {
      // bark: long plates separated by deep furrows along the grain
      let ang = atan2(f.y, f.x);
      let ca = cos(ang) * r; let sa = sin(ang) * r;
      let rn = abs(gn3(vec3f(f.u * 6.0, ca * 95.0, sa * 95.0) + f32(i) * 2.0));
      let rn2 = gn3(vec3f(f.u * 2.5, ca * 30.0, sa * 30.0) - f32(i));
      let furrow = 1.0 - smoothstep(0.0, 0.2, rn);
      disp += bk * (0.0035 * furrow - 0.0009 * rn2) * fa;
    }
  }
  if (capF > 0.01 && ch < 0.98) {
    // chainsaw kerf relief on fresh end grain
    let hc = hqi(vec3i(i, 19, select(0, 1, f.u > f.len * 0.5)));
    let kd = normalize(hc.yz - 0.5 + 1e-4);
    disp += 0.00035 * sin(dot(vec2f(f.x, f.y), kd) * 900.0 + 3.0 * gn3(vec3f(f.x * 25.0, f.y * 25.0, 1.0))) * capF * dfade(0.004);
  }
  return vec2f(disp, stp);
}
fn logSDF(p: vec3f, i: i32, detail: bool) -> f32 {
  let A = logs[i * 2]; let B = logs[i * 2 + 1];
  let bnd = sdCapsule(p, A.xyz, B.xyz, A.w * 1.55 + 0.01);
  if (bnd > 0.012) { return bnd; }
  let cls = logClass(A, B);
  let f = logFrame(p, A.xyz, B.xyz, i, cls);
  var d = logBase(p, f, A.w, cls, i, B.w) + burnThin(p, A.w, B.w);
  if (d < 0.03) { d = logHollows(p, i, d); }
  if (detail && d < 0.012 && !pf(8u)) {
    // relief faded in over the outer band (continuous field); steep crack walls get smaller steps
    let ld = logDisp(p, f, A, B, i, cls);
    let fo = smoothstep(0.009, 0.012, d);
    return (d + ld.x * (1.0 - fo)) * mix(ld.y, 0.9, fo);
  }
  return d * 0.85;
}

// ---------------------------------------------------------------- stones
fn stoneKind(S: vec4f) -> i32 { return i32(floor(S.w)); }
// Rock cut from an ellipsoid by 14 deep planes (45-75% of the support distance, most facing up/out), 1-2 deeper
// fracture planes taking a corner off, edges rounded by a fixed ~5 mm, low-frequency warp against symmetry.
fn stoneSDF(p: vec3f, i: i32, detail: bool) -> f32 {
  let C = stones[i * 2]; let S = stones[i * 2 + 1];
  let ext = max(S.x, max(S.y, S.z));
  let bound = max(length(p - C.xyz) - ext * 0.98, p.y - C.y - S.y * 0.9 - 0.008);
  if (bound > 0.02) { return bound; }
  var lp = rotY(p - C.xyz, C.w);
  let hs = hqi(vec3i(i, 61, 1));
  let wf = 4.0 / ext;
  lp += 0.1 * ext * vec3f(sin(lp.y * wf + hs.x * 6.0), sin(lp.z * wf * 0.8 + hs.y * 6.0), sin(lp.x * wf * 0.9 + hs.z * 6.0));
  var d = sdEllipsoid(lp, S.xyz);
  for (var k = 0; k < 16; k++) {
    let h = hqi(vec3i(i, k, 67));
    var nk = h * 2.0 - 1.0;
    if (k < 12) { nk.y = abs(nk.y) * 0.85 + 0.05; }
    nk = normalize(nk);
    let sup = length(S.xyz * nk);
    d = smax(d, dot(lp, nk) - sup * (0.47 + 0.22 * fract(h.z * 9.7)), 0.0035 + 0.002 * h.x);
  }
  // fracture planes (sharper edge)
  let nf = i32(1.0 + 1.99 * hs.z);
  for (var k = 0; k < nf; k++) {
    let h = hqi(vec3i(i, k, 71));
    let nk = normalize(vec3f(h.x - 0.5, 0.25 + 0.5 * h.y, h.z - 0.5));
    let sup = length(S.xyz * nk);
    d = smax(d, dot(lp, nk) - sup * (0.42 + 0.08 * h.y), 0.003);
  }
  if (detail && d < 0.008) {
    let sd = fract(S.w) * 97.0;
    d += ext * 0.02 * gn3(lp / ext * 5.0 + sd + 3.0) * dfade(0.01);
    let a = dfade(0.005);
    let q = lp * 38.0 + sd;
    let chip = 1.0 - abs(gn3(q));
    d += (gn3(q * 0.6) * 0.0011 - chip * chip * 0.001) * a + gn3(lp * 150.0 + sd) * 0.0003 * dfade(0.0015);
  }
  return d * 0.8;
}

// ---------------------------------------------------------------- scene
// x: distance, y: material (1 log, 2 stone, 3 ground, 4 coal chunk, 5 pebble), z: index
fn mapScene(p: vec3f, detail: bool) -> vec3f {
  let r = length(p.xz);
  var res = vec3f((p.y - groundMax(r)) * 0.9, 3.0, 0.0);
  let nearG = res.x < 0.015;
  let nearC = r < EMBER_R + 0.03 && p.y < coalMax(r) + 0.008;
  var ter = 0.0;
  if (nearG || nearC) { ter = terrain(p.xz); }
  if (nearG) {
    let gh = groundParts(p.xz, ter, detail).h;
    res.x = (p.y - gh) * select(0.85, 0.6, detail);
    if (gDet > 0.01 && r > 0.42 && p.y - gh < 0.016 && !pf(2u)) {
      let pb = pebbleField(p, gh);
      if (pb.x < res.x) { res = vec3f(pb.x, 5.0, pb.y); }
    }
  }
  if (nearC && !pf(4u)) {
    let cf = coalField(p, ter, detail);
    if (cf.x < res.x) { res = vec3f(cf.x, 4.0, cf.y); }
  } else if (r < EMBER_R + 0.03) { res.x = min(res.x, p.y - coalMax(r)); }
  // stones (bedded into the ground; blended only where the true stone SDF was evaluated)
  var dS = max(max(STONE_RMIN - r, r - STONE_RMAX), p.y - STONE_YMAX);
  if (dS < 0.02) {
    var si = -1.0; var dTrue = 1e9;
    let ns = i32(R.nStones);
    for (var i = 0; i < ns; i++) {
      let d = stoneSDF(p, i, detail);
      if (d < dTrue) { dTrue = d; si = f32(i); }
    }
    if (dTrue < res.x) { res = vec3f(dTrue, 2.0, si); }
  } else { res.x = min(res.x, dS); }
  let nl = i32(R.nLogs);
  for (var i = 0; i < nl; i++) {
    let d = logSDF(p, i, detail);
    if (d < res.x) { res = vec3f(d, 1.0, f32(i)); }
  }
  return res;
}
// Shadow casters only: logs and simplified stones (ground bumps would self-shadow into rings).
fn mapShadow(p: vec3f) -> f32 {
  var d = 1e9;
  let nl = i32(R.nLogs);
  for (var i = 0; i < nl; i++) {
    let A = logs[i * 2]; let B = logs[i * 2 + 1];
    d = min(d, sdCapsule(p, A.xyz, B.xyz, A.w) + burnThin(p, A.w, B.w));
  }
  let rr = length(p.xz);
  if (rr > STONE_RMIN - 0.03 && rr < STONE_RMAX + 0.03 && p.y < STONE_YMAX + 0.03) {
    let ns = i32(R.nStones);
    for (var i = 0; i < ns; i++) {
      let C = stones[i * 2]; let S = stones[i * 2 + 1];
      let q = p - C.xyz;
      let ext = max(S.x, max(S.y, S.z)) * 0.8;
      if (length(q) - ext > d) { continue; }
      d = min(d, sdEllipsoid(rotY(q, C.w), S.xyz * 0.72));
    }
  }
  return d;
}
// SDF of the object that was hit only (cheaper normals).
fn mapHit(p: vec3f, mat: f32, idx: i32, detail: bool) -> f32 {
  if (mat < 1.5) { return logSDF(p, idx, detail); }
  if (mat < 2.5) { return stoneSDF(p, idx, detail); }
  if (mat < 4.5) { return coalField(p, terrain(p.xz), detail).x; }
  let gh = groundH(p.xz, detail);
  return pebbleField(p, gh).x;
}
fn hitNormal(p: vec3f, e: f32, mat: f32, idx: i32, detail: bool) -> vec3f {
  let k = vec2f(1.0, -1.0);
  return normalize(k.xyy * mapHit(p + k.xyy * e, mat, idx, detail) + k.yyx * mapHit(p + k.yyx * e, mat, idx, detail) +
                   k.yxy * mapHit(p + k.yxy * e, mat, idx, detail) + k.xxx * mapHit(p + k.xxx * e, mat, idx, detail));
}
fn groundNormal(xz: vec2f, e: f32, detail: bool) -> vec3f {
  let h0 = groundH(xz, detail);
  let hx = groundH(xz + vec2f(e, 0.0), detail);
  let hz = groundH(xz + vec2f(0.0, e), detail);
  return normalize(vec3f(h0 - hx, e, h0 - hz));
}
// Soft shadow with a symmetric penumbra: a ray grazing an occluder's edge sees half the source (k = distance to the
// source / source radius); the ray keeps marching through thin occluders so umbra and penumbra both come out right.
fn softShadow(ro: vec3f, rd: vec3f, maxT: f32, k: f32, nIt: i32) -> f32 {
  var res = 1.0; var t = 0.012;
  for (var i = 0; i < nIt; i++) {
    let p = ro + rd * t;
    if (p.y > 0.58 || t > maxT) { break; }
    let h = mapShadow(p);
    res = min(res, sat(0.5 + 0.5 * k * h / t));
    if (res < 0.002) { break; }
    t += clamp(abs(h), 0.006, 0.09);
  }
  return res * res * (3.0 - 2.0 * res);
}
fn ambientOcc(p: vec3f, n: vec3f) -> f32 {
  var occ = 0.0; var sca = 1.0;
  for (var i = 1; i <= 3; i++) {
    let h = 0.006 + 0.018 * f32(i);
    occ += (h - mapScene(p + n * h, false).x) * sca;
    sca *= 0.65;
  }
  return clamp(1.0 - 3.5 * occ, 0.0, 1.0);
}
`;

// Surface pass: primary rays against the SDF scene, materials, fire lighting. Writes HDR colour + ray depth.
export const SURFACE_PASS = (NLIGHTS) => /* wgsl */`
@group(0) @binding(5) var bbT: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> lights: array<Light>;
@group(0) @binding(7) var colorOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var depthOut: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var simS: texture_3d<f32>;
@group(0) @binding(10) var lin: sampler;

fn camRay(px: vec2f) -> vec3f {
  let ndc = vec2f(px.x / R.width * 2.0 - 1.0, 1.0 - px.y / R.height * 2.0);
  let fwd = vec3f(R.fwdX, R.fwdY, R.fwdZ); let rgt = vec3f(R.rgtX, R.rgtY, R.rgtZ); let up = vec3f(R.upX, R.upY, R.upZ);
  return normalize(fwd + rgt * ndc.x * R.tanFov * R.aspect + up * ndc.y * R.tanFov);
}
// Gas temperature (K) from the fire simulation near a surface point.
fn gasTemp(p: vec3f) -> f32 {
  let g = (p - vec3f(R.ox, R.oy, R.oz)) / vec3f(R.sx, R.sy, R.sz);
  if (any(g < vec3f(0.0)) || any(g > vec3f(1.0))) { return 293.0; }
  return 293.0 + textureSampleLevel(simS, lin, g, 0.0).x * R.tempScale;
}
fn ggxSpec(n: vec3f, v: vec3f, l: vec3f, rough: f32) -> f32 {
  let h = normalize(v + l);
  let a = rough * rough; let a2 = a * a;
  let nh = max(dot(n, h), 0.0);
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let nl = max(dot(n, l), 0.0); let nv = max(dot(n, v), 1e-3);
  let k = a * 0.5;
  let G = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);
  let F = 0.04 + 0.2 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  return D * G * F / (4.0 * nv * max(nl, 1e-3));
}
// Visibility of a fire light from far away (analytic): the ray must clear the stone ring (~0.64 m, tops ~0.1 m)
// and the log pile (~0.42 m, ~0.12 m), with penumbrae from the source size; the teepee clutter takes a share.
fn farVis(p: vec3f, lp: vec3f, rs: f32) -> f32 {
  let rx = max(length(p.xz), 0.7);
  let rl = min(length(lp.xz), 0.3);
  let y1 = lp.y + (p.y - lp.y) * sat((0.64 - rl) / (rx - rl));
  let y2 = lp.y + (p.y - lp.y) * sat((0.42 - rl) / (rx - rl));
  let w1 = 2.0 * rs * (rx - 0.64) / rx + 0.02;
  let w2 = 2.0 * rs * (rx - 0.42) / rx + 0.02;
  let v1 = sat(0.5 + (y1 - 0.095) / w1);
  let v2 = mix(0.45, 1.0, sat(0.5 + (y2 - 0.12) / w2));
  return v1 * v2 * 0.9;
}
struct Shade { diff: vec3f, spec: vec3f };
// Fire light: the 8 extracted emitters as spherical area lights (inverse square with the source spread; the horizon
// wrap is limited to each source's angular size) with symmetric soft SDF shadows.
// nsh: 0 analytic visibility only, 1 rays to lights 1, 4, 7 reused by their neighbours (real time), 2 every other
// light (reused by its neighbour), 3 all lights (offline). Ray visibility fades into the analytic far-field
// visibility between 1.4 and 2.4 m from the fire, so there is no visible break.
fn fireLight(p: vec3f, n: vec3f, v: vec3f, rough: f32, nsh: i32, nIt: i32) -> Shade {
  var o = Shade(vec3f(0.0), vec3f(0.0));
  let fFar = select(1.0, smoothstep(1.4, 2.4, length(p.xz)), nsh > 0);
  // unshadowed share of each band (weak bands borrow the last traced visibility instead of tracing their own)
  var tot = 0.0;
  for (var i = 0; i < ${NLIGHTS}; i++) {
    let L = lights[i];
    if (L.pos.w <= 0.0) { continue; }
    let toL = L.pos.xyz - p;
    tot += dot(L.col.rgb, vec3f(0.3, 0.6, 0.1)) / (dot(toL, toL) + L.col.w * 0.6 + 0.004);
  }
  var shPrev = 1.0;
  for (var i = 0; i < ${NLIGHTS}; i++) {
    let L = lights[i];
    if (L.pos.w <= 0.0) { continue; }
    let toL = L.pos.xyz - p;
    let d2 = dot(toL, toL);
    let dL = sqrt(d2);
    let l = toL / max(dL, 1e-4);
    let rs = sqrt(max(L.col.w, 1e-4));
    let ndl = dot(n, l);
    let sa = clamp(rs / max(dL, 1e-3), 0.02, 0.95);
    var cosT = max(ndl, 0.0);
    if (ndl < sa) { cosT = select(0.0, (ndl + sa) * (ndl + sa) / (4.0 * sa), ndl > -sa); }
    let kk = clamp(dL / rs, 2.5, 14.0) * R.shadowSoft / 6.0;
    let share = dot(L.col.rgb, vec3f(0.3, 0.6, 0.1)) / (d2 + L.col.w * 0.6 + 0.004) / max(tot, 1e-12);
    var sh = 1.0;
    if (fFar < 1.0) {
      if (nsh > 1) {
        if ((nsh >= 3 || (i & 1) == 0) && share > 0.05) {
          shPrev = 1.0;
          if (cosT > 0.0) { shPrev = softShadow(p + n * 0.004, l, dL, kk, nIt); }
        }
        sh = shPrev;
      } else if (nsh == 1) {
        if (i % 3 == 0) {
          let Ls = lights[min(i + 1, ${NLIGHTS} - 1)];
          let toS = Ls.pos.xyz - p;
          let dS = length(toS);
          shPrev = 1.0;
          if (Ls.pos.w > 0.0 && dot(n, toS) > -0.3 * dS) {
            shPrev = softShadow(p + n * 0.004, toS / dS, dS, clamp(dS / sqrt(max(Ls.col.w, 1e-4)), 2.5, 14.0) * R.shadowSoft / 6.0, nIt);
          }
        }
        sh = shPrev;
      }
    }
    if (cosT <= 0.0) { continue; }
    if (fFar > 0.0) { sh = mix(sh, farVis(p, L.pos.xyz, rs), fFar); }
    let E = L.col.rgb * sh / (d2 + L.col.w * 0.6 + 0.004);
    o.diff += E * cosT;
    o.spec += E * max(ndl, 0.0) * ggxSpec(n, v, l, rough);
  }
  o.diff *= R.lightK; o.spec *= R.lightK;
  return o;
}
// One bounce from the fire-lit ground the face looks at: horizontal irradiance at the ground point p + n_xz * 0.22 m,
// times that point's albedo and visibility of the fire, seen by the face's view factor of the ground ((1 - n.y)/2).
// Faces turned away from the fire look at ground in their own shadow -> near-black silhouettes, as in photos.
fn groundBounce(p: vec3f, n: vec3f) -> vec3f {
  let g = vec3f(p.x + n.x * 0.22, 0.02, p.z + n.z * 0.22);
  var E = vec3f(0.0);
  for (var i = 0; i < ${NLIGHTS}; i++) {
    let L = lights[i];
    if (L.pos.w <= 0.0) { continue; }
    let toL = L.pos.xyz - g;
    let d2 = dot(toL, toL);
    let e = L.col.rgb * max(toL.y, 0.0) * inverseSqrt(d2 + 1e-6) / (d2 + L.col.w * 0.6 + 0.004);
    E += e;
  }
  let rg = length(g.xz);
  // the fire's visibility from that ground: the face's own object shadows the ground it looks at when it faces
  // away from the fire; the clutter of logs and stones takes ~35% elsewhere, the ring horizon further out
  let nxz = n.xz / max(length(n.xz), 1e-4);
  let toF = -p.xz / max(length(p.xz), 1e-4);
  let selfS = mix(0.3, 1.0, sat(0.5 + 0.8 * dot(nxz, toF)));
  var vis = 0.65;
  if (rg > 0.7) { vis = farVis(g, vec3f(0.0, 0.3, 0.0), 0.15); }
  vis *= selfS;
  let ashG = 1.0 - smoothstep(0.45, 0.6, rg);
  let alb = mix(0.1, 0.42, ashG);
  return E * R.lightK * alb * vis * 0.5 * (1.0 - n.y);
}
// Mean bed temperature (K) with its slow breathing; used for the bed's own glow light.
fn bedMeanT() -> f32 {
  return 1040.0 + 35.0 * R.glowTemp + 18.0 * gn3(vec3f(0.3, 1.7, R.time * 0.35)) * R.emberFlicker;
}
// The glowing coal bed as a wide, low, red area light (disk on the mound, Lambertian); outside the ring it is hidden
// behind the stones unless the point is high enough to see over them.
fn bedLight(p: vec3f, n: vec3f) -> vec3f {
  let c = vec3f(0.0, 0.05, 0.0);
  let toB = c - p;
  let d2 = dot(toB, toB);
  let l = toB * inverseSqrt(d2 + 1e-6);
  let a2 = 0.3 * 0.3 * 0.8;
  let cosB = sat(0.3 - 0.7 * l.y);
  let ndl = dot(n, l);
  let cosT = max(ndl * 0.8 + 0.2, 0.0) * smoothstep(-0.2, 0.15, ndl);
  let rx = length(p.xz);
  var vis = 1.0;
  if (rx > 0.6) {
    let yr = 0.05 + (p.y - 0.05) * (0.64 - 0.2) / (rx - 0.2);
    vis = sat(0.5 + (yr - 0.1) / 0.05);
  }
  let Lb = bbLoad(bbT, bedMeanT()) * 0.35 * R.emitK * R.bedGlow;   // ~35% of the bed area glows
  return Lb * PI * a2 * cosB * cosT * vis / (d2 + a2);
}

// alb: albedo, emit: blackbody radiance (x emitK later), occ: cavity occlusion of incident light,
// lit: share of the surface that still reflects fire light (glowing coal gets none), f0: specular scale
struct Surf { alb: vec3f, emit: vec3f, rough: f32, n: vec3f, occ: f32, lit: f32, spec: f32 };

// Crack/lip glow: blackbody radiance around a crack of half-width w at distance eb from its centre line: floor at Tc,
// glow climbing the lips (length Lc) to the face at Tf. Anti-aliased: features below the pixel footprint are widened
// with their energy preserved.
fn crackGlow(eb: f32, w: f32, Lc: f32, Tc: f32, Tf: f32) -> vec3f {
  let ef = bbLoad(bbT, Tf);
  let lc = 0.45 * w + 1e-5; let lcE = lc + 0.4 * gFoot;
  let ll = 0.5 * w + Lc;    let llE = ll + 0.5 * gFoot;
  let wc = exp(-eb / lcE) * lc / lcE;
  let wl = exp(-eb / llE) * ll / llE;
  let Tm = Tf + 0.55 * (Tc - Tf);
  return ef + (bbLoad(bbT, Tm) - ef) * wl + (bbLoad(bbT, Tc) - bbLoad(bbT, Tm)) * wc;
}

// ---- log end grain: off-centre pith, wobbling irregular rings, slight ovality, bark rind, sapwood band, kerf marks,
// radial drying checks; chars from the rim inwards (1-3 cm rim, all of it on hot ends) into ring/ray checking.
struct CapS { alb: vec3f, emit: vec3f, chc: f32, occ: f32 };
fn capShade(p: vec3f, f: LogF, r0: f32, cls: i32, li: i32, burn: f32, X: f32, br: f32, dSide: f32) -> CapS {
  let hc = hqi(vec3i(li, 19, select(0, 1, f.u > f.len * 0.5)));
  var pc = normalize(hc.xy - 0.5 + 1e-4) * r0 * (0.1 + 0.15 * hc.z);
  if (cls == 1) { pc = vec2f(0.0, -r0 * SPLIT_RW * SPLIT_APEX); }
  var q = vec2f(f.x, f.y) - pc;
  q.y *= 1.0 + 0.08 * hc.x;
  let rp = length(q) + 0.0025 * gn3(vec3f(q * 14.0, f32(li) * 1.3));
  let ang = atan2(q.y, q.x);
  let edge = max(-dSide, 0.0);
  let ph = rp * 230.0 + 2.6 * gnoise2(vec2f(rp * 24.0, hc.z * 50.0), 9);
  let ring = fract(ph);
  let rc = fract(floor(ph) * 0.618 + hc.x);
  let fR = dfade(0.004);
  let late = smoothstep(0.6, 0.88, ring) * (1.0 - smoothstep(0.93, 1.0, ring)) * (0.35 + 0.65 * rc) * fR;
  var wood = vec3f(0.40, 0.28, 0.16) * (0.92 + 0.12 * gn3(vec3f(q * 60.0, 3.0)));
  wood *= 1.0 - 0.3 * late;
  let rindW = 0.006 + 0.003 * gn3(vec3f(q * 30.0, 7.0));
  let sap = smoothstep(rindW, rindW + 0.004, edge) * (1.0 - smoothstep(0.022, 0.032, edge));
  wood = mix(wood * vec3f(0.85, 0.78, 0.72), wood * 1.1, sap + (1.0 - smoothstep(rindW, rindW + 0.004, edge)));
  let rind = (1.0 - smoothstep(rindW - 0.001, rindW + 0.001, edge)) * select(1.0, smoothstep(-0.2, 0.2, f.y), cls == 1);
  wood = mix(wood, vec3f(0.055, 0.04, 0.03) * (0.8 + 0.4 * gn3(vec3f(q * 200.0, 2.0))), rind);
  // radial drying checks, V-shaped, widest at the rim
  var chk = 0.0;
  let nck = 4 + i32(hc.z * 4.99);
  for (var k = 0; k < 8; k++) {
    if (k >= nck) { break; }
    let hk = hqi(vec3i(li, 23, k));
    let da = abs(fract((ang - hk.x * TAU) / TAU + 0.5) - 0.5) * TAU * rp;
    let wv = (0.0002 + 0.0012 * hk.y) * sat((rp / r0 - 0.2 - 0.35 * hk.z) * 2.0);
    chk = max(chk, (1.0 - smoothstep(0.0, wv + 0.5 * gFoot, da)) * (wv / (wv + 0.5 * gFoot + 1e-6)));
  }
  let kd = normalize(hc.yz - 0.5 + 1e-4);
  let kerf = sin(dot(q, kd) * 900.0 + 3.0 * gn3(vec3f(q * 25.0, 1.0))) * dfade(0.004);
  wood *= (1.0 - 0.75 * chk) * (1.0 + 0.03 * kerf);
  // charring from the rim in
  let heatE = heatAt(p) * burn;
  let rimW = 0.008 + 0.03 * heatE;
  var chc = max(smoothstep(-0.004, 0.004, rimW - edge), smoothstep(0.42, 0.62, heatE));
  if (cls == 2) { chc = 1.0; }
  // charred end grain: concentric checks every ~2 rings + short radial checks between them
  let ph2 = ph * 0.5;
  let ri = floor(ph2); let fr = fract(ph2);
  let dRing = min(fr, 1.0 - fr) / 115.0;
  let gate = step(0.3, fract(ri * 0.73 + hc.y));
  let Mk = max(3.0, round(TAU * (ri + 0.5) / 115.0 / 0.0075));
  let a01 = (ang / TAU + 0.5) * Mk + fract(ri * 0.37);
  let dRad = (0.5 - abs(fract(a01) - 0.5)) * TAU * max(rp, 1e-4) / Mk;
  let eb = min(select(1.0, dRing, gate > 0.5), dRad);
  let wc = 0.0003 + 0.001 * X;
  let cov = (1.0 - smoothstep(0.0, wc + 0.5 * gFoot, eb)) * (wc / (wc + 0.5 * gFoot));
  let charC = vec3f(0.03, 0.029, 0.03) * (0.8 + 0.4 * fract(ri * 0.37 + floor(a01) * 0.13));
  let alb = mix(wood, charC * (1.0 - 0.5 * cov), chc);
  let Tc = 860.0 + 250.0 * X + 40.0 * br * R.emberFlicker;
  let Tf = 420.0 + 450.0 * pow(X, 1.6) + 20.0 * br * R.emberFlicker;
  let em = crackGlow(eb, wc, 0.0006 + 0.0015 * X, Tc, Tf) * smoothstep(0.6, 1.0, chc);
  return CapS(alb, em, chc, 1.0 - 0.6 * cov * chc - 0.5 * chk * (1.0 - chc));
}

// ---- logs: bark / fresh wood -> scorch -> charcoal cobbles with glowing crack floors and lips, ash on top
fn shadeLog(p: vec3f, n: vec3f, li: i32) -> Surf {
  let A = logs[li * 2]; let B = logs[li * 2 + 1];
  let r = A.w; let cls = logClass(A, B);
  let f = logFrame(p, A.xyz, B.xyz, li, cls);
  let capF = logCap(f, r);
  let ch = logChar(p, B.w, li);
  let bark = logBarkness(f, r, cls) * (1.0 - capF);
  // exposure to flames and coals (0 .. ~1.2): sim gas temperature, coal-bed contact, static fire proximity
  let gT = gasTemp(p + n * 0.02);
  let gh = sat((gT - 650.0) / 550.0);
  let bedTop = coalTer(p.xz) + bedBase(bedRho(p.xz)) + 0.02;
  let rr = length(p.xz);
  let inBedR = 1.0 - smoothstep(0.32, 0.42, rr);
  let inBed = (1.0 - smoothstep(0.0, 0.05, p.y - bedTop)) * inBedR;
  // undersides above the coal bed are bathed in its heat
  let overBed = sat(-n.y * 1.5) * (1.0 - smoothstep(0.04, 0.25, p.y - bedTop)) * inBedR;
  let toC = normalize(vec3f(-p.x, 0.25 - p.y, -p.z));
  let facing = sat(0.5 + 0.5 * dot(n, toC));
  var X = max(max(gh, max(inBed * 0.95, overBed)), heatAt(p) * B.w * (0.4 + 0.6 * facing));
  X = clamp(X * R.glowTemp, 0.0, 1.25);
  // --- unburnt skin: bark plates and furrows / pale split wood with long dashed fibres
  let ang = atan2(f.y, f.x);
  let ca = cos(ang) * r; let sa = sin(ang) * r;
  let rn = abs(gn3(vec3f(f.u * 6.0, ca * 95.0, sa * 95.0) + f32(li) * 2.0));
  let furrow = 1.0 - smoothstep(0.0, 0.2 + gFoot * 10.0, rn);
  let g1 = gn3(vec3f(f.u * 30.0, ca * 160.0, sa * 160.0) + f32(li)) * 0.5 + 0.5;
  let barkC = mix(vec3f(0.062, 0.047, 0.036) * (0.75 + 0.5 * g1), vec3f(0.018, 0.013, 0.01), furrow) *
              (0.8 + 0.4 * (gn3(vec3f(f.u * 3.0, ca * 20.0, sa * 20.0)) * 0.5 + 0.5));
  let fib = gn3(vec3f(f.u * 8.0, f.x * 420.0, f.y * 420.0)) * dfade(0.0015);
  let fib2 = gn3(vec3f(f.u * 2.0, f.x * 90.0, f.y * 90.0)) * 0.5 + 0.5;
  let dash = smoothstep(0.2, 0.5, gn3(vec3f(f.u * 40.0, f.x * 300.0, f.y * 300.0) + 3.0));
  let woodC = vec3f(0.42, 0.30, 0.17) * (0.8 + 0.25 * fib2) * (1.0 - 0.18 * smoothstep(0.3, 0.6, fib) * dash);
  var alb = mix(woodC, barkC, bark);
  // --- scorch, then charcoal
  let scorch = smoothstep(0.0, 0.25, ch);
  alb = mix(alb, mix(vec3f(0.07, 0.035, 0.018), vec3f(0.03, 0.02, 0.015), sat(ch * 3.0)), scorch * (1.0 - 0.4 * bark));
  let cz = charCell(cls, B.w);
  let pc = logPerim(f, r, cls, li);
  let wk = 0.55 + 1.7 * charExposure(p, B.w);
  let cc = charCells(f.u, pc.x, pc.y, cz.x, cz.y, li, wk);
  let charAmt = smoothstep(0.25, 0.6, ch) * (1.0 - capF);
  let eb = max(cc.e + cc.w, 0.0);               // distance to the nearest active crack's centre line
  let inCr = sat(1.0 - eb / max(cc.w, 1e-5));  // 1 at a crack floor .. 0 at its lips
  let wEff = cc.w + 0.5 * gFoot;
  let crack = (1.0 - smoothstep(0.0, wEff, eb)) * (cc.w / wEff) * charAmt;
  // charcoal: velvety near-black 0.025-0.035, darker in the cracks
  let charC = vec3f(0.031, 0.03, 0.032) * (0.82 + 0.36 * fract(cc.id * 5.1));
  alb = mix(alb, charC, charAmt);
  alb *= 1.0 - 0.55 * crack;
  // ash: 2-6 cm patches on upward faces and cooler char, never in the cracks
  let lq = vec3f(f.u, f.x, f.y);
  let an = fb3(lq * 28.0 + f32(li) * 3.0, 2) * 0.5 + 0.5;
  let an2 = gn3(lq * 160.0) * 0.5 + 0.5;
  var ash = smoothstep(0.55, 0.7, an + 0.3 * n.y - 0.25 * X + 0.15 * (1.0 - heatAt(p))) * smoothstep(0.45, 0.85, ch);
  ash *= smoothstep(0.0, 0.0015 + gFoot, cc.e) * (0.6 + 0.4 * cc.dome) * (1.0 - capF);
  let ashC = vec3f(0.44, 0.43, 0.42) * (0.82 + 0.3 * an2);
  alb = mix(alb, ashC, ash * 0.92);
  // --- temperatures: crack floors 880-1260 K, glow climbing the lips, faces 430-900 K, the hottest band 950-1050 K;
  // per-cell breathing 0.1-0.5 Hz, not every crack reaches the hot core
  let fq = 0.1 + 0.4 * fract(cc.id * 17.3);
  let br = sin(TAU * (R.time * fq + cc.id * 7.0)) * 0.6 + gn3(vec3f(p * 7.0 + vec3f(0.0, 0.0, R.time * 0.25))) * 0.4;
  let openC = 0.6 + 0.4 * fract(cc.id2 * 3.7 + cc.id);
  let hot = smoothstep(0.85, 1.1, X) * smoothstep(0.9, 1.0, B.w);
  let Tc = 900.0 + 290.0 * X * openC + 40.0 * br * R.emberFlicker;
  let Tf = mix(430.0 + 470.0 * pow(X, 1.6), 960.0 + 70.0 * cc.dome, hot) + 25.0 * br * R.emberFlicker;
  let Lc = 0.0006 + 0.0026 * X * X;
  let glowable = smoothstep(0.35, 0.75, ch) * (1.0 - capF);
  var emit = crackGlow(eb, cc.w, Lc, Tc, Tf) * (1.0 - 0.9 * ash) * glowable;
  // a few glowing ember flecks on hot char
  let fl = gn3(lq * 520.0 + f32(li) * 3.7);
  let fleck = smoothstep(0.66, 0.76, fl) * glowable * smoothstep(0.55, 0.95, X) * (1.0 - ash) * dfade(0.002);
  emit += bbLoad(bbT, Tc - 40.0) * fleck * 0.7;
  gDbg = vec3f(X, ch, crack);
  var rough = mix(mix(0.72, 0.85, bark), 0.74 - 0.12 * cc.dome, charAmt);
  rough = mix(rough, 0.95, ash);
  var occ = 1.0 - 0.7 * crack - 0.45 * furrow * bark * (1.0 - charAmt);
  // --- end grain
  if (capF > 0.001) {
    let dSide = logSide(p, f, r, cls, li, B.w);
    let cs = capShade(p, f, r, cls, li, B.w, X, br, dSide);
    alb = mix(alb, cs.alb, capF);
    emit = mix(emit, cs.emit, capF);
    occ = mix(occ, cs.occ, capF);
    rough = mix(rough, mix(0.75, 0.6, cs.chc), capF);
  }
  // porous char is velvety: only a faint sheen (half the dielectric specular)
  return Surf(alb, emit * R.logGlow, rough, n, occ, 1.0, mix(1.0, 0.5, charAmt));
}

// ---- coal chunks: crusted tops (650-850 K, glowing only in fissures), crevices and undersides at 1100-1250 K;
// fresh / crusted / ashy chunks; fissures and grain in each chunk's own 3D frame; single embers and dead char bits
// outside the bed. Glowing coal reflects no fire light (its diffuse/specular term is dropped).
fn shadeCoal(p: vec3f, n: vec3f, id: f32) -> Surf {
  let cl = chunkCell(id);
  let ci = cl.xy; let L = cl.z;
  let pl = chunkPlace(ci, L, coalTer(p.xz), bedRadDir(p.xz / max(length(p.xz), 1e-5)));
  let G = chunkGeo(ci, L, pl.w);
  let q = (p - pl.xyz) * G.F;
  let hh = hqi(vec3i(ci, 161 + L));
  let rc = length(pl.xz);
  let rho = rc / bedRadDir(p.xz / max(length(p.xz), 1e-5));
  let fq = 0.1 + 0.4 * hh.y;
  let br = sin(TAU * (R.time * fq + hh.z * 5.0)) * 0.55 + gn3(vec3f(p.xz * 9.0, R.time * 0.3)) * 0.45;
  // fissures: transverse checks across the grain (4-8 mm apart) + splits along it, in the chunk's frame
  let sp = 0.0045 + 0.0035 * hh.x;
  let tx = q.x / sp + hh.z * 7.0 + 0.3 * gn3(q * 110.0 + hh * 9.0);
  let dT = (0.5 - abs(fract(tx) - 0.5)) * sp;
  let gate = smoothstep(-0.25, 0.15, gn3(vec3f(floor(tx) * 1.7, q.y * 80.0, q.z * 80.0) + hh * 5.0));
  let lz = abs(gn3(vec3f(q.x * 35.0, q.y * 170.0, q.z * 170.0) + hh * 3.0)) / 250.0;
  let wF = 0.00045;
  let wE = wF + 0.5 * gFoot;
  let fis = max((1.0 - smoothstep(0.0, wE, dT)) * gate, (1.0 - smoothstep(0.0, wE * 0.8, lz)) * 0.8) * (wF / wE);
  var alb = vec3f(0.03, 0.029, 0.03) * (0.85 + 0.35 * hh.x);
  var T = 400.0;
  var ashC = 0.0;
  var spec = 0.6;
  if (rho < 1.04) {
    // depth in the pile: 0 at the mean top surface .. 1 two cm down; undersides count as deep
    // where the surface faces open air at the top of the pile it is crusted (650-850 K, glowing only in fissures);
    // sides facing neighbouring chunks, undersides and everything deeper glow (980-1130 K), deepest = hot core
    let topEnv = coalTer(p.xz) + bedBase(bedRho(p.xz)) + 0.022;
    let cav = sat((topEnv - p.y) / 0.022);
    let up = sat(n.y);
    let exposed = sat((1.0 - cav) * (0.6 + 0.4 * up) * 1.6);
    let deep = smoothstep(0.55, 0.95, max(cav, sat(-n.y * 1.2)));
    let kindU = fract(hh.z * 7.7);
    let fresh = step(kindU, 0.25);
    let ashy = step(0.9 - 0.3 * smoothstep(0.6, 1.0, rho), kindU);
    let gT = gasTemp(p + vec3f(0.0, 0.02, 0.0));
    let gh = sat((gT - 650.0) / 600.0);
    // (camera response: 1000-1080 K reads deep orange, 1100 orange, >1140 K goes peach/cream -> hot core only)
    let bedHeat = sat(1.0 - rho * rho);
    let Tbody = 975.0 + 85.0 * bedHeat + 30.0 * br * R.emberFlicker + 20.0 * gh + 50.0 * (hh.y - 0.5) - select(0.0, 45.0, L == 1);
    let Tcrust = mix(650.0 + 180.0 * hh.x + 40.0 * gh, Tbody - 60.0, fresh * 0.7);
    let crustA = smoothstep(0.25, 0.7, exposed) * (1.0 - 0.75 * fis);
    T = mix(Tbody, Tcrust, crustA) + 95.0 * deep * (0.5 + 0.5 * bedHeat);
    T = mix(T, 740.0 + 150.0 * bedHeat, 0.6 * smoothstep(0.85, 1.04, rho));
    // ashy chunks: grey ash crust on the exposed top, red veins (ref. 21)
    let an = gn3(q * 260.0 + hh * 4.0) * 0.5 + 0.5;
    ashC = ashy * smoothstep(0.3, 0.65, exposed + 0.3 * an - 0.25 * fis);
    alb = mix(alb, vec3f(0.36, 0.35, 0.34) * (0.7 + 0.4 * an), ashC);
    spec = mix(0.5, 0.0, ashC);
  } else {
    // outside the bed: ~30% still-glowing embers (850-1000 K), the rest dead char bits
    if (hh.x < 0.3) { T = 850.0 + 150.0 * hh.y + 40.0 * br * R.emberFlicker - 120.0 * sat(n.y) * fract(hh.z * 3.3); }
  }
  let emit = bbLoad(bbT, T) * (1.0 - 0.92 * ashC) * R.bedGlow;
  let glow = smoothstep(760.0, 950.0, T) * (1.0 - ashC);
  gDbg = vec3f(T / 1300.0, glow, fis);
  return Surf(alb, emit, mix(0.7, 0.95, ashC), n, 1.0 - 0.4 * fis, 1.0 - glow, spec * (1.0 - glow));
}

// ---- stones: four rock kinds, +-40% tone between stones, 3-8 cm mottling (weathering rind, iron stain, damp bottom),
// mineral grains near the camera only, soot on the fire face with a fairly sharp lower edge, ash dust on fire-side tops
fn shadeStone(p: vec3f, n: vec3f, si: i32) -> Surf {
  let C = stones[si * 2]; let S = stones[si * 2 + 1];
  let lp = rotY(p - C.xyz, C.w);
  let kind = stoneKind(S);
  let sd = fract(S.w) * 97.0;
  let hs = hqi(vec3i(si, 61, 3));
  let tone = 0.62 + 0.76 * hs.x;
  let m1 = fb3(lp * 15.0 + sd, 3);
  let m2 = gn3(lp * 38.0 + sd + 7.0);
  var base = vec3f(0.29, 0.28, 0.27);
  if (kind == 1) { base = vec3f(0.36, 0.27, 0.18); }
  else if (kind == 2) { base = vec3f(0.105, 0.105, 0.11); }
  else if (kind == 3) { base = vec3f(0.43, 0.41, 0.38); }
  if (kind == 0 && hs.y > 0.6) { base = vec3f(0.33, 0.27, 0.25); }   // pink granite
  var alb = base * tone * (0.82 + 0.32 * m1);
  // weathering rind on the exposed upper faces: lighter, greyer
  let lum = dot(alb, vec3f(0.3, 0.59, 0.11));
  alb = mix(alb, vec3f(lum * 1.2), 0.3 * sat(n.y) * sat(m2 + 0.6));
  // iron stain patches
  let iron = smoothstep(0.1, 0.45, fb3(lp * 10.0 + sd + 5.0, 2)) * step(0.4, hs.z);
  alb = mix(alb, alb * vec3f(1.3, 0.85, 0.55), iron * 0.55);
  // sandstone bedding
  if (kind == 1) {
    let band = sin(dot(lp, normalize(vec3f(0.2, 1.0, 0.15))) * 220.0 + m1 * 5.0) * 0.5 + 0.5;
    alb *= 0.9 + 0.18 * band * dfade(0.006);
  }
  // mineral grains (1-4 mm), visible only close to the camera
  let fG = dfade(0.0005);
  if (fG > 0.0) {
    let gA = gn3(lp * 380.0 + sd);
    let gB = gn3(lp * 290.0 + sd + 11.0);
    let gw = 0.07 + gFoot * 100.0;
    var g = alb;
    if (kind == 0) {
      g = mix(g, g * vec3f(1.25, 1.05, 1.0), smoothstep(0.3 - gw, 0.3 + gw, gB));
      g = mix(g, vec3f(lum * 1.5), smoothstep(-0.45 + gw, -0.45 - gw, gB));
      g = mix(g, vec3f(0.05), smoothstep(0.5 - gw, 0.5 + gw, gA));
    } else if (kind == 2) {
      g *= 0.85 + 0.35 * (gA * 0.5 + 0.5);
    } else {
      g *= 0.9 + 0.2 * (gA * 0.5 + 0.5);
    }
    alb = mix(alb, g, 0.7 * fG);
  }
  // thin quartz veins on some stones, thermal fissures on others
  if (kind != 1 && hs.z > 0.55) {
    let vn = abs(gn3(lp * 11.0 + sd + 3.3));
    let vw = 0.02 + gFoot * 20.0;
    let vein = (1.0 - smoothstep(0.0, vw, vn)) * (0.02 / vw) * smoothstep(-0.2, 0.3, gn3(lp * 5.0 + sd));
    alb = mix(alb, vec3f(0.5, 0.49, 0.47), vein * 0.5);
  }
  let fz = abs(gn3(lp * 9.0 + sd + 21.0));
  let fw = 0.012 + gFoot * 8.0;
  alb *= 1.0 - 0.6 * (1.0 - smoothstep(0.0, fw, fz)) * (0.012 / fw) * step(0.5, fract(sd * 3.3));
  // damp, darker bottom; dirt where it is bedded
  let gh = groundH(p.xz, false);
  let above = p.y - gh;
  alb *= 1.0 - 0.32 * (1.0 - smoothstep(0.0, 0.035, above));
  alb = mix(alb, vec3f(0.08, 0.062, 0.047), (1.0 - smoothstep(0.0, 0.012 + 0.008 * m1, above)) * 0.7);
  // soot on the fire face: gradient upwards from a fairly sharp lower edge, patchy, strongest on the upper middle
  let toFire = normalize(vec3f(-p.x, 0.0, -p.z));
  let face = dot(n, toFire);
  let edgeH = 0.018 + 0.014 * gn3(lp * 22.0 + sd);
  let sootV = smoothstep(edgeH - 0.003, edgeH + 0.003, above) * (1.0 - 0.45 * smoothstep(0.05, 0.13, above));
  let sn = fb3(lp * 13.0 + sd + 11.0, 2) * 0.5 + 0.5;
  let soot = sat(smoothstep(-0.15, 0.55, face) * sootV * smoothstep(0.25, 0.55, sn + 0.4 * face)) * (0.7 + 0.3 * hs.y);
  alb = mix(alb, vec3f(0.02, 0.017, 0.015), soot * 0.9);
  // ash dust film on the fire-side tops
  let dust = smoothstep(0.45, 0.85, n.y) * smoothstep(-0.4, 0.3, face) * smoothstep(0.45, 0.75, sn + 0.2 * (gn3(lp * 70.0) * 0.5 + 0.5));
  alb = mix(alb, vec3f(0.42, 0.41, 0.4), dust * 0.45);
  return Surf(alb, vec3f(0.0), mix(0.62, 0.88, max(soot, dust)), n, 1.0, 1.0, 1.0);
}

// ---- pebbles: own colour (+-30%, greys, browns, the odd white quartz), darker damp underside
fn shadePebble(p: vec3f, n: vec3f, id: f32) -> Surf {
  let k = i32(id + 0.5);
  let ci = vec2i(k % 2048 - 1024, k / 2048 - 1024);
  let h = hqi(vec3i(ci, 149));
  var c = mix(vec3f(0.17, 0.15, 0.13), vec3f(0.22, 0.22, 0.21), h.x);
  if (h.y > 0.9) { c = vec3f(0.36, 0.35, 0.33); }
  else if (h.y < 0.2) { c = vec3f(0.16, 0.1, 0.07); }
  c *= 0.7 + 0.6 * h.z;
  // dirt / soot on the lower half (they sit in the soil)
  c = mix(c, vec3f(0.07, 0.055, 0.045), (1.0 - smoothstep(-0.2, 0.5, n.y)) * 0.6);
  c *= 0.85 + 0.25 * gn3(p * 300.0) * dfade(0.002);
  c *= 0.7 + 0.3 * sat(n.y + 0.3);
  return Surf(c, vec3f(0.0), 0.55, n, 1.0, 1.0, 1.0);
}

// ---- ground: forest floor (crumbly soil, needles, leaves, twigs) -> scorched ring -> ash (patchy, clumps, char bits)
// -> the coal bed's glowing base layer between the chunks
fn shadeGround(p: vec3f, n: vec3f) -> Surf {
  let xz = p.xz; let r = length(xz);
  let gp = groundParts(xz, terrain(xz), gDet > 0.01);
  let g1 = fbm2(xz * 2.2, 3, 61);
  let g2 = gnoise2(xz * 13.0, 63) * 0.5 + 0.5;
  let g3 = gnoise2(xz * 55.0, 65) * 0.5 + 0.5;
  let f3 = dfade(0.006);
  var soil = vec3f(0.085, 0.064, 0.047) * (0.7 + 0.45 * (g1 * 0.5 + 0.5)) * (0.85 + 0.3 * mix(0.5, g3, f3));
  // crumbs: lighter dry tops, dark gaps
  soil *= 0.8 + 0.35 * gp.crumb;
  // needle litter
  let pc = vec2i(floor(xz * 9.0));
  let pa = hqi(vec3i(pc, 67)).x * 3.14159;
  let rq = mat2x2f(cos(pa), sin(pa), -sin(pa), cos(pa)) * xz;
  let needle = (1.0 - smoothstep(0.0, 0.18, abs(gnoise2(rq * vec2f(30.0, 700.0), 69)))) * dfade(0.002);
  let litter = smoothstep(0.75, 1.05, r + 0.15 * g1) * (0.5 + 0.5 * smoothstep(0.3, 0.7, g2));
  soil = mix(soil, vec3f(0.16, 0.085, 0.04) * (0.8 + 0.4 * g3), needle * litter * 0.75);
  let leaf = smoothstep(0.6, 0.75, gnoise2(xz * 24.0, 71) * 0.5 + 0.5 + 0.2 * g1) * smoothstep(0.85, 1.1, r);
  soil = mix(soil, mix(vec3f(0.17, 0.085, 0.035), vec3f(0.11, 0.075, 0.03), g3), leaf * 0.8);
  // berm: freshly pushed-up, drier dirt
  soil = mix(soil, soil * vec3f(1.15, 1.1, 1.05), gp.berm * 0.5);
  // scorched soil around the ring
  let scorched = 1.0 - smoothstep(0.5, 0.82 + 0.08 * g1, r);
  soil = mix(soil, soil * vec3f(0.5, 0.47, 0.45), scorched);
  // twigs: colour varies along each twig, some ends charred
  if (gp.twig > 0.0) {
    let tA = fract(gp.twigH * 37.0);
    var twigC = mix(vec3f(0.13, 0.085, 0.05), vec3f(0.07, 0.055, 0.04), tA) * (0.75 + 0.5 * gnoise2(vec2f(gp.twigT * 9.0, gp.twigH * 40.0), 77));
    let burnt = select(0.0, smoothstep(0.55, 0.85, select(gp.twigT, 1.0 - gp.twigT, tA > 0.5)), fract(gp.twigH * 13.0) < 0.45 || r < 0.6);
    twigC = mix(twigC, vec3f(0.028, 0.025, 0.023), max(burnt, smoothstep(0.62, 0.5, r)));
    soil = mix(soil, twigC, sat(gp.twig * 1500.0));
  }
  // ash: patchy grey-white powder (0.4-0.55), brighter fluffy clumps, thin flaky edge
  let ashN = fbm2(xz * 18.0, 2, 73) * 0.5 + 0.5;
  var ashC = vec3f(0.47, 0.46, 0.445) * (0.82 + 0.3 * ashN) * (0.9 + 0.2 * mix(0.5, g3, f3));
  ashC = mix(ashC, vec3f(0.56, 0.55, 0.535), gp.clump);
  let ashAmt = gp.ash * smoothstep(0.0, 0.6, gp.ash + 0.35 * (ashN - 0.5));
  var alb = mix(soil, ashC, ashAmt);
  var emit = vec3f(0.0);
  var rough = 0.92;
  var lit = 1.0;
  var occ = 1.0 - 0.25 * (1.0 - gp.crumb) * dfade(0.004) * (1.0 - gp.ash);
  // coal bed floor: granular charcoal fines between the chunks. Enclosed crevices (chunks close above) glow at
  // 1050-1250 K; open patches are crusted (700-800 K) with ~25-45% of the granules still glowing.
  if (gp.rho < 1.1) {
    let inB = 1.0 - smoothstep(0.92, 1.1, gp.rho);
    let bh = sat(1.0 - gp.rho * gp.rho);
    let encl = 1.0 - smoothstep(0.002, 0.011, coalField(p + vec3f(0.0, 0.008, 0.0), terrain(xz), false).x);
    let fq = 0.1 + 0.4 * fract(gp.granH * 13.1);
    let br = sin(TAU * (R.time * fq + gp.granH * 5.0)) * 0.55 + gn3(vec3f(xz * 12.0, R.time * 0.3)) * 0.45;
    let gT = gasTemp(p + vec3f(0.0, 0.03, 0.0));
    let gh = sat((gT - 650.0) / 600.0);
    let gHot = step(fract(gp.granH * 7.31), 0.15 + 0.2 * bh + 0.45 * encl);
    let Thot = 960.0 + 110.0 * bh + 50.0 * (fract(gp.granH * 3.7) - 0.5) + 35.0 * br * R.emberFlicker + 25.0 * gh;
    let Tcr = 690.0 + 90.0 * fract(gp.granH * 5.3) + 30.0 * br;
    let hotA = max(gHot * mix(0.55, 1.0, gp.gran), encl * 0.85);
    let T = mix(Tcr, Thot + 125.0 * encl * encl, hotA);
    let fines = vec3f(0.038, 0.036, 0.036) * (0.75 + 0.5 * fract(gp.granH * 9.1));
    // a little grey ash on crusted granules
    let ashF = (1.0 - hotA) * smoothstep(0.55, 0.85, fract(gp.granH * 17.7)) * gp.gran;
    alb = mix(alb, mix(fines, vec3f(0.38, 0.37, 0.36), ashF * 0.7), inB);
    emit = bbLoad(bbT, T) * inB * (1.0 - 0.8 * ashF) * R.bedGlow;
    lit = 1.0 - inB * smoothstep(760.0, 950.0, T);
    rough = mix(rough, 0.8, inB);
  }
  return Surf(alb, emit, rough, n, occ, lit, mix(0.3, 0.6, gp.ash));
}

// ---- distant environment: tree trunks 4-12 m out (jittered, clustered, leaning, tapering, root flare), barely lit
const NTREES = 32;
// (x, z, base radius (0 = none), seed)
fn treeAt(k: i32) -> vec4f {
  let h = hqi(vec3i(k, 77, 3));
  if (k >= 26) {   // cluster mates next to some of the near trees
    let T = treeAt0(k - 26);
    let a = h.x * TAU; let d = 0.45 + 0.6 * h.y;
    return vec4f(T.x + cos(a) * d, T.y + sin(a) * d, select(0.0, 0.07 + 0.1 * h.z, T.z > 0.0 && h.z > 0.3), h.z * 17.0);
  }
  return treeAt0(k);
}
fn treeAt0(k: i32) -> vec4f {
  let h = hqi(vec3i(k, 77, 3));
  let rg = k / 13; let kk = k % 13;
  let a = (f32(kk) + 0.5 + 0.9 * (h.x - 0.5)) / 13.0 * TAU + f32(rg) * 0.24;
  let d = select(4.0 + 3.0 * h.y, 7.0 + 5.0 * h.y, rg == 1);
  let present = fract(h.z * 7.1) > 0.22;
  return vec4f(cos(a) * d, sin(a) * d, select(0.0, 0.09 + 0.17 * fract(h.z * 3.7), present), h.x * 31.0);
}
fn treeLean(k: i32) -> vec2f {
  let h = hqi(vec3i(k, 79, 5));
  return normalize(h.xy - 0.5 + 1e-4) * (0.035 + 0.07 * h.z);
}
fn trunkR(r0: f32, y: f32) -> f32 { return r0 * (max(1.0 - 0.05 * y, 0.45) + 0.42 * exp(-max(y, 0.0) / 0.3)); }
// -> (t, tree index) of the nearest trunk hit, t = 1e6 when none. Sheared frame removes the lean; a bounding
// cylinder (incl. the root flare) is intersected analytically, then the flared, tapering trunk is sphere-traced.
fn trunkHit(ro: vec3f, rd: vec3f) -> vec2f {
  var best = vec2f(1e6, -1.0);
  for (var k = 0; k < NTREES; k++) {
    let T = treeAt(k);
    if (T.z <= 0.0) { continue; }
    let ln = treeLean(k);
    let o = vec2f(ro.x - ln.x * ro.y - T.x, ro.z - ln.y * ro.y - T.y);
    let dd = vec2f(rd.x - ln.x * rd.y, rd.z - ln.y * rd.y);
    let a = dot(dd, dd);
    if (a < 1e-8) { continue; }
    let rb = T.z * 1.45;
    let b = dot(o, dd);
    let c = dot(o, o) - rb * rb;
    let disc = b * b - a * c;
    if (disc <= 0.0) { continue; }
    let sq = sqrt(disc);
    var t = max((-b - sq) / a, 0.0);
    let t1 = (-b + sq) / a;
    if (t1 < 0.0 || t > best.x) { continue; }
    for (var j = 0; j < 12; j++) {
      let y = ro.y + rd.y * t;
      let f = length(o + dd * t) - trunkR(T.z, y);
      if (f < 0.002) {
        if (y > -0.05 && y < 16.0 && t < best.x) { best = vec2f(t, f32(k)); }
        break;
      }
      t += f / sqrt(a) * 0.85;
      if (t > t1) { break; }
    }
  }
  return best;
}
fn stars(rd: vec3f, pixAng: f32) -> vec3f {
  if (rd.y < 0.2) { return vec3f(0.0); }
  let cell = 0.012;
  let az = atan2(rd.z, rd.x); let el = asin(clamp(rd.y, -1.0, 1.0));
  let q = vec2f(az * cos(el), el) / cell;
  let ip = vec2i(floor(q));
  let h = hqi(vec3i(ip, 91));
  if (h.z > 0.22) { return vec3f(0.0); }
  let c = (vec2f(ip) + 0.25 + 0.5 * h.xy) * cell;
  let d = length(vec2f(az * cos(el), el) - c);
  let s = max(pixAng * 0.75, 1e-5);
  let b = pow(fract(h.z * 37.0), 6.0) * 0.0016 + 0.00008;
  let tint = mix(vec3f(0.8, 0.88, 1.0), vec3f(1.0, 0.9, 0.75), fract(h.z * 91.0));
  // canopy: dark crowns hide many stars, more of them low in the sky
  let canopy = smoothstep(-0.1, 0.25, fbm2(vec2f(az * 2.2, el * 3.0), 3, 93) + 0.35 * (1.0 - smoothstep(0.35, 0.9, rd.y)));
  return tint * b * exp(-0.5 * d * d / (s * s)) * smoothstep(0.2, 0.42, rd.y) * (1.0 - canopy);
}

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= R.width || f32(id.y) >= R.height) { return; }
  gDet = clamp(R.surfDetail, 0.0, 1.0);
  let detail = gDet > 0.01;
  let px = vec2f(id.xy) + 0.5 + vec2f(R.jitX, R.jitY);
  let ro = vec3f(R.camX, R.camY, R.camZ);
  let rd = camRay(px);
  let pixAng = 2.0 * R.tanFov / R.height;
  var t = 1e6; var hit = false; var m = vec3f(0.0);
  let bmin = vec3f(-2.0, -0.05, -2.0); let bmax = vec3f(2.0, 0.62, 2.0);
  let inv = 1.0 / rd;
  let t0 = (bmin - ro) * inv; let t1 = (bmax - ro) * inv;
  let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tf > max(tn, 0.0)) {
    var tt = max(tn, 0.0);
    var det = false;
    for (var i = 0; i < 320; i++) {
      gIter = i;
      let p = ro + rd * tt;
      gFoot = max(tt * pixAng, 2e-5);
      m = mapScene(p, det);
      if (detail && !det && m.x < 0.0045) { det = true; continue; }   // raised detail relief is < ~3.3 mm
      if (m.x < 0.4 * gFoot) { hit = true; t = tt; break; }
      if (det && m.x > 0.025) { det = false; }
      tt += m.x * select(0.95, 0.8, det);
      if (tt > tf) { break; }
    }
  }
  var farGround = false;
  if (!hit && rd.y < -1e-4) {
    // ground outside the box: plane hit refined onto the height field (no seam at the box edge)
    var tg = -ro.y / rd.y;
    if (tg > 0.0) {
      for (var k = 0; k < 3; k++) {
        gFoot = tg * pixAng;
        let q = ro + rd * tg;
        tg = (groundH(q.xz, false) - ro.y) / rd.y;   // no mm-scale relief this far out (sub-pixel anyway)
      }
      t = tg; hit = true; m = vec3f(0.0, 3.0, 0.0); farGround = true;
    }
  }
  var tr = vec2f(1e6, -1.0);
  if (!hit || t > 3.5) { tr = trunkHit(ro, rd); }
  var col = vec3f(0.0);
  let moonDir = normalize(vec3f(-0.4, 0.8, -0.45));
  // light haze: one extinction for everything (ground, trunks), so distant ground and trunks fall off together
  let haze = 0.03 * R.fogK;
  if (tr.x < t) {
    // ---- tree trunk (leaning, tapering, flared roots, furrowed bark)
    t = tr.x;
    let p = ro + rd * t;
    let k = i32(tr.y);
    let T = treeAt(k);
    let ln = treeLean(k);
    let lc = vec2f(p.x - ln.x * p.y - T.x, p.z - ln.y * p.y - T.y);
    let nb = normalize(vec3f(lc.x, 0.0, lc.y));
    let a = atan2(nb.z, nb.x);
    let fA = a * T.z * 40.0;
    let fu = gn3(vec3f(fA, p.y * 4.0, T.w));
    let furrow = 1.0 - abs(fu);
    let dr = (trunkR(T.z, p.y + 0.01) - trunkR(T.z, p.y)) / 0.01;
    let tng = vec3f(-nb.z, 0.0, nb.x);
    let n = normalize(nb - vec3f(0.0, dr, 0.0) + tng * 0.5 * (gn3(vec3f(fA + 0.3, p.y * 4.0, T.w)) - fu) / 0.3 * 0.3);
    let alb = vec3f(0.075, 0.062, 0.052) * (0.5 + 0.7 * furrow * furrow) * (0.8 + 0.4 * gn3(vec3f(p.y * 0.7, a, T.w)));
    let sh = fireLight(p, n, -rd, 0.9, 0, 0);
    let moon = vec3f(0.00006, 0.00008, 0.00013) * R.moon * (max(dot(n, moonDir), 0.0) + 0.3);
    col = alb / PI * sh.diff + alb * moon * R.ambient;
    col *= exp(-max(t - 2.0, 0.0) * haze);
  } else if (hit) {
    let p = ro + rd * t;
    let v = -rd;
    gFoot = max(t * pixAng, 2e-5);
    let eps = clamp(gFoot * 0.5, 0.0002, 0.02);
    let mat = m.y;
    var n = vec3f(0.0, 1.0, 0.0);
    if (mat > 2.5 && mat < 3.5) { n = groundNormal(p.xz, eps, detail && !farGround); } else { n = hitNormal(p, eps, mat, i32(m.z), detail); }
    var S: Surf;
    if (mat < 1.5) { S = shadeLog(p, n, i32(m.z)); }
    else if (mat < 2.5) { S = shadeStone(p, n, i32(m.z)); }
    else if (mat < 3.5) { S = shadeGround(p, n); }
    else if (mat < 4.5) { S = shadeCoal(p, n, m.z); }
    else { S = shadePebble(p, n, m.z); }
    let rxz = length(p.xz);
    var nsh = 1;
    if (gDet > 0.5) { nsh = select(2, 3, gDet > 0.95); }
    if (pf(128u)) { nsh = 1; }
    let nIt = select(16, 22, gDet > 0.5);
    // glowing coal reflects nothing worth lighting: skip the light loop (and its shadow rays) there
    var sh = Shade(vec3f(0.0), vec3f(0.0));
    if (S.lit > 0.02) { sh = fireLight(p, S.n, v, S.rough, nsh, nIt); }
    // contact AO (fades out by 1 m so the far ground matches the box edge)
    var ao = 1.0;
    if (gDet > 0.3 && rxz < 1.0 && mat != 4.0 && !(mat == 3.0 && rxz < 0.3) && !pf(64u)) { ao = mix(ambientOcc(p, S.n), 1.0, smoothstep(0.8, 1.0, rxz)); }
    let bl = bedLight(p, S.n) * smoothstep(0.24, 0.4, rxz) * mix(1.0, ao, 0.7) + groundBounce(p, S.n) * ao;
    let moon = vec3f(0.00006, 0.00008, 0.00013) * R.moon * (max(dot(S.n, moonDir), 0.0) + 0.3) * ao;
    let aoD = mix(1.0, ao, 0.5);
    col = (S.alb / PI * (sh.diff * aoD + bl) + sh.spec * aoD * S.spec) * S.occ * S.lit + S.alb * moon * R.ambient * S.occ
        + S.emit * R.emitK * R.glowK;
    col *= exp(-max(t - 2.0, 0.0) * haze);
    if (R.matC > 0.5) {
      if (R.matC < 1.5) { col = S.emit * R.emitK * R.glowK; }
      else if (R.matC < 2.5) { col = S.alb * 0.02; }
      else if (R.matC < 3.5) { col = gDbg * 0.02; }
      else if (R.matC < 4.5) { col = S.alb / PI * sh.diff * S.occ * S.lit; }
      else if (R.matC < 5.5) { col = sh.diff * R.matA; }   // raw fire-light irradiance x matA (debug)
      else if (R.matC < 6.5) { col = (sh.diff + bl * PI) * R.matA; }        // irradiance incl. bounce and bed light
      else { col = vec3f(f32(gIter) / 160.0, f32(gIter) / 640.0, 0.0) * 0.06; }   // march step count (debug)
    }
  } else {
    // night sky: faint, fading to black below ~12 degrees (trees and undergrowth hide the horizon)
    let h = max(rd.y, 0.0);
    col = mix(vec3f(0.000012, 0.000014, 0.00002), vec3f(0.000004, 0.000004, 0.000008), sqrt(h)) * R.moon * smoothstep(0.1, 0.3, rd.y);
    col += stars(rd, pixAng) * R.moon;
  }
  if (R.matC > 0.5 && !hit && tr.x >= 1e6) { col = vec3f(0.0); }
  textureStore(colorOut, id.xy, vec4f(col, 1.0));
  textureStore(depthOut, id.xy, vec4f(t, 0.0, 0.0, 0.0));
}`;
export const SCENE_SURFACE = SURFACE_PASS;
