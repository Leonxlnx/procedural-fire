// ---------------- surface.wgsl : fire-lit ground / logs, char + glowing ember cracks ----------------
struct VPL { posRadius: vec4f, intensity: vec4f };
struct SurfParams {
  fire    : vec4f,     // same R, contrast as the volume -> emission colours match exactly
  misc    : vec4f,     // x = time [s], y = ember emissivity (0.9), z = moon/sky irradiance scale, w = unused
  moonDir : vec4f,     // xyz = direction to moon, w = unused
  moonCol : vec4f,     // rgb = moon irradiance (scene units, after camera WB this is bluish)
};
@group(0) @binding(0) var<uniform> SP : SurfParams;
@group(0) @binding(1) var<storage, read> surfVpls : array<VPL, 8>;   // 8 clustered lights (octants of the 8^3 grid)
@group(0) @binding(2) var bbLut    : texture_2d<f32>;
@group(0) @binding(3) var linClamp : sampler;

// Irradiance from the fire at surface point p with normal n (Lambert receivers).
// I = sum(j dV) is radiant intensity, so E = I cos(theta) / d^2; the r^2 term softens like a sphere light.
fn fireIrradiance(p: vec3f, n: vec3f) -> vec3f {
  var E = vec3f(0.0);
  for (var k = 0u; k < 8u; k++) {
    let v = surfVpls[k];
    let d = v.posRadius.xyz - p;
    let d2 = dot(d, d);
    let l = d * inverseSqrt(max(d2, 1e-8));
    let r2 = v.posRadius.w * v.posRadius.w;
    // horizon-wrapped cosine for a source of angular radius ~ r/d (avoids hard terminator on logs)
    let sinA = sqrt(r2 / (d2 + r2));
    let ndl = clamp((dot(n, l) + sinA) / (1.0 + sinA), 0.0, 1.0);
    E += v.intensity.rgb * ndl / (d2 + r2);
  }
  return E;
}

// 3D Voronoi F1, F2 (distances).  F2 - F1 ~ distance to the nearest cell border -> crack lines.
fn voronoiF1F2(p: vec3f) -> vec2f {
  let c = floor(p);
  let f = fract(p);
  var d1 = 8.0;
  var d2 = 8.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3f(f32(x), f32(y), f32(z));
        let r = o + hash33(c + o) - f;
        let d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
      }
    }
  }
  return sqrt(vec2f(d1, d2));
}

struct CharSample { albedo: vec3f, emission: vec3f, roughness: f32 };

// pObj in metres (object space), grainAxis = log axis, heat in 0..1 (how strongly this spot is burning:
// e.g. from fireIrradiance luminance, height above the coals, or a painted/simulated "burn" attribute).
fn charredWood(pObj: vec3f, grainAxis: vec3f, heat: f32, time: f32) -> CharSample {
  // "alligator" char: blocks elongated across the grain -> compress the coordinate along the grain
  let along = dot(pObj, grainAxis);
  let q = (pObj - grainAxis * along) + grainAxis * (along * 0.4);
  let v = voronoiF1F2(q * 70.0);                       // ~1-2 cm char blocks
  let edge = v.y - v.x;
  let crack = 1.0 - smoothstep(0.03, 0.14, edge);      // 1 inside the crack
  let cellId = hash33(floor(q * 70.0) + vec3f(3.1, 7.7, 1.3));
  // embers "breathe" with oxygen supply: 0.1-0.5 Hz per cell + slow global drift
  let breath = 0.8 + 0.2 * sin(time * (0.6 + 2.4 * cellId.x) + 6.2831 * cellId.y);
  let Tcrack = mix(650.0, 1250.0, heat) * breath;      // crack bottoms: 900-1250 K when hot
  let Tface  = mix(450.0, 1000.0, heat * heat);        // block faces: mostly dark, glow only when very hot
  let T = mix(Tface, Tcrack, crack);
  var s: CharSample;
  // Kirchhoff again: emissivity ~0.9 for char; same R and contrast as the flames so hues are coherent.
  s.emission = SP.misc.y * SP.fire.x * blackbody(T, SP.fire.y) * smoothstep(760.0, 880.0, T);
  let ash = smoothstep(0.5, 0.95, heat) * (1.0 - crack) * (0.4 + 0.6 * cellId.z);
  s.albedo = mix(vec3f(0.028, 0.024, 0.021), vec3f(0.42, 0.40, 0.37), ash);   // char 0.02-0.04, wood ash 0.35-0.5
  s.roughness = mix(0.8, 0.95, ash);
  return s;
}

// Soot deposit on stones/ground near the fire: darken albedo with height above the fire base and proximity.
fn sootDarken(albedo: vec3f, p: vec3f, fireBase: vec3f, radius: f32) -> vec3f {
  let dxz = length(p.xz - fireBase.xz);
  let s = (1.0 - smoothstep(0.3 * radius, 1.2 * radius, dxz)) * smoothstep(-0.05, 0.25, p.y - fireBase.y);
  return albedo * mix(1.0, 0.25, s);
}

@fragment
fn fsLog(@location(0) pObj: vec3f, @location(1) nW: vec3f, @location(2) pW: vec3f) -> @location(0) vec4f {
  let n = normalize(nW);
  let E = fireIrradiance(pW, n);
  let heat = clamp(log2(1.0 + dot(E, vec3f(0.2126, 0.7152, 0.0722))) * 0.1, 0.0, 1.0);
  let s = charredWood(pObj, vec3f(1.0, 0.0, 0.0), heat, SP.misc.x);
  let moon = SP.moonCol.rgb * max(dot(n, SP.moonDir.xyz), 0.0);
  let Lo = s.albedo / PI * (E + moon) + s.emission;
  return vec4f(Lo, 1.0);
}
