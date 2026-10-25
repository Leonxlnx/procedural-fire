// ---------------- common.wgsl (prepend to every fire module) ----------------
// blackbody.wgsl additionally needs the module to declare  bbLut: texture_2d<f32>  and  linClamp: sampler.
const PI = 3.14159265359;

// Jimenez 2014 interleaved gradient noise. The per-frame scroll of 5.588238 px (Jimenez via demofox)
// makes each pixel's sequence low-discrepancy over 64 frames -> ideal for temporal accumulation.
fn ign(pix: vec2f, frame: u32) -> f32 {
  let p = pix + 5.588238 * f32(frame % 64u);
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3f) -> vec3f {
  var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + 33.33);
  return fract((q.xxy + q.yxx) * q.zyx);
}

// Slab test. Returns (tNear, tFar); empty if x >= y.
fn rayBox(ro: vec3f, rd: vec3f, bmin: vec3f, bmax: vec3f) -> vec2f {
  let safe = select(rd, vec3f(1e-8), abs(rd) < vec3f(1e-8));
  let inv = 1.0 / safe;
  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  return vec2f(max(max(tmin.x, tmin.y), max(tmin.z, 0.0)), min(min(tmax.x, tmax.y), tmax.z));
}

// Integral of exp(-sigT*s) over [0, dt]  (Hillaire 2015 "energy-conserving" step integration).
fn segIntegral(sigT: f32, dt: f32) -> f32 {
  let x = sigT * dt;
  return select((1.0 - exp(-x)) / max(sigT, 1e-12), dt * (1.0 - 0.5 * x), x < 1e-3);
}

// Henyey-Greenstein; cosTheta = dot(light propagation dir, scattered dir towards camera)
fn hgPhase(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

