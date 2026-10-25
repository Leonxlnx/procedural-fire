// Shared WGSL snippets.

export const HASH = /* wgsl */`
fn pcg3d(vin: vec3u) -> vec3u {
  var v = vin * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn pcg(n: u32) -> u32 {
  let s = n * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash3f(p: vec3i) -> vec3f { return vec3f(pcg3d(bitcast<vec3u>(p))) * (1.0 / 4294967295.0); }
fn hash1u(n: u32) -> f32 { return f32(pcg(n)) * (1.0 / 4294967295.0); }
fn rand2(n: u32, m: u32) -> f32 { return f32(pcg(n ^ pcg(m))) * (1.0 / 4294967295.0); }
`;

// Interleaved gradient noise for ray-march jitter.
export const IGN = /* wgsl */`
fn ign(px: vec2f, frame: f32) -> f32 {
  let p = px + 5.588238 * (frame % 64.0);
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}
`;

// Analytic 3D gradient noise (value range ~[-1,1]).
export const GNOISE = /* wgsl */`
fn gradDot(ci: vec3i, f: vec3f) -> f32 {
  let h = hash3f(ci) * 2.0 - 1.0;
  return dot(normalize(h + vec3f(1e-4)), f);
}
fn gnoise(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n000 = gradDot(i + vec3i(0,0,0), f - vec3f(0,0,0));
  let n100 = gradDot(i + vec3i(1,0,0), f - vec3f(1,0,0));
  let n010 = gradDot(i + vec3i(0,1,0), f - vec3f(0,1,0));
  let n110 = gradDot(i + vec3i(1,1,0), f - vec3f(1,1,0));
  let n001 = gradDot(i + vec3i(0,0,1), f - vec3f(0,0,1));
  let n101 = gradDot(i + vec3i(1,0,1), f - vec3f(1,0,1));
  let n011 = gradDot(i + vec3i(0,1,1), f - vec3f(0,1,1));
  let n111 = gradDot(i + vec3i(1,1,1), f - vec3f(1,1,1));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
             mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z) * 1.6;
}
fn fbm3(p: vec3f, oct: i32) -> f32 {
  var a = 0.5; var s = 0.0; var q = p;
  for (var i = 0; i < oct; i++) { s += a * gnoise(q); q = q * 2.03 + vec3f(17.1, 3.7, 9.2); a *= 0.5; }
  return s;
}
`;

export const SDF = /* wgsl */`
fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
// returns (distance, h along axis)
fn sdCapsuleH(p: vec3f, a: vec3f, b: vec3f, r: f32) -> vec2f {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return vec2f(length(pa - ba * h) - r, h);
}
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
`;
