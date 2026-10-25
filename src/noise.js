// Generates a tileable 3D noise texture on the GPU (rgba16float).
// Channels: three independent fbm gradient noises (for curl/displacement) + one Worley-ish cell noise.
import { tex3D, computePipeline, bindGroup } from './gpu.js';
import { HASH } from './shaders/lib.js';

export function makeNoiseTexture(device, N = 128, seed = 1) {
  const tex = tex3D(device, [N, N, N], 'rgba16float', 'noise');
  const code = /* wgsl */`
${HASH}
const N = ${N}.0;
fn wrap(c: vec3i, per: i32) -> vec3i { return ((c % per) + per) % per; }
fn grad(c: vec3i, per: i32, ch: i32) -> vec3f {
  let h = hash3f(wrap(c, per) + vec3i(ch * 1013 + ${seed}, ch * 7919, ch * 104729));
  return normalize(h * 2.0 - 1.0 + vec3f(1e-5));
}
fn tnoise(p: vec3f, per: i32, ch: i32) -> f32 {
  let i = vec3i(floor(p)); let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var r = 0.0;
  var acc = array<f32, 8>();
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    acc[k] = dot(grad(i + o, per, ch), f - vec3f(o));
  }
  let x0 = mix(acc[0], acc[1], u.x); let x1 = mix(acc[2], acc[3], u.x);
  let x2 = mix(acc[4], acc[5], u.x); let x3 = mix(acc[6], acc[7], u.x);
  return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z);
}
fn tfbm(p01: vec3f, ch: i32) -> f32 {
  var s = 0.0; var a = 0.5; var per = 4;
  for (var o = 0; o < 5; o++) {
    s += a * tnoise(p01 * f32(per), per, ch);
    per *= 2; a *= 0.5;
  }
  return s * 1.5;
}
fn worley(p01: vec3f, per: i32) -> f32 {
  let p = p01 * f32(per);
  let i = vec3i(floor(p)); let f = fract(p);
  var d = 9.0;
  for (var k = 0; k < 27; k++) {
    let o = vec3i(k % 3 - 1, (k / 3) % 3 - 1, k / 9 - 1);
    let h = hash3f(wrap(i + o, per) + vec3i(77, 33, ${seed}));
    d = min(d, length(vec3f(o) + h - f));
  }
  return d;
}
@group(0) @binding(0) var outT: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = (vec3f(id) + 0.5) / N;
  let a = tfbm(p, 0); let b = tfbm(p, 1); let c = tfbm(p, 2);
  let w = 0.65 * worley(p, 8) + 0.35 * worley(p, 16);
  textureStore(outT, id, vec4f(a * 0.5 + 0.5, b * 0.5 + 0.5, c * 0.5 + 0.5, clamp(w, 0.0, 1.0)));
}`;
  const pipe = computePipeline(device, code, 'main', 'noiseGen');
  const bg = bindGroup(device, pipe, [tex]);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(N / 4, N / 4, N / 4);
  pass.end();
  device.queue.submit([enc.finish()]);
  return tex;
}

// Curl of the (r,g,b) vector potential of the noise texture, baked once (periodic central differences).
export function makeCurlTexture(device, noiseTex, N = 128) {
  const tex = tex3D(device, [N, N, N], 'rgba16float', 'curl');
  const code = /* wgsl */`
@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var outT: texture_storage_3d<rgba16float, write>;
fn L(c: vec3i) -> vec3f { return textureLoad(src, (c + ${N}) % ${N}, 0).xyz * 2.0 - 1.0; }
@compute @workgroup_size(4, 4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  let c = vec3i(id);
  let dx = L(c + vec3i(1, 0, 0)) - L(c - vec3i(1, 0, 0));
  let dy = L(c + vec3i(0, 1, 0)) - L(c - vec3i(0, 1, 0));
  let dz = L(c + vec3i(0, 0, 1)) - L(c - vec3i(0, 0, 1));
  let curl = vec3f(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) * (${N}.0 / 2.0);
  textureStore(outT, id, vec4f(curl, length(curl)));
}`;
  const pipe = computePipeline(device, code, 'main', 'curlGen');
  const bg = bindGroup(device, pipe, [noiseTex, tex]);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(N / 4, N / 4, N / 4);
  pass.end();
  device.queue.submit([enc.finish()]);
  return tex;
}
