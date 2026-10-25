// ---------------- bloom.wgsl : Jimenez 2014 (CoD:AW) mip-chain bloom, energy conserving ----------------
struct BloomParams { srcTexel: vec2f, firstPass: u32, radius: f32 };   // srcTexel = 1/size of the texture being sampled
@group(0) @binding(0) var srcTex   : texture_2d<f32>;    // downsample: previous level. upsample: lower (smaller) up-chain level
@group(0) @binding(1) var linClamp : sampler;
@group(0) @binding(2) var dstTex   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> B : BloomParams;
@group(0) @binding(4) var curTex   : texture_2d<f32>;    // upsample only: this level's downsample result

fn tap(uv: vec2f, o: vec2f) -> vec3f { return textureSampleLevel(srcTex, linClamp, uv + o * B.srcTexel, 0.0).rgb; }
fn karisW(c: vec3f) -> f32 { return 1.0 / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722))); }

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dstTex);
  if (any(gid.xy >= dims)) { return; }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
  let a = tap(uv, vec2f(-2.0, -2.0)); let b = tap(uv, vec2f(0.0, -2.0)); let c = tap(uv, vec2f(2.0, -2.0));
  let d = tap(uv, vec2f(-2.0,  0.0)); let e = tap(uv, vec2f(0.0,  0.0)); let f = tap(uv, vec2f(2.0,  0.0));
  let g = tap(uv, vec2f(-2.0,  2.0)); let h = tap(uv, vec2f(0.0,  2.0)); let i = tap(uv, vec2f(2.0,  2.0));
  let j = tap(uv, vec2f(-1.0, -1.0)); let k = tap(uv, vec2f(1.0, -1.0));
  let l = tap(uv, vec2f(-1.0,  1.0)); let m = tap(uv, vec2f(1.0,  1.0));
  var o: vec3f;
  if (B.firstPass == 1u) {
    // Karis average on the first downsample only: weights 1/(1+luma) per 2x2 group kill flame fireflies
    let g0 = (j + k + l + m) * 0.25;
    let g1 = (a + b + d + e) * 0.25; let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25; let g4 = (e + f + h + i) * 0.25;
    let w0 = 0.5 * karisW(g0);
    let w1 = 0.125 * karisW(g1); let w2 = 0.125 * karisW(g2);
    let w3 = 0.125 * karisW(g3); let w4 = 0.125 * karisW(g4);
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    // weights sum to 1: 0.125 + 4*0.03125 + 4*0.0625 + 4*0.125
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  textureStore(dstTex, vec2i(gid.xy), vec4f(o, 1.0));
}

// up_i = down_i + tent3x3(up_{i+1});   final bloom = up_0 / numLevels  (each level energy-preserving)
@compute @workgroup_size(8, 8)
fn upsample(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dstTex);
  if (any(gid.xy >= dims)) { return; }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
  let r = B.radius;   // in source texels, 1.0 is a good default
  var s = tap(uv, vec2f(0.0)) * 4.0;
  s += (tap(uv, vec2f(-r, 0.0)) + tap(uv, vec2f(r, 0.0)) + tap(uv, vec2f(0.0, -r)) + tap(uv, vec2f(0.0, r))) * 2.0;
  s += tap(uv, vec2f(-r, -r)) + tap(uv, vec2f(r, -r)) + tap(uv, vec2f(-r, r)) + tap(uv, vec2f(r, r));
  s *= 1.0 / 16.0;
  let cur = textureLoad(curTex, vec2i(gid.xy), 0).rgb;
  textureStore(dstTex, vec2i(gid.xy), vec4f(cur + s, 1.0));
}
