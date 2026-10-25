// HDR post chain: bloom mip pyramid (Jimenez 2014 / COD:AW style), AgX tonemap, dithered output.
// Textures: hdr (rgba16float, full res), down (rgba16float or rg11b10ufloat w/ texture-formats-tier1, mips 1..N),
// up (same format, mips 0..N-1). Bind single-mip views for storage writes; sampling another mip of the same
// texture in the same dispatch is legal (different subresources).

struct BloomParams { srcTexel: vec2<f32>, radius: f32, firstPass: u32 };
@group(0) @binding(0) var<uniform> B: BloomParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;            // linear, clamp-to-edge
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn karis(c: vec3<f32>) -> f32 { return 1.0 / (1.0 + luma(c) * 0.25); }  // 0.25: each block is a sum of 4 taps

// 13-tap downsample. uv = centre of the destination texel in source UV space.
@compute @workgroup_size(8, 8)
fn bloomDown(@builtin(global_invocation_id) id: vec3<u32>) {
  let od = textureDimensions(dst);
  if (any(id.xy >= od)) { return; }
  let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(od);
  let t = B.srcTexel;
  let S = vec2<f32>(2.0) * t;
  let a = textureSampleLevel(src, lin, uv + vec2<f32>(-S.x,  S.y), 0.0).rgb;
  let b = textureSampleLevel(src, lin, uv + vec2<f32>( 0.0,  S.y), 0.0).rgb;
  let c = textureSampleLevel(src, lin, uv + vec2<f32>( S.x,  S.y), 0.0).rgb;
  let d = textureSampleLevel(src, lin, uv + vec2<f32>(-S.x,  0.0), 0.0).rgb;
  let e = textureSampleLevel(src, lin, uv, 0.0).rgb;
  let f = textureSampleLevel(src, lin, uv + vec2<f32>( S.x,  0.0), 0.0).rgb;
  let g = textureSampleLevel(src, lin, uv + vec2<f32>(-S.x, -S.y), 0.0).rgb;
  let h = textureSampleLevel(src, lin, uv + vec2<f32>( 0.0, -S.y), 0.0).rgb;
  let i = textureSampleLevel(src, lin, uv + vec2<f32>( S.x, -S.y), 0.0).rgb;
  let j = textureSampleLevel(src, lin, uv + vec2<f32>(-t.x,  t.y), 0.0).rgb;
  let k = textureSampleLevel(src, lin, uv + vec2<f32>( t.x,  t.y), 0.0).rgb;
  let l = textureSampleLevel(src, lin, uv + vec2<f32>(-t.x, -t.y), 0.0).rgb;
  let m = textureSampleLevel(src, lin, uv + vec2<f32>( t.x, -t.y), 0.0).rgb;
  var o: vec3<f32>;
  if (B.firstPass == 1u) {
    // Karis average per 2x2 block: kills fireflies from tiny ultra-bright voxels (sparks)
    let g0 = (a + b + d + e) * 0.25; let g1 = (b + c + e + f) * 0.25;
    let g2 = (d + e + g + h) * 0.25; let g3 = (e + f + h + i) * 0.25;
    let g4 = (j + k + l + m) * 0.25;
    let w0 = karis(g0) * 0.125; let w1 = karis(g1) * 0.125; let w2 = karis(g2) * 0.125;
    let w3 = karis(g3) * 0.125; let w4 = karis(g4) * 0.5;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  textureStore(dst, id.xy, vec4<f32>(max(o, vec3<f32>(1e-4)), 1.0));
}

// 3x3 tent upsample of the coarser 'up' mip (src) added to this level's 'down' mip (curDown).
@group(0) @binding(4) var curDown: texture_2d<f32>;
@compute @workgroup_size(8, 8)
fn bloomUp(@builtin(global_invocation_id) id: vec3<u32>) {
  let od = textureDimensions(dst);
  if (any(id.xy >= od)) { return; }
  let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(od);
  let r = B.radius;                                   // in UV units, e.g. 0.005
  var s = textureSampleLevel(src, lin, uv, 0.0).rgb * 4.0;
  s += (textureSampleLevel(src, lin, uv + vec2<f32>(-r, 0.0), 0.0).rgb + textureSampleLevel(src, lin, uv + vec2<f32>(r, 0.0), 0.0).rgb
      + textureSampleLevel(src, lin, uv + vec2<f32>(0.0, -r), 0.0).rgb + textureSampleLevel(src, lin, uv + vec2<f32>(0.0, r), 0.0).rgb) * 2.0;
  s += textureSampleLevel(src, lin, uv + vec2<f32>(-r, -r), 0.0).rgb + textureSampleLevel(src, lin, uv + vec2<f32>(r, -r), 0.0).rgb
     + textureSampleLevel(src, lin, uv + vec2<f32>(-r, r), 0.0).rgb + textureSampleLevel(src, lin, uv + vec2<f32>(r, r), 0.0).rgb;
  textureStore(dst, id.xy, vec4<f32>(textureLoad(curDown, id.xy, 0).rgb + s / 16.0, 1.0));
}

// ---------------- composite + AgX + dither (fragment, fullscreen triangle) ----------------
struct Post { exposure: f32, bloomStrength: f32, frame: u32, pad: u32 };
@group(1) @binding(0) var<uniform> PP: Post;
@group(1) @binding(1) var hdr: texture_2d<f32>;
@group(1) @binding(2) var bloom: texture_2d<f32>;   // up-chain mip 0 (half res)
@group(1) @binding(3) var lin2: sampler;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn fsTriangle(@builtin(vertex_index) i: u32) -> VSOut {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));        // (0,0) (2,0) (0,2)
  var o: VSOut;
  o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2<f32>(p.x, 1.0 - p.y);
  return o;
}

// AgX (Blender/Filament/three.js formulation, Rec.2020 working space). Input/output linear sRGB.
const LIN_SRGB_TO_REC2020 = mat3x3<f32>(vec3<f32>(0.6274, 0.0691, 0.0164), vec3<f32>(0.3293, 0.9195, 0.0880), vec3<f32>(0.0433, 0.0113, 0.8956));
const LIN_REC2020_TO_SRGB = mat3x3<f32>(vec3<f32>(1.6605, -0.1246, -0.0182), vec3<f32>(-0.5876, 1.1329, -0.1006), vec3<f32>(-0.0728, -0.0083, 1.1187));
const AGX_INSET = mat3x3<f32>(vec3<f32>(0.856627153315983, 0.137318972929847, 0.11189821299995),
                              vec3<f32>(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
                              vec3<f32>(0.0482516061458583, 0.101439036467562, 0.811302368396859));
const AGX_OUTSET = mat3x3<f32>(vec3<f32>(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
                               vec3<f32>(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
                               vec3<f32>(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
fn agxContrast(x: vec3<f32>) -> vec3<f32> {
  let x2 = x * x; let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn agx(cIn: vec3<f32>) -> vec3<f32> {
  let minEv = -12.47393; let maxEv = 4.026069;
  var c = AGX_INSET * (LIN_SRGB_TO_REC2020 * cIn);
  c = clamp((log2(max(c, vec3<f32>(1e-10))) - minEv) / (maxEv - minEv), vec3<f32>(0.0), vec3<f32>(1.0));
  c = AGX_OUTSET * agxContrast(c);
  c = pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
  return clamp(LIN_REC2020_TO_SRGB * c, vec3<f32>(0.0), vec3<f32>(1.0));
}
fn srgbOetf(c: vec3<f32>) -> vec3<f32> {
  return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}
fn ign(p: vec2<f32>, frame: u32) -> f32 {                   // interleaved gradient noise, animated
  let q = p + 5.588238 * f32(frame & 63u);
  return fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn composite(in: VSOut) -> @location(0) vec4<f32> {
  let h = textureSampleLevel(hdr, lin2, in.uv, 0.0).rgb;
  let b = textureSampleLevel(bloom, lin2, in.uv, 0.0).rgb;
  let lc = mix(h, b, PP.bloomStrength) * PP.exposure;
  var o = srgbOetf(agx(lc));                      // canvas format bgra8unorm (non-sRGB) => encode here
  o += (ign(in.pos.xy, PP.frame) - 0.5) / 255.0;    // +-0.5 LSB dither against banding in smoke gradients
  return vec4<f32>(o, 1.0);
}
