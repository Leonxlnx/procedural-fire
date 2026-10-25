// ---------------- composite.wgsl ----------------
// Pass A (fsHdrComposite): heat haze -> volume over scene -> lateral CA      => HDR target (rgba16float)
// [bloom chain built from that HDR target: bloom.wgsl]
// Pass B (fsFinalize): bloom blend -> cos^4 vignette -> exposure -> tone map -> OETF -> grain -> dither
struct PostParams {
  exposure      : f32,   // scene-linear multiplier (see report: 1/(1.2*2^EV100) model or artist value)
  bloomStrength : f32,   // 0.03 .. 0.06
  bloomLevels   : f32,   // number of levels summed into up_0
  caAmount      : f32,   // 0.002 .. 0.004 radial scale difference R vs B
  vignette      : f32,   // 0..1 blend towards natural cos^4 falloff
  tanHalfFovY   : f32,
  grain         : f32,   // 0.015 .. 0.04 (display units)
  hazeGain      : f32,   // 1 = physical (Gladstone-Dale) deflection; 1..3 to exaggerate slightly
  focalPx       : f32,   // (height_px/2) / tanHalfFovY
  aspect        : f32,
  frame         : u32,
  tonemapper    : u32,   // 0 = AgX-Fire (default), 1 = ACES fitted (Hill)
};
@group(0) @binding(0) var<uniform> PP : PostParams;
@group(0) @binding(1) var sceneHdr   : texture_2d<f32>;  // opaque scene radiance (full res)
@group(0) @binding(2) var volRad     : texture_2d<f32>;  // march result upsampled to full res: rgb = L, a = Tr
@group(0) @binding(3) var volAux     : texture_2d<f32>;  // r = ∫(n_amb-n)ds, g = emission depth, b = front depth
@group(0) @binding(4) var sceneDistT : texture_2d<f32>;  // full-res distance to opaque surfaces
@group(0) @binding(5) var hdrTex     : texture_2d<f32>;  // pass B: output of pass A
@group(0) @binding(6) var bloomTex   : texture_2d<f32>;  // pass B: up_0 of the bloom chain
@group(0) @binding(7) var linClamp   : sampler;

// ---- tone mappers (input: scene-linear Rec.709/sRGB primaries; output: display-linear 0..1) ----
fn agxCurve(x: vec3f) -> vec3f {   // Sobotka AgX sigmoid, 6th-order polynomial fit (Filament / three.js)
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
// NOTE: WGSL matrices are column-major; with `v * M` and M built from ROWS, the product equals ROWS * v.
const AGXF_IN  = mat3x3f(vec3f(0.86, 0.11, 0.03), vec3f(0.16, 0.79, 0.05), vec3f(0.05, 0.10, 0.85));   // rows: sensor-like crosstalk
const AGXF_OUT = mat3x3f(vec3f(1.272727, -0.227273, -0.045455),
                         vec3f(-0.297521, 1.384298, -0.086777),
                         vec3f(-0.049587, -0.185950, 1.235537));                                        // rows: inverse of a milder crosstalk
fn tonemapAgxFire(hdr: vec3f) -> vec3f {
  let minEv = -12.47393;   // log2(0.18 * 2^-10)
  let maxEv = 4.026069;    // log2(0.18 * 2^6.5)
  var v = max(hdr, vec3f(0.0)) * AGXF_IN;
  v = clamp((log2(max(v, vec3f(1e-10))) - minEv) / (maxEv - minEv), vec3f(0.0), vec3f(1.0));
  v = agxCurve(v);
  // "punchy" look in the encoded domain (AgX look convention): power 1.2, saturation 1.2
  v = pow(max(v, vec3f(0.0)), vec3f(1.2));
  let luma = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  v = luma + 1.2 * (v - luma);
  v = pow(max(v, vec3f(0.0)), vec3f(2.2));
  return clamp(v * AGXF_OUT, vec3f(0.0), vec3f(1.0));
}

const ACES_IN  = mat3x3f(vec3f(0.59719, 0.35458, 0.04823), vec3f(0.07600, 0.90834, 0.01566), vec3f(0.02840, 0.13383, 0.83777));   // rows
const ACES_OUT = mat3x3f(vec3f(1.60475, -0.53108, -0.07367), vec3f(-0.10208, 1.10813, -0.00605), vec3f(-0.00327, -0.07276, 1.07602)); // rows
fn tonemapAcesHill(hdr: vec3f) -> vec3f {
  let v = (hdr / 0.6) * ACES_IN;    // /0.6 keeps mid-grey placement comparable to AgX (three.js convention)
  let rrt = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081);
  return clamp(rrt * ACES_OUT, vec3f(0.0), vec3f(1.0));
}

fn srgbOETF(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2f(1.0, 0.0)), u.x),
             mix(hash12(i + vec2f(0.0, 1.0)), hash12(i + vec2f(1.0, 1.0)), u.x), u.y);
}

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> VSOut {   // fullscreen triangle, draw(3)
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: VSOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

@fragment
fn fsHdrComposite(vin: VSOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(sceneHdr));
  let uv = vin.uv;
  let texel = 1.0 / dims;

  // heat haze: physical deflection = lateral gradient of ∫(n_amb - n) ds (rays bend away from the hot core)
  let aux = textureSampleLevel(volAux, linClamp, uv, 0.0);
  let gx = textureSampleLevel(volAux, linClamp, uv + vec2f(texel.x, 0.0), 0.0).r - textureSampleLevel(volAux, linClamp, uv - vec2f(texel.x, 0.0), 0.0).r;
  let gy = textureSampleLevel(volAux, linClamp, uv + vec2f(0.0, texel.y), 0.0).r - textureSampleLevel(volAux, linClamp, uv - vec2f(0.0, texel.y), 0.0).r;
  let zPlume = max(aux.g, 1e-3);
  let zBg = textureSampleLevel(sceneDistT, linClamp, uv, 0.0).r;
  // d/dx in metres = d/dpx * focalPx / zPlume ; image offset in px = angle * focalPx * (1 - zPlume/zBg)
  let angle = vec2f(gx, gy) * 0.5 * (PP.focalPx / zPlume);
  var off = -angle * PP.focalPx * max(1.0 - zPlume / max(zBg, 1e-3), 0.0) * PP.hazeGain;
  // sub-voxel shimmer the simulation cannot resolve: rising value noise scaled by the same deficit
  let q = vec2f(uv.x * PP.aspect, uv.y) * 40.0 + vec2f(0.0, f32(PP.frame) * 0.05);
  off += (vec2f(vnoise(q), vnoise(q + vec2f(7.3, 2.1))) - 0.5) * aux.r * PP.focalPx * 0.02 * PP.hazeGain;
  let behind = select(0.0, 1.0, zBg > aux.b);              // never refract things in front of the plume
  let sUV = uv + off * texel * behind;

  // lateral chromatic aberration on the background (R scaled out, B scaled in)
  let cc = sUV - 0.5;
  let sR = textureSampleLevel(sceneHdr, linClamp, 0.5 + cc * (1.0 + PP.caAmount), 0.0).r;
  let sG = textureSampleLevel(sceneHdr, linClamp, sUV, 0.0).g;
  let sB = textureSampleLevel(sceneHdr, linClamp, 0.5 + cc * (1.0 - PP.caAmount), 0.0).b;

  // volume over scene (premultiplied): L + Tr * background
  let vol = textureSampleLevel(volRad, linClamp, uv, 0.0);
  return vec4f(vol.rgb + vol.a * vec3f(sR, sG, sB), 1.0);
}

@fragment
fn fsFinalize(vin: VSOut) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(hdrTex));
  let uv = vin.uv;
  let pix = uv * dims;
  var hdr = textureSampleLevel(hdrTex, linClamp, uv, 0.0).rgb;

  // energy-conserving bloom blend (bloom chain is normalised by its level count)
  let bloom = textureSampleLevel(bloomTex, linClamp, uv, 0.0).rgb / PP.bloomLevels;
  hdr = mix(hdr, bloom, PP.bloomStrength);

  // natural vignetting cos^4(theta)
  let ndc = uv * 2.0 - 1.0;
  let tt = vec2f(ndc.x * PP.aspect, ndc.y) * PP.tanHalfFovY;
  let cos2 = 1.0 / (1.0 + dot(tt, tt));
  hdr *= mix(1.0, cos2 * cos2, PP.vignette);

  // exposure -> tone map -> OETF
  hdr *= PP.exposure;
  let ldr = select(tonemapAgxFire(hdr), tonemapAcesHill(hdr), PP.tonemapper == 1u);
  var d = srgbOETF(ldr);

  // grain: display space, mid-tone weighted, ~1.5 px clumps, re-seeded every frame
  let gp = floor(pix / 1.5) + f32(PP.frame % 1024u) * vec2f(17.0, 59.0);
  let gn = hash12(gp) + hash12(gp + vec2f(0.37, 0.71)) - 1.0;      // triangular, zero mean
  let y = dot(d, vec3f(0.2126, 0.7152, 0.0722));
  d += gn * PP.grain * 4.0 * y * (1.0 - y);
  // triangular dither to 8 bit (prevents banding in dark smoke gradients and after video encoding)
  let fr = f32(PP.frame % 64u);
  let dn = hash12(pix + fr * 13.0) + hash12(pix + vec2f(0.5, 0.25) + fr * 7.0) - 1.0;
  d += dn / 255.0;
  return vec4f(clamp(d, vec3f(0.0), vec3f(1.0)), 1.0);
}
