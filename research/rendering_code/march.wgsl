// ---------------- march.wgsl : emission-absorption-scattering ray march (compute, half res) ----------------
struct MarchParams {
  invViewProj  : mat4x4f,
  camPos       : vec4f,   // xyz camera position (world, metres)
  boxMin       : vec4f,   // xyz world AABB of the sim grid (or tighter occupancy AABB)
  boxMax       : vec4f,
  stepInfo     : vec4f,   // x = step length [m] (0.5..1 voxel), y = max steps, z = frame index, w = T_ambient [K]
  fire         : vec4f,   // x = R radiance scale of an optically thick blackbody, y = contrast, z = T_onset [K], w = kappa_flame [1/m per unit flame]
  smoke        : vec4f,   // x = soot sigma_a [1/m per unit density], y = sigma_s [1/m per unit density], z = HG g, w = unused
  ambient      : vec4f,   // rgb = isotropic sky/moon in-scatter radiance (already * phase), w unused
  fireCentroid : vec4f,   // xyz = luminance-weighted emission centroid (from the reduction pass)
  outSize      : vec4f,   // xy = target size, zw = 1/size
};

@group(0) @binding(0) var<uniform> P : MarchParams;
@group(0) @binding(1) var fields    : texture_3d<f32>;   // rgba16float: r = smoke/soot density, g = temperature [K], b = flame/reaction 0..1
@group(0) @binding(2) var fireLight : texture_3d<f32>;   // rgba16float, low res: rgb = fluence from fire = sum_k I_k Tr_k / d_k^2
@group(0) @binding(3) var bbLut     : texture_2d<f32>;
@group(0) @binding(4) var linClamp  : sampler;
@group(0) @binding(5) var sceneDist : texture_2d<f32>;   // r32float: distance along the ray to the opaque scene (min-downsampled)
@group(0) @binding(6) var outRad    : texture_storage_2d<rgba16float, write>;  // rgb = radiance added, a = transmittance
@group(0) @binding(7) var outAux    : texture_storage_2d<rgba16float, write>;  // r = ∫(n_amb - n)ds [m], g = emission-weighted depth, b = front depth

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2u(P.outSize.xy);
  if (any(gid.xy >= size)) { return; }
  let pix = vec2f(gid.xy) + 0.5;
  let ndc = vec2f(pix.x * P.outSize.z * 2.0 - 1.0, 1.0 - pix.y * P.outSize.w * 2.0);
  let farH = P.invViewProj * vec4f(ndc, 1.0, 1.0);
  let ro = P.camPos.xyz;
  let rd = normalize(farH.xyz / farH.w - ro);

  var seg = rayBox(ro, rd, P.boxMin.xyz, P.boxMax.xyz);
  seg.y = min(seg.y, textureLoad(sceneDist, vec2i(gid.xy), 0).r);
  if (seg.x >= seg.y) {
    textureStore(outRad, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 1.0));
    textureStore(outAux, vec2i(gid.xy), vec4f(0.0, 0.0, 1e4, 0.0));
    return;
  }

  let dt = P.stepInfo.x;
  let maxSteps = u32(P.stepInfo.y);
  let frame = u32(P.stepInfo.z);
  let Tamb = P.stepInfo.w;
  let invExt = 1.0 / (P.boxMax.xyz - P.boxMin.xyz);
  let g = P.smoke.z;

  var t = seg.x + dt * ign(pix, frame);          // jittered start: turns banding into noise that TAA/accumulation removes
  var L = vec3f(0.0);
  var Tr = 1.0;
  var dnInt = 0.0;                                // ∫ (n_amb - n) ds for heat haze
  var zAcc = 0.0; var wAcc = 0.0; var zFront = 1e4;

  for (var i = 0u; i < maxSteps; i++) {
    if (t >= seg.y || Tr < 0.003) { break; }
    let x = ro + rd * t;
    let uvw = (x - P.boxMin.xyz) * invExt;
    let f = textureSampleLevel(fields, linClamp, uvw, 0.0);
    let rho = f.r;
    let Tk = max(f.g, Tamb);
    let flame = f.b;

    // Extinction. Flame soot both absorbs and emits (Kirchhoff): thick flame saturates at R*B(T).
    let sigA = P.smoke.x * rho + P.fire.w * flame;
    let sigS = P.smoke.y * rho;
    let sigT = sigA + sigS;

    // Emission  j = sigma_a * B(T)   (B from the Planck LUT, R = global radiance scale)
    let onset = smoothstep(P.fire.z, P.fire.z + 200.0, Tk);
    let j = P.fire.x * blackbody(Tk, P.fire.y) * (onset * sigA);

    // Single scattering of fire light (HG towards the emission centroid) + ambient
    let F = textureSampleLevel(fireLight, linClamp, uvw, 0.0).rgb;
    let dIn = normalize(x - P.fireCentroid.xyz + vec3f(1e-5));
    let Ls = F * hgPhase(dot(dIn, -rd), g) + P.ambient.rgb;

    let S = j + sigS * Ls;
    let w = segIntegral(sigT, dt);
    L += Tr * S * w;

    // bookkeeping for post: refractive-index deficit (Gladstone-Dale, n-1 = 2.93e-4 * 273/T), depths
    dnInt += 2.93e-4 * 273.0 * (1.0 / Tamb - 1.0 / Tk) * dt;
    let lum = dot(j, vec3f(0.2126, 0.7152, 0.0722)) * Tr;
    zAcc += lum * t; wAcc += lum;
    if (Tk > Tamb + 60.0 && zFront > 9e3) { zFront = t; }

    Tr *= exp(-sigT * dt);
    t += dt;
  }
  textureStore(outRad, vec2i(gid.xy), vec4f(L, Tr));
  textureStore(outAux, vec2i(gid.xy), vec4f(dnInt, select(seg.y, zAcc / wAcc, wAcc > 0.0), zFront, 0.0));
}
