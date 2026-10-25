// Volume ray marcher for the fire grid (emission-absorption + optional smoke lighting) and a depth-aware
// upsampler for half-resolution rendering.
// scal texture (rgba16float, filterable): x = temperature (normalised 0..1+), y = smoke density, z = fuel, w = age
// occ texture (r16float): per-brick (8^3 voxels) max extinction, for empty-space skipping.
// blackbody LUT (rgba16float 256x1): linear-sRGB radiance vs normalised temperature (built on the CPU from Planck).

struct Cam {
  invViewProj: mat4x4<f32>,     // world <- clip
  worldToVol: mat4x4<f32>,      // world -> volume [0,1]^3
  camPos: vec4<f32>,            // world
  res: vec2<f32>,               // render-target size (half res when upsampling)
  stepVox: f32,                 // step in voxels (0.5 = Nyquist-ish)
  frame: u32,
  gridDims: vec3<f32>,
  emissionScale: f32,
  sigmaSmoke: f32,              // extinction per unit density per voxel length
  sigmaFire: f32,
  glowStrength: f32,
  pad: f32,
};
@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;                   // linear, clamp-to-edge
@group(0) @binding(3) var bbLut: texture_2d<f32>;
@group(0) @binding(4) var occ: texture_3d<f32>;
@group(0) @binding(5) var glow: texture_3d<f32>;          // emission volume with a mip chain (blurred light)
@group(0) @binding(6) var outColor: texture_storage_2d<rgba16float, write>;   // rgb = radiance, a = 1 - T
@group(0) @binding(7) var outDepth: texture_storage_2d<r32float, write>;      // transmittance-weighted depth

fn ign(p: vec2<f32>, frame: u32) -> f32 {
  let q = p + 5.588238 * f32(frame & 63u);
  return fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
}
fn slab(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  let inv = 1.0 / rd;                                    // rd components of exactly 0 give +-inf: fine for slabs
  let t0 = -ro * inv; let t1 = (vec3<f32>(1.0) - ro) * inv;
  let tmin = min(t0, t1); let tmax = max(t0, t1);
  return vec2<f32>(max(max(tmin.x, tmin.y), max(tmin.z, 0.0)), min(min(tmax.x, tmax.y), tmax.z));
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(vec2<f32>(id.xy) >= C.res)) { return; }
  let ndc = vec2<f32>((f32(id.x) + 0.5) / C.res.x * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / C.res.y * 2.0);
  let pn = C.invViewProj * vec4<f32>(ndc, 0.0, 1.0);    // WebGPU clip depth is [0,1]
  let pf = C.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  let roW = pn.xyz / pn.w;
  let rdW = normalize(pf.xyz / pf.w - roW);
  let ro = (C.worldToVol * vec4<f32>(roW, 1.0)).xyz;
  let rdU = (C.worldToVol * vec4<f32>(rdW, 0.0)).xyz;   // not unit length in volume space
  let worldPerVol = 1.0 / length(rdU);                  // world length per unit of volume-space t
  let rd = rdU * worldPerVol;
  let hit = slab(ro, rd);
  var L = vec3<f32>(0.0);
  var T = 1.0;
  var depthAcc = 0.0; var wAcc = 0.0;
  if (hit.x < hit.y) {
    let vox = 1.0 / max(C.gridDims.x, max(C.gridDims.y, C.gridDims.z));
    let dt = C.stepVox * vox;                          // volume-space step
    var t = hit.x + dt * ign(vec2<f32>(id.xy), C.frame);  // jittered start: banding -> noise
    let brick = 8.0 * vox;
    loop {
      if (t >= hit.y || T < 0.003) { break; }
      let p = ro + rd * t;
      // empty-space skipping: jump to the end of an empty 8^3 brick
      // (occ must be built over each brick plus a 1-voxel halo so trilinear footprints are covered)
      if (textureSampleLevel(occ, lin, p, 0.0).x < 1e-3) {
        let cell = floor(p / brick);
        let tb = slab((ro - cell * brick) / brick, rd);     // brick exit, measured from ro, in brick units
        t = max(tb.y * brick + 0.01 * dt, t + dt);
        continue;
      }
      let s = textureSampleLevel(scal, lin, p, 0.0);
      let sigma = (C.sigmaSmoke * s.y + C.sigmaFire * s.x) / vox;   // per volume-space unit
      var Le = textureSampleLevel(bbLut, lin, vec2<f32>(clamp(s.x, 0.0, 1.0), 0.5), 0.0).rgb * C.emissionScale;
      // smoke lit by the fire: sample a coarse mip of the emission volume (a cheap "blurred irradiance")
      Le += s.y * C.glowStrength * textureSampleLevel(glow, lin, p, 3.0).rgb;
      let a = exp(-sigma * dt);
      // Hillaire 2015: integrate emission analytically over the step (energy conserving for any sigma*dt)
      L += T * Le * select((1.0 - a) / max(sigma, 1e-8), dt, sigma < 1e-5);
      depthAcc += T * (1.0 - a) * t; wAcc += T * (1.0 - a);
      T *= a;
      t += dt;
    }
  }
  textureStore(outColor, id.xy, vec4<f32>(L, 1.0 - T));
  textureStore(outDepth, id.xy, vec4<f32>(select(1e9, depthAcc / max(wAcc, 1e-6), wAcc > 1e-4), 0.0, 0.0, 0.0));
}

// ---- depth-aware (joint bilateral) 2x upsample of the half-res volume buffer ----
@group(1) @binding(0) var lowColor: texture_2d<f32>;
@group(1) @binding(1) var lowDepth: texture_2d<f32>;
@group(1) @binding(2) var fullDepth: texture_2d<f32>;   // scene depth (logs / ground) in the same units as t
@group(1) @binding(3) var fullOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn upsample(@builtin(global_invocation_id) id: vec3<u32>) {
  let od = textureDimensions(fullOut);
  if (any(id.xy >= od)) { return; }
  let ld = vec2<i32>(textureDimensions(lowColor));
  let zf = textureLoad(fullDepth, id.xy, 0).x;
  let pl = (vec2<f32>(id.xy) + 0.5) * 0.5 - 0.5;        // position in low-res texel space
  let base = vec2<i32>(floor(pl));
  let f = pl - vec2<f32>(base);
  var acc = vec4<f32>(0.0); var wsum = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2<i32>(k & 1, k >> 1);
    let q = clamp(base + o, vec2<i32>(0), ld - 1);
    let wb = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    let zl = textureLoad(lowDepth, q, 0).x;
    // volume samples that lie behind the opaque surface get no weight
    let wd = select(1.0, 0.02, zl > zf);
    let w = wb * wd + 1e-4;
    acc += textureLoad(lowColor, q, 0) * w; wsum += w;
  }
  textureStore(fullOut, id.xy, acc / wsum);
}
