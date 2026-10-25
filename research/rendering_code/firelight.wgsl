// ---------------- firelight.wgsl : emission -> virtual point lights -> fluence volume ----------------
struct VPL { posRadius: vec4f, intensity: vec4f };   // xyz = luminance-weighted centroid, w = softening radius; rgb = radiant intensity, a = luminance
struct RParams {
  boxMin    : vec4f,
  voxelSize : vec4f,   // xyz = world size of one sim voxel
  fire      : vec4f,   // same as MarchParams.fire  (R, contrast, T_onset, kappa_flame)
  smoke     : vec4f,   // same as MarchParams.smoke (soot sigma_a, sigma_s, g, -)
  ext       : vec4f,   // xyz = box extent (max - min); w = shadow sigma_t per unit low-res density
  counts    : vec4u,   // x = number of VPLs used by buildFireLight
};
@group(0) @binding(0) var<uniform> RP : RParams;
@group(0) @binding(1) var fields   : texture_3d<f32>;
@group(0) @binding(2) var bbLut    : texture_2d<f32>;
@group(0) @binding(3) var linClamp : sampler;
@group(0) @binding(4) var<storage, read_write> vplsOut : array<VPL>;   // 8*8*8 coarse cells

var<workgroup> accP : array<vec4f, 64>;
var<workgroup> accX : array<vec4f, 64>;

// Pass 1: one 64-thread workgroup per coarse cell; dispatch (8,8,8). Radiant intensity of a cell = sum(j * dV)  [W/sr-like]
@compute @workgroup_size(64)
fn reduceEmission(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let simDims = textureDimensions(fields);
  let cellVox = max(simDims / vec3u(8u), vec3u(1u));
  let base = wg * cellVox;
  let nVox = cellVox.x * cellVox.y * cellVox.z;
  var p = vec3f(0.0);
  var lx = vec3f(0.0);
  var lum = 0.0;
  for (var n = li; n < nVox; n += 64u) {
    let v = base + vec3u(n % cellVox.x, (n / cellVox.x) % cellVox.y, n / (cellVox.x * cellVox.y));
    let f = textureLoad(fields, v, 0);
    let sigA = RP.smoke.x * f.r + RP.fire.w * f.b;
    let onset = smoothstep(RP.fire.z, RP.fire.z + 200.0, f.g);
    let j = RP.fire.x * blackbody(f.g, RP.fire.y) * (onset * sigA);
    let y = dot(j, vec3f(0.2126, 0.7152, 0.0722));
    p += j;
    lum += y;
    lx += y * (RP.boxMin.xyz + (vec3f(v) + 0.5) * RP.voxelSize.xyz);
  }
  accP[li] = vec4f(p, lum);
  accX[li] = vec4f(lx, 0.0);
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) {
      accP[li] += accP[li + s];
      accX[li] += accX[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let idx = wg.x + 8u * (wg.y + 8u * wg.z);
    let dV = RP.voxelSize.x * RP.voxelSize.y * RP.voxelSize.z;
    let tot = accP[0];
    let cellCentre = RP.boxMin.xyz + (vec3f(base) + 0.5 * vec3f(cellVox)) * RP.voxelSize.xyz;
    let c = select(cellCentre, accX[0].xyz / max(tot.w, 1e-20), tot.w > 0.0);
    vplsOut[idx].posRadius = vec4f(c, 0.5 * RP.voxelSize.x * f32(cellVox.x));
    vplsOut[idx].intensity = vec4f(tot.rgb * dV, tot.w * dV);
  }
}
