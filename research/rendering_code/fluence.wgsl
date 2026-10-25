// ---------------- fluence.wgsl : fire-light volume for smoke self-illumination ----------------
struct VPL { posRadius: vec4f, intensity: vec4f };
struct LParams {
  boxMin : vec4f,
  boxMax : vec4f,
  shadow : vec4f,    // x = sigma_t per unit low-res density (for 4-tap shadow), y = shadow taps (unused: fixed 4)
  counts : vec4u,    // x = number of VPLs (e.g. 512 from 8^3, or 64 from a 4^3 merge)
};
@group(0) @binding(0) var<uniform> LP : LParams;
@group(0) @binding(1) var<storage, read> vpls : array<VPL>;
@group(0) @binding(2) var densLow  : texture_3d<f32>;    // r = smoke density at 1/4 res (mip or separate texture)
@group(0) @binding(3) var linClamp : sampler;
@group(0) @binding(4) var fluenceOut : texture_storage_3d<rgba16float, write>;   // e.g. 32^3 .. 48^3

@compute @workgroup_size(4, 4, 4)
fn buildFireLight(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(fluenceOut);
  if (any(id >= dims)) { return; }
  let ext = LP.boxMax.xyz - LP.boxMin.xyz;
  let uvw = (vec3f(id) + 0.5) / vec3f(dims);
  let x = LP.boxMin.xyz + uvw * ext;
  var F = vec3f(0.0);
  for (var k = 0u; k < LP.counts.x; k++) {
    let v = vpls[k];
    if (v.intensity.a <= 0.0) { continue; }
    let d = v.posRadius.xyz - x;
    let dist = length(d);
    // 4-tap transmittance towards the light through low-res smoke (soft, cheap self-shadowing)
    var dens = 0.0;
    for (var s = 1u; s <= 4u; s++) {
      dens += textureSampleLevel(densLow, linClamp, uvw + (d / ext) * (f32(s) / 5.0), 0.0).r;
    }
    let trans = exp(-LP.shadow.x * dens * dist * 0.2);
    F += v.intensity.rgb * trans / (dist * dist + v.posRadius.w * v.posRadius.w);
  }
  textureStore(fluenceOut, id, vec4f(F, 0.0));
}
