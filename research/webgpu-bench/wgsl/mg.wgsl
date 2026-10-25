// Geometric multigrid building blocks for the pressure Poisson equation  sum_nb(p) - 6 p = h2 * b
// (cell-centred, 7-point stencil). Level 0: b = divergence (cell units), h2 = 1. Level l: h2 = 4^l.
// BCs: solid floor (y < 0) -> Neumann (ghost = p_c); open sides/top -> Dirichlet p = 0 on the boundary FACE
// (ghost = -p_c). The face form keeps the boundary at the same physical place on every level, which is what
// makes the V-cycle converge (ghost = 0 stalls/diverges; see poisson.mjs results).
// Each level is one mip of an r32float 3D texture (p, b, r); bind single-mip views.

struct Level { h2: f32, parity: u32, omega: f32, pad: f32 };
@group(0) @binding(0) var<uniform> L: Level;

// ---- red-black Gauss-Seidel / SOR smoother (in place, read_write storage) ----
@group(0) @binding(1) var pRW: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var bT: texture_3d<f32>;

fn pNb(c: vec3<i32>, d: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }                                   // solid floor: dp/dn = 0
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return -pc; }     // open: p = 0 on the face
  return textureLoad(pRW, c).x;
}

// dispatch: ceil(dims.x/2 / 8), ceil(dims.y / 8), ceil(dims.z / 4); once with parity 0, once with parity 1
@compute @workgroup_size(8, 8, 4)
fn smoothRB(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = vec3<i32>(textureDimensions(pRW));
  let c = vec3<i32>(i32(g.x) * 2 + ((i32(g.y) + i32(g.z) + i32(L.parity)) & 1), i32(g.y), i32(g.z));
  if (any(c >= d)) { return; }
  let pc = textureLoad(pRW, c).x;
  let s = pNb(c - vec3<i32>(1, 0, 0), d, pc) + pNb(c + vec3<i32>(1, 0, 0), d, pc)
        + pNb(c - vec3<i32>(0, 1, 0), d, pc) + pNb(c + vec3<i32>(0, 1, 0), d, pc)
        + pNb(c - vec3<i32>(0, 0, 1), d, pc) + pNb(c + vec3<i32>(0, 0, 1), d, pc);
  let pg = (s - L.h2 * textureLoad(bT, c, 0).x) / 6.0;
  textureStore(pRW, c, vec4<f32>(mix(pc, pg, L.omega), 0.0, 0.0, 0.0));
}

// ---- residual r = b - A p ----
@group(0) @binding(1) var pT: texture_3d<f32>;
@group(0) @binding(3) var rOut: texture_storage_3d<r32float, write>;

fn pNbT(c: vec3<i32>, d: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return -pc; }
  return textureLoad(pT, c, 0).x;
}

@compute @workgroup_size(8, 8, 4)
fn residual(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = vec3<i32>(textureDimensions(pT));
  let c = vec3<i32>(g);
  if (any(c >= d)) { return; }
  let pc = textureLoad(pT, c, 0).x;
  let s = pNbT(c - vec3<i32>(1, 0, 0), d, pc) + pNbT(c + vec3<i32>(1, 0, 0), d, pc)
        + pNbT(c - vec3<i32>(0, 1, 0), d, pc) + pNbT(c + vec3<i32>(0, 1, 0), d, pc)
        + pNbT(c - vec3<i32>(0, 0, 1), d, pc) + pNbT(c + vec3<i32>(0, 0, 1), d, pc);
  textureStore(rOut, c, vec4<f32>(textureLoad(bT, c, 0).x - (s - 6.0 * pc) / L.h2, 0.0, 0.0, 0.0));
}

// ---- damped-Jacobi smoother (ping-pong pT -> pOut), omega = 6/7 is the optimal 3D smoothing weight ----
@group(0) @binding(9) var pOut: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, 4)
fn jacobiSmooth(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = vec3<i32>(textureDimensions(pT));
  let c = vec3<i32>(g);
  if (any(c >= d)) { return; }
  let pc = textureLoad(pT, c, 0).x;
  let s = pNbT(c - vec3<i32>(1, 0, 0), d, pc) + pNbT(c + vec3<i32>(1, 0, 0), d, pc)
        + pNbT(c - vec3<i32>(0, 1, 0), d, pc) + pNbT(c + vec3<i32>(0, 1, 0), d, pc)
        + pNbT(c - vec3<i32>(0, 0, 1), d, pc) + pNbT(c + vec3<i32>(0, 0, 1), d, pc);
  let pj = (s - L.h2 * textureLoad(bT, c, 0).x) / 6.0;
  textureStore(pOut, c, vec4<f32>(mix(pc, pj, L.omega), 0.0, 0.0, 0.0));
}

// ---- restriction: coarse b = mean of the 8 fine residuals; also zero the coarse initial guess ----
@group(0) @binding(4) var rFine: texture_3d<f32>;
@group(0) @binding(5) var bCoarse: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var pCoarse: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn restrictLevel(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = vec3<i32>(textureDimensions(bCoarse));
  let c = vec3<i32>(g);
  if (any(c >= d)) { return; }
  var s = 0.0;
  for (var k = 0; k < 8; k++) {
    s += textureLoad(rFine, 2 * c + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1), 0).x;
  }
  textureStore(bCoarse, c, vec4<f32>(s * 0.125, 0.0, 0.0, 0.0));
  textureStore(pCoarse, c, vec4<f32>(0.0));
}

// ---- prolongation: p_fine += trilinear(p_coarse). For cell-centred grids the (3/4, 1/4) weights are exactly
// hardware trilinear filtering at the fine cell centre, so one textureSampleLevel does it (r32float needs the
// 'float32-filterable' feature; weights 0.25/0.75 are exact in the 8-bit filter fraction).
@group(0) @binding(7) var pCoarseT: texture_3d<f32>;
@group(0) @binding(8) var lin: sampler;   // linear, clamp-to-edge

@compute @workgroup_size(8, 8, 4)
fn prolongAdd(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = vec3<i32>(textureDimensions(pRW));
  let c = vec3<i32>(g);
  if (any(c >= d)) { return; }
  let uvw = (vec3<f32>(c) + 0.5) / vec3<f32>(d);
  let e = textureSampleLevel(pCoarseT, lin, uvw, 0.0).x;
  textureStore(pRW, c, vec4<f32>(textureLoad(pRW, c).x + e, 0.0, 0.0, 0.0));
}
