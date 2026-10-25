// Fluid kernels exactly as benchmarked in bench.js (workgroup 8x8x4, rgba16float fields).
// Binding numbers overlap between kernels: create one shader module per section.

// ======================= advect =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var srcTex: texture_3d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let v = textureLoad(velTex, c, 0).xyz;              // cells / second
  let p = vec3<f32>(id) + 0.5 - P.dt * v;             // back-trace (cell units)
  let s = textureSampleLevel(srcTex, samp, p * P.invDims, 0.0);
  textureStore(dst, c, s * P.decay);
}

// ======================= mccorrect =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var srcTex: texture_3d<f32>;
@group(0) @binding(3) var hatTex: texture_3d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let v = textureLoad(velTex, c, 0).xyz;
  let x = vec3<f32>(id) + 0.5;
  let hat  = textureLoad(hatTex, c, 0);
  let back = textureSampleLevel(hatTex, samp, (x + P.dt * v) * P.invDims, 0.0);
  let orig = textureLoad(srcTex, c, 0);
  var r = hat + 0.5 * (orig - back);
  let i0 = vec3<i32>(floor(x - P.dt * v - 0.5));
  var mn = vec4<f32>(3.0e38); var mx = vec4<f32>(-3.0e38);
  for (var k = 0; k < 8; k++) {
    let s = textureLoad(srcTex, cl(i0 + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1)), 0);
    mn = min(mn, s); mx = max(mx, s);
  }
  r = clamp(r, mn, mx);
  textureStore(dst, c, r * P.decay);
}

// ======================= div =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<r32float, write>;
fn V(c: vec3<i32>) -> vec3<f32> { return textureLoad(velTex, cl(c), 0).xyz; }
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let vC = V(c);
  let vL = V(c - vec3<i32>(1,0,0)).x; let vR = V(c + vec3<i32>(1,0,0)).x;
  var vD = V(c - vec3<i32>(0,1,0)).y; let vU = V(c + vec3<i32>(0,1,0)).y;
  let vB = V(c - vec3<i32>(0,0,1)).z; let vF = V(c + vec3<i32>(0,0,1)).z;
  if (c.y == 0) { vD = -vC.y; }                       // solid floor (free-slip mirror)
  textureStore(dst, c, vec4<f32>(0.5 * ((vR - vL) + (vU - vD) + (vF - vB)), 0.0, 0.0, 0.0));
}

// ======================= jacobi =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var pIn: texture_3d<f32>;
@group(0) @binding(2) var divT: texture_3d<f32>;
@group(0) @binding(3) var pOut: texture_storage_3d<r32float, write>;
fn Pn(c: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }                                            // floor: Neumann
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.dims))) { return 0.0; } // open: Dirichlet p=0
  return textureLoad(pIn, c, 0).x;
}
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let pc = textureLoad(pIn, c, 0).x;
  let s = Pn(c - vec3<i32>(1,0,0), pc) + Pn(c + vec3<i32>(1,0,0), pc)
        + Pn(c - vec3<i32>(0,1,0), pc) + Pn(c + vec3<i32>(0,1,0), pc)
        + Pn(c - vec3<i32>(0,0,1), pc) + Pn(c + vec3<i32>(0,0,1), pc);
  let pj = (s - textureLoad(divT, c, 0).x) / 6.0;
  textureStore(pOut, c, vec4<f32>(mix(pc, pj, P.omega), 0.0, 0.0, 0.0));
}

// ======================= jacobi2 =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var pIn: texture_3d<f32>;
@group(0) @binding(2) var divT: texture_3d<f32>;
@group(0) @binding(3) var pOut: texture_storage_3d<r32float, write>;
const SX = 12u; const SY = 12u; const SZ = 8u; const NS = 1152u;
var<workgroup> A: array<f32, NS>;
var<workgroup> B: array<f32, NS>;
var<workgroup> D: array<f32, NS>;
fn sidx(l: vec3<u32>) -> u32 { return l.x + SX * (l.y + SY * l.z); }
fn inDom(g: vec3<i32>) -> bool { return all(g >= vec3<i32>(0)) && all(g < vec3<i32>(P.dims)); }
// neighbour value with BCs evaluated in global coords; T selects source array
fn nb(T: u32, l: vec3<u32>, g: vec3<i32>, pc: f32) -> f32 {
  if (g.y < 0) { return pc; }
  if (!inDom(g)) { return -pc; }                 // face-centred Dirichlet (MG-consistent)
  if (T == 0u) { return A[sidx(l)]; } return B[sidx(l)];
}
fn relax(T: u32, l: vec3<u32>, g: vec3<i32>, pc: f32) -> f32 {
  let s = nb(T, l - vec3<u32>(1u,0u,0u), g - vec3<i32>(1,0,0), pc) + nb(T, l + vec3<u32>(1u,0u,0u), g + vec3<i32>(1,0,0), pc)
        + nb(T, l - vec3<u32>(0u,1u,0u), g - vec3<i32>(0,1,0), pc) + nb(T, l + vec3<u32>(0u,1u,0u), g + vec3<i32>(0,1,0), pc)
        + nb(T, l - vec3<u32>(0u,0u,1u), g - vec3<i32>(0,0,1), pc) + nb(T, l + vec3<u32>(0u,0u,1u), g + vec3<i32>(0,0,1), pc);
  return mix(pc, (s - D[sidx(l)]) / 6.0, P.omega);
}
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(local_invocation_index) li: u32, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let origin = vec3<i32>(wg * vec3<u32>(8u, 8u, 4u)) - vec3<i32>(2);
  for (var i = li; i < NS; i += 256u) {
    let l = vec3<u32>(i % SX, (i / SX) % SY, i / (SX * SY));
    let g = origin + vec3<i32>(l);
    var p = 0.0; var d = 0.0;
    if (inDom(g)) { p = textureLoad(pIn, g, 0).x; d = textureLoad(divT, g, 0).x; }
    A[i] = p; D[i] = d;
  }
  workgroupBarrier();
  // iteration 1 on the 10x10x6 inner region (600 cells, ~2.3 per thread)
  for (var i = li; i < 600u; i += 256u) {
    let l = vec3<u32>(1u + i % 10u, 1u + (i / 10u) % 10u, 1u + i / 100u);
    let g = origin + vec3<i32>(l);
    B[sidx(l)] = select(0.0, relax(0u, l, g, A[sidx(l)]), inDom(g));
  }
  workgroupBarrier();
  // iteration 2 on the 8x8x4 core
  let l = lid + vec3<u32>(2u);
  let g = origin + vec3<i32>(l);
  if (inDom(g)) { textureStore(pOut, g, vec4<f32>(relax(1u, l, g, B[sidx(l)]), 0.0, 0.0, 0.0)); }
}

// ======================= rbgs =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var p: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var divT: texture_3d<f32>;
fn Pn(c: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.dims))) { return 0.0; }
  return textureLoad(p, c).x;
}
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x * 2u + ((gid.y + gid.z + P.parity) & 1u);
  let id = vec3<u32>(x, gid.y, gid.z);
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let pc = textureLoad(p, c).x;
  let s = Pn(c - vec3<i32>(1,0,0), pc) + Pn(c + vec3<i32>(1,0,0), pc)
        + Pn(c - vec3<i32>(0,1,0), pc) + Pn(c + vec3<i32>(0,1,0), pc)
        + Pn(c - vec3<i32>(0,0,1), pc) + Pn(c + vec3<i32>(0,0,1), pc);
  let pg = (s - textureLoad(divT, c, 0).x) / 6.0;
  textureStore(p, c, vec4<f32>(mix(pc, pg, P.omega), 0.0, 0.0, 0.0));
}

// ======================= project =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var pT: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
fn Pn(c: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.dims))) { return 0.0; }
  return textureLoad(pT, c, 0).x;
}
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let pc = textureLoad(pT, c, 0).x;
  let g = 0.5 * vec3<f32>(Pn(c + vec3<i32>(1,0,0), pc) - Pn(c - vec3<i32>(1,0,0), pc),
                          Pn(c + vec3<i32>(0,1,0), pc) - Pn(c - vec3<i32>(0,1,0), pc),
                          Pn(c + vec3<i32>(0,0,1), pc) - Pn(c - vec3<i32>(0,0,1), pc));
  var v = textureLoad(velTex, c, 0).xyz - g;
  if (c.y == 0) { v.y = max(v.y, 0.0); }
  textureStore(dst, c, vec4<f32>(v, 0.0));
}

// ======================= curl =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
fn V(c: vec3<i32>) -> vec3<f32> { return textureLoad(velTex, cl(c), 0).xyz; }
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let L = V(c - vec3<i32>(1,0,0)); let R = V(c + vec3<i32>(1,0,0));
  let D = V(c - vec3<i32>(0,1,0)); let U = V(c + vec3<i32>(0,1,0));
  let B = V(c - vec3<i32>(0,0,1)); let F = V(c + vec3<i32>(0,0,1));
  let w = 0.5 * vec3<f32>((U.z - D.z) - (F.y - B.y), (F.x - B.x) - (R.z - L.z), (R.y - L.y) - (U.x - D.x));
  textureStore(dst, c, vec4<f32>(w, length(w)));
}

// ======================= confine =======================

struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }

@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var curlTex: texture_3d<f32>;
@group(0) @binding(3) var scalTex: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
fn W(c: vec3<i32>) -> f32 { return textureLoad(curlTex, cl(c), 0).w; }
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let eta = 0.5 * vec3<f32>(W(c + vec3<i32>(1,0,0)) - W(c - vec3<i32>(1,0,0)),
                            W(c + vec3<i32>(0,1,0)) - W(c - vec3<i32>(0,1,0)),
                            W(c + vec3<i32>(0,0,1)) - W(c - vec3<i32>(0,0,1)));
  let N = eta / (length(eta) + 1e-5);
  let w = textureLoad(curlTex, c, 0).xyz;
  let s = textureLoad(scalTex, c, 0);           // x = temperature, y = smoke
  var v = textureLoad(velTex, c, 0).xyz;
  v += P.dt * (P.eps * cross(N, w) + vec3<f32>(0.0, P.buoy * s.x - 0.1 * s.y, 0.0));
  textureStore(dst, c, vec4<f32>(v, 0.0));
}
