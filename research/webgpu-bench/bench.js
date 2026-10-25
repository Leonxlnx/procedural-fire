// WebGPU fluid-kernel micro benchmarks (advection, MacCormack, Jacobi, RBGS, project, vorticity, ray march, readback)
const logEl = document.getElementById('log');
const log = (...a) => { logEl.textContent += a.join(' ') + '\n'; console.log(...a); };
const qs = new URLSearchParams(location.search);
const MODE = qs.get('mode') || 'all';
const out = { ua: navigator.userAgent, errors: [], results: {} };
window.__out = out;

const HEADER = /* wgsl */`
struct Params {
  dims: vec3<u32>, dt: f32,
  invDims: vec3<f32>, decay: f32,
  parity: u32, omega: f32, eps: f32, buoy: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }
`;

const K = {
init: `
@group(0) @binding(1) var vOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var sOut: texture_storage_3d<rgba16float, write>;
fn hash(p: vec3<u32>) -> f32 {
  var h = (p.x * 73856093u) ^ (p.y * 19349663u) ^ (p.z * 83492791u);
  h = (h ^ (h >> 13u)) * 1274126177u; h = h ^ (h >> 16u);
  return f32(h & 0xffffu) / 65535.0;
}
@compute @workgroup_size(WX, WY, WZ)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let x = (vec3<f32>(id) + 0.5) * P.invDims;
  let s = f32(P.dims.x) / 64.0;
  let v = vec3<f32>(sin(x.y * 12.0) * 3.0, 4.0 + cos(x.x * 9.0) * 2.0, sin(x.z * 7.0 + x.x * 5.0) * 3.0) * s;
  let r = length((x - vec3<f32>(0.5, 0.35, 0.5)) * vec3<f32>(1.0, 0.6, 1.0));
  let blob = smoothstep(0.35, 0.0, r);
  let n = hash(id);
  textureStore(vOut, vec3<i32>(id), vec4<f32>(v, 0.0));
  textureStore(sOut, vec3<i32>(id), vec4<f32>(blob * (0.7 + 0.3 * n) * 2.0, blob * 0.5, n * blob, 0.0));
}`,

advect: `
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var srcTex: texture_3d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var dst: texture_storage_3d<FMT, write>;
@compute @workgroup_size(WX, WY, WZ)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let v = textureLoad(velTex, c, 0).xyz;              // cells / second
  let p = vec3<f32>(id) + 0.5 - P.dt * v;             // back-trace (cell units)
  let s = textureSampleLevel(srcTex, samp, p * P.invDims, 0.0);
  textureStore(dst, c, s * P.decay);
}`,

// MacCormack correction + Selle et al. limiter. hatTex = forward SL result.
mccorrect: `
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var srcTex: texture_3d<f32>;
@group(0) @binding(3) var hatTex: texture_3d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(WX, WY, WZ)
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
}`,

div: `
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<r32float, write>;
fn V(c: vec3<i32>) -> vec3<f32> { return textureLoad(velTex, cl(c), 0).xyz; }
@compute @workgroup_size(WX, WY, WZ)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let vC = V(c);
  let vL = V(c - vec3<i32>(1,0,0)).x; let vR = V(c + vec3<i32>(1,0,0)).x;
  var vD = V(c - vec3<i32>(0,1,0)).y; let vU = V(c + vec3<i32>(0,1,0)).y;
  let vB = V(c - vec3<i32>(0,0,1)).z; let vF = V(c + vec3<i32>(0,0,1)).z;
  if (c.y == 0) { vD = -vC.y; }                       // solid floor (free-slip mirror)
  textureStore(dst, c, vec4<f32>(0.5 * ((vR - vL) + (vU - vD) + (vF - vB)), 0.0, 0.0, 0.0));
}`,

jacobi: `
@group(0) @binding(1) var pIn: texture_3d<f32>;
@group(0) @binding(2) var divT: texture_3d<f32>;
@group(0) @binding(3) var pOut: texture_storage_3d<r32float, write>;
fn Pn(c: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }                                            // floor: Neumann
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.dims))) { return 0.0; } // open: Dirichlet p=0
  return textureLoad(pIn, c, 0).x;
}
@compute @workgroup_size(WX, WY, WZ)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let pc = textureLoad(pIn, c, 0).x;
  let s = Pn(c - vec3<i32>(1,0,0), pc) + Pn(c + vec3<i32>(1,0,0), pc)
        + Pn(c - vec3<i32>(0,1,0), pc) + Pn(c + vec3<i32>(0,1,0), pc)
        + Pn(c - vec3<i32>(0,0,1), pc) + Pn(c + vec3<i32>(0,0,1), pc);
  let pj = (s - textureLoad(divT, c, 0).x) / 6.0;
  textureStore(pOut, c, vec4<f32>(mix(pc, pj, P.omega), 0.0, 0.0, 0.0));
}`,

jacobibuf: `
@group(0) @binding(1) var<storage, read> pIn: array<f32>;
@group(0) @binding(2) var<storage, read> divB: array<f32>;
@group(0) @binding(3) var<storage, read_write> pOut: array<f32>;
fn idx(c: vec3<i32>) -> i32 { return c.x + i32(P.dims.x) * (c.y + i32(P.dims.y) * c.z); }
fn Pn(c: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.dims))) { return 0.0; }
  return pIn[idx(c)];
}
@compute @workgroup_size(WX, WY, WZ)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let i = idx(c);
  let pc = pIn[i];
  let s = Pn(c - vec3<i32>(1,0,0), pc) + Pn(c + vec3<i32>(1,0,0), pc)
        + Pn(c - vec3<i32>(0,1,0), pc) + Pn(c + vec3<i32>(0,1,0), pc)
        + Pn(c - vec3<i32>(0,0,1), pc) + Pn(c + vec3<i32>(0,0,1), pc);
  pOut[i] = mix(pc, (s - divB[i]) / 6.0, P.omega);
}`,

// Temporal blocking: 2 Jacobi iterations per dispatch through workgroup memory (8x8x4 tile + 2-cell halo).
jacobi2: `
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
}`,

// red-black Gauss-Seidel / SOR, in place on an r32float read_write storage texture
rbgs: `
@group(0) @binding(1) var p: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var divT: texture_3d<f32>;
fn Pn(c: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.dims))) { return 0.0; }
  return textureLoad(p, c).x;
}
@compute @workgroup_size(WX, WY, WZ)
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
}`,

project: `
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var pT: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
fn Pn(c: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.dims))) { return 0.0; }
  return textureLoad(pT, c, 0).x;
}
@compute @workgroup_size(WX, WY, WZ)
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
}`,

curl: `
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
fn V(c: vec3<i32>) -> vec3<f32> { return textureLoad(velTex, cl(c), 0).xyz; }
@compute @workgroup_size(WX, WY, WZ)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let L = V(c - vec3<i32>(1,0,0)); let R = V(c + vec3<i32>(1,0,0));
  let D = V(c - vec3<i32>(0,1,0)); let U = V(c + vec3<i32>(0,1,0));
  let B = V(c - vec3<i32>(0,0,1)); let F = V(c + vec3<i32>(0,0,1));
  let w = 0.5 * vec3<f32>((U.z - D.z) - (F.y - B.y), (F.x - B.x) - (R.z - L.z), (R.y - L.y) - (U.x - D.x));
  textureStore(dst, c, vec4<f32>(w, length(w)));
}`,

confine: `
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var curlTex: texture_3d<f32>;
@group(0) @binding(3) var scalTex: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
fn W(c: vec3<i32>) -> f32 { return textureLoad(curlTex, cl(c), 0).w; }
@compute @workgroup_size(WX, WY, WZ)
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
}`,
};

const RAYMARCH = /* wgsl */`
struct RP { res: vec2<f32>, nSteps: f32, shadowSteps: f32, camPos: vec4<f32>, };
@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;
fn ign(p: vec2<f32>) -> f32 { return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715)))); }
fn boxHit(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  let inv = 1.0 / rd;
  let t0 = (vec3<f32>(0.0) - ro) * inv; let t1 = (vec3<f32>(1.0) - ro) * inv;
  let tmin = min(t0, t1); let tmax = max(t0, t1);
  return vec2<f32>(max(max(tmin.x, tmin.y), max(tmin.z, 0.0)), min(min(tmax.x, tmax.y), tmax.z));
}
fn blackbody(t: f32) -> vec3<f32> { // cheap ramp stand-in
  return vec3<f32>(1.0, 0.35, 0.08) * t * t * 4.0 + vec3<f32>(0.6, 0.5, 0.3) * max(t - 0.8, 0.0) * 8.0;
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(vec2<f32>(id.xy) >= R.res)) { return; }
  let uv = (vec2<f32>(id.xy) + 0.5) / R.res * 2.0 - 1.0;
  let ro = R.camPos.xyz;
  let fw = normalize(vec3<f32>(0.5, 0.45, 0.5) - ro);
  let rt = normalize(cross(fw, vec3<f32>(0.0, 1.0, 0.0)));
  let up = cross(rt, fw);
  let rd = normalize(fw + uv.x * rt * (R.res.x / R.res.y) * 0.6 - uv.y * up * 0.6);
  let hit = boxHit(ro, rd);
  var L = vec3<f32>(0.0); var T = 1.0;
  if (hit.x < hit.y) {
    let dt = 1.0 / R.nSteps;                      // nSteps = samples per unit length
    var t = hit.x + dt * ign(vec2<f32>(id.xy));
    let lightPos = vec3<f32>(0.5, 0.3, 0.5);
    loop {
      if (t >= hit.y) { break; }
      let p = ro + rd * t;
      let s = textureSampleLevel(scal, samp, p, 0.0);
      let sigma = (s.y * 6.0 + s.x * 0.5) * 8.0;
      var Le = blackbody(s.x) * 30.0;
      if (R.shadowSteps > 0.0 && s.y > 0.02) {   // secondary march toward a light
        var od = 0.0; let ld = normalize(lightPos - p); let sd = 0.6 / R.shadowSteps;
        for (var k = 1.0; k <= R.shadowSteps; k += 1.0) { od += textureSampleLevel(scal, samp, p + ld * sd * k, 0.0).y * sd * 48.0; }
        Le += vec3<f32>(1.0, 0.5, 0.2) * exp(-od) * s.y * 3.0;
      }
      let a = exp(-sigma * dt);
      L += T * Le * (1.0 - a) / max(sigma, 1e-4);  // analytic emission over the step (Hillaire 2015)
      T *= a;
      t += dt;
    }
  }
  textureStore(outTex, vec2<i32>(id.xy), vec4<f32>(L, 1.0 - T));
}`;

K.jacobi16 = K.jacobi.replace('texture_storage_3d<r32float, write>', 'texture_storage_3d<r16float, write>');

async function benchTB(dims) {
  const R = {};
  const v = t => t.createView();
  const wg = [8, 8, 4], wc = wgCount(dims, wg);
  const U = ubuf(dims, { omega: 1.0 });
  const pA = tex3(dims, 'r32float'), pB = tex3(dims, 'r32float'), divT = tex3(dims, 'r32float');
  const hA = tex3(dims, 'r16float'), hB = tex3(dims, 'r16float');
  { const pp = pipe('jacobi', wg); const gAB = bg(pp, [U, v(pA), v(divT), v(pB)]), gBA = bg(pp, [U, v(pB), v(divT), v(pA)]);
    R.jacobi_r32f_perIter = await time((p, r) => { p.setPipeline(pp.p); for (let i = 0; i < r; i++) { p.setBindGroup(0, i & 1 ? gBA : gAB); p.dispatchWorkgroups(...wc); } }, 40); }
  { const pp = pipe('jacobi2', wg); const gAB = bg(pp, [U, v(pA), v(divT), v(pB)]), gBA = bg(pp, [U, v(pB), v(divT), v(pA)]);
    const r = await time((p, r) => { p.setPipeline(pp.p); for (let i = 0; i < r; i++) { p.setBindGroup(0, i & 1 ? gBA : gAB); p.dispatchWorkgroups(...wc); } }, 20);
    R.jacobi2_tempBlocked_perIter = { gpu: +(r.gpu / 2).toFixed(4), wall: +(r.wall / 2).toFixed(4) }; }
  { const pp = pipe('jacobi16', wg); const gAB = bg(pp, [U, v(hA), v(divT), v(hB)]), gBA = bg(pp, [U, v(hB), v(divT), v(hA)]);
    R.jacobi_r16f_perIter = await time((p, r) => { p.setPipeline(pp.p); for (let i = 0; i < r; i++) { p.setBindGroup(0, i & 1 ? gBA : gAB); p.dispatchWorkgroups(...wc); } }, 40); }
  // empty-ish dispatch overhead: 1 workgroup dispatches
  { const pp = pipe('jacobi', wg); const g = bg(pp, [U, v(pA), v(divT), v(pB)]);
    R.dispatchOverhead_1wg = await time((p, r) => { p.setPipeline(pp.p); p.setBindGroup(0, g); for (let i = 0; i < r; i++) p.dispatchWorkgroups(1, 1, 1); }, 200); }
  for (const t of [pA, pB, divT, hA, hB]) t.destroy();
  return R;
}

// Determinism check: run the full step K times from the same init, hash the velocity + scalar fields.
async function determinism(N, K) {
  const dims = [N, N, N], wg = [8, 8, 4], wc = wgCount(dims, wg), v = t => t.createView();
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const T = () => tex3(dims, 'rgba16float');
  const velA = T(), velB = T(), velH = T(), scA = T(), scB = T(), scH = T(), curlT = T();
  const pA = tex3(dims, 'r32float'), pB = tex3(dims, 'r32float'), divT = tex3(dims, 'r32float');
  const U = ubuf(dims, { dt: 0.5 }), Uw = ubuf(dims, { omega: 6 / 7 });
  const pa = pipe('advect', wg), pc = pipe('mccorrect', wg), p1 = pipe('curl', wg), p2 = pipe('confine', wg), pd = pipe('div', wg), pj = pipe('jacobi', wg), pr = pipe('project', wg), pi = pipe('init', wg);
  const g = {
    init: bg(pi, [U, v(velA), v(scA)]),
    v1: bg(pa, [U, v(velA), v(velA), linear, v(velH)]), v2: bg(pc, [U, v(velA), v(velA), v(velH), linear, v(velB)]),
    s1: bg(pa, [U, v(velA), v(scA), linear, v(scH)]), s2: bg(pc, [U, v(velA), v(scA), v(scH), linear, v(scB)]),
    c1: bg(p1, [U, v(velB), v(curlT)]), c2: bg(p2, [U, v(velB), v(curlT), v(scB), v(velH)]),
    d: bg(pd, [U, v(velH), v(divT)]),
    jAB: bg(pj, [Uw, v(pA), v(divT), v(pB)]), jBA: bg(pj, [Uw, v(pB), v(divT), v(pA)]),
    pr: bg(pr, [U, v(velH), v(pA), v(velA)]),
  };
  // scalars: copy scB -> scA each step via a second MC pair writing back (ping-pong by copy for simplicity)
  const bpr = Math.ceil(N * 8 / 256) * 256;
  const rbuf = device.createBuffer({ size: bpr * N * N, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const hashes = [];
  for (let run = 0; run < 2; run++) {
    { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pi.p); p.setBindGroup(0, g.init); p.dispatchWorkgroups(...wc); p.end();
      e.clearBuffer; device.queue.submit([e.finish()]); }
    device.queue.writeTexture({ texture: pA }, new Float32Array(N * N * N), { bytesPerRow: N * 4, rowsPerImage: N }, [N, N, N]);
    for (let k = 0; k < K; k++) {
      const e = device.createCommandEncoder(); const p = e.beginComputePass();
      const D = (pp, gg) => { p.setPipeline(pp.p); p.setBindGroup(0, gg); p.dispatchWorkgroups(...wc); };
      D(pa, g.v1); D(pc, g.v2); D(pa, g.s1); D(pc, g.s2); D(p1, g.c1); D(p2, g.c2); D(pd, g.d);
      for (let i = 0; i < 40; i++) D(pj, i & 1 ? g.jBA : g.jAB);
      D(pr, g.pr);
      p.end();
      e.copyTextureToTexture({ texture: scB }, { texture: scA }, dims);
      device.queue.submit([e.finish()]);
    }
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({ texture: velA }, { buffer: rbuf, bytesPerRow: bpr, rowsPerImage: N }, dims);
    device.queue.submit([e.finish()]);
    await rbuf.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(rbuf.getMappedRange()); let h = 2166136261 >>> 0;
    for (let i = 0; i < u.length; i++) { h ^= u[i]; h = Math.imul(h, 16777619) >>> 0; }
    rbuf.unmap(); hashes.push(h.toString(16));
  }
  for (const t of [velA, velB, velH, scA, scB, scH, curlT, pA, pB, divT]) t.destroy();
  rbuf.destroy();
  return hashes;
}

// GPU multigrid V-cycle (wgsl/mg.wgsl) on the same RHS as poisson.mjs; returns residual history + timings.
async function benchMG(N) {
  const code = await (await fetch('wgsl/mg.wgsl')).text();
  const m = device.createShaderModule({ code });
  const P = {};
  for (const ep of ['smoothRB', 'residual', 'restrictLevel', 'prolongAdd', 'jacobiSmooth']) P[ep] = device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: ep } });
  let levels = 1; for (let n = N; n % 2 === 0 && n / 2 >= 3; n /= 2) levels++;   // 128->4, 96->3, 160->5
  const mk = () => device.createTexture({ size: [N, N, N], dimension: '3d', format: 'r32float', mipLevelCount: levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
  const pT = mk(), bT = mk(), rT = mk(), pT2 = mk();
  const mv = (t, l) => t.createView({ baseMipLevel: l, mipLevelCount: 1 });
  const lin = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  // RHS: same analytic plume divergence as poisson.mjs
  const I = (x, y, z) => x + N * (y + N * z);
  const vel = (x, y, z) => { const u = (x + 0.5) / N, v = (y + 0.5) / N, w = (z + 0.5) / N; const r = Math.hypot(u - 0.5, w - 0.5);
    const plume = Math.exp(-(r * r) / 0.01) * Math.min(1, v * 4);
    return [Math.sin(v * 12) * 0.3 + Math.sin(w * 31 + u * 7) * 0.2, 1.5 * plume + Math.cos(u * 9) * 0.2, Math.sin(w * 7 + u * 5) * 0.3]; };
  const b = new Float32Array(N * N * N); const C = a => Math.max(0, Math.min(N - 1, a));
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const vD = y === 0 ? -vel(x, y, z)[1] : vel(x, C(y - 1), z)[1];
    b[I(x, y, z)] = 0.5 * ((vel(C(x + 1), y, z)[0] - vel(C(x - 1), y, z)[0]) + (vel(x, C(y + 1), z)[1] - vD) + (vel(x, y, C(z + 1))[2] - vel(x, y, C(z - 1))[2]));
  }
  device.queue.writeTexture({ texture: bT }, b, { bytesPerRow: N * 4, rowsPerImage: N }, [N, N, N]);
  const U = [];
  for (let l = 0; l < levels; l++) U.push([0, 1].map(par => { const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const ab = new ArrayBuffer(16); new Float32Array(ab)[0] = 4 ** l; new Uint32Array(ab)[1] = par; new Float32Array(ab)[2] = 1.0; device.queue.writeBuffer(ub, 0, ab); return ub; }));
  const BG = (pipe, entries) => device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: Object.entries(entries).map(([k, r]) => ({ binding: +k, resource: r instanceof GPUBuffer ? { buffer: r } : r })) });
  const G = [];
  for (let l = 0; l < levels; l++) {
    const n = N >> l;
    G.push({ n,
      sm: [0, 1].map(par => BG(P.smoothRB, { 0: U[l][par], 1: mv(pT, l), 2: mv(bT, l) })),
      res: BG(P.residual, { 0: U[l][0], 1: mv(pT, l), 2: mv(bT, l), 3: mv(rT, l) }),
      rst: l + 1 < levels ? BG(P.restrictLevel, { 4: mv(rT, l), 5: mv(bT, l + 1), 6: mv(pT, l + 1) }) : null,
      pro: l + 1 < levels ? BG(P.prolongAdd, { 1: mv(pT, l), 7: mv(pT, l + 1), 8: lin }) : null });
  }
  for (let l = 0; l < levels; l++) { const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); const ab = new ArrayBuffer(16); new Float32Array(ab)[0] = 4 ** l; new Float32Array(ab)[2] = 6 / 7; device.queue.writeBuffer(ub, 0, ab);
    G[l].jp = BG(P.jacobiSmooth, { 0: ub, 1: mv(pT, l), 2: mv(bT, l), 9: mv(pT2, l) }); G[l].jq = BG(P.jacobiSmooth, { 0: ub, 1: mv(pT2, l), 2: mv(bT, l), 9: mv(pT, l) }); }
  let SM = 'rb';
  const wc = (n, half) => [Math.ceil((half ? n / 2 : n) / 8), Math.ceil(n / 8), Math.ceil(n / 4)];
  const smooth = (pass, l, k) => { if (SM === 'jac' && l < levels - 1) { for (let i = 0; i < k; i++) { pass.setPipeline(P.jacobiSmooth); pass.setBindGroup(0, G[l].jp); pass.dispatchWorkgroups(...wc(G[l].n)); pass.setBindGroup(0, G[l].jq); pass.dispatchWorkgroups(...wc(G[l].n)); } return; } for (let i = 0; i < k; i++) for (const par of [0, 1]) { pass.setPipeline(P.smoothRB); pass.setBindGroup(0, G[l].sm[par]); pass.dispatchWorkgroups(...wc(G[l].n, true)); } };
  const vcycle = (pass, l, pre, post, coarseIters) => {
    if (l === levels - 1) { smooth(pass, l, coarseIters); return; }
    smooth(pass, l, pre);
    pass.setPipeline(P.residual); pass.setBindGroup(0, G[l].res); pass.dispatchWorkgroups(...wc(G[l].n));
    pass.setPipeline(P.restrictLevel); pass.setBindGroup(0, G[l].rst); const nc = G[l + 1].n; pass.dispatchWorkgroups(Math.ceil(nc / 4), Math.ceil(nc / 4), Math.ceil(nc / 4));
    vcycle(pass, l + 1, pre, post, coarseIters);
    pass.setPipeline(P.prolongAdd); pass.setBindGroup(0, G[l].pro); pass.dispatchWorkgroups(...wc(G[l].n));
    smooth(pass, l, post);
  };
  const bpr = Math.ceil(N * 4 / 256) * 256;     // copyTextureToBuffer: bytesPerRow must be a multiple of 256
  const rb = device.createBuffer({ size: bpr * N * N, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const rms = async () => {
    const e = device.createCommandEncoder(); const p = e.beginComputePass();
    p.setPipeline(P.residual); p.setBindGroup(0, G[0].res); p.dispatchWorkgroups(...wc(N)); p.end();
    e.copyTextureToBuffer({ texture: rT, mipLevel: 0 }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: N }, [N, N, N]);
    device.queue.submit([e.finish()]); await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange()); let s = 0; const rowF = bpr / 4;
    for (let r = 0; r < N * N; r++) for (let x = 0; x < N; x++) { const v = a[r * rowF + x]; s += v * v; }
    const n = N * N * N; rb.unmap();               // NOTE: unmap() detaches the mapped ArrayBuffer (a.length becomes 0)
    return Math.sqrt(s / n);
  };
  const clearP = () => { const e = device.createCommandEncoder(); e.clearBuffer; device.queue.writeTexture({ texture: pT }, new Float32Array(N * N * N), { bytesPerRow: N * 4, rowsPerImage: N }, [N, N, N]); };
  const R = {};
  for (const [pre, post, cIt, sm] of [[2, 2, 16, 'rb'], [1, 1, 16, 'rb'], [1, 1, 16, 'jac'], [2, 2, 16, 'jac']]) {
    SM = sm; clearP();
    const r0 = await rms(); const hist = [];
    for (let k = 0; k < 6; k++) { const e = device.createCommandEncoder(); const p = e.beginComputePass(); vcycle(p, 0, pre, post, cIt); p.end(); device.queue.submit([e.finish()]); hist.push(+((await rms()) / r0).toExponential(2)); }
    clearP();
    const t = await time((p, r) => { for (let i = 0; i < r; i++) vcycle(p, 0, pre, post, cIt); }, 4);
    R[`V(${pre},${pre})-${sm}`] = { residualRatioPerCycle: hist, msPerCycle: t.gpu, wallPerCycle: t.wall };
  }
  // plain reference: RB-GS sweeps on level 0 only
  SM = 'rb'; clearP();
  { const r0 = await rms(); const e = device.createCommandEncoder(); const p = e.beginComputePass(); smooth(p, 0, 40); p.end(); device.queue.submit([e.finish()]); R.rbgs40_residualRatio = +((await rms()) / r0).toExponential(2); }
  pT.destroy(); bT.destroy(); rT.destroy(); pT2.destroy(); rb.destroy();
  return R;
}

// ---------- helpers ----------
let device, ts, qset, resolveBuf, readBuf;
const pipeCache = new Map();
function E(binding, kind, opt = {}) {
  const e = { binding, visibility: GPUShaderStage.COMPUTE };
  if (kind === 'u') e.buffer = { type: 'uniform' };
  if (kind === 'tex') e.texture = { sampleType: opt.s || 'float', viewDimension: opt.dim || '3d' };
  if (kind === 'samp') e.sampler = { type: 'filtering' };
  if (kind === 'st') e.storageTexture = { access: opt.a || 'write-only', format: opt.f || 'rgba16float', viewDimension: opt.dim || '3d' };
  if (kind === 'ro') e.buffer = { type: 'read-only-storage' };
  if (kind === 'rw') e.buffer = { type: 'storage' };
  return e;
}
const LAYOUTS = {
  init: [E(0,'u'), E(1,'st'), E(2,'st')],
  advect: (f) => [E(0,'u'), E(1,'tex'), E(2,'tex'), E(3,'samp'), E(4,'st',{f})],
  mccorrect: [E(0,'u'), E(1,'tex'), E(2,'tex'), E(3,'tex'), E(4,'samp'), E(5,'st')],
  div: [E(0,'u'), E(1,'tex'), E(2,'st',{f:'r32float'})],
  jacobi: [E(0,'u'), E(1,'tex',{s:'unfilterable-float'}), E(2,'tex',{s:'unfilterable-float'}), E(3,'st',{f:'r32float'})],
  jacobibuf: [E(0,'u'), E(1,'ro'), E(2,'ro'), E(3,'rw')],
  jacobi2: [E(0,'u'), E(1,'tex',{s:'unfilterable-float'}), E(2,'tex',{s:'unfilterable-float'}), E(3,'st',{f:'r32float'})],
  jacobi16: [E(0,'u'), E(1,'tex',{s:'unfilterable-float'}), E(2,'tex',{s:'unfilterable-float'}), E(3,'st',{f:'r16float'})],
  rbgs: [E(0,'u'), E(1,'st',{f:'r32float',a:'read-write'}), E(2,'tex',{s:'unfilterable-float'})],
  project: [E(0,'u'), E(1,'tex'), E(2,'tex',{s:'unfilterable-float'}), E(3,'st')],
  curl: [E(0,'u'), E(1,'tex'), E(2,'st')],
  confine: [E(0,'u'), E(1,'tex'), E(2,'tex'), E(3,'tex'), E(4,'st')],
};
function pipe(name, wg, fmt = 'rgba16float') {
  const key = `${name}|${wg}|${fmt}`;
  if (pipeCache.has(key)) return pipeCache.get(key);
  const code = HEADER + K[name].replaceAll('WX', wg[0]).replaceAll('WY', wg[1]).replaceAll('WZ', wg[2]).replaceAll('FMT', fmt);
  const module = device.createShaderModule({ code, label: name });
  let entries = LAYOUTS[name]; if (typeof entries === 'function') entries = entries(fmt);
  if (fmt === 'rgba32float' && name === 'advect') entries = entries.map(e => e.texture ? { ...e, texture: { ...e.texture, sampleType: device.features.has('float32-filterable') ? 'float' : 'unfilterable-float' } } : e);
  const bgl = device.createBindGroupLayout({ entries });
  const p = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint: 'main' } });
  const r = { p, bgl };
  pipeCache.set(key, r);
  return r;
}
function bg(pp, resources) {
  return device.createBindGroup({ layout: pp.bgl, entries: resources.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })) });
}
function ubuf(dims, extra = {}) {
  const b = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const ab = new ArrayBuffer(48), u = new Uint32Array(ab), f = new Float32Array(ab);
  u[0] = dims[0]; u[1] = dims[1]; u[2] = dims[2]; f[3] = extra.dt ?? 0.5;
  f[4] = 1 / dims[0]; f[5] = 1 / dims[1]; f[6] = 1 / dims[2]; f[7] = extra.decay ?? 0.999;
  u[8] = extra.parity ?? 0; f[9] = extra.omega ?? 1.0; f[10] = extra.eps ?? 0.3; f[11] = extra.buoy ?? 1.0;
  device.queue.writeBuffer(b, 0, ab);
  return b;
}
const median = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
async function time(encodeFn, reps, trials = 3) {
  const g = [], w = [];
  { const e = device.createCommandEncoder(); const p = e.beginComputePass(); encodeFn(p, reps); p.end(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); }
  for (let t = 0; t < trials; t++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(ts ? { timestampWrites: { querySet: qset, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
    encodeFn(pass, reps);
    pass.end();
    if (ts) { enc.resolveQuerySet(qset, 0, 2, resolveBuf, 0); enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16); }
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    w.push((performance.now() - t0) / reps);
    if (ts) { await readBuf.mapAsync(GPUMapMode.READ); const a = new BigUint64Array(readBuf.getMappedRange()); g.push(Number(a[1] - a[0]) / 1e6 / reps); readBuf.unmap(); }
  }
  return { gpu: ts ? +median(g).toFixed(4) : null, wall: +median(w).toFixed(4) };
}
const wgCount = (dims, wg) => [Math.ceil(dims[0] / wg[0]), Math.ceil(dims[1] / wg[1]), Math.ceil(dims[2] / wg[2])];

function tex3(dims, format, extraUsage = 0) {
  return device.createTexture({ size: dims, dimension: '3d', format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | extraUsage });
}

// ---------- grid benchmarks ----------
async function benchGrid(dims, wg) {
  const R = {};
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
  const velA = tex3(dims, 'rgba16float'), velB = tex3(dims, 'rgba16float'), velH = tex3(dims, 'rgba16float');
  const scA = tex3(dims, 'rgba16float'), scB = tex3(dims, 'rgba16float'), scH = tex3(dims, 'rgba16float');
  const curlT = tex3(dims, 'rgba16float');
  const pA = tex3(dims, 'r32float'), pB = tex3(dims, 'r32float'), divT = tex3(dims, 'r32float');
  const n = dims[0] * dims[1] * dims[2];
  const bufA = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
  const bufB = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
  const bufD = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
  const U = ubuf(dims, { omega: 1.0 }), Uw = ubuf(dims, { omega: 6 / 7 });
  const U0 = ubuf(dims, { parity: 0, omega: 1.85 }), U1 = ubuf(dims, { parity: 1, omega: 1.85 });
  const v = t => t.createView();
  const wc = wgCount(dims, wg);
  const wcHalfX = wgCount([Math.ceil(dims[0] / 2), dims[1], dims[2]], wg);

  // init
  { const pp = pipe('init', wg); const e = device.createCommandEncoder(); const p = e.beginComputePass();
    p.setPipeline(pp.p); p.setBindGroup(0, bg(pp, [U, v(velA), v(scA)])); p.dispatchWorkgroups(...wc);
    p.setBindGroup(0, bg(pp, [U, v(velB), v(scB)])); p.dispatchWorkgroups(...wc); p.end(); device.queue.submit([e.finish()]); }

  const reps = 10;
  // semi-Lagrangian advection (one rgba16f field)
  { const pp = pipe('advect', wg); const g = bg(pp, [U, v(velA), v(scA), linear, v(scB)]);
    R.advectSL = await time((p, r) => { p.setPipeline(pp.p); p.setBindGroup(0, g); for (let i = 0; i < r; i++) p.dispatchWorkgroups(...wc); }, reps); }
  // rgba32float advection, if filterable
  if (device.features.has('float32-filterable')) {
    const a32 = tex3(dims, 'rgba32float'), b32 = tex3(dims, 'rgba32float');
    const pp = pipe('advect', wg, 'rgba32float'); const g = bg(pp, [U, v(velA), v(a32), linear, v(b32)]);
    R.advectSL_rgba32f = await time((p, r) => { p.setPipeline(pp.p); p.setBindGroup(0, g); for (let i = 0; i < r; i++) p.dispatchWorkgroups(...wc); }, reps);
    a32.destroy(); b32.destroy();
  }
  // MacCormack: forward SL + correction/limiter (2 dispatches)
  { const pa = pipe('advect', wg), pc = pipe('mccorrect', wg);
    const g1 = bg(pa, [U, v(velA), v(scA), linear, v(scH)]);
    const g2 = bg(pc, [U, v(velA), v(scA), v(scH), linear, v(scB)]);
    R.maccormack = await time((p, r) => { for (let i = 0; i < r; i++) { p.setPipeline(pa.p); p.setBindGroup(0, g1); p.dispatchWorkgroups(...wc); p.setPipeline(pc.p); p.setBindGroup(0, g2); p.dispatchWorkgroups(...wc); } }, reps); }
  // divergence
  { const pp = pipe('div', wg); const g = bg(pp, [U, v(velA), v(divT)]);
    R.divergence = await time((p, r) => { p.setPipeline(pp.p); p.setBindGroup(0, g); for (let i = 0; i < r; i++) p.dispatchWorkgroups(...wc); }, reps); }
  // Jacobi (texture ping-pong), per iteration
  { const pp = pipe('jacobi', wg); const gAB = bg(pp, [Uw, v(pA), v(divT), v(pB)]), gBA = bg(pp, [Uw, v(pB), v(divT), v(pA)]);
    R.jacobiTex = await time((p, r) => { p.setPipeline(pp.p); for (let i = 0; i < r; i++) { p.setBindGroup(0, i & 1 ? gBA : gAB); p.dispatchWorkgroups(...wc); } }, 40); }
  // Jacobi (storage buffers)
  { const pp = pipe('jacobibuf', wg); const gAB = bg(pp, [Uw, bufA, bufD, bufB]), gBA = bg(pp, [Uw, bufB, bufD, bufA]);
    R.jacobiBuf = await time((p, r) => { p.setPipeline(pp.p); for (let i = 0; i < r; i++) { p.setBindGroup(0, i & 1 ? gBA : gAB); p.dispatchWorkgroups(...wc); } }, 40); }
  // red-black SOR, per full iteration (red + black)
  { const pp = pipe('rbgs', wg); const g0 = bg(pp, [U0, v(pA), v(divT)]), g1 = bg(pp, [U1, v(pA), v(divT)]);
    R.rbSOR = await time((p, r) => { p.setPipeline(pp.p); for (let i = 0; i < r; i++) { p.setBindGroup(0, g0); p.dispatchWorkgroups(...wcHalfX); p.setBindGroup(0, g1); p.dispatchWorkgroups(...wcHalfX); } }, 40); }
  // projection
  { const pp = pipe('project', wg); const g = bg(pp, [U, v(velA), v(pA), v(velB)]);
    R.project = await time((p, r) => { p.setPipeline(pp.p); p.setBindGroup(0, g); for (let i = 0; i < r; i++) p.dispatchWorkgroups(...wc); }, reps); }
  // vorticity: curl + confinement/buoyancy
  { const p1 = pipe('curl', wg), p2 = pipe('confine', wg);
    const g1 = bg(p1, [U, v(velA), v(curlT)]), g2 = bg(p2, [U, v(velA), v(curlT), v(scA), v(velB)]);
    R.vorticity = await time((p, r) => { for (let i = 0; i < r; i++) { p.setPipeline(p1.p); p.setBindGroup(0, g1); p.dispatchWorkgroups(...wc); p.setPipeline(p2.p); p.setBindGroup(0, g2); p.dispatchWorkgroups(...wc); } }, reps); }

  // full step: MC(vel) + MC(scalars) + vorticity/buoyancy + div + 40x Jacobi + project
  { const pa = pipe('advect', wg), pc = pipe('mccorrect', wg), p1 = pipe('curl', wg), p2 = pipe('confine', wg), pd = pipe('div', wg), pj = pipe('jacobi', wg), pr = pipe('project', wg);
    const gv1 = bg(pa, [U, v(velA), v(velA), linear, v(velH)]), gv2 = bg(pc, [U, v(velA), v(velA), v(velH), linear, v(velB)]);
    const gs1 = bg(pa, [U, v(velA), v(scA), linear, v(scH)]), gs2 = bg(pc, [U, v(velA), v(scA), v(scH), linear, v(scB)]);
    const gc1 = bg(p1, [U, v(velB), v(curlT)]), gc2 = bg(p2, [U, v(velB), v(curlT), v(scB), v(velH)]);
    const gd = bg(pd, [U, v(velH), v(divT)]);
    const gAB = bg(pj, [Uw, v(pA), v(divT), v(pB)]), gBA = bg(pj, [Uw, v(pB), v(divT), v(pA)]);
    const gp = bg(pr, [U, v(velH), v(pA), v(velA)]);
    const D = (pp, g) => { };
    R.fullStep40 = await time((p, r) => {
      for (let k = 0; k < r; k++) {
        p.setPipeline(pa.p); p.setBindGroup(0, gv1); p.dispatchWorkgroups(...wc);
        p.setPipeline(pc.p); p.setBindGroup(0, gv2); p.dispatchWorkgroups(...wc);
        p.setPipeline(pa.p); p.setBindGroup(0, gs1); p.dispatchWorkgroups(...wc);
        p.setPipeline(pc.p); p.setBindGroup(0, gs2); p.dispatchWorkgroups(...wc);
        p.setPipeline(p1.p); p.setBindGroup(0, gc1); p.dispatchWorkgroups(...wc);
        p.setPipeline(p2.p); p.setBindGroup(0, gc2); p.dispatchWorkgroups(...wc);
        p.setPipeline(pd.p); p.setBindGroup(0, gd); p.dispatchWorkgroups(...wc);
        p.setPipeline(pj.p); for (let i = 0; i < 40; i++) { p.setBindGroup(0, i & 1 ? gBA : gAB); p.dispatchWorkgroups(...wc); }
        p.setPipeline(pr.p); p.setBindGroup(0, gp); p.dispatchWorkgroups(...wc);
      }
    }, 3);
  }

  // ray march (worst case: no early-out), 2 steps per voxel along the ray
  if (dims[0] === dims[1] || true) {
    const mod = device.createShaderModule({ code: RAYMARCH });
    const bgl = device.createBindGroupLayout({ entries: [E(0,'u'), E(1,'tex'), E(2,'samp'), E(3,'st',{dim:'2d'})] });
    const rp = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: mod, entryPoint: 'main' } });
    for (const [W, H] of [[960, 540], [1920, 1080]]) {
      for (const sh of [0, 6]) {
        const ot = device.createTexture({ size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
        const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(ub, 0, new Float32Array([W, H, 2 * Math.max(...dims), sh, 0.5, 0.55, -1.1, 0]));
        const g = device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: scA.createView() }, { binding: 2, resource: linear }, { binding: 3, resource: ot.createView() }] });
        R[`raymarch_${W}x${H}_shadow${sh}`] = await time((p, r) => { p.setPipeline(rp); p.setBindGroup(0, g); for (let i = 0; i < r; i++) p.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8)); }, 4);
        ot.destroy();
      }
    }
  }
  for (const t of [velA, velB, velH, scA, scB, scH, curlT, pA, pB, divT]) t.destroy();
  for (const b of [bufA, bufB, bufD]) b.destroy();
  return R;
}

async function benchWG(dims) {
  // workgroup shape sweep for advection and Jacobi
  const R = {};
  for (const wg of [[4,4,4], [8,8,1], [8,4,4], [8,8,4], [16,4,4], [32,2,2]]) {
    const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const velA = tex3(dims, 'rgba16float'), scA = tex3(dims, 'rgba16float'), scB = tex3(dims, 'rgba16float');
    const pA = tex3(dims, 'r32float'), pB = tex3(dims, 'r32float'), divT = tex3(dims, 'r32float');
    const U = ubuf(dims, { omega: 6 / 7 });
    const wc = wgCount(dims, wg);
    const v = t => t.createView();
    { const pp = pipe('init', wg); const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pp.p); p.setBindGroup(0, bg(pp, [U, v(velA), v(scA)])); p.dispatchWorkgroups(...wc); p.end(); device.queue.submit([e.finish()]); }
    const pa = pipe('advect', wg); const ga = bg(pa, [U, v(velA), v(scA), linear, v(scB)]);
    const adv = await time((p, r) => { p.setPipeline(pa.p); p.setBindGroup(0, ga); for (let i = 0; i < r; i++) p.dispatchWorkgroups(...wc); }, 10);
    const pj = pipe('jacobi', wg); const gAB = bg(pj, [U, v(pA), v(divT), v(pB)]), gBA = bg(pj, [U, v(pB), v(divT), v(pA)]);
    const jac = await time((p, r) => { p.setPipeline(pj.p); for (let i = 0; i < r; i++) { p.setBindGroup(0, i & 1 ? gBA : gAB); p.dispatchWorkgroups(...wc); } }, 40);
    R[wg.join('x')] = { advect: adv, jacobi: jac };
    for (const t of [velA, scA, scB, pA, pB, divT]) t.destroy();
  }
  return R;
}

// ---------- readback / transfer ----------
async function benchReadback(W, H, frames, post, label) {
  const tex = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT });
  const bytesPerRow = Math.ceil(W * 4 / 256) * 256;
  const NBUF = 3;
  const staging = Array.from({ length: NBUF }, () => device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
  const mod = device.createShaderModule({ code: `
    @group(0) @binding(0) var o: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(1) var<uniform> f: vec4<f32>;
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      let d = textureDimensions(o); if (any(id.xy >= d)) { return; }
      let uv = vec2<f32>(id.xy) / vec2<f32>(d);
      textureStore(o, vec2<i32>(id.xy), vec4<f32>(uv, 0.5 + 0.5 * sin(f.x * 0.2), 1.0));
    }` });
  const pl = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const g = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: { buffer: ub } }] });
  if (post === 'ffmpeg') await fetch(`/ffmpeg/start?w=${W}&h=${H}&out=${encodeURIComponent(qs.get('outdir') + `/rb_${W}x${H}.mp4`)}`, { method: 'POST' });
  const packed = new Uint8Array(W * H * 4);
  const t0 = performance.now();
  let tRead = 0, tSend = 0;
  for (let i = 0; i < frames; i++) {
    device.queue.writeBuffer(ub, 0, new Float32Array([i, 0, 0, 0]));
    const e = device.createCommandEncoder();
    const p = e.beginComputePass(); p.setPipeline(pl); p.setBindGroup(0, g); p.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8)); p.end();
    const sb = staging[i % NBUF];
    e.copyTextureToBuffer({ texture: tex }, { buffer: sb, bytesPerRow, rowsPerImage: H }, [W, H]);
    device.queue.submit([e.finish()]);
    const a = performance.now();
    await sb.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(sb.getMappedRange());
    if (bytesPerRow === W * 4) packed.set(src);
    else for (let y = 0; y < H; y++) packed.set(src.subarray(y * bytesPerRow, y * bytesPerRow + W * 4), y * W * 4);
    sb.unmap();
    const b = performance.now(); tRead += b - a;
    if (post === 'post' || post === 'ffmpeg') await fetch('/frame', { method: 'POST', body: packed });
    if (post === 'expose') {
      // base64 through a puppeteer-exposed binding (string-only channel)
      let s = ''; const CH = 0x8000; for (let k = 0; k < packed.length; k += CH) s += String.fromCharCode.apply(null, packed.subarray(k, k + CH));
      await window.__sendFrame(btoa(s));
    }
    tSend += performance.now() - b;
  }
  const total = performance.now() - t0;
  let ff = null;
  if (post === 'ffmpeg') ff = await (await fetch('/ffmpeg/stop', { method: 'POST' })).json();
  tex.destroy(); staging.forEach(b => b.destroy());
  return { label, W, H, bytesPerRow, frames, fps: +(frames / total * 1000).toFixed(2), msReadback: +(tRead / frames).toFixed(2), msSend: +(tSend / frames).toFixed(2), ff };
}

// GPU-side BT.709 limited-range RGB -> planar YUV 4:2:0 (8 or 10 bit) with TPDF dither. One thread = 8x2 pixel block.
function yuvShader(bits) {
  const ten = bits === 10;
  return /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> frame: vec4<u32>;
fn hash(p: vec3<u32>) -> f32 {
  var h = (p.x * 1597334677u) ^ (p.y * 3812015801u) ^ (p.z * 2798796415u);
  h = (h ^ (h >> 15u)) * 2246822519u; h = h ^ (h >> 13u);
  return f32(h) * (1.0 / 4294967296.0);
}
const SCALE = ${ten ? '4.0' : '1.0'};
const MAXV = ${ten ? '1023.0' : '255.0'};
fn q(v: f32, p: vec3<u32>) -> u32 {      // TPDF dither (+-1 LSB) then round
  let d = hash(p) - hash(p + vec3<u32>(7u, 11u, 13u));
  return u32(clamp(round(v + d), 0.0, MAXV));
}
fn Yv(c: vec3<f32>) -> f32 { return (16.0 + 219.0 * dot(c, vec3<f32>(0.2126, 0.7152, 0.0722))) * SCALE; }
fn UVv(c: vec3<f32>) -> vec2<f32> {
  let y = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  return (vec2<f32>(128.0) + 224.0 * vec2<f32>((c.b - y) / 1.8556, (c.r - y) / 1.5748)) * SCALE;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let d = textureDimensions(src);
  let bx = id.x * 8u; let by = id.y * 2u;
  if (bx >= d.x || by >= d.y) { return; }
  let W = d.x; let H = d.y; let f = frame.x;
  var ys: array<u32, 16>; var us: array<u32, 4>; var vs: array<u32, 4>;
  for (var i = 0u; i < 8u; i++) {
    let c0 = clamp(textureLoad(src, vec2<u32>(bx + i, by), 0).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let c1 = clamp(textureLoad(src, vec2<u32>(bx + i, by + 1u), 0).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    ys[i] = q(Yv(c0), vec3<u32>(bx + i, by, f));
    ys[8u + i] = q(Yv(c1), vec3<u32>(bx + i, by + 1u, f));
  }
  for (var j = 0u; j < 4u; j++) {
    let x = bx + 2u * j;
    let c = 0.25 * (textureLoad(src, vec2<u32>(x, by), 0).rgb + textureLoad(src, vec2<u32>(x + 1u, by), 0).rgb
                  + textureLoad(src, vec2<u32>(x, by + 1u), 0).rgb + textureLoad(src, vec2<u32>(x + 1u, by + 1u), 0).rgb);
    let uv = UVv(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
    us[j] = q(uv.x, vec3<u32>(x, by, f + 101u)); vs[j] = q(uv.y, vec3<u32>(x, by, f + 202u));
  }
${ten ? `
  // 16-bit little-endian samples, 2 per u32
  let yRow0 = (by * W + bx) / 2u; let yRow1 = ((by + 1u) * W + bx) / 2u;
  for (var k = 0u; k < 4u; k++) {
    dst[yRow0 + k] = ys[2u * k] | (ys[2u * k + 1u] << 16u);
    dst[yRow1 + k] = ys[8u + 2u * k] | (ys[8u + 2u * k + 1u] << 16u);
  }
  let cIdx = ((by / 2u) * (W / 2u) + bx / 2u);
  let uBase = (W * H) / 2u; let vBase = uBase + (W * H) / 8u;
  dst[uBase + cIdx / 2u] = us[0] | (us[1] << 16u); dst[uBase + cIdx / 2u + 1u] = us[2] | (us[3] << 16u);
  dst[vBase + cIdx / 2u] = vs[0] | (vs[1] << 16u); dst[vBase + cIdx / 2u + 1u] = vs[2] | (vs[3] << 16u);
` : `
  let yRow0 = (by * W + bx) / 4u; let yRow1 = ((by + 1u) * W + bx) / 4u;
  dst[yRow0]      = ys[0] | (ys[1] << 8u) | (ys[2] << 16u) | (ys[3] << 24u);
  dst[yRow0 + 1u] = ys[4] | (ys[5] << 8u) | (ys[6] << 16u) | (ys[7] << 24u);
  dst[yRow1]      = ys[8] | (ys[9] << 8u) | (ys[10] << 16u) | (ys[11] << 24u);
  dst[yRow1 + 1u] = ys[12] | (ys[13] << 8u) | (ys[14] << 16u) | (ys[15] << 24u);
  let cIdx = ((by / 2u) * (W / 2u) + bx / 2u) / 4u;
  let uBase = (W * H) / 4u; let vBase = uBase + (W * H) / 16u;
  dst[uBase + cIdx] = us[0] | (us[1] << 8u) | (us[2] << 16u) | (us[3] << 24u);
  dst[vBase + cIdx] = vs[0] | (vs[1] << 8u) | (vs[2] << 16u) | (vs[3] << 24u);
`}
}`;
}

async function benchXfer(W, H, frames, channel, fmt, label) {
  // channel: 'post' | 'ws'; fmt: 'rgba' | 'yuv420p' | 'yuv420p10le' | 'none'(no ffmpeg)
  const tex = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
  const gen = device.createComputePipeline({ layout: 'auto', compute: { entryPoint: 'main', module: device.createShaderModule({ code: `
    @group(0) @binding(0) var o: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(1) var<uniform> f: vec4<u32>;
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      let d = textureDimensions(o); if (any(id.xy >= d)) { return; }
      let uv = vec2<f32>(id.xy) / vec2<f32>(d);
      let t = f32(f.x) * 0.05;
      textureStore(o, vec2<i32>(id.xy), vec4<f32>(pow(uv.y, 0.5), 0.45 * uv.y * (0.5 + 0.5 * sin(uv.x * 20.0 + t)), 0.1 * uv.x, 1.0));
    }` }) } });
  const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const gg = device.createBindGroup({ layout: gen.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: { buffer: ub } }] });
  const yuv = fmt.startsWith('yuv');
  const ten = fmt === 'yuv420p10le';
  const frameBytes = yuv ? W * H * 3 / 2 * (ten ? 2 : 1) : W * H * 4;
  const bytesPerRow = Math.ceil(W * 4 / 256) * 256;
  const stagingSize = yuv ? frameBytes : bytesPerRow * H;
  let conv = null, cg = null, yb = null;
  if (yuv) {
    conv = device.createComputePipeline({ layout: 'auto', compute: { entryPoint: 'main', module: device.createShaderModule({ code: yuvShader(ten ? 10 : 8) }) } });
    yb = device.createBuffer({ size: frameBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    cg = device.createBindGroup({ layout: conv.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: { buffer: yb } }, { binding: 2, resource: { buffer: ub } }] });
  }
  const NB = 3;
  const staging = Array.from({ length: NB }, () => device.createBuffer({ size: stagingSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
  let ws = null;
  if (channel === 'ws') {
    ws = new WebSocket(`ws://${location.host}/ws`); ws.binaryType = 'arraybuffer';
    await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  }
  const pendingAcks = [];
  if (ws) ws.onmessage = () => { const r = pendingAcks.shift(); r && r(); };
  const sendWS = (data) => new Promise(r => { pendingAcks.push(r); ws.send(data); });
  if (fmt !== 'none') await fetch(`/ffmpeg/start?w=${W}&h=${H}&pix=${fmt}&out=${encodeURIComponent(qs.get('outdir') + `/x_${channel}_${fmt}_${W}x${H}.mp4`)}`, { method: 'POST' });
  const packed = new Uint8Array(frameBytes);
  const DL_FRAMES = 15; let dlBatch = [], dlCount = 0;
  if (channel === 'download') await window.__resetDownloads();
  let tGpu = 0, tSend = 0;
  const t0 = performance.now();
  for (let i = 0; i < frames; i++) {
    const a = performance.now();
    device.queue.writeBuffer(ub, 0, new Uint32Array([i, 0, 0, 0]));
    const e = device.createCommandEncoder();
    const p = e.beginComputePass(); p.setPipeline(gen); p.setBindGroup(0, gg); p.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    if (yuv) { p.setPipeline(conv); p.setBindGroup(0, cg); p.dispatchWorkgroups(Math.ceil(W / 64), Math.ceil(H / 16)); }
    p.end();
    const sb = staging[i % NB];
    if (yuv) e.copyBufferToBuffer(yb, 0, sb, 0, frameBytes);
    else e.copyTextureToBuffer({ texture: tex }, { buffer: sb, bytesPerRow, rowsPerImage: H }, [W, H]);
    device.queue.submit([e.finish()]);
    await sb.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(sb.getMappedRange());
    if (yuv || bytesPerRow === W * 4) packed.set(src.subarray(0, frameBytes));
    else for (let y = 0; y < H; y++) packed.set(src.subarray(y * bytesPerRow, y * bytesPerRow + W * 4), y * W * 4);
    sb.unmap();
    const b = performance.now(); tGpu += b - a;
    if (channel === 'ws') await sendWS(packed);
    else if (channel === 'post') await fetch('/frame', { method: 'POST', body: packed });
    else if (channel === 'expose') {
      const b64 = packed.toBase64 ? packed.toBase64() : await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result.slice(fr.result.indexOf(',') + 1)); fr.readAsDataURL(new Blob([packed])); });
      await window.__sendFrame(b64);
    } else if (channel === 'download') {
      dlBatch.push(packed.slice());
      if (dlBatch.length === DL_FRAMES || i === frames - 1) {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob(dlBatch)); a.download = `frames_${String(i).padStart(5, '0')}.raw`;
        document.body.appendChild(a); a.click(); a.remove();
        dlCount++; dlBatch = [];
      }
    }
    tSend += performance.now() - b;
  }
  if (channel === 'download') await window.__waitDownloads(dlCount);
  const total = performance.now() - t0;
  let ff = null;
  if (fmt !== 'none') ff = await (await fetch('/ffmpeg/stop', { method: 'POST' })).json();
  if (ws) ws.close();
  tex.destroy(); staging.forEach(s => s.destroy()); yb && yb.destroy();
  return { label, channel, fmt, W, H, frameMB: +(frameBytes / 1048576).toFixed(2), frames, fps: +(frames / total * 1000).toFixed(2), msGpuAndMap: +(tGpu / frames).toFixed(2), msSend: +(tSend / frames).toFixed(2), MBps: +(frameBytes * frames / (tSend / 1000) / 1048576).toFixed(1), ff };
}

async function benchBW() {
  const R = {};
  const MB = 64 * 1024 * 1024;
  const a = device.createBuffer({ size: MB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const b = device.createBuffer({ size: MB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  // copyBufferToBuffer (read 64MB + write 64MB)
  {
    const reps = 10, g = [];
    for (let t = 0; t < 3; t++) {
      const e = device.createCommandEncoder();
      const p = e.beginComputePass({ timestampWrites: { querySet: qset, beginningOfPassWriteIndex: 0 } }); p.end();
      for (let i = 0; i < reps; i++) e.copyBufferToBuffer(a, 0, b, 0, MB);
      const p2 = e.beginComputePass({ timestampWrites: { querySet: qset, endOfPassWriteIndex: 1 } }); p2.end();
      e.resolveQuerySet(qset, 0, 2, resolveBuf, 0); e.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16);
      device.queue.submit([e.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ); const x = new BigUint64Array(readBuf.getMappedRange()); g.push(Number(x[1] - x[0]) / 1e6 / reps); readBuf.unmap();
    }
    const ms = median(g); R.copyBufferToBuffer = { ms: +ms.toFixed(3), GBps: +(2 * MB / ms / 1e6).toFixed(1) };
  }
  for (const wgs of [64, 256]) {
    const m = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read> s: array<vec4<f32>>;
      @group(0) @binding(1) var<storage, read_write> d: array<vec4<f32>>;
      @compute @workgroup_size(${wgs}) fn main(@builtin(global_invocation_id) i: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
        let idx = i.x + i.y * n.x * ${wgs}u;
        if (idx < arrayLength(&s)) { d[idx] = s[idx] * 1.0001; }
      }` });
    const pl = device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
    const g = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } }] });
    const groups = MB / 16 / wgs;
    const r = await time((p, n) => { p.setPipeline(pl); p.setBindGroup(0, g); for (let i = 0; i < n; i++) p.dispatchWorkgroups(Math.min(groups, 32768), Math.ceil(groups / 32768)); }, 10);
    R['bufCopyKernel_vec4_wg' + wgs] = { ms: r.gpu, GBps: +(2 * MB / r.gpu / 1e6).toFixed(1) };
  }
  // 3D texture r32float 256^3 (64MB) load->store, and rgba16float 256^3 (128MB)
  for (const [fmt, N] of [['r32float', 256], ['rgba16float', 256], ['rgba16float', 128]]) {
    for (const wg of [[4, 4, 4], [8, 8, 4], [64, 1, 1], [8, 8, 1]]) {
      const ta = tex3([N, N, N], fmt), tb = tex3([N, N, N], fmt);
      const m = device.createShaderModule({ code: `
        @group(0) @binding(0) var s: texture_3d<f32>;
        @group(0) @binding(1) var d: texture_storage_3d<${fmt}, write>;
        @compute @workgroup_size(${wg}) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
          if (any(i >= vec3<u32>(${N}u))) { return; }
          textureStore(d, i, textureLoad(s, i, 0) * 1.0001);
        }` });
      const pl = device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
      const g = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: [{ binding: 0, resource: ta.createView() }, { binding: 1, resource: tb.createView() }] });
      const wc = wgCount([N, N, N], wg);
      const r = await time((p, n) => { p.setPipeline(pl); p.setBindGroup(0, g); for (let i = 0; i < n; i++) p.dispatchWorkgroups(...wc); }, 10);
      const bytes = N * N * N * (fmt === 'r32float' ? 4 : 8) * 2;
      R[`tex3d_${fmt}_${N}_wg${wg.join('x')}`] = { ms: r.gpu, GBps: +(bytes / r.gpu / 1e6).toFixed(1) };
      ta.destroy(); tb.destroy();
    }
  }
  // 2D texture equivalent: 4096x4096 r32float (64MB)
  {
    const mk = () => device.createTexture({ size: [4096, 4096], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    const ta = mk(), tb = mk();
    const m = device.createShaderModule({ code: `
      @group(0) @binding(0) var s: texture_2d<f32>;
      @group(0) @binding(1) var d: texture_storage_2d<r32float, write>;
      @compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
        textureStore(d, i.xy, textureLoad(s, i.xy, 0) * 1.0001);
      }` });
    const pl = device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
    const g = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: [{ binding: 0, resource: ta.createView() }, { binding: 1, resource: tb.createView() }] });
    const r = await time((p, n) => { p.setPipeline(pl); p.setBindGroup(0, g); for (let i = 0; i < n; i++) p.dispatchWorkgroups(256, 256); }, 10);
    R.tex2d_r32float_4096 = { ms: r.gpu, GBps: +(2 * MB / r.gpu / 1e6).toFixed(1) };
    ta.destroy(); tb.destroy();
  }
  // ALU throughput: 1M threads x 4096 FMAs (x4 lanes)
  {
    const m = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read_write> d: array<vec4<f32>>;
      @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
        var x = vec4<f32>(f32(i.x) * 1e-7, 0.5, 0.25, 0.125);
        var y = vec4<f32>(0.999, 0.998, 0.997, 0.996);
        for (var k = 0; k < 1024; k++) { x = fma(x, y, vec4<f32>(1e-6)); y = fma(y, x, vec4<f32>(-1e-6)); x = fma(x, y, vec4<f32>(1e-6)); y = fma(y, x, vec4<f32>(-1e-6)); }
        if (x.x == 123.0) { d[i.x] = x + y; }
      }` });
    const pl = device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
    const g = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: b } }] });
    const r = await time((p, n) => { p.setPipeline(pl); p.setBindGroup(0, g); for (let i = 0; i < n; i++) p.dispatchWorkgroups(4096); }, 5);
    const flops = 4096 * 256 * 4096 * 4 * 2;
    R.aluFma = { ms: r.gpu, TFLOPS: +(flops / r.gpu / 1e9).toFixed(2) };
  }
  a.destroy(); b.destroy();
  return R;
}

async function main() {
  try {
    if (!navigator.gpu) throw new Error('navigator.gpu undefined (secure context? flags?)');
    out.wgslLanguageFeatures = [...navigator.gpu.wgslLanguageFeatures].sort();
    out.preferredCanvasFormat = navigator.gpu.getPreferredCanvasFormat();
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('requestAdapter returned null');
    const info = adapter.info || {};
    out.adapterInfo = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter, subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize,
      driver: info.driver, backend: info.backend, type: info.type, d3dShaderModel: info.d3dShaderModel, powerPreference: info.powerPreference,
      memoryHeaps: info.memoryHeaps && [...info.memoryHeaps].map(h => ({ sizeGB: +(Number(h.size) / 2 ** 30).toFixed(2), properties: h.properties })) };
    out.features = [...adapter.features].sort();
    out.limits = {}; for (const k in adapter.limits) out.limits[k] = adapter.limits[k];
    const want = ['timestamp-query', 'float32-filterable', 'float32-blendable', 'shader-f16', 'texture-formats-tier1', 'texture-formats-tier2', 'rg11b10ufloat-renderable', 'subgroups'].filter(f => adapter.features.has(f));
    const L = adapter.limits;
    device = await adapter.requestDevice({ requiredFeatures: want, requiredLimits: {
      maxStorageTexturesPerShaderStage: L.maxStorageTexturesPerShaderStage,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxBufferSize: L.maxBufferSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    } });
    out.deviceFeatures = [...device.features].sort();
    device.lost.then(i => { out.errors.push('LOST ' + i.reason + ' ' + i.message); log('DEVICE LOST', i.reason, i.message); });
    device.onuncapturederror = e => { out.errors.push(e.error.message); log('ERR', e.error.message); };
    ts = device.features.has('timestamp-query');
    if (ts) {
      qset = device.createQuerySet({ type: 'timestamp', count: 2 });
      resolveBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      readBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    // capability probes: does rgba16float read_write storage compile & validate?
    const probe = async (code, entries) => {
      device.pushErrorScope('validation');
      try {
        const m = device.createShaderModule({ code });
        const bgl = device.createBindGroupLayout({ entries });
        device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: m, entryPoint: 'main' } });
      } catch (e) { await device.popErrorScope(); return 'throw: ' + e.message; }
      const err = await device.popErrorScope();
      return err ? 'invalid: ' + err.message.split('\n')[0] : 'ok';
    };
    out.probes = {
      r32float_rw: await probe(`@group(0) @binding(0) var t: texture_storage_3d<r32float, read_write>;
        @compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) i: vec3<u32>) { textureStore(t, i, textureLoad(t, i) + 1.0); }`,
        [E(0, 'st', { f: 'r32float', a: 'read-write' })]),
      rgba16float_rw: await probe(`@group(0) @binding(0) var t: texture_storage_3d<rgba16float, read_write>;
        @compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) i: vec3<u32>) { textureStore(t, i, textureLoad(t, i) * 0.5); }`,
        [E(0, 'st', { f: 'rgba16float', a: 'read-write' })]),
      rgba16float_readonly: await probe(`@group(0) @binding(0) var t: texture_storage_3d<rgba16float, read>;
        @group(0) @binding(1) var o: texture_storage_3d<rgba16float, write>;
        @compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) i: vec3<u32>) { textureStore(o, i, textureLoad(t, i)); }`,
        [E(0, 'st', { f: 'rgba16float', a: 'read-only' }), E(1, 'st')]),
      r16float_write: await probe(`@group(0) @binding(0) var t: texture_storage_3d<r16float, write>;
        @compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id) i: vec3<u32>) { textureStore(t, i, vec4<f32>(1.0)); }`,
        [E(0, 'st', { f: 'r16float', a: 'write-only' })]),
      rg11b10_write: await probe(`@group(0) @binding(0) var t: texture_storage_2d<rg11b10ufloat, write>;
        @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) i: vec3<u32>) { textureStore(t, i.xy, vec4<f32>(1.0)); }`,
        [E(0, 'st', { f: 'rg11b10ufloat', a: 'write-only', dim: '2d' })]),
    };
    log(JSON.stringify({ adapterInfo: out.adapterInfo, features: out.features, probes: out.probes }, null, 1));
    if (MODE === 'probe') { out.done = true; return; }

    if (MODE === 'sustain') {
      const MB = 64 * 1024 * 1024;
      const a = device.createBuffer({ size: MB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const b = device.createBuffer({ size: MB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const series = [];
      const t0 = performance.now();
      while (performance.now() - t0 < 8000) {
        const e = device.createCommandEncoder();
        const p = e.beginComputePass({ timestampWrites: { querySet: qset, beginningOfPassWriteIndex: 0 } }); p.end();
        for (let i = 0; i < 20; i++) e.copyBufferToBuffer(a, 0, b, 0, MB);
        const p2 = e.beginComputePass({ timestampWrites: { querySet: qset, endOfPassWriteIndex: 1 } }); p2.end();
        e.resolveQuerySet(qset, 0, 2, resolveBuf, 0); e.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16);
        device.queue.submit([e.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ); const x = new BigUint64Array(readBuf.getMappedRange()); const ms = Number(x[1] - x[0]) / 1e6; readBuf.unmap();
        series.push(+(2 * MB * 20 / ms / 1e6).toFixed(1));
      }
      out.results.sustainGBps = series;
      log(series.join(' '));
    }
    if (MODE === 'compile') {
      // compile + validate the reference WGSL files (wgsl/*.wgsl)
      const files = { 'mg.wgsl': ['smoothRB', 'residual', 'restrictLevel', 'prolongAdd', 'jacobiSmooth'], 'post.wgsl': ['bloomDown', 'bloomUp'], 'volume.wgsl': ['march', 'upsample'] };
      out.results.compile = {};
      for (const [f, eps] of Object.entries(files)) {
        const code = await (await fetch('wgsl/' + f)).text();
        device.pushErrorScope('validation');
        const m = device.createShaderModule({ code, label: f });
        const info = await m.getCompilationInfo();
        const res = { messages: info.messages.map(x => `${x.type} ${x.lineNum}:${x.linePos} ${x.message}`), pipelines: {} };
        for (const ep of eps) {
          device.pushErrorScope('validation');
          device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: ep } });
          const e = await device.popErrorScope();
          res.pipelines[ep] = e ? e.message.split('\n')[0] : 'ok';
        }
        if (f === 'post.wgsl') {
          device.pushErrorScope('validation');
          device.createRenderPipeline({ layout: 'auto', vertex: { module: m, entryPoint: 'fsTriangle' }, fragment: { module: m, entryPoint: 'composite', targets: [{ format: 'bgra8unorm' }] }, primitive: { topology: 'triangle-list' } });
          const e = await device.popErrorScope();
          res.pipelines.composite = e ? e.message.split('\n')[0] : 'ok';
        }
        const e0 = await device.popErrorScope();
        if (e0) res.moduleError = e0.message.split('\n').slice(0, 4).join(' | ');
        out.results.compile[f] = res;
      }
      // the sim reference: one module per section
      const ref = await (await fetch('wgsl/sim_reference.wgsl')).text();
      const parts = ref.split('// ======================= ').slice(1);
      for (const p of parts) {
        const name = p.split(' ')[0];
        device.pushErrorScope('validation');
        const m = device.createShaderModule({ code: p.slice(p.indexOf('\n')) });
        device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
        const e = await device.popErrorScope();
        out.results.compile['sim:' + name] = e ? e.message.split('\n')[0] : 'ok';
      }
      log(JSON.stringify(out.results.compile, null, 1));
    }
    if (MODE === 'determinism') {
      out.results.determinism = await determinism(64, 30);
      log('hashes', out.results.determinism.join(' '));
    }
    if (MODE === 'mg') {
      out.results.mg = {};
      for (const N of (qs.get('sizes') || '64,128').split(',').map(Number)) { out.results.mg[N] = await benchMG(N); log(N, JSON.stringify(out.results.mg[N])); }
    }
    if (MODE === 'tb') {
      for (const n of [64, 128]) { out.results['tb_' + n] = await benchTB([n, n, n]); log(n, JSON.stringify(out.results['tb_' + n])); }
    }
    if (MODE === 'xfer') {
      const list = [];
      const W = 1080, H = 1920;
      const plan = qs.get('plan') === 'verify' ? [['ws', 'rgba', 15], ['ws', 'yuv420p', 15], ['ws', 'yuv420p10le', 15]] : qs.get('plan') === 'alt'
        ? [['expose', 'none', 20], ['download', 'none', 60], ['ws', 'none', 20]]
        : [['post', 'none', 10], ['ws', 'none', 30], ['ws', 'rgba', 90], ['ws', 'yuv420p', 90], ['ws', 'yuv420p10le', 60]];
      out.hasToBase64 = !!Uint8Array.prototype.toBase64;
      for (const [ch, fmt, n] of plan) {
        log('xfer', ch, fmt);
        list.push(await benchXfer(W, H, n, ch, fmt, `${ch}/${fmt}`));
        log(JSON.stringify(list[list.length - 1]));
      }
      out.results.xfer = list;
    }
    if (MODE === 'all' || MODE === 'bw') {
      out.results.bw = await benchBW();
      log(JSON.stringify(out.results.bw, null, 1));
    }
    if (MODE === 'all' || MODE === 'grid') {
      const sizes = (qs.get('sizes') || '64,96,128,160,128x192x128').split(',');
      for (const s of sizes) {
        const d = s.includes('x') ? s.split('x').map(Number) : [+s, +s, +s];
        log('grid', s, '...');
        out.results['grid_' + s] = await benchGrid(d, [8, 8, 4]);
        log(JSON.stringify(out.results['grid_' + s]));
      }
      log('wg sweep 128^3');
      out.results.wgSweep128 = await benchWG([128, 128, 128]);
      log(JSON.stringify(out.results.wgSweep128));
    }
    if (MODE === 'all' || MODE === 'readback') {
      const list = [];
      list.push(await benchReadback(1920, 1080, 60, 'none', 'readback only'));
      list.push(await benchReadback(1080, 1920, 60, 'none', 'readback only (padded rows)'));
      list.push(await benchReadback(1080, 1920, 60, 'post', 'readback + HTTP POST'));
      if (window.__sendFrame) list.push(await benchReadback(1080, 1920, 20, 'expose', 'readback + exposeFunction(base64)'));
      if (qs.get('outdir')) list.push(await benchReadback(1080, 1920, 90, 'ffmpeg', 'readback + POST + ffmpeg x264 veryfast'));
      out.results.readback = list;
      log(JSON.stringify(list, null, 1));
    }
  } catch (e) {
    out.errors.push(String(e && e.stack || e)); log('FATAL', e && e.stack || e);
  } finally { out.done = true; }
}
main();
