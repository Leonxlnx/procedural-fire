// 3D Eulerian combustion solver on the GPU (low-Mach, constant-density projection with expansion source).
// Fields (cell-centred, rgba16float):
//   vel : xyz velocity (m/s), w = target divergence phi from gas expansion (1/s)
//   S   : x theta = (T - 293 K) / 1000, y fuel (pyrolysis gas mass fraction), z soot (ppm), w heat-release rate q (1e4 K/s)
//   S2  : x combustion products (mass fraction), y gas age since leaving the fuel (s), z organic smoke,
//         w fraction of the fuel that is CO from the coal bed (burns blue, makes no soot)
//   air (oxygen carrier) mass fraction = 1 - fuel - products
// Step: MacCormack advection -> curl -> sources/combustion/forces -> divergence -> multigrid Poisson -> projection.
// Model follows research/physics.md §10 (mixing-limited burning with oxygen, age-gated soot for the blue base,
// soot oxidation at the tips, soot radiative cooling, McCaffrey-calibrated buoyancy).
import { tex3D, UniformStruct, computePipeline, bindGroup, dispatch3 } from './gpu.js';
import { HASH, GNOISE, SDF } from './shaders/lib.js';

export const SIM_FIELDS = [
  'nx', 'ny', 'nz', 'dx', 'dt', 'time', 'frame', 'seed',
  'ox', 'oy', 'oz', 'nLogs',
  'fuelSrc', 'bedFuel', 'srcTemp', 'bedTemp', 'srcVel', 'srcFreq', 'srcSpeed', 'srcThresh',
  'ignT', 'kMix', 'rs', 'heatOut', 'expansion', 'sootForm', 'sootOx', 'radCool',
  'coolRate', 'sootDissip', 'smokeDissip', 'vitDecay', 'fuelDissip', 'buoyancy', 'vortEps', 'turbAmp',
  'turbFreq', 'turbRise', 'turbEvolve', 'windX', 'windZ', 'gust', 'intensity', 'velDissip',
  'bedRadius', 'logSrc', 'burstX', 'burstY', 'burstZ', 'burstAmt', 'smolder', 'sponge', 'hotR0', 'hotR1', 'hotY1',
  'visc', 'lamAge', 'anchorFreq', 'srcFloor', 'anchorGain',
];

export const SIM_DEFAULTS = {
  fuelSrc: 0.13,     // fuel mass fraction at active pyrolysis patches
  bedFuel: 0.16,     // weak CO-like fuel over the coal bed (blue flames)
  srcTemp: 0.75,     // patch temperature (1043 K): hot char surface acts as the pilot
  bedTemp: 0.82,     // coal bed gas temperature (1113 K)
  srcVel: 0.25,      // pyrolysis gas outflow (m/s)
  srcFreq: 35.0, srcSpeed: 0.06, srcThresh: 0.0,
  ignT: 550, kMix: 30, rs: 4.7, heatOut: 7700, expansion: 0.7,
  sootForm: 25, sootOx: 30, radCool: 0.25,
  coolRate: 3.0, sootDissip: 0.4, smokeDissip: 0.3, vitDecay: 1.5, fuelDissip: 0.3,
  buoyancy: 26.8,    // beta*g*1000/293 with beta = 0.8 (McCaffrey C^2)
  vortEps: 8, turbAmp: 0.8, turbFreq: 1.5, turbRise: 1.5, turbEvolve: 0.6,
  windX: 0, windZ: 0, gust: 0, intensity: 1, velDissip: 0.05,
  bedRadius: 0.22, hotR0: 0.12, hotR1: 0.45, hotY1: 0.5, logSrc: 1, burstX: 0, burstY: 0, burstZ: 0, burstAmt: 0, smolder: 0.02, sponge: 1,
  visc: 0,          // hot-gas viscosity in units of molecular air viscosity: nu = visc*1.5e-5*(T/293)^1.7 m2/s; 0 = inviscid (as before)
  lamAge: 0,        // fuel-gas age (s) below which the flame base stays laminar (no confinement / sub-grid noise); 0 = off
  anchorFreq: 1.6,  // spatial frequency (1/m) of the static pyrolysis anchor field (base wavelength 1/(4*anchorFreq) m)
  srcFloor: 0.25,   // pyrolysis release between anchors; < 0 acts as a cut-off (no flaming gas between tongues)
  anchorGain: 0.95, // pyrolysis release added at a full anchor
};

const WG = [8, 8, 4];

export class FireSim {
  constructor(device, { res = [80, 120, 80], size = [1.2, 1.8, 1.2], logs, vcycles = 2, noiseTex, curlTex, dbg = 0 }) {
    this.device = device;
    this.res = res;
    this.size = size;
    this.dx = size[0] / res[0];
    this.origin = [-size[0] / 2, 0, -size[2] / 2];
    this.vcycles = vcycles;
    this.dbg = dbg;
    this.time = 0;
    this.frame = 0;
    this.params = { ...SIM_DEFAULTS };
    this.noiseTex = noiseTex;
    this.curlTex = curlTex;

    this.U = new UniformStruct(device, 'SimParams', SIM_FIELDS);
    this.logBuf = device.createBuffer({ size: logs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.logBuf, 0, logs);
    this.nLogs = logs.length / 8;

    this.lin = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    this.rep = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' });

    const T = (l) => tex3D(device, res, 'rgba16float', l);
    this.vel = T('vel'); this.velX = T('velX'); this.velY = T('velY');
    this.S = T('S'); this.SX = T('SX'); this.SY = T('SY');
    this.S2 = T('S2'); this.S2X = T('S2X'); this.S2Y = T('S2Y');
    this.vort = T('vort');
    this.geo = T('geo');   // r: log pyrolysis band*heat, g: coal bed, b: solid, a: log sdf

    // Multigrid hierarchy.
    this.levels = [];
    let [x, y, z] = res;
    let h = this.dx;
    while (true) {
      const i = this.levels.length;
      this.levels.push({ dims: [x, y, z], h, p: tex3D(device, [x, y, z], 'r32float', `p${i}`), b: tex3D(device, [x, y, z], 'r32float', `b${i}`) });
      if (x % 2 || y % 2 || z % 2 || Math.min(x, y, z) <= 6) break;
      x /= 2; y /= 2; z /= 2; h *= 2;
    }
    // Enough red-black sweeps on the coarsest grid to cut its smoothest error mode to ~5 %
    // (sides/top Dirichlet, floor Neumann -> the vertical wavelength doubles).
    const [cx, cy, cz] = this.levels[this.levels.length - 1].dims;
    const mu = (Math.cos(Math.PI / cx) + Math.cos(Math.PI / (2 * cy)) + Math.cos(Math.PI / cz)) / 3;
    this.coarseIters = Math.max(8, Math.min(64, Math.ceil(Math.log(0.05) / Math.log(mu * mu))));
    for (const L of this.levels) {
      L.u = [0, 1].map(color => {
        const b = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(b, 0, new Float32Array([L.dims[0], L.dims[1], L.dims[2], L.h, color, 0, 0, 0]));
        return b;
      });
    }
    this.build();
  }

  header() {
    const [nx, ny, nz] = this.res;
    return /* wgsl */`
${this.U.wgsl()}
${HASH}
${GNOISE}
${SDF}
const DBG = ${this.dbg}u;
const DIMS = vec3i(${nx}, ${ny}, ${nz});
const DIMSF = vec3f(${nx}.0, ${ny}.0, ${nz}.0);
fn inside(id: vec3u) -> bool { return all(vec3i(id) < DIMS); }
fn ld(t: texture_3d<f32>, c: vec3i) -> vec4f { return textureLoad(t, clamp(c, vec3i(0), DIMS - 1), 0); }
`;
  }

  build() {
    const d = this.device;
    const H = this.header();

    // ---------------- Advection (MacCormack, 2 passes) ----------------
    const advCommon = /* wgsl */`
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var velT: texture_3d<f32>;
@group(0) @binding(2) var sT: texture_3d<f32>;
@group(0) @binding(3) var s2T: texture_3d<f32>;
@group(0) @binding(4) var lin: sampler;
fn sampleVel(c: vec3f) -> vec3f { return textureSampleLevel(velT, lin, c / DIMSF, 0.0).xyz; }
fn backtrace(c: vec3f, dt: f32) -> vec3f {
  let v0 = sampleVel(c);
  let mid = c - 0.5 * dt * v0 / P.dx;
  let v1 = sampleVel(mid);
  return c - dt * v1 / P.dx;
}
// Scalars entering from outside the domain are ambient (0).
fn outsideFade(c: vec3f) -> f32 {
  let lo = min(c.x, c.z); let hi = max(c.x - DIMSF.x, max(c.y - DIMSF.y, c.z - DIMSF.z));
  return clamp(1.0 - max(-lo, hi), 0.0, 1.0);
}
`;
    this.pAdv1 = computePipeline(d, H + advCommon + /* wgsl */`
@group(0) @binding(5) var velOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var sOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var s2Out: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (!inside(id)) { return; }
  let c = vec3f(id) + 0.5;
  let b = backtrace(c, P.dt);
  let f = outsideFade(b);
  let uvw = b / DIMSF;
  // air entering through the open sides carries the ambient wind (stronger with height above the ground)
  let wind = vec3f(P.windX, 0.0, P.windZ) * smoothstep(0.03, 0.4, b.y * P.dx);
  textureStore(velOut, id, vec4f(mix(wind, textureSampleLevel(velT, lin, uvw, 0.0).xyz, f), 0.0));
  textureStore(sOut, id, textureSampleLevel(sT, lin, uvw, 0.0) * f);
  textureStore(s2Out, id, textureSampleLevel(s2T, lin, uvw, 0.0) * f);
}`, 'main', 'adv1');

    this.pAdv2 = computePipeline(d, H + advCommon + /* wgsl */`
@group(0) @binding(5) var velHat: texture_3d<f32>;
@group(0) @binding(6) var sHat: texture_3d<f32>;
@group(0) @binding(7) var s2Hat: texture_3d<f32>;
@group(0) @binding(8) var velOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var sOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var s2Out: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (!inside(id)) { return; }
  let ci = vec3i(id);
  let c = vec3f(id) + 0.5;
  let v = textureLoad(velT, ci, 0).xyz;
  // reverse advection of the forward result
  let fw = (c + P.dt * v / P.dx) / DIMSF;
  let vH = textureLoad(velHat, ci, 0).xyz;
  let sH = textureLoad(sHat, ci, 0);
  let s2H = textureLoad(s2Hat, ci, 0);
  var vN = vH + 0.5 * (v - textureSampleLevel(velHat, lin, fw, 0.0).xyz);
  var sN = sH + 0.5 * (textureLoad(sT, ci, 0) - textureSampleLevel(sHat, lin, fw, 0.0));
  var s2N = s2H + 0.5 * (textureLoad(s2T, ci, 0) - textureSampleLevel(s2Hat, lin, fw, 0.0));
  // clamp to the range of the source stencil at the backtraced point (Selle et al. 2008)
  let b = backtrace(c, P.dt) - 0.5;
  let b0 = vec3i(floor(b));
  var vmn = vec3f(1e9); var vmx = vec3f(-1e9);
  var smn = vec4f(1e9); var smx = vec4f(-1e9);
  var tmn = vec4f(1e9); var tmx = vec4f(-1e9);
  for (var k = 0; k < 8; k++) {
    let q = b0 + vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let vv = ld(velT, q).xyz; let ss = ld(sT, q); let tt = ld(s2T, q);
    vmn = min(vmn, vv); vmx = max(vmx, vv); smn = min(smn, ss); smx = max(smx, ss); tmn = min(tmn, tt); tmx = max(tmx, tt);
  }
  if (outsideFade(b + 0.5) < 1.0) { vN = vH; sN = sH; s2N = s2H; }
  vN = clamp(vN, min(vmn, vH), max(vmx, vH));
  sN = clamp(sN, min(smn, sH), max(smx, sH));
  s2N = clamp(s2N, min(tmn, s2H), max(tmx, s2H));
  textureStore(velOut, id, vec4f(vN, 0.0));
  textureStore(sOut, id, max(sN, vec4f(0.0)));
  textureStore(s2Out, id, max(s2N, vec4f(0.0)));
}`, 'main', 'adv2');

    // ---------------- Curl ----------------
    this.pCurl = computePipeline(d, H + /* wgsl */`
@group(0) @binding(0) var velT: texture_3d<f32>;
@group(0) @binding(1) var vortOut: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (!inside(id)) { return; }
  let c = vec3i(id);
  let L = ld(velT, c - vec3i(1,0,0)).xyz; let R = ld(velT, c + vec3i(1,0,0)).xyz;
  let B = ld(velT, c - vec3i(0,1,0)).xyz; let T = ld(velT, c + vec3i(0,1,0)).xyz;
  let K = ld(velT, c - vec3i(0,0,1)).xyz; let F = ld(velT, c + vec3i(0,0,1)).xyz;
  let w = vec3f((T.z - B.z) - (F.y - K.y), (F.x - K.x) - (R.z - L.z), (R.y - L.y) - (T.x - B.x)) * 0.5;
  textureStore(vortOut, id, vec4f(w, length(w)));
}`, 'main', 'curl');

    // ---------------- Sources, combustion, forces ----------------
    this.pForces = computePipeline(d, H + /* wgsl */`
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var velT: texture_3d<f32>;
@group(0) @binding(2) var sT: texture_3d<f32>;
@group(0) @binding(3) var s2T: texture_3d<f32>;
@group(0) @binding(4) var vortT: texture_3d<f32>;
@group(0) @binding(5) var noiseT: texture_3d<f32>;
@group(0) @binding(6) var curlT: texture_3d<f32>;
@group(0) @binding(7) var rep: sampler;
@group(0) @binding(8) var geoT: texture_3d<f32>;
@group(0) @binding(9) var velOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var sOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(11) var s2Out: texture_storage_3d<rgba16float, write>;

fn noise4(q: vec3f) -> vec4f { return textureSampleLevel(noiseT, rep, q, 0.0) * 2.0 - 1.0; }

@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (!inside(id)) { return; }
  let ci = vec3i(id);
  let p = vec3f(P.ox, P.oy, P.oz) + (vec3f(id) + 0.5) * P.dx;
  let dt = P.dt;
  var v = textureLoad(velT, ci, 0).xyz;
  var s = textureLoad(sT, ci, 0);
  var s2 = textureLoad(s2T, ci, 0);
  let geo = textureLoad(geoT, ci, 0);
  let I = P.intensity;

  // ---- sources: pyrolysis on the hot log surfaces + the coal bed.
  // A static field anchors the strongest gas release at a handful of spots (log gaps, undersides), so the fire
  // grows several distinct tongues; a finer, slow noise (patch lifetimes ~3-5 s) modulates each spot. The total
  // supply only varies by ~±10%, so the puffing rhythm comes from the buoyant instability, not the source.
  // (Noise lookups only in source cells: incoherent texture reads are very slow on iGPUs.)
  var src = 0.0;
  var anchor = 0.0;
  if (geo.r > 0.001) {
    let qa = p * P.anchorFreq + vec3f(P.seed * 0.37, 0.0, 0.11);
    anchor = smoothstep(-0.05, 0.25, noise4(qa).z);
    let qn = p * (P.srcFreq * 0.06) + vec3f(P.seed * 0.131, 0.0, P.time * P.srcSpeed);
    let pulse = smoothstep(P.srcThresh - 0.25, P.srcThresh + 0.25, noise4(qn).x);
    // srcFloor < 0 is a cut-off: wood between anchors releases no flaming gas (dark gaps between tongues)
    src = clamp(geo.r * P.logSrc * (P.srcFloor + P.anchorGain * anchor) * mix(0.55, 1.0, pulse) * 1.2, 0.0, 1.0) * I;
    src *= 1.0 - geo.b;                                // no gas release inside the solid wood
  }
  let srcW = min(src, 1.0);                            // blend weight (src itself may exceed 1 when I > 1)
  // the coal bed stays hot even when the fire is low; its CO 'fuel' (blue flamelets, flame anchor) follows intensity
  let bedHot = geo.g;
  var bed = geo.g * clamp(I * 1.4, 0.3, 1.0);
  // CO burns over the coals in small separate flamelets (2-5 cm), not as a uniform blue sheet
  if (bed > 0.001) {
    let qb = p * 2.4 + vec3f(0.17, -P.time * 0.12, P.time * 0.05);
    bed *= smoothstep(0.0, 0.3, noise4(qb).y);
  }
  s.y = max(s.y, src * P.fuelSrc);
  s.y = max(s.y, bed * P.bedFuel);
  s.x = max(s.x, src * P.srcTemp);
  s.x = max(s.x, bedHot * P.bedTemp);
  let srcAll = max(srcW, bed);
  // tag where the fuel came from: char-bed CO vs. wood pyrolysis gas
  s2.w = mix(s2.w, 1.0, bed * (1.0 - srcW));
  s2.w = mix(s2.w, 0.0, srcW);
  // s2.y holds the fuel-gas age weighted by Z = fuel + products, so entrained (aged) air carries no weight;
  // it restarts at the fuel surface
  s2.y = s2.y * (1.0 - srcAll);
  // pyrolysis gas is fuel + inert (steam, CO2): no oxygen inside the source, so burning happens only
  // where it mixes with air (a diffusion flame)
  s2.x = max(s2.x, (1.0 - s.y) * srcW);
  s2.x = max(s2.x, (1.0 - s.y) * bed * 0.6);
  let wsrc = 1.0 - pow(1.0 - clamp(srcW + bed * 0.4, 0.0, 1.0) * 0.5, dt * 90.0);
  v.y = mix(v.y, max(v.y, P.srcVel), wsrc);            // weak outflow (dt-consistent blend)
  // smouldering, non-flaming wood gives off a little organic smoke
  s2.z += geo.r * (1.0 - anchor) * P.smolder * dt * I;

  // burst (log shift / resin pop): a puff of hot, near-stoichiometric gas that flares within 1-2 frames
  // (no outward push, so the flame brightens instead of being blown apart)
  if (P.burstAmt > 0.0) {
    let bd = length(p - vec3f(P.burstX, P.burstY, P.burstZ));
    let bw = clamp(exp(-bd * bd / 0.006) * P.burstAmt, 0.0, 1.0);
    s.y = max(s.y, 0.17 * bw);
    s.x = max(s.x, 0.9 * bw);
    s2.x *= 1.0 - bw;
    s2.w *= 1.0 - bw;
    s2.y = max(s2.y, 0.05 * bw * clamp(s.y + s2.x, 0.0, 1.0));
  }
  var ageF = s2.y / max(s.y + s2.x, 1e-4);             // age of the fuel-derived gas (s)

  // ---- mixing-limited combustion (fast chemistry)
  let T0 = 293.0 + 1000.0 * s.x;
  let o = clamp(1.0 - s.y - s2.x, 0.0, 1.0);           // air available in the local mixture
  let ign = smoothstep(P.ignT - 50.0, P.ignT + 50.0, T0);
  let b = min(s.y, o / P.rs) * (1.0 - exp(-P.kMix * ign * dt));
  s.y -= b;
  s2.x = min(s2.x + (1.0 + P.rs) * b, 1.0);
  let dT = min(P.heatOut * b, max(2100.0 - T0, 0.0));
  var T = T0 + dT;
  let q = dT / dt;                                     // K/s
  let phi = select(min(P.expansion * q / T, 60.0), 0.0, geo.b > 0.5);   // gas expansion (1/s), none in solids

  // ---- soot: forms in hot, fuel-rich, aged gas; oxidised where hot and oxygen is available
  ageF = min(ageF + dt, 4.0);
  // soot appears on the flame sheet once the gas has aged a few tens of ms (gives the blue base)
  s.z += P.sootForm * b * smoothstep(0.015, 0.05, ageF) * smoothstep(1000.0, 1400.0, T) * (1.0 - s2.w);
  s.z *= exp(-P.sootOx * (0.25 + o) * smoothstep(1100.0, 1400.0, T) * dt);
  // tail burnout: soot mixed into air at 850-1100 K still oxidises, so glowing tails end above the reaction zone
  s.z *= exp(-P.sootOx * 0.5 * o * smoothstep(850.0, 1100.0, T) * (1.0 - smoothstep(1100.0, 1300.0, T)) * dt);
  s.z *= exp(-P.sootDissip * dt);

  // ---- cooling: soot radiation (optically thin, attenuated) + unresolved entrainment where not burning
  let th = T * 0.001;
  T -= min(P.radCool * 970.0 * s.z * th * th * (th * th * th * th - 0.00737), 6000.0) * dt;
  let notBurning = 1.0 - smoothstep(500.0, 5000.0, q);
  T = 293.0 + max(T - 293.0, 0.0) * exp(-P.coolRate * notBurning * dt);
  s.x = (T - 293.0) * 0.001;
  s.w = q * 1e-4;
  s.y *= exp(-P.fuelDissip * dt);
  s2.x *= exp(-P.vitDecay * dt);
  s2.z *= exp(-P.smokeDissip * dt);
  s2.y = ageF * clamp(s.y + s2.x, 0.0, 1.0);

  // ---- buoyancy: a = beta g (T - Tamb) / Tamb
  v.y += P.buoyancy * s.x * dt;

  // ---- hot-gas viscosity: nu = visc * 1.5e-5 m2/s * (T/293)^1.7 (visc = 1: molecular, ~2.4e-4 m2/s at 1500 K).
  // At dx = 7.5 mm the flame gas has a grid Reynolds number of ~100, so its viscous cut-off is resolvable: 1-3 cell
  // eddies decay within tens of ms while 10 cm tongues and the puffing vortex are barely touched.
  // Explicit Laplacian of the advected velocity; diffusion number capped at 0.15 (< 1/6). visc = 0: off (as before).
  if (P.visc > 0.0) {
    let lapV = ld(velT, ci + vec3i(1, 0, 0)).xyz + ld(velT, ci - vec3i(1, 0, 0)).xyz
             + ld(velT, ci + vec3i(0, 1, 0)).xyz + ld(velT, ci - vec3i(0, 1, 0)).xyz
             + ld(velT, ci + vec3i(0, 0, 1)).xyz + ld(velT, ci - vec3i(0, 0, 1)).xyz
             - 6.0 * textureLoad(velT, ci, 0).xyz;
    let nu = P.visc * 1.5e-5 * pow(max(T, 293.0) / 293.0, 1.7);
    v += min(nu * dt / (P.dx * P.dx), 0.15) * lapV;
  }
  // ---- laminar flame base (McCaffrey continuous zone z < 0.08 Q^0.4 ~ 0.4 m, ~0.2 s of rise): fuel-derived gas
  // younger than lamAge gets no confinement and no sub-grid noise; both ramp in over [lamAge/2, lamAge].
  // lamAge = 0: lam = 1 exactly (unchanged behaviour).
  let lam = select(1.0, smoothstep(0.5 * P.lamAge, max(P.lamAge, 1e-3), ageF), P.lamAge > 0.0);

  // ---- vorticity confinement, ramped up in the 500-1300 K tongue/plume region and away from the source
  let w = textureLoad(vortT, ci, 0).xyz;
  let g = vec3f(ld(vortT, ci + vec3i(1,0,0)).w - ld(vortT, ci - vec3i(1,0,0)).w,
                ld(vortT, ci + vec3i(0,1,0)).w - ld(vortT, ci - vec3i(0,1,0)).w,
                ld(vortT, ci + vec3i(0,0,1)).w - ld(vortT, ci - vec3i(0,0,1)).w);
  let gl = length(g);
  if (gl > 1e-6) {
    let gate = smoothstep(450.0, 900.0, T) * (1.0 - srcAll) * smoothstep(0.02, 0.08, p.y) * lam;
    v += P.vortEps * gate * cross(g / gl, w) * dt;
  }

  // ---- sub-grid turbulence: rising, evolving curl noise gated by heat release and temperature
  let tq = (p - vec3f(0.0, P.time * P.turbRise, 0.0)) * P.turbFreq + vec3f(0.0, 0.0, P.time * P.turbEvolve);
  let cn = textureSampleLevel(curlT, rep, tq, 0.0).xyz;
  // stronger lateral mixing aloft (10-20 cm eddies between 0.3 and 1 m) so tongues fork, hook and lean
  let tgate = max(smoothstep(500.0, 1100.0, T), smoothstep(0.0, 3.0, s.w)) * (1.0 - srcAll) * mix(0.6, 1.8, smoothstep(0.2, 0.75, p.y)) * lam;
  v += cn * (P.turbAmp / 12.0) * tgate * dt * 10.0;

  // ---- wind above the logs
  // weak pull towards the ambient wind inside the domain (the inflow at the sides does most of the work)
  let windK = smoothstep(0.05, 0.5, p.y) * (1.0 + P.gust);
  v.x += (P.windX * windK - v.x) * dt * 0.35 * windK;
  v.z += (P.windZ * windK - v.z) * dt * 0.35 * windK;
  v *= exp(-P.velDissip * dt);

  // ---- solids & sponge layers (sides relax to rest, top fades scalars)
  if (geo.b > 0.5) { v = vec3f(0.0); s.y = 0.0; s2.y = 0.0; }
  let fp = vec3f(id) + 0.5;
  let side = min(min(fp.x, DIMSF.x - fp.x), min(fp.z, DIMSF.z - fp.z));
  let sp = (1.0 - smoothstep(0.0, 5.0, side)) * P.sponge;
  let wSide = vec2f(P.windX, P.windZ) * smoothstep(0.03, 0.4, p.y);
  // per-second rates (calibrated at dt = 1/90), so the look does not depend on the time step
  let ks = 1.0 - pow(1.0 - 0.15 * clamp(sp, 0.0, 1.0), dt * 90.0);
  v.x = mix(v.x, wSide.x, ks); v.z = mix(v.z, wSide.y, ks);
  let top = smoothstep(DIMSF.y - 10.0, DIMSF.y, fp.y);
  let kt = pow(1.0 - 0.1 * top, dt * 90.0);
  s.z *= kt; s2.z *= kt;

  let vl = length(v);
  if (vl > 15.0) { v *= 15.0 / vl; }
  if (any(s != s) || any(s2 != s2) || any(v != v)) { s = vec4f(0.0); s2 = vec4f(0.0); v = vec3f(0.0); }
  textureStore(velOut, id, vec4f(v, phi));
  textureStore(sOut, id, s);
  textureStore(s2Out, id, s2);
}`, 'main', 'forces');

    // ---------------- Divergence (minus the expansion target) ----------------
    this.pDiv = computePipeline(d, H + /* wgsl */`
@group(0) @binding(0) var velT: texture_3d<f32>;
@group(0) @binding(1) var<uniform> P: SimParams;
@group(0) @binding(2) var divOut: texture_storage_3d<r32float, write>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (!inside(id)) { return; }
  let c = vec3i(id);
  let me = textureLoad(velT, c, 0);
  let L = ld(velT, c - vec3i(1,0,0)).x; let R = ld(velT, c + vec3i(1,0,0)).x;
  var B = ld(velT, c - vec3i(0,1,0)).y; let T = ld(velT, c + vec3i(0,1,0)).y;
  let K = ld(velT, c - vec3i(0,0,1)).z; let F = ld(velT, c + vec3i(0,0,1)).z;
  if (c.y == 0) { B = -me.y; }   // solid floor
  let div = ((R - L) + (T - B) + (F - K)) / (2.0 * P.dx) - me.w;
  textureStore(divOut, id, vec4f(div, 0.0, 0.0, 0.0));
}`, 'main', 'div');

    // ---------------- Multigrid (cell-centred, Dirichlet sides/top, Neumann floor) ----------------
    const lvl = 'struct Lvl { nx: f32, ny: f32, nz: f32, h: f32, color: f32, _a: f32, _b: f32, _c: f32 };';
    this.pSmooth = computePipeline(d, /* wgsl */`
${lvl}
@group(0) @binding(0) var<uniform> L: Lvl;
@group(0) @binding(1) var p: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var b: texture_3d<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(i32(L.nx), i32(L.ny), i32(L.nz));
  let y = i32(gid.y); let z = i32(gid.z);
  let x = i32(gid.x) * 2 + ((y + z + i32(L.color)) & 1);
  if (x >= n.x || y >= n.y || z >= n.z) { return; }
  let c = vec3i(x, y, z);
  // open sides/top: p = 0 on the boundary face (ghost = -p), floor: Neumann (ghost = p)
  var sum = 0.0; var diag = 6.0;
  if (x > 0) { sum += textureLoad(p, c - vec3i(1,0,0)).x; } else { diag += 1.0; }
  if (x < n.x - 1) { sum += textureLoad(p, c + vec3i(1,0,0)).x; } else { diag += 1.0; }
  if (z > 0) { sum += textureLoad(p, c - vec3i(0,0,1)).x; } else { diag += 1.0; }
  if (z < n.z - 1) { sum += textureLoad(p, c + vec3i(0,0,1)).x; } else { diag += 1.0; }
  if (y > 0) { sum += textureLoad(p, c - vec3i(0,1,0)).x; } else { diag -= 1.0; }
  if (y < n.y - 1) { sum += textureLoad(p, c + vec3i(0,1,0)).x; } else { diag += 1.0; }
  textureStore(p, c, vec4f((sum - L.h * L.h * textureLoad(b, c, 0).x) / diag, 0.0, 0.0, 0.0));
}`, 'main', 'mgSmooth');

    this.pRestrict = computePipeline(d, /* wgsl */`
${lvl}
@group(0) @binding(0) var<uniform> L: Lvl;
@group(0) @binding(1) var p: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var b: texture_3d<f32>;
@group(0) @binding(3) var bc: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var pc: texture_storage_3d<r32float, write>;
fn P(c: vec3i, n: vec3i, me: f32) -> f32 {
  if (c.x < 0 || c.z < 0 || c.x >= n.x || c.z >= n.z || c.y >= n.y) { return -me; }
  if (c.y < 0) { return me; }
  return textureLoad(p, c).x;
}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(i32(L.nx), i32(L.ny), i32(L.nz));
  let cc = vec3i(gid);
  if (any(cc * 2 >= n)) { return; }
  var acc = 0.0;
  let ih2 = 1.0 / (L.h * L.h);
  for (var k = 0; k < 8; k++) {
    let c = cc * 2 + vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let me = textureLoad(p, c).x;
    let lap = (P(c - vec3i(1,0,0), n, me) + P(c + vec3i(1,0,0), n, me) + P(c - vec3i(0,1,0), n, me) +
               P(c + vec3i(0,1,0), n, me) + P(c - vec3i(0,0,1), n, me) + P(c + vec3i(0,0,1), n, me) - 6.0 * me) * ih2;
    acc += textureLoad(b, c, 0).x - lap;
  }
  textureStore(bc, cc, vec4f(acc * 0.125, 0.0, 0.0, 0.0));
  textureStore(pc, cc, vec4f(0.0));
}`, 'main', 'mgRestrict');

    this.pProlong = computePipeline(d, /* wgsl */`
${lvl}
@group(0) @binding(0) var<uniform> L: Lvl;
@group(0) @binding(1) var p: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var pc: texture_3d<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(i32(L.nx), i32(L.ny), i32(L.nz));
  let c = vec3i(gid);
  if (any(c >= n)) { return; }
  let nc = n / 2;
  let g = (vec3f(c) + 0.5) * 0.5 - 0.5;
  let g0 = vec3i(floor(g));
  let f = g - vec3f(g0);
  var acc = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let q = g0 + o;
    // mirror at the boundary faces: odd (Dirichlet) on sides/top, even (Neumann) at the floor
    // one sign flip per Dirichlet axis the node lies outside on (edges/corners: reflections combine)
    var val = textureLoad(pc, clamp(q, vec3i(0), nc - 1), 0).x;
    if (q.x < 0 || q.x >= nc.x) { val = -val; }
    if (q.z < 0 || q.z >= nc.z) { val = -val; }
    if (q.y >= nc.y) { val = -val; }
    let wv = mix(1.0 - f, f, vec3f(o));
    acc += val * wv.x * wv.y * wv.z;
  }
  textureStore(p, c, vec4f(textureLoad(p, c).x + acc, 0.0, 0.0, 0.0));
}`, 'main', 'mgProlong');

    // ---------------- Static emitter geometry (computed once) ----------------
    this.pGeo = computePipeline(d, H + /* wgsl */`
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> logs: array<vec4f>;
@group(0) @binding(2) var geoOut: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (!inside(id)) { return; }
  let p = vec3f(P.ox, P.oy, P.oz) + (vec3f(id) + 0.5) * P.dx;
  let nl = i32(P.nLogs);
  var dmin = 1e9; var band = 0.0;
  for (var i = 0; i < nl; i++) {
    let A = logs[i * 2]; let B = logs[i * 2 + 1];
    var dd = sdCapsule(p, A.xyz, B.xyz, A.w);
    dd += A.w * 0.3 * (1.0 - smoothstep(0.0, 0.16, length(p.xz))) * B.w;
    dmin = min(dmin, dd);
    // pyrolysis band just outside the hot part of each log
    let bnd = 1.0 - smoothstep(0.0, 1.6 * P.dx, abs(dd - 0.6 * P.dx));
    let hot = B.w * (1.0 - smoothstep(P.hotR0, P.hotR1, length(p.xz))) * (1.0 - smoothstep(0.08, P.hotY1, p.y));
    band = max(band, bnd * hot);
  }
  // coal bed: shallow mound in the middle
  let r = length(p.xz);
  let bedH = 0.05 * (1.0 - smoothstep(0.0, P.bedRadius, r)) + 0.02;
  let bed = (1.0 - smoothstep(0.0, 2.0 * P.dx, p.y - bedH)) * (1.0 - smoothstep(P.bedRadius * 0.6, P.bedRadius * 1.15, r));
  let solid = select(0.0, 1.0, dmin < -0.5 * P.dx);
  textureStore(geoOut, id, vec4f(band, bed, solid, dmin));
}`, 'main', 'geo');

    // ---------------- Projection ----------------
    this.pProject = computePipeline(d, H + /* wgsl */`
@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var velT: texture_3d<f32>;
@group(0) @binding(2) var pT: texture_3d<f32>;
@group(0) @binding(3) var geoT: texture_3d<f32>;
@group(0) @binding(4) var velOut: texture_storage_3d<rgba16float, write>;
fn pr(c: vec3i, me: f32) -> f32 {
  if (c.x < 0 || c.z < 0 || c.x >= DIMS.x || c.z >= DIMS.z || c.y >= DIMS.y) { return -me; }
  if (c.y < 0) { return me; }
  return textureLoad(pT, c, 0).x;
}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (!inside(id)) { return; }
  let c = vec3i(id);
  let me = textureLoad(pT, c, 0).x;
  let g = vec3f(pr(c + vec3i(1,0,0), me) - pr(c - vec3i(1,0,0), me),
                pr(c + vec3i(0,1,0), me) - pr(c - vec3i(0,1,0), me),
                pr(c + vec3i(0,0,1), me) - pr(c - vec3i(0,0,1), me)) / (2.0 * P.dx);
  var v = textureLoad(velT, c, 0).xyz - g;
  if (textureLoad(geoT, c, 0).b > 0.5) { v = vec3f(0.0); }
  if (c.y == 0) { v.y = max(v.y, 0.0); }
  let vl = length(v);
  if (vl > 20.0 || vl != vl) { v = select(v * 20.0 / vl, vec3f(0.0), vl != vl); }
  textureStore(velOut, id, vec4f(v, 0.0));
}`, 'main', 'project');

    this.makeBindGroups();
  }

  makeBindGroups() {
    const d = this.device, U = this.U.buffer;
    this.bgAdv1 = bindGroup(d, this.pAdv1, [U, this.vel, this.S, this.S2, this.lin, this.velX, this.SX, this.S2X]);
    this.bgAdv2 = bindGroup(d, this.pAdv2, [U, this.vel, this.S, this.S2, this.lin, this.velX, this.SX, this.S2X, this.velY, this.SY, this.S2Y]);
    this.bgCurl = bindGroup(d, this.pCurl, [this.velY, this.vort]);
    this.bgForces = bindGroup(d, this.pForces, [U, this.velY, this.SY, this.S2Y, this.vort, this.noiseTex, this.curlTex, this.rep, this.geo, this.velX, this.S, this.S2]);
    this.bgGeo = bindGroup(d, this.pGeo, [U, this.logBuf, this.geo]);
    const L0 = this.levels[0];
    this.bgDiv = bindGroup(d, this.pDiv, [this.velX, U, L0.b]);
    for (let i = 0; i < this.levels.length; i++) {
      const L = this.levels[i];
      L.bgSmooth = L.u.map(u => bindGroup(d, this.pSmooth, [u, L.p, L.b]));
      if (i + 1 < this.levels.length) {
        const C = this.levels[i + 1];
        L.bgRestrict = bindGroup(d, this.pRestrict, [L.u[0], L.p, L.b, C.b, C.p]);
        L.bgProlong = bindGroup(d, this.pProlong, [L.u[0], L.p, C.p]);
      }
    }
    this.bgProject = bindGroup(d, this.pProject, [U, this.velX, L0.p, this.geo, this.vel]);
  }

  smooth(pass, i, iters) {
    const L = this.levels[i];
    const [x, y, z] = L.dims;
    for (let k = 0; k < iters; k++) {
      for (const color of [0, 1]) {
        pass.setPipeline(this.pSmooth);
        pass.setBindGroup(0, L.bgSmooth[color]);
        pass.dispatchWorkgroups(Math.ceil(x / 2 / WG[0]), Math.ceil(y / WG[1]), Math.ceil(z / WG[2]));
      }
    }
  }

  vcycle(pass, pre = 2, post = 2, coarse = this.coarseIters) {
    const n = this.levels.length;
    for (let i = 0; i < n - 1; i++) {
      this.smooth(pass, i, pre);
      dispatch3(pass, this.pRestrict, this.levels[i].bgRestrict, this.levels[i + 1].dims, WG);
    }
    this.smooth(pass, n - 1, coarse);
    for (let i = n - 2; i >= 0; i--) {
      dispatch3(pass, this.pProlong, this.levels[i].bgProlong, this.levels[i].dims, WG);
      this.smooth(pass, i, post);
    }
  }

  setParams(p) {
    if ('hotR0' in p || 'hotR1' in p || 'hotY1' in p || 'bedRadius' in p) this.geoReady = false;
    Object.assign(this.params, p);
  }

  reset() {
    const [x, y, z] = this.res;
    const zeros = new Uint16Array(x * y * 4);
    for (const t of [this.vel, this.velX, this.velY, this.S, this.SX, this.SY, this.S2, this.S2X, this.S2Y, this.vort]) {
      for (let k = 0; k < z; k++) this.device.queue.writeTexture({ texture: t, origin: [0, 0, k] }, zeros, { bytesPerRow: x * 8, rowsPerImage: y }, [x, y, 1]);
    }
    for (const L of this.levels) {
      const [lx, ly, lz] = L.dims;
      const zf = new Float32Array(lx * ly);
      for (let k = 0; k < lz; k++) this.device.queue.writeTexture({ texture: L.p, origin: [0, 0, k] }, zf, { bytesPerRow: lx * 4, rowsPerImage: ly }, [lx, ly, 1]);
    }
    this.time = 0; this.frame = 0;
  }

  uploadUniforms(dt) {
    const [nx, ny, nz] = this.res;
    this.U.set({ ...this.params, nx, ny, nz, dx: this.dx, dt, time: this.time, frame: this.frame,
      ox: this.origin[0], oy: this.origin[1], oz: this.origin[2], nLogs: this.nLogs, seed: this.params.seed ?? 0 }).upload();
  }

  // NOTE: uniforms are written with queue.writeBuffer, so callers submit one step per command buffer.
  step(enc, dt, prof) {
    this.uploadUniforms(dt);
    const split = prof?.enabled && this.profileStages;
    let pass = enc.beginComputePass({ timestampWrites: prof?.pass(split ? 'adv' : 'sim') });
    const stage = (label) => { if (!split) return; pass.end(); pass = enc.beginComputePass({ timestampWrites: prof.pass(label) }); };
    if (!this.geoReady) { dispatch3(pass, this.pGeo, this.bgGeo, this.res, WG); this.geoReady = true; }
    dispatch3(pass, this.pAdv1, this.bgAdv1, this.res, WG);
    dispatch3(pass, this.pAdv2, this.bgAdv2, this.res, WG);
    stage('curl');
    dispatch3(pass, this.pCurl, this.bgCurl, this.res, WG);
    stage('forces');
    dispatch3(pass, this.pForces, this.bgForces, this.res, WG);
    stage('div');
    dispatch3(pass, this.pDiv, this.bgDiv, this.res, WG);
    stage('mg');
    for (let i = 0; i < this.vcycles; i++) this.vcycle(pass);
    stage('project');
    dispatch3(pass, this.pProject, this.bgProject, this.res, WG);
    pass.end();
    this.time += dt;
    this.frame++;
  }
}
