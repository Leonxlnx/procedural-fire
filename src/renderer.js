// Renderer: fire light extraction, SDF scene (logs, coals, stones, ground), volumetric fire ray-marcher,
// depth-aware composite with heat haze, sparks, bloom, accumulation (motion blur) and filmic output.
import { tex2D, tex3D, UniformStruct, computePipeline, bindGroup, layoutFromWGSL } from './gpu.js';
import { HASH, GNOISE, SDF, IGN } from './shaders/lib.js';
import { blackbodyLUT } from './blackbody.js';
import { SCENE_SDF, SCENE_SURFACE } from './shaders/scene.js';

export const RP_FIELDS = [
  'width', 'height', 'vw', 'vh', 'time', 'frame', 'nLogs', 'nStones',
  'camX', 'camY', 'camZ', 'tanFov', 'fwdX', 'fwdY', 'fwdZ', 'aspect', 'rgtX', 'rgtY', 'rgtZ', 'jitX', 'upX', 'upY', 'upZ', 'jitY',
  'ox', 'oy', 'oz', 'dx', 'sx', 'sy', 'sz', 'tempScale', 'nx', 'ny', 'nz', 'nLights',
  'exposure', 'stepMul', 'sootAbs', 'sootEmit', 'emitK', 'blueK', 'smokeScat', 'smokeG',
  'ambient', 'detailAmt', 'detailFreq', 'detailSpeed', 'hazeK', 'glowK', 'bloomK', 'tonemap',
  'grain', 'vignette', 'saturation', 'contrast', 'bedGlow', 'logGlow', 'lightK', 'accumN',
  'bbTmin', 'bbTmax', 'bbN', 'emberFlicker', 'sootEmitMin', 'flameSharp', 'moon', 'fogK',
  'shadowSoft', 'lift', 'emitTempLo', 'glowTemp', 'inset', 'knee', 'surfDetail', 'matA', 'matB', 'matC', 'sootScat', 'chroma', 'lensX', 'lensY', 'focus', 'lightDt', 'flickerK', 'cubic', 'aperture', 'sootEdge',
];

export const RP_DEFAULTS = {
  exposure: 15.0, stepMul: 0.7, sootAbs: 4.0, sootEmit: 10.0, emitK: 4.0, blueK: 0.05, smokeScat: 0.02, smokeG: 0.5,
  ambient: 1.0, detailAmt: 0.6, detailFreq: 2.2, detailSpeed: 0.9, hazeK: 1.0, glowK: 1.0, bloomK: 0.04, tonemap: 2,
  grain: 0.02, vignette: 0.6, saturation: 1.0, contrast: 1.0, bedGlow: 1.0, logGlow: 1.0, lightK: 1.0,
  emberFlicker: 1.0, sootEmitMin: 0.0, flameSharp: 0.0, moon: 1.0, fogK: 1.0, shadowSoft: 6.0, lift: 0.0,
  emitTempLo: 900, glowTemp: 1.0, tempScale: 1000, inset: 0.06, knee: 0.6, surfDetail: 1, matA: 0, matB: 0, matC: 0, sootScat: 0.015, chroma: 0.0025, lensX: 0, lensY: 0, focus: 0, lightDt: -1, flickerK: 0.35, cubic: 0, aperture: 0, sootEdge: 0,
};

const NLIGHTS = 8;

export class Renderer {
  constructor(device, { sim, scene, noiseTex, curlTex, logsF32, stonesF32, format }) {
    this.curlTex = curlTex;
    this.device = device;
    this.sim = sim;
    this.noiseTex = noiseTex;
    this.format = format;
    this.params = { ...RP_DEFAULTS };
    this.U = new UniformStruct(device, 'RP', RP_FIELDS);
    this.Usurf = new UniformStruct(device, 'RP', RP_FIELDS, 'RPsurf');
    this.nLogs = scene.logs.length;
    this.nStones = scene.stones.length;

    this.logBuf = device.createBuffer({ size: logsF32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.logBuf, 0, logsF32);
    this.stoneBuf = device.createBuffer({ size: stonesF32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.stoneBuf, 0, stonesF32);
    this.lightBuf = device.createBuffer({ size: NLIGHTS * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.lightState = device.createBuffer({ size: NLIGHTS * 3 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.lastLightTime = null;

    this.lin = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    this.rep = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' });

    this.setWhiteBalance(5000);
    this.build();
  }

  setWhiteBalance(Twb) {
    const lut = blackbodyLUT({ n: 1024, Tmin: 400, Tmax: 3000, Twb });
    this.bb = lut;
    if (!this.bbTex) this.bbTex = this.device.createTexture({ size: [lut.n, 2], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.device.queue.writeTexture({ texture: this.bbTex }, lut.data, { bytesPerRow: lut.n * 16 }, [lut.n, 2]);
    this.Twb = Twb;
  }

  header() {
    return /* wgsl */`
${this.U.wgsl()}
${HASH}
${GNOISE}
${SDF}
${IGN}
@group(0) @binding(0) var<uniform> R: RP;
const PI = 3.14159265;
fn bbRow(t: texture_2d<f32>, T: f32, row: u32) -> vec3f {
  let x = clamp((T - R.bbTmin) / (R.bbTmax - R.bbTmin), 0.0, 1.0) * (R.bbN - 1.0);
  let i = u32(floor(x)); let f = x - floor(x);
  let a = textureLoad(t, vec2u(i, row), 0).rgb;
  let b = textureLoad(t, vec2u(min(i + 1u, u32(R.bbN) - 1u), row), 0).rgb;
  return mix(a, b, f);
}
fn bbLoad(t: texture_2d<f32>, T: f32) -> vec3f { return bbRow(t, T, 0u); }
fn bbSoot(t: texture_2d<f32>, T: f32) -> vec3f { return bbRow(t, T, 1u); }
struct Light { pos: vec4f, col: vec4f };
// Thin-lens camera: rays start on the aperture (lens offset in the camera plane) and pass through the
// point where the pinhole ray meets the focal plane. focus <= 0 is a pinhole camera.
struct Ray { o: vec3f, d: vec3f };
fn lensRay(dPin: vec3f) -> Ray {
  let c = vec3f(R.camX, R.camY, R.camZ);
  if (R.focus <= 0.0) { return Ray(c, dPin); }
  let fwd = vec3f(R.fwdX, R.fwdY, R.fwdZ);
  let o = c + vec3f(R.rgtX, R.rgtY, R.rgtZ) * R.lensX + vec3f(R.upX, R.upY, R.upZ) * R.lensY;
  let q = c + dPin * (R.focus / dot(dPin, fwd));
  return Ray(o, normalize(q - o));
}
`;
  }

  build() {
    const d = this.device;
    const H = this.header();
    const sim = this.sim;

    // -------- Fire light extraction: one workgroup per height band --------
    this.pLights = computePipeline(d, H + /* wgsl */`
@group(0) @binding(1) var sT: texture_3d<f32>;
@group(0) @binding(2) var bbT: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> lights: array<Light>;
// temporal state per band: [0,N) smoothed pos, [N,2N) smoothed colour, [2N,3N) 1.5 s running mean colour
@group(0) @binding(4) var<storage, read_write> lstate: array<vec4f>;
var<workgroup> accP: array<vec4f, 256>;
var<workgroup> accC: array<vec4f, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u) {
  let n = vec3u(u32(R.nx), u32(R.ny), u32(R.nz));
  let bandH = n.y / ${NLIGHTS}u;
  let y0 = wg.x * bandH;
  var P = vec4f(0.0); var C = vec4f(0.0);
  let total = (n.x / 2u) * (bandH / 2u) * (n.z / 2u);
  for (var i = li; i < total; i += 256u) {
    let ix = (i % (n.x / 2u)) * 2u;
    let iy = ((i / (n.x / 2u)) % (bandH / 2u)) * 2u + y0;
    let iz = (i / ((n.x / 2u) * (bandH / 2u))) * 2u;
    let s = textureLoad(sT, vec3u(ix, iy, iz), 0);
    let T = 293.0 + s.x * R.tempScale;
    if (T < 700.0) { continue; }
    let e = s.z * R.sootEmit * bbSoot(bbT, T) * R.emitK;
    let w = dot(e, vec3f(0.2126, 0.7152, 0.0722));
    let pos = vec3f(R.ox, R.oy, R.oz) + (vec3f(f32(ix), f32(iy), f32(iz)) + 1.0) * R.dx;
    P += vec4f(pos * w, w);
    C += vec4f(e, dot(pos, pos) * w);
  }
  accP[li] = P; accC[li] = C;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { accP[li] += accP[li + s]; accC[li] += accC[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let w = accP[0].w;
    let cen = select(vec3f(0.0, (f32(y0) + f32(bandH) * 0.5) * R.dx, 0.0), accP[0].xyz / w, w > 1e-6);
    // spatial spread of the emitter (second moment) -> finite irradiance next to the flames
    let spread = select(0.04, max(accC[0].w / w - dot(cen, cen), 0.0), w > 1e-6);
    // 8x for the 2x2x2 subsampling; times cell volume -> radiant intensity
    // scale the 2x2x2 point sampling up to the whole band (exact even when the band height is odd)
    let J = accC[0].rgb * (f32(n.x * bandH * n.z) / f32(total)) * R.dx * R.dx * R.dx;
    // Real firelight flickers gently (sigma/mu 6-10 %, mostly < 4 Hz): low-pass the band's output (~80 ms)
    // and position (~150 ms), and compress it around a 1.5 s running mean.
    let N = ${NLIGHTS}u;
    let k = wg.x;
    var sp = lstate[k]; var sc = lstate[N + k]; var sm = lstate[2u * N + k];
    if (R.lightDt < 0.0 || sm.w <= 0.0) { sp = vec4f(cen, spread); sc = vec4f(J, 0.0); sm = vec4f(J, 1.0); }
    else {
      let a = 1.0 - exp(-R.lightDt / 0.08); let ap = 1.0 - exp(-R.lightDt / 0.15); let am = 1.0 - exp(-R.lightDt / 1.5);
      sp = mix(sp, vec4f(cen, spread), ap); sc = vec4f(mix(sc.rgb, J, a), 0.0); sm = vec4f(mix(sm.rgb, J, am), 1.0);
    }
    lstate[k] = sp; lstate[N + k] = sc; lstate[2u * N + k] = sm;
    let lum = vec3f(0.2126, 0.7152, 0.0722);
    let ratio = dot(sc.rgb, lum) / max(dot(sm.rgb, lum), 1e-9);
    let Jc = sc.rgb * pow(max(ratio, 1e-4), R.flickerK - 1.0);
    // gate on the smoothed output (not the raw weight), so surfaces and smoke follow the same filtered light
    let wl = dot(Jc, lum);
    let on = wl > 1e-3 * dot(sm.rgb, lum) && wl > 0.0;
    lights[wg.x] = Light(vec4f(sp.xyz, select(0.0, wl, on)), vec4f(select(vec3f(0.0), Jc, on), sp.w));
  }
}`, 'main', 'lights');

    // -------- Surfaces --------
    this.pSurface = computePipeline(d, H + SCENE_SDF + SCENE_SURFACE(NLIGHTS), 'main', 'surface');

    // -------- Flame-field prefilter for cubic reconstruction --------
    // s + k (s - mean of the 6 neighbours) is the first-order inverse of the [1 4 1]/6 B-spline kernel along
    // each axis: cubic B-spline sampling of the prefiltered field approaches interpolation, so grid-scale
    // flame sheets stay crisp instead of being smoothed a second time.
    this.pSharpen = computePipeline(d, H + /* wgsl */`
@group(0) @binding(1) var sIn: texture_3d<f32>;
@group(0) @binding(2) var sOut: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec3i(textureDimensions(sIn));
  let c = vec3i(id);
  if (any(c >= dims)) { return; }
  let hi = dims - 1;
  let s = textureLoad(sIn, c, 0);
  var n = vec4f(0.0); var lo = s; var up = s;
  for (var k = 0; k < 6; k++) {
    let o = select(vec3i(0), vec3i(select(-1, 1, (k & 1) == 0)), vec3<bool>(k / 2 == 0, k / 2 == 1, k / 2 == 2));
    let v = textureLoad(sIn, clamp(c + o, vec3i(0), hi), 0);
    n += v; lo = min(lo, v); up = max(up, v);
  }
  // clamped to the local range: edges steepen but peaks are never raised (emission grows ~exp(-1/T), so any
  // temperature overshoot would brighten the whole fire instead of sharpening it)
  textureStore(sOut, c, clamp(s + R.flameSharp * (s - n / 6.0), lo, up));
}`, 'main', 'sharpen');

    // -------- Volume ray-march --------
    this.pVolume = computePipeline(d, H + /* wgsl */`
@group(0) @binding(1) var sT: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var noiseT: texture_3d<f32>;
@group(0) @binding(4) var rep: sampler;
@group(0) @binding(5) var bbT: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> lights: array<Light>;
@group(0) @binding(7) var depthT: texture_2d<f32>;
@group(0) @binding(8) var volOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var auxOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(10) var s2T: texture_3d<f32>;
@group(0) @binding(11) var curlT: texture_3d<f32>;

fn camRay(px: vec2f) -> vec3f {
  let ndc = vec2f(px.x / R.vw * 2.0 - 1.0, 1.0 - px.y / R.vh * 2.0);
  let fwd = vec3f(R.fwdX, R.fwdY, R.fwdZ); let rgt = vec3f(R.rgtX, R.rgtY, R.rgtZ); let up = vec3f(R.upX, R.upY, R.upZ);
  return normalize(fwd + rgt * ndc.x * R.tanFov * R.aspect + up * ndc.y * R.tanFov);
}
fn hg(c: f32, g: f32) -> f32 { let g2 = g * g; return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * c, 1.5)); }
// Cubic B-spline reconstruction of the sim grid from 8 trilinear fetches (Sigg & Hadwiger 2005): removes the
// faceted 'brick' look of trilinear interpolation in close-ups. g is in texel units (cell centres at i + 0.5).
fn sampleCubic(g: vec3f, dims: vec3f) -> vec4f {
  let c = g - 0.5;
  let i = floor(c); let f = c - i;
  let f2 = f * f; let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1; let g1 = w2 + w3;
  let h0 = (i - 0.5 + w1 / g0) / dims; let h1 = (i + 1.5 + w3 / g1) / dims;
  var acc = vec4f(0.0);
  for (var k = 0; k < 8; k++) {
    let o = vec3<bool>((k & 1) != 0, (k & 2) != 0, (k & 4) != 0);
    let p = select(h0, h1, o);
    let wv = select(g0, g1, o);
    acc += textureSampleLevel(sT, lin, p, 0.0) * (wv.x * wv.y * wv.z);
  }
  return acc;
}

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= R.vw || f32(id.y) >= R.vh) { return; }
  let px = vec2f(id.xy) + 0.5 + vec2f(R.jitX, R.jitY) * (R.vw / R.width);
  let ray = lensRay(camRay(px));
  let ro = ray.o;
  let rd = ray.d;
  // depth from full-res surface buffer (centre sample)
  let ddim = vec2i(textureDimensions(depthT));
  let fpx = vec2i(px / vec2f(R.vw, R.vh) * vec2f(ddim));
  let sd = textureLoad(depthT, clamp(fpx, vec2i(0), ddim - 1), 0).x;
  let bmin = vec3f(R.ox, R.oy, R.oz); let bmax = bmin + vec3f(R.sx, R.sy, R.sz);
  let inv = 1.0 / rd;
  let t0 = (bmin - ro) * inv; let t1 = (bmax - ro) * inv;
  let tn = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), 0.0);
  let tf = min(min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z)), sd);
  var L = vec3f(0.0); var Tr = 1.0; var haze = 0.0;
  if (tf > tn) {
    let ds = R.dx * R.stepMul;
    let jit = ign(vec2f(id.xy), R.frame);
    var t = tn + ds * jit;
    let dims = vec3f(R.nx, R.ny, R.nz);
    let nlit = i32(R.nLights);
    let fwdDot = 0.0;
    for (var i = 0; i < 1200; i++) {
      if (t > tf || Tr < 0.003) { break; }
      let p = ro + rd * t;
      var g = (p - bmin) / R.dx;
      // sub-voxel detail: divergence-free (curl) warp of the lookup, advected upward in two phases that
      // cross-fade (no visible scrolling or stretching); rises faster with height like real tongues
      var mottle = 0.0;
      if (R.detailAmt > 0.0) {
        let rise = R.detailSpeed * (0.5 + 1.5 * smoothstep(0.0, 1.0, p.y));
        let ph = R.time / 0.7;
        let f0 = fract(ph); let f1 = fract(ph + 0.5);
        let w0 = 1.0 - abs(2.0 * f0 - 1.0);
        let o0 = vec3f(0.37, 0.0, 0.61) * floor(ph); let o1 = vec3f(0.71, 0.0, 0.29) * floor(ph + 0.5);
        let q0 = (p - vec3f(0.0, rise * f0 * 0.7, 0.0)) * R.detailFreq * 0.25 + o0;
        let q1 = (p - vec3f(0.0, rise * f1 * 0.7, 0.0)) * R.detailFreq * 0.25 + o1;
        let c0 = textureSampleLevel(curlT, rep, q0, 0.0); let c1 = textureSampleLevel(curlT, rep, q1, 0.0);
        let cw = c0 * w0 + c1 * (1.0 - w0);
        g += cw.xyz * (R.detailAmt * 0.12) * vec3f(1.0, 0.6, 1.0);
        mottle = cw.w * 0.08 - 0.6;
      }
      var s = vec4f(0.0);
      if (R.cubic > 0.5) { s = sampleCubic(g, dims); } else { s = textureSampleLevel(sT, lin, g / dims, 0.0); }
      // fade towards the open sides/top so the simulation box never shows
      let edge = min(min(g.x, dims.x - g.x), min(g.z, dims.z - g.z));
      let fade = smoothstep(0.02, 0.15, edge * R.dx) * smoothstep(0.0, 0.4, (dims.y - g.y) * R.dx);
      s *= fade;
      let T = 293.0 + s.x * R.tempScale + mottle * 60.0 * R.detailAmt * step(0.3, s.x);
      var soot = s.z;                       // ppm
      // flame-sheet reconstruction (sootEdge > 0, ppm): the unresolved soot-oxidation front is drawn as a steep transfer
      // around an iso-level of the smooth (cubic) soot field: crisp sheet edges and no glow from dilute soot between
      // tongues; monotone, never raises peaks, emission and absorption stay consistent. 0 = off.
      if (R.sootEdge > 0.0) { soot *= smoothstep(0.6 * R.sootEdge, 1.4 * R.sootEdge, soot); }
      var smoke = 0.0;
      if (R.smokeScat > 0.0) { smoke = textureSampleLevel(s2T, lin, g / dims, 0.0).z * fade; }
      if (soot > 1e-4 || s.w > 1e-3 || smoke > 1e-4) {
        // soot: kappa ~ 10 /m per ppm, albedo ~0.2; organic smoke: albedo ~0.9
        // emission uses the physical soot coefficient; absorption is lowered because the grid cannot
        // resolve the thin sheets that make real flames see-through
        let sigSoot = soot * R.sootAbs;
        let sigS = sigSoot * R.sootScat + smoke * R.smokeScat;
        let sigT = sigSoot + smoke * R.smokeScat * 1.1 + 1e-6;
        var e = vec3f(0.0);
        if (T > R.emitTempLo) { e = soot * R.sootEmit * bbSoot(bbT, T) * R.emitK; }
        // blue chemiluminescence (CH*/C2*) tracks heat release where soot hasn't formed yet
        // (concentrated in the strongest reaction zones so it reads as thin fringes, not a blue veil)
        e += vec3f(0.15, 0.3, 1.0) * pow(s.w, 1.6) * (R.blueK / 3.0) * exp(-soot * 2.0);
        // single scattering of fire light
        var ins = vec3f(0.0);
        if (sigS * Tr > 1e-6) {
          for (var k = 0; k < nlit; k++) {
            let Lk = lights[k];
            let toL = Lk.pos.xyz - p;
            let r2 = dot(toL, toL) + Lk.col.w * 0.6 + 0.004;
            ins += Lk.col.rgb * hg(dot(rd, toL * inverseSqrt(r2)), R.smokeG) / r2;
          }
          ins *= sigS * R.lightK;
        }
        let a = exp(-sigT * ds);
        let w = select((1.0 - a) / sigT, ds, sigT * ds < 1e-4);
        L += Tr * (e + ins) * w;
        Tr *= a;
      }
      haze += max(s.x - 0.05, 0.0) * ds;
      t += ds;
    }
  }
  textureStore(volOut, id.xy, vec4f(L, Tr));
  textureStore(auxOut, id.xy, vec4f(haze, sd, 0.0, 0.0));
}`, 'main', 'volume');

    // -------- Composite (depth-aware upsample + heat haze) --------
    this.pComposite = computePipeline(d, H + /* wgsl */`
@group(0) @binding(1) var surfT: texture_2d<f32>;
@group(0) @binding(2) var depthT: texture_2d<f32>;
@group(0) @binding(3) var volT: texture_2d<f32>;
@group(0) @binding(4) var auxT: texture_2d<f32>;
@group(0) @binding(5) var noiseT: texture_3d<f32>;
@group(0) @binding(6) var rep: sampler;
@group(0) @binding(7) var lin: sampler;
@group(0) @binding(8) var hdrOut: texture_storage_2d<rgba16float, write>;

fn volAt(uv: vec2f, dFull: f32) -> vec4f {
  let vs = vec2f(R.vw, R.vh);
  let g = uv * vs - 0.5;
  let g0 = vec2i(floor(g)); let f = g - floor(g);
  var acc = vec4f(0.0); var wsum = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1);
    let q = clamp(g0 + o, vec2i(0), vec2i(vs) - 1);
    let wb = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    let dv = textureLoad(auxT, q, 0).y;
    let rel = abs(dv - dFull) / max(min(dv, dFull), 0.05);
    let w = wb * exp(-rel * 8.0) + 1e-5;
    acc += textureLoad(volT, q, 0) * w; wsum += w;
  }
  return acc / wsum;
}

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= R.width || f32(id.y) >= R.height) { return; }
  let res = vec2f(R.width, R.height);
  let uv0 = (vec2f(id.xy) + 0.5) / res;
  var uv = uv0;
  // heat haze: refraction by hot air shifts what is seen *behind* it (surfaces), not the flames themselves;
  // a few pixels at most, rippling upward
  let hz = textureSampleLevel(auxT, lin, uv0, 0.0).x;
  if (R.hazeK > 0.0 && hz > 1e-4) {
    let q = vec3f(uv0 * vec2f(R.aspect, 1.0) * 14.0, 0.0) + vec3f(0.0, R.time * 2.2, R.time * 0.4);
    let n = textureSampleLevel(noiseT, rep, q, 0.0).xy - 0.5;
    // a fixed fraction of the frame height (max ~5 px at 1080x1920), isotropic in pixels
    uv += n * min(hz, 0.6) * R.hazeK * (16.875 / 1920.0) * vec2f(1.0 / R.aspect, 1.0);
  }
  let sdim = vec2f(textureDimensions(surfT));
  let sp = vec2i(clamp(uv * sdim, vec2f(0.0), sdim - 1.0));
  var surf = textureLoad(surfT, sp, 0).rgb;
  if (sdim.x < res.x) { surf = textureSampleLevel(surfT, lin, uv, 0.0).rgb; }
  // guard against NaN/Inf from grazing ground rays at the horizon
  if (any(surf != surf) || any(abs(surf) > vec3f(6e4))) { surf = vec3f(0.0); }
  let sp0 = vec2i(clamp(uv0 * sdim, vec2f(0.0), sdim - 1.0));
  let dFull = textureLoad(depthT, sp0, 0).x;
  let v = volAt(uv0, dFull);
  textureStore(hdrOut, id.xy, vec4f(surf * v.a + v.rgb, 1.0));
}`, 'main', 'composite');

    // -------- Temporal accumulation for real time (neighbourhood-clamped history, like TAA) --------
    this.pTemporal = computePipeline(d, /* wgsl */`
@group(0) @binding(0) var cur: texture_2d<f32>;
@group(0) @binding(1) var hist: texture_2d<f32>;
@group(0) @binding(2) var outT: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> a: vec4f;   // x: weight of the current frame, y: 1 = reset history
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  let s = vec2i(textureDimensions(cur));
  let p = vec2i(id.xy);
  if (any(p >= s)) { return; }
  let c = textureLoad(cur, p, 0);
  var mn = c; var mx = c;
  for (var k = 0; k < 9; k++) {
    let q = clamp(p + vec2i(k % 3 - 1, k / 3 - 1), vec2i(0), s - 1);
    let v = textureLoad(cur, q, 0); mn = min(mn, v); mx = max(mx, v);
  }
  let h = clamp(textureLoad(hist, p, 0), mn, mx);
  textureStore(outT, p, select(mix(h, c, a.x), c, a.y > 0.5));
}`, 'main', 'temporal');

    // -------- Accumulate (shutter / supersampling) --------
    this.pAccum = computePipeline(d, /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var acc: texture_storage_2d<rgba32float, read_write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  let s = textureDimensions(src);
  if (id.x >= s.x || id.y >= s.y) { return; }
  textureStore(acc, id.xy, textureLoad(acc, id.xy) + textureLoad(src, id.xy, 0));
}`, 'main', 'accum');
    this.pAccumResolve = computePipeline(d, /* wgsl */`
@group(0) @binding(0) var acc: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> n: vec4f;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  let s = textureDimensions(acc);
  if (id.x >= s.x || id.y >= s.y) { return; }
  textureStore(dst, id.xy, textureLoad(acc, id.xy) / n.x);
  textureStore(acc, id.xy, vec4f(0.0));
}`, 'main', 'accumResolve');

    // -------- Bloom (dual filter) --------
    this.pDown = computePipeline(d, /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> o: vec4f; // x: first level (karis)
fn s(uv: vec2f) -> vec3f { return textureSampleLevel(src, lin, uv, 0.0).rgb; }
fn karis(c: vec3f) -> f32 { return 1.0 / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722)) * 0.25); }
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  let ds = vec2f(textureDimensions(dst));
  if (f32(id.x) >= ds.x || f32(id.y) >= ds.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / ds;
  let t = 1.0 / vec2f(textureDimensions(src));
  let a = s(uv + t * vec2f(-2, -2)); let b = s(uv + t * vec2f(0, -2)); let c = s(uv + t * vec2f(2, -2));
  let dd = s(uv + t * vec2f(-2, 0)); let e = s(uv); let f = s(uv + t * vec2f(2, 0));
  let g = s(uv + t * vec2f(-2, 2)); let h = s(uv + t * vec2f(0, 2)); let i = s(uv + t * vec2f(2, 2));
  let j = s(uv + t * vec2f(-1, -1)); let k = s(uv + t * vec2f(1, -1)); let l = s(uv + t * vec2f(-1, 1)); let m = s(uv + t * vec2f(1, 1));
  var col = vec3f(0.0);
  if (o.x > 0.5) {
    let g0 = (a + b + dd + e) * 0.25; let g1 = (b + c + e + f) * 0.25; let g2 = (dd + e + g + h) * 0.25; let g3 = (e + f + h + i) * 0.25; let g4 = (j + k + l + m) * 0.25;
    let w0 = karis(g0) * 0.125; let w1 = karis(g1) * 0.125; let w2 = karis(g2) * 0.125; let w3 = karis(g3) * 0.125; let w4 = karis(g4) * 0.5;
    col = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    col = e * 0.125 + (a + c + g + i) * 0.03125 + (b + dd + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  textureStore(dst, id.xy, vec4f(max(col, vec3f(0.0)), 1.0));
}`, 'main', 'bloomDown');
    this.pUp = computePipeline(d, /* wgsl */`
@group(0) @binding(0) var low: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var cur: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> o: vec4f; // x: radius
fn s(uv: vec2f) -> vec3f { return textureSampleLevel(low, lin, uv, 0.0).rgb; }
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
  let ds = vec2f(textureDimensions(dst));
  if (f32(id.x) >= ds.x || f32(id.y) >= ds.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / ds;
  let t = o.x / vec2f(textureDimensions(low));
  var c = s(uv) * 4.0;
  c += (s(uv + vec2f(-t.x, 0)) + s(uv + vec2f(t.x, 0)) + s(uv + vec2f(0, -t.y)) + s(uv + vec2f(0, t.y))) * 2.0;
  c += s(uv + vec2f(-t.x, -t.y)) + s(uv + vec2f(t.x, -t.y)) + s(uv + vec2f(-t.x, t.y)) + s(uv + vec2f(t.x, t.y));
  c /= 16.0;
  let here = textureLoad(cur, vec2i(id.xy), 0).rgb;
  textureStore(dst, id.xy, vec4f(here + c, 1.0));
}`, 'main', 'bloomUp');

    // -------- Final: tonemap + grade (render pipeline into any target format) --------
    const finalCode = H + /* wgsl */`
@group(0) @binding(1) var hdrT: texture_2d<f32>;
@group(0) @binding(2) var bloomT: texture_2d<f32>;
@group(0) @binding(3) var lin: sampler;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO; o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0); o.uv = vec2f(p.x, 1.0 - p.y); return o;
}
// AgX (Blender) base, polynomial sigmoid fit.
fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x; let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn agx(c: vec3f) -> vec3f {
  let m = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                  0.0784335999999992, 0.878468636469772, 0.0784336,
                  0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let mi = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                   -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                   -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let minEv = -12.47393; let maxEv = 4.026069;
  var v = m * max(c, vec3f(1e-10));
  v = clamp(log2(v), vec3f(minEv), vec3f(maxEv));
  v = (v - minEv) / (maxEv - minEv);
  v = agxContrast(v);
  v = mi * v;
  return clamp(v, vec3f(0.0), vec3f(1.0));
}
// Per-channel filmic (Hable-ish) with natural per-channel clipping, like a camera sensor.
fn filmic(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  let y = clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
  return pow(y, vec3f(1.0 / 2.2));
}
// Camera-like response: slight channel crosstalk (path to white), per-channel soft clip, sRGB encode.
// Channels saturate individually (R first), reproducing orange -> amber -> yellow -> cream in bright cores.
fn srgbEnc(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308)); }
fn camera(c: vec3f) -> vec3f {
  let k = R.inset;
  let kn = R.knee;
  // channel crosstalk only on the part above the knee (bright cores go to white; mid-tones/embers stay saturated)
  let cc = max(c, vec3f(0.0));
  let ex = max(cc - kn, vec3f(0.0));
  // symmetric spill: bright cores roll off to cream/white like the measured photo ramp (not lemon)
  let m = mat3x3f(1.0 - 2.0 * k, k, k, k, 1.0 - 2.0 * k, k, k, k, 1.0 - 2.0 * k);
  var v = cc - ex + m * ex;
  // soft knee: linear below the knee, exponential roll-off to 1 above it
  v = select(kn + (1.0 - kn) * (1.0 - exp(-(v - kn) / (1.0 - kn))), v, v < vec3f(kn));
  return srgbEnc(clamp(v, vec3f(0.0), vec3f(1.0)));
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let res = vec2f(R.width, R.height);
  // lateral chromatic aberration: R and B magnified slightly differently
  let cuv = i.uv - 0.5;
  var c = vec3f(textureSampleLevel(hdrT, lin, 0.5 + cuv * (1.0 + R.chroma), 0.0).r,
                textureSampleLevel(hdrT, lin, i.uv, 0.0).g,
                textureSampleLevel(hdrT, lin, 0.5 + cuv * (1.0 - R.chroma), 0.0).b);
  var bl = textureSampleLevel(bloomT, lin, i.uv, 0.0).rgb;
  if (any(c != c)) { c = vec3f(0.0); }
  if (any(bl != bl)) { bl = vec3f(0.0); }
  c = mix(c, bl / 7.0, R.bloomK);   // the upsample chain sums 7 levels: normalise to keep energy
  // natural vignetting: cos^4 of the field angle (partially applied)
  let ndc = cuv * 2.0 * vec2f(R.aspect, 1.0) * R.tanFov;
  let cos2 = 1.0 / (1.0 + dot(ndc, ndc));
  c *= mix(1.0, cos2 * cos2, R.vignette);
  c *= R.exposure;
  c += R.lift * vec3f(0.6, 0.7, 1.0) * 0.001;
  var o = vec3f(0.0);
  if (R.tonemap < 0.5) { o = agx(c); } else if (R.tonemap < 1.5) { o = filmic(c); } else { o = camera(c); }
  let lum = dot(o, vec3f(0.2126, 0.7152, 0.0722));
  o = mix(vec3f(lum), o, R.saturation);
  o = clamp((o - 0.5) * R.contrast + 0.5, vec3f(0.0), vec3f(1.0));
  // film grain + dither
  let px = i.pos.xy;
  // grain: ~1.5 px clumps (value noise on a 1.5 px lattice), mid-tone weighted 4y(1-y), partly per channel;
  // plus a separate TPDF dither
  let gp = px / 1.5;
  let g0 = vec2i(floor(gp)); let gf = fract(gp); let gu = gf * gf * (3.0 - 2.0 * gf);
  var gn = vec3f(0.0);
  for (var k = 0; k < 4; k++) {
    let o2 = vec2i(k & 1, k >> 1);
    let hh = hash3f(vec3i(g0 + o2, i32(R.frame))) - 0.5;
    let wv = select(1.0 - gu, gu, vec2<bool>(o2.x == 1, o2.y == 1));
    gn += (vec3f(hh.x) * 0.75 + hh * 0.25) * (wv.x * wv.y);   // mostly luminance grain, some per channel
  }
  let gw = 4.0 * lum * (1.0 - lum) + 0.15;
  o += gn * R.grain * 3.0 * gw;
  let h = hash3f(vec3i(vec2i(px), i32(R.frame) + 911));
  o += (h.x + h.y - 1.0) / 255.0;
  return vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), 1.0);
}`;
    const fmod = d.createShaderModule({ label: 'final', code: finalCode });
    fmod.getCompilationInfo().then(info => { for (const m of info.messages) if (m.type === 'error') console.error(`[final] ${m.lineNum}:${m.linePos} ${m.message}`); });
    this.finalPipes = {};
    this.finalModule = fmod;
    const finalBgl = layoutFromWGSL(d, finalCode, GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX);
    this.getFinalPipe = (format) => this.finalPipes[format] ||= Object.assign(d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [finalBgl] }), vertex: { module: fmod, entryPoint: 'vs' },
      fragment: { module: fmod, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' },
    }), { bgl: finalBgl });
    this.getFinalPipe(this.format);
  }

  // surfScale < 1 renders the SDF surfaces at reduced resolution (bilinear upsample in the composite).
  resize(w, h, volScale = 0.5, surfScale = this.surfScale ?? 1) {
    const d = this.device;
    w = Math.max(16, Math.floor(w)); h = Math.max(16, Math.floor(h));
    if (this.w === w && this.h === h && this.volScale === volScale && this._surfScale === surfScale) return;
    this.w = w; this.h = h; this.volScale = volScale; this._surfScale = surfScale;
    this.vw = Math.max(8, Math.round(w * volScale)); this.vh = Math.max(8, Math.round(h * volScale));
    this.sw = Math.max(8, Math.round(w * surfScale)); this.sh = Math.max(8, Math.round(h * surfScale));
    for (const t of this.targets || []) t.destroy();
    const RA = GPUTextureUsage.RENDER_ATTACHMENT;
    this.surf = tex2D(d, [this.sw, this.sh], 'rgba16float', 'surf');
    this.depth = tex2D(d, [this.sw, this.sh], 'r32float', 'depth');
    this.vol = tex2D(d, [this.vw, this.vh], 'rgba16float', 'vol');
    this.aux = tex2D(d, [this.vw, this.vh], 'rg32float', 'aux');
    this.hdr = tex2D(d, [w, h], 'rgba16float', 'hdr', RA);
    this.hdrAvg = tex2D(d, [w, h], 'rgba16float', 'hdrAvg', RA);
    this.hist = tex2D(d, [w, h], 'rgba16float', 'hist');
    this.histReset = true;
    this.acc = tex2D(d, [w, h], 'rgba32float', 'acc');
    this.blur = [];
    let bw = w, bh = h;
    for (let i = 0; i < 7; i++) {
      bw = Math.max(1, Math.ceil(bw / 2)); bh = Math.max(1, Math.ceil(bh / 2));
      this.blur.push({ down: tex2D(d, [bw, bh], 'rgba16float', `bd${i}`), up: tex2D(d, [bw, bh], 'rgba16float', `bu${i}`), w: bw, h: bh });
    }
    this.targets = [this.surf, this.depth, this.vol, this.aux, this.hdr, this.hdrAvg, this.hist, this.acc, ...this.blur.flatMap(b => [b.down, b.up])];
    this.makeBindGroups();
  }

  makeBindGroups() {
    const d = this.device, U = this.U.buffer, sim = this.sim;
    this.bgLights = bindGroup(d, this.pLights, [U, sim.S, this.bbTex, this.lightBuf, this.lightState]);
    this.bgSurface = bindGroup(d, this.pSurface, [this.Usurf.buffer, this.logBuf, this.stoneBuf, this.noiseTex, this.rep, this.bbTex, this.lightBuf, this.surf, this.depth, sim.S, this.lin]);
    this.bgVolume = bindGroup(d, this.pVolume, [U, sim.S, this.lin, this.noiseTex, this.rep, this.bbTex, this.lightBuf, this.depth, this.vol, this.aux, sim.S2, this.curlTex]);
    if (!this.sSharp) this.sSharp = tex3D(d, sim.res, 'rgba16float', 'sSharp');
    this.bgSharpen = bindGroup(d, this.pSharpen, [U, sim.S, this.sSharp]);
    this.bgVolumeSharp = bindGroup(d, this.pVolume, [U, this.sSharp, this.lin, this.noiseTex, this.rep, this.bbTex, this.lightBuf, this.depth, this.vol, this.aux, sim.S2, this.curlTex]);
    this.bgComposite = bindGroup(d, this.pComposite, [U, this.surf, this.depth, this.vol, this.aux, this.noiseTex, this.rep, this.lin, this.hdr]);
    const ub = (v) => { const b = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); d.queue.writeBuffer(b, 0, new Float32Array(v)); return b; };
    this.accN = ub([1, 0, 0, 0]);
    this.bgAccum = bindGroup(d, this.pAccum, [this.hdr, this.acc]);
    this.bgResolve = bindGroup(d, this.pAccumResolve, [this.acc, this.hdrAvg, this.accN]);
    this.bgDown = (src) => this.blur.map((b, i) => bindGroup(d, this.pDown, [i === 0 ? src : this.blur[i - 1].down, this.lin, b.down, ub([i === 0 ? 1 : 0, 0, 0, 0])]));
    this.bgDownHdr = this.bgDown(this.hdr);
    this.bgDownAvg = this.bgDown(this.hdrAvg);
    const n = this.blur.length;
    this.bgUp = [];
    for (let i = n - 2; i >= 0; i--) {
      const low = i === n - 2 ? this.blur[n - 1].down : this.blur[i + 1].up;
      this.bgUp.push({ bg: bindGroup(d, this.pUp, [low, this.lin, this.blur[i].down, this.blur[i].up, ub([1.0, 0, 0, 0])]), w: this.blur[i].w, h: this.blur[i].h });
    }
    this.bloomOut = this.blur[0].up;
    const fp = this.getFinalPipe(this.format);
    this.bgFinalHdr = bindGroup(d, fp, [U, this.hdr, this.bloomOut, this.lin]);
    this.bgFinalAvg = bindGroup(d, fp, [U, this.hdrAvg, this.bloomOut, this.lin]);
    this.bgFinalCache = new Map();
  }

  setCamera(cam) { this.cam = cam; }

  uploadParams(time, frame, jitter = [0, 0]) {
    const c = this.cam, sim = this.sim;
    // light smoothing step: sim time elapsed since the last render (-1 = reset the filter state)
    const lightDt = this.lastLightTime === null || time < this.lastLightTime || time - this.lastLightTime > 0.5 ? -1 : time - this.lastLightTime;
    this.lastLightTime = time;
    this.U.set({
      ...this.params, lightDt, width: this.w, height: this.h, vw: this.vw, vh: this.vh, time, frame,
      nLogs: this.nLogs, nStones: this.nStones, nLights: NLIGHTS,
      camX: c.pos[0], camY: c.pos[1], camZ: c.pos[2], fwdX: c.fwd[0], fwdY: c.fwd[1], fwdZ: c.fwd[2],
      rgtX: c.right[0], rgtY: c.right[1], rgtZ: c.right[2], upX: c.up[0], upY: c.up[1], upZ: c.up[2],
      tanFov: Math.tan(c.fovY / 2), aspect: this.w / this.h, jitX: jitter[0], jitY: jitter[1],
      ox: sim.origin[0], oy: sim.origin[1], oz: sim.origin[2], dx: sim.dx, sx: sim.size[0], sy: sim.size[1], sz: sim.size[2],
      nx: sim.res[0], ny: sim.res[1], nz: sim.res[2],
      bbTmin: this.bb.Tmin, bbTmax: this.bb.Tmax, bbN: this.bb.n,
    }).upload();
    // the surface pass sees its own (possibly reduced) resolution
    this.Usurf.data.set(this.U.data);
    const sj = this.sw / this.w;
    const su = { width: this.sw, height: this.sh, jitX: jitter[0] * sj, jitY: jitter[1] * sj };
    const p = this.params;
    if (p.focus > 0) {
      // Thin lens as an equivalent pinhole: move the eye onto the aperture sample and shear the frustum so the
      // focal plane stays fixed (a pixel offset of -lens/F in tangent space).
      const tf = Math.tan(c.fovY / 2), asp = this.w / this.h;
      su.camX = c.pos[0] + c.right[0] * p.lensX + c.up[0] * p.lensY;
      su.camY = c.pos[1] + c.right[1] * p.lensX + c.up[1] * p.lensY;
      su.camZ = c.pos[2] + c.right[2] * p.lensX + c.up[2] * p.lensY;
      su.jitX += -(p.lensX / p.focus) / (tf * asp) * this.sw * 0.5;
      su.jitY += (p.lensY / p.focus) / tf * this.sh * 0.5;
    }
    this.Usurf.set(su).upload();
  }

  // Renders the scene into this.hdr (linear HDR).
  renderHDR(enc, prof) {
    const W = this.w, H = this.h;
    let pass = enc.beginComputePass({ timestampWrites: prof?.pass('lights') });
    pass.setPipeline(this.pLights); pass.setBindGroup(0, this.bgLights); pass.dispatchWorkgroups(NLIGHTS);
    pass.end();
    pass = enc.beginComputePass({ timestampWrites: prof?.pass('surface') });
    pass.setPipeline(this.pSurface); pass.setBindGroup(0, this.bgSurface); pass.dispatchWorkgroups(Math.ceil(this.sw / 8), Math.ceil(this.sh / 8));
    pass.end();
    // prefilter once per simulation state (several jittered renders of one sub-step share it)
    const sharp = this.params.flameSharp > 0 && this.params.cubic > 0.5;
    if (sharp) {
      const stamp = `${this.sim.frame}:${this.sim.time}:${this.params.flameSharp}`;
      if (stamp !== this.sharpStamp) {
        this.sharpStamp = stamp;
        const [x, y, z] = this.sim.res;
        pass = enc.beginComputePass();
        pass.setPipeline(this.pSharpen); pass.setBindGroup(0, this.bgSharpen); pass.dispatchWorkgroups(Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4));
        pass.end();
      }
    }
    pass = enc.beginComputePass({ timestampWrites: prof?.pass('volume') });
    pass.setPipeline(this.pVolume); pass.setBindGroup(0, sharp ? this.bgVolumeSharp : this.bgVolume); pass.dispatchWorkgroups(Math.ceil(this.vw / 8), Math.ceil(this.vh / 8));
    pass.end();
    pass = enc.beginComputePass({ timestampWrites: prof?.pass('composite') });
    pass.setPipeline(this.pComposite); pass.setBindGroup(0, this.bgComposite); pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    pass.end();
  }

  // Total visible light output of the flames (sum of the extracted lights' luminance) and its centroid.
  // Non-blocking: returns the last completed reading; used to drive audio loudness and capture stats.
  readLights(enc) {
    if (!this.lightRead) { this.lightRead = this.device.createBuffer({ size: NLIGHTS * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); this.lightStat = { L: 0, y: 0.3 }; this.lightBusy = false; }
    if (this.lightBusy) return false;
    enc.copyBufferToBuffer(this.lightBuf, 0, this.lightRead, 0, NLIGHTS * 32);
    this.lightPending = true;
    this.lightPendingT = performance.now();
    return true;
  }
  async collectLights() {
    if (!this.lightPending || this.lightBusy) return this.lightStat;
    this.lightBusy = true; this.lightPending = false;
    await this.lightRead.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(this.lightRead.getMappedRange().slice(0));
    this.lightRead.unmap();
    let L = 0, y = 0;
    for (let i = 0; i < NLIGHTS; i++) {
      const l = 0.2126 * f[i * 8 + 4] + 0.7152 * f[i * 8 + 5] + 0.0722 * f[i * 8 + 6];
      L += l; y += l * f[i * 8 + 1];
    }
    this.lightSeq = (this.lightSeq | 0) + 1;
    this.lightStat = { L, y: L > 0 ? y / L : 0.3, t: this.lightPendingT, seq: this.lightSeq };
    this.lightBusy = false;
    return this.lightStat;
  }

  // Real-time: blend this frame with the clamped history into hdrAvg (then bloom/final from hdrAvg).
  temporal(enc, weight = 0.35) {
    const d = this.device;
    if (!this.taU) this.taU = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (this._taHdr !== this.hdr) { this._taHdr = this.hdr; this.bgTemporal = bindGroup(d, this.pTemporal, [this.hdr, this.hist, this.hdrAvg, this.taU]); this.histReset = true; }
    d.queue.writeBuffer(this.taU, 0, new Float32Array([weight, this.histReset ? 1 : 0, 0, 0]));
    this.histReset = false;
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pTemporal); pass.setBindGroup(0, this.bgTemporal); pass.dispatchWorkgroups(Math.ceil(this.w / 8), Math.ceil(this.h / 8));
    pass.end();
    enc.copyTextureToTexture({ texture: this.hdrAvg }, { texture: this.hist }, [this.w, this.h]);
  }

  accumulate(enc) {
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pAccum); pass.setBindGroup(0, this.bgAccum); pass.dispatchWorkgroups(Math.ceil(this.w / 8), Math.ceil(this.h / 8));
    pass.end();
  }
  resolveAccum(enc, n) {
    this.device.queue.writeBuffer(this.accN, 0, new Float32Array([n, 0, 0, 0]));
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pAccumResolve); pass.setBindGroup(0, this.bgResolve); pass.dispatchWorkgroups(Math.ceil(this.w / 8), Math.ceil(this.h / 8));
    pass.end();
  }

  bloom(enc, fromAvg, prof) {
    const pass = enc.beginComputePass({ timestampWrites: prof?.pass('bloom') });
    const downs = fromAvg ? this.bgDownAvg : this.bgDownHdr;
    this.blur.forEach((b, i) => { pass.setPipeline(this.pDown); pass.setBindGroup(0, downs[i]); pass.dispatchWorkgroups(Math.ceil(b.w / 8), Math.ceil(b.h / 8)); });
    for (const u of this.bgUp) { pass.setPipeline(this.pUp); pass.setBindGroup(0, u.bg); pass.dispatchWorkgroups(Math.ceil(u.w / 8), Math.ceil(u.h / 8)); }
    pass.end();
  }

  final(enc, view, format, fromAvg) {
    const fp = this.getFinalPipe(format);
    let bg = fromAvg ? this.bgFinalAvg : this.bgFinalHdr;
    if (format !== this.format) {
      const key = format + (fromAvg ? 'a' : 'h');
      if (!this.bgFinalCache.has(key)) this.bgFinalCache.set(key, bindGroup(this.device, fp, [this.U.buffer, fromAvg ? this.hdrAvg : this.hdr, this.bloomOut, this.lin]));
      bg = this.bgFinalCache.get(key);
    }
    const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(fp); pass.setBindGroup(0, bg); pass.draw(3);
    pass.end();
  }
}
