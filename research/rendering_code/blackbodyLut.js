// blackbodyLut.js — procedural blackbody -> linear-sRGB LUT (no assets).
// CIE 1931 2° colour matching functions, 380–780 nm @ 10 nm (CVRL table, 4 sig. digits).
// 10 nm integration reproduces 1 nm-integrated chromaticity to < 1e-4 in x,y for 800–6500 K.
const CIE10 = [
  [0.001368,0.000039,0.00645],[0.004243,0.00012,0.02005],[0.01431,0.000396,0.06785],[0.04351,0.00121,0.2074],
  [0.1344,0.004,0.6456],[0.2839,0.0116,1.386],[0.3483,0.023,1.747],[0.3362,0.038,1.772],[0.2908,0.06,1.669],
  [0.1954,0.09098,1.288],[0.09564,0.139,0.813],[0.03201,0.208,0.4652],[0.0049,0.323,0.272],[0.0093,0.503,0.1582],
  [0.06327,0.71,0.07825],[0.1655,0.862,0.04216],[0.2904,0.954,0.0203],[0.4334,0.995,0.00875],[0.5945,0.995,0.0039],
  [0.7621,0.952,0.0021],[0.9163,0.87,0.00165],[1.026,0.757,0.0011],[1.062,0.631,0.0008],[1.003,0.503,0.00034],
  [0.8544,0.381,0.00019],[0.6424,0.265,0.00005],[0.4479,0.175,0.00002],[0.2835,0.107,0],[0.1649,0.061,0],
  [0.0874,0.032,0],[0.04677,0.017,0],[0.0227,0.00821,0],[0.01136,0.004102,0],[0.00579,0.002091,0],
  [0.002899,0.001047,0],[0.00144,0.00052,0],[0.0006901,0.0002492,0],[0.0003323,0.00012,0],[0.0001662,0.00006,0],
  [0.00008308,0.00003,0],[0.00004151,0.00001499,0]];
const XYZ_TO_SRGB = [[3.2404542,-1.5371385,-0.4985314],[-0.9692660,1.8760108,0.0415560],[0.0556434,-0.2040259,1.0572252]];
const BRADFORD = [[0.8951,0.2664,-0.1614],[-0.7502,1.7135,0.0367],[0.0389,-0.0685,1.0296]];
const BRADFORD_INV = [[0.9869929,-0.1470543,0.1599627],[0.4323053,0.5183603,0.0492912],[-0.0085287,0.0400428,0.9684867]];
const D65 = [0.95047, 1.0, 1.08883];
const mv = (m, v) => m.map(r => r[0]*v[0] + r[1]*v[1] + r[2]*v[2]);

/** Spectral radiance of a blackbody integrated against CIE 1931 -> XYZ (Y in cd/m^2). */
export function blackbodyXYZ(T, emissivity = null) {
  const c1 = 1.191042972e-16;      // 2hc^2  [W m^2 sr^-1]
  const c2 = 1.438776877e-2;       // hc/k   [m K]
  let X = 0, Y = 0, Z = 0;
  for (let i = 0; i < 41; i++) {
    const nm = 380 + 10 * i, l = nm * 1e-9;
    let B = c1 / (l ** 5 * Math.expm1(c2 / (l * T))) * 10e-9;   // W m^-2 sr^-1 per 10 nm bin
    if (emissivity) B *= emissivity(nm);
    X += B * CIE10[i][0]; Y += B * CIE10[i][1]; Z += B * CIE10[i][2];
  }
  return [683 * X, 683 * Y, 683 * Z];
}

/** Bradford chromatic adaptation from the white of a `wbKelvin` blackbody to D65 (i.e. a camera WB setting). */
function adaptToD65(xyz, wbKelvin) {
  const w = blackbodyXYZ(wbKelvin); const ws = [w[0]/w[1], 1, w[2]/w[1]];
  const S = mv(BRADFORD, ws), D = mv(BRADFORD, D65), V = mv(BRADFORD, xyz);
  return mv(BRADFORD_INV, [V[0]*D[0]/S[0], V[1]*D[1]/S[1], V[2]*D[2]/S[2]]);
}

/**
 * Build a LUT over T in [tMin, tMax]. Per texel (RGBA float):
 *   rgb = linear-sRGB chromaticity scaled to luminance 1 (negatives clamped, then re-normalised)
 *   a   = log2(Y(T) / Y(tRef))  — physical luminance, relative to tRef (default 1500 K)
 * Shader: emission = intensity * rgb * exp2(contrast * a)   (contrast 1 = physical, 0.4–0.7 = production look)
 */
export function buildBlackbodyLUT({ tMin = 500, tMax = 4000, size = 256, tRef = 1500, wbKelvin = null, sootAlpha = 0 } = {}) {
  const emis = sootAlpha > 0 ? (nm => Math.pow(nm / 600, -sootAlpha)) : null; // optically-thin soot: eps ~ lambda^-alpha
  const Yref = blackbodyXYZ(tRef, emis)[1];
  const out = new Float32Array(size * 4);
  for (let i = 0; i < size; i++) {
    const T = tMin + (tMax - tMin) * i / (size - 1);
    let xyz = blackbodyXYZ(T, emis);
    const Yphys = xyz[1];
    if (wbKelvin) xyz = adaptToD65(xyz, wbKelvin);
    const rgb = mv(XYZ_TO_SRGB, xyz).map(v => Math.max(v, 0));
    const Y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    out[i*4+0] = rgb[0] / Y; out[i*4+1] = rgb[1] / Y; out[i*4+2] = rgb[2] / Y;
    out[i*4+3] = Math.log2(Math.max(Yphys, 1e-30) / Yref);       // fp16-safe range (about -40..+15)
  }
  return out;
}

/** float32 -> IEEE half (for an rgba16float texture; rgba32float is not filterable without 'float32-filterable'). */
export function toHalf(f) {
  const b = new DataView(new ArrayBuffer(4)); b.setFloat32(0, f);
  const x = b.getUint32(0), s = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 0xff) - 127 + 15, m = x & 0x7fffff;
  if (e <= 0) { if (e < -10) return s; m = (m | 0x800000) >> (1 - e); return s | ((m + 0x1000) >> 13); }
  if (e >= 31) return s | 0x7c00;
  const hv = s | (e << 10) | ((m + 0x1000) >> 13); return hv;   // rounding may carry into exponent: fine
}

/** Create the 256x1 rgba16float texture (2D so it can be sampled with textureSampleLevel in compute shaders). */
export function createBlackbodyTexture(device, opts = {}) {
  const size = opts.size ?? 256;
  const lut = buildBlackbodyLUT({ ...opts, size });
  const half = new Uint16Array(lut.length); for (let i = 0; i < lut.length; i++) half[i] = toHalf(lut[i]);
  const tex = device.createTexture({ size: [size, 1], format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, half, { bytesPerRow: size * 8 }, [size, 1]);
  return tex;
}
