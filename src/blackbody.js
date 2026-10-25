// Blackbody (Planck) spectrum integrated against the CIE 1931 2° observer (CVRL table, 10 nm)
// -> XYZ -> linear Rec.709/sRGB, with an optional camera white balance (Bradford).

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
const h = 6.62607015e-34, c = 2.99792458e8, kB = 1.380649e-23;

function g(x, mu, s1, s2) { const t = (x - mu) / (x < mu ? s1 : s2); return Math.exp(-0.5 * t * t); }
export function cie(lnm) {
  const x = 1.056 * g(lnm, 599.8, 37.9, 31.0) + 0.362 * g(lnm, 442.0, 16.0, 26.7) - 0.065 * g(lnm, 501.1, 20.4, 26.2);
  const y = 0.821 * g(lnm, 568.8, 46.9, 40.5) + 0.286 * g(lnm, 530.9, 16.3, 31.1);
  const z = 1.217 * g(lnm, 437.0, 11.8, 36.0) + 0.681 * g(lnm, 459.0, 26.0, 13.8);
  return [x, y, z];
}
export function planck(lnm, T) {
  const l = lnm * 1e-9;
  return (2 * h * c * c) / (l ** 5 * (Math.exp((h * c) / (l * kB * T)) - 1));
}
// soot: spectral emissivity ~ lambda^-1.39 (Hottel-Broughton), normalised at 550 nm
export function blackbodyXYZ(T, soot = false) {
  let X = 0, Y = 0, Z = 0;
  for (let l = 380; l <= 780; l += 2) {
    const p = planck(l, T) * (soot ? Math.pow(l / 550, -1.39) : 1);
    const [x, y, z] = cie(l);
    X += p * x; Y += p * y; Z += p * z;
  }
  return [X, Y, Z];
}
const XYZ2RGB = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.9692660, 1.8760108, 0.0415560],
  [0.0556434, -0.2040259, 1.0572252],
];
const XYZ2LMS = [ // Bradford
  [0.8951, 0.2664, -0.1614],
  [-0.7502, 1.7135, 0.0367],
  [0.0389, -0.0685, 1.0296],
];
const mul = (M, v) => M.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
function inv3(m) {
  const [a, b, c2] = m[0], [d, e, f] = m[1], [g2, h2, i] = m[2];
  const A = e * i - f * h2, B = -(d * i - f * g2), C = d * h2 - e * g2;
  const det = a * A + b * B + c2 * C;
  return [
    [A / det, -(b * i - c2 * h2) / det, (b * f - c2 * e) / det],
    [B / det, (a * i - c2 * g2) / det, -(a * f - c2 * d) / det],
    [C / det, -(a * h2 - b * g2) / det, (a * e - b * d) / det],
  ];
}
const LMS2XYZ = inv3(XYZ2LMS);
const D65 = [0.95047, 1.0, 1.08883];

// Returns a function XYZ->linear sRGB that adapts the white of a blackbody at Twb to D65 (camera WB).
export function whiteBalance(Twb) {
  if (!Twb) return xyz => mul(XYZ2RGB, xyz);
  const w = blackbodyXYZ(Twb);
  const wn = w.map(v => v / w[1]);
  const src = mul(XYZ2LMS, wn), dst = mul(XYZ2LMS, D65);
  return xyz => {
    const lms = mul(XYZ2LMS, xyz).map((v, i) => v * dst[i] / src[i]);
    return mul(XYZ2RGB, mul(LMS2XYZ, lms));
  };
}

// 2-row LUT over [Tmin, Tmax] K: row 0 blackbody, row 1 soot (grey-ish, slightly bluer).
// Values are radiance relative to the luminance of a 1500 K blackbody.
export function blackbodyLUT({ n = 1024, Tmin = 400, Tmax = 3000, Twb = 5000 } = {}) {
  const toRGB = whiteBalance(Twb);
  const ref = blackbodyXYZ(1500)[1];
  const data = new Float32Array(n * 4 * 2);
  for (let row = 0; row < 2; row++) {
    for (let i = 0; i < n; i++) {
      const T = Tmin + (Tmax - Tmin) * i / (n - 1);
      const xyz = blackbodyXYZ(T, row === 1).map(v => v / ref);
      const rgb = toRGB(xyz).map(v => Math.max(0, v));
      data.set([rgb[0], rgb[1], rgb[2], xyz[1]], (row * n + i) * 4);
    }
  }
  return { data, n, Tmin, Tmax };
}
