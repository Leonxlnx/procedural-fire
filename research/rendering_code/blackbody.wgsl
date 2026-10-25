// Blackbody LUT: 256x1 rgba16float over T in [500, 4000] K.
//   rgb = linear-sRGB chroma normalised to luminance 1, a = log2(Y(T)/Y(1500 K)).
// contrast = 1 -> physical luminance ratio (~T^16 around 1500 K); 0.4..0.7 -> production look.
fn blackbody(Tk: f32, contrast: f32) -> vec3f {
  let u = clamp((Tk - 500.0) / 3500.0, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0;
  let e = textureSampleLevel(bbLut, linClamp, vec2f(u, 0.5), 0.0);
  return e.rgb * exp2(contrast * e.a);
}
