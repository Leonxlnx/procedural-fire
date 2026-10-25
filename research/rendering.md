I couldn't create `D:\fire\research\rendering.md`: the tool environment blocks subagents from writing `.md` report files and says to return the content as text. The full report is below; the summary comes first.

**Summary**
1. **Emission model:** fire's visible light is glowing soot, so use emission = σa·R·B(T) (absorption times blackbody radiance, Kirchhoff's law). Flames can then never get brighter than a blackbody: thick flame bodies look solid, edges stay translucent, and red tips come out on their own.
2. **Colour lookup:** temperature → colour comes from a 256×1 table built at startup from Planck's law and the official CIE 1931 tables, storing chroma plus log2 brightness. Don't use the popular analytic colour-matching fit: in the far red it is off by about 2×, which makes 1000 K embers too orange.
3. **Brightness range:** physical brightness rises about as T^16 near 1500 K, so 1200→1800 K is 9.7 stops. Compress it with a contrast exponent γ ≈ 0.55 in real time, the same idea as Arnold's `blackbody_contrast` and Houdini's Burn/Adaptation controls.
4. **Temperatures:** map tips to 900–1100 K, body 1250–1550 K, core 1600–1900 K, embers 900–1250 K. Nothing glows visibly below about 800 K.
5. **White balance and exposure:** camera white balance matters more than anything else; use 3500–4000 K. Put the flame body 3–5 stops above mid-grey; a flame is roughly 100× brighter than the ground it lights at 1 m.
6. **Tone mapping:** I ran each candidate on blackbody colours. Khronos PBR Neutral turns fire highlights pink and plain AgX turns them salmon, so both are rejected. ACES-Hill and my "AgX-Fire" variant both go orange → yellow → white like a real camera. AgX-Fire is the default; it holds about one more stop of detail inside the bright flame. Its output matches the reference values on the GPU.
7. **Smoke lit by the fire:** sum the emission into 8³ virtual point lights, build a 32³ light volume with 4-tap shadows, and use Henyey-Greenstein scattering with g ≈ 0.6 and albedo 0.6–0.85.
8. **Surfaces:** light logs and ground from 8 clustered lights using the same units, so the flicker comes for free. Glowing cracks use Voronoi noise and the same colour table, so embers and flames match.
9. **Ray marching on the 780M:** half resolution, 0.75–1 voxel steps, jittered starts, tight bounding box and empty-brick skipping, stop at transmittance 0.003. That should cost about 3–6 ms.
10. **Heat haze:** compute it physically from the hot-air refractive index; it comes to 1–5 px at 1080p.
11. **Post chain:** energy-conserving bloom (strength 0.04, 6–7 levels), slight chromatic aberration, cos⁴ vignette, mid-tone grain at 0.02, then dither.
12. **Video render:** 30 fps with a 180° shutter (1/60 s), 16–32 accumulated subframes per frame with motion blur and depth of field, tone-mapped once after accumulating, then x264 at CRF 14 with BT.709 tags.

All shader files compile and pass pipeline validation in Chrome WebGPU on the 780M. The colour-table generator was run in Node, and the tone mapper's GPU output matches the JS reference exactly.

Files are in `D:\fire\research`:
- `rendering_tonemap_swatches.png`
- `rendering_code\` (contains the files below)
  - `blackbodyLut.js`
  - `common.wgsl`
  - `blackbody.wgsl`
  - `march.wgsl`
  - `firelight.wgsl`
  - `fluence.wgsl`
  - `surface.wgsl`
  - `bloom.wgsl`
  - `composite.wgsl`

---

# Photoreal rendering of volumetric fire: research notes and implementation guide

**Scope.** This covers how to make a procedural, no-asset WebGPU fire look photographic on the 780M in real time, and how to render a 20 s cinematic offline. Simulation is covered elsewhere. The input here is a `density` field (smoke/soot), a `temperature` field in Kelvin, and an optional `flame` field (reaction or heat).

**How the material was checked**
- The blackbody and tone-mapping numbers were computed from the CIE 1931 2° tables (CVRL, 1 nm). As a cross-check, 2856 K gives x,y = 0.4475, 0.4074; Illuminant A is 0.4476, 0.4075.
- All WGSL compiles and validates in Chrome WebGPU on the 780M.
- Code is in `D:\fire\research\rendering_code\`.
- The tone-mapper comparison image is `D:\fire\research\rendering_tonemap_swatches.png`. Each panel runs 800→2400 K left to right, and +8 EV at the top to −4 EV at the bottom.

## 0. Recommended pipeline
1. **Emission.** Use `j = σa·R·B(T)` with `σa = σa,soot·ρ + κflame·flame`.
   - B(T) comes from a 256×1 rgba16float table covering 500–4000 K.
   - rgb holds the chroma scaled to luminance 1; a holds log2(Y/Y(1500 K)).
2. **Brightness.** Shade with `rgb·exp2(γ·a)`. Use γ ≈ 0.55 in real time and 0.7–1.0 offline.
3. **Temperatures.**
   - Tips: 900–1100 K.
   - Body: 1250–1550 K.
   - Core: 1600–1900 K.
   - Embers: 900–1250 K.
   - Fade emission in starting at about 800 K (the Draper point, where things first glow visibly, is 798 K).
4. **White balance.** Apply a 3500–4000 K camera white balance as a Bradford matrix on the whole frame, or bake it into the table, but not both.
5. **Exposure.** Put the flame body at +3 to +5 EV above mid-grey; the fire-lit ground at 1 m lands around −1 to 0 EV.
6. **Tone mapping.** Use AgX-Fire (§5), or ACES-Hill as a punchier alternative.
7. **Smoke lighting.** 512 virtual point lights feed a 32³ light volume, with Henyey-Greenstein scattering at g = 0.6.
8. **Surfaces.** Use 8 clustered lights and glowing char cracks from the same table.
9. **Ray marching.** Half resolution, 0.75–1 voxel steps, jittered with interleaved gradient noise (IGN), energy-conserving per-step integral, skipping, depth-aware upsample, temporal blend.
10. **Post chain, in order:**
    1. Heat haze
    2. Chromatic aberration
    3. Compositing the volume over the scene
    4. Bloom
    5. Vignette
    6. White balance
    7. Exposure
    8. Tone map
    9. sRGB transfer curve
    10. Grain
    11. Dither
11. **Offline.** 30 fps, 180° shutter, 16–32 subframes accumulated in linear HDR, tone-mapped once.

## 1. Physics that matters for the image

**What each part of a fire is, and what it means for rendering**

- **Yellow and orange flame**
  - Physics: soot particles glowing at about 1200–1900 K. Gas emission from CO₂ and H₂O is infrared and invisible.
  - Rendering: emission is soot absorption × Planck.
- **Blue base**
  - Physics: chemiluminescence, not heat glow. CH* emits at 431 nm and C₂ Swan bands at 516/474/564 nm.
  - Rendering: add a small extra term where fuel and air mix. Linear sRGB is about (0.0, 0.45, 1.0); pure CH* is violet. Keep it 2–4 EV below the yellow body.
- **Red tips**
  - Physics: soot at about 1000 K is 10³–10⁴× dimmer than at 1500 K, and the tips are thin.
  - Rendering: comes automatically from B(T).
- **Smoke above**
  - Physics: cooled soot absorbs but no longer glows.
  - Rendering: σa from density, plus scattering lit by the fire below.
- **Shimmer**
  - Physics: hot air has a lower refractive index, n−1 ∝ 1/T.
  - Rendering: screen-space refraction (§10.1).
- **Glowing coals**
  - Physics: char surface at 900–1300 K with emissivity about 0.9.
  - Rendering: the same table, applied on surfaces.

**Temperature reference** (linear sRGB, max channel = 1, D65, no white balance):

| Zone | Temperature | Colour | Brightness Y relative to 1500 K |
|---|---|---|---|
| Draper point | 798 K | (1, 0, 0) | 7.5e-3 cd/m² absolute |
| Dull red tips | 900–1000 K | (1, 0–0.009, 0) | 2.6e-5 to 3.5e-4 |
| Coals / outer flame | 1100–1300 K | (1, 0.033–0.083, 0) | |
| Flame body | 1300–1600 K | (1, 0.083–0.158, 0) | |
| Core | 1700–1900 K | (1, 0.18–0.23, 0) | reads white only through clipping |

**Brightness sanity check.**
- A thick 1500 K blackbody has a luminance Y of 7.7·10³ cd/m²; 1600 K gives 2.1·10⁴.
- A measured candle flame peaks at about 1.6·10⁴ cd/m², with 2·10⁴ quoted as an upper value (amper.ped.muni.cz).
- That agrees with soot at 1500–1700 K with emissivity below 1.

**Soot optics**
- Absorption in the small-particle (Rayleigh) limit is κλ = 6π·E(m)·f_v/λ, with E(m) ≈ 0.26 for m = 1.57−0.56i.
- That gives about 8.9 m⁻¹ per ppm of soot at 550 nm. Dalzell–Sarofim measured κ/f_v of 9.3 µm⁻¹ at 436 nm and 5.0 µm⁻¹ at 806 nm.
- The visible-range exponent is λ^−1.39 (Hottel–Broughton). For RGB channels this gives σa ∝ (0.88, 1.0, 1.22).
- Soot fraction in luminous flames is 0.1–10 ppm; one forest-fuel burner peaked at 6.8 ppm (CSIRO WF06043).
- Across a 0.3 m flame at 1 ppm, the optical depth τ ≈ 2.7, so the luminous body is nearly opaque.

**Smoke optics**
- Smoke from flaming combustion has a near-universal mass extinction of 8.7 ± 1.1 m²/g (Mulholland & Croarkin 2000).
- A plume at 10–100 mg/m³ therefore has σt ≈ 0.09–0.9 m⁻¹. These concentrations are my estimate.
- Single-scattering albedo depends on how the fire burns:
  - Flaming, soot-rich smoke: about 0.3–0.6.
  - Smouldering, organic smoke: about 0.8–0.95.
  - Campfire: use 0.6–0.85 per channel, e.g. (0.85, 0.80, 0.72).
- Henyey-Greenstein g ≈ 0.5–0.7.

**Flicker frequency**
- Pool fires "puff" at f ≈ 1.5·D^−½ Hz (Cetegen & Ahmed); 1.68 is also reported as the constant.
- A 0.5 m campfire gives about 2.1 Hz.
- Check the simulation: total emission should have a dominant 1.5–3 Hz component.

**Gas speeds** (McCaffrey correlations, from memory): about 2–4 m/s in the flame zone.

## 2. Volume rendering equation and discretisation

```
L = ∫₀ᴰ Tr(0,t)·[σa·Le + σs·Ls] dt + Tr(0,D)·L_bg
Tr(a,b) = exp(−∫ₐᵇ σt ds),   σt = σa + σs
Ls = ∫ p·Li dω
HG:  p = (1−g²) / (4π(1+g²−2g·cosθ)^{3/2})
```

- **Phase function check.** At 90° with g = 0.6, p = 0.032, against 0.080 for isotropic. Assuming isotropic scattering overestimates side-scatter from smoke by 2.5×.
- **Volume as a light source** (Villemin & Hery, JCGT 2013). Each voxel has radiant intensity I = e·V. Irradiance at a surface is E = Σ I·Tr·cosθ/d². Using these units everywhere keeps how bright the flames look and how much light they cast consistent with each other.
- **Per-step integration** (Hillaire 2015, energy-conserving). For each step:
  ```
  S   = σa·R·B(T) + σs·(F·p + L_amb)
  w   = (1 − e^{−σt·Δ}) / σt          // use Δ(1 − σt·Δ/2) when σt·Δ < 1e-3
  L  += Tr·S·w
  Tr *= e^{−σt·Δ}
  ```
  Stop when Tr < 0.003. Composite as `L + Tr·L_bg`.
- **Step size.** Wrenninge's rule is at most ½ voxel. In real time, 0.75–1 voxel with jitter and temporal accumulation is enough.
- **Opaque objects.** Logs and ground cut the march: `seg.y = min(seg.y, sceneDist)`.
- **Extra outputs to store:** transmittance, emission-weighted depth, and the integrated refractive-index deficit.

**What the reference renderers do, and which controls to copy**
- **Nguyen, Fedkiw & Jensen 2002.**
  - Ray-marched blackbody emission, a separately modelled blue core, and photon-mapped smoke.
  - Visual adaptation maps the hottest part of the flame to white. This is a white-balance choice:
    - Balanced to 2000 K, a 1600 K flame looks pale (1, 0.53, 0.26).
    - Balanced to 5500 K, the same flame looks deep orange (1, 0.18, 0).
- **Pegoraro & Parker 2006.**
  - Fully spectral, with properties derived from the chemical species present, curved rays through the refractive index, and adaptation-based tone mapping.
  - Practical take-away: per-channel soot absorption, screen-space refraction, and white balance plus exposure.
- **Houdini Pyro (classic).**
  - Temperature→colour modes: Constant, Ramp, or Physical Blackbody.
  - "Color Temp in Kelvin" sets the Kelvin value that a field value of 1.0 maps to.
  - "Adaptation" adjusts the low end of intensity and "Burn" the high end.
  - Smoke scattering phase runs from −1 to 1, default 0.
- **Karma Pyro.**
  - Colour comes from a Ramp or Blackbody mode, with a separate intensity ramp.
  - Blackbody mode folds radiant power into the colour, so set the intensity scale to a constant 1 for physical results.
  - "Temperature Scale" multiplies the field before the lookup.
- **Arnold standard_volume.**
  - `emission_mode` set to blackbody; `blackbody_kelvin` (default 5000) scales a 0–1 temperature channel to Kelvin.
  - `blackbody_intensity` = 1 is physically correct; `blackbody_contrast` (added in 5.2) compresses brightness non-physically.
- **EmberGen.**
  - Official docs were unreachable; this comes from forum threads.
  - Fire colour is a gradient indexed by temperature, with a "blackbody max temperature" (e.g. 4000 K) and a "fire temperature adjustment" multiplier.
- **Wrenninge, "Production Volume Rendering".**
  - Step size and jitter, early termination, opaque cut-outs, deep shadow maps, and tracking estimators.
  - The "Oz" multiple-scattering approximation is `Σ b^i·L(a^i·σt, b^i·σs, c^i·g)` with N = 4–8 and a = b = c = 0.5, where a ≤ b. Use it on smoke.

**How our controls map to theirs:**
- R ↔ intensity
- γ ↔ blackbody_contrast / Burn
- T_onset ↔ low-end cut
- White-balance Kelvin ↔ Nguyen's adaptation
- κflame ↔ how solid the flame looks

## 3. Blackbody colour

**Equations**
- Planck: `B = 2hc² / (λ⁵·(e^{hc/λkT} − 1))`, with 2hc² = 1.191042972e−16 and hc/k = 1.438776877e−2.
- XYZ: `XYZ = 683·∫B·CMF dλ` over 380–780 nm, which puts Y in cd/m².
- Convert to linear sRGB with the standard D65 matrix.

**Choices that matter**
- **Colour-matching data.** The analytic fit by Wyman et al. (2013) underestimates x̄ at 700 nm by 2× (0.0056 against 0.0114). That puts 1000 K at xy (0.6419, 0.3554) instead of (0.6528, 0.3445).
  - Use the real table instead. The 41-row 10 nm CIE table is embedded in `blackbodyLut.js`.
  - It reproduces the 1 nm result to better than 1e-4 in xy.
- **Out-of-gamut blue.** Below about 1900 K the blue channel goes negative, down to about −0.09·Y.
  - Clamp it and renormalise the luminance.
- **Normalisation.** Store chroma at luminance 1 plus log2 of luminance, which is fp16-safe. Max-normalised colours are only for swatches.
- **Weighting.** Weight by energy, not photon count; photon-count weighting moves white to about 5300 K.

**Code**
- **JS generator:** `D:\fire\research\rendering_code\blackbodyLut.js`.
  - Exports `buildBlackbodyLUT({tMin, tMax, size, tRef, wbKelvin, sootAlpha})` and `createBlackbodyTexture(device, opts)`.
  - Uses a 2D 256×1 rgba16float texture, because `textureSampleLevel` has no `texture_1d` overload.
  - Spot checks: `at(1500)` = (3.249, 0.432, 0); `at(1000)` = (4.570, 0.040, 0).
- **Shader side:**

```wgsl
fn blackbody(Tk: f32, contrast: f32) -> vec3f {
  let u = clamp((Tk - 500.0) / 3500.0, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0;
  let e = textureSampleLevel(bbLut, linClamp, vec2f(u, 0.5), 0.0);
  return e.rgb * exp2(contrast * e.a);
}
```

**Brightness relative to 1500 K**

| T (K) | 800 | 900 | 1000 | 1100 | 1200 | 1300 | 1400 | 1500 | 1600 | 1800 | 2000 | 2200 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Y/Y(1500 K) | 1.0e-6 | 2.6e-5 | 3.5e-4 | 3.0e-3 | 1.8e-2 | 8.4e-2 | 0.32 | 1 | 2.76 | 15.2 | 60 | 186 | 727 | 3900 |

- The log-slope d lnY / d lnT is 28.7 at 800 K, 23.5 at 1000 K, 19.9 at 1200 K, 16.2 at 1500 K, 12.4 at 2000 K and 8.5 at 3000 K.
- Going from 1200 K to 1800 K is 9.7 EV at γ = 1, but only 5.3 EV at γ = 0.55.
- Luminous efficacy is only 0.085 lm/W at 1500 K (1.6 at 2000 K). Derive light from the visible emission, never from heat release.

**Sample values**

| T (K) | x, y | lin sRGB, Y=1 | lin sRGB, max=1 | sRGB 8-bit (max=1) | log2(Y/Y1500) | Y cd/m² (ε=1) | WB 3200 K, max=1 |
|---|---|---|---|---|---|---|---|
| 800 | 0.6813, 0.3179 | 4.704, 0, 0 | 1, 0, 0 | 255, 0, 0 | -19.87 | 0.00809 | 1, 0, 0 |
| 900 | 0.6670, 0.3315 | 4.704, 0, 0 | 1, 0, 0 | 255, 0, 0 | -15.24 | 0.200 | 1, 0.001, 0 |
| 1000 | 0.6528, 0.3445 | 4.571, 0.039, 0 | 1, 0.0086, 0 | 255, 23, 0 | -11.49 | 2.69 | 1, 0.035, 0 |
| 1100 | 0.6388, 0.3565 | 4.236, 0.139, 0 | 1, 0.0328, 0 | 255, 51, 0 | -8.40 | 23.0 | 1, 0.070, 0 |
| 1200 | 0.6250, 0.3675 | 3.941, 0.227, 0 | 1, 0.0575, 0 | 255, 68, 0 | -5.80 | 139 | 1, 0.106, 0 |
| 1300 | 0.6116, 0.3772 | 3.681, 0.304, 0 | 1, 0.0826, 0 | 255, 81, 0 | -3.58 | 648 | 1, 0.145, 0 |
| 1400 | 0.5985, 0.3858 | 3.452, 0.372, 0 | 1, 0.1078, 0 | 255, 92, 0 | -1.67 | 2.44e3 | 1, 0.184, 0 |
| 1500 | 0.5857, 0.3931 | 3.250, 0.432, 0 | 1, 0.1330, 0 | 255, 102, 0 | 0.00 | 7.74e3 | 1, 0.224, 0.005 |
| 1600 | 0.5732, 0.3993 | 3.070, 0.486, 0 | 1, 0.1582, 0 | 255, 111, 0 | 1.47 | 2.14e4 | 1, 0.265, 0.020 |
| 1700 | 0.5611, 0.4043 | 2.910, 0.533, 0 | 1, 0.1832, 0 | 255, 119, 0 | 2.76 | 5.26e4 | 1, 0.307, 0.040 |
| 1800 | 0.5492, 0.4082 | 2.767, 0.576, 0 | 1, 0.2080, 0 | 255, 126, 0 | 3.92 | 1.17e5 | 1, 0.349, 0.064 |
| 1900 | 0.5378, 0.4112 | 2.639, 0.614, 0 | 1, 0.2325, 0 | 255, 132, 0 | 4.96 | 2.42e5 | 1, 0.393, 0.094 |
| 2000 | 0.5267, 0.4133 | 2.520, 0.647, 0.020 | 1, 0.2567, 0.0081 | 255, 139, 22 | 5.90 | 4.64e5 | 1, 0.437, 0.129 |
| 2200 | 0.5056, 0.4152 | 2.314, 0.704, 0.065 | 1, 0.3041, 0.0282 | 255, 150, 47 | 7.54 | 1.44e6 | 1, 0.526, 0.215 |
| 2400 | 0.4861, 0.4147 | 2.143, 0.750, 0.114 | 1, 0.3499, 0.0533 | 255, 160, 65 | 8.90 | 3.70e6 | 1, 0.617, 0.326 |
| 2600 | 0.4682, 0.4123 | 1.999, 0.787, 0.166 | 1, 0.3939, 0.0828 | 255, 168, 81 | 10.06 | 8.28e6 | 1, 0.711, 0.459 |
| 2800 | 0.4519, 0.4086 | 1.876, 0.818, 0.218 | 1, 0.4363, 0.1165 | 255, 176, 96 | 11.06 | 1.65e7 | 1, 0.805, 0.616 |
| 3000 | 0.4369, 0.4041 | 1.771, 0.844, 0.272 | 1, 0.4768, 0.1537 | 255, 184, 109 | 11.93 | 3.02e7 | 1, 0.902, 0.797 |

**White balance comparison** (max channel = 1)

| T | WB 2000 K | WB 2700 K | WB 3200 K | WB 4000 K | WB 5500 K |
|---|---|---|---|---|---|
| 1000 | 1, .086, .011 | 1, .047, 0 | 1, .035, 0 | 1, .023, 0 | 1, .014, 0 |
| 1200 | 1, .211, .028 | 1, .131, 0 | 1, .107, 0 | 1, .085, 0 | 1, .067, 0 |
| 1400 | 1, .359, .100 | 1, .224, .011 | 1, .184, 0 | 1, .150, 0 | 1, .122, 0 |
| 1600 | 1, .535, .258 | 1, .323, .055 | 1, .265, .020 | 1, .217, 0 | 1, .178, 0 |
| 1800 | 1, .745, .541 | 1, .430, .131 | 1, .350, .064 | 1, .285, .023 | 1, .233, 0 |
| 2000 | 1, 1, 1 | 1, .544, .245 | 1, .437, .129 | 1, .353, .059 | 1, .288, .017 |

**White-balance matrices** (linear sRGB → linear sRGB, Bradford, written as rows):
- 2700 K: [0.7415 −0.5110 −0.1249; 0.0237 1.1999 −0.0522; 0.0609 0.2286 3.6404]
- 3200 K: [0.8002 −0.3686 −0.0759; 0.0173 1.1324 −0.0297; 0.0361 0.1326 2.5302]
- 4000 K: [0.8693 −0.2257 −0.0383; 0.0107 1.0737 −0.0136; 0.0175 0.0625 1.7196]
- 5500 K: [0.9514 −0.0831 −0.0093; 0.0040 1.0272 −0.0023; 0.0038 0.0119 1.1410]

In WGSL: `hdr = max(hdr * mat3x3f(row0,row1,row2), vec3f(0))` — the `v * M` form with M built from rows computes rows·v. Apply it before exposure.

**Optional extras**
- **Thin soot** (`sootAlpha: 1.39`): at 1500 K the green channel rises from 0.133 to 0.168, so wispy flames look slightly yellower.
- **Blue chemiluminescence:** `k·flame·(1−soot)·(0, 0.45, 1)`.

## 4. Why fire photos look the way they do
1. **Dynamic range.**
   - The flame body is 10³–10⁵ cd/m².
   - Ground at 1 m: E ≈ L·Ω ≈ 5e3·0.2 = 1000 lx; with albedo 0.15 that is about 48 cd/m².
   - So the flame is about 100× (6.7 EV) brighter than the ground at 1 m, and about 1000× at 3 m. Any exposure that shows the surroundings clips the flames.
2. **Per-channel clipping, the "path to white".**
   - R clips first (orange turns yellow), then G. B only rises through sensor crosstalk and the camera's highlight handling.
   - That gives white-yellow cores, an orange body and red tips.
   - A pure per-channel clip ends at (255, 255, 0) and never reaches white.
   - Hue-preserving curves go salmon instead (see the pixls.us discussion).
3. **White balance.**
   - Auto or night white balance (2700–4000 K) makes flames orange-yellow and the ambient light blue.
   - Daylight white balance makes the same fire deep red-orange.
4. **Shutter.** At 1/50–1/60 s, flames smear 3–8 cm per frame and look silky. At 1/1000 s they look crisp and ragged.
5. **Lens and camera.**
   - Veiling glare and bloom.
   - Chromatic aberration on bright edges.
   - Vignetting.
   - High-ISO grain.
6. **Smoke.** Lit orange from below, with a cool rim from the moon or sky.

## 5. Tone mapping

**Test:** a 1500 K blackbody colour at luminance 0.18·2^EV, output as 8-bit sRGB; "h" is hue in degrees.

| EV | Sensor clip | ACES-Hill | AgX (three.js) | Khronos PBR Neutral | **AgX-Fire** |
|---|---|---|---|---|---|
| −2 | 107/38/0 | 102/17/1 | 120/41/5 | 107/38/0 | **113/0/0** |
| 0 | 202/79/0 h23 | 214/81/9 h21 | 191/102/56 h20 | 202/79/0 h23 | **187/68/0** |
| +2 | 255/152/0 h36 | 255/188/76 h38 | 243/169/124 h22 | 251/143/113 h13 | **241/154/1** |
| +4 | 255/255/0 h60 | 255/240/184 h47 | 255/222/191 h29 | 254/205/196 h9 | **255/220/134** |
| +6 | 255/255/0 | 255/253/239 h53 | 255/248/235 h39 | 255/239/237 h7 | **255/249/209** |
| grey 0.18 | 118 | 127 | 128 | 105 | 111 |

**Verdicts**
- **Khronos PBR Neutral:** hue-preserving, so fire highlights go pink. Rejected for fire.
- **Plain AgX:** goes peach/salmon and is low-contrast. Blender's rotated variant (Godot/EaryChow matrices) behaved almost the same on this test.
- **Tony McMapface:** needs a 48³ lookup table, which is an asset, so it doesn't fit the no-assets rule.
- **ACES-Hill:** a good photographic path with crimson tips, but highlights reach white about 1 EV earlier than AgX-Fire.
- **AgX-Fire (chosen):**
  - Path: deep red → amber → yellow → warm white.
  - Greys stay neutral.
  - About 1 EV more detail inside bright flames.
  - No lookup table.
  - Mid-grey maps to 111, so add +0.3 EV if you need to match other content.

```wgsl
fn agxCurve(x: vec3f) -> vec3f {
  let x2 = x * x; let x4 = x2 * x2;
  return 15.5*x4*x2 - 40.14*x4*x + 31.96*x4 - 6.868*x2*x + 0.4298*x2 + 0.1191*x - 0.00232;
}
// rows; v * M == ROWS · v
const AGXF_IN  = mat3x3f(vec3f(0.86,0.11,0.03), vec3f(0.16,0.79,0.05), vec3f(0.05,0.10,0.85));
const AGXF_OUT = mat3x3f(vec3f(1.272727,-0.227273,-0.045455), vec3f(-0.297521,1.384298,-0.086777), vec3f(-0.049587,-0.185950,1.235537));
fn tonemapAgxFire(hdr: vec3f) -> vec3f {
  let minEv = -12.47393; let maxEv = 4.026069;
  var v = max(hdr, vec3f(0.0)) * AGXF_IN;
  v = clamp((log2(max(v, vec3f(1e-10))) - minEv) / (maxEv - minEv), vec3f(0.0), vec3f(1.0));
  v = agxCurve(v);
  v = pow(max(v, vec3f(0.0)), vec3f(1.2));
  let luma = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  v = luma + 1.2 * (v - luma);
  v = pow(max(v, vec3f(0.0)), vec3f(2.2));
  return clamp(v * AGXF_OUT, vec3f(0.0), vec3f(1.0));   // then sRGB OETF
}
```

ACES-Hill is included in `composite.wgsl` as `tonemapper = 1`.

**Tuning**
- Larger off-diagonals in AGXF_IN make highlights turn yellow sooner.
- A stronger AGXF_OUT raises mid-tone saturation, but past rows of about (.74 .20 .06) a crease appears at the red/amber boundary.

**Exposure**
- **Camera model** (Frostbite EV100): L_max = 1.2·2^EV100. A campfire scene is about EV100 4–7.
- **Relative units (recommended):**
  - Set R so a thick 1500 K blackbody is about 0.18·2⁵ ≈ 6.
  - Fire-lit ground at 1 m should average 0.05–0.15.
  - Night ambient light is 0.002–0.01.
- **Auto-exposure.** Keep exposure fixed. If you use auto-exposure, smooth it over 1–2 s and exclude the flames from metering.

## 6. Smoke and self-illumination

**Starting values**

| Quantity | Value | Why |
|---|---|---|
| σa,soot per unit density | 1–5 m⁻¹ | plume σt 0.1–1 m⁻¹ at density 0.1–0.3 |
| Albedo | 0.6–0.85, per channel (0.85, 0.80, 0.72) | campfire smoke mixture |
| Henyey-Greenstein g | 0.55–0.7 | sub-micron particles scatter forward |
| κflame per unit flame | 5–30 m⁻¹ | optical depth 1–5 across a 20–30 cm flame body |
| "Oz" multiple scattering | N = 4, a = b = c = 0.5 | cheap brightening of thick smoke |

**Three passes** (verified):
1. `reduceEmission` (`firelight.wgsl`).
   - Uses 8³ coarse cells. Each 64-thread workgroup sums `j·dV` and the luminance-weighted centroid.
   - Output: 512 virtual point lights.
2. `buildFireLight` (`fluence.wgsl`).
   - Builds a 32³ light volume: `F = Σ I_k·Tr_k / (d² + r²)`, where Tr_k is a 4-tap shadow through ¼-resolution density.
   - About 1–3 ms.
3. March. In-scattering is `σs·F·p_HG(dot(normalize(x−centroid), −rd))`.

**Cheaper fallback:** a 3D mip pyramid of emission, summed as `Σ w_l·ē_l·s_l` with `w_l` calibrated once against the full sum. It has no shadows and costs about 0.3 ms.

**Data layout:** pack density, temperature (K) and flame into one rgba16float 3D texture so each step needs one fetch.

## 7. Lighting the environment
- **Surface lights.** Merge the 512 point lights into 8 (2×2×2 octants).
  - Irradiance: `E = Σ I_k·cos/(d² + r²)`, using a horizon-wrapped cosine so logs don't get a hard edge.
  - Because the intensities *are* the volume's emission total, the 1.5–3 Hz flicker comes for free.
  - Don't add synthetic flicker; at most low-pass over 15–30 ms.
- **Shadows.**
  - Use one cube shadow map at the luminance centroid, with soft-edged filtering (PCF) whose radius matches the cluster radius.
  - If logs are SDFs, cone-traced soft shadows with k = 8–16 work too.
- **Char and embers** (`surface.wgsl`):
  - Albedo: char 0.02–0.04, ash 0.35–0.5.
  - Cracks: anisotropic Voronoi, `F2−F1 < 0.03–0.14`, with the grain axis compressed ×0.4 and about 70 cells/m.
  - Temperatures: crack bottoms 900–1250 K, faces 450–1000 K, with per-cell "breathing" at 0.1–0.5 Hz.
  - Emission: `0.9·R·B(T)` with the same R and γ as the flames.
- **Soot staining.** `sootDarken` darkens nearby stones up to 75 %.

## 8. Performance on the 780M
**Hardware rough numbers**
- About 8 TFLOPS peak and about 48 texture units.
- About 100–120 GB/s shared memory bandwidth, which is the real limit.
- A 128×256×128 rgba16f field is 33.5 MB.

**Techniques**
- **Half-resolution march.** Upsample with depth-aware weights: `w = bilinear·exp(−|Δz|/(0.05·z))`. About 4×.
- **Tight bounding box** from the reduction pass. About 2–3×.
- **Brick skipping.** 8³-voxel bricks storing max σt and max emission. About 1.5–3×.
- **Early stop** at transmittance below 0.003.
- **Adaptive step:** `Δ = clamp(c/σt,max, 0.5, 2 voxels)`, doubled once transmittance drops below 0.3.
- **Jitter.** IGN with a 5.588238 px scroll per frame.
- **Temporal reuse.**
  - Reproject using the emission-weighted depth, offset by the simulation velocity (`x − u·Δt`).
  - Clamp the history to the 3×3 neighbourhood range (min/max in YCoCg).
  - Keep 70–85 % history.

**Expected cost:** 960×540 × about 96 steps × 3 fetches ≈ 150 M samples ≈ 3–6 ms.

## 9. Detail amplification
- **Wavelet Turbulence** (Kim et al. 2008).
  - High-resolution velocity: `U = I(u) + 2^{−5/6}·I(ê)·z(I(c))`.
    - `ê` is the energy of the coarse field's highest resolved band.
    - `z` is the curl of wavelet noise, evaluated at advected texture coordinates.
  - Octave weights are 2^{−5/6} ≈ 0.56 (the paper notes this is close to Perlin's ½).
  - Regenerate the texture coordinates when the eigenvalues of their Jacobian leave [½, 2].
  - Real-time approximation: `ê ≈ ½|u − blur(u)|²`.
- **Curl noise** (Bridson 2007).
  - `v = ∇×ψ` is divergence-free by construction.
  - Near obstacles, scale by `ramp(r) = 15/8·r − 10/8·r³ + 3/8·r⁵`.
  - Use length scales of 4–8 coarse voxels and 2–3 octaves.
- **Advected texture coordinates** (Neyret 2003).
  - Two coordinate sets, phase-shifted by ½, each reset every 0.5–1.5 s.
  - Blend with `w₀ = |1 − 2·fract(t/T)|`.
- **Render-time detail (do this first):**
  - Warp the lookup: `x' = x + A·curl(x·f − (0, v·t, 0))`, with A = 0.5–1.5 voxels and f = 1/(4–8 voxels). Weight it by flame strength and by |∇flame|.
  - Sharpen the flame front: `flame' = smoothstep(0.08, 0.35, flame + 0.15·(fbm − 0.5))`.
  - Mottle the colour: `T' = T + 60 K·(fbm − 0.5)`.
  - Add smoke edge detail: `ρ·(1 + 0.6·(fbm − 0.5))·edgeMask`.
- **Upres.** Simulate velocity at 64×128×64 and advect the scalars at 2× resolution.

## 10. Post-processing

### 10.1 Heat haze
- Refractive index (Gladstone–Dale): n−1 ≈ 2.93e−4·273/T, so 2.67e-4 at 300 K and 0.89e-4 at 900 K.
- Rays bend by the lateral gradient of ∫(n_amb − n) ds, away from the hot core.
- Worked example: a 0.3 m path with a 3 cm gradient gives about 1.8e-3 rad, or 1–5 px at 1080p.
- `fsHdrComposite` implements it:
  1. Take the screen-space gradient of `aux.r`.
  2. Convert it to an angle with `focalPx/z_plume`.
  3. Turn the angle into a background shift with `(1 − z_plume/z_bg)`.
  4. Add rising noise for fine shimmer.
  5. Skip pixels in front of the plume.
- `hazeGain` 1 is physical; 1.5–2 reads better on phones.

### 10.2 Bloom
- Jimenez 2014 (Call of Duty: Advanced Warfare) mip chain.
- **Downsample:** 13 taps as 5 overlapping boxes weighted 0.5 + 4×0.125. Use the Karis average `1/(1+luma)` on the first pass only to kill fireflies.
- **Upsample:** 3×3 tent, added into the level above.
- **Final blend:** `mix(hdr, bloom/levels, 0.04)`. This is energy-conserving and needs no threshold.
- Use 6–7 levels and a strength of 0.03–0.06.

### 10.3 Lens and film effects

| Effect | How | Value |
|---|---|---|
| Lateral chromatic aberration | R sampled at `0.5 + (uv−0.5)(1+k)`, B at `(1−k)` | k = 0.002–0.003 |
| Vignette | `cos²θ = 1/(1 + |ndc·(aspect, 1)·tan(fov/2)|²)`, blended toward cos⁴ | 0.5–0.8 |
| Grain | display space, triangular noise, 1.5 px clumps, amplitude `g·4y(1−y)`, re-seeded per frame | g = 0.015–0.03 |
| Dither | triangular, ±1 LSB | always |

### 10.4 Offline video
- **Shutter:** 180°, i.e. 1/(2·fps), centred on the frame time.
- **Subframes:** 16–32 per frame at stratified times.
  - Pixel jitter: R2 sequence `(0.7548776662·n, 0.5698402910·n) mod 1`.
  - Ray jitter: IGN with a global subframe counter.
- **Depth of field:** thin lens with aperture radius f/(2N), e.g. 12.5 mm for 50 mm at f/2.
- **Accumulation:** sum linear HDR in rgba32float. Run haze, bloom and tone mapping once, after accumulating. Tone-mapping before averaging breaks the path to white.
- **Format:** 1080×1920 or 1920×1080, 30 fps × 20 s = 600 frames. Optionally render at 2× and downsample.
- **Encode:**

```
ffmpeg -framerate 30 -i f%04d.png -c:v libx264 -preset slow -crf 14 -pix_fmt yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart fire.mp4
```

Keep a 10-bit HEVC master as well.

## 11. Parameter cheat sheet

| Group | Parameter | Real time | Offline |
|---|---|---|---|
| March | step size | 0.75–1 voxel | 0.25–0.5 voxel |
| | max steps | 96–160 | 512 |
| | resolution | ½ | full or 2× |
| Fire | γ (contrast) | 0.5–0.6 | 0.6–0.8 |
| | κflame | 10–20 m⁻¹ | 5–30 m⁻¹ |
| | T_onset | 750–850 K | 750–850 K |
| Smoke | σa,soot | 1–5 m⁻¹ | 1–5 m⁻¹ |
| | albedo | 0.6–0.85 | 0.6–0.85 |
| | g | 0.6 | 0.6 |
| Camera | white balance | 3500–4000 K | 3500–4000 K |
| | flame body exposure | +3 to +5 EV | +3 to +5 EV |
| | tone mapper | AgX-Fire | AgX-Fire |
| Post | bloom levels / strength | 6 / 0.04 | 7 / 0.04 |
| | haze gain | 1.5 | 1–1.5 |
| | chromatic aberration | 0.0025 | 0.002 |
| | vignette | 0.6 | 0.6 |
| | grain | 0.02 | 0.015 |
| Video | shutter / subframes | — | 180° / 16–32 |

**"Looks CG" checklist**

| Symptom | Likely cause |
|---|---|
| No yellow-white core | exposure too low, or a hue-preserving tone mapper |
| No red tips | γ too low or T_onset too high |
| Flame opaque everywhere | κflame too high at the edges |
| Flames rise too slowly | real gas moves 2–4 m/s and puffs at 1.5–3 Hz |
| Black smoke with no orange underside | missing self-illumination |

Also check for: ground that doesn't flicker, no heat haze against a textured background, no bloom, no grain or chromatic aberration, and embers that don't use the same colour table.

## 12. Sources
- Nguyen, Fedkiw & Jensen 2002: https://graphics.ucsd.edu/~henrik/papers/fire/
- Pegoraro & Parker 2006: https://diglib.eg.org/items/60274f05-4f1f-451c-8e86-bc203abd016d/full
- Summary of both: https://ar5iv.arxiv.org/html/1804.05261
- Villemin & Hery 2013: https://jcgt.org/published/0002/02/10/paper.pdf
- Simon et al. 2017 (FNEE): https://cg.ivd.kit.edu/english/fnee.php
- Production Volume Rendering: https://graphics.pixar.com/library/ProductionVolumeRendering
- Wrenninge & Zafar 2011: https://cg.informatik.uni-freiburg.de/intern/seminar/volumeRenderingProductionFundamentals2011.pdf
- Oz 2013: https://history.siggraph.org/learning/oz-the-great-and-volumetric-by-wrenninge-kulla-and-lundqvist
- Hillaire 2015: https://advances.realtimerendering.com/s2015/
- Houdini Pyro: https://sidefx.com/docs/houdini/nodes/shop/pyro.html
- Karma Pyro: https://www.sidefx.com/docs/houdini/nodes/vop/kma_pyrofireemission.html
- Arnold standard_volume: https://help.autodesk.com/view/ARNOL/ENU/?guid=arnold_core_ac_standard_volume_html
- Arnold 5.2.0 notes: https://docs.arnoldrenderer.com/display/A5AFMUG/5.2.0
- EmberGen forum: https://forums.jangafx.com/t/fire-temperature-adjustment-when-exporting/508
- CIE CMF (CVRL): http://www.cvrl.org/database/data/cmfs/ciexyz31_1.csv
- Wyman et al. 2013: https://research.nvidia.com/labs/rtr/publication/wyman2013simple/
- Scratchapixel blackbody: https://scratchapixel.com/lessons/cg-gems/blackbody/blackbody.html
- Khronos PBR Neutral: https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/README.md
- three.js tone mapping: https://github.com/mrdoob/three.js/blob/dev/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js
- Godot AgX: https://github.com/godotengine/godot/blob/master/servers/rendering/renderer_rd/shaders/effects/tonemap.glsl
- EaryChow AgX_LUT_Gen: https://github.com/EaryChow/AgX_LUT_Gen
- Tony McMapface: https://github.com/h3r2tic/tony-mc-mapface
- Bevy tonemapper notes: https://doc.qu1x.dev/bevy_trackball/bevy/core_pipeline/tonemapping/enum.Tonemapping.html
- AgMax notes: https://cgmeerkat.gumroad.com/l/blender-AgMax
- pixls.us filmic discussion: https://discuss.pixls.us/t/comparing-filmic-color-science-v5-v6/31646?page=6
- Jimenez 2014: https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
- LearnOpenGL bloom: https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
- IGN (demofox): https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/
- Wavelet Turbulence: https://www.tkim.graphics/WTURB/
- Curl noise: https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
- Advected Textures: https://diglib.eg.org/items/c4bf80cc-23f0-4107-958d-e03c0b9c8ac4/full
- Soot optics: https://www.thermalradiation.net/webappendix/webappendixDSoot1.html
- Soot spectral correlation: https://www.thermalradiation.net/webappendix/webappendixDSoot.html
- Forest-fuel soot fraction: https://publish.csiro.au/wf/WF06043
- Mulholland & Croarkin: https://firedoc.nist.gov/article/JHcyXYQBWEcjUZEY8eM6
- Smoke optics by fire regime: https://journals.sagepub.com/doi/reader/10.1177/1178622118774803
- Candle luminance: https://amper.ped.muni.cz/light/luminance/candle/
- Pool-fire puffing: https://arxiv.org/pdf/1803.10400
- Refractive index of air: https://arxiv.org/pdf/2204.02603