# Physics of Real Fire for a Physically Based 3D Grid Fire Simulation

Target: real-time WebGPU campfire (about 1.5 m domain, 64^3 to 192^3 cells) and an offline 20 s cinematic render.
Everything is in SI units (m, s, K, kg, W). Values marked **[est]** are my own derivations or order-of-magnitude
estimates. Values with a citation come from the cited source. All numeric derivations were computed and checked.

---

## 0. Cheat sheet: the numbers that matter

| Quantity | Value | Source |
|---|---|---|
| Adiabatic flame temperature, wood/hydrocarbons in air | ~2200 K at constant p, about 2000-2100 K with dissociation | Wikipedia "Adiabatic flame temperature"; Thornton's-rule check in Sec. 2.4 |
| Local instantaneous flame-sheet peak in turbulent wood/pool fires | ~1400-1800 K **[est]** (soot radiation drains 20-40%) | Sec. 2.4 |
| Time-averaged centreline gas temperature, continuous flame zone | T∞ + ~850 K ≈ 1150 K (constant with height) | McCaffrey 1979 (derived from his constants) |
| Soot-luminous (yellow/orange) zone | ~1100-1500 K | Sec. 2 |
| Mean temperature rise at the visible flame tip | ΔT ≈ 320 °C (≈ 610-620 K absolute) | McCaffrey 1979 |
| Glowing char/ember surface | 750-955 °C (1020-1230 K); colour-pyrometry mean 931 °C ≈ 1200 K | UMD / Berkeley ember pyrometry |
| Onset of visible red glow (Draper point) | ~798 K (525 °C) | classic |
| Puffing/flicker frequency | f ≈ 1.5 / sqrt(D) Hz = 0.48 sqrt(g/D) | Cetegen & Ahmed 1993; Xia & Zhang 2018 |
| Mean flame height | L = 0.235 Q^(2/5) − 1.02 D  (L, D in m; Q in kW) | Heskestad |
| McCaffrey zones (z/Q^(2/5), m·kW^-2/5) | continuous < 0.08 < intermittent < 0.20 < plume | McCaffrey 1979 |
| Peak time-averaged centreline velocity | u ≈ 1.9 Q^(1/5) m/s (≈ 4.2 m/s at 50 kW) | McCaffrey 1979 |
| Apparent rise speed of visible flame structures | 1.1-1.8 m/s while gas moves at 3.3-4.3 m/s (factor 2.5-3 slower) | McCaffrey 1979 |
| Centreline pressure/velocity fluctuation | ~35% of mean on axis, rising to ~100% at the flame edges; energy concentrated near the puffing frequency | McCaffrey 1979 |
| Heat release per burning area, cellulosics | 100-200 kW/m² | McCaffrey 1979 (citing literature) |
| Typical campfire | Q ≈ 20-150 kW, fuel-bed D ≈ 0.4-0.8 m, flames 0.4-1.2 m, f ≈ 1.7-2.4 Hz | Sec. 4.4 |
| Soot volume fraction in flames | ~1-7 ppm peak (forest-fuel flame: 6.8 ppm) | frames.gov forest-fuel burner study |
| Soot Planck-mean absorption | κ ≈ 1817-1862 · f_v · T  [m⁻¹] | Williams/Shaddix via RadLib; Kent & Honnery family |
| Soot spectral emissivity (visible) | κ_λ ∝ λ^-1.39 (Hottel-Broughton) | Hottel & Broughton |
| Visible luminance vs temperature | L_v ∝ T^~20 at 1200 K, T^~14 at 1800 K | my CIE integration, Sec. 2.6 |
| Expansion from heat release (low Mach) | ∇·u = q''' / (p0 c_p / R) = q''' / 3.55e5  [1/s per W/m³] | Sec. 3.2 |
| Buoyancy acceleration of hot gas | a ≈ 0.81 · g (T − T∞)/T∞ (≈ 23 m/s² in the flame) | from McCaffrey's C = 0.9; Sec. 3.3 |

---

## 1. What a campfire physically is

A wood fire is a **non-premixed (diffusion) flame fed by a solid that gasifies itself**:

1. **Heat feedback.** Radiation and convection from the flame heat the wood surface.
2. **Drying, then pyrolysis.** Free water boils off first (≈ 373 K, with steam pockets as the source of crackle, Sec. 9).
   Then the polymers decompose. Hemicellulose decomposes mainly at **220-315 °C** and cellulose at **315-400 °C**.
   Lignin decomposes slowly over **160-900 °C** and leaves about 40 wt% residue (Yang et al., *Fuel* 86:1781, 2007).
   The products are a mix of combustible gas and tar (CO, H2, CH4, other hydrocarbons, tar aerosols, plus CO2 and H2O),
   leaving a porous **char** layer. Roughly 75-85% of dry wood leaves as volatiles and 15-25% stays as char **[est, typical]**.
3. **Low-momentum fuel release.** Pyrolysis gas leaves the surface at only about **1-3 cm/s** **[est]**.
   Take a mass flux of 5-15 g/(m²·s) and a gas density of about 0.5 kg/m³ at ~600 K. With essentially no initial momentum,
   the fire is **purely buoyancy-driven** (Richardson number → ∞, Froude number → 0). Every flame structure you see
   comes from buoyancy, not injection.
4. **Mixing-limited burning.** The fuel gas can burn only where it mixes with entrained air. The reaction sits on a
   thin, wrinkled sheet near the stoichiometric surface. Wood volatiles need roughly 4.5-5 kg of air per kg **[est]**.
   Inside the sheet is a fuel-rich, hot, oxygen-free core where **soot forms**. Outside it are air and products.
5. **Heat release rate.** The effective heat of combustion of wood is about **14 MJ/kg** (fire-science measurement found
   in search). Burning cellulosic surfaces release about **100-200 kW/m²**.
   So a campfire with 0.2-0.5 m² of actively flaming surface gives Q ≈ 20-100 kW.
6. **Char glowing combustion.** Once volatiles are exhausted, the char surface oxidises directly (C + O2 → CO/CO2)
   at ~1000-1250 K. These are the glowing coals. CO escaping from the coal bed can burn in faint blue flames just above it.

Consequence for the sim: you inject **gaseous fuel** at low speed from the log surfaces. Burning happens where that fuel
meets air. Buoyancy, expansion and entrainment produce everything else.

---

## 2. Flame structure, temperatures and colour

### 2.1 Layer by layer (moving outward and upward from the log surface)

| Zone | What happens | T (K) | What you see |
|---|---|---|---|
| Wood surface/char | pyrolysis front under the char; char oxidation at surface and cracks | 600-900 under char; 1000-1250 at glowing char | black char; orange-red glow in cracks and on coals |
| Fuel-rich core (just above surface) | hot pyrolysis gas, no O2 | 500-900 | transparent, sometimes brownish tar smoke or a faint blue base at the leading edge |
| Reaction sheet, leading edge | fuel meets air; heat release peaks; CH*, C2*, OH* formed | 1600-2000 locally | **thin blue/blue-green fringe**, 1-3 cm tall at the base of each flame, only where soot has not formed yet |
| Soot inception and growth (fuel side of the sheet) | PAH → nascent soot; growth by surface reactions | 1300-1600 needed for inception | onset of **yellow** a short distance (tens of ms of residence time) above the blue base |
| Luminous zone | hot soot particles radiate as a nearly-grey body | 1100-1500 | yellow-orange body of the flame, very bright, nearly opaque where thick |
| Soot oxidation and burnout (flame tip) | OH/O2 burns the soot as it mixes with air | 1000-1300 falling | orange to deep red tips; the visible flame **ends abruptly where the soot is gone** |
| Detached flame tongues | blobs pinched off by the puffing vortex burn out within ~0.1-0.3 s | 900-1300 | isolated orange "licks" above the continuous flame |
| Plume | entrainment-diluted products, CO2, H2O, residual soot and tar aerosol | ΔT ∝ z^-5/3 | invisible hot gas (refractive shimmer); grey/blue-white smoke when combustion is incomplete |

Sources and checks:
- The blue colour comes from chemiluminescence of short-lived radicals. In methane diffusion flames the bands are
  **OH\* 309.1 nm** (UV), **CH\* 431.4 nm**, and **C2\* Swan bands 463-563 nm** (strongest heads near 473.7, 516.5 and 563.5 nm)
  (spectroscopy study, researching.cn; Liu lecture slides, fluids.ac.uk).
  Sootless hydrocarbon flames emit almost all their visible light below ~565 nm.
- Soot inception needs about **1300 K or more**; formation is usually quoted at 1300-1600 K (BYU soot lecture;
  inverse-diffusion-flame studies). In inverse diffusion flames the soot layer sits on the fuel side of the OH sheet,
  with PAH further on the fuel side. Soot growth continues downstream even below the inception temperature.
- **Why the base is blue:** gas near the base is young. Soot needs a few tens of ms of hot, fuel-rich residence time
  to appear, so for the first 1-3 cm (at about 0.5-1 m/s) only chemiluminescence is visible. Higher up, soot incandescence
  outshines the radicals by orders of magnitude. **This is the key to a natural blue base: gate soot formation by gas age.**
- **Blue flames over coals:** CO from char oxidation burns above the coal bed with a weak, mostly continuous blue emission
  (Nature 109:683, 1922 letter). It is common late in the fire when coals are hot and air is plentiful.
  It reads as wispy, transparent violet-blue sheets 2-10 cm tall dancing over the embers.

### 2.2 Soot amounts and optics (needed for emission and absorption)

- Soot volume fraction f_v: about **1-7 ppm** in luminous regions (6.8 ppm peak measured in the persistent region of a
  forest-fuel flame, frames.gov catalog 45982). Wood flames are lightly sooting compared with heptane or toluene.
- Spectral absorption coefficient, Rayleigh regime: `κ_λ = 6π E(m) f_v / λ`, with E(m) ≈ 0.26-0.35.
  At 550 nm: **κ ≈ 10 m⁻¹ per ppm of soot** **[computed]**. So a 1 ppm flame 10 cm thick has optical depth ≈ 1.
  Thick flame cores at 3-7 ppm are effectively opaque within 1-3 cm, while thin tongues stay translucent.
  This matches how real campfires look.
- Visible spectral shape: Hottel-Broughton gives `ε_λ = 1 − exp(−K L / λ^α)` with **α = 1.39** in the visible.
  This makes soot emission slightly bluer than a blackbody at the same T (see the colour table in Sec. 2.6).
- Planck-mean absorption for radiative cooling: **κ_P = F_s f_v T**, with F_s ≈ 1817 K⁻¹m⁻¹
  (Shaddix refractive index, RadLib documentation) or ≈ 1862 in other model variants.
- Smoke optics **[literature-typical]**:
  - Black-carbon soot smoke is strongly absorbing (single-scattering albedo ~0.2-0.3), so it looks dark grey or black.
  - Wood smoke from smouldering or tar-rich burning is mostly organic aerosol (albedo ~0.85-0.95) with sub-micron
    particles. Viewed sideways against a dark background it looks **bluish** (Rayleigh-like scattering, λ⁻⁴); seen
    against the light, or once particles grow, it turns white or grey (atoptics.co.uk).

### 2.3 Smoke

- **Flaming wood fires that burn well** release little soot. Most visible smoke comes from:
  - smouldering or pyrolysing parts that are not in a flame (white-grey organic aerosol, plus condensed water early on);
  - soot that escapes oxidation when flame tips are quenched by cold air (dark wisps above the tips).
- Smoke temperature above the flame follows the plume law (Sec. 4.3). At 1 m above a 50 kW fire it is ~220 K above ambient;
  at 2 m it is ~80 K above ambient. Smoke is therefore still strongly buoyant across the whole 1.5 m domain.

### 2.4 Temperatures in detail

- **Adiabatic flame temperature** of wood-like fuels in air: ~2200 K at constant pressure (Wikipedia); wood-syngas
  calculations give 1850-2250 K.
  Check with Thornton's rule (≈ 13.1 MJ released per kg of O2 consumed): a stoichiometric mix of 1 kg volatiles with
  ~5 kg air contains 1.165 kg O2, which releases ≈ 15.3 MJ. Spread over 6 kg at c_p ≈ 1300 J/(kg K), that gives
  ΔT ≈ 1960 K, so T ≈ 2250 K before dissociation **[computed]**.
- **Real turbulent flames never reach that.** Soot radiation removes 20-40% of the heat (radiative fraction for wood is
  roughly 0.3 **[typical]**), and turbulent mixing dilutes the sheet. Local instantaneous peaks are ~1400-1800 K **[est]**.
- **Time-averaged centreline temperature** in the continuous flame zone is constant with height and Q-independent.
  McCaffrey's constants give ΔT/T∞ = (κ/C)²/(2g) = (6.8/0.9)²/19.62 = **2.91**, so ΔT ≈ 850 K and T ≈ 1150 K **[computed]**.
  The time average includes cold-air intermittency, so instantaneous values swing widely around it.
  McCaffrey notes that bare thermocouples read about 20% low at 1300 K (≈ 1000 K reading) because of radiation error.
- **Visible flame tip:** ΔT ≈ 320 °C (McCaffrey: the intermittent-plume boundary z/Q^(2/5) = 0.20 coincides with the tip
  seen in a 1 s exposure). In other words, where the mean gas is ~600 K, the visible flame is gone.
- **Soot-luminous zone:** ~1100-1500 K. Because visible luminance scales like T^20 (Sec. 2.6), anything below ~1000 K is
  visually negligible next to 1300 K gas. That is why flames have sharp-looking edges even though the temperature field is smooth.

### 2.5 Pyrolysis gas, ignition and extinction thresholds (for the burn model)

- Pyrolysis gas leaves the char at ~600-800 K **[est]**.
- Piloted ignition of the volatiles happens readily wherever existing flame or hot gas above ~600-700 K meets a
  flammable mixture. Autoignition of the volatile mix needs roughly 700-900 K **[est]**.
  In a graphics model, **T_ign ≈ 550-700 K** works well and keeps flames anchored to the logs.
- Extinction happens when strain or heat loss is too high: flame tips that get stretched and diluted go out,
  which is where soot escapes as smoke.

### 2.6 Light emission: colour and brightness of hot soot

Emission per unit volume is `j_λ = κ_λ(f_v) · B_λ(T)` (Planck). The table below was computed by integrating Planck
spectra against the CIE 1931 colour-matching functions (Wyman-Sloan-Shirley 2013 analytic fit), then converting
XYZ → linear sRGB and normalising so the largest channel is 1.

| T (K) | Rel. luminance (1500 K = 1) | Blackbody, linear sRGB | Soot (λ^-1.39), linear sRGB | Blackbody, eye adapted to 2000 K white (Bradford) |
|---|---|---|---|---|
| 800 | 1.1e-6 | 1, 0.000, 0 | 1, 0.000, 0 | 1, 0.045, 0.002 |
| 900 | 2.6e-5 | 1, 0.009, 0 | 1, 0.018, 0 | 1, 0.084, 0 |
| 1000 | 3.5e-4 | 1, 0.029, 0 | 1, 0.042, 0 | 1, 0.130, 0 |
| 1100 | 3.0e-3 | 1, 0.050, 0 | 1, 0.067, 0 | 1, 0.185, 0.004 |
| 1200 | 1.8e-2 | 1, 0.073, 0 | 1, 0.094, 0 | 1, 0.246, 0.019 |
| 1300 | 8.4e-2 | 1, 0.097, 0 | 1, 0.122, 0 | 1, 0.313, 0.048 |
| 1400 | 0.32 | 1, 0.121, 0 | 1, 0.150, 0 | 1, 0.387, 0.095 |
| 1500 | 1.0 | 1, 0.145, 0 | 1, 0.179, 0 | 1, 0.467, 0.163 |
| 1600 | 2.8 | 1, 0.170, 0 | 1, 0.208, 0 | 1, 0.555, 0.257 |
| 1800 | 15 | 1, 0.218, 0 | 1, 0.267, 0.006 | 1, 0.756, 0.542 |
| 2000 | 60 | 1, 0.266, 0.008 | 1, 0.325, 0.031 | 1, 1, 1 |

Takeaways:
- **Brightness is extremely temperature-sensitive:** d ln L / d ln T ≈ 20 at 1200 K and ≈ 14 at 1800 K.
  A ±5% temperature swing changes brightness by a factor of 0.36 to 2.7. 1000 K gas is 50× dimmer than 1200 K gas,
  and 800 K gas is 17,000× dimmer. Use physical emission, not a linear temperature-to-colour ramp. That steepness is what produces
  crisp flame edges and violent flicker.
- **Unadapted blackbody colours are deep red-orange.** Fire looks yellow-orange because:
  (a) the eye or camera white-balances toward a warm source (Nguyen et al. 2002 use a von Kries transform with white =
  the fire's maximum-temperature spectrum);
  (b) bright cores saturate or clip in the camera toward yellow-white.
  For video, combine physical emission, warm white balance (~2000-3000 K) and a filmic shoulder.
- **Blue base colour:** CH\* alone is at chromaticity xy = (0.165, 0.007), far outside sRGB (linear ≈ 0.12, 0, 1).
  A CH\* + C2\* mix sits at xy ≈ (0.16, 0.27), linear sRGB ≈ (0, 0.66, 1) **[computed]**.
  Use a deep blue to cyan-blue emission colour such as linear (0.05-0.15, 0.25-0.6, 1). Scale its intensity with the
  local **heat-release rate**: CH\*/OH\* chemiluminescence is the standard optical marker of heat release.
  Keep it 10²-10³ times dimmer than the soot-luminous flame body.
- **Embers** at 1100-1250 K: adapted colour ≈ (1, 0.19-0.28, 0.01-0.03), deep orange.

---

## 3. Fluid mechanics of fire

### 3.1 Governing equations (low-Mach, variable density)

```
∂ρ/∂t + ∇·(ρu) = 0
ρ Du/Dt = −∇p' + (ρ − ρ∞) g + ∇·τ
ρ c_p DT/Dt = q'''_chem − q'''_rad + ∇·(k∇T)
p0 = ρ R T   (thermodynamic pressure constant; p' ≪ p0)
```

### 3.2 Expansion: why "gas release = positive divergence" is physics, not a hack

From the low-Mach equations with an ideal gas at constant thermodynamic pressure:

```
∇·u = (1/T) DT/Dt = (q'''_chem − q'''_rad + ∇·k∇T) / (ρ c_p T) = (R / (p0 c_p)) (q'''_chem − q'''_rad + …)
     ≈ q''' / 3.55e5   [s⁻¹, with q''' in W/m³]   (since ρT = p0/R)
```

- The divergence depends **only on the volumetric heat-release rate**, not on the local temperature.
  For example, 1 MW/m³ gives ∇·u ≈ 2.8 s⁻¹. Inside the thin reaction sheet it is much higher.
- Integrated along a parcel's path, ∫∇·u dt = ln(T_final / T_initial). Burning from 300 K to 2000 K means a 6.7× volume
  increase (ln ≈ 1.9). If that happens in ~50 ms of local burning, ∇·u ≈ 38 s⁻¹ in the burning cells.
- Radiative cooling gives *negative* divergence (contraction) — small, but physically consistent.
- **Implementation:** set the pressure-projection target `∇·u = φ` with `φ = (ΔT_burn / Δt) / T` (Feldman et al. 2003
  solve `∇²p = (ρ/Δt)(∇·u* − φ)`). This keeps expansion exactly consistent with the temperature rise.
  Expansion is what makes flames look "full", pushes fresh gas outward across the front, and drives the rolling at the base
  (Nguyen 2002 Fig. 7; Houdini "Expansion"; FumeFX "Expansion"; Houdini Pyro "Gas Released").

### 3.3 Buoyancy: Boussinesq vs non-Boussinesq

The exact buoyancy term per unit mass is `(ρ∞ − ρ)/ρ · g = g (T − T∞)/T∞` (ideal gas, equal pressure).
At 1150 K that is ≈ 29 m/s² (≈ 3 g). The textbook Boussinesq form `g (ρ∞ − ρ)/ρ∞ = g (T − T∞)/T` is capped at 1 g
and gives far too little acceleration in flames.

**Calibration from McCaffrey's data:**
- In the continuous zone u = 6.8 sqrt(z), so u² = 46 z, which means a constant effective acceleration of **23 m/s²**.
- McCaffrey's buoyancy relation `V = C sqrt(2 g z ΔT/T∞)` with **C ≈ 0.9 in all three zones** implies
  a_eff = C² g ΔT/T∞ = 0.81 · g ΔT/T∞ (= 23 m/s² for ΔT = 850 K).

So in a constant-density incompressible solver use:

```
f_buoy = β · g · (T − T∞)/T∞ · ẑ ,   β ≈ 0.7-1.0   (start at 0.8)
equivalently α = β g/T∞ ≈ 0.027 m s⁻² K⁻¹ in Nguyen's f = α (T − T∞) ẑ
```

For comparison, Nguyen et al. used α = 0.15 m/(K s²) for a 30 m/s flamethrower, which is far stronger.
A good regression test is to measure V/sqrt(2 g z ΔT/T∞) on the time-averaged centreline: it should come out ≈ 0.9.

### 3.4 Vorticity generation (where the swirls come from)

```
Dω/Dt = (ω·∇)u − ω(∇·u) + (∇ρ × ∇p)/ρ² + (ρ_A/ρ²)(∇ρ × g) + ν∇²ω
```
(Xia & Zhang 2018, Eq. 1)

- The **gravitational-baroclinic term ∇ρ × g** is the source of essentially all large-scale fire vorticity. It is
  non-zero wherever the density gradient is horizontal, i.e. on the flame's side walls.
  It builds a vortex sheet along the flame surface that rolls up into the toroidal puffing vortex (Sec. 5).
- The **dilatation term −ω(∇·u)** weakens vorticity in expanding gas. That partly explains why the hot core is
  smoother ("laminarised") and why wrinkles live at the edges.
- Both effects appear automatically in a constant-density solver that has temperature-driven buoyancy plus a
  divergence source. Vorticity confinement only puts back what numerical diffusion removes.

### 3.5 Dimensionless groups

- Froude/heat-release number: `Q* = Q / (ρ∞ c_p T∞ sqrt(g D) D²)`. Campfires have Q* ≈ 0.15-0.5
  (20 kW/0.4 m → 0.18; 50 kW/0.5 m → 0.26; 100 kW/0.6 m → 0.32), the regime of "normal" pool-fire-like flames
  with L/D ≈ 1-1.5 **[computed]**.
- Richardson number `Ri = (1 − ρ_A/ρ_f) g D / U0²` is effectively ∞ for wood fires (U0 ≈ 0).
- Strouhal number `St = f D / U0` against Froude `Fr = U0²/(g D)`: pool fires follow St ∝ Fr^-1/2, which is the same
  statement as f ∝ sqrt(g/D).

---

## 4. Plume zones and flame height

### 4.1 McCaffrey (1979), NBSIR 79-1910: purely buoyant diffusion flames

Measurements: 0.3 m porous burner, natural gas, Q = 14-58 kW, time-averaged centreline data.

```
u0 / Q^(1/5) = κ (z / Q^(2/5))^η
ΔT0 / T∞ = (κ / C)² / (2g) · (z / Q^(2/5))^(2η − 1),   C = 0.9
(z in m, Q in kW, u in m/s)
```

| Zone | z/Q^(2/5) [m kW^-2/5] | κ | η | Centreline u | Centreline ΔT |
|---|---|---|---|---|---|
| Continuous flame | < 0.08 | 6.8 m^1/2/s | 1/2 | ∝ z^1/2 (accelerating from 0) | constant ≈ 850 K |
| Intermittent | 0.08-0.20 | 1.9 m/(s kW^1/5) | 0 | constant ≈ 1.9 Q^1/5 | ∝ z^-1 |
| Plume | > 0.20 | 1.1 m^4/3/(s kW^1/3) | −1/3 | ∝ z^-1/3 | ∝ z^-5/3 |

The values above are exactly as printed in McCaffrey's summary table, checked against the scanned report text.

Observations quoted from the report:
- The intermittent zone shows "clumps of visible flame … rise periodically … at a regular frequency of a few Hz"
  (about 3 Hz for his 0.3 m burner). The cycle is **eruption, stretching, and breaking of the flame envelope**.
- For 33 kW the zone boundaries are at 0.3 m and 0.8 m.
- The visible tip in a 1 s exposure coincides with z/Q^(2/5) = 0.2, where ΔT ≈ 320 °C.
- The pattern velocity of flame clumps (1.1-1.8 m/s) is 2.5-3× slower than the gas velocity (3.3-4.3 m/s).
  **The visible flame structures should rise noticeably slower than the gas.** In a sim this emerges because the
  structures are vortices and burning fronts, not passive tracers.
- Low-frequency pressure (velocity) fluctuations are about 35% of the mean on the centreline and approach 100% at the edges.
  Radial profiles are not Gaussian in the flame zones, and the velocity profile is wider than the temperature profile.
- Without soot as a tracer, the inward **necking and pinching** of the flame is hard to see. Sooty fires show it clearly.

### 4.2 Heskestad flame height and virtual origin

```
L = 0.235 Q^(2/5) − 1.02 D      [m; Q in kW; D = equivalent fuel-bed diameter in m]
  (dimensionless form: L/D = −1.02 + 3.7 Q*^(2/5))
z0 / D = −1.02 + 1.4 Q*^(2/5)   (virtual origin; equivalently z0 = 0.083 Q^(2/5) − 1.02 D)
```

- L is the mean (50% intermittency) luminous height. Secondary sources give a validity range of roughly
  7 < Q^(2/5)/D < 700 kW^(2/5)/m.
- Zukoski, Cetegen & Kubota (1985; 0.10-0.50 m burners, 10-200 kW) found that for flames taller than about three
  diameters the burner diameter stops mattering. They describe "large scale axisymmetric structures" formed near the
  burner "with more or less regular frequency" that rise through the flame and make the tip fluctuate.

### 4.3 Plume above the flames (Heskestad; NIST TN 1984 reproduces the equation)

```
ΔT0 = 9.1 (T∞ / (g c_p² ρ∞²))^(1/3) Qc^(2/3) (z − z0)^(−5/3)   → ≈ 25 Qc^(2/3)(z − z0)^(−5/3) K
u0  = 3.4 (g / (c_p ρ∞ T∞))^(1/3) Qc^(1/3) (z − z0)^(−1/3)        → ≈ 1.03 Qc^(1/3)(z − z0)^(−1/3) m/s
Qc = convective HRR ≈ 0.6-0.75 Q
```

Worked example, 50 kW, D = 0.5 m, Qc = 35 kW, z0 = −0.11 m **[computed]**:

| z (m) | ΔT0 (K) | u0 (m/s) |
|---|---|---|
| 1.0 | 224 | 3.25 |
| 1.5 | 121 | 2.87 |
| 2.0 | 77 | 2.63 |
| 3.0 | 40 | 2.31 |

### 4.4 Campfire design table **[computed]**

| Q (kW) | D (m) | Heskestad L (m) | Continuous to (m) | Intermittent to (m) | Peak mean u (m/s) | f_puff (Hz) |
|---|---|---|---|---|---|---|
| 20 | 0.4 | 0.37 | 0.27 | 0.66 | 3.5 | 2.4 |
| 50 | 0.5 | 0.61 | 0.38 | 0.96 | 4.2 | 2.1 |
| 100 | 0.6 | 0.87 | 0.50 | 1.26 | 4.8 | 1.9 |
| 200 | 0.8 | 1.14 | 0.67 | 1.67 | 5.5 | 1.7 |

**Recommended hero fire for a 1.5 m tall domain: Q ≈ 50-80 kW, D ≈ 0.5 m.**
The continuous flame reaches about 0.4 m, intermittent tongues reach about 1 m, and roughly 0.5 m of plume and smoke
stays in frame. The required pyrolysis rate is 50 kW / 14 MJ/kg ≈ 3.6 g/s, which over ~0.3 m² of flaming log surface
is ≈ 12 g/(m²·s) — realistic.

### 4.5 Intermittency model (for flame-tip statistics)

Let I(z) be the fraction of time flame is present on the axis.
- I ≈ 1 below 0.08 Q^(2/5).
- I passes 0.5 near the Heskestad L.
- I reaches ≈ 0 by about 0.2 Q^(2/5) (the tip seen in a 1 s exposure).

A usable fit **[est]** is `I(z) = 0.5 erfc((z/L − 1)/(√2 · 0.35))`.
Use it as a test: render a long exposure of the sim and compare against these statistics.

---

## 5. Puffing, vortex roll-up, necking and flame-tongue detachment

### 5.1 Frequency

```
f ≈ 1.5 / sqrt(D)  Hz      (D in m)   — Cetegen & Ahmed (1993), Combust. Flame 93:157; also Trefethen & Panton
f = 0.48 sqrt(g / D)                    — fit to pool-fire data incl. varied gravity (Xia & Zhang 2018)
Hamins et al. (1992): St ∝ Fr^-0.57 over the full jet + pool range; the buoyancy-dominated limit is exponent −0.5
```

Examples: D = 0.3 m → 2.7 Hz (McCaffrey observed about 3 Hz); D = 0.5 m → 2.1 Hz; D = 1 m → 1.5 Hz; candle → 10-15 Hz.
Frequency depends on g and on physical size, so **the simulation must use real metres, seconds and g = 9.81**.
A miniaturised domain flickers too fast and looks like a toy.

### 5.2 Mechanism (vortex dynamics)

1. Hot gas over the fuel bed accelerates upward (continuous zone, u ∝ sqrt(z)). By continuity the column **necks in**.
   Entrained air flows radially inward at the base. The neck sits roughly 0.5-1 D above the bed
   (about 1 D is reported for pool fires: arXiv 2202.06567).
2. The baroclinic term ∇ρ × g turns the flame's side wall into a **vortex sheet**. Its circulation grows at the rate
   `dΓ/dt = ρ_A g Δh (1/ρ_f − 1/ρ_A)`. That depends only on the vertical extent of the sheet, not its shape (Xia & Zhang 2018, Eq. 6).
3. The sheet rolls up into a **toroidal vortex** just above the bed: a Kelvin-Helmholtz-type instability of the buoyant
   shear layer. The vortex swells the flame base into a **bulge**.
4. When the non-dimensional circulation of the ring reaches a threshold (a vortex-ring formation-number analogue,
   C ≈ 25 for r* = T_f/T_A ≈ 6.7), the ring **detaches**. It rises, squeezes the neck, and **pinches off** the upper
   part of the flame. The detached **flame tongue/puff** is carried upward with the vortex, burns out within a fraction
   of a cycle, and leaves smoke and hot gas. A new bulge forms at the base and the cycle repeats.
   For pool fires this gives `f = sqrt((r* − 1) g / (C D))`.
5. In larger or more turbulent fires (D ≳ 1 m) the toroid breaks into smaller structures and the periodicity weakens.
   A broad spectral peak near 1.5/sqrt(D) remains, smeared by turbulence.

**Multiple flames (several logs):** neighbouring flames couple through their inner shear layers. Close together they
flicker in phase or merge; at intermediate spacing they lock **anti-phase**, similar to a Kármán street
(Xia & Zhang et al., Phys. Rev. Fluids 4, 053202, 2019). A campfire with several flaming sites therefore shows a
partially coherent, rocking flicker rather than one pulsing column. This comes for free if the sources are separate
patches on the logs.

### 5.3 What makes it look real

- The flame base always looks somewhat anchored and rolling: fresh blue-fringed fuel curls up the log and outward,
  driven by expansion and the inflow.
- Flame sheets **stretch upward** into long tongues, then **neck and tear**. Torn pieces burn out within ~0.1-0.3 s
  while rising at ~1-2 m/s.
- Tip height jumps intermittently, ±50% of L. The puffing frequency is clearly visible in the motion of the tip.
- Small-scale wrinkles (cm scale) ride on the large-scale cycle. Strain from the big eddies stretches the
  wrinkles into vertical striations.

---

## 6. Turbulence scales (what the grid resolves and what it cannot)

Campfire, Q ≈ 50 kW, D ≈ 0.5 m, u ≈ 3-4 m/s **[computed]**:

| Quantity | Value |
|---|---|
| Kinematic viscosity at 300 / 1000 / 1500 K | 1.6e-5 / 1.2e-4 / 2.2e-4 m²/s (hot gas is 7-14× more viscous) |
| Reynolds number, Re = uD/ν | ~1e4 in hot gas, ~1e5 in entrained cold air |
| Dissipation, ε ~ u³/L | ~50 m²/s³ |
| Integral scale / time | ~0.3-0.5 m / ~0.1-0.15 s |
| Taylor microscale | ~1.5-2 cm |
| Kolmogorov scale η = (ν³/ε)^1/4 | ~0.3-0.5 mm (hot); time scale ~1.5 ms |
| Eddy velocity at scale ℓ, (εℓ)^1/3 | 5 cm → ~1.4 m/s; 2 cm → ~1.0 m/s |
| Eddy turnover at scale ℓ, ℓ^2/3 ε^-1/3 | 5 cm → ~37 ms; 2 cm → ~20 ms |
| Laminar flame thickness | ~1 mm |

Implications:
- A 1.5 m domain at 64-192 cells has h ≈ 0.8-2.3 cm, so the effective resolution (3-4h) is ≈ 3-8 cm.
  The grid resolves only the top ~1-1.5 decades of an inertial range that spans ~3 decades.
- **The puffing vortex and the large tongues are resolvable.** The cm and mm wrinkling of the flame sheet is not.
  It has to come from procedural detail (curl noise, wavelet-turbulence-style upsampling, or noise in the renderer)
  following a Kolmogorov spectrum: E(k) ∝ k^-5/3, eddy velocity ∝ ℓ^1/3, eddy lifetime ∝ ℓ^2/3.
- Sub-frame scales (< 20 ms) are invisible at 24-60 fps except as **motion blur**. Use a physically plausible shutter
  (180° → 1/48 s at 24 fps) for the video.

---

## 7. Combustion models used in graphics, and their physical meaning

### 7.1 Nguyen, Fedkiw & Jensen 2002, "Physically Based Modeling and Animation of Fire" (SIGGRAPH)

- **Thin-flame model.** A level set φ marks the blue reaction core (φ > 0 = gaseous fuel). It moves with
  `w = u_f + S n`, where S is the flame speed (≈ 0.44 m/s for premixed propane; they use **S = 0.5 m/s** for most examples
  and **S = 0.1 m/s** for the flamethrower). A smaller S gives a larger blue core, which is how they mimic diffusion flames.
  Blue-core area estimate: `v_f A_f = S A_S`.
- **Expansion across the front.** The fuel and hot products are two incompressible fluids with jump conditions
  `ρ_h(V_h − D) = ρ_f(V_f − D)` and `ρ_h(V_h − D)² + p_h = ρ_f(V_f − D)² + p_f`.
  Ghost velocity: `V_h^G = V_f + (ρ_f/ρ_h − 1) S`.
  They use ρ_f = 1 kg/m³ with ρ_h = 0.2 / 0.1 / 0.05 kg/m³ (more expansion → fuller flames).
  For solid fuel, gas is injected along the surface normal at `V_f = V_s + (ρ_s/ρ_f − 1) S`. Their campfire uses
  two cylindrical logs that inject gas along their normals.
- **Reaction coordinate** `Y_t = −(u·∇)Y − k`, with k = 1 and Y = 1 in the fuel. Then 1 − Y = time since the fluid
  element crossed the blue core. The temperature curve is set from Y: T = T_ign at Y = 1, rising (for example linearly)
  to T_max at Y = 0.9, then cooling. For large flames they emit at T_max directly from the core.
- **Cooling** `T_t = −(u·∇)T − c_T ((T − T_air)/(T_max − T_air))^4`, integrated analytically.
- **Buoyancy** `f = α(T − T_air) z`.
- **Vorticity confinement** `f = ε h (N × ω)`.
- **Flamethrower parameters:** 8 m domain, 160 cells (h = 5 cm), 30 m/s injection, 0.4 m pipe, S = 0.1 m/s,
  ρ_f = 1, ρ_h = 0.01 kg/m³, **c_T = 3000 K/s**, **α = 0.15 m/(K s²)**, **ε = 16 (fuel), 60 (hot products)** (so εh = 0.8 and 3 m/s).
  Grids up to 120³. The level set is substepped, with the flow step ~5× the level-set CFL step.
- **Rendering.** Stochastic ray marching through a participating medium with a Henyey-Greenstein phase function.
  Emission is Planck (C1 ≈ 3.7418e-16 W m², C2 ≈ 1.4388e-2 m K), converted to XYZ. Chromatic adaptation is a
  **von Kries** transform to the white point of the fire's maximum-temperature spectrum.
- **Physics check:** c_T = 3000 K/s is close to real optically-thin soot radiative cooling at 1300 K and 1 ppm
  (≈ 4700 K/s; Sec. 8.2). The ^4 shape is a sensible proxy, though the true soot dependence is ~f_v T^5.

### 7.2 Feldman, O'Brien & Arikan 2003, "Animating Suspended Particle Explosions" (SIGGRAPH)

- Incompressible Euler flow with a **divergence source**: `∇·u = φ`, solved as `∇²p = (ρ/Δt)(∇·u − φ)`.
- Temperature equation: `Ṫ = −(u·∇)T − c_r((T − T_a)/(T_max − T_a))^4 + c_k∇²T + Ḣ/(ρ c_v)`. They deliberately use a
  large c_k so the diffusion term also stands in for radiative transfer.
- **Fuel particles** have mass, temperature and thermal mass.
  - They ignite above an ignition temperature and burn at a constant rate z.
  - Heat release: `Ḣ = b_h z`.
  - Gas release into the cell: `Δφ = b_g z / V`.
  - Soot production: `ṡ = b_s z`, emitted as new soot particles.
  - Drag: `f = α_d r² (ū − ẋ)|ū − ẋ|` (quadratic).
  - Heat exchange: `Ḣ = α_h r² (T̄ − Y)`.
- Table 2 values were picked by eye (c_r = 0-2000, c_k = 0-10, b_h up to 3·10^5, …), with grids of 35-75 cells at
  h = 0.13-0.5 m. Initial velocities are perturbed with small random values to avoid symmetry.
- **Lesson for us:** particle-carried fuel combined with a grid divergence source is a clean way to add sparks, burning
  debris or fuel sprays. The same φ bookkeeping applies to gas fuel on the grid.

### 7.3 Horvath & Geiger 2009, "Directable, High-Resolution Simulation of Fire on the GPU" (SIGGRAPH; ILM patents US8289327, US8760449)

- **Stage 1:** a coarse 3D particle-grid simulation (≤ 50³ grid, 20k-100k particles; each particle carries position,
  velocity, radius, fuel, mass, impulse, age and a validity flag). Artists direct it through particle behaviour.
- **Stage 2:** many **view-aligned 2D slices** (16-32 for a torch or fireball, 96-128 for medium effects, 500-2000+ for
  very complex shots; up to 2048² each). Particles are splatted into the slices with distance weighting.
  - Each slice runs an independent 2D incompressible NS solve on the GPU: semi-Lagrangian advection of density,
    temperature, fuel, a "texture" detail channel and velocity; Jacobi pressure; buoyancy, turbulence, vorticity
    confinement and damping.
  - Slices ignore coupling along the view axis, which is invisible from the camera.
- **Rendering:** light from temperature through a blackbody model; opacity = exp(−density × slice spacing); motion blur
  from the stored velocity.
- **Lesson for us:** fine detail mostly has to look right only perpendicular to the view. For the offline video, a
  camera-aligned 2D refinement layer (or noise advected in screen space) is a cheap path to film-like detail on top of a
  coarse 3D sim.

### 7.4 Lamorlette & Foster 2002, "Structural Modeling of Flames for a Production Environment" (TOG 21(3))

Flames are modelled as parametric spines in a wind field, with stochastic flicker, buoyant-diffusion models and
Kolmogorov noise for detail (the PDI approach used in *Shrek*). It is the artist-friendly alternative to grids and a
reminder that **Kolmogorov-spectrum noise plus a correct flicker rate** carry much of the realism.

### 7.5 Production solvers: Houdini Pyro, EmberGen, FumeFX, TurbulenceFD

Common field set: velocity, **fuel**, **temperature**, **flame / heat** (the burn "lifetime"), **density/soot**,
**divergence**.

**Houdini Pyro Solver 2.0 (classic) combustion block** (SideFX docs):

| Parameter | Behaviour | Physical meaning |
|---|---|---|
| Ignition Temperature | burn only where T > threshold (negative value = always ignite) | piloted ignition |
| Burn Rate | fraction of fuel burned per second (0.9 → 90% gone after 1 s, i.e. exponential k = −ln(0.1) = 2.3 s⁻¹) | mixing-limited consumption rate |
| Fuel Inefficiency | 0 = burned fuel removed; 1 = fuel never depleted | incomplete combustion |
| Temperature Output | temperature added per unit fuel burned | ΔH_c / c_p |
| Gas Released | divergence added where fuel burns | expansion, ∇·u = q'''/(ρ c_p T) |
| Flame Height | unitless; its inverse sets cooling of the "heat" (flame) field — lower value = faster cooling = shorter flames | how long burning gas stays visibly luminous: soot burnout plus mixing time |
| Smoke emission | from the burn field or where the heat field drops below a cutoff (e.g. 0.2), with a blend falloff | soot escaping at the flame tip |
| Buoyancy Lift / Direction | force ∝ (T − T_ambient) | Sec. 3.3 |
| Cooling Rate | fraction of temperature lost per second (0.9 → 90%) | entrainment and radiation; ≈ 2-3 s⁻¹ matches the plume decay in Sec. 4.3 **[computed]** |
| Dissipation | fraction of smoke removed per 1/24 s | dilution below visibility |
| Shredding | rotates velocity by temperature gradient: stretch below a threshold, squash above it | flame-sheet tearing proxy |
| Disturbance | block-based random forces (Block Size, Locality) gated by a field cutoff | sub-grid turbulence |
| Turbulence | curl noise (Swirl Size, Grain, Pulse Length, levels that halve size) | large-scale noise injection |
| Confinement | vorticity confinement with a control-field ramp (more where curl is high) | numerical-dissipation compensation |

**Houdini Sparse Pyro (H18+):**
- A **flame** field holds the "remaining lifetime of reactants".
- **Flame Lifespan** L_f is the seconds needed to consume a flame value of 1 (depletion takes L_f · f seconds).
- Outputs are driven from flame within a **Flame Range** remapped to 0-1: Temperature (Add, or Pull with Strength),
  **Expansion** (goal divergence, set by Expansion Rate; negative values pull inward), and Smoke emission
  (Max or Add per 1/24 s).
- Buoyancy uses Ambient and Reference temperatures (K) with gravity.
- **Physical calibration for a campfire [computed]:** flame lifespan ≈ (visible flame length)/(rise speed)
  ≈ 0.6 m / (2-3 m/s) ≈ **0.2-0.35 s**.

**EmberGen and FumeFX:** both use the same fuel → heat + expansion ("gas pressure release" in EmberGen since 0.5.6) → smoke
model, with blackbody colouring of hot smoke and ambient-temperature boundaries (EmberGen release notes and JangaFX forum).

### 7.6 Vorticity confinement and turbulence injection: how much

- **Confinement** `f = ε h (N × ω)`, N = ∇|ω|/|∇|ω||. ε has units of 1/s; εh has units of m/s.
  Nguyen used εh = 0.8-3 m/s for a 30 m/s jet. For a buoyant campfire with u ≈ 4 m/s, scale down to
  **εh ≈ 0.1-0.3 m/s**: ε ≈ 10-30 s⁻¹ at h = 1 cm. Rules:
  - Ramp it up in the 500-1300 K tongue and plume region.
  - Keep it near 0 in the first 2-3 cells above the source, or the base turns to noisy "boiling noodles".
  - Keep the confinement acceleration below ~30-50% of local buoyancy (~20 m/s² in the flame).
- **Curl-noise forcing** (divergence-free; Bridson et al. 2007):
  - Base scale ℓ0 ≈ 6-10 cm with 2-3 octaves (ℓ halves each octave, amplitude ∝ ℓ^1/3).
  - Amplitude ≈ 0.3-0.8 m/s at ℓ0, correlation time τ ≈ 0.1 s × (ℓ/0.1 m)^2/3.
  - Gate it by the burn/flame field and temperature.
  - Inject only below the grid's own resolved range (ℓ ≲ 6-8 h), otherwise you fight the resolved puffing.
- **Advection quality matters more than either.** MacCormack/BFECC with clamping (Selle et al. 2008) roughly halves
  the confinement you need and keeps flame edges sharp.

---

## 8. Embers and sparks

### 8.1 What they are

Glowing fragments of **char** (plus some burning resin droplets and flying bark). They are released by:
- collapsing coals;
- steam or resin "pops" that throw fragments out (Sec. 9);
- convective lift of loose char.

Release is **bursty**: firebrand generation is "highly intermittent, occurring in large bursts" (UC Irvine, via PreventionWeb).
Wildfire firebrands are larger, ~3-4 mm × 40-53 mm cylinders (NIST Douglas-fir burns). Campfire sparks are mostly
**0.2-2 mm**, with some 2-5 mm flakes.

### 8.2 Physics per particle

```
m dv/dt = m g (1 − ρ_g/ρ_p) + (m/τ_p) · F(Re) · (u_gas − v)
τ_p = ρ_p d² / (18 μ_g),   F(Re) = 1 + 0.15 Re^0.687 (Schiller-Naumann),   Re = ρ_g |u − v| d / μ_g
ρ_p ≈ 150-350 kg/m³ (charred wood char; decreases while burning)
```

Terminal settling velocity in still air (my Schiller-Naumann solution, Sutherland viscosity) **[computed]**:

| d | ρ_p = 150, 300 K air | ρ_p = 300, 300 K air | ρ_p = 300, 1000 K gas |
|---|---|---|---|
| 0.2 mm | 0.14 m/s | 0.26 m/s | 0.15 m/s |
| 0.5 mm | 0.54 m/s | 0.89 m/s | 0.74 m/s |
| 1 mm | 1.14 m/s | 1.80 m/s | 1.94 m/s |
| 2 mm | 2.2 m/s | 3.3 m/s | 4.2 m/s |

Stokes response time at 600 K: 0.2 mm → 0.01-0.02 s; 0.5 mm → 0.07-0.14 s; 1 mm → 0.3-0.55 s; 2 mm → 1.1-2.2 s.

Consequences:
- The plume's 3-5 m/s **lifts everything below about 1-1.5 mm** easily.
- Small sparks (< 0.5 mm) follow eddies closely. They trace the swirls, which makes them good flow visualisers.
- Larger sparks (1-2 mm) lag and overshoot. They take ballistic arcs, leave the plume sideways when it slows or
  turns, and some fall back.
- Gravity on the particle matters only for the big ones.

**Temperature and lifetime (the key insight)** **[computed]:**
- An *inert* 1 mm char sphere at 1200 K cools below the Draper point (800 K) in only **~0.06-0.13 s**.
  Radiative h ≈ 350 W/m²K and convective h ≈ 200 W/m²K at d = 1 mm; at 0.2 mm, under 10 ms.
- Sparks that stay visible for seconds must therefore be **burning**. Char oxidation heat balances the losses and
  holds the surface at ~1000-1300 K (colour-pyrometry embers 920-955 °C; hotter in faster airflow).
- Diffusion-limited char burnout: `t_b ≈ ρ_p d0² / (8 ρD ln(1 + B))` with B ≈ 0.175 (C → CO) and ρD ≈ 5e-5 kg/(m s).

| d0 | t_b at ρ_p = 150 | t_b at ρ_p = 300 |
|---|---|---|
| 0.2 mm | 0.09 s | 0.19 s |
| 0.5 mm | 0.6 s | 1.2 s |
| 1 mm | 2.3 s | 4.7 s |
| 2 mm | 9 s | 19 s |

Forced convection shortens these (burning rate ∝ U^1/2; CSIRO firebrand-under-wind study).
**Practical rule [est]:** t_b ≈ 1.5-2.5 s × (d/1 mm)² × (ρ_p/200); then extinguish and cool in 0.05-0.3 s.

- **Colour and brightness:** use the blackbody at the particle temperature (Sec. 2.6). Because L ∝ T^20, a 1300 K spark
  is ~5× brighter than a 1200 K one and ~30× brighter than a 1100 K one. Real spark populations have a few blazing
  yellow-white sparks and many dim orange-red ones. Sample T0 from about N(1180 K, 60 K).
- **Twinkle:** tumbling, non-uniform burning and ash spalling modulate brightness by roughly ±20-40% at 5-20 Hz **[est]**.
  Fragmentation (a spark splitting in two, near-instant "branching") happens occasionally, more often in conifer wood.
- **Ash:** after burnout, a grey ash speck may drift on in the plume; it is invisible against flame but visible against light.

### 8.3 Population, trajectories and rendering **[est, observational]**

- **Rate:** about 0-5 sparks/s from a calm fire; bursts of 20-100 within 0.1-0.3 s when a log shifts or a resin
  pocket pops. Tie bursts to the audio crackle events.
- **Sizes:** log-normal with median ≈ 0.5 mm, σ_ln ≈ 0.5, clamped to 0.15-2.5 mm. Most are tiny and short-lived.
- **Trajectories:** launch from coals at 0.5-3 m/s plus a random kick (pops: 2-6 m/s). Particles accelerate into the
  plume and swirl with the puffing vortices.
  - Small sparks die within the first ~0.2-0.5 m.
  - 0.5-1 mm sparks rise 1-3 m over 0.5-2 s and drift with any breeze.
  - The largest arc out sideways.
- **Rendering:** sparks are sub-pixel. Render them as motion-blurred **streaks** with length = velocity × shutter
  (at 3 m/s and 1/48 s: 6 cm). Use energy-conserving intensity: radiant power ∝ d² T^4 in-band, spread along the streak.
  Optional: let them light the scene as tiny point lights.

---

## 9. Wood campfire specifics

### 9.1 Char and glowing surfaces

- **Ember and char surface temperature:**
  - Colour-camera ratio pyrometry: mean **931 ± 6 °C (≈ 1204 K)** (UMD).
  - Under 1 m/s simulated movement 922 °C, under 2 m/s 955 °C (UMD). A 33 K change means about **1.7× brightness**,
    so coals visibly "breathe" with every gust and every puffing inflow pulse.
  - A mapping study reports a 750-950 °C range.
  - Bare thermocouples read more than 200 °C low.
- The surface **emissivity changes** as char turns to ash, and ash is shed intermittently (Berkeley CFP). Grey-white ash
  flakes partly veil the glow. When an ash flake drops off, a fresh orange-yellow patch brightens.
- **Charring rate:** for reference, ~0.65 mm/min for softwood under the standard fire exposure (EN 1995-1-2 design value).
  Campfire logs facing high radiant flux char faster **[typical]**.

### 9.2 Crack patterns ("alligatoring")

- The **primary macro-crack pattern** forms early, *below* pyrolysis temperatures, through a thermo-mechanical buckling
  instability of the hot surface layer on the cold core.
  - In fir the main cracks run **perpendicular to the grain** with **roughly constant spacing**.
  - Isotropic boards (MDF) form 2D polygonal patterns.
  - **Secondary** cracks form chemically above pyrolysis temperature.
  (Baroudi et al., arXiv 1604.01249)
- Together these give the familiar **rectangular "alligator" blocks**: cross-grain cracks plus shrinkage cracks along
  the grain. Block size grows with char depth, from a few mm on fresh char to several cm on deep char **[est]**.
- **Procedural recipe [est]:**
  - Anisotropic Voronoi or cell noise stretched 1.5-3× along the grain.
  - Crack width 0.5-3 mm, widening as the log burns.
  - Crack floors **glow hotter** than block tops (+50-150 K) because of better O2 access and no ash.
  - Block faces: dark char with a slight sheen and a grey ash rim.
  - Glow temperature driven by local sim velocity and heat flux: brighter where flames lick and air flows in.

### 9.3 Crackle, pops and hiss (physics hooks for the audio team)

- **Mechanism:** water and sap in pores and resin pockets heat, vaporise and build pressure. When the pressure exceeds
  the local wood strength the pocket ruptures: a sharp **pop**, often throwing out a spark burst.
  - Wetter wood and resinous softwoods crackle more.
  - Seasoned hardwood crackles less. (Conversation/Wonderopolis level sources; the mechanism is well accepted.)
- **Acoustic character [est]:** individual fracture clicks are sub-millisecond, broadband impulses (energy mostly
  ~1-10 kHz), arriving roughly as a Poisson process at about 0.5-10 per second depending on moisture.
  Bigger pops add a short low-mid "thump" and spark ejection.
  Hissing or whistling comes from steam and volatile jets escaping through cracks and end grain; you can see sap bubbling
  at the log ends.
- **Combustion roar from the sim:** unsteady heat release radiates sound as an acoustic monopole:
  ```
  p'(r, t) = (γ − 1) / (4π r c0²) · dQ̇(t − r/c0)/dt        (Strahle; standard combustion-noise result)
  ```
  The sim's total HRR Q̇(t) gives the low-frequency envelope, including puffing modulation at ~2 Hz.
  Synthesize the audible band (~50-1000 Hz flutter/roar) as noise whose amplitude follows |dQ̇/dt| and the local burn
  rate. Visible puffs then line up with audible "whumps".

---

## 10. Recommended simulation pipeline (WebGPU, Radeon 780M; plus an offline video mode)

### 10.1 Domain, grid and budget

- **Domain:** 1.0 (W) × 1.5 (H) × 1.0 (D) m, fuel bed centred about 0.05 m above the floor.
  The 1.5 m height holds a 50-80 kW campfire (Sec. 4.4).

| Tier | Grid | h | Cells | Use |
|---|---|---|---|---|
| Low | 64 × 96 × 64 | 1.56 cm | 0.39 M | 60 fps with margin |
| Default | 80 × 120 × 80 | 1.25 cm | 0.77 M | 60 fps target on the 780M |
| High | 96 × 144 × 96 | 1.04 cm | 1.33 M | 30-60 fps |
| Video (offline) | 128 × 192 × 128 to 160 × 240 × 160 | 0.78-0.63 cm | 3.1-6.1 M | not real time; 4-8 substeps per frame |

- **Bandwidth is the bottleneck** **[est]:** the 780M shares ~90-120 GB/s of system memory bandwidth.
  - Store scalars in `r16float` and velocity in `rgba16float`; keep pressure in r32float.
  - One step costs roughly 300-600 B/cell, so ≈ 0.4-0.8 GB per step at 1.3 M cells → 5-10 ms.
  - Budget 2 substeps per frame at the default tier.
- **Time step:** u_max ≈ 5-6 m/s. Keep the semi-Lagrangian CFL ≤ 3:
  - h = 1.25 cm: Δt ≤ ~7 ms → 2-3 substeps at 60 fps.
  - Video tier: CFL ≤ 1.5.
  - Do *not* take one big 16.7 ms step: a CFL of ~10 smears the flame sheet and kills the necking.

### 10.2 Fields

| Field | Format | Notes |
|---|---|---|
| u (velocity) | rgba16f, collocated with a staggered-style projection, or MAC | m/s |
| p (pressure) | r32f, warm-started | |
| T (temperature) | r16f | K; or store θ = T/1000 for precision |
| f (fuel) | r16f | pyrolysis-gas mass fraction; 1 at the log surface |
| o (oxygen availability) | r16f | air = 1; consumed by burning; products ≈ 0 |
| S (soot) | r16f | ppm; drives emission and absorption |
| M (organic smoke) | r16f, optional | white/grey/bluish scattering aerosol from smouldering parts |
| A (gas age since leaving the fuel source) | r16f | Nguyen-style reaction coordinate (A = 1 − Y); reset to 0 in source cells, +Δt per step, advected; gates soot formation and the blue fringe |
| q (burn rate) | r16f, transient | drives expansion, blue chemiluminescence, audio, sparks |

### 10.3 Per-substep order

**1. Sources** (log voxels or SDF, coal bed)
- Fuel: in flaming patches set `f = max(f, f_src)` with f_src = 1.
  - Patch strength = 0.5-1 × slow noise (time scale 0.5-3 s; spatial scale 5-15 cm) × a log "burn-state" map.
  - Normal injection velocity v_n = 0.05-0.3 m/s (physical is ~2 cm/s; slightly more helps resolution).
  - Set the source cells to T ≈ 700-900 K.
  - Target the total HRR (Sec. 10.5) at 50-80 kW, i.e. ≈ 3.6-5.7 g/s of pyrolysis gas.
- Coal bed: cells at 1000-1150 K with a weak CO "fuel" (f_src ≈ 0.1-0.2, no soot) to get faint blue flames over the coals.
- Optional cool smouldering spots emit M (organic smoke) at T ≈ 500-600 K.

**2. Forces** (add to u)
- Buoyancy: `a_z = β g (T − T∞)/T∞` with **β = 0.8** (range 0.6-1.0), T∞ = 293 K.
  Optionally clamp a_z ≤ 5 g. Soot weight is negligible.
- Vorticity confinement: `ε h = 0.1-0.3 m/s`, weighted by smoothstep(500 K, 900 K, T) and by height above the source
  (0 in the first 2-3 cells).
- Curl-noise turbulence: ℓ0 = 6-10 cm, 2-3 octaves, u_ℓ0 = 0.3-0.8 m/s, τ ≈ 0.1 s, gated by q and T.

**3. Advect** u, T, f, o, S, M, A
- Use MacCormack/BFECC with min-max clamping; trilinear sampling of 16-bit float textures.
- For extensive quantities (S, M, f) apply `c *= exp(−φ Δt)` after advection, so expansion dilutes them consistently.

**4. Combustion** (fast-chemistry, mixing-limited)
```
ign   = smoothstep(T_ign − 50, T_ign + 50, T)            T_ign = 600 K (550-700)
rate  = k_mix · ign                                       k_mix = 20-50 s⁻¹ (higher = shorter, crisper flames)
b     = min(f, o / r_s) · (1 − exp(−rate Δt))             r_s = 4.7 (stoich "air units" per fuel unit)
f    -= b ;  o -= r_s · b
ΔT    = η · ΔT_f · b                                      ΔT_f ≈ ΔH_c / c_p ≈ 14e6 / 1250 ≈ 11 000 K per unit fuel
                                                          η = 0.6-0.8 (radiative fraction + incomplete burning)
T    += ΔT ;   q = ΔT / Δt (K/s) ;   A += Δt   (A = 0 in source cells; advected like a scalar)
φ     = γ_exp · q / T                                     γ_exp = 0.5-1.0 (1 = physical); clamp φ ≤ ~60 s⁻¹
```
- Check: a stoichiometric parcel (f = 1/(1 + r_s) ≈ 0.175) gets ΔT ≈ 0.7 × 1925 ≈ 1350 K, so it peaks near 1650 K.
  Mixing and cooling bring the zone mean down to ~1150 K, consistent with Sec. 2.4.
- Alternative (Houdini-like, simpler): drop o and burn `b = f(1 − e^(−kΔt))` wherever T > T_ign, with k ≈ 2-5 s⁻¹.
  It is cheaper, but flames become fuller fireball-like blobs with no fuel-rich core. The oxygen field is what gives
  long, hollow, sheet-like diffusion flames that tear.

**5. Soot** (ppm)
- Formation needs **age**, fuel richness and heat:
  ```
  dS/dt = +k_sf · f_rich · gate(T, 1300 K, 1800 K) · gate(A, 0.02 s, 0.06 s)
  ```
  with k_sf tuned so S reaches 2-4 ppm in the flame core (f_rich ≈ f/(f + o/r_s)).
  The age gate is what creates the **blue base**.
- Oxidation:
  ```
  dS/dt = −k_ox · S · o · gate(T, 1100 K, 1300 K),   k_ox = 10-30 s⁻¹
  ```
  This ends the luminous flame abruptly at the tip.
- Whatever soot survives once T < ~1000 K becomes visible smoke (dark). Optionally decay it slowly (0.05-0.2 s⁻¹) as
  sub-visible dilution.

**6. Cooling**
- Radiative (optically thin soot, then attenuated):
  ```
  dT/dt = −χ · 970 · (S / 1 ppm) · θ² (θ⁴ − θ∞⁴)   [K/s],  θ = T/1000 K,  χ = 0.3-0.6 (self-absorption)
  ```
  Derivation: κ_P = 1817 f_v T and ρ c_p = p0 c_p/(R T). At 1300 K and 1 ppm this is ≈ 4700 K/s × χ.
- Unresolved entrainment, only where q ≈ 0:
  ```
  dT/dt = −k_T (T − T∞),   k_T = 0.5-2 s⁻¹ (tier dependent: coarser grid needs more)
  ```
  Tune so the time-averaged plume ΔT matches Heskestad (Sec. 4.3).
- Integrate both exactly or implicitly per cell (`T∞ + (T − T∞)e^(−kΔt)`), so they stay stable at any Δt.

**7. Projection**
- Solve `∇²p = (∇·u* − φ)/Δt` (constant ρ).
- **Boundaries:**
  - Floor: solid (free-slip or no-slip).
  - Sides and top: **open** (Dirichlet p = 0) so air can be **entrained**. A closed box kills necking and puffing.
  - Add a 3-5-cell sponge at the sides and top that relaxes u toward 0 (sides) or toward outflow (top).
- **Solver:** multigrid-preconditioned CG, or 2-3 multigrid V-cycles (red-black Gauss-Seidel smoother), warm-started.
  The minimum is 40-80 Jacobi iterations, warm-started; fewer leaves divergence that looks like spurious pumping.
- Log solids: voxelised SDF, velocity = 0 inside, with fuel and temperature sources on the surface.

### 10.4 Rendering hooks (physics-side contract)

- Emission per voxel = `κ_λ(S) B_λ(T)`, using the soot λ^-1.39 colour (Sec. 2.6), plus a blue chemiluminescence term
  `∝ q` coloured ≈ linear (0.1, 0.4, 1.0).
- Absorption: soot (κ_550 ≈ 10 m⁻¹/ppm, albedo ≈ 0.2-0.3) and organic smoke (albedo ≈ 0.9, λ-dependent toward blue).
- Ray-march with emission integrated in linear HDR. White-balance to a ~2000-3000 K white (von Kries/Bradford).
  Use a filmic tone curve with a soft highlight shoulder so cores clip toward yellow-white.
- The steep T^20 brightness response is what makes edges look crisp. Add sub-voxel detail by perturbing the
  T and S lookups with advected noise at 1-3 cm (Kolmogorov amplitudes, Sec. 6).
- Heat haze: refraction from ∇T in the plume (Δn ∝ ρ, so cooler-hotter interfaces bend light).

### 10.5 Calibration and validation tests (run them; this is how you know it is "real")

1. **Heat release:** Q̇ = Σ_cells b · ρ_cell · V_cell · ΔH_c / Δt, with ρ_cell = p0/(R T). Hold it at 50-80 kW.
2. **Puffing frequency:** FFT of the total visible emission (or of Q̇) over 20 s. Expect a clear peak at **1.5/sqrt(D)**
   (≈ 2.1 Hz for D = 0.5 m) within ±20%. If it is too high, check units and g, and whether expansion or buoyancy is too
   strong. If it is too low or absent, check open boundaries, advection diffusion and the CFL.
3. **Centreline profiles (time-averaged):**
   - u ≈ 6.8 sqrt(z) up to 0.08 Q^(2/5), then a plateau at ≈ 1.9 Q^(1/5) (≈ 4.2 m/s at 50 kW).
   - T ≈ 1100-1200 K in the continuous zone.
   - V/sqrt(2gzΔT/T∞) ≈ 0.9.
4. **Flame height:** 50% intermittency at the Heskestad L (≈ 0.6-0.75 m); visible tongues to ≈ 0.2 Q^(2/5) (≈ 1 m).
5. **Pattern speed:** visible tongues should rise at ~1-2 m/s, well below the peak gas speed.
6. **Necking:** inward inflow of ~0.3-1 m/s near the base, with the narrowest flame width about 0.5-1 D above the bed.
7. **Look tests:** compare side by side against real campfire footage frame by frame, in particular:
   - blue fringe height (1-3 cm);
   - how fast torn-off tongues burn out (0.1-0.3 s);
   - the amount of smoke (thin and grey above tips, not a black column);
   - spark density and lifetime.

### 10.6 Parameter summary (start here, then tune)

| Parameter | Start | Range | Anchor |
|---|---|---|---|
| g | 9.81 m/s² | fixed | flicker frequency ∝ sqrt(g) |
| T∞ | 293 K | — | |
| β (buoyancy) | 0.8 | 0.6-1.0 | McCaffrey C² = 0.81 |
| T_ign | 600 K | 550-700 K | piloted ignition |
| k_mix | 30 s⁻¹ | 15-60 s⁻¹ | grid-scale mixing time (ℓ = h → τ ≈ 13-17 ms → ~60-80 s⁻¹ upper bound) |
| r_s | 4.7 | 4-5.5 | stoichiometric air:volatiles ≈ 4.5-5 kg/kg |
| ΔT_f · η | 7700 K | 6000-9000 K | ΔH_c/c_p × (1 − χ_rad) |
| γ_exp (expansion) | 0.7 | 0.3-1.0 | ∇·u = q'''/(ρ c_p T) |
| Soot peak | 2-3 ppm | 0.5-6 ppm | 6.8 ppm in forest-fuel flames |
| Soot age gate | 20-60 ms | — | blue base 1-3 cm |
| k_ox | 20 s⁻¹ | 10-30 s⁻¹ | luminous tip ends where soot is gone |
| Radiative χ | 0.4 | 0.3-0.6 | 4700 K/s at 1300 K and 1 ppm (thin) |
| k_T (mixing cooling) | 1 s⁻¹ | 0.5-3 s⁻¹ | Heskestad plume decay ≈ 3 s⁻¹ total |
| εh (confinement) | 0.2 m/s | 0.1-0.3 m/s | Nguyen 0.8-3 m/s at 30 m/s jet, scaled |
| Curl noise | 0.5 m/s @ 8 cm, τ = 0.1 s | 0.3-0.8 m/s | Kolmogorov (εℓ)^1/3 |
| Fuel injection | 3.6-5.7 g/s total | — | Q = 50-80 kW at 14 MJ/kg |
| CFL | ≤ 3 (RT), ≤ 1.5 (video) | — | sharpness |
| Sparks | 2/s baseline + bursts of 20-100 | — | Sec. 8 |

### 10.7 Common failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Fire flickers like a toy (5-10 Hz) | domain not in metres, or g scaled | use real units (f = 1.5/sqrt(D)) |
| Blobby fireball look, no tall tongues | burning everywhere above T_ign | add the oxygen field and mixing-limited burning |
| No necking or puffing; fire "leans" or drifts | closed box, too much damping, CFL too big | open boundaries plus sponge, BFECC, CFL ≤ 3 |
| Uniformly orange, soft-edged flames | linear colour ramp | physical Planck emission plus soot plus white balance; the T^20 sensitivity gives the crisp edges |
| No blue base | soot formed instantly | age gate on soot formation; blue term ∝ heat-release rate |
| Black column of smoke | soot never oxidised | k_ox with an oxygen dependence; the tip should be clean |
| Noodly, noisy base | too much confinement or noise near the source | ramp them up with height and temperature |
| Everything too fast, structures zip upward | buoyancy too strong (Boussinesq α too high) or too much expansion | β ≈ 0.8, γ_exp ≤ 1; check the pattern-speed test |
| Sparks look like dots | no motion blur, no burning model | streaks with shutter-length blur; burning lifetimes from Sec. 8.2 |

---

## 11. References (primary sources first)

- McCaffrey, B. J. (1979). *Purely Buoyant Diffusion Flames: Some Experimental Results.* NBSIR 79-1910. https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nbsir79-1910.pdf (centreline table: κ = 6.8/1.9/1.1, η = 1/2, 0, −1/3, zone limits 0.08/0.20, C = 0.9, ~3 Hz, flame tip ΔT = 320 °C, pattern speed 1.1-1.8 m/s)
- McCaffrey & Cox (1982). *Entrainment and Heat Flux of Buoyant Diffusion Flames.* NBSIR 82-2473. https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nbsir82-2473.pdf
- Heskestad, G. SFPE Handbook, "Fire Plumes, Flame Height and Air Entrainment" (plume equation reproduced in NIST TN 1984: https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1984.pdf); flame-height form via arXiv 2008.12209 / 2008.13457.
- Zukoski, Cetegen & Kubota (1985). Visible structure of buoyant diffusion flames. 20th Symp. (Int.) on Combustion. https://resolver.caltech.edu/CaltechAUTHORS:20101206-110336671
- Cetegen & Ahmed (1993). Experiments on the periodic instability of buoyant plumes and pool fires. Combust. Flame 93:157-184.
- Hamins, Yang & Kashiwagi (1992). An experimental investigation of the pulsation frequency of flames. Proc. Combust. Inst. 24.
- Xia & Zhang (2018). Flickering of buoyant diffusion flames: a vortex dynamics revisit. https://arxiv.org/abs/1803.10400 ; dual-flame coupling: https://arxiv.org/abs/1803.10411 (PRF 4, 053202, 2019)
- Swirl and necking in pool fires: https://arxiv.org/abs/2202.06567
- Nguyen, Fedkiw & Jensen (2002). Physically Based Modeling and Animation of Fire. SIGGRAPH. https://graphics.stanford.edu/papers/fire-sg02/
- Fedkiw, Stam & Jensen (2001). Visual Simulation of Smoke. SIGGRAPH. https://www.graphics.stanford.edu/papers/smoke/
- Feldman, O'Brien & Arikan (2003). Animating Suspended Particle Explosions. SIGGRAPH. http://graphics.berkeley.edu/papers/Feldman-ASP-2003-08/
- Horvath & Geiger (2009). Directable, High-Resolution Simulation of Fire on the GPU. SIGGRAPH. https://history.siggraph.org/?p=108799 ; patents US8289327B1 / US8760449B1.
- Lamorlette & Foster (2002). Structural Modeling of Flames for a Production Environment. ACM TOG 21(3):729-735.
- SideFX Houdini docs: Pyro Solver 2.0 https://www.sidefx.com/docs/houdini/nodes/dop/pyrosolver.html ; Sparse Pyro https://www.sidefx.com/docs/houdini/nodes/dop/pyrosolver_sparse.html ; Flames https://www.sidefx.com/docs/houdini/pyro/flames.html
- JangaFX EmberGen forum thread on combustion expansion: https://forums.jangafx.com/t/feature-request-pressure-gain-from-combustion/392
- Yang et al. (2007). Characteristics of hemicellulose, cellulose and lignin pyrolysis. Fuel 86:1781.
- Soot: Hottel & Broughton α = 1.39; RadLib Planck-mean soot (F_s = 1817): https://ignite.byu.edu/radlib_documentation/classrad__planck__mean.html ; Sandia Fuego absorptivity: https://www.sandia.gov/files/sierra/Fuego_Users_5_20/theory/mathmodels/absorptivity.html ; BYU soot lecture: https://ignite.byu.edu/che522/lectures/soot.pdf ; forest-fuel soot 6.8 ppm: https://www.frames.gov/catalog/45982
- Chemiluminescence bands: https://researching.cn/articles/OJ71454e6da154baca ; https://fluids.ac.uk/files/sig/Liu.1525267346.pdf ; CO flame over coals: Nature 109:683 (1922) https://www.nature.com/articles/109683a0.pdf
- Ember temperatures: UMD imaging pyrometry https://drum.lib.umd.edu/handle/1903/21928 , https://drum.lib.umd.edu/handle/1903/28770 ; Berkeley CFP https://cfp.berkeley.edu/research/temperature-measurement-of-glowing-firebrands
- Firebrands: NIST Douglas-fir burns https://publish.csiro.au/wf/WF06079 ; burning under wind https://publish.csiro.au/WF/WF23151 ; ember bursts https://www.preventionweb.net/quick/89843
- Char cracks: Baroudi et al., https://arxiv.org/abs/1604.01249
- Crackle mechanism (popular level): https://theconversation.com/curious-kids-why-does-wood-crackle-in-a-fire-126346
- Smoke colour: https://www.atoptics.co.uk/fz300.htm
- CIE CMF fit used for the colour table: Wyman, Sloan & Shirley (2013), JCGT 2(2). Bradford chromatic adaptation.
- Curl noise: Bridson, Hourihan & Nordenstam (2007), SIGGRAPH. MacCormack: Selle et al. (2008), J. Sci. Comput. Wavelet turbulence: Kim et al. (2008), SIGGRAPH.
