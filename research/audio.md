# Procedural fire sound: physics, DSP design, Web Audio + deterministic offline render

Scope: fully procedural (no samples) campfire / bonfire audio for the WebGPU fire. It covers live playback in the browser (AudioWorklet) and a bit-exact offline render for the 20 s social video. Everything below is backed by a **working, measured reference implementation** in `D:\fire\research\audio-ref\`:

| File | What it is |
|---|---|
| `fire-audio-dsp.mjs` | Dependency-free DSP core (engine, filters, voices, IR generator, BS.1770 meter, TP limiter). Runs unchanged in an AudioWorklet and in Node. |
| `fire-worklet.js` | `AudioWorkletProcessor` wrapper (k-rate params, event messages, event log back to the main thread) |
| `live-demo.html` | Minimal live test page: sliders and event buttons, ConvolverNode reverb, audio-to-visual flash sync. Tested in the desktop browser pane; serve over http. |
| `render-offline.mjs` | Node offline renderer: timeline JSON → 48 kHz/24-bit WAV + `events.json` (frame-stamped) + loudness/spectral stats |
| `calibrate.mjs` | Re-measures every layer and prints the calibration table (`CAL`) |
| `timeline.example.json` | 20 s cinematic cue sheet (intensity/distance curves, hero events incl. the 3.2 s big pop) |
| `demo_fire_20s.wav`, `demo_events.json` | Rendered result of the example timeline: **-14.0 LUFS, -1.3 dBTP (ffmpeg ebur128), LRA 3.0 LU** |

---

## 0. Key takeaways

1. **Fire = 3 physical sources + scene events.** Farnell's decomposition (*lapping* = combustion of gas in air, *crackling* = small fracture explosions in the fuel, *hissing* = outgassing of trapped vapour) is the right backbone. Realism also needs pops (steam/resin pocket ruptures), micro-ticks, charcoal tinkles, log shifts, gust whooshes and an outdoor acoustic space.
2. **The single most important perceptual cue is broadband, co-modulated transients.** McDermott et al. (2009) found that fire's crackles produce correlated envelopes across frequency bands. Synthesis that matches only per-band statistics tends to sound like water. So every crackle must contain a **raw broadband component** (15-45 %) and not just band-passed noise.
3. **Combustion roar spectrum:** above a peak/knee frequency, acoustic power falls as **P(f) ∝ f^-α with α ≈ 2.1-3.4** (Rajaram & Lieuwen 2009). Chadwick & James (2011) found **α = 3.0 sounds best** (2.5 too bright, 3.5 too dull). Below the peak, power rises roughly as f². For a campfire the true peak is very low (field fires peak at about 2-60 Hz). The audible roar is the f^-3 tail plus a 100-500 Hz "body".
4. **Flames puff at f = 1.5/√D Hz** (D = fire base diameter in m; equivalently 0.48·√(g/D)). A 0.7 m campfire puffs at about 1.8 Hz and a 2 m bonfire at about 1.06 Hz. Use the same frequency for visual flicker and audio AM. Add 5-12 Hz tongue flutter and about 0.1 Hz slow wander, combined log-normally.
5. **Crackle timing is clustered, not Poisson.** Use a self-exciting (Hawkes) process: base rate about 16/s at full intensity, branching ratio about 0.45, kernel τ = 70 ms. Big events add Omori-like aftershocks. Wood fracture acoustic-emission energies follow **P(E) ∝ E^-b with b ≈ 1.4-1.5**. Use b ≈ 1.6 over a 20 dB energy range, with amplitude ∝ E^0.4, or the result cannot be mastered.
6. **Pops** combine a sub-millisecond rupture click, a cavity/fragment ring at **1.2-4.5 kHz** (Helmholtz estimate; larger pocket = lower), a 2-6 ms crack-noise burst, a 90-200 Hz body for big pops, a micro-crackle aftershock cluster, an optional 60-250 ms steam "psssh", and a spark-sizzle tick burst. The popcorn analogy (Virot & Ponomarenko 2015) supports this: the audible pop comes from vapour release exciting the cavity, not from the fracture itself.
7. **Mid-band "breath" is mandatory.** Chadwick & James name a missing mid-frequency "whooshing" as their main limitation. Add a separate flutter-modulated 300 Hz-3 kHz band.
8. **Mastering a fire to -14 LUFS is a crest-factor problem.** Sparse crackles have high peak-to-loudness ratios. The measured fixes were: multi-sample ticks (not single-sample impulses), amplitude ∝ E^0.4, soft saturation of the roar at 3× running RMS, then a memoryless soft clipper and a 1.5 ms lookahead true-peak limiter. The result is -14.0 LUFS and -1.3 dBTP, with about 2 crackle overs per second of ≤ 6 dB on 1-3 ms noise bursts (inaudible).
9. **Determinism:** seeded per-layer RNG streams, all control work on a global 32-sample grid, and events scheduled by absolute sample index. The Node render is then **bit-identical for any block size** (verified for 128/333/997 frames) and identical to what the worklet produces for the same events.
10. **Sync:** one timeline JSON drives both renderers. The audio engine logs every hero event (scripted and random) with frame numbers, and the video renderer spawns sparks and flares from that log. Live, use `AudioContext.getOutputTimestamp()` to show the visual exactly when the pop leaves the speakers. Audio may lag the picture by tens of ms but should never lead it (ITU-R BT.1359).

---

## 1. Physics: what is actually making the sound

### 1.1 Source taxonomy (what to synthesize)

| Layer | Physical cause | Time scale | Spectrum | Statistics |
|---|---|---|---|---|
| **Roar / rumble** (Farnell "lapping") | Unsteady heat release of turbulent combustion → volume expansion → monopole sound | continuous, AM 0.1-15 Hz | peak ≤ 60 Hz for open fires; f^-3 tail; 100-500 Hz body | Gaussian carrier, log-normal envelope |
| **Breath / whoosh** | turbulent flame-tongue mixing, flapping | continuous, flutter 5-12 Hz | 300 Hz-3 kHz | deeply modulated |
| **Hiss** | vapour/gas jetting from cracks and wood ends ("singing" wood), outgassing | 0.1-2 s surges + floor | 1-12 kHz | very peaky envelope (Farnell: 4th power) |
| **Crackles** | micro-fractures as wood chars, shrinks and dries ("small scale explosions caused by stresses in the fuel", Farnell via Kersten & Purwins) | 1-30 ms each | broadband with a coloured peak 1.5-12 kHz | heavy-tailed sizes, clustered |
| **Ticks** | tiny char/ember fractures, surface cracking | 0.1-0.5 ms | 2.5-15 kHz | dense (hundreds/s) |
| **Pops** | rupture of pressurised steam/resin pockets | click < 1 ms, ring 5-30 ms, tail ≤ 0.3 s | broadband click + 1-5 kHz ring + LF body | rare (0.05-0.5/s), big |
| **Charcoal tinkles / embers** | brittle char pieces impacting, settling | 5-20 ms rings | 2-10 kHz modal ("glassy") | bouncing sequences |
| **Log shift** | log collapses onto coals | thump 60-250 ms, debris ≤ 1.5 s | 55-90 Hz thump, 350-700 Hz knock, tinkles | rare scripted |
| **Gust** | wind feeds oxygen → flare, flame leans, flapping | 1-3 s | 250-1000 Hz whoosh sweep; all layers up | events |
| **Ambience** | crickets, distant space | continuous | 4-5 kHz chirps | periodic |

### 1.2 Combustion noise (roar), with equations

The dominant source of open-flame sound is **direct combustion noise**: unsteady heat release makes the gas expand unsteadily, which acts as a monopole. Vortex/aerodynamic noise is secondary (Ihme et al. 2009, cited by Chadwick & James). The linear wave equation with a heat-release source (Crighton et al. 1992; Rajaram & Lieuwen 2009), as used in Chadwick & James eq. (2)-(3):

```
(1/c0²) ∂²p/∂t² − ∇²p = −∂/∂t ( ρ0 (γ−1) q / (ρ c²) )

p(x,t) = (1/4π) · ((γ−1)/c0²) · ∂/∂t ∫ [q](y, t − |x−y|/c0) / |x−y| d³y
```

So **far-field pressure ∝ d/dt (total heat-release rate)**. Chadwick & James assume heat release on a flame-front patch ∝ the velocity flux through it, which gives eq. (6):

```
p(t) = d/dt ∫_S(t) u(x,t)·n(x,t) dS  =  dI/dt
```

Practical consequence for us: if the WebGPU sim can export a per-frame scalar **H(t) = Σ voxels (fuel burn rate)**, or summed emitted radiance as a proxy, then **dH/dt is a physically motivated low-frequency roar driver**. Visual frame rates only resolve this up to 30 Hz (at 60 fps), so everything above that must come from bandwidth extension (section 3).

**Spectral shape** (measured on lab burners):
- Abugov & Obrezkov (1978): above the peak, **P(f) ∝ f^-5/2**.
- Clavin & Siggia (1991): f^-5/2 follows if the flame front is wrinkled by Kolmogorov turbulence (E(k) ∝ k^-5/3).
- Rajaram & Lieuwen (2009): four parameters describe the spectrum: peak frequency (a Strouhal number based on mean velocity and flame length), low-frequency slope (≈ **f²**, from the f² transfer function between heat-release and acoustic spectra), high-frequency slope **α = 2.1-3.4**, and overall SPL.
- Practical burners peak at 200-800 Hz (IOA review), with the peak wavelength about 50× the nozzle diameter.

**Scaling to a campfire.** The flame length is about 0.5-1.5 m and buoyant velocities about 2-5 m/s, so f_peak ≈ U/L is single-digit Hz. Field data agree:
- Crown fire: 0-400 Hz, amplitude peak at about 25 Hz (initial) to 60 Hz (strong). Surface fire: peak about 220 Hz, energy up to about 15 kHz (Zhang et al. 2019, Sensors).
- Prescribed burns radiate 2-90 Hz infrasound, with 2-40 Hz attributed to the active combustion zone (Marcillo et al. 2025).

**Design consequence:** the audible campfire "roar" is the **f^-3 tail above a knee of about 150-600 Hz (rising with intensity/wind)**, a moderate 100-500 Hz body (needed for phone speakers), and sub-bass of 45-100 Hz, high-passed at 45 Hz because phones cannot reproduce it and it wastes headroom.

### 1.3 Flame puffing / flicker (the AM of everything)

Buoyant diffusion flames shed toroidal vortices periodically. Pool-fire data collapse onto

```
f_puff ≈ 1.5 · D^-1/2  [Hz, D in m]   (Cetegen & Ahmed 1990)  ≡ 0.48·√(g/D)
(Malalasekera et al. review fit: 1.68 · D^-1/2)
```

| Fire | D (m) | f_puff (Hz) |
|---|---|---|
| candle / single tongue | 0.02-0.03 | 8.7-10.6 (real candles flicker at about 10-12 Hz) |
| small tongue / twig fire | 0.1 | 4.7 |
| small campfire | 0.5 | 2.1 |
| campfire | 0.7-0.8 | 1.7-1.8 |
| bonfire | 2 | 1.06 |

The puffing is quasi-periodic, so model it as **narrow-band Gaussian noise** (2nd-order band-pass at f_puff, Q ≈ 2.5), not a sine. Many small flame tongues (D of 2-10 cm) add a **5-12 Hz flutter** (Q ≈ 1.2), and wind adds 10-20 Hz flapping.

### 1.4 Crackling statistics

- "Crackling noise" (Sethna, Dahmen & Myers 2001) means discrete impulsive events spanning a broad range of sizes with a power-law size distribution, D(s) ∝ s^-(1+β).
- **Wood-like fracture AE** (bamboo chopstick, 44.1 kHz, 5 cm mic): pulse energies follow **P(E) ∝ E^-b with b = 1.45** (MLE; 1.40-1.48 across halves; spaghetti 1.35). Aftershock rates follow **Omori's law k/(c+t)^p with p = 1.68, c = 0.07 s**. That paper also notes earthquake b values of 0.5-2. Clustering is real: up to about 8 aftershocks per main event.
- Kersten & Purwins (DAFx-12) model the fire foreground as **Poisson-distributed "molecules"** (clustered wavelet atoms), with inter-event intervals drawn from a kernel density estimate, plus a background of LPC-coded coloured noise. Frame 1024, hop 256, 20 spectral + 20 temporal envelope coefficients. Their artefacts are useful lessons: transients got smeared, and the **"explosion tails" following transients** were not captured. Pops need tails.
- Yedinak et al. (2017, JASA): crackle impulse content in the **6-15 kHz** band correlates with vegetation species and water stress. Fuel moisture is audible up there.
- Zhu & Wyse (DAFx-04) and Athineos & Ellis describe fire as a "din" plus a sequence of micro-transients. Farnell's patch triggers crackles when a slowly varying noise level crosses a narrow window. That is a *doubly stochastic* (Cox) process with built-in clustering, sharing its slow noise with the hiss modulator.

**Implementation choice:** a Hawkes process (self-exciting) for crackles: λ(t) = μ·I(t)^1.3 + Σ_i n_i/τ · e^-(t−t_i)/τ. Its mean cluster size is 1/(1−n̄). Big events (pops, log shifts) inject Omori-like bursts.

### 1.5 Pops: steam / resin pocket ruptures

Mechanism: heating turns moisture into steam (or volatilises resin) inside closed cells and pockets. Pressure builds until the wall fails. The popcorn study (Virot & Ponomarenko 2015, synchronized high-speed video and audio) found the audible "pop" coincides with **vapour release**, not with the fracture (about 100 ms later in popcorn). It is attributed to the pressure drop exciting the cavity's acoustic modes. For wood, model the event as an impulsive pressure release plus a short cavity/fragment ring.

**Helmholtz estimate of the ring:** f_H = (c / 2π) · √(A / (V · L_eff)), with c ≈ 343 m/s (air, 20 °C) to about 470-480 m/s (steam at 100 °C or air at 300 °C).

| Pocket | V | crack area A | L_eff | f_H (c = 343) | f_H (hot, c ≈ 475) |
|---|---|---|---|---|---|
| small resin blister | 0.05 cm³ | 1 mm² | 1 mm | 7.7 kHz | 10.7 kHz |
| medium steam pocket | 0.5 cm³ | 3 mm² | 2 mm | 3.0 kHz | 4.1 kHz |
| large pocket / split | 5 cm³ | 10 mm² | 4 mm | 1.2 kHz | 1.7 kHz |

That gives the **1.2-4.5 kHz / size^0.4** law used below: bigger pops are lower and longer. A vented cavity alone has low Q (about 5-20). Wood or char fragments add higher-Q rings, so τ = 4-12 ms is used. Shorten it if pops sound "plinky".

Wet or resinous wood pops more often and louder. Popular sources disagree on softwood versus hardwood, so treat `popRate` as a fuel parameter, typically 0.05-0.5 /s.

### 1.6 Perception: what makes a texture read as "fire"

McDermott, Oxenham & Simoncelli (2009) used 30 ERB-spaced bands (20 Hz-14 kHz) with half-cosine filters and envelope statistics on log Hilbert envelopes:
- All natural textures have **subband kurtosis > 3** (Gaussian = 3).
- **Fire specifically** shows vertical broadband streaks (crackles and pops), giving **non-zero correlations between neighbouring subband envelopes**.
- Imposing only marginal statistics (variance + kurtosis) worked for water, but made non-water textures **sound like water**.
- Adding envelope cross-correlations and autocorrelations (25 lags, 2-500 ms) made textures recognisable.

Design rules that follow:
1. Every transient is **broadband first, coloured second** (raw fraction 15-45 % plus a band-pass).
2. Sparsity (kurtosis) comes from heavy-tailed event sizes and silence between events, not from noise modulation alone.
3. Envelope autocorrelation structure comes from Hawkes clustering, puffing AM, and event cascades (pop → aftershock → sizzle).

Note that procedural fire models in a recent MUSHRA test (Nemisindo engine, QuAP paper, arXiv 2606.00629) still scored only "slight resemblance" (about 29-40/100). Naive crackle/hiss/roar mixes are known not to be enough, so the details above matter.

---

## 2. Farnell's *Designing Sound* fire model, decoded from the original Pd patches

Source: MIT Press companion code, Practical 11 (`fire_crackling1-3.pd`, `fire_hissing1-3.pd`, `fire_lapping1-3.pd`, `fire_generator.pd`, `fire_all.pd`). Values below are the exact numbers in the patches. Pd objects: `lop~`/`hip~` are one-pole low/high-pass; `bp~ f Q` is a two-pole resonator.

### 2.1 Crackling (`fire_crackling3` / generator subpatch)
```
noise~ → lop~ 1 → env~ → moses 50 → (≥50) moses 51 → (<51) bang       // trigger when slow level ∈ [50,51) dB
bang → random 30 → t f f
   ├─ msg "1, 0 $1" → line~ → (x)² → envelope        // jump to 1, ramp to 0 over 0..29 ms, squared
   └─ * 500 → + 1500 → bp~ centre                    // centre 1500..16000 Hz, Q = 1 (generator: bp~ 4000 1)
noise~ → bp~ (centre, Q 1) × envelope → out
```
- The trigger is a level-crossing detector on 1 Hz low-passed noise. `env~` outputs every 512 samples (about 11.6 ms at 44.1 kHz), so while the slow noise hovers near the window, several consecutive blocks fire. That gives **natural clusters of crackles 11.6 ms apart**.
- Each crackle: noise through a Q = 1 band-pass with a random centre of 1.5-16 kHz and a **squared linear decay of 0-29 ms**. The squared ramp has an energy-equivalent duration of about D/5.

### 2.2 Hissing (`fire_hissing3`)
```
carrier:   noise~ → hip~ 1000
modulator: noise~ → lop~ 1 → *~ 10 → (x)² → (x)² → *~ 600      // 600·(10·LP1Hz(noise))^4
out = carrier × modulator
```
The 4th power of a slow Gaussian makes hiss that is near-silent most of the time with occasional **surges** (E[z⁴] = 3σ⁴, and peaks at 3σ are 81× the median). It is the outgassing "pssh". In the generator, the **same 1 Hz LP noise** drives both the crackle trigger and the hiss surges.

### 2.3 Lapping / roar (`fire_lapping3`)
```
noise~ → bp~ 30 5 → *~ 100 → hip~ 25 → clip~ -0.9 0.9 → hip~ 25 → *~ 0.6
```
This is a narrow 30 Hz band of noise (bandwidth 6 Hz), so a lumpy random-phase 30 Hz wobble, driven hard into a clipper. The clipping creates odd harmonics at 90/150/210 Hz, giving a "flapping flag" quality that stays audible on small speakers.

### 2.4 Mixing and the "big fire"
- `fire_generator`: one shared `noise~` ("factor common noise") feeds crackles ×0.2, hissing ×0.3, lapping ×0.6.
- `fire_all`: **four generators** through `bp~ 600 0.2`, `bp~ 1200 0.6`, `bp~ 2600 0.4`, `hip~ 1000`, summed ×0.2. These are differently coloured sub-fires (polyphony = size).

### 2.5 What we keep and what we change
| Farnell | Kept / changed in our design | Why |
|---|---|---|
| crackle: random BP 1.5-16 kHz, Q 1, 0-29 ms | kept the ranges, plus a **raw broadband fraction**, sub-impulses, Hawkes clustering, power-law sizes | cross-band correlation (McDermott); realistic size statistics |
| hiss = HP1k noise × (LP1Hz)^4 | kept the HP carrier and z⁴ surges, **added a 40 % steady floor**, capped z⁴ at 16 | pure z⁴ is extremely kurtotic (long-term kurtosis in the hundreds) |
| lapping = BP30 Q5 → clip | kept (tanh instead of hard clip), plus a separate f^-3 combustion roar and a mid "breath" layer | Farnell's lapping alone has no broadband roar |
| shared noise couples the layers | coupling is explicit: puffing modulates roar, breath, crackle rate and tick density | controllable and synced with the visuals |
| 4 coloured generators = big fire | `intensity` and `fireDiameter` parameters | continuous control |

---

## 3. Chadwick & James 2011, "Animating Fire with Sound" (SIGGRAPH)

**Pipeline:** (1) a low-frequency physically based sound from a visual flame simulation run at only **360 Hz** (Houdini Pyro/Flame), and (2) the high-frequency content added either by **spectral bandwidth extension** (power-law noise) or by **data-driven texture synthesis** from recordings. We can't use the latter (no assets).

**Low-frequency model:** p(t) = dI/dt with I(t) = ∫_S u·n dS over the flame front S, extracted as an isosurface of the burn field. A cubic B-spline reconstruction is needed to avoid jagged-normal artefacts. I(t) is interpolated to audio rate with a **Mitchell-Netravali cubic**, and the derivative is taken analytically: p(t) = Σ ψ'_i(t) I(t_i).

**Spectral bandwidth extension (Algorithm 1):**
1. Split the LF pressure into triangular windows of half-width **w_H = 500 samples (11.34 ms @ 44.1 kHz)**.
2. For each window, generate noise N(t) = IFFT(f^-α/2 · e^{-iθ}) with uniform random phases, high-passed above f_cut, and scale it by |p(t)·w(t)| so it follows the LF signal's temporal structure.
3. Blend the spectra: p̃_blend = F_low·p̃_W + β·F_high·Ñ_W, linear cross-fade over **[165, 195] Hz**.
4. Choose β by matching power in a Gaussian window (mean f_cut, σ = 10 Hz) around **f_cut = 180 Hz** (the Nyquist frequency of the 360 Hz simulation). This is a quadratic in β.
5. Pre-process with a **30 Hz high-pass** (otherwise inaudible LF fluctuations get extended into loud content).

| α | f_cut | σ_cut | w_H | blend | finding |
|---|---|---|---|---|---|
| 2.5-3.5 | 180 Hz | 10 Hz | 500 samples | 165-195 Hz | **α = 3.0 best**; 2.5 too much HF, 3.5 too much LF |

**Their stated limitations:**
- Pure BWE sounds "bursty", very uniform across scenarios, and **lacks the mid-frequency "whooshing"** of real flames.
- Texture synthesis needs recordings.
- Spatialization of the extended content is open.

**How we use it without FFTs or recordings (time-domain BWE):**
- The roar carrier is a fixed **f^-3 shaped noise**: Kellet pink (−3 dB/oct) followed by a one-pole low-pass at the knee (−6 dB/oct), giving −9 dB/oct ≡ P ∝ f^-3 above the knee.
- Its envelope plays the role of |p(t)w(t)|. It comes either from (a) the internal puffing/flutter modulators, or (b) optionally the sim's per-frame heat-release proxy H(t): e(t) = smooth(|dH/dt|) with a 30-60 ms one-pole, normalised, then log-compressed.
- The missing "whoosh" is a separate **breath** layer (section 4.4).

---

## 4. DSP design (all numbers are the reference implementation's)

Sample rate 48 kHz. Control rate = fs/32 = 1500 Hz on a **global sample grid** (control work runs when `absSample % 32 == 0`, never "per process() call"). Pre-master headroom: continuous layers are calibrated so **mix 0 dB = -20 LKu** (ungated K-weighted level) at intensity 1 and distance 2 m. For pops and log shifts, mix 0 dB means a size-1 event peaks at about -6 dBFS.

### 4.0 Layer overview at intensity I = 1

| Layer | Mix (dB rel. -20 LKu) | Rate / scaling with I | Reverb send |
|---|---|---|---|
| crackles | −4 | 16/s·I^1.3 base, Hawkes n ≈ 0.45 (≈ 25/s total) | 0.20 |
| ticks | −9 | 260/s per channel·I^1.2 (≈ 520/s) | 0.12 |
| roar | −7 | amp ∝ I^1.5 | 0.08 |
| breath | −8 | amp ∝ I^1.3 | 0.08 |
| lapping | −15 | amp ∝ I^1.2 | 0 |
| hiss | −16 | amp ∝ I | 0.10 |
| pops | +2 (peak-calibrated) | random 0.12/s·I^1.5 + scripted | 0.40 |
| logs | −1 (peak-calibrated) | scripted | 0.30 |
| embers | −6 (peak −20 dBFS @0 dB) | 0.15/s | 0.35 |
| whoosh | −9 | gust envelope^1.5 | 0.15 |
| crickets | −22 | 3 individuals | 0.70 |

### 4.1 Randomness (deterministic)
- `mulberry32` PRNG. A **separate stream per layer**, seeded by hashing (seed, layer name), so changing one layer never reshuffles another's randomness. This matters a lot when iterating on a mix.
- Audio-rate noise uses uniform `bi()` ∈ [−1, 1) (variance 1/3) rather than Box-Muller. Heavy filtering makes it Gaussian (CLT), and it is about 5× cheaper. Gaussian draws are used only at control rate and for event parameters.
- Distributions:
  - log-uniform: `a·(b/a)^u`
  - truncated Pareto, pdf ∝ x^-b on [x0, x1]: `x0·(1 − u(1 − (x0/x1)^(b−1)))^(−1/(b−1))`
  - geometric sub-impulse counts
  - Box-Muller N(0, 1)

### 4.2 Filters and formulas

**RBJ biquads** (W3C Audio EQ Cookbook). Compute w0 = 2π f0/Fs, α = sin(w0)/(2Q), A = 10^(dB/40); for shelves α = sin(w0)/2 · √((A + 1/A)(1/S − 1) + 2). Normalise all coefficients by a0. Implement as transposed direct form II: `y = b0 x + z1; z1 = b1 x − a1 y + z2; z2 = b2 x − a2 y`.

| Type | b0 | b1 | b2 | a0 | a1 | a2 |
|---|---|---|---|---|---|---|
| LPF | (1−cos)/2 | 1−cos | (1−cos)/2 | 1+α | −2cos | 1−α |
| HPF | (1+cos)/2 | −(1+cos) | (1+cos)/2 | 1+α | −2cos | 1−α |
| BPF (0 dB peak) | α | 0 | −α | 1+α | −2cos | 1−α |
| Peaking | 1+αA | −2cos | 1−αA | 1+α/A | −2cos | 1−α/A |
| Low shelf | A((A+1)−(A−1)cos+2√Aα) | 2A((A−1)−(A+1)cos) | A((A+1)−(A−1)cos−2√Aα) | (A+1)+(A−1)cos+2√Aα | −2((A−1)+(A+1)cos) | (A+1)+(A−1)cos−2√Aα |
| High shelf | A((A+1)+(A−1)cos+2√Aα) | −2A((A−1)+(A+1)cos) | A((A+1)+(A−1)cos−2√Aα) | (A+1)−(A−1)cos+2√Aα | 2((A−1)−(A+1)cos) | (A+1)−(A−1)cos−2√Aα |

**One-pole:** `y += a·(x − y)` with `a = 1 − exp(−2π fc/Fs)`. Its output std for unit-variance white input is √(a/(2−a)), which is used to normalise slow modulators.

**Cytomic/Simper TPT SVF** for anything modulated (crackle and whoosh filters, puffing modulators). It is stable under fast cutoff changes:
```js
g = tan(π f/fs); k = 1/Q; a1 = 1/(1 + g(g + k)); a2 = g·a1; a3 = g·a2;
v3 = v0 − ic2; v1 = a1·ic1 + a2·v3; v2 = ic2 + a2·ic1 + a3·v3;
ic1 = 2v1 − ic1; ic2 = 2v2 − ic2;   // lp = v2, bp = v1 (peak gain Q → use k·v1 for 0 dB), hp = v0 − k·v1 − v2
```

**Narrow-band modulator normalisation.** A unity-peak 2nd-order band-pass has noise-equivalent bandwidth **ENBW = (π/2)·f/Q**. For unit-variance input at rate f_r, output variance = ENBW/(f_r/2), so scale by `1/√((π/2)(f/Q)/(f_r/2))`. The same formula normalises Farnell's 30 Hz/Q5 lapping band at audio rate.

**Pink noise:** Paul Kellet's refined 7-pole filter. Coefficients 0.99886/0.0555179, 0.99332/0.0750759, 0.969/0.153852, 0.8665/0.3104856, 0.55/0.5329522, −0.7616/−0.016898, plus 0.5362·w and 0.115926·w. ⚠ It has **large LF gain** (DC gain about 70× before scaling). This made the first roar draft +2 LUFS hot, so always calibrate by measurement.

### 4.3 Roar (combustion)
```
m, s = uniform noise;  L = m + 0.45 s,  R = m − 0.45 s         // inter-channel correlation ≈ 0.66
x = Pink(·) → OnePoleLP(knee) → HP 45 Hz (Q .7) → Peak +4 dB @120 Hz (Q .8) → Peak +2 dB @420 Hz (Q 1)
x = 3σ·tanh(x / 3σ),  σ = running RMS (one-pole on x², τ ≈ 1 s)    // soft saturation: LF peak control + harmonics
y = x · I^1.5 · exp(lnAmp) · (1 + 1.4·gust) · mix
knee = max(60, 150 + 220·min(I,1.5) + 260·gust + 40·u_puff)  Hz   // brighter when bigger/windier
lnAmp = 2.2σ_A·tanh((0.32 u_puff + 0.16 u_flutter + (0.12+0.35 gust) u_flap + 0.22 u_slow)/(2.2σ_A)) − σ_A²/2
u_puff: NB Gaussian f = 1.5/√D, Q 2.5 · u_flutter: 7 Hz, Q 1.2 · u_flap: 14 Hz, Q 2 · u_slow: one-pole 0.12 Hz
```
The log-normal AM has unit mean, so loudness does not drift with modulation depth. The tanh soft-cap at ±2.2σ stops rare giant swells from slamming the limiter.

### 4.4 Breath (mid-band flame whoosh): Chadwick & James' missing piece
Independent L/R uniform noise → pink → SVF band-pass **750 Hz, Q 0.55** (unity peak; covers about 300 Hz-3 kHz) → HP 220 Hz.
Amplitude: `I^1.3 · exp(0.25 u_puff + 0.4 u_flutter + (0.2 + 0.3 gust) u_flap − ½σ²) · (1 + gust)`.
It is modulated more deeply and faster than the roar. This is the "fluttering" of tongues.

### 4.5 Lapping (Farnell)
White (Gaussianised uniform) → RBJ BPF **30 Hz, Q 5** → ×1/√((π/2)(30/5)/(fs/2)) (unit RMS) → `tanh(1.6x)` → HP 25 Hz ×2 → × I^1.2 · (1 + 1.2 gust). Mono (centre). Level −15 dB: it is felt more than heard.

### 4.6 Hiss (outgassing)
Per channel: uniform noise → HP **1.2 kHz** (Q 0.6) → high shelf **−4 dB @ 9 kHz**.
Envelope: `I · (0.4 + 0.6 · min(z⁴, 16)/3) · (1 + 0.8 gust)`, with z the normalised 1 Hz one-pole LP Gaussian (Farnell's `lop~ 1` → x⁴). Both channels share the envelope; the carriers are independent.

### 4.7 Crackles (the main texture)
Spawned by a per-sample Bernoulli draw with p = λ/fs. That is equivalent to a Poisson process with time-varying rate and is deterministic.
```
λ = 16·I^1.3·(1 + 0.2 u_puff)·(1 + 1.5 gustLag) + η(t)          [1/s]
η(t): Hawkes state; on each crackle  η += n_i / τ, n_i = min(0.85, 0.45·(0.6 + 0.8·sz)); decays with τ = 70 ms
pops: η += 35·s (≈ 2.5·s extra crackles);  log shift: η += 25·s
```
Per crackle:

| Parameter | Value |
|---|---|
| energy | E ~ Pareto(b = 1.6) on [1, 100] (20 dB); sz = log10(E)/2 |
| amplitude | 0.02·**E^0.4** (16 dB peak range) |
| decay | exponential, τ = 2·E^0.15·logU(0.5, 2.5) ms (≈ 1-7 ms; same energy duration as Farnell's 0-29 ms squared ramps) |
| sub-impulses | 1 + geometric(0.45), max 5; gaps U(0.3, 1)·(1.6 + 2 sz) ms; amplitudes U(0.25, 0.9) of the first; envelope = max(current, new) |
| colour | SVF band-pass fc = logU(1.8, 11 kHz)/(1 + 0.6 sz) (bigger = lower), Q U(0.8, 2.5), ×2.2 make-up |
| broadband | raw noise fraction U(0.15, 0.45) mixed in |
| position | pan U(−0.55, +0.55) (fire width), equal-power |

Voice pool: 48, with deterministic round-robin stealing.

### 4.8 Ticks (micro-crackle bed)
Two independent streams (L and R), 260/s each at I = 1, density wandering log-normally (0.6 Hz one-pole, σ 0.35) and +25 % with puffing.
Each tick is a **tiny decaying noise burst**, not a 1-sample impulse:
- amplitude log-normal, median −17 dB, σ 3.5 dB
- τ U(0.08, 0.3) ms
- 55 % routed through HP 2.5 kHz (broadband), 45 % through BP 5.5 kHz (Q 0.9, ×1.5)

Measured: single-sample impulses caused about 12 limiter hits/s for very little loudness; multi-sample bursts at a higher density fixed it. After big pops, tick rate gets +700·s /s decaying with τ = 0.25 s (spark/resin sizzle).

### 4.9 Pops (hero events)
Size s: scripted, or random Pareto(b = 2.2) on [0.25, 1.2]. L = pops mix gain.

| # | Component | Recipe |
|---|---|---|
| 1 | rupture click | noise burst τ 0.5 ms, 85 % raw + 15 % BP 7 kHz, amp 0.5·L·s^0.8 |
| 2 | cavity/fragment ring | 3 damped sines: f1 = logU(1.2, 4.5 kHz)/s^0.4, f2 = f1·U(1.4, 2.3), f3 = f1·U(2.6, 3.8); τ1 = U(4, 12) ms·√s, τ2 = 0.6τ1, τ3 = 0.4τ1; amps 0.11·L·s·(1, 0.5, 0.25) |
| 3 | crack noise | SVF BP U(2.5, 5 kHz)/√s, Q 0.7, attack 0.1 ms, τ U(2, 6) ms·√s, amp 0.16·L·s |
| 4 | LF body (s > 0.5) | damped sine U(90, 200) Hz, τ 20 ms·s, amp 0.32·L·s, pan × 0.5 |
| 4b | spark sizzle (s > 0.7) | tick-rate burst +700·s /s, τ 0.25 s |
| 5 | fracture cascade | round(3 + 9·s·U) crackles, gaps U(2, 12) ms, size 0.35·s, same pan |
| 6 | steam jet "psssh" (p = 0.3 + 0.3 s) | BP U(3.5, 7 kHz) Q 0.9, attack 8 ms, τ U(60, 250) ms·s, 40 Hz "sputter" AM depth 0.6, amp 0.03·L·s |
| 7 | aftershock | Hawkes η += 35·s |
| 8 | log | `{t, type:'pop', size, pan}` pushed to `eventLog` (→ sparks) |

Damped sinusoid via a click-free recursion (phase 0): `s[n] = 2r·cosθ·s[n−1] − r²·s[n−2]`, s[0] = 0, s[1] = A·r·sinθ, r = e^(−1/(τ fs)), θ = 2πf/fs. Stop after 9.2τ (−80 dB).

### 4.10 Log shift, charcoal tinkles, ember settling
Log shift (size s):
- thump: 55-90 Hz damped sine (τ 60 ms·s, amp 0.3·L·s) + 2nd mode ×2.1-2.9 (τ 30 ms)
- LF noise: SVF LP 180 Hz, attack 3 ms, τ 45 ms
- knock: 350-700 Hz plus ×2.4-2.9 (τ 25/15 ms)
- debris rustle: HP 2.5 kHz, attack 20 ms, τ 280 ms, 25 Hz sputter
- tinkles: N = 6 + 18·s charcoal tinkles over about 1 s. Gaps U(10, 80) ms for the first half, then U(30, 180) ms; levels fall to 50 %; pan ±0.25 around the log.
- **flare** +0.35·s on intensity (decays τ 2 s) and Hawkes +25·s
- logged → the visual drops the log and flares the flames

Charcoal tinkle: 3 modes, f1 = logU(2.2, 6.5 kHz), ratios 1 : U(1.55, 2.4) : U(2.9, 4.6) (inharmonic, brittle plate-like), τ U(4, 18) ms (×0.7, ×0.5), amps 1/0.6/0.35. Random ember settles happen at 0.15/s.

### 4.11 Gusts / whoosh
Gust event {strength g, dur d}: raised-cosine attack over 0.35·d, then exponential release τ = 0.4·d. The total `gust` (plus steady wind) drives:
- roar ×(1 + 1.4 g), knee +260·g Hz, flap AM depth 0.12 → 0.47
- breath ×(1 + g)
- lapping ×(1 + 1.2 g)
- hiss ×(1 + 0.8 g)
- crackle rate ×(1 + 1.5 gustLag) and ticks ×(1 + 0.8 gustLag), where gustLag = one-pole τ 0.4 s (fuel heats up *after* the oxygen arrives)
- pop rate ×(1 + gustLag)
- **whoosh layer:** pink noise → SVF BP **(280 + 750·min(g, 1)) Hz**, Q 0.9 (R ×1.12), amp g^1.5, pan sweeping −0.3 → +0.3 (wind direction)

### 4.12 Crickets (optional, scale and night)
- Dolbear's law: T(°F) = 50 + (N − 40)/4, N = chirps/min. About 110/min ≈ 20 °C.
- Field crickets (*Gryllus*): carrier **3.9-4.8 kHz**, which does not change with temperature. Chirps of **3-5 syllables**. Chirp and syllable rates rise linearly with temperature (*G. firmus*: ×4 and ×2 over 12-30 °C).

Per cricket (3 individuals):
- carrier U(4.1, 4.9 kHz) plus 8 % 2nd harmonic
- chirp period U(0.48, 0.66) s ± 6 % jitter, 8 % of chirps skipped
- 3-4 syllables, period U(36, 44) ms, duration U(14, 18) ms
- 3 ms raised-cosine attack, 5 ms release, 1.5 % downward glide per syllable
- one-pole LP 6 kHz (distance), pan U(±0.85), 40 % dry / 0.7 send (they are 10-30 m away)

### 4.13 Spatialization
- **Equal-power pan:** θ = (p + 1)·π/4, g_L = cos θ, g_R = sin θ.
- **Fire width:** point events (crackles, pops, tinkles) are scattered within ±0.55. Hero events take scripted pans that match where the visual sparks come from: `pan = x_screen / (fire half-width on screen)`.
- **Bed:** roar is mid + 0.45 side (correlation about 0.66); breath, ticks and hiss carriers are independent per channel. Measured final mix: **L/R correlation 0.58, mono fold-down −1.2 dB** (mono-safe; phones often play mono).
- **Avoid Haas delays > 1 ms on the bed.** They comb-filter in mono. Use decorrelated sources instead.
- **Distance (camera):**
  - gain (2/r)^0.8. A fire is an extended source, so attenuation is softer than 1/r near it.
  - high shelf at 5 kHz with gain −3.5 dB per doubling beyond 2 m (+ when closer), clamped to [−9, +4] dB.
  - The reverb send is not distance-scaled, so the direct-to-reverb ratio falls naturally as the camera pulls back.
- **Ground reflection:** source 0.3 m high, ear/mic 1.0-1.6 m, 2 m away gives a path difference of 0.3 m ≈ 0.9 ms. Implemented as 0.85 ms (L) / 1.05 ms (R) delays, one-pole LP 2.5 kHz (soft ground absorbs HF), gain 0.25.

### 4.14 Procedural outdoor reverb (IR from shaped noise)
Forest acoustics (Microsoft forest reverberator, MMSP 2021): a forest IR = direct + ground reflection + many single-scattering trunks. Measured forest energy decay curves fall about 40-50 dB in 0.4-1.0 s. Decay gets slower with distance, echo density is nearly constant, and T60 is frequency dependent.

IR recipe (`makeOutdoorIR`, deterministic per channel):
- Length 1.6 s.
- **Early reflections:** 10 taps at logU(9, 140 ms), gain −14 dB at 10 ms falling 9 dB per doubling of delay (σ 3 dB), random sign, one-pole LP 4.5 kHz, ×3.
- **Tail:** velvet noise (one ±1 pulse per grid cell), density 600 → 2500 pulses/s over the first 300 ms (sparse and "grainy" early, smooth later).
- Split into 3 bands (LP 400 Hz / BP 400-3k / HP 3 kHz, HF −4 dB) with **T60 = 0.55 / 0.85 / 0.6 s**, raised-cosine fade-in 12-60 ms, tail level −24 dB.
- Stereo: independent seeds per channel plus 15 % cross-feed after convolution.
- Use: offline FFT overlap-add (`render-offline.mjs`); live `ConvolverNode` with `disableNormalization: true`.

---

## 5. Coupling to the visuals

### 5.1 Shared timeline (the contract between the audio and video renderers)
`timeline.example.json`:
```json
{ "duration": 20, "sampleRate": 48000, "fps": 60, "seed": 20261025,
  "params": { "fireDiameter": 0.8, "width": 0.55, "crickets": 1 },
  "curves": {
    "intensity": [[0,0.75],[2.9,0.8],[3.3,1.05],[6,0.95],[9.5,1.0],[10.5,1.25],[13,1.1],[16,1.3],[18.5,1.2],[20,1.1]],
    "distance":  [[0,1.3],[5,1.5],[12,2.2],[20,3.0]],
    "wind":      [[0,0],[20,0]] },
  "events": [
    {"t":0.25,"type":"pop","size":0.6,"pan":0.22},
    {"t":3.2,"type":"pop","size":1.25,"pan":-0.1},
    {"t":5.7,"type":"pop","size":0.45,"pan":0.35},
    {"t":7.9,"type":"ember","pan":-0.3,"level":1},
    {"t":9.8,"type":"gust","strength":0.9,"dur":2.4},
    {"t":12.6,"type":"logShift","size":0.9,"pan":0.15},
    {"t":15.1,"type":"pop","size":0.8,"pan":0.3},
    {"t":17.4,"type":"crackleBurst","count":10,"pan":-0.2,"spacing":0.025,"size":1.5},
    {"t":18.3,"type":"pop","size":1.0,"pan":-0.25} ] }
```
- **intensity** I(t) (0-1.5) = the fire's heat-release proxy. The visual sim should use the same curve for fuel injection/brightness, or better, export its measured H(t) and use that as I(t).
- **distance** = camera-to-fire distance in metres (pull-back reveal: 1.3 → 3.0 m).
- Hero events are placed on frame boundaries (3.2 s × 60 = frame 192 exactly).
- The engine also creates **random** pops, gusts and log shifts. `render-offline.mjs --events events.json` writes every fired hero event with its frame number:
```json
{"t":3.024875,"type":"pop","size":0.398,"pan":0.533,"frame":181}
{"t":3.2,"type":"pop","size":1.25,"pan":-0.1,"frame":192}
```
- The video renderer spawns from this log:
  - pops → spark bursts (suggest about 120·size^1.5 sparks, launch speed ∝ √size, from x = pan·fireRadius)
  - logShift → log drop + flare
  - gust → flame lean
- Physical nicety: sound from 2 m arrives 5.8 ms after the light, about a third of a frame at 60 fps. Align exactly or delay the audio by r/343 s.

### 5.2 Physically driven mode (optional, Chadwick-style)
If the sim exports H(t) per frame:
1. Interpolate H(t) to the control rate with a Catmull-Rom or Mitchell-Netravali cubic.
2. Compute d = dH/dt, high-pass at 0.5 Hz, then take e = one-pole(|d|, τ = 40 ms).
3. Normalise e by its median and set `lnAmp_roar = 0.6·ln(e)` (clamped ±1.5), instead of (or blended with) the internal puffing modulator.

The roar AM then follows the flames' visible surging frame by frame, and the f^-3 carrier is the bandwidth extension.

### 5.3 A/V sync tolerances
ITU-R BT.1359 detectability thresholds: audio **early** by more than about 45 ms, or **late** by more than about 125 ms. Acceptability: about +90 / −185 ms. Never let pops lead the sparks. A few tens of ms late is invisible.

### 5.4 Live sync in the browser
- **Audio → visual** (random events originate in the worklet): the worklet posts `{t, contextTime}`. The main thread converts with `getOutputTimestamp()`:
```js
const ts = ctx.getOutputTimestamp();
const showAt = ts.performanceTime + (ev.contextTime - ts.contextTime) * 1000;  // performance.now() domain
// spawn sparks on the first rendered frame with rAF timestamp >= showAt
```
- **Visual → audio** (e.g., the physics sim decides a log falls): post `{type:'event', event:{type:'logShift', t: ctx.currentTime + 0.03}}`. The 30 ms scheduling margin makes it sample-accurate. Delay the visual by `ctx.outputLatency` (or `baseLatency`) if you want perfect alignment.
- Continuous controls: `AudioParam` k-rate `intensity`, `wind`, `distance`, updated once per frame with `setValueAtTime`. The engine smooths them with τ = 150 ms.

---

## 6. Web Audio implementation

### 6.1 Graph
```
AudioWorkletNode 'fire-audio' (0 inputs, 2 outputs: [dry 2ch], [send 2ch])
   out0 ──────────────────────────────► Gain (make-up ≈ +5 dB) ─► limiter ─► destination
   out1 ─► ConvolverNode(procedural IR, disableNormalization) ─┘
```
For the live limiter, `DynamicsCompressorNode {threshold −6, knee 3, ratio 20, attack 0.001, release 0.12}` is adequate. For mastering quality, port `masterChain()` to a streaming second worklet.

### 6.2 Worklet wrapper (complete file: `audio-ref/fire-worklet.js`)
```js
import { FireAudioEngine } from './fire-audio-dsp.mjs';   // static import OK in worklets (Chrome ≥ 80); import() is not
class FireProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() { return [
    { name: 'intensity', defaultValue: 0.8, minValue: 0, maxValue: 1.6, automationRate: 'k-rate' },
    { name: 'wind', defaultValue: 0, minValue: 0, maxValue: 1.5, automationRate: 'k-rate' },
    { name: 'distance', defaultValue: 2, minValue: 0.4, maxValue: 30, automationRate: 'k-rate' } ]; }
  constructor(o) { super();
    this.engine = new FireAudioEngine({ sampleRate, seed: o.processorOptions?.seed ?? 1 });
    this.t0 = currentTime; this.sent = 0;
    this.port.onmessage = (e) => { if (e.data.type === 'event') this.engine.schedule({ ...e.data.event, t: e.data.event.t - this.t0 }); };
  }
  process(inputs, outputs, p) {
    const e = this.engine, dry = outputs[0], send = outputs[1], n = dry[0].length; // never assume 128
    e.p.intensity = p.intensity[0]; e.p.wind = p.wind[0]; e.p.distance = p.distance[0];
    e.render(dry[0], dry[1], send[0], send[1], n);
    if (e.eventLog.length > this.sent) { this.port.postMessage({ type: 'events',
      events: e.eventLog.slice(this.sent).map(x => ({ ...x, contextTime: x.t + this.t0 })) }); this.sent = e.eventLog.length; }
    return true;
  }
}
registerProcessor('fire-audio', FireProcessor);
```
Main thread:
```js
const ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });   // create/resume on a user gesture (autoplay policy)
await ctx.audioWorklet.addModule('fire-worklet.js');                              // needs http(s)/localhost, not file://
const node = new AudioWorkletNode(ctx, 'fire-audio', { numberOfInputs: 0, numberOfOutputs: 2, outputChannelCount: [2, 2], processorOptions: { seed: 1337 } });
const [irL, irR] = makeOutdoorIR(ctx.sampleRate, 1337, { length: 1.6, t60: [0.55, 0.85, 0.6], tailDb: -24 });
const ir = ctx.createBuffer(2, irL.length, ctx.sampleRate); ir.copyToChannel(irL, 0); ir.copyToChannel(irR, 1);
const conv = new ConvolverNode(ctx, { buffer: ir, disableNormalization: true });
const makeup = new GainNode(ctx, { gain: 10 ** (5 / 20) });
const lim = new DynamicsCompressorNode(ctx, { threshold: -6, knee: 3, ratio: 20, attack: 0.001, release: 0.12 });
node.connect(makeup, 0); node.connect(conv, 1); conv.connect(makeup); makeup.connect(lim).connect(ctx.destination);
```

### 6.3 Real-time rules followed by the core
- **No allocation in `render()`.** Voice pools are preallocated (48 crackle, 40 modal, 16 noise voices) with deterministic stealing. Scratch arrays are reused.
- **No `Math.random`/`Date`.** All parameter smoothing is per control tick on the absolute sample grid, so the output is independent of quantum size.
- `process()` checks `outputs[0][0].length`. MDN warns the quantum may not stay 128. It returns `true`; a missing return equals `false` and the node stops.
- **Denormals:** V8 on x86 does not flush denormal doubles. IIR tails that decay with zero input can hit slow denormal arithmetic. Most filters here are always fed noise, and voice filters reset on start. If you add gated layers, add a 1e-20 offset or reset idle filters.
- **CPU:** measured **≈ 0.36 ms CPU per 128-frame quantum (budget 2.67 ms) while the machine was at 100 % load** from other jobs. That is about 7× real time, roughly 13 % of the audio thread. Cost is spread fairly evenly: roar 0.18, hiss 0.17, breath 0.16, ticks 0.15, crickets 0.14, pops 0.12, crackles 0.10 ms (each soloed together with the base cost). Cheap levers if needed: drop crickets, run the hiss carrier mono, share one pink generator.
- **Live test done:** `live-demo.html` in the desktop browser pane. The worklet loaded via static import, random pops appeared in the log at 4.7 s and 8.1 s, a button pop round-tripped to the main thread, and there were no console errors.

---

## 7. Deterministic offline render (Node) for the video

```bash
cd D:\fire\research\audio-ref
node render-offline.mjs --timeline timeline.example.json --out fire.wav --events events.json --stats
#   --solo crackles     render one layer (calibration / debugging)
#   --block 997         prove block-size independence (dryHash must not change)
#   --nomaster          skip loudness normalisation;  --gainDb x  fixed gain
node calibrate.mjs      # after any DSP change: prints a new CAL table (mix 0 dB = -20 LKu)
ffmpeg -hide_banner -i fire.wav -af ebur128=peak=true -f null -     # independent loudness / TP check
```

Pipeline inside `render-offline.mjs`:
1. Construct `FireAudioEngine({sampleRate: 48000, seed, params, curves})` and `schedule()` every timeline event.
2. Render in 128-frame blocks into dry and send buffers. This is the same code path as the worklet. The SHA-256 of the dry/send output is printed as `dryHash` and was **identical for 128/333/997-frame blocks**.
3. Convolve the send with `makeOutdoorIR` (FFT overlap-add, FFT size ≥ 2× IR length), cross-feed 15 %, sum with dry.
4. Master (section 8), measure with the built-in BS.1770-4 meter, iterate the gain 2-3 times to ±0.05 LU.
5. Write 24-bit PCM WAV with TPDF dither from a seeded LCG. Write `events.json` with frame numbers.

Mux with the frames:
```bash
ffmpeg -framerate 60 -i frames/%05d.png -i fire.wav -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \
       -c:a aac -b:a 320k -ar 48000 -movflags +faststart -shortest fire_20s.mp4
```
- Audio starts at sample 0 = frame 0; ffmpeg's MP4 edit list handles AAC priming.
- Verify sync once: put a 1-sample click at t = 3.2 s and a white frame at frame 192, then inspect in an editor.
- Re-check loudness on the final MP4 (`ffmpeg -i fire_20s.mp4 -af ebur128=peak=true -f null -`). AAC can raise true peaks by a few tenths of a dB, so keep ≥ 1 dB TP headroom.

---

## 8. Mastering for social media

Targets:
- **−14 LUFS integrated, ≤ −1 dBTP.** YouTube publishes −14 LUFS / −1 dBTP and only turns loud masters down. Instagram, TikTok and Facebook publish no official target; about −14 LUFS is observed. Meta uses adaptive loudness.
- Masters hotter than about −12 LUFS just get turned down and lose punch.
- Phone speakers roll off below about 200-300 Hz, which is why the roar has a 120/420 Hz body, tanh harmonics, and Farnell's clipped lapping harmonics.

**Chain** (`masterChain()`):
1. HP 30 Hz, 4th-order Butterworth (two biquads, Q 0.5412 and 1.3066). Sub-30 Hz costs headroom and is inaudible on target devices.
2. Peak +1.5 dB @ 220 Hz, Q 0.9 ("phone body"). High shelf −1.5 dB @ 10 kHz, S 0.7, to tame sizzle before AAC.
3. **Glue compressor:** stereo-linked RMS detector (10 ms), threshold = pre-master integrated loudness − 2 dB, **ratio 1.6:1, knee 8 dB, attack 20 ms, release 250 ms** (1-2 dB of gain reduction on swells).
4. Normalisation gain (iterated to −14.0 LUFS).
5. **Memoryless soft clipper:** linear below ceiling − 4.5 dB, tanh into ceiling + 1 dB. It shaves sub-millisecond crackle and pop peaks with **no release, so no pumping**. Clipping a 1-3 ms noise burst by ≤ 6 dB is perceptually transparent.
6. **Lookahead true-peak limiter:**
   - required gain from a 4×-oversampled peak estimate (16-tap/phase windowed sinc)
   - sliding minimum over 1.5 ms (72 samples)
   - **box-filter average of that minimum over the same window.** This provably guarantees g ≤ g_required at the delayed sample.
   - release one-pole 80 ms; delay compensated so events stay frame-accurate
   - internal ceiling −1.6 dBTP. The estimator reads about 0.4 dB under ffmpeg's, so this yields −1.3 dBTP measured.
7. 24-bit TPDF-dithered WAV, then AAC-LC 48 kHz 256-320 kbps in the MP4.

**Why it needs all of this: crest factor.** Measured on the first drafts:
- Pre-master the mix sat at about −18 LUFS. The biggest crackles peaked at **+4.5 dBFS pre-gain**. After +5 dB of make-up they needed up to 11 dB of clipping, about 12 times per second.
- Per-layer attribution (solo renders at the full-mix gain) found the culprits:

| Culprit | Fix | Result |
|---|---|---|
| 1-sample ticks | 4-14-sample bursts, denser and quieter | 2.6 → 0.7 overs/s, max 2.6 dB |
| roar LF peaks (K-weighting discounts LF, so peaks are big relative to loudness) | 3×RMS tanh + ±2.2σ AM cap, HP 45 Hz | 0.5 → 0.25 /s, max 8.5 → 3.4 dB |
| crackle amplitude ∝ √E over 32 dB | ∝ E^0.4 over 20 dB energy, b = 1.6 | max excess 6 dB |

- Avoid `loudnorm` single-pass dynamic mode (it pumps on fire). Two-pass `linear=true` falls back to dynamic mode when the TP target can't be met, which is likely with fire, so prefer the clipper + limiter above. If you must use ffmpeg:
```bash
ffmpeg -i mix.wav -af loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json -f null -             # pass 1
ffmpeg -i mix.wav -af loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=..:measured_TP=..:measured_LRA=..:measured_thresh=..:offset=..:linear=true -ar 48000 out.wav
```

**Measured result** (`demo_fire_20s.wav`, the example timeline):

| Metric | Value |
|---|---|
| Integrated loudness | **−14.0 LUFS** (internal −14.03; ffmpeg −14.0) |
| True peak | **−1.3 dBTP** (ffmpeg); sample peak −1.6 dBFS |
| LRA | 3.0 LU (−15.0 … −12.0) |
| Short-term (3 s windows, 1 s hop) | −15.3, −14.2, −13.9, −13.9, −14.2, −14.3, −14.8, −15.1, **−13.3, −12.2, −11.6** (gust), −12.5, −13.3, −13.7, −14.0, −14.2, −14.9, −16.0 |
| Pre-master loudness / make-up | −17.8 LUFS / +5.2 dB |
| Pre-limiter overs, full mix (gain applied before the glue comp, so slightly pessimistic) | 8.7 events/s over ceiling, max 7.9 dB, 0.39 % of samples; handled by soft clip + TP limiter |
| L/R correlation, mono fold-down | 0.58, −1.2 dB |
| Octave levels (dB, 31.5 Hz … 16 kHz) | −29.4, −27.2, −25.1, −27.2, −31.0, −30.9, −32.3, −31.2, −31.9, −37.9 |
| Octave kurtosis (Gaussian = 3) | 3.1, 6.3, 7.7, 8.2, 9.3, 9.4, 23.3, 24.3, 38.1, 27.7 |
| Spectrogram check | broadband vertical crackle streaks; 2.9 kHz ring + aftershock cluster + sizzle after the 3.2 s pop; cricket chirps at about 4.7 kHz; gust brightening of 100-1500 Hz at 10-12 s |

The 2-16 kHz kurtosis of 23-38 is sparse/peaky. No real-recording reference was available (no assets). If the texture sounds too "clicky" by ear, raise `tickRate` and lower the `ticks` mix, or add 1-2 dB of hiss. To make it more "crackly", do the opposite.

---

## 9. Suggested 20 s sound script (matches `timeline.example.json`)

| t (s) | Frame @60 | Sound | Picture |
|---|---|---|---|
| 0.00 | 0 | Starts **at full texture, no fade-in** (autoplay hook). Close camera 1.3 m: bright, intimate crackle bed, I = 0.75 | macro on flames and embers |
| 0.25 | 15 | pop s = 0.6 (hook) | small spark puff |
| ~3.02 | 181 | random small pop (engine-generated, in the event log) | tiny spark |
| **3.20** | **192** | **HERO pop s = 1.25:** click + 2-4 kHz ring + 90-200 Hz body + steam "psssh" + aftershock crackles + sizzle; intensity ramps 0.8 → 1.05 | **big spark burst (about 170 sparks)**, flame flare |
| 5.7 | 342 | pop s = 0.45 | sparks |
| 7.9 | 474 | ember tinkle | ember settles |
| 9.8-12.2 | 588-732 | **gust:** whoosh sweeps 280 → 1000 Hz, roar +, flapping, crackle surge after 0.4 s (loudest moment, −11.6 LUFS short-term) | flames lean, stream sideways |
| 12.6 | 756 | **log shift:** thump + knock + charcoal tinkle cascade (about 1 s) + flare | log drops, sparks and embers tumble |
| 15.1 | 906 | pop s = 0.8 | sparks |
| 17.4 | 1044 | crackle burst (10 × 25 ms) | flicker |
| 18.3 | 1098 | pop s = 1.0 (final punctuation) | sparks |
| 20.0 | 1200 | end. Camera has pulled back to 3 m: darker, more reverberant | wide shot |

The opening sits about 1-2 LU below the middle. If the edit needs a bigger hook, start at I = 0.9-1.0 and move the hero pop earlier (frame 60-120).

---

## 10. Tuning checklist and pitfalls

1. **Calibrate by measurement, not by guessing.** Filter-chain gains are wildly non-obvious (Kellet pink DC gain about 70×; band-passed noise loses 10-20 dB). Run `calibrate.mjs` after every DSP change, then mix in "dB re −20 LKu".
2. **Measure overs per layer** (`--solo X --gainDb <full-mix gain>`) before blaming the limiter.
3. **Check by ear on a phone speaker and in mono.** If the roar disappears on the phone, add harmonics (tanh drive, 120/420 Hz body) rather than more sub.
4. **Crackles sound like rain or water?** Increase the broadband raw fraction, sub-impulses and Hawkes clustering. Independent band-limited blips read as water (McDermott).
5. **Too metronomic?** You probably have a fixed-rate or plain Poisson spawn. Use Hawkes plus intensity/puffing-modulated rate plus aftershocks.
6. **Pops sound like clicks?** Add the body (90-200 Hz), the ring, the fracture cascade and a reverb send of 0.4. Pops are the only events that should clearly excite the space.
7. **Roar sounds like a waterfall or wind?** Too bright: lower the knee (150-250 Hz), keep α ≈ 3. Make sure puffing AM uses f = 1.5/√D and isn't too deep (σ_ln ≈ 0.4).
8. Keep `fireDiameter` consistent with the visual fire size, and drive the visual flicker from the same puffing frequency.
9. **Never use per-block control updates** (output becomes block-size dependent). Never use `Math.random`. Keep one RNG stream per layer.
10. AudioWorklet: secure context, user-gesture `resume()`, static imports only, ConvolverNode `disableNormalization: true`. Request 48 kHz; Chrome resamples if the device runs at 44.1 kHz.

---

## 11. References
- A. Farnell, *Designing Sound* (MIT Press, 2010), Practical 11 "Fire". Companion Pd patches with the exact values above: https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/8375/designing_sound.zip/practical11.html
- J. N. Chadwick, D. L. James, "Animating Fire with Sound", ACM TOG 30(4), SIGGRAPH 2011. Project: https://www.cs.cornell.edu/projects/Sound/fire/ · PDF: https://www.cs.cornell.edu/projects/Sound/fire/FireSound2011.pdf
- R. Rajaram, T. Lieuwen, "Acoustic radiation from turbulent premixed flames", J. Fluid Mech. 637 (2009): https://resolve.cambridge.org/core/journals/journal-of-fluid-mechanics/article/acoustic-radiation-from-turbulent-premixed-flames/BC21E9A40FCE66B7B4A808A8A107CB99 · thesis: https://repository.gatech.edu/entities/publication/ddd71da2-2094-4628-8e77-faca08d347f5
- D. I. Abugov, O. I. Obrezkov (1978); P. Clavin, E. Siggia (1991). Via Chadwick & James §2.3.
- S. Kersten, H. Purwins, "Fire texture sound re-synthesis using sparse decomposition and noise modelling", DAFx-12: https://dafx.de/paper-archive/2012/papers/dafx12_submission_65.pdf
- J. McDermott, A. Oxenham, E. Simoncelli, "Sound texture synthesis via filter statistics", WASPAA 2009: https://mcdermottlab.mit.edu/papers/McDermott_Oxenham_Simoncelli_2009_sound_texture_synthesis.pdf
- J. Sethna, K. Dahmen, C. Myers, "Crackling noise" (Nature 2001): https://www.cac.cornell.edu/myers/papers/crackling_preprint.pdf
- Acoustic emission from breaking a bamboo chopstick (b ≈ 1.45, Omori p ≈ 1.68): https://ar5iv.labs.arxiv.org/html/1509.00631
- Flicker scaling (f = 0.48√(g/D) ≡ 1.5/√D, Cetegen & Ahmed): https://arxiv.org/abs/1803.10400 · NIST record: https://firedoc.nist.gov/article/LncyXYQBWEcjUZEYkZ_8
- Zhang et al., "Wildfire detection using sound spectrum analysis", Sensors 2019: https://pmc.ncbi.nlm.nih.gov/articles/PMC6929188/
- Yedinak et al., "Vegetation effects on impulsive events in the acoustic signature of fires", JASA 141 (2017): https://nrfirescience.org/node/20898
- Acoustic observations of a prescribed burn (infrasound 2-90 Hz): https://impact.ornl.gov/en/publications/acoustic-observations-of-a-prescribed-burn/
- E. Virot, A. Ponomarenko, "Popcorn: critical temperature, jump and sound", J. R. Soc. Interface 2015: https://hal-polytechnique.archives-ouvertes.fr/hal-01178414
- "A fast forest reverberator using single scattering cylinders", MMSP 2021: https://microsoft.com/en-us/research/uploads/prod/2021/10/A-fast-forest-reverberator-using-single-scattering-cylinders-MMSP-2021.pdf
- W3C Audio EQ Cookbook (RBJ): https://www.w3.org/TR/audio-eq-cookbook/
- A. Simper (Cytomic), "Linear trapezoidal integrated SVF": https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
- MDN `AudioWorkletProcessor.process()`: https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor/process · Worklet static imports: https://developer.mozilla.org/en-US/docs/Web/API/Worklet
- ITU-R BS.1770-4 loudness/true-peak (K-weighting implemented via the libebur128 analog prototype: 1681.97 Hz shelf +4 dB, 38.135 Hz HP): https://www.itu.int/rec/R-REC-BS.1770
- Platform loudness targets (YouTube −14 LUFS/−1 dBTP; others unpublished): https://www.forasoft.com/learn/audio-for-video/articles-audio/lufs-targets-per-platform-2026
- ffmpeg loudnorm two-pass notes: https://ffmpeg.org/pipermail/ffmpeg-devel/2018-August/233372.html
- Gryllus calling song vs temperature: https://link.springer.com/article/10.1007/BF00195962 · T. J. Walker cricket song data: https://entnemdept.ifas.ufl.edu/walker/buzz/s610wf14jor23p1.pdf
- QuAP / Nemisindo procedural model MUSHRA scores: https://arxiv.org/abs/2606.00629
- ITU-R BT.1359-1 (1998), relative timing of sound and vision for broadcasting.
