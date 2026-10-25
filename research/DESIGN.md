# DESIGN.md: procedural campfire (WebGPU real-time, procedural audio, 20 s cinematic)

Status: authoritative build spec, version 1. Owner: lead TD. This spec merges the research reports:

| Source | Location | Used for |
|---|---|---|
| physics | `research/physics.md` | sim model, constants, calibration tests |
| rendering | agent report (not written to disk); code in `research/rendering_code/` | emission/LUT, tone mapper, firelight, post |
| audio | `research/audio.md`, `research/audio-ref/` | audio model, mastering |
| visual reference | `research/visual_reference.md` (+ `refs_tools/analyze.py`, `video.py` in the session scratchpad) | targets, checklist |
| WebGPU | `research/webgpu.md`, `research/webgpu-bench/` | formats, solver, budgets, capture, ffmpeg |

Where the reports disagree, this document decides. Do not tune by eye alone: every look parameter has a measurable target in §6.

---

## 0. Decisions at a glance

| Topic | Decision |
|---|---|
| Subject | Night campfire: 2 base logs + 5-log teepee + charred chunk on a coal bed, stone ring, forest floor, distant tree silhouettes. **Q0 = 60 kW, fuel-bed D = 0.6 m.** |
| Physical targets | Mean flame height **0.60 m** (Heskestad). Continuous flame to 0.41 m, intermittent tongues to about 1.0 m. Peak mean gas speed **4.3 m/s**. **Puffing at 1.94 Hz.** Pyrolysis rate 4.3 g/s. |
| Units | SI everywhere: m, s, K, g = 9.81. Velocity in m/s; convert to cells only inside the back-trace. |
| Domain | 1.2 × 1.8 × 1.2 m, origin at the fire centre on the ground, y up (same as the prototype). |
| Sim | Low-Mach, constant-density projection with an expansion source; fuel + oxidiser mixing-limited burning; soot formation gated by gas age; MacCormack advection; multigrid V(2,2) with face-Dirichlet open boundaries. |
| Real-time grid | 80×120×80 (h = 1.5 cm), dt = 1/120 s, 1 V-cycle. Low tier 64×96×64. |
| Video grids | Hero: **160×240×160** (h = 7.5 mm), dt = 1/360 s, 3 V-cycles; stretch 192×288×192. Macro (shot 1 only): **192×224×192 over 0.8×0.93×0.8 m** (h = 4.17 mm), dt = 1/600 s. |
| Emission | Kirchhoff soot emission `j = σa,soot(S)·B(T)` from a CIE-integrated blackbody LUT, brightness contrast γ (0.6 real-time / 0.75 offline). Blue chemiluminescence ∝ heat-release rate. |
| Camera colour | White balance **3200 K** (Bradford, applied once in post). **AgX-Fire** tone mapper (ACES-Hill as alternate). Per-channel path to white. |
| Light | Fire light = the volume's own emission: 512 virtual point lights → 32³/64³ smoke light volume, and 8 clustered surface lights. No synthetic flicker. |
| Surfaces | SDF scene: alligator-cracked char with glowing crack floors, granular ember bed, ash, stones, ground, trees. Same LUT, γ and scale as the flames. |
| Sparks | GPU particles that burn: log-normal sizes, drag driven by the sim velocity, d²-law burnout, energy-conserving motion-blurred streaks with depth of field. Bursts come from audio events. |
| Audio | Existing engine `src/audio/fire-dsp.js` (Farnell layers + Hawkes crackles + pops + Chadwick roar). Coupled to the sim's HRR and dHRR/dt. Offline render with `tools/render_audio.mjs`, mastered to **−14 LUFS, ≤ −1 dBTP**. |
| Video | **1080×1920 vertical, 30 fps**, 180° shutter. 20.0 s, 5 shots; the hook is a slow-motion macro of the flame root plus a hero pop at 1.05 s. Key action stays inside the centre 4:5 safe area. H.264 High CRF 14 delivery with BT.709 tags, plus a 10-bit HEVC master. Optional 16:9 render from the same sims. |
| Capture | System Chrome in new headless mode via puppeteer-core; render to your own texture; RGB→YUV on the GPU; binary WebSocket → ffmpeg stdin; deterministic re-simulation and checkpoints. |

### 0.1 Current prototype state (from `out/shots/*.png`) and what this spec changes

The prototype (`src/main.js`, `sim.js`, `renderer.js`, …) already follows most of the physics recipe and renders a recognisable campfire. Gaps visible in the shots:

1. **Flames too tall, with vertical streaks** (`m.png`: tongues reach about 3× the teepee height, with striated, noise-like texture). Run the flame-height and intermittency test (§2.13 T4) and the HRR controller (§2.10). Change render-time detail to advected two-phase noise (§3.4) instead of stretched static noise.
2. **Far too much bright smoke and fog** (`e32.png`). The smoke source must drop about 5–10×, the plume target is σt 0.1–0.4 m⁻¹ (§2.6), and smoke scattering must use the HG phase function with g = 0.6.
3. **Logs read as brown bark**, not black alligator char with glowing crack rims. **Stones are evenly lit** and the **ground is flat**: implement §3.7.
4. The white balance default is 5000 K and baked into the LUT. Change to a 3200 K post matrix with an unbalanced D65 LUT (§3.2, §3.10).
5. Some test shots are nearly black (`u1.png`). Fix the exposure to relative units (§3.1) and stop relying on ad-hoc `exposure: 25`.

---

## 1. Architecture

### 1.1 Files and modules

Keep the existing layout and add the modules marked **new**.

```
index.html                 real-time app shell (canvas + HUD + "click for sound" overlay)
src/main.js                entry; URL params (?q=low|med|high, ?test, ?capture, ?timeline=…); window.FIRE API
src/gpu.js                 device init (features + limits, §1.7), tex3D/2D helpers, UniformStruct, Profiler
src/sim.js                 FireSim: fields, kernels, multigrid, HRR reduction + controller (§2)
src/emitters.js      (new) fuel-source shaping: per-log burn state, pyrolysis patches, coal bed, dynamic log
                           transforms (log shift); writes the `geo` texture (§2.4)
src/scene.js               seeded procedural layout: logs, stones, coal bed, ground, trees
src/shaders/lib.js         HASH (PCG3D), GNOISE, curl noise, SDF primitives, IGN
src/shaders/scene.js       SDF scene + materials: char cracks, embers, ash, stones, ground, trees, sky (§3.7)
src/blackbody.js           LUT generator: port of research/rendering_code/blackbodyLut.js (CIE 1931 table), §3.2
src/renderer.js            render graph (§1.4): pack volume, firelight, surfaces, march, upsample/TAA, composite, post
src/firelight.js     (new) VPL reduction, light clusters, fluence volume (from rendering_code/firelight.wgsl, fluence.wgsl)
src/sparks.js        (new) spark particle system: spawn, integrate, render streaks (§3.8)
src/camera.js              orbit camera + thin-lens physical camera + keyframed path evaluation
src/timeline.js      (new) timeline JSON evaluation: curves, events, time-warp, shot cameras, gust envelope
                           (one envelope function shared with the audio engine)
src/capture.js       (new) capture mode: fixed-step loop, subframe accumulation, GPU YUV, WebSocket send, checkpoints
src/audio/fire-dsp.js      audio DSP core (existing, canonical); engine.js / worklet.js (live); master-offline.js (offline)
tools/server.mjs           static server on 127.0.0.1 (existing); add the /frames WebSocket sink from webgpu-bench/ws.mjs
tools/shot.mjs             headless stills harness (existing)
tools/capture.mjs    (new) headless video capture → ffmpeg (§5.4)
tools/render_audio.mjs     offline audio (existing)
tools/mux.mjs        (new) audio + video mux + verification (ffprobe tags, ebur128, sync)
tools/fire_metrics.py      image and video metrics vs the visual reference (existing; extend with analyze.py metrics)
video/timeline.json  (new) single source of truth for the 20 s film: shots, cameras, curves, events (§5.3)
out/ckpt/, out/video/      checkpoints, frames, deliverables
```

### 1.2 Data flow

```
timeline.json ──► timeline.js ──► sim controls (Q target, wind, gusts, events) ──► FireSim.step()   × N substeps
                        │                                                              │
                        ├──► camera keys ──► Renderer (per subframe) ◄────── fields ────┘
                        │                         │
                        │                 sparks.js ◄── spark bursts (events, pops) ── sim velocity
                        │                         ▼
                        │                linear HDR accum ──► post ──► YUV ──► WS ──► ffmpeg ──► video.mp4
                        │
                        └──► audio events ──► render_audio.mjs ◄── per-frame {intensity, hrr, dhrr, camDist, camAz}
                                                   │                  (written by the capture run: out/video/frames.json)
                                                   ▼
                                               fire.wav ──► mux.mjs ──► fire_1080x1920.mp4
```

Real-time: the same modules. Audio runs in an AudioWorklet and receives `intensity`, `hrr` and `dhrr` from an async HRR readback, about 2–3 frames late, which is fine. Worklet pops are posted back and spawn spark bursts at `getOutputTimestamp()` time.

### 1.3 Simulation pipeline (one substep; mostly fused, as in the prototype)

| # | Pass | Dispatches | Notes |
|---|---|---|---|
| 0 | Upload per-step uniforms | – | **One uniform slot per substep** (256-B dynamic offsets) or one submit per substep. Several `writeBuffer`s before one submit: the last one wins. |
| 1 | `emitters` (only if logs moved or burn state changed) | 1 | Rewrites `geo` (SDF, normal, pyrolysis weight, bed mask). Otherwise cached. |
| 2 | Advect velocity: MacCormack, RK2 back-trace, min/max limiter | 2 | Velocity in m/s; back-trace in cells = dt·u/h. Sanitise NaN/Inf with a bit test; clamp \|u\| to 4 cells per substep. |
| 3 | Advect scalars S, S2 together (MacCormack) | 2 | One back-trace advects 8 channels (two textures). |
| 4 | Curl ω (rgba16f: ω, \|ω\|) | 1 | |
| 5 | **React + forces** (fused) | 1 | Sources → combustion → soot → cooling → expansion target φ → buoyancy → confinement → curl noise → wind → solids/sponge → ambient shell. Writes vel (w = φ), S, S2. |
| 6 | Divergence − φ | 1 | |
| 7 | Multigrid V(2,2) × N, warm-started | about 60–115 | Ghost value −p at open faces, +p at the floor and solids. |
| 8 | Project, plus solid and floor velocity rules | 1 | |
| 9 | Sparks integrate (offline: every substep; real-time: every frame) | 1 | Reads velocity and temperature. |
| 10 | HRR reduction + controller (once per frame) | 2 | Deterministic tree reduction, no float atomics (§2.10). |

### 1.4 Render graph

**Real-time, per frame:**
1. `packVolume`: from S and S2, write `R` (rgba16f, sim resolution): x = T [K]/1000, y = σa,soot [m⁻¹], z = smoke M, w = blue emission weight. Also writes 8³ brick occupancy (max σt, max emission, 1-voxel halo) and the bounding box of non-empty bricks.
2. `firelight`: reduceEmission (8³ coarse cells → 512 VPLs, each with intensity rgb, luminance-weighted centroid and radius) → cluster into 8 surface lights (2×2×2 octants) → buildFireLight fluence volume F (32³, 4-tap shadow through ¼-res σt).
3. `surfaces`: full-res SDF ray march of ground, stones, logs, ember bed and trees. Output HDR radiance (rgba16f) + linear depth (r32f).
4. `march`: half-res volume march, clipped at surface depth. Output premultiplied radiance + alpha (rgba16f), emission-weighted depth (r32f), haze integral (r16f).
5. `upsample` (depth-aware joint bilateral) + `TAA` (history 0.75–0.8, YCoCg 3×3 clamp, reprojection by emission-weighted depth).
6. `sparks`: additive streak rasterisation into HDR, depth-tested against surfaces and attenuated by the volume alpha when behind the emission depth.
7. `composite`: heat haze displaces the surfaces' background lookup, then volume over, then sparks.
8. `bloom`: 6 levels.
9. `final`: chromatic aberration → bloom mix → vignette → WB matrix → exposure → AgX-Fire → sRGB OETF → grain → dither → canvas.

**Offline, per output frame:** for each subframe (§3.11) run steps 1–7 at full resolution with no TAA and no upsample. Accumulate linear HDR in `rgba32float`. Once per frame, run steps 8–9 on the accumulated average, then GPU RGB→YUV with TPDF dither (8-bit delivery, 10-bit master).

Never tone-map before averaging subframes: it breaks the path to white.

### 1.5 Textures and buffers

| Resource | Format | Size | Notes |
|---|---|---|---|
| `vel` A/B (+ MacCormack hat) | rgba16float 3D | sim | xyz m/s, w = φ (1/s) |
| `S` A/B (+ hat) | rgba16float 3D | sim | x θ = (T−293)/1000, y fuel f, z soot S [ppm], w q [1e4 K/s] |
| `S2` A/B (+ hat) | rgba16float 3D | sim | x products P (oxygen o = 1 − f − P), y age A [s], z smoke M [×100 mg/m³], w tar fraction τ (sooting propensity) |
| `vort` | rgba16float 3D | sim | ω, \|ω\| |
| `geo` | rgba16float 3D | sim | x pyrolysis weight, y coal-bed weight, z solid, w log SDF [m] |
| `p`, `b`, residual mips | r32float 3D | MG chain | needs `float32-filterable` for prolongation |
| `R` (render volume) | rgba16float 3D | sim | one fetch per march step |
| bricks | rgba16float 3D | sim/8 | max σt, max emission |
| VPL buffer | storage | 512 × 32 B | rgb intensity, centroid, radius |
| clusters | storage | 8 × 32 B (+ 4 bed lights) | surface lights |
| fluence F | rgba16float 3D | 32³ (rt) / 64³ (offline) | fire irradiance for smoke scattering |
| blackbody LUT | rgba16float 2D | 256×1 (or the existing 1024×2 rgba32f) | rgb = chroma at Y = 1 (D65, unbalanced), a = log2(Y/Y1500) |
| HDR scene | rgba16float | screen | values kept in relative units below 1e4 (f16 max is 65504) |
| surface depth | r32float | screen | |
| volume colour / depth / haze | rgba16f / r32f / r16f | ½ (rt), full (offline) | |
| TAA history | rgba16float | screen | real-time only |
| accum | rgba32float | screen | offline only |
| bloom chain | rg11b10ufloat (or rgba16f) | 6–7 mips | |
| sparks | storage | 4k (rt) / 16k (offline) × 64 B | CPU-managed ring buffer, no atomics |
| HRR partials / controller | storage | ⌈cells/256⌉ × 16 B + 64 B | GPU-side controller state |
| YUV out | storage | W·H·1.5 B (8-bit) / ×2 (10-bit) | packed u32, no 256-B row padding |

Memory: about 67 B/cell for the sim: 80×120×80 → 52 MB, 160×240×160 → 410 MB, 192×288×192 → 710 MB. All fit easily.

### 1.6 Quality tiers

| | RT low | **RT default** | RT high | **Video hero** | **Video macro (shot 1)** |
|---|---|---|---|---|---|
| Sim grid | 64×96×64 | 80×120×80 | 96×144×96 | 160×240×160 (stretch 192×288×192) | 192×224×192 |
| Domain | 1.2×1.8×1.2 | same | same | same | 0.8×0.933×0.8 around the fire centre |
| h | 1.875 cm | 1.5 cm | 1.25 cm | 7.5 mm | 4.17 mm |
| dt | 1/120 | 1/120 | 1/120 | 1/360 | 1/600 |
| CFL at 4.3 m/s (at 6 m/s) | 1.9 (2.7) | 2.4 (3.3) | 2.9 (4.0) | 1.6 (2.2) | 1.7 (2.4) |
| MG V(2,2) per step | 1 | 1 | 1–2 | 3 | 3 |
| MG levels (coarsest) | 64→4×6×4 (5 levels) | 80→10×15×10 (4) | 96→6×9×6 (5) | 160→10×15×10 (5) | 192→6×7×6 (6) |
| March resolution | ½ | ½ | ½–⅔ | full 1080×1920 | full |
| Step size | 1 voxel | 0.75 | 0.75 | 0.35 | 0.35 |
| Max steps | 128 | 160 | 192 | 768 | 768 |
| Fluence | 32³, 4 taps | 32³, 4 taps | 32³, 4 taps | 64³, 8 taps | 64³, 8 taps |
| Surface shadows | 2 of 8 lights | 4 of 8 | 8 | 8, 64 SDF steps | 8 |
| Temporal | TAA | TAA | TAA | 12 subframes/frame | 12 subframes/frame |
| Sparks max | 2k | 4k | 4k | 16k | 16k |
| Target | 60 fps | 45–60 fps idle, ≥ 30 fps loaded | 30 fps | about 4–5 s/frame (loaded) | same |

When the coarsest MG level has odd dims (10×15×10), do 32 red-black sweeps there instead of 16.

The real-time loop is a fixed-dt accumulator: `acc += min(frameDt, 0.05); while (acc >= dt && n++ < 4) step()`. If the GPU cannot keep up, sim time dilates. Never enlarge dt.

### 1.7 Device setup (`src/gpu.js`)

- `requestAdapter({ powerPreference: 'high-performance' })`.
- Request features when present: `texture-formats-tier2` (plural name), `float32-filterable`, `shader-f16`, `timestamp-query`, `subgroups`.
- Request limits: `maxStorageTexturesPerShaderStage: 8`, `maxComputeWorkgroupStorageSize: 32768`, `maxComputeInvocationsPerWorkgroup: 1024`, `maxBufferSize` and `maxStorageBufferBindingSize: 1 GiB`. Otherwise you get the spec defaults.
- Use explicit bind-group layouts, not `layout: 'auto'`.
- Workgroup size 8×8×4.
- Rules: never name an entry point `smooth`. `bytesPerRow` must be a multiple of 256. Copy mapped data out before `unmap()`. Parenthesise mixed bitwise and arithmetic operators.
- Keep each submit under 0.5 s (the Windows TDR limit is 2 s). In capture mode, await `onSubmittedWorkDone()` between substeps.

### 1.8 Determinism (required for capture, resume and A/V sync)

- Fixed dt and fixed substep counts. The RNG is `pcg3d(cell, step, seed)`. Never use `Math.random()` or `performance.now()` in sim or capture paths.
- Red-black multigrid only (no racy Gauss-Seidel). No float atomics. Use fixed-order tree reductions.
- Sparks: the CPU assigns ring-buffer slots `[head, head+N)` per spawn event, and per-particle parameters come from `hash(eventId, i)`. No GPU allocation.
- The webgpu agent measured that the same code hashes identically across runs, processes, and headless versus headful. Add an `FIRE.hash()` debug call (FNV over a readback) and check it in CI-style runs.

### 1.9 Control surface (`window.FIRE`, used by tools and the UI)

`FIRE.step(n)`, `FIRE.set(state)`, `FIRE.setSim(params)`, `FIRE.setRender(params)` (existing), plus:
- `FIRE.loadTimeline(json)`, `FIRE.seek(videoTime)` (re-simulate or load a checkpoint)
- `FIRE.event({type, …})`, `FIRE.hrr()`, `FIRE.hash()`
- `FIRE.checkpoint()` and `FIRE.restore(buf)`
- `FIRE.capture({...})` (§5.4)

Real-time UI keys:
- **G** gust
- **L** drop a new log
- **click on embers**: stir (spark burst + crackle)
- **P** photo mode (64 accumulated jittered frames)
- **H** sliders: intensity, wind, white balance, exposure, quality
- **T** play `video/timeline.json` in real time (previz)

---

## 2. Simulation model

### 2.1 Fire sizing

| Quantity | Value | Formula / source |
|---|---|---|
| Q0 | 60 kW (timeline intensity I(t) scales it, range 0.6–1.6) | physics §4.4 |
| D (active bed) | 0.6 m | |
| Mean flame height L | 0.235·Q^0.4 − 1.02·D = 0.60 m | Heskestad |
| Continuous / intermittent limit | 0.08·Q^0.4 = 0.41 m / 0.20·Q^0.4 = 1.03 m | McCaffrey |
| Peak mean centreline u | 1.9·Q^0.2 = 4.3 m/s | McCaffrey |
| Puffing f | 1.5/√D = 1.94 Hz | Cetegen & Ahmed |
| Plume at z = 1.0 / 1.5 m (Qc = 42 kW, z0 = −0.19 m) | ΔT0 = 228 / 127 K; u0 = 3.4 / 3.0 m/s | Heskestad plume |
| Pyrolysis rate | Q/ΔHc = 60 kW / 14 MJ/kg = 4.3 g/s | |
| Audio `fireDiameter` | set to (1.5/f_measured)² once T2 has measured the sim's puffing frequency, so sound and picture puff together | |

### 2.2 Domain, units and boundaries

- **Domain:** 1.2 × 1.8 × 1.2 m, cell-centred collocated grid, T∞ = 293 K, p0 = 101.3 kPa.
- **Floor (y = 0):** solid, free-slip. Pressure ghost = +p_c. After projection, v.y = max(v.y, 0) at y = 0.
- **Sides and top:** open, pressure ghost = −p_c (face Dirichlet). The usual "ghost = 0" makes multigrid stall.
- **Ambient shell:** each step, force the outermost voxel layer on the sides to ambient scalars (T∞, f = 0, P = 0, S = 0, M = 0, A = large). Clamp-to-edge would otherwise recirculate interior values back in.
- **Sponge:** over the 4 outer cells at the sides, relax horizontal velocity toward the wind profile at 5 s⁻¹·ramp. At the top, use 6 cells of scalar fade (S, M ×(1 − 0.5·ramp·dt·10)) so smoke never ends in a hard line.
- **Logs:** solids from `geo`. Inside a solid, velocity = solid velocity (zero unless shifting) and scalars are zeroed. Pressure ghost at a solid face = +p_c. After projection, overwrite the normal velocity at solid faces.

### 2.3 Fields

The table in §1.5 lists the fields. Derived quantities:
- oxygen availability `o = clamp(1 − f − P, 0, 1)`
- gas density `ρ = p0/(R·T)` with R = 287
- `T = 293 + 1000·θ`

Keep T in θ form for fp16 precision (step 0.5–1 K near 1500 K).

### 2.4 Fuel source: procedural logs and coal bed (`src/emitters.js`)

**Geometry** (`src/scene.js`, seeded; the existing layout):
- Two base logs: r = 6 cm, length about 0.8 m, burn 0.8.
- Five teepee logs: r = 3.2–4.4 cm, tops at 0.36–0.46 m, burn 1.0.
- A charred chunk.
- Coal bed: a disc of radius 0.24 m, height 3–5 cm.
- 13 stones on a ring of radius 0.6 m.

Logs are capsules with SDF displacement: char blocks ±1.5 mm, bark ±3 mm on unburnt parts.

**Pyrolysis weight** W(x) on the voxel shell 0 < sdf < 2h outside each log. Multiply these factors:
- **burn** (per log, 0–1).
- **Heat exposure:** `0.35 + 0.65·smoothstep(−0.2, 0.6, dot(n, normalize(c_fire − x)))`, where c_fire = (0, 0.15, 0). Faces toward the fire burn.
- **Underside:** ×1.5 where n.y < −0.3.
- **Gap / chimney:** ×2.0 where a second log is within 4 cm (second-nearest SDF < 0.04 m). The strongest flames come from gaps between logs.
- **Patches:** `smoothstep(0.1, 0.45, fbm3(x_log/0.08 + drift))`, in log-local coordinates. Two octaves, 5–15 cm scale. The patch pattern drifts along and around the log at 0.5–3 s time scales. Target 30–50 % active coverage.
- **Breathing:** per-patch slow modulation `1 + 0.3·noise(t/1.5 s)`.

**Injection** in shell cells:

```
f   = max(f, G·W·f_src)      f_src = 0.4 (pyrolysis-gas mass fraction at the surface)
τ   = 1 (sooting tar)
T   = max(T, 750 K + 150 K·W)            hot char surface acts as the pilot
u  += (v_n·n − u)·(1 − exp(−20·dt))      v_n = 0.15 m/s  (physical ~2 cm/s, raised so a coarse grid resolves it)
A   = 0
```

G is the HRR controller gain (§2.10).

**Coal bed:**
- Cells with bed weight get T = 1050–1150 K.
- Weak CO fuel: f_src = 0.12, τ = 0 (no soot) → faint blue flamelets, 2–10 cm.
- Small organic smoke at smouldering edges: M += 0.02·dt.

**Smouldering spots:** 3–5 seeded spots on low-burn log ends at 550 K, emitting M at 0.05 units/s each. They are the only steady source of white smoke.

**Log shift** (event): animate the capsule endpoints along a keyframed path (slip 4 cm, rotate 6°, over 0.25 s with ease-in, then a 1 cm bounce). Recompute `geo` every substep while moving and give solid cells the capsule's velocity. When it lands: fuel G ×1.6 decaying with τ = 1.5 s, a spark burst, and the audio `log-shift` event. **Fallback:** keep the log static and use a pop, a spark burst and a flare.

### 2.5 Combustion (mixing-limited, fast chemistry)

```
ign  = smoothstep(T_ign − 50, T_ign + 50, T)                    T_ign = 600 K
b    = min(f, o/r_s) · (1 − exp(−k_mix·ign·dt))                 r_s = 4.7, k_mix = 30 s⁻¹ (rt) / 45 s⁻¹ (video)
f   −= b;  P += (1 + r_s)·b                                     (products consume oxygen; o recomputed)
ΔT   = η·ΔT_f·b = 7700 K · b                                    (ΔHc/cp ≈ 11 000 K × η 0.7)
T   += ΔT;  q = ΔT/dt  [K/s]
```

Check: a stoichiometric parcel (f = 0.175) gains about 1350 K and peaks near 1650 K. Mixing and cooling bring the continuous-zone mean to about 1150 K.

**Do not** use the simpler "burn wherever T > T_ign" model. The oxygen limit is what produces hollow, tearing flame sheets instead of fireball blobs.

### 2.6 Soot (ppm) and smoke

```
f_rich = f/(f + o/r_s + 1e-4)
dS/dt  = + k_sf · τ · f_rich · gate(T, 1300, 1800 K) · gate(A, 0.02, 0.06 s)      k_sf tuned so S_peak ≈ 1–2 ppm in sheets
dS/dt  = − k_ox · S · o · gate(T, 1100, 1300 K)                                  k_ox = 20 s⁻¹
dS/dt  = − 0.1 s⁻¹ · S where T < 900 K                                           sub-visible dilution
dM/dt  = + 2.0 s⁻¹ · f · (1 − ign) · (1 − gate(T, 450, 650 K))                   unburnt pyrolysis gas cooled below
         ~500 K condenses to tar aerosol; f −= same amount
dM/dt  = − 0.15 s⁻¹ · M                                                          dilution
```

`gate(x, a, b) = smoothstep(a, b, x)`.
- The **age gate** produces the 1–3 cm blue base.
- The **oxidation gate** makes the luminous tip end abruptly.

Calibration targets:
- **Sheets:** S_peak 1–2 ppm; mean S over the flame body 0.2–0.5 ppm. Median optical depth through the flame body along camera rays should be **0.3–1.5**, so logs remain visible through flames at 40–60 %.
- **Plume:** σt = 0.1–0.4 m⁻¹ at 0.3 m above the tips. Smoke is a subtle warm haze, not fog.

### 2.7 Cooling (both terms integrated exactly per cell)

```
radiative:  dT/dt = −χ · 970 · S[ppm] · θ²(θ⁴ − θ∞⁴)   with θ = T/1000, χ = 0.4         (≈ 4700 K/s·χ at 1300 K, 1 ppm)
mixing:     T = T∞ + (T − T∞)·exp(−k_T·(1 − ign·[q > 0])·dt)   k_T = 1.5 s⁻¹ (rt, h 1.5 cm) / 0.8 s⁻¹ (video, h 7.5 mm)
```

Tune k_T with T3 so the time-averaged plume matches §2.1 (ΔT0 = 228 K at 1 m). The prototype's 3.0 s⁻¹ is likely too high for the video grid.

### 2.8 Forces

**Buoyancy:**
```
a_y = β·g·(T − T∞)/T∞,   β = 0.8,   clamp a_y ≤ 49 m/s²
```
This is the McCaffrey C² fit. The Boussinesq g·ΔT/T form is about 3× too weak.

**Vorticity confinement:**
- `f = C_c·(N × ω)` with N = ∇|ω|/(|∇|ω|| + 1e-5).
- **C_c = ε·h** in world units: 0.15 m/s (rt), 0.10 m/s (video; the finer grid resolves more).
- Gate by `smoothstep(500, 900 K, T) · smoothstep(2h, 6h, sdf_to_source)`, so the base stays calm.
- Clamp |f| ≤ 0.4·a_y,local + 2 m/s².

**Curl-noise turbulence** (sub-grid Kolmogorov injection, divergence-free):
- Base scale ℓ0 = 5h (rt 7.5 cm, video 3.75 cm), 2 octaves (ℓ halves; amplitude ∝ ℓ^(1/3)).
- u_ℓ0 = 0.5 m/s · (ℓ0/0.08 m)^(1/3).
- Correlation time τ = 0.1 s · (ℓ/0.1 m)^(2/3): evolve the 4D noise by time, advected upward at 1.5 m/s.
- Gate by `max(q/1e5, smoothstep(600, 1200 K, T))`, and zero within 2h of solids.
- Applied as relaxation `u += (u_curl − u_curl,prev)` per step, i.e. added increments of a velocity field, not an acceleration. This keeps the amplitude independent of dt.

**Wind and gusts** (shared envelope function in `src/timeline.js`, identical to the audio engine's):
```
U_w(t, y)   = (U_base + U_gust(t)) · (max(y, 0.05)/0.5)^0.2 · dir(t)
U_base      = 0.25 m/s; dir wanders ±20° with 0.05 Hz noise
U_gust(t)   = g·1.3 m/s · env(t);  env = raised-cosine attack over 0.35·d, then exp release τ = 0.4·d
a_wind      = 3 s⁻¹ · ((U_w + 0.25·|U_w|·curl_h(x/0.3 m, t)) − u_h) · (1 − smoothstep(400, 800 K, T))
```
- Push only ambient air: the hot core leans through pressure and entrainment, as in reality.
- Lean targets (AGA correlation, `cos θ = u*^(−1/2)` for u* > 1, u* = U/(g·ṁ''·D/ρa)^(1/3) ≈ U/0.42 m/s) **[est]**:
  - 0.25 m/s → 0–10° wobble
  - 0.5 m/s → about 25°
  - 1.0 m/s → about 50°
  - 1.5 m/s → about 57°
- Gusts also raise k_mix by ×(1 + 0.5·env), because there is more air. Coal-bed T rises by 33 K per m/s of local inflow (ember breathing, physics §9.1).

### 2.9 Expansion and projection

```
φ = γ_exp · q / T,  γ_exp = 0.7,  clamp φ ≤ 60 s⁻¹ (rt) / 80 s⁻¹ (video)
radiative cooling contributes −(dT/dt)_rad / T (small contraction; optional)
solve ∇²p = (∇·u* − φ)/dt;  u = u* − dt∇p
```

### 2.10 HRR measurement and controller (GPU-side, deterministic)

- **Q̇** = Σ_cells b·ρ·V·ΔHc/dt, where ρ = p0/(R·T_before) and ΔHc·b is per unit fuel mass fraction, so Q̇ = Σ ρ·V·b·14 MJ/kg / dt. It is reduced every frame: 256-thread workgroup sums, then one single-workgroup pass in fixed order.
- **Controller** (one-thread dispatch, state in a storage buffer):
  - Track the target `Q_t(t) = Q0·I(t)·(1 + flare(t))` with a PI loop on the log error of the 1 s running mean Q̄: `e = ln(Q_t/Q̄)`, `I_e += e·dt_f`, `G = clamp(G_0·exp(0.3·e + 0.1·I_e), 0.2, 5)`. Effective time constant about 3 s.
  - It never sees per-puff fluctuations, so puffing survives.
  - Gust flare-ups are deliberately not counteracted: include `+0.35·env` in Q_t.
- **Outputs per frame** (for audio): `hrr = Q̇/Q0` and `dhrr = d(hrr)/dt` (central difference). Written to `out/video/frames.json` in capture mode, or read back asynchronously in real time.

### 2.11 Sim-side events

| Event | Sim action |
|---|---|
| `pop` (size s, at fire-local x, z) | Spark burst N = 120·s^1.5 (§3.8). Local fuel puff: f += 0.3·s in a 3 cm sphere at the log surface. Light flare via HRR +10 % for 0.3 s. |
| `crackle-burst` | 3–10 small sparks (s 0.2) |
| `log-shift` | §2.4 animation + flare ×1.6 (τ 1.5 s) + burst N = 150 |
| `gust` (strength g, duration d) | §2.8 |
| `ember` | Collapse of one coal chunk: bed T +80 K locally (τ 1 s) + 5 sparks |
| `ignite` (real-time only) | Reset fields; burn ramps 0 → 1 over 4 s with spreading patches |

### 2.12 Parameter table (start values)

| Parameter | RT (h 1.5 cm) | Video (h 7.5 mm / 4.17 mm) | Range | Anchor |
|---|---|---|---|---|
| g, T∞ | 9.81, 293 K | same | fixed | flicker ∝ √g |
| β buoyancy | 0.8 | 0.8 | 0.6–1.0 | McCaffrey C² |
| T_ign | 600 K | 600 K | 550–700 | piloted ignition |
| k_mix | 30 s⁻¹ | 45 s⁻¹ | 15–60 | grid mixing time |
| r_s | 4.7 | 4.7 | 4–5.5 | stoichiometry |
| ΔT_f·η | 7700 K | 7700 K | 6000–9000 | ΔHc/cp·(1−χ_rad) |
| γ_exp | 0.7 | 0.7 | 0.3–1.0 | ∇·u = q/T |
| f_src / v_n | 0.4 / 0.15 m/s | 0.4 / 0.10 m/s | | |
| soot gates | T 1300–1800 K, A 20–60 ms | same | | blue base 1–3 cm |
| k_ox | 20 s⁻¹ | 20 s⁻¹ | 10–30 | clean tips |
| χ radiative | 0.4 | 0.4 | 0.3–0.6 | 4700 K/s at 1 ppm |
| k_T | 1.5 s⁻¹ | 0.8 s⁻¹ | 0.5–3 | Heskestad plume |
| C_c = εh | 0.15 m/s | 0.10 m/s | 0.05–0.3 | |
| curl noise | 0.5 m/s @ 7.5 cm | 0.4 m/s @ 3.75 cm | | Kolmogorov |
| U_base wind | 0.25 m/s | 0.25 m/s | 0–0.5 | lean 5–20° |
| Q0 | 60 kW | 60 kW | 40–90 | L = 0.6 m |
| sparks baseline | 2 /s | 2 /s | 0–5 | physics §8.3 |

### 2.13 Physics calibration tests (`tools/shot.mjs` + `tools/fire_metrics.py`; run after every sim change)

| ID | Test | Pass |
|---|---|---|
| T1 | HRR over 20 s | mean within ±10 % of 60 kW; σ/μ 5–15 % |
| T2 | FFT of total visible emission over 20 s | dominant peak at **1.94 Hz ± 20 %**; 80 % of power below 4 Hz |
| T3 | Time-averaged centreline | u ≈ 6.8√z up to 0.41 m, plateau about 4.3 m/s; T 1100–1200 K in the continuous zone; ΔT(1 m) ≈ 230 K ± 30 %; V/√(2gzΔT/T∞) ≈ 0.9 |
| T4 | Flame intermittency (emission threshold on the axis) | I = 0.5 at 0.6 m ± 0.1; I < 0.05 above 1.1 m |
| T5 | Pattern speed (tracked tongue tips) | 1–2 m/s, well below the gas speed |
| T6 | Necking | inflow 0.3–1 m/s at the base; narrowest width at 0.3–0.6 m |
| T7 | Blue base / tip | blue fringe 1–3 cm tall; detached tongues burn out in 0.1–0.3 s; no black smoke column |
| T8 | Determinism | `FIRE.hash()` after 600 steps is identical across two runs |

---

## 3. Rendering model

### 3.1 Units and exposure

- **Relative radiance units** (linear, unbalanced D65 sRGB): a **thick 1500 K blackbody has luminance R = 6.0** (= 0.18·2⁵, so the flame body sits at +5 EV).
- Firelight uses the same units, which gives at 1 m: ground radiance ≈ 0.15/π · (6 · 0.3 m²)/1 m² ≈ 0.09. Target 0.05–0.15, so the flame is about 70× (6 EV) brighter than the lit ground, as in physics.
- Night ambient light: 0.002–0.01.
- **Exposure is fixed per shot** (no auto-exposure): default 0 EV.
  - Video offsets: S1 −0.5 EV, S3 +0.7 EV, others 0 (§5.2).
  - Tuning range −1.5 to +0.5 EV, chosen by the R-clip fraction (§6, C2).
- Keep all HDR values below 1e4, since fp16 overflow turns into NaN in bloom.

### 3.2 Blackbody LUT (`src/blackbody.js`, from `rendering_code/blackbodyLut.js`)

```
B_λ(T) = 2hc² / (λ⁵ (exp(hc/(λkT)) − 1)),   2hc² = 1.191042972e−16 W·m²/sr, hc/k = 1.438776877e−2 m·K
XYZ    = 683 ∫_{380}^{780} B_λ · {x̄,ȳ,z̄} dλ     — the real CIE 1931 2° table (10 nm, 41 rows); NOT the Wyman analytic fit
rgb    = M_XYZ→sRGB(D65) · XYZ;  clamp b ≥ 0 and renormalise to Y = 1  (below ~1900 K blue is out of gamut)
M = [ 3.2406 −1.5372 −0.4986 ; −0.9689 1.8758 0.0415 ; 0.0557 −0.2040 1.0570 ]
texel  = (rgb_Y=1, log2(Y(T)/Y(1500 K))) over 500–4000 K
```

Spot checks:
- `at(1500)` = (3.249, 0.432, 0)
- `at(1000)` = (4.570, 0.040, 0)
- 2856 K → xy (0.4475, 0.4074)

Do **not** bake white balance into the LUT; WB is a post matrix (§3.10).

```wgsl
fn blackbody(Tk: f32, gamma: f32) -> vec3f {   // relative radiance of a thick blackbody, R applied by caller
  let u = clamp((Tk - 500.0) / 3500.0, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0;
  let e = textureSampleLevel(bbLut, linClamp, vec2f(u, 0.5), 0.0);
  return e.rgb * exp2(gamma * e.a) * smoothstep(750.0, 850.0, Tk);   // Draper onset
}
```

**γ (brightness contrast):** 1.0 is physical (9.7 EV from 1200 K to 1800 K). Use **0.6 in real time and 0.75 offline**. Flames, sparks, embers and firelight all use the same γ and R, so they stay consistent.

### 3.3 Volume optical properties

```
c_soot      = (0.88, 1.00, 1.22)                       λ^−1.39 spectral shape over R,G,B
σa_soot     = 8.9 m⁻¹ · S[ppm] · c_soot · k_opt          k_opt = 1 (optical scale knob, 0.5–2)
smoke (M):  σt_M = 0.87 m⁻¹ · M   (M in 100 mg/m³; mass extinction 8.7 m²/g), albedo ω = (0.85, 0.80, 0.72)
            σa_M = σt_M (1 − ω), σs_M = σt_M ω
σt          = σa_soot + σt_M
emission    j = R · σa_soot ⊙ blackbody(T', γ) · ε_soot      (Kirchhoff; ε_soot = 1)
blue        j_b = k_b · (q/1e5 K/s) · (1 − smoothstep(0.05, 0.3, S)) · (0.12, 0.28, 1.0) / 0.29
              k_b chosen so peak blue radiance ≈ 1/16 (−4 EV) of the flame-body radiance; reads violet/lavender in camera
scattering  S_s = σs_M · ( F(x)·p_HG(cosθ_fire, g = 0.6) + L_amb(sky/moon)·p_HG(cosθ_moon, 0.6) )
              cosθ_fire = dot(normalize(x − c_lum), −rd) with c_lum = the luminance-weighted fire centroid
offline: Oz multiple scattering on smoke: Σ_{i=0..3} 0.5^i · L(0.5^i σt, 0.5^i σs, 0.5^i g)  (approximated as 1 + 0.5 + … gain on F with softened g)
```

The flame is almost purely emissive with modest absorption. Soot emission is automatically capped at B(T), so thick folds read solid and thin veils translucent, and red tips come out with no extra work.

### 3.4 Render-time detail (sub-voxel; the main fix for streaky "noise flames")

1. **Advected two-phase detail coordinates** (Neyret):
   - `x_k = x − u(x)·φ_k`, with φ_k = T_p·fract(t/T_p + k/2), k ∈ {0, 1}, T_p = 0.6 s.
   - Blend weights `w0 = |1 − 2·fract(t/T_p)|`, `w1 = 1 − w0`.
   - Noise therefore moves with the actual flow and never scrolls rigidly.
   - Use 3D gradient fBm with gain 0.6, 3 octaves, y scaled by 0.5 (features about 2× taller than wide), base frequency 1/(6 voxels).
2. **Lookup warp:** sample R at `x + A·curl(detail)·w_edge`.
   - A = 0.75 voxel (rt) / 0.6 voxel (offline).
   - `w_edge = smoothstep(0.2, 1.0, |∇T|·h/100 K)`: detail only on edges; cores stay smooth.
3. **Temperature mottle:** T' = T + 60 K·(fbm − 0.5)·w_edge. Through the steep B(T), this gives bright fold lines and dim veils.
4. **No** additive noise on density in the flame core, **no** static noise textures stretched over time, and **no** vertical-streak bias beyond the 2× aspect. The visual reference (§3.4) targets a PSD slope of −2.5 to −4 and ACF aspect 1.2–3.

### 3.5 Volume integration (`march`, from `rendering_code/march.wgsl`)

- Ray setup from inverse view-projection; slab test against the tight non-empty brick box; clip `tFar` at surface depth.
- Start jittered by IGN with a frame scroll of 5.588238 px (offline: R2 subframe sequence).
- Empty-space skipping: if brick σt and emission are 0, jump to the brick exit.
- Step Δ = stepVox·h. Adaptive: Δ = clamp(0.5/σt,max, 0.5, 2)·base, doubled once Tr < 0.3.
- **Per step** (energy-conserving, Hillaire 2015):

```
Src = j + j_b + S_s
w   = (σt·Δ < 1e-3) ? Δ·(1 − σt·Δ/2) : (1 − exp(−σt·Δ))/σt          (per channel)
L  += Tr ⊙ Src ⊙ w
Tr *= exp(−σt·Δ)
haze += (n∞ − n(T))·Δ  where n − 1 = 2.93e−4·273/T
zEm  += lum(Tr ⊙ j ⊙ w)·t   (emission-weighted depth; normalise at the end)
stop when max(Tr) < 0.003
```

Output `L` (premultiplied), `α = 1 − mean(Tr)`, `zEm`, `haze`.

### 3.6 Firelight (`src/firelight.js`, from `rendering_code/firelight.wgsl` and `fluence.wgsl`)

1. **VPLs:** 8³ coarse cells over the sim box; each workgroup sums `I_k = Σ j·dV` (rgb, relative W/sr) and the luminance-weighted centroid c_k. Radius r_k = 0.6 × cell size.
2. **Smoke light volume F:** `F(x) = Σ_k I_k·Tr_k(x)/(|x − c_k|² + r_k²)`. Tr_k is a 4-tap (rt) or 8-tap (offline) march through ¼-res σt. Grid is 32³ (rt) or 64³ (offline), over the sim box plus a 0.3 m skirt. About 1–3 ms.
3. **Surface lights:** merge VPLs into 8 clusters (2×2×2 octants of the bright region), intensity-weighted centroid and radius.
   - **Add the ember bed** as 4 low lights on a 0.15 m ring at y = 0.04, with intensity from the bed emission integral, so log undersides glow red. Expect 20–40 % of the flame light.
   - No synthetic flicker: the flicker is the sim's own (target σ/μ 6–10 %, §6). At most, low-pass the cluster intensities with τ = 20 ms.

### 3.7 Surfaces and environment (`src/shaders/scene.js`)

Lighting for every surface:

```
E   = Σ_{k∈clusters ∪ bed} I_k · V_k · wrap(n·l_k, 0.2) / (d_k² + r_k²)   +  E_sky(n) + E_moon·V_moon·max(n·l_m, 0)
L   = (albedo/π)·E + spec(GGX, F0 = 0.04, roughness) + R·ε·blackbody(T_surf, γ)·(1 − ash)
V_k = SDF cone soft shadow toward c_k, k_soft = d_k/(r_k·1.5), 24 (rt) / 64 (offline) steps
```

| Material | Albedo (linear) | Roughness | Notes |
|---|---|---|---|
| Char block tops | 0.03 | 0.55 (slight sheen) | T_top = T_crack − 150…400 K, so mostly below the Draper point |
| Crack floors | 0.02 | – | T_crack (below), width 0.6–2.5 mm growing with burn |
| Ash | 0.45 (cream under firelight) | 0.9 | 20–40 % coverage on upward faces × burn; flakes randomly shed |
| Fresh wood end grain | (0.42, 0.30, 0.17) | 0.7 | only on unburnt ends |
| Bark | (0.06, 0.045, 0.035) | 0.8 | only where burn < 0.3 |
| Coal crust | 0.04 + ash flecks | 0.7 | |
| Stones | 0.25–0.35 granite speckle | 0.6 | soot darkening up to 75 % on faces toward the fire |
| Ground | 0.12–0.18 forest floor (dirt, needles, twigs) | 0.9 | ash-blackened disc of radius 0.55 m inside the ring |
| Trees | 0.08 bark | 0.9 | 7–12 trunks, r 0.1–0.25 m, at 4–7 m; rim-lit by the moon only |

**Char cracks.** Anisotropic Voronoi in log-local coordinates (u = axial m, v = arc m):
- `F = voronoi(vec2(u/1.6, v)/0.015)`, which gives blocks about 2.4 cm along the grain × 1.5 cm across.
- `crack = 1 − smoothstep(0, w, F2 − F1)`.
- Block doming via a normal perturbation from F1.
- Block size scales with burn ×(0.7 → 1.3).

Crack temperature:
```
T_crack = 900 + 300·heat + 33 K·(|u_air| − 0.5)/(1 m/s) + 40 K·noise(cell, t/2.5 s)      (900–1250 K)
heat    = sim near-surface T and q sampled at the surface (glows where flames lick)
```

**Late stage** (burn > 0.85): the pattern inverts. Block faces glow salmon at 1050–1150 K with darker cracks.

**Ember bed.** An SDF pebble field of Voronoi chunks 1.5–5 cm across, 2–5 cm high, in a disc of radius 0.24 m:
- Crevices at 1050–1200 K: cavity factor from the SDF ambient-occlusion proxy adds +80 K.
- Tops at 850–950 K with ash flecks.
- Breathing: +33 K per m/s of local inflow (sim-sampled) + 25 K·noise(0.1–0.5 Hz).
- Embers come out redder than flames automatically, because they are cooler.

**Sky and moon.**
- Sky: radiance gradient zenith (0.0006, 0.0009, 0.0016) → horizon (0.0012, 0.0014, 0.0020).
- Stars: hashed, at most 0.02, visible only in wide shots.
- Moon: off-screen directional light (0.004, 0.006, 0.010) at 35° elevation, behind-left, for rim light on smoke, trees and stones.
- Moonlit ground should be about 2–5 % of the fire-lit ground at 1 m.

### 3.8 Sparks (`src/sparks.js`)

**Spawning.**
- Baseline: 2 per second from the coal bed and crack points.
- Bursts come from events: N = 120·s^1.5 within 0.15 s at (pan·0.3 m, 0.1–0.25 m).
- Launch: 0.5–3 m/s (pops: 2–6 m/s × √s) in a 35° cone around up-and-outward.

**Per-particle state.**
- Diameter d: log-normal, median 0.5 mm, σ_ln 0.5, clamped to 0.15–2.5 mm.
- Density ρ_p: U(150, 300) kg/m³.
- Initial temperature T0: N(1180, 60) K (pops: N(1250, 60) K).
- Twinkle on 40 % of sparks: ×(1 + 0.3·sin(2π·U(5, 20) Hz·t + φ)).
- Fragmentation: 2 %/s chance to split into two of half volume.

**Motion** (semi-implicit, per substep):
```
dv/dt = (u_gas(x) − v)·(1 + 0.15 Re^0.687)/τ_p + g(1 − ρ_g/ρ_p) ŷ
τ_p = ρ_p d²/(18 μ(T_gas)),  μ = 1.716e−5 (T/273)^1.5 (384)/(T + 111),  Re = ρ_g|u − v|d/μ
```
Outside the sim box, use the analytic plume `u0 = 1.03·Qc^(1/3)(z − z0)^(−1/3)` with a Gaussian radial profile (σ = 0.12·(z − z0)), plus curl turbulence of amplitude 0.3·u0 at ℓ = 0.2 m. That lets the spark column continue above 1.8 m in wide shots.

**Burning.** T stays at T0 while burning; d² shrinks linearly to 0 over `t_b = 2.0 s·(d/1 mm)²·(ρ_p/200)`. Then the spark is extinguished and cools: T → 293 K with τ = 0.1 s. It dies below 850 K.

**Rendering.**
- Energy-conserving anti-aliased capsule from the position at subframe-shutter open to close; real time uses the full 1/60 s.
- Radiant intensity `I = R·ε·π(d/2)²·blackbody(T, γ)·k_spark`, with k_spark = 1, the same units as everything else.
- Drawn width = max(1.2 px, projected d, CoC), with energy spread over width × length, so tiny sparks become 1–2.5 px streaks.
- Depth of field: CoC = A·f·|1/s_f − 1/s|·px_per_mm. Near sparks become soft bokeh streaks.
- Depth-tested against surfaces; behind the flame's emission depth, multiplied by (1 − α_vol).
- Real-time path: additive instanced quads in a render pass. Offline path: the same, once per subframe.

### 3.9 Heat haze (in composite)

- Gladstone–Dale relation: n − 1 = 2.93e−4·273/T. `haze` holds ∫(n∞ − n)ds along the ray.
- Deflection α = ∇_screen(haze)·(1/px_world).
- Background offset in pixels = α·f_px·(1 − z_em/z_bg).
- Apply only where the surface depth is behind the plume (`z_bg > z_em + 5 cm`). Add a rising fine shimmer: noise at 5–20 Hz advected at 1.5 m/s, amplitude 30 % of the physical offset.
- **Gain 1.0** (offline), 1.3 (rt). Typical result: 1–5 px at 1080p, at most 6 px.
- Visible only against lit stones, trees and the ground. Overdone haze is a classic CG tell.

### 3.10 Post chain (`final`, from `rendering_code/bloom.wgsl` and `composite.wgsl`)

Order:
1. CA
2. Bloom mix
3. Vignette
4. WB
5. Exposure
6. Tone map
7. OETF
8. Grain
9. Dither

| Stage | Spec |
|---|---|
| Lateral chromatic aberration | R sampled at 0.5 + (uv − 0.5)(1 + k), B at (1 − k); k = 0.002 |
| Bloom | Jimenez 2014 chain, 7 levels (rt 6). 13-tap downsample with Karis 1/(1 + luma) on the first pass, 3×3 tent upsample. `hdr = mix(hdr, bloom/levels, 0.04)`. Energy-conserving, no threshold. |
| Vignette | cos⁴ law, blend 0.6 |
| White balance | Bradford 3200 K matrix (rows, linear sRGB): `[0.8002 −0.3686 −0.0759; 0.0173 1.1324 −0.0297; 0.0361 0.1326 2.5302]`. Apply as `max(hdr * mat3x3f(row0, row1, row2), vec3f(0))`. Tuning range 2700–4000 K, computed by `blackbody.js`. |
| Exposure | × 2^EV |
| Tone map | **AgX-Fire** (below). ACES-Hill as `tonemap = 1`. Khronos Neutral and plain AgX are banned for fire: they push highlights pink or salmon. |
| OETF | sRGB piecewise curve |
| Grain | display-space triangular noise, 1.5 px clumps, amplitude g·4y(1 − y), g = 0.015 (offline) / 0.02 (rt), re-seeded each frame (deterministic seed = frame) |
| Dither | TPDF ±1 LSB, in the YUV conversion for capture and in the canvas write for real time |

```wgsl
fn agxCurve(x: vec3f) -> vec3f {
  let x2 = x * x; let x4 = x2 * x2;
  return 15.5*x4*x2 - 40.14*x4*x + 31.96*x4 - 6.868*x2*x + 0.4298*x2 + 0.1191*x - 0.00232;
}
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

Expected path on 1500 K emission: −2 EV deep red → 0 EV orange (187/68/0) → +2 EV amber (241/154/1) → +4 EV cream (255/220/134) → +6 EV warm white. Highlights must never turn lemon (hue > 55° with saturation > 0.5).

### 3.11 Temporal: TAA, subframes, motion blur, depth of field

- **Real time:** TAA with 0.75 history (0.8 when the camera is still), YCoCg neighbourhood clamp, and reprojection with `x − u_sim·dt` at the emission depth. Sparks bypass TAA (drawn after).
- **Offline (180° shutter):** the shutter is open for half the frame interval in **sim time**, centred on the frame time. In slow motion the shutter scales with sim time, like a high-speed camera.
  - Render **12 subframes per output frame**, spread across the open-shutter interval. At 1× speed the hero sim takes 12 substeps per frame, and the open half covers 6 of them: render 2 jittered views after each of those 6 substeps (sim state is not interpolated inside a substep).
  - General rule: spread the 12 views evenly (round-robin) over the substeps that end inside the open half. Slow motion has fewer substeps per frame, so it gets more views per substep. The macro sim at 1× has 10 open substeps, so it gets 1–2 views each.
  - Per subframe: R2 pixel jitter `(0.7548776662·n, 0.5698402910·n) mod 1`, march-start jitter with a global subframe counter, and a thin-lens DoF sample (aperture radius f/(2N), uniform disc) through the focus plane.
  - Sparks are drawn per subframe as short segments (v·Δt_sub), so the accumulation builds physically correct streaks.
- Accumulate linear HDR in rgba32f. Run bloom and tone mapping once, after averaging.

### 3.12 Render parameter summary

| Parameter | RT | Offline |
|---|---|---|
| R (thick 1500 K BB luminance) | 6.0 | 6.0 |
| γ | 0.6 | 0.75 |
| WB | 3200 K | 3200 K |
| k_opt (soot optical scale) | 1.0 | 1.0 (tune with C4/C8) |
| blue k_b | −4 EV vs body | same |
| smoke albedo / g | (0.85, 0.80, 0.72) / 0.6 | same |
| detail warp A | 0.75 vox | 0.6 vox |
| haze gain | 1.3 | 1.0 |
| bloom levels / strength | 6 / 0.04 | 7 / 0.04 |
| CA / vignette / grain | 0.002 / 0.6 / 0.02 | 0.002 / 0.6 / 0.015 |
| tone mapper | AgX-Fire | AgX-Fire |

---

## 4. Audio model

### 4.1 Engine (existing: `src/audio/fire-dsp.js`; design reference `research/audio.md`)

| Layer | Mix (dB rel. −20 LKu) | Model |
|---|---|---|
| Roar | −7 | Pink noise → one-pole LP at the knee (150 + 220·I + 260·gust Hz) → P ∝ f⁻³ tail; HP 45 Hz; +4 dB at 120 Hz, +2 dB at 420 Hz; 3σ tanh soft saturation. Log-normal AM from puffing (1.5/√D Hz, Q 2.5), flutter (7 Hz), flap (14 Hz) and slow wander. |
| Breath | −8 | Pink → SVF BP 750 Hz, Q 0.55 (300 Hz–3 kHz), deeper and faster AM. This is Chadwick & James's missing "whoosh". |
| Lapping | −15 | Farnell: BPF 30 Hz Q 5 → tanh(1.6x) → HP 25 Hz ×2 |
| Hiss | −16 | HP 1.2 kHz, shelf −4 dB at 9 kHz; envelope 0.4 + 0.6·min(z⁴, 16)/3 (Farnell surges) |
| Crackles | −4 | Hawkes process: base 16/s·I^1.3, branching 0.45, τ 70 ms (about 25/s total). Pareto energy (b 1.6, 20 dB), amplitude ∝ E^0.4, decay 1–7 ms, 1–5 sub-impulses, SVF BP 1.8–11 kHz + 15–45 % raw broadband. |
| Ticks | −9 | 2 × 260/s multi-sample bursts (0.08–0.3 ms), HP 2.5 k / BP 5.5 k |
| Pops | +2 (peak cal.) | click + 3-mode ring 1.2–4.5 kHz/s^0.4 + crack noise + 90–200 Hz body + fracture cascade + optional steam "psssh" + sizzle + Hawkes aftershock |
| Log shift, tinkles, ember | −1 / −6 | thump 55–90 Hz, knock 350–700 Hz, charcoal modal tinkles 2.2–6.5 kHz |
| Whoosh (gust) | −9 | pink → BP (280 + 750·g) Hz, pan sweep |
| Crickets | −22 | 3 Gryllus at 4.1–4.9 kHz, 3–4 syllable chirps every 0.48–0.66 s, far away (LP 6 kHz) |
| Space | – | ground reflection 0.85/1.05 ms; procedural outdoor IR (1.6 s; T60 0.55/0.85/0.6 s) |

### 4.2 Coupling to the sim

- **Continuous controls per frame:**
  - `intensity = Q_t/Q0`
  - `hrr = Q̇/Q0` (§2.10)
  - `dhrr`
  - `camDist` (metres, from the camera)
  - `camAz`
  - `wind`
- `simCoupling` 0.6: roar AM follows the HRR fluctuation; discrete "whumps" fire on fast dHRR rises, following Chadwick's p ∝ dQ/dt.
- The puffing frequency is shared through `fireDiameter`, derived from T2.
- **Events:** one list in `video/timeline.json` (`pop`, `crackle-burst`, `log-shift`, `gust`, `ember`, `flare`, `whoosh`) drives both the sim (§2.11) and the audio.
  - **For the video, set the engine's `popRate` to 0** and list every pop explicitly in the timeline, including the small ones.
  - This removes a circular dependency: random pops depend on intensity and gust state, while the audio render needs the HRR trace from the video capture.
  - The order is then strictly: timeline → video capture (writes `frames.json`) → audio render → mux.
  - `emberRate` may stay random; those sounds need no visuals.
- **Sync rule:** audio may lag the picture by up to about 40 ms but must never lead it (ITU-R BT.1359). Sparks appear on the frame at or just before the pop: `frame = floor(t·fps)`.
- **Real time:**
  - Create the AudioContext on the first click (autoplay policy).
  - Worklet random pops → `port.postMessage` → main thread → spark burst at `getOutputTimestamp()`.
  - User actions → `FIRE.event` → worklet `schedule(t = currentTime + 0.03)`.

### 4.3 Slow-motion treatment (shot 1, 0–1.6 s video time)

Add a `slowmo` curve (0..1) to the timeline. The engine maps it to:
- crackle and tick rates ×(1 − 0.75·s)
- crackle band-pass centres ×(1 − 0.5·s) (pitch-down feel)
- roar knee ×(1 − 0.4·s)
- reverb send ×(1 + 0.5·s)
- plus a sub-swell (40–80 Hz noise, 1.1 s rise) into the hero pop at 1.05 s

Use a tiny parameter patch in `fire-dsp.js` (the control-grid smoothing already exists). **Fallback:** no slow motion; shot 1 runs at 1×.

### 4.4 Offline render and mastering

```
node tools/render_audio.mjs video/audio_timeline.json out/video/fire.wav --events out/video/audio_events.json --stats
```

`audio_timeline.json` = the timeline events plus `frames[]` from `out/video/frames.json` (per-frame intensity, hrr, dhrr, camDist, camAz).

Master chain (`master-offline.js`):
1. HP 30 Hz (4th order)
2. +1.5 dB at 220 Hz ("phone body")
3. −1.5 dB high shelf at 10 kHz
4. Glue compressor 1.6:1, knee 8 dB
5. Gain normalised to **−14.0 LUFS**
6. Memoryless soft clipper (ceiling −4.5 dB knee)
7. Lookahead (1.5 ms) true-peak limiter, internal ceiling −1.6 dBTP

Then 24-bit TPDF WAV → AAC-LC 48 kHz 320 kbps.

Targets (verify on the final MP4 with `ffmpeg -af ebur128=peak=true`):
- integrated −14 ± 0.3 LUFS
- **TP ≤ −1.0 dBTP after AAC**
- LRA 3–6 LU
- mono fold-down loss ≤ 2 dB

Listening test on a phone speaker: the roar must stay audible through the 120/420 Hz body and harmonics.

---

## 5. Video plan

### 5.1 Format decisions

| | Decision | Why |
|---|---|---|
| Aspect | **1080×1920 (9:16) primary** | Fire is a vertical subject (flames plus a spark column). Native on Reels, TikTok and Shorts, and fills the X mobile feed. Keep key action inside the centre **1080×1350 (4:5) safe area** (y = 285–1635 px) so a 4:5 crop also works. Keep the bottom 300 px and right 150 px free of essential action (platform UI). |
| Secondary | 1920×1080 (16:9), **optional**, from the same sims with re-framed cameras, rendered in the same capture pass (two cameras per subframe, two ffmpeg pipes) | X desktop, YouTube. Doubles render cost but not sim cost. |
| Frame rate | **30 fps** | Native to phone footage and platforms; the fire looks like real phone or camera footage. 24 fps reads more "film" but gets juddery on platform re-encodes. |
| Shutter | **180°** in sim time (1/60 s at 1×), 12 subframes | Matches real footage: slightly silky tongues and 5–20 px spark streaks. |
| Duration | 20.0 s = 600 frames, hard start (no fade-in), sound from sample 0 | |
| Colour | BT.709, limited range, all four VUI tags | §5.6 |

### 5.2 Shot list

Camera conventions:
- World coordinates in metres; fire centre at the origin on the ground.
- Full-frame-equivalent focal length on a vertical 20.25 × 36 mm gate: vFOV = 2·atan(18/f).
- Camera moves are Catmull-Rom curves through the keys, eased with smootherstep.
- Focus pulls follow the target distance.
- Cuts land on audio transients.

| # | Video time | Shot | Sim / time-warp | Camera (pos → pos; target) | Lens | Content and beats |
|---|---|---|---|---|---|---|
| **S1 HOOK** | 0.00–2.20 (66 fr) | **Macro flame root + hero pop**, speed-ramp | **Macro sim**. Speed 0.30× for 0–1.10 s, smoothstep ramp to 1.0× by 1.60 s, then 1.0×. About 1.26 s of sim time after a 2.5 s warm-up. | (0.20, 0.07, 0.60) → (0.16, 0.08, 0.50), push-in; target (0.03, 0.10, 0.14) = flame root under the front teepee log | 40 mm (vFOV 48°), **f/4**, focus 0.48 m; frame height about 0.45 m at focus (about 108 voxels, 18 px/voxel) | **Frame 0 is already bright and moving.** Violet roots peel off alligator char; crack rims glow; sheets fold in slow motion. **1.05 s: HERO pop (size 1.25)** from the char at screen centre; sparks burst toward and past the lens (bokeh streaks) as the speed ramps up to real time; whump plus ring plus sizzle. |
| S2 | 2.20–6.20 (120 fr) | **Climb the tongues** | Hero sim, 1×. Sim time = 5.0 s warm-up + (t − 2.2). | (0.35, 0.18, 0.80) → (0.27, 0.85, 0.82), crane up with a 12° orbit; target (0, 0.25, 0) → (0, 0.78, 0) | 50 mm (vFOV 40°), f/2.8, focus tracks about 0.85 m | Base → yellow body with clipping folds → necking tongues and detached pockets → tips at 0.9–1 m, sparks rising into darkness. Pop at 3.40 s (size 0.6) sends a spark burst across the frame. Cut at 6.20 on a crackle burst. |
| S3 | 6.20–9.60 (102 fr) | **Ember bed breathing** | Hero sim, 1× | (−0.55, 0.07, 0.40) → (−0.50, 0.06, 0.32), slow creep; target (0, 0.05, 0) | 35 mm (vFOV 54°), f/4, focus 0.65 m; EV +0.7 | Low angle across the glowing coals: tiny blue CO flamelets, glowing cracks on the near log, firelight flickering on stones. 7.90 s: ember settle (tinkle + 5 sparks). **8.60 s: gentle gust (0.4, 1.2 s)** makes the coals visibly brighten (breathing) and flames lean. |
| S4 | 9.60–14.40 (144 fr) | **Gust and collapse** | Hero sim, 1× | (0.90, 0.55, 1.35) → (0.80, 0.52, 1.20), slow push; target (0, 0.45, 0) | 40 mm, f/5.6, focus 1.6 m | Full fire in frame (frame height about 1.4 m, 10 px/voxel). **9.80–12.20 s gust (0.9, 2.4 s)**: flames lean about 45–50°, stream and flap; whoosh. **12.60 s: log shift (0.9)**: teepee log slips, spark burst of about 150 plus flare, thump and charcoal tinkles. |
| S5 | 14.40–20.00 (168 fr) | **Pull-back reveal** | Hero sim, 1× | (0.60, 0.50, 1.20) → (1.60, 1.25, 3.00), continuous dolly back and rise; target (0, 0.55, 0) → (0, 0.80, 0) | 35 mm, f/8, focus tracks | Reveals the stone ring, the ground light pool falling off as r⁻², faint lit smoke, the spark column climbing above the frame (analytic plume above the box), moonlit tree trunks and faint stars. Pops at 15.10 s (0.8) and 18.30 s (1.0) with spark plumes; crackle burst at 17.40 s. Hold the last 0.5 s. |

Notes:
- **Resolution rule:** never frame flames tighter than about 0.4 m frame height on the hero sim (h = 7.5 mm), or 0.3 m on the macro sim. Shallow depth of field in S1 and S2 is a deliberate cover for sub-voxel limits.
- **Hook alternative B** (if the macro sim fails): "rekindle". Frame 0 shows only glowing alligator cracks and coals (bright, not black). At 0.3 s a breath gust (strength 1.0) makes the coals flare, and flames leap up the log within 0.8 s with a whoosh. Uses the hero sim and the `ignite`-style burn ramp, at 1×.
- **Poster/thumbnail:** S1 frame 40 (sparks mid-flight) or S4 frame at 12.75 s.
- Exposure per shot: S1 −0.5 EV, S2 0, S3 +0.7, S4 0, S5 0, with 3-frame ease at cuts.
- The audio `camDist` curve: 0.48 (S1), 0.85 (S2), 0.65 (S3), 1.6 (S4), 1.3 → 3.4 (S5), smoothed over 150 ms at cuts.

### 5.3 `video/timeline.json` (contract shared by video and audio)

```json
{
  "duration": 20.0, "fps": 30, "sampleRate": 48000, "seed": 20261025, "width": 1080, "height": 1920,
  "params": { "Q0_kW": 60, "fireDiameter": 0.6, "wbKelvin": 3200, "shutter": 0.5, "subframes": 12 },
  "sims": {
    "hero":  { "res": [160, 240, 160], "size": [1.2, 1.8, 1.2], "origin": [-0.6, 0, -0.6], "dt": 0.0027777778, "vcycles": 3, "warmup": 5.0 },
    "macro": { "res": [192, 224, 192], "size": [0.8, 0.9333, 0.8], "origin": [-0.4, 0, -0.4], "dt": 0.0016666667, "vcycles": 3, "warmup": 2.5 }
  },
  "shots": [
    { "id": "S1", "t0": 0.0, "t1": 2.2, "sim": "macro", "simT0": 2.5,
      "speed": [[0, 0.3], [1.1, 0.3], [1.6, 1.0], [2.2, 1.0]], "ev": -0.5,
      "cam": [ { "t": 0.0, "pos": [0.20, 0.07, 0.60], "target": [0.03, 0.10, 0.14], "f": 40, "N": 4, "focus": "target" },
               { "t": 2.2, "pos": [0.16, 0.08, 0.50], "target": [0.03, 0.10, 0.14], "f": 40, "N": 4, "focus": "target" } ] },
    { "id": "S2", "t0": 2.2, "t1": 6.2, "sim": "hero", "simT0": 5.0, "speed": 1.0, "ev": 0, "cam": [ "…" ] },
    { "id": "S3", "t0": 6.2, "t1": 9.6, "sim": "hero", "speed": 1.0, "ev": 0.7, "cam": [ "…" ] },
    { "id": "S4", "t0": 9.6, "t1": 14.4, "sim": "hero", "speed": 1.0, "ev": 0, "cam": [ "…" ] },
    { "id": "S5", "t0": 14.4, "t1": 20.0, "sim": "hero", "speed": 1.0, "ev": 0, "cam": [ "…" ] }
  ],
  "curves": {
    "intensity": [[0, 1.0], [6.2, 0.85], [9.6, 1.0], [14.4, 1.1], [20, 1.1]],
    "wind":      [[0, 0.25], [20, 0.25]],
    "slowmo":    [[0, 1], [1.1, 1], [1.6, 0], [20, 0]],
    "camDist":   "derived from shot cameras"
  },
  "events": [
    { "t": 1.05,  "type": "pop", "size": 1.25, "pan": 0.0, "x": 0.03, "y": 0.10, "z": 0.14, "id": "hero" },
    { "t": 2.18,  "type": "crackle-burst", "count": 6, "pan": -0.2 },
    { "t": 3.40,  "type": "pop", "size": 0.6, "pan": 0.25 },
    { "t": 4.65,  "type": "pop", "size": 0.3, "pan": -0.35 },
    { "t": 5.35,  "type": "pop", "size": 0.25, "pan": 0.4 },
    { "t": 6.18,  "type": "crackle-burst", "count": 8, "pan": 0.1 },
    { "t": 7.90,  "type": "ember", "pan": -0.3 },
    { "t": 8.60,  "type": "gust", "strength": 0.4, "dur": 1.2 },
    { "t": 9.80,  "type": "gust", "strength": 0.9, "dur": 2.4 },
    { "t": 11.10, "type": "pop", "size": 0.35, "pan": -0.1 },
    { "t": 12.60, "type": "log-shift", "size": 0.9, "pan": 0.15 },
    { "t": 13.70, "type": "pop", "size": 0.3, "pan": 0.35 },
    { "t": 15.10, "type": "pop", "size": 0.8, "pan": 0.3 },
    { "t": 16.20, "type": "pop", "size": 0.3, "pan": -0.4 },
    { "t": 17.40, "type": "crackle-burst", "count": 10, "pan": -0.2 },
    { "t": 18.30, "type": "pop", "size": 1.0, "pan": -0.25 },
    { "t": 19.25, "type": "pop", "size": 0.25, "pan": 0.2 }
  ]
}
```

Rules:
- Event times are in **video time**. Each shot maps an event to its sim time through `speed`, i.e. sim t = simT0 + ∫speed.
- An event earlier than a shot's start (the sim keeps running behind cuts) is still applied to the hero sim, so continuity holds: the S4 log stays shifted in S5.
- The hero sim runs continuously from its warm-up through S2–S5. S1 uses its own macro sim.
- Field names follow `src/audio/fire-dsp.js` (`strength` 0..1 and/or `size`).

### 5.4 Capture pipeline (`tools/capture.mjs` + `src/capture.js`)

```
node tools/capture.mjs --timeline video/timeline.json --out out/video --aspect 9x16 [--also 16x9] [--from S2] [--preview]
```

1. Start `tools/server.mjs` on 127.0.0.1 with the `/frames` binary WebSocket sink, which acks after `ffmpeg.stdin.write` drains.
2. puppeteer-core launches `C:/Program Files/Google/Chrome/Application/chrome.exe` with `headless: true` (new headless), `protocolTimeout: 0`, and flags `--enable-unsafe-webgpu --ignore-gpu-blocklist --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows`. Profile dir: `D:/fire/.chrome-capture-profile`.
3. The page runs `?capture=1&timeline=…`. For each shot:
   - **Sims:** restore the checkpoint `out/ckpt/<sim>_warm.bin` if present. Otherwise warm up and save one (the hero warm-up is 1800 steps, about 9 min loaded).
   - **Frames:** for each frame, advance the sim through the substeps for sim time `[t_f, t_f+1)`, rendering subframes during the open half. Then post, then GPU YUV, then WS send with await ack. Await `onSubmittedWorkDone` after every substep and subframe.
   - **Per-frame data:** append `{frame, shot, intensity, hrr, dhrr, camDist, camAz}` to `frames.json`.
   - **Checkpoints:** every 100 frames, write `out/ckpt/hero_f<N>.bin`. A crashed run resumes with `--from <frame>`, by loading the nearest checkpoint and re-simulating.
4. ffmpeg (Node argv array, no shell). Delivery and master use the **tagged raw-YUV input** commands:
   - **8-bit delivery:** `-f rawvideo -pix_fmt yuv420p -s 1080x1920 -r 30 -colorspace bt709 -color_range tv -color_primaries bt709 -color_trc bt709 -i - -c:v libx264 -preset slow -crf 14 -profile:v high -x264-params "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv:aq-mode=3" -movflags +faststart out/video/fire_video_1080x1920.mp4`
   - **10-bit master:** same input tags with `yuv420p10le`, then `-c:v libx265 -preset slow -crf 12 -pix_fmt yuv420p10le -tag:v hvc1 -x265-params "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited" out/video/master_10bit.mp4`
5. **Preview mode** (`--preview`):
   - hero sim 80×120×80 at dt 1/120
   - macro sim 96×112×96
   - 2 subframes, 540×960
   - about 5–8 min for all 20 s
   - Always run it before the hero render to check cameras, timing and events.

### 5.5 Budget (loaded machine per webgpu.md; expect 2–3× faster when idle)

| Item | Estimate |
|---|---|
| Hero warm-up 5 s (1800 steps × about 0.3 s) | about 9 min, once (checkpointed) |
| Hero S2–S5: 534 frames × 12 steps × 0.3 s | about 32 min |
| Macro: warm-up 1500 steps + 1.26 s × 600 = 756 steps, × about 0.4 s | about 15 min |
| Rendering: 600 frames × 12 subframes × about 60 ms + post/capture 0.1 s | about 8–9 min |
| **Total hero render** | **about 65 min loaded / 25–35 min idle**. Use the 192×288×192 stretch tier only if idle time allows (×1.75 sim). |

### 5.6 Mux, delivery and verification (`tools/mux.mjs`)

```
ffmpeg -i out/video/fire_video_1080x1920.mp4 -i out/video/fire.wav -map 0:v -map 1:a -c:v copy \
       -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart out/video/fire_1080x1920.mp4
```

- **Social upload copy:** re-encode with `-crf 16 -maxrate 20M -bufsize 40M` if the CRF 14 file is above about 40 Mbit/s. Grain and spark detail are expensive; keep grain at 0.015.
- **Verify:**
  1. `ffprobe` shows bt709/tv on all four tags.
  2. Colour patch round-trip error ≤ 1 code value (`webgpu-bench/color/cmp.mjs`).
  3. `ebur128`: −14 ± 0.3 LUFS, ≤ −1.0 dBTP.
  4. Sync: debug build with a 1-sample click at t = 1.05 s and a white frame at frame 31 (= floor(1.05·30)), checked in the waveform and video.
  5. Simulated platform re-encode (x264 at 6 Mbit/s): no banding in the smoke and near-black, sparks survive.
  6. Watch on a phone at full brightness and at 50 %.
- **Deliverables:**
  - `out/video/fire_1080x1920.mp4`
  - `out/video/master_10bit.mp4`
  - optional `out/video/fire_1920x1080.mp4`
  - `out/video/contact_sheet.jpg` (one frame per 0.5 s)
  - `out/video/poster.png`

---

## 6. Realism checklist (reviewer scoring)

Score each row 0 (wrong or absent), 1 (partly) or 2 (convincing). Rows marked ★ are blocking: they must score 2 before the hero render.

Tools:
- `tools/fire_metrics.py` with `analyze.py`/`video.py` (`refs_tools`, in the session scratchpad)
- side-by-side with `08_closeup_of_fire.jpg`, `25_knisternd.jpg`, `14_fireplace_rostov.jpg`, `01_campfire_flames_night.jpg`, `10_flames3.jpg`, `16_flames_52633350622.jpg`, `17_campfire_15188333373.jpg`, `22_swirling_sparks_hilsenhain.jpg` (contact sheet `refs_crops/best_refs_contact.jpg`)

### Colour and tone

| ID | Check | Pass criterion |
|---|---|---|
| C1 ★ | Luma → colour ramp | within ±1 bin of the natural ramp: `#421908` @0.13, `#B5611E` @0.43, `#FEC27A` @0.79, `#FFE8B2` @0.91 |
| C2 ★ | Clipping order and amounts | R clipped in 20–60 % of flame pixels; R+G 2–15 %; all three < 5 % of flame pixels |
| C3 ★ | No lemon or green highlights | no saturated (> 0.5) pixels with hue > 55° |
| C4 | Dim veils dominate | area(luma 0.10–0.55) ≥ 1.3× (target about 2×) area(> 0.55) |
| C5 | Base → tip profile | brightest at 0.1–0.5 H; tips about 50 % luma and 5–10° redder; coverage < 15 % at the top |
| C6 | Blue/violet root | visible on zoom at log roots; < 5 % of flame pixels; never a "gas log" look |
| C7 | Warm near-black | near-fire pixels with luma 0.015–0.12 have hue 5–25°; no banding (with dither) |
| C8 | Embers vs flames | embers 2–15° hue at luma 0.13–0.43 (redder than flames at equal luma) |

### Shape

| ID | Check | Pass criterion |
|---|---|---|
| H1 ★ | Thin folded sheets, see-through | logs visible through flames at 40–60 %; bright fold lines |
| H2 ★ | Bimodal edges | some 2–6 px edges (tongue sides and bottoms) plus soft tips; edge median 10–35 px @1920 |
| H3 | Tongues | 3–8 distinct tongues, forking in the upper half; asymmetric silhouette; lean 5–20° |
| H4 ★ | Detached pockets | present in about 60 % of frames, about 7/s, lifetime 67–190 ms |
| H5 | Holes and dark interior zones | yes; not a solid gradient blob |
| H6 | Flames anchored to fuel | emerging from gaps, undersides and cracks, wrapping logs |
| H7 | Multiple scales | big plume + medium tongues + tiny flamelets on char and coals |
| H8 | Texture statistics | PSD slope −2.5 to −4; ACF aspect 1.2–3; no visible streak pattern |
| H9 | Flame height | mean 0.6 m; intermittent to about 1 m (T4) |

### Motion (video)

| ID | Check | Pass criterion |
|---|---|---|
| M1 ★ | Puffing | dominant 1.6–2.3 Hz (T2); height σ 10–18 % |
| M2 ★ | Speed | tips grow 1.5–2.5 flame heights/s; tip region about 70 ms decorrelation; base 3–4× slower |
| M3 | No rigid scrolling or loops | detail is advected with the flow (§3.4); no repetition over 20 s |
| M4 | Motion blur | 180° shutter: silky tongues, spark streaks 5–20 px |
| M5 | Gust behaviour | lean about 45–50° at strength 0.9; coals brighten (+33 K/(m/s)); recovery within about 1 s after the gust |

### Light

| ID | Check | Pass criterion |
|---|---|---|
| L1 ★ | Light flicker | σ/μ 6–10 %, about 3 % per-frame steps, 80 % of power below 4 Hz, 1/f-like; no strobing |
| L2 | Moving shadows | log and stone shadows jitter (multi-centroid) with soft penumbrae |
| L3 | Ground pool | 50 % at 1.5 fire radii, 15–25 % at 2 r, then about r⁻²; lit surfaces saturated orange; log undersides red from the coals |
| L4 | Smoke lighting | orange underside from below; cool moon rim; fades within 1–2 flame heights |

### Particles and atmosphere

| ID | Check | Pass criterion |
|---|---|---|
| P1 ★ | Sparks | 1–2.5 px streaks, deep orange `#A2431D`–`#D96A33`, pale cores only when fresh; log-normal sizes; straight near the fire, curly higher; 40 % twinkle; cool to red and wink out |
| P2 | Bursty emission | baseline 0–5/s, bursts synced to pops within 0–1 frames (never before the sound) |
| P3 | Heat haze | 1–5 px, only above the flames against lit background detail |
| P4 | Smoke amount | subtle (luma 0.03–0.15 near the fire); no fog, no black column |

### Fuel bed

| ID | Check | Pass criterion |
|---|---|---|
| F1 ★ | Alligator char | black blocks elongated along the grain, glowing crack floors, sheen on tops |
| F2 | Ember bed | granular, brightest in crevices `#FC7719` → `#FFB04A`, slow breathing |
| F3 | Ash and fresh wood | white-grey ash (albedo 0.45); fresh end grain clearly paler than char |

### Audio

| ID | Check | Pass criterion |
|---|---|---|
| A1 ★ | Reads as fire, not rain or water | broadband co-modulated crackles (spectrogram shows vertical streaks); clustered timing |
| A2 | Roar | f⁻³ tail, knee 150–400 Hz, AM at the same puffing frequency as the picture; audible on a phone |
| A3 | Pops | click + ring + body + aftershock; sparks visible at the same frame or the next |
| A4 ★ | Loudness | −14 ± 0.3 LUFS, ≤ −1 dBTP after AAC, mono-safe |

### Video and encoding

| ID | Check | Pass criterion |
|---|---|---|
| V1 ★ | Hook | frame 0 bright and moving; spark burst plus pop within 1.1 s |
| V2 | Safe areas | key action inside the 4:5 centre; nothing essential under platform UI |
| V3 ★ | Colour tags and accuracy | BT.709 tags; ≤ 1 code value round-trip; no banding after a 6 Mbit/s re-encode |
| V4 | Cuts | on transients; no continuity breaks (hero sim continuous S2–S5) |

**Scoring:** ≥ 38/50 on the 25 visual-reference items (C1–C7, H1–H8, M1–M4, L1–L3, P1–P3, roughly as in `visual_reference.md` §11) = photoreal territory. Every ★ row must pass for sign-off.

---

## 7. Risks and fallbacks

| # | Risk | Detection | Fallback |
|---|---|---|---|
| 1 | Headless WebGPU fails or produces blank frames | `navigator.gpu` undefined, adapter null, or all-zero YUV frames (check frame 0 mean) | Headful Chrome (`headless: false`, window off-screen, same 3 anti-throttling flags), then Edge (`msedge.exe`), then a manual run in a visible browser tab with the same WS sink. Never use `page.screenshot`/`toDataURL`. |
| 2 | Device lost / TDR reset mid-render | `device.lost` (reason ≠ 'destroyed') | Recreate the device, restore the nearest checkpoint, re-simulate deterministically, continue; cap 3 retries. Keep dispatches under 100–200 ms and submits under 0.5 s; tile the full-res march into 4 if > 100 ms. |
| 3 | Real-time too slow on the 780M | Profiler: frame > 25 ms | Automatic ladder: 80→64 grid, march ½→0.4 scale, fluence shadows off, 6→4 bloom levels, surface shadows 4→2; then sim at 60 Hz with 1 step (CFL 4.5, MacCormack limiter) as a last resort. |
| 4 | Offline render too slow (> 2 h) | timing log | Hero 160→128×192×128; subframes 12→8; macro sim → hero sim with f/2.8 depth of field (hook alternative B); skip the 16:9 render. |
| 5 | NaN/Inf blow-up | debug NaN counter, black or white frames | Bit-test sanitise in advection writes; clamp \|u\| ≤ 4 cells/substep and φ ≤ 60; reduce γ_exp to 0.5; SL+RK2 for velocity. |
| 6 | Multigrid stalls on odd coarsest dims | residual after 1 cycle > 0.5 | 32 coarse sweeps; or pad the grid to a 2ⁿ-friendly size; or warm-started temporally-blocked Jacobi (40 iterations) plus more confinement. |
| 7 | Streaky, noise-looking flames (prototype issue) | H2/H8 fail, PSD slope > −2.3 | Lower detail amplitude A; advected coordinates only; raise the grid tier; lower γ is NOT the fix (it kills edges). |
| 8 | Flames too tall or too short | T4 | Rely on the HRR controller; then k_mix and k_ox; then β (last resort). |
| 9 | Too much smoke or fog (prototype issue) | P4, C4 | Cut M sources; k_ox up; smoke albedo down to 0.6; check the HG g = 0.6 is used. |
| 10 | Colour wrong (salmon, lemon, too red) | C1–C3 | Sweep WB 2700–4000 K and EV −1.5 to +0.5 with `analyze.py`; switch to ACES-Hill; check that WB is not applied twice. |
| 11 | No blue base, or a gas-log look | C6 | Adjust the age gate (20–60 ms) and k_b. |
| 12 | Flames opaque or see-through wrong | H1 | k_opt 0.5–2; soot target 1–2 ppm in sheets. |
| 13 | Log-shift artefacts (pressure spike, fuel inside solids) | visual pop, divergence spike | Shorten the move to 2 cm and smooth it; or replace it with pop + flare + burst (no geometry change). |
| 14 | Macro sim boundary artefacts (missing whole-fire entrainment) | S1 looks laminar or lifeless | Add an inflow profile (0.4 m/s toward the axis at the domain bottom sides); or alternative B. |
| 15 | Audio clicky, watery or pumping | A1 by ear, kurtosis 2–16 kHz > 30 | Raise tick rate and lower the tick mix; +1–2 dB hiss; more Hawkes clustering; never use single-pass loudnorm. |
| 16 | A/V desync | click/flash test | Events by frame index; audio rendered from the same timeline; delay audio rather than lead. |
| 17 | ffmpeg colour shift | patch test | Use the tagged raw-YUV input command; never a naive `-pix_fmt yuv420p`. |
| 18 | Platform re-encode destroys dark gradients and sparks | V3 simulation | aq-mode 3; grain ≤ 0.015; TPDF dither; slightly larger spark minimum width (1.4 px); upload the highest-quality file. |
| 19 | Machine contention (benchmarks ran at 13–24 GB/s) | per-step time × 2 | Render when idle; checkpoints make long runs resumable; preview tier for iteration. |
| 20 | fp16 overflow or precision | Inf in bloom, temperature banding | Relative units < 1e4; T stored as θ; pressure in r32f. |
| 21 | Spark nondeterminism | hash mismatch with sparks on | CPU ring-buffer allocation and hashed parameters; no atomics. |

---

## 8. Build order (milestones; each ends with screenshots, metrics and a checklist score in `out/`)

1. **M1 Physics calibration:** units, boundaries, HRR controller, T1–T8 passing at the RT default tier. Fix flame height and smoke amount.
2. **M2 Emission and camera:** unbalanced LUT, R = 6 relative units, WB post matrix at 3200 K, AgX-Fire, γ; C1–C4 at a fixed reference view against `08`/`14`/`01`.
3. **M3 Detail:** advected two-phase detail, edge-gated warp; pass H1, H2 and H8.
4. **M4 Fuel bed and surfaces:** char cracks, ember bed, ash, stones, ground, trees, cluster lights + bed lights + soft shadows; F1–F3, L2, L3.
5. **M5 Firelight on smoke + haze + sparks:** L4, P1–P4.
6. **M6 Audio coupling:** hrr/dhrr, events → sparks, slow-motion params; A1–A4 live and offline.
7. **M7 Capture pipeline:** timeline, shots, checkpoints, YUV/WS/ffmpeg, preview render of the full 20 s; V1–V4 and M1–M5 on the preview.
8. **M8 Hero render**, review against §6, fix, re-render the failing shots only (checkpoints), mux, verify, deliver.
9. **M9 Real-time polish:** UI keys, quality ladder, previz playback of the timeline.
