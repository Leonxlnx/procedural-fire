# WebGPU/WGSL implementation guide: real-time 3D grid fire, volume ray marcher, deterministic offline video capture

Target machine: Windows 11, Ryzen 7 260 + **Radeon 780M** (RDNA3, 12 CU), 62 GB RAM (DDR5-5600, 16 GB + 48 GB SODIMMs), **Chrome 155.0.8059.85**, Node 22.22, ffmpeg 8.0.1 (gyan full build).

This guide is built on two kinds of evidence. Facts marked **[measured]** come from a benchmark harness I wrote and ran on this machine. It lives in `D:\fire\research\webgpu-bench\` and you can rerun it. Facts marked **[spec]** or **[source]** come from the WebGPU/WGSL spec, Chrome release notes, papers and similar references (sources are listed at the end). Everything else is engineering judgement and is labelled as such.

> **Measurement caveat (read this first).** While these benchmarks ran, other jobs were heavily loading the machine. CPU load sat between 40 % and 100 %. A Node→Node loopback TCP test reached only 21–139 MB/s, when GB/s is normal. A single-threaded `memcpy` managed 7–10 GB/s. GPU streaming bandwidth came out at **only 13–24 GB/s**, against a theoretical 89.6 GB/s for dual-channel DDR5-5600. Treat the absolute numbers as a **pessimistic bound**. The relative comparisons (which solver, which format, which transfer channel) are what matter. To get clean numbers, rerun `node run.mjs new grid|mg|tb|xfer` on an idle machine.

---

## 0. TL;DR — decisions

| Topic | Decision | Why (evidence) |
|---|---|---|
| Field storage | **3D textures**. Velocity and scalars in `rgba16float`, pressure/divergence in `r32float`. Ping-pong textures plus write-only storage; sample with a linear sampler. | Hardware trilinear filtering makes semi-Lagrangian advection one fetch. rgba16f advection was 1.4–2.5× cheaper than rgba32f **[measured]**. Buffers are only about 10 % faster for the pure Jacobi stencil **[measured]**. |
| Extra formats | Request `texture-formats-tier2` (gives read-write `rgba16float`/`r16float`/`rgba32float`), `float32-filterable`, `shader-f16`, `timestamp-query`, `subgroups`. All are available on the 780M. | Probe **[measured]**. |
| Limits | Request `maxStorageTexturesPerShaderStage: 8`, `maxComputeWorkgroupStorageSize: 32768`, `maxComputeInvocationsPerWorkgroup: 1024`, plus large buffer limits. Otherwise you get spec defaults (4 / 16 KB / 256). | Adapter limits **[measured]**. |
| Advection | **MacCormack with min/max limiter** for velocity and for the packed scalars (temperature, smoke, fuel, age in one rgba16f). It is 2 dispatches and about 2.0–2.8× the cost of plain semi-Lagrangian. | **[measured]** |
| Pressure | **Geometric multigrid V(2,2)** with a red-black Gauss-Seidel smoother and *face-centred* Dirichlet open boundaries. Warm-start from last frame's pressure. 1–2 cycles in real time, 3–4 cycles offline. | One V-cycle cuts the residual about 10×, independent of resolution. 40 Jacobi iterations only reach 0.67 (128³) **[measured]**. At 64³ a V(2,2) cycle costs **1.4–2.0 ms**, versus 7.8 ms for 40 Jacobi iterations **[measured]**. |
| Workgroup size | `8×8×4` (256) by default, `4×4×4` also fine. | Shape sweep at 128³: all within ±10 % **[measured]**. |
| Real-time grid on this iGPU | **64×96×64** (or 64³) sim at 60 Hz. 80–96³ only with a 30 Hz sim. Ray march at half resolution (960×540). | Full 64³ step with multigrid is about 6 ms; 96³ is about 33 ms **[derived from measured kernels, contended]**. |
| Offline (video) grid | **160×240×160** (6.1 M cells), 4 substeps per 30 fps frame. Go to 192×288×192 if time allows. | About 1.1–1.5 s of sim per frame (contended estimate) → about 16–20 min for 600 frames. |
| Capture | **System Chrome, `headless: true` (new headless), puppeteer-core.** Render into your own texture, convert RGB→YUV 4:2:0 BT.709 on the GPU with dither, read back with `copyBufferToBuffer` + `mapAsync`, send over a **binary WebSocket** to Node, pipe into `ffmpeg` stdin. | Headless WebGPU works on the 780M with no special flags **[measured]**. WS + GPU-YUV was the fastest channel tested: 2.7× fewer bytes and 2.7× more fps than RGBA **[measured]**. |
| Colour | Use an explicit BT.709 matrix and tag all four VUI fields. Raw YUV input must be tagged *on the input side*. | A naive `-pix_fmt yuv420p` shifts fire orange by up to 39 code values; untagged raw YUV input is silently re-converted by ffmpeg 8 **[measured]**. |
| Determinism | Fixed `dt`, fixed substeps, hash-based RNG seeded by frame/substep, no wall clock, no float atomics. | 30 full steps at 64³ hashed to the same value across runs, processes, headless and headful **[measured]**. |

---

## 1. Platform facts on this machine [measured]

These come from `results/result-new-probe.json`, run in headless Chrome 155 with `--enable-webgpu-developer-features` so that `adapter.info` is unredacted.

```
adapter.info: vendor "amd", architecture "rdna-3", description "AMD Radeon 780M Graphics",
              backend "D3D12", driver "D3D12 driver version 32.0.32015.7029", type "integrated GPU",
              d3dShaderModel 68, subgroupMinSize 32, subgroupMaxSize 64,
              memoryHeaps [{ size 45.8 GB, properties 27 = device-local|host-visible|host-coherent|host-cached }]
preferredCanvasFormat: "bgra8unorm"
```

**Adapter features:** `bgra8unorm-storage`, `clip-distances`, `depth-clip-control`, `depth32float-stencil8`, `dual-source-blending`, `float32-blendable`, **`float32-filterable`**, `indirect-first-instance`, `primitive-index`, **`rg11b10ufloat-renderable`**, **`shader-f16`**, **`subgroups`**, `subgroup-size-control`, `texture-component-swizzle`, `texture-compression-bc`, `texture-compression-bc-sliced-3d`, **`texture-formats-tier1`**, **`texture-formats-tier2`**, **`timestamp-query`**, `chromium-experimental-timestamp-query-inside-passes` (only with the dev-features flag), `atomic-vec2u-min-max`.

**WGSL language features (`navigator.gpu.wgslLanguageFeatures`):** `readonly_and_readwrite_storage_textures`, `texture_formats_tier1`, `immediate_address_space` (push-constant-like `var<immediate>`), `uniform_buffer_standard_layout` (removes the 16-byte uniform array-stride rule), `pointer_composite_access`, `unrestricted_pointer_parameters`, `packed_4x8_integer_dot_product`, `subgroup_id`, `subgroup_uniformity`, `swizzle_assignment`, `texture_and_sampler_let`, `linear_indexing`, `buffer_view`.

**Limits (adapter max vs. spec default you get if you don't ask):**

| limit | 780M adapter | spec default |
|---|---|---|
| maxTextureDimension3D | 2048 | 2048 |
| maxStorageTexturesPerShaderStage | **8** | 4 |
| maxSampledTexturesPerShaderStage | 48 | 16 |
| maxStorageBuffersPerShaderStage | 16 | 8 |
| maxComputeWorkgroupStorageSize | **32768** | 16384 |
| maxComputeInvocationsPerWorkgroup | **1024** | 256 |
| maxComputeWorkgroupSizeX/Y/Z | 1024/1024/64 | 256/256/64 |
| maxComputeWorkgroupsPerDimension | 65535 | 65535 |
| maxBufferSize | 2 GiB | 256 MiB |
| maxStorageBufferBindingSize | 2 GiB − 4 | 128 MiB |
| maxUniformBufferBindingSize | 65536 | 65536 |
| min{Uniform,Storage}BufferOffsetAlignment | 256 | 256 |
| maxColorAttachmentBytesPerSample | 128 | 32 |
| maxImmediateSize | 64 | — |

**Capability probes (create pipeline in a validation error scope):** `r32float` read_write ✔, `rgba16float` read_write ✔ (tier2), `rgba16float` read-only ✔, `r16float` write ✔ (tier1), `rg11b10ufloat` storage write ✔ (tier1).

**Headless:** Plain `--headless` (new headless, the Chrome 132+ default) with **no flags at all** still gets the hardware AMD adapter with `timestamp-query`. Description and device are just blank without the dev-features flag. **[measured]**

**D3D11 backend (`--use-webgpu-adapter=d3d11`):** has no `timestamp-query`, no `shader-f16` and no tier2. Don't use it.

**Raw device numbers (contended):** FMA throughput 3.0–3.4 TFLOPS. `copyBufferToBuffer` 12.7–16.3 GB/s. Storage-buffer copy kernel 13–17 GB/s. 3D r32float texture copy 12.5–18 GB/s. 2D r32float texture copy 18–24 GB/s. Sustained 8 s copy test held flat at about 13 GB/s, so clock ramping is not the issue. **The fluid sim is bandwidth-bound.** Every byte per cell counts: f16 storage, packing, fusion, fewer solver sweeps.

---

## 2. 3D textures and storage in WebGPU

### 2.1 Storage format / access matrix [spec + measured]

| format | core storage access | + `texture-formats-tier1` | + `texture-formats-tier2` | sampleable with linear filter |
|---|---|---|---|---|
| `r32float` | write-only, read-only, **read-write** | | | only with `float32-filterable` |
| `r32uint` / `r32sint` | write/read/**read-write** | | | no (integer) |
| `rgba16float` | write-only, read-only | | **read-write** | **yes** |
| `rgba32float` | write-only, read-only | | **read-write** | only with `float32-filterable` |
| `rg32float` | write-only, read-only | | | only with `float32-filterable` |
| `r16float` / `rg16float` | — | write-only, read-only | `r16float` read-write | yes |
| `rgba8unorm` | write-only, read-only | | read-write | yes |
| `r8unorm` | — | write/read | read-write | yes |
| `rg11b10ufloat` | — | write/read (+ renderable via `rg11b10ufloat-renderable`) | | yes |
| `bgra8unorm` | needs `bgra8unorm-storage` (available) | | | yes |

- Core `read_write` exists only for the three 32-bit single-channel formats. The `readonly_and_readwrite_storage_textures` language feature shipped in Chrome 124. On Chrome 155 `texture_storage_3d<r32float, read_write>` compiled **without** a `requires` directive **[measured]**. Chrome still recommends adding `requires readonly_and_readwrite_storage_textures;` for portability.
- Tier2 (Chrome 142+) adds `read_write` for: `r8unorm, r8uint, r8sint, rgba8unorm, rgba8uint, rgba8sint, r16uint, r16sint, r16float, rgba16uint, rgba16sint, rgba16float, rgba32uint, rgba32sint, rgba32float`. Enabling tier2 also enables tier1 and `rg11b10ufloat-renderable`. Survey support: 98 % of AMD reports, 99.8 % of RDNA3.
- Feature-name gotcha: the Chrome 142 blog sample says `"texture-format-tier1"`, but the actual name is **`"texture-formats-tier1"`/`"texture-formats-tier2"`** (plural). Always check with `adapter.features.has()`.
- A storage texture **cannot be sampled** (`textureLoad` only, no sampler). For advection you need a separate sampled view, so ping-pong anyway.
- `textureBarrier()` exists for read-write storage textures (workgroup-scope sync of texture memory).

### 2.2 Sampling in compute shaders

- **`textureSampleLevel(t, s, uvw, 0.0)` works in compute.** `textureSample` (implicit derivatives) is fragment-only.
- `rgba16float` / `r16float` / `rg16float` are filterable in core. `*32float` needs **`float32-filterable`** (available here, about 99.9 % of AMD Windows reports). Without it, bind as `sampleType: 'unfilterable-float'` and do manual trilinear (8 × `textureLoad`).
- Texel centres: cell `i` maps to `uvw = (i + 0.5) / dims`. The back-trace is `uvw = (vec3f(id) + 0.5 - dt*v) / dims` with `v` in **cells per second**.
- **Address modes:** WebGPU has only `clamp-to-edge`, `repeat` and `mirror-repeat`. There is **no clamp-to-border**. Clamp-to-edge gives zero-gradient (outflow) behaviour for free. For ambient inflow at open sides, force a 1-voxel border shell to ambient values each step, or test `any(p < 0 | p > dims)` and return ambient.
- **Filter weight precision:** GPU texture units compute interpolation weights in fixed point with about **8 fractional bits** (1/256 steps). This is fine for advection of noisy fields. For very smooth low-gradient fields you may see stair-stepping; switch that field to manual trilinear with `textureLoad`. The (¾, ¼) weights used by multigrid prolongation are exact.

### 2.3 Storage buffers vs storage textures [measured]

| | storage texture (3D) | storage buffer (`array<f32>`/`array<vec4<f16>>`) |
|---|---|---|
| Trilinear fetch | 1 hardware sample | 8 loads + 7 lerps |
| Layout | driver-tiled 3D swizzle, neighbours local in all axes | linear (x fastest); z-neighbours are N² apart |
| Half-precision storage | `rgba16float`, f32 math | needs `shader-f16` (`array<vec4<f16>>`) |
| read_write | r32* core, more with tier2 | always |
| Atomics | `r32uint/sint` only | yes |
| Jacobi iteration 128³ | 2.08 ms | 1.88 ms (−10 %) |
| 3D copy kernel bandwidth | 12.5–18 GB/s | 13–17 GB/s |

**Recommendation:** use textures for everything that is advected or sampled (velocity, scalars, curl, emission/glow mips, occupancy). Pressure in `r32float` textures, because the multigrid mip chain is natural there and prolongation is one hardware sample. Use buffers for reductions (max |u| for the CFL check, NaN counters), indirect args and particles (sparks).

### 2.4 Bind-group layouts, usage scopes, ping-pong

- **Use explicit `GPUBindGroupLayout`s.** With `layout: 'auto'`, bindings the shader doesn't statically use are dropped, so bind groups built from a shared resource list fail. Auto layouts also guess `sampleType`.
- A texture subresource cannot be **sampled and storage-written in the same dispatch**; that is a validation error. **Different mips of the same texture are different subresources.** You can sample mip *l+1* while writing mip *l* (multigrid prolongation, bloom upsample). **[measured: works]**
- Texture usage flags: `TEXTURE_BINDING | STORAGE_BINDING` for ping-pong fields, `| COPY_SRC` for readback/debug, `| COPY_DST` for `writeTexture`/clears. Render targets need `RENDER_ATTACHMENT`.
- Inside a single compute pass, Dawn inserts the needed barriers between dispatches. You don't need to split passes for correctness. Split passes only for timestamp granularity.
- Per-dispatch GPU overhead is small: **1.3–3.9 µs** for a 1-workgroup dispatch on the GPU timeline. On the CPU side, encoding cost about 34 µs per dispatch in the wall-clock loop **[measured]**. A 115-dispatch multigrid V-cycle is fine.

### 2.5 Memory budget

| texture | format | B/cell |
|---|---|---|
| vel A/B | rgba16float (xyz + spare) | 16 |
| scalars A/B (T, smoke, fuel, age) | rgba16float | 16 |
| MacCormack temp (shared by vel and scalars) | rgba16float | 8 |
| curl (xyz, \|ω\|) | rgba16float | 8 |
| divergence (MG `b` mip 0) | r32float | 4 |
| pressure (MG `p` mips) | r32float | 4 × 8/7 |
| MG residual mips | r32float | 4 × 8/7 |
| occupancy (8³ bricks), glow mips | small | about 2 |
| **total** | | **about 67 B/cell** |

That works out to 64³ → 18 MB; 128³ → 140 MB; 160×240×160 → 410 MB; 256³ → 1.1 GB. Comfortable in the 45.8 GB UMA heap. Every dimension must stay ≤ 2048.

---

## 3. Simulation kernels (WGSL)

All the kernels below compile and run on Chrome 155 / 780M. They were benchmarked in `bench.js`, and the reference copies are in `webgpu-bench/wgsl/sim_reference.wgsl`, `mg.wgsl`, `volume.wgsl` and `post.wgsl` (all validated with `node run.mjs new compile`).

### 3.1 Uniforms and alignment

```wgsl
struct Params {             // offset  size
  dims: vec3<u32>,          //  0      12   (vec3 has align 16, size 12)
  dt: f32,                  // 12       4   (packs into the vec3's tail)
  invDims: vec3<f32>,       // 16      12
  decay: f32,               // 28       4
  parity: u32,              // 32       4
  omega: f32,               // 36
  eps: f32,                 // 40
  buoy: f32,                // 44  -> struct size 48 (multiple of 16)
};
@group(0) @binding(0) var<uniform> P: Params;
fn cl(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), vec3<i32>(P.dims) - vec3<i32>(1)); }
```

```js
const ab = new ArrayBuffer(48), u = new Uint32Array(ab), f = new Float32Array(ab);
u[0]=nx; u[1]=ny; u[2]=nz; f[3]=dt; f[4]=1/nx; f[5]=1/ny; f[6]=1/nz; f[7]=decay;
u[8]=parity; f[9]=omega; f[10]=eps; f[11]=buoy;
device.queue.writeBuffer(ubo, 0, ab);
```

Layout rules: `vec3` has align 16 and size 12, so put a scalar after it. `mat4x4` is 64 B. Struct size rounds up to its largest alignment. In uniform buffers, array strides and nested-struct offsets must be multiples of 16, unless the `uniform_buffer_standard_layout` language feature is present (it is on Chrome 155).

**Crucial `writeBuffer` semantics.** `queue.writeBuffer` is a queue operation that lands **before the next `submit`**. If you write the same uniform buffer twice while encoding one command buffer (say parity 0, then parity 1), every dispatch sees the **last** value. For per-dispatch constants use one of these:
1. separate small uniform buffers (what the benchmark does),
2. one buffer with 256-byte-aligned slots plus **dynamic offsets** (`hasDynamicOffset: true`, `setBindGroup(0, bg, [slot*256])`),
3. **immediates** (Chrome 149+, `immediate_address_space` is in the list): `requires immediate_address_space; var<immediate> k: vec4<u32>;` then `pass.setImmediates(0, new Uint32Array([parity, level, 0, 0]))`.

### 3.2 Semi-Lagrangian advection (1 dispatch) [measured: 64³ 0.51 ms, 128³ 4.5 ms]

```wgsl
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var srcTex: texture_3d<f32>;
@group(0) @binding(3) var samp: sampler;                       // linear, clamp-to-edge
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let v = textureLoad(velTex, c, 0).xyz;              // cells / second
  let p = vec3<f32>(id) + 0.5 - P.dt * v;             // Euler back-trace (cell units)
  let s = textureSampleLevel(srcTex, samp, p * P.invDims, 0.0);
  textureStore(dst, c, s * P.decay);
}
```

- For velocity self-advection, bind the velocity texture as both `velTex` and `srcTex`. Two sampled bindings of one texture are fine.
- **RK2 back-trace** (judgement): use it for velocity when CFL > 1. It costs one extra sample: `let vm = textureSampleLevel(velTex, samp, (x - 0.5*dt*v)*invDims, 0).xyz; p = x - dt*vm;`. Midpoint tracing keeps vortices round.
- **Pack scalars**: temperature, smoke, fuel and age in one `rgba16float`. One back-trace and one fetch advect all four.

### 3.3 MacCormack / BFECC with limiter [measured: 64³ 1.04 ms, 128³ 12.0 ms for 2 dispatches]

Equations (Selle, Fedkiw, Kim, Liu, Rossignac 2008), with A = semi-Lagrangian advection:

- φ̂ⁿ⁺¹ = A(φⁿ) (forward, `-dt`)
- φ̂ⁿ = Aᴿ(φ̂ⁿ⁺¹) (backward, `+dt`)
- φⁿ⁺¹ = φ̂ⁿ⁺¹ + ½(φⁿ − φ̂ⁿ)
- **Limiter:** clamp φⁿ⁺¹ to [min, max] of the 8 φⁿ texels used by the forward trilinear fetch. Selle et al. instead *revert* to φ̂ⁿ⁺¹ when out of range; both work, and revert is less "plateau-y".
- BFECC (Kim et al. 2005, "FlowFixer") does φ̄ = Aᴿ(A(φ)), φ' = φ + ½(φ − φ̄), φⁿ⁺¹ = A(φ'). That is 3 advections; MacCormack does the same job with 2.

Pass 1 is the SL kernel above (writes `hatTex`). Pass 2 fuses the backward step and the correction:

```wgsl
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var srcTex: texture_3d<f32>;   // φⁿ
@group(0) @binding(3) var hatTex: texture_3d<f32>;   // φ̂ⁿ⁺¹ from pass 1
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let v = textureLoad(velTex, c, 0).xyz;
  let x = vec3<f32>(id) + 0.5;
  let hat  = textureLoad(hatTex, c, 0);
  let back = textureSampleLevel(hatTex, samp, (x + P.dt * v) * P.invDims, 0.0);   // φ̂ⁿ
  let orig = textureLoad(srcTex, c, 0);
  var r = hat + 0.5 * (orig - back);
  let i0 = vec3<i32>(floor(x - P.dt * v - 0.5));          // 8 texels of the forward fetch
  var mn = vec4<f32>(3.0e38); var mx = vec4<f32>(-3.0e38);
  for (var k = 0; k < 8; k++) {
    let s = textureLoad(srcTex, cl(i0 + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1)), 0);
    mn = min(mn, s); mx = max(mx, s);
  }
  r = clamp(r, mn, mx);                    // or: select(r, hat, any(r < mn) || any(r > mx))
  textureStore(dst, c, r * P.decay);
}
```

**Cost-saving variant (judgement):** if the frame budget is tight, use MacCormack for the scalars (where the fire detail lives) and SL+RK2 for velocity, then compensate with vorticity confinement.

### 3.4 Vorticity confinement + buoyancy [measured: 2 dispatches, 64³ 0.87 ms, 128³ 10.2 ms]

Following Fedkiw, Stam, Jensen 2001: ω = ∇×u, η = ∇|ω|, N = η/|η|, f_conf = ε·h·(N × ω). Buoyancy (Boussinesq) is f_b = (α(T − T_amb) − β·ρ_smoke)·ŷ.

```wgsl
// pass A: curl -> rgba16float (ω, |ω|)
fn V(c: vec3<i32>) -> vec3<f32> { return textureLoad(velTex, cl(c), 0).xyz; }
@compute @workgroup_size(8, 8, 4)
fn curl(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let L = V(c - vec3<i32>(1,0,0)); let R = V(c + vec3<i32>(1,0,0));
  let D = V(c - vec3<i32>(0,1,0)); let U = V(c + vec3<i32>(0,1,0));
  let B = V(c - vec3<i32>(0,0,1)); let F = V(c + vec3<i32>(0,0,1));
  let w = 0.5 * vec3<f32>((U.z - D.z) - (F.y - B.y), (F.x - B.x) - (R.z - L.z), (R.y - L.y) - (U.x - D.x));
  textureStore(dst, c, vec4<f32>(w, length(w)));
}
// pass B: confinement + buoyancy
fn W(c: vec3<i32>) -> f32 { return textureLoad(curlTex, cl(c), 0).w; }
@compute @workgroup_size(8, 8, 4)
fn confine(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let eta = 0.5 * vec3<f32>(W(c + vec3<i32>(1,0,0)) - W(c - vec3<i32>(1,0,0)),
                            W(c + vec3<i32>(0,1,0)) - W(c - vec3<i32>(0,1,0)),
                            W(c + vec3<i32>(0,0,1)) - W(c - vec3<i32>(0,0,1)));
  let N = eta / (length(eta) + 1e-5);              // epsilon: normalize(0) would be NaN
  let w = textureLoad(curlTex, c, 0).xyz;
  let s = textureLoad(scalTex, c, 0);              // x = temperature, y = smoke
  var v = textureLoad(velTex, c, 0).xyz;
  v += P.dt * (P.eps * cross(N, w) + vec3<f32>(0.0, P.buoy * s.x - 0.1 * s.y, 0.0));
  textureStore(dst, c, vec4<f32>(v, 0.0));
}
```

ε (grid units, h = 1): start around 0.2–0.4 and tune. Making ε grow with temperature gives turbulent flame tips with a calmer smoke plume. If you change resolution and want the same look, keep ε·Δx constant in world units. This is two neighbour-heavy passes; it is the second-most-expensive item after the pressure solve.

### 3.5 Divergence and boundary conditions (open top/sides, solid floor)

```wgsl
fn V(c: vec3<i32>) -> vec3<f32> { return textureLoad(velTex, cl(c), 0).xyz; }   // clamp => zero-gradient at open faces
@compute @workgroup_size(8, 8, 4)
fn div(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let vC = V(c);
  let vL = V(c - vec3<i32>(1,0,0)).x; let vR = V(c + vec3<i32>(1,0,0)).x;
  var vD = V(c - vec3<i32>(0,1,0)).y; let vU = V(c + vec3<i32>(0,1,0)).y;
  let vB = V(c - vec3<i32>(0,0,1)).z; let vF = V(c + vec3<i32>(0,0,1)).z;
  if (c.y == 0) { vD = -vC.y; }      // solid floor: mirrored ghost => zero normal velocity on the face
  textureStore(dst, c, vec4<f32>(0.5 * ((vR - vL) + (vU - vD) + (vF - vB)), 0.0, 0.0, 0.0));
}
```

**Pressure ghost values. This matters for multigrid [measured]:**

| boundary | velocity | pressure ghost |
|---|---|---|
| solid floor (y < 0) | free-slip: v.y = 0 (mirror in div, clamp v.y ≥ 0 after projection if you allow upward flow) | **Neumann**: ghost = p_c |
| open sides/top | zero-gradient (clamp) | **Dirichlet at the face**: ghost = **−p_c** |

GPU Gems uses ghost = 0, which puts the boundary at the ghost cell's *centre*. That is fine for Jacobi/SOR. But on coarser multigrid levels the boundary then sits at a different physical place, and **the V-cycle stalls/diverges**. At 64³, V(2,2) with ghost = 0 reached 0.45 at cycle 5 and then grew; V(1,1) blew up to 750×. With ghost = −p_c, every cycle gives about 10× reduction. Use −p_c everywhere.

Solid obstacles inside the domain (logs) follow GPU Gems 3 ch. 30: use the obstacle's velocity for solid neighbours in `div`, use ghost = p_c for solid pressure neighbours, overwrite the normal velocity component after projection, and zero advected quantities inside solids.

### 3.6 Pressure solve — options, convergence, cost

**Convergence [measured, CPU `poisson.mjs` and GPU `mg` mode agree].** Plume-like RHS, face Dirichlet. Values are RMS residual relative to the initial residual (the residual *is* the divergence left after projection).

| 128³ | after 5 | 10 | 20 | 40 | 80 | 160 iterations |
|---|---|---|---|---|---|---|
| Jacobi ω=1 | 0.83 | 0.78 | 0.73 | **0.67** | 0.59 | 0.52 |
| Jacobi ω=6/7 | 0.84 | 0.79 | 0.74 | 0.68 | 0.61 | 0.53 |
| RB-GS ω=1 | 1.1 | 1.0 | 0.94 | 0.84 | 0.73 | 0.60 |
| RB-SOR ω=1.85 | 6.5 | 6.3 | 4.9 | 3.1 | 1.5 | 0.70 |
| RB-SOR ω_opt=2/(1+sin(π/N))=1.952 | 10 | 13 | 11 | 4.0 | 1.5 | 0.30 |

| multigrid, 6 levels (128→4) | 1 cycle | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **V(2,2), RBGS smoother** | **0.31** | **0.022** | **0.0022** | 0.00025 | 3e-5 (f32 floor about 2e-5) |
| V(1,1), RBGS | 3.3 | 1.8 | 0.62 | 0.22 | 0.08 |
| V(2,2), damped-Jacobi (ω=6/7, 4 + 4 sweeps) | 0.38 | 0.036 | 0.0027 | 0.0004 | |

Multigrid convergence is resolution-independent. GPU V(2,2), residual after 1 / 2 / 3 cycles: 64³ 0.23 / 0.018 / 0.0018, 96³ 0.28 / 0.020 / 0.0020, 128³ 0.31 / 0.022 / 0.0022, 160³ 0.34 / 0.024 / 0.0024. For comparison, 40 RB-GS sweeps reach only 0.69 / 0.79 / 0.84 / 0.86 at those sizes. In theory, every doubling of N needs about 4× more Jacobi iterations (spectral radius 1 − O(h²)) and about 2× more optimal-SOR iterations for the same reduction. SOR's residual first *grows* (it amplifies high-frequency error) and only beats Jacobi after more than 100 iterations. Warm-starting from last frame's pressure helps every method a lot in real time, because consecutive frames have similar pressure fields.

**GPU cost [measured, ms, contended machine]:**

| solver step | 64³ | 96³ | 128³ | 160³ |
|---|---|---|---|---|
| Jacobi iteration (texture ping-pong, r32f) | 0.20 | 0.75 | 2.08 | 4.67 |
| Jacobi iteration (storage buffers) | 0.16 | 0.59 | 1.88 | 4.37 |
| Jacobi, **2 iterations per dispatch through workgroup memory** (per iteration) | 0.19 | – | **1.02** | – |
| Jacobi iteration, r16float pressure | 0.054* | – | 1.35 | – |
| RB-SOR full iteration (red + black dispatch) | 0.23 | 1.27 | 4.32 | 9.12 |
| **MG V(2,2) cycle** (RBGS, 16 coarse sweeps) | **1.4–2.0** | 10.8–11.4 | 24.4–25.3 | 44.8–51.5 |
| MG V(1,1) cycle | 0.87–1.34 | 6.6–6.9 | 15.0–16.7 | 27.0–32.6 |

\*At 64³ the r16f working set (p ping-pong 2 × 0.5 MB + div) is about 2 MB, roughly the size of the 780M's L2, so it is largely cache-resident. That is also why 64³ multigrid is so cheap: about 18× cheaper than 128³ for 8× fewer cells.

**Cost to reach about 2 % residual at 128³:** 2 MG cycles take **49 ms**. Jacobi never gets there in a real-time budget (40 iterations = 83 ms and still at 0.67).

**Recommendation:**
- **Real time (64³ to 64×96×64):** 1–2 × V(2,2), warm-started, in **about 1.4–4 ms**.
- **Offline:** 3 × V(2,2) per substep (residual about 0.002).
- If you skip multigrid in v1: **warm-started, temporally-blocked Jacobi**, 20–40 iterations. Accept the compressible-looking motion this leaves and lean on vorticity confinement. Upgrade to MG next.

**Multigrid WGSL (`wgsl/mg.wgsl`, validated and benchmarked).** Each level is one mip of an `r32float` 3D texture. Bind single-mip views with `tex.createView({ baseMipLevel: l, mipLevelCount: 1 })`. The equation is Σ_nb p − 6p = h²b, with h² = 4^l.

```wgsl
struct Level { h2: f32, parity: u32, omega: f32, pad: f32 };
@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var pRW: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var bT: texture_3d<f32>;
fn pNb(c: vec3<i32>, d: vec3<i32>, pc: f32) -> f32 {
  if (c.y < 0) { return pc; }                                   // floor: Neumann
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return -pc; }     // open: p = 0 on the face
  return textureLoad(pRW, c).x;
}
@compute @workgroup_size(8, 8, 4)        // dispatch ceil(nx/2/8), ceil(ny/8), ceil(nz/4)
fn smoothRB(@builtin(global_invocation_id) g: vec3<u32>) {     // NOTE: 'smooth' is a reserved word
  let d = vec3<i32>(textureDimensions(pRW));
  let c = vec3<i32>(i32(g.x) * 2 + ((i32(g.y) + i32(g.z) + i32(L.parity)) & 1), i32(g.y), i32(g.z));
  if (any(c >= d)) { return; }
  let pc = textureLoad(pRW, c).x;
  let s = pNb(c - vec3<i32>(1,0,0), d, pc) + pNb(c + vec3<i32>(1,0,0), d, pc)
        + pNb(c - vec3<i32>(0,1,0), d, pc) + pNb(c + vec3<i32>(0,1,0), d, pc)
        + pNb(c - vec3<i32>(0,0,1), d, pc) + pNb(c + vec3<i32>(0,0,1), d, pc);
  textureStore(pRW, c, vec4<f32>(mix(pc, (s - L.h2 * textureLoad(bT, c, 0).x) / 6.0, L.omega), 0.0, 0.0, 0.0));
}
// residual: r = b - (Σnb - 6p)/h2 (same neighbour rule, pT sampled view) -> rOut (write-only)
// restrictLevel: bCoarse = mean of 8 fine residuals; pCoarse = 0
// prolongAdd: pFine += textureSampleLevel(pCoarse, linear, (c + 0.5)/dFine, 0)
//   (cell-centred (3/4, 1/4) trilinear == hardware trilinear at the fine cell centre; needs float32-filterable)
```

V-cycle driver. Everything goes in one compute pass; it is about 115 dispatches for 6 levels.

```js
function vcycle(pass, l, pre = 2, post = 2) {
  if (l === levels - 1) { smooth(pass, l, 16); return; }  // 3^3..5^3 coarsest: just sweep (or 1-workgroup kernel)
  smooth(pass, l, pre);                                   // each sweep = red dispatch + black dispatch
  dispatch(pass, P.residual, G[l].res, n[l]);
  dispatch(pass, P.restrictLevel, G[l].rst, n[l + 1]);    // also clears p[l+1]
  vcycle(pass, l + 1, pre, post);
  dispatch(pass, P.prolongAdd, G[l].pro, n[l]);           // samples mip l+1, read_write mip l: legal
  smooth(pass, l, post);
}
```

Grid sizes must halve cleanly. 128 → 64 → … → 4 gives 6 levels. 96 → 3 and 160 → 5 also give 6 levels. For tall grids like 64×96×64 → 2×3×2, the level count is limited by the smallest axis. Note the readback alignment trap I hit at 96³: `bytesPerRow` = 96 × 4 = 384 is not a multiple of 256 (§7.4).

**Temporally-blocked Jacobi (2 iterations per dispatch, about 2× faster) [measured].** Load a 12×12×8 tile (8×8×4 core plus a 2-cell halo) of p and b into `var<workgroup>` (3 × 4.6 KB). Do iteration 1 on the inner 10×10×6 cells, then iteration 2 on the 8×8×4 core, then write. Global traffic per iteration halves. The full kernel is `jacobi2` in `sim_reference.wgsl`. The same trick works for the finest-level multigrid smoother.

**Collocated vs MAC (judgement):** central-difference gradient and divergence on a collocated grid don't exactly match the compact 7-point Laplacian. So a converged p still leaves some checkerboard-mode divergence. This is fine visually and it is what GPU Gems does. A staggered MAC grid (u at x-faces and so on) is exact but needs 3 offset samples per velocity back-trace. Revisit only if smoke looks grainy or compressible.

### 3.7 Projection [measured: 64³ 0.44 ms, 128³ 3.4 ms]

```wgsl
@compute @workgroup_size(8, 8, 4)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= P.dims)) { return; }
  let c = vec3<i32>(id);
  let pc = textureLoad(pT, c, 0).x;
  let g = 0.5 * vec3<f32>(Pn(c + vec3<i32>(1,0,0), pc) - Pn(c - vec3<i32>(1,0,0), pc),
                          Pn(c + vec3<i32>(0,1,0), pc) - Pn(c - vec3<i32>(0,1,0), pc),
                          Pn(c + vec3<i32>(0,0,1), pc) - Pn(c - vec3<i32>(0,0,1), pc));
  var v = textureLoad(velTex, c, 0).xyz - g;
  if (c.y == 0) { v.y = max(v.y, 0.0); }        // floor
  textureStore(dst, c, vec4<f32>(v, 0.0));
}
```

`Pn` must use the **same ghost rule** as the solver: −p_c at open faces, p_c at the floor.

### 3.8 Workgroup shape [measured, 128³, ms]

| WG | advect | Jacobi |
|---|---|---|
| 4×4×4 | 4.51 | 2.13 |
| 8×8×1 | 4.23 | 2.18 |
| 8×4×4 | 4.56 | 2.20 |
| 8×8×4 | 4.47 | 2.19 |
| 16×4×4 | 4.50 | 2.16 |
| 32×2×2 | 3.98 | 2.08 |

Shape is not a lever here: the kernels are bandwidth-bound and textures are tiled. Use 8×8×4 (256 threads, matches workgroup-memory tiles) or 4×4×4. On RDNA3 (wave32/64) keep the total a multiple of 64. In the pure-bandwidth copy tests (128³/256³ textures), `64×1×1` on 3D textures was the worst shape (−10…−40 % vs 8×8×4).

### 3.9 Measured per-kernel cost table [measured, ms GPU timestamps, contended]

Headless new Chrome 155. A headful run gave similar numbers, from 0 to +40 % depending on kernel, which is load noise.

| kernel | 64³ | 96³ | 128³ | 160³ | 128×192×128 |
|---|---|---|---|---|---|
| SL advect rgba16f | 0.51 | 1.21 | 4.49 | 9.00 | 4.78 |
| SL advect rgba32f | 0.70 | 1.90 | 7.75 | 13.1 | 11.7 |
| MacCormack (2 dispatches) | 1.04 | 2.81 | 12.0 | 25.1 | 12.0 |
| divergence | 0.23 | 0.91 | 2.31 | 5.17 | 3.76 |
| vorticity + buoyancy (2 dispatches) | 0.87 | 2.72 | 10.2 | 21.7 | 14.5 |
| projection | 0.44 | 1.28 | 3.43 | 7.91 | 5.61 |
| MG V(2,2) | 1.4–2.0 | 10.8–11.4 | 24.4–25.3 | 44.8–51.5 | – |
| *full step, old style* (2×MC + vort + div + **40 Jacobi** + project) | 11.8 | 44.4 | 126 | 270 | 178 |
| *full step, recommended* (2×MC + vort + div + **2×MG V(2,2)** + project), derived | **6.3** | **33** | **89** | **188** | – |

Expect up to about 3–4× better numbers on an idle machine, because these kernels hit only 13–18 GB/s.

**Budgets (judgement, from the table):**
- **60 fps real time:** 64³ or 64×96×64 sim (about 6–9 ms) + half-res ray march (about 1–2 ms) + upsample/bloom/tonemap (about 1 ms). 96³ needs a 30 Hz sim. 128³ is not real time on this iGPU.
- **Offline video:** 160×240×160 is about 6.1 M cells (1.5× 160³). That is roughly 0.28 s per substep with 2 MG cycles, or 0.36 s with 3. At 4 substeps per frame that is 1.1–1.45 s of sim per frame. Add 0.1–0.3 s for 2–4 motion-blur sub-renders at 1080×1920 (with shadow taps) and about 0.3 s capture: **about 1.6–2 s per frame → about 16–20 min for 600 frames** (contended). 192×288×192 is about 1.75× that.

---

## 4. Volume ray marcher

`wgsl/volume.wgsl` (validated) contains `march` (compute, writes radiance + alpha and a transmittance-weighted depth) and `upsample` (joint bilateral 2×).

### 4.1 Ray setup from a fullscreen compute pass

```wgsl
let ndc = vec2<f32>((f32(id.x) + 0.5) / C.res.x * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / C.res.y * 2.0);
let pn = C.invViewProj * vec4<f32>(ndc, 0.0, 1.0);     // WebGPU clip-space depth is [0, 1]
let pf = C.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
let roW = pn.xyz / pn.w;  let rdW = normalize(pf.xyz / pf.w - roW);
let ro = (C.worldToVol * vec4<f32>(roW, 1.0)).xyz;     // volume space [0,1]^3
let rd = normalize((C.worldToVol * vec4<f32>(rdW, 0.0)).xyz);
// slab test
let inv = 1.0 / rd; let t0 = -ro * inv; let t1 = (vec3<f32>(1.0) - ro) * inv;
let tn = max(max(min(t0, t1).x, min(t0, t1).y), max(min(t0, t1).z, 0.0));
let tf = min(min(max(t0, t1).x, max(t0, t1).y), max(t0, t1).z);
```

A compute pass writing an `rgba16float` storage texture and a fullscreen-triangle fragment pass into an `rgba16float` render target are equivalent in cost. Compute makes it easy to also write a depth or alpha side channel. A fragment pass can render straight into the HDR scene target with blending.

### 4.2 Emission–absorption integration (energy-conserving step)

Per step of length Δ with extinction σ and emission ε (radiance per unit length), integrate analytically (Hillaire, SIGGRAPH 2015):

L += T · ε · (1 − e^(−σΔ)) / σ,   T *= e^(−σΔ)

```wgsl
let s = textureSampleLevel(scal, lin, p, 0.0);                    // x = T, y = smoke
let sigma = (C.sigmaSmoke * s.y + C.sigmaFire * s.x) / vox;
var Le = textureSampleLevel(bbLut, lin, vec2<f32>(clamp(s.x, 0.0, 1.0), 0.5), 0.0).rgb * C.emissionScale;
Le += s.y * C.glowStrength * textureSampleLevel(glow, lin, p, 3.0).rgb;   // smoke lit by fire (see 4.4)
let a = exp(-sigma * dt);
L += T * Le * select((1.0 - a) / max(sigma, 1e-8), dt, sigma < 1e-5);
T *= a;
```

The naive `L += T*Le*dt` overshoots badly for thick steps. The analytic form is stable for any σΔ, which lets you take bigger steps through dense smoke.

### 4.3 Step size, noise, early-out, empty space

- **Step:** ≤ 0.5 voxel (`stepVox = 0.5`). Larger steps band.
- **Jitter** the start by interleaved gradient noise (`fract(52.98·fract(dot(p, (0.0671, 0.00584))))`), animated by `frame & 63`. Real time: follow with TAA (history 0.8–0.9 plus neighbourhood clamp, since fire moves fast). Offline: average many jittered sub-frames instead (§7.3).
- **Early-out:** when T < 0.003.
- **Empty-space skipping:** a coarse occupancy texture (one texel per 8³ brick, storing max extinction over the brick **plus a 1-voxel halo**, rebuilt each frame by a reduction dispatch). When it reads 0, jump to the brick exit. The fire usually fills only 20–40 % of the box, so expect about 2–3× fewer samples (judgement).
- **Measured cost** (worst case: full box, no early-out, 2 samples per voxel):

| ray march | 64³ | 96³ | 128³ | 160³ |
|---|---|---|---|---|
| 960×540 | 0.86 | 0.97 | 2.10 | 3.57 |
| 960×540 + 6 shadow taps per sample | 1.59 | 1.85 | 8.62 | (29.5)† |
| 1920×1080 | 3.32 | 4.03 | 4.47 | 6.03 |
| 1920×1080 + 6 shadow taps | 4.77 | 5.94 | 10.3 | 22.5 |

†Noisy under load. Primary ray marching is cheap: texture fetches hit cache well and the sampler is fast. Secondary shadow rays per sample are what cost (2–5×). Keep them for offline only, or cache them in a light volume.

### 4.4 Lighting the smoke with the fire (cheap)

Fire is the light source. A **3D mip chain of the emission volume** (`rgba16float`, 2×2×2 box-filter compute pass per mip each frame) sampled at LOD 2–4 gives a blurred "nearby fire irradiance". Multiply it by smoke density and albedo, and smoke above the flame gets the characteristic orange underside for about one extra fetch per sample. For the offline video, optionally add a precomputed per-frame **light/transmittance volume** (e.g. 64³). March each voxel toward 2–4 representative points in the flame core, then the camera ray needs one fetch instead of N shadow taps.

### 4.5 Half-res render + depth-aware upsample

Ray march at 960×540 into colour (premultiplied radiance + alpha = 1 − T) plus a transmittance-weighted depth `r32float`. Upsample to 1080p:

```wgsl
let pl = (vec2<f32>(id.xy) + 0.5) * 0.5 - 0.5;          // low-res texel space
let base = vec2<i32>(floor(pl)); let f = pl - vec2<f32>(base);
for (var k = 0; k < 4; k++) {
  let o = vec2<i32>(k & 1, k >> 1);
  let q = clamp(base + o, vec2<i32>(0), ld - 1);
  let wb = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
  let wd = select(1.0, 0.02, textureLoad(lowDepth, q, 0).x > zf);   // behind the opaque surface => ~0 weight
  acc += textureLoad(lowColor, q, 0) * (wb * wd + 1e-4); wsum += wb * wd + 1e-4;
}
```

With no opaque geometry intersecting the volume (fire on a black background), plain bilinear upsampling of premultiplied colour is enough. Once logs or ground cut through the flame, the depth test removes halos. Measured gain from half res: about 2–3× (not 4×, because of fixed overheads). Offline: always render full resolution.

---

## 5. HDR targets and post chain (`wgsl/post.wgsl`, validated)

- **Targets:** scene + volume in `rgba16float` (renderable, blendable, filterable). Bloom pyramid in `rg11b10ufloat` (4 B/px, storage-write via tier1, renderable via `rg11b10ufloat-renderable`) or `rgba16float`. `float32-blendable` is available but not needed.
- **f16 range:** the maximum is 65504. Keep emission in **exposure-scaled units** (e.g. 1.0 = paper white) so hot cores stay below about 1e4. An f16 overflow is +Inf, and Inf − Inf = NaN, which poisons the bloom chain.
- **Bloom (Jimenez, COD:AW 2014):** 5–6 mips.
  - Downsample with the 13-tap filter (centre 0.125, 4 corners 0.03125, 4 edges 0.0625, inner 4 × 0.125). On the first mip use a **Karis average** (2×2 blocks weighted 1/(1 + luma/4)) to stop single hot voxels or sparks from flickering.
  - Upsample with a 3×3 tent (1-2-1 / 16, radius about 0.005 UV), adding each level to the next finer one.
  - Composite: `mix(hdr, bloom, 0.03–0.15)`.
- **Tonemap:** use **AgX** (the Blender 4.x default; WGSL port in `post.wgsl`, matrices from three.js/Filament). Fire is highly saturated and hot. Per-channel curves (Reinhard/ACES fit) skew orange to flat yellow and clip; AgX desaturates bright cores toward white the way film and sensors do (judgement backed by the AgX design goal). ACES-fit (Narkowicz/three.js) and Khronos PBR Neutral are alternatives.
- **Output encoding:** `getPreferredCanvasFormat()` is `bgra8unorm` (non-sRGB) on Windows, so apply the sRGB OETF in the shader, or render through a `bgra8unorm-srgb` view (`viewFormats`). **Dither** ±0.5 LSB before 8-bit quantization; smoke gradients band visibly otherwise, and x264 makes it worse.
- **HDR display (optional, live):** `context.configure({ device, format: 'rgba16float', toneMapping: { mode: 'extended' } })` (Chrome 129+) shows values > 1 on HDR monitors. The video path tonemaps itself.

```js
const ctx = canvas.getContext('webgpu');
ctx.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque',
                usage: GPUTextureUsage.RENDER_ATTACHMENT });           // add COPY_SRC only if you copy from it
new ResizeObserver(([e]) => {                                          // device pixels, no blur
  const s = e.devicePixelContentBoxSize?.[0];
  canvas.width = s ? s.inlineSize : Math.round(e.contentRect.width * devicePixelRatio);
  canvas.height = s ? s.blockSize : Math.round(e.contentRect.height * devicePixelRatio);
}).observe(canvas, { box: 'device-pixel-content-box' });
```

---

## 6. Profiling with timestamp queries

- Request the `timestamp-query` feature. Chrome **quantizes timestamps to 100 µs** unless started with `--enable-webgpu-developer-features` (the same as chrome://flags/#enable-webgpu-developer-features). That flag also unlocks `chromium-experimental-timestamp-query-inside-passes` (`pass.writeTimestamp`, dev only) and the detailed `adapter.info`.
- Granularity is per pass (`timestampWrites`). Put each logical stage in its own compute pass; pass boundaries are cheap.
- Deltas can be negative or garbage; discard them. Subtract as BigInt before `Number()`.

```js
class GpuProfiler {
  constructor(device, maxPasses = 32) {
    this.qs = device.createQuerySet({ type: 'timestamp', count: maxPasses * 2 });
    this.resolve = device.createBuffer({ size: maxPasses * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.ring = [0, 1, 2].map(() => device.createBuffer({ size: maxPasses * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    this.names = []; this.frame = 0;
  }
  pass(name) {             // spread into beginComputePass / beginRenderPass descriptors
    const i = this.names.push(name) - 1;
    return { timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } };
  }
  end(encoder) {           // call before finish()
    const n = this.names.length, rb = this.ring[this.frame++ % 3];
    encoder.resolveQuerySet(this.qs, 0, 2 * n, this.resolve, 0);
    if (rb.mapState === 'unmapped') encoder.copyBufferToBuffer(this.resolve, 0, rb, 0, 16 * n);
    const names = this.names; this.names = [];
    return async () => {   // call after submit()
      if (rb.mapState !== 'unmapped') return null;
      await rb.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(rb.getMappedRange().slice(0));   // copy BEFORE unmap (unmap detaches)
      rb.unmap();
      return Object.fromEntries(names.map((nm, i) => [nm, Number(t[2 * i + 1] - t[2 * i]) / 1e6]));
    };
  }
}
// usage: const p = enc.beginComputePass(prof.pass('advect')); ... p.end(); const read = prof.end(enc); queue.submit(...); (await read())
```

Cross-check timestamps with wall time: `submit` → `await queue.onSubmittedWorkDone()`. In the benchmarks the two agreed within 5–10 % for long passes.

---

## 7. Offline deterministic video capture

### 7.1 Architecture

```
capture.mjs (Node 22)
 ├─ http.createServer on 127.0.0.1:PORT  – serves the app (WebGPU needs a secure context: localhost or 127.0.0.1)
 │    └─ WebSocket /frames (binary)       – one message per frame, ack after ffmpeg.stdin accepted it (backpressure)
 ├─ spawn('ffmpeg', [... '-i', '-' ...])  – raw YUV420p (8- or 10-bit) on stdin
 └─ puppeteer-core → C:\Program Files\Google\Chrome\Application\chrome.exe, headless: true
        page ?capture=1&w=1080&h=1920&fps=30&frames=600&substeps=4&seed=7
        for f in 0..599:
          for s in 0..substeps-1: simStep(dtFixed, f*substeps+s); renderSubframe(accumulate)   // motion blur
          tonemap → rgb2yuv compute (dither) → copyBufferToBuffer(staging) → mapAsync → ws.send → await ack
```

### 7.2 Launching Chrome [measured]

```js
import puppeteer from 'puppeteer-core';                 // npm i puppeteer-core (v24 used here)
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true,                                       // = new headless (Chrome >= 132 has no --headless=old)
  args: [
    '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', // belt and braces; not required on this machine
    '--enable-webgpu-developer-features',               // precise timestamps (dev only)
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',         // matters for headful/minimised windows
    '--no-first-run', '--no-default-browser-check',
  ],
  userDataDir: 'D:/fire/.chrome-capture-profile',
  protocolTimeout: 0,                                   // long renders: never time out CDP calls
});
const page = await browser.newPage();
page.on('console', m => console.log('[page]', m.text()));
await page.goto(`http://127.0.0.1:${PORT}/?capture=1&frames=600`);
await page.waitForFunction('window.__captureDone === true', { timeout: 0, polling: 1000 });
```

- **New headless on Windows uses the real 780M** (D3D12, same features and limits as headful). It works even with no flags. Headful gave identical sim hashes and comparable timing (0 to +40 %, load noise).
- Unlike Linux, no Xvfb/display is needed. Rendering into **your own texture plus `copyTextureToBuffer`** sidesteps every canvas/compositor issue (luma.gl reports blank WebGPU canvas screenshots in some headless configurations). Do not rely on `page.screenshot()` or `canvas.toDataURL()`.
- Edge works the same way (`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`, Chromium 155).
- Playwright equivalent (not tested here): `chromium.launch({ executablePath, headless: true, args })`.
- `pipe: true` (CDP over stdio) worked but did not speed anything up.

### 7.3 Fixed timestep, substeps, determinism, motion blur

```js
const FPS = 30, SUB = 4, DT = 1 / (FPS * SUB);         // sim dt decoupled from wall clock
for (let f = 0; f < FRAMES; f++) {
  clearAccum();
  for (let s = 0; s < SUB; s++) {
    const step = f * SUB + s;
    simStep(DT, step);                                  // seeds = hash(step), never Math.random()/performance.now()
    if (s >= SUB / 2) renderSubframe(step);             // 180° shutter: average the 2nd half of the frame interval
    await device.queue.onSubmittedWorkDone();           // keep each submit short (TDR, §8)
  }
  resolveAccumToTonemapped(); await captureFrame(f);
}
```

- **WGSL RNG:** use an integer hash (PCG/xxhash-style) of (cell, step, seed). Never feed wall-clock time or `Math.random()` into the sim.
- **Avoid nondeterminism:** no float atomics, no unordered in-place updates (plain Gauss-Seidel without red-black colouring races), no reductions whose order depends on scheduling.
- **[measured]** 30 full steps at 64³ (MacCormack, vorticity, Jacobi, …) produced the **same FNV hash** in two runs in one page, across two processes, and headless vs headful. A crashed render can therefore be **resumed by re-simulating** to frame *f* without rendering (sim is cheaper than render plus capture). For long jobs, also checkpoint state to disk every about 100 frames via the same WS.
- **Motion blur:** fire filmed at 24/30 fps with a 180° shutter shows streaks. Average 2–4 jittered sub-frame renders over the open-shutter half of each frame, in an `rgba32float` accumulator (or rgba16f). Different jitter per sub-frame also anti-aliases and removes ray-march noise. This is the biggest "looks like real footage" win per millisecond in the offline path (judgement).
- **Live mode:** use the same `simStep` with an accumulator. `acc += min(frameDt, 0.1); while (acc >= DT && n++ < 2) { simStep(DT); acc -= DT; }`. Cap the steps so a slow frame doesn't spiral.

### 7.4 Readback

```js
const W = 1080, H = 1920;
const bytesPerRow = Math.ceil(W * 4 / 256) * 256;         // 4320 -> 4352 (MUST be a multiple of 256)
const staging = [0, 1, 2].map(() => device.createBuffer({ size: bytesPerRow * H,
                 usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
enc.copyTextureToBuffer({ texture: outTex }, { buffer: staging[i % 3], bytesPerRow, rowsPerImage: H }, [W, H]);
device.queue.submit([enc.finish()]);
await staging[i % 3].mapAsync(GPUMapMode.READ);
const src = new Uint8Array(staging[i % 3].getMappedRange());
for (let y = 0; y < H; y++) packed.set(src.subarray(y * bytesPerRow, y * bytesPerRow + W * 4), y * W * 4);  // strip padding
staging[i % 3].unmap();                                   // detaches `src` – copy out first!
```

- Measured: 1920×1080 RGBA8 readback and copy-out takes 11.9 ms (82 fps). 1080×1920 with padded rows takes 14.5 ms (68 fps). Readback is never the bottleneck.
- **Better:** write the frame tightly packed into a **storage buffer** in a compute pass (the YUV kernel below does this), then `copyBufferToBuffer`. That avoids the 256-byte row rule and the CPU un-padding loop.
- 3D texture copies need `rowsPerImage`, and the same 256 rule applies. My 96³ residual readback (384 B rows) failed validation.

### 7.5 Transfer channel benchmark (1080×1920, Chrome → Node) [measured, loaded machine]

| channel | payload / frame | fps | effective MB/s |
|---|---|---|---|
| HTTP `fetch` POST (Uint8Array body) | 7.9 MB RGBA | 0.55–0.86 | 4–7 |
| `page.exposeFunction` + base64 (CDP over WebSocket) | 7.9 MB → 10.5 MB b64 | 1.1 | about 9 |
| `exposeFunction` with `pipe: true` | " | 0.65 | 5 |
| Browser download of Blobs (`Browser.setDownloadBehavior`) | 15 frames per file | 0.16 | — (disk + Defender scan) |
| **WebSocket binary, RGBA** | 7.9 MB | 1.45–2.1 | 12–17 |
| **WebSocket binary, GPU-converted YUV420p** | **3.0 MB** | **3.9–5.4** | 12–17 |
| WebSocket binary, YUV420p10le | 5.9 MB | 1.6–2.5 | 10–15 |

The ceiling here was the machine's degraded loopback (Node↔Node TCP was 21–139 MB/s during the run). On an idle machine expect hundreds of MB/s and 20+ fps RGBA. Either way, the offline render (about 1–2 s per frame) dominates, so **WebSocket + GPU YUV420** is plenty.

Node side: a dependency-free WS sink with backpressure is in `webgpu-bench/ws.mjs` and `server.mjs`. Its essential part:

```js
attachWS(server, '/frames', buf => new Promise(ok => {          // resolves -> server sends 'ok' ack
  if (!ff.stdin.write(buf)) ff.stdin.once('drain', ok); else ok();
}));
```

Page side:

```js
const ws = new WebSocket(`ws://${location.host}/frames`); ws.binaryType = 'arraybuffer';
const acks = []; ws.onmessage = () => acks.shift()?.();
const send = data => new Promise(r => { acks.push(r); ws.send(data); });   // await => backpressure
```

### 7.6 GPU RGB → YUV 4:2:0 BT.709 (limited range) with dither (`yuvShader()` in `bench.js`, validated)

Input is the tonemapped, sRGB-encoded **float** texture, so quantize only once, in YUV. One thread handles an 8×2 pixel block and writes packed `u32`s for the Y, U and V planes. W must be a multiple of 8 (1080 and 1920 are).

```wgsl
fn Yv(c: vec3<f32>) -> f32 { return 16.0 + 219.0 * dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn UVv(c: vec3<f32>) -> vec2<f32> {
  let y = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  return vec2<f32>(128.0) + 224.0 * vec2<f32>((c.b - y) / 1.8556, (c.r - y) / 1.5748);
}
fn q(v: f32, p: vec3<u32>) -> u32 {            // TPDF dither (two uniform hashes) then round
  let d = hash(p) - hash(p + vec3<u32>(7u, 11u, 13u));
  return u32(clamp(round(v + d), 0.0, 255.0));
}
// Y plane: W*H bytes; U: (W/2)*(H/2); V: same. Chroma = average of each 2x2 block (centre-sited).
// 10-bit variant: scale by 4 (64..940 / 64..960), clamp to 1023, pack 2 x u16 per u32 -> yuv420p10le.
```

**[measured]** The GPU-YUV path and the RGBA → swscale path decode to the same picture: PSNR 46.5 dB between the two x264 CRF-16 encodes (dither and coding noise only, no colour shift).

### 7.7 ffmpeg command lines [measured with ffmpeg 8.0.1]

Colour test: fire-colour patches encoded lossless (`-qp 0`), then decoded as a correct BT.709 player would. Max error in 8-bit code values:

| encode path | ffprobe tags | max error | (200,30,0) became |
|---|---|---|---|
| `-pix_fmt yuv420p` only (naive) | all unknown | **39** | 215,44,0 |
| `scale=out_color_matrix=bt709:out_range=tv,format=yuv420p` + `-colorspace/-color_*` options | space/range ok, **trc/primaries "unknown"** | 2 | 200,30,0 |
| same + `:flags=accurate_rnd+full_chroma_int` + **`setparams=…`** | all bt709/tv | **1** | 200,30,1 |
| raw YUV from GPU, **untagged input** + output tags | | **25** | 182,12,1 (ffmpeg re-converted it as BT.601!) |
| raw YUV from GPU, **tagged on the input side** | all bt709/tv | **1** | 200,30,1 |
| 10-bit x265 from RGBA via scale + setparams | all bt709/tv, Main 10, hvc1 | **0** | 200,30,0 |

Findings:
- swscale's default matrix is BT.601. Players assume BT.709 for HD, so fire hues shift.
- In ffmpeg 8 the output `-color_primaries`/`-color_trc` options did **not** reach the libx264 VUI in my test. Use `setparams` or `-x264-params`.
- ffmpeg 8 negotiates colour spaces automatically, so an untagged raw-YUV input gets "converted" from its assumed BT.601 to the BT.709 you asked for. **Tag raw YUV input before `-i`.**

**A. RGBA frames on stdin → H.264 delivery file**
```
ffmpeg -f rawvideo -pix_fmt rgba -s 1080x1920 -r 30 -i - ^
  -vf "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709" ^
  -c:v libx264 -preset slow -crf 14 -profile:v high -x264-params aq-mode=3 ^
  -movflags +faststart fire_1080x1920.mp4
```

**B. GPU YUV420p (8-bit) frames on stdin (recommended capture path)**
```
ffmpeg -f rawvideo -pix_fmt yuv420p -s 1080x1920 -r 30 ^
  -colorspace bt709 -color_range tv -color_primaries bt709 -color_trc bt709 -i - ^
  -c:v libx264 -preset slow -crf 14 -profile:v high ^
  -x264-params "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv:aq-mode=3" ^
  -movflags +faststart fire_1080x1920.mp4
```

**C. 10-bit master (GPU yuv420p10le on stdin, or RGBA with `format=yuv420p10le` in the filter)**
```
ffmpeg -f rawvideo -pix_fmt yuv420p10le -s 2160x3840 -r 30 ^
  -colorspace bt709 -color_range tv -color_primaries bt709 -color_trc bt709 -i - ^
  -c:v libx265 -preset slow -crf 12 -pix_fmt yuv420p10le -tag:v hvc1 ^
  -x265-params "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited" master_10bit.mp4
```

**D. Delivery from the master, with procedural audio**
```
ffmpeg -i master_10bit.mp4 -i fire_audio.wav ^
  -vf "scale=1080:1920:flags=lanczos,format=yuv420p,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709" ^
  -c:v libx264 -preset slow -crf 15 -profile:v high -x264-params aq-mode=3 ^
  -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart fire_social.mp4
```

Notes:
- `-r 30` belongs before `-i -` for rawvideo. Frame size must match exactly.
- Use `^` in cmd.exe and `` ` `` in PowerShell for line continuation. From Node, pass an argv array; no shell quoting is needed.
- Platforms re-encode everything. Uploading a high-bitrate 2160×3840 file to YouTube Shorts gets a better encode ladder. Instagram/TikTok scale to 1080 wide.
- `aq-mode=3` biases bits toward dark regions, which helps the dark smoke around the flame.

### 7.8 Robustness during a long render

- `device.lost.then(info => …)`: `info.reason === 'destroyed'` means you called `destroy()`. Anything else (`undefined`/`'unknown'`) is a real loss. Then: request a new device, rebuild all resources, **re-simulate deterministically to the current frame**, and continue. Cap retries.
- Capture uncaptured errors (`device.onuncapturederror`) to the Node log. Use `pushErrorScope('validation'|'out-of-memory')` around resource creation.
- Node: `ffmpeg.on('close', code => …)`, and make sure stdin is ended. A half-written MP4 without `+faststart` is still recoverable, but one without a moov atom is not, so always end stdin cleanly.

---

## 8. Pitfalls checklist (★ = hit during this research)

1. **Windows TDR.** The default `TdrDelay` is 2 s (no registry override on this machine). A single GPU job that runs longer than that without preemption resets the driver; all WebGPU devices are then lost and the screen flickers. Rules:
   - keep every dispatch ≲ 100–200 ms;
   - keep every `submit` ≲ 0.5–1 s;
   - in offline mode, `await device.queue.onSubmittedWorkDone()` between substeps and sub-renders;
   - split huge ray marches into tiles (e.g. 4 dispatches of 1/4 of the image).

   At 160³ the biggest single dispatch measured was about 25 ms, so the risk only comes from piling up work. The WebGPU spec also lets the browser lose the device if "shader execution does not end in a reasonable amount of time".
2. **Device lost** after a GPU process crash, driver update or TDR: see §7.8. Test it with `device.destroy()`.
3. ★ **`bytesPerRow` must be a multiple of 256** for `copyTextureToBuffer`/`copyBufferToTexture` (96³ r32f → 384 B failed). `queue.writeTexture` has no such rule.
4. ★ **`unmap()` detaches the mapped ArrayBuffer.** Reading `.length` afterwards gives 0 and silently produces Infinity/NaN. Copy or reduce before unmapping.
5. ★ **WGSL needs parentheses** when mixing some operators (`a * b ^ c` is a compile error). Bitwise ops, comparisons and arithmetic don't share a precedence chain.
6. ★ **`smooth` is a reserved word** in WGSL (interpolation), and so are many others (`sample`, `flat`, `filter`, …). Name entry points distinctively.
7. **Multiple `writeBuffer`s to one uniform before one `submit` → last write wins** (§3.1).
8. **Uniform layout:** vec3 align 16, struct size rounded, array stride 16 in uniform (unless `uniform_buffer_standard_layout`); dynamic offsets 256-aligned.
9. **Usage-scope conflicts:** the same subresource sampled and storage-written in one dispatch is a validation error. Different mips are fine.
10. **Texture usage flags** must cover every use (`TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC | COPY_DST | RENDER_ATTACHMENT`). The canvas texture defaults to `RENDER_ATTACHMENT` only.
11. **Default limits are low:** request `maxStorageTexturesPerShaderStage`, `maxComputeWorkgroupStorageSize`, `maxComputeInvocationsPerWorkgroup`, `maxBufferSize`, `maxStorageBufferBindingSize` explicitly. Shaders that exceed the defaults fail pipeline creation.
12. **`layout: 'auto'`** drops unused bindings and guesses sample types. Use explicit layouts for shared bind-group code.
13. **NaN/Inf propagation** in fluid sims. One NaN cell spreads through advection and the pressure solve in a few frames. Sources:
    - `normalize(0)` (vorticity N, use ε);
    - `x/0` (density-normalized quantities);
    - `log2(0)`, `pow(neg, y)`;
    - f16 overflow above 65504 (Inf − Inf);
    - MacCormack without a limiter.

    WGSL does not guarantee IEEE NaN/Inf behaviour: there is no `isnan` builtin, and compilers may fold `x != x` to false (spec §15.7.2 "Differences from IEEE-754"; Dawn notes WGSL "does not support NaNs or Infinities"). Use bit tests:

    ```wgsl
    fn bad(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u; }   // NaN or Inf
    fn sane4(v: vec4<f32>) -> vec4<f32> {
      return select(v, vec4<f32>(0.0), vec4<bool>(bad(v.x), bad(v.y), bad(v.z), bad(v.w)));
    }
    ```

    Sanitize in the advection write and clamp |u| to a CFL-safe maximum (e.g. 4 cells per substep). In debug builds, run a reduction that `atomicAdd`s a NaN counter into a storage buffer each frame and log it.
14. **Out-of-bounds `textureLoad`** returns an indeterminate (but safe) value. Clamp coordinates yourself (`cl()`), and use a guard `if (any(id >= dims)) { return; }` for partial workgroups.
15. **Clamp-to-edge only, no border colour.** Handle inflow and ambient boundaries manually (§2.2).
16. **Timestamp quantization** is 100 µs without the dev flag, and negative deltas happen.
17. **Secure context:** `navigator.gpu` is undefined on `about:blank`, `page.setContent()` pages and some `file:` setups. Serve from `http://127.0.0.1`.
18. **Background throttling** (headful): use the three `--disable-*backgrounding*/throttling` flags. Never drive the capture loop from `requestAnimationFrame`.
19. **ffmpeg colour** (§7.7): explicit BT.709 matrix, all four tags, tagged raw-YUV input.
20. **Banding:** dither before any 8-bit quantization (tonemap output, YUV conversion).
21. **`float32-filterable`** is needed to sample `r32float` (MG prolongation via hardware trilinear). Otherwise bind it as `unfilterable-float` and interpolate by hand.
22. **Mixed asymmetric DIMMs** (16 + 48 GB): memory beyond the interleaved 32 GB runs single-channel. That is harmless here because GPU allocations are small, but keep total working sets modest (judgement).

---

## 9. Recommended configurations

**Real-time (browser, 60 fps target on the 780M)**
- Grid 64×96×64 (or 64³), cell size about 1.5 cm for a 1–1.5 m flame. Velocity in cells/s, CFL ≤ 2 per step, fixed 1/60 s sim.
- Per step: source/combustion → MacCormack(velocity, RK2) → MacCormack(scalars) → curl + confinement/buoyancy → divergence → MG V(2,2) ×1–2 warm-started → projection. About 6–9 ms (contended).
- Ray march at 960×540 with stepVox 0.5, IGN jitter, empty-space skipping, glow mips for smoke lighting. Joint-bilateral upsample, TAA (0.85), bloom (5 mips, rg11b10ufloat), AgX, dither.

**Offline video (20 s, 30 fps, 1080×1920 delivery / 2160×3840 master)**
- Grid 160×240×160 (or 192×288×192). 4 substeps per frame (dt = 1/120 s). MacCormack everywhere. MG V(2,2) ×3. Higher ε near the flame.
- Render at full output resolution, stepVox 0.5. Shadow/light volume or 4–6 shadow taps. 2–4 jittered sub-frames over a 180° shutter, accumulated in rgba32f. Then bloom → AgX → GPU YUV (10-bit for the master) → WS → ffmpeg (§7.7 C), and a delivery encode with the audio track (§7.7 D).
- Expected: about 1.6–2 s per frame → about 16–20 min for 600 frames at 160×240×160 (contended estimate; likely 2–3× faster on an idle machine).

---

## 10. Benchmark harness (`D:\fire\research\webgpu-bench\`)

```
npm i puppeteer-core            # node_modules not committed
node run.mjs new probe          # adapter info / features / limits / format probes
node run.mjs new grid           # all fluid kernels at 64,96,128,160,128x192x128 (SIZES=... to override)
node run.mjs new mg             # GPU multigrid convergence + cost (SIZES=64,128)
node run.mjs new tb             # temporally-blocked + r16f Jacobi, dispatch overhead
node run.mjs new bw             # bandwidth + ALU
node run.mjs new xfer           # Chrome->Node transfer channels + ffmpeg (PLAN=alt|verify)
node run.mjs new determinism    # hash of 30 sim steps, twice
node run.mjs new compile        # validates wgsl/*.wgsl
node run.mjs false grid         # same, headful window
node poisson.mjs 128 face       # CPU convergence study (Jacobi / SOR / MG), ghost=zero|face
```

- **Files:** `bench.js` (all kernels plus harness), `server.mjs`/`ws.mjs` (static server, POST/WS sink, ffmpeg spawn), `wgsl/` (`mg.wgsl`, `post.wgsl`, `volume.wgsl`, `sim_reference.wgsl`), `color/` (ffmpeg colour-matrix test generator and comparer), `results/` (all JSON outputs, `poisson-128-face.txt`).
- **Env vars:** `NOBASE=1` (no extra Chrome flags), `PIPE=1` (CDP over pipes), `SIZES`, `PLAN`.

---

## Sources

- WebGPU spec (W3C CR): https://www.w3.org/TR/webgpu/ — limits, usage scopes, texture-format capabilities, device loss.
- WGSL spec: https://www.w3.org/TR/WGSL/ — language extensions `readonly_and_readwrite_storage_textures`, `uniform_buffer_standard_layout`, `immediate_address_space`; §15.7.2 Differences from IEEE-754. Dawn commit on NaN/Inf handling: https://dawn.googlesource.com/dawn/+/157e414792326956e28c37c01ad5b3e8d7cbf0ca
- Chrome "New in WebGPU": 119 (float32-filterable) https://developer.chrome.com/blog/new-in-webgpu-119 · 121 (timestamp-query) https://developer.chrome.com/blog/new-in-webgpu-121 · 124 (read-only/read-write storage textures, textureBarrier) https://developer.chrome.com/blog/new-in-webgpu-124 · 129 (HDR canvas toneMapping extended) https://developer.chrome.com/blog/new-in-webgpu-129 · 142 (texture-formats-tier1/tier2) https://developer.chrome.com/blog/new-in-webgpu-142 · 149–150 (immediates) https://developer.chrome.com/blog/new-in-webgpu-149-150
- MDN `GPUDevice.createTexture` (exact tier1/tier2 format lists): https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createTexture
- Chrome WebGPU developer features (timestamp quantization, adapter.info extensions, strictMath): https://developer.chrome.com/docs/web-platform/webgpu/developer-features
- Chrome WebGPU troubleshooting: https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips · Removing `--headless=old`: https://developer.chrome.com/blog/removing-headless-old-from-chrome · Headless WebGPU testing: https://developer.chrome.com/blog/supercharge-web-ai-testing
- Intent to Ship, uniform buffer standard layout: https://groups.google.com/a/chromium.org/g/blink-dev/c/Ww2eL6b74V0/m/D8AT9DWlAQAJ
- web3dsurvey (feature/limit availability: float32-filterable, texture-formats-tier2, maxStorageTexturesPerShaderStage, maxComputeWorkgroupStorageSize): https://web3dsurvey.com/webgpu
- webgpufundamentals — storage textures: https://webgpufundamentals.org/webgpu/lessons/webgpu-storage-textures.html · timing: https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html
- luma.gl issue #2874 (headless WebGPU canvas capture blank; offscreen readback robust): https://github.com/visgl/luma.gl/issues/2874
- Crane, Llamas, Tariq — GPU Gems 3 ch. 30 "Real-Time Simulation and Rendering of 3D Fluids": https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids
- Selle, Fedkiw, Kim, Liu, Rossignac — "An Unconditionally Stable MacCormack Method", J. Sci. Comput. 35 (2008): https://research.adobe.com/publication/an-unconditionally-stable-maccormack-method
- Kim, Liu, Llamas, Rossignac — "FlowFixer: Using BFECC for Fluid Simulation" (2005).
- Fedkiw, Stam, Jensen — "Visual Simulation of Smoke", SIGGRAPH 2001 (vorticity confinement). Nguyen, Fedkiw, Jensen — "Physically Based Modeling and Animation of Fire", SIGGRAPH 2002.
- McAdams, Sifakis, Teran — "A Parallel Multigrid Poisson Solver for Fluids Simulation on Large Grids", SCA 2010: https://pages.cs.wisc.edu/~sifakis/project_pages/mgpcg.html
- Hillaire — "Physically Based and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015: https://advances.realtimerendering.com/s2015/
- Jimenez — "Next Generation Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014: https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/ · LearnOpenGL physically based bloom: https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
- three.js tone mapping chunk (AgX / ACES / Neutral constants, MIT): https://github.com/mrdoob/three.js/blob/dev/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js
- Inigo Quilez — hardware interpolation precision (24.8 fixed point): https://iquilezles.org/articles/hwinterpolation/
- FFmpeg trac #979 (scale/zscale BT.709 conversion pitfalls): https://trac.ffmpeg.org/ticket/979 · scale in/out_color_matrix commit: https://ffmpeg.org/pipermail/ffmpeg-cvslog/2013-July/066579.html
- MDN `GPUDeviceLostInfo.reason`: https://developer.mozilla.org/docs/Web/API/GPUDeviceLostInfo/reason
