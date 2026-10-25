# Visual Reference Study: Real Fire

How real wood fires (campfire, bonfire, log/fireplace fire, embers, sparks, torch) look in photographs and video, measured from 30 reference photos and 8 videos. Each claim is labelled either as a **measurement** I made on the references or as **literature**. The goal is a target spec and a review checklist for a fully procedural WebGPU fire and a 20 s offline cinematic.

---

## 0. TL;DR: the 15 numbers that matter most

| # | Finding | Value (measured unless noted) |
|---|---|---|
| 1 | **Camera "fire ramp"** (pixel brightness → colour) is the same in almost every photo. | dark red `#471803` → orange `#CB5C03` → `#FC8E05` → amber `#FFB418` → yellow `#FFDF29` → pale `#FFFF79`/white (vivid JPEG). Neutral/cinematic version: `#421908` → `#B5611E` → `#EE8C46` → `#FEC27A` → `#FFE8B2` → `#FEFCD2` (§2.1) |
| 2 | Hue climbs with brightness: **~11–18° at the dim tips → 25–30° in the body → 45–60° in the yellow core**. | median flame hue 26–29°, almost no flame pixels beyond 60° |
| 3 | **R clips first.** R=255 in a median ~45% of flame pixels (11–66%). R+G both clip in ~5% (1–48%, high only in candle and torch). All three clip in ~1–2% (0–10%, candle 29%). | white core is **< 1% of the image and < 5% of the flame** in campfire shots |
| 4 | **Dim flame dominates by area.** Flame pixels with luma 0.10–0.55 cover **~2× (1.3–5.5×)** the area of pixels with luma > 0.55. | e.g. #01: faint 23% + edge 12% vs body 11.5% + inner 5.4% |
| 5 | Going up the flame it gets **darker, redder and more saturated**. Flame coverage peaks at 0.3–0.5 H. The top 15% is dull `#A74109`…`#833B11`, hue ~20–23°, sat > 0.9. | §2.3 |
| 6 | **Blue/violet base is subtle.** It shows in 0–13% of flame pixels and only at flame roots on wood. In photos it reads as lavender/violet (`#8E527E`, bright `#E5A4CB`, deep `#221351`), not pure blue. | §2.6 |
| 7 | Night **background is ~black but warm**: image median luma 0.01–0.05. Near-black pixels next to the fire are tinted `#150503`–`#270400` (hue 6–21°), not neutral. | §2.9 |
| 8 | **Edges are bimodal.** Some edges are razor-sharp (10–90% rise in 2–6 px at 1920 px) and many are soft fades (median 6–35 px, p90 40–70 px). | §3.3 |
| 9 | **Texture spectrum** of flame luma follows a power law with radial PSD slope **−2.5 … −4.1 (median ≈ −3.2)**. Features are **vertically elongated**: power at horizontal frequencies is 1.4–8.8× that at vertical ones. | §3.4 |
| 10 | **Light-output flicker**: σ/μ = **6–10%**. Frame-to-frame change is 2.4–3.6% (median), 6–9% (p90). Max/min within 1 s is **1.15–1.32**. 80% of flicker power lies **below ~3–4 Hz**. | §4.2 (3 night videos) |
| 11 | **Puffing frequency** (literature): f ≈ 1.5/√D (D in m) → 2.1 Hz for a 0.5 m fire, 0.9 Hz for a 3 m bonfire. A candle disturbed into flicker oscillates at 10–15 Hz. | §4.1 |
| 12 | **Flame tips** decorrelate in **~70 ms** (pixel ACF < 0.5). The base is **3–4× slower** (170–300 ms). Tips grow at **~1.5–2.3 flame heights/s**. Detached pockets are visible in **~60% of frames**, about **7 episodes/s**, and live **67 ms (median) / 170–190 ms (p90)**. | §4.3 (3 videos) |
| 13 | **Sparks are thin.** Long-exposure trails are **2–2.5 px wide** (p90 2.3–4.4 px) at 1920 px. Colour is `#A2431D`…`#D96A33` (hue 17–22°), cooling to `#833A33`/`#A64921` aloft. Only young sparks have pale cores (`#FFE9B5`). Paths are straight near the fire and curl in loops higher up. | §6 |
| 14 | **Ground light pool** (night): 50% at ~1.5 fire radii, 15–25% at 2 r, then roughly r⁻² (4% at 5 r). Lit-ground colour runs `#994013` → `#3F1507` → black. | §7 |
| 15 | **Embers are redder than flames at equal brightness** (hue 2–15° vs 18–27° at luma 0.13–0.43). Dying embers go crimson (`#E52342`, hue ~350°). Ash is white-grey (`#948B8C` in daylight). | §2.8, §9 |

---

## 1. Reference set and method

### 1.1 Downloaded references

All images are in `C:\Users\User\AppData\Local\Temp\claude\D--fire\bc550da3-92c2-4cba-b445-153d7412a165\scratchpad\refs\` and come from Wikimedia Commons (1920 px wide thumbnails; licences vary, CC-BY/CC-BY-SA, for internal reference only). Videos are in `...\scratchpad\refs_video\`.

| File | Commons source | Content / why useful | Shutter character |
|---|---|---|---|
| `01_campfire_flames_night.jpg` | File:Campfire_flames_at_night.jpg | Brushwood campfire at night: flame sheets behind silhouetted twigs, glowing red twig filaments | medium (~1/60), slight streak |
| `02_campfire_at_night.jpg` | File:Campfire_at_night.jpg | **Rejected**: heavy processing and flare. Ignore. | – |
| `03_anttoora_sparks1.jpg` | File:Campfire_and_sparks_in_Anttoora_1.jpg | Long-exposure spark arcs next to the flame wall, dashed (tumbling) trails | long (~1 s) |
| `04_anttoora_sparks3.jpg` | ..._3.jpg | Dense curly spark trails, random-walk loops | long |
| `05_anttoora_sparks4.jpg` | ..._4.jpg | Thick out-of-focus spark trails above flames | long |
| `06_campfire_tulppio.jpg` | File:Campfire_in_Tulppio.jpg | Small fire in a forest at dusk: scale and context, ambient-dominated | – |
| `07_campfire_closeup_red.jpg` | File:Campfire_Close_Up_-_Red_Flames.jpg | Close-up of thin, folded, translucent flame **sheets** with sharp folds | fast |
| `08_closeup_of_fire.jpg` | File:Close-up_of_a_Fire.jpg | Log with **alligator char** and glowing crack rims, translucent flame, tiny blue root, sparks | medium |
| `09_flames1.jpg`, `10_flames3.jpg` | File:Flames_1.jpg, File:Flames_3.jpg | Natural, underexposed colour. Wispy tongues, **detached pockets**, faint violet at the roots | medium-fast |
| `11_flames_spark_trails.jpg` | File:Flames_with_Spark_Trails.jpg | Long exposure: flames become vertical veils, sparks curl | long |
| `12_wood_closeup_bs.jpg` | File:Flames_of_fire_burning_wood_close-up_(2026)-bs.jpg | Phone shot against a stone wall: you can see **through** the flames, and curling tips | phone HDR |
| `13_fireplace_wood_rostov.jpg` | File:Fireplace,_Wood_fire,_Rostov-on-Don,_Russia.jpg | Fireplace overview: **white clipped core**, light spill on brick, ash | fast |
| `14_fireplace_rostov.jpg` | File:Fire,_Fireplace,_Rostov-on-Don,_Russia.jpg | Logs on a **white ash bed**, lavender flame roots, white cores, orange tips | fast |
| `15_candle_closeup.jpg` | File:Candle_flame_closeup.jpg | Candle anatomy: dark wick zone, blue rim at the base, white core, orange envelope | – |
| `16_flames_52633350622.jpg` | File:Flames_(52633350622).jpg | **Lavender/violet flame sheet** at the root, curl and detached tip, glowing ember bed under a grate | fast |
| `17_campfire_15188333373.jpg` | File:Campfire_(15188333373).jpg | Stirred campfire **spark burst**, glowing coal bed, grass lit red | ~1/10 |
| `18_fire_closeup_texture.jpg` | File:Fire_close_up_texture.jpg | Overlapping translucent sheets with **bright limb lines** | fast |
| `19_glowing_embers_unsplash.jpg` | File:Glowing_Embers_(Unsplash).jpg | Low-key ember bed: glow from crevices, tiny blue flames | – |
| `20_hot_embers_night.jpg` | File:Hot_embers_in_night.jpg | **Dying embers**: crimson only | phone |
| `21_gluehende_kohle.jpg` | File:Gluehende_Kohle_(CloseUp)_001.JPG | Charcoal with grey **ash crust** and glowing veins | daylight |
| `22_swirling_sparks_hilsenhain.jpg` | File:Swirling_sparks,_shrovetide_bonfire_at_Hilsenhain,_Germany.jpg | Bonfire **spark plume**, smoke haze lit orange from below | long |
| `23_long_exposure_sparks_ee.jpg` | File:Pika_säriga_jäädvustatud_sädemed_lõkkes.jpg | Bonfire sparks against the night sky, moonlit cloud | long |
| `24_sparks_fade_ru.jpg` | File:Искры_гаснут_на_лету.jpg | Clipped white core, short straight spark streaks, person lit from below | ~1/15 |
| `25_knisternd.jpg` | File:Knisternd.JPG | **Best ember bed + glowing cracked log + white/blue flame + spark trails** | ~1/8 |
| `26_lagerfeuer_glut.jpg` | File:Lagerfeuer_mit_Glut_Mai_2012.JPG | Daylight: char cracks, ash coat, fresh wood vs charred wood | daylight |
| `27_bonfire_sparks.jpg` | File:Bonfire_Sparks.jpg | Spark burst over an ember field: thick, short streaks | ~1/10 |
| `28_fire_torch_geograph.jpg` | View_of_a_fire_torch...5965831.jpg | Gas torch fireball at dusk: **rolling vortices**, detached flakes | fast |
| `29_wood_in_the_fire.jpg` | File:Wood_In_The_Fire.jpg | Late-stage log glowing all over with **rectangular char cells** | – |
| `30_people_campfire.jpg` | File:People_sitting_around_a_camp_fire.jpg | Wide night shot: people lit by the fire, falloff, sparks as dots | phone |

Videos (Commons): `v2` Campfire_at_night_-_Claytor_Lake_State_Park (fire pit at dusk, 30 s), `v3` Fireplace_1_2021-02-14 (gas log, blue), `v4` Feuer.webm (bonfire; frame cadence is odd, so it was excluded from motion stats), `v5` Peterfeuer…mit_Funkenflug (bonfire teepee with a huge spark column), `v6`/`v7` Campfire_Video_1/2 (phone, night, slightly top-down), `v8` Fire_burning_in_a_fireplace_in_Bosnia (outdoor fireplace, possibly slowed, so it was excluded from speed stats).

### 1.2 Tools

These can be reused to score renders. They live in `...\scratchpad\refs_tools\`. Run them with `...\scratchpad\venv\Scripts\python.exe -I <script>`; the venv has Pillow, numpy and scipy.

* `analyze.py <config.json> <img_dir> <out.json>`: per-image ROI analysis. It covers the flame mask, brightness classes (median hex and equivalent blackbody T), luma→colour ramp, base→tip profile, hue/sat/value histograms, clipping, edge widths, PSD slope, anisotropy, autocorrelation aspect, spark stats and ember/ash stats. Configs used: `config1.json`, `config2.json`, `config_scene.json`. Outputs: `out1.json`, `out2.json`, `out3.json`.
* `summarize.py <out.json> main|ramp|prof|bb` prints the tables. `ramp_agg.py` aggregates the ramps by family.
* `video.py <video> x0 y0 x1 y1 t0 dur` computes the light-flicker spectrum, intermittency, tip height and pixel decorrelation. `tipspeed.py` gives tip rise speed and pocket lifetimes. `lightfall.py` gives the radial light falloff.
* `crop.py` (crop and colour stats) and `contact.py` (labelled contact sheet). Crops and contact sheets are in `...\scratchpad\refs_crops\`. `best_refs_contact.jpg` shows the 8 best references side by side.

**To score a render:** save the screenshot, add it to a config with an ROI whose bottom edge is the flame base, and run `analyze.py`. Then compare against §2–§3: luma ramp hexes, class fractions, clip fractions, edge widths, PSD slope, anisotropy and near-black tint.

Flame mask definition used throughout: luma > 0.25, R ≥ B, and (hue ≤ 65° or ≥ 335° or sat < 0.18). Luma is Rec.709 weights applied to sRGB values.

---

## 2. Colour

### 2.1 The luma → colour ramp

For every image I binned warm pixels by sRGB luma and took the median colour of each bin. The curves are very consistent across photographers and cameras. They fall into two families.

**Vivid family** (consumer JPEG, "punchy"): images 01, 07, 08, 11, 12.

| luma | 0.07 | 0.13 | 0.19 | 0.25 | 0.31 | 0.37 | 0.43 | 0.49 | 0.55 | 0.61 | 0.67 | 0.73 | 0.79 | 0.85 | 0.91 | 0.97 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hex | `#290A03` | `#471803` | `#652503` | `#823203` | `#9D4003` | `#B64D03` | `#CB5C03` | `#DE6D04` | `#F07C04` | `#FC8E05` | `#FFA20A` | `#FFB418` | `#FFCA22` | `#FFDF29` | `#FFED56` | `#FFFF79` |
| hue° | 11 | 19 | 21 | 22 | 24 | 25 | 27 | 29 | 31 | 33 | 37 | 41 | 46 | 51 | 54 | 60 |

**Natural/cinematic family** (neutral profile, less saturation): images 09, 10, 13, 14, 17, 18.

| luma | 0.07 | 0.13 | 0.19 | 0.25 | 0.31 | 0.37 | 0.43 | 0.49 | 0.55 | 0.61 | 0.67 | 0.73 | 0.79 | 0.85 | 0.91 | 0.97 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hex | `#270C06` | `#421908` | `#5C2708` | `#6F350C` | `#844412` | `#9D5318` | `#B5611E` | `#CC6F27` | `#DF7D35` | `#EE8C46` | `#F99D53` | `#FEAF62` | `#FEC27A` | `#FFD595` | `#FFE8B2` | `#FEFCD2` |
| sat | .85 | .89 | .91 | .89 | .87 | .84 | .83 | .81 | .76 | .71 | .67 | .61 | .52 | .42 | .30 | .17 |

Reading the tables:

* Below luma ~0.55 both families are **fully saturated deep orange-red**. The blue channel is essentially 0 in the vivid family, and G/R ≈ 0.25–0.45.
* Above ~0.6 **R is clipped** and only G (and B in the natural family) rises. That is what moves the hue to amber/yellow and desaturates toward cream and white.
* The **natural family desaturates gracefully** (cream highlights `#FFD595`, `#FFE8B2`). The vivid family goes **lemon** (`#FFED56`, `#FFFF79`), which looks "digital" and is a common CG tell.
* **Recommendation for the hero look:** target the natural ramp, or a 70/30 mix toward natural. Never let the highlights end up lemon-green-yellow (hue > 55° with sat > 0.5).

### 2.2 Brightness classes per image (median sRGB)

Classes are defined as: core = all channels ≥ 0.94; inner = luma 0.80–1.0; body = 0.55–0.80; edge = 0.30–0.55; faint = 0.10–0.30. Percentages are of ROI pixels.

| Image | core | inner | body | edge | faint tips/glow |
|---|---|---|---|---|---|
| 01 campfire night | `#FFFFF8` 0.3% | `#FFE865` 5.4% | `#FE9F27` 11.5% | `#C55510` 12.1% | `#6A2105` 23.1% |
| 07 sheets close-up | `#FFFFFB` 0.1% | `#FFF43A` 6.2% | `#FD9C05` 16.1% | `#CA5A03` 23.9% | `#622303` 18.6% |
| 08 log close-up | – | `#FFF238` 2.5% | `#FD9607` 8.7% | `#C95801` 20.7% | `#722101` 31.8% |
| 09 flames1 (natural) | – | `#FFD579` 2.6% | `#EFA35B` 6.0% | `#985A2A` 11.1% | `#5B2613` 23.2% |
| 10 flames3 (natural) | – | `#FFD980` 1.3% | `#E49D59` 4.9% | `#9B5E2D` 13.4% | `#592512` 20.4% |
| 13 fireplace | `#FFFEFC` 9.0% | `#FFECC3` 11.1% | `#F69945` 13.7% | `#AB5D11` 21.0% | `#542B06` 37.9% |
| 14 fireplace ash bed | `#FFFEFB` 2.8% | `#FFE3BA` 11.9% | `#FA9D52` 12.4% | `#B65F1B` 13.7% | `#4F240B` 21.2% |
| 16 lavender root | `#FEFBF7` 0.5% | `#FFDAB9` 5.3% | `#F8A16E` 10.6% | `#B25844` 8.7% | `#4F2108` 13.6% |
| 17 spark burst | `#FEFDF8` 0.4% | `#FFE6B7` 7.3% | `#F99E5F` 8.1% | `#C45722` 10.2% | `#6D2003` 15.2% |
| 18 sheet texture | – | `#FFD494` 6.2% | `#FE9B4C` 33.0% | `#CD5D0F` 30.4% | `#7C2301` 20.0% |
| 25 fireplace embers | `#FFFFFF` 3.6% | `#FFE7A9` 5.4% | `#FF9836` 9.2% | `#E54D0D` 16.4% | `#891304` 29.8% |
| 28 gas torch (dusk) | `#FFFFF9` 8.7% | `#FFFECA` 20.3% | (sky contamination) | `#A35C27` 1.3% | – |
| 15 candle | `#FFFFFF` 28.9% | `#FFF8B5` 16.9% | `#CFAD7F` | `#B75D1C` | `#6D3916` |

**Typical values to hit:** core `#FFFEF8`; inner `#FFE6B0`…`#FFE865`; body `#F99E50`…`#FD9C10`; edge `#C55A14`…`#B65F1B`; faint `#6A2105`…`#542B06`.

**Area rule (measured):** faint + edge ≈ 1.3–5.5× (median ≈ 2×) of (inner + body + core) in campfire and fireplace shots. Per image: 01: 2.0, 07: 1.9, 08: 4.7, 09: 4.0, 10: 5.5, 13: 1.7, 14: 1.3, 16: 1.4, 17: 1.6, 18: 1.3, 25: 2.5. Big, dim, red-orange translucent veils are the bulk of a real flame. The bright parts are thin cores and folds.

### 2.3 Base → tip colour profile

ROI height is normalized: 0 = flame base, 1 = top of the visible flame. "cov" is flame pixel coverage. Bright = median of the top 25% brightest flame pixels in the band.

**01 campfire (night, typical):**

| h | cov | mean luma | median | bright | hue |
|---|---|---|---|---|---|
| 0.04 | 0.11 | 0.57 | `#E87B23` | `#FFDE59` | 28 |
| 0.29 | 0.52 | 0.61 | `#F88A22` | `#FFEE6D` | 29 |
| 0.46 | 0.62 | 0.63 | `#FD9D23` | `#FFE662` | 33 |
| 0.62 | 0.42 | 0.52 | `#EA7317` | `#FFB735` | 26 |
| 0.79 | 0.20 | 0.45 | `#CC5C0F` | `#FE941D` | 25 |
| 0.96 | 0.06 | 0.29 | `#9B3706` | `#A63D07` | 20 |

**08 log close-up:** bright stays yellow `#FEDA22` up to h ≈ 0.4, then falls to `#EA7201` at h 0.88 and `#D75B01` at the top. Hue drops from 33° to 24°.

**11 long exposure:** h 0.04 `#FFD703` (hue 50°) → h 0.38 `#FEA700` → h 0.62 `#9A3C00` (hue 23°). Coverage drops from 92% to 15%.

**09/10 natural:** the brightest band is at h ≈ 0.1–0.2 (`#FFDB84`). Above h 0.55 coverage is under 20% and colour is `#87461A`…`#813E15`.

Rules extracted:

1. **Coverage** peaks at 0.3–0.5 H. Above ~0.6 H the flame breaks into tongues, so coverage falls quickly to ~5% at the top. This is the intermittent zone.
2. **Brightness** peaks low (0.1–0.5 H). Mean flame luma at the top is about 50% of the peak.
3. **Hue drifts redder** by 5–10° from body to tip (≈ 28–33° → 20–24°). **Saturation rises** toward the tips (0.85 → 0.95). Tips are **never** clipped.
4. In the **lowest ~5–10%**, where the flame meets fuel, luma often dips again: the dark fuel-rich zone, plus logs and occlusion. Clipped white is in the 0.1–0.5 band, where sheets fold and overlap.

### 2.4 Hue and saturation statistics (flame pixels)

* **Hue histogram** (5° bins, 0–70°): mass is concentrated at **15–35°**. Medians are 20–36°: 20° for spark-dominated images, 26–29° for flames, 36° for the long exposure. Only 3–14% of pixels are above 45°, and these are the yellow cores.
* **Saturation** for vivid images: 84–88% of flame pixels have sat > 0.9. For natural images the median sat is 0.65–0.75, with the bulk at 0.5–0.8. The candle and white cores pull the median sat down to 0.23.
* **Value**: the distribution is broad and roughly flat from 0.25 to 1. It is not peaked at the top.

### 2.5 Clipping: where and how much

| Image | flame px R clipped | R&G clipped | all 3 clipped | image px any-channel clipped | image px all-3 clipped |
|---|---|---|---|---|---|
| 01 | 0.44 | 0.05 | 0.00 | 9.1% | 0.10% |
| 07 | 0.35 | 0.05 | 0.00 | 12.6% | 0.06% |
| 08 | 0.23 | 0.03 | 0.00 | 3.6% | 0.00% |
| 10 | 0.11 | 0.01 | 0.00 | 1.6% | 0.00% |
| 11 (long exp) | 0.58 | 0.03 | 0.00 | 13.1% | 0.04% |
| 13 fireplace | 0.40 | 0.17 | 0.10 | 2.2% | 0.55% |
| 14 fireplace | 0.47 | 0.07 | 0.04 | 8.1% | 0.71% |
| 15 candle | 0.56 | 0.40 | 0.29 | 7.2% | 1.76% |
| 24 hot campfire | 0.61 | – | – | 7.9% | – |
| 25 embers | 0.46 | 0.11 | 0.07 | 9.4% | 1.44% |
| 28 gas torch | 0.66 | 0.48 | 0.09 | 8.1% | 1.09% |

* **Where:** in the lower-middle flame (0.1–0.5 H), along fold lines where sheets overlap (longer optical path), where flames hug log surfaces, and in the hottest ember crevices. The upper third and the edges never clip.
* **Order:** R first, then G, then B. A "white" core is surrounded by a yellow R+G-clipped ring, which is surrounded by orange with only R clipped.
* **Consequence for the renderer:** compute emission in HDR (linear, unbounded) and apply a **per-channel** (not hue-preserving) filmic curve. This reproduces orange → yellow → white with brightness for free. Hue-preserving tonemappers (e.g. luminance-only Reinhard) keep the hot core orange, which looks fake.

### 2.6 Blue and violet at the base

* **Candle (15):** the classic thin blue rim at the very base and lower edges. A dark, non-luminous zone surrounds the wick and the luminous zone is a hollow shell.
* **Wood fires:** blue/violet appears only where a flame sheet **leaves a wood surface**, with fresh pyrolysis gas and good air, and mostly on dry split wood in fireplaces (14, 16, 25, 13, 08 small root).
  * Measured violet/pink pixels in 16 (crop near the base): **13% of the crop**. Median `#8E527E`, bright `#E5A4CB`, hue 321°. Darkest blue `#221351` (hue 254°).
  * In 13/14 the bright lavender fringes are `#FFD0E9` / `#EEA4C5` (hue 333–335°).
  * In typical campfire shots (01, 07, 09, 10, 11, 17) the violet fraction is **~0%**: hidden by logs, outshone by soot glow, or clipped.
* Physics (literature): blue/violet comes from CH* (431 nm) and C₂* Swan-band (~470–550 nm) chemiluminescence in the thin reaction zone at the flame root, before soot forms. It is a few % of the light and is easily swamped by soot incandescence above.
* **Render spec:** a thin, low-intensity violet-blue (`#3A2A8A`…`#6A4FB0` in linear-ish terms, appearing lavender where added on top of warm emission) **skirt** at the lowest ~5–15% of flame sheets that are attached to logs. It should be visible mainly at the lower *edges* and in darker exposure. It should be **≤ 5% of flame area** in a campfire shot. Large blue regions read as a gas log (`v3` shows that look: almost all blue with yellow tips), not wood.

### 2.7 Blackbody comparison: what temperature the camera colours imply

The blackbody chromaticity in linear sRGB (computed with CIE 1931 CMFs) normalized to max = 1, with relative luminance:

| T (K) | sRGB hex (max-normalized) | rel. luminance vs 1500 K |
|---|---|---|
| 1000 | `#FF2F00` | 0.00035 |
| 1200 | `#FF4C00` | 0.018 |
| 1400 | `#FF6200` | 0.32 |
| 1500 | `#FF6A00` | 1 |
| 1600 | `#FF7200` | 2.8 |
| 1700 | `#FF7A00` | 6.8 |
| 1800 | `#FF8100` | 15 |
| 1900 | `#FF8700` | 31 |
| 2000 | `#FF8D15` | 60 |
| 2200 | `#FF982F` | 185 |
| 2400 | `#FFA141` | 477 |
| 2700 | `#FFAE59` | 1527 |
| 3000 | `#FFB96E` | 3892 |

* Fitting each class median to the nearest blackbody chromaticity gives roughly: faint ≈ 1250–1500 K, edge ≈ 1500–1700 K, body ≈ 2100–2500 K, inner ≈ 3100–4500 K. **This is a tone-mapping artifact.** Soot in wood flames is physically about 1200–1600 K (≈ 1000–1400 °C; literature: Wikipedia *Flame*, American Scientist *Soot*). Yellow and white in photos come from **channel clipping of over-exposed 1400–1700 K emission**, not from 3000 K+ gas.
* **Brightness vs T is extremely steep.** +100 K at 1500 K roughly doubles visible luminance (1500 → 1600 K is ×2.8; 1500 → 1800 K is ×15). Small temperature variations therefore create huge brightness contrast. That is why real flames have bright folds next to dim veils. Use physically steep emission (∝ Planck integral) or an equivalent exponential mapping, not a linear temperature → brightness ramp.
* **Practical renderer mapping:**
  1. Soot temperature 1100–1700 K.
  2. Emission = density × Planck(T) in linear RGB.
  3. Set exposure so ~1550–1650 K reaches display value ~1.0.
  4. Apply a per-channel filmic curve (ACES-like, or a custom shoulder).
  5. Apply partial chromatic adaptation (white point ~2500–3500 K in von Kries/CAT02) if you want the "natural" family. Nguyen/Fedkiw/Jensen 2002 and Pegoraro & Parker 2006 adapt to the fire's own maximum temperature; without adaptation, everything is very orange.
  * Check: the resulting luma→colour mapping should match the table in §2.1.

### 2.8 Embers vs flames vs sparks: different ramps

Median colour per luma bin (aggregated):

| luma | flame (natural) | flame (vivid) | **embers** (25, 29, 19) | **sparks** (03, 04, 05, 22, 23, 27) |
|---|---|---|---|---|
| 0.13 | `#421908` (18°) | `#471803` (19°) | `#6F0E0B` (2°) | `#501504` (13°) |
| 0.25 | `#6F350C` (25°) | `#823203` (22°) | `#AF2213` (6°) | `#8E2E0A` (16°) |
| 0.37 | `#9D5318` (27°) | `#B64D03` (25°) | `#DB421A` (12°) | `#BA4A14` (20°) |
| 0.49 | `#CC6F27` (26°) | `#DE6D04` (29°) | `#F86517` (21°) | `#DA6A2D` (21°) |
| 0.61 | `#EE8C46` (25°) | `#FC8E05` (33°) | `#FE8B23` (29°) | `#F88A44` (23°) |
| 0.73 | `#FEAF62` (30°) | `#FFB418` (41°) | `#FFB04A` (34°) | `#FFAE65` (29°) |
| 0.85 | `#FFD595` (36°) | `#FFDF29` (51°) | `#FFD77A` (42°) | `#FFD58A` (39°) |

* **Embers at low and mid brightness are clearly redder and more crimson** than flames (hue 2–15° vs 18–27°). Their glow is closer to 900–1100 K solid incandescence: a deep red-orange "lava" look, `#AF2213` → `#DB421A` → `#FC7719`. Hot spots go `#FFB04A` → `#FFE8A2`.
* **Dying/cool embers** (20) are crimson to magenta-red, hue ~350°: `#E52342`, `#7B0E1E`. Phone cameras push them pink.
* **Sparks** sit between: hue 13–23°, redder than flame at equal luma. They cool to red aloft (§6).
* Glowing ember bed fraction: in 25, 62% of the ember ROI glows. Ember glow hue histogram for 25: 16% at 0–5°, 45% at 5–15°, 28% at 15–30°. In 29 (late-stage log), 83% is below 10°.

### 2.9 Background darkness and "black"

* Night shots: image median luma **0.008–0.05** (p5 ≈ 0.000–0.004). 50–75% of the frame is essentially black.
* The darkest half of the frame: median `#000000`…`#100300`.
* **Near-black pixels (luma 0.015–0.12) are warm**: `#170601`, `#1F0503`, `#1B0702`, `#250701`, `#270400` (hue 6–21°). This comes from smoke and air glow, lens veiling glare, and lit ground. A CG fire on a pure `#000000` or a neutral grey `#101010` background looks pasted on.
* Fireplaces: walls inside the firebox are `#2E1708`…`#1C0807` (dark-half medians).
* Dusk scenes (06, 28): the ambient light is blue/teal (`#262F34`, sky-lit), the fire is a small clipped warm spot, and the orange/teal complementary contrast is strong. A blue-hour scene with a fire is a strong cinematic option.

---

## 3. Shape and structure

### 3.1 Vocabulary of real flame shapes (observed)

1. **Thin folded sheets** (07, 18, 12, 16). A flame is a thin luminous *surface* (reaction sheet plus soot layer), not a solid blob.
   * Where a sheet folds or is seen edge-on, it is **brighter**: bright limb lines and a "rim" effect, because the line of sight crosses more emitting material. Interiors between folds are darker.
   * You can **see through** flames to logs and walls (08, 12, 13). Opacity is low except in folds.
2. **Tongues / licks.** Vertical lobes 2–6× taller than wide. They taper into thin, often hooked or curled tips that lean with local flow.
   * A typical campfire shows **3–8 major tongues** and many minor ones at any instant (01, 08, 13, 24).
   * Tongues are attached to specific fuel surfaces: log edges, crevices, twig crossings. They are not spread uniformly.
3. **Forking.** Tongues split into 2–3 sub-tongues in the upper half (01, 09, 10, 25), producing a branching, flickering silhouette.
4. **Detached pockets.** Blobs of flame separate from the tongue tips and rise for ~2–6 frames (70–200 ms at 30 fps) while shrinking and reddening, then vanish (09, 10 top, 16 curl, 28 flakes, `v6`/`v7`/`v8`).
   * Measured: pockets are present in **~60% of frames**, at about **7 episodes/s** (§4.3).
5. **Rolling vortices at edges.** Large fires and torches show cauliflower and curl structures that roll *outward-up* along the sides (28 torch, 13, 22 base haze). Small campfires show S-curls and hooks at the tips (16 is a classic "?" curl).
   * Each roll is the visible signature of buoyancy-driven toroidal vortex shedding: the puffing in §4.1.
6. **Necking / pinch-off.** A tongue narrows at 0.5–0.7 H, then the top part separates. This is the origin of pockets.
7. **Flame roots hug the fuel.** Flames emerge as flat sheets *along* log undersides and edges (13, 14, 25). They curve around the log and stand up only once they clear it. Flames on the top of a log are thinner. The strongest flames come from **gaps between logs**: the chimney effect.
8. **Multiple scales.** One or two big central tongues, medium tongues on log ends, and many tiny 1–3 cm flamelets ("candle flames") on char surfaces and embers (19, 25, 26, 08 bottom).
9. **Dark internal zones.** Fuel-rich, not yet luminous gas near the wick or log surface (15 candle dark zone, 09/10 holes). Real flames have **holes and gaps**, not a solid gradient.
10. **Asymmetry.** No real reference is left/right symmetric. The main plume leans by 5–20° from local draft. The fire's silhouette is lopsided because fuel geometry is irregular.

### 3.2 Translucency and layering

* Sheets overlap with partial transparency. Overlapping regions are brighter (additive emission), so overlaps produce the "lattice of bright lines" in 07 and 18.
* Through-flame visibility: in 08 the log behind the main flame shows clearly at roughly 40–60% of its unobstructed brightness plus the flame's emission. Flames are **emissive with low absorption**. Treat them as almost purely emissive volume (absorption σa small, σa/emission ratio low). Use a separate, darker, absorbing smoke and soot volume above.

### 3.3 Edge sharpness (measured)

Edge width is the distance (px, image 1920 px wide) over which luma rises from 0.15 to 0.55 across a flame boundary.

| Image | horizontal edges p10 / median / p90 | vertical edges p10 / median / p90 |
|---|---|---|
| 01 campfire (night) | 2 / 10 / 48 | 2 / 6 / 41 |
| 07 sheets | 6 / 26 / 63 | 5 / 24 / 66 |
| 08 log close-up | 6 / 34 / 62 | 5 / 41 / 75 |
| 09 / 10 natural | 7–8 / 25–27 / 63–65 | 10–13 / 37–38 / 70–72 |
| 13 / 14 fireplace | 2–3 / 14–18 / 55–60 | 2 / 12–13 / 53–58 |
| 16 lavender | 11 / 34 / 65 | 11 / 32 / 71 |
| 18 texture | 10 / 35 / 70 | 10 / 27 / 71 |
| 24 (overexposed) | 1 / 2 / 31 | 2 / 12 / 51 |

* **Bimodal.** Every image has *some* edges as sharp as 2–6 px: sheet folds, the lower and side edges of tongues, bright cores against dark. Most boundaries are soft (median 10–35 px ≈ 0.5–2% of frame width). The softest fades (p90 40–75 px) are the tips and translucent veils.
* Fireplace and campfire overview shots at 1/100+ s have sharper edges than close-ups.
* **CG tells:** uniformly blurry edges (Gaussian-blurred noise) or uniformly hard edges (alpha-cut billboards). Real flames are **sharp on the bottom and sides of tongues and soft at their tips**.

### 3.4 Texture statistics (measured; can be used as render targets)

| Image | radial PSD slope | power(horiz freq) / power(vert freq) | ACF half-width aspect y/x |
|---|---|---|---|
| 01 | −2.49 | 1.68 | 1.16 |
| 07 | −2.67 | 0.74 | 0.97 |
| 08 | −3.82 | 8.52 | 3.12 |
| 09 | −3.16 | 2.23 | 1.88 |
| 10 | −3.18 | 2.43 | 2.46 |
| 11 (long exp) | −3.09 | 8.81 | – |
| 12 | −2.83 | 1.05 | 1.07 |
| 13 | −3.10 | 0.41* | 1.08 |
| 14 | −3.20 | 0.47* | 0.93 |
| 16 | −4.11 | 4.10 | 1.02 |
| 18 | −3.49 | 1.40 | 0.89 |
| sparks 03/04/05 | −1.2 … −1.3 | – | – |

\* Fireplace ROIs include horizontal logs and brick courses, which dominate the anisotropy there.

* **Slope ≈ −3.2 (range −2.5 to −4.1)** is smoother than natural images (≈ −2). Flames are dominated by large smooth gradients and a few sharp edges.
* Mapping to fBm: radial PSD ∝ k^(2·log2(gain) − 2). Slope −3.2 corresponds to **gain ≈ 0.66** per octave (lacunarity 2); −2.5 → 0.84; −4 → 0.5. For the *density/temperature* field this suggests **gain 0.5–0.65 and 4–6 octaves**, with sharpening (e.g. a steep emission curve) to create the edges.
* **Vertical stretch:** wild flames have 1.7–8.8× more power at horizontal frequencies, meaning structures are elongated vertically, with an ACF aspect of 1.2–3. **Stretch noise and turbulence about 1.5–3× vertically** and advect it upward faster with height.
* Spark images have a much flatter spectrum (≈ −1.2), dominated by thin, high-frequency lines.

---

## 4. Motion

### 4.1 Literature

* **Puffing/flicker of buoyant flames:** f ≈ 1.5·D^−0.5 Hz (D = fire diameter in m). Equivalent form: f ≈ 0.48√(g/D) (Cetegen & Ahmed 1993; Xia & Zhang, arXiv 1803.10400). A Loughborough review fits 1.68·D^−0.5.
  * D = 0.3 m → 2.7 Hz; 0.5 m → 2.1 Hz; 1 m → 1.5 Hz; 3 m bonfire → 0.9 Hz.
  * Lab-scale flames and candles: 10–20 Hz, typically 12–15 Hz (Chen et al. 1988; Buckmaster & Peters predicted ~17 Hz).
  * The mechanism is periodic shedding of buoyancy-driven toroidal vortices. Each puff is a vortex ring that rolls up at the flame edge, pinches the flame, and lifts a pocket.
* **McCaffrey (NBSIR 79-1910) three-zone structure** of a fire plume. Height is scaled by Q^(2/5) (Q in kW, z in m). Constants are the standard textbook values (Drysdale; SFPE Handbook).
  * Continuous flame (z/Q^0.4 < 0.08): gas speed u ≈ 6.8·(z/Q^0.4)^0.5·Q^0.2 m/s, rising with height, ΔT ≈ 800–900 K.
  * Intermittent (0.08–0.2): u ≈ 1.9·Q^0.2 m/s, roughly constant. "Pulsating flame".
  * Plume (> 0.2): u ≈ 1.1·(z/Q^0.4)^(−1/3)·Q^0.2, decaying.
  * Example, 50 kW campfire: continuous zone up to ≈ 0.38 m, intermittent up to ≈ 0.96 m, peak gas speed ≈ 4 m/s.
  * **Visible flame features move slower than peak centerline gas, about 1–3 m/s.**
* **Mean flame height** (Heskestad): L = 0.235·Q^0.4 − 1.02·D (m, kW). For Q = 50 kW and D = 0.6 m: L ≈ 0.5 m. For Q = 150 kW and D = 0.8 m: L ≈ 0.9 m.

### 4.2 Measured light-output flicker (what your point light should do)

Linear luminance summed over the fire ROI in night videos `v2`, `v4` and `v7`, at 30 fps. This mixes flame area and brightness, which is exactly what a fire's light output does.

| Video | σ/μ | frame-to-frame abs(ΔI)/I (median / p90) | 1-s window max/min (median) | f50 (half of power below) | f80 | power by band 0–0.5 / 0.5–1 / 1–2 / 2–4 / 4–8 / 8–15 Hz | slope 1–12 Hz |
|---|---|---|---|---|---|---|---|
| v2 fire pit | 0.064 | 0.029 / 0.074 | 1.23 | 1.2 Hz | 4.3 Hz | .40/.07/.18/.13/.15/.07 | −1.3 |
| v4 bonfire | 0.074 | 0.024 / 0.062 | 1.25 | 1.7 Hz | 3.1 Hz | .27/.10/.20/.31/.11/.03 | −2.3 |
| v7 campfire | 0.105 | 0.036 / 0.091 | 1.32 | 0.6 Hz | 3.2 Hz | .48/.13/.10/.16/.10/.03 | −1.8 |

**Light flicker spec.** Intensity I(t) = I₀·(1 + n(t)), where:

* std(n) = **0.06–0.10**;
* the spectrum of n is roughly **1/f^1.3–2.3** between 1 and 12 Hz;
* most energy is **below 4 Hz**, with a slight bump at the puffing frequency (≈ 1–3 Hz);
* per-frame steps at 30 fps are ~3% median and 7–9% p90;
* occasional larger dips and surges (±20–30% over ~0.3–1 s) when a log shifts or a big tongue forms.

Do **not** use a sine or a single noise octave, and do not use white random jitter per frame. Both look wrong.

Use **several (3–5) point/area lights** spread over the flame volume, each with partially independent flicker. That moves the *shadows*, not only the brightness. Real firelight shadows jitter in position because the effective light centroid moves by ~10–30% of the flame width.

### 4.3 Measured flame-shape motion

Three phone and night videos (`v2`, `v6`, `v7`, 30 fps). Units are the ROI height (≈ 1.4–1.7× the mean continuous flame height).

| Metric | v7 | v6 | v2 |
|---|---|---|---|
| main-body top height, std/mean | 0.108/0.61 (18%) | 0.077/0.59 (13%) | 0.065/0.76 (9%) |
| tip rise speed when growing, median (p75) | 1.38 (2.16) ROI-h/s | 0.90 (1.35) | 1.15 (1.85) |
| … in mean flame heights/s | **≈ 2.3** | **≈ 1.5** | **≈ 1.5** |
| fraction of frames tip is rising | 0.54 | 0.56 | 0.52 |
| collapse per frame, median / p90 (ROI-h) | 0.044 / 0.145 | 0.035 / 0.099 | 0.038 / 0.119 |
| frames with a detached pocket | 66% | 58% | 61% |
| pocket episodes per second | 6.8 | 7.4 | 6.9 |
| pocket lifetime, median / p90 | 67 / 193 ms | 67 / 167 ms | 67 / 167 ms |
| pocket rise speed | 1.25 ROI-h/s | 1.84 | 1.96 |
| pixel decorrelation (ACF < 0.5), **upper** flame | 67 ms | – | 67 ms |
| pixel decorrelation (ACF < 0.5), **lower** flame | 300 ms | – | 167 ms |
| 1-frame pixel correlation, upper / lower | 0.68 / 0.82 | – | 0.53 / 0.76 |
| intermittency: P(flame) ≥ 0.95 up to / 0.5 up to / 0.05 up to (approximate: glowing logs are in the ROI) | 0.70 / 0.79 / ≥ 1.0 (ROI top) | – | – |

Interpretation and spec:

* **Tongue life cycle.** A tongue grows upward at ~1.5–2.5 flame heights per second for ~0.2–0.5 s. It necks and pinches off (a 10–15% drop in a single frame at p90), and the released pocket rises at ~2 flame heights/s while shrinking. It is gone in 1–6 frames (50–200 ms). At 30 fps the visible tip region is **mostly new every 2–3 frames**.
* For a 0.5–0.8 m campfire flame, that is **~1–2 m/s apparent rise speed**. The camera views in `v6`/`v7` were angled from above, so true vertical speed is ~1.3× higher. This is consistent with McCaffrey (§4.1).
* **The base is slow and the tip is fast.** The base region decorrelates 2.5–4.5× slower than the top: 170–300 ms vs ~70 ms. The procedural field must advect faster and evolve faster with height. A good rule is to make the noise time-scale ∝ 1/(1 + 2·h/H) and the upward advection speed rise from ~0.5 to ~2 flame heights/s between base and tip.
* **Intermittency.** The flame is "always there" up to ~0.7 of the maximum tip height, present half the time up to ~0.8, and occasional up to 1.0+. In literature terms the intermittent zone spans from ~L/2 to ~1.5–2 L.
* **Flame height** fluctuates with σ ≈ 10–18% of mean height. That fluctuation is a mix of fast pinch-offs and slow (≥ 1 s) surges.
* **What too slow and too fast look like.** Too slow is the most common CG error (forum consensus, realtimevfx): tongues that ooze upward over 1–2 s look like lava lamps or under water. Too fast, or per-frame random, looks like noise flicker. Target: a tip moves **5–15% of the flame height per frame at 30 fps**.

### 4.4 Shutter, frame rate and exposure

| Capture | Appearance | Use for |
|---|---|---|
| **1/1000–1/250 s** (07, 13, 14, 16, 28) | Crisp sheets, razor folds, visible detached flakes, curls, edges 2–6 px. Sparks become dots or very short dashes. | "frozen" beauty shots, slow motion |
| **1/50–1/30 s** (24/30 fps video, 180° shutter) (01, 08, 24, `v6`/`v7`) | Slight vertical streaking (motion blur of ~3–10% of flame height). Sparks are **short straight streaks 5–20 px** at 1080p. | the main realtime look and the cinematic video |
| **1/15–1/4 s** (17, 25, 27) | Tongues smear into vertical veils. Sparks become longer and slightly curved streaks. | moody shots |
| **≥ 1 s long exposure** (03, 04, 05, 11, 22, 23) | Flames become smooth glowing columns (the vertical-streak anisotropy hits 8.8 in 11). Spark trails become **long curly loops**. | stylized "hero" shots, not realism |
| **Slow motion** (120–1000 fps) | Rolling vortices and peeling sheets become legible. Each pocket's life becomes visible: it rises, rolls and dims. Gives a strong "wow" first-scene effect if motion is physically consistent (keep buoyant acceleration and timing consistent when slowed). | 20 s cinematic opener |

For the 20 s video: render at 24 or 30 fps with **true temporal motion blur at a 180° shutter** (1/48 s or 1/60 s). Use sub-frame accumulation of 8–16 samples, or velocity-based blur that includes the volume. Spark streak length should be v·t_exp. A 2 m/s spark at 1/48 s gives a 4 cm streak.

---

## 5. Smoke (at night)

* **Mostly invisible at night.** Smoke is seen only where lit by the fire. It appears as an **orange-brown glowing haze just above and around the flames** (22: a broad orange glow fog around the base; `v5` bonfire: a grey-orange smoke wall drifting sideways at the base, lit warm). It fades to invisible ~1–2 flame heights above the tips, where it leaves the light.
* **Colour.** Lit from below by firelight it is `#5A2A10`…`#9A5A30` (warm). Unlit regions are dark grey-blue, visible only against a moonlit or blue-hour sky (23, 06).
* **Daylight and dusk.** Thin grey-blue wisps (`#948B8C`-ish against the ground in 26), more visible with smouldering wet wood.
* **Behaviour.** Smoke is the continuation of the plume above the flame. It is slower (plume velocity decays as z^(−1/3)), wider, and wanders and leans with draft. It rises in billowy lumps at roughly the puffing frequency.
* **Render spec.** A low-albedo (0.2–0.5) absorbing and scattering volume above the flame that **receives light from the flame**: forward scattering g ≈ 0.3–0.6. Its brightness should be dominated by in-scattered firelight, falling off quickly with height. At normal campfire framing it is a subtle (luma 0.03–0.15) warm haze that softens the flame tips. It also produces the warm "near black" (§2.9). **No smoke = cut-out look. Too much bright smoke = foggy fake look.**

---

## 6. Sparks and embers in flight

Measured from the long-exposure trails in 03, 04, 05, 11, 17, 22, 23 and 27:

| Image | trail width median / p90 (px @1920) | trail colour median | brightest/core | spark hue |
|---|---|---|---|---|
| 03 | 2.0 / 2.3 | `#A1512F` | `#FFE9B5` | 20° |
| 04 | 2.0 / 2.5 | `#A2431D` | `#EC895A` | 18° |
| 05 (defocused) | 3.5 / 9.3 | `#A63706` | `#E36512` | 19° |
| 11 | 2.2 / 3.1 | `#B7550B` | `#FDA921` | 26° |
| 17 burst | 2.2 / 2.8 | `#D96A33` | `#FFE8D4` | 22° |
| 22 bonfire plume | 2.2 / 2.7 | `#833A33` | `#E5A48E` | 9° |
| 23 | 2.2 / 3.2 | `#843A1E` | `#C65C2C` | 17° |
| 27 burst | 2.5 / 4.4 | `#F99869` | `#FFFCBD` | 22° |

* **Size.** Sparks are tiny: sub-mm to few-mm glowing particles. **In focus they render 1–2.5 px wide** at 1080–1920 px. Out-of-focus or near-lens sparks become larger soft streaks or bokeh discs: 05 has median 3.5 px and p90 9.3 px; 27 has p90 4.4 px.
* **Colour.** Deep orange (hue 17–22°). Only fresh or fast sparks close to the fire have pale yellow-white cores. Sparks cool and redden as they rise: in 22 hue drops from 19° to 8–10°; in 23 brightness drops from 0.43 to 0.30 luma and hue from 22° to 15°. Then they wink out.
* **Paths.**
  * Near the fire, in the strong plume, they are **straight and fanning upward** (17, 24, 27, 30).
  * Higher up they become **curly and looping**. Loop radii are ~2–10% of frame height in 04, 05 and 23. Turbulent eddies pick them up, and some sparks even descend.
  * Many trails show **dashed or beaded brightness**: tumbling, non-spherical particles with a glowing side (03 bottom-centre). Use per-spark brightness flicker of roughly 5–30 Hz on 30–50% of sparks.
* **Count and timing.**
  * A steady campfire emits occasional sparks, about **0–5 per second**, mostly from crackling (pops) and log shifts.
  * **Bursts** of dozens to hundreds happen when logs collapse or are stirred (17, 27, `v5`).
  * A big bonfire (`v5`) has a continuous column of hundreds of sparks with a plume ~2–3× the fire height, widening like a cone of ~20–30°.
  * Bursts should be synchronized with **sound** (pops and crackle) and with brief light surges.
* **Lifetime.** Most sparks fade within ~0.5–2 s and rise ~0.5–3 m. A few large firebrands last longer. Firebrand literature notes burnout time scales mainly with particle mass (Fire Technology 2010).
* **Render spec.**
  * Emissive streak particles: streak length = |v|·t_shutter, width 1–2 px plus bloom.
  * Temperature-driven colour from ~1300 K (yellow-orange in camera) down to ~900 K (deep red), then out.
  * Brightness ∝ size², with log-normal size distribution.
  * Motion = plume velocity + curl-noise turbulence that grows with height + slight gravity and drag.
  * Spawn mostly from the ember bed and log cracks. Every distribution parameter should be wide: no uniform sizes, speeds or lifetimes.

---

## 7. Light spill on the surroundings

### 7.1 Measured falloff

| Scene | r = 1.5 r_fire | 2 r | 3 r | 4 r | 5 r | 10 r | colour near → far |
|---|---|---|---|---|---|---|---|
| v6 campfire on ground (night, phone) | 0.42 | 0.16 | 0.015 | 0.007 | 0.004 | – | `#994013` → `#662810` → `#3F1507` → `#0C0302` |
| 30 people around campfire (phone) | 0.48 | 0.24 | 0.16 | 0.07 | 0.04 | 0.026 | `#634531` → `#2F2018` → `#0C090C` |
| 13 fireplace interior/brick | 0.86 | 0.43 | 0.32 | 0.29 | 0.29 | – | `#7F440B` → `#4D2608` → `#3A200B` (enclosed, inter-reflection) |

Values are linear luminance relative to r = 1 fire radius.

* **Open-air campfire.** Light drops to ~50% within 1.5 fire radii and to 15–25% at 2 r. Further out it falls roughly as **r⁻²** (30: 0.24 → 0.04 from 2 r to 5 r, exponent ≈ 1.9).
  * People 1–1.5 m away are lit at 5–15% of the near-fire ground.
  * Beyond ~4–5 m the scene is black except for sky (30, 24).
* **Enclosed fireplace.** Inter-reflection makes a flat, warm bounce (~30% plateau). Brick takes on an amber-brown cast (`#3A200B`…`#7F440B`).
* **Colour of lit surfaces.** Deep orange-red (hue ~15–25°). In camera it is more saturated than the flame body itself, because surfaces are lit mostly by the dominant 1400–1700 K emission without clipping. Neutral materials (grey stone, skin, ash) become orange-brown. Shadow sides are near black.
* **Contact shadows and occlusion.** Logs cast long, jittering shadows radially outward. The ground right next to the fire ring is partly shadowed by stones (`v6`: the sharp drop at 3 r is the stone ring). Faces are lit **from below** (24, 30), giving a strong underlight signature.
* **Temporal.** Spill brightness follows the flicker spec of §4.2 (σ/μ 6–10%). Shadow edges shift as the light centroid moves (multi-light approach).
* **Render spec.**
  * Treat the fire as an area/volume light, or 3–5 point lights distributed over the flame volume at 0.2–0.6 H, with physically correct inverse-square falloff.
  * Use soft shadows: penumbra from a source the size of the flame, 0.3–0.6 m.
  * Add the ember bed as a separate low, red, wide area light (`#C83010`-ish hue, ~20–40% of the flame light), which lights the undersides of nearby logs and stones.
  * Add a tiny ambient (moon/sky) term, ~0.5–3% of fire light, cool blue, so silhouettes read at the frame edges.

### 7.2 Light on the fuel itself

* Log tops and sides facing the flames are lit bright orange. Undersides above the ember bed glow red from the coals.
* Char is very dark (albedo ~0.03–0.06) but has **specular-ish sheen** on cell tops (08, 26).

---

## 8. Heat haze

* No clean still references; this section is estimated from physics and visual observation.
* Refractive index of air: n − 1 = G·ρ with G ≈ 2.26–2.33×10⁻⁴ m³/kg (Gladstone–Dale). This gives ≈ 2.8×10⁻⁴ at 290 K, ≈ 1.4×10⁻⁴ at 580 K, ≈ 5×10⁻⁵ in flame gas.
* Thermal eddies of ~5–10 cm with ΔT ~ 300 K across a ~0.5 m path deflect rays by ~1 mrad. A background 2–4 m behind the fire is displaced by **~2–4 mm**, which at typical framing is **~1–5 px at 1080p**. Displacements grow just above the flame tips, where gradients are largest.
* **Look:** a shimmering, upward-rippling distortion of the background in a column ~1–1.5× the fire width. It is strongest from the flame tips up to ~2 flame heights, then decays. It evolves at **5–20 Hz** and travels upward at plume speed (~1–2 m/s). It is invisible against pure black and shows clearly against lit backgrounds: brick, trees, stars, people across the fire.
* **Render spec:** a screen-space UV offset from upward-advected, vertically stretched high-frequency noise. Amplitude ≤ 3–6 px at 1080p, masked to the plume region and only where there is background detail. Keep it **extremely subtle** (realtimevfx: "extremely subtle"). Overdone haze is a classic fake tell.

---

## 9. Fuel bed: logs, char, embers, ash

* **Fresh wood** (26): pale tan `#C8A878`-ish end grain. Bark is dark brown. Fresh wood blackens progressively from the flame side.
* **Char:**
  * A dull black layer, slightly blue-grey where ash forms.
  * It cracks into **"alligator" cells**: roughly rectangular blocks elongated along the grain, with deep transverse cracks (08, 26, 29). Cracks preferentially follow rays and growth-ring boundaries (char-cracking literature, arXiv 1604.01249).
  * Cell size observed is ~1–3 cm on campfire logs, ≈ 5–15% of log diameter, with larger cells on hotter or longer-burning logs. Cells are convex and slightly domed. Edges are rounded by burning.
* **Glow pattern on burning logs:**
  * *Active stage* (08, 25, 16): dark cell tops with **thin, bright orange-red glowing lines along crack edges and log undersides**. Colour `#D84010`…`#FF6A20`, hot spots yellow. Glowing ember dots and flakes sit scattered on char (16: `#E04018`-ish speckles).
  * *Late stage* (29): the whole surface glows salmon-red `#E06050`…`#FF865D`, with **darker cracks** (the inverse of the active stage) and yellow-orange in deep crevices.
  * Dying (20): dull crimson patches `#E52342` → `#7B0E1E`.
  * Thin twigs glow along their whole length as red filaments (01: curly glowing twig `#E04010`).
* **Ember/coal bed** (25, 16, 17, 19):
  * Granular 1–5 cm chunks. Glowing interiors, `#FC7719` to `#FFB04A`, hot spots `#FFE8A2`.
  * Darker crusts and edges.
  * The glow is strongest in **crevices and undersides** (cavities act like blackbody cavities).
  * The surface pattern shimmers slowly: glow intensity pulses on ~0.5–3 s scales as air flows over it.
  * Tiny blue flamelets (CO burning) float just above hot coals (19, 25, 14).
* **Ash:**
  * Powdery white-grey coat on burnt surfaces and fluffy white clumps on the bed (14, 26).
  * Daylight colour `#948B8C` / `#6E7075` (slightly blue-grey). Under firelight it looks cream or warm white and often clips (25/16: `#FFFFFD`).
  * Ash is the brightest *reflective* material in the scene. Use albedo ~0.5–0.7 versus char ~0.04.
  * Charcoal ash crust with glowing veins (21): grey crust, orange-red veins `#FF4020` with yellow centres.

---

## 10. What makes CG fire look fake (and the measured real counterpart)

| # | Fake tell | What real fire shows (refs, measurement) |
|---|---|---|
| 1 | **Uniform smooth noise:** fBm "lava lamp" blobs with the same softness everywhere | Thin folded sheets, bimodal edges (2–6 px sharp folds plus soft tips), PSD slope ≈ −3.2, holes and gaps (§3) |
| 2 | **Saturated orange everywhere / no white-hot core** (hue-preserving tonemap, LDR gradient texture) | R clips in ~40% of flame pixels and goes to amber → yellow → cream → white with brightness. Core is < 5% of the flame but present (§2.1, §2.5) |
| 3 | **Lemon or greenish-yellow highlights** | Highlights end cream `#FFE8B2` / `#FEFCD2` (natural) or at worst `#FFED56`. Never hue > 60° |
| 4 | **Too much bright area**, a solid glowing blob | Dim translucent orange-red veils (luma 0.1–0.55) cover 1.5–3× the bright area |
| 5 | **Obviously scrolling or looping noise**: texture slides upward rigidly, or a visible loop period | Shapes are born, stretch, neck, pinch off and die. Tips are new every 2–3 frames; the base evolves 3–4× slower (§4.3). No repetition |
| 6 | **Too slow** (most common), or per-frame random jitter | Tips rise ~1.5–2.5 flame heights/s. Pockets live 50–200 ms. Light flicker is 1/f-ish below 4 Hz with ~3% per-frame steps |
| 7 | **Billboards / camera-facing sprites:** flatness on camera moves, intersection lines with logs, identical silhouettes | Volumetric sheets wrapping around logs. Parallax between near and far tongues. Flames emerge from specific fuel surfaces and gaps |
| 8 | **Symmetry and a cone-shaped "fireball" silhouette** | Lopsided, multi-tongued, leaning 5–20°. 3–8 major tongues of different heights |
| 9 | **No detachment:** a continuous flame column with smooth tops | Detached pockets in ~60% of frames, ~7 per second, plus forking and necking |
| 10 | **Opaque flames** | You can see logs and walls through them. Brightness adds where sheets overlap |
| 11 | **No blue base, or too much blue** | Faint violet skirt at the roots (≤ 5% of area). Gas-log look if overdone |
| 12 | **Flat lighting:** a constant point light, no flicker, no moving shadows | Light σ/μ 6–10%, multi-centroid shadows that move, inverse-square ground pool, faces lit from below, ember-bed red underlight |
| 13 | **No smoke / glow haze** (flame on pure black) | Warm near-black `#150503`…`#270400` around the fire, faint lit smoke haze above tips |
| 14 | **Sparks too uniform**: same size, speed, colour and lifetime, emitted at a constant rate, straight-line paths, round dots | Bursty emission synced with crackles, log-normal sizes, cooling colour, straight then curly paths, streak motion blur, tumbling flicker |
| 15 | **Sparks too big, bright or white** | 1–2.5 px streaks, deep orange `#A2431D`…`#D96A33`, pale only when fresh |
| 16 | **Wrong fuel bed:** brown textured logs with orange paint, no char cells, no ash, no ember glow | Black alligator char with glowing crack rims, white ash, granular glowing coal bed brightest in crevices |
| 17 | **Temperature ramp linear in brightness** | Emission ∝ Planck: ×15 luminance from 1500 → 1800 K. Steep contrast between folds and veils |
| 18 | **No motion blur** on fast features in video | 180° shutter: slight vertical smearing of tongues and spark streaks |
| 19 | **Heat haze overdone** (wobbly glass) | 1–5 px shimmer, only above the flames and only where background detail exists |
| 20 | **Bloom as a crutch:** huge soft glow hiding the flame | Real lens glare is a mild veil (near-black lifts to luma ~0.02–0.05), not a fog. Flame edges stay readable |
| 21 | **Scale cues wrong** (flames too big or too slow for the logs) | Campfire flames 0.3–1 m, puff 1.5–3 Hz. Bonfire flames 2–6 m, puff ~1 Hz. Timing must match size (f ∝ D^−0.5) |
| 22 | **Colour constant along height** | Tips 5–10° redder and darker (luma ~50%), coverage falls from ~0.6 to ~0.05 at the top |

---

## 11. Realism checklist for scoring screenshots and clips

Score each item 0 (absent or wrong), 1 (partly) or 2 (convincing). **≥ 38/50 is "photoreal" territory; < 25 reads as CG.** Items marked (V) need a video clip.

**Colour and tone**
1. Flame pixels follow the natural luma→colour ramp of §2.1 to within ±1 bin: deep red `#421908` at luma 0.13, orange `#B5611E` at 0.43, `#FEC27A` at 0.79.
2. White-hot clipped core exists but is small (all-3-clipped < 5% of flame pixels). R is clipped in roughly 20–60% of flame pixels.
3. Highlights roll off to cream or pale yellow, not lemon or green (no saturated pixels with hue > 55°).
4. Dim translucent orange-red regions (luma 0.10–0.55) cover at least ~1.3×, typically ~2×, the area of bright regions (> 0.55).
5. Base → tip: brightest at 0.1–0.5 H. Tips darker (≈ 50% luma) and 5–10° redder. Coverage at the top under 15%.
6. Subtle violet/blue at flame roots on wood (visible on zoom), under 5% of flame area, not a blue "gas" flame.
7. Background black is warm-tinted near the fire (near-black hue 5–25°), not neutral, with no visible banding.

**Shape**
8. Flame reads as thin folded sheets with brighter fold lines; partially see-through (logs visible through it).
9. Edges are bimodal: some 2–6 px sharp boundaries (sides and bottoms of tongues) plus soft tips. Not uniformly blurred.
10. 3–8 distinct tongues of different heights, forking in the upper half. The silhouette is asymmetric and leans.
11. At least one detached pocket or flake above the main body (in ~60% of random frames).
12. Holes, gaps and dark interior zones exist. It is not a solid gradient blob.
13. Flames emerge from specific fuel features (log gaps, undersides, char cracks) and wrap around logs.
14. Multiple scales: a big central plume, medium tongues and tiny flamelets on char and embers.
15. Texture is vertically elongated (ACF aspect ~1.2–3). Running `analyze.py` gives a PSD slope of −2.5 … −4.

**Motion (V)**
16. Tip region changes substantially every 2–3 frames at 30 fps. Tongues grow at ~1.5–2.5 flame heights/s and pinch off.
17. The base evolves visibly slower than the tips. No rigid upward scrolling, no visible loop in 20 s.
18. Puffing rhythm matches size (≈ 1.5/√D Hz), with irregular height fluctuation σ ≈ 10–18%.
19. Proper motion blur (180° shutter) on tongues and sparks.

**Light**
20. Scene light flickers with σ/μ ≈ 6–10%, 1/f-like, mostly < 4 Hz; no strobing; occasional slower surges (V).
21. Shadows from logs, stones and people move and jitter (multi-centroid source) and have soft penumbrae.
22. Ground light pool falls to ~50% at 1.5 fire radii and ~r⁻² beyond. Lit surfaces are saturated orange. Undersides of nearby logs are lit red by embers.

**Particles and atmosphere**
23. Sparks are 1–2.5 px streaks, deep orange, with varied sizes and lifetimes. Straight near the fire and curly above. Some tumbling flicker. They cool to red and wink out.
24. Spark emission is bursty (a few per second baseline, bursts on log collapse or crackle), synchronized with audio pops (V).
25. Faint lit smoke and haze above the flames softens the tips and fades within 1–2 flame heights. Heat shimmer is subtle (≤ 5 px) and only where there is background detail.

**Fuel bed (bonus check, folded into the items above when scoring)**
* Black alligator-cracked char with glowing crack rims.
* Granular ember bed brightest in crevices (`#FC7719` → `#FFB04A`).
* White-grey ash.
* Fresh wood distinctly paler than char.

(25 items × 2 = 50 points.)

---

## 12. Best reference images for side-by-side comparison

All are in `C:\Users\User\AppData\Local\Temp\claude\D--fire\bc550da3-92c2-4cba-b445-153d7412a165\scratchpad\refs\`. A labelled contact sheet of the eight is at `...\scratchpad\refs_crops\best_refs_contact.jpg`.

1. `08_closeup_of_fire.jpg`: campfire log at night. Alligator char with glowing crack rims, translucent flame (log visible through it), base-to-tip colour drift, tiny blue root, spark streaks. **Best all-round campfire target.**
2. `25_knisternd.jpg`: fireplace log on a glowing ember bed. Cracked glowing log underside, white-hot flame with violet edge, curly spark trails, warm-lit back wall. **Best embers, fuel bed and sparks target.**
3. `14_fireplace_rostov.jpg`: logs on a white ash bed. White clipped cores, orange tips, lavender flame roots, light spill on the firebox. **Best natural-colour and clipping target.**
4. `01_campfire_flames_night.jpg`: brushwood campfire at night. Many tongues and sheets behind silhouetted twigs, glowing twig filaments, warm near-black. **Best wide-campfire composition and colour-ramp target.**
5. `10_flames3.jpg` (or the twin `09_flames1.jpg`): neutral-colour wispy tongues with forking, detached pockets at the top and faint violet roots. **Best shape and detachment target.**
6. `16_flames_52633350622.jpg`: close-up of a curling flame with a lavender/blue root sheet, detached curl at the tip, glowing coal bed under a grate. **Best blue-base and curl target.**
7. `17_campfire_15188333373.jpg`: stirred campfire with a fan-shaped spark burst, glowing coal bed, grass lit red. **Best spark-burst and light-spill target.**
8. `22_swirling_sparks_hilsenhain.jpg`: bonfire spark plume (long exposure), smoke haze glowing orange above the flames, sparks reddening with height. **Best bonfire, smoke and spark-plume target.**

Secondary: `07` (sheet folds), `13` (wide fireplace and light spill), `18` (overlapping sheets and limb brightening), `28` (rolling vortices on a big gas flame), `29` (late-stage glowing log), `21` (ash crust with glowing veins), `30` (people lit by the fire, falloff), `03`/`04` (spark trail curl statistics).

---

## 13. Sources

Image and video sources are the Wikimedia Commons file pages named in §1.1 (search API: commons.wikimedia.org/w/api.php). Literature:

* Puffing frequency f ∝ D^−1/2 (≈ 1.5): [Xia & Zhang, *Flickering of buoyant diffusion flames: a vortex dynamics revisit*, arXiv 1803.10400](https://arxiv.org/pdf/1803.10400); [Loughborough review of pool-fire pulsation (f = 1.68 D^−0.5)](https://repository.lboro.ac.uk/articles/journal_contribution/A_review_of_research_and_experimental_study_on_the_pulsation_of_buoyant_diffusion_flames_and_pool_fires/9547682); [Xia, *Vortex-dynamical scaling theory*, PolyU](https://ira.lib.polyu.edu.hk/bitstream/10397/106470/1/Xia_Vortex-Dynamical_Scaling_Theory.pdf); [USTC SKLFS abstract (cm-pools puff ~10 Hz)](https://en.sklfs.ustc.edu.cn/2016/0301/c5760a374423/page.htm)
* Candle / lab flame flicker 10–20 Hz: [Diffusion-flame flickering as a hydrodynamic global mode, arXiv 1912.03642](https://arxiv.org/pdf/1912.03642); [AIChE 2010 abstract](https://proceedings.aiche.org/conferences/aiche-annual-meeting/2010/proceeding/paper/469h-control-flame-instabilities-oscillating-electric-fields); [IIT Madras burner study](https://publications.iitm.ac.in/publication/effects-of-burner-configurations-on-the-natural-oscillation/pdf/publisher-pdf-fulltext-effects-of-burner-configurations-on-the-natural-oscillation.pdf); [KAIST record (Buckmaster & Peters ~17 Hz)](https://dspace.kaist.ac.kr/handle/10203/19409)
* McCaffrey plume regimes: [McCaffrey, NBSIR 79-1910](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nbsir79-1910.pdf); [NIST FireDOC](https://firedoc.nist.gov/article/qHcxXYQBWEcjUZEYxRPV). Coefficients 6.8 / 1.9 / 1.1 and boundaries 0.08 / 0.2 are the standard textbook values (Drysdale; SFPE Handbook).
* Heskestad flame height: [calculator.academy summary](https://calculator.academy/flame-height-calculator/); [AIDIC paper](https://www.aidic.it/cet/12/26/061.pdf)
* Soot colour and temperature: [Wikipedia – Flame](https://en.wikipedia.org/wiki/Flame); [Wikipedia – Luminous flame](https://en.wikipedia.org/wiki/Luminous_flame); [American Scientist – Soot: Giver and Taker of Light](https://www.americanscientist.org/node/3267)
* Blue chemiluminescence CH* / C₂*: [Univ. Duisburg-Essen chemiluminescence](https://www.uni-due.de/ivg/vg/chemilumineszenz_en.php); [colorSpec "blue flame" vignette](https://www.vps.fmvz.usp.br/CRAN/web/packages/colorSpec/vignettes/blueflame.pdf); [NASA NTRS diffusion-flame chemiluminescence](https://ntrs.nasa.gov/api/citations/20020027537/downloads/20020027537.pdf)
* Fire rendering with blackbody, blue core and chromatic adaptation: [Nguyen, Fedkiw, Jensen 2002, *Physically Based Modeling and Animation of Fire*](https://graphics.ucsd.edu/~henrik/papers/fire/); [FieryGS 2026 (CAT02 adaptation following Nguyen 2002 / Pegoraro & Parker 2006)](https://arxiv.org/pdf/2605.00177); [Wikipedia – Chromatic adaptation](https://en.wikipedia.org/wiki/Chromatic_adaptation)
* Real-time VFX fire practice (speed, blackbody, translucency, subtle distortion): [realtimevfx.com thread](https://realtimevfx.com/t/looking-for-some-tips-for-creating-realistic-fire-effects-in-real-time-vfx/28143); [Boing Boing – why CG fire looks terrible](https://boingboing.net/2023/10/29/why-cg-fire-looks-terrible-even-in-high-budget-movies.html); [vitrina.ai VFX fire techniques](https://vitrina.ai/blog/realistic-fire-explosion-vfx-simulation-techniques/)
* Fire photography (shutter effects): [Canon – tips for fire photography](https://www.canon.co.uk/get-inspired/tips-and-techniques/tips-for-fire-photography/); [Digital Photography School](https://digital-photography-school.com/hot-images-fire/); [Expedition Portal](https://expeditionportal.com/how-to-photograph-fire/)
* Heat haze (Gladstone–Dale): [Harvard schlieren demo](https://sciencedemonstrations.fas.harvard.edu/presentations/schlieren-optics); [COMSOL schlieren blog](https://www.comsol.com/blogs/creating-schlieren-type-visualizations-in-comsol-multiphysics); [BOS tomography arXiv 2509.14442](https://arxiv.org/pdf/2509.14442)
* Char cracking: [thermo-mechanical explanation of char crack patterns, arXiv 1604.01249](https://arxiv.org/pdf/1604.01249); [beech char cracking study](https://doc.rero.ch/record/321406/files/226_2013_Article_545.pdf)
* Firebrands and embers: [Fire Technology 2010 firebrand burnout](https://link.springer.com/article/10.1007/s10694-010-0181-x); [NIST emberometer](https://www.nist.gov/system/files/documents/2022/08/18/NIST%20WUI%20FIRE%20DAYS%202022%20-%203.2%20-%20NIST%20Emberometer%20Research.pdf)
