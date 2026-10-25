// Entry point: real-time viewer + deterministic API used by the test/capture tools.
import { initGPU, Profiler } from './gpu.js';
import { FireSim } from './sim.js';
import { Renderer } from './renderer.js';
import { Sparks } from './sparks.js';
import { makeNoiseTexture, makeCurlTexture } from './noise.js';
import { buildCampfire, packLogs, packStones } from './scene.js';
import { orbit, lookAt } from './camera.js';
import * as TL from './timeline.js';

const Q = new URLSearchParams(location.search);
const num = (k, d) => (Q.has(k) ? Number(Q.get(k)) : d);
const TEST = Q.has('test');

const PRESETS = {
  low: { res: [64, 96, 64], vcycles: 1, surfScale: 0.5, volScale: 0.5, stepMul: 1.0, dt: 1 / 60 },
  med: { res: [80, 120, 80], vcycles: 1, surfScale: 0.6, volScale: 0.5, stepMul: 0.8, dt: 1 / 75 },
  high: { res: [96, 144, 96], vcycles: 2, surfScale: 0.75, volScale: 0.6, stepMul: 0.7, dt: 1 / 90 },
  ultra: { res: [128, 192, 128], vcycles: 3, volScale: 1.0, stepMul: 0.5, dt: 1 / 120 },
  hero: { res: [160, 240, 160], vcycles: 3, volScale: 1.0, stepMul: 0.45, dt: 1 / 240 },
  // finer grid for a large GPU (5.4 mm cells, ~17 M cells, ~2.5 GB): offline only
  cine: { res: [224, 336, 224], vcycles: 3, volScale: 1.0, stepMul: 0.45, dt: 1 / 360 },
};

async function main() {
  const canvas = document.getElementById('c');
  const { device, features } = await initGPU();
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });

  const preset = PRESETS[Q.get('q') || (TEST ? 'high' : 'med')];
  const res = Q.has('res') ? [num('res', 80), num('res', 80) * 1.5, num('res', 80)] : preset.res;
  const scene = buildCampfire(num('seed', 7));
  const logsF32 = packLogs(scene.logs);
  const noiseTex = makeNoiseTexture(device, 128, 1);
  const curlTex = makeCurlTexture(device, noiseTex, 128);
  const sim = new FireSim(device, { res, size: [1.2, 1.8, 1.2], logs: logsF32, vcycles: num('vc', preset.vcycles), noiseTex, curlTex, dbg: num('dbg', 0) });
  const renderer = new Renderer(device, { sim, scene, noiseTex, curlTex, logsF32, stonesF32: packStones(scene.stones), format });
  renderer.params.stepMul = num('step', preset.stepMul);
  renderer.surfScale = num('ss', TEST ? 1 : (preset.surfScale ?? 1));
  const sparks = new Sparks(device, { count: num('sparks', 8192), sim, renderer, curlTex });
  const prof = new Profiler(device, Q.has('prof'));
  sim.profileStages = prof.enabled;

  const state = {
    yaw: num('yaw', 0.35), pitch: num('pitch', 0.12), dist: num('dist', 2.2), ty: num('ty', 0.5), fov: num('fov', 38),
    paused: false, simDt: num('dt', TEST ? 1 / 90 : preset.dt), volScale: num('vs', preset.volScale),
  };

  function size() {
    const dpr = TEST ? 1 : Math.min(devicePixelRatio, 1.5);
    const w = TEST ? num('w', 960) : Math.floor(canvas.clientWidth * dpr);
    const h = TEST ? num('h', 1080) : Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    renderer.resize(w, h, state.volScale);
  }
  if (TEST) {
    canvas.style.width = num('w', 960) + 'px'; canvas.style.height = num('h', 1080) + 'px';
    document.querySelectorAll('#start, #panel, #hud').forEach(e => e.remove());
  }
  size();

  function camera() {
    if (state.camT !== undefined && state.camT !== null) { const c = TL.cameraAt(state.camT); return lookAt(c.pos, c.target, c.fov); }
    if (state.cam) return lookAt(state.cam.pos, state.cam.target, state.cam.fov || 40);
    return orbit({ yaw: state.yaw, pitch: state.pitch, dist: state.dist, target: [0, state.ty, 0], fov: state.fov });
  }

  function simStep(dt) {
    const enc = device.createCommandEncoder();
    sim.step(enc, dt, prof);
    sparks.rate = 1.0 * (sim.params.intensity ?? 1);   // a calm fire only drizzles sparks; bursts carry the show
    sparks.step(enc, dt);
    device.queue.submit([enc.finish()]);
  }
  // Scripted/random events -> spark bursts (+ sim bursts are handled by the timeline params).
  // Resin pops throw a handful of sparks; hero pops and log shifts throw a fan of 100-200 that clears the flame.
  function fireEvent(e) {
    const s = e.strength ?? 1;
    const pos = e.pos || [0, 0.12, 0];
    if (e.type === 'ignite') sparks.burst({ n: 140, pos, speed: 3.2, radius: 0.12, dur: 0.5 });
    else if (e.type === 'pop' && s >= 0.8) sparks.burst({ n: Math.round(110 * s), pos, speed: 3.5 + 2.5 * s, radius: 0.03, dur: 0.05, fan: 0.9, size: 1.4, aim: e.aim || null });
    else if (e.type === 'pop') sparks.burst({ n: Math.round(3 + 12 * s), pos, speed: 3 + 3 * s, radius: 0.02, dur: 0.04, fan: 0.8, size: 1.4 });
    else if (e.type === 'log-shift') sparks.burst({ n: Math.round(190 * s), pos, speed: 4.5, radius: 0.12, dur: 0.15, fan: 0.7, size: 1.5 });
    else if (e.type === 'crackle-burst') sparks.burst({ n: Math.round(12 * s), pos, speed: 3, radius: 0.08, dur: 0.3, fan: 0.6 });
  }

  let frameNo = 0;
  const halton = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };
  // Real-time frame: jittered render, sparks, temporal accumulation (AA + noise), bloom, tone map.
  function draw(taWeight = 0.4) {
    size();
    renderer.setCamera(camera());
    const j = (frameNo % 16) + 1;
    renderer.uploadParams(sim.time, frameNo++, TEST ? [0, 0] : [halton(j, 2) - 0.5, halton(j, 3) - 0.5]);
    const enc = device.createCommandEncoder();
    prof.begin();
    renderer.renderHDR(enc, prof);
    sparks.draw(enc, 1 / 120);
    const ta = !TEST;
    if (ta) renderer.temporal(enc, taWeight);
    renderer.bloom(enc, ta, prof);
    renderer.final(enc, ctx.getCurrentTexture().createView(), format, ta);
    if (!TEST) renderer.readLights(enc);
    prof.end(enc);
    device.queue.submit([enc.finish()]);
    prof.collect();
  }
  // Supersampled still: n jittered renders of the current sim state, averaged before bloom/tonemap.
  function drawAccum(n, view = null, fmt = format) {
    size();
    renderer.setCamera(camera());
    for (let i = 0; i < n; i++) {
      renderer.uploadParams(sim.time, frameNo++, [halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5]);
      const enc = device.createCommandEncoder();
      renderer.renderHDR(enc);
      sparks.draw(enc, 1 / 120);
      renderer.accumulate(enc);
      device.queue.submit([enc.finish()]);
    }
    const enc = device.createCommandEncoder();
    renderer.resolveAccum(enc, n);
    renderer.bloom(enc, true);
    renderer.final(enc, view || ctx.getCurrentTexture().createView(), fmt, true);
    device.queue.submit([enc.finish()]);
  }

  // ---------- offline capture (timeline-driven, motion-blurred sub-frames) ----------
  const cap = { on: false };
  function capCamera(t) { const c = TL.cameraAt(t); return lookAt(c.pos, c.target, c.fov); }
  // Moving logs (log shift): rewrite the log buffers shared by the sim and the renderer, rebuild the emitter.
  const baseLogs = logsF32.slice();
  let logKey = '';
  function applyLogs(o) {
    const key = o ? `${o.index}:${o.s.toFixed(3)}` : '';
    if (key === logKey) return;
    logKey = key;
    const arr = baseLogs.slice();
    if (o) {
      const i = o.index * 8;
      const ox = arr[i], oz = arr[i + 2], l = Math.hypot(ox, oz) || 1;
      const dx = ox / l, dz = oz / l;
      arr[i] += dx * o.footOut * o.s; arr[i + 2] += dz * o.footOut * o.s; arr[i + 1] -= 0.004 * o.s;
      arr[i + 4] += dx * o.topOut * o.s; arr[i + 6] += dz * o.topOut * o.s; arr[i + 5] -= o.topDrop * o.s;
    }
    device.queue.writeBuffer(sim.logBuf, 0, arr);
    device.queue.writeBuffer(renderer.logBuf, 0, arr);
    sim.geoReady = false;
  }
  // Capture time is driven by one integer step index j (t = j / (fps * n)), so adjacent event windows are
  // computed by the same function and can never overlap or leave gaps; a frame is exactly n steps.
  function capStep(j) {
    const t = cap.stepT(j), tNext = cap.stepT(j + 1);
    sim.setParams(TL.simAt(t));
    applyLogs(TL.logOffsetsAt(t));
    for (const e of TL.EVENTS) if (e.t >= t && e.t < tNext) fireEvent(e);
    simStep(cap.simDt);
    cap.t = tNext;
  }
  async function captureBegin({ w = 1080, h = 1920, simDt = 1 / 240, shutter = 0.5, aa = 2, preroll = TL.PREROLL, dof = false } = {}) {
    cap.dof = dof;
    canvas.width = w; canvas.height = h;
    renderer.resize(w, h, state.volScale);
    cap.w = w; cap.h = h; cap.shutter = shutter; cap.aa = aa;
    cap.n = Math.max(1, Math.round(1 / (TL.FPS * simDt)));
    cap.simDt = 1 / (TL.FPS * cap.n);                   // snapped so a frame is exactly n steps
    cap.stepT = (j) => j / (TL.FPS * cap.n);
    cap.baseExposure = renderer.params.exposure;
    cap.bpr = Math.ceil(w * 4 / 256) * 256;
    cap.tex = device.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    cap.buf = device.createBuffer({ size: cap.bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    cap.stats = [];
    sim.reset();
    sparks.reset();
    applyLogs(null);
    const P = Math.round(preroll * TL.FPS) * cap.n;     // preroll in whole steps
    for (let j = -P; j < 0; j++) { capStep(j); if ((j + P) % 30 === 29) await device.queue.onSubmittedWorkDone(); }
    cap.t = 0;
    await device.queue.onSubmittedWorkDone();
    cap.on = true;
  }
  // Renders frame i (time i/FPS): steps the sim through the frame interval, accumulating jittered renders
  // over the open-shutter part, then reads back RGBA8 (row padding removed).
  // skip = true advances the sim (and events) through the frame without rendering it.
  async function captureFrame(i, fps = TL.FPS, skip = false) {
    const t0 = i / fps;
    const n = cap.n;
    const open = skip ? 0 : Math.max(1, Math.round(n * cap.shutter));
    let samples = 0;
    for (let k = 0; k < n; k++) {
      const t = cap.stepT(i * n + k);
      if (k < open) {
        for (let a = 0; a < cap.aa; a++) {
          const j = samples + 1 + i * 7;
          renderer.setCamera(capCamera(t + (a + 0.5) / cap.aa * cap.simDt));
          if (cap.dof) {
            // thin lens: concentric-disk sample of the aperture (R2 sequence), focus on the camera target
            const L = TL.lensAt(t);
            const u = (0.5 + j * 0.7548776662) % 1, v = (0.5 + j * 0.5698402910) % 1;
            const ap = L.aperture * (cap.apScale ?? 1);
            const r = Math.sqrt(u) * ap, ph = v * Math.PI * 2;
            Object.assign(renderer.params, { lensX: r * Math.cos(ph), lensY: r * Math.sin(ph), focus: L.focus, aperture: ap });
          }
          renderer.params.exposure = cap.baseExposure * TL.exposureAt(t);
          renderer.uploadParams(sim.time, frameNo++, [halton(j, 2) - 0.5, halton(j, 3) - 0.5]);
          const enc = device.createCommandEncoder();
          renderer.renderHDR(enc);
          sparks.draw(enc, cap.simDt);   // each sample shows the whole sub-step's motion (energy-normalised)
          renderer.accumulate(enc);
          device.queue.submit([enc.finish()]);
          samples++;
        }
      }
      capStep(i * n + k);
      if (k % 4 === 3) await device.queue.onSubmittedWorkDone();
    }
    if (skip) {
      await device.queue.onSubmittedWorkDone();
      return null;
    }
    const enc = device.createCommandEncoder();
    renderer.resolveAccum(enc, samples);
    renderer.bloom(enc, true);
    renderer.final(enc, cap.tex.createView(), 'rgba8unorm', true);
    enc.copyTextureToBuffer({ texture: cap.tex }, { buffer: cap.buf, bytesPerRow: cap.bpr }, [cap.w, cap.h]);
    renderer.readLights(enc);
    device.queue.submit([enc.finish()]);
    const ls = await renderer.collectLights();
    const cam = TL.cameraAt(t0);
    cap.stats.push({ frame: i, t: t0, intensity: sim.params.intensity, light: ls.L, lightY: ls.y,
      camDist: Math.hypot(cam.pos[0], cam.pos[1] - 0.35, cam.pos[2]), camAz: Math.atan2(cam.pos[0], cam.pos[2]) });
    await cap.buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(cap.buf.getMappedRange());
    const out = new Uint8Array(cap.w * cap.h * 4);
    for (let y = 0; y < cap.h; y++) out.set(src.subarray(y * cap.bpr, y * cap.bpr + cap.w * 4), y * cap.w * 4);
    cap.buf.unmap();
    return out;
  }

  // ---------- deterministic API ----------
  window.FIRE = {
    device, sim, renderer, state, features: [...features],
    async step(n = 1, dt = state.simDt) { for (let i = 0; i < n; i++) simStep(dt); await device.queue.onSubmittedWorkDone(); },
    async draw() { draw(); await device.queue.onSubmittedWorkDone(); },
    async drawAccum(n = 16) { drawAccum(n); await device.queue.onSubmittedWorkDone(); },
    captureBegin,
    // Renders frame i and POSTs the raw RGBA to the capture server.
    async captureFrame(i, url = '/frame') {
      const px = await captureFrame(i);
      const r = await fetch(`${url}?i=${i}`, { method: 'POST', body: px });
      return r.status;
    },
    // Advances the capture timeline through frame i without rendering (fast-forward for --start).
    async captureSkip(i) { await captureFrame(i, TL.FPS, true); return 200; },
    // Changes capture settings between frames (shutter, aa, dof, apScale) for A/B tests.
    captureSet(p) { Object.assign(cap, p); },
    timeline: TL,
    captureStats() { return cap.stats; },
    set(p) { Object.assign(state, p); },
    setSim(p) { sim.setParams(p); },
    async reset() { sim.reset(); sparks.reset(); await device.queue.onSubmittedWorkDone(); },
    event(e) { fireEvent(e); },
    sparks,
    setRender(p) { Object.assign(renderer.params, p); },
    prof,
    // Reads back a 3D field and returns per-channel min/max/mean (+ cells with NaN/Inf).
    async stats(name = 'S') {
      const tex = sim[name];
      const [x, y, z] = sim.res;
      const bpr = Math.ceil(x * 8 / 256) * 256;
      const buf = device.createBuffer({ size: bpr * y * z, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: y }, [x, y, z]);
      device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const u16 = new Uint16Array(buf.getMappedRange());
      const h2f = (h) => { const s = h >> 15, e = (h >> 10) & 31, m = h & 1023; const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15); return s ? -v : v; };
      const out = [0, 1, 2, 3].map(() => ({ min: Infinity, max: -Infinity, sum: 0, bad: 0, argmax: null }));
      for (let k = 0; k < z; k++) for (let j = 0; j < y; j++) for (let i = 0; i < x; i++) {
        const o = (k * y * bpr + j * bpr) / 2 + i * 4;
        for (let c = 0; c < 4; c++) {
          const v = h2f(u16[o + c]);
          const S = out[c];
          if (!Number.isFinite(v)) { S.bad++; continue; }
          if (v < S.min) S.min = v;
          if (v > S.max) { S.max = v; S.argmax = [i, j, k]; }
          S.sum += v;
        }
      }
      buf.unmap(); buf.destroy();
      const n = x * y * z;
      return out.map(S => ({ min: +S.min.toPrecision(4), max: +S.max.toPrecision(4), mean: +(S.sum / n).toPrecision(4), bad: S.bad, argmax: S.argmax }));
    },
    // Finds NaN/Inf pixels in the intermediate 2D buffers (surf, depth, vol, aux, hdr); returns counts + rows.
    async nanScan() {
      const h2f = (h) => { const s = h >> 15, e = (h >> 10) & 31, m = h & 1023; const v = e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15); return s ? -v : v; };
      const out = {};
      for (const [name, bpp, half] of [['surf', 8, true], ['depth', 4, false], ['vol', 8, true], ['aux', 8, false], ['hdr', 8, true]]) {
        const tex = renderer[name];
        const w = tex.width, h = tex.height, bpr = Math.ceil(w * bpp / 256) * 256;
        const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
        device.queue.submit([enc.finish()]);
        await buf.mapAsync(GPUMapMode.READ);
        const ab = buf.getMappedRange();
        let bad = 0, neg = 0; const rows = new Set();
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          const ch = bpp / (half ? 2 : 4);
          for (let c = 0; c < ch; c++) {
            const v = half ? h2f(new Uint16Array(ab, y * bpr + x * bpp + c * 2, 1)[0]) : new Float32Array(ab, y * bpr + x * bpp + c * 4, 1)[0];
            if (!Number.isFinite(v)) { bad++; rows.add(y); } else if (v < 0) neg++;
          }
        }
        buf.unmap(); buf.destroy();
        out[name] = { w, h, bad, neg, rows: [...rows].slice(0, 12) };
      }
      return out;
    },
    // Physical diagnostics: heat release rate (kW) and centreline/height profiles of T, soot, fuel.
    async profile() {
      const tex = sim.S;
      const [x, y, z] = sim.res;
      const bpr = Math.ceil(x * 8 / 256) * 256;
      const buf = device.createBuffer({ size: bpr * y * z, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: y }, [x, y, z]);
      device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const u16 = new Uint16Array(buf.getMappedRange());
      const h2f = (h) => { const e = (h >> 10) & 31, m = h & 1023; return e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15); };
      const dx = sim.dx, V = dx * dx * dx;
      let Q = 0;
      const rows = [];
      for (let j = 0; j < y; j += 4) {
        let Tmax = 0, Tc = 0, sootMax = 0, fuelMax = 0, hot = 0, nC = 0;
        for (let k = 0; k < z; k++) for (let i = 0; i < x; i++) {
          const o = (k * y * bpr + j * bpr) / 2 + i * 4;
          const T = 293 + 1000 * h2f(u16[o]);
          Tmax = Math.max(Tmax, T); sootMax = Math.max(sootMax, h2f(u16[o + 2])); fuelMax = Math.max(fuelMax, h2f(u16[o + 1]));
          if (T > 1000) hot++;
          if (Math.abs(i - x / 2) < 3 && Math.abs(k - z / 2) < 3) { Tc += T; nC++; }
        }
        rows.push({ h: +((j + 0.5) * dx).toFixed(2), Tc: Math.round(Tc / nC), Tmax: Math.round(Tmax), soot: +sootMax.toFixed(2), fuel: +fuelMax.toFixed(2), hotCells: hot });
      }
      for (let k = 0; k < z; k++) for (let j = 0; j < y; j++) for (let i = 0; i < x; i++) {
        const o = (k * y * bpr + j * bpr) / 2 + i * 4;
        const q = h2f(u16[o + 3]) * 1e4; if (q <= 0) continue;
        const T = 293 + 1000 * h2f(u16[o]);
        Q += 354800 * q / T * V;
      }
      buf.unmap(); buf.destroy();
      return { Q_kW: +(Q / 1000).toFixed(1), rows };
    },
    // One sim step + one frame in a single profiled command buffer; returns ms per stage.
    async profileFrame(n = 4) {
      const out = {};
      for (let i = 0; i < n; i++) {
        size();
        renderer.setCamera(camera());
        renderer.uploadParams(sim.time, frameNo++);
        const enc = device.createCommandEncoder();
        prof.begin();
        sim.step(enc, state.simDt, prof);
        renderer.renderHDR(enc, prof);
        renderer.bloom(enc, false, prof);
        renderer.final(enc, ctx.getCurrentTexture().createView(), format, false);
        prof.end(enc);
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        await prof.collect();
      }
      for (const [k, v] of Object.entries(prof.results)) out[k] = +v.toFixed(2);
      return out;
    },
  };

  if (TEST) { window.__ready = true; return; }

  // ---------- interactive ----------
  const ui = setupUI();
  let audio = null;
  const fire = { target: 0.03, intensity: 0.03, ignited: false, windT: 0, popClock: 0, burstUntil: 0 };
  sim.setParams({ intensity: fire.intensity });

  async function startAudio() {
    try {
      const { FireAudio } = await import('./audio/engine.js');
      audio = await FireAudio.create();
      await audio.start();
      audio.setVolume(ui.volume());
      audio.onEvent(ev => {
        // spark bursts in sync with the sound of random pops (events are scheduled slightly ahead)
        const delay = Math.max(0, (ev.contextTime ?? audio.ctx.currentTime) - audio.ctx.currentTime) * 1000;
        const pos = [(Math.random() - 0.5) * 0.3, 0.08 + Math.random() * 0.12, (Math.random() - 0.5) * 0.3];
        setTimeout(() => fireEvent({ type: ev.type === 'log-shift' ? 'log-shift' : 'pop', strength: ev.strength ?? 0.5, pos }), delay);
      });
    } catch (e) { console.warn('audio unavailable', e); audio = null; }
  }
  function trigger(type, strength = 1) {
    const pos = [(Math.random() - 0.5) * 0.2, 0.1, (Math.random() - 0.5) * 0.2];
    fireEvent({ type, strength, pos });
    if (type === 'log-shift' || type === 'ignite') { sim.setParams({ burstX: pos[0], burstY: 0.08, burstZ: pos[2] }); fire.burstUntil = performance.now() + 250; }
    if (type === 'gust') fire.windT = 2.4;
    audio?.trigger({ type, strength, pan: Math.max(-1, Math.min(1, pos[0] * 4)) });
  }
  function ignite() {
    if (fire.ignited) return;
    fire.ignited = true; fire.target = ui.size();
    trigger('ignite', 1);
  }
  ui.on('start', async () => { await startAudio(); ignite(); });
  ui.on('stoke', () => trigger('log-shift', 1));
  ui.on('pop', () => trigger('pop', 0.9));
  ui.on('gust', () => trigger('gust', 0.8));
  ui.on('size', v => { if (fire.ignited) fire.target = v; });
  ui.on('volume', v => audio?.setVolume(v));
  ui.on('mute', m => audio?.setVolume(m ? 0 : ui.volume()));
  setTimeout(() => { if (!fire.ignited) { ignite(); ui.hideStart(); } }, 7000);   // light it anyway, silently

  // orbit camera with inertia; when idle, a slow cinematic drift
  let dragging = false, lx = 0, ly = 0, vyaw = 0, vpitch = 0, idle = 0;
  canvas.addEventListener('pointerdown', e => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); idle = 0; });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    vyaw = -(e.clientX - lx) * 0.005; vpitch = (e.clientY - ly) * 0.004;
    state.yaw += vyaw; state.pitch = Math.min(1.25, Math.max(-0.02, state.pitch + vpitch));
    lx = e.clientX; ly = e.clientY; idle = 0; renderer.histReset = true;
  });
  canvas.addEventListener('dblclick', () => { Object.assign(state, { yaw: 0.35, pitch: 0.14, dist: 2.6, ty: 0.5 }); renderer.histReset = true; });
  canvas.addEventListener('wheel', e => { state.dist = Math.min(7, Math.max(0.7, state.dist * Math.exp(e.deltaY * 0.001))); idle = 0; e.preventDefault(); }, { passive: false });
  addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { state.paused = !state.paused; e.preventDefault(); }
    else if (e.code === 'KeyS') trigger('log-shift', 1);
    else if (e.code === 'KeyP') trigger('pop', 0.9);
    else if (e.code === 'KeyG') trigger('gust', 0.8);
    else if (e.code === 'KeyH') ui.toggle();
    else if (e.code === 'KeyF') (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()).catch(() => {});
  });
  state.dist = num('dist', 2.6); state.pitch = num('pitch', 0.14);

  let last = performance.now(), acc = 0, fps = 60, Lref = 0, prevH = 1, prevSeq = 0, prevT = 0, dhrrHeld = 0;
  function loop(now) {
    const dtReal = Math.min(0.1, (now - last) / 1000); last = now;
    fps = fps * 0.95 + (1 / Math.max(dtReal, 1e-3)) * 0.05;
    if (!dragging) {
      state.yaw += vyaw; state.pitch = Math.min(1.25, Math.max(-0.02, state.pitch + vpitch));
      vyaw *= 0.9; vpitch *= 0.85;
      idle += dtReal;
      if (idle > 8) state.yaw += dtReal * 0.035 * Math.min((idle - 8) / 4, 1);
      if (Math.abs(vyaw) > 1e-4) renderer.histReset = true;
    }
    // fire size eases towards the target (ignition = fast surge)
    const k = fire.intensity < fire.target ? (fire.target - fire.intensity > 0.3 ? 4.0 : 1.5) : 0.8;
    fire.intensity += (fire.target - fire.intensity) * (1 - Math.exp(-k * dtReal));
    fire.windT = Math.max(0, fire.windT - dtReal);
    const gw = fire.windT > 0 ? Math.sin(Math.PI * (1 - fire.windT / 2.4)) : 0;
    sim.setParams({ intensity: fire.intensity, windX: -1.4 * gw, windZ: 0.5 * gw, burstAmt: now < fire.burstUntil ? 1 : 0 });
    // random pops when the audio engine isn't generating them
    if (!audio && fire.ignited) {
      fire.popClock -= dtReal;
      if (fire.popClock <= 0) {
        fire.popClock = 0.6 + Math.random() * 2.5;
        fireEvent({ type: 'pop', strength: 0.2 + Math.random() * 0.6, pos: [(Math.random() - 0.5) * 0.3, 0.1, (Math.random() - 0.5) * 0.3] });
      }
    }
    if (!state.paused) {
      acc += dtReal;
      let n = 0;
      while (acc >= state.simDt && n < 2) { simStep(state.simDt); acc -= state.simDt; n++; }
      if (n === 2) acc = Math.min(acc, state.simDt);
    }
    draw(dragging ? 0.8 : 0.4);
    renderer.collectLights().then(ls => {
      // only a new reading updates hrr/dhrr; its slope uses the real time between the two GPU copies
      if (ls.seq && ls.seq !== prevSeq) {
        const dt = prevSeq ? Math.max((ls.t - prevT) / 1000, 1e-3) : Infinity;
        Lref = Lref ? Lref + (ls.L - Lref) * (1 - Math.exp(-Math.min(dt, 1) / 3.3)) : ls.L;
        const hrr = Lref > 0 ? ls.L / Lref : 1;
        dhrrHeld = Number.isFinite(dt) ? (hrr - prevH) / dt : 0;
        prevH = hrr; prevT = ls.t; prevSeq = ls.seq;
      }
      const c = camera();
      audio?.update({ intensity: fire.intensity, hrr: prevH, dhrr: dhrrHeld, camDist: Math.hypot(c.pos[0], c.pos[1] - 0.35, c.pos[2]), camAz: Math.atan2(c.pos[0], c.pos[2]) });
    });
    if (frameNo % 20 === 0) ui.status(`${fps.toFixed(0)} fps · ${sim.res.join('×')}`);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

// Minimal overlay UI (start screen + compact controls).
function setupUI() {
  const $ = (q) => document.querySelector(q);
  const handlers = {};
  const emit = (k, v) => (handlers[k] || []).forEach(f => f(v));
  const start = $('#start'), panel = $('#panel');
  start?.addEventListener('click', () => { start.classList.add('gone'); emit('start'); });
  document.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => emit(b.dataset.act)));
  const size = $('#size'), vol = $('#vol'), mute = $('#mute');
  size?.addEventListener('input', () => emit('size', +size.value));
  vol?.addEventListener('input', () => emit('volume', +vol.value));
  mute?.addEventListener('click', () => {
    const on = mute.classList.toggle('on');
    mute.textContent = on ? 'Sound off' : 'Sound on';
    emit('mute', on);
  });
  const q = new URLSearchParams(location.search).get('q') || 'med';
  document.querySelectorAll('[data-q]').forEach(a => {
    if (a.dataset.q === q) a.classList.add('cur');
    a.addEventListener('click', e => { e.preventDefault(); const u = new URL(location.href); u.searchParams.set('q', a.dataset.q); location.href = u.toString(); });
  });
  return {
    on(k, f) { (handlers[k] ||= []).push(f); },
    size: () => +(size?.value ?? 1),
    volume: () => +(vol?.value ?? 0.8),
    status(t) { const s = $('#hud'); if (s) s.textContent = t; },
    toggle() { panel?.classList.toggle('hidden'); $('#hud')?.classList.toggle('hidden'); },
    hideStart() { start?.classList.add('gone'); },
  };
}

main().catch(e => {
  console.error(e);
  document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f88;position:fixed;top:10px;left:10px">${e.message}</pre>`);
  window.__error = String(e && e.message || e);
});
