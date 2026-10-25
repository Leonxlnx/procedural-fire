#!/usr/bin/env node
// render_audio.mjs — deterministic offline render of the procedural campfire soundtrack for the video.
//
//   node tools/render_audio.mjs <timeline.json> <out.wav> [--events out_events.json]
//   node tools/render_audio.mjs --demo [out.wav] [--events out_events.json]     (writes out/audio/demo_timeline.json)
//
// Timeline: { fps, duration, seed?, params?, frames: [{ intensity, hrr, dhrr, camDist, camAz }],
//             events: [{ t, type: 'ignite'|'pop'|'crackle-burst'|'log-shift'|'whoosh'|'gust', strength, pan }] }
// Output:   48 kHz / 24-bit stereo WAV, mastered to -14 LUFS integrated, true peak <= -1 dBTP (checked with ffmpeg
//           ebur128 when available, closed loop), bit-identical for any --block size.
// Events:   { sampleRate, fps, events: [random pops: {t, frame, strength, size, pan, x, y, z, stereoPan}], all: [...] }
//           frame = floor(t * fps): the spark appears on the frame at or just before the sound (audio never leads).
// Options:  --block N (default 128)  --seed N  --no-master  --solo <layer>  --gain-db x  --stats  --no-ffmpeg
//           --target -14  --tp -1  --json report.json  --quiet
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { FireAudioEngine, RNG, deriveSeed, REVERB_DEFAULTS, Biquad } from '../src/audio/fire-dsp.js';
import { mixWithReverb, masterChain, integratedLoudness, loudnessRange, shortTermLoudness, truePeakDb, makeTruePeak, encodeWav24, kWeightingCoefs } from '../src/audio/master-offline.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// social master: 24 dB/oct high-pass (phone speakers reproduce nothing below ~150 Hz; sub energy only costs headroom)
export const MASTER_HP_HZ = 45; // (55 Hz took the 25-40 Hz bands below every real recording)
const PREROLL_S = 2, LOOP_XFADE_S = 0.25, LOOP_PAD_S = 1;
// engine parameters of the picture-locked render (the timeline's params override them): level jumps at camera cuts are
// 0.6x the live "closer = louder" law (a cut to a macro shot is +2.6 dB instead of +4.3 dB)
export const VIDEO_PARAMS = { distDepth: 0.6 };
// loop (video) masters: upward leveler, quiet stretches of the bed lifted towards integrated - 2.5 LU by at most 6 dB
const LIFT = { floorLu: 2.5, maxDb: 6 };
const VALUE_FLAGS = new Set(['events', 'block', 'seed', 'solo', 'gain-db', 'json', 'target', 'tp', 'sr']);
// --loop / --no-loop force the loop mode (default: auto, see isLoopTimeline), --no-preroll starts the engine cold at 0
function parseArgs(argv) {
  const pos = [], opt = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) { const k = a.slice(2); if (VALUE_FLAGS.has(k)) opt[k] = argv[++i]; else opt[k] = true; }
    else pos.push(a);
  }
  return { pos, opt };
}

// ------------------------------------------------------------------ demo timeline (20 s)
const smooth = (pts, t) => {
  if (t <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) if (t <= pts[i][0]) {
    const [t0, v0] = pts[i - 1], [t1, v1] = pts[i], x = (t - t0) / (t1 - t0);
    return v0 + (v1 - v0) * x * x * (3 - 2 * x);
  }
  return pts[pts.length - 1][1];
};
export function makeDemoTimeline(seed = 20261025) {
  const fps = 60, duration = 20, N = duration * fps;
  const intensity = [[0, 0], [0.3, 0], [0.9, 0.5], [1.7, 0.95], [3.0, 1.0], [3.3, 1.08], [5.0, 1.0], [8.9, 1.0], [9.3, 1.3], [11, 1.12], [12.9, 1.1], [13.8, 1.3], [15.6, 1.12], [17, 1.0], [20, 0.88]];
  const camDist = [[0, 1.3], [1.5, 1.3], [5, 1.55], [12, 1.9], [20, 2.2]];
  // stand-in for the sim's heat-release proxy: lagged intensity x quasi-periodic puffing (1.5/sqrt(D) Hz) x noise,
  // plus surges where the visuals flare (big pop, log shift, gust)
  const r = new RNG(deriveSeed(seed, 'demo-hrr'));
  const surge = (t, t0, a, rise, fall) => (t < t0 ? 0 : a * (1 - Math.exp(-(t - t0) / rise)) * Math.exp(-(t - t0) / fall));
  let lag = 0, nb1 = 0, nb2 = 0;
  const hrr = [];
  for (let i = 0; i < N; i++) {
    const t = i / fps, I = smooth(intensity, t);
    lag += (1 - Math.exp(-1 / (0.35 * fps))) * (I - lag);
    nb1 += 0.25 * (r.gauss() - nb1); nb2 += 0.05 * (nb1 - nb2);
    const flick = 1 + 0.06 * Math.sin(2 * Math.PI * 1.75 * t + 0.4 * Math.sin(2 * Math.PI * 0.21 * t)) + 0.12 * nb2;
    const s = surge(t, 3.2, 0.25, 0.05, 0.4) + surge(t, 9.0, 0.6, 0.12, 0.9) + surge(t, 13.0, 0.35, 0.3, 1.2);
    hrr.push(Math.max(0, lag * flick + s * lag));
  }
  const frames = [];
  for (let i = 0; i < N; i++) {
    const t = i / fps;
    const dh = (hrr[Math.min(N - 1, i + 1)] - hrr[Math.max(0, i - 1)]) * fps / ((i > 0 && i < N - 1) ? 2 : 1);
    frames.push({
      intensity: +smooth(intensity, t).toFixed(4), hrr: +hrr[i].toFixed(4), dhrr: +dh.toFixed(4),
      camDist: +smooth(camDist, t).toFixed(4), camAz: +(0.35 + 0.025 * t).toFixed(4),
    });
  }
  return {
    fps, duration, seed,
    notes: 'Demo cue sheet. 0.2 s ignite (match strike + whoosh); steady crackling; 3.2 s big pop; 6.4 s crackle burst; 9.0 s log shift (+hrr surge); 13.0 s gust; settling from 17 s. pan = fire-local lateral position (x = pan * fireRadius).',
    frames,
    events: [
      { t: 0.2, type: 'ignite', strength: 1.0, pan: 0.1 },
      { t: 3.2, type: 'pop', strength: 1.0, pan: -0.25 },
      { t: 6.4, type: 'crackle-burst', strength: 0.5, pan: 0.4 },
      { t: 9.0, type: 'log-shift', strength: 0.9, pan: 0.2 },
      { t: 13.0, type: 'gust', strength: 0.85 },
      { t: 17.6, type: 'pop', strength: 0.4, pan: 0.5 },
      { t: 18.9, type: 'ember', strength: 0.8, pan: -0.3 },
    ],
  };
}

// ------------------------------------------------------------------ ffmpeg ebur128
function ffmpegEbur128(file) {
  const r = spawnSync('ffmpeg', ['-hide_banner', '-nostats', '-i', file, '-af', 'ebur128=peak=true:framelog=quiet', '-f', 'null', '-'], { encoding: 'utf8' });
  if (r.error || r.status !== 0) return null;
  const s = r.stderr, sum = s.slice(s.lastIndexOf('Summary:'));
  const num = (re) => { const m = sum.match(re); return m ? parseFloat(m[1]) : null; };
  return { I: num(/I:\s+(-?[\d.]+) LUFS/), LRA: num(/LRA:\s+(-?[\d.]+) LU/), TP: num(/True peak:\s+Peak:\s+(-?[\d.inf]+) dBFS/) };
}

// ------------------------------------------------------------------ render
// o.preroll (s): the engine runs this long before timeline time 0 (whole frames; frame 0 held, or with o.loop the last
// frames of the timeline), so the soundtrack starts warm: no fade-in from filter / smoother / reverb start-up.
// o.tail (s): extra samples rendered after the end (controls held at the last frame), the source of the loop crossfade.
// Returned buffers cover [-preroll, duration + tail); `pre` = the pre-roll in samples. Without both: [0, duration).
export function renderTimeline(tl, o = {}) {
  const sr = o.sr || tl.sampleRate || 48000, fps = tl.fps || 60;
  const duration = tl.duration ?? (tl.frames ? tl.frames.length / fps : 20);
  const N = Math.round(duration * sr), block = Math.max(1, o.block || 128);
  const seed = (o.seed ?? tl.seed ?? 1) >>> 0;
  const frames0 = tl.frames || [];
  const preF = frames0.length ? Math.max(0, Math.round((o.preroll || 0) * fps)) : 0, pre = Math.round(preF * sr / fps), offS = pre / sr;
  const T = Math.max(0, Math.round((o.tail || 0) * sr)), NT = pre + N + T;
  let frames = frames0;
  if (preF) { const pf = o.loop ? frames0.slice(Math.max(0, frames0.length - preF)) : []; while (pf.length < preF) pf.unshift(frames0[0]); frames = pf.concat(frames0); }
  const params = o.videoParams === false ? tl.params || {} : { ...VIDEO_PARAMS, ...(tl.params || {}) };
  const eng = new FireAudioEngine({ sampleRate: sr, seed, params, solo: o.solo || null });
  eng.setTimeline({ fps, frames });
  const warnings = [];
  for (const ev of tl.events || []) {
    if (!eng.schedule(offS && typeof ev.t === 'number' ? { ...ev, t: ev.t + offS } : ev)) warnings.push(`event ignored (unknown type or queue full): ${JSON.stringify(ev)}`);
    else if (typeof ev.t === 'number' && ev.t >= duration) warnings.push(`event after the end (${duration} s) will not sound: ${JSON.stringify(ev)}`);
  }
  const dL = new Float32Array(NT), dR = new Float32Array(NT), sL = new Float32Array(NT), sR = new Float32Array(NT);
  const raw = []; let since = 0;
  const t0 = performance.now();
  for (let i = 0; i < NT; i += block) {
    const n = Math.min(block, NT - i);
    eng.render(dL.subarray(i, i + n), dR.subarray(i, i + n), sL.subarray(i, i + n), sR.subarray(i, i + n), n);
    since = eng.log.read(since, raw);
  }
  const engineMs = performance.now() - t0;
  // event times back on the timeline clock; pre-roll events dropped (none are scheduled there)
  const log = []; for (const e of raw) { e.n -= pre; e.t = e.n / sr; if (e.n >= 0) log.push(e); }
  let bad = 0; for (const a of [dL, dR, sL, sR]) for (let i = 0; i < NT; i++) if (!Number.isFinite(a[i])) bad++;
  const hash = crypto.createHash('sha256');
  for (const a of [dL, dR, sL, sR]) hash.update(Buffer.from(a.buffer, a.byteOffset, a.byteLength));
  return { sr, fps, duration, N, pre, tail: T, seed, block, dL, dR, sL, sR, log, engineMs, nonFinite: bad, dryHash: hash.digest('hex').slice(0, 16), queueDropped: eng.queue.dropped, warnings };
}

// Loop: is the timeline meant to loop (last frame back on the first frame's framing and fire)? tl.loop overrides.
export function isLoopTimeline(tl) {
  if (typeof tl.loop === 'boolean') return tl.loop;
  const F = tl.frames || []; if (F.length < 2) return false;
  const a = F[0], b = F[F.length - 1], num = (v) => (typeof v === 'number' && isFinite(v) ? v : null);
  if ([a.camDist, b.camDist, a.camAz, b.camAz, a.intensity, b.intensity].some((v) => num(v) === null)) return false;
  let dAz = b.camAz - a.camAz; dAz -= 2 * Math.PI * Math.round(dAz / (2 * Math.PI));
  return a.intensity > 0.2 && Math.abs(b.intensity - a.intensity) < 0.15 && Math.abs(Math.log(b.camDist / a.camDist)) < 0.15 && Math.abs(dAz) < 0.2;
}

// Loop level match: the bed loudness (K-weighted 50 ms windows below their 70th percentile, so pops do not count) of the
// last second is brought to that of the first second by a gain ramp over the last rampS seconds (smoothstep, at most
// +-maxDb), applied to [pre-roll | 0 .. N | tail] buffers before cropLoop (the tail, which becomes the head's cross-fade
// source, gets the full gain). Returns the gain in dB.
export function matchLoopLevel(chans, pre, N, T, sr, rampS = 3, maxDb = 4) {
  const [s1, s2] = kWeightingCoefs(sr), w = Math.round(0.05 * sr);
  const bed = (a, b) => {
    const v = [];
    const f = chans.map(() => [new Biquad().setCoefs(s1.b[0], s1.b[1], s1.b[2], s1.a[0], s1.a[1]), new Biquad().setCoefs(1, -2, 1, s2.a[0], s2.a[1])]);
    let acc = 0, k = 0;
    for (let i = Math.max(0, a - sr); i < b; i++) { // 1 s of filter warm-up before the range
      let e = 0; for (let c = 0; c < chans.length; c++) { const y = f[c][1].tick(f[c][0].tick(chans[c][i])); e += y * y; }
      if (i < a) continue; acc += e; if (++k === w) { v.push(acc / w); acc = 0; k = 0; }
    }
    v.sort((x, y) => x - y); const g = v.slice(0, Math.max(1, Math.round(0.7 * v.length)));
    return 10 * Math.log10(g.reduce((x, y) => x + y, 0) / g.length + 1e-30);
  };
  const S = Math.round(sr), head = bed(pre, pre + S), last = bed(pre + N - S, pre + N);
  const gDb = Math.max(-maxDb, Math.min(maxDb, head - last)), R = Math.round(rampS * sr);
  if (Math.abs(gDb) < 0.05) return 0;
  for (const x of chans) for (let i = pre + N - R; i < pre + N + T; i++) {
    const u = Math.min(1, (i - (pre + N - R)) / R), s = u * u * (3 - 2 * u);
    x[i] *= Math.pow(10, gDb * s / 20);
  }
  return gDb;
}

// [pre-roll | 0 .. N | tail] buffers -> N samples. With loop: the first T samples cross-fade (equal power) from the tail, the
// continuation of the last sample, into the head, so sample N-1 runs straight into sample 0 when the file repeats.
export function cropLoop(chans, pre, N, T, loop) {
  return chans.map((x) => {
    const y = new Float64Array(N); for (let i = 0; i < N; i++) y[i] = x[pre + i];
    if (loop) for (let i = 0; i < Math.min(T, N); i++) { const a = (Math.PI / 2) * (i / T); y[i] = x[pre + i] * Math.sin(a) + x[pre + N + i] * Math.cos(a); }
    return y;
  });
}

// Upward leveler (offline, zero-phase): where the 400 ms K-weighted loudness falls more than floorLu below the integrated
// loudness, raise it towards that floor by at most maxDb (gain smoothed forwards and backwards, tau 0.25 s). Quiet
// stretches of the bed between events stay audible without touching the loud moments. Returns per-sample gains.
export function liftGains(L, R, sr, floorLu = 3.5, maxDb = 4) {
  const N = L.length, [s1, s2] = kWeightingCoefs(sr), hop = Math.round(0.05 * sr), w = Math.round(0.4 * sr);
  const cs = new Float64Array(N + 1);
  { const sq = new Float64Array(N); for (const x of [L, R]) { const f1 = new Biquad().setCoefs(s1.b[0], s1.b[1], s1.b[2], s1.a[0], s1.a[1]), f2 = new Biquad().setCoefs(1, -2, 1, s2.a[0], s2.a[1]); for (let i = 0; i < N; i++) { const y = f2.tick(f1.tick(x[i])); sq[i] += y * y; } } for (let i = 0; i < N; i++) cs[i + 1] = cs[i] + sq[i]; }
  const floor = integratedLoudness([L, R], sr) - floorLu, nb = Math.floor(N / hop) + 1, g = new Float64Array(nb);
  for (let b = 0; b < nb; b++) { const c = b * hop, a = Math.max(0, c - (w >> 1)), z = Math.min(N, a + w); const m = -0.691 + 10 * Math.log10((cs[z] - cs[a]) / Math.max(1, z - a) + 1e-30); g[b] = Math.max(0, Math.min(maxDb, floor - m)); }
  const k = 1 - Math.exp(-hop / (0.25 * sr));
  for (let pass = 0; pass < 1; pass++) { let y = g[0]; for (let b = 0; b < nb; b++) { y += k * (g[b] - y); g[b] = y; } y = g[nb - 1]; for (let b = nb - 1; b >= 0; b--) { y += k * (g[b] - y); g[b] = y; } }
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) { const x = i / hop, b = Math.min(nb - 2, Math.floor(x)), f = x - b; out[i] = Math.pow(10, (g[b] + (g[b + 1] - g[b]) * f) / 20); }
  return out;
}

// o.loopPad (samples): master the signal as a loop (wrapped by loopPad on both sides), so filters, soft clip and limiter
// see across the seam and the mastered file still repeats without a step; loudness and limiter statistics cover the loop.
export function master(L, R, sr, o = {}) {
  const target = o.target ?? -14, tp = makeTruePeak(32), N = L.length, K = Math.min(N, Math.max(0, o.loopPad || 0));
  const wrap = (x) => { if (!K) return x; const y = new Float64Array(N + 2 * K); y.set(x.subarray(N - K), 0); y.set(x, K); y.set(x.subarray(0, K), K + N); return y; };
  let inL = wrap(L), inR = wrap(R); const mid = (out) => (K ? out.map((x) => x.subarray(K, K + N)) : out);
  if (o.lift) { // upward leveler on the (wrapped) loop
    const g = liftGains(inL, inR, sr, o.lift.floorLu, o.lift.maxDb);
    inL = Float64Array.from(inL, (v, i) => v * g[i]); inR = Float64Array.from(inR, (v, i) => v * g[i]);
  }
  const pre = integratedLoudness([L, R], sr);
  let gainDb = o.gainDb ?? (target - pre), out = [L, R];
  // glue comp off by default (it flattened the slow level variation real fires have; o.comp re-enables it)
  const mopts = { ceilingDb: o.ceilingDb ?? -1.4, compThresholdDb: o.comp ? pre - (o.compOffsetDb ?? 2) : undefined, hpHz: o.hpHz ?? MASTER_HP_HZ, clipPasses: o.clipPasses ?? 3, phoneBodyDb: o.phoneBodyDb ?? 0.5, airDb: -1.5, kneeDb: o.kneeDb ?? 4.5, releaseMs: o.releaseMs ?? 80, tp, measureFrom: K, measureTo: K + N };
  let st = null;
  if (o.gainDb !== undefined) { const m = masterChain(inL, inR, sr, { ...mopts, gainDb }); st = m; out = mid(m); }
  else for (let pass = 0; pass < 6; pass++) {
    const m = masterChain(inL, inR, sr, { ...mopts, gainDb }); st = m; out = mid(m);
    const got = integratedLoudness(out, sr);
    if (Math.abs(got - target) < 0.03) break;
    gainDb += target - got;
  }
  return { out, pre, gainDb, ceilingDb: mopts.ceilingDb, limiterMaxGrDb: st.limiterMaxGrDb ?? 0, limiterStats: st.limiterStats ?? null };
}

async function main() {
  const { pos, opt } = parseArgs(process.argv.slice(2));
  const say = (...a) => { if (!opt.quiet) console.error(...a); };
  let tl, outWav, eventsPath = opt.events;
  if (opt.demo) {
    tl = makeDemoTimeline(opt.seed !== undefined ? +opt.seed : undefined);
    const dir = path.join(ROOT, 'out', 'audio'); fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'demo_timeline.json'), JSON.stringify(tl));
    outWav = pos[0] || path.join(dir, 'demo.wav');
    eventsPath = eventsPath || path.join(dir, 'demo_events.json');
    say('demo timeline -> ' + path.join(dir, 'demo_timeline.json'));
  } else {
    if (pos.length < 2) { console.error('usage: node tools/render_audio.mjs <timeline.json> <out.wav> [--events out_events.json] | --demo [out.wav]'); process.exit(2); }
    tl = JSON.parse(fs.readFileSync(pos[0], 'utf8'));
    outWav = pos[1];
  }
  const target = opt.target !== undefined ? +opt.target : -14, tpTarget = opt.tp !== undefined ? +opt.tp : -1.0;

  // loop: the file repeats seamlessly (tail cross-faded into the head, mastered as a loop); auto-detected from the
  // camera/fire of the first and last frame unless tl.loop or --loop / --no-loop say otherwise
  const loop = opt.loop ? true : opt['no-loop'] ? false : isLoopTimeline(tl);
  // pre-roll: warm start (no fade-in) whenever the fire burns at frame 0 (a loop pre-rolls the end of the timeline)
  const preroll = opt['no-preroll'] ? 0 : loop || ((tl.frames || [])[0]?.intensity ?? 0) > 0.05 ? PREROLL_S : 0;
  const R = renderTimeline(tl, { block: opt.block ? +opt.block : 128, seed: opt.seed !== undefined ? +opt.seed : undefined, solo: opt.solo, sr: opt.sr ? +opt.sr : undefined, preroll, tail: loop ? LOOP_XFADE_S : 0, loop });
  if (R.nonFinite) { console.error(`engine produced ${R.nonFinite} non-finite samples`); process.exit(1); }
  for (const w of R.warnings) console.error('warning: ' + w);
  say(`engine: ${R.duration}s in ${(R.engineMs / 1000).toFixed(2)}s (${(R.duration * 1000 / R.engineMs).toFixed(1)}x realtime), dryHash ${R.dryHash}${loop ? ', loop' : ''}${preroll ? `, pre-roll ${preroll} s` : ''}`);
  const mixed = mixWithReverb(R.dL, R.dR, R.sL, R.sR, R.sr, R.seed, { ...REVERB_DEFAULTS, ...(tl.reverb || {}) });
  const loopGainDb = loop ? matchLoopLevel(mixed, R.pre, R.N, R.tail, R.sr) : 0;
  const [L, Rr] = cropLoop(mixed, R.pre, R.N, R.tail, loop);
  if (loop) for (const e of R.log) if (e.n >= R.N) { e.n -= R.N; e.t = e.n / R.sr; } // tail events sound in the head cross-fade
  R.log = R.log.filter((e) => e.n < R.N).sort((a, b) => a.n - b.n);

  let M, ff = null, ceilingDb = -1.4, out, iterations = 0;
  if (opt['no-master']) { out = [L, Rr]; M = { pre: integratedLoudness([L, Rr], R.sr), gainDb: 0, ceilingDb: null, limiterMaxGrDb: 0 }; }
  for (let it = 0; it < 4 && !opt['no-master']; it++) {
    iterations = it + 1;
    M = master(L, Rr, R.sr, { target, ceilingDb, gainDb: opt['gain-db'] !== undefined ? +opt['gain-db'] : undefined, loopPad: loop ? Math.round(LOOP_PAD_S * R.sr) : 0, lift: loop && !opt['no-lift'] ? LIFT : null });
    out = M.out;
    fs.mkdirSync(path.dirname(path.resolve(outWav)), { recursive: true });
    fs.writeFileSync(outWav, encodeWav24(out, R.sr));
    if (opt['no-ffmpeg']) break;
    ff = ffmpegEbur128(outWav);
    if (!ff || ff.TP === null || ff.TP <= tpTarget - 0.05) break;
    ceilingDb -= ff.TP - (tpTarget - 0.1); // closed loop on ffmpeg's true-peak meter
    say(`ffmpeg TP ${ff.TP} dBTP > ${tpTarget}: lowering limiter ceiling to ${ceilingDb.toFixed(2)} dB`);
  }
  if (opt['no-master']) { fs.writeFileSync(outWav, encodeWav24(out, R.sr)); if (!opt['no-ffmpeg']) ff = ffmpegEbur128(outWav); }

  // events
  const fps = R.fps;
  const toOut = (e) => ({ t: +e.t.toFixed(6), frame: Math.floor(e.t * fps + 1e-9), type: e.type, strength: e.strength, size: e.size, pan: e.pan, x: e.x, y: e.y, z: e.z, stereoPan: e.stereoPan, source: e.source, ...(e.id >= 0 ? { id: e.id } : {}) });
  const randomPops = R.log.filter((e) => e.type === 'pop' && e.source === 'random').map(toOut);
  if (eventsPath) {
    fs.mkdirSync(path.dirname(path.resolve(eventsPath)), { recursive: true });
    fs.writeFileSync(eventsPath, JSON.stringify({ sampleRate: R.sr, fps, duration: R.duration, seed: R.seed, note: 'events = random pops generated by the audio engine (spawn sparks at frame/t); all = every hero event fired, scripted included', events: randomPops, all: R.log.map(toOut) }, null, 1));
  }

  const wavHash = crypto.createHash('sha256').update(fs.readFileSync(outWav)).digest('hex').slice(0, 16);
  const tp = makeTruePeak(32);
  let spk = 0; for (const c of out) for (let i = 0; i < c.length; i++) spk = Math.max(spk, Math.abs(c[i]));
  let dc = out.map((c) => c.reduce((a, b) => a + b, 0) / c.length);
  const report = {
    out: path.resolve(outWav), events: eventsPath ? path.resolve(eventsPath) : null,
    seconds: R.duration, sampleRate: R.sr, block: R.block, seed: R.seed, loop, prerollS: preroll, loopEndGainDb: +loopGainDb.toFixed(2),
    engineRealtimeFactor: +(R.duration * 1000 / R.engineMs).toFixed(1),
    preMasterLufs: +M.pre.toFixed(2), masterGainDb: +M.gainDb.toFixed(2), limiterCeilingDb: M.ceilingDb, limiterMaxGrDb: +(M.limiterMaxGrDb || 0).toFixed(2), limiter: M.limiterStats ?? null, masterIterations: iterations,
    lufs: +integratedLoudness(out, R.sr).toFixed(2), truePeakDbTP: +truePeakDb(out, tp).toFixed(2), samplePeakDbFS: +(20 * Math.log10(spk)).toFixed(2),
    lra: +loudnessRange(out, R.sr).toFixed(2), dcOffset: dc.map((v) => +v.toExponential(2)),
    ffmpeg: ff, dryHash: R.dryHash, wavHash,
    heroEvents: R.log.length, randomPops: randomPops.length, queueDropped: R.queueDropped,
  };
  if (opt.stats) {
    const st = shortTermLoudness(out, R.sr, 1, 3); report.shortTerm3s = st.map((v) => +v.toFixed(1));
    let c = 0, ll = 0, rr = 0; for (let i = 0; i < R.N; i++) { c += out[0][i] * out[1][i]; ll += out[0][i] ** 2; rr += out[1][i] ** 2; }
    report.lrCorrelation = +(c / Math.sqrt(ll * rr)).toFixed(3);
  }
  if (opt.json) fs.writeFileSync(opt.json, JSON.stringify(report, null, 1));
  console.log(JSON.stringify(report, null, 1));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((e) => { console.error(e); process.exit(1); });
