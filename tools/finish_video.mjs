// Builds the soundtrack for a rendered video from its per-frame capture stats + the shared timeline,
// renders it with the procedural audio engine and muxes the deliverable.
// Usage: node tools/finish_video.mjs --video out/video/fire_raw.mp4 [--stats out/video/fire_raw_frames.json]
//        [--out out/video/fire_final.mp4] [--crf 17]
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import * as TL from '../src/timeline.js';

const A = {};
for (let i = 2; i < process.argv.length; i++) if (process.argv[i].startsWith('--')) A[process.argv[i].slice(2)] = process.argv[i + 1];
const video = path.resolve(A.video || 'out/video/fire_raw.mp4');
const base = video.replace(/\.mp4$/, '');
const statsFile = A.stats || `${base}_frames.json`;
const out = path.resolve(A.out || `${base.replace(/_raw$/, '')}_final.mp4`);
const stats = JSON.parse(fs.readFileSync(statsFile, 'utf8'));

// heat-release proxy: visible light output normalised to the steady fire
const steady = stats.filter(s => s.t > 3).map(s => s.light).sort((a, b) => a - b);
const Lref = steady.length ? steady[Math.floor(steady.length / 2)] : 1;
const fps = TL.FPS;
const frames = stats.map((s, i) => {
  const hrr = Lref > 0 ? s.light / Lref : 1;
  const prev = i > 0 ? stats[i - 1].light / Lref : hrr;
  return { intensity: s.intensity, hrr, dhrr: (hrr - prev) * fps, camDist: s.camDist, camAz: s.camAz };
});
// light the dhrr a little (frame-to-frame jitter is not combustion noise)
for (let i = 1; i < frames.length; i++) frames[i].dhrr = 0.6 * frames[i].dhrr + 0.4 * frames[i - 1].dhrr;

// Stereo position of each event as seen by the camera at that moment (screenPan, already camera-relative,
// so the engine must not project it again), plus the fire-local position for its sub-events.
// The soundtrack is clip-relative: a render started with --start begins at stats[0].t.
const sub = (a, b) => a.map((v, i) => v - b[i]);
const t0 = stats.length ? stats[0].t : 0;
const dur = stats.length / fps;
const events = TL.EVENTS.filter(e => e.t >= t0 && e.t < t0 + dur).map(e => {
  const c = TL.cameraAt(e.t);
  const fwd = sub(c.target, c.pos); const fl = Math.hypot(...fwd); fwd.forEach((v, i) => (fwd[i] = v / fl));
  const right = [fwd[2] * -1, 0, fwd[0]]; const rl = Math.hypot(...right) || 1;   // right = fwd x up
  const pos = e.pos || [0, 0.2, 0];
  const rel = sub(pos, c.pos);
  const x = (rel[0] * -fwd[2] + rel[2] * fwd[0]) / rl;
  const pan = Math.max(-1, Math.min(1, x / Math.max(Math.hypot(...rel) * 0.6, 0.2)));
  return { t: +(e.t - t0).toFixed(6), type: e.type, strength: e.strength ?? 1, screenPan: +pan.toFixed(3), x: pos[0], y: pos[1], z: pos[2] };
});
// every audible pop comes from TL.EVENTS (so every pop has sparks); no crickets (a regular chirp reads as a beep)
const timeline = { fps, duration: dur, seed: 20261025, params: { popRate: 0, mix: { crickets: -90 } }, frames, events };
const tlFile = `${base}_audio_timeline.json`;
fs.writeFileSync(tlFile, JSON.stringify(timeline));
const wav = `${base}.wav`;
console.log('rendering audio ...');
let r = spawnSync('node', ['tools/render_audio.mjs', tlFile, wav, '--events', `${base}_audio_events.json`], { stdio: 'inherit' });
if (r.status !== 0) process.exit(r.status || 1);

console.log('muxing ...');
// ffmpeg's native AAC encoder overshoots the deliberately clipped crackle transients by up to +4 dB after decoding
// (measured: 19 samples above full scale). The MediaFoundation encoder (Windows) stays below -0.5 dBFS; elsewhere,
// band-limit to 16 kHz first, which removes most of the overshoot.
const encs = spawnSync('ffmpeg', ['-hide_banner', '-encoders'], { encoding: 'utf8' }).stdout || '';
const aacArgs = /\baac_mf\b/.test(encs) ? ['-c:a', 'aac_mf', '-b:a', '320k'] : ['-af', 'lowpass=f=16000:p=2', '-c:a', 'aac', '-b:a', '320k'];
r = spawnSync('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-i', video, '-i', wav,
  '-map', '0:v:0', '-map', '1:a:0',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', String(A.crf || 17), '-tune', 'grain', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
  '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv',
  '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
  ...aacArgs, '-ar', '48000', '-shortest', '-movflags', '+faststart', out], { stdio: 'inherit' });
if (r.status !== 0) process.exit(r.status || 1);
console.log('wrote', out, (fs.statSync(out).size / 1e6).toFixed(1), 'MB');
