// Offline cinematic renderer: drives the page in headless Chrome, renders motion-blurred frames on the GPU
// and streams raw RGBA into ffmpeg.
// Usage: node tools/render_video.mjs [--w 1080] [--h 1920] [--q ultra] [--frames 600] [--start 0]
//        [--aa 2] [--shutter 0.5] [--simDt 0.0041667] [--out out/video/fire.mp4] [--stills 30] [--dof]
//        [--render '{renderer params}'] [--sim '{sim params}']
// The final cinematic settings live in tools/render_final.mjs.
import { spawn } from 'node:child_process';
import { launchChrome } from './chrome.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { createServer } from './server.mjs';

const A = Object.fromEntries(process.argv.slice(2).reduce((acc, v, i, arr) => (v.startsWith('--') ? [...acc, [v.slice(2), arr[i + 1]?.startsWith('--') || arr[i + 1] === undefined ? '1' : arr[i + 1]]] : acc), []));
const W = Number(A.w || 1080), H = Number(A.h || 1920);
const FRAMES = Number(A.frames || 600), START = Number(A.start || 0);
const OUT = path.resolve(A.out || 'out/video/fire_raw.mp4');
const STILLS = Number(A.stills || 30);
fs.mkdirSync(path.dirname(OUT), { recursive: true });
const stillDir = path.join(path.dirname(OUT), path.basename(OUT, path.extname(OUT)) + '_stills');
fs.mkdirSync(stillDir, { recursive: true });

const fps = 30;
const ff = spawn('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error',
  '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${H}`, '-r', String(fps), '-i', '-',
  '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', String(A.crf || 12), '-tune', 'grain',
  '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv',
  '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
  '-movflags', '+faststart', OUT], { stdio: ['pipe', 'inherit', 'inherit'] });

let received = 0;
let finishing = false;
// fail fast: a dead ffmpeg must never leave the page loop waiting on a 'drain' forever or drop frames silently
const fail = (msg) => { console.error('FATAL:', msg); process.exit(1); };
ff.on('error', e => fail(`ffmpeg spawn: ${e.message}`));
ff.stdin.on('error', e => fail(`ffmpeg stdin: ${e.message}`));
ff.on('exit', (code, sig) => { if (!finishing || code !== 0) fail(`ffmpeg exited (code ${code}, signal ${sig})`); });
const port = 8700 + Math.floor(Math.random() * 80);
const server = await createServer(port, (req, res) => {
  if (req.method !== 'POST' || !req.url.startsWith('/frame')) return false;
  const chunks = [];
  req.on('data', c => chunks.push(c));
  req.on('end', () => {
    const buf = Buffer.concat(chunks);
    const i = Number(new URL(req.url, 'http://x').searchParams.get('i'));
    if (buf.length !== W * H * 4) { res.writeHead(400).end('bad size'); return; }
    if (i < START) { res.writeHead(200).end('skip'); return; }   // frames before --start are simulated, not kept
    if (ff.stdin.destroyed || ff.exitCode !== null) { res.writeHead(500).end('ffmpeg gone'); return; }
    const ok = ff.stdin.write(buf);
    if (STILLS && (i % STILLS === 0)) {
      const p = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${H}`, '-i', '-', '-frames:v', '1', path.join(stillDir, `f${String(i).padStart(4, '0')}.png`)]);
      p.on('error', () => {});
      p.stdin.on('error', e => console.log('still', i, e.message));
      p.stdin.end(buf);
    }
    received++;
    const done = () => res.writeHead(200).end('ok');
    if (ok) done(); else ff.stdin.once('drain', done);
  });
  return true;
});

const browser = await launchChrome();
const page = await browser.newPage();
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[${m.type()}] ${m.text()}`); });
page.on('pageerror', e => console.log('[pageerror]', e.message));
await page.setViewport({ width: 400, height: 400 });
await page.goto(`http://localhost:${port}/index.html?test&w=${W}&h=${H}&q=${A.q || 'ultra'}${A.vs ? `&vs=${A.vs}` : ''}${A.step ? `&step=${A.step}` : ''}`);
await page.waitForFunction('window.__ready || window.__error', { timeout: 120000 });
const err = await page.evaluate('window.__error');
if (err) { console.log('INIT ERROR', err); process.exit(1); }
console.log('GPU:', await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter(); const i = a?.info || {}; return `${i.vendor || '?'} ${i.architecture || ''} ${i.device || ''} ${i.description || ''}`.trim(); }));
if (A.render) await page.evaluate(p => FIRE.setRender(p), JSON.parse(A.render));
if (A.sim) await page.evaluate(p => FIRE.setSim(p), JSON.parse(A.sim));

const t0 = Date.now();
await page.evaluate(o => FIRE.captureBegin(o), { w: W, h: H, simDt: Number(A.simDt || 1 / 240), aa: Number(A.aa || 2), shutter: Number(A.shutter || 0.5), dof: !!A.dof, ...(A.preroll ? { preroll: Number(A.preroll) } : {}) });
console.log(`preroll done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
// frames before START are simulated (same steps and events) but not rendered
for (let i = 0; i < START + FRAMES; i++) {
  const tf = Date.now();
  const status = await (i < START ? page.evaluate(i => FIRE.captureSkip(i), i) : page.evaluate(i => FIRE.captureFrame(i), i));
  if (status !== 200) fail(`frame ${i} status ${status}`);
  const el = (Date.now() - t0) / 1000;
  if (i % 10 === 0 || i === START + FRAMES - 1) console.log(`frame ${i + 1}/${START + FRAMES} ${((Date.now() - tf) / 1000).toFixed(2)}s/frame, elapsed ${(el / 60).toFixed(1)} min, eta ${((el / (i + 1)) * (START + FRAMES - i - 1) / 60).toFixed(1)} min`);
}
const stats = await page.evaluate(() => FIRE.captureStats());
fs.writeFileSync(OUT.replace(/.mp4$/, '') + '_frames.json', JSON.stringify(stats.filter(s => s.frame >= START).map(s => ({ ...s, frame: s.frame - START }))));
finishing = true;
if (received !== FRAMES) fail(`received ${received} frames, expected ${FRAMES}`);
ff.stdin.end();
const code = await new Promise(r => ff.on('close', r));
if (code !== 0) fail(`ffmpeg exit ${code}`);
await browser.close();
server.close();
console.log(`wrote ${OUT} (${received} frames) in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
