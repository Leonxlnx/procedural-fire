// Tiny static server + frame sink (HTTP POST and WebSocket) for the WebGPU benchmarks.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { attachWS } from './ws.mjs';

const root = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript' };

export function startServer(port = 8765) {
  let ff = null, frames = 0, bytes = 0;
  const sink = (buf) => new Promise(res => {
    frames++; bytes += buf.length;
    if (!ff) return res('ok');
    if (!ff.stdin.write(buf)) ff.stdin.once('drain', () => res('ok')); else res('ok');
  });
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    if (req.method === 'POST' && url.pathname === '/ffmpeg/start') {
      const w = url.searchParams.get('w'), h = url.searchParams.get('h');
      const pix = url.searchParams.get('pix') || 'rgba';
      const out = url.searchParams.get('out');
      const vf = pix === 'rgba'
        ? ['-vf', 'scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p']
        : [];
      const args = ['-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', pix, '-s', `${w}x${h}`, '-r', '30',
        ...(pix === 'rgba' ? [] : ['-colorspace', 'bt709', '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709']), // tag raw YUV input or ffmpeg 8 re-converts it
        '-i', '-', ...vf,
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-pix_fmt', 'yuv420p',
        '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv', out];
      ff = spawn('ffmpeg', args, { stdio: ['pipe', 'inherit', 'inherit'] });
      frames = 0; bytes = 0;
      res.end('ok'); return;
    }
    if (req.method === 'POST' && url.pathname === '/frame') {
      const chunks = [];
      req.on('data', c => chunks.push(c));
      req.on('end', () => sink(Buffer.concat(chunks)).then(r => res.end(r)));
      return;
    }
    if (req.method === 'POST' && url.pathname === '/ffmpeg/stop') {
      if (ff) { const f = ff; ff = null; f.stdin.end(); f.on('close', code => res.end(JSON.stringify({ code, frames, bytes }))); }
      else res.end(JSON.stringify({ frames, bytes }));
      return;
    }
    const p = path.join(root, decodeURIComponent(url.pathname === '/' ? '/bench.html' : url.pathname));
    fs.readFile(p, (err, data) => {
      if (err) { res.statusCode = 404; res.end('nf'); return; }
      res.setHeader('Content-Type', types[path.extname(p)] || 'application/octet-stream');
      res.end(data);
    });
  });
  attachWS(server, '/ws', sink);
  return new Promise(r => server.listen(port, '127.0.0.1', () => r(server)));
}
