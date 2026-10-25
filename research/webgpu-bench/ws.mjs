// Minimal dependency-free WebSocket server endpoint (binary messages in, short text acks out).
import crypto from 'node:crypto';

export function attachWS(server, path, onMessage) {
  server.on('upgrade', (req, socket) => {
    if (!req.url.startsWith(path)) { socket.destroy(); return; }
    const key = req.headers['sec-websocket-key'];
    const accept = crypto.createHash('sha1').update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
    socket.write('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n' +
      `Sec-WebSocket-Accept: ${accept}\r\n\r\n`);
    socket.setNoDelay(true);
    let chunks = [], total = 0, frags = [];
    const send = (text) => { const p = Buffer.from(text); socket.write(Buffer.concat([Buffer.from([0x81, p.length]), p])); };
    socket.on('data', chunk => {
      chunks.push(chunk); total += chunk.length;
      for (;;) {
        if (total < 2) return;
        if (chunks[0].length < 14 && chunks.length > 1) { chunks = [Buffer.concat(chunks, total)]; }
        const h = chunks[0];
        if (h.length < 2) return;
        const fin = h[0] & 0x80, op = h[0] & 0x0f, masked = h[1] & 0x80;
        let len = h[1] & 0x7f, off = 2;
        if (len === 126) { if (h.length < 4) return; len = h.readUInt16BE(2); off = 4; }
        else if (len === 127) { if (h.length < 10) return; len = Number(h.readBigUInt64BE(2)); off = 10; }
        const mo = off, po = off + (masked ? 4 : 0), need = po + len;
        if (total < need) return;                       // wait for the whole frame, concat once
        const all = chunks.length === 1 ? chunks[0] : Buffer.concat(chunks, total);
        const payload = Buffer.allocUnsafeSlow(len);    // 4-byte aligned, fresh
        all.copy(payload, 0, po, need);
        if (masked) {
          const m = all.subarray(mo, mo + 4), m32 = m.readUInt32LE(0), n4 = len >> 2;
          const u32 = new Uint32Array(payload.buffer, 0, n4);
          for (let i = 0; i < n4; i++) u32[i] ^= m32;
          for (let i = n4 * 4; i < len; i++) payload[i] ^= m[i & 3];
        }
        const rest = all.subarray(need);
        chunks = rest.length ? [rest] : []; total = rest.length;
        if (op === 0x8) { socket.end(); return; }
        if (op === 0x9 || op === 0xA) continue;
        frags.push(payload);
        if (fin) { const msg = frags.length === 1 ? frags[0] : Buffer.concat(frags); frags = []; Promise.resolve(onMessage(msg)).then(r => send(r ?? 'ok')); }
      }
    });
  });
}
