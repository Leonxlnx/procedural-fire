// Generate an RGBA test frame with fire-like patches, plus a CPU reference of the GPU YUV shader math.
import fs from 'node:fs';
const W = 1080, H = 1920;
const patches = [[255,80,0],[255,160,20],[255,220,120],[200,30,0],[40,10,5],[255,0,0],[0,255,0],[0,0,255],[128,128,128],[255,255,255],[16,16,16],[90,40,20]];
const rgba = Buffer.alloc(W * H * 4);
const PW = 270, PH = 320; // 4 x 3 grid of patches + ramp below
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  let c;
  const py = Math.floor(y / PH), px = Math.floor(x / PW);
  if (py < 3) c = patches[py * 4 + px];
  else { const v = Math.round((x / (W - 1)) * 255); c = [v, Math.round(v * 0.45), Math.round(v * 0.1)]; }
  const i = (y * W + x) * 4; rgba[i] = c[0]; rgba[i + 1] = c[1]; rgba[i + 2] = c[2]; rgba[i + 3] = 255;
}
fs.writeFileSync('src.rgba', rgba);
// GPU-shader-equivalent BT.709 limited conversion (no dither) -> yuv420p
const Y = Buffer.alloc(W * H), U = Buffer.alloc(W * H / 4), V = Buffer.alloc(W * H / 4);
const px = (x, y) => { const i = (y * W + x) * 4; return [rgba[i] / 255, rgba[i + 1] / 255, rgba[i + 2] / 255]; };
const q = v => Math.max(0, Math.min(255, Math.round(v)));
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const [r, g, b] = px(x, y); Y[y * W + x] = q(16 + 219 * (0.2126 * r + 0.7152 * g + 0.0722 * b)); }
for (let y = 0; y < H / 2; y++) for (let x = 0; x < W / 2; x++) {
  let r = 0, g = 0, b = 0;
  for (const [dx, dy] of [[0,0],[1,0],[0,1],[1,1]]) { const c = px(2 * x + dx, 2 * y + dy); r += c[0] / 4; g += c[1] / 4; b += c[2] / 4; }
  const yy = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  U[y * W / 2 + x] = q(128 + 224 * (b - yy) / 1.8556); V[y * W / 2 + x] = q(128 + 224 * (r - yy) / 1.5748);
}
fs.writeFileSync('gpu.yuv', Buffer.concat([Y, U, V]));
fs.writeFileSync('patches.json', JSON.stringify(patches));
