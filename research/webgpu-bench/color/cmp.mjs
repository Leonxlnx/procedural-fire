// Compare decoded rgb24 frames (decoded assuming BT.709 limited) against the source at patch centres.
import fs from 'node:fs';
const W = 1080, H = 1920, PW = 270, PH = 320;
const src = fs.readFileSync('src.rgba');
const patches = JSON.parse(fs.readFileSync('patches.json'));
for (const f of process.argv.slice(2)) {
  const d = fs.readFileSync(f);
  let maxErr = 0, sumErr = 0, n = 0; const worst = [];
  for (let k = 0; k < 12; k++) {
    const x = (k % 4) * PW + PW / 2, y = Math.floor(k / 4) * PH + PH / 2;
    const i = y * W + x;
    const s = [src[i * 4], src[i * 4 + 1], src[i * 4 + 2]], o = [d[i * 3], d[i * 3 + 1], d[i * 3 + 2]];
    const e = Math.max(...s.map((v, j) => Math.abs(v - o[j])));
    maxErr = Math.max(maxErr, e); sumErr += e; n++;
    if (k < 4) worst.push(`${s.join(',')}->${o.join(',')}`);
  }
  console.log(f.padEnd(14), 'max|err|', String(maxErr).padStart(3), ' mean', (sumErr / n).toFixed(1), ' fire patches:', worst.join('  '));
}
