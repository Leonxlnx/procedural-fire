// Extract the benchmarked WGSL kernels from bench.js into wgsl/sim_reference.wgsl (one module per kernel).
import fs from 'node:fs';
const src = fs.readFileSync('bench.js', 'utf8');
const bt = String.fromCharCode(96);
const header = src.split('const HEADER = /* wgsl */' + bt)[1].split(bt + ';')[0];
const names = ['advect', 'mccorrect', 'div', 'jacobi', 'jacobi2', 'rbgs', 'project', 'curl', 'confine'];
let out = '// Fluid kernels exactly as benchmarked in bench.js (workgroup 8x8x4, rgba16float fields).\n' +
  '// Binding numbers overlap between kernels: create one shader module per section.\n';
for (const n of names) {
  const start = src.indexOf('\n' + n + ': ' + bt);
  if (start < 0) { console.error('missing', n); continue; }
  const body = src.slice(start + n.length + 4).split(bt + ',')[0];
  out += `\n// ======================= ${n} =======================\n` + header +
    body.replaceAll('WX', '8').replaceAll('WY', '8').replaceAll('WZ', '4').replaceAll('FMT', 'rgba16float') + '\n';
}
fs.mkdirSync('wgsl', { recursive: true });
fs.writeFileSync('wgsl/sim_reference.wgsl', out);
console.log('kernels:', (out.match(/@compute/g) || []).length);
