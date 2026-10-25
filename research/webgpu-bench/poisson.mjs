// CPU convergence study for the pressure Poisson solve used by the GPU kernels.
// Same discretisation & BCs as the WGSL: 7-point Laplacian, cell-centred, solid floor (Neumann, ghost = p_c),
// open sides/top (Dirichlet; ghost = 0 like the shader, or ghost = -p_c "face" Dirichlet).
// Usage: node poisson.mjs N [ghostMode=zero|face]
const N = +(process.argv[2] || 64);
const GHOST = process.argv[3] || 'zero';

function makeLevel(n) { return { n, p: new Float32Array(n * n * n), b: new Float32Array(n * n * n), r: new Float32Array(n * n * n) }; }
const I = (n, x, y, z) => x + n * (y + n * z);
function nbSum(L, x, y, z, pc) {
  const { n, p } = L;
  const g = (xx, yy, zz) => {
    if (yy < 0) return pc;                                              // floor: Neumann
    if (xx < 0 || yy >= n || zz < 0 || xx >= n || zz >= n) return GHOST === 'face' ? -pc : 0; // open
    return p[I(n, xx, yy, zz)];
  };
  return g(x - 1, y, z) + g(x + 1, y, z) + g(x, y - 1, z) + g(x, y + 1, z) + g(x, y, z - 1) + g(x, y, z + 1);
}
// Equation per level: sum_nb(p) - 6 p = h2 * b
function jacobi(L, h2, omega, tmp) {
  const { n, p, b } = L;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = I(n, x, y, z), pc = p[i];
    tmp[i] = pc + omega * ((nbSum(L, x, y, z, pc) - h2 * b[i]) / 6 - pc);
  }
  p.set(tmp);
}
function rbgs(L, h2, omega) {
  const { n, p, b } = L;
  for (let color = 0; color < 2; color++)
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = (y + z + color) & 1; x < n; x += 2) {
      const i = I(n, x, y, z), pc = p[i];
      p[i] = pc + omega * ((nbSum(L, x, y, z, pc) - h2 * b[i]) / 6 - pc);
    }
}
function residual(L, h2) {
  const { n, p, b, r } = L; let s = 0;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = I(n, x, y, z), pc = p[i];
    r[i] = b[i] - (nbSum(L, x, y, z, pc) - 6 * pc) / h2; s += r[i] * r[i];
  }
  return Math.sqrt(s / (n * n * n));
}
function restrict(F, C) { // average of 8 children
  const n = C.n;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let s = 0;
    for (let k = 0; k < 8; k++) s += F.r[I(F.n, 2 * x + (k & 1), 2 * y + ((k >> 1) & 1), 2 * z + ((k >> 2) & 1))];
    C.b[I(n, x, y, z)] = s / 8; C.p[I(n, x, y, z)] = 0;
  }
}
function prolongAdd(C, F) { // trilinear (3/4, 1/4 per axis) for cell-centred grids
  const n = C.n, cl = v => Math.max(0, Math.min(n - 1, v));
  for (let z = 0; z < F.n; z++) for (let y = 0; y < F.n; y++) for (let x = 0; x < F.n; x++) {
    const cx = x >> 1, cy = y >> 1, cz = z >> 1;
    const ox = (x & 1) ? 1 : -1, oy = (y & 1) ? 1 : -1, oz = (z & 1) ? 1 : -1;
    let v = 0;
    for (let k = 0; k < 8; k++) {
      const dx = k & 1, dy = (k >> 1) & 1, dz = (k >> 2) & 1;
      const w = (dx ? 0.25 : 0.75) * (dy ? 0.25 : 0.75) * (dz ? 0.25 : 0.75);
      v += w * C.p[I(n, cl(cx + dx * ox), cl(cy + dy * oy), cl(cz + dz * oz))];
    }
    F.p[I(F.n, x, y, z)] += v;
  }
}
function vcycle(levels, l, pre = 2, post = 2) {
  const L = levels[l], h2 = 4 ** l;
  if (l === levels.length - 1) { for (let i = 0; i < 30; i++) rbgs(L, h2, 1.0); return; }
  for (let i = 0; i < pre; i++) rbgs(L, h2, 1.0);
  residual(L, h2);
  restrict(L, levels[l + 1]);
  vcycle(levels, l + 1, pre, post);
  prolongAdd(levels[l + 1], L);
  for (let i = 0; i < post; i++) rbgs(L, h2, 1.0);
}

// Right-hand side: divergence of a buoyant, swirly plume velocity field (cells/step units)
function rhs() {
  const b = new Float32Array(N * N * N);
  const vel = (x, y, z) => {
    const u = (x + 0.5) / N, v = (y + 0.5) / N, w = (z + 0.5) / N;
    const r = Math.hypot(u - 0.5, w - 0.5);
    const plume = Math.exp(-(r * r) / 0.01) * Math.min(1, v * 4);
    return [Math.sin(v * 12) * 0.3 + Math.sin(w * 31 + u * 7) * 0.2, 1.5 * plume + Math.cos(u * 9) * 0.2, Math.sin(w * 7 + u * 5) * 0.3];
  };
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const c = (a, b2, cc) => vel(Math.max(0, Math.min(N - 1, a)), Math.max(0, Math.min(N - 1, b2)), Math.max(0, Math.min(N - 1, cc)));
    const vD = y === 0 ? -vel(x, y, z)[1] : c(x, y - 1, z)[1];
    b[I(N, x, y, z)] = 0.5 * ((c(x + 1, y, z)[0] - c(x - 1, y, z)[0]) + (c(x, y + 1, z)[1] - vD) + (c(x, y, z + 1)[2] - c(x, y, z - 1)[2]));
  }
  return b;
}
const B = rhs();
const fresh = () => { const L = makeLevel(N); L.b.set(B); return L; };
const r0 = (() => { const L = fresh(); return residual(L, 1); })();
const report = (name, costPerIter, hist) => console.log(name.padEnd(30), hist.map(([k, r]) => `${k}:${(r / r0).toExponential(1)}`).join('  '), ` (cost/iter = ${costPerIter} Jacobi-eq)`);

console.log(`N=${N}^3 ghost=${GHOST}  initial rms residual ${r0.toExponential(3)}; values = residual / initial`);
const checkpoints = [5, 10, 20, 40, 80, 160];
for (const omega of [1.0, 6 / 7]) {
  const L = fresh(), tmp = new Float32Array(N ** 3), hist = [];
  for (let k = 1; k <= 160; k++) { jacobi(L, 1, omega, tmp); if (checkpoints.includes(k)) hist.push([k, residual(L, 1)]); }
  report(`Jacobi w=${omega.toFixed(3)}`, 1, hist);
}
const wOpt = 2 / (1 + Math.sin(Math.PI / N));
for (const omega of [1.0, 1.5, 1.7, 1.85, +wOpt.toFixed(3)]) {
  const L = fresh(), hist = [];
  for (let k = 1; k <= 160; k++) { rbgs(L, 1, omega); if (checkpoints.includes(k)) hist.push([k, residual(L, 1)]); }
  report(`RB-SOR w=${omega}`, 2, hist);
}
{
  const levels = []; for (let n = N; n >= 4; n >>= 1) levels.push(makeLevel(n));
  levels[0].b.set(B);
  const hist = [];
  for (let k = 1; k <= 8; k++) { vcycle(levels, 0); hist.push([k, residual(levels[0], 1)]); }
  report(`MG V(2,2) RBGS, ${levels.length} lvls`, '~12', hist);
}
{
  const levels = []; for (let n = N; n >= 4; n >>= 1) levels.push(makeLevel(n));
  levels[0].b.set(B);
  const hist = [];
  for (let k = 1; k <= 8; k++) { vcycle(levels, 0, 1, 1); hist.push([k, residual(levels[0], 1)]); }
  report(`MG V(1,1) RBGS, ${levels.length} lvls`, '~7', hist);
}
