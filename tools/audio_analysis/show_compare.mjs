#!/usr/bin/env node
// show_compare.mjs — print the rows of one or more compare.py --json outputs side by side (worst |z| first).
//   node tools/audio_analysis/show_compare.mjs a_compare.json [b_compare.json ...] [--top 40] [--filter regex] [--no-bands]
import fs from 'node:fs';
const a = process.argv.slice(2), val = (k, d) => { const i = a.indexOf('--' + k); return i >= 0 ? a[i + 1] : d; };
const files = a.filter((x, i) => !x.startsWith('--') && !(i > 0 && a[i - 1].startsWith('--') && ['--top', '--filter'].includes(a[i - 1])));
const C = files.map((f) => JSON.parse(fs.readFileSync(f, 'utf8')));
const top = +val('top', 40), re = val('filter') ? new RegExp(val('filter')) : null;
console.log('index: ' + C.map((c) => c.candidate + ' ' + c.deviation_index.toFixed(3)).join(' | '));
const cats = Object.keys(C[0].category_scores);
for (const k of cats) console.log('  ' + k.padEnd(18) + C.map((c) => (c.category_scores[k] / c.category_real_loo_mean[k]).toFixed(2).padStart(7)).join(''));
const rows = C[0].rows.filter((r) => (!re || re.test(r.feature)) && !(a.includes('--no-bands') && r.category === 'envelope_bands'));
const maps = C.map((c) => new Map(c.rows.map((r) => [r.feature, r])));
rows.sort((x, y) => Math.abs(y.z) - Math.abs(x.z));
for (const r of rows.slice(0, top)) {
  const f = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));
  console.log(r.feature.padEnd(30) + maps.map((m) => { const q = m.get(r.feature); return q ? (f(+q.value) + (q.out ? '*' : ' ')).padStart(10) + q.z.toFixed(1).padStart(6) : ''.padStart(16); }).join('') + '   ref ' + f(r.ref_mean) + ' [' + f(r.ref_min) + ', ' + f(r.ref_max) + ']');
}
