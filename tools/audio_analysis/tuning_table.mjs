#!/usr/bin/env node
// tuning_table.mjs — markdown table of deviation index and per-category ratios (score / typical real-vs-real score)
// for a list of compare.py --json outputs.   node tools/audio_analysis/tuning_table.mjs a_compare.json b_compare.json ...
import fs from 'node:fs';
import path from 'node:path';
const files = process.argv.slice(2);
const C = files.map((f) => ({ name: path.basename(f).replace(/_compare\.json$/, ''), ...JSON.parse(fs.readFileSync(f, 'utf8')) }));
const cats = Object.keys(C[0].category_scores);
const r = (c, k) => (c.category_scores[k] / c.category_real_loo_mean[k]).toFixed(2);
console.log('| run | index | out of range | ' + cats.join(' | ') + ' |');
console.log('|---|---|---|' + cats.map(() => '---').join('|') + '|');
for (const c of C) console.log(`| ${c.name} | **${c.deviation_index.toFixed(3)}** | ${c.n_out}/${c.rows.length} | ` + cats.map((k) => r(c, k)).join(' | ') + ' |');
