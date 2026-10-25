#!/usr/bin/env python
"""
Compare one texture_stats.py result against the real-fire reference.

Usage
-----
  python -I compare.py <stats.json> [--ref reference_stats.json] [--top 25] [--all]
                       [--category NAME] [--json out.json]

For every feature it prints the candidate value, the reference mean and [min, max],
a z-score (x - mean) / scale and an OUT flag when the value is outside the range of
the real recordings. scale = max(reference std, a per-feature floor), so a feature on
which the few references happen to agree closely cannot blow up the score. Features
that are positive and multiplicative by nature (rates, kurtosis, Hz, Fano factors)
are compared in log space.

Scores
  category score : mean |z| (clipped at 6) over the category's features.
  raw score      : mean of the category scores (so the 32 per-band features do not dominate).
  calibration    : every real recording is scored leave-one-out against the others, which
                   gives the typical ("real avg") and worst ("real max") category score a
                   genuine fire recording gets. The references differ a lot (small campfire vs
                   bonfire, recorder noise, lossy codecs), so raw scores must be read against it.
  deviation index: mean over categories of score / real avg. ~1 = as different as one real
                   recording is from the others; categories above "real max" are flagged.
"""
import argparse
import json
import math
import os
import sys

LOG_KEYS = ("rate_", "kurt", "centroid", "rolloff", "fano", "okurt", "energy_frac")


def is_log(k):
    return any(s in k for s in LOG_KEYS)


def floor_for(k, cat):
    if is_log(k):
        return 0.15                      # ~15 % in log space
    if cat in ("ltas", "modulation", "dynamics") or k.endswith("_db") or "_db_" in k:
        return 1.5 if "slope" not in k else 0.5
    if cat == "cross_band_corr" or k.startswith("stereo.corr"):
        return 0.04
    return None                           # relative floor, see scale()


def tf(k, v):
    if is_log(k):
        return math.log(max(v, 1e-9))
    return v


def stats_of(values):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return m, sd, min(values), max(values)


def score(features, ref_rows, cats):
    """features: {k: v}; ref_rows: list of {k: v} (the reference recordings)."""
    rows = []
    for k, cat in cats.items():
        if k not in features:
            continue
        vals = [tf(k, r[k]) for r in ref_rows if k in r]
        if len(vals) < 2:
            continue
        m, sd, lo, hi = stats_of(vals)
        fl = floor_for(k, cat)
        if fl is None:
            fl = 0.05 * abs(m) + 0.02
        s = max(sd, fl)
        x = tf(k, features[k])
        z = (x - m) / s
        excess = (x - hi) / s if x > hi else (lo - x) / s if x < lo else 0.0
        rows.append({"feature": k, "category": cat, "value": features[k], "z": z,
                     "excess": excess, "out": excess > 0,
                     "ref_mean": (math.exp(m) if is_log(k) else m),
                     "ref_min": (math.exp(lo) if is_log(k) else lo),
                     "ref_max": (math.exp(hi) if is_log(k) else hi),
                     "log": is_log(k)})
    bycat = {}
    for r in rows:
        bycat.setdefault(r["category"], []).append(min(abs(r["z"]), 6.0))
    cat_scores = {c: sum(v) / len(v) for c, v in bycat.items()}
    overall = sum(cat_scores.values()) / max(len(cat_scores), 1)
    return rows, cat_scores, overall


def fmt(v):
    if v is None:
        return "-"
    a = abs(v)
    if a >= 1000:
        return f"{v:.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats")
    ap.add_argument("--ref", default=os.path.join(here, "reference_stats.json"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--all", action="store_true", help="print every feature")
    ap.add_argument("--category", help="only print this category")
    ap.add_argument("--json", help="write the full comparison here")
    a = ap.parse_args()

    ref = json.load(open(a.ref, encoding="utf-8"))
    cand = json.load(open(a.stats, encoding="utf-8"))
    feats = cand["features"]
    cats = {k: v["category"] for k, v in ref["features"].items()}
    per_ref = ref["per_reference"]
    rows, cat_scores, overall = score(feats, list(per_ref.values()), cats)

    # leave-one-out calibration
    loo, loo_out, loo_cat = {}, {}, {}
    for name, f in per_ref.items():
        others = [g for n, g in per_ref.items() if n != name]
        r_, cs, s = score(f, others, cats)
        loo[name] = s
        loo_out[name] = sum(r["out"] for r in r_)
        loo_cat[name] = cs
    cat_loo_mean = {c: sum(v[c] for v in loo_cat.values() if c in v) /
                    max(1, sum(1 for v in loo_cat.values() if c in v)) for c in cat_scores}
    cat_loo_max = {c: max(v[c] for v in loo_cat.values() if c in v) for c in cat_scores}

    def realism(cs):
        """mean over categories of score / (typical leave-one-out score of a real recording)."""
        q = [cs[c] / max(cat_loo_mean[c], 1e-6) for c in cs if c in cat_loo_mean]
        return sum(q) / max(len(q), 1)

    index = realism(cat_scores)
    ref_index = sorted(realism(cs) for cs in loo_cat.values())

    n_out = sum(r["out"] for r in rows)
    print(f"candidate : {cand.get('name')}  ({cand.get('duration_s', 0):.1f} s, {cand.get('channels')} ch)")
    print(f"reference : {len(per_ref)} real recordings ({', '.join(per_ref)})")
    print(f"DEVIATION INDEX {index:.2f}   (category score / typical real-vs-real score, averaged; "
          f"real recordings: {ref_index[0]:.2f}-{ref_index[-1]:.2f}; lower is better)")
    lv = sorted(loo.values())
    print(f"  raw score {overall:.2f} (mean |z| per category); real recordings leave-one-out: "
          f"min {lv[0]:.2f}  mean {sum(lv)/len(lv):.2f}  max {lv[-1]:.2f}")
    print(f"  features outside the real range: {n_out} / {len(rows)}"
          f"   (each real recording vs the other {len(per_ref) - 1}: "
          f"{min(loo_out.values())}-{max(loo_out.values())}, mean {sum(loo_out.values()) / len(loo_out):.0f})")
    print("  (for log-scaled features - rates, kurtosis, Hz, Fano - 'ref mean' is the geometric mean)")
    print()
    print(f"  {'category':<18} {'score':>6} {'real avg':>8} {'real max':>8} {'ratio':>6}  out")
    for c, s in sorted(cat_scores.items(), key=lambda x: -x[1] / max(cat_loo_mean[x[0]], 1e-6)):
        nc = sum(1 for r in rows if r["category"] == c)
        no = sum(1 for r in rows if r["category"] == c and r["out"])
        worse = "  << worse than every real recording" if s > cat_loo_max[c] else ""
        print(f"  {c:<18} {s:6.2f} {cat_loo_mean[c]:8.2f} {cat_loo_max[c]:8.2f} "
              f"{s / max(cat_loo_mean[c], 1e-6):6.2f}  {no:>2}/{nc}{worse}")
    print()

    sel = rows if a.all else sorted(rows, key=lambda r: (-r["excess"], -abs(r["z"])))[: a.top]
    if a.category:
        sel = [r for r in rows if r["category"] == a.category]
    if not a.all and not a.category:
        print(f"top {len(sel)} discrepancies (sorted by distance outside the real range, then |z|):")
    print(f"  {'feature':<36} {'value':>9} {'ref mean':>9} {'ref min':>9} {'ref max':>9} {'z':>7}  flag")
    for r in sel:
        flag = ("OUT " + ("high" if r["z"] > 0 else "low")) if r["out"] else ""
        print(f"  {r['feature']:<36} {fmt(r['value']):>9} {fmt(r['ref_mean']):>9} {fmt(r['ref_min']):>9} "
              f"{fmt(r['ref_max']):>9} {r['z']:>7.2f}  {flag}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"candidate": cand.get("name"), "deviation_index": index, "raw_score": overall,
                       "category_scores": cat_scores, "category_real_loo_mean": cat_loo_mean,
                       "category_real_loo_max": cat_loo_max, "real_deviation_index": ref_index,
                       "loo_reference_scores": loo, "loo_reference_n_out": loo_out, "n_out": n_out,
                       "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
