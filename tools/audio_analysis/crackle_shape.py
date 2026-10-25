#!/usr/bin/env python
"""
Per-crackle shape statistics: what does one isolated crackle look like in a real fire?

Usage
-----
  python -I crackle_shape.py [--refs references.json] [wav ...] [--zoom-dir DIR] [--json out.json]

For every input (all non-excluded references from references.json, plus any extra WAVs)
it detects transients the same way texture_stats.py does (>= 1.5 kHz, 0.5 ms frame peaks,
ratio to a 250 ms running median), keeps strong (>= 15 dB) onsets that are isolated
(no other >= 6 dB onset within -40..+25 ms), and measures

  excess spectrum   octave-band power of an 8 ms window starting 1 ms before the onset,
                    minus the power of the 40 ms of background before it (dB, re the max band)
  decay             time for the >1.5 kHz envelope to fall 10 / 20 dB below its peak
  sub-peaks         local maxima >= peak-12 dB within 30 ms after the onset
  stereo (stereo files only): max |normalised cross-correlation| of L/R > 1.5 kHz in the
                    crackle window and the lag at which it occurs

It also writes 2 s zoomed spectrograms (0-16 kHz, 2.7 ms hop) for visual comparison.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                       # noqa: E402
from scipy import signal                 # noqa: E402
import texture_stats as ts               # noqa: E402

FS = ts.FS
OCT = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def onsets(mono):
    x = signal.sosfiltfilt(ts.butter_sos("highpass", ts.TRANSIENT_HP), mono)
    hop = 24
    nfr = x.size // hop
    pk = np.abs(x[: nfr * hop]).reshape(nfr, hop).max(axis=1)
    frs = FS / hop
    win, step = int(0.25 * frs), int(0.01 * frs)
    padded = np.pad(pk, (win // 2, win - win // 2), mode="reflect")
    view = np.lib.stride_tricks.sliding_window_view(padded, win)[::step]
    bg = np.interp(np.arange(nfr), np.arange(view.shape[0]) * step, np.median(view, axis=1))
    ratio = 20 * np.log10(np.maximum(pk, 1e-12) / np.maximum(bg, 1e-12))
    p, pr = signal.find_peaks(ratio, height=6, distance=int(0.005 * frs))
    return x, pk, bg, ratio, p, pr["peak_heights"], frs, hop


def band_power(seg, nfft=4096):
    w = np.hanning(seg.size)
    S = np.abs(np.fft.rfft(seg * w, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / FS)
    out = []
    for fc in OCT:
        m = (f >= fc / math.sqrt(2)) & (f < min(fc * math.sqrt(2), FS / 2))
        out.append(S[m].sum() / max(seg.size, 1))
    return np.array(out)


def analyse(chans, name, zoom_dir=None):
    mono = chans.mean(axis=0)
    mono = mono / math.sqrt(np.mean(mono ** 2)) * 0.1
    x, pk, bg, ratio, p, h, frs, hop = onsets(mono)
    t = p / frs
    strong = []
    for i in range(p.size):
        if h[i] < 15:
            continue
        near = (np.abs(t - t[i]) < 0.04) & (t - t[i] > -0.04) & (t - t[i] < 0.025)
        near[i] = False
        if near.any() or t[i] < 0.1 or t[i] > mono.size / FS - 0.1:
            continue
        strong.append(i)
    strong = sorted(strong, key=lambda i: -h[i])[:150]
    exc, dec10, dec20, subs, xc, lag = [], [], [], [], [], []
    stereo = chans.shape[0] >= 2 and np.corrcoef(chans[0], chans[-1])[0, 1] < 0.98
    if stereo:
        hpL = signal.sosfiltfilt(ts.butter_sos("highpass", ts.TRANSIENT_HP), chans[0])
        hpR = signal.sosfiltfilt(ts.butter_sos("highpass", ts.TRANSIENT_HP), chans[1])
    for i in strong:
        n0 = p[i] * hop
        a = n0 - int(0.001 * FS)
        seg = mono[a: a + int(0.008 * FS)]
        bgp = np.mean([band_power(mono[b: b + int(0.008 * FS)]) for b in range(n0 - int(0.045 * FS), n0 - int(0.008 * FS), int(0.008 * FS))], axis=0)
        cp = band_power(seg)
        e = 10 * np.log10(np.maximum(cp - bgp, 1e-12 * cp.max()))
        exc.append(e - e.max())
        k0 = p[i]
        peak = pk[k0]
        tail = pk[k0: k0 + int(0.06 * frs)] / peak
        b_rel = bg[k0] / peak
        def first_below(db):
            thr = max(10 ** (db / 20), b_rel * 1.4)
            idx = np.nonzero(tail < thr)[0]
            return idx[0] / frs * 1000 if idx.size else 60.0
        dec10.append(first_below(-10))
        dec20.append(first_below(-20))
        w = pk[k0: k0 + int(0.03 * frs)]
        lp, _ = signal.find_peaks(w, height=peak * 10 ** (-12 / 20), distance=int(0.0015 * frs))
        subs.append(1 + int((lp > 2).sum()))
        if stereo:
            L = hpL[a: a + int(0.008 * FS)]
            R = hpR[a: a + int(0.008 * FS)]
            c = signal.correlate(L, R, mode="full") / math.sqrt(np.sum(L * L) * np.sum(R * R) + 1e-30)
            lags = np.arange(-L.size + 1, L.size)
            m = np.abs(lags) <= int(0.0015 * FS)
            j = np.argmax(np.abs(c[m]))
            xc.append(float(c[m][j]))
            lag.append(float(lags[m][j] / FS * 1000))
    # aligned envelope profiles (1 ms RMS frames, dB above the pre-onset floor), median over the strong onsets
    LAGS = [-5, 0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]
    prof = {}
    for bname, (flo, fhi) in {"mid_1k_2k": (1000, 2000), "high_4k_8k": (4000, 8000), "low_150_400": (150, 400)}.items():
        y = signal.sosfiltfilt(ts.butter_sos("bandpass", [flo, fhi], FS, 3), mono)
        rows = []
        for i in strong:
            n0 = p[i] * hop
            a0, a1 = n0 - int(0.07 * FS), n0 + int(0.15 * FS)
            if a0 < 0 or a1 > y.size:
                continue
            seg = y[a0:a1] ** 2
            fr = seg[: (seg.size // 48) * 48].reshape(-1, 48).mean(axis=1)
            fdb = 10 * np.log10(fr + 1e-20)
            floor = np.median(fdb[:60])                       # -70 .. -10 ms
            rows.append(fdb[70 - 10: 70 + 150] - floor)
        if rows:
            m = np.median(np.array(rows), axis=0)
            prof[bname] = {str(l): round(float(np.max(m[l + 10: l + 11])), 1) for l in LAGS}
            prof[bname]["peak"] = round(float(m[8:16].max()), 1)
    exc = np.array(exc) if exc else np.zeros((0, len(OCT)))
    res = {
        "name": name, "n_isolated_strong": len(strong),
        "excess_spectrum_db_median": np.median(exc, axis=0).round(1).tolist() if len(exc) else None,
        "excess_spectrum_db_p25": np.percentile(exc, 25, axis=0).round(1).tolist() if len(exc) else None,
        "excess_spectrum_db_p75": np.percentile(exc, 75, axis=0).round(1).tolist() if len(exc) else None,
        "decay10_ms_median": float(np.median(dec10)) if dec10 else None,
        "decay20_ms_median": float(np.median(dec20)) if dec20 else None,
        "decay20_ms_p25_p75": [float(np.percentile(dec20, 25)), float(np.percentile(dec20, 75))] if dec20 else None,
        "subpeaks_mean": float(np.mean(subs)) if subs else None,
        "stereo_hf_xcorr_median": float(np.median(xc)) if xc else None,
        "stereo_hf_abs_lag_ms_median": float(np.median(np.abs(lag))) if lag else None,
        "octave_centers_hz": OCT,
        "profile_db_above_floor_at_ms": prof,
    }
    if zoom_dir:
        os.makedirs(zoom_dir, exist_ok=True)
        # 2 s window around the densest strong-crackle region
        if strong:
            tt = np.sort(t[strong])
            best = max(range(len(tt)), key=lambda k: np.sum((tt >= tt[k]) & (tt < tt[k] + 2)))
            s0 = max(0.0, min(tt[best] - 0.2, mono.size / FS - 2))
        else:
            s0 = 0.0
        zoom(mono[int(s0 * FS): int((s0 + 2) * FS)], os.path.join(zoom_dir, f"{name}_zoom2s.png"), f"{name} @ {s0:.1f}s")
    return res


def zoom(x, path, title, fmax=16000, dyn=80):
    from PIL import Image, ImageDraw
    f, tt, S = signal.spectrogram(x, FS, window="hann", nperseg=512, noverlap=512 - 128, mode="psd")
    m = f <= fmax
    S = ts.db(S[m])
    top = np.percentile(S, 99.9)
    img = ts.colormap((S - (top - dyn)) / dyn)[::-1]
    W, H = 1200, 420
    im = Image.fromarray(img).resize((W, H), Image.BILINEAR)
    L, T, Rm, B = 64, 28, 16, 30
    canvas = Image.new("RGB", (W + L + Rm, H + T + B), (255, 255, 255))
    canvas.paste(im, (L, T))
    d = ImageDraw.Draw(canvas)
    fnt = ts._font()
    d.text((L, 6), f"{title} | 2 s zoom, {dyn} dB, 0-{fmax/1000:g} kHz", fill=(0, 0, 0), font=fnt)
    for khz in range(0, int(fmax / 1000) + 1, 2):
        y = T + H - int(H * khz * 1000 / fmax)
        d.text((8, y - 7), f"{khz:>2} kHz", fill=(0, 0, 0), font=fnt)
    for s in np.arange(0, 2.0001, 0.25):
        xx = L + int(W * s / 2)
        d.line([(xx, T + H), (xx, T + H + 5)], fill=(0, 0, 0))
        d.text((xx - 8, T + H + 8), f"{s:g}", fill=(0, 0, 0), font=fnt)
    canvas.save(path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="*")
    ap.add_argument("--refs", default=None, help="references.json (analyse every non-excluded reference)")
    ap.add_argument("--zoom-dir", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None)
    a = ap.parse_args()
    out = []
    if a.refs:
        cfg = json.load(open(a.refs, encoding="utf-8"))
        for r in cfg["references"]:
            if r.get("exclude"):
                continue
            ch = ts.load_wav(os.path.expandvars(r["wav"]), r.get("start", 0.0), r.get("dur"))
            out.append(analyse(ch, r["name"], a.zoom_dir))
            print(json.dumps(out[-1]), flush=True)
    for w in a.wav:
        ch = ts.load_wav(w, a.start, a.dur)
        out.append(analyse(ch, os.path.splitext(os.path.basename(w))[0], a.zoom_dir))
        print(json.dumps(out[-1]), flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
