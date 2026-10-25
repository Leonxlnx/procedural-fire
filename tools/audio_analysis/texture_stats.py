#!/usr/bin/env python
"""
Sound-texture statistics for fire recordings, after McDermott & Simoncelli (2011).

Usage
-----
  python -I texture_stats.py <in.wav> [more.wav ...] [--out stats.json] [--png-dir DIR]
                             [--name NAME] [--start S] [--dur D] [--no-png]
  (default outputs: stats/<name>.json and png/<name>_*.png next to this script)
  python -I texture_stats.py --build-reference references.json [--out reference_stats.json]

Each analysis writes one JSON with
  "features": flat {name: float}, the values compare.py uses
  "categories": {name: category}
  "detail": arrays (1/3-oct LTAS, per-band envelope moments, the correlation matrix, ...)
and, unless --no-png is given, <name>_spectrogram.png (0-16 kHz, log magnitude) and
<name>_envelopes.png (ERB band envelopes).

All level-dependent statistics are computed after normalising the mono mix to
-20 dBFS RMS, so recordings with different gains are comparable. Only numpy,
scipy and Pillow are needed.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import median_filter

FS = 48000
ENV_FS = 400                 # envelope sample rate (McDermott & Simoncelli 2011)
COMPRESS = 0.3               # envelope compression exponent (ibid.)
N_BANDS = 32
BAND_LO, BAND_HI = 50.0, 14000.0
GROUPS = {"low": (0, 500), "mid": (500, 3000), "high": (3000, 1e9)}
MOD_CENTERS = [0.5, 1, 2, 4, 8, 16, 32, 64]
THIRD_OCT = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
             1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
OCTAVES = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
ONSET_THRESH_DB = [6, 10, 15, 20]
TRANSIENT_HP = 1500.0        # transient detector works above this
TARGET_RMS = 0.1             # -20 dBFS


# --------------------------------------------------------------------------- I/O
def load_wav(path, start=0.0, dur=None):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    else:
        x = x.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if sr != FS:
        g = math.gcd(int(sr), FS)
        x = signal.resample_poly(x, FS // g, int(sr) // g, axis=0)
    i0 = int(round(start * FS))
    i1 = x.shape[0] if dur is None else min(x.shape[0], i0 + int(round(dur * FS)))
    return x[i0:i1].T.copy()          # (channels, samples)


# ------------------------------------------------------------------ filterbank
def erb_num(f):
    return 9.265 * np.log1p(np.asarray(f, dtype=np.float64) / (24.7 * 9.265))


def erb_inv(e):
    return 24.7 * 9.265 * np.expm1(np.asarray(e, dtype=np.float64) / 9.265)


def erb_filterbank(nfft, n=N_BANDS, lo=BAND_LO, hi=BAND_HI, fs=FS):
    """Half-cosine filters equally spaced on the ERB scale; adjacent filters overlap
    by 50 % and their squared responses sum to 1 between the outer centres."""
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    e = erb_num(f)
    edges = np.linspace(erb_num(lo), erb_num(hi), n + 2)
    H = np.zeros((n, f.size))
    for i in range(n):
        a, c, b = edges[i], edges[i + 1], edges[i + 2]
        m = (e > a) & (e < b)
        H[i, m] = np.cos(np.pi * (e[m] - c) / (b - a))
    return H, erb_inv(edges[1:-1])


def band_envelopes(x, seg_s=30.0, pad_s=1.0):
    """Hilbert envelopes of N_BANDS ERB bands, downsampled to ENV_FS. Processed in
    padded segments so long files fit in memory. Each band's one-sided spectrum is
    shifted to baseband and inverse-transformed at FS/8 = 6 kHz (every band is
    < 3 kHz wide, so the complex analytic signal is exactly represented and its
    magnitude, the envelope, is unchanged by the shift), then resampled to ENV_FS."""
    DEC = 8                                   # 48 kHz -> 6 kHz complex baseband
    seg, pad = int(seg_s * FS), int(pad_s * FS)
    out = []
    cache = {}
    centers = None
    for s0 in range(0, x.size, seg):
        a, b = max(0, s0 - pad), min(x.size, s0 + seg + pad)
        chunk = x[a:b]
        nfft = int(2 ** math.ceil(math.log2(chunk.size)))
        if nfft not in cache:
            cache[nfft] = erb_filterbank(nfft)
        H, centers = cache[nfft]
        X = np.fft.rfft(chunk, nfft)
        M = nfft // DEC
        lead, keep = (s0 - a) // DEC, int(math.ceil(min(seg, x.size - s0) / DEC))
        envs = []
        for i in range(H.shape[0]):
            nz = np.nonzero(H[i])[0]
            ka, kb = nz[0], nz[-1]
            kc = (ka + kb) // 2
            if kb - ka >= M:
                raise RuntimeError("band too wide for baseband decimation")
            Z = np.zeros(M, dtype=np.complex128)
            Z[(np.arange(ka, kb + 1) - kc) % M] = 2.0 * X[ka:kb + 1] * H[i, ka:kb + 1]
            env = np.abs(np.fft.ifft(Z)) * (M / nfft)
            envs.append(env[lead: lead + keep])
        envs = np.asarray(envs)
        envs = signal.resample_poly(envs, 1, FS // DEC // ENV_FS, axis=1)
        out.append(np.maximum(envs, 0.0))
    return np.concatenate(out, axis=1), centers


# --------------------------------------------------------------------- helpers
def moments(v):
    m = v.mean()
    s = v.std()
    if s <= 0 or m == 0:
        return float(m), 0.0, 0.0, 3.0
    z = (v - m) / s
    return float(m), float(s / m), float((z ** 3).mean()), float((z ** 4).mean())


def butter_sos(kind, f, fs=FS, order=4):
    return signal.butter(order, f, btype=kind, fs=fs, output="sos")


def db(x, floor=1e-20):
    return 10.0 * np.log10(np.maximum(x, floor))


def powerlaw_mle(x, xmin):
    """Clauset, Shalizi & Newman (2009) continuous MLE: pdf ~ x^-alpha for x >= xmin."""
    x = np.asarray(x)
    x = x[x >= xmin]
    n = x.size
    if n < 10:
        return None, None, n
    alpha = 1.0 + n / np.sum(np.log(x / xmin))
    err = (alpha - 1.0) / math.sqrt(n)
    return float(alpha), float(err), n


# ------------------------------------------------------------------ statistics
def ltas_third_octave(mono):
    nper = 32768 if mono.size >= 65536 else 8192
    f, P = signal.welch(mono, FS, window="hann", nperseg=nper, noverlap=nper // 2)
    df = f[1] - f[0]
    total = P.sum() * df
    bands = []
    for fc in THIRD_OCT:
        lo, hi = fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6)
        m = (f >= lo) & (f < hi)
        bands.append(P[m].sum() * df if m.any() else 0.0)
    bands = np.asarray(bands)
    rel = db(bands / total)
    cen = float((f * P).sum() / P.sum())
    cum = np.cumsum(P) / P.sum()
    roll85 = float(f[np.searchsorted(cum, 0.85)])
    fc = np.asarray(THIRD_OCT, dtype=float)

    def slope(lo, hi):
        m = (fc >= lo) & (fc <= hi)
        k, _ = np.polyfit(np.log2(fc[m]), rel[m], 1)
        return float(k)

    # effective bandwidth: highest 1/3-oct band whose level is within 50 dB of the max
    top = rel.max()
    eff_bw = float(fc[np.nonzero(rel > top - 50)[0].max()])
    return {"third_oct_db": rel.tolist(), "centroid_hz": cen, "rolloff85_hz": roll85,
            "slope_125_1k": slope(125, 1000), "slope_1k_12k": slope(1000, 12500),
            "slope_100_8k": slope(100, 8000), "eff_bandwidth_hz": eff_bw}


def envelope_stats(env, centers):
    envc = env ** COMPRESS
    n = envc.shape[0]
    mom = np.array([moments(envc[i]) for i in range(n)])          # mean, cv, skew, kurt
    C = np.corrcoef(envc)
    # fast co-modulation: envelopes high-passed at 2 Hz
    hp = signal.sosfiltfilt(butter_sos("highpass", 2.0, ENV_FS, 2), envc, axis=1)
    Cf = np.corrcoef(hp)

    def summarize(M):
        res = {"adjacent": float(np.mean([M[i, i + 1] for i in range(n - 1)]))}
        for octs in (1, 2, 4):
            vals = []
            for i in range(n):
                target = centers[i] * 2 ** octs
                if target > centers[-1] * 1.03:
                    continue
                j = int(np.argmin(np.abs(np.log2(centers / target))))
                if abs(np.log2(centers[j] / target)) < 0.15:
                    vals.append(M[i, j])
            res[f"oct{octs}"] = float(np.mean(vals)) if vals else float("nan")
        iu = np.triu_indices(n, 1)
        res["all"] = float(M[iu].mean())
        return res

    # modulation power, octave bands, as a fraction of each band's envelope variance
    nper = min(int(16 * ENV_FS), envc.shape[1] // 2)
    nper = max(nper, 256)
    f, P = signal.welch(envc - envc.mean(axis=1, keepdims=True), ENV_FS, window="hann",
                        nperseg=nper, noverlap=nper // 2, axis=1)
    var = envc.var(axis=1)
    df = f[1] - f[0]
    modfrac = np.zeros((n, len(MOD_CENTERS)))
    for k, fc in enumerate(MOD_CENTERS):
        m = (f >= fc / math.sqrt(2)) & (f < fc * math.sqrt(2))
        modfrac[:, k] = P[:, m].sum(axis=1) * df / np.maximum(var, 1e-30)
    grp = {}
    for g, (lo, hi) in GROUPS.items():
        sel = (centers >= lo) & (centers < hi)
        grp[g] = {
            "mod_frac_db": db(modfrac[sel].mean(axis=0)).tolist(),
            "env_cv": float(mom[sel, 1].mean()),
            "env_skew": float(mom[sel, 2].mean()),
            "env_kurt": float(mom[sel, 3].mean()),
            "env_mean": float(mom[sel, 0].mean()),
        }
    return {"centers_hz": centers.tolist(), "mean": mom[:, 0].tolist(), "cv": mom[:, 1].tolist(),
            "skew": mom[:, 2].tolist(), "kurt": mom[:, 3].tolist(),
            "corr_matrix": np.round(C, 4).tolist(), "corr": summarize(C),
            "corr_fast": summarize(Cf), "mod_centers_hz": MOD_CENTERS,
            "mod_frac_db": db(modfrac).tolist(), "groups": grp}


def transient_stats(mono, dur_s):
    x = signal.sosfiltfilt(butter_sos("highpass", TRANSIENT_HP), mono)
    hop = 24                                              # 0.5 ms frames -> 2 kHz
    nfr = x.size // hop
    fr = np.abs(x[: nfr * hop]).reshape(nfr, hop)
    pk = fr.max(axis=1)
    en = (x[: nfr * hop] ** 2).reshape(nfr, hop).sum(axis=1)
    frs = FS / hop
    # local background: running median of frame peaks over 250 ms, evaluated every
    # 10 ms and interpolated (much faster than a per-frame median filter)
    win, step = int(0.25 * frs), int(0.01 * frs)
    padded = np.pad(pk, (win // 2, win - win // 2), mode="reflect")
    view = np.lib.stride_tricks.sliding_window_view(padded, win)[::step]
    bgc = np.median(view, axis=1)
    bg = np.interp(np.arange(nfr), np.arange(bgc.size) * step, bgc)
    ratio = 20 * np.log10(np.maximum(pk, 1e-12) / np.maximum(bg, 1e-12))
    min_dist = int(0.005 * frs)
    res = {"rates": {}}
    peaks_all, props = signal.find_peaks(ratio, height=ONSET_THRESH_DB[0], distance=min_dist)
    hts = props["peak_heights"]
    for th in ONSET_THRESH_DB:
        res["rates"][str(th)] = float((hts >= th).sum() / dur_s)
    # power law of peak amplitude (ratio to local background), pdf ~ a^-alpha
    amp = 10 ** (hts / 20)
    a6, e6, n6 = powerlaw_mle(amp, 10 ** (6 / 20))
    a10, e10, n10 = powerlaw_mle(amp, 10 ** (10 / 20))
    res["alpha_amp_6db"] = a6
    res["alpha_amp_10db"] = a10
    res["alpha_amp_6db_err"] = e6
    res["b_energy_6db"] = None if a6 is None else (a6 + 1) / 2   # E ~ a^2 -> pdf(E) ~ E^-(alpha+1)/2
    # absolute-size power law (peak re global median of frame peaks)
    gmed = np.median(pk)
    absamp = pk[peaks_all] / max(gmed, 1e-12)
    aa, _, _ = powerlaw_mle(absamp, 10 ** (10 / 20))
    res["alpha_absamp_10db"] = aa
    # inter-onset intervals / clustering (onsets >= 10 dB)
    for th in (6, 10):
        t = peaks_all[hts >= th] / frs
        key = f"{th}db"
        if t.size >= 8:
            ioi = np.diff(t)
            cv = float(ioi.std() / ioi.mean())
            lv = float(3.0 / (ioi.size - 1) * np.sum(((ioi[:-1] - ioi[1:]) / (ioi[:-1] + ioi[1:])) ** 2))
            cnt, _ = np.histogram(t, bins=np.arange(0, dur_s + 1e-9, 0.1))
            fano = float(cnt.var() / cnt.mean()) if cnt.mean() > 0 else None
            cnt1, _ = np.histogram(t, bins=np.arange(0, dur_s + 1e-9, 1.0))
            fano1 = float(cnt1.var() / cnt1.mean()) if cnt1.mean() > 0 else None
        else:
            cv = lv = fano = fano1 = None
        res[f"ioi_cv_{key}"] = cv
        res[f"ioi_lv_{key}"] = lv
        res[f"fano_100ms_{key}"] = fano
        res[f"fano_1s_{key}"] = fano1
    # fraction of HF energy carried by transient frames (>= 10 dB over background)
    res["energy_frac_10db"] = float(en[ratio >= 10].sum() / max(en.sum(), 1e-30))
    res["energy_frac_6db"] = float(en[ratio >= 6].sum() / max(en.sum(), 1e-30))
    res["onset_times_10db"] = (peaks_all[hts >= 10] / frs).round(4).tolist()[:4000]
    return res


def waveform_stats(chans, mono):
    rms = math.sqrt(np.mean(mono ** 2))
    res = {"crest_db": float(20 * np.log10(np.abs(mono).max() / rms)),
           "p999_db": float(20 * np.log10(np.quantile(np.abs(mono), 0.999) / rms)),
           "kurtosis": float(np.mean((mono / rms) ** 4))}
    okurt = []
    for fc in OCTAVES:
        lo, hi = fc / math.sqrt(2), min(fc * math.sqrt(2), FS / 2 * 0.98)
        y = signal.sosfiltfilt(butter_sos("bandpass", [lo, hi], FS, 3), mono)
        r = np.mean(y ** 2)
        okurt.append(float(np.mean(y ** 4) / r ** 2) if r > 0 else 3.0)
    res["octave_kurtosis"] = okurt
    # slow level variation
    for name, w in (("400ms", 0.4), ("3s", 3.0)):
        n = int(w * FS)
        k = mono.size // n
        if k >= 3:
            lv = 10 * np.log10(np.mean(mono[: k * n].reshape(k, n) ** 2, axis=1) + 1e-20)
            res[f"level_std_{name}_db"] = float(lv.std())
        else:
            res[f"level_std_{name}_db"] = None
    # stereo (files whose channels correlate > 0.98 are treated as mono: they carry no
    # usable stereo-image information and would distort the reference range)
    res["near_mono"] = bool(chans.shape[0] < 2 or np.corrcoef(chans[0], chans[-1])[0, 1] > 0.98)
    if not res["near_mono"]:
        L, R = chans[0], chans[1]
        res["stereo_corr"] = float(np.corrcoef(L, R)[0, 1])
        for g, (lo, hi) in GROUPS.items():
            if lo == 0:
                sos = butter_sos("lowpass", hi)
            elif hi > FS:
                sos = butter_sos("highpass", lo)
            else:
                sos = butter_sos("bandpass", [lo, hi])
            l, r = signal.sosfiltfilt(sos, L), signal.sosfiltfilt(sos, R)
            res[f"stereo_corr_{g}"] = float(np.corrcoef(l, r)[0, 1])
        res["mono_loss_db"] = float(10 * np.log10(np.mean(((L + R) / 2) ** 2) /
                                                 (0.5 * (np.mean(L ** 2) + np.mean(R ** 2)))))
    else:
        res["stereo_corr"] = None
        for g in GROUPS:
            res[f"stereo_corr_{g}"] = None
        res["mono_loss_db"] = None
    return res


# ------------------------------------------------------------------- features
def flatten(d):
    """Flat feature dict + category map used by compare.py."""
    F, C = {}, {}

    def put(k, v, cat):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return
        F[k] = float(v)
        C[k] = cat

    sp = d["spectrum"]
    for fc, v in zip(THIRD_OCT, sp["third_oct_db"]):
        if fc <= 12500:            # 16 kHz band unreliable for lossy references
            put(f"ltas.{fc:g}Hz", v, "ltas")
    for k in ("slope_125_1k", "slope_1k_12k", "slope_100_8k"):
        put(f"spec.{k}_db_oct", sp[k], "spectral_shape")
    put("spec.centroid_hz", sp["centroid_hz"], "spectral_shape")
    put("spec.rolloff85_hz", sp["rolloff85_hz"], "spectral_shape")
    ev = d["envelope"]
    for g, s in ev["groups"].items():
        put(f"env.cv.{g}", s["env_cv"], "envelope_moments")
        put(f"env.skew.{g}", s["env_skew"], "envelope_moments")
        put(f"env.kurt.{g}", s["env_kurt"], "envelope_moments")
        put(f"env.mean.{g}", s["env_mean"], "envelope_moments")
        for fc, v in zip(MOD_CENTERS, s["mod_frac_db"]):
            put(f"mod.{g}.{fc:g}Hz", v, "modulation")
    for i, fc in enumerate(ev["centers_hz"]):
        tag = f"b{i:02d}_{fc:.0f}Hz"
        put(f"envband.cv.{tag}", ev["cv"][i], "envelope_bands")
        put(f"envband.kurt.{tag}", ev["kurt"][i], "envelope_bands")
    for k, v in ev["corr"].items():
        put(f"corr.{k}", v, "cross_band_corr")
    for k, v in ev["corr_fast"].items():
        put(f"corrfast.{k}", v, "cross_band_corr")
    tr = d["transients"]
    for th, v in tr["rates"].items():
        put(f"trans.rate_{th}db_per_s", v, "transients")
    for k in ("alpha_amp_6db", "alpha_amp_10db", "alpha_absamp_10db", "b_energy_6db",
              "ioi_cv_6db", "ioi_cv_10db", "ioi_lv_6db", "ioi_lv_10db", "fano_100ms_10db",
              "fano_1s_10db", "energy_frac_10db", "energy_frac_6db"):
        put(f"trans.{k}", tr.get(k), "transients")
    wv = d["waveform"]
    put("wave.crest_db", wv["crest_db"], "waveform")
    put("wave.p999_db", wv["p999_db"], "waveform")
    put("wave.kurtosis", wv["kurtosis"], "waveform")
    for fc, v in zip(OCTAVES, wv["octave_kurtosis"]):
        if fc <= 8000:
            put(f"wave.okurt.{fc:g}Hz", v, "waveform")
    put("level.std_400ms_db", wv["level_std_400ms_db"], "dynamics")
    put("level.std_3s_db", wv["level_std_3s_db"], "dynamics")
    for k in ("stereo_corr", "stereo_corr_low", "stereo_corr_mid", "stereo_corr_high", "mono_loss_db"):
        put(f"stereo.{k}", wv[k], "stereo")
    return F, C


# ----------------------------------------------------------------- PNG output
_CMAP = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                  [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=float)


def colormap(v):
    v = np.clip(v, 0, 1) * (len(_CMAP) - 1)
    i = np.minimum(v.astype(int), len(_CMAP) - 2)
    t = (v - i)[..., None]
    return (_CMAP[i] * (1 - t) + _CMAP[i + 1] * t).astype(np.uint8)


def _font():
    from PIL import ImageFont
    for p in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 13)
            except Exception:
                pass
    return ImageFont.load_default()


def save_spectrogram(mono, path, title, max_s=30.0, fmax=16000, dyn=80):
    from PIL import Image, ImageDraw
    x = mono[: int(max_s * FS)]
    f, t, S = signal.spectrogram(x, FS, window="hann", nperseg=1024, noverlap=1024 - 240,
                                 mode="psd")
    m = f <= fmax
    S = db(S[m])
    top = np.percentile(S, 99.9)
    img = colormap((S - (top - dyn)) / dyn)[::-1]                 # high freq at top
    W, H = 1200, 520
    im = Image.fromarray(img).resize((W, H), Image.BILINEAR)
    L, T, Rm, B = 64, 28, 16, 36
    canvas = Image.new("RGB", (W + L + Rm, H + T + B), (255, 255, 255))
    canvas.paste(im, (L, T))
    d = ImageDraw.Draw(canvas)
    fnt = _font()
    d.text((L, 6), f"{title}  |  log-magnitude spectrogram, {dyn} dB range, 0-{fmax/1000:g} kHz", fill=(0, 0, 0), font=fnt)
    for khz in range(0, int(fmax / 1000) + 1, 2):
        y = T + H - int(H * khz * 1000 / fmax)
        d.line([(L - 5, y), (L, y)], fill=(0, 0, 0))
        d.text((8, y - 7), f"{khz:>2} kHz", fill=(0, 0, 0), font=fnt)
    dur = x.size / FS
    step = 1 if dur <= 12 else 2 if dur <= 30 else 5 if dur <= 90 else 20 if dur <= 300 else 60
    for s in np.arange(0, dur + 1e-9, step):
        xx = L + int(W * s / dur)
        d.line([(xx, T + H), (xx, T + H + 5)], fill=(0, 0, 0))
        d.text((xx - 8, T + H + 8), f"{s:g}s", fill=(0, 0, 0), font=fnt)
    canvas.save(path)


def save_envelopes(env, centers, path, title, max_s=20.0):
    from PIL import Image, ImageDraw
    n = int(max_s * ENV_FS)
    E = env[:, :n] ** COMPRESS
    E = E / np.maximum(np.percentile(E, 99.5, axis=1, keepdims=True), 1e-12)   # per-band normalised
    W, Hh = 1200, 384
    img = colormap(E[::-1])
    im = Image.fromarray(img).resize((W, Hh), Image.NEAREST)
    # line traces for three bands
    Ht = 300
    L, T, Rm, B = 70, 28, 16, 36
    canvas = Image.new("RGB", (W + L + Rm, Hh + Ht + T + B + 20), (255, 255, 255))
    canvas.paste(im, (L, T))
    d = ImageDraw.Draw(canvas)
    fnt = _font()
    d.text((L, 6), f"{title}  |  {len(centers)} ERB band envelopes (compressed ^{COMPRESS}, per-band normalised); traces below", fill=(0, 0, 0), font=fnt)
    for i in range(0, len(centers), 4):
        y = T + Hh - int((i + 0.5) * Hh / len(centers))
        d.text((4, y - 7), f"{centers[i]:>5.0f}", fill=(0, 0, 0), font=fnt)
    y0 = T + Hh + 20
    picks = [int(np.argmin(np.abs(centers - fc))) for fc in (150, 1200, 6000)]
    cols = [(200, 60, 20), (20, 120, 60), (30, 60, 200)]
    for k, (bi, col) in enumerate(zip(picks, cols)):
        tr = E[bi]
        yy = y0 + Ht / 3 * (k + 1) - 4 - tr * (Ht / 3 - 12)
        xs = L + np.arange(tr.size) * W / max(tr.size - 1, 1)
        d.line(list(zip(xs.tolist(), yy.tolist())), fill=col, width=1)
        d.text((4, y0 + Ht / 3 * k + 4), f"{centers[bi]:.0f}Hz", fill=col, font=fnt)
    dur = E.shape[1] / ENV_FS
    step = 1 if dur <= 12 else 2
    for s in np.arange(0, dur + 1e-9, step):
        xx = L + int(W * s / dur)
        d.line([(xx, y0 + Ht), (xx, y0 + Ht + 5)], fill=(0, 0, 0))
        d.text((xx - 8, y0 + Ht + 8), f"{s:g}s", fill=(0, 0, 0), font=fnt)
    canvas.save(path)


# --------------------------------------------------------------------- driver
def analyze(path, name=None, start=0.0, dur=None, png_dir=None):
    chans = load_wav(path, start, dur)
    mono = chans.mean(axis=0)
    rms = math.sqrt(np.mean(mono ** 2))
    if rms <= 0:
        raise SystemExit("silent input")
    g = TARGET_RMS / rms
    mono = mono * g
    chans = chans * g
    dur_s = mono.size / FS
    name = name or os.path.splitext(os.path.basename(path))[0]
    env, centers = band_envelopes(mono)
    d = {
        "name": name, "file": os.path.abspath(path), "start_s": start, "duration_s": dur_s,
        "channels": int(chans.shape[0]), "input_rms_dbfs": float(20 * np.log10(rms)),
        "spectrum": ltas_third_octave(mono),
        "envelope": envelope_stats(env, centers),
        "transients": transient_stats(mono, dur_s),
        "waveform": waveform_stats(chans, mono),
        "settings": {"fs": FS, "env_fs": ENV_FS, "compress": COMPRESS, "n_bands": N_BANDS,
                     "band_range_hz": [BAND_LO, BAND_HI], "transient_hp_hz": TRANSIENT_HP,
                     "onset_thresholds_db": ONSET_THRESH_DB, "normalised_rms_dbfs": -20},
    }
    d["spectrum"]["third_oct_centers_hz"] = THIRD_OCT
    d["waveform"]["octave_centers_hz"] = OCTAVES
    F, C = flatten(d)
    d["features"], d["categories"] = F, C
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
        save_spectrogram(mono, os.path.join(png_dir, f"{name}_spectrogram.png"), name)
        save_envelopes(env, centers, os.path.join(png_dir, f"{name}_envelopes.png"), name)
    return d


def build_reference(cfg_path, out_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(cfg_path))
    stats_dir = os.path.join(base, cfg.get("stats_dir", "stats"))
    png_dir = os.path.join(base, cfg.get("png_dir", "png"))
    os.makedirs(stats_dir, exist_ok=True)
    results = []
    for r in cfg["references"]:
        if r.get("exclude"):
            continue
        wav = os.path.expandvars(r["wav"])
        print(f"analysing {r['name']} ...", file=sys.stderr, flush=True)
        d = analyze(wav, r["name"], r.get("start", 0.0), r.get("dur"), png_dir)
        d["source"] = {k: r.get(k) for k in ("title", "url", "license", "author", "notes")}
        with open(os.path.join(stats_dir, f"{r['name']}.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
        results.append(d)
    keys = sorted(set().union(*[d["features"].keys() for d in results]))
    cats = {}
    for d in results:
        cats.update(d["categories"])
    summary = {}
    for k in keys:
        v = np.array([d["features"][k] for d in results if k in d["features"]])
        summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                      "min": float(v.min()), "max": float(v.max()), "n": int(v.size),
                      "category": cats[k]}
    ref = {
        "description": "Texture statistics of real campfire/bonfire recordings (McDermott & Simoncelli 2011 style). "
                       "Produced by tools/audio_analysis/texture_stats.py --build-reference references.json.",
        "settings": results[0]["settings"],
        "references": [{"name": d["name"], "duration_s": round(d["duration_s"], 2), "start_s": d["start_s"],
                        "channels": d["channels"], **d["source"]} for d in results],
        "features": summary,
        "per_reference": {d["name"]: d["features"] for d in results},
        "curves": {
            "third_oct_centers_hz": THIRD_OCT,
            "third_oct_db_mean": np.mean([d["spectrum"]["third_oct_db"] for d in results], axis=0).round(2).tolist(),
            "env_band_centers_hz": results[0]["envelope"]["centers_hz"],
            "env_kurt_mean": np.mean([d["envelope"]["kurt"] for d in results], axis=0).round(3).tolist(),
            "env_cv_mean": np.mean([d["envelope"]["cv"] for d in results], axis=0).round(4).tolist(),
            "corr_matrix_mean": np.mean([d["envelope"]["corr_matrix"] for d in results], axis=0).round(3).tolist(),
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, indent=1)
    print(f"wrote {out_path} ({len(results)} references, {len(keys)} features)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="*", help="one or more WAV files (several files share one "
                    "interpreter start-up; --out/--name then apply only to a single file)")
    ap.add_argument("--out")
    ap.add_argument("--png-dir")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--build-reference", metavar="REFERENCES_JSON")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    if a.build_reference:
        build_reference(a.build_reference, a.out or os.path.join(here, "reference_stats.json"))
        return
    if not a.wav:
        ap.error("need a wav file or --build-reference")
    if len(a.wav) > 1 and (a.out or a.name):
        ap.error("--out/--name need a single input file")
    png_dir = None if a.no_png else (a.png_dir or os.path.join(here, "png"))
    for wav in a.wav:
        name = a.name or os.path.splitext(os.path.basename(wav))[0]
        d = analyze(wav, name, a.start, a.dur, png_dir)
        out = a.out or os.path.join(here, "stats", f"{name}.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
        print(out, flush=True)


if __name__ == "__main__":
    main()
