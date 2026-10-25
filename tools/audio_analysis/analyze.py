"""analyze.py - objective checks + spectrogram for rendered fire audio (no listening possible, so measure).

usage: python -I tools/audio_analysis/analyze.py <in.wav> [--png out.png] [--events events.json]
                                                 [--t0 s] [--t1 s] [--json out.json] [--title text]
Prints JSON: NaN/inf count, DC offset, peaks, largest sample-to-sample jumps, isolated single-sample spikes
(digital clicks: a sample far from both neighbours, relative to the local high-pass RMS), octave levels,
octave kurtosis (sparsity; McDermott et al.), L/R correlation. Draws a log-frequency spectrogram with a
momentary-loudness strip and event markers.
"""
import json, sys, struct, argparse
import numpy as np
from scipy import signal
from PIL import Image, ImageDraw, ImageFont


def read_wav(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'RIFF' and data[8:12] == b'WAVE', 'not a WAV'
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b'fmt ':
            fmt = struct.unpack('<HHIIHH', body[:16])
        elif cid == b'data':
            raw = body
        pos += 8 + size + (size & 1)
    tag, nch, sr, _, _, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(raw, dtype='<f4').astype(np.float64)
    elif bits == 16:
        x = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0
    elif bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float64) / 8388608.0
    elif bits == 32:
        x = np.frombuffer(raw, dtype='<i4').astype(np.float64) / 2147483648.0
    else:
        raise ValueError('unsupported format')
    return x.reshape(-1, nch).T, sr


def stats(x, sr):
    out = {}
    out['nonFinite'] = int(np.sum(~np.isfinite(x)))
    x = np.nan_to_num(x)
    out['dcOffset'] = [float(f'{v:.2e}') for v in x.mean(axis=1)]
    out['samplePeakDbFS'] = round(float(20 * np.log10(np.abs(x).max() + 1e-30)), 2)
    d = np.abs(np.diff(x, axis=1))
    flat = d.max(axis=0)
    top = np.argsort(flat)[::-1]
    jumps, used = [], []
    for i in top:
        if any(abs(i - u) < sr * 0.02 for u in used):
            continue
        used.append(i)
        jumps.append({'t': round(i / sr, 4), 'jump': round(float(flat[i]), 4)})
        if len(jumps) >= 8:
            break
    out['largestJumps'] = jumps
    # isolated single-sample spikes: deviation from neighbour mean vs local RMS of the 2nd difference
    spikes = 0
    worst = []
    for c in range(x.shape[0]):
        y = x[c]
        dev = np.abs(y[1:-1] - 0.5 * (y[:-2] + y[2:]))
        loc = np.sqrt(signal.convolve(dev ** 2, np.ones(481) / 481, mode='same')) + 1e-6  # 10 ms
        ratio = dev / loc
        idx = np.where((ratio > 12) & (dev > 0.05))[0]
        spikes += len(idx)
        worst += [(float(ratio[i]), (i + 1) / sr) for i in idx[:20]]
    out['isolatedSpikes'] = spikes
    out['isolatedSpikeExamples'] = [{'t': round(t, 4), 'ratio': round(r, 1)} for r, t in sorted(worst, reverse=True)[:5]]
    m = 0.5 * (x[0] + x[1]) if x.shape[0] > 1 else x[0]
    octs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    lv, ku = {}, {}
    for fc in octs:
        lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), sr / 2 * 0.95)
        sos = signal.butter(3, [lo, hi], btype='band', fs=sr, output='sos')
        y = signal.sosfilt(sos, m)[sr // 2:]
        p = np.mean(y ** 2)
        lv[str(fc)] = round(float(10 * np.log10(p + 1e-30)), 1)
        ku[str(fc)] = round(float(np.mean(y ** 4) / (p * p + 1e-60)), 1)
    out['octaveDb'] = lv
    out['octaveKurtosis'] = ku
    if x.shape[0] > 1:
        out['lrCorrelation'] = round(float(np.sum(x[0] * x[1]) / np.sqrt(np.sum(x[0] ** 2) * np.sum(x[1] ** 2) + 1e-30)), 3)
    return out


def colormap(v):
    # v in [0,1] -> RGB (perceptual-ish "inferno" approximation)
    stops = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=np.float64)
    pos = np.linspace(0, 1, len(stops))
    v = np.clip(v, 0, 1)
    rgb = np.stack([np.interp(v, pos, stops[:, k]) for k in range(3)], axis=-1)
    return rgb.astype(np.uint8)


def k_weight(x, sr):
    # BS.1770 K-weighting (libebur128 analog prototype, any sr)
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / sr); Vh = 10 ** (G / 20); Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / sr); a0 = 1 + K / Q + K * K
    a2 = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    return signal.lfilter([1, -2, 1], a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def draw(x, sr, png, events=None, t0=0.0, t1=None, title=''):
    W, H, top, left, bottom = 1800, 640, 150, 60, 40
    n0 = int(t0 * sr); n1 = x.shape[1] if t1 is None else min(x.shape[1], int(t1 * sr))
    m = 0.5 * (x[0, n0:n1] + x[1, n0:n1])
    dur = (n1 - n0) / sr
    nfft = 2048 if dur > 4 else 1024
    hop = max(32, int((n1 - n0) / W))
    f, t, S = signal.stft(m, fs=sr, window='hann', nperseg=nfft, noverlap=nfft - hop if nfft > hop else nfft // 2, boundary=None, padded=False)
    P = 20 * np.log10(np.abs(S) * 2 / (np.sum(np.hanning(nfft)) / nfft * nfft) + 1e-12)
    fmin, fmax = 30.0, min(20000.0, sr / 2)
    ys = np.arange(H)
    fy = fmin * (fmax / fmin) ** (1 - ys / (H - 1))
    cols = np.interp(np.linspace(0, len(t) - 1, W), np.arange(len(t)), np.arange(len(t)))
    ci = np.clip(np.round(cols).astype(int), 0, len(t) - 1)
    img = np.empty((H, W))
    for r in range(H):
        k = np.searchsorted(f, fy[r])
        k = min(max(k, 1), len(f) - 1)
        img[r] = P[k, ci]
    lo, hi = -120.0, -30.0
    rgb = colormap((img - lo) / (hi - lo))
    canvas = Image.new('RGB', (W + left + 20, H + top + bottom + 30), (16, 16, 18))
    canvas.paste(Image.fromarray(rgb, 'RGB'), (left, top))
    dr = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    # momentary loudness strip (400 ms, 50 ms hop)
    kx = k_weight(x[:, n0:n1], sr)
    pw = np.sum(kx ** 2, axis=0)
    cs = np.concatenate([[0], np.cumsum(pw)])
    w4, h5 = int(0.4 * sr), int(0.05 * sr)
    pts = []
    for s in range(0, len(pw) - w4, h5):
        v = -0.691 + 10 * np.log10((cs[s + w4] - cs[s]) / w4 + 1e-20)
        pts.append(((s + w4 / 2) / sr, v))
    y0s, y1s = 20, top - 20
    lmin, lmax = -40.0, -5.0
    for lvl in (-30, -20, -14, -10):
        yy = y1s - (lvl - lmin) / (lmax - lmin) * (y1s - y0s)
        dr.line([(left, yy), (left + W, yy)], fill=(60, 60, 64) if lvl != -14 else (90, 70, 40))
        dr.text((8, yy - 6), f'{lvl}', fill=(150, 150, 150), font=font)
    xy = [(left + p[0] / dur * W, y1s - (np.clip(p[1], lmin, lmax) - lmin) / (lmax - lmin) * (y1s - y0s)) for p in pts]
    if len(xy) > 1:
        dr.line(xy, fill=(255, 170, 60), width=2)
    dr.text((left, 2), (title + '  ' if title else '') + 'momentary loudness (LUFS, 400 ms) / spectrogram (dBFS, log f)', fill=(200, 200, 200), font=font)
    for fr in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
        if fr < fmin or fr > fmax:
            continue
        yy = top + (1 - np.log(fr / fmin) / np.log(fmax / fmin)) * (H - 1)
        dr.line([(left - 5, yy), (left, yy)], fill=(200, 200, 200))
        dr.text((4, yy - 6), f'{fr // 1000}k' if fr >= 1000 else str(fr), fill=(200, 200, 200), font=font)
    step = 1 if dur <= 5 else 2 if dur <= 12 else 5 if dur <= 40 else 10
    tt = np.ceil(t0 / step) * step
    while tt <= t0 + dur + 1e-9:
        xx = left + (tt - t0) / dur * W
        dr.line([(xx, top + H), (xx, top + H + 5)], fill=(200, 200, 200))
        dr.text((xx - 8, top + H + 8), f'{tt:g}s', fill=(200, 200, 200), font=font)
        tt += step
    if events:
        for i, e in enumerate(events):
            if not (t0 <= e['t'] <= t0 + dur):
                continue
            xx = left + (e['t'] - t0) / dur * W
            col = (120, 220, 255) if e.get('source') == 'random' else (140, 255, 140)
            dr.line([(xx, top), (xx, top + 10)], fill=col, width=2)
            dr.line([(xx, top + H - 10), (xx, top + H)], fill=col, width=2)
            dr.text((xx + 3, top + H + 22 if i % 2 else top + 12), e['type'] + (' (rnd)' if e.get('source') == 'random' else ''), fill=col, font=font)
    canvas.save(png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav')
    ap.add_argument('--png')
    ap.add_argument('--events')
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--t1', type=float)
    ap.add_argument('--json')
    ap.add_argument('--title', default='')
    ap.add_argument('--no-stats', action='store_true')
    a = ap.parse_args()
    x, sr = read_wav(a.wav)
    out = {'file': a.wav, 'sampleRate': sr, 'channels': int(x.shape[0]), 'seconds': round(x.shape[1] / sr, 3)}
    if not a.no_stats:
        out.update(stats(x, sr))
    if a.png:
        ev = None
        if a.events:
            j = json.load(open(a.events))
            ev = j.get('all', j.get('events', []))
        draw(x, sr, a.png, ev, a.t0, a.t1, a.title)
        out['png'] = a.png
    s = json.dumps(out, indent=1)
    if a.json:
        open(a.json, 'w').write(s)
    print(s)


if __name__ == '__main__':
    main()
