"""Scores a fire render against measured real-fire statistics (research/visual_reference.md).

usage: .venv/Scripts/python.exe tools/fire_metrics.py img.png [img2.png ...] [--roi x0,y0,x1,y1]
Prints: clipping fractions, natural-ramp colour error, dim/bright area ratio, base->tip profile, near-black tint.
"""
import sys, json
import numpy as np
from PIL import Image

NAT = [(0.07, '270C06'), (0.13, '421908'), (0.19, '5C2708'), (0.25, '6F350C'), (0.31, '844412'), (0.37, '9D5318'),
       (0.43, 'B5611E'), (0.49, 'CC6F27'), (0.55, 'DF7D35'), (0.61, 'EE8C46'), (0.67, 'F99D53'), (0.73, 'FEAF62'),
       (0.79, 'FEC27A'), (0.85, 'FFD595'), (0.91, 'FFE8B2'), (0.97, 'FEFCD2')]
NAT = [(l, np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)) for l, h in NAT]


def hue(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = img.max(-1); mn = img.min(-1); d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return h, s


def analyze(path, roi=None):
    im = np.asarray(Image.open(path).convert('RGB')).astype(float)
    if roi: x0, y0, x1, y1 = roi; im = im[y0:y1, x0:x1]
    n = im / 255.0
    luma = n @ np.array([0.2126, 0.7152, 0.0722])
    h, s = hue(n)
    flame = (luma > 0.25) & (n[..., 0] >= n[..., 2]) & ((h <= 65) | (h >= 335) | (s < 0.18))
    fp = im[flame]
    out = {'file': path.split('/')[-1].split('\\')[-1], 'flame_frac': round(float(flame.mean()), 4)}
    if fp.size == 0:
        return out
    out['R_clip'] = round(float((fp[:, 0] >= 250).mean()), 3)
    out['RG_clip'] = round(float(((fp[:, 0] >= 250) & (fp[:, 1] >= 250)).mean()), 3)
    out['all_clip'] = round(float((fp.min(1) >= 245).mean()), 3)
    # area ratio dim (0.10-0.55) vs bright (>0.55) among warm pixels
    warm = (n[..., 0] >= n[..., 2]) & ((h <= 65) | (h >= 335) | (s < 0.18))
    dim = ((luma > 0.10) & (luma <= 0.55) & warm).sum(); bright = ((luma > 0.55) & warm).sum()
    out['dim_bright_ratio'] = round(float(dim / max(bright, 1)), 2)
    # ramp error
    errs = []; ramp = []
    for l, tgt in NAT:
        m = warm & (np.abs(luma - l) < 0.03)
        if m.sum() < 30: continue
        med = np.median(im[m], axis=0)
        e = float(np.linalg.norm(med - tgt)); errs.append(e)
        ramp.append((l, '#%02X%02X%02X' % tuple(int(v) for v in med), int(e)))
    out['ramp_err'] = round(float(np.mean(errs)), 1) if errs else None
    out['ramp'] = ramp
    # base -> tip profile: rows with flame, normalized height
    rows = np.where(flame.any(1))[0]
    if len(rows) > 10:
        top, base = rows.min(), rows.max()
        prof = []
        for f in (0.05, 0.25, 0.45, 0.65, 0.85, 0.97):
            y = int(base - f * (base - top))
            band = flame[max(y - 3, 0):y + 3]
            cov = float(band.mean()) if band.size else 0
            px = im[max(y - 3, 0):y + 3][band]
            hm = float(np.median(hue(px[None] / 255.0)[0])) if px.size else 0
            lm = float(np.median(px @ np.array([0.2126, 0.7152, 0.0722]) / 255)) if px.size else 0
            prof.append((f, round(cov, 3), round(lm, 2), round(hm, 1)))
        out['profile(h,cov,luma,hue)'] = prof
    # near-black tint
    nb = (luma > 0.015) & (luma < 0.12)
    if nb.sum() > 100:
        med = np.median(im[nb], axis=0)
        out['near_black'] = '#%02X%02X%02X' % tuple(int(v) for v in med)
    out['image_median_luma'] = round(float(np.median(luma)), 3)
    return out


if __name__ == '__main__':
    args = sys.argv[1:]; roi = None
    if '--roi' in args:
        i = args.index('--roi'); roi = [int(v) for v in args[i + 1].split(',')]; del args[i:i + 2]
    for p in args:
        r = analyze(p, roi)
        ramp = r.pop('ramp', [])
        print(json.dumps(r))
        if ramp: print('   ramp:', ' '.join(f'{l:.2f}{c}({e})' for l, c, e in ramp))
    print('targets: R_clip 0.2-0.6 (median .45), RG_clip ~.05, all_clip <.05, dim_bright 1.3-5.5 (~2), ramp_err <15, '
          'profile: brightest at .1-.5H, cov at top <.15, tips 5-10deg redder; near-black warm #150503-#270400; median luma .01-.05')
