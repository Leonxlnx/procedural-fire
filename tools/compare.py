"""Side-by-side comparison strip with labels and an optional crop.
Usage: python tools/compare.py out.png [--crop x0,y0,x1,y1] [--h 900] img1 img2 ...
"""
import sys
from PIL import Image, ImageDraw

args = sys.argv[1:]
out = args.pop(0)
crop = None
height = 900
while args and args[0].startswith('--'):
    k = args.pop(0)
    if k == '--crop':
        crop = tuple(int(v) for v in args.pop(0).split(','))
    elif k == '--h':
        height = int(args.pop(0))
tiles = []
for p in args:
    im = Image.open(p).convert('RGB')
    if crop:
        im = im.crop(crop)
    w = round(im.width * height / im.height)
    im = im.resize((w, height), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    label = p.replace('\\', '/').split('/')[-1].rsplit('.', 1)[0]
    d.rectangle([0, 0, 8 + 7 * len(label), 16], fill=(0, 0, 0))
    d.text((4, 2), label, fill=(120, 200, 255))
    tiles.append(im)
W = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
strip = Image.new('RGB', (W, height), (40, 40, 40))
x = 0
for t in tiles:
    strip.paste(t, (x, 0))
    x += t.width + 6
strip.save(out)
print(out, strip.size)
