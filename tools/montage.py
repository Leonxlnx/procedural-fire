# Side-by-side montage: uv run --with pillow python tools/montage.py out.png a.png b.png ... [--scale 0.5] [--labels a,b,c]
import sys
from PIL import Image, ImageDraw
args = sys.argv[1:]; scale = 1.0; labels = None
if '--scale' in args: i = args.index('--scale'); scale = float(args[i+1]); del args[i:i+2]
if '--labels' in args: i = args.index('--labels'); labels = args[i+1].split(','); del args[i:i+2]
out, files = args[0], args[1:]
imgs = [Image.open(f).convert('RGB') for f in files]
imgs = [im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS) for im in imgs]
H = max(i.height for i in imgs); W = sum(i.width for i in imgs) + 4 * (len(imgs) - 1)
canvas = Image.new('RGB', (W, H)); x = 0
d = ImageDraw.Draw(canvas)
for k, im in enumerate(imgs):
    canvas.paste(im, (x, 0))
    if labels and k < len(labels): d.text((x + 6, 6), labels[k], fill=(120, 200, 255))
    x += im.width + 4
canvas.save(out); print(out, W, H)
