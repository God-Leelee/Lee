"""Create original procedural inputs, not an AI-detection accuracy dataset."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/review_demo'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(10241500)
    y, x = np.mgrid[:480, :640]
    pixels = np.stack((70 + x / 6, 120 + y / 8, 200 - y / 5), axis=2)
    pixels = np.clip(pixels + rng.normal(0, 3, pixels.shape), 0, 255).astype('uint8')
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.polygon([(0, 350), (150, 190), (320, 360)], fill=(55, 100, 90))
    draw.rectangle((0, 360, 640, 480), fill=(55, 75, 85))
    for left in range(330, 600, 64):
        draw.rectangle((left, 210, left + 44, 360), fill=(188, 191, 196))
        for top in range(225, 340, 28):
            draw.rectangle((left + 8, top, left + 20, top + 15), fill=(70, 106, 132))
    draw.text((20, 20), 'CLEARTRACE / PROCEDURAL ENGINEERING INPUT', fill='white')
    image.save(OUT / 'sample01.png')
    edited = image.copy()
    edited.paste(image.crop((342, 220, 406, 320)), (170, 260))
    edited.save(OUT / 'sample02.png')
    mask = Image.new('L', image.size)
    ImageDraw.Draw(mask).rectangle((170, 260, 233, 359), fill=255)
    mask.save(OUT / 'sample02_operation_mask.png')
    with av.open(str(OUT / 'sample03.mp4'), 'w') as container:
        stream = container.add_stream('libx264', rate=6)
        stream.width, stream.height, stream.pix_fmt = 640, 480, 'yuv420p'
        for index in range(12):
            current = image.copy()
            ImageDraw.Draw(current).ellipse((40 + index * 20, 385, 70 + index * 20, 415), fill=(240, 175, 70))
            frame = av.VideoFrame.from_image(current)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    names = [('sample01.png', 'image'), ('sample02.png', 'image'), ('sample03.mp4', 'video')]
    manifest = {'scope': 'Procedural engineering inputs; no accuracy labels and no AI-generation truth',
                'source': 'Original deterministic code in scripts/make_demo_samples.py',
                'samples': [{'path': name, 'material_type': kind,
                             'sha256': hashlib.sha256((OUT / name).read_bytes()).hexdigest()} for name, kind in names]}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'samples': len(names), 'output': str(OUT), 'scope': manifest['scope']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
