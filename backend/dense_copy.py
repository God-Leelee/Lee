"""Pixel-verified exact-copy candidate; no authenticity verdict is produced.

Dense translation hashing complements sparse SIFT points. Hash collisions,
flat colour and small repeated letters are rejected by pixel and area checks.
Natural repeated graphics can still match; training/calibration must include
them, and copy direction remains unknown.
"""
import time
import cv2
import numpy as np
from PIL import Image


def windows(a, side):
    # Summed-area tables give translation-invariant candidate hashes. Overflow
    # in uint64 is intentional modular arithmetic, followed by exact verification.
    z = np.pad(a, ((1, 0), (1, 0))).cumsum(0, dtype=a.dtype).cumsum(1, dtype=a.dtype)
    return z[side:, side:] - z[:-side, side:] - z[side:, :-side] + z[:-side, :-side]


def dense_copy(image, max_side=1024, block=16, query_stride=8, min_variance=100.):
    start = time.perf_counter()
    width, height = image.size
    scale = min(1., max_side / max(width, height))
    rgb = np.asarray(image.resize((max(1, round(width * scale)), max(1, round(height * scale)))))
    h, w = rgb.shape[:2]
    result = {'verified_pair_count': 0, 'verified_area_ratio': 0., 'largest_component_ratio': 0.,
              'coordinate_space': 'original_raw_pixels', 'direction': 'unknown',
              'method': 'dense uint64 candidate hash + exact RGB patch verification', 'pairs': []}
    if block < 4 or query_stride < 1 or not np.isfinite(min_variance) or min_variance <= 0:
        raise ValueError('Invalid dense-copy parameters')
    result['min_patch_variance'] = float(min_variance)
    if min(h, w) < block * 3:
        return {**result, 'elapsed_ms': (time.perf_counter() - start) * 1000}
    colour = rgb[:, :, 0].astype(np.uint64) | (rgb[:, :, 1].astype(np.uint64) << 8) | (rgb[:, :, 2].astype(np.uint64) << 16)
    # SplitMix64 maps pixel colours to a full-width deterministic hash.
    hashed = colour + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    hashed = (hashed ^ (hashed >> 27)) * np.uint64(0x94D049BB133111EB)
    hashed ^= hashed >> 31
    key = windows(hashed, block)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mean = windows(gray, block) / block ** 2
    variance = windows(gray ** 2, block) / block ** 2 - mean ** 2
    valid = variance >= min_variance
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return {**result, 'elapsed_ms': (time.perf_counter() - start) * 1000}
    keys = key[ys, xs]
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    qy, qx = np.nonzero(valid[::query_stride, ::query_stride])
    qy, qx = qy * query_stride, qx * query_stride
    qkeys = key[qy, qx]
    lower = np.searchsorted(sorted_keys, qkeys, side='left')
    upper = np.searchsorted(sorted_keys, qkeys, side='right')
    translations = {}
    min_distance = max(24, min(w, h) * .07)
    for index in np.flatnonzero((upper - lower >= 2) & (upper - lower <= 32)):
        x, y = int(qx[index]), int(qy[index])
        for k in order[lower[index]:upper[index]]:
            tx, ty = int(xs[k]), int(ys[k])
            if (tx - x) ** 2 + (ty - y) ** 2 < min_distance ** 2:
                continue
            if not np.array_equal(rgb[y:y + block, x:x + block], rgb[ty:ty + block, tx:tx + block]):
                continue
            if (tx, ty) < (x, y):
                x0, y0, x1, y1 = tx, ty, x, y
            else:
                x0, y0, x1, y1 = x, y, tx, ty
            shift = (x1 - x0, y1 - y0)
            translations.setdefault(shift, set()).add((x0, y0, x1, y1))
    if translations:
        shift, pairs = max(translations.items(), key=lambda item: len(item[1]))
        mask = np.zeros((h, w), np.uint8)
        for x, y, tx, ty in pairs:
            mask[y:y + block, x:x + block] = 1
            mask[ty:ty + block, tx:tx + block] = 1
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = sorted(stats[1:], key=lambda s: s[4], reverse=True)
        # Area is evidence, not a hand-set fake decision; even small components
        # remain numeric features and may be rejected by the learned edit head.
        result.update(verified_pair_count=len(pairs), verified_area_ratio=float(mask.mean()),
                      largest_component_ratio=float(components[0][4] / (h * w)) if components else 0.,
                      translation_xy=[shift[0] / scale, shift[1] / scale],
                      pairs=[[x / scale, y / scale, tx / scale, ty / scale] for x, y, tx, ty in sorted(pairs)[:24]],
                      regions=[[float(x / scale), float(y / scale), float(rw / scale), float(rh / scale)]
                               for x, y, rw, rh, area in components[:2]])
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000
    return result
