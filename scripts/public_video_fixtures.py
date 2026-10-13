"""Build train-parent video controls and known local edits with frame-level GT.

CPU only. Looped/resampled genuine frames are engineered motion, not an original
continuous capture. The AI-source patch is a composite, not generative inpainting.
These related-parent fixtures are never an independent contest acceptance set.
"""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT, atomic_json, sha256

WIDTH, HEIGHT, FPS, FRAME_COUNT = 640, 480, 24, 96
REQUESTED_INTERVAL = (1.2, 2.8)
PATCH_BOX = (220, 160, 260, 220)
DONOR_BOX = (160, 100, 260, 220)
SCOPE = 'Train-parent known-operation engineering fixtures; not independent contest or business acceptance.'
MOTION_LIMIT = 'Repeated/resampled decoded frames and loop seams are engineered motion, not genuine continuous source motion.'


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes source/output root: ' + relative)
    return path




def decode_parent(source: Path, row: dict):
    path = inside(source, row['relative_path'])
    if sha256(path) != row['sha256']:
        raise ValueError('Parent hash mismatch: ' + row['sample_id'])
    pixels, records = [], []
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == 'video']
        if len(streams) != 1:
            raise ValueError('Fixture parent must have one video track')
        stream = streams[0]
        declared_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        declared_rate = float(stream.average_rate) if stream.average_rate else None
        first_time = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Parent frame has no original PTS/time base')
            absolute_time = float(frame.pts * frame.time_base)
            first_time = absolute_time if first_time is None else first_time
            relative_time = absolute_time - first_time
            if records and relative_time <= records[-1]['relative_timestamp_seconds']:
                raise ValueError('Non-increasing parent PTS')
            records.append({'decoded_frame_index': index, 'original_pts': frame.pts,
                            'original_time_base': str(frame.time_base),
                            'original_timestamp_seconds': absolute_time,
                            'relative_timestamp_seconds': relative_time,
                            'original_width': frame.width, 'original_height': frame.height})
            pixels.append(np.asarray(frame.to_image().convert('RGB').resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)))
            # Include the first boundary frame, so selection below can cover the
            # entire requested four-second prefix without decoding long parents.
            if relative_time >= FRAME_COUNT / FPS:
                break
    if not records:
        raise ValueError('No decoded frames: ' + row['sample_id'])
    deltas = [b['relative_timestamp_seconds'] - a['relative_timestamp_seconds'] for a, b in zip(records, records[1:])]
    period = 1 / declared_rate if declared_rate and declared_rate > 0 else float(np.median(deltas)) if deltas else None
    if period is None or period <= 0:
        raise ValueError('Cannot establish source-frame exposure duration')
    decoded_duration = records[-1]['relative_timestamp_seconds'] + period
    usable_duration = min(FRAME_COUNT / FPS, decoded_duration,
                          declared_duration if declared_duration and declared_duration > 0 else decoded_duration)
    if usable_duration <= 0:
        raise ValueError('No usable parent duration')
    timing = {'sample_id': row['sample_id'], 'sha256': row['sha256'], 'split': row['split'],
              'truth_label': row['truth_label'], 'source_group': row['source_group'],
              'relative_path': row['relative_path'], 'declared_duration_seconds': declared_duration,
              'declared_average_rate': declared_rate, 'decoded_prefix_only': decoded_duration >= FRAME_COUNT / FPS,
              'usable_cycle_duration_seconds': usable_duration,
              'original_decoded_frame_pts': records, 'motion_limit': MOTION_LIMIT}
    return {'row': row, 'pixels': pixels, 'records': records,
            'times': [r['relative_timestamp_seconds'] for r in records],
            'cycle_duration': usable_duration, 'timing': timing}


def parent_frame(parent, output_time):
    cycle = int(output_time / parent['cycle_duration'])
    phase = output_time - cycle * parent['cycle_duration']
    index = max(0, bisect_right(parent['times'], phase + 1e-9) - 1)
    record = parent['records'][index]
    mapping = {'parent_sample_id': parent['row']['sample_id'], 'parent_sha256': parent['row']['sha256'],
               'decoded_frame_index': record['decoded_frame_index'], 'original_pts': record['original_pts'],
               'original_time_base': record['original_time_base'],
               'original_timestamp_seconds': record['original_timestamp_seconds'],
               'source_cycle_index': cycle, 'source_phase_seconds': phase}
    return parent['pixels'][index], mapping


def verify_encoded(path):
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError('Export lost PTS')
            frames.append({'output_frame_index': index, 'pts': frame.pts, 'time_base': str(frame.time_base),
                           'timestamp_seconds': float(frame.pts * frame.time_base) - origin,
                           'width': frame.width, 'height': frame.height})
    if len(frames) != FRAME_COUNT:
        raise ValueError(f'Export has {len(frames)} frames, expected {FRAME_COUNT}')
    for frame in frames:
        if (frame['width'], frame['height']) != (WIDTH, HEIGHT) or abs(frame['timestamp_seconds'] - frame['output_frame_index'] / FPS) > 1e-6:
            raise ValueError('Export dimensions/PTS differ from declared CFR fixture')
    return frames


def create_fixture(out, name, host, donor, operation):
    path = out / (name + '.mp4')
    if path.exists():
        raise FileExistsError('Existing fixture preserved; choose a new --output: ' + str(path))
    mappings = []
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('libx264', rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, 'yuv420p'
        stream.time_base = Fraction(1, FPS)
        stream.options = {'crf': '18', 'preset': 'medium'}
        for index in range(FRAME_COUNT):
            timestamp = index / FPS
            base, host_map = parent_frame(host, timestamp)
            pixels = base.copy()
            active = donor is not None and REQUESTED_INTERVAL[0] <= timestamp < REQUESTED_INTERVAL[1]
            donor_map = None
            if active:
                donor_pixels, donor_map = parent_frame(donor, timestamp)
                x, y, width, height = PATCH_BOX
                sx, sy, _, _ = DONOR_BOX
                pixels[y:y+height, x:x+width] = donor_pixels[sy:sy+height, sx:sx+width]
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
            mappings.append({'gt_active': active, 'host_source': host_map, 'patch_source': donor_map})
        for packet in stream.encode():
            container.mux(packet)
    encoded = verify_encoded(path)
    frames = [{**actual, **mapping,
               'mask_path': name + '_truth.png' if mapping['gt_active'] else 'empty_truth.png'}
              for actual, mapping in zip(encoded, mappings)]
    active_frames = [f for f in frames if f['gt_active']]
    actual_interval = ([active_frames[0]['timestamp_seconds'], active_frames[-1]['timestamp_seconds'] + 1 / FPS]
                       if active_frames else None)
    truth = np.zeros((HEIGHT, WIDTH), np.uint8)
    if donor:
        x, y, width, height = PATCH_BOX
        truth[y:y+height, x:x+width] = 255
    mask_path = out / (name + '_truth.png')
    Image.fromarray(truth).save(mask_path)
    sidecar = out / (name + '_frames.json')
    atomic_json(sidecar, frames)
    parent_rows = [host['row']] + ([donor['row']] if donor else [])
    used_host = [f['host_source'] for f in frames]
    used_donor = [f['patch_source'] for f in frames if f['patch_source']]
    looped = any(p['source_cycle_index'] > 0 for p in used_host + used_donor)
    return {'sample_id': name, 'relative_path': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'material_type': 'video', 'truth_label': 'fake' if donor else 'genuine',
            'material_class': operation, 'split': 'engineering', 'source_group': host['row']['sha256'],
            'generator_tool': 'known_public_AI_source_patch_composite' if operation == 'ai_patch_composite' else 'not_applicable',
            'parent_sha256': [r['sha256'] for r in parent_rows],
            'parents': [{'sample_id': r['sample_id'], 'split': r['split'], 'truth_label': r['truth_label'],
                         'sha256': r['sha256'], 'source_group': r['source_group'],
                         'timing_path': 'parents/' + r['sample_id'] + '_timing.json'} for r in parent_rows],
            'width': WIDTH, 'height': HEIGHT, 'fps': FPS, 'frame_count': FRAME_COUNT, 'duration_seconds': FRAME_COUNT / FPS,
            'frame_ground_truth_path': sidecar.name, 'frame_ground_truth_sha256': sha256(sidecar),
            'mask_path': mask_path.name, 'mask_sha256': sha256(mask_path),
            'empty_mask_path': 'empty_truth.png', 'empty_mask_sha256': sha256(out / 'empty_truth.png'),
            'operation_ground_truth': {'requested_interval_seconds': list(REQUESTED_INTERVAL) if donor else None,
                                       'actual_frame_exposure_interval_seconds': actual_interval,
                                       'interval_convention': 'start inclusive, end exclusive; quantized output-frame exposure',
                                       'edited_frame_indices': [f['output_frame_index'] for f in active_frames],
                                       'patch_box_xywh': list(PATCH_BOX) if donor else None,
                                       'donor_crop_xywh': list(DONOR_BOX) if donor else None},
            'host_looped': any(p['source_cycle_index'] > 0 for p in used_host),
            'donor_looped': any(p['source_cycle_index'] > 0 for p in used_donor), 'any_parent_looped': looped,
            'host_unique_source_frames_used': len({p['original_pts'] for p in used_host}),
            'loop_and_resampling_limit': MOTION_LIMIT,
            'is_unmodified_original_capture': False, 'audio_policy': 'No audio exported; audio is outside this fixture scope.',
            'ai_operation_limit': 'Composite of a labelled train-only generated-source patch; no inpainting/generator invocation.' if operation == 'ai_patch_composite' else None,
            'source_url': 'public_genvideo_v1/manifest.json', 'scope': SCOPE, 'formal_acceptance': False}




