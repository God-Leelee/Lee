"""Evaluate real Engine outputs on related-parent known-operation video fixtures.

Run serially with other GPU work. Every planned fixture remains in denominators,
including model/startup, analysis, persistence, GT-alignment and missing-mask
failures. Discrete sampled mask points are never bridged into a true interval.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT, atomic_json, load_config, sha256

SCOPE = 'Train-parent known-operation video engineering test; not independent contest/business accuracy or authentic motion validation.'


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes fixture/evidence root: ' + relative)
    return path


def check_hash(path, expected):
    if not expected or sha256(path) != expected:
        raise ValueError('File hash mismatch: ' + str(path))


def load_gt(data, row, allowed_parent_splits=('train',)):
    if row.get('split') != 'engineering' or row.get('material_type') != 'video':
        raise ValueError('Only declared video engineering fixtures are supported')
    if not row.get('parents') or any(p.get('split') not in allowed_parent_splits for p in row['parents']):
        raise ValueError('Every fixture parent must use a declared allowed split')
    if row.get('parent_sha256') != [p['sha256'] for p in row['parents']]:
        raise ValueError('Parent provenance hash list is inconsistent')
    for parent in row['parents']:
        timing = json.loads(inside(data, parent['timing_path']).read_text(encoding='utf-8'))
        if any(timing.get(key) != parent.get(key) for key in ('sample_id', 'sha256', 'split', 'truth_label', 'source_group')):
            raise ValueError('Original parent PTS sidecar disagrees with train-parent provenance')
    source = inside(data, row['relative_path'])
    check_hash(source, row['sha256'])
    gt_path = inside(data, row['frame_ground_truth_path'])
    check_hash(gt_path, row['frame_ground_truth_sha256'])
    check_hash(inside(data, row['mask_path']), row['mask_sha256'])
    check_hash(inside(data, row['empty_mask_path']), row['empty_mask_sha256'])
    frames = json.loads(gt_path.read_text(encoding='utf-8'))
    if len(frames) != row['frame_count']:
        raise ValueError('GT frame-count mismatch')
    for index, frame in enumerate(frames):
        timestamp = frame['timestamp_seconds']
        pts_time = float(Fraction(frame['pts']) * Fraction(frame['time_base']))
        if frame['output_frame_index'] != index or not math.isfinite(timestamp) or abs(pts_time - timestamp) > 1e-6:
            raise ValueError('Invalid GT frame PTS/timestamp')
        if index and timestamp <= frames[index-1]['timestamp_seconds']:
            raise ValueError('Non-increasing GT timestamps')
        if (frame['width'], frame['height']) != (row['width'], row['height']):
            raise ValueError('GT coordinate-space mismatch')
        if frame['mask_path'] != (row['mask_path'] if frame['gt_active'] else row['empty_mask_path']):
            raise ValueError('GT mask schedule inconsistent with operation')
    active = [f['output_frame_index'] for f in frames if f['gt_active']]
    if active != row['operation_ground_truth']['edited_frame_indices']:
        raise ValueError('Operation interval and frame GT disagree')
    interval = row['operation_ground_truth']['actual_frame_exposure_interval_seconds']
    if active:
        expected_interval = [frames[active[0]]['timestamp_seconds'], frames[active[-1]]['timestamp_seconds'] + 1 / row['fps']]
        if interval is None or any(abs(a-b) > 1e-6 for a, b in zip(interval, expected_interval)):
            raise ValueError('GT interval disagrees with actual encoded frame exposure')
    elif interval is not None:
        raise ValueError('Unedited control must not carry a positive operation interval')
    return source, frames


def aligned_gt(frame, gt, timestamps):
    timestamp = frame.get('timestamp_seconds')
    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError('Engine sampled frame lacks actual timestamp')
    if frame.get('pts') is None or not frame.get('time_base'):
        raise ValueError('Engine sampled frame lacks actual PTS/time base')
    pts_time = float(Fraction(frame['pts']) * Fraction(frame['time_base']))
    if abs(pts_time - timestamp) > 1e-6:
        raise ValueError('Engine timestamp disagrees with PTS in zero-origin fixture')
    index = bisect_left(timestamps, timestamp)
    candidates = [i for i in (index-1, index) if 0 <= i < len(gt)]
    nearest = min(candidates, key=lambda i: abs(timestamps[i]-timestamp))
    if abs(timestamps[nearest]-timestamp) > 1e-6:
        raise ValueError('Sampled PTS does not correspond to an exported GT frame')
    return gt[nearest]


def score_localization(result, evidence, data, row, gt):
    sampled, localized, points, errors = [], [], [], []
    timestamps = [g['timestamp_seconds'] for g in gt]
    seen, sampled_ids = set(), set()
    aggregate = next((b for b in result.get('branches', []) if b.get('branch_id') == 'localization'), None)
    selected_ids = aggregate.get('details', {}).get('localized_frame_ids', aggregate.get('details', {}).get('selected_frame_ids')) if aggregate else None
    if not isinstance(selected_ids, list) or any(not isinstance(i, int) for i in selected_ids) or len(set(selected_ids)) != len(selected_ids):
        errors.append({'frame_id': None, 'error': 'Missing/invalid aggregate localization selected_frame_ids provenance'})
        selected_ids = []
    selected_ids = set(selected_ids)
    for frame in result.get('media', {}).get('frames', []):
        truth_frame = aligned_gt(frame, gt, timestamps)
        if truth_frame['output_frame_index'] in seen:
            raise ValueError('Duplicate sampled PTS cannot create additional GT coverage')
        if frame['frame_id'] in sampled_ids:
            raise ValueError('Duplicate Engine sampled frame IDs')
        seen.add(truth_frame['output_frame_index'])
        sampled_ids.add(frame['frame_id'])
        sample = {'frame_id': frame['frame_id'], 'timestamp_seconds': frame['timestamp_seconds'],
                  'pts': frame['pts'], 'time_base': frame['time_base'],
                  'output_frame_index': truth_frame['output_frame_index'], 'gt_active': truth_frame['gt_active'],
                  'localization_status': 'not_selected'}
        local = frame.get('localization')
        if local is not None or frame['frame_id'] in selected_ids:
            sample['localization_status'] = local.get('status', 'failed') if local else 'failed'
            try:
                if local is None:
                    raise ValueError('Selected frame has no nested localization result')
                if frame['frame_id'] not in selected_ids:
                    raise ValueError('Nested localization is absent from selected-frame provenance')
                details = local.get('details', {})
                nested_timestamp = details.get('timestamp_seconds')
                if (details.get('frame_id') != frame['frame_id'] or
                        isinstance(nested_timestamp, bool) or
                        not isinstance(nested_timestamp, (int, float)) or
                        not math.isfinite(nested_timestamp) or
                        abs(nested_timestamp-frame['timestamp_seconds']) > 1e-6):
                    raise ValueError('Nested localization belongs to a different frame/PTS')
                if local.get('status') != 'success':
                    raise ValueError(local.get('reason') or 'Selected localization branch did not succeed')
                mask_name = details.get('mask')
                if not mask_name:
                    raise ValueError('Successful localization has no mask artifact')
                mask_path = inside(evidence, mask_name)
                prediction = np.asarray(Image.open(mask_path).convert('L')) > 0
                truth = np.asarray(Image.open(inside(data, truth_frame['mask_path'])).convert('L')) > 0
                if prediction.shape != truth.shape or truth.shape != (row['height'], row['width']):
                    raise ValueError('Mask dimensions differ from original fixture coordinates; no implicit resize')
                predicted, truth_count = int(prediction.sum()), int(truth.sum())
                intersection = int((prediction & truth).sum())
                outside_count = int((prediction & ~truth).sum())
                outside_total = int((~truth).sum())
                spatial = {**sample, 'mask_path': mask_name, 'mask_sha256': sha256(mask_path),
                           'predicted_area_pixels': predicted, 'predicted_area_ratio': predicted / prediction.size,
                           'gt_area_pixels': truth_count, 'intersection_pixels': intersection,
                           'outside_gt_predicted_pixels': outside_count, 'outside_gt_pixels': outside_total,
                           'outside_gt_area_ratio': outside_count / outside_total if outside_total else None,
                           'pixel_iou': intersection / max(1, predicted + truth_count - intersection) if truth_count else None,
                           'pixel_f1': 2 * intersection / max(1, predicted + truth_count) if truth_count else None}
                localized.append(spatial)
                if predicted:
                    points.append({'frame_id': frame['frame_id'], 'timestamp_seconds': frame['timestamp_seconds'],
                                   'output_frame_index': truth_frame['output_frame_index'], 'mask_path': mask_name,
                                   'predicted_area_ratio': predicted / prediction.size, 'gt_active': truth_frame['gt_active']})
            except Exception as exc:
                sample['localization_status'] = 'failed'
                errors.append({'frame_id': frame['frame_id'], 'timestamp_seconds': frame['timestamp_seconds'],
                               'error': f'{type(exc).__name__}: {exc}'})
        sampled.append(sample)
    for missing_id in sorted(selected_ids - sampled_ids):
        errors.append({'frame_id': missing_id, 'error': 'Selected localization frame absent from sampled media records'})
    gt_indices = {g['output_frame_index'] for g in gt if g['gt_active']}
    sampled_hits = [s for s in sampled if s['gt_active']]
    positive_masks = [s for s in localized if s['gt_active']]
    outside_time = [s for s in localized if not s['gt_active']]
    outside_pixels = sum(s['outside_gt_predicted_pixels'] for s in localized)
    outside_total = sum(s['outside_gt_pixels'] for s in localized)
    iou = float(np.mean([s['pixel_iou'] for s in positive_masks])) if positive_masks else None
    f1 = float(np.mean([s['pixel_f1'] for s in positive_masks])) if positive_masks else None
    complete = result.get('execution_status') == 'complete' and not errors
    positive = bool(gt_indices)
    localization_valid = complete and bool(localized) and (bool(positive_masks) if positive else True)
    selected_indices = {s['output_frame_index'] for s in positive_masks}
    # The measured score describes the masks actually produced. Coverage-aware
    # effective scores additionally keep every GT-positive sampled point in the
    # denominator, including points outside the fixed spatial-selection budget.
    effective_iou = sum(s['pixel_iou'] for s in positive_masks)/len(sampled_hits) if positive and localization_valid else 0.0 if positive else None
    effective_f1 = sum(s['pixel_f1'] for s in positive_masks)/len(sampled_hits) if positive and localization_valid else 0.0 if positive else None
    return {'selected_spatial_frame_ids': sorted(selected_ids),
            'sampled_frames': sampled, 'localized_frames': localized, 'mask_errors': errors,
            'sampled_frame_count': len(sampled), 'gt_edited_frame_count': len(gt_indices),
            'sampled_gt_hit_count': len(sampled_hits), 'sampled_gt_hit_times': [s['timestamp_seconds'] for s in sampled_hits],
            'gt_unsampled_frame_indices': sorted(gt_indices - seen),
            'gt_unsampled_frame_timestamps': [g['timestamp_seconds'] for g in gt if g['output_frame_index'] in gt_indices - seen],
            'gt_time_sampling_hit': bool(sampled_hits), 'gt_spatial_mask_hit_count': len(positive_masks),
            'at_least_one_gt_spatial_mask_available': bool(positive_masks),
            'gt_sampled_points_without_spatial_mask': [s['timestamp_seconds'] for s in sampled_hits if s['output_frame_index'] not in selected_indices],
            'gt_spatial_mask_coverage_among_sampled_points': len(positive_masks)/len(sampled_hits) if sampled_hits else None,
            'measured_pixel_iou_gt_macro': iou, 'measured_pixel_f1_gt_macro': f1,
            'measured_pixel_score_scope': 'Macro over successful selected spatial masks at actual GT-positive sampled PTS only.',
            'effective_positive_iou': effective_iou, 'effective_positive_f1': effective_f1,
            'effective_pixel_score_scope': 'All unique GT-positive sampled PTS are the per-fixture denominator; missing/unselected masks contribute zero. An incomplete pipeline zeroes the fixture score.',
            'outside_gt_predicted_pixels': outside_pixels, 'outside_gt_pixels_observed': outside_total,
            'outside_gt_area_ratio_observed': outside_pixels / outside_total if outside_total else None,
            'outside_gt_time_predicted_area_ratio_observed': float(np.mean([s['predicted_area_ratio'] for s in outside_time])) if outside_time else None,
            'outside_gt_time_mask_sample_count': len(outside_time),
            'outside_gt_time_sampled_without_spatial_mask_count': sum(not s['gt_active'] for s in sampled) - len(outside_time),
            'control_predicted_area_ratio_observed': float(np.mean([s['predicted_area_ratio'] for s in localized])) if not positive and localized else None,
            'localization_evaluation_valid': localization_valid,
            'predicted_mask_points': points, 'predicted_continuous_intervals': [],
            'temporal_representation': 'Separate actual sampled PTS points only; unsampled gaps are never bridged into true intervals.',
            'temporal_window_outputs_are_ground_truth': False}


def summarize(rows, records, config, model_info, load_error, load_ms):
    positives = [r for r in records if r['positive_fixture']]
    failed = [r for r in records if not r.get('localization_evaluation_valid', False)]
    latencies = [r['external_persisted_pipeline_ms'] for r in records if r.get('external_persisted_pipeline_ms') is not None]
    completed_latency = [r for r in records if r.get('external_persisted_pipeline_ms') is not None and r.get('execution_status') == 'complete']
    return {'status': 'completed' if not failed else 'completed_with_failures', 'scope': SCOPE, 'formal_acceptance': False,
            'configuration_id': config.get('configuration_id'), 'model_manifest': model_info,
            'model_initialization_ms_excluded_from_per_material_timer': load_ms, 'model_initialization_error': load_error,
            'metrics': {'planned_total': len(rows), 'positive_total': len(positives),
                        'control_total': len(rows)-len(positives), 'invalid_or_failed_total': len(failed),
                        'execution_incomplete_total': sum(r.get('execution_status') != 'complete' for r in records),
                        'valid_localization_rate_all_planned': (len(rows)-len(failed))/len(rows),
                        'positive_macro_iou_all_planned': sum(r.get('effective_positive_iou') or 0 for r in positives)/len(positives) if positives else None,
                        'positive_macro_f1_all_planned': sum(r.get('effective_positive_f1') or 0 for r in positives)/len(positives) if positives else None,
                        'positive_time_sampling_hit_rate_all_planned': sum(bool(r.get('gt_time_sampling_hit')) for r in positives)/len(positives) if positives else None,
                        'positive_spatial_mask_hit_rate_all_planned': sum(bool(r.get('at_least_one_gt_spatial_mask_available')) for r in positives)/len(positives) if positives else None,
                        'latency_observed_total': len(latencies),
                        'latency_p95_ms_observed_including_failed_attempts': float(np.percentile(latencies, 95)) if latencies else None,
                        'latency_max_ms_observed': max(latencies) if latencies else None,
                        'complete_within_5_seconds_rate_all_planned': sum(r['external_persisted_pipeline_ms'] <= 5000 for r in completed_latency)/len(rows),
                        'valid_localization_within_5_seconds_rate_all_planned': sum(r.get('localization_evaluation_valid', False) and r['external_persisted_pipeline_ms'] <= 5000 for r in completed_latency)/len(rows),
                        'outside_gt_area_ratios_observed': [{'sample_id': r['sample_id'], 'ratio': r.get('outside_gt_area_ratio_observed'),
                                                           'observed_mask_frames': len(r.get('localized_frames', []))} for r in records],
                        'controls_predicted_area_ratios_observed': [{'sample_id': r['sample_id'], 'ratio': r.get('control_predicted_area_ratio_observed')} for r in records if not r['positive_fixture']]},
            'metric_policy': 'Every planned fixture remains in denominators. Within each positive fixture, all unique GT-positive sampled PTS are the effective pixel-score denominator, with missing/unselected masks zeroed. Partial/failed/not-run or no-GT-mask fixtures contribute zero. Selected-mask measured scores and coverage are reported separately. Empty control GT never raises positive pixel scores. Missing-mask false-positive area is unknown, never zero.',
            'temporal_policy': 'Only discrete sampled PTS mask points reported; no interval connecting separated anomalies.',
            'results': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'data/known_video_edits')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark_results/known-operation-video-localization')
    args = parser.parse_args()
    data, out = args.fixtures.resolve(), args.output.resolve()
    rows = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Fixture manifest must be nonempty and uniquely identified')
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / 'frozen_manifest.json'
    text = json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2)
    if frozen.exists() and frozen.read_text(encoding='utf-8') != text:
        raise ValueError('Frozen fixture manifest changed; use a new --output')
    frozen.write_text(text, encoding='utf-8')
    config = load_config()
    records = [{'sample_id': r['sample_id'], 'operation': r.get('material_class'),
                'positive_fixture': bool(r.get('operation_ground_truth', {}).get('edited_frame_indices')),
                'parent_sha256': r.get('parent_sha256'), 'fixture_sha256': r.get('sha256'),
                'execution_status': 'not_run', 'localization_evaluation_valid': False,
                'effective_positive_iou': 0.0, 'effective_positive_f1': 0.0,
                'external_persisted_pipeline_ms': None} for r in rows]
    atomic_json(out / 'records.json', records)
    engine, models, load_error, load_ms = None, None, None, None
    load_start = time.perf_counter()
    try:
        # Imports/loading are deliberately here: importing metric helpers and
        # CPU fixture generation do not instantiate or call GPU models.
        from backend.models import Models
        from backend.engine import Engine
        models = Models(config)
        models.load()
        engine = Engine(config, models)
        load_ms = (time.perf_counter()-load_start)*1000
        for index, row in enumerate(rows):
            record = records[index]
            started = None
            try:
                source, gt = load_gt(data, row)
                evidence = out / 'evidence' / row['sample_id']
                started = time.perf_counter()
                result = engine.analyze(source, evidence, row['sample_id'], material_type='video', received_at=started,
                                        original_name='engineering_video.mp4')
                # Engine has returned after its final evidence/report write.
                persisted = json.loads((evidence / 'report.json').read_text(encoding='utf-8'))
                if persisted.get('sha256') != row['sha256'] or persisted.get('material_id') != row['sample_id']:
                    raise ValueError('Persisted Engine report does not match fixture identity')
                record['external_persisted_pipeline_ms'] = (time.perf_counter()-started)*1000
                record['external_timer_definition'] = 'Immediately before real Engine.analyze through final Engine report/evidence persistence, return and readable persisted-report verification; evaluation metrics excluded.'
                record.update({'execution_status': result.get('execution_status', 'failed'),
                               'material_risk': result.get('fusion', {}).get('risk'),
                               'verdict': result.get('fusion', {}).get('verdict'), 'engine_timing': result.get('timing'),
                               'branch_status': {b['branch_id']: b['status'] for b in result.get('branches', [])},
                               'operation_ground_truth': row['operation_ground_truth']})
                record.update(score_localization(persisted, evidence, data, row, gt))
                if not record['localization_evaluation_valid']:
                    record['evaluation_error'] = ('Incomplete Engine execution or selected-mask failure' if record['execution_status'] != 'complete' or record['mask_errors'] else
                                                  'No successful selected spatial mask inside GT time interval' if record['positive_fixture'] else
                                                  'No successful spatial-mask control observation')
            except Exception as exc:
                if started is not None and record['external_persisted_pipeline_ms'] is None:
                    record['external_persisted_pipeline_ms'] = (time.perf_counter()-started)*1000
                record['execution_status'] = 'failed' if record['execution_status'] == 'not_run' else record['execution_status']
                record['localization_evaluation_valid'] = False
                record['effective_positive_iou'] = record['effective_positive_f1'] = 0.0
                record['evaluation_error'] = f'{type(exc).__name__}: {exc}'
            atomic_json(out / 'records.json', records)
            print(json.dumps({'sample_id': record['sample_id'], 'execution_status': record['execution_status'],
                              'gt_mask_hits': record.get('gt_spatial_mask_hit_count'),
                              'effective_iou': record['effective_positive_iou'],
                              'external_ms': record['external_persisted_pipeline_ms'],
                              'error': record.get('evaluation_error')}, ensure_ascii=False), flush=True)
    except Exception as exc:
        load_error = f'{type(exc).__name__}: {exc}'
        load_ms = (time.perf_counter()-load_start)*1000 if load_ms is None else load_ms
        for record in records:
            if record['execution_status'] == 'not_run':
                record.update({'execution_status': 'failed', 'evaluation_error': 'Model/run initialization failed: ' + load_error})
    finally:
        if engine is not None:
            engine.cpu.shutdown(wait=True)
        for record in records:
            if record['execution_status'] == 'not_run':
                record['evaluation_error'] = 'Not attempted; retained in planned denominator'
        atomic_json(out / 'records.json', records)
        summary = summarize(rows, records, config, getattr(models, 'info', {}), load_error, load_ms)
        summary['frozen_manifest_sha256'] = sha256(frozen)
        summary['configuration_sha256'] = sha256(ROOT / 'config.json')
        atomic_json(out / 'summary.json', summary)
        print(json.dumps(summary['metrics'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
