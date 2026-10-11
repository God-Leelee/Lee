"""Measure a preselected original-image partition through the running app.

This never fits parameters or passes labels/captions/source tools to inference.
Use a distinct run name for each frozen configuration; completed inputs are
not silently remeasured after a restart. Acquisition must have finished first.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT, atomic_json, sha256
from scripts.benchmark import wilson


def freeze(path, value):
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError(f'Frozen input changed: {path.name}; choose a new run')
    if not path.exists():
        atomic_json(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/public_images_standalone_v1')
    parser.add_argument('--split', choices=['fusion', 'calibration', 'threshold', 'public_holdout'], required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--base', default='http://127.0.0.1:8765')
    parser.add_argument('--regression', action='store_true', help='Already evaluated parents; not new independent accuracy evidence')
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError('A run name must be one local directory component')
    data = args.data.resolve()
    provenance = json.loads((data/'provenance.json').read_text(encoding='utf-8'))
    full = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    rows = [row for row in full if row['split'] == args.split]
    assert len(rows) == 100 and sum(row['truth_label'] == 'genuine' for row in rows) == 70
    assert len({row['sample_id'] for row in full}) == len(full)
    assert len({row['sha256'] for row in full}) == len(full)
    groups = {split: {r['source_group'] for r in full if r['split'] == split}
              for split in ['fusion', 'calibration', 'threshold', 'public_holdout']}
    for left in groups:
        for right in groups:
            assert left == right or not groups[left] & groups[right]
    session = requests.Session()
    ready_deadline = time.perf_counter()+180
    while True:
        try:
            health = session.get(args.base+'/api/health', timeout=10).json()
            if health['status'] == 'ready':
                break
        except requests.RequestException:
            pass
        if time.perf_counter() > ready_deadline:
            raise TimeoutError('Service did not finish loading before evaluation; no inputs submitted')
        time.sleep(1)
    assert health['status'] == 'ready' and not health['model_errors']
    config = health['configuration']
    output = ROOT/'benchmark_results'/args.run_name
    output.mkdir(parents=True, exist_ok=True)
    freeze(output/'frozen_manifest.json', rows)
    freeze(output/'frozen_configuration.json', config)
    freeze(output/'frozen_model_manifest.json', health['models'])
    unused = {'development_texture.py', 'standalone_generation.py'} if not config.get('image_origin_adapter', {}).get('enabled') else set()
    runtime_files = [p for p in (ROOT/'backend').glob('*.py') if p.name not in unused]
    implementation = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(runtime_files+[Path(__file__).resolve()])}
    freeze(output/'frozen_implementation.json', implementation)
    path_records = output/'records.json'
    records = json.loads(path_records.read_text(encoding='utf-8')) if path_records.exists() else []
    previous = {r['sample_id']: r for r in records}
    assert len(previous) == len(records) and set(previous) <= {r['sample_id'] for r in rows}
    for index, row in enumerate(rows):
        if row['sample_id'] in previous:
            assert previous[row['sample_id']]['sha256'] == row['sha256']
            continue
        media = (data/row['relative_path']).resolve()
        assert media.is_relative_to(data) and sha256(media) == row['sha256']
        record = {key: row[key] for key in ['sample_id', 'sha256', 'truth_label', 'generator_tool', 'source_group']}
        record.update({'generator_version': row.get('generator_version'), 'split': args.split})
        began = time.perf_counter()
        try:
            with media.open('rb') as stream:
                response = session.post(args.base+'/api/cases', files={'files': ('evaluation_input'+media.suffix, stream)},
                                        data={'title': '独立图片核验'}, timeout=30)
            response.raise_for_status()
            case_id = response.json()['case_id']
            deadline = time.perf_counter()+90
            while True:
                case = session.get(args.base+'/api/cases/'+case_id, timeout=15).json()
                material = case['materials'][0]
                result = material.get('result', {})
                if case['status'] in {'complete', 'partial', 'failed'} and 'api_total_ms' in result.get('timing', {}):
                    break
                if time.perf_counter() > deadline:
                    raise TimeoutError('Persisted terminal result timeout')
                time.sleep(.025)
            latency = (time.perf_counter()-began)*1000
            assert result.get('sha256') == row['sha256'] and result['version'] == config['version']
            record.update({'case_id': case_id, 'material_id': material['material_id'],
                           'execution_status': result['execution_status'], 'prediction': result['fusion']['verdict'],
                           'risk': result['fusion']['risk'], 'latency_ms': latency,
                           'server_latency_ms': result['timing']['api_total_ms'], 'within_target': latency <= 5000,
                           'branch_scores': {b['branch_id']: b['score'] for b in result['branches']},
                           'branch_status': {b['branch_id']: b['status'] for b in result['branches']},
                           'fusion': result['fusion'],
                           'localization': next(b['details'] for b in result['branches'] if b['branch_id'] == 'localization'),
                           'copy_move': next(b['details'] for b in result['branches'] if b['branch_id'] == 'copy_move')})
            if args.split == 'public_holdout':
                prefix = f'/api/cases/{case_id}/materials/{material["material_id"]}'
                original = session.get(args.base+prefix+'/original', timeout=30)
                original.raise_for_status()
                assert hashlib.sha256(original.content).hexdigest() == row['sha256']
                for asset in [result['media']['preview'], record['localization']['mask'], record['localization']['heatmap']]:
                    session.get(args.base+prefix+'/assets/'+asset, timeout=30).raise_for_status()
                for kind in ['json', 'csv', 'html']:
                    session.get(args.base+f'/api/cases/{case_id}/report/{kind}', timeout=30).raise_for_status()
                record['evidence_and_exports_verified'] = True
        except Exception as exc:
            record.update({'execution_status': 'failed', 'prediction': 'undetermined', 'risk': None,
                           'latency_ms': (time.perf_counter()-began)*1000,
                           'error': f'{type(exc).__name__}: {exc}'})
        records.append(record)
        atomic_json(path_records, records)
        print(f'{args.split} {index+1}/{len(rows)}: {record["prediction"]}, {record["latency_ms"]/1000:.3f}s', flush=True)
    final_health = session.get(args.base+'/api/health', timeout=15).json()
    if final_health['configuration'] != config or any(sha256(ROOT/p) != digest for p, digest in implementation.items()):
        raise RuntimeError('Runtime changed during evaluation; partial records retained, no summary published')
    genuine = [r for r in records if r['truth_label'] == 'genuine']
    fake = [r for r in records if r['truth_label'] == 'fake']
    valid = lambda r: r['execution_status'] == 'complete' and r['prediction'] in {'lower_risk', 'suspected_fake'}
    tp = sum(valid(r) and r['prediction'] == 'suspected_fake' for r in fake)
    tn = sum(valid(r) and r['prediction'] == 'lower_risk' for r in genuine)
    fp = sum(valid(r) and r['prediction'] == 'suspected_fake' for r in genuine)
    fn = sum(valid(r) and r['prediction'] == 'lower_risk' for r in fake)
    failed = sum(not valid(r) for r in records)
    latencies = [r['latency_ms'] for r in records]
    metrics = {'total': len(records), 'genuine': len(genuine), 'fake': len(fake), 'tp': tp, 'tn': tn,
               'fp': fp, 'fn': fn, 'failed': failed, 'accuracy': (tp+tn)/len(records),
               'genuine_fpr': fp/len(genuine), 'fake_recall': tp/len(fake),
               'accuracy_wilson_95': wilson(tp+tn, len(records)), 'genuine_fpr_wilson_95': wilson(fp, len(genuine)),
               'latency_p95_ms': float(np.percentile(latencies, 95)), 'latency_max_ms': max(latencies),
               'over_target': sum(t > 5000 for t in latencies),
               'traceable_tools': sorted({r['generator_tool'] for r in fake})}
    summary = {'status': 'completed', 'formal_acceptance': False, 'dataset_name': args.run_name,
               'configuration_id': config['configuration_id'], 'version': config['version'], 'split': args.split,
               'independent_accuracy_test': args.split == 'public_holdout' and not args.regression, 'metrics': metrics,
               'per_generator': {tool: {'total': len(part := [r for r in fake if r['generator_tool'] == tool]),
                                       'correct': sum(valid(r) and r['prediction'] == 'suspected_fake' for r in part)}
                                 for tool in metrics['traceable_tools']},
               'frozen_manifest_sha256': sha256(output/'frozen_manifest.json'),
               'configuration_sha256': sha256(output/'frozen_configuration.json'),
               'acquisition_provenance_sha256': sha256(data/'provenance.json'),
               'source': provenance,
               'acceptance_checks': {'accuracy_95': metrics['accuracy'] >= .95, 'genuine_fpr_3': metrics['genuine_fpr'] <= .03,
                                     'every_material_within_5_seconds': metrics['over_target'] == 0,
                                     '100_cases_70_30': True, '3_generator_sources': len(metrics['traceable_tools']) >= 3,
                                     'image_and_video_scope': False, 'contest_business_sample_scope': False,
                                     'spatial_temporal_ground_truth': False, 'no_required_execution_failures': failed == 0},
               'notes': ('已评测原图片的交付策略回归，不增加独立精度样本。' if args.regression else '')+'按字幕父源分组的公开原始图片；失败与拒判计入分母。图像来源标签不替代保险业务及局部篡改、视频验收。耗时从HTTP上传开始到完整证据持久化，额外下载和导出核查在计时后。'}
    atomic_json(output/'summary.json', summary)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
