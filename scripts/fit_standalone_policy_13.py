"""Freeze the full IMAGE policy using threshold-development inputs only.

The classifier is already fitted on fusion/calibration partitions. Local-edit
and copy guards participate in threshold selection, not just the new head.
No held-out images or outcomes are opened, and the live config is not changed.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT, atomic_json, sha256
from backend.standalone_generation import predict, validate_spec


def main():
    source = ROOT/'models/candidates/public_image_adapter_v1.3.json'
    spec = json.loads(source.read_text(encoding='utf-8'))
    validate_spec(spec, spec['encoder_sha256'], spec['wave_checkpoint_sha256'])
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    work = ROOT/'data/public_images_standalone_v1/training_v1.3'
    data = np.load(work/'features.npz', allow_pickle=False)
    lookup = {str(value): i for i, value in enumerate(data['sample_ids'])}
    assert len(lookup) == len(data['sample_ids'])
    records_path = ROOT/'benchmark_results/standalone-threshold-baseline-v1.2/records.json'
    records = json.loads(records_path.read_text(encoding='utf-8'))
    assert len(records) == 100 and set(spec['sample_ids']['threshold']) == {r['sample_id'] for r in records}
    examples = []
    for row in records:
        assert row['execution_status'] == 'complete'
        index = lookup[row['sample_id']]
        assert str(data['source_hashes'][index]) == row['sha256']
        probability = predict(spec, data['features'][index])
        scores = row['branch_scores']
        loc = row['localization']
        copy = row['copy_move']
        risk = probability
        overrides = []
        if scores['localization'] >= config.get('image_strong_localization_min_score', .85) and loc.get('reliable_area_ratio', 0) >= .005:
            risk = max(risk, scores['localization'])
            overrides.append('strong_localization')
        if scores['copy_move'] >= .85 and copy.get('inliers', 0) >= 16:
            risk = max(risk, scores['copy_move'])
            overrides.append('strong_geometric_copy')
        examples.append({'sample_id': row['sample_id'], 'sha256': row['sha256'], 'truth_label': row['truth_label'],
                         'adapter_probability': probability, 'risk': risk, 'rule_overrides': overrides})
    risks = np.asarray([r['risk'] for r in examples])
    fake = np.asarray([r['truth_label'] == 'fake' for r in examples])
    assert np.sum(~fake) == 70 and np.sum(fake) == 30
    # Midpoints avoid defining a decision boundary on one serialized score.
    distinct = np.unique(risks)
    candidates = np.unique(np.r_[np.linspace(.01, .99, 99), (distinct[:-1]+distinct[1:])/2, 1.])
    best = None
    for threshold in candidates:
        high = risks >= threshold
        fp = int(np.sum(high & ~fake))
        if fp/70 > .03:
            continue
        correct = int(np.sum(high == fake))
        item = (correct, float(threshold))
        if best is None or item > best[0]:
            best = (item, high)
    assert best is not None
    high = best[1]
    result = deepcopy(spec)
    result.update({'version': '1.3.0', 'threshold': best[0][1], 'classifier_candidate_sha256': sha256(source),
                   'full_image_policy': 'max(calibrated four-view source probability, valid strong localization, valid geometric copy)',
                   'full_policy_threshold_metrics': {'total': 100, 'genuine': 70, 'fake': 30,
                      'accuracy': float(np.mean(high == fake)), 'genuine_fpr': float(np.mean(high[~fake])),
                      'fake_recall': float(np.mean(high[fake])), 'threshold': best[0][1]},
                   'threshold_policy_record_sha256': sha256(records_path),
                   'strong_rules': {'localization_min_score': config.get('image_strong_localization_min_score', .85),
                                    'localization_min_reliable_area': .005, 'copy_min_score': .85, 'copy_min_inliers': 16},
                   'threshold_selection': 'threshold100 only; full-policy correct/ALL with FPR<=3%; midpoint and grid boundaries; larger threshold on ties',
                   'runtime_deployed': False, 'formal_acceptance': False})
    validate_spec(result, result['encoder_sha256'], result['wave_checkpoint_sha256'])
    destination = ROOT/'models/candidates/public_image_policy_v1.3.json'
    if destination.exists():
        assert json.loads(destination.read_text(encoding='utf-8')) == result
    else:
        atomic_json(destination, result)
    atomic_json(ROOT/'runs/public_image_policy_v1.3.json', {'model_sha256': sha256(destination),
                'development_only': True, 'holdout_used': False, 'metrics': result['full_policy_threshold_metrics'],
                'classifier_threshold_before_full_policy': spec['threshold'], 'examples': examples,
                'not_approved_as_business_fraud_probability': True})
    print(json.dumps(result['full_policy_threshold_metrics']), flush=True)


if __name__ == '__main__':
    main()
