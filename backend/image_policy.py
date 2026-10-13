"""Separate, calibrated generation and editing heads with a monotone union.

All numbers come from frozen development fits. The union cannot suppress a
strong head merely because the other head addresses a different manipulation.
No filenames, source labels, EXIF or OCR outcomes become authenticity features.
"""
import math
import numpy as np

from .evidence_fusion import evidence_features

SCHEMA = 'separate_risk_heads_v1'


def sigmoid(value):
    if not math.isfinite(value):
        raise ValueError('Non-finite fitted evidence')
    return 1. / (1. + math.exp(-max(-50., min(50., value))))


def numeric_raw(spec, values):
    """JSON-only estimator; no pickle or executable estimator deserialization."""
    kind = spec['estimator_type']
    if kind == 'max_experts_v1':
        if not 1 <= len(spec['heads']) <= 8:
            raise ValueError('Invalid expert-head count')
        return max(numeric_probability(head, values) for head in spec['heads'])
    vector = np.asarray([values[key] for key in spec['features']], dtype=float)
    mean, scale = (np.asarray(spec[key], dtype=float) for key in ('scaler_mean', 'scaler_scale'))
    if vector.ndim != 1 or vector.shape != mean.shape or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError('Invalid fitted head dimensions')
    if not np.isfinite(vector).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Non-finite fitted head features')
    vector = (vector - mean) / scale
    if kind == 'logistic_evidence_v1':
        coefficient = np.asarray(spec['coefficients'], dtype=float)
        if coefficient.shape != vector.shape or not np.isfinite(coefficient).all():
            raise ValueError('Invalid fitted head coefficients')
        return float(vector @ coefficient + spec['intercept'])
    if kind != 'numeric_evidence_boosted_trees_v1' or not 1 <= len(spec['trees']) <= 256:
        raise ValueError('Unsupported fitted head')
    raw = float(spec['baseline_logit'])
    for tree in spec['trees']:
        if not 1 <= len(tree) <= 63:
            raise ValueError('Invalid fitted head tree')
        node = 0
        while True:
            entry = tree[node]
            if not math.isfinite(entry['value']) or not math.isfinite(entry['threshold']):
                raise ValueError('Non-finite fitted head tree')
            if entry['leaf']:
                raw += entry['value']
                break
            feature = entry['feature']
            if not 0 <= feature < len(vector):
                raise ValueError('Invalid fitted head feature')
            child = entry['left'] if vector[feature] <= entry['threshold'] else entry['right']
            if not node < child < len(tree):
                raise ValueError('Cyclic fitted head tree')
            node = child
    return raw


def numeric_probability(spec, values):
    slope, intercept = float(spec['platt_coefficient']), float(spec['platt_intercept'])
    if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
        raise ValueError('Invalid monotone calibration')
    return sigmoid(numeric_raw(spec, values) * slope + intercept)


def generation_values(raw, origin_probability):
    from .generation_experts import FEATURES
    from .representation_adapter import FEATURES as ADAPTED_FEATURES
    if set(raw) not in (set(FEATURES), set(ADAPTED_FEATURES)) or not all(math.isfinite(float(v)) for v in raw.values()):
        raise ValueError('Invalid complementary expert features')
    if origin_probability is None or not math.isfinite(origin_probability) or not 0 <= origin_probability <= 1:
        raise ValueError('Image-origin adapter unavailable')
    values = dict(raw)
    for name in ('commfor224', 'commfor384'):
        if name + '.center_logit' in raw:
            values[name + '.max_logit'] = max(raw[name + '.center_logit'], raw[name + '.global_logit'])
    bounded = min(.99999, max(.00001, origin_probability))
    values['standalone_generation.logit'] = math.log(bounded / (1 - bounded))
    return values


def fuse_separate(branches, config):
    spec = config['image_fusion']
    by = {b.branch_id: b for b in branches}
    failed = [b.branch_id for b in branches if b.status == 'failed']
    required = spec['required_features']
    missing = [name for name in required if name not in by or by[name].status != 'success'
               or by[name].score is None or not math.isfinite(by[name].score) or not 0 <= by[name].score <= 1]
    base = {'calibration_status': 'fitted_separate_generation_edit_heads',
            'threshold_status': 'selected_on_separate_public_threshold_partition',
            'threshold': spec['threshold'], 'weights': {}, 'rule_overrides': [],
            'valid_visual_branches': [name for name in required if name not in missing],
            'score_kind': 'public_reference_calibrated_risk', 'fusion_method': SCHEMA,
            'probability_scope': spec['scope'],
            'meaning': '材料生成或编辑的研究风险线索；低分不是真实性证明，也不等于保险欺诈概率。'}
    try:
        if spec.get('trained') is not True or spec.get('test_used') is not False:
            raise ValueError('Untrained or contaminated image policy')
        if missing:
            raise ValueError('Missing required model evidence: ' + ', '.join(missing))
        raw = by['generation_experts'].details['raw_features']
        gen_values = generation_values(raw, by['standalone_generation'].score)
        generation = numeric_probability(spec['generation_head'], gen_values)
        if abs(generation - by['generation_experts'].score) > 1e-6:
            raise ValueError('Generation score/configuration mismatch')
        edit = numeric_probability(spec['editing_head'], evidence_features(branches))
        union = max(generation, edit)
        slope = float(spec['union_platt_coefficient'])
        if not math.isfinite(slope) or slope <= 0:
            raise ValueError('Union calibration must preserve risk ordering')
        risk = sigmoid(union * slope + float(spec['union_platt_intercept']))
        # Compare calibrated source heads. Raw logits from differently trained
        # classifiers have different scales and cannot establish disagreement.
        votes = [generation, by['standalone_generation'].score]
        disagreement = max(votes) - min(votes)
        incomplete = bool(failed)
        verdict = 'suspected_fake' if risk >= spec['threshold'] else 'undetermined' if incomplete else 'lower_risk'
        return {**base, 'status': 'partial' if incomplete else 'success',
                'execution_complete': not incomplete, 'verdict': verdict,
                'risk': round(risk, 6), 'weighted_risk': round(union, 6),
                'public_reference_probability': risk, 'public_video_reference_probability': None,
                'head_probabilities': {'generated_origin': generation, 'local_edit': edit},
                'head_semantics': 'Two distinct reference labels; independent calibrated heads; monotone max union',
                'model_disagreement': round(disagreement, 6), 'review_recommended': disagreement >= .8,
                'failed_branches': failed, 'missing_required_branches': [],
                'low_score_limit': '未检出显著证据不能确认真实；未覆盖的生成器、截图或压缩可能漏检。'}
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        return {**base, 'status': 'insufficient_evidence', 'execution_complete': False,
                'verdict': 'undetermined', 'risk': None, 'public_reference_probability': None,
                'failed_branches': sorted(set(failed + ['fusion'])), 'missing_required_branches': missing,
                'reason': str(error), 'calibration_status': 'not_applicable_to_incomplete_or_invalid_evidence'}
