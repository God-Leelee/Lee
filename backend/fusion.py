from __future__ import annotations

import math
import numpy as np
from .contracts import Branch
from .evidence_fusion import SCHEMA as EVIDENCE_SCHEMA, fitted_probability


VISUAL = {'generation', 'forensic_generation', 'localization', 'copy_move', 'temporal', 'standalone_generation', 'video_origin'}


def fuse(branches: list[Branch], config: dict, material_type: str) -> dict:
    review_origin = config.get('image_origin_adapter', {}).get('decision_role') == 'review_only'
    valid = [b for b in branches if b.branch_id in VISUAL and not (review_origin and b.branch_id == 'standalone_generation') and b.status == 'success' and
             b.score is not None and math.isfinite(b.score)]
    failed = [b.branch_id for b in branches if b.status == 'failed']
    if len(valid) < 2:
        return {'status': 'failed' if failed else 'insufficient_evidence', 'risk': None, 'verdict': 'undetermined',
                'valid_visual_branches': [b.branch_id for b in valid], 'weights': {},
                'reason': '少于两个有效视觉分支，不能形成融合判断。',
                'failed_branches': failed, 'calibration_status': config['calibration_status']}
    weights = {b.branch_id: config['weights'].get(b.branch_id, 1) for b in valid}
    denominator = sum(weights.values())
    weights = {key: val / denominator for key, val in weights.items()}
    weighted = sum(weights[b.branch_id] * b.score for b in valid)
    calibration_status = config.get(f'{material_type}_calibration_status', config['calibration_status'])
    threshold_status = config['threshold_status']
    reference_probability = None
    spec = config.get(f'{material_type}_fusion')
    evidence_policy = bool(spec and spec.get('schema') == EVIDENCE_SCHEMA)
    adapter = config.get('image_origin_adapter', {}) if material_type == 'image' else {}
    adapter_active = bool(adapter.get('enabled')) and not review_origin
    valid_scores = {b.branch_id:b.score for b in valid}
    required = spec.get('required_features',spec['features'][:4]) if spec else []
    if evidence_policy and all(name in valid_scores for name in required):
        try:
            reference_probability = fitted_probability(spec, branches)
            weighted = reference_probability
            threshold_status = 'selected_on_separate_public_threshold_partition'
        except (ValueError, KeyError, TypeError) as exc:
            calibration_status = 'invalid_fitted_evidence_policy'
            failed.append('fusion')
    elif adapter_active:
        required = ['generation', 'forensic_generation', 'localization', 'standalone_generation']
        if all(name in valid_scores for name in required):
            reference_probability = valid_scores['standalone_generation']
            weighted = reference_probability
            threshold_status = 'selected_on_separate_public_threshold_partition'
            calibration_status = 'fitted_public_standalone_image_calibration_partition'
        else:
            calibration_status = 'not_applicable_to_missing_required_model_output'
            threshold_status = 'unvalidated_research_default'
    elif spec and all(name in valid_scores for name in required):
        values = []
        for name in spec['features']:
            score = valid_scores.get(name)
            bounded = min(.999,max(.001,score)) if score is not None else None
            values.append(math.log(bounded/(1-bounded)) if bounded is not None else 0)
        values += [int(name not in valid_scores) for name in spec['features']]
        standardized = (np.asarray(values)-np.asarray(spec['scaler_mean']))/np.asarray(spec['scaler_scale'])
        raw = float(standardized @ np.asarray(spec['coefficients']) + spec['intercept'])
        logit = min(50,max(-50,raw*spec['platt_coefficient']+spec['platt_intercept']))
        reference_probability = 1/(1+math.exp(-logit))
        weighted = reference_probability
        threshold_status = 'selected_on_separate_public_threshold_partition'
    elif spec:
        calibration_status = 'not_applicable_to_missing_required_model_output'
        threshold_status = 'unvalidated_research_default'
    # A strong independently localized edit must not be averaged away by a low
    # whole-image generation score. This rule is visible and versioned.
    loc = next((b for b in valid if b.branch_id == 'localization'), None)
    copy = next((b for b in valid if b.branch_id == 'copy_move'), None)
    risk = weighted
    overrides = []
    if not evidence_policy and loc and loc.score >= config.get(f'{material_type}_strong_localization_min_score',.85) and loc.details.get('reliable_area_ratio', 0) >= .005:
        risk = max(risk, loc.score)
        overrides.append('strong_localization')
    if not evidence_policy and copy and copy.score >= .85 and copy.details.get('inliers', 0) >= 16:
        risk = max(risk, copy.score)
        overrides.append('strong_geometric_copy')
    threshold = spec['threshold'] if evidence_policy and reference_probability is not None else adapter['threshold'] if adapter_active and reference_probability is not None else spec['threshold'] if reference_probability is not None else .7 if spec else config[f'{material_type}_threshold']
    incomplete = bool(failed) or bool((spec or adapter_active) and reference_probability is None)
    verdict = 'suspected_fake' if risk >= threshold else 'undetermined' if incomplete else 'lower_risk'
    return {'status': 'partial' if incomplete else 'success', 'risk': round(risk, 6),
            'verdict': verdict, 'execution_complete':not incomplete,
            'score_kind': 'uncalibrated_risk', 'threshold': threshold,
            'threshold_status': threshold_status, 'calibration_status': calibration_status,
            'public_video_reference_probability': reference_probability,
            'public_reference_probability': reference_probability,
            'probability_scope': spec['scope'] if evidence_policy and reference_probability is not None else adapter['scope'] if adapter_active and reference_probability is not None else spec['scope'] if reference_probability is not None else None,
            'valid_visual_branches': [b.branch_id for b in valid], 'weights': weights,
            'weighted_risk': round(weighted, 6), 'rule_overrides': overrides, 'failed_branches': failed,
            'fusion_method': 'jointly_trained_generation_and_local_evidence' if evidence_policy else 'trained_image_origin_with_local_edit_guards' if adapter_active else 'logit_stacker_with_local_edit_guards' if spec else 'weighted_visual_evidence',
            'meaning': '材料真实性风险线索；不等同于保险欺诈概率，也不能单独证明真伪。'}
