"""Versioned numeric ensemble inputs, shared by training and live inference.

No names, source labels, annotations, captions or metadata become features.
Missing optional geometry remains a missingness flag, never a genuine label.
"""
import math
import numpy as np
import cv2

SCHEMA='evidence_stacker_v2'


def localization_statistics(probability,reliability,mask_threshold=.5):
    binary=((probability>=mask_threshold)&(reliability>=.5)).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(binary,8)
    regions=stats[1:,4];contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    largest=max(contours,key=cv2.contourArea) if contours else None
    area=cv2.contourArea(largest) if largest is not None else 0.
    perimeter=cv2.arcLength(largest,True) if largest is not None else 0.
    inside=probability[binary.astype(bool)];outside=probability[~binary.astype(bool)]
    return {'reliable_area_ratio':float(binary.mean()),'mean_reliability':float(reliability.mean()),
      'largest_region_area_ratio':float(max(regions,default=0)/binary.size),
      'largest_region_compactness':float(4*np.pi*area/(perimeter*perimeter)) if perimeter else 0.,
      'region_count':min(12,int(np.sum(regions>=max(9,binary.size*.0002)))),
      'inside_outside_probability_delta':float(inside.mean()-outside.mean()) if inside.size and outside.size else 0.}


def evidence_features(branches):
    by={b.branch_id:b for b in branches}
    def score(name):
        b=by.get(name)
        if b is None or b.status!='success' or b.score is None:return None
        return float(b.score) if math.isfinite(b.score) and 0<=b.score<=1 else None
    def logit(value):
        if value is None:return 0.
        p=min(.99999,max(.00001,value));return math.log(p/(1-p))
    def details(name):
        b=by.get(name);return b.details if b is not None and b.status=='success' else {}
    result={}
    for name in ('generation','forensic_generation','standalone_generation','video_origin','video_generation_experts','localization','copy_move','temporal'):
        value=score(name);result[name+'.logit']=logit(value);result[name+'.missing']=float(value is None)
    loc=details('localization');copy=details('copy_move')
    for name in ('reliable_area_ratio','mean_reliability','largest_region_area_ratio','largest_region_compactness','inside_outside_probability_delta'):
        result['localization.'+name]=float(loc.get(name,0.))
    result['localization.log_regions']=math.log1p(max(0.,float(loc.get('region_count',0))))
    result['localization.log_area']=math.log(max(1e-5,float(loc.get('reliable_area_ratio',0.))))
    result['copy_move.log_inliers']=math.log1p(max(0.,float(copy.get('inliers',0))))
    result['copy_move.inlier_ratio']=float(copy.get('inlier_ratio',0.))
    result['copy_move.photometric_missing'] = float(not copy.get('photometric_available', False))
    for key in ('photometric_agreement_ratio', 'photometric_normalized_error', 'photometric_correlation'):
        result['copy_move.' + key] = float(copy.get(key, 0.))
    dense = copy.get('dense_exact', {})
    result['copy_move.dense_verified_log_pairs'] = math.log1p(max(0., float(dense.get('verified_pair_count', 0))))
    result['copy_move.dense_verified_area_ratio'] = float(dense.get('verified_area_ratio', 0.))
    result['copy_move.dense_largest_component_ratio'] = float(dense.get('largest_component_ratio', 0.))
    size=loc.get('original_size',[0,0]);pixels=float(size[0]*size[1]) if len(size)==2 else 0.
    result['copy_move.dense_area_ratio']=float(copy.get('dense_agreement_area',0.))/max(1.,pixels)
    temporal=details('temporal');windows=temporal.get('windows',[])
    risks=[float(w['temporal_risk']) for w in windows if w.get('temporal_risk') is not None]
    result['temporal.window_disagreement']=float(np.ptp(risks)) if risks else 0.
    result['temporal.window_mean']=float(np.mean(risks)) if risks else 0.
    result['video_origin.temporal_agreement_logit']=min(result['video_origin.logit'],result['temporal.logit'])
    result['video_origin.localization_agreement_logit']=min(result['video_origin.logit'],result['localization.logit'])
    frame_experts = details('video_generation_experts').get('frames', [])
    frame_probabilities = [r['generation_probability'] for r in frame_experts if r.get('generation_probability') is not None]
    result['video_generation_experts.frame_disagreement'] = float(np.ptp(frame_probabilities)) if frame_probabilities else 0.
    result['video_generation_experts.temporal_agreement_logit'] = min(result['video_generation_experts.logit'], result['temporal.logit'])
    result['video_generation_experts.localization_agreement_logit'] = min(result['video_generation_experts.logit'], result['localization.logit'])
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Non-finite ensemble evidence')
    return result


def fitted_probability(spec, branches):
    if spec.get('schema')!=SCHEMA or not spec.get('trained') or spec.get('test_used') is not False:
        raise ValueError('Ensemble is untrained or test-contaminated')
    values=evidence_features(branches)
    vector=np.asarray([values[name] for name in spec['features']],float)
    mean,scale,coefficient=(np.asarray(spec[name],float) for name in ('scaler_mean','scaler_scale','coefficients'))
    if vector.shape!=mean.shape or scale.shape!=mean.shape or coefficient.shape!=mean.shape or np.any(scale<=0):
        raise ValueError('Invalid ensemble feature/scaler dimensions')
    standardized=(vector-mean)/scale
    if spec.get('estimator_type')=='numeric_evidence_boosted_trees_v1':
        raw=float(spec['baseline_logit']);trees=spec['trees']
        if not math.isfinite(raw) or not 1<=len(trees)<=256:raise ValueError('Invalid numeric evidence-tree ensemble')
        for tree in trees:
            if not 1<=len(tree)<=63:raise ValueError('Invalid evidence-tree size')
            node=0
            while True:
                entry=tree[node]
                if not math.isfinite(entry['value']) or not math.isfinite(entry['threshold']):raise ValueError('Non-finite evidence-tree node')
                if entry['leaf']:
                    raw+=entry['value'];break
                if not 0<=entry['feature']<len(standardized):raise ValueError('Invalid evidence-tree feature')
                child=entry['left'] if standardized[entry['feature']]<=entry['threshold'] else entry['right']
                if not node<child<len(tree):raise ValueError('Cyclic or invalid evidence-tree child')
                node=child
    elif spec.get('estimator_type') in (None,'logistic_evidence_v1'):
        raw=float(standardized@coefficient+spec['intercept'])
    else:raise ValueError('Unsupported fitted evidence estimator')
    logit=raw*spec['platt_coefficient']+spec['platt_intercept']
    if not math.isfinite(logit) or spec['platt_coefficient']<=0:raise ValueError('Invalid fitted ensemble calibration')
    return 1./(1.+math.exp(-max(-50.,min(50.,logit))))
