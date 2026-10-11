"""Validated video-origin descriptor pooling; no metadata or source names."""
import math
import re
import numpy as np
from .standalone_generation import FEATURE_SPEC,ALLOWED_BLOCKS,transform_descriptor

ARCHITECTURE='frozen_clip_wave_two_keyframe_video_origin_v1'


def validate_spec(spec,clip_sha,wave_sha):
    if spec.get('architecture')!=ARCHITECTURE or spec.get('feature_spec')!=FEATURE_SPEC or spec.get('trained') is not True:
        raise ValueError('Video-origin head is untrained or incompatible')
    if spec.get('encoder_sha256')!=clip_sha or spec.get('wave_checkpoint_sha256')!=wave_sha:raise ValueError('Video-origin backbone differs')
    if spec.get('holdout_used_for_fit') is not False or spec.get('holdout_used_for_candidate_selection') is not False:raise ValueError('Video-origin reserved sources contaminated')
    blocks=spec['feature_blocks']
    if [b['name'] for b in blocks] not in ALLOWED_BLOCKS:raise ValueError('Invalid video-origin feature blocks')
    for b in blocks:
        if b['dimensions']!=(53 if b['name']=='texture' else 768) or not math.isfinite(b['scale']) or b['scale']<=0:raise ValueError('Invalid video-origin block')
        if b['name']=='texture':
            mean,std=np.asarray(b['mean']),np.asarray(b['std'])
            if mean.shape!=(53,) or std.shape!=(53,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid video-origin texture transform')
    values=[*spec['coefficients'],spec['intercept'],spec['platt_coefficient'],spec['platt_intercept'],spec['threshold']]
    if len(spec['coefficients'])!=sum(b['dimensions'] for b in blocks) or not np.isfinite(values).all() or spec['platt_coefficient']<=0 or not 0<spec['threshold']<1:raise ValueError('Invalid video-origin fitted parameters')
    if spec.get('aggregation')!='mean_raw_two_keyframe_logits' or spec.get('frame_selection')!='sampled_frame_quarter_positions':raise ValueError('Unvalidated video-origin sampling/pooling')
    groups=spec['source_groups'];ids=spec['sample_ids'];seen_groups=set();seen_ids=set()
    for split,group_count,frame_count in [('train',200,1200),('calibration',100,264),('threshold',100,264)]:
        g,i=groups[split],ids[split]
        if len(g)!=group_count or len(set(g))!=len(g) or set(g)&seen_groups:raise ValueError('Video-origin parent groups overlap/missing')
        if len(i)!=frame_count or len(set(i))!=len(i) or set(i)&seen_ids:raise ValueError('Video-origin frame IDs overlap/missing')
        seen_groups.update(g);seen_ids.update(i)
    for name in ('frame_manifest_sha256','feature_cache_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',spec.get(name,'')):raise ValueError('Video-origin provenance hash missing')


def predict(spec,descriptors):
    if len(descriptors)!=2:raise ValueError('Video-origin needs exactly two declared keyframes')
    coef=np.asarray(spec['coefficients'],float);logits=[]
    for descriptor in descriptors:
        features=transform_descriptor(spec,descriptor)
        if features.shape!=coef.shape:raise ValueError('Invalid video-origin descriptor dimensions')
        logits.append(float(features@coef+spec['intercept']))
    calibrated=float(np.mean(logits))*spec['platt_coefficient']+spec['platt_intercept']
    if not np.isfinite([*logits,calibrated]).all():raise ValueError('Non-finite video-origin prediction')
    return 1./(1.+math.exp(-max(-50.,min(50.,calibrated)))),logits
