"""Frozen public-image origin adapter; no filenames, captions or metadata input.

This probability describes image origin within the measured public reference
domain. It is not an insurance-fraud probability or a claim-legitimacy decision.
"""
from __future__ import annotations

import math
import re
import numpy as np

FEATURE_SPEC='standalone_clip_wave_four_view_l2_v1'
TEXTURE_SPEC='gray_residual_frequency_dct_uniform_lbp_cpu_v1'
BLOCK_SLICES={'clip_center':(0,768),'clip_global':(768,1536),
              'wave_center':(1536,2304),'wave_global':(2304,3072),'texture':(3072,3125)}
ALLOWED_BLOCKS=[['clip_center','clip_global'],['wave_center','wave_global'],
                ['clip_center','clip_global','wave_center','wave_global'],
                ['clip_center','clip_global','wave_center','wave_global','texture']]


def descriptor(clip_center,clip_global,wave_center,wave_global,texture=None):
    values=[]
    for raw in [clip_center,clip_global,wave_center,wave_global]:
        value=np.asarray(raw,np.float64)
        if value.shape!=(768,) or not np.isfinite(value).all():raise ValueError('Invalid frozen 768-dimensional view')
        with np.errstate(over='ignore',invalid='ignore'):norm=float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm<1e-8:raise ValueError('Invalid norm or zero frozen view cannot establish image-origin evidence')
        values.append(value/norm)
    if texture is not None:
        value=np.asarray(texture,np.float64)
        if value.shape!=(53,) or not np.isfinite(value).all():raise ValueError('Invalid CPU texture descriptor')
        values.append(value)
    return np.concatenate(values)


def transform_descriptor(spec,feature):
    feature=np.asarray(feature,np.float64)
    if feature.shape not in [(3072,),(3125,)] or not np.isfinite(feature).all():raise ValueError('Invalid four-view descriptor')
    values=[]
    for block in spec['feature_blocks']:
        lo,hi=BLOCK_SLICES[block['name']]
        if hi>len(feature):raise ValueError('Selected adapter requires missing texture features')
        value=feature[lo:hi].copy()
        if block['name']=='texture':
            with np.errstate(over='ignore',invalid='ignore'):value=(value-np.asarray(block['mean']))/np.asarray(block['std'])
        with np.errstate(over='ignore',invalid='ignore'):norm=float(np.linalg.norm(value))
        if not np.isfinite(value).all() or not np.isfinite(norm) or (block['name']!='texture' and norm<1e-8):
            raise ValueError('Invalid norm or zero required deep feature block')
        value/=max(norm,1e-8)
        values.append(value*block['scale'])
    return np.concatenate(values)


def predict(spec,feature):
    value=transform_descriptor(spec,feature)
    if value.shape!=(len(spec['coefficients']),):raise ValueError('Incompatible adapter coefficients')
    with np.errstate(over='ignore',invalid='ignore'):raw=float(value@np.asarray(spec['coefficients'],np.float64)+spec['intercept'])
    if not np.isfinite(raw):raise ValueError('Invalid fitted public-image raw score')
    logit=max(-50.,min(50.,raw*spec['platt_coefficient']+spec['platt_intercept']))
    return 1./(1.+math.exp(-logit))


def validate_spec(spec,clip_sha,wave_sha):
    if spec.get('trained') is not True or spec.get('feature_spec')!=FEATURE_SPEC:raise ValueError('Public-image adapter is untrained or incompatible')
    if spec.get('encoder_sha256')!=clip_sha or spec.get('wave_checkpoint_sha256')!=wave_sha:raise ValueError('Public-image adapter backbone checkpoint mismatch')
    if spec.get('holdout_used_for_fit') is not False or spec.get('holdout_used_for_candidate_selection') is not False:raise ValueError('Public-image holdout contaminated fitting or selection')
    blocks=spec.get('feature_blocks',[])
    if [b.get('name') for b in blocks] not in ALLOWED_BLOCKS:raise ValueError('Unsupported public-image feature blocks')
    for block in blocks:
        dimension=53 if block['name']=='texture' else 768
        if block.get('dimensions')!=dimension or not np.isfinite(block['scale']) or block['scale']<=0:raise ValueError('Invalid public-image block dimension or scale')
        if block['name']=='texture':
            mean,std=np.asarray(block['mean']),np.asarray(block['std'])
            if (block.get('normalization')!='standardized_l2' or block.get('texture_spec')!=TEXTURE_SPEC
                or mean.shape!=(53,) or std.shape!=(53,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std<=0)):raise ValueError('Invalid train-only texture transform')
        elif block.get('normalization')!='l2':raise ValueError('Invalid frozen-view transform')
    dims=sum(b['dimensions'] for b in blocks)
    numbers=[spec['intercept'],spec['platt_coefficient'],spec['platt_intercept'],spec['threshold'],*spec['coefficients']]
    if len(spec['coefficients'])!=dims or not np.isfinite(numbers).all() or not 0<spec['threshold']<=1 or spec['platt_coefficient']<=0:raise ValueError('Invalid public-image fitted parameters')
    ids=spec.get('sample_ids',{});groups=spec.get('source_groups',{});seen_ids=set();seen_groups=set()
    expanded = spec.get('architecture') == 'frozen_clip_wave_multiblock_logistic_v2'
    partitions = [('fusion',3000,500),('calibration',100,100),('threshold',100,100)] if expanded else [('fusion',600,100),('calibration',100,100),('threshold',100,100)]
    for split,size,group_count in partitions:
        this_ids=ids.get(split,[]);this_groups=groups.get(split,[])
        if len(this_ids)!=size or len(set(this_ids))!=size or seen_ids&set(this_ids):raise ValueError('Public-image development IDs missing, duplicate or overlapping')
        if len(this_groups)!=group_count or len(set(this_groups))!=group_count or seen_groups&set(this_groups):raise ValueError('Public-image caption source groups overlap or are missing')
        seen_ids.update(this_ids);seen_groups.update(this_groups)
    for key in ['development_manifest_sha256','feature_cache_sha256']:
        if not re.fullmatch('[0-9a-f]{64}',spec.get(key,'')):raise ValueError('Public-image adapter lacks frozen input hashes')
