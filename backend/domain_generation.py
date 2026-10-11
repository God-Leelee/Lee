"""A small, versioned AI-origin head. Labels and filenames never enter inference."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageOps

FEATURE_SPEC = 'clip_l14_center224_and_letterbox224_l2_v1'
MULTIBLOCK_FEATURE_SPEC = 'wechat_multiblock_frozen_v2'


def block_descriptor(spec, clip_features, texture):
    """Apply the transforms fitted on train only, never refit on an input."""
    available = {'clip_center': clip_features[:768],
                 'clip_global': clip_features[768:], 'texture': texture}
    transformed = []
    for block in spec['feature_blocks']:
        value = np.array(available[block['name']], dtype=np.float32, copy=True)
        if value.shape != (block['dimensions'],) or not np.isfinite(value).all():
            raise ValueError('Invalid domain feature block')
        if block['normalization'] == 'standardized_l2':
            # Match sklearn StandardScaler's float32 in-place transform.
            value -= np.asarray(block['mean'], dtype=np.float64)
            value /= np.asarray(block['std'], dtype=np.float64)
        value /= max(float(np.linalg.norm(value)), 1e-8)
        value *= block['scale']
        transformed.append(value)
    return np.concatenate(transformed)


def global_view(image: Image.Image) -> Image.Image:
    return ImageOps.pad(image.convert('RGB'), (224, 224), method=Image.Resampling.BICUBIC,
                        color=(127, 127, 127), centering=(.5, .5))


def descriptor(center, global_feature):
    views = np.asarray([center, global_feature], np.float32)
    views /= np.maximum(np.linalg.norm(views, axis=1, keepdims=True), 1e-8)
    return np.concatenate(views)


def predict(spec, features, support_features=None):
    features = np.asarray(features, np.float64)
    if features.shape != (len(spec['coefficients']),) or not np.isfinite(features).all():
        raise ValueError('Invalid fitted domain descriptor')
    support_features = features if support_features is None else np.asarray(support_features, np.float64)
    if support_features.shape != (1536,) or not np.isfinite(support_features).all():
        raise ValueError('Invalid two-view CLIP support descriptor')
    raw = float(features @ np.asarray(spec['coefficients']) + spec['intercept'])
    logit = max(-50., min(50., raw * spec['platt_coefficient'] + spec['platt_intercept']))
    probability = 1. / (1. + math.exp(-logit))
    references = np.asarray(spec['support_vectors'], np.float32)
    similarity = float(np.max(references @ support_features / 2.))
    return probability, similarity >= spec['support_min_cosine'], similarity


def validate_spec(spec, encoder_sha256):
    if spec.get('trained') is not True or spec.get('feature_spec') not in {FEATURE_SPEC, MULTIBLOCK_FEATURE_SPEC}:
        raise ValueError('Domain head lacks completed training or has incompatible features')
    if spec.get('encoder_sha256') != encoder_sha256:
        raise ValueError('Domain head was fitted with a different CLIP checkpoint')
    splits = spec['sample_ids']
    seen = set()
    for name in ['train', 'calibration', 'threshold', 'holdout']:
        ids = set(splits[name])
        if len(ids) != len(splits[name]) or seen & ids:
            raise ValueError('Domain-head partitions overlap')
        seen.update(ids)
    if spec.get('holdout_used_for_fit') is not False:
        raise ValueError('Held-out samples were used for fitting')
    dimensions = 1536
    if spec['feature_spec'] == MULTIBLOCK_FEATURE_SPEC:
        blocks = spec.get('feature_blocks', [])
        if [b.get('name') for b in blocks] != ['clip_center', 'clip_global', 'texture']:
            raise ValueError('Unsupported domain feature blocks')
        for block, expected in zip(blocks, [768, 768, 53]):
            if block.get('dimensions') != expected or not np.isfinite(block['scale']) or block['scale'] <= 0:
                raise ValueError('Invalid domain block dimensions or scale')
            if block['name'] == 'texture':
                mean, std = np.asarray(block['mean']), np.asarray(block['std'])
                if (block.get('normalization') != 'standardized_l2' or mean.shape != (53,)
                        or std.shape != (53,) or not np.isfinite(mean).all()
                        or not np.isfinite(std).all() or np.any(std <= 0)):
                    raise ValueError('Invalid train-fitted texture transform')
            elif block.get('normalization') != 'l2':
                raise ValueError('Invalid frozen deep feature normalization')
        dimensions = 1589
    references = np.asarray(spec['support_vectors'], np.float32)
    if (len(spec['coefficients']) != dimensions or references.ndim != 2
            or references.shape[0] == 0 or references.shape[1] != 1536
            or not np.isfinite(references).all()):
        raise ValueError('Domain head has invalid fitted dimensions')
    numeric = [spec['intercept'], spec['platt_coefficient'], spec['platt_intercept'],
               spec['threshold'], spec['support_min_cosine'], *spec['coefficients']]
    if (not np.isfinite(numeric).all() or not 0 < spec['threshold'] <= 1
            or not 0 <= spec['support_min_cosine'] <= 1 or spec['platt_coefficient'] <= 0):
        raise ValueError('Domain head has invalid numeric parameters')
