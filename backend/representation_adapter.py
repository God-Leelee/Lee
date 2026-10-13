"""Versioned frozen CF representation and safe numeric adaptation head."""
import numpy as np
import torch

FEATURE_SPEC = 'cf224_center_global_cls_patchmean_patchstd_float32_v1'
EXPERT_SPEC = 'cf224_representation_adapter_float32_v1'
NATIVE_FEATURE_SPEC = 'cf224_global2_native5_cls_patchstats_meanmax_float32_batch7_v2'
NATIVE_EXPERT_SPEC = 'cf224_native_representation_adapter_float32_v2'
FEATURES = ['commfor224.center_logit', 'commfor224.global_logit', 'adapted_generation.logit']


def descriptor(tokens, feature_spec=FEATURE_SPEC):
    count = 7 if feature_spec == NATIVE_FEATURE_SPEC else 2
    if feature_spec not in (FEATURE_SPEC, NATIVE_FEATURE_SPEC) or tokens.shape != (count, 197, 384):
        raise ValueError('Adaptation token layout differs from training')
    patch = tokens[:, 1:].float()
    described = torch.cat([tokens[:, 0].float(), patch.mean(1), patch.std(1, unbiased=False)], 1)
    if count == 2:
        return described.flatten()
    return torch.cat([described[:2].flatten(), described[2:].mean(0), described[2:].max(0).values])


def views(image, feature_spec, global_transform):
    """Shared extraction/serving transforms, including unresized native crops."""
    from PIL import Image, ImageOps
    from torchvision import transforms
    global_image = ImageOps.pad(image, (512, 512), method=Image.Resampling.BICUBIC, color=(127, 127, 127))
    result = [global_transform(image), global_transform(global_image)]
    if feature_spec == FEATURE_SPEC:
        return result
    if feature_spec != NATIVE_FEATURE_SPEC:
        raise ValueError('Unknown adaptation preprocessing')
    px, py = max(0, 224 - image.width), max(0, 224 - image.height)
    padded = ImageOps.expand(image, (px // 2, py // 2, px - px // 2, py - py // 2), fill=(127, 127, 127))
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    return result + [normalize(v) for v in transforms.FiveCrop(224)(padded)]


def validate(spec, backbone_sha):
    if spec.get('feature_spec') not in (FEATURE_SPEC, NATIVE_FEATURE_SPEC) or spec.get('backbone_sha256') != backbone_sha or spec.get('test_used') is not False:
        raise ValueError('Adaptation source, layout or fitting boundary differs')
    dimension = 4608 if spec['feature_spec'] == NATIVE_FEATURE_SPEC else 2304
    arrays = [np.asarray(spec[k], float) for k in ('scaler_mean', 'scaler_scale')]
    if any(a.shape != (dimension,) or not np.isfinite(a).all() for a in arrays) or np.any(arrays[1] <= 0):
        raise ValueError('Invalid adaptation dimensions/scales')
    if spec.get('estimator_type', 'linear') == 'mlp_representation_v1':
        weights = np.asarray(spec['hidden_weights'], float)
        width = len(spec['hidden_biases'])
        if not 1 <= width <= 256 or weights.shape != (width, dimension) or not np.isfinite(weights).all():
            raise ValueError('Invalid adaptation hidden layer')
        for name in ('hidden_biases', 'output_coefficients'):
            value = np.asarray(spec[name], float)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError('Invalid adaptation output layer')
    elif spec.get('estimator_type', 'linear') == 'linear':
        coefficient = np.asarray(spec['coefficients'], float)
        if coefficient.shape != (dimension,) or not np.isfinite(coefficient).all():
            raise ValueError('Invalid adaptation coefficients')
    else:
        raise ValueError('Unknown representation estimator')
    if not np.isfinite(spec['intercept']) or not np.isfinite(spec['platt_intercept']) or not np.isfinite(spec['platt_coefficient']) or spec['platt_coefficient'] <= 0:
        raise ValueError('Invalid adaptation calibration')


def raw_logit(spec, feature):
    value = np.asarray(feature, float)
    dimension = 4608 if spec.get('feature_spec') == NATIVE_FEATURE_SPEC else 2304
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise ValueError('Invalid adaptation features')
    standardized = (value - spec['scaler_mean']) / spec['scaler_scale']
    if spec.get('estimator_type', 'linear') == 'mlp_representation_v1':
        hidden = np.maximum(0., np.asarray(spec['hidden_weights']) @ standardized + spec['hidden_biases'])
        result = float(hidden @ spec['output_coefficients'] + spec['intercept'])
    else:
        result = float(standardized @ spec['coefficients'] + spec['intercept'])
    if not np.isfinite(result):
        raise ValueError('Non-finite adaptation output')
    return result


def numeric_arrays(spec):
    """Compile safe numeric arrays once after validation; never deserialize code."""
    result = dict(spec)
    for name in ('scaler_mean', 'scaler_scale', 'coefficients', 'hidden_weights', 'hidden_biases', 'output_coefficients'):
        if name in result:
            result[name] = np.asarray(result[name], dtype=np.float64)
    return result
