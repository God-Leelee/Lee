"""Protect deployable public-image evidence, provenance and decision boundaries."""
import json
import math

import numpy as np
import pytest

from backend.contracts import Branch
from backend.fusion import fuse
from backend.standalone_generation import (
    FEATURE_SPEC, TEXTURE_SPEC, descriptor, predict, transform_descriptor, validate_spec,
)


CLIP_SHA = 'a' * 64
WAVE_SHA = 'b' * 64


def checkpoint(with_texture=False):
    names = ['clip_center', 'clip_global', 'wave_center', 'wave_global']
    blocks = [
        {'name': name, 'dimensions': 768, 'normalization': 'l2', 'scale': scale}
        for name, scale in zip(names, [1.0, 0.75, 0.5, 0.25])
    ]
    if with_texture:
        blocks.append({
            'name': 'texture', 'dimensions': 53, 'normalization': 'standardized_l2',
            'texture_spec': TEXTURE_SPEC, 'scale': 0.2,
            'mean': list(np.linspace(-1.0, 1.0, 53)), 'std': [2.0] * 53,
        })
    sizes = {'fusion': 600, 'calibration': 100, 'threshold': 100}
    return {
        'trained': True, 'feature_spec': FEATURE_SPEC,
        'encoder_sha256': CLIP_SHA, 'wave_checkpoint_sha256': WAVE_SHA,
        'holdout_used_for_fit': False, 'holdout_used_for_candidate_selection': False,
        'feature_blocks': blocks, 'coefficients': [0.0] * (3125 if with_texture else 3072),
        'intercept': -0.3, 'platt_coefficient': 1.2, 'platt_intercept': 0.1,
        'threshold': 0.9,
        # Fusion has six source variants for each of 100 caption families;
        # calibration and threshold each reserve one variant per family.
        'sample_ids': {
            split: [f'{split}-caption-{i // 6 if split == "fusion" else i}-source-{i % 6}'
                    for i in range(size)]
            for split, size in sizes.items()
        },
        'source_groups': {split: [f'{split}-caption-{i}' for i in range(100)] for split in sizes},
        'development_manifest_sha256': 'c' * 64, 'feature_cache_sha256': 'd' * 64,
    }


def frozen_views():
    views = []
    for positions, values in [((0, 1), (3.0, 4.0)), ((2, 3), (5.0, 12.0)),
                              ((4, 5), (8.0, 15.0)), ((6, 7), (5.0, 12.0))]:
        vector = np.zeros(768, dtype=np.float64)
        vector[list(positions)] = values
        views.append(vector)
    return views


@pytest.mark.parametrize('with_texture', [False, True])
def test_frozen_valid_development_provenance_can_be_loaded(with_texture):
    validate_spec(checkpoint(with_texture), CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('field', ['encoder_sha256', 'wave_checkpoint_sha256'])
def test_changed_backbone_cannot_reuse_fitted_head(field):
    spec = checkpoint()
    spec[field] = 'e' * 64
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('field', ['trained', 'holdout_used_for_fit', 'holdout_used_for_candidate_selection'])
def test_untrained_or_test_selected_head_cannot_be_deployed(field):
    spec = checkpoint()
    spec[field] = False if field == 'trained' else True
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('destination', ['calibration', 'threshold'])
def test_shared_caption_family_cannot_cross_development_partitions(destination):
    spec = checkpoint()
    spec['source_groups'][destination][0] = spec['source_groups']['fusion'][0]
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('destination', ['calibration', 'threshold'])
def test_same_original_sample_cannot_cross_development_partitions(destination):
    spec = checkpoint()
    spec['sample_ids'][destination][0] = spec['sample_ids']['fusion'][0]
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('collection', ['sample_ids', 'source_groups'])
def test_duplicate_or_missing_training_provenance_is_not_a_valid_fit(collection):
    spec = checkpoint()
    spec[collection]['fusion'][1] = spec[collection]['fusion'][0]
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)
    spec = checkpoint()
    spec[collection]['calibration'].pop()
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('field', ['development_manifest_sha256', 'feature_cache_sha256'])
def test_input_receipt_is_required_to_reproduce_fitted_head(field):
    spec = checkpoint()
    spec[field] = 'not-a-frozen-sha256'
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('bad_std', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_training_texture_scaler_is_rejected(bad_std):
    spec = checkpoint(with_texture=True)
    spec['feature_blocks'][-1]['std'][0] = bad_std
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('bad_scale', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_block_scale_is_rejected_before_inference(bad_scale):
    spec = checkpoint()
    spec['feature_blocks'][0]['scale'] = bad_scale
    with pytest.raises(ValueError):
        validate_spec(spec, CLIP_SHA, WAVE_SHA)


@pytest.mark.parametrize('bad_view', [None, np.zeros(768), np.full(768, np.nan), np.ones(767)])
def test_missing_invalid_or_zero_backbone_output_cannot_form_evidence(bad_view):
    views = frozen_views()
    views[0] = bad_view
    with pytest.raises(ValueError):
        descriptor(*views)


@pytest.mark.parametrize('bad_feature', [None, np.ones(3071), np.full(3072, np.nan)])
def test_invalid_direct_prediction_feature_cannot_return_normal_probability(bad_feature):
    with pytest.raises(ValueError):
        predict(checkpoint(), bad_feature)


@pytest.mark.parametrize('block', range(4))
def test_direct_prediction_cannot_treat_zero_selected_backbone_view_as_valid(block):
    feature = descriptor(*frozen_views())
    feature[block * 768:(block + 1) * 768] = 0.0
    with pytest.raises(ValueError):
        predict(checkpoint(), feature)


def test_missing_selected_texture_is_not_silently_replaced_by_zero():
    with pytest.raises(ValueError):
        predict(checkpoint(with_texture=True), descriptor(*frozen_views()))


def test_zero_centered_texture_with_valid_visual_views_is_not_missing_evidence():
    spec = checkpoint(with_texture=True)
    feature = descriptor(*frozen_views(), texture=np.asarray(spec['feature_blocks'][-1]['mean']))
    transformed = transform_descriptor(spec, feature)
    np.testing.assert_array_equal(transformed[-53:], np.zeros(53))
    assert math.isfinite(predict(spec, feature))


def test_extreme_finite_backbone_values_cannot_normalize_into_zero_evidence():
    views = frozen_views()
    views[0] = np.full(768, 1e308)
    with pytest.raises(ValueError):
        descriptor(*views)


def test_serialized_transform_and_probability_preserve_measured_mathematics():
    spec = checkpoint(with_texture=True)
    spec['coefficients'][0] = 0.4
    spec['coefficients'][768 + 3] = -0.6
    spec['coefficients'][1536 + 4] = 0.8
    spec['coefficients'][2304 + 7] = 0.5
    spec['coefficients'][3072] = -0.2
    texture = np.arange(53, dtype=np.float64) / 10.0
    feature = descriptor(*frozen_views(), texture=texture)
    restored = json.loads(json.dumps(spec, allow_nan=False))
    validate_spec(restored, CLIP_SHA, WAVE_SHA)
    actual = transform_descriptor(restored, feature)
    expected = np.zeros(3125)
    expected[0:2] = [3.0 / 5.0, 4.0 / 5.0]
    expected[768 + 2:768 + 4] = np.array([5.0 / 13.0, 12.0 / 13.0]) * 0.75
    expected[1536 + 4:1536 + 6] = np.array([8.0 / 17.0, 15.0 / 17.0]) * 0.5
    expected[2304 + 6:2304 + 8] = np.array([5.0 / 13.0, 12.0 / 13.0]) * 0.25
    standardized_texture = (texture - np.linspace(-1.0, 1.0, 53)) / 2.0
    expected[-53:] = standardized_texture / np.linalg.norm(standardized_texture) * 0.2
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    raw = (0.4 * expected[0] - 0.6 * expected[771] + 0.8 * expected[1540]
           + 0.5 * expected[2311] - 0.2 * expected[3072] - 0.3)
    expected_probability = 1.0 / (1.0 + math.exp(-(raw * 1.2 + 0.1)))
    assert predict(restored, feature) == pytest.approx(expected_probability, abs=1e-12)
    assert predict(restored, feature) == pytest.approx(predict(spec, feature), abs=1e-12)


def test_non_ai_origin_probability_cannot_clear_independent_copy_forgery():
    spec = checkpoint()
    spec['intercept'] = -12.0
    origin_probability = predict(spec, descriptor(*frozen_views()))
    assert origin_probability < spec['threshold']
    config = {'weights': {'generation': 0.3, 'forensic_generation': 0.3,
                          'localization': 0.2, 'copy_move': 0.2},
              'calibration_status': 'not_fitted', 'threshold_status': 'unvalidated_research_default',
              'image_threshold': 0.7}
    result = fuse([
        Branch('generation', 'success', origin_probability),
        Branch('forensic_generation', 'success', 0.01),
        Branch('localization', 'success', 0.01),
        Branch('copy_move', 'success', 0.95, details={'inliers': 24}),
    ], config, 'image')
    assert result['verdict'] == 'suspected_fake'
    assert 'strong_geometric_copy' in result['rule_overrides']
    assert result['score_kind'] == 'uncalibrated_risk'
