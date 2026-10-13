from copy import deepcopy
import numpy as np
import pytest
from PIL import Image
import torch
from backend.dense_copy import dense_copy
from backend.representation_adapter import descriptor, validate, raw_logit, FEATURE_SPEC, NATIVE_FEATURE_SPEC, views, numeric_arrays


def test_dense_verifies_copy_with_non_grid_translation():
    array = np.random.default_rng(7).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    edited = array.copy()
    edited[111:183, 207:287] = array[23:95, 31:111]
    result = dense_copy(Image.fromarray(edited))
    assert result['verified_pair_count'] > 20
    assert result['translation_xy'] == [176., 88.]
    assert result['direction'] == 'unknown'
    assert len(result['regions']) == 2


def test_dense_does_not_treat_flat_or_independent_noise_as_copy():
    assert dense_copy(Image.new('RGB', (240, 240), 'white'))['verified_pair_count'] == 0
    noise = np.random.default_rng(4).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    assert dense_copy(Image.fromarray(noise))['verified_pair_count'] == 0


def test_descriptor_matches_training_order_and_rejects_wrong_tokens():
    tokens = torch.arange(2 * 197 * 384, dtype=torch.float32).reshape(2, 197, 384)
    value = descriptor(tokens)
    assert value.shape == (2304,)
    assert torch.equal(value[:384], tokens[0, 0])
    assert torch.equal(value[1152:1536], tokens[1, 0])
    with pytest.raises(ValueError): descriptor(tokens[:, :196])


def test_numeric_adapter_rejects_invalid_scale_and_heldout_feedback():
    spec = {'feature_spec': FEATURE_SPEC, 'backbone_sha256': 'abc', 'test_used': False,
            'scaler_mean': [0.] * 2304, 'scaler_scale': [1.] * 2304,
            'coefficients': [1.] * 2304, 'intercept': -2., 'platt_coefficient': 1., 'platt_intercept': 0.}
    validate(spec, 'abc')
    assert raw_logit(spec, np.ones(2304)) == 2302.
    broken = deepcopy(spec); broken['scaler_scale'][0] = 0
    with pytest.raises(ValueError): validate(broken, 'abc')
    broken = deepcopy(spec); broken['test_used'] = True
    with pytest.raises(ValueError): validate(broken, 'abc')
    with pytest.raises(ValueError): raw_logit(spec, np.full(2304, np.nan))


def test_native_descriptor_retains_regional_mean_and_max_without_resizing():
    tokens = torch.zeros(7, 197, 384)
    for i in range(7): tokens[i] = i
    value = descriptor(tokens, NATIVE_FEATURE_SPEC)
    assert value.shape == (4608,)
    assert value[2304] == 4  # mean of crop CLS values2..6
    assert value[3456] == 6  # max of crop CLS values2..6
    with pytest.raises(ValueError): descriptor(tokens[:2], NATIVE_FEATURE_SPEC)
    array = np.zeros((320, 480, 3), np.uint8)
    array[:224, :224] = [37, 61, 89]
    image = Image.fromarray(array)
    parts = views(image, NATIVE_FEATURE_SPEC, lambda im: torch.zeros(3, 224, 224))
    expected = (torch.tensor([37, 61, 89])/255 - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    assert len(parts) == 7
    assert torch.allclose(parts[2][:, 100, 100], expected, atol=1e-6)


def test_low_texture_copy_tier_still_rejects_flat_patches():
    rng = np.random.default_rng(19)
    array = rng.integers(116, 140, (240, 320, 3), dtype=np.uint8)
    array[111:183, 207:287] = array[23:95, 31:111]
    assert dense_copy(Image.fromarray(array))['verified_pair_count'] == 0
    assert dense_copy(Image.fromarray(array), min_variance=4.)['verified_pair_count'] > 20
    assert dense_copy(Image.new('RGB', (240, 320), 'white'), min_variance=4.)['verified_pair_count'] == 0


def test_safe_mlp_export_and_layer_validation():
    spec = {'feature_spec': NATIVE_FEATURE_SPEC, 'backbone_sha256': 'abc', 'test_used': False,
        'scaler_mean': [0.] * 4608, 'scaler_scale': [1.] * 4608, 'estimator_type': 'mlp_representation_v1',
        'hidden_weights': [[1.] * 4608, [-1.] * 4608], 'hidden_biases': [2., 3.],
        'output_coefficients': [2., 5.], 'intercept': -1., 'platt_coefficient': 1., 'platt_intercept': 0.}
    validate(spec, 'abc')
    compiled = numeric_arrays(spec)
    assert raw_logit(compiled, np.ones(4608)) == 9219.
    broken = deepcopy(spec); broken['hidden_weights'][0][0] = float('inf')
    with pytest.raises(ValueError): validate(broken, 'abc')
    broken = deepcopy(spec); broken['output_coefficients'].pop()
    with pytest.raises(ValueError): validate(broken, 'abc')
