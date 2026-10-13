from copy import deepcopy
import math

import pytest

from backend.contracts import Branch
from backend.fusion import fuse
from backend.image_policy import generation_values, numeric_probability


def linear(feature):
    return {'estimator_type': 'logistic_evidence_v1', 'features': [feature],
            'scaler_mean': [0.], 'scaler_scale': [1.], 'coefficients': [1.], 'intercept': 0.,
            'platt_coefficient': 1., 'platt_intercept': 0.}


def fixture():
    raw = {'commfor224.center_logit': 4., 'commfor224.global_logit': 3.,
           'commfor384.center_logit': -4., 'commfor384.global_logit': -3., 'bfree.logit': -5.}
    gen = linear('commfor224.max_logit')
    score = numeric_probability(gen, generation_values(raw, .01))
    policy = {'schema': 'separate_risk_heads_v1', 'trained': True, 'test_used': False,
              'required_features': ['standalone_generation', 'generation_experts', 'localization', 'copy_move'],
              'generation_head': gen, 'editing_head': linear('copy_move.logit'),
              'union_platt_coefficient': 10., 'union_platt_intercept': -5., 'threshold': .8, 'scope': 'test only'}
    branches = [Branch('standalone_generation', 'success', .01),
                Branch('generation_experts', 'success', score, details={'raw_features': raw}),
                Branch('localization', 'success', .001), Branch('copy_move', 'success', .001)]
    return {'image_fusion': policy}, branches


def set_generation(config, branches, logit):
    branches[1].details['raw_features']['commfor224.center_logit'] = logit
    branches[1].details['raw_features']['commfor224.global_logit'] = logit
    branches[1].score = numeric_probability(config['image_fusion']['generation_head'],
        generation_values(branches[1].details['raw_features'], branches[0].score))


def test_strong_generation_is_not_diluted_by_negative_edit_models():
    config, branches = fixture()
    result = fuse(branches, config, 'image')
    assert result['execution_complete'] and result['verdict'] == 'suspected_fake'
    assert result['head_probabilities']['generated_origin'] > .98
    assert result['head_probabilities']['local_edit'] < .01


def test_strong_copy_is_not_suppressed_by_low_generation_origin():
    config, branches = fixture()
    set_generation(config, branches, -8)
    branches[-1].score = .99
    result = fuse(branches, config, 'image')
    assert result['verdict'] == 'suspected_fake'
    assert result['head_probabilities']['generated_origin'] < .001
    assert result['head_probabilities']['local_edit'] > .98


@pytest.mark.parametrize('mode', ['missing', 'failed', 'nan', 'score_mismatch', 'bad_calibration', 'test_contaminated'])
def test_invalid_or_missing_required_evidence_never_becomes_low_risk(mode):
    config, branches = fixture()
    if mode == 'missing': branches.pop(1)
    if mode == 'failed': branches[1].status = 'failed'
    if mode == 'nan': branches[1].details['raw_features']['bfree.logit'] = float('nan')
    if mode == 'score_mismatch': branches[1].score = .1
    if mode == 'bad_calibration': config['image_fusion']['union_platt_coefficient'] = float('nan')
    if mode == 'test_contaminated': config['image_fusion']['test_used'] = True
    result = fuse(branches, config, 'image')
    assert result['verdict'] == 'undetermined' and result['risk'] is None and not result['execution_complete']


def test_union_preserves_monotonicity_and_reference_head_semantics():
    config, branches = fixture()
    set_generation(config, branches, -8)
    low = fuse(branches, config, 'image')
    branches[-1].score = .8
    higher = fuse(branches, config, 'image')
    assert low['risk'] < higher['risk']
    assert 'claims' not in higher['head_semantics']
    assert higher['fusion_method'] == 'separate_risk_heads_v1'


def test_numeric_tree_rejects_cycle_instead_of_hanging():
    spec = {**linear('x'), 'estimator_type': 'numeric_evidence_boosted_trees_v1', 'baseline_logit': 0.,
            'trees': [[{'leaf': False, 'feature': 0, 'threshold': 0., 'left': 0, 'right': 0, 'value': 0.}]]}
    with pytest.raises(ValueError, match='Cyclic'):
        numeric_probability(spec, {'x': 0.})
