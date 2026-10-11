import math
import pytest
from backend.contracts import Branch
from backend.evidence_fusion import fitted_probability,SCHEMA


def specification():
    return {'schema':SCHEMA,'trained':True,'test_used':False,'features':['video_origin.logit'],
      'scaler_mean':[0.],'scaler_scale':[1.],'coefficients':[0.],'intercept':0.,
      'estimator_type':'numeric_evidence_boosted_trees_v1','baseline_logit':0.,
      'trees':[[{'leaf':False,'feature':0,'threshold':0.,'value':0.,'left':1,'right':2},
        {'leaf':True,'value':-2.,'threshold':0.},{'leaf':True,'value':2.,'threshold':0.}]],
      'platt_coefficient':1.,'platt_intercept':0.}


def test_conditional_tree_uses_actual_branch_evidence_and_calibration():
    spec=specification()
    assert fitted_probability(spec,[Branch('video_origin','success',.2)])==pytest.approx(1/(1+math.exp(2)))
    assert fitted_probability(spec,[Branch('video_origin','success',.8)])==pytest.approx(1/(1+math.exp(-2)))


def test_invalid_tree_cannot_silently_mark_material_genuine():
    spec=specification();spec['trees'][0][0]['left']=0
    with pytest.raises(ValueError,match='Cyclic'):fitted_probability(spec,[Branch('video_origin','success',.2)])


def test_unknown_estimator_and_nonfinite_tree_are_rejected():
    spec=specification();spec['estimator_type']='unknown'
    with pytest.raises(ValueError,match='Unsupported'):fitted_probability(spec,[])
    spec=specification();spec['trees'][0][1]['value']=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):fitted_probability(spec,[Branch('video_origin','success',.2)])
