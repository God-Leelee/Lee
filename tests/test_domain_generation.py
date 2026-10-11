"""Protect checkpoint provenance and the experimental head's decision boundary."""
import numpy as np
import pytest

from backend.contracts import Branch
from backend.domain_generation import FEATURE_SPEC, MULTIBLOCK_FEATURE_SPEC, block_descriptor, descriptor, predict, validate_spec
from backend.fusion import fuse


def checkpoint():
    return {'trained':True,'feature_spec':FEATURE_SPEC,'encoder_sha256':'encoder-a',
            'coefficients':[0.]*1536,'intercept':12.,'platt_coefficient':1.,'platt_intercept':0.,
            'threshold':.8,'support_min_cosine':.8,'support_vectors':[[1.,*([0.]*767),1.,*([0.]*767)]],
            'sample_ids':{s:[s+'-sample'] for s in ['train','calibration','threshold','holdout']},
            'holdout_used_for_fit':False}


@pytest.mark.parametrize('change',[
    {'trained':False}, {'encoder_sha256':'encoder-b'}, {'holdout_used_for_fit':True},
    {'coefficients':[float('nan')]+[0.]*1535},
    {'sample_ids':{'train':['same'],'calibration':['same'],'threshold':['t'],'holdout':['h']}},
])
def test_untrained_changed_leaked_or_invalid_checkpoint_is_rejected(change):
    spec=checkpoint();spec.update(change)
    with pytest.raises(ValueError):validate_spec(spec,'encoder-a')


def test_unsupported_input_abstains_even_with_high_classifier_probability():
    center=np.zeros(768);center[7]=4
    features=descriptor(center,center)
    probability,supported,similarity=predict(checkpoint(),features)
    assert probability>.99 and supported is False and similarity==0


def test_review_head_cannot_change_final_verdict_or_clear_geometric_tampering():
    cfg={'weights':{'generation':.6,'localization':.4},'calibration_status':'not_fitted',
         'threshold_status':'unvalidated_research_default','image_threshold':.7}
    base=[Branch('generation','success',.1),Branch('localization','success',.1)]
    original=fuse(base,cfg,'image')
    assert fuse(base+[Branch('domain_generation','success',.999)],cfg,'image')==original
    copied=Branch('copy_move','success',.95,details={'inliers':24})
    output=fuse(base+[copied,Branch('domain_generation','success',.001)],cfg,'image')
    assert output['verdict']=='suspected_fake' and 'strong_geometric_copy' in output['rule_overrides']


def multiblock_checkpoint():
    spec=checkpoint()
    spec.update({'feature_spec':MULTIBLOCK_FEATURE_SPEC,'coefficients':[0.]*1589,
                 'feature_blocks':[{'name':name,'dimensions':768,'normalization':'l2','scale':1.}
                                    for name in ['clip_center','clip_global']] +
                 [{'name':'texture','dimensions':53,'normalization':'standardized_l2','scale':.25,
                   'mean':[2.]*53,'std':[3.]*53}]})
    return spec


@pytest.mark.parametrize('bad_std',[0.,-1.,float('nan'),float('inf')])
def test_invalid_training_scaler_cannot_be_deployed(bad_std):
    spec=multiblock_checkpoint();spec['feature_blocks'][2]['std'][0]=bad_std
    with pytest.raises(ValueError):validate_spec(spec,'encoder-a')


def test_texture_never_changes_the_clip_support_gate():
    spec=multiblock_checkpoint();validate_spec(spec,'encoder-a')
    out_of_domain=np.zeros(768);out_of_domain[7]=1
    support=descriptor(out_of_domain,out_of_domain)
    features=block_descriptor(spec,support,np.arange(53,dtype=np.float32))
    probability,supported,similarity=predict(spec,features,support)
    assert probability>.99 and supported is False and similarity==0
    assert np.linalg.norm(features[-53:])==pytest.approx(.25)
    with pytest.raises(ValueError):predict(spec,features)
