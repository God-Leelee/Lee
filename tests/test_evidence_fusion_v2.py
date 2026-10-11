import numpy as np
from backend.contracts import Branch
from backend.evidence_fusion import evidence_features,localization_statistics
from backend.fusion import fuse


def example():
    branches=[Branch('generation','success',.1),Branch('forensic_generation','success',.1),
      Branch('standalone_generation','success',.01),
      Branch('localization','success',.996,details={'reliable_area_ratio':.01})]
    # A learned decision in a test fixture; this isn't an accuracy measurement.
    spec={'schema':'evidence_stacker_v2','trained':True,'test_used':False,
      'features':['standalone_generation.logit','localization.reliable_area_ratio'],
      'required_features':['generation','forensic_generation','localization','standalone_generation'],
      'scaler_mean':[0,0],'scaler_scale':[1,1],'coefficients':[1,10],'intercept':0,
      'platt_coefficient':1,'platt_intercept':0,'threshold':.8,'scope':'unit-test policy'}
    config={'weights':{},'calibration_status':'test','threshold_status':'test','image_threshold':.9,'image_fusion':spec,
      'image_origin_adapter':{'enabled':True,'decision_role':'ensemble_feature','threshold':.9,'scope':'source'}}
    return config,branches


def test_raw_high_local_score_cannot_bypass_joint_policy():
    config,branches=example();r=fuse(branches,config,'image')
    assert r['verdict']=='lower_risk' and r['execution_complete'] and not r['rule_overrides']
    assert r['fusion_method']=='jointly_trained_generation_and_local_evidence'
    branches[2].score=.99
    assert fuse(branches,config,'image')['verdict']=='suspected_fake'


def test_invalid_joint_calibration_remains_incomplete():
    config,branches=example();config['image_fusion']['platt_coefficient']=-1
    r=fuse(branches,config,'image')
    assert not r['execution_complete'] and r['verdict']=='undetermined'
    assert 'fusion' in r['failed_branches']
    branches[2].status='failed';branches[2].score=None
    assert not fuse(branches,config,'image')['execution_complete']


def test_spatial_summary_distinguishes_few_reliable_pixels_from_whole_image():
    p=np.zeros((64,64),np.float32);p[16:32,20:40]=.9;reliability=np.ones_like(p)
    stats=localization_statistics(p,reliability)
    assert stats['largest_region_area_ratio']==320/4096 and stats['region_count']==1
    stats2=localization_statistics(p,np.zeros_like(p))
    assert stats2['reliable_area_ratio']==0 and stats2['region_count']==0
    values=evidence_features([Branch('localization','success',.8,details=stats)])
    assert values['copy_move.missing']==1 and np.isfinite(list(values.values())).all()
