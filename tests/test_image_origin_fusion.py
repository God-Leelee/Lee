"""A source classifier cannot clear a local edit or hide a required failure."""
import pytest
from backend.contracts import Branch
from backend.fusion import fuse


def setup():
    config = {'weights': {'generation': .5, 'localization': .5}, 'calibration_status': 'not_fitted',
              'threshold_status': 'unvalidated_research_default', 'image_threshold': .9,
              'image_origin_adapter': {'enabled': True, 'threshold': .93, 'scope': 'public image origin'},
              'image_strong_localization_min_score': .85}
    branches = [Branch('generation', 'success', .99), Branch('forensic_generation', 'success', .99),
                Branch('localization', 'success', .1, details={'reliable_area_ratio': .1}),
                Branch('standalone_generation', 'success', .01)]
    return config, branches


def test_low_source_probability_keeps_strong_geometric_copy_visible():
    config, branches = setup()
    branches.append(Branch('copy_move', 'success', .95, details={'inliers': 24}))
    result = fuse(branches, config, 'image')
    assert result['verdict'] == 'suspected_fake' and result['execution_complete']
    assert result['public_reference_probability'] == .01
    assert 'strong_geometric_copy' in result['rule_overrides']


def test_low_source_probability_keeps_reliable_splice_visible():
    config, branches = setup()
    branches[2].score = .97
    result = fuse(branches, config, 'image')
    assert result['verdict'] == 'suspected_fake' and 'strong_localization' in result['rule_overrides']


@pytest.mark.parametrize('required', ['generation', 'forensic_generation', 'localization', 'standalone_generation'])
def test_required_model_failure_cannot_be_a_correct_negative(required):
    config, branches = setup()
    for branch in branches:
        if branch.branch_id == required:
            branch.status = 'failed'; branch.score = None
    result = fuse(branches, config, 'image')
    assert not result['execution_complete'] and result['verdict'] == 'undetermined'
    assert result['public_reference_probability'] is None


def test_reference_probability_scope_does_not_inherit_old_video_calibration():
    config, branches = setup()
    config['image_fusion'] = {'features': ['generation', 'forensic_generation', 'localization'],
                              'threshold': .01, 'scope': 'old video frames'}
    result = fuse(branches, config, 'image')
    assert result['threshold'] == .93 and result['probability_scope'] == 'public image origin'
    assert result['verdict'] == 'lower_risk'


@pytest.mark.parametrize('probability',[.00001,.99999])
def test_rejected_origin_candidate_in_review_mode_never_changes_main_risk(probability):
    config, branches=setup()
    config['image_origin_adapter']['decision_role']='review_only'
    branches=branches[:3]
    original=fuse(branches,config,'image')
    new=fuse(branches+[Branch('standalone_generation','success',probability)],config,'image')
    assert original==new and 'standalone_generation' not in new['valid_visual_branches']
