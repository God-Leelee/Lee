"""Localization provenance must not turn malformed timing into valid masks."""
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from scripts.evaluate_video_localization import score_localization


@pytest.fixture
def localization_case(tmp_path):
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 255
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    Image.fromarray(mask).save(tmp_path / 'truth.png')
    ground_truth = [
        {'output_frame_index': 0, 'timestamp_seconds': 1.0,
         'gt_active': True, 'mask_path': 'truth.png'},
    ]
    result = {
        'execution_status': 'complete',
        'branches': [{'branch_id': 'localization',
                      'details': {'localized_frame_ids': [0]}}],
        'media': {'frames': [
            {'frame_id': 0, 'timestamp_seconds': 1.0, 'pts': 1, 'time_base': '1',
             'localization': {'status': 'success',
                              'details': {'frame_id': 0, 'timestamp_seconds': 1.0,
                                          'mask': 'mask.png'}}},
        ]},
    }
    return result, ground_truth, tmp_path


@pytest.mark.parametrize('timestamp', [
    float('nan'), float('inf'), float('-inf'), True, False, '1.0', None, {}, 2.0,
])
def test_invalid_nested_timestamp_cannot_count_as_valid_localization(localization_case, timestamp):
    result, ground_truth, directory = localization_case
    result = deepcopy(result)
    result['media']['frames'][0]['localization']['details']['timestamp_seconds'] = timestamp
    score = score_localization(result, directory, directory, {'width': 2, 'height': 2}, ground_truth)
    assert not score['localization_evaluation_valid']
    assert score['effective_positive_iou'] == 0.0
    assert score['effective_positive_f1'] == 0.0
    assert len(score['mask_errors']) == 1


@pytest.mark.parametrize('timestamp', [1, 1.0, 1.0 + 5e-7])
def test_actual_nested_pts_within_tolerance_preserves_valid_mask(localization_case, timestamp):
    result, ground_truth, directory = localization_case
    result['media']['frames'][0]['localization']['details']['timestamp_seconds'] = timestamp
    score = score_localization(result, directory, directory, {'width': 2, 'height': 2}, ground_truth)
    assert score['localization_evaluation_valid']
    assert score['effective_positive_iou'] == 1.0
    assert score['effective_positive_f1'] == 1.0
    assert not score['mask_errors']
    assert not score['predicted_continuous_intervals']
