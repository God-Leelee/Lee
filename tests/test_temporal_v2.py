import numpy as np
import pytest
import torch
from backend.temporal_v2 import TemporalNetV2, bag_logit, validate_window


def test_pts_validation_rejects_repeated_valid_frame_and_nan():
    features=np.ones((8,768));mask=np.ones(8,bool);times=np.arange(8)*.125;quality=np.zeros((8,2))
    validate_window(features,mask,times,quality)
    times[3]=times[2]
    with pytest.raises(ValueError):validate_window(features,mask,times,quality)
    mask[3]=False
    validate_window(features,mask,times,quality)
    quality[0,0]=np.nan
    with pytest.raises(ValueError):validate_window(features,mask,times,quality)


def test_time_aware_network_uses_actual_time_and_mask_isolation():
    torch.manual_seed(8);net=TemporalNetV2().eval()
    features=torch.randn(1,8,768);mask=torch.tensor([[True,True,True,True,True,True,False,False]])
    times=torch.arange(8)[None].float()*.125;quality=torch.zeros(1,8,2)
    with torch.inference_mode():
        baseline=net(features,mask,times,quality)
        slower=net(features,mask,times*2,quality)
        changed=features.clone();changed[:,6:]=1000
        masked=net(changed,mask,times,quality)
    assert not torch.allclose(baseline,slower)
    assert torch.allclose(baseline,masked,atol=1e-6)
    assert torch.allclose(bag_logit(torch.full((2,2),.7)),torch.full((2,),.7))
