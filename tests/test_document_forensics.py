import numpy as np
import pytest
from PIL import Image
from backend.document_forensics import tile_plan,reconstruct,mask_grid,patch_average,validate,DIMENSION,FEATURE_SPEC


@pytest.mark.parametrize('size',[(19,37),(224,224),(751,333),(1400,2200)])
def test_dense_coverage_and_constant_reprojection(size):
    image=Image.new('RGB',size,'white')
    working,padded,positions,factor=tile_plan(image)
    out=reconstruct(np.full((len(positions),196),.37),positions,working.size,size)
    assert out.shape==(size[1],size[0])
    assert np.allclose(out,.37,atol=1e-6)
    assert max(working.size)<=768 and padded.width>=224 and padded.height>=224
    assert 0<factor<=1


def test_partial_padding_does_not_label_outside_document_as_forged():
    image=Image.new('RGB',(16,16),'white')
    working,_,positions,_=tile_plan(image)
    fraction=mask_grid(Image.new('L',image.size,255),positions,working.size)
    assert fraction[0,0]==1 and fraction[0,1:].sum()==0
    grid=np.arange(224*224).reshape(224,224)
    assert patch_average(grid)[0]==grid[:16,:16].mean()


def test_document_head_rejects_nonfinite_and_contaminated_spec():
    spec={'feature_spec':FEATURE_SPEC,'backbone_sha256':'abc','test_used':False,
        'scaler_mean':[0.]*DIMENSION,'scaler_scale':[1.]*DIMENSION,
        'coefficients':[0.]*DIMENSION,'intercept':0.}
    validate(spec,'abc')
    spec['scaler_scale'][0]=float('nan')
    with pytest.raises(ValueError):validate(spec,'abc')
    spec['scaler_scale'][0]=1;spec['test_used']=True
    with pytest.raises(ValueError):validate(spec,'abc')
