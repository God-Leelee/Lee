
import av
import numpy as np
from PIL import Image

from backend.contracts import Branch, load_config
from backend.copy_move import detect_copy_move
from backend.fusion import fuse
from backend.media import d3_statistics, sample_video, metadata_signals, read_image
from backend.reports import case_csv, case_html


def test_missing_scores_do_not_become_genuine():
    config = load_config()
    decision = fuse([Branch('generation','failed'), Branch('localization','insufficient_evidence')],config,'image')
    assert decision['risk'] is None and decision['verdict']=='undetermined'
    decision=fuse([Branch('generation','success',.1),Branch('metadata','success',.0)],config,'image')
    assert decision['risk'] is None  # metadata is not a visual detector


def test_legacy_large_local_edit_guard_is_not_diluted():
    config=load_config()
    # This verifies the retained legacy guard. New trained joint policies require
    # complete evidence and do not treat a raw localization score as proof.
    config.pop('image_fusion',None)
    config['image_origin_adapter']={'enabled':False}
    decision=fuse([Branch('generation','success',.03),Branch('localization','success',.96,details={'reliable_area_ratio':.08})],config,'image')
    assert decision['verdict']=='suspected_fake'
    assert 'strong_localization' in decision['rule_overrides']


def test_joint_localization_alone_does_not_override_missing_models():
    config=load_config()
    decision=fuse([Branch('generation','success',.03),Branch('localization','success',.96,
      details={'reliable_area_ratio':.08})],config,'image')
    assert not decision['execution_complete']
    assert decision['verdict']=='undetermined'
    assert decision['rule_overrides']==[]


def test_copy_pair_geometry_and_genuine_repetitions():
    rng=np.random.default_rng(41)
    raw=rng.integers(0,256,(480,640,3),np.uint8)
    edited=raw.copy();edited[80:240,360:520]=raw[40:200,40:200]
    positive=detect_copy_move(Image.fromarray(edited))
    assert positive.score>.85
    pair=positive.evidence[0]
    assert pair['direction']=='unknown'
    transform=np.asarray(positive.details['affine_matrix'])
    assert np.allclose(transform[:,2],[320,40],atol=2)
    for bbox in [pair['region_a'],pair['region_b']]:
        assert 0<=bbox[0]<640 and 0<=bbox[1]<480
    assert detect_copy_move(Image.fromarray(raw)).score<.5
    repeat=np.tile(np.array([[0,255],[255,0]],np.uint8),(240,320))
    repeated=detect_copy_move(Image.fromarray(repeat).convert('RGB'))
    assert repeated.score is None or repeated.score<.5


def test_original_orientation_is_not_silently_rotated(tmp_path):
    path=tmp_path/'orientation.jpg'
    image=Image.new('RGB',(100,200),(10,20,30));exif=Image.Exif();exif[274]=6;image.save(path,exif=exif)
    decoded,meta=read_image(path,50000)
    assert decoded.size==(100,200)
    assert meta['exif']['Orientation']==6
    assert meta['coordinate_space']=='original_raw_pixels'


def test_missing_exif_is_not_forgery_evidence():
    assert metadata_signals({'info':{},'exif':{}})==[]


def write_video(path, timestamps):
    from fractions import Fraction
    with av.open(str(path),'w') as c:
        stream=c.add_stream('libx264',rate=24);stream.width=160;stream.height=120;stream.pix_fmt='yuv420p'
        stream.time_base=Fraction(1,1000)
        stream.codec_context.time_base=Fraction(1,1000)
        for i,t in enumerate(timestamps):
            frame=av.VideoFrame.from_ndarray(np.full((120,160,3),i*3,np.uint8),format='rgb24')
            frame.pts=t;frame.time_base=Fraction(1,1000)
            for packet in stream.encode(frame):c.mux(packet)
        for packet in stream.encode():c.mux(packet)


def test_sampling_keeps_pts_and_window_masks(tmp_path):
    path=tmp_path/'vfr.mp4'
    write_video(path,list(range(0,1000,40))+list(range(1000,2000,80)))
    frames,windows,meta=sample_video(path,load_config())
    assert 1<=len(frames)<=32
    assert len({f.pts for f in frames})==len(frames)
    assert np.all(np.diff([f.timestamp for f in frames])>0)
    for window in windows:
        assert sum(window['valid_mask'])==len(set(window['frame_ids']))
        assert window['actual_timestamps']==[frames[i].timestamp for i in window['frame_ids']]
    assert meta['frames_decoded_including_preroll']<100  # no repeated GOP decoding


def test_static_feature_and_padding_are_not_invented_temporal_change():
    features=np.ones((8,768),np.float32)
    evidence=d3_statistics(features,[True]*8,[i*.125 for i in range(8)])
    assert evidence['second_order_std']==0
    duplicate=d3_statistics(features,[True,True,False,False,False,False,False,False],[0,.1,.1,.1,.1,.1,.1,.1])
    assert duplicate['status']=='insufficient_evidence'


def test_exports_escape_ocr_and_csv_formula():
    case={'title':'<script>alert(1)</script>','materials':[{'material_id':'1','name':'=HYPERLINK("bad")',
          'material_type':'image','status':'failed','result':{'explanation':[{'text':'<script>bad</script>'}]}}]}
    html=case_html(case)
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert "'=HYPERLINK" in case_csv(case)
