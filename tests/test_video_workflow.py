"""Artifact/failure contracts with fake networks; no GPU or detector accuracy claims."""

import numpy as np
from PIL import Image

from backend.contracts import load_config
from backend.engine import Engine
from backend.media import VideoFrame


class FakeNetworks:
    def __init__(self, encode_error=False):
        self.encode_error=encode_error
        self.waverep=object()
        self.device='cpu'
        self.info={'generation':{'preprocess':'test'},'temporal':{'trained':True},
                   'video_origin':{'model':'synthetic workflow fixture','source_groups':{}},
                   'generation_experts':{'model':'synthetic workflow fixture'}}
        self.errors={}
        self.ocr=lambda raw:([],None)

    def encode(self,images):
        if self.encode_error:raise OSError('encoder failure')
        return np.ones((len(images),768),np.float32),[.1]*len(images),np.zeros((len(images),7))

    def wave_scores(self,images):return [-2.0]*len(images)
    def temporal_score(self,features,mask):return .1
    def video_origin_score(self,images,features):return .1,[-2.,-2.]
    def video_expert_score(self,image):return .1,{'adapted_generation.logit':-2.}
    def localize_batch(self,images,max_side=320,batch_size=4):return [self.localize(image,max_side) for image in images]

    def localize(self,image,max_side=None):
        size=(16,12) if max_side else image.size
        probability=np.zeros((size[1],size[0]),np.float32)
        probability[3:7,4:9]=.8
        return .3,probability,np.ones_like(probability),size


def fixture_video():
    frames=[VideoFrame(i,i*.125,i,'1/8',Image.new('RGB',(96,72),(i*3,20,30))) for i in range(8)]
    windows=[{'window_id':0,'frame_ids':list(range(8)),'valid_mask':[True]*8,
              'actual_timestamps':[f.timestamp for f in frames],'target_timestamps':[f.timestamp for f in frames],
              'start_seconds':0,'end_seconds':.875}]
    return frames,windows,{'audio_present':False,'width':96,'height':72,'tags':{},'tracks':[],
                           'global_frame_ids':list(range(0,8,2))}


def test_all_sampled_frames_have_persisted_masks_and_identical_evidence_ids(tmp_path,monkeypatch):
    monkeypatch.setattr('backend.engine.sample_video',lambda *args:fixture_video())
    source=tmp_path/'input.mp4';source.write_bytes(b'fake transport fixture, not detector evaluation')
    engine=Engine(load_config(),FakeNetworks())
    try:result=engine.analyze(source,tmp_path/'out','material1',material_type='video')
    finally:engine.cpu.shutdown(wait=True)
    assert result['execution_status']=='complete'
    loc=next(b for b in result['branches'] if b['branch_id']=='localization')
    ids={e['evidence_id'] for e in loc['evidence']}
    assert len(ids)==len(loc['evidence'])
    assert len(result['media']['frames'])==8
    assert len(loc['details']['selected_frame_ids'])==2
    for frame in result['media']['frames']:
        nested=frame['localization']
        assert nested['status']=='success'
        assert nested['details']['frame_id']==frame['frame_id']
        assert nested['evidence'][0]['evidence_id'] in ids
        assert nested['evidence'][0]['timestamp_seconds']==frame['timestamp_seconds']
        mask=Image.open(tmp_path/'out'/nested['details']['mask'])
        assert mask.size==(frame['width'],frame['height'])
    assert (tmp_path/'out/report.json').is_file()


def test_encoder_failure_preserves_other_evidence_and_cannot_produce_low_risk(tmp_path,monkeypatch):
    monkeypatch.setattr('backend.engine.sample_video',lambda *args:fixture_video())
    source=tmp_path/'input.mp4';source.write_bytes(b'fake transport fixture')
    engine=Engine(load_config(),FakeNetworks(encode_error=True))
    try:result=engine.analyze(source,tmp_path/'out','material2',material_type='video')
    finally:engine.cpu.shutdown(wait=True)
    branches={b['branch_id']:b for b in result['branches']}
    assert result['execution_status']=='partial'
    assert result['fusion']['verdict']!='lower_risk'
    assert branches['metadata']['status']=='success'
    assert branches['generation']['status']=='failed'
    assert branches['temporal']['score'] is None
    assert branches['localization']['status']=='success'
    assert len(result['media']['frames'])==8
    assert all('generation_risk' not in row for row in result['media']['frames'])
