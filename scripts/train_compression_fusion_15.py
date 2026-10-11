"""Development-only compression robustness iteration; no1.4 holdout is fitted.

Keep original temporal/visual networks. Fit video ensemble with paired source
clips transcoded to MPEG-4 AVI and strongly compressed H264, both classes.
"""
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import av
import numpy as np
import requests
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from backend.contracts import ROOT,Branch,atomic_json,sha256
from backend.evidence_fusion import evidence_features
from scripts.evaluate_public_images_live import freeze
from train_joint_image_fusion_14 import metrics
from train_joint_video_fusion_14 import FEATURES

WORK=ROOT/'data/compression_development_v1.5';SOURCE=ROOT/'data/public_genvideo_v1.4';BASE=ROOT/'data/joint_video_development_v1.4-r02';SEED=10241501


def prepare():
    WORK.mkdir(exist_ok=True);rows=json.loads((SOURCE/'manifest.json').read_text('utf-8'));plan=[];rng=random.Random(SEED)
    for split in ('fusion','calibration','threshold'):
        for truth in ('genuine','fake'):
            available=sorted([r for r in rows if r['split']==split and r['truth_label']==truth],key=lambda r:r['sample_id']);rng.shuffle(available)
            for parent in available[:8]:
                for codec in ('mpeg4','libx264'):
                    name='compress15_'+hashlib.sha256((parent['sample_id']+codec).encode()).hexdigest()[:24]
                    plan.append({'sample_id':name,'parent':parent,'split':split,'truth_label':truth,'codec':codec,
                      'relative_path':name+('.avi' if codec=='mpeg4' else '.mp4'),'material_type':'video',
                      'material_class':'transcoded_'+parent['material_class'],'source_groups':[parent['source_group']]})
    freeze(WORK/'selection_registration.json',{'seed':SEED,'rows':plan,'registered_before_transcoded_pixels_or_predictions':True,
      'parent_rule':'Eight metadata-shuffled original full-source clips per label and development split;1.4 reserved sources never accessed',
      'transforms':'2second source prefix, nearest source exposure sampling to24fps, aspect-preserving max320, no fabricated frames',
      'source_labels_inherited_for_nonsemantic_transcoding':True,'holdout_used':False,'formal_acceptance':False})
    finished=[]
    for index,row in enumerate(plan):
        destination=WORK/row['relative_path'];parent=row['parent'];source=SOURCE/parent['relative_path']
        if sha256(source)!=parent['sha256']:raise ValueError('Original development parent changed')
        if not destination.exists():
            pixels=[];times=[]
            with av.open(str(source)) as container:
                stream=container.streams.video[0];first=None
                for frame in container.decode(stream):
                    if frame.pts is None:raise ValueError('No source PTS')
                    t=float(frame.pts*frame.time_base);first=t if first is None else first
                    image=frame.to_image().convert('RGB');scale=320/max(image.size);size=tuple(max(64,(round(v*scale)//2)*2) for v in image.size)
                    pixels.append(image.resize(size));times.append(t-first)
                    if t-first>=2.:break
                period=1/float(stream.average_rate) if stream.average_rate else float(np.median(np.diff(times)))
            duration=min(2.,times[-1]+period);count=max(8,int(duration*24))
            with av.open(str(destination),'w') as out:
                track=out.add_stream(row['codec'],rate=24);track.width,track.height=pixels[0].size;track.pix_fmt='yuv420p';track.codec_context.thread_count=2
                if row['codec']=='libx264':track.options={'preset':'ultrafast','crf':'36','bf':'0'}
                else:track.bit_rate=200000
                for i in range(count):
                    source_index=max(0,min(len(pixels)-1,bisect_right(times,i/24)-1));frame=av.VideoFrame.from_image(pixels[source_index]);frame.pts=i;frame.time_base=Fraction(1,24)
                    for packet in track.encode(frame):out.mux(packet)
                for packet in track.encode():out.mux(packet)
            atomic_json(WORK/(row['sample_id']+'_parent_pts.json'),{'parent_sha256':parent['sha256'],'source_relative_timestamps':times,
              'source_period':period,'output_fps':24,'output_frames':count,'no_looped_frames':True})
        finished.append({**row,'workspace_relative_path':destination.relative_to(ROOT).as_posix(),'sha256':sha256(destination)})
        if index%16==0:print('Compression originals',index+1,len(plan),flush=True)
    freeze(WORK/'manifest.json',finished);return finished


def main():
    rows=prepare();session=requests.Session();base='http://127.0.0.1:8765'
    health=session.get(base+'/api/health',timeout=10).json()
    if health['status']!='ready' or health['configuration']['configuration_id']!='v1.4-joint-evidence-candidate-002':raise ValueError('Expected frozen1.4 inference for development features')
    original=json.loads((BASE/'manifest.json').read_text('utf-8'));examples=[json.loads((BASE/(r['sample_id']+'.json')).read_text('utf-8')) for r in original]
    for index,row in enumerate(rows):
        cache=WORK/(row['sample_id']+'.json')
        if cache.exists():value=json.loads(cache.read_text('utf-8'))
        else:
            media=ROOT/row['workspace_relative_path']
            with media.open('rb') as stream:response=session.post(base+'/api/cases',files={'files':('开发材料'+media.suffix,stream)},data={'title':'压缩鲁棒开发'},timeout=30)
            response.raise_for_status();case_id=response.json()['case_id'];deadline=time.perf_counter()+120
            while True:
                case=session.get(base+'/api/cases/'+case_id,timeout=15).json();material=case['materials'][0]
                if case['status'] in ('complete','partial','failed'):break
                if time.perf_counter()>deadline:raise TimeoutError('Development inference timeout')
                time.sleep(.025)
            result=material['result']
            if result['execution_status']!='complete' or result['sha256']!=row['sha256']:raise ValueError('Compressed development inference failed')
            branches=[Branch(**b) for b in result['branches']]
            value={**row,'features':evidence_features(branches),'branch_scores':{b.branch_id:b.score for b in branches},
              'case_id':case_id,'material_id':material['material_id'],'test_used':False}
            atomic_json(cache,value)
        if value['sha256']!=row['sha256']:raise ValueError('Compression cache differs')
        examples.append(value)
        if index%8==0:print('Actual compression evidence',index+1,len(rows),flush=True)
    tr,cal,th=([e for e in examples if e['split']==s] for s in ('fusion','calibration','threshold'))
    def matrix(part):return np.asarray([[e['features'][name] for name in FEATURES] for e in part])
    scaler=StandardScaler().fit(matrix(tr));head=LogisticRegression(C=.3,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(matrix(tr)),[e['truth_label']=='fake' for e in tr])
    raw=head.decision_function(scaler.transform(matrix(cal)));platt=LogisticRegression(C=10,max_iter=2000,random_state=SEED).fit(raw[:,None],[e['truth_label']=='fake' for e in cal])
    if platt.coef_[0,0]<=0:raise ValueError('Compression calibration reverses risk')
    def probability(part):return expit(head.decision_function(scaler.transform(matrix(part)))*platt.coef_[0,0]+platt.intercept_[0])
    scores=probability(th);boundaries=np.unique(np.r_[0.,scores,1.]);candidates=[]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        threshold=float((lo+hi)/2);m=metrics(th,scores,threshold)
        if m['genuine_fpr']<=.03:candidates.append((m['prior70_30_accuracy'],hi-lo,threshold,m))
    _,_,threshold,m=max(candidates,key=lambda r:(r[0],r[1]))
    fitted={**json.loads((ROOT/'models/candidates/video_fusion_v1.4.json').read_text('utf-8')),
      'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coefficients':head.coef_[0].tolist(),'intercept':float(head.intercept_[0]),
      'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),'threshold':threshold,
      'sample_ids':{s:[e['sample_id'] for e in examples if e['split']==s] for s in ('fusion','calibration','threshold')},
      'compression_manifest_sha256':sha256(WORK/'manifest.json'),'calibration_metrics':metrics(cal,probability(cal),threshold),
      'threshold_metrics':m,'test_used':False,'version':'1.5.0-candidate','origin_networks_changed':False}
    freeze(ROOT/'models/candidates/video_fusion_v1.5.json',fitted)
    atomic_json(ROOT/'runs/compression_video_fusion_v1.5.json',{'specification':fitted,'whole_policy_development_passed':m['prior70_30_accuracy']>=.95 and m['genuine_fpr']<=.03,
      'new_transcodes':len(rows),'training_or_selection_from1_4_holdout':False,'live_config_modified':False,'formal_acceptance':False,
      'development_rows':[{'sample_id':e['sample_id'],'truth_label':e['truth_label'],'split':e['split'],'probability':float(p)} for e,p in zip(examples,probability(examples))]})
    print(json.dumps({'calibration':fitted['calibration_metrics'],'threshold':m,'threshold_value':threshold}),flush=True)


if __name__=='__main__':main()
