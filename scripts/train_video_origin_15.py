"""Source-grouped two-keyframe visual origin head, paired compression training.

Frozen CLIP/Wave views are shared with the image chain. Whole-generated/real
source labels train this head; edited clips never invent frame-level labels.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from backend.contracts import ROOT,load_config,atomic_json,sha256
from backend.media import sample_video
from backend.standalone_generation import FEATURE_SPEC
from scripts.evaluate_public_images_live import freeze
from scripts.train_joint_image_fusion_14 import metrics
import train_standalone_image_adapter_13 as extractor

WORK=ROOT/'data/video_origin_development_v1.5';SEED=10241507


def main():
    WORK.mkdir(exist_ok=True);frames_dir=WORK/'frames';frames_dir.mkdir(exist_ok=True)
    source=ROOT/'data/public_genvideo_v1.4';compression=ROOT/'data/compression_development_v1.5'
    temporal=ROOT/'data/temporal_compression_v1.5';joint=ROOT/'data/joint_video_development_v1.4-r02'
    originals=[{**r,'workspace_relative_path':(source/r['relative_path']).relative_to(ROOT).as_posix()} for r in json.loads((source/'manifest.json').read_text('utf-8')) if r['split']=='train']
    rows={r['sample_id']:r for r in originals+json.loads((temporal/'training_augmentations.json').read_text('utf-8'))+
      json.loads((joint/'manifest.json').read_text('utf-8'))+json.loads((compression/'manifest.json').read_text('utf-8'))}
    protocol={'seed':SEED,'source_videos':len(rows),'train_source_groups':200,'train_videos_with_paired_transforms':600,
      'frame_rule':'two decoded sampled exposures at floor(n/4) and floor(3*n/4), actual source PTS',
      'head_candidates':['clip_wave_equal','clip_wave_texture'],'C_candidates':[.3,3.,30.],
      'selection':'three-fold parent-source-group partial AUC at FPR.03; no edited source frame labels',
      'calibration':'mean raw two-frame logits then positive Platt on separate source calibration partition',
      'threshold':'separate threshold partition, source-real FPR<=.03; largest adjacent-score margin',
      'reserved_sources_accessed':False,'formal_business_truth':False}
    freeze(WORK/'protocol.json',protocol);config=load_config();frame_rows=[]
    manifest_path=WORK/'frame_manifest.json'
    if manifest_path.exists():frame_rows=json.loads(manifest_path.read_text('utf-8'))
    else:
        for index,row in enumerate(rows.values()):
            path=ROOT/row['workspace_relative_path']
            if sha256(path)!=row['sha256']:raise ValueError('Video source changed')
            frames,_,_=sample_video(path,config);chosen=[len(frames)//4,3*len(frames)//4]
            for slot,frame_id in enumerate(chosen):
                frame=frames[frame_id];name=row['sample_id']+'_view'+str(slot);destination=frames_dir/(name+'.png');frame.image.save(destination)
                group=row.get('source_group') or row.get('parent',{}).get('source_group') or '|'.join(sorted(row['source_groups']))
                edited=not (row.get('original') or row.get('split')=='train' or row.get('codec'))
                fit_split=row['split'] if not edited and row['split'] in ('train','calibration','threshold') else 'feature_only'
                frame_rows.append({'sample_id':name,'video_sample_id':row['sample_id'],'source_group':group,'split':fit_split,
                  'truth_label':row['truth_label'],'official_label_a':int(row['truth_label']=='fake'),
                  'relative_path':destination.relative_to(WORK).as_posix(),'sha256':sha256(destination),
                  'parent_video_sha256':row['sha256'],'frame_id':frame_id,'pts':frame.pts,'time_base':frame.time_base,
                  'timestamp_seconds':frame.timestamp,'slot':slot,'whole_source_frame_label_applicable':not edited})
            if index%40==0:print('Actual source keyframes',index+1,len(rows),flush=True)
        freeze(manifest_path,frame_rows)
    digest=sha256(manifest_path);feature_dir=WORK/'features';feature_dir.mkdir(exist_ok=True)
    extractor.DATA=WORK;extractor.WORK=feature_dir;extractor.extract(frame_rows,digest,'cuda')
    with np.load(feature_dir/'features.npz',allow_pickle=False) as cache:features=cache['features'].copy()
    groups=np.asarray([r['source_group'] for r in frame_rows]);y=np.asarray([r['official_label_a'] for r in frame_rows])
    splits=np.asarray([r['split'] for r in frame_rows]);tr=np.where(splits=='train')[0]
    if len(tr)!=1200 or len(set(groups[tr]))!=200:raise ValueError('Paired source training coverage differs')
    for a,b in [('train','calibration'),('train','threshold'),('calibration','threshold')]:
        if set(groups[splits==a])&set(groups[splits==b]):raise ValueError('Visual source fit/calibration groups overlap')
    trials=[];folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(features[tr],y[tr],groups[tr]))
    for name in protocol['head_candidates']:
        for c in protocol['C_candidates']:
            scores=[]
            for left,right in folds:
                a,b=tr[left],tr[right];blocks=extractor.make_blocks('clip_wave_texture_weak' if name=='clip_wave_texture' else name,features[a]);x=extractor.transform_matrix(blocks,features[np.r_[a,b]])
                head=LogisticRegression(C=c,class_weight='balanced',max_iter=2500,random_state=SEED).fit(x[:len(a)],y[a])
                p=head.predict_proba(x[len(a):])[:,1];scores.append(roc_auc_score(y[b],p,max_fpr=.03))
            trials.append({'candidate':name,'C':c,'mean_group_cv_partial_auc':float(np.mean(scores))});print(trials[-1],flush=True)
    choice=max(trials,key=lambda r:(r['mean_group_cv_partial_auc'],-r['C']))
    blocks=extractor.make_blocks('clip_wave_texture_weak' if choice['candidate']=='clip_wave_texture' else choice['candidate'],features[tr]);x=extractor.transform_matrix(blocks,features)
    head=LogisticRegression(C=choice['C'],class_weight='balanced',max_iter=2500,random_state=SEED).fit(x[tr],y[tr]);raw=head.decision_function(x)
    video_rows=[];raw_by_video={}
    for i,row in enumerate(frame_rows):raw_by_video.setdefault(row['video_sample_id'],[]).append(float(raw[i]))
    for key,values in raw_by_video.items():
        row=rows[key];video_rows.append({**row,'origin_raw_logit':float(np.mean(values)),
          'head_split':next(r['split'] for r in frame_rows if r['video_sample_id']==key)})
    cal=[r for r in video_rows if r['head_split']=='calibration'];th=[r for r in video_rows if r['head_split']=='threshold']
    platt=LogisticRegression(C=10,max_iter=2000,random_state=SEED).fit(np.asarray([r['origin_raw_logit'] for r in cal])[:,None],[r['truth_label']=='fake' for r in cal])
    if platt.coef_[0,0]<=0:raise ValueError('Video source calibration reverses meaning')
    def probability(part):return expit(np.asarray([r['origin_raw_logit'] for r in part])*platt.coef_[0,0]+platt.intercept_[0])
    p=probability(th);boundaries=np.unique(np.r_[0.,p,1.]);trials_t=[]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        threshold=float((lo+hi)/2);m=metrics(th,p,threshold)
        if m['genuine_fpr']<=.03:trials_t.append((m['prior70_30_accuracy'],hi-lo,threshold,m))
    _,_,threshold,m=max(trials_t,key=lambda r:(r[0],r[1]))
    spec={'architecture':'frozen_clip_wave_two_keyframe_video_origin_v1','feature_spec':FEATURE_SPEC,'trained':True,
      'feature_blocks':blocks,'encoder_sha256':extractor.CLIP_SHA,'wave_checkpoint_sha256':extractor.WAVE_SHA,
      'coefficients':head.coef_[0].tolist(),'intercept':float(head.intercept_[0]),'platt_coefficient':float(platt.coef_[0,0]),
      'platt_intercept':float(platt.intercept_[0]),'threshold':threshold,'aggregation':'mean_raw_two_keyframe_logits',
      'frame_selection':'sampled_frame_quarter_positions','frame_manifest_sha256':digest,'feature_cache_sha256':sha256(feature_dir/'features.npz'),
      'source_groups':{s:sorted(set(groups[splits==s])) for s in ('train','calibration','threshold')},
      'sample_ids':{s:[r['sample_id'] for r in frame_rows if r['split']==s] for s in ('train','calibration','threshold')},
      'holdout_used_for_fit':False,'holdout_used_for_candidate_selection':False,'calibration_metrics':metrics(cal,probability(cal),threshold),
      'threshold_metrics':m,'scope':'Public source-video origin reference domain; not insurance-fraud probability',
      'version':'1.5.0-candidate','formal_acceptance':False}
    freeze(ROOT/'models/candidates/video_origin_v1.5.json',spec)
    predictions={r['sample_id']:{'probability':float(p),'raw_frame_logits':raw_by_video[r['sample_id']]} for r,p in zip(video_rows,probability(video_rows))}
    freeze(WORK/'development_predictions.json',predictions)
    atomic_json(ROOT/'runs/video_origin_training_v1.5.json',{'chosen':choice,'trials':trials,
      'calibration_metrics':spec['calibration_metrics'],'threshold_metrics':m,'threshold':threshold,
      'whole_head_development_passed':m['prior70_30_accuracy']>=.95 and m['genuine_fpr']<=.03,
      'candidate_sha256':sha256(ROOT/'models/candidates/video_origin_v1.5.json'),'reserved_sources_used':False})
    print(json.dumps({'calibration':spec['calibration_metrics'],'threshold':m}),flush=True)


if __name__=='__main__':main()
