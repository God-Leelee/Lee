"""Video generation/temporal/local/copy fusion on new development sources.

Actual original PTS sampling, batched320 spatial screening, two detailed frames
and matched known-operation controls. Reserved originals are never decoded.
"""
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from backend.contracts import ROOT, atomic_json, sha256
from backend.models import Models
from backend.engine import Engine
from backend.evidence_fusion import SCHEMA,evidence_features
from scripts.make_video_edit_fixtures import decode_parent,create_fixture
from scripts.evaluate_public_images_live import freeze
from train_joint_image_fusion_14 import candidate_config,metrics

DATA=ROOT/'data/public_genvideo_v1.4';WORK=ROOT/'data/joint_video_development_v1.4-r02';SEED=10241407
FEATURES=['generation.logit','forensic_generation.logit','temporal.logit','temporal.window_disagreement','temporal.window_mean',
  'localization.logit','localization.reliable_area_ratio','localization.mean_reliability',
  'localization.largest_region_area_ratio','localization.largest_region_compactness','localization.inside_outside_probability_delta',
  'localization.log_regions','copy_move.logit','copy_move.log_inliers','copy_move.inlier_ratio','copy_move.dense_area_ratio','copy_move.missing']


def prepare():
    WORK.mkdir(exist_ok=True);fixtures=WORK/'fixtures';fixtures.mkdir(exist_ok=True);(fixtures/'parents').mkdir(exist_ok=True)
    if not (fixtures/'empty_truth.png').exists():Image.new('L',(640,480),0).save(fixtures/'empty_truth.png')
    all_rows=json.loads((DATA/'manifest.json').read_text('utf-8'));development=[r for r in all_rows if r['split'] in ('fusion','calibration','threshold')];del all_rows
    plan=[]
    for split in ('fusion','calibration','threshold'):
        part=[r for r in development if r['split']==split];reals=[r for r in part if r['truth_label']=='genuine']
        for r in part:
            plan.append({**r,'workspace_relative_path':(DATA/r['relative_path']).relative_to(ROOT).as_posix(),'original':True})
        for i in range(12):
            host,donor=reals[i*2:i*2+2]
            for control in (False,True):
                plan.append({'sample_id':f'jointvideo14_{split}_{i:02d}'+('_control' if control else ''),
                  'truth_label':'genuine' if control else 'fake','split':split,'material_type':'video',
                  'material_class':'matched_reencoded_control' if control else 'traditional_video_splice',
                  'host':host,'donor':donor,'control':control,'source_groups':[host['source_group'],donor['source_group']]})
    freeze(WORK/'selection_registration.json',{'seed':SEED,'planned_rows':plan,'source_manifest_sha256':sha256(DATA/'manifest.json'),
      'reserved_videos_decoded':False,'registered_before_control_operation_or_predictions':True,
      'known_operations':'12 source-disjoint host/donor pairs per development partition, each with matched control',
      'engineering_motion_limit':'Four-second resampling/loops can change genuine motion; explicit controls retained',
      'formal_acceptance':False})
    finished=[]
    for row in plan:
        if row.get('original'):finished.append(row);continue
        receipt=fixtures/(row['sample_id']+'_receipt.json')
        if receipt.exists():result=json.loads(receipt.read_text('utf-8'))
        else:
            host=decode_parent(DATA,row['host']);donor=None if row['control'] else decode_parent(DATA,row['donor'])
            for parent in [host]+([donor] if donor else []):atomic_json(fixtures/'parents'/(parent['row']['sample_id']+'_timing.json'),parent['timing'])
            result=create_fixture(fixtures,row['sample_id'],host,donor,'matched_reencoded_control' if row['control'] else 'traditional_video_splice')
            result.update(split=row['split'],source_groups=row['source_groups'],workspace_relative_path=(fixtures/result['relative_path']).relative_to(ROOT).as_posix())
            atomic_json(receipt,result)
        finished.append(result)
    freeze(WORK/'manifest.json',finished);return finished


def main():
    rows=prepare();config=candidate_config();freeze(WORK/'extraction_config.json',config)
    models=Models(config);models.load()
    if models.errors:raise RuntimeError(models.errors)
    engine=Engine(config,models);examples=[]
    try:
        for index,row in enumerate(rows):
            target=WORK/(row['sample_id']+'.json')
            if target.exists():
                value=json.loads(target.read_text('utf-8'))
                if value['sha256']!=row['sha256'] or value['temporal_sha256']!=config['temporal_checkpoint_sha256']:raise ValueError('Frozen video evidence cache differs')
            else:
                source=ROOT/row['workspace_relative_path']
                if sha256(source)!=row['sha256']:raise ValueError('Development original bytes differ')
                out=WORK/'evidence'/row['sample_id'];out.mkdir(parents=True,exist_ok=True)
                branches,media=engine.video(source,out)
                failures=[b.branch_id for b in branches if b.status=='failed']
                required={'generation','forensic_generation','temporal','localization'}
                if failures or not required.issubset({b.branch_id for b in branches if b.status=='success'}):raise RuntimeError('Required development model failed:'+str(failures))
                value={**row,'temporal_sha256':config['temporal_checkpoint_sha256'],'features':evidence_features(branches),
                  'branch_scores':{b.branch_id:b.score for b in branches},'branches':[b.to_dict() for b in branches],
                  'actual_sampled_timestamps':[f['timestamp_seconds'] for f in media['frames']],'test_used':False}
                atomic_json(target,value)
            examples.append(value)
            if index%10==0:print('Joint video development',index+1,len(rows),flush=True)
    finally:engine.cpu.shutdown(wait=True)
    tr,cal,th=([e for e in examples if e['split']==s] for s in ('fusion','calibration','threshold'))
    def matrix(part):return np.asarray([[e['features'][name] for name in FEATURES] for e in part])
    scaler=StandardScaler().fit(matrix(tr));classifier=LogisticRegression(C=.3,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(matrix(tr)),[e['truth_label']=='fake' for e in tr])
    raw=classifier.decision_function(scaler.transform(matrix(cal)));platt=LogisticRegression(C=10,max_iter=2000,random_state=SEED).fit(raw[:,None],[e['truth_label']=='fake' for e in cal])
    if platt.coef_[0,0]<=0:raise ValueError('Video calibration reverses risk ordering')
    def probability(part):return expit(classifier.decision_function(scaler.transform(matrix(part)))*platt.coef_[0,0]+platt.intercept_[0])
    scores=probability(th);bounds=np.unique(np.r_[0.,scores,1.]);trials=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        threshold=float((lo+hi)/2);m=metrics(th,scores,threshold)
        if m['genuine_fpr']<=.03:trials.append((m['prior70_30_accuracy'],hi-lo,threshold,m))
    if not trials:raise ValueError('No complete video policy threshold meets development FPR')
    _,_,threshold,m=max(trials,key=lambda r:(r[0],r[1]))
    fitted={'schema':SCHEMA,'trained':True,'features':FEATURES,'required_features':['generation','forensic_generation','temporal','localization'],
      'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coefficients':classifier.coef_[0].tolist(),'intercept':float(classifier.intercept_[0]),
      'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),'threshold':threshold,
      'scope':'Public original generated video and controlled splice reference mixture; not insurance-fraud probability',
      'sample_ids':{s:[e['sample_id'] for e in examples if e['split']==s] for s in ('fusion','calibration','threshold')},
      'development_manifest_sha256':sha256(WORK/'manifest.json'),'temporal_checkpoint_sha256':config['temporal_checkpoint_sha256'],
      'test_used':False,'calibration_metrics':metrics(cal,probability(cal),threshold),'threshold_metrics':m,
      'threshold_policy':'complete temporal/generation/local geometry evidence; no uncalibrated max override'}
    freeze(ROOT/'models/candidates/video_fusion_v1.4.json',fitted)
    atomic_json(ROOT/'runs/joint_video_fusion_v1.4.json',{'specification':fitted,'whole_policy_development_passed':m['prior70_30_accuracy']>=.95 and m['genuine_fpr']<=.03,
      'test_consumed':False,'formal_acceptance':False,'live_config_modified':False,
      'development_rows':[{'sample_id':e['sample_id'],'split':e['split'],'truth_label':e['truth_label'],'probability':float(p)} for e,p in zip(examples,probability(examples))]})
    print(json.dumps({'calibration':fitted['calibration_metrics'],'threshold':m,'threshold_value':threshold}),flush=True)


if __name__=='__main__':main()
