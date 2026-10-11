"""Complete image evidence fusion: original generation + actual copy/splice.

Frozen source-head OOF fits exclude host and donor caption groups together.
Every operation has an unchanged matched control and an exact operation mask.
Only development splits are used. Live config and reserved data are untouched.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import train_standalone_image_adapter_13 as base
from train_image_adapter_14 import DATA, WORK as SOURCE_WORK, load_development
from backend.contracts import ROOT,Branch,load_config,atomic_json,sha256
from backend.models import Models
from backend.media import read_image
from backend.copy_move import detect_copy_move
from backend.evidence_fusion import SCHEMA, evidence_features, localization_statistics
from backend.standalone_generation import predict
from scripts.evaluate_public_images_live import freeze

WORK=ROOT/'data/joint_image_development_v1.4';SEED=10241406
FEATURES=['standalone_generation.logit','localization.logit','localization.reliable_area_ratio',
  'localization.mean_reliability','localization.largest_region_area_ratio','localization.largest_region_compactness',
  'localization.inside_outside_probability_delta','localization.log_regions','copy_move.logit',
  'copy_move.log_inliers','copy_move.inlier_ratio','copy_move.dense_area_ratio','copy_move.missing']


def candidate_config():
    baseline=ROOT/'models/versions/1.3-final-preserved-on-14/config.json'
    config=deepcopy(json.loads(baseline.read_text('utf-8')) if baseline.exists() else load_config());path=ROOT/'models/candidates/public_image_adapter_v1.4.json';spec=json.loads(path.read_text('utf-8'))
    config['version']='1.4.0';config['configuration_id']='v1.4-joint-development-001'
    config['image_origin_adapter']={'enabled':True,'checkpoint':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),
      'decision_role':'ensemble_feature','threshold':spec['threshold'],'scope':spec['scope'],'validation_note':'候选：完整融合尚未独立验收'}
    temporal=ROOT/'models/candidates/temporal_tcn_v1.4.pth'
    config.update(temporal_checkpoint=temporal.relative_to(ROOT).as_posix(),temporal_checkpoint_sha256=sha256(temporal),
      video_screening_max_side=320,video_screening_batch_size=4,ocr_field_min_confidence=.85)
    return config


def prepare(rows):
    WORK.mkdir(exist_ok=True);fixtures=WORK/'fixtures';fixtures.mkdir(exist_ok=True)
    plan=[];parents={};fusion_groups={}
    for split in ('fusion','calibration','threshold'):
        part=[r for r in rows if r['split']==split];reals=sorted([r for r in part if not r['official_label_a']],key=lambda r:r['source_group'])
        if split=='fusion':
            by=defaultdict(dict)
            for r in part:by[r['source_group']][int(r['official_label_b'])]=r
            selected=[by[group][1+i%5] for i,group in enumerate(sorted(by))]
            originals=reals+selected;ncopy,nsplice=100,100
            fusion_groups={g:g for g in by}
        else:originals=part;ncopy,nsplice=15,15
        for r in originals:
            plan.append({'sample_id':r['sample_id'],'split':split,'truth_label':r['truth_label'],'material_class':r['material_class'],
              'source_groups':[r['source_group']],'workspace_relative_path':(DATA/r['relative_path']).relative_to(ROOT).as_posix(),'sha256':r['sha256'],'original':True})
        for i in range(ncopy+nsplice):
            host=reals[i];donor=None if i<ncopy else reals[ncopy+nsplice+i-ncopy]
            if split=='fusion' and donor:fusion_groups[donor['source_group']]=host['source_group']
            name=f'joint14_{split}_{i:03d}'
            groups=[host['source_group']]+([donor['source_group']] if donor else [])
            for control in (False,True):
                plan.append({'sample_id':name+('_control' if control else ''),'split':split,'truth_label':'genuine' if control else 'fake',
                  'material_class':'matched_reencoded_control' if control else 'traditional_copy_move' if donor is None else 'traditional_splice',
                  'source_groups':groups,'host':host,'donor':donor,'control':control})
    freeze(WORK/'selection_registration.json',{'seed':SEED,'planned_rows':plan,'source_manifest_sha256':sha256(DATA/'manifest.json'),
      'reserved_images_consumed':False,'registered_before_operation_pixels_and_scores':True,'source_group_pairing_for_OOF':fusion_groups,
      'source_real_is_published_label_not_verified_claim':True,'formal_acceptance':False})
    finished=[]
    for r in plan:
        if r.get('original'):finished.append(r);continue
        destination=fixtures/(r['sample_id']+'.png');truth=fixtures/(r['sample_id']+'_truth.png')
        if not destination.exists():
            host,_=read_image(DATA/r['host']['relative_path'],50000000);image=host.resize((640,480),Image.Resampling.LANCZOS)
            mask=np.zeros((480,640),np.uint8)
            if not r['control']:
                if r['donor'] is None:image.paste(image.crop((40,40,200,200)),(380,90));mask[90:250,380:540]=255
                else:
                    donor,_=read_image(DATA/r['donor']['relative_path'],50000000);donor=donor.resize((640,480),Image.Resampling.LANCZOS)
                    image.paste(donor.crop((160,100,420,320)),(220,160));mask[160:380,220:480]=255
            image.save(destination);Image.fromarray(mask).save(truth)
        finished.append({**r,'workspace_relative_path':destination.relative_to(ROOT).as_posix(),'sha256':sha256(destination),
          'mask_path':truth.relative_to(ROOT).as_posix(),'mask_sha256':sha256(truth)})
    freeze(WORK/'manifest.json',finished)
    return finished,fusion_groups


def make_oof(source_rows,group_map,spec):
    path=WORK/'oof_heads.json'
    if path.exists():return json.loads(path.read_text('utf-8'))
    with np.load(SOURCE_WORK/'features.npz',allow_pickle=False) as data:features=data['features'].copy()
    tr=np.asarray([i for i,r in enumerate(source_rows) if r['split']=='fusion']);y=np.asarray([int(r['official_label_a']) for r in source_rows])
    groups=np.asarray([group_map[r['source_group']] for r in source_rows if r['split']=='fusion'])
    folds=StratifiedGroupKFold(3,shuffle=True,random_state=SEED);heads=[];assignment={};probabilities={}
    for fold,(a,b) in enumerate(folds.split(features[tr],y[tr],groups)):
        left,right=tr[a],tr[b];blocks=base.make_blocks(spec['candidate'],features[left])
        head=LogisticRegression(C=spec['selected_C'],class_weight='balanced',max_iter=2500,random_state=SEED).fit(base.transform_matrix(blocks,features[left]),y[left])
        frozen={**spec,'feature_blocks':blocks,'coefficients':head.coef_[0].tolist(),'intercept':float(head.intercept_[0])};heads.append(frozen)
        for i in right:
            group=source_rows[i]['source_group'];assignment[group]=fold;probabilities[source_rows[i]['sample_id']]=predict(frozen,features[i])
    for r in source_rows:
        if r['split']=='fusion':
            for g in group_map:
                if group_map[g]==group_map[r['source_group']] and assignment[g]!=assignment[r['source_group']]:raise ValueError('Host/donor crossed OOF fitting boundary')
    result={'heads':heads,'source_group_fold':assignment,'original_oof_probabilities':probabilities,
      'source_adapter_sha256':sha256(ROOT/'models/candidates/public_image_adapter_v1.4.json'),'test_used':False}
    freeze(path,result);return result


def metrics(examples,probability,threshold):
    y=np.asarray([r['truth_label']=='fake' for r in examples]);p=np.asarray(probability)>=threshold
    real=y==0;fake=~real;fp=int((p&real).sum());tp=int((p&fake).sum())
    # Match the contest prior while retaining every declared development example.
    fpr=float(p[real].mean());recall=float(p[fake].mean())
    return {'samples':len(y),'genuine':int(real.sum()),'fake':int(fake.sum()),'fp':fp,'tp':tp,
      'accuracy':float((p==y).mean()),'genuine_fpr':fpr,'fake_recall':recall,'prior70_30_accuracy':.7*(1-fpr)+.3*recall}


def main():
    source_rows,_=load_development();rows,group_map=prepare(source_rows);config=candidate_config();freeze(WORK/'extraction_config.json',config)
    spec=json.loads((ROOT/'models/candidates/public_image_adapter_v1.4.json').read_text('utf-8'));oof=make_oof(source_rows,group_map,spec)
    source_probability={r['sample_id']:r['probability'] for r in json.loads((ROOT/'runs/public_image_adapter_training_v1.4.json').read_text('utf-8'))['development_rows']}
    models=Models(config);models.load()
    if models.errors:raise RuntimeError(models.errors)
    geometry=ThreadPoolExecutor(2);examples=[]
    try:
        for index,row in enumerate(rows):
            cache=WORK/(row['sample_id']+'.json')
            if cache.exists():
                value=json.loads(cache.read_text('utf-8'))
                if value['sha256']!=row['sha256'] or value['source_adapter_sha256']!=sha256(ROOT/'models/candidates/public_image_adapter_v1.4.json'):raise ValueError('Frozen development features differ')
            else:
                image,_=read_image(ROOT/row['workspace_relative_path'],50000000)
                future=geometry.submit(detect_copy_move,image,config['copy_max_side'])
                score,p,rel,size=models.localize(image)
                loc=Branch('localization','success',score,details={**localization_statistics(p,rel,config['mask_threshold']),'original_size':list(image.size),'model_input_size':size})
                if row.get('original'):
                    origin=oof['original_oof_probabilities'][row['sample_id']] if row['split']=='fusion' else source_probability[row['sample_id']]
                else:
                    feature=models.encode([image])[0][0]
                    origin,_,_,descriptor=models.image_adapter_score(image,feature,return_descriptor=True)
                    if row['split']=='fusion':
                        fold=oof['source_group_fold'][row['source_groups'][0]]
                        if any(oof['source_group_fold'][g]!=fold for g in row['source_groups']):raise ValueError('Edit donor crossed OOF group boundary')
                        origin=predict(oof['heads'][fold],descriptor)
                copy=future.result();branches=[Branch('standalone_generation','success',origin),loc,copy]
                value={**row,'source_adapter_sha256':sha256(ROOT/'models/candidates/public_image_adapter_v1.4.json'),
                  'features':evidence_features(branches),'branch_scores':{b.branch_id:b.score for b in branches},'test_used':False}
                atomic_json(cache,value)
            examples.append(value)
            if index%40==0:print('Joint image evidence',index+1,len(rows),flush=True)
    finally:geometry.shutdown(wait=True)
    tr=[e for e in examples if e['split']=='fusion'];cal=[e for e in examples if e['split']=='calibration'];th=[e for e in examples if e['split']=='threshold']
    def matrix(part):return np.asarray([[e['features'][name] for name in FEATURES] for e in part])
    scaler=StandardScaler().fit(matrix(tr));head=LogisticRegression(C=.3,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(matrix(tr)),[e['truth_label']=='fake' for e in tr])
    raw=head.decision_function(scaler.transform(matrix(cal)));platt=LogisticRegression(C=10,max_iter=2000,random_state=SEED).fit(raw[:,None],[e['truth_label']=='fake' for e in cal])
    if platt.coef_[0,0]<=0:raise ValueError('Joint calibration reverses score direction')
    def probability(part):return expit(head.decision_function(scaler.transform(matrix(part)))*platt.coef_[0,0]+platt.intercept_[0])
    scores=probability(th);bounds=np.unique(np.r_[0.,scores,1.]);trials=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        threshold=float((lo+hi)/2);m=metrics(th,scores,threshold)
        if m['genuine_fpr']<=.03:trials.append((m['prior70_30_accuracy'],hi-lo,threshold,m))
    if not trials:raise ValueError('No complete joint image threshold meets development FPR')
    _,_,threshold,m=max(trials,key=lambda r:(r[0],r[1]))
    fitted={'schema':SCHEMA,'trained':True,'features':FEATURES,'required_features':['generation','forensic_generation','localization','standalone_generation'],
      'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coefficients':head.coef_[0].tolist(),'intercept':float(head.intercept_[0]),
      'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),'threshold':threshold,
      'scope':'Public generated-origin and controlled copy/splice reference mixture; not insurance-fraud probability',
      'sample_ids':{s:[e['sample_id'] for e in examples if e['split']==s] for s in ('fusion','calibration','threshold')},
      'development_manifest_sha256':sha256(WORK/'manifest.json'),'source_adapter_sha256':sha256(ROOT/'models/candidates/public_image_adapter_v1.4.json'),
      'original_and_edit_host_donor_OOF':True,'test_used':False,'calibration_metrics':metrics(cal,probability(cal),threshold),'threshold_metrics':m,
      'threshold_policy':'complete trained generation, localization shape/reliability and optional copy geometry; no uncalibrated max override'}
    freeze(ROOT/'models/candidates/image_fusion_v1.4.json',fitted)
    atomic_json(ROOT/'runs/joint_image_fusion_v1.4.json',{'specification':fitted,'whole_policy_development_passed':m['prior70_30_accuracy']>=.95 and m['genuine_fpr']<=.03,
      'test_consumed':False,'formal_acceptance':False,'live_config_modified':False,
      'development_rows':[{'sample_id':e['sample_id'],'split':e['split'],'truth_label':e['truth_label'],'probability':float(p)} for e,p in zip(examples,probability(examples))]})
    print(json.dumps({'calibration':fitted['calibration_metrics'],'threshold':m,'threshold_value':threshold}),flush=True)


if __name__=='__main__':main()
