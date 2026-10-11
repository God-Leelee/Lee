"""Expanded, source-grouped training. No reserved image is opened or predicted.

Only the offline feature extractor is reused from 1.3; all caches, protocols and
outputs are separate. This creates a candidate, never changes live config.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_standalone_image_adapter_13 as base
from backend.contracts import ROOT, atomic_json, sha256
from backend.standalone_generation import FEATURE_SPEC, predict, validate_spec

DATA=ROOT/'data/public_images_v1.4'
WORK=DATA/'training_v1.4'
MODEL=ROOT/'models/candidates/public_image_adapter_v1.4.json'
REPORT=ROOT/'runs/public_image_adapter_training_v1.4.json'
SEED=10241402
CS=[.3,1.,3.,10.,30.]


def load_development():
    all_rows=json.loads((DATA/'manifest.json').read_text('utf-8'))
    # Filter before resolving paths or checking labels. Reserved pixels stay sealed.
    rows=[r for r in all_rows if r['split'] in ('fusion','calibration','threshold')]
    del all_rows
    if Counter(r['split'] for r in rows)!={'fusion':3000,'calibration':100,'threshold':100}:
        raise ValueError('Registered expanded development counts differ')
    seen_groups=set();seen_hashes=set();seen_ids=set()
    for split,count in [('fusion',500),('calibration',100),('threshold',100)]:
        part=[r for r in rows if r['split']==split];groups={r['source_group'] for r in part}
        if len(groups)!=count or groups&seen_groups:raise ValueError('Caption grouping leakage')
        seen_groups.update(groups)
        labels=Counter(int(r['official_label_b']) for r in part)
        expected=Counter({i:500 for i in range(6)}) if split=='fusion' else Counter({0:70,1:6,2:6,3:6,4:6,5:6})
        if labels!=expected:raise ValueError('Source stratification differs')
        siblings=defaultdict(list)
        for r in part:
            siblings[r['source_group']].append(int(r['official_label_b']))
            if r['source_revision']!=base.REVISION or int(r['official_label_a'])!=int(r['official_label_b']!=0):raise ValueError('Pinned annotation mismatch')
            path=(DATA/r['relative_path']).resolve()
            if not path.is_relative_to(DATA.resolve()) or sha256(path)!=r['sha256']:raise ValueError('Original byte hash/path mismatch')
            if r['sha256'] in seen_hashes or r['sample_id'] in seen_ids:raise ValueError('Exact duplicate development sample')
            seen_hashes.add(r['sha256']);seen_ids.add(r['sample_id'])
        if split=='fusion' and any(sorted(values)!=list(range(6)) for values in siblings.values()):raise ValueError('Incomplete source siblings')
    return rows,hashlib.sha256(base.canonical(rows)).hexdigest()


def predeclare(digest):
    WORK.mkdir(parents=True,exist_ok=True)
    protocol={'version':'1.4-candidate','seed':SEED,'development_manifest_sha256':digest,
      'candidates':base.CANDIDATES,'C_grid':CS,'fusion_images':3000,'fusion_groups':500,
      'calibration_images':100,'threshold_images':100,'holdout_consumed':False,
      'selection':'three-fold caption-group CV: mean standardized partial AUC at FPR .03; tie full AUC then smaller C',
      'calibration':'Platt C=10 on separate calibration100, positive slope required',
      'threshold':'separate threshold100, maximize accuracy with source-real FPR<=.03; tie widest adjacent-score interval midpoint',
      'live_config_modified':False,'complete_ensemble_validated':False,'formal_acceptance':False}
    path=WORK/'training_protocol.json'
    if path.exists() and json.loads(path.read_text('utf-8'))!=protocol:raise ValueError('Frozen training experiment changed')
    if not path.exists():atomic_json(path,protocol)


def fit(rows,digest):
    began=time.perf_counter();cache=WORK/'features.npz'
    with np.load(cache,allow_pickle=False) as data:
        if (data['development_manifest_sha256'].item()!=digest or data['feature_spec'].item()!=FEATURE_SPEC
            or data['clip_checkpoint_sha256'].item()!=base.CLIP_SHA or data['wave_checkpoint_sha256'].item()!=base.WAVE_SHA
            or data['sample_ids'].astype(str).tolist()!=[r['sample_id'] for r in rows]
            or data['source_hashes'].astype(str).tolist()!=[r['sha256'] for r in rows]):raise ValueError('Frozen feature cache differs')
        features=data['features'].copy()
    if features.shape!=(3200,3125) or not np.isfinite(features).all():raise ValueError('Invalid expanded feature matrix')
    y=np.asarray([int(r['official_label_a']) for r in rows]);groups=np.asarray([r['source_group'] for r in rows]);splits=np.asarray([r['split'] for r in rows])
    indices={s:np.where(splits==s)[0] for s in ('fusion','calibration','threshold')};tr,cal,th=(indices[s] for s in indices)
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(features[tr],y[tr],groups[tr]))
    trials=[]
    for name in base.CANDIDATES:
        for c in CS:
            oof=np.empty(len(tr));fold_metrics=[]
            for a,b in folds:
                left,right=tr[a],tr[b]
                if set(groups[left])&set(groups[right]):raise ValueError('CV siblings leaked')
                blocks=base.make_blocks(name,features[left]);xfit=base.transform_matrix(blocks,features[left]);xval=base.transform_matrix(blocks,features[right])
                head=LogisticRegression(C=c,class_weight='balanced',max_iter=2500,random_state=SEED).fit(xfit,y[left])
                p=head.predict_proba(xval)[:,1];oof[b]=p
                fold_metrics.append({'auc':float(roc_auc_score(y[right],p)), 'partial_auc_fpr003':float(roc_auc_score(y[right],p,max_fpr=.03))})
            item={'candidate':name,'C':c,'mean_partial_auc':float(np.mean([v['partial_auc_fpr003'] for v in fold_metrics])),
              'mean_auc':float(np.mean([v['auc'] for v in fold_metrics])),'folds':fold_metrics}
            trials.append(item);print(json.dumps(item),flush=True)
    choice=max(trials,key=lambda r:(r['mean_partial_auc'],r['mean_auc'],-r['C'],-base.CANDIDATES.index(r['candidate'])))
    blocks=base.make_blocks(choice['candidate'],features[tr]);x=base.transform_matrix(blocks,features)
    head=LogisticRegression(C=choice['C'],class_weight='balanced',max_iter=2500,random_state=SEED).fit(x[tr],y[tr])
    raw=head.decision_function(x)
    platt=LogisticRegression(C=10,max_iter=2000,random_state=SEED).fit(raw[cal,None],y[cal])
    if platt.coef_[0,0]<=0:raise ValueError('Calibration slope reverses origin semantics')
    p=expit(raw*platt.coef_[0,0]+platt.intercept_[0])
    boundaries=np.unique(np.r_[0.,p[th],1.]);thresholds=[]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        t=float((lo+hi)/2);m=base.metrics(y[th],p[th],t)
        if m['non_ai_fpr']<=.03:thresholds.append({'threshold':t,'gap':float(hi-lo),'metrics':m})
    if not thresholds:raise ValueError('No threshold satisfies development FPR limit')
    selected=max(thresholds,key=lambda r:(r['metrics']['accuracy_all_samples'],r['gap']))
    spec={'version':'1.4.0-candidate','architecture':'frozen_clip_wave_multiblock_logistic_v2','feature_spec':FEATURE_SPEC,
      'candidate':choice['candidate'],'selected_C':choice['C'],'trained':True,'feature_blocks':blocks,
      'encoder_sha256':base.CLIP_SHA,'wave_checkpoint_sha256':base.WAVE_SHA,
      'coefficients':head.coef_[0].tolist(),'intercept':float(head.intercept_[0]),
      'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),'threshold':selected['threshold'],
      'threshold_scope':'image-origin head only, complete local-edit fusion must be independently frozen and measured',
      'threshold_metrics':selected['metrics'],'calibration_metrics':base.metrics(y[cal],p[cal],selected['threshold']),
      'sample_ids':{s:[rows[i]['sample_id'] for i in idx] for s,idx in indices.items()},
      'source_groups':{s:sorted(set(groups[idx].tolist())) for s,idx in indices.items()},
      'development_manifest_sha256':digest,'feature_cache_sha256':sha256(cache),'source_revision':base.REVISION,
      'dataset_source':'https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset',
      'generator_families':['Stable Diffusion','DALL-E','Midjourney'],'holdout_used_for_fit':False,
      'holdout_used_for_candidate_selection':False,'runtime_deployed':False,'formal_acceptance':False,
      'scope':'public image-origin reference domain; source-real does not mean untampered or legitimate insurance claim',
      'support_gate':None,'labels_captions_metadata_used_as_live_inputs':False,'upstream_pretraining_overlap_known':False}
    validate_spec(spec,base.CLIP_SHA,base.WAVE_SHA)
    reconstructed=np.asarray([predict(spec,f) for f in features]);delta=float(np.max(np.abs(reconstructed-p)))
    if delta>1e-10:raise ValueError('Serialized candidate does not reproduce training probabilities')
    if MODEL.exists() and json.loads(MODEL.read_text('utf-8'))!=spec:raise ValueError('Candidate is frozen; use another version')
    if not MODEL.exists():atomic_json(MODEL,spec)
    report={'model_file':str(MODEL.relative_to(ROOT)),'model_sha256':sha256(MODEL),'chosen':choice,'trials':trials,
      'calibration_metrics':spec['calibration_metrics'],'threshold_metrics':spec['threshold_metrics'],'head_threshold':spec['threshold'],
      'development_images':len(rows),'holdout_consumed':False,'runtime_config_modified':False,'complete_ensemble_validated':False,
      'serialized_probability_max_difference':delta,'formal_acceptance':False,'elapsed_seconds':time.perf_counter()-began,
      'development_rows':[{'sample_id':r['sample_id'],'source_group':r['source_group'],'split':r['split'],
          'label':int(y[i]),'source_label':int(r['official_label_b']),'probability':float(p[i])} for i,r in enumerate(rows)]}
    atomic_json(REPORT,report)
    print(json.dumps({k:report[k] for k in ('chosen','calibration_metrics','threshold_metrics','head_threshold','elapsed_seconds')},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['preflight','extract','fit','all'],required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=parser.parse_args()
    rows,digest=load_development();predeclare(digest)
    print('Verified development-only originals',len(rows),digest,flush=True)
    if args.phase in ('extract','all'):
        base.DATA=DATA;base.WORK=WORK
        base.extract(rows,digest,args.device)
    if args.phase in ('fit','all'):fit(rows,digest)


if __name__=='__main__':main()
