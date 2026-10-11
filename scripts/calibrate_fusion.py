"""Three separate public partitions: fusion fit / probability fit / threshold.

Public holdout is never opened. Config changes only after successful complete
development runs and records all sample IDs and source hashes.
"""
import hashlib
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT,load_config,atomic_json,sha256
from backend.models import Models
from backend.engine import Engine

FEATURES=['generation','forensic_generation','localization','temporal','copy_move']


def vector(branch_scores):
    values=[]
    for name in FEATURES:
        score=branch_scores.get(name)
        # An absent score has a neutral logit AND a missingness flag. It is not
        # treated as negative evidence or counted as a valid visual branch.
        values.append(0.0 if score is None else float(np.log(np.clip(score,.001,.999)/(1-np.clip(score,.001,.999)))))
    return values+[int(branch_scores.get(name) is None) for name in FEATURES]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['image','video'],default='video');parser.add_argument('--cached-only',action='store_true');args=parser.parse_args()
    global FEATURES
    if args.mode=='image':FEATURES=['generation','forensic_generation','localization','copy_move']
    data=ROOT/'data/public_genvideo_v1'
    rows=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    dev=[r for r in rows if r['split'] in ['fusion','calibration','threshold']]
    frozen_text=json.dumps(dev,sort_keys=True)
    config=load_config();engine=None
    if args.cached_only:
        models=SimpleNamespace(temporal=True,waverep=True,info={'temporal':{'sha256':sha256(ROOT/'models/temporal_tcn.pth')},'forensic_generation':{'sha256':sha256(ROOT/'models/weights_dinov2_G4.ckpt')}})
    else:
        models=Models(config);models.load();engine=Engine(config,models)
    if models.temporal is None or models.waverep is None:raise RuntimeError('All trained networks are required before fitting ensemble')
    work=ROOT/('data/fusion_features_pts_v3' if args.mode=='video' else 'data/fusion_features_image_pts_v2');work.mkdir(parents=True,exist_ok=True)
    examples=[]
    try:
        for i,row in enumerate(dev):
            cache=work/(row['sample_id']+'.json')
            if cache.exists():
                stored=json.loads(cache.read_text(encoding='utf-8'))
                if stored['source_sha256']!=row['sha256'] or stored['temporal_sha256']!=models.info['temporal']['sha256']:
                    raise RuntimeError('Development feature cache has a different source or temporal model. Use a separate cache version.')
                result=stored['result']
            else:
                if args.cached_only:raise RuntimeError('No verified development cache: '+row['sample_id'])
                media_path=data/row['relative_path']
                if args.mode=='image':
                    import av
                    media_path=work/(row['sample_id']+'_frame.png')
                    with av.open(str(data/row['relative_path'])) as container:
                        stream=container.streams.video[0]
                        duration=float(stream.duration*stream.time_base) if stream.duration else container.duration/1e6
                        desired=(stream.start_time or 0)+int(duration*.5/float(stream.time_base))
                        container.seek(desired,stream=stream)
                        for frame in container.decode(stream):
                            if frame.pts is not None and frame.pts>=desired:break
                        frame.to_image().convert('RGB').save(media_path)
                result=engine.analyze(media_path,work/row['sample_id'],row['sample_id'],material_type=args.mode,original_name='development'+media_path.suffix)
                atomic_json(cache,{'source_sha256':row['sha256'],'temporal_sha256':models.info['temporal']['sha256'],'result':result})
            if result['execution_status']!='complete':
                raise RuntimeError('Required development execution failed: '+row['sample_id'])
            scores={b['branch_id']:b['score'] for b in result['branches'] if b['status']=='success'}
            override=0.0
            for branch in result['branches']:
                if branch['status']!='success' or branch['score'] is None:continue
                if branch['branch_id']=='localization' and branch['score']>=config.get(f'{args.mode}_strong_localization_min_score',.85) and branch['details'].get('reliable_area_ratio',0)>=.005:
                    override=max(override,branch['score'])
                if branch['branch_id']=='copy_move' and branch['score']>=.85 and branch['details'].get('inliers',0)>=16:
                    override=max(override,branch['score'])
            examples.append({'sample_id':row['sample_id'],'split':row['split'],'y':int(row['truth_label']=='fake'),
                             'x':vector(scores),'scores':scores,'sha256':row['sha256'],'override_risk':override})
            print('development',i+1,len(dev),row['sample_id'],scores,flush=True)
    finally:
        if engine:engine.cpu.shutdown(wait=True)
    fit=[e for e in examples if e['split']=='fusion'];cal=[e for e in examples if e['split']=='calibration'];threshold=[e for e in examples if e['split']=='threshold']
    scaler=StandardScaler().fit([e['x'] for e in fit])
    classifier=LogisticRegression(C=.3,class_weight='balanced',random_state=102411,max_iter=2000).fit(scaler.transform([e['x'] for e in fit]),[e['y'] for e in fit])
    logits=classifier.decision_function(scaler.transform([e['x'] for e in cal]))
    platt=LogisticRegression(C=10,random_state=102411,max_iter=2000).fit(logits[:,None],[e['y'] for e in cal])
    def predict(group):
        base=classifier.decision_function(scaler.transform([e['x'] for e in group]))
        return platt.predict_proba(base[:,None])[:,1]
    # Threshold the complete live policy, including visible localization/copy
    # overrides. Calibrating only the stacker understates genuine false alarms.
    thscores=np.maximum(predict(threshold),[e['override_risk'] for e in threshold]);y=np.asarray([e['y'] for e in threshold])
    candidates=np.unique(np.r_[np.linspace(.01,.99,99),thscores+1e-8]);best=None
    for threshold_value in candidates:
        p=thscores>=threshold_value
        false_positive=int(np.sum(p & (y==0)));g=int(np.sum(y==0))
        if false_positive/max(1,g)>.03:continue
        # Target prior 70/30 is specified before test. Select only threshold data.
        balanced=.7*np.mean(~p[y==0])+.3*np.mean(p[y==1])
        item=(balanced,threshold_value)
        if best is None or item>best[0]:best=(item,float(threshold_value))
    if best is None:raise ValueError('No valid threshold candidate')
    threshold_value=best[1]
    calscores=predict(cal);cal_y=np.asarray([e['y'] for e in cal])
    spec={'schema':'logit_stacker_v1','features':FEATURES,'required_features':FEATURES[:4] if args.mode=='video' else FEATURES[:3], 'missingness_flags':True,
          'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
          'coefficients':classifier.coef_[0].tolist(),'intercept':float(classifier.intercept_[0]),
          'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),
          'threshold':threshold_value,'scope':f'GenVideo development {args.mode}; full genuine / full generated source only; not insurance fraud probability',
          'temporal_checkpoint_sha256':models.info['temporal']['sha256'],
          'wave_checkpoint_sha256':models.info['forensic_generation']['sha256'],
          'development_manifest_sha256':hashlib.sha256(frozen_text.encode()).hexdigest(),
          'sample_ids':{split:[e['sample_id'] for e in examples if e['split']==split] for split in ['fusion','calibration','threshold']},
          'calibration_metrics':{'auc':float(roc_auc_score(cal_y,calscores)),'brier':float(brier_score_loss(cal_y,calscores)),
                                 'samples':len(cal),'class_prior':float(cal_y.mean())},
          'threshold_metrics':{'samples':len(threshold),'genuine':int(np.sum(y==0)),'fake':int(np.sum(y==1)),
                               'accuracy':float(np.mean((thscores>=threshold_value)==y)),
                               'genuine_fpr':float(np.mean(thscores[y==0]>=threshold_value))},
          'test_used':False}
    spec['threshold_policy']='max(public reference probability, strong localization/copy evidence)'
    config=load_config()
    config[f'{args.mode}_fusion']=spec;config[f'{args.mode}_threshold']=threshold_value
    config['configuration_id']='v1.1-public-calibrated-008' if args.mode=='image' else 'v1.1-public-calibrated-007'
    config[f'{args.mode}_calibration_status']='fitted_on_independent_public_calibration_partition'
    atomic_json(ROOT/'config.json',config);atomic_json(ROOT/f'runs/{args.mode}_fusion_calibration.json',spec)
    print('Fitted actual stacker, independent calibration and threshold:',spec['calibration_metrics'],spec['threshold_metrics'],flush=True)


if __name__=='__main__':main()
