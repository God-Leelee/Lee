"""Source-isolated fusion selection, including trained two-keyframe origin.

Original and paired-augmentation temporal candidates compete only in fusion
partition grouped CV. Calibration/threshold/test never select this choice.
"""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from backend.contracts import ROOT,sha256,atomic_json
from scripts.evaluate_public_images_live import freeze
from scripts.train_joint_image_fusion_14 import metrics
from scripts.train_joint_video_fusion_14 import FEATURES as OLD_FEATURES

WORK=ROOT/'data/joint_video_source_development_v1.5-r02';SEED=10241508
FEATURES=OLD_FEATURES+['video_origin.logit','video_origin.temporal_agreement_logit','video_origin.localization_agreement_logit']


def logit(p):
    p=min(.99999,max(.00001,p));return math.log(p/(1-p))


def main():
    WORK.mkdir(exist_ok=True)
    origin=json.loads((ROOT/'data/video_origin_development_v1.5/development_predictions.json').read_text('utf-8'))
    temporal=json.loads((ROOT/'data/temporal_compression_v1.5/development_temporal_predictions.json').read_text('utf-8'))
    rows=[]
    for folder in (ROOT/'data/joint_video_development_v1.4-r02',ROOT/'data/compression_development_v1.5'):
        for row in json.loads((folder/'manifest.json').read_text('utf-8')):
            saved=json.loads((folder/(row['sample_id']+'.json')).read_text('utf-8'))
            features=dict(saved['features']);features['video_origin.logit']=logit(origin[row['sample_id']]['probability'])
            groups=row.get('source_groups') or [row['source_group']]
            rows.append({**saved,'features':features,'source_groups':groups,'origin_prediction':origin[row['sample_id']],
              'augmented_temporal_prediction':temporal[row['sample_id']],'non_temporal_cache_sha256':sha256(folder/(row['sample_id']+'.json'))})
    freeze(WORK/'development_evidence.json',rows)
    protocol={'seed':SEED,'features':FEATURES,'C_candidates':[.1,.3,1.],
      'temporal_candidates':['original1.4','paired_compression1.5'],'selection':'three-fold fusion source-connected-component partial AUC at FPR.03',
      'calibration':'separate calibration Platt C10, positive slope','threshold':'separate threshold FPR<=.03, maximize prior70/30 accuracy',
      'reserved_used':False,'head_training_sources_disjoint_from_fusion':True}
    freeze(WORK/'protocol.json',protocol)
    parts={s:[r for r in rows if r['split']==s] for s in ('fusion','calibration','threshold')};tr=parts['fusion']
    parent={g:g for r in tr for g in r['source_groups']}
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    for row in tr:
        left=find(row['source_groups'][0])
        for g in row['source_groups'][1:]:parent[find(g)]=left
    groups=np.asarray([find(r['source_groups'][0]) for r in tr]);y=np.asarray([r['truth_label']=='fake' for r in tr])
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(np.zeros((len(tr),1)),y,groups))
    def matrix(part,temporal_choice):
        output=[]
        for row in part:
            f=dict(row['features'])
            if temporal_choice=='paired_compression1.5':
                t=row['augmented_temporal_prediction'];f.update({'temporal.logit':logit(t['temporal_risk']),
                  'temporal.window_mean':t['window_mean'],'temporal.window_disagreement':t['window_disagreement']})
            f['video_origin.temporal_agreement_logit']=min(f['video_origin.logit'],f['temporal.logit'])
            f['video_origin.localization_agreement_logit']=min(f['video_origin.logit'],f['localization.logit'])
            output.append([f[name] for name in FEATURES])
        return np.asarray(output)
    trials=[]
    for temporal_choice in protocol['temporal_candidates']:
        x=matrix(tr,temporal_choice)
        for c in protocol['C_candidates']:
            scores=[]
            for a,b in folds:
                if set(groups[a])&set(groups[b]):raise ValueError('Video fusion source connected components leaked')
                scaler=StandardScaler().fit(x[a]);head=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(x[a]),y[a])
                p=head.predict_proba(scaler.transform(x[b]))[:,1];scores.append(float(roc_auc_score(y[b],p,max_fpr=.03)))
            trials.append({'temporal':temporal_choice,'C':c,'mean_source_cv_partial_auc':float(np.mean(scores))})
    choice=max(trials,key=lambda r:(r['mean_source_cv_partial_auc'],-r['C'],r['temporal']=='original1.4'))
    x=matrix(tr,choice['temporal']);scaler=StandardScaler().fit(x)
    head=LogisticRegression(C=choice['C'],class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(x),y)
    cal,th=parts['calibration'],parts['threshold'];raw=head.decision_function(scaler.transform(matrix(cal,choice['temporal'])))
    platt=LogisticRegression(C=10,max_iter=2000,random_state=SEED).fit(raw[:,None],[r['truth_label']=='fake' for r in cal])
    if platt.coef_[0,0]<=0:raise ValueError('Video fusion calibration reverses meaning')
    def probability(part):return expit(head.decision_function(scaler.transform(matrix(part,choice['temporal'])))*platt.coef_[0,0]+platt.intercept_[0])
    p=probability(th);bounds=np.unique(np.r_[0.,p,1.]);choices=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        threshold=float((lo+hi)/2);m=metrics(th,p,threshold)
        if m['genuine_fpr']<=.03:choices.append((m['prior70_30_accuracy'],hi-lo,threshold,m))
    _,_,threshold,m=max(choices,key=lambda r:(r[0],r[1]));cal_metrics=metrics(cal,probability(cal),threshold)
    checkpoint=ROOT/('models/candidates/temporal_tcn_v1.4.pth' if choice['temporal']=='original1.4' else 'models/candidates/temporal_tcn_v1.5-compression.pth')
    spec={**json.loads((ROOT/'models/candidates/video_fusion_v1.4.json').read_text('utf-8')),
      'features':FEATURES,'required_features':['generation','forensic_generation','temporal','localization','video_origin'],
      'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coefficients':head.coef_[0].tolist(),
      'intercept':float(head.intercept_[0]),'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),
      'threshold':threshold,'version':'1.5.0-source-candidate','temporal_checkpoint_sha256':sha256(checkpoint),
      'video_origin_checkpoint_sha256':sha256(ROOT/'models/candidates/video_origin_v1.5.json'),
      'sample_ids':{s:[r['sample_id'] for r in part] for s,part in parts.items()},
      'development_evidence_sha256':sha256(WORK/'development_evidence.json'),'calibration_metrics':cal_metrics,'threshold_metrics':m,
      'test_used':False,'selection':choice,'source_partition_groups':{s:len({g for r in part for g in r['source_groups']}) for s,part in parts.items()}}
    freeze(ROOT/'models/candidates/video_fusion_v1.5-source-r02.json',spec)
    passed=all(value['prior70_30_accuracy']>=.95 and value['genuine_fpr']<=.03 for value in (cal_metrics,m))
    atomic_json(ROOT/'runs/joint_video_source_v1.5-r02.json',{'specification':spec,'trials':trials,'chosen':choice,
      'whole_policy_development_passed':passed,'temporal_checkpoint':checkpoint.relative_to(ROOT).as_posix(),
      'live_config_modified':False,'reserved_used':False,'formal_acceptance':False,
      'development_rows':[{'sample_id':r['sample_id'],'truth_label':r['truth_label'],'split':r['split'],'probability':float(p)} for r,p in zip(rows,probability(rows))]})
    print(json.dumps({'chosen':choice,'calibration':cal_metrics,'threshold':m,'threshold_value':threshold,'development_passed':passed}),flush=True)


if __name__=='__main__':main()
