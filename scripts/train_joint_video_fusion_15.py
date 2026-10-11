"""Refit fixed video ensemble after paired compression temporal training.

Spatial branches are byte-identical frozen development predictions. Temporal
values are regenerated from actual features/PTS through the new checkpoint.
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
from backend.contracts import ROOT,sha256,atomic_json
from scripts.evaluate_public_images_live import freeze
from scripts.train_joint_image_fusion_14 import metrics
from scripts.train_joint_video_fusion_14 import FEATURES

WORK=ROOT/'data/joint_video_development_v1.5'
BASE=ROOT/'data/joint_video_development_v1.4-r02'
COMPRESSION=ROOT/'data/compression_development_v1.5'
TEMPORAL=ROOT/'data/temporal_compression_v1.5'
SEED=10241506


def main():
    WORK.mkdir(exist_ok=True)
    checkpoint=ROOT/'models/candidates/temporal_tcn_v1.5-compression.pth'
    temporal=json.loads((TEMPORAL/'development_temporal_predictions.json').read_text('utf-8'))
    rows=[]
    for folder in (BASE,COMPRESSION):
        manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
        for row in manifest:
            stored=json.loads((folder/(row['sample_id']+'.json')).read_text('utf-8'))
            old_features=dict(stored['features']);new=temporal[row['sample_id']]
            p=min(.99999,max(.00001,new['temporal_risk']))
            old_features.update({'temporal.logit':math.log(p/(1-p)),
              'temporal.window_mean':new['window_mean'],'temporal.window_disagreement':new['window_disagreement']})
            rows.append({**stored,'features':old_features,'temporal_sha256':sha256(checkpoint),
              'temporal_prediction':new,'non_temporal_cache_sha256':sha256(folder/(row['sample_id']+'.json'))})
    freeze(WORK/'development_evidence.json',rows)
    tr,cal,th=([r for r in rows if r['split']==s] for s in ('fusion','calibration','threshold'))
    def matrix(part):return np.asarray([[r['features'][f] for f in FEATURES] for r in part])
    scaler=StandardScaler().fit(matrix(tr));head=LogisticRegression(C=.3,class_weight='balanced',max_iter=2000,random_state=SEED).fit(scaler.transform(matrix(tr)),[r['truth_label']=='fake' for r in tr])
    raw=head.decision_function(scaler.transform(matrix(cal)))
    platt=LogisticRegression(C=10,max_iter=2000,random_state=SEED).fit(raw[:,None],[r['truth_label']=='fake' for r in cal])
    if platt.coef_[0,0]<=0:raise ValueError('New ensemble calibration reverses risk')
    def probability(part):return expit(head.decision_function(scaler.transform(matrix(part)))*platt.coef_[0,0]+platt.intercept_[0])
    scores=probability(th);bounds=np.unique(np.r_[0.,scores,1.]);trials=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        threshold=float((lo+hi)/2);m=metrics(th,scores,threshold)
        if m['genuine_fpr']<=.03:trials.append((m['prior70_30_accuracy'],hi-lo,threshold,m))
    _,_,threshold,m=max(trials,key=lambda r:(r[0],r[1]))
    spec={**json.loads((ROOT/'models/candidates/video_fusion_v1.4.json').read_text('utf-8')),
      'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coefficients':head.coef_[0].tolist(),
      'intercept':float(head.intercept_[0]),'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),
      'threshold':threshold,'temporal_checkpoint_sha256':sha256(checkpoint),'version':'1.5.0-compression-temporal-candidate',
      'sample_ids':{s:[r['sample_id'] for r in rows if r['split']==s] for s in ('fusion','calibration','threshold')},
      'development_evidence_sha256':sha256(WORK/'development_evidence.json'),'test_used':False,
      'calibration_metrics':metrics(cal,probability(cal),threshold),'threshold_metrics':m,
      'compression_rule':'Paired real/generated transcodes, both labels, source-disjoint partitions; no codec-name shortcut'}
    freeze(ROOT/'models/candidates/video_fusion_v1.5-compression-temporal.json',spec)
    atomic_json(ROOT/'runs/joint_video_fusion_v1.5-compression-temporal.json',{'specification':spec,
      'whole_policy_development_passed':m['prior70_30_accuracy']>=.95 and m['genuine_fpr']<=.03,
      'live_config_modified':False,'formal_acceptance':False,'reserved_inference_used':False,
      'development_rows':[{'sample_id':r['sample_id'],'split':r['split'],'truth_label':r['truth_label'],'probability':float(p)} for r,p in zip(rows,probability(rows))]})
    print(json.dumps({'calibration':spec['calibration_metrics'],'threshold':m,'threshold_value':threshold}),flush=True)


if __name__=='__main__':main()
