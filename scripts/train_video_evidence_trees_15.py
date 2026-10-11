"""Shallow conditional evidence ensemble, selected by parent-group fusion CV.

Only finite numeric model/geometry evidence enters this classifier. No labels,
codec, source identity, prompt, name, or known-operation metadata enter runtime.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from backend.contracts import ROOT,atomic_json,sha256
from scripts.evaluate_public_images_live import freeze
from scripts.train_joint_image_fusion_14 import metrics
from scripts.train_joint_video_source_15_r02 import FEATURES,logit

WORK=ROOT/'data/video_evidence_trees_v1.5';SEED=10241509


def main():
    WORK.mkdir(exist_ok=True)
    rows=json.loads((ROOT/'data/joint_video_source_development_v1.5-r02/development_evidence.json').read_text('utf-8'))
    parts={s:[r for r in rows if r['split']==s] for s in ('fusion','calibration','threshold')}
    protocol={'seed':SEED,'features':FEATURES,'iterations':100,'learning_rate':.05,'l2_regularization':1.,
      'depth_grid':[2,3],'min_leaf_grid':[8,12],'temporal_candidates':['original1.4','paired_compression1.5'],
      'selection':'3-fold fusion connected parent-source groups partial AUC at FPR .03',
      'calibration':'separate calibration Platt C10 positive slope','threshold':'separate threshold FPR<=.03, maximum prior70/30 accuracy',
      'reserved_used':False,'early_stopping':False,'codec_metadata_source_names_as_model_inputs':False}
    freeze(WORK/'protocol.json',protocol)
    tr=parts['fusion'];parent={g:g for r in tr for g in r['source_groups']}
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    for r in tr:
        for g in r['source_groups'][1:]:parent[find(g)]=find(r['source_groups'][0])
    groups=np.asarray([find(r['source_groups'][0]) for r in tr]);y=np.asarray([r['truth_label']=='fake' for r in tr])
    def matrix(part,temporal):
        values=[]
        for r in part:
            f=dict(r['features'])
            if temporal=='paired_compression1.5':
                t=r['augmented_temporal_prediction'];f.update({'temporal.logit':logit(t['temporal_risk']),
                  'temporal.window_mean':t['window_mean'],'temporal.window_disagreement':t['window_disagreement']})
            f['video_origin.temporal_agreement_logit']=min(f['video_origin.logit'],f['temporal.logit'])
            f['video_origin.localization_agreement_logit']=min(f['video_origin.logit'],f['localization.logit'])
            values.append([f[k] for k in FEATURES])
        return np.asarray(values)
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(np.zeros((len(tr),1)),y,groups))
    def model(depth,leaf):return HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_depth=depth,min_samples_leaf=leaf,l2_regularization=1.,early_stopping=False,random_state=SEED)
    def weights(labels):return np.where(labels,len(labels)/(2*labels.sum()),len(labels)/(2*(~labels).sum()))
    trials=[]
    for temporal in protocol['temporal_candidates']:
        x=matrix(tr,temporal)
        for depth in (2,3):
            for leaf in (8,12):
                aucs=[]
                for a,b in folds:
                    if set(groups[a])&set(groups[b]):raise ValueError('Evidence-tree source group leakage')
                    head=model(depth,leaf).fit(x[a],y[a],sample_weight=weights(y[a]));p=head.predict_proba(x[b])[:,1]
                    aucs.append(float(roc_auc_score(y[b],p,max_fpr=.03)))
                trial={'temporal':temporal,'depth':depth,'leaf':leaf,'mean_source_cv_partial_auc':float(np.mean(aucs))};trials.append(trial);print(trial,flush=True)
    choice=max(trials,key=lambda r:(r['mean_source_cv_partial_auc'],-r['depth'],r['leaf'],r['temporal']=='original1.4'))
    head=model(choice['depth'],choice['leaf']).fit(matrix(tr,choice['temporal']),y,sample_weight=weights(y))
    cal,th=parts['calibration'],parts['threshold'];raw=head.decision_function(matrix(cal,choice['temporal']))
    platt=LogisticRegression(C=10,max_iter=2000,random_state=SEED).fit(raw[:,None],[r['truth_label']=='fake' for r in cal])
    if platt.coef_[0,0]<=0:raise ValueError('Tree calibration reverses meaning')
    def probability(part):return expit(head.decision_function(matrix(part,choice['temporal']))*platt.coef_[0,0]+platt.intercept_[0])
    p=probability(th);bounds=np.unique(np.r_[0.,p,1.]);trials_t=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        threshold=float((lo+hi)/2);m=metrics(th,p,threshold)
        if m['genuine_fpr']<=.03:trials_t.append((m['prior70_30_accuracy'],hi-lo,threshold,m))
    _,_,threshold,m=max(trials_t,key=lambda r:(r[0],r[1]));cal_metrics=metrics(cal,probability(cal),threshold)
    trees=[]
    for iteration in head._predictors:
        if len(iteration)!=1:raise ValueError('Only binary numeric trees are supported')
        trees.append([{'value':float(n['value']),'feature':int(n['feature_idx']),'threshold':float(n['num_threshold']),
          'left':int(n['left']),'right':int(n['right']),'leaf':bool(n['is_leaf'])} for n in iteration[0].nodes])
    baseline=float(head._baseline_prediction[0,0])
    def export_raw(x):
        total=baseline
        for tree in trees:
            node=0
            while not tree[node]['leaf']:
                n=tree[node];node=n['left'] if x[n['feature']]<=n['threshold'] else n['right']
            total+=tree[node]['value']
        return total
    xall=matrix(rows,choice['temporal']);difference=float(np.max(np.abs(np.asarray([export_raw(x) for x in xall])-head.decision_function(xall))))
    if difference>1e-10:raise ValueError('Exported numeric trees do not reproduce the fitted estimator')
    checkpoint=ROOT/('models/candidates/temporal_tcn_v1.4.pth' if choice['temporal']=='original1.4' else 'models/candidates/temporal_tcn_v1.5-compression.pth')
    old=json.loads((ROOT/'models/candidates/video_fusion_v1.4.json').read_text('utf-8'))
    spec={**old,'features':FEATURES,'required_features':['generation','forensic_generation','temporal','localization','video_origin'],
      'estimator_type':'numeric_evidence_boosted_trees_v1','trees':trees,'baseline_logit':baseline,
      'scaler_mean':[0.]*len(FEATURES),'scaler_scale':[1.]*len(FEATURES),'coefficients':[0.]*len(FEATURES),'intercept':0.,
      'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),'threshold':threshold,
      'version':'1.5.0-source-tree-candidate','temporal_checkpoint_sha256':sha256(checkpoint),
      'video_origin_checkpoint_sha256':sha256(ROOT/'models/candidates/video_origin_v1.5.json'),
      'sample_ids':{s:[r['sample_id'] for r in part] for s,part in parts.items()},'calibration_metrics':cal_metrics,
      'threshold_metrics':m,'test_used':False,'selection':choice,'exported_raw_logit_max_difference':difference,
      'development_evidence_sha256':sha256(ROOT/'data/joint_video_source_development_v1.5-r02/development_evidence.json')}
    freeze(ROOT/'models/candidates/video_fusion_v1.5-trees.json',spec)
    passed=all(v['prior70_30_accuracy']>=.95 and v['genuine_fpr']<=.03 for v in (cal_metrics,m))
    atomic_json(ROOT/'runs/video_evidence_trees_v1.5.json',{'specification':spec,'chosen':choice,'trials':trials,
      'whole_policy_development_passed':passed,'temporal_checkpoint':checkpoint.relative_to(ROOT).as_posix(),
      'reserved_used':False,'live_config_modified':False,'formal_acceptance':False,
      'development_rows':[{'sample_id':r['sample_id'],'split':r['split'],'truth_label':r['truth_label'],'probability':float(p)} for r,p in zip(rows,probability(rows))]})
    print(json.dumps({'calibration':cal_metrics,'threshold':m,'threshold_value':threshold,'passed':passed,'export_delta':difference}),flush=True)


if __name__=='__main__':main()
