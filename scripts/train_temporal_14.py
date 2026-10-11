"""New PTS/velocity/quality TCN with whole-video MIL. Test is never decoded."""
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from backend.contracts import ROOT, load_config, atomic_json, sha256
from backend.models import Models
from backend.media import sample_video
from backend.temporal_v2 import ARCHITECTURE, TemporalNetV2, bag_logit, frame_quality, validate_window
DATA=ROOT/'data/public_genvideo_v1.4';WORK=DATA/'temporal_features_v1.4';SEED=10241405


def main():
    WORK.mkdir(exist_ok=True);torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(4)
    all_rows=json.loads((DATA/'manifest.json').read_text('utf-8'));rows=[r for r in all_rows if r['split'] in ('train','calibration')];del all_rows
    if len(rows)!=300 or len({r['source_group'] for r in rows})!=300:raise ValueError('New temporal development counts differ')
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest();config=load_config()
    protocol={'architecture':ARCHITECTURE,'seed':SEED,'train_videos':200,'calibration_videos':100,'epochs':32,
      'development_sha256':digest,'supervision':'Whole full-real/full-generated video MIL bag; no window edit pseudo-GT',
      'quality':'actual sampled frame Laplacian variance and decoded size','test_consumed':False,
      'epoch_selection':'none; fixed32 before calibration','aggregation':'logsumexp temperature .5, normalized by window count'}
    path=WORK/'protocol.json'
    if path.exists() and json.loads(path.read_text('utf-8'))!=protocol:raise ValueError('Frozen TCN experiment differs')
    if not path.exists():atomic_json(path,protocol)
    models=Models(config);models.load(generation_only=True)
    if models.errors:raise RuntimeError(models.errors)
    examples=[]
    for index,row in enumerate(rows):
        source=DATA/row['relative_path'];target=WORK/(row['sample_id']+'.npz')
        if sha256(source)!=row['sha256']:raise ValueError('Original video hash mismatch')
        if not target.exists():
            frames,windows,_=sample_video(source,config);features,_,_=models.encode([f.image for f in frames]);qualities=np.asarray([frame_quality(f.image) for f in frames])
            values=np.asarray([features[w['frame_ids']] for w in windows],np.float32)
            masks=np.asarray([w['valid_mask'] for w in windows],bool);times=np.asarray([w['actual_timestamps'] for w in windows],np.float32)
            quality=np.asarray([qualities[w['frame_ids']] for w in windows],np.float32)
            for f,m,t,q in zip(values,masks,times,quality):validate_window(f,m,t,q)
            np.savez_compressed(target,features=values,masks=masks,times=times,quality=quality,
              source_sha256=row['sha256'],encoder_sha256=models.info['generation']['encoder_sha256'])
        with np.load(target,allow_pickle=False) as stored:
            if stored['source_sha256'].item()!=row['sha256'] or stored['encoder_sha256'].item()!=models.info['generation']['encoder_sha256']:raise ValueError('Cached temporal source/backbone differs')
            examples.append((stored['features'].copy(),stored['masks'].copy(),stored['times'].copy(),stored['quality'].copy(),int(row['truth_label']=='fake'),row))
        if index%20==0:print('PTS temporal development',index+1,len(rows),flush=True)
    encoder_sha=models.info['generation']['encoder_sha256'];del models;torch.cuda.empty_cache()
    device='cuda' if torch.cuda.is_available() else 'cpu';net=TemporalNetV2().to(device);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    train=[e for e in examples if e[-1]['split']=='train'];cal=[e for e in examples if e[-1]['split']=='calibration']
    def tensors(part):
        return [torch.as_tensor(np.asarray([e[i] for e in part]),device=device,dtype=torch.bool if i==1 else torch.float32) for i in range(5)]
    tx,tm,tt,tq,ty=tensors(train);cx,cm,ct,cq,cy=tensors(cal)
    def logits(x,m,t,q):
        b,w,f,d=x.shape
        return net(x.reshape(b*w,f,d),m.reshape(b*w,f),t.reshape(b*w,f),q.reshape(b*w,f,2)).reshape(b,w)
    history=[]
    for epoch in range(32):
        net.train();order=torch.randperm(len(train),device=device);losses=[]
        for start in range(0,len(order),32):
            idx=order[start:start+32];bag=bag_logit(logits(tx[idx],tm[idx],tt[idx],tq[idx]))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(bag,ty[idx])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.item()))
        history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses))})
        if epoch%4==0:print(history[-1],flush=True)
    net.eval()
    with torch.inference_mode():
        p=bag_logit(logits(cx,cm,ct,cq)).sigmoid().cpu().numpy()
        permutation=torch.tensor([0,4,1,6,2,7,3,5],device=device)
        shuffled=bag_logit(logits(cx[:,:,permutation],cm[:,:,permutation],ct,cq[:,:,permutation])).sigmoid().cpu().numpy()
        slow=bag_logit(logits(cx,cm,ct*2,cq)).sigmoid().cpu().numpy()
    y=cy.cpu().numpy();metrics={'calibration_video_auc':float(roc_auc_score(y,p)),
      'calibration_accuracy_at_half':float(np.mean((p>=.5)==y)),'shuffled_feature_order_mean_abs_change':float(np.mean(abs(p-shuffled))),
      'doubled_actual_time_mean_abs_change':float(np.mean(abs(p-slow))),'calibration_video_count':len(cal),'test_used':False}
    checkpoint={'architecture':ARCHITECTURE,'trained':True,'state_dict':{k:v.cpu() for k,v in net.state_dict().items()},
      'training_manifest_sha256':digest,'encoder_sha256':encoder_sha,'training_sample_ids':[e[-1]['sample_id'] for e in train],
      'calibration_sample_ids':[e[-1]['sample_id'] for e in cal],'seed':SEED,'epochs':32,'metrics':metrics,
      'labels':'Whole-video MIL; no edit interval/window truth inferred from source labels','bag_aggregation':'logsumexp_temperature_0.5',
      'score_kind':'uncalibrated_risk','training_source':'new source-isolated public videos; insurance truth unknown'}
    candidate=ROOT/'models/candidates/temporal_tcn_v1.4.pth'
    if candidate.exists():raise ValueError('Frozen TCN candidate exists; use a new version')
    torch.save(checkpoint,candidate)
    atomic_json(ROOT/'runs/temporal_training_v1.4.json',{'protocol':protocol,'metrics':metrics,'history':history,'checkpoint_sha256':sha256(candidate),
      'live_config_modified':False,'formal_acceptance':False,'calibration_rows':[{'sample_id':e[-1]['sample_id'],'label':int(y[i]),'bag_probability':float(p[i])} for i,e in enumerate(cal)]})
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
