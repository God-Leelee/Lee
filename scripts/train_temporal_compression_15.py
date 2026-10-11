"""Paired, development-only transcoding augmentation for actual-PTS TCN.

Do not use any reserved inference or labels. The previous fusion-only candidate
is retained as a failed experiment; this candidate fixes the temporal network.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from bisect import bisect_right
from fractions import Fraction
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from backend.contracts import ROOT,load_config,sha256,atomic_json
from backend.media import sample_video
from backend.models import Models
from backend.temporal_v2 import TemporalNetV2,ARCHITECTURE,frame_quality,validate_window,bag_logit
from scripts.evaluate_public_images_live import freeze

DATA=ROOT/'data/public_genvideo_v1.4'
OLD=DATA/'temporal_features_v1.4'
WORK=ROOT/'data/temporal_compression_v1.5'
COMPRESSION=ROOT/'data/compression_development_v1.5'
JOINT=ROOT/'data/joint_video_development_v1.4-r02'
SEED=10241502


def augment(parent,codec):
    source=DATA/parent['relative_path'];name='temporal15_'+hashlib.sha256((parent['sample_id']+codec).encode()).hexdigest()[:24]
    destination=WORK/(name+('.avi' if codec=='mpeg4' else '.mp4'))
    if sha256(source)!=parent['sha256']:raise ValueError('Temporal parent hash changed')
    if not destination.exists():
        images=[];times=[]
        with av.open(str(source)) as container:
            stream=container.streams.video[0];first=None
            for frame in container.decode(stream):
                if frame.pts is None:raise ValueError('Missing source exposure time')
                stamp=float(frame.pts*frame.time_base);first=stamp if first is None else first
                image=frame.to_image().convert('RGB');factor=320/max(image.size)
                size=tuple(max(64,round(v*factor)//2*2) for v in image.size)
                images.append(image.resize(size));times.append(stamp-first)
                if stamp-first>=2.:break
            period=1/float(stream.average_rate) if stream.average_rate else float(np.median(np.diff(times)))
        duration=min(2.,times[-1]+period);count=max(8,int(duration*24))
        with av.open(str(destination),'w') as container:
            stream=container.add_stream(codec,rate=24);stream.width,stream.height=images[0].size
            stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2
            if codec=='libx264':stream.options={'crf':'36','preset':'ultrafast','bf':'0'}
            else:stream.bit_rate=200000
            for index in range(count):
                exposure=max(0,min(len(images)-1,bisect_right(times,index/24)-1))
                frame=av.VideoFrame.from_image(images[exposure]);frame.pts=index;frame.time_base=Fraction(1,24)
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        atomic_json(WORK/(name+'_parent_pts.json'),{'parent_sha256':parent['sha256'],'actual_parent_relative_pts':times,
          'source_period':period,'output_frames':count,'output_fps':24,'looped':False})
    return {'sample_id':name,'split':'train','truth_label':parent['truth_label'],'parent_sample_id':parent['sample_id'],
      'source_group':parent['source_group'],'codec':codec,'workspace_relative_path':destination.relative_to(ROOT).as_posix(),
      'sha256':sha256(destination)}


def main():
    WORK.mkdir(exist_ok=True);torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(4)
    originals=json.loads((DATA/'manifest.json').read_text('utf-8'))
    parents=[r for r in originals if r['split']=='train']
    if len(parents)!=200 or len({r['source_group'] for r in parents})!=200:raise ValueError('Temporal train source groups differ')
    protocol={'seed':SEED,'architecture':ARCHITECTURE,'epochs':48,'learning_rate':.001,'weight_decay':.01,
      'parents':[{'sample_id':r['sample_id'],'sha256':r['sha256'],'source_group':r['source_group']} for r in parents],
      'paired_augmentation':['mpeg4_avi_320_200kbps','h264_mp4_320_crf36'],'classes_receive_identical_codec_transforms':True,
      'temporal_supervision':'whole full-source real/generated video bag, no fabricated window edit truth',
      'epoch_selection':'fixed48 before any calibration predictions','quality_context_retained':True,
      'reserved1_4_used':False,'formal_acceptance':False}
    freeze(WORK/'protocol.json',protocol)
    augmented=[]
    for index,parent in enumerate(parents):
        augmented.extend(augment(parent,codec) for codec in ('mpeg4','libx264'))
        if index%20==0:print('Temporal paired parents',index+1,len(parents),flush=True)
    freeze(WORK/'training_augmentations.json',augmented)
    original_rows=[{**r,'workspace_relative_path':(DATA/r['relative_path']).relative_to(ROOT).as_posix()} for r in originals if r['split'] in ('train','calibration')]
    compression=json.loads((COMPRESSION/'manifest.json').read_text('utf-8'))
    joint=json.loads((JOINT/'manifest.json').read_text('utf-8'))
    rows={r['sample_id']:r for r in original_rows+augmented+joint+compression}
    config=load_config();freeze(WORK/'sampling_configuration.json',{k:config[k] for k in ('global_frames','continuous_windows','window_frames','window_step_seconds','max_visual_frames')})
    models=Models(config);models.load(generation_only=True)
    if models.errors:raise RuntimeError(models.errors)
    encoder=models.info['generation']['encoder_sha256'];features={}
    for index,row in enumerate(rows.values()):
        path=ROOT/row['workspace_relative_path']
        if sha256(path)!=row['sha256']:raise ValueError('Temporal development media changed')
        cache=WORK/(row['sample_id']+'.npz');old=OLD/(row['sample_id']+'.npz')
        stored_path=old if old.exists() else cache
        if not stored_path.exists():
            frames,windows,_=sample_video(path,config);values,_,_=models.encode([f.image for f in frames])
            qualities=np.asarray([frame_quality(f.image) for f in frames],np.float32)
            arrays={'features':np.asarray([values[w['frame_ids']] for w in windows],np.float32),
              'masks':np.asarray([w['valid_mask'] for w in windows],bool),
              'times':np.asarray([w['actual_timestamps'] for w in windows],np.float32),
              'quality':np.asarray([qualities[w['frame_ids']] for w in windows],np.float32),
              'source_sha256':row['sha256'],'encoder_sha256':encoder}
            for f,m,t,q in zip(*(arrays[k] for k in ('features','masks','times','quality'))):validate_window(f,m,t,q)
            np.savez_compressed(cache,**arrays)
        with np.load(stored_path,allow_pickle=False) as arrays:
            if arrays['source_sha256'].item()!=row['sha256'] or arrays['encoder_sha256'].item()!=encoder:raise ValueError('Temporal feature cache differs')
            features[row['sample_id']]=tuple(arrays[k].copy() for k in ('features','masks','times','quality'))
        if index%40==0:print('Temporal actual-PTS features',index+1,len(rows),flush=True)
    del models;torch.cuda.empty_cache()
    device='cuda' if torch.cuda.is_available() else 'cpu';net=TemporalNetV2().to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    train=[r for r in original_rows if r['split']=='train']+augmented
    calibration=[r for r in original_rows+compression if r['split']=='calibration']
    def tensors(part):
        return [torch.as_tensor(np.asarray([features[r['sample_id']][i] for r in part]),device=device,
                dtype=torch.bool if i==1 else torch.float32) for i in range(4)]+[torch.as_tensor([r['truth_label']=='fake' for r in part],device=device,dtype=torch.float32)]
    tx,tm,tt,tq,ty=tensors(train);cx,cm,ct,cq,cy=tensors(calibration)
    def logits(x,m,t,q):
        b,w,f,d=x.shape
        return net(x.reshape(b*w,f,d),m.reshape(b*w,f),t.reshape(b*w,f),q.reshape(b*w,f,2)).reshape(b,w)
    history=[]
    for epoch in range(protocol['epochs']):
        net.train();order=torch.randperm(len(train),device=device);losses=[]
        for start in range(0,len(order),32):
            idx=order[start:start+32];prediction=bag_logit(logits(tx[idx],tm[idx],tt[idx],tq[idx]))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(prediction,ty[idx])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(loss.item())
        history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses))})
        if epoch%8==0:print(history[-1],flush=True)
    net.eval();predictions={}
    with torch.inference_mode():
        p=bag_logit(logits(cx,cm,ct,cq)).sigmoid().cpu().numpy();y=cy.cpu().numpy()
        for row in joint+compression:
            x,m,t,q=(torch.as_tensor(v,device=device,dtype=torch.bool if i==1 else torch.float32) for i,v in enumerate(features[row['sample_id']]))
            window_logits=net(x,m,t,q);risks=window_logits.sigmoid().cpu().numpy()
            predictions[row['sample_id']]={'temporal_risk':float(bag_logit(window_logits).sigmoid().item()),
              'window_risks':risks.tolist(),'window_mean':float(risks.mean()),'window_disagreement':float(np.ptp(risks))}
    metrics={'calibration_videos':len(calibration),'calibration_auc':float(roc_auc_score(y,p)),
      'accuracy_at_half':float(np.mean((p>=.5)==y)),'genuine_fp_at_half':int(np.sum((p>=.5)&(y==0))),
      'fake_tp_at_half':int(np.sum((p>=.5)&(y==1))),'reserved_used':False}
    candidate=ROOT/'models/candidates/temporal_tcn_v1.5-compression.pth'
    if candidate.exists():raise ValueError('Frozen temporal candidate exists')
    checkpoint={'architecture':ARCHITECTURE,'trained':True,'state_dict':{k:v.cpu() for k,v in net.state_dict().items()},
      'encoder_sha256':encoder,'training_manifest_sha256':sha256(WORK/'training_augmentations.json'),
      'training_sample_ids':[r['sample_id'] for r in train],'calibration_sample_ids':[r['sample_id'] for r in calibration],
      'epochs':protocol['epochs'],'seed':SEED,'metrics':metrics,'labels':'Whole source-video MIL; original and paired nonsemantic transcodes',
      'score_kind':'uncalibrated_risk','bag_aggregation':'logsumexp_temperature_0.5','training_source':'Public development only; no reserved sources'}
    torch.save(checkpoint,candidate)
    freeze(WORK/'development_temporal_predictions.json',predictions)
    atomic_json(ROOT/'runs/temporal_compression_v1.5.json',{'protocol':protocol,'metrics':metrics,'history':history,
      'candidate_sha256':sha256(candidate),'live_config_modified':False,'reserved_used':False,
      'calibration_rows':[{'sample_id':r['sample_id'],'truth_label':r['truth_label'],'probability':float(prob)} for r,prob in zip(calibration,p)]})
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
