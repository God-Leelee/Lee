"""Actual standalone-image training; only registered development images are used.

Usage (run extraction only when the inference GPU is available):
  .venv/Scripts/python.exe -X utf8 scripts/train_standalone_image_adapter_13.py --phase preflight
  .venv/Scripts/python.exe -X utf8 scripts/train_standalone_image_adapter_13.py --phase extract --device cuda
  .venv/Scripts/python.exe -X utf8 scripts/train_standalone_image_adapter_13.py --phase fit

The 100 public_holdout images/targets are never accessed here. A temporary
head-only development threshold is NOT the complete platform decision policy.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageOps
from scipy.optimize import OptimizeWarning
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from backend.contracts import ROOT,atomic_json,sha256
from backend.development_texture import texture_features,TEXTURE_SPEC
from backend.media import read_image
from backend.standalone_generation import FEATURE_SPEC, predict, transform_descriptor, validate_spec

DATA=ROOT/'data/public_images_standalone_v1'
WORK=DATA/'training_v1.3'
MODEL=ROOT/'models/candidates/public_image_adapter_v1.3.json'
REPORT=ROOT/'runs/public_image_adapter_training_v1.3.json'
CLIP_SHA='9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e'
WAVE_SHA='50d639049d928986ba7d69861a4fe4f3e7afbba1843e3e089cdf6b4748f53d5b'
REVISION='787334f7857fa54f29027a7f09c30e895ad486ef'
SEED=10241303
CS=[.1,1.,10.]
CANDIDATES=['clip_two_view','wave_two_view','clip_wave_equal','clip_wave_texture_weak']
BLOCKS={'clip_two_view':['clip_center','clip_global'],'wave_two_view':['wave_center','wave_global'],
        'clip_wave_equal':['clip_center','clip_global','wave_center','wave_global'],
        'clip_wave_texture_weak':['clip_center','clip_global','wave_center','wave_global','texture']}
warnings.filterwarnings('ignore',message='Unknown solver options: iprint',category=OptimizeWarning)
threadpool_limits(limits=4)


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')


def load_development(manifest,expansion):
    # Filter before examining any image path, target, source group or hash.
    base=json.loads(manifest.read_text('utf-8'))
    rows=[r for r in base if r['split'] in ('calibration','threshold')]
    del base
    sibling=json.loads(expansion.read_text('utf-8'))
    if any(r['split']!='fusion' for r in sibling):raise ValueError('Expansion contains non-fusion rows')
    rows=sibling+rows
    if len(rows)!=800 or len({r['sample_id'] for r in rows})!=800:raise ValueError('Need exactly 600 fusion + 100 calibration + 100 threshold images')
    seen_groups=set();seen_hashes=set()
    for split,size in [('fusion',600),('calibration',100),('threshold',100)]:
        part=[r for r in rows if r['split']==split];groups={r['source_group'] for r in part}
        if len(part)!=size or len(groups)!=100 or seen_groups&groups:raise ValueError('Registered development caption groups overlap or counts differ')
        seen_groups.update(groups)
        labels=Counter(int(r['official_label_b']) for r in part)
        expected=Counter({i:100 for i in range(6)}) if split=='fusion' else Counter({0:70,1:6,2:6,3:6,4:6,5:6})
        if labels!=expected:raise ValueError('Registered source stratification differs')
        if split=='fusion':
            by_group=defaultdict(list)
            for r in part:by_group[r['source_group']].append(int(r['official_label_b']))
            if any(sorted(v)!=list(range(6)) for v in by_group.values()):raise ValueError('Fusion caption siblings are incomplete or duplicated')
        for r in part:
            if r.get('source_revision')!=REVISION or int(r['official_label_a'])!=int(r['official_label_b']!=0):raise ValueError('Pinned source binary/generator annotation differs')
            if r['sha256'] in seen_hashes:raise ValueError('Exact development image duplicates')
            seen_hashes.add(r['sha256'])
            path=(DATA/r['relative_path']).resolve()
            if not path.is_relative_to(DATA.resolve()) or sha256(path)!=r['sha256']:raise ValueError('Original source image hash/path mismatch')
    frozen=[{k:r[k] for k in ['sample_id','source_group','split','sha256','relative_path','official_label_a','official_label_b','source_revision']} for r in rows]
    return rows,hashlib.sha256(canonical(frozen)).hexdigest()


def predeclare(development_sha):
    WORK.mkdir(parents=True,exist_ok=True)
    protocol={'seed':SEED,'candidates':CANDIDATES,'C_grid':CS,'development_manifest_sha256':development_sha,
        'fusion':'600 images / 100 caption groups; every six siblings remain in one CV fold',
        'candidate_and_C_selection':'3-fold stratified GROUP CV on fusion600: mean fold AUC; tie pooled AUC, smaller C, earlier declared candidate',
        'calibration':'Platt C=1 fitted only on calibration100; reject nonpositive slope',
        'threshold':'head-only threshold100: maximize full-denominator accuracy subject to non-AI FPR<=0.03; tie higher threshold',
        'support_gate':'none; WeChat-specific CLIP support gate does not apply to this separately measured public domain',
        'holdout_images_labels_predictions_consumed':False,'complete_platform_policy_measured':False,
        'scope':'registered public researcher shard sample, not insurance adjudication or formal contest acceptance'}
    destination=WORK/'training_protocol.json'
    if destination.exists():
        if json.loads(destination.read_text('utf-8'))!=protocol:raise ValueError('Frozen training protocol differs; create a new experiment version')
    else:atomic_json(destination,protocol)
    return protocol


def cache_valid(path,rows,development_sha,backbone_sha):
    if not path.exists():return False
    with np.load(path,allow_pickle=False) as value:
        if (value['development_manifest_sha256'].item()!=development_sha or value['checkpoint_sha256'].item()!=backbone_sha
            or value['sample_ids'].astype(str).tolist()!=[r['sample_id'] for r in rows]
            or value['source_hashes'].astype(str).tolist()!=[r['sha256'] for r in rows]):raise ValueError('Feature cache registration differs')
        if value['features'].shape!=(len(rows),1536) or not np.isfinite(value['features']).all():raise ValueError('Invalid frozen two-view cache')
    return True


def extract(rows,development_sha,device):
    import torch
    from torchvision import transforms
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no silent CPU substitution')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    clip_files=list((ROOT/'models/cache').glob('models--timm--vit_large_patch14_clip_224.openai/snapshots/*/open_clip_model.safetensors'))
    if len(clip_files)!=1 or sha256(clip_files[0])!=CLIP_SHA:raise ValueError('Expected one exact frozen CLIP checkpoint')
    wave_file=ROOT/'models/weights_dinov2_G4.ckpt'
    if sha256(wave_file)!=WAVE_SHA:raise ValueError('Frozen WaveRep checkpoint differs')
    began=time.perf_counter();timings={};native_match=None
    for kind,checkpoint in [('clip',CLIP_SHA),('wave',WAVE_SHA)]:
        target=WORK/(kind+'_features.npz')
        if cache_valid(target,rows,development_sha,checkpoint):
            print('Reusing exact frozen development cache',kind,flush=True);continue
        if kind=='clip':
            import open_clip
            net,_,_=open_clip.create_model_and_transforms('ViT-L-14',pretrained=str(clip_files[0]),device=device,
                precision='fp16' if device=='cuda' else 'fp32',force_quick_gelu=True)
            size=224;preprocess=transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),
                transforms.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])])
        else:
            import timm
            net=timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',num_classes=1,pretrained=False,img_size=504)
            release=torch.load(wave_file,map_location='cpu',weights_only=False)
            state={k[6:]:v for k,v in release['state_dict'].items() if k.startswith('model.')} if 'state_dict' in release else release
            net.load_state_dict(state,strict=True);net.to(device);del release,state
            size=504;preprocess=transforms.Compose([transforms.CenterCrop((504,504)),transforms.ToTensor(),
                transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        net.eval();features=[];native_logits=[];forward_times=[]
        for index,row in enumerate(rows):
            image,_=read_image(DATA/row['relative_path'],50000000)
            whole=ImageOps.pad(image,(size,size),method=Image.Resampling.BICUBIC,color=(127,127,127),centering=(.5,.5))
            batch=torch.stack([preprocess(image),preprocess(whole)]).to(device);started=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):
                if kind=='clip':vectors=net.encode_image(batch,normalize=False)
                else:
                    tokens=net.forward_features(batch);vectors=net.forward_head(tokens,pre_logits=True)
                    logits=net.forward_head(tokens).flatten();native_logits.append(logits.float().cpu().numpy())
                    if index==0:
                        native_match=float(torch.max(torch.abs(net(batch).flatten()-logits)).item())
                        if native_match>1e-5:raise ValueError('Wave feature/native head extraction disagrees')
            value=vectors.float().cpu().numpy()
            if value.shape!=(2,768) or not np.isfinite(value).all():raise ValueError('Frozen backbone feature dimensions/values differ')
            value/=np.maximum(np.linalg.norm(value,axis=1,keepdims=True),1e-8);features.append(value.reshape(-1))
            forward_times.append(time.perf_counter()-started)
            if index%50==0 or index==len(rows)-1:print(f'{kind} development features {index+1}/{len(rows)}',flush=True)
        temporary=target.with_suffix('.pending.npz')
        np.savez_compressed(temporary,features=np.asarray(features,np.float32),sample_ids=np.asarray([r['sample_id'] for r in rows]),
            source_hashes=np.asarray([r['sha256'] for r in rows]),development_manifest_sha256=development_sha,
            checkpoint_sha256=checkpoint,feature_spec=FEATURE_SPEC,device=device,
            native_logits=np.asarray(native_logits,np.float32) if native_logits else np.empty((0,2),np.float32))
        temporary.replace(target);timings[kind]={'median_forward_ms':float(np.median(forward_times)*1000),'maximum_forward_ms':float(max(forward_times)*1000)}
        del net,batch,vectors;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    texture_target=WORK/'texture_features.npz'
    if texture_target.exists():
        with np.load(texture_target,allow_pickle=False) as data:
            if data['development_manifest_sha256'].item()!=development_sha or data['sample_ids'].astype(str).tolist()!=[r['sample_id'] for r in rows]:raise ValueError('Texture cache registration differs')
            textures=data['features'].copy();names=data['feature_names'].astype(str).tolist()
    else:
        textures=[];names=None
        for index,row in enumerate(rows):
            image,_=read_image(DATA/row['relative_path'],50000000)
            value,feature_names=texture_features(image)
            if names is None:names=feature_names
            if names!=feature_names:raise ValueError('CPU texture feature order differs')
            textures.append(value)
            if index%50==0 or index==len(rows)-1:print(f'CPU texture development features {index+1}/{len(rows)}',flush=True)
        textures=np.asarray(textures,np.float64)
        temporary=texture_target.with_suffix('.pending.npz')
        np.savez_compressed(temporary,features=textures,feature_names=np.asarray(names),feature_spec=TEXTURE_SPEC,
            development_manifest_sha256=development_sha,sample_ids=np.asarray([r['sample_id'] for r in rows]))
        temporary.replace(texture_target)
    if textures.shape!=(len(rows),53) or not np.isfinite(textures).all():raise ValueError('Invalid CPU texture matrix')
    with np.load(WORK/'clip_features.npz',allow_pickle=False) as data:clip=data['features'].copy()
    with np.load(WORK/'wave_features.npz',allow_pickle=False) as data:wave=data['features'].copy()
    values=np.concatenate([clip,wave,textures],axis=1)
    temporary=WORK/'features.pending.npz';target=WORK/'features.npz'
    np.savez_compressed(temporary,features=values,sample_ids=np.asarray([r['sample_id'] for r in rows]),source_hashes=np.asarray([r['sha256'] for r in rows]),
        development_manifest_sha256=development_sha,clip_checkpoint_sha256=CLIP_SHA,wave_checkpoint_sha256=WAVE_SHA,
        feature_spec=FEATURE_SPEC,texture_spec=TEXTURE_SPEC,texture_feature_names=np.asarray(names))
    temporary.replace(target)
    receipt={'development_samples':len(rows),'caption_groups':len({r['source_group'] for r in rows}),'holdout_images_targets_consumed':False,'feature_cache_sha256':sha256(target),
        'development_manifest_sha256':development_sha,'clip_checkpoint_sha256':CLIP_SHA,'wave_checkpoint_sha256':WAVE_SHA,
        'device':device,'precision':'CUDA mixed FP16' if device=='cuda' else 'CPU FP32','feature_spec':FEATURE_SPEC,
        'view_adaptations':['official CLIP center224; no resize','CLIP full-image gray letterbox224','official Wave center504; no resize','Wave full-image gray letterbox504'],
        'Wave_native_head_max_difference':native_match,'timing_scope':'offline feature extraction, not HTTP end-to-end latency',
        'forward_timings':timings,'elapsed_seconds':time.perf_counter()-began,'runtime_config_modified':False}
    atomic_json(WORK/'features_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


def metrics(y,p,threshold):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=threshold
    tp=int(np.sum(pred&(y==1)));tn=int(np.sum(~pred&(y==0)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)))
    return {'samples':len(y),'ai':int(np.sum(y==1)),'non_ai':int(np.sum(y==0)),'tp':tp,'tn':tn,'fp':fp,'fn':fn,
        'abstentions':0,'accuracy_all_samples':float((pred==y).mean()),'non_ai_fpr':fp/max(1,int(np.sum(y==0))),
        'ai_recall':tp/max(1,int(np.sum(y==1))),'auc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))}


def make_blocks(name,training_features):
    blocks=[]
    for block in BLOCKS[name]:
        item={'name':block,'scale':.25 if block=='texture' else 1.,'dimensions':53 if block=='texture' else 768,
              'normalization':'standardized_l2' if block=='texture' else 'l2'}
        if block=='texture':
            scaler=StandardScaler().fit(training_features[:,3072:3125])
            item.update(mean=scaler.mean_.tolist(),std=scaler.scale_.tolist(),texture_spec=TEXTURE_SPEC)
        blocks.append(item)
    return blocks


def transform_matrix(blocks,features):return np.asarray([transform_descriptor({'feature_blocks':blocks},f) for f in features])


def fit(rows,development_sha):
    began=time.perf_counter();cache=WORK/'features.npz'
    with np.load(cache,allow_pickle=False) as data:
        if (data['development_manifest_sha256'].item()!=development_sha or data['feature_spec'].item()!=FEATURE_SPEC
            or data['clip_checkpoint_sha256'].item()!=CLIP_SHA or data['wave_checkpoint_sha256'].item()!=WAVE_SHA
            or data['sample_ids'].astype(str).tolist()!=[r['sample_id'] for r in rows]
            or data['source_hashes'].astype(str).tolist()!=[r['sha256'] for r in rows]):raise ValueError('Frozen extracted cache differs from registered originals')
        features=data['features'].copy()
    if features.shape!=(800,3125) or not np.isfinite(features).all():raise ValueError('Invalid frozen four-view/texture feature matrix')
    labels=np.asarray([int(r['official_label_a']) for r in rows]);groups=np.asarray([r['source_group'] for r in rows]);splits=np.asarray([r['split'] for r in rows])
    indices={s:np.where(splits==s)[0] for s in ['fusion','calibration','threshold']};tr,cal,th=(indices[s] for s in ['fusion','calibration','threshold'])
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(features[tr],labels[tr],groups[tr]))
    trials=[]
    for name in CANDIDATES:
        for c in CS:
            oof=np.zeros(len(tr));fold_metrics=[]
            for fold,(a,b) in enumerate(folds):
                left,right=tr[a],tr[b]
                if set(groups[left])&set(groups[right]):raise ValueError('Caption siblings leaked across CV folds')
                blocks=make_blocks(name,features[left]);xfit=transform_matrix(blocks,features[left]);xval=transform_matrix(blocks,features[right])
                head=LogisticRegression(C=c,class_weight='balanced',max_iter=2500,random_state=SEED).fit(xfit,labels[left])
                oof[b]=head.predict_proba(xval)[:,1]
                fold_metrics.append({'fold':fold,'fit_images':len(left),'validation_images':len(right),'fit_groups':len(set(groups[left])),
                    'validation_groups':len(set(groups[right])),'auc':float(roc_auc_score(labels[right],oof[b]))})
            item={'candidate':name,'C':c,'mean_fold_auc':float(np.mean([r['auc'] for r in fold_metrics])),
                'fold_auc_std':float(np.std([r['auc'] for r in fold_metrics])),'oof_metrics_at_uncalibrated_half':metrics(labels[tr],oof,.5),
                'folds':fold_metrics}
            trials.append(item);print(json.dumps({k:item[k] for k in ['candidate','C','mean_fold_auc','fold_auc_std']}),flush=True)
    choice=max(trials,key=lambda r:(r['mean_fold_auc'],r['oof_metrics_at_uncalibrated_half']['auc'],-r['C'],-CANDIDATES.index(r['candidate'])))
    blocks=make_blocks(choice['candidate'],features[tr]);x=transform_matrix(blocks,features)
    head=LogisticRegression(C=choice['C'],class_weight='balanced',max_iter=2500,random_state=SEED).fit(x[tr],labels[tr])
    raw=head.decision_function(x)
    platt=LogisticRegression(C=1.,max_iter=2000,random_state=SEED).fit(raw[cal,None],labels[cal])
    if platt.coef_[0,0]<=0:raise ValueError('Independent public calibration reverses model direction')
    probability=expit(raw*platt.coef_[0,0]+platt.intercept_[0])
    threshold_trials=[{'threshold':float(t),'metrics':metrics(labels[th],probability[th],float(t))} for t in np.unique(np.r_[probability[th],1.])]
    eligible=[r for r in threshold_trials if r['metrics']['non_ai_fpr']<=.03]
    if not eligible:raise ValueError('No temporary head threshold meets the declared development FPR policy')
    threshold=max(eligible,key=lambda r:(r['metrics']['accuracy_all_samples'],r['threshold']))
    spec={'version':'1.3.0-candidate','architecture':'frozen_clip_wave_multiblock_logistic_v1','feature_spec':FEATURE_SPEC,
        'candidate':choice['candidate'],'selected_C':choice['C'],'trained':True,'feature_blocks':blocks,
        'encoder_sha256':CLIP_SHA,'wave_checkpoint_sha256':WAVE_SHA,'coefficients':head.coef_[0].tolist(),'intercept':float(head.intercept_[0]),
        'platt_coefficient':float(platt.coef_[0,0]),'platt_intercept':float(platt.intercept_[0]),'threshold':threshold['threshold'],
        'threshold_scope':'temporary head-only threshold; complete ensemble/local-edit policy needs its own development freeze',
        'threshold_metrics':threshold['metrics'],'calibration_metrics':metrics(labels[cal],probability[cal],threshold['threshold']),
        'support_gate':None,'sample_ids':{s:[rows[i]['sample_id'] for i in idx] for s,idx in indices.items()},
        'source_groups':{s:sorted(set(groups[idx].tolist())) for s,idx in indices.items()},
        'development_manifest_sha256':development_sha,'feature_cache_sha256':sha256(cache),'source_revision':REVISION,
        'dataset_source':'https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset',
        'generator_families':['Stable Diffusion','DALL-E','Midjourney'],'holdout_used_for_fit':False,'holdout_used_for_candidate_selection':False,
        'runtime_deployed':False,'formal_acceptance':False,'scope':'public standalone AI/non-AI image origin; no insurance fraud or tamper-free truth',
        'labels_captions_metadata_used_as_live_inputs':False,'upstream_pretraining_overlap_known':False}
    validate_spec(spec,CLIP_SHA,WAVE_SHA)
    recovered=np.asarray([predict(spec,f) for f in features]);difference=float(np.max(np.abs(recovered-probability)))
    if difference>1e-10:raise ValueError('Serialized adapter does not reproduce fitted sklearn probabilities')
    if MODEL.exists():
        old=json.loads(MODEL.read_text('utf-8'))
        if old!=spec:raise ValueError('Frozen candidate differs; use a new candidate filename/version')
    else:atomic_json(MODEL,spec)
    result={'model_file':str(MODEL.relative_to(ROOT)),'model_sha256':sha256(MODEL),'trained':True,'runtime_config_modified':False,
        'candidate':choice['candidate'],'C':choice['C'],'development_manifest_sha256':development_sha,'feature_cache_sha256':sha256(cache),
        'holdout_images_labels_predictions_consumed':False,'formal_acceptance':False,'complete_platform_policy_measured':False,
        'candidate_C_trials':trials,'selected_train_cv':choice,'calibration_metrics':spec['calibration_metrics'],
        'head_only_threshold_metrics':spec['threshold_metrics'],'temporary_head_threshold':spec['threshold'],
        'threshold_trials':threshold_trials,'serialized_probability_max_difference':difference,
        'development_rows':[{'sample_id':r['sample_id'],'source_group':r['source_group'],'split':r['split'],
            'ai_label':int(labels[i]),'official_label_b':int(r['official_label_b']),'probability':float(probability[i])} for i,r in enumerate(rows)],
        'elapsed_seconds':time.perf_counter()-began,'limits':['Public benchmark origin labels, not insurance legitimacy/tamper-free adjudication.',
            'Candidate selection shares fusion CV folds; independent held-out evaluation remains necessary.',
            'Threshold scores measure the image-origin head only; strong local editing guards can alter complete-platform metrics.',
            'Public upstream backbone pretraining overlap cannot be excluded.']}
    atomic_json(REPORT,result)
    lines=['# 独立图片适配候选 1.3','',f"真实训练：600 张 / 100 个 caption 分组；独立校准 100 张；阈值开发 100 张。公共保留测试图像和标签未消费。",'',
        '| 候选 | C | 分组 CV 平均 AUC | 折间标准差 |','|---|---:|---:|---:|']
    lines += [f"| {r['candidate']} | {r['C']} | {r['mean_fold_auc']:.5f} | {r['fold_auc_std']:.5f} |" for r in trials]
    lines += ['',f"选中 `{choice['candidate']}` / C={choice['C']}。校准区正确率 {spec['calibration_metrics']['accuracy_all_samples']:.1%}；临时头阈值 {spec['threshold']:.6f} 下，阈值开发正确率 {threshold['metrics']['accuracy_all_samples']:.1%}，非 AI 误报 {threshold['metrics']['non_ai_fpr']:.1%}。",'',
        '这些是图像来源分类头的开发成绩。完整平台还须结合局部编辑/复制证据重新冻结策略，随后只做一次独立测试；本报告不声称比赛达标。', '',
        f"冻结模型 SHA-256：`{sha256(MODEL)}`。导出参数与原 sklearn 最大概率差 {difference:.3g}。"]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['model_file','model_sha256','candidate','C','calibration_metrics','head_only_threshold_metrics','serialized_probability_max_difference']},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['preflight','extract','fit','all'],required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--manifest',type=Path,default=DATA/'manifest.json')
    parser.add_argument('--expansion',type=Path,default=DATA/'training_expansion_manifest.json')
    args=parser.parse_args();rows,development_sha=load_development(args.manifest,args.expansion);protocol=predeclare(development_sha)
    print(json.dumps({'phase':args.phase,'development_images':len(rows),'development_caption_groups':len({r['source_group'] for r in rows}),
        'development_manifest_sha256':development_sha,'holdout_images_labels_predictions_consumed':False},ensure_ascii=False),flush=True)
    if args.phase=='preflight':
        atomic_json(ROOT/'runs/public_image_adapter_preflight_v1.3.json',{'passed':True,'development_images':len(rows),
            'development_manifest_sha256':development_sha,'protocol':protocol,'original_development_hashes_verified':True,'GPU_loaded':False});return
    if args.phase in ['extract','all']:extract(rows,development_sha,args.device)
    if args.phase in ['fit','all']:fit(rows,development_sha)


if __name__=='__main__':main()
