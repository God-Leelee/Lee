"""Frozen frequency-forensics teacher features from development images only.

No held-out media or label is consumed. This does not install a new live model.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageOps
import torch
import timm
from torchvision import transforms
from backend.contracts import ROOT,atomic_json,sha256
from backend.media import read_image

WORK=ROOT/'data/wechat/candidate_v1.3'
MODEL_HASH='50d639049d928986ba7d69861a4fe4f3e7afbba1843e3e089cdf6b4748f53d5b'
FEATURE_SPEC='waverep_g4_prelogits_center504_letterbox504_l2_v1'


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'data/wechat/training_v1.2/split_manifest.json'
    # The filtering expression does not examine the held-out target field.
    rows=[r for r in json.loads(manifest_path.read_text(encoding='utf-8')) if r['split'] in ['train','calibration','threshold']]
    assert len(rows)==99
    frozen=json.dumps(rows,sort_keys=True).encode('utf-8')
    import hashlib
    dev_hash=hashlib.sha256(frozen).hexdigest()
    target=WORK/'wave_features.npz'
    checkpoint=ROOT/'models/weights_dinov2_G4.ckpt';assert sha256(checkpoint)==MODEL_HASH
    if target.exists():
        cache=np.load(target,allow_pickle=False)
        assert cache['development_manifest_sha256'].item()==dev_hash and cache['model_sha256'].item()==MODEL_HASH
        print('Existing matching development teacher features preserved.');return
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',num_classes=1,pretrained=False,img_size=504)
    released=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state={k[6:]:v for k,v in released['state_dict'].items() if k.startswith('model.')} if 'state_dict' in released else released
    net.load_state_dict(state,strict=True);net.to(device).eval()
    del released,state
    preprocess=transforms.Compose([transforms.CenterCrop((504,504)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    features=[];logits=[];elapsed=[];native_match=None;began=time.perf_counter()
    for index,row in enumerate(rows):
        path=ROOT/'data/wechat/images'/row['relative_path'];assert sha256(path)==row['sha256']
        image,_=read_image(path,50000000)
        whole=ImageOps.pad(image,(504,504),method=Image.Resampling.BICUBIC,color=(127,127,127),centering=(.5,.5))
        batch=torch.stack([preprocess(image),preprocess(whole)]).to(device)
        start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'):
            tokens=net.forward_features(batch)
            feature=net.forward_head(tokens,pre_logits=True)
            logit=net.forward_head(tokens).flatten()
            if index==0:
                ordinary=net(batch).flatten()
                native_match=float(torch.max(torch.abs(ordinary-logit)).item())
                assert native_match<=1e-5
        vectors=feature.float().cpu().numpy()
        assert vectors.shape==(2,768)
        vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-8)
        features.append(vectors.reshape(-1));logits.append(logit.float().cpu().numpy());elapsed.append(time.perf_counter()-start)
        if index%10==0 or index==len(rows)-1:print(f'Development WaveRep features {index+1}/{len(rows)}',flush=True)
    temp=WORK/'wave_features.pending.npz'
    np.savez_compressed(temp,features=np.asarray(features,np.float32),native_logits=np.asarray(logits,np.float32),
        sample_ids=np.asarray([r['sample_id'] for r in rows]),source_hashes=np.asarray([r['sha256'] for r in rows]),
        development_manifest_sha256=dev_hash,model_sha256=MODEL_HASH,feature_spec=FEATURE_SPEC,
        device=device,precision='CUDA mixed FP16' if device=='cuda' else 'CPU FP32')
    temp.replace(target)
    atomic_json(WORK/'wave_features_receipt.json',{'features':str(target.relative_to(ROOT)),'count':len(rows),'source_groups':len({r['source_group'] for r in rows}),
        'source_manifest_sha256':dev_hash,'model_sha256':MODEL_HASH,'feature_sha256':sha256(target),'feature_spec':FEATURE_SPEC,
        'pool':net.global_pool,'heldout_used':False,'native_head_check_max_difference':native_match,
        'seconds':round(time.perf_counter()-began,3),'median_forward_ms':float(np.median(elapsed)*1000),
        'views':['official center crop 504','full image aspect-preserving letterbox 504'],
        'adaptation':'Frozen pre-classification feature extraction; no backbone fitting; letterbox is a recorded new prototype view'})
    print(json.dumps({'count':len(rows),'heldout_used':False,'seconds':round(time.perf_counter()-began,2),'cache_sha256':sha256(target)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
