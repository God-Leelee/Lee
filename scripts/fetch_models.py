"""Fetch missing frozen runtime assets; never download competition materials."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.part')
    request=urllib.request.Request(url,headers={'User-Agent':'CLEARTRACE research preview'})
    with urllib.request.urlopen(request,timeout=120) as response,part.open('wb') as dest:shutil.copyfileobj(response,dest)
    part.replace(path)


def main():
    manifest=json.loads((ROOT/'models/runtime_assets.json').read_text('utf8'))
    for item in manifest['weights']:
        path=ROOT/item['path']
        if path.exists():
            if digest(path)!=item['sha256']:raise ValueError('Existing asset hash mismatch: '+item['path'])
            print('Verified',item['path'],flush=True);continue
        if path.name=='trufor.pth.tar':
            archive=ROOT/'models/TruFor_weights.zip';download(item['source'],archive)
            with zipfile.ZipFile(archive) as z:
                names=[n for n in z.namelist() if n.endswith('trufor.pth.tar')]
                if len(names)!=1:raise ValueError('Unexpected TruFor archive')
                with z.open(names[0]) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
        elif path.name=='open_clip_model.safetensors':
            cached=hf_hub_download('timm/vit_large_patch14_clip_224.openai','open_clip_model.safetensors',
                revision='18d0535469bb561bf468d76c1d73aa35156c922b',cache_dir=ROOT/'models/cache')
            if Path(cached).resolve()!=path.resolve():path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(cached,path)
        elif path.name=='commfor_224.safetensors':
            cached=hf_hub_download('OwensLab/commfor-model-224','model.safetensors',
                revision='26afc31e6b40c312c3fd42c05a758be62446215b')
            path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(cached,path)
        elif path.name=='ufd_fc.pth':
            download('https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/pretrained_weights/fc_weights.pth',path)
        elif path.name=='weights_dinov2_G4.ckpt':download(item['source'],path)
        else:raise FileNotFoundError('Download complete source release containing trained local head: '+item['path'])
        if digest(path)!=item['sha256']:raise ValueError('Downloaded asset hash mismatch: '+item['path'])
        print('Downloaded and verified',item['path'],flush=True)


if __name__=='__main__':main()
