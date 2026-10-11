from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT,sha256,atomic_json

repositories=[]
for name in ['UniversalFakeDetect','TruFor','D3','WaveRep']:
    path=ROOT/'third_party'/name
    if not path.exists():continue
    commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(path),'remote','get-url','origin'],text=True).strip()
    licenses=[{'path':str(p.relative_to(path)),'sha256':sha256(p)} for p in path.rglob('LICENSE*') if p.is_file()]
    repositories.append({'name':name,'commit':commit,'source':remote,'licenses':licenses})
weights=[]
for path in sorted((ROOT/'models').rglob('*')):
    if path.is_file() and path.suffix in ['.pth','.ckpt','.bin','.safetensors'] or path.is_file() and path.name.endswith('.pth.tar'):
        weights.append({'file':str(path.relative_to(ROOT/'models')),'bytes':path.stat().st_size,'sha256':sha256(path)})
domain=ROOT/'models/wechat_generation_v1.2.json'
if domain.exists():
    weights.append({'file':str(domain.relative_to(ROOT/'models')),'bytes':domain.stat().st_size,'sha256':sha256(domain),
                    'architecture':'frozen_clip_two_view_logistic_v1','source':'user-authorized local resource labels; source images stay local'})
public_head=ROOT/'models/public_image_policy_v1.3.json'
if public_head.exists():
    weights.append({'file':str(public_head.relative_to(ROOT/'models')),'bytes':public_head.stat().st_size,'sha256':sha256(public_head),
                    'architecture':'frozen_clip_wave_multiblock_logistic_v1',
                    'source':'Defactify original developer images, pinned revision787334f7857fa54f29027a7f09c30e895ad486ef; public source probability, not insurance fraud'})
atomic_json(ROOT/'models/reproducibility_manifest.json',{'repositories':repositories,'weights':weights})
print('Recorded',len(repositories),'upstream commits and',len(weights),'actual checkpoint hashes')
