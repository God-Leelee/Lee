"""Download only from recorded original upstream locations; register hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models'
OUT.mkdir(exist_ok=True)


def download(url, target):
    if target.exists():
        print('Already present:', target.name, flush=True)
        return
    temp = target.with_suffix(target.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'ZAForensics/1.1 research'})
    with urllib.request.urlopen(request, timeout=60) as response, temp.open('wb') as dest:
        total = 0
        while block := response.read(1024 * 1024):
            dest.write(block)
            total += len(block)
            if total % (50 * 1024 * 1024) == 0:
                print(target.name, total // 1048576, 'MiB', flush=True)
    temp.replace(target)


download('https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip', OUT / 'TruFor_weights.zip')
archive_path = OUT / 'TruFor_weights.zip'
if hashlib.md5(archive_path.read_bytes()).hexdigest() != '7bee48f3476c75616c3c5721ab256ff8':
    raise RuntimeError('TruFor original published MD5 does not match')
with zipfile.ZipFile(archive_path) as archive:
    matches = [name for name in archive.namelist() if name.endswith('trufor.pth.tar')]
    if len(matches) != 1:
        raise RuntimeError('Unexpected upstream archive structure')
    target = OUT / 'trufor.pth.tar'
    if not target.exists():
        with archive.open(matches[0]) as src, target.open('wb') as dst:
            while block := src.read(1024 * 1024):
                dst.write(block)
head = ROOT / 'third_party/UniversalFakeDetect/pretrained_weights/fc_weights.pth'
if not head.exists():
    raise RuntimeError('Clone the recorded UFD repository first')
(OUT / 'ufd_fc.pth').write_bytes(head.read_bytes())
records = []
clip_revision='18d0535469bb561bf468d76c1d73aa35156c922b'
clip_path=OUT/'cache/models--timm--vit_large_patch14_clip_224.openai/snapshots'/clip_revision/'open_clip_model.safetensors'
if not clip_path.exists():
    from huggingface_hub import hf_hub_download
    clip_path=Path(hf_hub_download('timm/vit_large_patch14_clip_224.openai','open_clip_model.safetensors',revision=clip_revision,cache_dir=str(OUT/'cache')))
clip_hash=hashlib.sha256()
with clip_path.open('rb') as clip_file:
    for block in iter(lambda:clip_file.read(1048576),b''):clip_hash.update(block)
if clip_hash.hexdigest()!='9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e':
    raise RuntimeError('Pinned OpenAI CLIP encoder hash mismatch')
download('https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/weights_dinov2_G4.ckpt', OUT / 'weights_dinov2_G4.ckpt')
if hashlib.md5((OUT / 'weights_dinov2_G4.ckpt').read_bytes()).hexdigest() != '8bf19e6f68a92bed600dd97fbed3f2cd':
    raise RuntimeError('WaveRep original published MD5 does not match')
(OUT / 'licenses').mkdir(exist_ok=True)
download('https://www.grip.unina.it/download/LICENSE_OPEN.txt', OUT / 'licenses/TruFor_LICENSE_OPEN.txt')
for file, url, scope in [
    (OUT / 'trufor.pth.tar', 'https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip', 'nonprofit research only; upstream terms retained'),
    (OUT / 'ufd_fc.pth', 'https://github.com/WisconsinAIVision/UniversalFakeDetect', 'MIT code; released research checkpoint'),
    (OUT / 'weights_dinov2_G4.ckpt', 'https://github.com/grip-unina/WaveRep-SyntheticVideoDetection', 'informational and nonprofit research only; upstream terms retained'),
    (clip_path,'https://huggingface.co/timm/vit_large_patch14_clip_224.openai/tree/'+clip_revision,'OpenAI CLIP encoder mirror; pinned upstream revision'),
]:
    h = hashlib.sha256()
    with file.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    records.append({'file': str(file.relative_to(OUT)), 'bytes': file.stat().st_size, 'sha256': h.hexdigest(), 'source': url, 'scope': scope})
(OUT / 'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print('Detector checkpoints downloaded and verified', flush=True)
