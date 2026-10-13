"""Acquire registered public originals and reconstruct the exact research100.

Downloads from recorded publishers only. No private119 or gated documents.
Sources retain their original terms. Every original and final input is hashed.
"""
import argparse
import io
import json
from pathlib import Path
import re
import struct
import sys
import time
import zipfile
import zlib
import requests
import numpy as np
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT,sha256,atomic_json
from backend.media import read_image
from scripts import public_video_fixtures as vf

METADATA=ROOT/'docs/validation/public100'
SESSION=requests.Session()
SESSION.headers.update({'User-Agent':'CLEARTRACE/1.8 noncommercial research reproduction','Accept-Encoding':'identity'})
SESSION.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504])))


def inside(relative):
    target=(ROOT/relative).resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Unsafe source plan path')
    return target


def get_range(url,start,end):
    if not 0<=start<=end or end-start>128*1024*1024:raise ValueError('Range exceeds safety boundary')
    with SESSION.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=120,stream=True) as response:
        response.raise_for_status()
        header=response.headers.get('Content-Range','')
        if response.status_code!=206 or not header.startswith(f'bytes {start}-{end}/'):
            raise RuntimeError('Publisher does not serve verified byte ranges. Retry or provide --archive-dir with original ZIP files.')
        payload=response.content
        if len(payload)!=end-start+1:raise ValueError('Truncated range response')
        return payload,int(header.split('/')[-1])


class RemoteReader(io.RawIOBase):
    def __init__(self,url):self.url=url;_,self.size=get_range(url,0,0);self.position=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if not 0<=self.position<=self.size:raise ValueError('Invalid remote seek')
        return self.position
    def read(self,length=-1):
        length=self.size-self.position if length<0 else min(length,self.size-self.position)
        if length==0:return b''
        result,_=get_range(self.url,self.position,self.position+length-1);self.position+=length;return result


def member_bytes(row,archive_cache,archive_dir=None):
    url=row['source_url'];name=row['archive_member']
    if archive_dir:
        local=archive_dir/Path(url.split('?')[0]).name
        if local.exists():
            with zipfile.ZipFile(local) as archive:return archive.read(name)
    if url not in archive_cache:archive_cache[url]=zipfile.ZipFile(RemoteReader(url))
    info=archive_cache[url].getinfo(name)
    if info.file_size>100*1024*1024 or info.flag_bits&1:raise ValueError('Unexpected oversized or encrypted input')
    header,_=get_range(url,info.header_offset,info.header_offset+29)
    if header[:4]!=b'PK\x03\x04':raise ValueError('ZIP local header differs')
    n,e=struct.unpack_from('<HH',header,26);offset=info.header_offset+30+n+e
    raw,_=get_range(url,offset,offset+info.compress_size-1)
    payload=zlib.decompress(raw,-15) if info.compress_type==8 else raw if info.compress_type==0 else None
    if payload is None or len(payload)!=info.file_size or zlib.crc32(payload)!=info.CRC:raise ValueError('ZIP member CRC/size mismatch')
    return payload


def acquire(rows,archive_dir):
    archives={}
    try:
        for i,row in enumerate(rows):
            path=inside(row['workspace_relative_path'])
            if path.exists():
                if sha256(path)!=row['sha256']:raise ValueError('Existing input differs: '+row['sample_id'])
                continue
            if row.get('archive_member'):payload=member_bytes(row,archives,archive_dir)
            else:
                response=SESSION.get(row['source_url'],timeout=120);response.raise_for_status();payload=response.content
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(payload)
            if sha256(temp)!=row['sha256']:raise ValueError('Publisher input hash differs: '+row['sample_id'])
            temp.replace(path)
            print('Verified original',i+1,'/',len(rows),row['sample_id'],flush=True)
    finally:
        for archive in archives.values():archive.close()


def fixtures(registration,expected,controls):
    out=ROOT/'data/complete_suite_v1.8';out.mkdir(parents=True,exist_ok=True)
    folder=out/'fixtures';folder.mkdir(exist_ok=True);(folder/'parents').mkdir(exist_ok=True)
    Image.new('L',(640,480),0).save(folder/'empty_truth.png')
    target_by={r['sample_id']:r for r in expected}
    for row in registration['rows']:
        if 'host' not in row:continue
        target=target_by[row['sample_id']]
        path=inside(target['workspace_relative_path'])
        if path.exists() and sha256(path)==target['sha256']:continue
        name=row['sample_id'];host,donor=row['host'],row.get('donor')
        if row['material_type']=='image':
            def image_for(r):
                key=r.get('workspace_relative_path','data/public_images_reserved_v1.8/'+r['relative_path'])
                image,_=read_image(inside(key),50000000)
                return image.resize((640,480),Image.Resampling.LANCZOS)
            base=image_for(host);edited=base.copy();base.save(folder/(name+'_control.png'))
            source=image_for(donor) if donor else base
            sx,sy,w,h=row['source_box_xywh'];tx,ty,_,_=row['target_box_xywh']
            edited.paste(source.crop((sx,sy,sx+w,sy+h)),(tx,ty));edited.save(path)
            mask=np.zeros((480,640),np.uint8);mask[ty:ty+h,tx:tx+w]=255
            Image.fromarray(mask).save(folder/(name+'_truth.png'))
        else:
            source=ROOT/'data/public_genvideo_reserved_v1.8'
            decoded_host=vf.decode_parent(source,host);decoded_donor=vf.decode_parent(source,donor)
            for parent in [decoded_host,decoded_donor]:atomic_json(folder/'parents'/(parent['row']['sample_id']+'_timing.json'),parent['timing'])
            vf.PATCH_BOX=tuple(row['target_box_xywh']);vf.DONOR_BOX=tuple(row['source_box_xywh']);vf.REQUESTED_INTERVAL=tuple(row['edit_interval'])
            vf.create_fixture(folder,name,decoded_host,decoded_donor,'traditional_video_splice')
            vf.create_fixture(folder,name+'_control',decoded_host,None,'matched_reencoded_control')
        print('Reconstructed controlled operation',name,flush=True)
    for row in expected+controls:
        path=inside(row['workspace_relative_path'])
        if not path.exists() or sha256(path)!=row['sha256']:
            raise ValueError('Reconstructed bytes differ: '+row['sample_id']+'; use the pinned Pillow/PyAV Windows environment, do not silently replace test input.')
    for filename in ['manifest.json','matched_controls.json','selection_registration.json','scope.json','provenance.json']:
        atomic_json(out/filename,json.loads((METADATA/filename).read_text('utf8')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive-dir',type=Path)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    rows=json.loads((METADATA/'source_inputs.json').read_text('utf8'))['assets']
    registration=json.loads((METADATA/'selection_registration.json').read_text('utf8'))
    expected=json.loads((METADATA/'manifest.json').read_text('utf8'))
    controls=json.loads((METADATA/'matched_controls.json').read_text('utf8'))
    if not args.check_only:acquire(rows,args.archive_dir);fixtures(registration,expected,controls)
    for row in expected+controls:
        path=inside(row['workspace_relative_path'])
        if not path.exists() or sha256(path)!=row['sha256']:raise ValueError('Missing/different registered input '+row['sample_id'])
    receipt={'passed':True,'registered100':len(expected),'supplementary_controls':len(controls),
        'all_input_sha256_verified':True,'private_materials_downloaded':False,'source_terms_retained':True,
        'classification_not_run':True,'same_machine_data_reconstruction_only':True}
    (ROOT/'runs').mkdir(exist_ok=True);atomic_json(ROOT/'runs/public100_input_verification.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
