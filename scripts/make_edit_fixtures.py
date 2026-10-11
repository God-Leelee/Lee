"""Known operations and genuine public-parent controls for localization checks.

Not independent business acceptance. AI patch comes from a labelled generated
public video; it is a composite, not a claimed generative-inpainting tool run.
"""
import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image,ImageFilter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT,atomic_json,sha256


def frame(path):
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_image().convert('RGB').resize((640,480))


def main():
    source=ROOT/'data/public_genvideo_v1'
    rows=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    real=[r for r in rows if r['split']=='train' and r['truth_label']=='genuine']
    fake=[r for r in rows if r['split']=='train' and r['truth_label']=='fake']
    if len(real)<2 or len(fake)<1:raise ValueError('Need two real training parents and one generated training parent')
    a,b,g=[frame(source/r['relative_path']) for r in [real[0],real[1],fake[0]]]
    out=ROOT/'data/known_edits';out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def save(name,im,label,kind,mask=None,parents=None,pair=None):
        path=out/(name+'.png');im.save(path)
        maskname=None
        if mask is not None:maskname=name+'_truth.png';Image.fromarray(mask).save(out/maskname)
        manifest.append({'sample_id':name,'relative_path':path.name,'material_type':'image',
            'truth_label':label,'material_class':kind,'split':'engineering','generator_tool':'known_public_AI_patch_composite' if kind=='ai_patch_composite' else 'not_applicable',
            'source_group':(parents or [real[0]['sha256']])[0], 'parent_sha256':parents or [real[0]['sha256']],
            'sha256':sha256(path),'mask_path':maskname,'copy_pair':pair,
            'source_url':'public_genvideo_v1/manifest.json','scope':'known-operation engineering test, not contest acceptance'})
    save('genuine_parent',a,'genuine','genuine_public_frame')
    save('genuine_repeat_texture',Image.fromarray(np.tile(np.array([[0,255],[255,0]],np.uint8),(240,320))).convert('RGB'),'genuine','engineering_pattern',parents=['deterministic_checkerboard'])
    arr=np.asarray(a).copy();arr[90:250,380:540]=arr[40:200,40:200]
    mask=np.zeros((480,640),np.uint8);mask[90:250,380:540]=255
    save('copy_move_known',Image.fromarray(arr),'fake','traditional_copy_move',mask,pair={'region_a':[40,40,160,160],'region_b':[380,90,160,160],'direction':'known_in_fixture_only'})
    for name,parent,kind in [('splice_known',b,'traditional_splice'),('ai_patch_known',g,'ai_patch_composite')]:
        edited=a.copy();edited.paste(parent.crop((160,100,420,320)),(220,160));mask=np.zeros((480,640),np.uint8);mask[160:380,220:480]=255
        save(name,edited,'fake',kind,mask,parents=[real[0]['sha256'],real[1]['sha256'] if kind=='traditional_splice' else fake[0]['sha256']])
    edited=a.copy();region=edited.crop((150,130,380,330)).filter(ImageFilter.GaussianBlur(8));edited.paste(region,(150,130));mask=np.zeros((480,640),np.uint8);mask[130:330,150:380]=255
    save('retouch_known',edited,'fake','traditional_retouch',mask)
    atomic_json(out/'manifest.json',manifest)
    print('Known operation samples:',len(manifest))


if __name__=='__main__':main()
