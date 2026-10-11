"""Register a 100-case mixed suite BEFORE any reserved inference, then edit.

50 images / 50 videos, 70 original source-real / 30 generated or controlled
edits. Nine known edits have masks/PTS, with nine extra matched controls.
Controlled edits and public source labels cannot certify insurance truth.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT,atomic_json,sha256
from backend.media import read_image
from scripts.evaluate_public_images_live import freeze
from scripts.make_video_edit_fixtures import decode_parent,create_fixture


def shuffled(rows,rng):
    rows=sorted(rows,key=lambda r:r['sample_id']);rng.shuffle(rows);return rows


def round_robin(rows,count):
    tools=sorted({r['generator_tool'] for r in rows});chosen=[]
    for i in range(count):
        part=[r for r in rows if r['generator_tool']==tools[i%len(tools)]]
        chosen.append(part[i//len(tools)])
    return chosen


def main():
    output=ROOT/'data/complete_suite_v1.3';output.mkdir(parents=True,exist_ok=True)
    fixtures=output/'fixtures';fixtures.mkdir(exist_ok=True)
    registered_path=output/'selection_registration.json'
    if (output/'manifest.json').exists():
        existing=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        assert len(existing)==100 and all(sha256(ROOT/r['workspace_relative_path'])==r['sha256'] for r in existing)
        print('Existing suite preserved:',len(existing));return
    rng=random.Random(10241304)
    image_data=ROOT/'data/public_images_standalone_v1';video_data=ROOT/'data/public_genvideo_independent_v1'
    images=[r for r in json.loads((image_data/'manifest.json').read_text(encoding='utf-8')) if r['split']=='public_holdout']
    videos=json.loads((video_data/'manifest.json').read_text(encoding='utf-8'))
    ir=shuffled([r for r in images if r['truth_label']=='genuine'],rng)
    vr=shuffled([r for r in videos if r['truth_label']=='genuine'],rng)
    ig=shuffled([r for r in images if r['truth_label']=='fake'],rng)
    vg=shuffled([r for r in videos if r['truth_label']=='fake'],rng)
    planned=[]
    for kind,originals,generated,data in [('image',ir[:35],round_robin(ig,9),image_data),
                                           ('video',vr[:35],round_robin(vg,12),video_data)]:
        for row in originals+generated:
            planned.append({'sample_id':row['sample_id'],'material_type':kind,'truth_label':row['truth_label'],
                            'material_class':'original_source_real' if row['truth_label']=='genuine' else 'fully_generated',
                            'workspace_relative_path':str((data/row['relative_path']).relative_to(ROOT)).replace('\\','/'),
                            'generator_tool':row['generator_tool'],'sha256':row['sha256'],
                            'source_groups':[row['source_group']]})
    for i in range(6):
        host=ir[35+i];donor=None if i<3 else ir[41+i-3]
        planned.append({'sample_id':f'suite13_edit_image_{i+1:02d}','material_type':'image','truth_label':'fake',
                        'material_class':'traditional_copy_move' if donor is None else 'traditional_splice',
                        'host':host,'donor':donor,'generator_tool':'not_applicable',
                        'source_groups':[host['source_group']]+([donor['source_group']] if donor else [])})
    for i in range(3):
        host,donor=vr[35+i],vr[38+i]
        planned.append({'sample_id':f'suite13_edit_video_{i+1:02d}','material_type':'video','truth_label':'fake',
                        'material_class':'traditional_video_splice','host':host,'donor':donor,
                        'generator_tool':'not_applicable','source_groups':[host['source_group'],donor['source_group']]})
    assert len(planned)==100 and Counter(r['truth_label'] for r in planned)=={'genuine':70,'fake':30}
    seen=set()
    for row in planned:
        assert not seen&set(row['source_groups']);seen.update(row['source_groups'])
    freeze(registered_path,{'seed':10241304,'rows':planned,'configuration_sha256':sha256(ROOT/'config.json'),
        'sampling_before_reserved_predictions':True,'edits_registered_before_image_access':True,
        'distribution':{'real_image':35,'real_video':35,'generated_image':9,'generated_video':12,
                        'traditional_image_edit':6,'traditional_video_edit':3},
        'scope':'Public original sources plus controlled known operations, not business acceptance; image/video专项测试与此套件有父源相关性，不能合并为300个独立案件。'})
    assert all(not (ROOT/'benchmark_results'/run/'records.json').exists() for run in
               ['public100-standalone-image-v1.3','public100-independent-video-v1.3'])
    Image.new('L',(640,480),0).save(fixtures/'empty_truth.png')
    (fixtures/'parents').mkdir(exist_ok=True)
    controls=[];finished=[]
    for row in planned:
        if 'host' not in row:
            assert sha256(ROOT/row['workspace_relative_path'])==row['sha256'];finished.append(row);continue
        name=row['sample_id'];host=row['host'];donor=row['donor']
        if row['material_type']=='image':
            image,_=read_image(image_data/host['relative_path'],50000000)
            assert sha256(image_data/host['relative_path'])==host['sha256']
            base=image.resize((640,480),Image.Resampling.LANCZOS)
            control=fixtures/(name+'_control.png');base.save(control)
            edited=base.copy();mask=np.zeros((480,640),np.uint8)
            if donor is None:
                edited.paste(base.crop((40,40,200,200)),(380,90));mask[90:250,380:540]=255
            else:
                patch,_=read_image(image_data/donor['relative_path'],50000000)
                assert sha256(image_data/donor['relative_path'])==donor['sha256']
                patch=patch.resize((640,480),Image.Resampling.LANCZOS)
                edited.paste(patch.crop((160,100,420,320)),(220,160));mask[160:380,220:480]=255
            destination=fixtures/(name+'.png');edited.save(destination)
            truth=fixtures/(name+'_truth.png');Image.fromarray(mask).save(truth)
            result={k:v for k,v in row.items() if k not in ['host','donor']}
            result.update({'workspace_relative_path':str(destination.relative_to(ROOT)).replace('\\','/'),
                           'sha256':sha256(destination),'mask_path':truth.name,'mask_sha256':sha256(truth),
                           'width':640,'height':480,'parents':[host,donor] if donor else [host]})
            controls.append({'sample_id':name+'_control','workspace_relative_path':str(control.relative_to(ROOT)).replace('\\','/'),
                             'sha256':sha256(control),'material_type':'image','truth_label':'genuine',
                             'material_class':'resized_reencoded_control','generator_tool':'not_applicable',
                             'source_groups':row['source_groups'],'mask_path':'empty_truth.png'})
        else:
            decoded_host=decode_parent(video_data,host);decoded_donor=decode_parent(video_data,donor)
            for parent in [decoded_host,decoded_donor]:
                atomic_json(fixtures/'parents'/(parent['row']['sample_id']+'_timing.json'),parent['timing'])
            result=create_fixture(fixtures,name,decoded_host,decoded_donor,'traditional_video_splice')
            result.update({'workspace_relative_path':str((fixtures/result['relative_path']).relative_to(ROOT)).replace('\\','/'),
                           'source_groups':row['source_groups'],'scope':'New-source controlled operation; source split is reserved, not train',
                           'source_url':'public_genvideo_independent_v1/manifest.json'})
            control=create_fixture(fixtures,name+'_control',decoded_host,None,'matched_reencoded_control')
            control.update({'workspace_relative_path':str((fixtures/control['relative_path']).relative_to(ROOT)).replace('\\','/'),
                            'source_groups':row['source_groups'],'scope':'Matched resampling control, not an original capture'})
            controls.append(control)
        finished.append(result)
    atomic_json(output/'manifest.json',finished);atomic_json(output/'matched_controls.json',controls)
    atomic_json(output/'provenance.json',{'registered_before_any_reserved_predictions':True,
        'registration_sha256':sha256(registered_path),'manifest_sha256':sha256(output/'manifest.json'),
        'cases':100,'materials':100,'controls':9,'source_groups_disjoint_within_suite':True,
        'formal_acceptance':False,'known_edit_scope':'deterministic copy/splice only; no generative inpainting or insurance truth',
        'source_caption_pairing_for_videos':'unknown; archive parent IDs conservatively excluded',
        'correlated_with专项100image_and100video':True})
    print('Prepared registered100 mixed cases and9 matched controls',flush=True)


if __name__=='__main__':main()
