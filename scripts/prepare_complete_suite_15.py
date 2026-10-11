"""Fresh mixed100 registered before reserved inference; operation geometry varies.

70 source-real /30 generated or known edited. All old parent groups excluded
by source acquisitions. Nine matched controls are supplementary, not100 cases.
"""
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
import scripts.make_video_edit_fixtures as video_fixtures
from scripts.prepare_complete_suite_13 import shuffled,round_robin

OUTPUT=ROOT/'data/complete_suite_v1.5';SEED=10241505


def main():
    OUTPUT.mkdir(exist_ok=True);fixtures=OUTPUT/'fixtures';fixtures.mkdir(exist_ok=True);(fixtures/'parents').mkdir(exist_ok=True)
    path=OUTPUT/'manifest.json'
    if path.exists():
        rows=json.loads(path.read_text('utf-8'))
        if len(rows)!=100 or any(sha256(ROOT/r['workspace_relative_path'])!=r['sha256'] for r in rows):raise ValueError('Existing new suite changed')
        print('Fresh mixed100 registration and bytes preserved');return
    image_data=ROOT/'data/public_images_reserved_v1.5';video_data=ROOT/'data/public_genvideo_reserved_v1.5'
    images=[r for r in json.loads((image_data/'manifest.json').read_text('utf-8')) if r['split']=='public_holdout']
    videos=[r for r in json.loads((video_data/'manifest.json').read_text('utf-8')) if r['split']=='public_holdout']
    rng=random.Random(SEED);ir=shuffled([r for r in images if r['truth_label']=='genuine'],rng);ig=shuffled([r for r in images if r['truth_label']=='fake'],rng)
    vr=shuffled([r for r in videos if r['truth_label']=='genuine'],rng);vg=shuffled([r for r in videos if r['truth_label']=='fake'],rng)
    planned=[]
    for kind,reals,generated,data in [('image',ir[:35],round_robin(ig,9),image_data),('video',vr[:35],round_robin(vg,12),video_data)]:
        for row in reals+generated:
            planned.append({'sample_id':row['sample_id'],'material_type':kind,'truth_label':row['truth_label'],
              'material_class':'original_source_real' if row['truth_label']=='genuine' else 'fully_generated',
              'workspace_relative_path':(data/row['relative_path']).relative_to(ROOT).as_posix(),'sha256':row['sha256'],
              'generator_tool':row['generator_tool'],'source_groups':[row['source_group']]})
    for i in range(6):
        donor=None if i<3 else ir[41+i-3];host=ir[35+i]
        w,h=(rng.randint(128,192),rng.randint(128,192)) if donor is None else (rng.randint(200,288),rng.randint(160,240))
        src=[rng.randint(24,80),rng.randint(24,64),w,h] if donor is None else [rng.randint(100,180),rng.randint(48,100),w,h]
        dst=[rng.randint(350,430),rng.randint(80,240),w,h] if donor is None else [rng.randint(160,260),rng.randint(128,200),w,h]
        if dst[0]+w>640 or dst[1]+h>480:raise ValueError('Registered geometry out of bounds')
        planned.append({'sample_id':f'suite15_edit_image_{i+1:02d}','material_type':'image','truth_label':'fake',
          'material_class':'traditional_copy_move' if donor is None else 'traditional_splice','host':host,'donor':donor,
          'source_box_xywh':src,'target_box_xywh':dst,'generator_tool':'not_applicable',
          'source_groups':[host['source_group']]+([donor['source_group']] if donor else [])})
    for i in range(3):
        host,donor=vr[35+i],vr[38+i];w,h=rng.randint(224,280),rng.randint(176,232)
        planned.append({'sample_id':f'suite15_edit_video_{i+1:02d}','material_type':'video','truth_label':'fake',
          'material_class':'traditional_video_splice','host':host,'donor':donor,'generator_tool':'not_applicable',
          'target_box_xywh':[rng.randint(160,260),rng.randint(112,184),w,h],
          'source_box_xywh':[rng.randint(112,200),rng.randint(64,112),w,h],
          'edit_interval':[1.+i*.15,2.6+i*.1],'source_groups':[host['source_group'],donor['source_group']]})
    seen=set()
    for row in planned:
        if seen&set(row['source_groups']):raise ValueError('Case parent sources overlap')
        seen.update(row['source_groups'])
    if len(planned)!=100 or Counter(r['truth_label'] for r in planned)!={'genuine':70,'fake':30}:raise ValueError('Required case distribution differs')
    freeze(OUTPUT/'selection_registration.json',{'seed':SEED,'rows':planned,'image_manifest_sha256':sha256(image_data/'manifest.json'),
      'video_manifest_sha256':sha256(video_data/'manifest.json'),'registered_before_any_reserved_predictions':True,
      'operation_geometry_selection':'declared RNG; independent locations and sizes from training; no score or image quality selection',
      'distribution':{'real_image':35,'real_video':35,'generated_image':9,'generated_video':12,'traditional_image':6,'traditional_video':3},
      'scope':'Source-isolated public mixture with controlled operations, not verified insurance claims',
      'formal_business_truth_known':False,'no_test_feedback_available_when_selected':True})
    Image.new('L',(640,480),0).save(fixtures/'empty_truth.png')
    finished=[];controls=[]
    for row in planned:
        if 'host' not in row:finished.append(row);continue
        name=row['sample_id'];host,donor=row['host'],row['donor']
        if row['material_type']=='image':
            image,_=read_image(image_data/host['relative_path'],50000000);base=image.resize((640,480),Image.Resampling.LANCZOS);edited=base.copy()
            control=fixtures/(name+'_control.png');base.save(control)
            if donor:
                source,_=read_image(image_data/donor['relative_path'],50000000);source=source.resize((640,480),Image.Resampling.LANCZOS)
            else:source=base
            sx,sy,w,h=row['source_box_xywh'];tx,ty,_,_=row['target_box_xywh'];edited.paste(source.crop((sx,sy,sx+w,sy+h)),(tx,ty))
            mask=np.zeros((480,640),np.uint8);mask[ty:ty+h,tx:tx+w]=255
            destination=fixtures/(name+'.png');truth=fixtures/(name+'_truth.png');edited.save(destination);Image.fromarray(mask).save(truth)
            result={k:v for k,v in row.items() if k not in ('host','donor')}
            result.update(workspace_relative_path=destination.relative_to(ROOT).as_posix(),sha256=sha256(destination),
              mask_path=truth.name,mask_sha256=sha256(truth),width=640,height=480,parents=[host]+([donor] if donor else []))
            controls.append({'sample_id':name+'_control','material_type':'image','truth_label':'genuine','material_class':'matched_reencoded_control',
              'workspace_relative_path':control.relative_to(ROOT).as_posix(),'sha256':sha256(control),'generator_tool':'not_applicable',
              'source_groups':row['source_groups'],'mask_path':'empty_truth.png'})
        else:
            decoded_host=video_fixtures.decode_parent(video_data,host);decoded_donor=video_fixtures.decode_parent(video_data,donor)
            for parent in [decoded_host,decoded_donor]:atomic_json(fixtures/'parents'/(parent['row']['sample_id']+'_timing.json'),parent['timing'])
            video_fixtures.PATCH_BOX=tuple(row['target_box_xywh']);video_fixtures.DONOR_BOX=tuple(row['source_box_xywh']);video_fixtures.REQUESTED_INTERVAL=tuple(row['edit_interval'])
            result=video_fixtures.create_fixture(fixtures,name,decoded_host,decoded_donor,'traditional_video_splice')
            result.update(workspace_relative_path=(fixtures/result['relative_path']).relative_to(ROOT).as_posix(),source_groups=row['source_groups'],
              scope='Fresh reserved-parent controlled operation, actual original parent PTS retained')
            control=video_fixtures.create_fixture(fixtures,name+'_control',decoded_host,None,'matched_reencoded_control')
            control.update(workspace_relative_path=(fixtures/control['relative_path']).relative_to(ROOT).as_posix(),source_groups=row['source_groups']);controls.append(control)
        finished.append(result)
    freeze(path,finished);freeze(OUTPUT/'matched_controls.json',controls)
    freeze(OUTPUT/'provenance.json',{'registration_sha256':sha256(OUTPUT/'selection_registration.json'),'manifest_sha256':sha256(path),
      'cases':100,'supplementary_matched_controls':9,'case_parents_disjoint':True,'reserved_inference_used_to_select':False,
      'formal_acceptance':False,'business_truth_known':False,'known_localization_GT_cases':9,'generative_inpainting_GT_cases':0,
      'video_caption_pairing_known':False,'upstream_pretraining_overlap_known':False})
    print('Fresh100 mixed cases and9 matched controls registered and generated',flush=True)


if __name__=='__main__':main()
