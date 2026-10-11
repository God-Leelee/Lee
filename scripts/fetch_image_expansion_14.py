"""Register source-isolated image development and NEW reserved sources first.

Reuse pinned original-byte acquisition code, without modifying1.3 manifests.
Never select originals from predictions. Old test groups are forbidden.
"""
from collections import Counter
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fetch_public_images as acquire
from backend.contracts import ROOT,atomic_json,sha256

OUTPUT=ROOT/'data/public_images_v1.4'
SEED=10241401

def make_row(parent,record,label,split):
    identity=acquire.h((record['parquet_member']+':'+str(record['row_index'])).encode())[:24]
    tool,version=acquire.TOOLS[label]
    return {'sample_id':'defactify_'+identity,'file_stem':identity,'material_type':'image',
      'truth_label':'genuine' if label==0 else 'fake','material_class':'original_source_real' if label==0 else 'fully_generated',
      'generator_tool':tool,'generator_version':version,'source_group':parent,'caption_sha256':parent,'split':split,
      'source_url':record['source_url'],'source_revision':acquire.REVISION,'original_split':record['original_split'],
      'parquet_member':record['parquet_member'],'parquet_row_index':record['row_index'],'row_group':record['row_group'],
      'row_in_group':record['row_in_group'],'official_label_a':int(label!=0),'official_label_b':label,
      'license':'Original source rights; no media redistribution','annotation_status':'published_source_labels_not_business_authenticity',
      'metadata_provenance':'pinned_dataset_original','scope':'public source development/reserved, not insurance truth'}

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    registration_path=OUTPUT/'selection_registration.json'
    old_path=ROOT/'data/public_images_standalone_v1/manifest.json'
    old=json.loads(old_path.read_text(encoding='utf-8'))
    forbidden={r['source_group'] for r in old}
    old_extra=json.loads((ROOT/'data/public_images_standalone_v1/training_expansion_manifest.json').read_text(encoding='utf-8'))
    forbidden.update(r['source_group'] for r in old_extra)
    old_hashes={r['sha256'] for r in old+old_extra}
    if registration_path.exists():
        registration=json.loads(registration_path.read_text(encoding='utf-8'))
        assert registration['seed']==SEED and registration['prior_manifest_sha256']==sha256(old_path)
        selected=registration['selected_rows']
    else:
        train,train_all,train_receipt=acquire.cached_metadata_scan(OUTPUT,'train',[('data/train-00000-of-00007.parquet',None),('data/train-00001-of-00007.parquet',None)])
        test,test_all,test_receipt=acquire.cached_metadata_scan(OUTPUT,'test',acquire.TEST_MEMBERS)
        rng=random.Random(SEED);available=sorted(set(train)-forbidden);rng.shuffle(available)
        reserved=sorted(set(test)-train_all-forbidden);rng.shuffle(reserved)
        if len(available)<700 or len(reserved)<100:raise ValueError(f'Insufficient new complete groups: {len(available)}/{len(reserved)}')
        parts={'fusion':available[:500],'calibration':available[500:600],'threshold':available[600:700],'public_holdout':reserved[:100]}
        selected=[]
        for split,parents in parts.items():
            if split=='fusion':
                for group in parents:
                    selected.extend(make_row(group,train[group][label],label,split) for label in range(6))
            else:
                assignments=[0]*70+[1,2,3,4,5]*6;rng.shuffle(assignments)
                for group,label in zip(parents,assignments):selected.append(make_row(group,(test if split=='public_holdout' else train)[group][label],label,split))
        registration={'version':'1.4','seed':SEED,'revision':acquire.REVISION,'prior_manifest_sha256':sha256(old_path),
          'selected_rows':selected,'development_images':3200,'train_groups':500,'calibration_groups':100,'threshold_groups':100,'new_reserved_groups':100,
          'prior_groups_excluded':len(forbidden),'source_group_overlap_with_prior':0,'registered_before_new_pixels_or_predictions':True,
          'train_receipt':train_receipt,'test_receipt':test_receipt,'no_test_based_selection':True,'source_selection':'fixed shuffled metadata only; whole six-label siblings in fusion',
          'formal_acceptance':False,'business_truth_known':False,'unselected_train_captions_excluded_from_reserved':True}
        acquire.freeze(registration_path,registration)
        print('Registered',len(selected),'new original bytes before inference',flush=True)
    rows=acquire.acquire_image_bytes(OUTPUT,selected)
    assert len(rows)==3300 and len({r['sample_id'] for r in rows})==3300 and len({r['sha256'] for r in rows})==3300
    assert not {r['sha256'] for r in rows}&old_hashes and not {r['source_group'] for r in rows}&forbidden
    groups={split:{r['source_group'] for r in rows if r['split']==split} for split in ['fusion','calibration','threshold','public_holdout']}
    assert all(a==b or not groups[a]&groups[b] for a in groups for b in groups)
    assert Counter(r['split'] for r in rows)=={'fusion':3000,'calibration':100,'threshold':100,'public_holdout':100}
    acquire.freeze(OUTPUT/'manifest.json',rows)
    atomic_json(OUTPUT/'provenance.json',{'registration_sha256':sha256(registration_path),'manifest_sha256':sha256(OUTPUT/'manifest.json'),
      'revision':acquire.REVISION,'original_bytes_reencoded':False,'unique_originals':3300,'prior_group_overlap':0,'prior_exact_hash_overlap':0,
      'development_vs_reserved_source_overlap':0,'source_labels_not_insurance_truth':True,'upstream_pretraining_overlap_known':False,
      'generator_families':['Stable Diffusion','DALL-E','Midjourney'],'caption_or_tool_used_as_inference_input':False,'formal_acceptance':False})
    print('Verified3300 new original bytes, all source groups isolated',flush=True)

if __name__=='__main__':main()
