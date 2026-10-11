"""Freeze 100 untouched caption groups before any1.5 detector predictions."""
import json
import random
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.fetch_public_images as acquire
from scripts.fetch_image_expansion_14 import make_row
from backend.contracts import ROOT,sha256

OUTPUT=ROOT/'data/public_images_reserved_v1.5';SEED=10241503


def main():
    OUTPUT.mkdir(exist_ok=True);previous=[]
    paths=['data/public_images_standalone_v1/manifest.json','data/public_images_standalone_v1/training_expansion_manifest.json',
      'data/public_images_v1.4/manifest.json']
    for relative in paths:previous.extend(json.loads((ROOT/relative).read_text('utf-8')))
    metadata=ROOT/'data/public_images_v1.4'
    train=json.loads((metadata/'train_metadata_rows.json').read_text('utf-8'))
    test,_,receipt=acquire.cached_metadata_scan(OUTPUT,'test',[
      ('data/test-00002-of-00008.parquet',None),('data/test-00003-of-00008.parquet',None)])
    forbidden={r['source_group'] for r in previous}|set(train['all_caption_groups'])
    for filename in ('selection_registration.json','selection_registration_integrity_r01.json'):
        value=json.loads((metadata/filename).read_text('utf-8'))
        forbidden.update(r['source_group'] for r in value.get('selected_rows',[]))
    rng=random.Random(SEED);groups=sorted(set(test)-forbidden);rng.shuffle(groups)
    if len(groups)<100:raise ValueError('Insufficient wholly unused reserved source groups')
    labels=[0]*70+[1,2,3,4,5]*6;rng.shuffle(labels)
    selected=[make_row(group,test[group][label],label,'public_holdout') for group,label in zip(groups[:100],labels)]
    acquire.freeze(OUTPUT/'selection_registration.json',{'seed':SEED,'version':'1.5','selected_rows':selected,
      'prior_manifest_receipts':{p:sha256(ROOT/p) for p in paths},'test_metadata_receipt':receipt,
      'registered_before_new_pixels_or_any_predictions':True,'all_scanned_training_caption_groups_excluded':True,
      'selection':'fixed metadata seed,70 source-real and30 generated; no detector or quality filtering',
      'formal_business_truth_known':False})
    rows=acquire.acquire_image_bytes(OUTPUT,selected)
    if len(rows)!=100 or len({r['sha256'] for r in rows})!=100:raise ValueError('Reserved original count/duplicates differ')
    if {r['sha256'] for r in rows}&{r['sha256'] for r in previous}:raise ValueError('Prior exact source reused; stop before any predictions')
    if {r['source_group'] for r in rows}&forbidden:raise ValueError('Prior source groups overlap')
    acquire.freeze(OUTPUT/'manifest.json',rows)
    acquire.freeze(OUTPUT/'provenance.json',{'manifest_sha256':sha256(OUTPUT/'manifest.json'),
      'source_revision':acquire.REVISION,'prior_source_group_overlap':0,'prior_original_sha256_overlap':0,
      'source_real':70,'generated':30,'original_bytes_reencoded':False,'no_model_called':True,
      'source_labels_not_business_authenticity':True,'upstream_pretraining_overlap_known':False,'formal_acceptance':False})
    print('Frozen100 wholly unused original image groups for1.5 acceptance',flush=True)


if __name__=='__main__':main()
