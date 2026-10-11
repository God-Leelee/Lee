"""600 new originals, disjoint old sources and disjoint new development/test.

Archive metadata selection occurs before decoding or detector predictions.
Existing Range/CRC/SHA and mutable-source fingerprint verification are reused.
"""
from collections import Counter
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fetch_independent_videos_13 as base
from backend.contracts import ROOT

OUTPUT=ROOT/'data/public_genvideo_v1.4'
SEED=10241404
PARTITIONS={'train':(100,100),'fusion':(70,30),'calibration':(70,30),'threshold':(70,30),'public_holdout':(70,30)}
TOTAL=sum(sum(counts) for counts in PARTITIONS.values())


def register(snapshot,items,old_members,old_suffixes,receipts):
    rng=random.Random(SEED);tools=('real_MSRVTT',)+base.TOOLS
    sources={tool:[] for tool in tools}
    for item in items:
        tool=base.source_tool(item.filename)
        if tool is None or item.filename in old_members:continue
        if tool!='real_MSRVTT' and (base.suffix(item.filename) is None or base.suffix(item.filename) in old_suffixes):continue
        sources[tool].append(item)
    for group in sources.values():group.sort(key=lambda i:i.filename);rng.shuffle(group)
    selected=[];used_fake_ids=set()
    for split,(nr,nf) in PARTITIONS.items():
        if len(sources['real_MSRVTT'])<nr:raise ValueError('Insufficient unused real sources')
        selected.extend((sources['real_MSRVTT'].pop(),'real_MSRVTT',split) for _ in range(nr))
        collected=0
        while collected<nf:
            advanced=False
            for tool in base.TOOLS:
                while sources[tool] and base.suffix(sources[tool][-1].filename) in used_fake_ids:sources[tool].pop()
                if not sources[tool]:continue
                item=sources[tool].pop();used_fake_ids.add(base.suffix(item.filename));selected.append((item,tool,split));collected+=1;advanced=True
                if collected==nf:break
            if not advanced:raise ValueError('Insufficient conservatively isolated fake IDs')
    rows=[]
    for item,tool,split in selected:
        identity=base.digest(item.filename.encode())[:24];real=tool=='real_MSRVTT'
        rows.append({'sample_id':'genvideo14_'+identity,'relative_path':identity+'.mp4','material_type':'video',
          'truth_label':'genuine' if real else 'fake','material_class':'original_source_real' if real else 'fully_generated',
          'generator_tool':'not_applicable' if real else tool,'generator_version':'not_applicable' if real else 'unknown_original_label',
          'split':split,'source_group':item.filename,'archive_member':item.filename,
          'conservative_fake_numeric_id':None if real else base.suffix(item.filename),'source_url':base.URL,
          'uncompressed_bytes':item.file_size,'compressed_bytes':item.compress_size,'archive_crc32':f'{item.CRC:08x}',
          'zip_compression_method':item.compress_type,'source_caption_or_prompt_pairing_known':False,
          'spatial_temporal_localization_ground_truth':'unknown','labels_or_filename_used_as_live_detector_inputs':False,
          'license':'Original GenVideo/MSRVTT source conditions; media not redistributed','scope':base.SCOPE})
    result={'version':'1.4','seed':SEED,'archive_snapshot':snapshot,'prior_source_manifests':receipts,'selected_rows':rows,
      'rows':TOTAL,'truth_counts':dict(Counter(r['truth_label'] for r in rows)),
      'generator_counts':dict(Counter(r['generator_tool'] for r in rows if r['truth_label']=='fake')),
      'partitions':PARTITIONS,'excluded_prior_archive_members':len(old_members),'excluded_prior_fake_numeric_ids':len(old_suffixes),
      'selection_rule':'Fixed seeded metadata shuffle, declared split/class counts; fake suffix excluded globally across tools and all splits',
      'sampling_before_model_prediction_or_decode':True,'development_test_overlap':0,'formal_contest_acceptance':False}
    base.freeze(OUTPUT/'selection_registration.json',result);return result


def verify(rows,registration,old_members,old_hashes,old_suffixes):
    if len(rows)!=TOTAL or len({r['source_group'] for r in rows})!=TOTAL or len({r['sha256'] for r in rows})!=TOTAL:raise ValueError('New original video count/group/hash differs')
    if {r['source_group'] for r in rows}&old_members or {r['sha256'] for r in rows}&old_hashes:raise ValueError('Video original reused from prior data')
    fake_ids=[r['conservative_fake_numeric_id'] for r in rows if r['truth_label']=='fake']
    if len(fake_ids)!=len(set(fake_ids)) or set(fake_ids)&old_suffixes:raise ValueError('Conservative source IDs overlap')
    selected={r['sample_id']:r for r in registration['selected_rows']}
    for split,(nr,nf) in PARTITIONS.items():
        if Counter(r['truth_label'] for r in rows if r['split']==split)!=Counter({'genuine':nr,'fake':nf}):raise ValueError('New video partition differs')
    for r in rows:
        if any(r[k]!=v for k,v in selected[r['sample_id']].items()):raise ValueError('Registration differs')
        data=(OUTPUT/r['relative_path']).read_bytes()
        if base.digest(data)!=r['sha256'] or len(data)!=r['bytes'] or len(data)!=r['uncompressed_bytes']:raise ValueError('Original source bytes differ')
        if f'{base.zlib.crc32(data)&0xffffffff:08x}'!=r['archive_crc32']:raise ValueError('Original ZIP CRC differs')


if __name__=='__main__':
    base.OUTPUT=OUTPUT;base.SEED=SEED;base.TOTAL_ROWS=TOTAL
    base.SCOPE='New public video development/reserved originals; source labels do not establish insurance truth'
    base.PROVENANCE_INPUTS=base.PROVENANCE_INPUTS+('data/public_genvideo_independent_v1/manifest.json','data/complete_suite_v1.3/manifest.json')
    base.register=register;base.verify=verify
    base.main()
