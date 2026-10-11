"""Complete per-material benchmark with failures, sources and independent scope.

Usage: python scripts/benchmark.py manifest.json --root media_dir --split public_holdout
Labels, summaries, source paths and tools never enter Engine.analyze().
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT, load_config, atomic_json, sha256
from backend.engine import Engine
from backend.models import Models


def wilson(k,n):
    if not n:return [None,None]
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    a=z*((p*(1-p)+z*z/(4*n))/n)**.5/d
    return [c-a,c+a]


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--split',default='public_holdout');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--run-name',default='public-development');p.add_argument('--freeze-public-100',action='store_true')
    a=p.parse_args();config=load_config()
    full=json.loads(a.manifest.read_text(encoding='utf-8'))
    rows=[r for r in full if r['split']==a.split]
    quarantine=ROOT/'data/leakage_quarantine.json'
    exclusions=json.loads(quarantine.read_text(encoding='utf-8')) if quarantine.exists() else []
    used={r['sample_id'] for r in exclusions}
    rows=[r for r in rows if r['sample_id'] not in used]
    if a.freeze_public_100:
        genuine=[r for r in rows if r['truth_label']=='genuine'][:70]
        fake=[r for r in rows if r['truth_label']=='fake']
        tools=sorted({r['generator_tool'] for r in fake})
        selected=[]
        for i in range(30):
            group=[r for r in fake if r['generator_tool']==tools[i%len(tools)]]
            if i//len(tools)>=len(group):raise ValueError('Not enough independently reserved fake sources')
            selected.append(group[i//len(tools)])
        rows=genuine+selected
        if len(genuine)!=70:raise ValueError('Not enough reserved genuine public videos for 70/30')
    if a.limit:rows=rows[:a.limit]
    if not rows:raise ValueError('Empty evaluation split')
    output=ROOT/'benchmark_results'/a.run_name;output.mkdir(parents=True,exist_ok=True)
    frozen=output/'frozen_manifest.json'
    manifest_text=json.dumps(rows,ensure_ascii=False,sort_keys=True,indent=2)
    if frozen.exists() and frozen.read_text(encoding='utf-8')!=manifest_text:
        raise ValueError('Frozen test manifest changed; use a new run name')
    frozen.write_text(manifest_text,encoding='utf-8')
    frozen_config=output/'frozen_configuration.json'
    config_text=json.dumps(config,ensure_ascii=False,sort_keys=True,indent=2)
    if frozen_config.exists() and frozen_config.read_text(encoding='utf-8')!=config_text:
        raise ValueError('Frozen evaluation configuration changed; use a new run name')
    frozen_config.write_text(config_text,encoding='utf-8')
    source_files=list((ROOT/'backend').glob('*.py'))+[ROOT/'scripts/benchmark.py']
    implementation={str(p.relative_to(ROOT)):sha256(p) for p in sorted(source_files)}
    implementation_path=output/'frozen_implementation.json'
    if implementation_path.exists() and json.loads(implementation_path.read_text(encoding='utf-8'))!=implementation:
        raise ValueError('Frozen implementation changed; use a new run name')
    atomic_json(implementation_path,implementation)
    models=Models(config);models.load();engine=Engine(config,models)
    records=[];tp=tn=fp=fn=failed=over=0;latencies=[]
    try:
        for index,row in enumerate(rows):
            media=a.root/row['relative_path']
            if sha256(media)!=row['sha256']:raise ValueError('Evaluation input hash mismatch: '+row['sample_id'])
            start=time.perf_counter()
            try:
                result=engine.analyze(media,output/'evidence'/row['sample_id'],row['sample_id'],material_type=row['material_type'],received_at=start,
                                      original_name='evaluation_input'+media.suffix)
            except Exception as exc:
                result={'execution_status':'failed','fusion':{'risk':None,'verdict':'undetermined'},
                        'branches':[],'error':f'{type(exc).__name__}: {exc}'}
            external_ms=(time.perf_counter()-start)*1000
            latencies.append(external_ms)
            predicted=result.get('fusion',{}).get('verdict')
            valid=result['execution_status']=='complete' and predicted in ['suspected_fake','lower_risk']
            true_fake=row['truth_label']=='fake'
            if not valid:failed+=1
            elif true_fake and predicted=='suspected_fake':tp+=1
            elif not true_fake and predicted=='lower_risk':tn+=1
            elif not true_fake:fp+=1
            else:fn+=1
            if external_ms>5000:over+=1
            record={'sample_id':row['sample_id'],'truth_label':row['truth_label'],'prediction':predicted,
                    'risk':result.get('fusion',{}).get('risk'),'execution_status':result['execution_status'],
                    'error':result.get('error'),
                    'latency_ms':external_ms,'within_target':external_ms<=5000,
                    'generator_tool':row['generator_tool'],'sha256':row['sha256'],
                    'branch_scores':{b['branch_id']:b['score'] for b in result.get('branches',[])},
                    'branch_status':{b['branch_id']:b['status'] for b in result.get('branches',[])}}
            records.append(record);atomic_json(output/'records.json',records)
            print(index+1,len(rows),row['sample_id'],row['truth_label'],predicted,round(external_ms),flush=True)
    finally:engine.cpu.shutdown(wait=True)
    n=len(rows);g=sum(r['truth_label']=='genuine' for r in rows);f=sum(r['truth_label']=='fake' for r in rows)
    tools=sorted({r['generator_tool'] for r in rows if r['truth_label']=='fake' and r['generator_tool'] not in ['unknown','not_applicable']})
    metrics={'total':n,'genuine':g,'fake':f,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'failed':failed,
             'accuracy':(tp+tn)/n,'genuine_fpr':fp/g if g else None,'accuracy_wilson_95':wilson(tp+tn,n),
             'genuine_fpr_wilson_95':wilson(fp,g),'fake_recall':tp/f if f else None,
             'latency_p50_ms':float(np.percentile(latencies,50)),'latency_p95_ms':float(np.percentile(latencies,95)),
             'latency_max_ms':max(latencies),'over_target':over,'traceable_tools':tools}
    checks={'accuracy_95':metrics['accuracy']>=.95,'genuine_fpr_3':g>0 and fp/g<=.03,'every_material_within_5_seconds':over==0,
            '100_cases_70_30':n==100 and g==70 and f==30,'3_generator_sources':len(tools)>=3,'no_required_execution_failures':failed==0,
            'contest_business_sample_scope':False,'image_and_video_scope':len({r['material_type'] for r in rows})==2,
            'spatial_temporal_ground_truth':False}
    if load_config()!=config or any(sha256(ROOT/p)!=digest for p,digest in implementation.items()):
        raise RuntimeError('Configuration or implementation changed during evaluation; records retained, summary not published')
    summary={'status':'completed','formal_acceptance':False,'dataset_name':a.run_name,'configuration_id':config['configuration_id'],
             'frozen_manifest_sha256':sha256(frozen),'configuration_sha256':sha256(frozen_config),
             'implementation_sha256':sha256(implementation_path),
             'metrics':metrics,'acceptance_checks':checks,'model_manifest':models.info,'quarantined_development_samples':exclusions,
             'notes':'公开开发保留集。失败保留在准确率分母，耗时是调用至完整证据持久化的外部计时；不能替代比赛保险材料与空间/时间定位验收。'}
    atomic_json(output/'summary.json',summary);atomic_json(ROOT/'benchmark_results/latest_summary.json',summary)
    print(json.dumps(summary['metrics'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
