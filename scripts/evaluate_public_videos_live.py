"""Fresh preselected source-video test through HTTP; no fitting or re-selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT, atomic_json, sha256
from scripts.benchmark import wilson
from scripts.evaluate_public_images_live import freeze


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='public100-independent-video-v1.3')
    parser.add_argument('--regression',action='store_true')
    args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:raise ValueError('Invalid local run name')
    data=ROOT/'data/public_genvideo_independent_v1'
    rows=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    provenance=json.loads((data/'provenance.json').read_text(encoding='utf-8'))
    assert len(rows)==100 and sum(r['truth_label']=='genuine' for r in rows)==70
    assert len({r['sample_id'] for r in rows})==100 and len({r['sha256'] for r in rows})==100
    assert provenance['exact_prior_sha_overlap']==0 and provenance['source_group_overlap']==0
    session=requests.Session();base='http://127.0.0.1:8765'
    health=session.get(base+'/api/health',timeout=15).json()
    assert health['status']=='ready' and health['version']=='1.3.0' and not health['model_errors']
    output=ROOT/'benchmark_results'/args.run_name;output.mkdir(parents=True,exist_ok=True)
    freeze(output/'frozen_manifest.json',rows);freeze(output/'frozen_configuration.json',health['configuration'])
    freeze(output/'frozen_model_manifest.json',health['models'])
    source_files=list((ROOT/'backend').glob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/evaluate_public_images_live.py']
    implementation={str(p.relative_to(ROOT)):sha256(p) for p in source_files}
    freeze(output/'frozen_implementation.json',implementation)
    records_path=output/'records.json'
    records=json.loads(records_path.read_text(encoding='utf-8')) if records_path.exists() else []
    previous={r['sample_id']:r for r in records};assert len(previous)==len(records)
    assert set(previous)<={r['sample_id'] for r in rows}
    for index,row in enumerate(rows):
        if row['sample_id'] in previous:assert previous[row['sample_id']]['sha256']==row['sha256'];continue
        media=(data/row['relative_path']).resolve();assert media.is_relative_to(data) and sha256(media)==row['sha256']
        record={k:row[k] for k in ['sample_id','truth_label','sha256','generator_tool','source_group']}
        began=time.perf_counter()
        try:
            with media.open('rb') as stream:
                response=session.post(base+'/api/cases',files={'files':('evaluation_input.mp4',stream,'video/mp4')},
                                      data={'title':'独立视频核验'},timeout=30)
            response.raise_for_status();case_id=response.json()['case_id'];deadline=time.perf_counter()+120
            while True:
                case=session.get(base+'/api/cases/'+case_id,timeout=15).json();material=case['materials'][0];result=material.get('result',{})
                if case['status'] in {'complete','partial','failed'} and 'api_total_ms' in result.get('timing',{}):break
                if time.perf_counter()>deadline:raise TimeoutError('Video terminal result timeout')
                time.sleep(.025)
            elapsed=(time.perf_counter()-began)*1000
            assert result['version']=='1.3.0' and result['sha256']==row['sha256']
            record.update({'case_id':case_id,'material_id':material['material_id'],'execution_status':result['execution_status'],
                           'prediction':result['fusion']['verdict'],'risk':result['fusion']['risk'],'latency_ms':elapsed,
                           'server_latency_ms':result['timing']['api_total_ms'],'within_target':elapsed<=5000,
                           'branch_scores':{b['branch_id']:b['score'] for b in result['branches']},
                           'branch_status':{b['branch_id']:b['status'] for b in result['branches']},
                           'media':result['media'],'fusion':result['fusion']})
            if result['execution_status']=='complete':
                prefix=f'/api/cases/{case_id}/materials/{material["material_id"]}'
                original=session.get(base+prefix+'/original',timeout=30);original.raise_for_status()
                assert hashlib.sha256(original.content).hexdigest()==row['sha256']
                frames=result['media']['frames'];assert frames and len(frames)<=32
                for frame in frames:
                    assert frame['pts'] is not None and math.isfinite(frame['timestamp_seconds'])
                    if frame.get('localization'):
                        local=frame['localization'];assert local['status']=='success'
                        assert local['details']['frame_id']==frame['frame_id']
                        assert math.isfinite(local['details']['timestamp_seconds'])
                        assert abs(local['details']['timestamp_seconds']-frame['timestamp_seconds'])<=1e-6
                selected={0,len(frames)//2,len(frames)-1}
                selected.update(i for i,f in enumerate(frames) if f.get('localization',{}).get('details',{}).get('resolution_tier')=='detailed_frame')
                for i in sorted(selected):
                    frame=frames[i];assets=[frame['preview']]
                    if frame.get('localization'):assets += [frame['localization']['details']['mask'],frame['localization']['details']['heatmap']]
                    for name in assets:session.get(base+prefix+'/assets/'+name,timeout=30).raise_for_status()
                for kind in ['json','csv','html']:session.get(base+f'/api/cases/{case_id}/report/{kind}',timeout=30).raise_for_status()
                record['original_pts_assets_reports_verified']=True
        except Exception as exc:
            record.update({'execution_status':'failed','prediction':'undetermined','risk':None,
                           'latency_ms':(time.perf_counter()-began)*1000,'error':f'{type(exc).__name__}: {exc}'})
        records.append(record);atomic_json(records_path,records)
        print(f'video {index+1}/100: {record["prediction"]}, {record["latency_ms"]/1000:.3f}s',flush=True)
    final=session.get(base+'/api/health',timeout=15).json()
    assert final['configuration']==health['configuration']
    assert all(sha256(ROOT/p)==digest for p,digest in implementation.items())
    valid=lambda r:r['execution_status']=='complete' and r['prediction'] in {'lower_risk','suspected_fake'}
    genuine=[r for r in records if r['truth_label']=='genuine'];fake=[r for r in records if r['truth_label']=='fake']
    tp=sum(valid(r) and r['prediction']=='suspected_fake' for r in fake);tn=sum(valid(r) and r['prediction']=='lower_risk' for r in genuine)
    fp=sum(valid(r) and r['prediction']=='suspected_fake' for r in genuine);fn=sum(valid(r) and r['prediction']=='lower_risk' for r in fake)
    latencies=[r['latency_ms'] for r in records];failed=sum(not valid(r) for r in records)
    metrics={'total':100,'genuine':70,'fake':30,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'failed':failed,
             'accuracy':(tp+tn)/100,'genuine_fpr':fp/70,'fake_recall':tp/30,
             'accuracy_wilson_95':wilson(tp+tn,100),'genuine_fpr_wilson_95':wilson(fp,70),
             'latency_p95_ms':float(np.percentile(latencies,95)),'latency_max_ms':max(latencies),
             'over_target':sum(t>5000 for t in latencies),'traceable_tools':sorted({r['generator_tool'] for r in fake})}
    summary={'status':'completed','formal_acceptance':False,'independent_accuracy_test':not args.regression,
             'performance_regression':args.regression,
             'dataset_name':args.run_name,'configuration_id':health['configuration']['configuration_id'],'version':'1.3.0',
             'metrics':metrics,'source':provenance,'configuration_sha256':sha256(output/'frozen_configuration.json'),
             'frozen_manifest_sha256':sha256(output/'frozen_manifest.json'),
             'acceptance_checks':{'accuracy_95':metrics['accuracy']>=.95,'genuine_fpr_3':metrics['genuine_fpr']<=.03,
                 'every_material_within_5_seconds':metrics['over_target']==0,'100_cases_70_30':True,
                 '3_generator_sources':len(metrics['traceable_tools'])>=3,'no_required_execution_failures':failed==0,
                 'image_and_video_scope':False,'contest_business_sample_scope':False,'spatial_temporal_ground_truth':False},
             'notes':'新父视频原字节公开专项测试；字幕父源关系与上游预训练交叉未知。失败和拒判进入分母，耗时含上传、排队及必要证据落盘。原件、PTS、预览掩码和三类报告另核查，不能代替真实局部编辑或保险业务验收。'}
    atomic_json(output/'summary.json',summary);print(json.dumps(metrics,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
