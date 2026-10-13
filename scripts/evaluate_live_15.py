"""Complete HTTP measurements; no fitting or label/GT input reaches inference.

--suite mixed: fresh frozen100, then9 supplementary matched controls.
--suite formats:22 native format/codec fixtures; compatibility, no accuracy claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT,atomic_json,sha256
from scripts.evaluate_public_images_live import freeze
from scripts.benchmark import wilson
from scripts.evaluate_video_localization import load_gt,score_localization


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=['mixed','formats'],required=True)
    parser.add_argument('--version',default='1.5');parser.add_argument('--configuration-id',default='v1.5-compression-source-candidate-001')
    parser.add_argument('--base',default='http://127.0.0.1:8765');parser.add_argument('--output')
    args=parser.parse_args()
    data=ROOT/(f'data/complete_suite_v{args.version}' if args.suite=='mixed' else 'data/format_matrix_v1.4')
    rows=json.loads((data/'manifest.json').read_text('utf-8'))
    controls=json.loads((data/'matched_controls.json').read_text('utf-8')) if args.suite=='mixed' else []
    session=requests.Session();base=args.base;deadline=time.perf_counter()+180
    while True:
        try:
            health=session.get(base+'/api/health',timeout=5).json()
            if health['status']=='ready':break
        except requests.RequestException:pass
        if time.perf_counter()>deadline:raise TimeoutError('Local service not ready')
        time.sleep(.5)
    config=health['configuration']
    if config['configuration_id']!=args.configuration_id or health['model_errors']:raise ValueError('Unexpected frozen candidate or model error')
    output=ROOT/'benchmark_results'/(args.output or (f'public100-mixed-independent-v{args.version}-001' if args.suite=='mixed' else f'native-formats-v{args.version}-001'));output.mkdir(exist_ok=True)
    implementation={p.relative_to(ROOT).as_posix():sha256(p) for p in list((ROOT/'backend').glob('*.py'))+[Path(__file__),ROOT/'scripts/evaluate_video_localization.py']}
    freeze(output/'frozen_manifest.json',rows);freeze(output/'frozen_controls.json',controls)
    freeze(output/'frozen_configuration.json',config);freeze(output/'frozen_implementation.json',implementation)
    freeze(output/'protocol.json',{'purpose':'fresh mixed independent research gate' if args.suite=='mixed' else 'native format compatibility only',
      'inference_labels_or_GT_passed':False,'every_file_in_denominator':True,'source_sampling_before_predictions':True,
      'latency':'HTTP upload to complete persisted result observed by polling; report/asset download verification follows',
      'formal_business_acceptance':False,'business_truth_verified':False,'case_scope':'one material per case; multi-material case threshold not validated'})
    path=output/'records.json';records=json.loads(path.read_text('utf-8')) if path.exists() else [];known={r['sample_id']:r for r in records}
    controlpath=output/'controls.json';controlrecords=json.loads(controlpath.read_text('utf-8')) if controlpath.exists() else [];knowncontrols={r['sample_id']:r for r in controlrecords}

    def fresh(row):
        media=ROOT/row['workspace_relative_path']
        if sha256(media)!=row['sha256']:raise ValueError('Frozen input bytes changed')
        record={k:row[k] for k in ('sample_id','material_type','sha256')};record.update(truth_label=row.get('truth_label'),material_class=row.get('material_class'),generator_tool=row.get('generator_tool'))
        began=time.perf_counter()
        try:
            with media.open('rb') as stream:response=session.post(base+'/api/cases',files={'files':('待核验材料'+media.suffix,stream)},data={'title':'独立核验' if args.suite=='mixed' else '格式兼容验证'},timeout=30)
            response.raise_for_status();cid=response.json()['case_id'];case_deadline=time.perf_counter()+120
            while True:
                case=session.get(base+'/api/cases/'+cid,timeout=15).json();material=case['materials'][0];result=material.get('result',{})
                if case['status'] in ('complete','partial','failed') and result.get('timing',{}).get('api_total_ms') is not None:break
                if time.perf_counter()>case_deadline:raise TimeoutError('Persisted result timeout')
                time.sleep(.025)
            record.update(case_id=cid,material_id=material['material_id'],prediction=result['fusion']['verdict'],risk=result['fusion']['risk'],
              execution_status=result['execution_status'],latency_ms=(time.perf_counter()-began)*1000,server_latency_ms=result['timing']['api_total_ms'],
              branch_scores={b['branch_id']:b['score'] for b in result['branches']},fusion=result['fusion'])
            if result['sha256']!=row['sha256'] or result['configuration_id']!=config['configuration_id']:raise ValueError('Result source/config mismatch')
            prefix=f'/api/cases/{cid}/materials/{material["material_id"]}'
            original=session.get(base+prefix+'/original',timeout=30);original.raise_for_status()
            if hashlib.sha256(original.content).hexdigest()!=row['sha256']:raise ValueError('Original download bytes differ')
            for kind in ('json','csv','html'):session.get(base+f'/api/cases/{cid}/report/{kind}',timeout=30).raise_for_status()
            evidence=ROOT/'data/cases'/cid/material['material_id']/'evidence'
            if row['material_type']=='image':
                loc=next(b for b in result['branches'] if b['branch_id']=='localization')
                mask_path=evidence/loc['details']['mask'];pred=np.asarray(Image.open(mask_path).convert('L'))>0
                if pred.shape!=(result['media']['height'],result['media']['width']):raise ValueError('Image mask coordinates differ')
                for name in (result['media']['preview'],loc['details']['mask'],loc['details']['heatmap']):session.get(base+prefix+'/assets/'+name,timeout=15).raise_for_status()
                if row.get('mask_path'):
                    truth=np.asarray(Image.open(data/'fixtures'/row['mask_path']).convert('L'))>0
                    if pred.shape!=truth.shape:raise ValueError('Known operation GT dimensions differ')
                    union=int((pred|truth).sum());intersection=int((pred&truth).sum())
                    record['localization']={'iou':intersection/union if union else 1.,'f1':2*intersection/max(1,int(pred.sum()+truth.sum())),
                      'false_positive_area_ratio':float(pred.mean()) if not truth.any() else None}
            else:
                frame_rows=result['media']['frames'];seen_pts=set()
                for frame in frame_rows:
                    nested=frame['localization'];key=(frame['pts'],frame['time_base'])
                    if key in seen_pts or nested['status']!='success' or abs(nested['details']['timestamp_seconds']-frame['timestamp_seconds'])>1e-6:raise ValueError('Frame localization/PTS provenance differs')
                    seen_pts.add(key);mask=np.asarray(Image.open(evidence/nested['details']['mask']))
                    if mask.shape!=(frame['height'],frame['width']):raise ValueError('Video mask coordinates differ')
                record['sampled_frames']=len(frame_rows);record['pts_and_masks_verified']=True
                if row.get('frame_ground_truth_path'):
                    _,gt=load_gt(data/'fixtures',row,allowed_parent_splits=('public_holdout',))
                    record['localization']=score_localization(result,evidence,data/'fixtures',row,gt)
            record['original_and_reports_verified']=True
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}',validation_failed=True)
            if 'latency_ms' not in record:record['latency_ms']=(time.perf_counter()-began)*1000
        return record

    for row in rows:
        if row['sample_id'] not in known:
            record=fresh(row);records.append(record);known[row['sample_id']]=record;atomic_json(path,records)
            print(len(records),len(rows),row['material_type'],row.get('truth_label'),record.get('prediction'),round(record['latency_ms']),record.get('error',''),flush=True)
    for row in controls:
        if row['sample_id'] not in knowncontrols:
            record=fresh(row);controlrecords.append(record);knowncontrols[row['sample_id']]=record;atomic_json(controlpath,controlrecords)
            print('Matched control',len(controlrecords),len(controls),record.get('prediction'),round(record['latency_ms']),record.get('error',''),flush=True)
    if session.get(base+'/api/health',timeout=10).json()['configuration']!=config or any(sha256(ROOT/p)!=digest for p,digest in implementation.items()):raise RuntimeError('Candidate code/config changed during frozen measurement')
    latencies=[r['latency_ms'] for r in records];valid=lambda r:r.get('execution_status')=='complete' and not r.get('validation_failed') and r.get('prediction') in ('lower_risk','suspected_fake')
    summary={'status':'completed','configuration_id':config['configuration_id'],'formal_acceptance':False,'dataset_name':output.name,
      'frozen_manifest_sha256':sha256(output/'frozen_manifest.json'),'configuration_sha256':sha256(output/'frozen_configuration.json'),
      'implementation_sha256':sha256(output/'frozen_implementation.json'),'fresh_source_independent':args.suite=='mixed',
      'notes':'新来源公开材料与已知编辑操作的独立研究实测。来源真实标签不等于理赔合法。每套1份材料，未验证多材料案件阈值。未采样视频间隔保持未知。',
      'latency_max_ms':max(latencies),'latency_p95_ms':float(np.percentile(latencies,95)),
      'over_target':sum(t>5000 for t in latencies),'failed':sum(not valid(r) for r in records)}
    if args.suite=='mixed':
        tp=sum(valid(r) and r['truth_label']=='fake' and r['prediction']=='suspected_fake' for r in records)
        tn=sum(valid(r) and r['truth_label']=='genuine' and r['prediction']=='lower_risk' for r in records)
        fp=sum(valid(r) and r['truth_label']=='genuine' and r['prediction']=='suspected_fake' for r in records)
        fn=sum(valid(r) and r['truth_label']=='fake' and r['prediction']=='lower_risk' for r in records)
        metrics={'total':len(records),'genuine':70,'fake':30,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'failed':summary['failed'],
          'accuracy':(tp+tn)/len(records),'genuine_fpr':fp/70,'fake_recall':tp/30,'accuracy_wilson_95':wilson(tp+tn,len(records)),
          'genuine_fpr_wilson_95':wilson(fp,70),'latency_max_ms':summary['latency_max_ms'],'latency_p95_ms':summary['latency_p95_ms'],'over_target':summary['over_target']}
        tools=sorted({r['generator_tool'] for r in rows if r['generator_tool'] not in ('not_applicable',None)})
        checks={'100_cases_70_30':len(records)==100,'accuracy_95':metrics['accuracy']>=.95,'genuine_fpr_3':metrics['genuine_fpr']<=.03,
          'every_material_within_5_seconds':summary['over_target']==0,'no_required_or_evidence_failures':summary['failed']==0,
          'both_image_and_video':{r['material_type'] for r in rows}=={'image','video'},'at_least3_generator_families':len(tools)>=3,
          'traditional_edit_images_and_videos':sum(r['material_class'].startswith('traditional') for r in rows)==9,
          'original_and_reports_verified':all(r.get('original_and_reports_verified') for r in records)}
        summary.update(metrics=metrics,acceptance_checks=checks,research_joint_numeric_gate_passed=all(checks.values()),traceable_tools=tools,
          supplementary_controls={'total':len(controlrecords),'failed':sum(not valid(r) for r in controlrecords),
            'false_positives':sum(valid(r) and r['prediction']=='suspected_fake' for r in controlrecords)},
          business_acceptance_pending=True,generative_inpainting_GT_pending=True,multi_material_case_threshold_pending=True)
        print(json.dumps({'metrics':metrics,'checks':checks,'supplementary_controls':summary['supplementary_controls']},ensure_ascii=False),flush=True)
    else:
        summary.update(fixtures=len(records),complete_pipeline_and_evidence_verified=all(valid(r) for r in records),
          metrics={'total':len(records),'failed':summary['failed'],'latency_max_ms':summary['latency_max_ms'],'latency_p95_ms':summary['latency_p95_ms'],'over_target':summary['over_target']},
          no_accuracy_claim=True,notes='22份开发父源的真实格式编码文件；用于接口、解码、原件、坐标、PTS、掩码与报告兼容验证，不是22个独立真伪样本。')
        print(json.dumps({k:summary[k] for k in ('fixtures','complete_pipeline_and_evidence_verified','latency_max_ms','over_target','failed')}),flush=True)
    atomic_json(output/'summary.json',summary)


if __name__=='__main__':main()
