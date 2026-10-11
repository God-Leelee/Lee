"""Frozen mixed regression: reuse91 actual HTTP results, test9 edits+9 controls.

Correlated with public image/video专项测试, never a third independent100.
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
from scripts.evaluate_video_localization import load_gt,score_localization
from scripts.benchmark import wilson


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--delivery-regression',action='store_true')
    args=parser.parse_args()
    data=ROOT/'data/complete_suite_v1.3';fixtures=data/'fixtures'
    rows=read(data/'manifest.json');controls=read(data/'matched_controls.json')
    suffix='public100-complete-suite-delivery-regression-v1.3' if args.delivery_regression else 'public100-complete-suite-v1.3'
    output=ROOT/'benchmark_results'/suffix;output.mkdir(exist_ok=True)
    session=requests.Session();base='http://127.0.0.1:8765'
    health=session.get(base+'/api/health',timeout=15).json();config=health['configuration']
    assert health['status']=='ready' and config['version']=='1.3.0'
    assert len(rows)==100 and sum(r['truth_label']=='genuine' for r in rows)==70
    seen=set()
    for row in rows:
        assert not seen & set(row['source_groups']);seen.update(row['source_groups'])
        assert sha256(ROOT/row['workspace_relative_path'])==row['sha256']
    freeze(output/'frozen_manifest.json',rows);freeze(output/'frozen_controls.json',controls)
    freeze(output/'frozen_configuration.json',config)
    implementation={p.relative_to(ROOT).as_posix():sha256(p) for p in list((ROOT/'backend').glob('*.py'))+[Path(__file__),ROOT/'scripts/evaluate_video_localization.py']}
    freeze(output/'frozen_implementation.json',implementation)
    reuse={};references={}
    runs=['public100-image-final-performance-v1.3','public100-video-final-performance-v1.3'] if args.delivery_regression else ['public100-image-delivery-regression-v1.3','public100-independent-video-v1.3']
    for run in runs:
        directory=ROOT/'benchmark_results'/run
        assert read(directory/'frozen_configuration.json')==config
        summary=read(directory/'summary.json');assert summary['status']=='completed'
        references[run]=sha256(directory/'records.json')
        for record in read(directory/'records.json'):
            assert record['sample_id'] not in reuse
            reuse[record['sample_id']]={**record,'measurement_reused_from':run}
    freeze(output/'reused_receipts.json',references)
    path=output/'records.json';records=read(path) if path.exists() else []
    measured={r['sample_id']:r for r in records}
    controlpath=output/'controls.json';controlrecords=read(controlpath) if controlpath.exists() else []
    measuredcontrols={r['sample_id']:r for r in controlrecords}

    def fresh(row):
        media=ROOT/row['workspace_relative_path'];assert sha256(media)==row['sha256']
        record={k:row[k] for k in ['sample_id','sha256','truth_label','material_type','material_class']}
        began=time.perf_counter()
        try:
            with media.open('rb') as stream:
                response=session.post(base+'/api/cases',files={'files':('evaluation_input'+media.suffix,stream)},data={'title':'材料核验 · 综合回归'},timeout=30)
            response.raise_for_status();case_id=response.json()['case_id'];deadline=time.perf_counter()+120
            while True:
                case=session.get(base+'/api/cases/'+case_id,timeout=15).json();material=case['materials'][0];result=material.get('result',{})
                if case['status'] in {'complete','partial','failed'} and 'api_total_ms' in result.get('timing',{}):break
                if time.perf_counter()>deadline:raise TimeoutError('Persisted result timeout')
                time.sleep(.025)
            elapsed=(time.perf_counter()-began)*1000
            assert result['sha256']==row['sha256'] and result['configuration_id']==config['configuration_id']
            record.update({'case_id':case_id,'material_id':material['material_id'],'prediction':result['fusion']['verdict'],'risk':result['fusion']['risk'],
                           'execution_status':result['execution_status'],'latency_ms':elapsed,'server_latency_ms':result['timing']['api_total_ms']})
            assert result['execution_status']=='complete'
            prefix=f'/api/cases/{case_id}/materials/{material["material_id"]}'
            original=session.get(base+prefix+'/original',timeout=30);original.raise_for_status()
            assert hashlib.sha256(original.content).hexdigest()==row['sha256']
            for kind in ['json','csv','html']:session.get(base+f'/api/cases/{case_id}/report/{kind}',timeout=30).raise_for_status()
            evidence=ROOT/'data/cases'/case_id/material['material_id']/'evidence'
            if row['material_type']=='image':
                branch=next(b for b in result['branches'] if b['branch_id']=='localization')
                assert branch['status']=='success'
                for name in [result['media']['preview'],branch['details']['mask'],branch['details']['heatmap']]:
                    session.get(base+prefix+'/assets/'+name,timeout=30).raise_for_status()
                truthpath=fixtures/row['mask_path']
                if row.get('mask_sha256'):assert sha256(truthpath)==row['mask_sha256']
                truth=np.asarray(Image.open(truthpath).convert('L'))>0
                prediction=np.asarray(Image.open(evidence/branch['details']['mask']).convert('L'))>0
                assert truth.shape==prediction.shape==(480,640)
                intersection=int((truth&prediction).sum());positive=int(truth.sum());pred=int(prediction.sum())
                record['localization']={'pixel_iou':intersection/max(1,positive+pred-intersection) if positive else None,
                    'pixel_f1':2*intersection/max(1,positive+pred) if positive else None,'predicted_area_ratio':pred/prediction.size,
                    'gt_area_pixels':positive,'mask_sha256':sha256(evidence/branch['details']['mask'])}
            else:
                _,gt=load_gt(fixtures,row,allowed_parent_splits=('public_holdout',))
                record['localization']=score_localization(result,evidence,fixtures,row,gt)
                assert record['localization']['localization_evaluation_valid']
                for frame in result['media']['frames']:
                    assets=[frame['preview']]
                    if frame.get('localization'):assets.extend([frame['localization']['details']['mask'],frame['localization']['details']['heatmap']])
                    for name in assets:session.get(base+prefix+'/assets/'+name,timeout=30).raise_for_status()
            record['original_assets_reports_verified']=True
        except Exception as exc:
            record.update({'execution_status':'failed','prediction':'undetermined','error':f'{type(exc).__name__}: {exc}','latency_ms':(time.perf_counter()-began)*1000})
        return record

    for row in rows:
        if row['sample_id'] in measured:continue
        if row['sample_id'] in reuse:
            record={**reuse[row['sample_id']],'material_type':row['material_type'],'material_class':row['material_class']}
            assert record['sha256']==row['sha256']
        else:
            record=fresh(row);print(row['sample_id'],record['prediction'],flush=True)
        records.append(record);atomic_json(path,records)
    for row in controls:
        if row['sample_id'] in measuredcontrols:continue
        record=fresh(row);controlrecords.append(record);atomic_json(controlpath,controlrecords)
        print(row['sample_id'],record['prediction'],flush=True)
    assert session.get(base+'/api/health',timeout=15).json()['configuration']==config
    assert all(sha256(ROOT/p)==digest for p,digest in implementation.items())
    valid=lambda r:r['execution_status']=='complete' and r['prediction'] in {'lower_risk','suspected_fake'}
    real=[r for r in records if r['truth_label']=='genuine'];fake=[r for r in records if r['truth_label']=='fake']
    tp=sum(valid(r) and r['prediction']=='suspected_fake' for r in fake);tn=sum(valid(r) and r['prediction']=='lower_risk' for r in real)
    fp=sum(valid(r) and r['prediction']=='suspected_fake' for r in real);fn=sum(valid(r) and r['prediction']=='lower_risk' for r in fake)
    latencies=[r['latency_ms'] for r in records]
    metrics={'total':100,'genuine':70,'fake':30,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'failed':sum(not valid(r) for r in records),
        'accuracy':(tp+tn)/100,'genuine_fpr':fp/70,'fake_recall':tp/30,'accuracy_wilson_95':wilson(tp+tn,100),
        'genuine_fpr_wilson_95':wilson(fp,70),'latency_max_ms':max(latencies),'latency_p95_ms':float(np.percentile(latencies,95)),
        'over_target':sum(t>5000 for t in latencies)}
    images=[r for r in records if r['material_class'] in {'traditional_copy_move','traditional_splice'}]
    videos=[r for r in records if r['material_class']=='traditional_video_splice']
    local={'positive_images':6,'positive_videos':3,'controls':9,
        'image_positive_macro_iou_all_planned':sum(r.get('localization',{}).get('pixel_iou') or 0 for r in images)/6,
        'image_positive_macro_f1_all_planned':sum(r.get('localization',{}).get('pixel_f1') or 0 for r in images)/6,
        'video_effective_positive_macro_iou_all_planned':sum(r.get('localization',{}).get('effective_positive_iou') or 0 for r in videos)/3,
        'video_effective_positive_macro_f1_all_planned':sum(r.get('localization',{}).get('effective_positive_f1') or 0 for r in videos)/3,
        'control_false_alarm_count':sum(valid(r) and r['prediction']=='suspected_fake' for r in controlrecords),
        'control_failed_count':sum(not valid(r) for r in controlrecords),
        'traditional_edit_detected_count':sum(valid(r) and r['prediction']=='suspected_fake' for r in images+videos),
        'continuous_intervals_claimed':False,'scope':'Known deterministic operations; video scores cover actual sampled GT-positive PTS only, unsampled times unknown.'}
    atomic_json(output/'localization.json',local)
    summary={'status':'completed','version':'1.3.0','configuration_id':config['configuration_id'],'dataset_name':output.name,
        'formal_acceptance':False,'independent_accuracy_test':False,'metrics':metrics,'localization':local,'source':read(data/'provenance.json'),
        'acceptance_checks':{'accuracy_95':metrics['accuracy']>=.95,'genuine_fpr_3':metrics['genuine_fpr']<=.03,
            'every_material_within_5_seconds':metrics['over_target']==0,'100_cases_70_30':True,'image_and_video_scope':True,
            'no_required_execution_failures':metrics['failed']==0,'contest_business_sample_scope':False},
        'notes':'预先登记100案混合功能回归：91个原件复用同一冻结配置的专项HTTP结果，9个编辑与9个额外对照实际新请求。与专项测试相关，不能累加称为300独立案件；原图分区已被候选模型测过。传统编辑只有复制与拼接，无生成式局部重绘、真实业务或跨设备验收。'}
    atomic_json(output/'summary.json',summary);print(json.dumps({'metrics':metrics,'localization':local},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
