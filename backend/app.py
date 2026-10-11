from __future__ import annotations

from contextlib import asynccontextmanager
from collections import OrderedDict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .contracts import ROOT, load_config, atomic_json
from .engine import Engine
from .models import Models
from .reports import case_csv, case_html

CONFIG = load_config()
DATA = ROOT / 'data/cases'
DATA.mkdir(parents=True, exist_ok=True)
MODEL = Models(CONFIG)
ENGINE = Engine(CONFIG, MODEL)
WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='forensics')
LOCK = threading.RLock()
CASE_CACHE = OrderedDict()
CASE_CACHE_LIMIT = 64
CASE_LIST_CACHE = {}
IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.avif', '.heic', '.heif'}
VIDEO_TYPES = {'.mp4', '.mov', '.mkv', '.webm', '.avi'}


def case_path(case_id):
    if not re.fullmatch(r'[a-f0-9]{32}', case_id):
        raise HTTPException(404, '案件不存在。')
    path = DATA / case_id
    if not path.is_dir():
        raise HTTPException(404, '案件不存在。')
    return path


def read_case(case_id):
    with LOCK:
        if case_id not in CASE_CACHE:
            path = case_path(case_id) / 'case.json'
            CASE_CACHE[case_id] = json.loads(path.read_text(encoding='utf-8'))
        CASE_CACHE.move_to_end(case_id)
        while len(CASE_CACHE) > CASE_CACHE_LIMIT:
            CASE_CACHE.popitem(last=False)
        return json.loads(json.dumps(CASE_CACHE[case_id], ensure_ascii=False))


def case_overview(case):
    """Keep list responses independent of the large evidence/configuration tree."""
    return {k: deepcopy(case.get(k)) for k in
            ('case_id', 'title', 'status', 'created_at', 'summary')} | {
                'material_count': len(case['materials'])}


def remember_overview(case, path):
    stat = path.stat()
    CASE_LIST_CACHE[case['case_id']] = ((stat.st_mtime_ns, stat.st_size), case_overview(case))


def save_case(case):
    with LOCK:
        snapshot = json.loads(json.dumps(case, ensure_ascii=False, allow_nan=False))
        path = DATA / snapshot['case_id'] / 'case.json'
        atomic_json(path, snapshot)
        CASE_CACHE[snapshot['case_id']] = snapshot
        CASE_CACHE.move_to_end(snapshot['case_id'])
        while len(CASE_CACHE) > CASE_CACHE_LIMIT:
            CASE_CACHE.popitem(last=False)
        remember_overview(snapshot, path)


def finish_case(case):
    complete = all(m.get('result', {}).get('execution_status') == 'complete' for m in case['materials'])
    valid = [m['result']['fusion']['risk'] for m in case['materials'] if m.get('result', {}).get('fusion', {}).get('risk') is not None]
    case['status'] = 'complete' if complete else 'partial' if valid else 'failed'
    # No independently selected suite threshold yet: expose maximum as a
    # candidate aggregation instead of asserting calibrated case accuracy.
    case['summary'] = {'candidate_max_material_risk': max(valid) if valid else None,
                       'aggregation_rule': 'candidate_max_requires_case_threshold_validation',
                       'case_verdict': 'review_required', 'execution_complete': complete,
                       'materials': len(case['materials']), 'completed': sum(m['status'] == 'complete' for m in case['materials']),
                       'failed': sum(m['status'] == 'failed' for m in case['materials']),
                       'case_threshold': CONFIG.get('case_threshold')}
    save_case(case)


def run_material(case_id, material_id, received):
    case = read_case(case_id)
    item = next(m for m in case['materials'] if m['material_id'] == material_id)
    item['status'] = 'analyzing'
    case['status'] = 'analyzing'
    save_case(case)
    path = DATA / case_id / material_id
    try:
        result = ENGINE.analyze(path / item['stored_name'], path / 'evidence', material_id,
                               case.get('description', ''), received, item['material_type'], item['name'])
        if result.get('execution_status') not in {'complete', 'partial', 'failed'}:
            raise ValueError('Engine returned an invalid execution status')
    except Exception as exc:
        # Artifact I/O can fail outside the engine's branch guards. Persist a
        # terminal failure so one material cannot keep the entire case pending.
        result = {'schema_version': '1.1', 'version': CONFIG['version'],
                  'material_id': material_id, 'name': item['name'],
                  'material_type': item['material_type'], 'execution_status': 'failed',
                  'error': f'{type(exc).__name__}: {exc}', 'branches': [],
                  'fusion': {'status': 'failed', 'risk': None, 'verdict': 'undetermined'},
                  'configuration_id': CONFIG.get('configuration_id')}
    case = read_case(case_id)
    item = next(m for m in case['materials'] if m['material_id'] == material_id)
    item.update({'result': result, 'status': result['execution_status']})
    save_case(case)
    if all(m['status'] in ['complete', 'partial', 'failed'] for m in case['materials']):
        finish_case(case)
    # Measure through the real first persisted case commit. Persisting this
    # timing footer is outside the interval; external benchmarks include it.
    timing=result.setdefault('timing',{})
    timing['api_total_ms']=round((time.perf_counter()-received)*1000,2)
    timing['api_end_definition']='evidence_report_and_case_result_persisted_before_timing_footer'
    timing['api_start_definition']='complete_multipart_body_received_by_server'
    timing['api_within_target']=timing['api_total_ms']<=5000
    report_path=path/'evidence/report.json'
    if report_path.exists():
        atomic_json(report_path,result)
    save_case(case)


@asynccontextmanager
async def lifespan(app):
    startup_began = time.perf_counter()
    # Load and warm all networks before accepting files; cold-start measured.
    # CUDA/cuDNN plans and handles can be thread-local. Use the same dedicated
    # worker for initialization and inference, before advertising readiness.
    await __import__('asyncio').wrap_future(WORKER.submit(MODEL.load))
    index_began = time.perf_counter()
    for entry in DATA.glob('*/case.json'):
        case = json.loads(entry.read_text(encoding='utf-8'))
        interrupted = False
        for material in case['materials']:
            if material['status'] in ['queued', 'analyzing']:
                material['status'] = 'failed'
                material['result'] = {'execution_status': 'failed', 'error': '上次服务停止时任务尚未完成；请重新提交材料。',
                                      'fusion': {'risk': None, 'verdict': 'undetermined'}}
                interrupted = True
        if interrupted:
            finish_case(case)
        with LOCK:
            remember_overview(case, entry)
    MODEL.info.setdefault('runtime', {}).update({
        'case_index_startup_seconds': round(time.perf_counter() - index_began, 3),
        'service_startup_seconds': round(time.perf_counter() - startup_began, 3)})
    yield
    WORKER.shutdown(wait=True)
    ENGINE.cpu.shutdown(wait=True)


app = FastAPI(title='澄鉴 · CLEARTRACE', version=CONFIG['version'], lifespan=lifespan)


class ReceivedBodyClock:
    def __init__(self, app):
        self.app=app

    async def __call__(self, scope, receive, send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        async def timed_receive():
            message=await receive()
            if message['type']=='http.request' and not message.get('more_body',False):
                scope.setdefault('state',{})['body_received_at']=time.perf_counter()
            return message
        await self.app(scope,timed_receive,send)


app.add_middleware(ReceivedBodyClock)


@app.get('/api/health')
def health():
    available = {
        'generation': getattr(MODEL, 'clip', None) is not None and getattr(MODEL, 'head', None) is not None,
        'forensic_generation': getattr(MODEL, 'waverep', None) is not None,
        'localization': getattr(MODEL, 'trufor', None) is not None,
        'temporal': MODEL.temporal is not None,
        'copy_move': True,
        'standalone_generation': getattr(MODEL, 'image_adapter', None) is not None,
        'video_origin': getattr(MODEL, 'video_adapter', None) is not None,
    }
    available = {branch: active and branch not in MODEL.errors for branch, active in available.items()}
    image_required = CONFIG.get('image_fusion', {}).get('required_features', ['generation', 'localization'])
    if CONFIG.get('image_origin_adapter', {}).get('enabled') and CONFIG['image_origin_adapter'].get('decision_role') != 'review_only':
        image_required = ['generation', 'forensic_generation', 'localization', 'standalone_generation']
    video_required = CONFIG.get('video_fusion', {}).get('required_features', ['generation', 'localization', 'temporal'])
    return {'status': 'ready' if MODEL.ready else 'loading', 'version': CONFIG['version'],
            'models': MODEL.info, 'model_errors': MODEL.errors,
            'capabilities': {'image': MODEL.ready and all(available.get(branch, False) for branch in image_required),
                             'video': MODEL.ready and all(available.get(branch, False) for branch in video_required),
                             'copy_move': True,
                             'multimodal': available['generation'] and MODEL.ocr is not None and 'ocr' not in MODEL.errors,
                             'temporal_trained': available['temporal'],
                             'domain_ai_trained': getattr(MODEL,'domain_head',None) is not None and 'domain_generation' not in MODEL.errors,
                             'public_image_origin_trained': available['standalone_generation'],
                             'public_video_origin_trained': available['video_origin'],
                             'metadata': True, 'reports': True},
            'configuration': CONFIG, 'formal_acceptance': 'not_yet_verified'}


@app.get('/api/cases')
def list_cases():
    entries = [(path, path.stat()) for path in DATA.glob('*/case.json')]
    entries.sort(key=lambda item: item[1].st_mtime_ns, reverse=True)
    result = []
    for path, stat in entries:
        case_id = path.parent.name
        stamp = (stat.st_mtime_ns, stat.st_size)
        with LOCK:
            cached = CASE_LIST_CACHE.get(case_id)
            if cached is None or cached[0] != stamp:
                # Loading a list must not populate the full-report LRU cache.
                case = json.loads(path.read_text(encoding='utf-8'))
                remember_overview(case, path)
            result.append(deepcopy(CASE_LIST_CACHE[case_id][1]))
    return result


@app.get('/api/cases/{case_id}')
def get_case(case_id: str):
    return read_case(case_id)


@app.post('/api/cases', status_code=202)
async def create_case(request: Request, files: list[UploadFile] = File(...), title: str = Form('新建材料核验'), description: str = Form('')):
    if not MODEL.ready:
        raise HTTPException(503, '模型仍在加载，请稍后。')
    if not 1 <= len(files) <= 12:
        raise HTTPException(400, '每案需上传1–12份材料。')
    if len(title) > 100 or len(description) > 4000:
        raise HTTPException(400, '标题限100字，描述限4000字。')
    for file in files:
        if Path(file.filename or '').suffix.lower() not in IMAGE_TYPES | VIDEO_TYPES:
            raise HTTPException(415, f'不支持 {Path(file.filename or "").suffix} 格式。')
    case_id = uuid.uuid4().hex
    path = DATA / case_id
    path.mkdir()
    case = {'case_id': case_id, 'title': title.strip() or '新建材料核验', 'description': description,
            'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'queued', 'materials': [], 'summary': None}
    times = []
    try:
        for file in files:
            material_id = uuid.uuid4().hex
            suffix = Path(file.filename or '').suffix.lower()
            stored_name = 'original' + suffix
            folder = path / material_id
            folder.mkdir()
            total = 0
            with (folder / stored_name).open('wb') as dest:
                while block := await file.read(1024 * 1024):
                    total += len(block)
                    if total > CONFIG['max_upload_mb'] * 1024 * 1024:
                        raise HTTPException(413, f'单份材料上限{CONFIG["max_upload_mb"]}MB。')
                    dest.write(block)
            if total == 0:
                raise HTTPException(400, '不能提交空文件。')
            received = getattr(request.state,'body_received_at',time.perf_counter())
            case['materials'].append({'material_id': material_id, 'name': Path(file.filename or '材料').name,
                                       'stored_name': stored_name, 'material_type': 'video' if suffix in VIDEO_TYPES else 'image',
                                       'bytes': total, 'status': 'queued'})
            times.append(received)
        save_case(case)
    except Exception:
        # Retain failed originals for traceability, but do not schedule a partial
        # upload as if it were a complete case.
        case['status'] = 'upload_failed'
        save_case(case)
        raise
    for material, received in zip(case['materials'], times):
        WORKER.submit(run_material, case_id, material['material_id'], received)
    return {'case_id': case_id, 'status': 'queued', 'materials': len(case['materials'])}


@app.get('/api/cases/{case_id}/materials/{material_id}/assets/{name}')
def asset(case_id: str, material_id: str, name: str):
    case = read_case(case_id)
    if not any(m['material_id'] == material_id for m in case['materials']):
        raise HTTPException(404, '材料不存在。')
    if not re.fullmatch(r'[a-zA-Z0-9_]+\.(?:png|jpg)', name):
        raise HTTPException(404, '证据不存在。')
    path = case_path(case_id) / material_id / 'evidence' / name
    if not path.is_file():
        raise HTTPException(404, '证据尚未生成。')
    return FileResponse(path)


@app.get('/api/cases/{case_id}/materials/{material_id}/original')
def original(case_id: str, material_id: str):
    case = read_case(case_id)
    material = next((m for m in case['materials'] if m['material_id'] == material_id), None)
    if material is None:
        raise HTTPException(404, '材料不存在。')
    suffix = Path(material['stored_name']).suffix
    media_type = {'.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm'}.get(suffix, 'application/octet-stream')
    return FileResponse(case_path(case_id) / material_id / material['stored_name'], media_type=media_type, filename=material['name'], content_disposition_type='inline' if material['material_type'] == 'video' else 'attachment')


@app.get('/api/cases/{case_id}/report/{format}')
def report(case_id: str, format: str):
    case = read_case(case_id)
    if format == 'json':
        return Response(json.dumps(case, ensure_ascii=False, indent=2), media_type='application/json', headers={'Content-Disposition': f'attachment; filename="cleartrace_{case_id}.json"'})
    if format == 'csv':
        return Response(case_csv(case), media_type='text/csv; charset=utf-8', headers={'Content-Disposition': f'attachment; filename="cleartrace_{case_id}.csv"'})
    if format == 'html':
        return HTMLResponse(case_html(case))
    raise HTTPException(404, '支持JSON、CSV和HTML报告。')


@app.get('/api/benchmark')
def benchmark(dataset: str='latest'):
    choices={'suite15':('public100-mixed-independent-v1.5-001','1.5 新来源混合100套 · 独立实测'),
             'formats15':('native-formats-v1.5-001','1.5 主流格式 · 兼容与重编码回归'),
             'suite14verified':('public100-mixed-independent-v1.4-002-verified','1.4 混合100套 · 证据修复复核'),
             'suite14initial':('public100-mixed-independent-v1.4-002','1.4 首轮实测 · 校验失败记录'),
             'formats14':('native-formats-v1.4-002','1.4 主流格式 · 兼容实测'),
             'suite13final':('public100-complete-suite-delivery-regression-v1.3','1.3 最终交付 · 混合100案回归'),
             'image13final':('public100-image-final-performance-v1.3','1.3 最终交付 · 图片性能回归'),
             'video13final':('public100-video-final-performance-v1.3','1.3 最终交付 · 视频性能回归'),
             'suite13':('public100-complete-suite-v1.3','1.3 混合100案 · 首轮回归'),
             'image13':('public100-standalone-image-v1.3','1.3 图片主策略候选 · 未通过准入'),
             'image13release':('public100-image-delivery-regression-v1.3','1.3 交付策略 · 图片首轮回归'),
             'video13':('public100-independent-video-v1.3','1.3 新父源公开视频 · 首轮独立'),
             'wechat':('wechat-ai-origin-v1.2','1.2 微信样本 · AI来源模型'),
             'video':('public100-v1.1','1.1 独立公开视频'),'image':('public100-image-v1.1','1.1 同来源图片'),'suite':('public100-paired-v1.1','1.1 图片＋视频套件')}
    if dataset not in {'latest',*choices}:
        raise HTTPException(400,'未知评测集合。')
    if dataset=='latest':
        if (ROOT/'benchmark_results/public100-mixed-independent-v1.5-001/summary.json').exists():dataset='suite15'
        elif (ROOT/'benchmark_results/public100-mixed-independent-v1.4-002-verified/summary.json').exists():dataset='suite14verified'
        elif (ROOT/'benchmark_results/public100-complete-suite-delivery-regression-v1.3/summary.json').exists():dataset='suite13final'
        elif (ROOT/'benchmark_results/public100-complete-suite-v1.3/summary.json').exists():dataset='suite13'
        elif (ROOT/'benchmark_results/public100-image-delivery-regression-v1.3/summary.json').exists():dataset='image13release'
        elif (ROOT/'benchmark_results/public100-standalone-image-v1.3/summary.json').exists():dataset='image13'
        elif (ROOT/'benchmark_results/wechat-ai-origin-v1.2/summary.json').exists():dataset='wechat'
        elif (ROOT/'benchmark_results/public100-paired-v1.1/summary.json').exists():dataset='suite'
    path=ROOT/'benchmark_results'/choices[dataset][0]/'summary.json' if dataset in choices else ROOT/'benchmark_results/latest_summary.json'
    if dataset=='suite' and (ROOT/'benchmark_results/public100-paired-performance-v1.1/summary.json').exists():
        path=ROOT/'benchmark_results/public100-paired-performance-v1.1/summary.json'
    result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status':'not_run','formal_acceptance':False}
    if dataset=='wechat' and path.exists():
        result.update({'status':'complete','task_kind':'ai_origin','dataset_name':'微信资源包：分组后20张保留样本，仅评测AI／非AI来源标签',
                       'configuration_id':'wechat_generation_v1.2 / '+result['model_sha256'][:12],
                       'notes':'119张原始图片按近似材料分组，训练44、概率校准26、阈值选择29、保留测试20。标签仅说明AI来源；非AI不代表无传统篡改。此处不是完整风控产品验收。样本适配分支为辅助复核。'})
        result['metrics']['total']=result['metrics']['samples']
    result['available_datasets']=[{'id':key,'name':value[1]} for key,value in choices.items() if (ROOT/'benchmark_results'/value[0]/'summary.json').exists()]
    result['selected_dataset']=dataset
    return result


FRONTEND = ROOT / 'frontend/dist'
if FRONTEND.exists():
    app.mount('/', StaticFiles(directory=FRONTEND, html=True), name='ui')
