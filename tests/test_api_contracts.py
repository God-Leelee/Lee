"""HTTP contracts without loading models, touching real cases, or using a GPU."""

from concurrent.futures import Future
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

from fastapi.testclient import TestClient
import pytest

import backend.contracts as contracts


class RecordingWorker:
    def __init__(self, *args, **kwargs):
        self.jobs = []

    def submit(self, function, *args, **kwargs):
        future = Future()
        if function.__name__ == 'load':
            future.set_result(function(*args, **kwargs))
        else:
            self.jobs.append((future, function, args, kwargs))
        return future

    def drain(self):
        for future, function, args, kwargs in self.jobs:
            if future.done():
                continue
            try:
                future.set_result(function(*args, **kwargs))
            except Exception as error:
                future.set_exception(error)

    def shutdown(self, wait=True):
        pass


class FakeModels:
    def __init__(self, config):
        self.ready = True
        self.info = {'runtime': {'device': 'cpu'}}
        self.errors = {}
        self.ocr = object()
        self.temporal = object()
        self.clip = self.head = self.trufor = self.waverep = object()

    def load(self):
        # The lifespan may run, but a real model is never imported or loaded.
        self.ready = True


class FakeEngine:
    def __init__(self, config, models):
        self.config = config
        self.cpu = RecordingWorker()
        self.calls = []
        self.results = {}
        self.exceptions = {}

    def analyze(self, path, out, material_id, description='', received_at=None,
                material_type=None, original_name=None):
        payload = path.read_bytes()
        self.calls.append({'path': path, 'out': out, 'material_id': material_id,
                           'description': description, 'received_at': received_at,
                           'material_type': material_type, 'original_name': original_name,
                           'payload': payload})
        if original_name in self.exceptions:
            raise self.exceptions[original_name]
        out.mkdir(parents=True, exist_ok=True)
        (out / 'preview.jpg').write_bytes(b'fake-evidence-image')
        result = {'material_id': material_id, 'execution_status': 'complete',
                  'sha256': hashlib.sha256(payload).hexdigest(),
                  'fusion': {'risk': .2, 'verdict': 'lower_risk'},
                  'timing': {'total_ms': 1., 'within_target': True},
                  'explanation': [{'text': '<script>OCR text</script>'}]}
        result.update(self.results.get(original_name, {}))
        contracts.atomic_json(out / 'report.json', result)
        return result


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Execute an isolated app module with fake dependencies and a local queue."""
    model_module = types.ModuleType('backend.models')
    model_module.Models = FakeModels
    engine_module = types.ModuleType('backend.engine')
    engine_module.Engine = FakeEngine
    monkeypatch.setitem(sys.modules, 'backend.models', model_module)
    monkeypatch.setitem(sys.modules, 'backend.engine', engine_module)
    monkeypatch.setattr(contracts, 'ROOT', tmp_path)
    monkeypatch.setattr(contracts, 'load_config', lambda: {
        'version': 'api-contract-test', 'max_upload_mb': 1,
        'configuration_id': 'test-only', 'case_threshold': None,
        'image_fusion': {'required_features': ['generation', 'forensic_generation', 'localization']},
        'video_fusion': {'required_features': ['generation', 'forensic_generation', 'localization', 'temporal']},
    })
    source = Path(__file__).resolve().parents[1] / 'backend' / 'app.py'
    spec = importlib.util.spec_from_file_location('backend._api_contract_test', source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.WORKER.shutdown(wait=True)
    module.WORKER = RecordingWorker()
    with TestClient(module.app) as client:
        yield types.SimpleNamespace(client=client, app=module, root=tmp_path,
                                    engine=module.ENGINE, worker=module.WORKER)


def upload(api, files=None, **fields):
    return api.client.post('/api/cases', data=fields,
                           files=files if files is not None else [
                               ('files', ('photo.jpg', b'original-image', 'image/jpeg'))])


def test_startup_and_inference_use_same_dedicated_thread(api, monkeypatch):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import threading
    worker=ThreadPoolExecutor(max_workers=1)
    loaded=[]
    monkeypatch.setattr(api.app, 'WORKER', worker)
    monkeypatch.setattr(api.app.MODEL, 'load', lambda: loaded.append(threading.get_ident()))
    async def check():
        async with api.app.lifespan(api.app.app):
            infer_thread=await asyncio.wrap_future(worker.submit(threading.get_ident))
            assert loaded == [infer_thread]
            assert infer_thread != threading.get_ident()
    asyncio.run(check())


def test_health_and_loading_rejection_do_not_load_real_models(api):
    health = api.client.get('/api/health').json()
    assert health['version'] == 'api-contract-test'
    assert health['status'] == 'ready'
    api.app.MODEL.ready = False
    assert upload(api).status_code == 503
    assert api.worker.jobs == []
    assert list(api.app.DATA.iterdir()) == []


def test_health_respects_new_required_experts_even_with_legacy_adapter_enabled(api):
    api.app.CONFIG['image_fusion']['required_features'].append('generation_experts')
    api.app.CONFIG['image_origin_adapter'] = {'enabled': True, 'decision_role': 'ensemble_feature'}
    api.app.MODEL.image_adapter = object()
    assert not api.client.get('/api/health').json()['capabilities']['image']
    api.app.MODEL.generation_experts = object()
    assert api.client.get('/api/health').json()['capabilities']['image']
    api.app.MODEL.errors['generation_experts'] = 'weight or calibration mismatch'
    assert not api.client.get('/api/health').json()['capabilities']['image']


def test_case_list_uses_small_index_and_never_copies_full_evidence(api, monkeypatch):
    case_id = upload(api).json()['case_id']
    api.worker.drain()
    expected = api.client.get('/api/cases').json()
    api.app.CASE_CACHE.clear()
    def unexpected_full_read(*args):
        raise AssertionError('Case list must not fetch a complete report')
    monkeypatch.setattr(api.app, 'read_case', unexpected_full_read)
    assert api.client.get('/api/cases').json() == expected
    assert not api.app.CASE_CACHE
    assert expected[0]['case_id'] == case_id
    assert expected[0]['status'] == 'complete'
    assert 'materials' not in expected[0]


def test_case_list_refreshes_changed_disk_record_without_caching_full_report(api):
    case_id = upload(api).json()['case_id']
    path = api.app.DATA / case_id / 'case.json'
    case = json.loads(path.read_text('utf-8'))
    case['title'] = 'Updated externally, different length'
    contracts.atomic_json(path, case)
    api.app.CASE_CACHE.clear()
    listed = api.client.get('/api/cases').json()
    assert listed[0]['title'] == case['title']
    assert not api.app.CASE_CACHE
    listed[0]['title'] = 'Client-local change'
    assert api.client.get('/api/cases').json()[0]['title'] == case['title']


def test_full_report_cache_is_bounded_and_evicted_original_reloads(api, monkeypatch):
    monkeypatch.setattr(api.app, 'CASE_CACHE_LIMIT', 2)
    ids = [upload(api).json()['case_id'] for _ in range(3)]
    assert len(api.app.CASE_CACHE) == 2
    assert ids[0] not in api.app.CASE_CACHE
    case = api.client.get('/api/cases/' + ids[0]).json()
    assert case['case_id'] == ids[0]
    assert len(api.app.CASE_CACHE) == 2
    assert list(api.app.CASE_CACHE)[-1] == ids[0]


def test_health_never_marks_failed_ocr_or_temporal_objects_available(api):
    api.app.MODEL.errors = {'ocr': 'OCR initialization failed',
                            'temporal': 'Temporal checkpoint failed'}
    capabilities = api.client.get('/api/health').json()['capabilities']
    assert capabilities['multimodal'] is False
    assert capabilities['temporal_trained'] is False


@pytest.mark.parametrize(('branch', 'image', 'video'), [
    ('generation', False, False),
    ('forensic_generation', False, False),
    ('localization', False, False),
    ('temporal', True, False),
])
def test_health_material_capabilities_require_successful_mandatory_models(api, branch, image, video):
    assert api.client.get('/api/health').json()['capabilities']['image'] is True
    assert api.client.get('/api/health').json()['capabilities']['video'] is True
    api.app.MODEL.errors[branch] = 'partial model loading failed'
    capabilities = api.client.get('/api/health').json()['capabilities']
    assert capabilities['image'] is image
    assert capabilities['video'] is video


@pytest.mark.parametrize(('files', 'fields', 'expected'), [
    ([('files', ('payload.exe', b'unsafe', 'application/octet-stream'))], {}, 415),
    ([('files', ('photo.jpg', b'', 'image/jpeg'))], {}, 400),
    ([('files', ('photo.jpg', b'x' * (1024 * 1024 + 1), 'image/jpeg'))], {}, 413),
    ([('files', (f'photo{i}.jpg', b'x', 'image/jpeg')) for i in range(13)], {}, 400),
    (None, {'title': 'x' * 101}, 400),
    (None, {'description': 'x' * 4001}, 400),
])
def test_invalid_uploads_never_schedule_analysis(api, files, fields, expected):
    assert upload(api, files, **fields).status_code == expected
    assert api.worker.jobs == []
    assert api.engine.calls == []


def test_all_uploads_must_finish_before_multi_material_case_is_queued(api):
    result = upload(api, [
        ('files', ('first.jpg', b'valid-original', 'image/jpeg')),
        ('files', ('empty.mp4', b'', 'video/mp4')),
    ])
    assert result.status_code == 400
    assert api.worker.jobs == []
    listed = api.client.get('/api/cases').json()
    assert len(listed) == 1 and listed[0]['status'] == 'upload_failed'
    originals = list((api.app.DATA / listed[0]['case_id']).glob('*/original.jpg'))
    assert len(originals) == 1 and originals[0].read_bytes() == b'valid-original'


def test_multi_material_dispatch_preserves_paths_types_names_and_original_bytes(api):
    payloads = [b'photo-original', b'video-original']
    response = upload(api, [
        ('files', (r'..\..\evidence.jpg', payloads[0], 'image/jpeg')),
        ('files', ('clip.mp4', payloads[1], 'video/mp4')),
    ], title='案卷', description='单号 ABCD-1234')
    assert response.status_code == 202
    case_id = response.json()['case_id']
    assert response.json()['materials'] == 2
    assert len(api.worker.jobs) == 2
    queued = api.client.get(f'/api/cases/{case_id}').json()
    assert [m['status'] for m in queued['materials']] == ['queued', 'queued']
    assert [m['name'] for m in queued['materials']] == ['evidence.jpg', 'clip.mp4']
    api.worker.drain()
    assert all(future.exception() is None for future, *_ in api.worker.jobs)
    completed = api.client.get(f'/api/cases/{case_id}').json()
    assert completed['status'] == 'complete'
    assert completed['summary']['materials'] == 2
    assert completed['summary']['completed'] == 2
    assert completed['summary']['case_verdict'] == 'review_required'
    for index, (material, call) in enumerate(zip(completed['materials'], api.engine.calls)):
        folder = api.app.DATA / case_id / material['material_id']
        assert call['path'] == folder / material['stored_name']
        assert call['out'] == folder / 'evidence'
        assert call['path'].resolve().is_relative_to(api.app.DATA.resolve())
        assert call['path'].read_bytes() == payloads[index] == call['payload']
        assert call['description'] == '单号 ABCD-1234'
        assert call['received_at'] > 0
        assert call['material_type'] == ['image', 'video'][index]
        original = api.client.get(f'/api/cases/{case_id}/materials/{material["material_id"]}/original')
        assert original.content == payloads[index]
        assert material['result']['sha256'] == hashlib.sha256(payloads[index]).hexdigest()


def test_case_result_reads_are_copies_and_routes_reject_unowned_materials(api):
    case_id = upload(api).json()['case_id']
    api.worker.drain()
    case = api.app.read_case(case_id)
    material_id = case['materials'][0]['material_id']
    case['title'] = 'local mutation'
    assert api.app.read_case(case_id)['title'] != 'local mutation'
    for target in ['not-a-case', 'a' * 32]:
        assert api.client.get(f'/api/cases/{target}').status_code == 404
    assert api.client.get(f'/api/cases/{case_id}/materials/{"b" * 32}/original').status_code == 404
    assert api.client.get(f'/api/cases/{case_id}/materials/{"b" * 32}/assets/preview.jpg').status_code == 404
    for name in ['report.json', 'case.json', 'original.jpg', 'no-image.png']:
        assert api.client.get(f'/api/cases/{case_id}/materials/{material_id}/assets/{name}').status_code == 404
    assert api.client.get(f'/api/cases/{case_id}/materials/{material_id}/assets/preview.jpg').content == b'fake-evidence-image'


def test_report_exports_escape_user_input_and_preserve_json_evidence(api):
    response = upload(api, [('files', ('=HYPERLINK("x").jpg', b'original', 'image/jpeg'))],
                      title='<script>title</script>')
    case_id = response.json()['case_id']
    api.worker.drain()
    html = api.client.get(f'/api/cases/{case_id}/report/html')
    assert html.status_code == 200
    assert '<script>' not in html.text
    assert '&lt;script&gt;title&lt;/script&gt;' in html.text
    assert '&lt;script&gt;OCR text&lt;/script&gt;' in html.text
    csv = api.client.get(f'/api/cases/{case_id}/report/csv')
    assert "'=HYPERLINK" in csv.text
    exported = api.client.get(f'/api/cases/{case_id}/report/json')
    assert exported.json()['materials'][0]['result']['explanation'][0]['text'] == '<script>OCR text</script>'
    assert 'attachment;' in exported.headers['content-disposition']
    assert api.client.get(f'/api/cases/{case_id}/report/not-supported').status_code == 404


def test_partial_and_failed_materials_never_produce_a_complete_case(api):
    api.engine.results['partial.jpg'] = {'execution_status': 'partial',
                                       'fusion': {'risk': None, 'verdict': 'undetermined'}}
    api.engine.results['failed.jpg'] = {'execution_status': 'failed',
                                      'fusion': {'risk': None, 'verdict': 'undetermined'}}
    case_id = upload(api, [
        ('files', ('partial.jpg', b'p', 'image/jpeg')),
        ('files', ('failed.jpg', b'f', 'image/jpeg')),
    ]).json()['case_id']
    api.worker.drain()
    case = api.client.get(f'/api/cases/{case_id}').json()
    assert case['status'] == 'failed'
    assert case['summary']['execution_complete'] is False
    assert case['summary']['candidate_max_material_risk'] is None
    assert case['summary']['case_verdict'] == 'review_required'


def test_unexpected_engine_failure_is_persisted_and_next_material_still_runs(api):
    api.engine.exceptions['unreadable.jpg'] = OSError('temporary evidence write failure')
    case_id = upload(api, [
        ('files', ('unreadable.jpg', b'u', 'image/jpeg')),
        ('files', ('good.jpg', b'g', 'image/jpeg')),
    ]).json()['case_id']
    api.worker.drain()
    case = api.client.get(f'/api/cases/{case_id}').json()
    assert [m['status'] for m in case['materials']] == ['failed', 'complete']
    assert case['status'] == 'partial'
    assert case['materials'][0]['result']['fusion']['risk'] is None
    assert case['materials'][0]['result']['fusion']['verdict'] == 'undetermined'
    assert len(api.engine.calls) == 2


def test_atomic_json_rejects_nonfinite_scores_without_replacing_valid_report(tmp_path):
    report = tmp_path / 'report.json'
    contracts.atomic_json(report, {'risk': .4})
    with pytest.raises(ValueError):
        contracts.atomic_json(report, {'risk': float('nan')})
    assert json.loads(report.read_text(encoding='utf-8')) == {'risk': .4}


def test_failed_case_write_does_not_publish_unpersisted_cache_state(api, monkeypatch):
    case_id = upload(api).json()['case_id']
    updated = api.app.read_case(case_id)
    updated['status'] = 'complete'

    def fail_write(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(api.app, 'atomic_json', fail_write)
    with pytest.raises(OSError):
        api.app.save_case(updated)
    assert api.app.read_case(case_id)['status'] == 'queued'
    persisted = json.loads((api.app.DATA / case_id / 'case.json').read_text(encoding='utf-8'))
    assert persisted['status'] == 'queued'


def test_restart_recomputes_case_summary_after_interrupting_pending_material(api):
    case_id = 'c' * 32
    case = {'case_id': case_id, 'title': 'Interrupted case', 'description': '',
            'created_at': '2026-10-02T00:00:00+00:00', 'status': 'analyzing',
            'summary': {'execution_complete': True, 'failed': 0},
            'materials': [
                {'material_id': 'd' * 32, 'name': 'finished.jpg', 'stored_name': 'original.jpg',
                 'material_type': 'image', 'status': 'complete',
                 'result': {'execution_status': 'complete', 'fusion': {'risk': .2, 'verdict': 'lower_risk'}}},
                {'material_id': 'e' * 32, 'name': 'pending.jpg', 'stored_name': 'original.jpg',
                 'material_type': 'image', 'status': 'queued'},
            ]}
    contracts.atomic_json(api.app.DATA / case_id / 'case.json', case)
    with TestClient(api.app.app) as restarted:
        recovered = restarted.get(f'/api/cases/{case_id}').json()
    assert recovered['status'] == 'partial'
    assert [m['status'] for m in recovered['materials']] == ['complete', 'failed']
    assert recovered['summary']['execution_complete'] is False
    assert recovered['summary']['completed'] == 1
    assert recovered['summary']['failed'] == 1
    assert recovered['summary']['candidate_max_material_risk'] == .2


@pytest.mark.parametrize(('branch', 'attributes'), [
    ('generation', ['clip', 'head', 'scene_text', 'preprocess']),
    ('forensic_generation', ['waverep', 'wave_preprocess']),
    ('localization', ['trufor']),
    ('ocr', ['ocr']),
    ('temporal', ['temporal']),
])
def test_model_load_failure_disables_partially_initialized_objects(branch, attributes):
    # Importing the class initializes no networks. Bypassing its constructor
    # also avoids CUDA availability checks; this test allocates no tensors.
    from backend.models import Models
    models = Models.__new__(Models)
    models.errors, models.info = {}, {branch: {'sha256': 'stale-manifest'}}
    for attribute in attributes:
        setattr(models, attribute, object())
    models._disable_model(branch, ValueError('incompatible checkpoint'))
    assert all(getattr(models, attribute) is None for attribute in attributes)
    assert branch not in models.info
    assert 'incompatible checkpoint' in models.errors[branch]


def test_error_marked_temporal_network_cannot_produce_a_random_score():
    from backend.models import Models
    models = Models.__new__(Models)
    models.temporal = object()
    models.errors = {'temporal': 'partial checkpoint load failed'}
    assert models.temporal_score(None, None) is None


def test_each_model_warmup_runs_even_after_another_model_fails():
    from backend.models import Models
    from PIL import Image
    models = Models.__new__(Models)
    models.config = {'window_frames': 8}
    models.errors, models.info = {}, {}
    models.clip = models.head = models.trufor = models.waverep = models.temporal = object()
    warmed = []

    def fail_generation(*args):
        warmed.append('generation')
        raise RuntimeError('generation warmup failed')

    models.encode = fail_generation
    models.localize = lambda *args: warmed.append('localization')
    models.wave_scores = lambda *args: warmed.append('forensic_generation')
    models.ocr = lambda *args: warmed.append('ocr')
    models.temporal_score = lambda *args: warmed.append('temporal')
    models._warmup_models(Image.new('RGB', (2, 2)))
    assert warmed == ['generation', 'localization', 'forensic_generation', 'ocr', 'temporal']
    assert models.clip is None and models.head is None
    assert models.info['warmup']['generation']['status'] == 'failed'
    assert all(models.info['warmup'][branch]['status'] == 'success' for branch in warmed[1:])
    assert set(models.errors) == {'generation'}
