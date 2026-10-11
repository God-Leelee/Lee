"""Real local HTTP replay of procedural inputs; never an accuracy benchmark."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text('utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    parsed = urlparse(args.base)
    require(parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}
            and not parsed.username and not parsed.password, 'Only a local HTTP service is allowed')
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / 'runs'), 'Output must be a new directory below runs/')
    output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False

    def get(path):
        response = session.get(args.base.rstrip('/') + path, timeout=120)
        response.raise_for_status()
        return response

    config = read(ROOT / 'config.json')
    health = get('/api/health').json()
    require(health['status'] == 'ready' and not health['model_errors'], 'Models not fully ready')
    require(health['configuration'] == config, 'Service configuration differs from this checkout')
    require(all(health['capabilities'].get(k) for k in ('image', 'video', 'multimodal', 'metadata', 'temporal_trained')),
            'Mandatory chain unavailable')
    manifest = read(ROOT / 'data/review_demo/manifest.json')
    records = []
    for index, sample in enumerate(manifest['samples']):
        row = {'input': sample['path'], 'material_type': sample['material_type'], 'passed': False}
        start = time.perf_counter()
        try:
            source = (ROOT / 'data/review_demo' / sample['path']).resolve()
            require(source.is_relative_to(ROOT / 'data/review_demo'), 'Unsafe sample path')
            payload = source.read_bytes()
            expected = hashlib.sha256(payload).hexdigest()
            require(expected == sample['sha256'], 'Input hash mismatch')
            begin = time.perf_counter()
            response = session.post(args.base.rstrip('/') + '/api/cases',
                data={'title': f'工程演示 {index + 1}', 'description': ''},
                files={'files': ('material' + source.suffix, payload, 'application/octet-stream')}, timeout=120)
            response.raise_for_status()
            cid = response.json()['case_id']
            row['case_id'] = cid
            while True:
                case = get('/api/cases/' + cid).json()
                if case['status'] in {'complete', 'partial', 'failed', 'upload_failed'}:
                    break
                require(time.perf_counter() - begin < args.timeout, 'Inference timed out')
                time.sleep(.1)
            row['latency_ms'] = round((time.perf_counter() - begin) * 1000, 3)
            item = case['materials'][0]
            result = item.get('result', {})
            require(case['status'] == 'complete' and result.get('execution_status') == 'complete', 'Incomplete inference')
            require(result.get('sha256') == expected, 'Result belongs to different bytes')
            require(result.get('configuration_id') == config['configuration_id'], 'Result configuration differs')
            require(result['material_type'] == sample['material_type'], 'Material kind differs')
            risk = result['fusion'].get('risk')
            require(isinstance(risk, (int, float)) and math.isfinite(risk) and 0 <= risk <= 1, 'Invalid risk')
            branches = {b['branch_id']: b for b in result['branches']}
            required = {'generation', 'forensic_generation', 'localization', 'copy_move', 'metadata', 'multimodal'}
            required |= {'temporal', 'video_origin'} if sample['material_type'] == 'video' else {'standalone_generation'}
            require(required <= branches.keys(), 'Required evidence branches missing')
            require(all(branches[k]['status'] != 'failed' for k in required), 'Evidence branch failed')
            evidence = output / f'sample-{index + 1}'
            evidence.mkdir()
            prefix = f'/api/cases/{cid}/materials/{item["material_id"]}'
            require(hashlib.sha256(get(prefix + '/original').content).hexdigest() == expected, 'Stored original changed')

            def asset(name, shape=None):
                require(Path(name).name == name and name.endswith(('.png', '.jpg')), 'Unsafe evidence name')
                content = get(prefix + '/assets/' + name).content
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    if shape:
                        require(image.size == shape, 'Evidence dimensions differ from source pixels')
                (evidence / name).write_bytes(content)

            media = result['media']
            asset(media['preview'])
            if sample['material_type'] == 'image':
                details = branches['localization']['details']
                asset(details['mask'], (media['width'], media['height']))
                asset(details['heatmap'])
            else:
                seen = set()
                require(bool(media['frames']), 'No sampled video frames')
                for frame in media['frames']:
                    key = (frame['pts'], frame['time_base'])
                    require(key not in seen, 'Duplicate sampled PTS')
                    seen.add(key)
                    local = frame['localization']
                    require(local['status'] == 'success', 'Sampled-frame localization failed')
                    require(abs(local['details']['timestamp_seconds'] - frame['timestamp_seconds']) < 1e-6,
                            'Mask timestamp mismatch')
                    asset(local['details']['mask'], (frame['width'], frame['height']))
                    asset(local['details']['heatmap'])
                row['verified_pts_frames'] = len(seen)
            for extension in ('json', 'csv', 'html'):
                report = get(f'/api/cases/{cid}/report/{extension}')
                require(bool(report.content), 'Empty report')
                if extension == 'json':
                    require(report.json()['materials'][0]['result']['sha256'] == expected, 'Report input mismatch')
                (evidence / ('report.' + extension)).write_bytes(report.content)
            row.update(passed=True, risk=risk, verdict=result['fusion']['verdict'], sha256=expected,
                       branch_statuses={k: branches[k]['status'] for k in sorted(required)})
        except Exception as error:
            row.update(error=f'{type(error).__name__}: {error}')
        row['verification_total_ms'] = round((time.perf_counter() - start) * 1000, 3)
        records.append(row)
        save(output / 'records.json', records)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {'profile': config['configuration_id'], 'passed': all(r['passed'] for r in records) and bool(records),
               'samples': len(records), 'failed': sum(not r['passed'] for r in records),
               'over_5_seconds': sum(r.get('latency_ms', float('inf')) > 5000 for r in records),
               'classification_accuracy_claimed': False, 'clean_machine_installation_verified': False,
               'scope': manifest['scope'], 'latency_scope': 'HTTP upload to readable terminal case; evidence downloads separately recorded'}
    save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    raise SystemExit(run(parser.parse_args()))
