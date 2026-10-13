"""Validate exact runtime assets without importing or loading model checkpoints."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def verify():
    manifest = json.loads((ROOT / 'models/runtime_assets.json').read_text('utf-8'))
    config = json.loads((ROOT / 'config.json').read_text('utf-8'))
    errors = []
    if config['configuration_id'] != manifest['profile']:
        errors.append('Configuration profile differs from runtime asset lock')
    expected = {item['path']: item['sha256'] for item in manifest['weights']}
    configured = [(config['temporal_checkpoint'], config['temporal_checkpoint_sha256'])]
    for key in ('image_origin_adapter', 'video_origin_adapter', 'domain_generation'):
        item = config.get(key, {})
        if item.get('enabled'):
            configured.append((item['checkpoint'], item['sha256']))
    for path, value in configured:
        if expected.get(path) != value:
            errors.append('Configured checkpoint missing or different in lock: ' + path)
    for item in config.get('generation_experts', {}).values():
        if isinstance(item, dict) and 'checkpoint' in item and expected.get(item['checkpoint']) != item['sha256']:
            errors.append('Generation checkpoint missing or different in lock: '+item['checkpoint'])
    for item in manifest['weights']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append('Missing/unsafe weight: ' + item['path'])
        elif digest(path) != item['sha256']:
            errors.append('SHA256 mismatch: ' + item['path'])
    for item in manifest['repositories']:
        path = ROOT / 'third_party' / item['name']
        if item.get('source_files'):
            for relative, expected_hash in item['source_files'].items():
                source=(path/relative).resolve()
                if not source.is_relative_to(path.resolve()) or not source.is_file() or digest(source)!=expected_hash:
                    errors.append('Missing/modified pinned upstream file: '+item['name']+'/'+relative)
            continue
        try:
            actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
            changes = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
            if actual != item['commit'] or changes:
                errors.append('Upstream commit/working tree differs: ' + item['name'])
        except (OSError, subprocess.CalledProcessError):
            errors.append('Missing upstream repository: ' + item['name'])
    report = {'passed': not errors, 'checked_weights': len(manifest['weights']),
              'checked_repositories': len(manifest['repositories']), 'errors': errors}
    if errors:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
