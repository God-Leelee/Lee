"""Audit the staged/tracked publication set; ignored private runtime data is never read."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'data', 'runs', 'third_party', '.venv', 'node_modules', '__pycache__', '.pytest_cache', 'tooling'}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9]{32,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?:cookie|authorization)\s*[:=]\s*[\"\x27](?:Bearer |eyJ)[^\"\x27]+)', re.I)


def audit():
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z'])
    names = [s.decode('utf-8') for s in raw.split(b'\0') if s]
    errors, files = [], []
    for name in names:
        path = ROOT / name
        if any(part in FORBIDDEN for part in Path(name).parts):
            errors.append('Forbidden tracked directory: ' + name)
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            errors.append('Unsafe tracked link: ' + name)
            continue
        if path.suffix.lower() in {'.log', '.zip', '.xlsx', '.docx', '.parquet', '.npz', '.bin', '.safetensors', '.ckpt', '.onnx'}:
            errors.append('Forbidden tracked asset: ' + name)
        content = path.read_bytes()
        if len(content) >= 50 * 1024 * 1024:
            errors.append('Large tracked asset: ' + name)
        if b'\0' not in content and path.suffix != '.pth':
            body = content.decode('utf-8', errors='replace')
            if SECRET.search(body):
                errors.append('Possible secret: ' + name)
            if re.search(r'[CD]:[\\/](?:Users|26211)[\\/]', body, re.I):
                errors.append('Local personal path: ' + name)
            if re.search(r'https://doc\.weixin\.qq\.com/smartsheet/\S+', body):
                errors.append('Private resource-sharing URL: ' + name)
        files.append({'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    report = {'passed': not errors and bool(files), 'files': len(files),
              'bytes': sum(p['bytes'] for p in files), 'errors': errors,
              'scope': 'Heuristic audit of all git tracked files; not a complete legal or secret-detection guarantee'}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report, files


if __name__ == '__main__':
    raise SystemExit(0 if audit()[0]['passed'] else 1)
