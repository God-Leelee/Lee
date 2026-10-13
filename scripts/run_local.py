"""Verified local-only entrypoint; original inputs stay in data/cases."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    from scripts.verify_assets import verify
    print(verify(), flush=True)
    if not (ROOT / 'frontend/dist/index.html').is_file():
        raise SystemExit('Frontend not built. Run npm ci and npm run build in frontend first.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print(f'CLEARTRACE: loading models; then open http://127.0.0.1:{args.port}/', flush=True)
    import uvicorn
    uvicorn.run('backend.app:app', host='127.0.0.1', port=args.port)
