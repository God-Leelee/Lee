"""Install the locked test environment with CPU PyTorch for GitHub CI only."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.8.0', 'torchvision==0.23.0',
                    '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
    requirements = [line.strip() for line in (ROOT / 'requirements.lock').read_text('utf-8').splitlines()
                    if line.strip() and not line.lstrip().startswith('#')
                    and line.split('==')[0].lower() not in {'torch', 'torchvision'}]
    subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
