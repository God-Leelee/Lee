"""Small, deterministic research subset from the original GenVideo archive.

HTTP Range avoids downloading the 13 GB archive. ZIP CRC and SHA256 are checked.
This is a development/public holdout set, never a contest acceptance certificate.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://modelscope.cn/datasets/cccnju/Gen-Video/resolve/master/GenVideo-Val.zip'


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-0', 'User-Agent': 'ZAForensics/1.1 research'})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.cache = {}

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos

    def read(self, count=-1):
        count = min(self.size - self.pos, count if count >= 0 else self.size - self.pos)
        if count <= 0:
            return b''
        start, end = self.pos, self.pos + count - 1
        key = (start, end)
        if key not in self.cache:
            for attempt in range(4):
                try:
                    req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'ZAForensics/1.1 research'})
                    with urllib.request.urlopen(req, timeout=90) as response:
                        if response.status != 206:
                            raise IOError('Server did not honor bounded Range')
                        content_range = response.headers.get('Content-Range', '')
                        if not content_range.startswith(f'bytes {start}-{end}/'):
                            raise IOError('Range response offset mismatch: ' + content_range)
                        value = response.read(count + 1)
                    if len(value) != count:
                        raise IOError('Incomplete Range response')
                    if count < 3_000_000:
                        self.cache[key] = value
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1 + attempt)
        else:
            value = self.cache[key]
        self.pos += count
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-group', type=int, default=30)
    parser.add_argument('--real-count', type=int, default=360)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'data/public_genvideo_v1'
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RangeFile(URL)) as archive:
        videos = [i for i in archive.infolist() if i.filename.lower().endswith('.mp4')
                  and '__MACOSX' not in i.filename.split('/')
                  and not Path(i.filename).name.startswith('.')]
        groups = {}
        for item in videos:
            parts = item.filename.split('/')
            group = 'real_MSRVTT' if 'Real' in parts else parts[parts.index('Fake') + 1] if 'Fake' in parts else 'unknown'
            groups.setdefault(group, []).append(item)
        print({g: len(v) for g, v in groups.items()}, flush=True)
        if args.list_only:
            return
        rows = []
        randomizer = random.Random(102411)
        for group, items in sorted(groups.items()):
            if group not in ['real_MSRVTT', 'Crafter', 'Gen2', 'HotShot', 'Lavie', 'ModelScope', 'Show_1', 'Sora']:
                continue
            items = sorted(items, key=lambda i: i.filename)
            randomizer.shuffle(items)
            take = args.real_count if group == 'real_MSRVTT' else args.per_group
            for index, item in enumerate(items[:take]):
                basename = f'{group}_{index:04d}.mp4'
                dest = target / basename
                if not dest.exists():
                    data = archive.read(item)  # zipfile verifies upstream CRC
                    dest.write_bytes(data)
                rows.append({'sample_id': basename[:-4], 'relative_path': basename,
                             'truth_label': 'genuine' if group == 'real_MSRVTT' else 'fake',
                             'material_type': 'video', 'source_group': item.filename,
                             'generator_tool': 'not_applicable' if group == 'real_MSRVTT' else group,
                             'source_url': URL, 'archive_member': item.filename,
                             'split': 'train' if index % 20 < 11 else 'fusion' if index % 20 < 13 else 'calibration' if index % 20 < 15 else 'threshold' if index % 20 < 16 else 'public_holdout',
                             'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                             'license': 'see GenVideo/MSRVTT original source; research use; media not redistributed'})
                (target / 'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(group, index + 1, dest.stat().st_size, flush=True)
    print('Selected subset downloaded. Public holdout is not the contest test.', flush=True)


if __name__ == '__main__':
    main()
