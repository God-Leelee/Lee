"""Acquire original bytes from a pinned researcher release, grouped by caption.

Optional acquisition dependency, isolated from the inference environment:
  .venv/Scripts/python.exe -m pip install --target tooling/pyarrow pyarrow==21.0.0 --no-deps

Test rows are shuffled, so one test row group has no complete sibling sets.
Scan inexpensive caption/label columns across two test shards first, freeze
the 300-development/100-reserved selection, then read the required original
image columns. Never use preview JPEGs or an image-derived sampling decision.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import threading
import time
import urllib.request

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tooling/pyarrow'))
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise RuntimeError('Install the optional pinned acquisition dependency in tooling/pyarrow; see script header') from exc
from fetch_public_videos import RangeFile

REPOSITORY = 'Rajarshi-Roy-research/Defactify_Image_Dataset'
REVISION = '787334f7857fa54f29027a7f09c30e895ad486ef'
CARD = f'https://huggingface.co/datasets/{REPOSITORY}/raw/{REVISION}/README.md'
SEED = 10241103
TOOLS = {0: ('not_applicable', 'not_applicable'), 1: ('Stable Diffusion', '2.1'),
         2: ('Stable Diffusion', 'XL'), 3: ('Stable Diffusion', '3'),
         4: ('DALL-E', '3'), 5: ('Midjourney', '6')}
TRAIN_MEMBERS = [('data/train-00000-of-00007.parquet', [0, 1])]
TEST_MEMBERS = [('data/test-00000-of-00008.parquet', None),
                ('data/test-00001-of-00008.parquet', None)]
POLICY = {
    'version': 2, 'repository': REPOSITORY, 'revision': REVISION, 'seed': SEED,
    'metadata_coverage': {'train': TRAIN_MEMBERS, 'test': TEST_MEMBERS},
    'source_group': 'SHA256 of normalized caption; exactly one of every label 0..5 in scanned metadata',
    'development': '300 complete train caption groups shuffled from sorted hashes by registered seed',
    'holdout': '100 complete test caption groups excluding every scanned train caption, shuffled by same seeded RNG',
    'partitions': {'fusion': 100, 'calibration': 100, 'threshold': 100, 'public_holdout': 100},
    'stratification': 'Every partition: 70 real, 30 generated; 6 per SD2.1/SDXL/SD3/DALL-E3/Midjourney6',
    'sampling_timing': 'Caption/label metadata only before selection; no detector or image-derived sampling decision',
    'representation': 'Pinned original parquet Image.bytes, no re-encoding and no previews',
    'duplicate_rule': 'Abort without final manifest on exact byte duplicates or caption overlap; never select replacements after inference',
    'scope': 'Restricted shard coverage, public image-origin benchmark; not insurance/business or contest acceptance',
    'upstream_pretraining_overlap_known': False,
}


def h(data):
    return hashlib.sha256(data).hexdigest()


def caption_group(caption):
    return h(re.sub(r'\s+', ' ', caption.strip().lower()).encode('utf-8'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')


def freeze(path, value):
    content = canonical(value)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f'Already registered content disagrees: {path.name}; do not silently overwrite protocol/selection')
    path.write_bytes(content)
    return h(content)


class ParquetRange(RangeFile):
    """Bounded parallel transfers with pinned-URL/offset disk checkpoints."""
    block_bytes = 4 * 1024 * 1024

    def __init__(self, url):
        self.url, self.pos, self.cache = url, 0, {}
        for attempt in range(6):
            try:
                response = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=45)
                response.raise_for_status()
                match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
                if response.status_code != 206 or not match or len(response.content) != 1:
                    raise IOError('Pinned source did not return the requested size probe')
                self.size = int(match.group(1))
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(1 + attempt)
        self.block_directory = ROOT / 'data/public_images_standalone_v1/range_cache'
        self.block_directory.mkdir(parents=True, exist_ok=True)

    def readable(self):
        return True

    def _range(self, bounds):
        start, end = bounds
        count = end - start + 1
        key = hashlib.sha256(f'{self.url}:{start}:{end}'.encode()).hexdigest()
        destination = self.block_directory / (key + '.bin')
        checksum_path = self.block_directory / (key + '.sha256')
        if destination.exists() and destination.stat().st_size == count and checksum_path.exists():
            value = destination.read_bytes()
            if h(value) != checksum_path.read_text(encoding='ascii').strip():
                raise IOError('Pinned source Range cache checksum differs')
            return value
        for attempt in range(6):
            try:
                request = urllib.request.Request(self.url, headers={
                    'Range': f'bytes={start}-{end}', 'User-Agent': 'ZAForensics/1.3 research'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206:
                        raise IOError('Server did not honor bounded Range')
                    observed = response.headers.get('Content-Range', '')
                    if observed != f'bytes {start}-{end}/{self.size}':
                        raise IOError('Range response offset/length differs from pinned source')
                    value = response.read(count + 1)
                if len(value) != count:
                    raise IOError('Incomplete Range response')
                temporary = destination.with_suffix('.tmp')
                temporary.write_bytes(value)
                temporary.replace(destination)
                checksum_path.write_text(h(value), encoding='ascii')
                return value
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(1 + attempt)

    def read(self, count=-1):
        count = min(self.size - self.pos, count if count >= 0 else self.size - self.pos)
        if count <= 0:
            return b''
        start = self.pos
        pieces = [(offset, min(start + count, offset + self.block_bytes) - 1)
                  for offset in range(start, start + count, self.block_bytes)]
        if len(pieces) == 1:
            value = self._range(pieces[0])
        else:
            with ThreadPoolExecutor(max_workers=4) as workers:
                value = b''.join(workers.map(self._range, pieces))
        self.pos += count
        return value


def cached_metadata_scan(output, original_split, members):
    """Cache full row references, not only caption counts, before byte acquisition."""
    destination = output / f'{original_split}_metadata_rows.json'
    specification = {'revision': REVISION, 'repository': REPOSITORY,
                     'original_split': original_split, 'members': members}
    # JSON serialization converts registered tuples to lists.
    specification = json.loads(canonical(specification))
    if destination.exists():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if saved['specification'] != specification:
            raise ValueError('Cached metadata coverage differs from registered protocol')
        print('Reusing source metadata', original_split, flush=True)
        return ({parent: {int(label): row for label, row in records.items()}
                 for parent, records in saved['candidates'].items()},
                set(saved['all_caption_groups']), saved['receipt'])
    candidates, captions, receipt = metadata_scan(original_split, members)
    freeze(destination, {'specification': specification, 'candidates': candidates,
                        'all_caption_groups': sorted(captions), 'receipt': receipt})
    return candidates, captions, receipt


def metadata_scan(original_split, members):
    groups = defaultdict(list)
    receipts = []
    for member, requested in members:
        url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{member}'
        stream = ParquetRange(url)
        parquet = pq.ParquetFile(stream)
        row_groups = requested if requested is not None else list(range(parquet.metadata.num_row_groups))
        offset = 0
        offsets = []
        for rg in range(parquet.metadata.num_row_groups):
            offsets.append(offset)
            offset += parquet.metadata.row_group(rg).num_rows
        for rg in row_groups:
            print('Scanning source metadata', original_split, member, rg, flush=True)
            checkpoint = ROOT / 'data/public_images_standalone_v1' / f'metadata_{h(member.encode())[:16]}_rg{rg}.json'
            expected = {'revision': REVISION, 'member': member, 'row_group': rg,
                        'offset': offsets[rg], 'row_count': parquet.metadata.row_group(rg).num_rows}
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding='utf-8'))
                if saved['source'] != expected:
                    raise ValueError('Metadata row-group checkpoint differs from pinned source')
                table = saved['rows']
            else:
                table = parquet.read_row_group(rg, columns=['Caption', 'Label_A', 'Label_B']).to_pylist()
                if len(table) != expected['row_count']:
                    raise ValueError('Metadata row-group length differs from pinned source')
                freeze(checkpoint, {'source': expected, 'rows': table})
            for index, record in enumerate(table):
                if int(record['Label_A']) != int(record['Label_B'] != 0):
                    raise ValueError('Official binary and source labels disagree')
                groups[caption_group(record['Caption'])].append({**record,
                    'row_index': offsets[rg] + index, 'row_group': rg, 'row_in_group': index,
                    'parquet_member': member, 'source_url': url, 'original_split': original_split})
        receipts.append({'original_split': original_split, 'parquet_member': member,
                         'revision': REVISION, 'url': url, 'parquet_bytes': stream.size,
                         'row_groups_scanned': row_groups, 'parquet_rows': parquet.metadata.num_rows})
    candidates = {parent: {int(record['Label_B']): record for record in group}
                  for parent, group in groups.items()
                  if sorted(int(record['Label_B']) for record in group) == list(range(6))}
    return candidates, set(groups), {
        'source_receipts': receipts, 'complete_distinct_caption_groups': len(candidates),
        'all_caption_groups': len(groups),
        'incomplete_or_duplicated_caption_groups_excluded': len(groups) - len(candidates)}


def select(train, train_caption_set, test):
    rng = random.Random(SEED)
    development = sorted(train)
    rng.shuffle(development)
    development = development[:300]
    heldout = sorted(set(test) - train_caption_set)
    rng.shuffle(heldout)
    heldout = heldout[:100]
    if len(development) != 300 or len(heldout) != 100:
        raise ValueError('Insufficient complete mutually exclusive source groups under registered coverage')
    selection = []
    for split_index, split in enumerate(('fusion', 'calibration', 'threshold', 'public_holdout')):
        parents = development[100 * split_index:100 * (split_index + 1)] if split != 'public_holdout' else heldout
        sources = train if split != 'public_holdout' else test
        assignments = [0] * 70 + [1, 2, 3, 4, 5] * 6
        rng.shuffle(assignments)
        for parent, label in zip(parents, assignments):
            record = sources[parent][label]
            tool, version = TOOLS[label]
            identity = h((record['parquet_member'] + ':' + str(record['row_index'])).encode())[:24]
            selection.append({
                'sample_id': 'defactify_' + identity, 'file_stem': identity,
                'material_type': 'image', 'truth_label': 'genuine' if label == 0 else 'fake',
                'material_class': 'genuine' if label == 0 else 'ai_generated',
                'generator_tool': tool, 'generator_version': version,
                'source_group': parent, 'caption_sha256': parent, 'split': split,
                'source_url': record['source_url'], 'source_revision': REVISION,
                'original_split': record['original_split'], 'parquet_member': record['parquet_member'],
                'parquet_row_index': record['row_index'], 'row_group': record['row_group'],
                'row_in_group': record['row_in_group'],
                'official_label_a': int(record['Label_A']), 'official_label_b': label,
                'caption': record['Caption'], 'metadata_provenance': 'dataset_embedded_original',
                'license': 'Dataset card has no separate license declaration; source rights apply; media not redistributed',
                'annotation_status': 'published_dataset_labels_no_independent_business_adjudication',
                'scope': 'standalone COCO/photos and generated images; not video midpoint frames; not insurance cases',
            })
    if len({row['source_group'] for row in selection}) != len(selection):
        raise ValueError('Source groups overlap')
    return selection


def prepare_training_expansion(output, original_registration):
    """Expand only registered fusion parents, before acquiring extra image bytes."""
    originals = original_registration['selected_rows']
    fusion = [row for row in originals if row['split'] == 'fusion']
    forbidden = {row['source_group'] for row in originals if row['split'] != 'fusion'}
    if len(fusion) != 100 or len({row['source_group'] for row in fusion}) != 100:
        raise ValueError('Fusion parent coverage differs from the frozen 400-row protocol')
    metadata = json.loads((output / 'train_metadata_rows.json').read_text(encoding='utf-8'))
    expansion = []
    for template in fusion:
        parent = template['source_group']
        if parent in forbidden:
            raise ValueError('Training expansion parent overlaps a reserved or calibration parent')
        candidates = metadata['candidates'][parent]
        if sorted(int(label) for label in candidates) != list(range(6)):
            raise ValueError('Registered fusion parent has no complete six-class sibling set')
        for label in range(6):
            record = candidates[str(label)]
            identity = h((record['parquet_member'] + ':' + str(record['row_index'])).encode())[:24]
            tool, version = TOOLS[label]
            row = {**template, 'sample_id': 'defactify_' + identity, 'file_stem': identity,
                'truth_label': 'genuine' if label == 0 else 'fake',
                'material_class': 'genuine' if label == 0 else 'ai_generated',
                'generator_tool': tool, 'generator_version': version,
                'source_url': record['source_url'], 'original_split': record['original_split'],
                'parquet_member': record['parquet_member'], 'parquet_row_index': record['row_index'],
                'row_group': record['row_group'], 'row_in_group': record['row_in_group'],
                'official_label_a': int(record['Label_A']), 'official_label_b': label,
                'caption': record['Caption'], 'development_expansion': True,
                'expansion_parent_original_sample_id': template['sample_id']}
            expansion.append(row)
    registered = {
        'version': 1, 'repository': REPOSITORY, 'revision': REVISION,
        'parent_selection_registration_sha256': h((output / 'selection_registration.json').read_bytes()),
        'selection_rule': 'All six published source-label siblings of the already registered fusion 100 caption groups',
        'selected_rows': expansion, 'rows': 600, 'caption_groups': 100,
        'source_label_counts': {str(label): 100 for label in range(6)},
        'additional_originals': 500, 'original_fusion_rows_included': 100,
        'calibration_threshold_public_holdout_expanded': False,
        'labels_captions_or_filename_used_as_detector_inputs': False,
        'sampling_timing': 'Extra rows frozen from published caption/label metadata before their bytes are acquired or any model inference',
        'scope': 'Training expansion only; does not alter the frozen 70/30 public image benchmark',
        'exact_content_cross_partition_duplicate_policy': 'Do not publish expansion manifest if extra bytes overlap any non-fusion partition',
    }
    freeze(output / 'training_expansion_registration.json', registered)
    return expansion


def acquire_image_bytes(output, selection, progress_name='acquisition_progress.json'):
    by_member = defaultdict(lambda: defaultdict(list))
    for row in selection:
        by_member[row['parquet_member']][row['row_group']].append(row)
    progress_path = output / progress_name
    result = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
    selected_ids = {row['sample_id'] for row in selection}
    if not set(result).issubset(selected_ids):
        raise ValueError('Acquisition progress contains rows outside frozen selection')
    for saved in result.values():
        destination = output / saved['relative_path']
        if not destination.exists() or h(destination.read_bytes()) != saved['sha256']:
            raise ValueError('Previously acquired original-byte file differs from progress receipt')
    result_lock = threading.Lock()

    def acquire_member(item):
        member, grouped_rows = item
        if all(row['sample_id'] in result for rows in grouped_rows.values() for row in rows):
            print('Previously acquired source member verified', member, flush=True)
            return
        stream = ParquetRange(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{member}')
        parquet = pq.ParquetFile(stream)
        for rg, rows in sorted(grouped_rows.items()):
            if all(row['sample_id'] in result for row in rows):
                print('Previously acquired row group verified', member, rg, flush=True)
                continue
            started = time.perf_counter()
            print('Reading selected original image row group', member, rg, 'selected', len(rows), flush=True)
            table = parquet.read_row_group(rg, columns=['Image']).column('Image')
            batch = {}
            for row in rows:
                payload = table[row['row_in_group']].as_py()
                original_bytes = payload['bytes']
                if not isinstance(original_bytes, bytes) or not original_bytes:
                    raise ValueError('Original embedded image bytes unavailable')
                with Image.open(io.BytesIO(original_bytes)) as image:
                    fmt, (width, height) = image.format, image.size
                    image.verify()
                suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(fmt)
                if not suffix:
                    raise ValueError('Unplanned source format: ' + str(fmt))
                filename = row['file_stem'] + suffix
                destination = output / filename
                checksum = h(original_bytes)
                if destination.exists() and h(destination.read_bytes()) != checksum:
                    raise ValueError('Existing content differs from pinned original')
                destination.write_bytes(original_bytes)
                saved = {key: value for key, value in row.items() if key not in ('file_stem', 'row_in_group')}
                saved.update({'relative_path': filename, 'original_embedded_path': payload['path'],
                    'sha256': checksum, 'bytes': len(original_bytes), 'original_width': width,
                    'original_height': height, 'original_format': fmt})
                batch[row['sample_id']] = saved
            del table
            with result_lock:
                result.update(batch)
                temporary = progress_path.with_suffix('.json.tmp')
                temporary.write_bytes(canonical(result))
                temporary.replace(progress_path)
                print('Saved original bytes', len(result), '/', len(selection),
                      'row group seconds', round(time.perf_counter() - started, 2), flush=True)
    # Acquire development originals early while a separate source reader fills
    # the already registered reserved set. Selection never depends on arrival.
    ordered = sorted(by_member.items(), key=lambda item: ('train-' not in item[0], item[0]))
    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(acquire_member, ordered))
    return [result[row['sample_id']] for row in selection]


def verify(output, selection):
    hashes = [row['sha256'] for row in selection]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Exact duplicate content found; selection not published')
    if len({row['source_group'] for row in selection}) != len(selection):
        raise ValueError('Source groups overlap; selection not published')
    for row in selection:
        content = (output / row['relative_path']).read_bytes()
        if len(content) != row['bytes'] or h(content) != row['sha256']:
            raise ValueError('Stored original-byte hash/length mismatch')
        with Image.open(io.BytesIO(content)) as image:
            if image.format != row['original_format'] or image.size != (row['original_width'], row['original_height']):
                raise ValueError('Stored format/dimension mismatch')
            image.verify()
    for split in POLICY['partitions']:
        rows = [row for row in selection if row['split'] == split]
        counts = Counter(row['official_label_b'] for row in rows)
        if counts != Counter({0: 70, 1: 6, 2: 6, 3: 6, 4: 6, 5: 6}):
            raise ValueError('Registered source stratification mismatch')


def publish_training_expansion(output, original_registration, original_manifest):
    selection = prepare_training_expansion(output, original_registration)
    acquired = acquire_image_bytes(output, selection, 'training_expansion_progress.json')
    if len(acquired) != 600 or len({row['sample_id'] for row in acquired}) != 600:
        raise ValueError('Training expansion row identity coverage differs')
    groups = defaultdict(list)
    for row in acquired:
        if row['split'] != 'fusion':
            raise ValueError('Expansion contains a non-fusion image')
        groups[row['source_group']].append(row['official_label_b'])
        content = (output / row['relative_path']).read_bytes()
        if len(content) != row['bytes'] or h(content) != row['sha256']:
            raise ValueError('Training expansion original-byte verification differs')
        with Image.open(io.BytesIO(content)) as image:
            if image.format != row['original_format'] or image.size != (row['original_width'], row['original_height']):
                raise ValueError('Training expansion format/dimension verification differs')
            image.verify()
    if len(groups) != 100 or any(sorted(labels) != list(range(6)) for labels in groups.values()):
        raise ValueError('Training expansion does not contain six siblings per registered fusion parent')
    forbidden_groups = {row['source_group'] for row in original_manifest if row['split'] != 'fusion'}
    forbidden_hashes = {row['sha256'] for row in original_manifest if row['split'] != 'fusion'}
    if set(groups) & forbidden_groups or {row['sha256'] for row in acquired} & forbidden_hashes:
        raise ValueError('Training expansion content or caption group crosses a calibration/threshold/reserved boundary')
    if len({row['sha256'] for row in acquired}) != 600:
        raise ValueError('Training expansion has exact duplicate bytes; group audit required before publishing')
    original_fusion = {row['sample_id']: row for row in original_manifest if row['split'] == 'fusion'}
    by_id = {row['sample_id']: row for row in acquired}
    if not set(original_fusion).issubset(by_id) or any(
            by_id[identity]['sha256'] != row['sha256'] for identity, row in original_fusion.items()):
        raise ValueError('Training expansion differs from the previously registered 100 fusion originals')
    checksum = freeze(output / 'training_expansion_manifest.json', acquired)
    print(json.dumps({'training_expansion_rows': 600, 'caption_groups': 100,
                      'additional_originals': 500, 'manifest_sha256': checksum}, ensure_ascii=False), flush=True)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    output = ROOT / 'data/public_images_standalone_v1'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        original_manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        verify(output, original_manifest)
        print('Existing frozen original-byte selection verified; nothing overwritten', flush=True)
        registration = json.loads((output / 'selection_registration.json').read_text(encoding='utf-8'))
        publish_training_expansion(output, registration, original_manifest)
        return
    protocol_sha = freeze(output / 'acquisition_protocol.json', POLICY)
    response = requests.get(CARD, timeout=30)
    response.raise_for_status()
    card = output / 'dataset_card.md'
    if card.exists() and card.read_bytes() != response.content:
        raise ValueError('Pinned researcher dataset card differs from recorded source')
    card.write_bytes(response.content)
    registration_path = output / 'selection_registration.json'
    if registration_path.exists():
        registration = json.loads(registration_path.read_text(encoding='utf-8'))
        if registration['acquisition_protocol_sha256'] != protocol_sha:
            raise ValueError('Frozen selection disagrees with acquisition protocol')
        selection = registration['selected_rows']
        train_receipt, test_receipt = registration['source_receipts']
        registration_sha = h(registration_path.read_bytes())
        print('Reusing frozen pre-inference selection', len(selection), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=2) as workers:
            train_job = workers.submit(cached_metadata_scan, output, 'train', TRAIN_MEMBERS)
            test_job = workers.submit(cached_metadata_scan, output, 'test', TEST_MEMBERS)
            train, train_caption_set, train_receipt = train_job.result()
            test, _, test_receipt = test_job.result()
        print('Complete source candidates', len(train), len(test), 'eligible reserved', len(set(test) - train_caption_set), flush=True)
        selection = select(train, train_caption_set, test)
        registration_sha = freeze(registration_path, {
            'acquisition_protocol_sha256': protocol_sha,
            'source_receipts': [train_receipt, test_receipt], 'selected_rows': selection})
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    prepare_training_expansion(output, registration)
    acquired = acquire_image_bytes(output, selection)
    verify(output, acquired)
    manifest_sha = freeze(output / 'manifest.json', acquired)
    provenance = {
        'repository': REPOSITORY, 'revision': REVISION, 'dataset_card_sha256': h(response.content),
        'paper': 'https://arxiv.org/abs/2601.00553', 'seed': SEED, 'selection_sha256': manifest_sha,
        'acquisition_protocol_sha256': protocol_sha, 'selection_registration_sha256': registration_sha,
        'policy': POLICY, 'source_receipts': [train_receipt, test_receipt], 'rows': len(acquired),
        'split_counts': dict(Counter(row['split'] for row in acquired)),
        'generator_families': ['Stable Diffusion', 'DALL-E', 'Midjourney'],
        'representation': 'original Image.bytes extracted from pinned parquet; no preview conversion, no re-encoding',
        'labels_and_captions_used_as_live_detector_inputs': False,
        'upstream_pretraining_overlap_known': False, 'complete_dataset_random_sample': False,
        'formal_contest_acceptance': False,
        'verification': {'stored_hashes': True, 'formats_and_dimensions': True,
                         'exact_content_duplicates': 0, 'source_groups_overlap': 0,
                         'source_stratification': True},
    }
    freeze(output / 'provenance.json', provenance)
    print(json.dumps({'selected': len(acquired), 'partitions': provenance['split_counts'],
                      'families': provenance['generator_families'], 'manifest_sha256': manifest_sha}, ensure_ascii=False), flush=True)
    publish_training_expansion(output, registration, acquired)


if __name__ == '__main__':
    main()
