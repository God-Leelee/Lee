"""Register and acquire 100 previously unused original GenVideo videos.

Only archive filenames and prior-source provenance determine selection. No
detector, decoded frames, duration, quality, or old predictions are consulted.
GenVideo's master archive is mutable: record its actual HTTP headers, size and
central-directory hash, rather than pretending that the URL is a revision.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import threading
import zipfile
import zlib

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fetch_public_videos import RangeFile, URL

OUTPUT = ROOT / 'data/public_genvideo_independent_v1'
SEED = 102413
TOOLS = ('Crafter', 'Gen2', 'HotShot', 'Lavie', 'ModelScope', 'Show_1', 'Sora')
PROVENANCE_INPUTS = (
    'data/public_genvideo/manifest.json', 'data/public_genvideo_v1/manifest.json',
    'data/public_frames_v1/manifest.json', 'data/known_edits/manifest.json',
    'data/known_video_edits/manifest.json',
)
SCOPE = 'Unused-source public GenVideo regression/reserved set; not an insurance/business or contest acceptance certificate.'
TOTAL_ROWS = 100


def digest(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')


def freeze(path, value):
    content = canonical(value)
    if path.exists() and path.read_bytes() != content:
        raise ValueError('Frozen registration differs: ' + path.name)
    path.write_bytes(content)
    return digest(content)


def walks(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walks(item)
    elif isinstance(value, list):
        for item in value:
            yield from walks(item)


def suffix(member):
    match = re.search(r'_(\d+)\.mp4$', member, re.IGNORECASE)
    return str(int(match.group(1))) if match else None


def prior_sources():
    members, checksums, fake_suffixes, receipts = set(), set(), set(), []
    for relative in PROVENANCE_INPUTS:
        path = ROOT / relative
        content = path.read_bytes()
        value = json.loads(content)
        receipts.append({'relative_path': relative, 'sha256': digest(content),
                         'rows': len(value) if isinstance(value, list) else None})
        for record in walks(value):
            for key in ('archive_member', 'source_group'):
                member = record.get(key)
                if isinstance(member, str) and '/GenVideo-Val/' in '/' + member and member.lower().endswith('.mp4'):
                    members.add(member.replace('\\', '/'))
                    if '/Fake/' in member:
                        number = suffix(member)
                        if number is not None:
                            fake_suffixes.add(number)
            for key in ('sha256', 'parent_sha256'):
                values = record.get(key)
                values = values if isinstance(values, list) else [values]
                checksums.update(v for v in values if isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v))
    return members, checksums, fake_suffixes, receipts


def http_identity():
    response = requests.get(URL, headers={'Range': 'bytes=0-0'}, timeout=60)
    response.raise_for_status()
    observed = response.headers.get('Content-Range', '')
    match = re.fullmatch(r'bytes 0-0/(\d+)', observed)
    if response.status_code != 206 or not match or len(response.content) != 1:
        raise ValueError('Original archive size probe did not honor Range')
    return {'source_url': URL, 'archive_bytes': int(match.group(1)),
            'http_etag': response.headers.get('ETag'),
            'http_last_modified': response.headers.get('Last-Modified')}


def directory_identity(stream, archive):
    end = zipfile._EndRecData(stream)
    if end is None:
        raise ValueError('Archive end-of-directory record is unavailable')
    size, offset = int(end[zipfile._ECD_SIZE]), int(archive.start_dir)
    stream.seek(offset)
    content = stream.read(size)
    if len(content) != size:
        raise ValueError('Incomplete central-directory bytes')
    return {'central_directory_offset': offset, 'central_directory_bytes': size,
            'central_directory_sha256': digest(content)}, content


def source_tool(member):
    parts = member.split('/')
    if '__MACOSX' in parts or Path(member).name.startswith('.') or not member.lower().endswith('.mp4'):
        return None
    if 'Real' in parts:
        return 'real_MSRVTT'
    if 'Fake' in parts:
        index = parts.index('Fake') + 1
        return parts[index] if index < len(parts) and parts[index] in TOOLS else None
    return None


def register(snapshot, items, old_members, old_suffixes, source_receipts):
    rng = random.Random(SEED)
    groups = {tool: [] for tool in ('real_MSRVTT',) + TOOLS}
    for item in items:
        tool = source_tool(item.filename)
        if tool is None or item.filename in old_members:
            continue
        if tool != 'real_MSRVTT' and (suffix(item.filename) is None or suffix(item.filename) in old_suffixes):
            continue
        groups[tool].append(item)
    for tool, group in groups.items():
        group.sort(key=lambda item: item.filename)
        rng.shuffle(group)
    if len(groups['real_MSRVTT']) < 70:
        raise ValueError('Insufficient unused genuine source filenames')
    selected = [(item, 'real_MSRVTT') for item in groups['real_MSRVTT'][:70]]
    available = [tool for tool in TOOLS if groups[tool]]
    used_suffixes = set()
    counts = Counter()
    while sum(counts.values()) < 30:
        advanced = False
        for tool in available:
            while groups[tool] and suffix(groups[tool][-1].filename) in used_suffixes:
                groups[tool].pop()
            if not groups[tool]:
                continue
            item = groups[tool].pop()
            selected.append((item, tool))
            used_suffixes.add(suffix(item.filename))
            counts[tool] += 1
            advanced = True
            if sum(counts.values()) == 30:
                break
        if not advanced:
            raise ValueError('Insufficient unused distinct conservative fake numeric-ID groups')
    if len(counts) < 3:
        raise ValueError('Fewer than three unused generation source families')
    rows = []
    for item, tool in selected:
        identity = digest(item.filename.encode('utf-8'))[:24]
        rows.append({'sample_id': 'genvideo_new_' + identity, 'relative_path': identity + '.mp4',
            'material_type': 'video', 'truth_label': 'genuine' if tool == 'real_MSRVTT' else 'fake',
            'material_class': 'genuine_public_video' if tool == 'real_MSRVTT' else 'ai_generated_public_video',
            'generator_tool': 'not_applicable' if tool == 'real_MSRVTT' else tool,
            'generator_version': 'not_applicable' if tool == 'real_MSRVTT' else 'unknown_original_release_label',
            'split': 'public_holdout', 'source_group': item.filename, 'archive_member': item.filename,
            'conservative_fake_numeric_id': None if tool == 'real_MSRVTT' else suffix(item.filename),
            'source_url': URL, 'uncompressed_bytes': item.file_size,
            'compressed_bytes': item.compress_size, 'archive_crc32': f'{item.CRC:08x}',
            'zip_compression_method': item.compress_type,
            'source_caption_or_prompt_pairing_known': False,
            'spatial_temporal_localization_ground_truth': 'unknown_not_provided_by_release',
            'labels_or_filename_used_as_live_detector_inputs': False,
            'license': 'See GenVideo/MSRVTT original source; original media are not redistributed',
            'scope': SCOPE})
    registration = {'version': 1, 'seed': SEED, 'archive_snapshot': snapshot,
        'prior_source_manifests': source_receipts, 'selected_rows': rows, 'rows': 100,
        'truth_counts': {'genuine': 70, 'fake': 30}, 'generator_counts': dict(counts),
        'excluded_prior_archive_members': len(old_members), 'excluded_prior_fake_numeric_ids': len(old_suffixes),
        'selection_rule': 'Archive metadata only: deterministic shuffle of unused originals; 70 real and 30 fake round robin across available source families; old and new fake numeric suffix IDs excluded across families',
        'sampling_timing': 'Registered before reading selected video bytes, decoding any frame or calling any model',
        'duplicate_rule': 'Abort final manifest on SHA overlap with any prior source, or exact duplicate within selection; no detector-dependent replacements',
        'all_original_msrvt_caption_groups_known': False,
        'fake_numeric_suffix_means_same_prompt_proven': False,
        'upstream_pretraining_overlap_known': False, 'formal_contest_acceptance': False,
        'mutable_master_url': True, 'scope': SCOPE}
    freeze(OUTPUT / 'selection_registration.json', registration)
    return registration


def verify(rows, registration, old_members, old_hashes, old_suffixes):
    if len(rows) != 100 or Counter(row['truth_label'] for row in rows) != Counter({'genuine': 70, 'fake': 30}):
        raise ValueError('Frozen 70/30 row count differs')
    selected = {row['sample_id']: row for row in registration['selected_rows']}
    if len(selected) != 100 or set(selected) != {row['sample_id'] for row in rows}:
        raise ValueError('Selected original identities differ from pre-inference registration')
    if len({row['source_group'] for row in rows}) != 100 or len({row['sha256'] for row in rows}) != 100:
        raise ValueError('New original source or SHA is duplicated')
    if {row['source_group'] for row in rows} & old_members or {row['sha256'] for row in rows} & old_hashes:
        raise ValueError('New video overlaps a previously used original source/SHA')
    fake_ids = [row['conservative_fake_numeric_id'] for row in rows if row['truth_label'] == 'fake']
    if len(set(fake_ids)) != 30 or set(fake_ids) & old_suffixes:
        raise ValueError('Conservative fake numeric source groups overlap')
    for row in rows:
        expected = selected[row['sample_id']]
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError('Acquired row metadata differs from frozen registration')
        data = (OUTPUT / row['relative_path']).read_bytes()
        if digest(data) != row['sha256'] or len(data) != row['bytes'] or len(data) != row['uncompressed_bytes']:
            raise ValueError('Saved original video bytes differ')
        if f'{zlib.crc32(data) & 0xffffffff:08x}' != row['archive_crc32']:
            raise ValueError('Saved original video CRC differs')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old_members, old_hashes, old_suffixes, source_receipts = prior_sources()
    identity = http_identity()
    stream = RangeFile(URL)
    archive = zipfile.ZipFile(stream)
    directory, original_directory = directory_identity(stream, archive)
    if identity['archive_bytes'] != stream.size:
        raise ValueError('Archive size changed between source probes')
    snapshot = {**identity, **directory}
    path = OUTPUT / 'selection_registration.json'
    if path.exists():
        registration = json.loads(path.read_text(encoding='utf-8'))
        if registration['archive_snapshot'] != snapshot or registration['prior_source_manifests'] != source_receipts:
            raise ValueError('Archive fingerprint/prior-source boundary changed since registration')
        print('Reusing frozen new-source registration', flush=True)
    else:
        registration = register(snapshot, archive.infolist(), old_members, old_suffixes, source_receipts)
        (OUTPUT / 'central_directory.bin').write_bytes(original_directory)
        print('Registered unused videos', registration['truth_counts'], registration['generator_counts'], flush=True)
    if (OUTPUT / 'manifest.json').exists():
        verify(json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8')), registration, old_members, old_hashes, old_suffixes)
        print('Existing new-source original video manifest verified', flush=True)
        return
    progress_path = OUTPUT / 'acquisition_progress.json'
    acquired = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
    selected_ids = {row['sample_id'] for row in registration['selected_rows']}
    if not set(acquired).issubset(selected_ids):
        raise ValueError('Checkpoint contains an unregistered row')
    for row in acquired.values():
        data = (OUTPUT / row['relative_path']).read_bytes()
        if digest(data) != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('Checkpoint original bytes differ')
    local = threading.local()
    result_lock = threading.Lock()
    original_small_range_cache = dict(stream.cache)

    def acquire(row):
        if row['sample_id'] in acquired:
            return
        if not hasattr(local, 'archive'):
            worker_stream = RangeFile(URL)
            if worker_stream.size != snapshot['archive_bytes']:
                raise ValueError('Original archive size changed during acquisition')
            worker_stream.cache.update(original_small_range_cache)
            local.archive = zipfile.ZipFile(worker_stream)
        info = local.archive.getinfo(row['archive_member'])
        if info.file_size != row['uncompressed_bytes'] or f'{info.CRC:08x}' != row['archive_crc32']:
            raise ValueError('Original archive member metadata changed')
        data = local.archive.read(info)  # Python checks the registered upstream ZIP CRC.
        checksum = digest(data)
        if checksum in old_hashes or f'{zlib.crc32(data) & 0xffffffff:08x}' != row['archive_crc32']:
            raise ValueError('Unused filename contains old source bytes, or source CRC differs')
        destination = OUTPUT / row['relative_path']
        if destination.exists() and digest(destination.read_bytes()) != checksum:
            raise ValueError('Previously saved pinned-source original differs')
        destination.write_bytes(data)
        saved = {**row, 'sha256': checksum, 'bytes': len(data), 'zip_crc_verified': True,
                 'original_video_reencoded': False}
        with result_lock:
            if any(existing['sha256'] == checksum for existing in acquired.values()):
                raise ValueError('Exact duplicate original video within frozen selection')
            acquired[row['sample_id']] = saved
            temporary = progress_path.with_suffix('.json.tmp')
            temporary.write_bytes(canonical(acquired))
            temporary.replace(progress_path)
            print('Saved original video', len(acquired), '/', TOTAL_ROWS, row['generator_tool'], len(data), flush=True)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(acquire, registration['selected_rows']))
    result = [acquired[row['sample_id']] for row in registration['selected_rows']]
    verify(result, registration, old_members, old_hashes, old_suffixes)
    final_stream = RangeFile(URL)
    final_archive = zipfile.ZipFile(final_stream)
    final_directory, _ = directory_identity(final_stream, final_archive)
    if {**http_identity(), **final_directory} != snapshot:
        raise ValueError('Mutable source archive changed before acquisition completed')
    checksum = freeze(OUTPUT / 'manifest.json', result)
    freeze(OUTPUT / 'provenance.json', {'version': 1, 'source_url': URL, 'archive_snapshot': snapshot,
        'selection_registration_sha256': digest(path.read_bytes()), 'manifest_sha256': checksum,
        'seed': SEED, 'rows': TOTAL_ROWS, 'truth_counts': registration['truth_counts'],
        'generator_counts': registration['generator_counts'], 'source_receipts': source_receipts,
        'excluded_prior_archive_members': len(old_members), 'excluded_prior_sha256_values': len(old_hashes),
        'excluded_prior_fake_numeric_ids': len(old_suffixes), 'source_group_overlap': 0,
        'exact_prior_sha_overlap': 0, 'new_selection_duplicate_sha': 0,
        'original_archive_crc_verified_for_all': True, 'source_fingerprint_unchanged_after_download': True,
        'no_model_called': True, 'no_decode_or_quality_based_sampling': True,
        'source_caption_pairing_known': False, 'fake_numeric_id_pairing_proven': False,
        'upstream_pretraining_overlap_known': False, 'original_bytes_reencoded': False,
        'formal_contest_acceptance': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': SCOPE})
    print(json.dumps({'original_videos': len(result), 'truth_counts': registration['truth_counts'],
        'generator_counts': registration['generator_counts'], 'manifest_sha256': checksum,
        'bytes': sum(row['bytes'] for row in result)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
