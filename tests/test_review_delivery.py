"""Release integrity checks fail before any checkpoint deserialization."""
import hashlib
import json

import pytest

from scripts import verify_assets


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_assets, 'ROOT', tmp_path)
    model = tmp_path / 'models/small.pth'
    model.parent.mkdir()
    model.write_bytes(b'not-a-pickle; only hashed')
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    config = {'configuration_id': 'test-review', 'temporal_checkpoint': 'models/small.pth',
              'temporal_checkpoint_sha256': sha}
    lock = {'profile': 'test-review', 'weights': [{'path': 'models/small.pth', 'sha256': sha}], 'repositories': []}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'models/runtime_assets.json').write_text(json.dumps(lock))
    return tmp_path, model, config, lock


def test_known_assets_verified_without_deserialization(assets):
    assert verify_assets.verify()['passed']


def test_tampered_checkpoint_rejected(assets):
    assets[1].write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        verify_assets.verify()


def test_configuration_cannot_silently_select_unlocked_weights(assets):
    root, _, config, _ = assets
    config['temporal_checkpoint_sha256'] = '0' * 64
    (root / 'config.json').write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match='Configured checkpoint'):
        verify_assets.verify()


def test_asset_lock_cannot_escape_project(assets):
    root, _, _, lock = assets
    lock['weights'].append({'path': '../private.pth', 'sha256': '0' * 64})
    (root / 'models/runtime_assets.json').write_text(json.dumps(lock))
    with pytest.raises(RuntimeError, match='Missing/unsafe weight'):
        verify_assets.verify()
