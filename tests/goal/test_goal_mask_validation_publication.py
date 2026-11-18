"""Fault injection for target-only diagnostic generations; no GPU/assets needed."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('fail_manifest', [False, True])
def test_interrupted_target_encoding_preserves_referenced_rows(tmp_path, fail_manifest):
    path = Path(__file__).resolve().parents[2] / 'scripts/evaluation/validate_goal_observation_fix.py'
    spec = importlib.util.spec_from_file_location('goal_fix_validator', path)
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    old = tmp_path / 'val/000008.pt'; old.parent.mkdir()
    torch.save({'target': torch.zeros(2), 'pending': True}, old)
    checksum = hashlib.sha256(old.read_bytes()).hexdigest()
    entry = {'file': 'val/000008.pt', 'source_index': 8}
    manifest = {'records': {'val': [entry]}, 'files': {entry['file']: checksum}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    def publish(path, value):
        if fail_manifest: raise OSError('injected publication failure')
        temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value)); temporary.replace(path)
    writer = SimpleNamespace(atomic_torch=lambda path, row: torch.save(row, path), atomic_json=publish)
    row = {'target': torch.ones(2), 'pending': False}
    if fail_manifest:
        with pytest.raises(OSError, match='injected'):
            validator.publish_target_encoding(tmp_path, manifest, entry, row, writer)
    else:
        validator.publish_target_encoding(tmp_path, manifest, entry, row, writer)
    published = json.loads((tmp_path / 'manifest.json').read_text())
    name = published['records']['val'][0]['file']; actual = tmp_path / name
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == published['files'][name]
    assert hashlib.sha256(old.read_bytes()).hexdigest() == checksum
    assert torch.load(actual, weights_only=True)['pending'] == fail_manifest
