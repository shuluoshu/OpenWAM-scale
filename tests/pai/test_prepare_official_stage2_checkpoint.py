import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/pai/prepare_official_stage2_checkpoint.py'


def prepare(source, output):
    assert SCRIPT.exists(), 'official artifact preparation utility must exist'
    spec = importlib.util.spec_from_file_location('official_stage2', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_checkpoint(source, output)


@pytest.fixture
def release(tmp_path):
    root = tmp_path / 'release'
    tokenizer = root / 'tokenizer/google/tiny'
    tokenizer.mkdir(parents=True)
    (tokenizer / 'tokenizer.json').write_text('{"identity": "original"}')
    config = {'model': {'architecture': {'action_dim': 80, 'state_dim': 80, 'other': True},
                        'video_backbone': {'tokenizer': {'subdir': 'tokenizer/google/tiny', 'kwargs': {'seq_len': 512}}}},
              'training': {'learning_rate': 0.001}, 'literal': '${unchanged}'}
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    state = {'action_backbone.action_encoder.weight': torch.arange(240).reshape(3, 80).to(torch.bfloat16),
             'action_backbone.action_decoder.weight': torch.arange(240).reshape(80, 3).float(),
             'action_backbone.action_decoder.bias': torch.arange(80).float(),
             'proprio_encoder.weight': torch.arange(160).reshape(2, 80).float(),
             'untouched': torch.tensor([[-3.0, 7.0]])}
    source = root / 'checkpoint_step_154000.safetensors'
    save_file(state, source, metadata={'official': 'yes', 'nested': '{"x": 1}'})
    return source, state, config


def test_preserves_values_metadata_config_and_records_full_hashes(release, tmp_path):
    source, state, config = release
    original_bytes = source.read_bytes()
    output = tmp_path / 'derived'
    manifest = prepare(source, output)
    converted = load_file(output / source.name)
    for key, old in state.items():
        actual = converted[key]
        assert actual.dtype == old.dtype
        if key == 'untouched':
            assert torch.equal(actual, old)
        else:
            axis = 1 if key in ('action_backbone.action_encoder.weight', 'proprio_encoder.weight') else 0
            assert actual.shape[axis] == 112
            assert torch.equal(actual.narrow(axis, 0, 80), old)
            assert torch.count_nonzero(actual.narrow(axis, 80, 32)) == 0
    with safe_open(output / source.name, framework='pt') as handle:
        assert handle.metadata() == {'official': 'yes', 'nested': '{"x": 1}'}
    expected_config = copy.deepcopy(config)
    expected_config['model']['architecture'].update(action_dim=112, state_dim=112)
    assert yaml.safe_load((output / 'config.yaml').read_text()) == expected_config
    assert source.read_bytes() == original_bytes
    assert manifest == json.loads((output / 'provenance.json').read_text())
    assert len(manifest['migration']['projections']) == 4
    for role, base in [('source', source.parent), ('output', output)]:
        for name, identity in manifest[role]['files'].items():
            data = (base / name).read_bytes()
            assert identity == {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
    before = {p: p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}
    assert prepare(source, output) == manifest
    assert before == {p: p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('role,name', [(r, n) for r in ('source', 'output') for n in
    ('checkpoint_step_154000.safetensors', 'config.yaml', 'tokenizer/google/tiny/tokenizer.json')])
def test_reuse_rejects_modified_files(release, tmp_path, role, name):
    source, _, _ = release
    output = tmp_path / 'derived'
    prepare(source, output)
    path = (source.parent if role == 'source' else output) / name
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError):
        prepare(source, output)


@pytest.mark.parametrize('change', ['missing', 'inconsistent', 'already112', 'rank', 'config'])
def test_requires_complete_exact_width80_release(release, tmp_path, change):
    source, state, _ = release
    if change == 'missing':
        del state['action_backbone.action_decoder.bias']
    elif change == 'inconsistent':
        state['proprio_encoder.weight'] = torch.zeros(2, 94)
    elif change == 'already112':
        state['action_backbone.action_encoder.weight'] = torch.zeros(3, 112)
    elif change == 'rank':
        state['proprio_encoder.weight'] = torch.zeros(1, 2, 80)
    else:
        cfg = yaml.safe_load((source.parent / 'config.yaml').read_text())
        cfg['model']['architecture']['state_dim'] = 94
        (source.parent / 'config.yaml').write_text(yaml.safe_dump(cfg))
    save_file(state, source)
    output = tmp_path / 'derived'
    with pytest.raises(ValueError):
        prepare(source, output)
    assert not output.exists()


def test_rejects_partial_artifact_and_extra_files(release, tmp_path):
    source, _, _ = release
    output = tmp_path / 'partial'
    output.mkdir()
    with pytest.raises(ValueError):
        prepare(source, output)
    complete = tmp_path / 'complete'
    prepare(source, complete)
    (complete / 'unexpected').write_text('x')
    with pytest.raises(ValueError):
        prepare(source, complete)


@pytest.mark.parametrize('subdir', ['../outside', '/tmp/tokenizer', '.', 'config.yaml'])
def test_rejects_unsafe_tokenizer_location(release, tmp_path, subdir):
    source, _, config = release
    config['model']['video_backbone']['tokenizer']['subdir'] = subdir
    (source.parent / 'config.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        prepare(source, tmp_path / 'derived')


def test_rejects_symlinks_and_source_output_nesting(release, tmp_path):
    source, _, _ = release
    with pytest.raises(ValueError):
        prepare(source, source.parent / 'derived')
    with pytest.raises(ValueError):
        prepare(source, tmp_path)
    tokenizer = source.parent / 'tokenizer/google/tiny/tokenizer.json'
    tokenizer.unlink()
    tokenizer.symlink_to(source.parent / 'config.yaml')
    with pytest.raises(ValueError):
        prepare(source, tmp_path / 'derived')


@pytest.mark.parametrize('failure', ['write', 'source_mutation'])
def test_failed_conversion_never_publishes_partial_directory(release, tmp_path, monkeypatch, failure):
    source, _, _ = release
    spec = importlib.util.spec_from_file_location('official_stage2_transaction', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_save = module.save_file

    def save_with_failure(state, path, **kwargs):
        if failure == 'write':
            Path(path).write_bytes(b'partial')
            raise OSError('simulated full filesystem')
        original_save(state, path, **kwargs)
        tokenizer = source.parent / 'tokenizer/google/tiny/tokenizer.json'
        tokenizer.write_text('mutated while serializing')

    monkeypatch.setattr(module, 'save_file', save_with_failure)
    output = tmp_path / 'derived'
    with pytest.raises((OSError, ValueError)):
        module.prepare_checkpoint(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.derived.artifact-*'))


def test_preserves_source_payload_on_cuda_forbidden_host(release, tmp_path, monkeypatch):
    source, _, _ = release
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('GPU initialized'))
    prepare(source, tmp_path / 'derived')


def test_provenance_matches_config_changed_before_first_hash(release, tmp_path, monkeypatch):
    source, _, _ = release
    spec = importlib.util.spec_from_file_location('official_stage2_snapshot', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_inventory = module._inventory
    first = True

    def inventory_with_replacement(directory, names):
        nonlocal first
        if first:
            first = False
            path = source.parent / 'config.yaml'
            cfg = yaml.safe_load(path.read_text())
            cfg['training']['learning_rate'] = 0.5
            path.write_text(yaml.safe_dump(cfg))
        return original_inventory(directory, names)

    monkeypatch.setattr(module, '_inventory', inventory_with_replacement)
    output = tmp_path / 'derived'
    module.prepare_checkpoint(source, output)
    assert yaml.safe_load((output / 'config.yaml').read_text())['training']['learning_rate'] == 0.5


def test_reuse_rejects_source_changed_during_output_verification(release, tmp_path, monkeypatch):
    source, _, _ = release
    output = tmp_path / 'derived'
    prepare(source, output)
    spec = importlib.util.spec_from_file_location('official_stage2_reuse_race', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_inventory = module._inventory

    def inventory_with_replacement(directory, names):
        result = original_inventory(directory, names)
        if directory == output.resolve():
            path = source.parent / 'tokenizer/google/tiny/tokenizer.json'
            path.write_text('changed during output verification')
        return result

    monkeypatch.setattr(module, '_inventory', inventory_with_replacement)
    with pytest.raises(ValueError, match='Source files changed'):
        module.prepare_checkpoint(source, output)


def test_publication_never_replaces_concurrent_empty_directory(release, tmp_path, monkeypatch):
    source, _, _ = release
    spec = importlib.util.spec_from_file_location('official_stage2_publish', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'derived'
    original_publish = module._publish_noreplace
    inode = None

    def concurrent_destination(staging, destination):
        nonlocal inode
        destination.mkdir()
        inode = destination.stat().st_ino
        original_publish(staging, destination)

    monkeypatch.setattr(module, '_publish_noreplace', concurrent_destination)
    with pytest.raises(FileExistsError):
        module.prepare_checkpoint(source, output)
    assert output.stat().st_ino == inode
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob('.derived.artifact-*'))


def test_publication_uses_owned_relative_symlink_and_rejects_foreign_targets(release, tmp_path):
    source, _, _ = release
    output = tmp_path / 'derived'
    prepare(source, output)
    assert output.is_symlink()
    assert not Path(output.readlink()).is_absolute()
    assert output.readlink().name.startswith('.derived.artifact-')
    foreign = tmp_path / 'foreign'
    foreign.symlink_to(output.resolve(), target_is_directory=True)
    with pytest.raises(ValueError, match='owned'):
        prepare(source, foreign)
    dangling = tmp_path / 'dangling'
    dangling.symlink_to('.dangling.artifact-missing0')
    with pytest.raises(ValueError):
        prepare(source, dangling)


def test_postpublication_fsync_failure_keeps_complete_backing_directory(release, tmp_path, monkeypatch):
    source, _, _ = release
    spec = importlib.util.spec_from_file_location('official_stage2_fsync', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_publish, original_fsync = module._publish_noreplace, module.os.fsync
    published = False

    def publish(staging, output):
        nonlocal published
        original_publish(staging, output)
        published = True

    def fsync(fd):
        if published:
            raise OSError('simulated parent fsync failure')
        original_fsync(fd)

    monkeypatch.setattr(module, '_publish_noreplace', publish)
    monkeypatch.setattr(module.os, 'fsync', fsync)
    output = tmp_path / 'derived'
    with pytest.raises(OSError, match='parent fsync'):
        module.prepare_checkpoint(source, output)
    assert output.is_symlink() and output.is_dir()
    assert (output / 'provenance.json').is_file()
    prepare(source, output)
