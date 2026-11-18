"""The production snapshot is complete, immutable and usable without old code directories."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shlex

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'training/stage1_longtrain_16gpu'


def load(name):
    path = BUNDLE / (name + '.py')
    assert path.is_file(), f'Missing repository tool: {path}'
    spec = importlib.util.spec_from_file_location('snapshot_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_detects_tampering_and_missing_live_source(tmp_path):
    verifier = load('verify_snapshot')
    source = tmp_path / 'live.py'
    source.write_text('verified source\n')
    data = source.read_bytes()
    (tmp_path / 'copied.py').write_bytes(data)
    (tmp_path / 'snapshot.json').write_text(json.dumps({'files': {
        'copied.py': {'source_path': str(source), 'sha256': hashlib.sha256(data).hexdigest(),
                      'size': len(data)}}}))
    assert verifier.verify(tmp_path, check_sources=True)['ok']
    source.unlink()
    result = verifier.verify(tmp_path, check_sources=True)
    assert not result['ok'] and 'source' in result['errors'][0]
    assert verifier.verify(tmp_path)['ok']
    (tmp_path / 'copied.py').write_text('modified\n')
    assert not verifier.verify(tmp_path)['ok']


def test_config_preparation_preserves_scientific_settings_and_existing_files(tmp_path):
    tool = load('prepare_config')
    original = json.loads((BUNDLE / 'runtime/G.json').read_text())
    model = json.loads((BUNDLE / 'runtime/G_model.json').read_text())
    target = tmp_path / 'config with spaces'
    tool.prepare(target, code_dir=ROOT)
    actual = json.loads((target / 'G.json').read_text())
    assert actual['code_dir'] == str(ROOT)
    assert actual['data_pipeline'] == original['data_pipeline']
    assert actual['global_batch_size'] == original['global_batch_size'] == 2560
    assert json.loads((target / 'G_model.json').read_text()) == model
    assert Path(actual['prepared_g']).is_absolute()
    assert Path(actual['prepared_cstar']).is_absolute()
    before = (target / 'G.json').read_bytes()
    import pytest
    with pytest.raises(FileExistsError):
        tool.prepare(target, code_dir=ROOT)
    assert (target / 'G.json').read_bytes() == before
    relocated = tmp_path / 'relocated'
    tool.prepare(relocated, code_dir=ROOT, prepared_g=tmp_path / 'g-data',
                 prepared_cstar=tmp_path / 'c-data')
    config = json.loads((relocated / 'G.json').read_text())
    assert config['data_pipeline']['validation_g'] == config['prepared_g']
    assert config['data_pipeline']['validation_cstar'] == config['prepared_cstar']


def test_bundled_runner_and_helpers_match_live_contract():
    result = load('verify_snapshot').verify(BUNDLE)
    assert result['ok'], result
    manifest = json.loads((BUNDLE / 'snapshot.json').read_text())
    contract = json.loads((BUNDLE / 'provenance/G.contract.json').read_text())
    for relative, expected in contract['runner'].items():
        assert manifest['files']['runtime/' + relative]['sha256'] == expected
    for relative, expected in contract['source']['reviewed_helpers'].items():
        assert manifest['files']['reviewed/' + relative]['sha256'] == expected


def test_repository_entry_imports_bundled_helpers_and_keeps_native_cli():
    program = """
import importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('repository_entry', root/'entrypoint.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
trainer = entry.load_trainer()
d = trainer.d
assert d.REVIEWED == root/'reviewed'
assert pathlib.Path(d.load_reviewed().__file__).resolve().is_relative_to(root)
# bind_code happens before native C* is imported, as in the production runner.
d.load_reviewed().base.bind_code(pathlib.Path(sys.argv[2]))
assert pathlib.Path(d.load_cstar().__file__).resolve().is_relative_to(root)
assert trainer.runner_identity() == json.loads((root/'provenance/G.contract.json').read_text())['runner']
assert trainer.source_identity(sys.argv[2])['reviewed_helpers'] == json.loads((root/'provenance/G.contract.json').read_text())['source']['reviewed_helpers']
assert trainer.source_identity(sys.argv[2])['pinned']['python_tree_sha256'] == json.loads((root/'snapshot.json').read_text())['core_python_tree_sha256']
# Both import-time defaults must use the selected clone, not the old live core.
import frozen_metadata_view, raw_metadata_overlay, data_pipeline
assert pathlib.Path(data_pipeline.DEFAULT_CODE) == pathlib.Path(sys.argv[2])
# Explicitly binding a different directory must still fail.
try:
    data_pipeline.bind_native('/this-is-a-different-code-directory')
except ValueError:
    pass
else:
    raise AssertionError('explicit code-directory guard was bypassed')
print('bundled imports and native identities verified')
"""
    result = subprocess.run([sys.executable, '-c', program, str(BUNDLE), str(ROOT)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([sys.executable, str(BUNDLE / 'entrypoint.py'), '--help'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--resume' in result.stdout and '--stop-after' in result.stdout


def test_launch_print_only_quotes_paths_without_executing_python(tmp_path):
    import os
    environment = {**os.environ, 'TRAIN_PYTHON': str(tmp_path / 'not an executable'), 'MASTER_PORT': '30682'}
    config = str(tmp_path / "config 'quoted' $(false).json")
    output = str(tmp_path / 'new output')
    checkpoint = str(tmp_path / 'resume checkpoint.json')
    result = subprocess.run(['bash', str(BUNDLE / 'launch_g_node.sh'), '--print-command',
                             '1', '127.0.0.1', config, output, '--resume', checkpoint],
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    assert argv[0] == environment['TRAIN_PYTHON']
    assert argv[argv.index('--node_rank') + 1] == '1'
    assert argv[argv.index('--master_port') + 1] == '30682'
    assert argv[argv.index('--config') + 1] == config
    assert argv[argv.index('--output') + 1] == output
    assert argv[-2:] == ['--resume', checkpoint]


def test_spawn_unpickles_frozen_metadata_using_clone_without_loading_live_core(tmp_path):
    fixture = tmp_path / 'spawn_fixture.py'
    fixture.write_text('''
import importlib.util, multiprocessing, pathlib, runpy
ENTRY = pathlib.Path(%r)
if __name__ == '__mp_main__':
    # spawn re-executes the production entrypoint this way before unpickling.
    runpy.run_path(str(ENTRY), run_name='__mp_main__')
if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('entry', ENTRY)
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    entry.load_trainer()
    import frozen_metadata_view
    reader = frozen_metadata_view.FrozenMetadataGoalDataset.__new__(frozen_metadata_view.FrozenMetadataGoalDataset)
    process = multiprocessing.get_context('spawn').Process(target=print, args=(reader,))
    process.start(); process.join(30)
    if process.is_alive():
        process.terminate(); process.join()
        raise AssertionError('spawn timed out')
    assert process.exitcode == 0, process.exitcode
''' % str(BUNDLE / 'entrypoint.py'))
    result = subprocess.run([sys.executable, str(fixture)], text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
