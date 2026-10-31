"""Offline proofs admit unchanged data without repeating global metadata scans."""
import copy
import json
from pathlib import Path
import pytest
from openwam.dataloader import goal_epoch
from test_goal_epoch_data import Source


def artifact(tmp_path):
    from openwam.train import prepared_runtime as runtime
    source = Source(tmp_path)
    dependency=tmp_path/'dependency';dependency.write_text('initial')
    episode=source._episode
    source._episode=lambda *args:dict(episode(*args),data=dependency)
    index = tmp_path/'population.sqlite'
    population = goal_epoch.build_epoch_index(source, index, sample_fps=3.)
    registry = tmp_path/'registry.json'; registry.write_text('{}')
    audit = tmp_path/'audit.json'; audit.write_text('{"errors": []}')
    baseline = tmp_path/'baseline'; baseline.mkdir()
    (baseline/'model.safetensors').write_bytes(b'weights')
    (baseline/'config.yaml').write_text('model: {}')
    stage1 = dict(dataset=dict(source_yaml=source.selection['source_yaml']), sample_fps=3.)
    stage2 = dict(source_yaml=source.selection['source_yaml'], normalization_registry=str(registry), seed=42)
    pipeline = dict(stage2_baseline=str(baseline), output_dir=str(tmp_path))
    proof = runtime.write_prepared_runtime(tmp_path, stage1=stage1, stage2_dataloader=stage2,
        pipeline=pipeline, population=population, audit_path=audit, dataset_contract={'reader': 1})
    return runtime, source, index, registry, audit, stage1, stage2, pipeline, proof


def test_warm_population_never_scans_sources_or_hashes_index(tmp_path, monkeypatch):
    runtime, source, index, registry, audit, stage1, stage2, pipeline, proof = artifact(tmp_path)
    monkeypatch.setattr(goal_epoch, 'validate_index', lambda *a, **k: pytest.fail('full index audit'))
    original = runtime.sha256_file
    def bounded(path):
        assert Path(path) not in (index, audit), 'warm path read immutable large artifact'
        return original(path)
    monkeypatch.setattr(runtime, 'sha256_file', bounded)
    monkeypatch.setattr(goal_epoch, 'sha256_file', bounded)
    from scripts.train_goal_epoch import validated_population
    finite = validated_population(source, index, sample_fps=3., prepared_runtime=proof, config=stage1)
    assert len(finite) > 0
    assert finite.sample(0)[0]['anchor'] >= 0
    loaded = runtime.load_prepared_runtime(proof, pipeline=pipeline, stage2_dataloader=stage2)
    assert loaded['dataset_contract'] == {'reader': 1}


@pytest.mark.parametrize('changed', ['source','registry','index','audit','stage1','stage2','pipeline','code','baseline','proof','missing'])
def test_prepared_runtime_rejects_stale_bindings(tmp_path, monkeypatch, changed):
    runtime, source, index, registry, audit, stage1, stage2, pipeline, proof = artifact(tmp_path)
    if changed in ('source','registry','index','audit','baseline'):
        path = {'source':Path(source.selection['source_yaml']), 'registry':registry,
                'index':index,'audit':audit,'baseline':Path(pipeline['stage2_baseline'])/'model.safetensors'}[changed]
        path.chmod(0o644)
        with path.open('ab') as stream: stream.write(b'changed')
    elif changed == 'stage1': stage1['sample_fps'] = 5.
    elif changed == 'stage2': stage2['seed'] = 1
    elif changed == 'pipeline': pipeline['world_size'] = 32
    elif changed == 'code': monkeypatch.setattr(runtime, 'implementation_identity', lambda: 'changed')
    elif changed == 'proof':
        Path(proof).chmod(0o644)
        Path(proof).write_text(Path(proof).read_text() + ' ')
    else: proof = str(proof) + '.missing'
    with pytest.raises((ValueError, FileNotFoundError), match='prepared|No such file'):
        runtime.load_prepared_runtime(proof, stage1=stage1, stage2_dataloader=stage2, pipeline=pipeline)


def test_explicit_proof_requires_matching_dataset_and_config(tmp_path):
    runtime, source, index, registry, audit, stage1, stage2, pipeline, proof = artifact(tmp_path)
    source.repos[0]['ranges'] = [(0,2,12)]
    from scripts.train_goal_epoch import validated_population
    with pytest.raises(ValueError, match='prepared'):
        validated_population(source, index, sample_fps=3., prepared_runtime=proof, config=stage1)


def test_legacy_population_still_audits(tmp_path, monkeypatch):
    from scripts.train_goal_epoch import validated_population
    source = Source(tmp_path)
    monkeypatch.setattr(goal_epoch, 'validate_index', lambda *a, **k: (_ for _ in ()).throw(ValueError('legacy audit')))
    with pytest.raises(ValueError, match='legacy audit'):
        validated_population(source, tmp_path/'index', sample_fps=3.)


def test_stage2_prepared_admission_skips_audit_but_legacy_keeps_it(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from openwam.train.utils.stage2 import validate_stage2_data
    from openwam.dataloader import curated_stage2, native_stage2_stats
    runtime, source, index, registry, audit, stage1, stage2, pipeline, proof = artifact(tmp_path)
    actual = object()
    monkeypatch.setattr(curated_stage2.CuratedStage2Dataset, 'from_config', lambda *a: actual)
    monkeypatch.setattr(native_stage2_stats, 'audit_selection', lambda *a: pytest.fail('full Stage2 audit'))
    cfg = OmegaConf.create({'dataloader': stage2})
    assert validate_stage2_data(cfg, prepared_runtime=proof) is actual
    with pytest.raises(pytest.fail.Exception, match='full Stage2 audit'):
        validate_stage2_data(cfg)
    cfg.dataloader.seed = 7
    with pytest.raises(ValueError, match='prepared config'):
        validate_stage2_data(cfg, prepared_runtime=proof)


def test_publication_rejects_index_changed_since_full_validation(tmp_path):
    runtime, source, index, registry, audit, stage1, stage2, pipeline, proof = artifact(tmp_path)
    population=runtime.load_prepared_runtime(proof)['population']
    index.chmod(0o644)
    with index.open('ab') as stream:stream.write(b'changed since validation')
    with pytest.raises(ValueError, match='prepared.*index'):
        runtime.write_prepared_runtime(tmp_path,stage1=stage1,stage2_dataloader=stage2,
            pipeline=pipeline,population=population,audit_path=audit,dataset_contract={})


def test_prepared_population_still_checks_lazy_source_dependencies(tmp_path):
    runtime, source, index, registry, audit, stage1, stage2, pipeline, proof = artifact(tmp_path)
    from scripts.train_goal_epoch import validated_population
    finite=validated_population(source,index,sample_fps=3.,prepared_runtime=proof,config=stage1)
    (tmp_path/'dependency').write_text('changed after admission')
    with pytest.raises(ValueError,match='source dependency changed'):
        finite.sample(0)


def test_stage2_resume_accepts_only_identical_packaged_data(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from openwam.train.utils.stage2 import validate_stage2_data
    from openwam.dataloader import curated_stage2, native_stage2_stats
    runtime, source, index, registry, audit, stage1, stage2, pipeline, proof = artifact(tmp_path)
    packaged=tmp_path/'normalization';packaged.mkdir()
    (packaged/'selection.yaml').write_bytes(Path(stage2['source_yaml']).read_bytes())
    (packaged/'registry.json').write_bytes(registry.read_bytes())
    cfg=OmegaConf.create({'dataloader':dict(stage2,source_yaml=str(packaged/'selection.yaml'),
        normalization_registry=str(packaged/'registry.json'),data_root=str(tmp_path))})
    actual=SimpleNamespace(stats_contract=lambda:{'reader':1})
    monkeypatch.setattr(curated_stage2.CuratedStage2Dataset,'from_config',lambda *a:actual)
    monkeypatch.setattr(native_stage2_stats,'audit_selection',lambda *a:pytest.fail('full Stage2 audit'))
    assert validate_stage2_data(cfg,prepared_runtime=proof,original_dataloader=stage2) is actual
    (packaged/'registry.json').write_text('{"changed": true}')
    with pytest.raises(ValueError,match='prepared.*relocated'):
        validate_stage2_data(cfg,prepared_runtime=proof,original_dataloader=stage2)
