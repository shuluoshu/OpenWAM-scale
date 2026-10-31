"""Save receipts eliminate redundant local reads; external verification stays fresh."""
from collections import Counter
import errno
import json
import os
from pathlib import Path
import pytest


def completed_run(tmp_path, monkeypatch):
    import openwam.train.goal_epoch as epoch
    import openwam.train.convergence as gate
    from test_goal_epoch_convergence import fixture_run, constant_metric
    monkeypatch.setattr(gate, 'fixed_flow_metric', constant_metric)
    expert, data, kwargs = fixture_run()
    calls = []
    original = epoch.sha256
    def count(path):
        calls.append(Path(path)); return original(path)
    monkeypatch.setattr(epoch, 'sha256', count)
    proof = epoch.train_epoch(expert, data, None, tmp_path, **kwargs)
    return epoch, proof, kwargs['identity'], calls


def test_fresh_training_hashes_each_saved_file_once_and_external_verify_is_fresh(tmp_path, monkeypatch):
    epoch, proof, identity, calls = completed_run(tmp_path, monkeypatch)
    assert calls and all(path.parent.name.startswith('.pending-') for path in calls)
    assert Counter(path.name for path in calls)['training.pt'] == 3  # baseline and two updates
    source = Path(proof['checkpoint']); alias = tmp_path/source.name
    assert source.stat().st_ino == alias.stat().st_ino
    calls.clear()
    assert epoch.verify_completion(tmp_path, identity) == proof
    assert Counter(path.name for path in calls)[source.name] == 1
    assert Counter(path.name for path in calls)['training.pt'] == 1


def saved_receipt(tmp_path):
    from openwam.train.goal_epoch import save_epoch_checkpoint
    from openwam.train.goal_overfit import FP32MasterAdamW
    from openwam.train.goal_curated import goal_parameter_groups
    from test_goal_epoch_convergence import fixture_run
    expert, _, kwargs = fixture_run()
    optimizer = FP32MasterAdamW(goal_parameter_groups(expert,1e-5,1e-4))
    receipt = save_epoch_checkpoint(expert,optimizer,tmp_path,identity=kwargs['identity'],cursor=0,update=0,
                                   dependencies=kwargs['dependencies'],_return_receipt=True)
    return receipt,kwargs['identity']


@pytest.mark.parametrize('name', ['training.pt','query_goal.safetensors','manifest.json'])
def test_receipt_rejects_corruption_even_when_size_and_mtime_are_restored(tmp_path, name):
    import openwam.train.goal_epoch as epoch
    receipt,identity = saved_receipt(tmp_path)
    target = receipt.directory/name; stamp = target.stat()
    data = bytearray(target.read_bytes())
    if name == 'manifest.json':
        manifest = json.loads(data); manifest['identity']['seed'] += 1
        data = json.dumps(manifest).encode()
    else: data[-1] ^= 1
    target.write_bytes(data); os.utime(target, ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='checksum|identity'):
        epoch._checkpoint(tmp_path,identity,receipt=receipt)


def test_receipt_does_not_validate_replaced_latest_generation(tmp_path, monkeypatch):
    import openwam.train.goal_epoch as epoch
    receipt,identity = saved_receipt(tmp_path)
    replacement,_ = saved_receipt(tmp_path)
    calls = []; original = epoch.sha256
    monkeypatch.setattr(epoch,'sha256',lambda path: (calls.append(Path(path)),original(path))[1])
    directory,_ = epoch._checkpoint(tmp_path,identity,receipt=receipt)
    assert directory == replacement.directory
    assert calls and all(p.parent == replacement.directory for p in calls)


def test_atomic_bundle_link_falls_back_to_copy(tmp_path, monkeypatch):
    import openwam.train.goal_epoch as epoch
    source = tmp_path/'source'; source.write_bytes(b'durable model')
    target = tmp_path/'alias'; target.write_bytes(b'old')
    def unsupported(*args,**kwargs): raise OSError(errno.EXDEV,'cross device')
    monkeypatch.setattr(epoch.os,'link',unsupported)
    epoch._publish_bundle(source,target)
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino
    assert not list(tmp_path.glob('*.pending-*'))


def test_corruption_after_alias_publication_invalidates_trainer_receipt(tmp_path, monkeypatch):
    import openwam.train.goal_epoch as epoch
    import openwam.train.convergence as gate
    from test_goal_epoch_convergence import fixture_run,constant_metric
    original = epoch._publish_bundle
    def corrupt(source,target,**kwargs):
        original(source,target,**kwargs)
        with source.open('r+b') as stream:
            stream.seek(-1,2); value=stream.read(1); stream.seek(-1,2); stream.write(bytes([value[0]^1]))
    monkeypatch.setattr(epoch,'_publish_bundle',corrupt)
    monkeypatch.setattr(gate,'fixed_flow_metric',constant_metric)
    expert,data,kwargs = fixture_run()
    with pytest.raises(RuntimeError,match='checksum'):
        epoch.train_epoch(expert,data,None,tmp_path,**kwargs)


def test_external_verifier_rejects_missing_mandatory_model_digest(tmp_path, monkeypatch):
    epoch,proof,identity,_ = completed_run(tmp_path,monkeypatch)
    source = Path(proof['checkpoint']); manifest_path = source.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text()); del manifest['files'][source.name]
    manifest_path.write_text(json.dumps(manifest))
    proof['checkpoint_sha256'] = None
    (tmp_path/'training_complete.json').write_text(json.dumps(proof))
    with pytest.raises((ValueError,KeyError),match='checkpoint|query_goal'):
        epoch.verify_completion(tmp_path,identity)


def test_external_verification_never_trusts_unchanged_metadata(tmp_path,monkeypatch):
    epoch,proof,identity,_ = completed_run(tmp_path,monkeypatch)
    source = Path(proof['checkpoint']); stamp = epoch._file_stamp(source)
    with source.open('r+b') as stream:
        stream.seek(-1,2); value=stream.read(1); stream.seek(-1,2); stream.write(bytes([value[0]^1]))
    # Simulate same-tick/restored metadata: the public verifier must still hash.
    original = epoch._file_stamp
    monkeypatch.setattr(epoch,'_file_stamp',lambda path: stamp if Path(path)==source else original(path))
    with pytest.raises(ValueError,match='checksum'):
        epoch.verify_completion(tmp_path,identity)


@pytest.mark.parametrize('completed', [False,True])
def test_resume_always_rehashes_and_rejects_corrupt_optimizer(tmp_path,monkeypatch,completed):
    from test_goal_epoch_convergence import fixture_run
    epoch,proof,identity,_ = completed_run(tmp_path,monkeypatch)
    target = Path(proof['checkpoint']).parent/'training.pt'
    old = epoch._file_stamp(target)
    with target.open('r+b') as stream:
        stream.seek(-1,2); value=stream.read(1); stream.seek(-1,2); stream.write(bytes([value[0]^1]))
    original = epoch._file_stamp
    monkeypatch.setattr(epoch,'_file_stamp',lambda path: old if Path(path)==target else original(path))
    if not completed: (tmp_path/'training_complete.json').unlink()
    expert,data,kwargs = fixture_run()
    with pytest.raises(RuntimeError,match='checksum'):
        epoch.train_epoch(expert,data,None,tmp_path,resume=True,**kwargs)


def test_save_rejects_file_changed_after_its_hash_before_receipt(tmp_path,monkeypatch):
    import openwam.train.goal_epoch as epoch
    original = epoch.sha256
    def changed(path):
        digest = original(path)
        if Path(path).name == 'training.pt':
            with Path(path).open('ab') as stream:stream.write(b'changed')
        return digest
    monkeypatch.setattr(epoch,'sha256',changed)
    with pytest.raises(ValueError,match='changed during save verification'):
        saved_receipt(tmp_path)
    assert not (tmp_path/'latest.json').exists()
