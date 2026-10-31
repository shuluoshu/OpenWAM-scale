"""Content-bound offline admission for a trusted, immutable shared filesystem.

The offline builder fully validates/hashes the population and audits sources.
Runtime checks exact configs, source/registry/code bytes and file stamps, without
reading the population, audit or baseline weights again. Size/mtime/mode stamps
are an operational immutability contract, NOT protection from an adversary that
rewrites bytes and restores metadata. Do not replace/mutate prepared files;
republish after changes. Source dependencies are checked lazily by the readers.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat

from openwam.dataloader.goal_epoch import sha256_file

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'openwam.prepared_runtime.v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def implementation_identity():
    digest = hashlib.sha256()
    for folder in ('openwam', 'scripts'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.sh'):
                digest.update(str(path.relative_to(ROOT)).encode() + b'\0')
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _stamp(path):
    path = Path(path).resolve()
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'prepared artifact must be a regular file: {path}')
    return dict(path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns,
                mode=stat.S_IMODE(info.st_mode))


def _asset(path, *, digest=None, readonly=False):
    path = Path(path).resolve()
    before = _stamp(path)
    checksum = digest or sha256_file(path)
    if _stamp(path) != before:
        raise ValueError(f'prepared artifact changed while hashing: {path}')
    if readonly:
        path.chmod(before['mode'] & ~0o222)
    return dict(_stamp(path), sha256=checksum)


def baseline_assets(path):
    path = Path(path)
    if path.is_file():
        return [path, path.parent/'config.yaml',
                *sorted(p for p in (path.parent/'tokenizer').rglob('*') if p.is_file())]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob('*') if p.is_file())


def _dataloader(config):
    result = dict(config)
    # This selects the admission mode, not data semantics. The effective seed
    # must already include project.seed, as scripts/train.py injects it.
    result.pop('metadata_only', None)
    return result


def write_prepared_runtime(output, *, stage1, stage2_dataloader, pipeline,
                           population, audit_path, dataset_contract, stage2_composed=None):
    """Publish after successful Stage1 --prepare and Stage2 metadata audit.

    population is the summary returned by build_epoch_index's final full audit;
    its index hash is reused, avoiding a second full index hash offline.
    """
    if population.get('complete') is not True or population.get('total_samples', 0) <= 0:
        raise ValueError('prepared population incomplete or empty')
    if len(population.get('index_sha256', '')) != 64:
        raise ValueError('prepared population requires offline index digest')
    source = stage1['dataset']['source_yaml']
    source_asset = _asset(source)
    if source_asset['sha256'] != population.get('source_yaml_sha256'):
        raise ValueError('prepared source changed since population validation')
    audit = json.loads(Path(audit_path).read_text())
    if audit.get('errors') != []:
        raise ValueError('prepared Stage2 audit has errors')
    index = Path(population['index_path'])
    _reject_journals(index)
    index_asset=_asset(index, digest=population['index_sha256'], readonly=True)
    if population.get('index_file_stamp') != {key:index_asset[key] for key in ('size','mtime_ns')}:
        raise ValueError('prepared index changed since full validation (or validation stamp missing)')
    aliases = sorted({str(Path(c['path_aliases']).resolve()) for c in
                      (stage1['dataset'], stage2_dataloader) if c.get('path_aliases')})
    payload = dict(format=FORMAT, implementation_sha256=implementation_identity(),
        stage1_sha256=_digest(stage1), stage2_dataloader_sha256=_digest(_dataloader(stage2_dataloader)),
        pipeline_sha256=_digest(pipeline), stage2_composed_sha256=_digest(stage2_composed), population=population, dataset_contract=dataset_contract,
        source=source_asset, registry=_asset(stage2_dataloader['normalization_registry']),
        index=index_asset,
        audit=_asset(audit_path, readonly=True), aliases=[_asset(path) for path in aliases],
        baseline=[_asset(path) for path in baseline_assets(pipeline['stage2_baseline'])])
    raw = _json(payload)
    target = Path(output)/('runtime-preparation-'+hashlib.sha256(raw).hexdigest()+'.json')
    temporary = target.with_suffix('.pending')
    temporary.write_bytes(raw)
    temporary.chmod(0o444)
    os.replace(temporary, target)
    return str(target.resolve())


def _reject_journals(index):
    for suffix in ('-wal', '-journal'):
        if Path(str(index)+suffix).exists():
            raise ValueError(f'prepared index has a mutable SQLite journal: {index}{suffix}')


def _validate_asset(asset, *, content=False):
    path = asset['path']
    if _stamp(path) != {key: asset[key] for key in ('path', 'size', 'mtime_ns', 'mode')}:
        raise ValueError(f'prepared artifact stamp mismatch: {path}')
    if content and sha256_file(path) != asset['sha256']:
        raise ValueError(f'prepared artifact content mismatch: {path}')


def load_prepared_runtime(path, *, stage1=None, stage2_dataloader=None, pipeline=None,
                          index_path=None, source_yaml=None, stage2_composed=None):
    """Fail closed for an explicitly requested artifact; never fall back to scans."""
    path = Path(path)
    raw = path.read_bytes()
    if path.name != 'runtime-preparation-'+hashlib.sha256(raw).hexdigest()+'.json':
        raise ValueError('prepared proof content address mismatch')
    proof = json.loads(raw)
    if proof.get('format') != FORMAT or proof.get('implementation_sha256') != implementation_identity():
        raise ValueError('prepared proof implementation/format mismatch')
    for key, value in (('stage1_sha256', stage1), ('pipeline_sha256', pipeline), ('stage2_composed_sha256', stage2_composed),
                       ('stage2_dataloader_sha256', _dataloader(stage2_dataloader) if stage2_dataloader is not None else None)):
        if value is not None and proof[key] != _digest(value):
            raise ValueError(f'prepared config mismatch: {key}')
    if index_path is not None and str(Path(index_path).resolve()) != proof['index']['path']:
        raise ValueError('prepared population index path mismatch')
    if source_yaml is not None and str(Path(source_yaml).resolve()) != proof['source']['path']:
        raise ValueError('prepared source path mismatch')
    for key in ('source', 'registry'):
        _validate_asset(proof[key], content=True)
    for asset in proof['aliases']:
        _validate_asset(asset, content=True)
    for key in ('index', 'audit'):
        if proof[key]['mode'] & 0o222:
            raise ValueError(f'prepared {key} must be immutable/read-only')
        _validate_asset(proof[key])
    _reject_journals(proof['index']['path'])
    for asset in proof['baseline']:
        _validate_asset(asset)
    # Detect added/removed baseline assets too; no weight bytes are read here.
    if pipeline is not None and sorted(str(p.resolve()) for p in baseline_assets(pipeline['stage2_baseline'])) != sorted(a['path'] for a in proof['baseline']):
        raise ValueError('prepared baseline asset set mismatch')
    if proof['population']['index_sha256'] != proof['index']['sha256']:
        raise ValueError('prepared population digest mismatch')
    return proof
