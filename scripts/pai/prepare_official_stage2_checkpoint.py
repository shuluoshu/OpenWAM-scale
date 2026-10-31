#!/usr/bin/env python3
"""Build a separately identified, immutable official80 -> relative112 warm start.

This is parameter initialization, not a trained checkpoint or a conversion of
normalization/action semantics. Existing directories are verified, never repaired.
All tensor operations are on CPU. No model or tokenizer is instantiated.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile

import torch
import yaml
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

# Support direct invocation without installing this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openwam.model.checkpoint_utils import expand_action_state_dict

AXES = {
    'action_backbone.action_encoder.weight': 1,
    'action_backbone.action_decoder.weight': 0,
    'action_backbone.action_decoder.bias': 0,
    'proprio_encoder.weight': 1,
}
FORMAT = 'openwam.official80-relative112-init.v1'


def _log(message):
    print(message, flush=True)


def _hash(path):
    digest = hashlib.sha256()
    size = 0
    next_progress = 2 * 1024**3
    with path.open('rb') as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
            size += len(chunk)
            if size >= next_progress:
                _log(f'SHA256 {path.name}: {size / 1024**3:.1f} GiB')
                next_progress += 2 * 1024**3
    return {'sha256': digest.hexdigest(), 'size_bytes': size}


def _files(directory):
    """Inventory regular files only; never traverse symlinks."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f'Expected real directory: {directory}')
    found = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f'Unsafe artifact entry: {path}')
        if path.is_file():
            found.append(path.relative_to(directory).as_posix())
    return found


def _inventory(directory, names):
    result = {}
    for name in names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or unsafe artifact file: {path}')
        result[name] = _hash(path)
    return result


def _source_layout(source):
    config_path = source.parent / 'config.yaml'
    if config_path.is_symlink():
        raise ValueError('Source config must not be a symlink')
    try:
        config = yaml.safe_load(config_path.read_text())
        architecture = config['model']['architecture']
        if architecture['action_dim'] != 80 or architecture['state_dim'] != 80:
            raise ValueError('Source config must declare action_dim=state_dim=80')
        subdir = config['model']['video_backbone']['tokenizer']['subdir']
        relative = Path(subdir)
        if not isinstance(subdir, str) or relative.is_absolute() or not relative.parts or '..' in relative.parts:
            raise ValueError('Tokenizer subdir must be a nonempty safe relative directory')
        tokenizer = source.parent / relative
        # Reject symlinks in every component, even if they point inside the release.
        for parent in (tokenizer, *tokenizer.parents):
            if parent == source.parent:
                break
            if parent.is_symlink():
                raise ValueError(f'Tokenizer path contains symlink: {parent}')
        token_files = _files(tokenizer)
        if not token_files:
            raise ValueError('Tokenizer directory must contain files')
    except (KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        raise ValueError(f'Invalid source config/tokenizer: {exc}') from exc
    names = [source.name, 'config.yaml'] + [(relative / name).as_posix() for name in token_files]
    return config, sorted(names)


def _projection_shapes(source):
    try:
        with safe_open(str(source), framework='pt', device='cpu') as handle:
            shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
            metadata = handle.metadata()
    except SafetensorError as exc:
        raise ValueError(f'Invalid source safetensors checkpoint: {exc}') from exc
    targets = dict(shapes)
    migrations = {}
    for key, axis in AXES.items():
        shape = shapes.get(key)
        if shape is None or len(shape) != (1 if key.endswith('.bias') else 2) or shape[axis] != 80:
            raise ValueError(f'Official source requires complete width80 projections: {key}={shape}')
        target = list(shape)
        target[axis] = 112
        targets[key] = tuple(target)
        migrations[key] = {'axis': axis, 'source_shape': list(shape), 'output_shape': target}
    return targets, migrations, metadata


def _verify_existing(output, source_identity, migration, names):
    manifest_path = output / 'provenance.json'
    try:
        if manifest_path.is_symlink():
            raise ValueError('Provenance must not be a symlink')
        manifest = json.loads(manifest_path.read_text())
        if manifest['format'] != FORMAT or manifest['source'] != source_identity or manifest['migration'] != migration:
            raise ValueError('Existing artifact source identity or migration differs')
        if _files(output) != sorted([*names, 'provenance.json']):
            raise ValueError('Existing artifact has missing or unexpected files')
        if manifest['output'] != {'files': _inventory(output, names)}:
            raise ValueError('Existing artifact output hashes differ')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Partial or invalid existing artifact: {output}: {exc}') from exc
    return manifest


def _fsync_tree(directory):
    for name in _files(directory):
        with (directory / name).open('rb') as handle:
            os.fsync(handle.fileno())
    # Persist child entries before publishing the enclosing directory.
    directories = [p for p in directory.rglob('*') if p.is_dir()] + [directory]
    for path in directories:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _artifact_directory(output):
    if not output.is_symlink():
        return output
    link = os.readlink(output)
    pattern = re.escape(f'.{output.name}.artifact-') + r'[a-z0-9_]{8}'
    if re.fullmatch(pattern, link) is None:
        raise ValueError('Output symlink must name its owned private sibling artifact')
    artifact = output.parent / link
    if artifact.is_symlink() or not artifact.is_dir():
        raise ValueError('Output symlink must reference a complete real artifact directory')
    return artifact


def _publish_noreplace(source, destination):
    # CPFS rejects renameat2(RENAME_NOREPLACE). Publishing a relative symlink
    # exposes the complete private directory atomically and never clobbers an
    # existing destination. Keep both siblings when relocating this artifact.
    if source.parent != destination.parent:
        raise ValueError('Artifact publication requires sibling directories')
    os.symlink(source.name, destination, target_is_directory=True)


def prepare_checkpoint(source_path, output_dir):
    """Atomically create an artifact or return its fully reverified manifest."""
    original_source = Path(source_path).absolute()
    original_output = Path(output_dir).absolute()
    if original_source.is_symlink():
        raise ValueError('Source checkpoint must not be a symlink')
    source = original_source.resolve(strict=True)
    output = original_output.parent.resolve() / original_output.name
    artifact = _artifact_directory(output)
    source_root = source.parent
    if source.suffix != '.safetensors' or not source.is_file():
        raise ValueError('Source must be a safetensors checkpoint file')
    for candidate in (output, artifact):
        if candidate == source_root or candidate.is_relative_to(source_root) or source_root.is_relative_to(candidate):
            raise ValueError('Source release and output directory must not be nested')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Serialize callers, including reuse verification; leave the lock inode stable.
    with (output.parent / f'.{output.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _, names = _source_layout(source)
        _log(f'Hashing official source and dependencies: {source}')
        source_identity = {'checkpoint': str(source), 'files': _inventory(source_root, names)}
        # Read content only after pinning its identity; final rehash detects
        # changes between this snapshot and publication.
        config, current_names = _source_layout(source)
        if current_names != names:
            raise ValueError('Source tokenizer inventory changed while hashing')
        targets, migrations, metadata = _projection_shapes(source)
        migration = {'source_dim': 80, 'target_dim': 112, 'initialization': 'zero_pad_appended_dimensions',
                     'projections': migrations}
        if output.exists() or output.is_symlink():
            manifest = _verify_existing(_artifact_directory(output), source_identity, migration, names)
            if _source_layout(source)[1] != names or _inventory(source_root, names) != source_identity['files']:
                raise ValueError('Source files changed during reuse verification')
            _log(f'Verified immutable artifact reuse: {output}')
            return manifest
        staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.artifact-', dir=output.parent))
        published = False
        try:
            _log('Loading source tensor views on CPU and expanding the four projections')
            with safe_open(str(source), framework='pt', device='cpu') as handle:
                state = {key: handle.get_tensor(key) for key in handle.keys()}
                expanded = expand_action_state_dict(state, targets, source_dim=80, target_dim=112)
                _log(f'Writing derived checkpoint: {staging / source.name}')
                save_file(expanded, str(staging / source.name), metadata=metadata)
                del expanded, state
            derived_config = copy.deepcopy(config)
            derived_config['model']['architecture'].update(action_dim=112, state_dim=112)
            (staging / 'config.yaml').write_text(yaml.safe_dump(derived_config, sort_keys=False))
            for name in names:
                if name in (source.name, 'config.yaml'):
                    continue
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / name, target)
            _log('Hashing completed artifact and reverifying source remained unchanged')
            output_identity = {'files': _inventory(staging, names)}
            if _source_layout(source)[1] != names or _inventory(source_root, names) != source_identity['files']:
                raise ValueError('Source files changed during conversion; refusing publication')
            manifest = {'format': FORMAT, 'source': source_identity, 'output': output_identity, 'migration': migration}
            (staging / 'provenance.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
            _fsync_tree(staging)
            _publish_noreplace(staging, output)
            published = True
            fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            _log(f'Published derived initialization: {output}')
            return manifest
        finally:
            if not published and staging.exists():
                shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = prepare_checkpoint(args.source, args.output_dir)
    _log(json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
    main()
