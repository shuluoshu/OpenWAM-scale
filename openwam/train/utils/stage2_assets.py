"""Portable native normalization registry and selected-source identity assets."""
import json
import os
from pathlib import Path
import shutil
from openwam.model.goal.runtime import goal_asset_sha256


def _relative_asset(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Checkpoint asset path must be relative')
    return Path(root) / relative


def save_native_normalization(output_dir, dataset):
    root = Path(output_dir)
    manifest_path = root / 'normalization/manifest.json'
    if manifest_path.exists():
        verify_native_normalization(root, dataset)
        return
    directory = manifest_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    registry = Path(dataset.registry_path)
    selection = Path(dataset.selection['source_yaml'])
    registry_target = directory / registry.name
    selection_target = directory / 'selection.yaml'
    # Registries contain transform arrays, contracts, source hashes and sampled
    # anchor identities inline; no external transform payload is referenced.
    for src, dst in ((registry, registry_target), (selection, selection_target)):
        shutil.copyfile(src, dst)
    manifest = dict(format='openwam.stage2.normalization.v1',
        registry=str(registry_target.relative_to(root)), registry_sha256=goal_asset_sha256(registry_target),
        selection=str(selection_target.relative_to(root)), selection_sha256=goal_asset_sha256(selection_target),
        contract=dataset.stats_contract())
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    os.replace(temporary, manifest_path)
    verify_native_normalization(root, dataset)


def load_native_normalization(output_dir):
    from openwam.dataloader.native_stage2_stats import read_registry
    root = Path(output_dir)
    manifest = json.loads((root/'normalization/manifest.json').read_text())
    if manifest.get('format') != 'openwam.stage2.normalization.v1':
        raise ValueError('Native normalization manifest format mismatch')
    for key in ('registry', 'selection'):
        asset = _relative_asset(root, manifest[key])
        if goal_asset_sha256(asset) != manifest[key+'_sha256']:
            raise ValueError(f'Native normalization {key} SHA256 mismatch')
    registry = read_registry(_relative_asset(root, manifest['registry']), manifest['contract'])
    if registry['contract'].get('selection_sha256') != manifest['selection_sha256']:
        raise ValueError('Native selection/registry fingerprint mismatch')
    return manifest, registry


def verify_native_normalization(output_dir, dataset):
    manifest, registry = load_native_normalization(output_dir)
    if (manifest['contract'] != dataset.stats_contract() or
            manifest['registry_sha256'] != goal_asset_sha256(dataset.registry_path) or
            registry != dataset.registry):
        raise ValueError('Stage2 resume selection/normalization identity mismatch; use warm-start')


def resolve_resume_normalization(cfg, checkpoint_dir):
    """Pin packaged stats/selection; data_root and source content remain checked."""
    from omegaconf import OmegaConf, open_dict
    manifest, _ = load_native_normalization(checkpoint_dir)
    root = OmegaConf.select(cfg, 'dataloader.data_root')
    if root and str(Path(root).resolve()) != str(Path(manifest['contract']['data_root']).resolve()):
        raise ValueError('Stage2 resume data_root identity mismatch; use warm-start')
    source = OmegaConf.select(cfg, 'dataloader.source_yaml')
    if source and Path(source).is_file() and goal_asset_sha256(source) != manifest['selection_sha256']:
        raise ValueError('Stage2 resume selection identity mismatch; use warm-start')
    with open_dict(cfg):
        OmegaConf.update(cfg, 'dataloader.normalization_registry',
                         str(_relative_asset(checkpoint_dir, manifest['registry']).resolve()))
        OmegaConf.update(cfg, 'dataloader.source_yaml',
                         str(_relative_asset(checkpoint_dir, manifest['selection']).resolve()))
        # YAML's original relative root was interpreted at its original location.
        # The registry pins that resolved data identity, even after packaging.
        OmegaConf.update(cfg, 'dataloader.data_root', manifest['contract']['data_root'])
