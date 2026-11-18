"""Immutable metadata view preserving the original native RAW camera policy.

Native CuratedGoalDataset._episode itself reads frozen info/episode files before
camera resolution, labels, projection configuration or video alignment. All
returned data/root/video paths are restored to their original physical sources.
The source repo selection retains its name, weights and ranges; a contract-only
metadata field binds the frozen overlay into native population identity.
"""
from collections import OrderedDict
import json
from pathlib import Path
import shutil

import data_pipeline as dp

dp.bind_native()
from openwam.dataloader.curated_goal import CuratedGoalDataset
from openwam.dataloader.native_cameras import resolve_native_cameras

OVERLAY_MANIFEST=dp.ROOT/'raw_metadata_overlay.json'
SOURCE_NAMES=('conterfactual_20261016','conterfactual_before_1008/im')
OLD_AUDIT=Path('/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/causal_cstar_h100_switch_20261109/h100_causal_text/stage2_metadata_audit.json')


def create_overlay_manifest(output=OVERLAY_MANIFEST):
    import pyarrow.parquet as pq
    output=Path(output);old_audit=json.loads(OLD_AUDIT.read_text());entries={}
    for name in SOURCE_NAMES:
        source=(Path('/mnt/data/cpfs/b5big/post_train_data/instruct_move')/name).resolve()
        backup=source/'meta/backup_before_undistorted_20261112'
        info=backup/'info.json';episode=backup/'episodes/chunk-000/file-000.parquet'
        if dp.sha256_file(info)!=old_audit['sources'][str(source)]['info_sha256']:
            raise ValueError('backup info differs from original completed source audit')
        a=pq.read_table(episode);b=pq.read_table(source/'meta/episodes/chunk-000/file-000.parquet')
        changed=[k for k in a.column_names if k not in b.column_names or not a[k].equals(b[k])]
        if changed:raise ValueError('original episode metadata fields changed: '+str(changed))
        oldinfo=json.loads(info.read_text());cameras=resolve_native_cameras(str(source),oldinfo)
        expected=[f'observation.images.{v}' for v in ('head','left','right')]
        if cameras!=expected:raise ValueError('original selected RAW camera contract differs')
        overlay=output.parent/'raw_metadata_view'/dp.digest_json(str(source))[:16]
        meta=overlay/'meta';meta.mkdir(parents=True,exist_ok=True)
        (meta/'episodes/chunk-000').mkdir(parents=True,exist_ok=True)
        copies={meta/'info.json':info,meta/'episodes/chunk-000/file-000.parquet':episode}
        for target,original in copies.items():
            if target.exists() and dp.sha256_file(target)!=dp.sha256_file(original):raise ValueError('immutable overlay already exists with different content')
            if not target.exists():shutil.copyfile(original,target);target.chmod(0o444)
        # Every native data/video open follows the original physical file; only
        # info and episode metadata use the immutable frozen view.
        for child in source.iterdir():
            if child.name=='meta':continue
            target=overlay/child.name
            if not target.exists():target.symlink_to(child,target_is_directory=child.is_dir())
        for child in (source/'meta').iterdir():
            if child.name in ('info.json','episodes') or child.name.startswith('backup_'):continue
            target=meta/child.name
            if not target.exists():target.symlink_to(child,target_is_directory=child.is_dir())
        files=[dict(path=str(p.resolve()),sha256=dp.sha256_file(p)) for p in copies]
        descriptor=dict(source=str(source),view_root=str(overlay.resolve()),cameras=cameras,
            files=files,original_info_sha256=dp.sha256_file(info),original_metadata_sha256=dp.sha256_file(episode),
            original_metadata_columns_equal_current=True,original_metadata_rows=a.num_rows,
            original_info_audit=str(OLD_AUDIT),original_info_audit_sha256=dp.sha256_file(OLD_AUDIT),
            policy='native-original-raw-info-and-episode-v1')
        descriptor['fingerprint']=dp.digest_json(descriptor);entries[str(source)]=descriptor
    manifest=dict(format='openwam.longtrain.raw-metadata-overlay.v1',sources=entries,
        source_files_modified=False,old_index_modified=False,undistorted_videos_used=False,
        policy='fresh current RAW video generation; preserve original raw camera/info/episode policy')
    manifest['fingerprint']=dp.digest_json(manifest)
    if output.exists():
        if json.loads(output.read_text())!=manifest:raise ValueError('immutable overlay manifest changed')
        return manifest
    dp.atomic_json(output,manifest);output.chmod(0o444);return manifest


def load_overlay_manifest(path=OVERLAY_MANIFEST):
    path=Path(path);manifest=json.loads(path.read_text());fingerprint=manifest.pop('fingerprint')
    if manifest.get('format')!='openwam.longtrain.raw-metadata-overlay.v1' or dp.digest_json(manifest)!=fingerprint:
        raise ValueError('RAW metadata overlay manifest content mismatch')
    manifest['fingerprint']=fingerprint
    for entry in manifest['sources'].values():
        descriptor={k:v for k,v in entry.items() if k!='fingerprint'}
        if dp.digest_json(descriptor)!=entry['fingerprint']:raise ValueError('RAW metadata overlay entry identity mismatch')
        for file in entry['files']:
            if dp.sha256_file(file['path'])!=file['sha256']:raise ValueError('immutable RAW metadata overlay bytes changed')
    return manifest


class RawPolicyCuratedGoalDataset(CuratedGoalDataset):
    def __init__(self,*args,overlay_manifest=OVERLAY_MANIFEST,**kwargs):
        super().__init__(*args,**kwargs)
        self.raw_overlay=load_overlay_manifest(overlay_manifest)
        self.overlay_path=str(Path(overlay_manifest).resolve())
        for repo in self.repos:
            entry=self.raw_overlay['sources'].get(repo['path'])
            if entry is not None:
                # Native _identity hashes the entire selected repo descriptor.
                # This binds overlay bytes while preserving original repo path.
                repo['raw_camera_policy']=dict(format=entry['policy'],fingerprint=entry['fingerprint'],
                    manifest_sha256=dp.sha256_file(overlay_manifest),overlay_manifest=self.overlay_path)
    def _episode(self,repo,episode):
        entry=self.raw_overlay['sources'].get(repo['path'])
        if entry is None:return super()._episode(repo,episode)
        key=(repo['path'],episode)
        if key in self._episodes:self._episodes.move_to_end(key);return self._episodes[key]
        view=Path(entry['view_root']);native_repo=dict(repo,path=str(view))
        # No camera/UV configuration has been parsed before this native call.
        m=super()._episode(native_repo,episode)
        self._episodes.pop((str(view),episode),None)
        if m['cameras']!=entry['cameras'] or m['head']!=entry['cameras'][0]:
            raise ValueError('native reader escaped immutable original RAW camera policy')
        m['root']=Path(repo['path'])
        m['data']=Path(m['data']).resolve();m['data_files']=[Path(p).resolve() for p in m['data_files']]
        if m.get('annotation_file'):m['annotation_file']=str(Path(m['annotation_file']).resolve())
        self._episodes[key]=m
        while len(self._episodes)>self.max_cached_episodes:self._episodes.popitem(last=False)
        return m


def make_reader(source_yaml,cache_dir,*,overlay_manifest=OVERLAY_MANIFEST,**kwargs):
    if Path(overlay_manifest).is_file():
        return RawPolicyCuratedGoalDataset(source_yaml,cache_dir,overlay_manifest=overlay_manifest,**kwargs)
    raise FileNotFoundError('explicit immutable RAW metadata overlay required: '+str(overlay_manifest))

if __name__=='__main__':
    print(dp.canonical(dict(manifest=str(OVERLAY_MANIFEST),fingerprint=create_overlay_manifest()['fingerprint'])))
