"""Manifest-bound native metadata views; physical RGB/data remain original.

Only the selected instruct_move roots are redirected. Native selection identity
reads the actual frozen info/tasks/subtasks via repo.path; native _episode reads
frozen episode fragments before camera/UV parsing. Original metadata can then
change without becoming an unread dependency. Data and camera generations retain
ordinary physical source guards.
"""
from collections import OrderedDict
import json
from pathlib import Path
import shutil
import sqlite3

import data_pipeline as dp

dp.bind_native()
from openwam.dataloader.curated_goal import CuratedGoalDataset,parse_curated_yaml
from openwam.dataloader.native_cameras import resolve_native_cameras

GENERATION_ROOT=Path('/mnt/data/cpfs/b5big/yeqingda/CodeX/longtrain_16gpu_20261112/metadata_generation_v2')
OLD_AUDIT=Path('/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/causal_cstar_h100_switch_20261109/h100_causal_text/stage2_metadata_audit.json')


def remap_dependency_rows(rows,mapping):
    return [(mapping.get(row[0],row[0]),*row[1:]) for row in rows]


def _stamp(path):
    try:s=Path(path).stat();return s.st_size,s.st_mtime_ns
    except FileNotFoundError:return -1,-1


def create_generation(root=GENERATION_ROOT):
    import pyarrow.parquet as pq
    root=Path(root).resolve()
    if (root/'manifest.json').exists():raise ValueError('metadata generation is immutable; choose a new generation path')
    root.mkdir(parents=True,exist_ok=True)
    selection=parse_curated_yaml(dp.DEFAULT_SOURCE)
    selected=[(i,r) for i,r in enumerate(selection['repos']) if 'instruct_move' in r['path']]
    if len(selected)!=15:raise ValueError('reviewed selected instruct_move scope changed')
    oldaudit=json.loads(OLD_AUDIT.read_text())
    old=sqlite3.connect(f'{dp.DEFAULT_ORIGINAL_INDEX.as_uri()}?mode=ro',uri=True)
    original_rows={p:(size,mtime) for p,size,mtime in old.execute('SELECT path,size,mtime_ns FROM sources')}
    from raw_metadata_overlay import load_overlay_manifest
    previous=load_overlay_manifest()
    entries={};aliases={};mapping={};proof=[]
    for ri,repo in selected:
        physical=Path(repo['path']);name=repo['repo'];view=root/'views'/dp.digest_json(name)[:20]
        view.mkdir(parents=True,exist_ok=True);(view/'meta').mkdir(exist_ok=True)
        backup=physical/'meta/backup_before_undistorted_20261112'
        prior=previous['sources'].get(str(physical))
        prior_by_relative={}
        if prior:
            for file in prior['files']:
                relative=str(Path(file['path']).relative_to(Path(prior['view_root'])))
                prior_by_relative[relative]=Path(file['path'])
        prefix=str(physical/'meta')+'/'
        paths=sorted(p for p in original_rows if p.startswith(prefix) and (Path(p).name in ('info.json','tasks.parquet','sub_tasks.parquet') or '/meta/episodes/' in p))
        if not paths:raise ValueError('original native metadata dependencies missing')
        files=[]
        for oldpath in paths:
            relative=str(Path(oldpath).relative_to(physical));target=view/relative;target.parent.mkdir(parents=True,exist_ok=True)
            expected=original_rows[oldpath];mapping[oldpath]=str(target)
            if expected==(-1,-1):
                if Path(oldpath).exists():raise ValueError('previously absent annotation appeared; cannot infer unchanged labels')
                files.append(dict(relative=relative,path=str(target),sha256='MISSING',old_path=oldpath,old_stamp=list(expected),proof='original-index-absent'))
                continue
            if relative in prior_by_relative:
                source=prior_by_relative[relative];kind='reused-6718-native-digest-proof'
            elif backup.is_dir() and (backup/Path(relative).relative_to('meta')).is_file() and (relative=='meta/info.json' or relative.startswith('meta/episodes/')):
                source=backup/Path(relative).relative_to('meta');kind='backup-original-needs-complete-native-proof'
            else:
                source=Path(oldpath);kind='unchanged-original-byte-copy'
                if _stamp(source)!=expected:raise ValueError('unreviewed source metadata changed: '+oldpath)
            before=dp.file_generation(source);sha=dp.sha256_file(source)
            if source.stat().st_size!=expected[0]:raise ValueError('original metadata byte count differs: '+str(source))
            if target.exists():
                if dp.sha256_file(target)!=sha:raise ValueError('unpublished metadata copy differs; choose a fresh generation')
            else:shutil.copyfile(source,target)
            if dp.sha256_file(target)!=sha or dp.file_generation(source)!=before:raise ValueError('metadata changed during frozen copy')
            target.chmod(0o444)
            files.append(dict(relative=relative,path=str(target),sha256=sha,old_path=oldpath,old_stamp=list(expected),
                copy_source=str(source),copy_source_generation=before,proof=kind))
        info=json.loads((view/'meta/info.json').read_text())
        info_sha=dp.sha256_file(view/'meta/info.json')
        if info_sha!=oldaudit['sources'][str(physical)]['info_sha256']:raise ValueError('frozen original info differs from completed source audit')
        if backup.is_dir():
            for item in files:
                if '/episodes/' not in item['relative']:continue
                a=pq.read_table(item['path']);b=pq.read_table(physical/item['relative'])
                if any(k not in b.column_names or not a[k].equals(b[k]) for k in a.column_names):
                    raise ValueError('original episode fields changed: '+str(physical))
        # Metadata is fully copied. No meta directory/file symlink may leak a
        # newly generated annotation or info field back into the native reader.
        for child in physical.iterdir():
            if child.name=='meta':continue
            target=view/child.name
            if target.is_symlink():
                if target.resolve()!=child.resolve():raise ValueError('physical source symlink differs')
            elif not target.exists():target.symlink_to(child,target_is_directory=child.is_dir())
        external=repo.get('native_metadata',{}).get('annotation_file')
        external_evidence=None
        if external:
            if _stamp(external)!=original_rows.get(str(Path(external).resolve())):raise ValueError('physical external annotation changed')
            external_evidence=dict(path=str(Path(external).resolve()),sha256=dp.sha256_file(external),generation=dp.file_generation(external),policy='genuinely read physical annotation; original strict source guard retained')
        count=old.execute('SELECT COUNT(*) FROM episodes WHERE repo_id=?',(ri,)).fetchone()[0]
        entry=dict(repo_id=ri,repo=name,physical_root=str(physical),view_root=str(view),files=files,
            cameras=resolve_native_cameras(name,info),selected_episodes=count,
            info_sha256=info_sha,original_info_matches_stage2_audit=True,external_annotation=external_evidence,
            original_info_audit_sha256=dp.sha256_file(OLD_AUDIT))
        entry['fingerprint']=dp.digest_json(entry);entries[name]=entry;aliases[name]=str(view)
        proof.append(dict(repo_id=ri,repo=name,selected_episodes=count,cameras=entry['cameras'],files=len(files),
            prior_full_proof_reused=prior is not None,backup_original=backup.is_dir()))
    aliases_path=root/'path_aliases.json';dp.atomic_json(aliases_path,aliases);aliases_path.chmod(0o444)
    manifest=dict(format='openwam.longtrain.frozen-metadata-view.v2',sources=entries,
        aliases_file=str(aliases_path),aliases_sha256=dp.sha256_file(aliases_path),
        source_yaml=str(dp.DEFAULT_SOURCE),source_yaml_sha256=dp.sha256_file(dp.DEFAULT_SOURCE),
        original_index=str(dp.DEFAULT_ORIGINAL_INDEX),original_index_sha256=dp.sha256_file(dp.DEFAULT_ORIGINAL_INDEX),
        previous_6718_receipt=str(dp.ROOT/'population_rebind_receipt.json'),
        previous_6718_receipt_sha256=dp.sha256_file(dp.ROOT/'population_rebind_receipt.json'),
        original_names_weights_ranges_preserved=True,physical_sources_modified=False,
        policy='each root original native camera/info/episode/annotation policy; metadata frozen; physical data/video generations guarded')
    manifest['fingerprint']=dp.digest_json(manifest);dp.atomic_json(root/'manifest.json',manifest);(root/'manifest.json').chmod(0o444)
    dp.atomic_json(root/'copy_proof.json',dict(roots=proof,metadata_path_mapping=mapping,selected_episodes=sum(x['selected_episodes'] for x in proof)))
    old.close();return manifest


def load_manifest(path):
    path=Path(path).resolve();manifest=json.loads(path.read_text());expected=manifest['fingerprint']
    if dp.digest_json({k:v for k,v in manifest.items() if k!='fingerprint'})!=expected or manifest['format']!='openwam.longtrain.frozen-metadata-view.v2':
        raise ValueError('frozen metadata manifest identity mismatch')
    if dp.sha256_file(manifest['aliases_file'])!=manifest['aliases_sha256']:raise ValueError('metadata path aliases changed')
    for entry in manifest['sources'].values():
        if dp.digest_json({k:v for k,v in entry.items() if k!='fingerprint'})!=entry['fingerprint']:raise ValueError('metadata entry identity mismatch')
        for file in entry['files']:
            if file['sha256']=='MISSING':
                if Path(file['path']).exists():raise ValueError('frozen absent annotation became present')
            elif dp.sha256_file(file['path'])!=file['sha256']:raise ValueError('frozen metadata bytes changed: '+file['path'])
    return manifest


class FrozenMetadataGoalDataset(CuratedGoalDataset):
    def __init__(self,*args,metadata_view_manifest,**kwargs):
        self.manifest_path=str(Path(metadata_view_manifest).resolve());self.metadata_view=load_manifest(self.manifest_path)
        supplied=kwargs.pop('path_aliases',None)
        if supplied is not None:
            supplied=json.loads(Path(supplied).read_text()) if isinstance(supplied,(str,Path)) else supplied
            if supplied!=json.loads(Path(self.metadata_view['aliases_file']).read_text()):raise ValueError('path aliases differ from immutable metadata view')
        super().__init__(*args,path_aliases=self.metadata_view['aliases_file'],**kwargs)
        self.raw_overlay=self.metadata_view
        self.entries_by_view={entry['view_root']:entry for entry in self.metadata_view['sources'].values()}
        for repo in self.repos:
            entry=self.entries_by_view.get(repo['path'])
            if entry is not None:
                if repo['repo']!=entry['repo']:raise ValueError('metadata view changed source name')
                repo['frozen_metadata']=dict(manifest_sha256=dp.sha256_file(self.manifest_path),entry=entry['fingerprint'])
    def _episode(self,repo,episode):
        metadata=super()._episode(repo,episode);entry=self.entries_by_view.get(repo['path'])
        if entry is None:return metadata
        if metadata['cameras']!=entry['cameras']:raise ValueError('original effective native camera policy changed')
        metadata['metadata_root']=Path(entry['view_root'])
        metadata['root']=Path(entry['physical_root'])
        metadata['data']=Path(metadata['data']).resolve();metadata['data_files']=[Path(p).resolve() for p in metadata['data_files']]
        # An annotation, when present, deliberately remains the frozen path.
        return metadata
    def metadata_dependencies(self,repo,episode):
        entry=self.entries_by_view.get(repo['path'])
        if entry is None:return None
        return [file['path'] for file in entry['files'] if file['sha256']!='MISSING']+[self.manifest_path,self.metadata_view['aliases_file']]


def make_reader(source_yaml,cache_dir,*,metadata_view_manifest,**kwargs):
    return FrozenMetadataGoalDataset(source_yaml,cache_dir,metadata_view_manifest=metadata_view_manifest,**kwargs)

if __name__=='__main__':
    manifest=create_generation();print(dp.canonical(dict(manifest=str(GENERATION_ROOT/'manifest.json'),fingerprint=manifest['fingerprint'],roots=len(manifest['sources']))))
