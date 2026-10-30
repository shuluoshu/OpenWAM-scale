"""Materialize complete source instruction spans, one bounded sample at a time.

The smoke YAML identifies candidate source episodes, never the display interval.
Frozen training statistics remain unchanged. Every source frame in the chosen
instruction is retained for rendering, while inference anchors use stride 10.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from scripts.experiments.infer_stage1_samecase import dump, sha, sample_identity


from openwam.dataloader.curated_stage2 import CuratedStage2Dataset


def terminal_raw_window(metadata, anchor):
    """Encode the real current state; synthetic duplicate is discarded before return.

    The training representation requires >=2 rows. No synthetic future is
    returned, decoded, supervised, or exposed as model input/report geometry.
    """
    from openwam.dataloader.native_stage2_io import merge_source_masks
    cols = {key: value[anchor:anchor+1] for key, value in metadata['columns'].items()}
    duplicated = {key: np.repeat(value, 2, axis=0) for key, value in cols.items()}
    raw, mask, state, sm = metadata['representation'].encode_window(duplicated)
    raw, mask = raw[:0], mask[:0]
    frame_valid = merge_source_masks(metadata['representation'], cols, mask, sm)
    if not metadata['head_calibration_eligible']: sm[:,34:] = False
    state[~sm] = 0
    return cols, raw, mask, state, sm, frame_valid


def normalize_terminal_timestamps(metadata):
    """Tolerate <=1e-4 second serialization roundoff, never a missing frame."""
    ep = dict(metadata['ep']); changes=[]
    for camera in metadata['profile']['cameras']:
        if camera is None: continue
        prefix='videos/'+camera+'/'
        key=prefix+'to_timestamp'
        if key not in ep: continue
        expected=float(ep[prefix+'from_timestamp'])+(metadata['length']-1)/float(metadata['info']['fps'])
        original=float(ep[key])
        if 0 < expected-original <= 1e-4:
            ep[key]=expected+1e-9
            changes.append(dict(camera=camera,original=original,adjusted=ep[key],reason='terminal timestamp serialization roundoff; decoder still verifies exact PTS'))
    metadata['ep']=ep
    if changes: metadata['fullspan_timestamp_rounding']=changes


def restore_visual_availability(dataset, metadata, sample):
    """Report-only visual validity follows decoded photons, not pose tracking.

    Retain every action/proprio/FK supervision mask exactly. The native loader
    normally excludes tracking-invalid anchors before building samples; full
    instruction inference must also support their real decoded observations.
    """
    from openwam.model.goal.observation import clean_multiview
    from openwam.model.goal.runtime import clean_goal_head
    from openwam.dataloader.utils.video_color import stamp_stage2_color
    trace=sample['trace'];cameras=metadata['profile']['cameras'];anchor=trace['anchor']
    original=trace.get('visual_availability') or dict(policy='decoded_source_images_independent_of_tracking_v1',
        native_video_supervision_mask=sample['video_mask'].tolist(),
        native_view_supervision_mask=sample['video_view_mask'].tolist(),
        native_current_view_valid=sample['goal_view_valid'].tolist())
    current,_=dataset._decode_views(metadata,[anchor]);current=current[0]
    mask=torch.tensor([camera is not None and camera in current for camera in cameras],dtype=torch.bool)
    sample['goal_view_valid']=mask
    sample['goal_observed_multiview']=(clean_multiview([current.get(camera) for camera in cameras],mask)
        if mask.any() else torch.full((3,1,384,320),-1.))
    head=current.get(cameras[0]);sample['goal_observed_head']=None if head is None else clean_goal_head(head)
    sample['goal_head_valid']=head is not None
    view_mask=torch.zeros((9,3),dtype=torch.bool)
    for i,frame in enumerate(trace['video_source_indices']):
        if frame>trace['last_source']:continue
        for j,camera in enumerate(cameras):
            evidence=trace['video_alignment'].get(camera,{}).get('alignment',{}).get(frame)
            if evidence is not None and evidence.get('source_rgb_sha256') and evidence.get('pts') is not None:
                view_mask[i,j]=True
    if not torch.equal(view_mask[0],mask):raise ValueError('decoded current image/evidence mismatch')
    sample['video_view_mask']=view_mask;sample['video_mask']=view_mask.any(-1)
    if dataset.history.enabled:
        from openwam.dataloader.head_history import materialize_head_history
        from openwam.dataloader.native_stage2_profiles import fingerprint
        original.setdefault('native_history_valid',sample['history_valid'].tolist())
        original.setdefault('native_history_indices',trace['head_history']['indices'])
        history_seed=int(fingerprint([sample['goal_seed'],'head_history',trace.get('draw_id')])[:15],16)
        fields,history_trace=materialize_head_history(metadata['root'],metadata['info'],metadata['ep'],cameras[0],anchor,
            trace['selected_range'][0],dataset.history,np.random.default_rng(history_seed),dataset.split=='train',
            sequential=metadata['profile']['family']=='mecka',source_valid=None,
            video_ordinals=metadata.get('video_ordinals'))
        sample.update(fields);trace['head_history']=history_trace
    trace['visual_availability']=original
    stamp_stage2_color(sample)
    return sample


class FullSpanDataset(CuratedStage2Dataset):
    def _episode(self, repo, episode):
        metadata=super()._episode(repo,episode)
        normalize_terminal_timestamps(metadata)
        return metadata

    def _sample_selected(self, repo, episode, metadata, option, anchor, level, *, draw_id=None):
        sample=super()._sample_selected(repo,episode,metadata,option,anchor,level,draw_id=draw_id)
        return restore_visual_availability(self,metadata,sample)

    def raw_window(self, metadata, option, anchor):
        if anchor == option['last_source']:
            return terminal_raw_window(metadata, anchor)
        return super().raw_window(metadata, option, anchor)


class DecodedFrameCache:
    """Byte-bounded cache of unmodified native RGB frames and alignment traces."""
    def __init__(self, decoder, max_bytes=512*1024*1024):
        from collections import OrderedDict
        self.decoder=decoder;self.max_bytes=max_bytes;self.bytes=0;self.cache=OrderedDict()

    def __call__(self,path,offset_s,fps,indices,*,sequential=False,video_ordinals=None):
        indices=sorted(set(indices));images={};traces={};missing=[]
        def key(index):
            ordinal=None if video_ordinals is None else int(video_ordinals[index])
            return str(path),float(offset_s),float(fps),sequential,index,ordinal
        for index in indices:
            k=key(index)
            if k in self.cache:
                image,trace,_=self.cache[k];self.cache.move_to_end(k)
                images[index]=image;traces[index]=trace
            else:missing.append(index)
        if missing:
            fresh,alignment=self.decoder(path,offset_s,fps,missing,sequential=sequential,video_ordinals=video_ordinals)
            for index in missing:
                image=fresh[index];trace=alignment[index];size=image.width*image.height*len(image.getbands())
                images[index]=image;traces[index]=trace
                if size<=self.max_bytes:
                    while self.cache and self.bytes+size>self.max_bytes:
                        _,(_,_,old_size)=self.cache.popitem(last=False);self.bytes-=old_size
                    self.cache[key(index)]=(image,trace,size);self.bytes+=size
        return images,traces


def anchor_frames(start, end_exclusive, stride=10):
    if not 0 <= start < end_exclusive or stride <= 0:
        raise ValueError('invalid instruction span or stride')
    return sorted(set([*range(start, end_exclusive, stride), end_exclusive - 1]))


def select_instruction(metadata):
    """Prefer the earliest meaningful complete subtask; otherwise complete task."""
    for scope, key in [('subtask', 'subtasks'), ('task', 'tasks')]:
        choices = [x for x in metadata[key] if x.get('valid', True)
                   and 0 <= x['start'] < x['end_exclusive'] <= metadata['length']]
        if not choices:
            continue
        substantial = [x for x in choices if x['end_exclusive'] - x['start'] >= 33
                       and 'stationary' not in x['text'].lower()]
        return scope, min(substantial or choices, key=lambda x: x['start'])
    raise ValueError('no complete valid instruction')


def choose_sequence(candidates):
    # Prefer grounded comparisons, then a manageable but still COMPLETE interval.
    return min(candidates, key=lambda x: (x['scope'] != 'subtask', x['valid_tcp'] == 0,
                                          x['frame_count'], x.get('repo', '')))


def validate_sample(sample):
    anchor = sample['trace']['anchor']; end = sample['instruction_end_exclusive']
    if not sample['instruction_start'] <= anchor < end:
        raise ValueError('anchor outside instruction')
    if sample['report_target']['goal_source_frame'] != end - 1 or sample['trace']['last_source'] != end - 1:
        raise ValueError('endpoint mismatch')
    action = np.asarray(sample['action_mask']); video = np.asarray(sample['video_mask'])
    if action.shape != (32,112) or video.shape != (9,):
        raise ValueError('sample shape mismatch')
    if action[np.arange(32) + anchor + 1 >= end].any() or video[np.arange(9)*4 + anchor >= end].any():
        raise ValueError('valid future beyond instruction boundary')


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    manifest['_manifest_path'] = str(Path(path).resolve())
    return manifest


def iter_anchor_records(manifest):
    if isinstance(manifest, (str, Path)):
        manifest = load_manifest(manifest)
    yield from manifest['anchors']


def load_sample(manifest, record):
    path = Path(record['sample_path'])
    if not path.is_absolute():
        path = Path(manifest['_manifest_path']).parent / path
    sample = torch.load(path, map_location='cpu', weights_only=False)
    validate_sample(sample)
    return sample


def source_rows(sequence, columns=None, start=None, end_exclusive=None):
    """Read exact source geometry rows for the whole selected display interval."""
    from openwam.dataloader.episode_data import read_episode_rows
    paths = sequence['source_parquets']
    start = sequence['start'] if start is None else start
    end_exclusive = sequence['end_exclusive'] if end_exclusive is None else end_exclusive
    return read_episode_rows(dict(data=paths[0], data_files=paths), sequence['episode'],
        columns=columns, filters=[('frame_index','>=',start),('frame_index','<',end_exclusive)]).sort_by('frame_index')


def decode_sequence_views(sequence, indices):
    """Yield the existing native decoder's RGB views and provenance for exact rows."""
    from scripts.data.prepare_subtask_goals import decode_frames
    views = {int(index): {} for index in indices}; traces = {}
    for camera, video in sequence['videos'].items():
        images, alignment = decode_frames(video['path'],video['offset_seconds'],sequence['fps'],indices,
            sequential=sequence['family']=='mecka', video_ordinals=sequence.get('video_ordinals'))
        for index, image in images.items(): views[index][camera] = image
        traces[camera] = alignment
    return views, traces


def build_inventory(dataset, output, config_path):
    from openwam.dataloader.stage2_diagnostics import report_goal_target
    candidates = []; errors = []
    for repo in dataset.repos:
        for episode in sorted({r[0] for r in repo['ranges']}):
            try:
                m = dataset._episode(repo, episode)
                scope, instruction = select_instruction(m)
                start, end = instruction['start'], instruction['end_exclusive']
                _, target = report_goal_target(dataset, repo, episode, m,
                    dict(report_target=dict(goal_source_frame=end-1,label_kind=scope+'_end',success_verified=False)))
                projection = target.get('projection', {})
                valid_tcp = int(np.asarray(projection.get('projection_valid', [False,False])).sum())
                row = dict(family=repo['family'], repo=repo['repo'], episode=episode,
                    source_folder=str(m['root']), source_parquets=[str(p) for p in m['data_files']],
                    scope=scope, prompt=instruction['text'], instruction=instruction,
                    start=start, end_exclusive=end, target_frame=end-1,
                    frame_count=end-start, source_episode_frames=m['length'], fps=float(m['info']['fps']),
                    anchors=anchor_frames(start,end), valid_tcp=valid_tcp,
                    calibration_eligible=m['head_calibration_eligible'], endpoint_report=target,
                    curated_ranges=[list(x[1:]) for x in repo['ranges'] if x[0]==episode],
                    annotation_file=m['annotation_file'], tasks=m['tasks'], subtasks=m['subtasks'],
                    cameras=m['profile']['cameras'], source_dense_complete_verified=True)
                row['videos'] = {}
                for camera in row['cameras']:
                    if camera is None: continue
                    prefix='videos/'+camera+'/'
                    row['videos'][camera]=dict(path=str(m['root']/m['info']['video_path'].format(
                        video_key=camera,episode_index=episode,chunk_index=int(m['ep'][prefix+'chunk_index']),
                        file_index=int(m['ep'][prefix+'file_index']))),offset_seconds=float(m['ep'][prefix+'from_timestamp']))
                if 'video_ordinals' in m: row['video_ordinals']=np.asarray(m['video_ordinals']).tolist()
                candidates.append(row)
                print(json.dumps(dict(event='candidate',family=row['family'],episode=episode,
                    scope=scope,frames=end-start,valid_tcp=valid_tcp)),flush=True)
            except Exception as exc:
                errors.append(dict(repo=repo['repo'],episode=episode,error=f'{type(exc).__name__}: {exc}'))
    families=sorted({r['family'] for r in dataset.repos}); sequences=[]; anchors=[]
    for i, family in enumerate(families):
        choices=[x for x in candidates if x['family']==family]
        if not choices:
            raise ValueError(f'No source candidate for {family}: {errors}')
        row=dict(choose_sequence(choices));row['sequence_id']=f'{i:02d}_{family}'
        row['selection_reason']='complete subtask preferred; valid endpoint TCP preferred; shorter whole interval; earliest meaningful subtask per episode'
        row['candidate_count']=len(choices);sequences.append(row)
        for anchor in row['anchors']:
            anchors.append(dict(id=len(anchors),sequence_id=row['sequence_id'],anchor=anchor,
                family=family,episode=row['episode'],
                sample_path=str((output/row['sequence_id']/f'anchor_{anchor:06d}.pt').resolve()),status='pending'))
    manifest=dict(schema='openwam.fullspan_samples.v1',status='inventory_ready',
        source_config=str(Path(config_path).resolve()),source_config_sha256=sha(config_path),
        source_yaml=dataset.selection['source_yaml'],source_yaml_sha256=dataset.source_yaml_sha256,
        normalization_registry=dataset.registry_path,
        normalization_registry_sha256=sha(dataset.registry_path),stride_source_frames=10,
        selection_policy='one full instruction per dataset family; source YAML clips identify candidate episodes only',
        history_policy='native reader, selected full instruction start, clean current views; no target conditioning',
        source_frames=sum(x['frame_count'] for x in sequences),anchor_count=len(anchors),
        family_count=len(families),sequences=sequences,anchors=anchors,candidate_inventory=candidates,
        candidate_errors=errors)
    dump(output/'manifest.json',manifest)
    return manifest


def add_video_metadata(dataset, manifest):
    repos = {r['repo']: r for r in dataset.repos}
    for sequence in manifest['sequences']:
        if 'videos' in sequence: continue
        m = dataset._episode(repos[sequence['repo']], sequence['episode'])
        sequence['videos'] = {}
        for camera in sequence['cameras']:
            if camera is None: continue
            prefix = 'videos/' + camera + '/'
            sequence['videos'][camera] = dict(path=str(m['root']/m['info']['video_path'].format(
                video_key=camera,episode_index=sequence['episode'],chunk_index=int(m['ep'][prefix+'chunk_index']),
                file_index=int(m['ep'][prefix+'file_index']))),offset_seconds=float(m['ep'][prefix+'from_timestamp']))
        if 'video_ordinals' in m: sequence['video_ordinals'] = np.asarray(m['video_ordinals']).tolist()


def materialize(dataset, manifest, output):
    from openwam.dataloader.native_stage2_stats import lookup_stats
    from openwam.dataloader.stage2_diagnostics import report_goal_target
    from openwam.dataloader.utils.video_color import validate_stage2_color
    repos={r['repo']:r for r in dataset.repos}; started=time.monotonic()
    manifest['status']='materializing';dump(output/'manifest.json',manifest)
    for sequence in manifest['sequences']:
        repo=repos[sequence['repo']];episode=sequence['episode'];m=dataset._episode(repo,episode)
        sequence['native_gripper_filter_bad_frames']=(np.flatnonzero(m['bad'][sequence['start']:sequence['end_exclusive']])+sequence['start']).tolist()
        sequence['terminal_timestamp_rounding']=m.get('fullspan_timestamp_rounding',[])
        stats=lookup_stats(dataset.registry,repo,m)
        active=np.zeros(112,bool);active[m['representation'].scatter]=np.asarray(stats['action']['count'])>0
        option=dict(instruction=sequence['instruction'],selected_start=sequence['start'],
                    selected_end_exclusive=sequence['end_exclusive'],last_source=sequence['end_exclusive']-1)
        goal_image, target=report_goal_target(dataset,repo,episode,m,
            dict(report_target=dict(goal_source_frame=sequence['target_frame'],label_kind=sequence['scope']+'_end',success_verified=False)))
        records=[a for a in manifest['anchors'] if a['sequence_id']==sequence['sequence_id']]
        for record in records:
            path=Path(record['sample_path']);path.parent.mkdir(parents=True,exist_ok=True)
            if record['status']=='ready' and path.is_file():
                if sha(path)!=record['sha256']:raise ValueError(f'changed artifact: {path}')
                continue
            if record['status']=='pending_visual_repair' and path.is_file():
                sample=torch.load(path,map_location='cpu',weights_only=False)
                restore_visual_availability(dataset,m,sample)
            else:
                sample=dataset._sample_selected(repo,episode,m,option,record['anchor'],sequence['scope'])
            sample['report_target']=target
            sample['_eval']=dict(family=sequence['family'],active_action_mask=active,action_stats=stats['action'],
                                  report_goal=goal_image,projection=target.get('projection'))
            sample['_fullspan']=dict(sequence_id=sequence['sequence_id'],anchor_id=record['id'],
                native_gripper_filter_rejected=bool(m['bad'][record['anchor']:min(record['anchor']+33,sequence['end_exclusive'])].any()),
                native_gripper_filter_bad_source_frames=(np.flatnonzero(m['bad'][record['anchor']:min(record['anchor']+33,sequence['end_exclusive'])])+record['anchor']).tolist(),
                terminal_timestamp_rounding=m.get('fullspan_timestamp_rounding',[]),
                start=sequence['start'],end_exclusive=sequence['end_exclusive'],
                future_source_frames=list(range(record['anchor']+1,min(record['anchor']+33,sequence['end_exclusive']))))
            validate_sample(sample);validate_stage2_color(sample)
            temp=path.with_suffix('.tmp');torch.save(sample,temp);temp.replace(path)
            record.update(status='ready',visual_policy_revision=2,sha256=sha(path),bytes=path.stat().st_size,
                future_frames=len(sample['_fullspan']['future_source_frames']),
                valid_action_steps=int(sample['action_mask'].any(-1).sum()),
                identity=sample_identity(sample))
            dump(output/'manifest.json',manifest)
            print(json.dumps(dict(event='anchor_ready',id=record['id'],sequence=sequence['sequence_id'],
                anchor=record['anchor'],total=len(manifest['anchors']),elapsed=time.monotonic()-started)),flush=True)
            del sample
    manifest.update(status='complete',elapsed_seconds=time.monotonic()-started,
                    ready_anchors=sum(x['status']=='ready' for x in manifest['anchors']))
    dump(output/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inventory-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    import scripts.data.prepare_subtask_goals as decoder_module
    decoder_module.decode_frames=DecodedFrameCache(decoder_module.decode_frames)
    cfg=yaml.safe_load(args.config.read_text());dataset=FullSpanDataset.from_config(cfg['dataloader'])
    path=args.output/'manifest.json'
    manifest=load_manifest(path) if path.exists() else build_inventory(dataset,args.output,args.config)
    manifest.pop('_manifest_path',None)
    for record in manifest['anchors']:
        if record['status']=='ready' and record.get('visual_policy_revision',1)<2:
            record.update(status='pending_visual_repair',previous_sample_sha256=record['sha256'])
    manifest['visual_validity_policy']='decoded_source_images_independent_of_tracking_v1; original supervision masks retained in sample trace; action/proprio/FK unchanged'
    add_video_metadata(dataset,manifest);dump(path,manifest)
    print(json.dumps(dict(event='inventory_ready',families=manifest['family_count'],
        source_frames=manifest['source_frames'],anchors=manifest['anchor_count'],manifest=str(path))),flush=True)
    if not args.inventory_only:materialize(dataset,manifest,args.output)


if __name__=='__main__':main()
