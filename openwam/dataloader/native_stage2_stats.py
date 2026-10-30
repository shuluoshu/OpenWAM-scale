"""Immutable native normalization registries from bounded deterministic populations."""
from pathlib import Path
import hashlib
import json
import numpy as np
from openwam.dataloader.native_stage2_profiles import fingerprint, resolve_native_profile
from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation
from openwam.dataloader.utils.normalization import apply_normalization

ROTATION_DIMS=[*range(3,9),*range(13,19),*range(37,43),*range(46,52)]


def summarize(parts):
    """Caller bounds the number of windows; invalid/padded values never enter stats."""
    values=np.concatenate([p[0] for p in parts]) if parts else np.empty((0,52))
    valid=np.concatenate([p[1] for p in parts]) if parts else np.empty((0,52),bool)
    result={k:np.zeros(52) for k in ('min','max','q01','q99','mean','std','count')}
    for i in range(52):
        a=values[:,i][valid[:,i]&np.isfinite(values[:,i])]
        result['count'][i]=len(a)
        if len(a):
            for key,value in zip(('min','max','q01','q99','mean','std'),
                                  (a.min(),a.max(),*np.quantile(a,[.01,.99]),a.mean(),a.std())):result[key][i]=value
        else:
            # Only inactive dimensions may use these identity fillers. normalize checks counts.
            for k in ('min','q01'):result[k][i]=-1
            for k in ('max','q99','std'):result[k][i]=1
    for i in ROTATION_DIMS:
        for k in ('min','q01'):result[k][i]=-1
        for k in ('max','q99','std'):result[k][i]=1
        result['mean'][i]=0
    return {k:v.astype(int).tolist() if k=='count' else v.tolist() for k,v in result.items()}


def normalize(values,mask,stats,*,source):
    counts=np.asarray(stats['count'])
    missing=np.flatnonzero(np.asarray(mask).any(0)&(counts==0))
    if len(missing):raise ValueError(f'{source}: stats_insufficient_valid_population raw dimensions {missing.tolist()}')
    arrays={k:np.asarray(v,np.float32) for k,v in stats.items() if k!='count'}
    result=apply_normalization(values,arrays,'quantile').astype(np.float32)
    result[:,[9,19]]=np.clip(values[:,[9,19]],0,1);result[~mask]=0
    return result


def audit_episode_metadata(repo):
    """Stream only episode identity/length; retain one repo's selected scalars.

    The digest records the audited selected metadata population, not a promise
    that files cannot later change. Runtime independently verifies uniqueness,
    dense frame indices, and selected bounds whenever an episode is loaded.
    """
    import numbers
    import pyarrow.parquet as pq
    selected={}
    for episode,start,end in repo['ranges']:
        selected.setdefault(episode,[]).append([start,end])
    seen={};errors=[]
    def fail(episode,reason):
        for bounds in selected[episode]:
            errors.append(dict(repo=repo['repo'],episode=episode,range=bounds,error=reason))
    files=sorted((Path(repo['path'])/'meta/episodes').glob('chunk-*/*.parquet'))
    for path in files:
        metadata=pq.ParquetFile(path)
        if not {'episode_index','length'}.issubset(metadata.schema_arrow.names):
            raise ValueError(f'{path}: missing episode_index/length metadata columns')
        for batch in metadata.iter_batches(columns=['episode_index','length'],batch_size=4096):
            for row in batch.to_pylist():
                episode=row['episode_index']
                if not isinstance(episode,numbers.Integral) or isinstance(episode,bool):
                    raise ValueError(f'{path}: invalid episode_index {episode!r}')
                if episode not in selected:continue
                if episode in seen:
                    fail(episode,'nonunique_episode_metadata');continue
                length=row['length'];seen[episode]=length
                if not isinstance(length,numbers.Integral) or isinstance(length,bool) or length<1:
                    fail(episode,f'invalid_episode_length: {length!r}');continue
                for start,end in selected[episode]:
                    if not 0<=start<end<=length:
                        errors.append(dict(repo=repo['repo'],episode=episode,range=[start,end],
                                           error=f'invalid_segment_bounds: length={length}'))
    for episode in selected.keys()-seen.keys():fail(episode,'missing_episode_metadata')
    identity=[[episode,seen[episode]] for episode in sorted(seen)]
    summary=dict(selected_episodes=len(selected),selected_ranges=len(repo['ranges']),
                 sha256=fingerprint(identity),projection=['episode_index','length'])
    return summary,errors


def audit_selection(dataset):
    """Audit every selected episode/range with bounded projected metadata reads."""
    from concurrent.futures import ThreadPoolExecutor
    from openwam.dataloader.native_stage2_io import calibration_file_hash
    groups={};sources={};errors=[];episodes_checked=0;ranges_checked=0
    def inspect(repo):
        try:
            raw=(Path(repo['path'])/'meta/info.json').read_bytes();info=json.loads(raw)
            profile=resolve_native_profile(repo['repo'],info)
            from openwam.dataloader.posttrain_physical import representation_info
            rep=MoraHeadTCPRepresentation(representation_info(info,profile),**profile['representation']);rep.contract['gripper_state_filter']=dataset.filter_config
            rep.contract['native_profile_id']=profile['profile_id']
            key=fingerprint(rep.contract)
            metadata,failures=audit_episode_metadata(repo)
            source=dict(info_sha256=hashlib.sha256(raw).hexdigest(),entry=key,profile_id=profile['profile_id'],
                        calibration_sha256=calibration_file_hash(repo['path']),episode_metadata_audit=metadata)
            return repo['repo'],source,rep.contract,profile,failures
        except Exception as exc:
            return repo['repo'],None,None,None,[dict(repo=repo['repo'],error=f'{type(exc).__name__}: {exc}')]
    # Consume every result in bounded batches: no thousands of live info objects/futures.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(dataset.repos),32):
            for name,source,contract,profile,failures in pool.map(inspect,dataset.repos[start:start+32]):
                if source:
                    episodes_checked+=source['episode_metadata_audit']['selected_episodes']
                    ranges_checked+=source['episode_metadata_audit']['selected_ranges']
                if failures:errors.extend(failures);continue
                sources[name]=source
                group=groups.setdefault(source['entry'],dict(contract=contract,profile=profile,repos=[]));group['repos'].append(name)

    return dict(contract=dataset.stats_contract(),sources=sources,groups=groups,errors=errors,
                all_videos_audited=False,metadata_repositories=len(dataset.repos),
                selected_episodes_checked=episodes_checked,selected_ranges_checked=ranges_checked)


def prepare_native_stats(dataset,output_dir,*,max_episodes_per_profile=4,max_windows_per_profile=128,sample_seed=42,audit=None):
    """Sample by profile, selected repo weight and selected episode duration.

    At most ``max_episodes_per_profile`` episodes and ``max_windows_per_profile``
    windows are encoded per contract. Sampling is explicit profile-population
    normalization, not a claim of all-window/per-repository quantiles.
    """
    if min(max_episodes_per_profile,max_windows_per_profile)<1:raise ValueError('statistics sample bounds must be positive')
    audit=audit_selection(dataset) if audit is None else audit
    if audit['errors']:raise ValueError(f'native metadata preflight failed: {audit["errors"][:8]}')
    repos={r['repo']:r for r in dataset.repos};entries={}
    for key,group in sorted(audit['groups'].items()):
        rng=np.random.default_rng(np.random.SeedSequence([sample_seed,int(key[:8],16)]))
        population=[];action_parts=[];state_parts=[];rejected=[]
        selected_repos=[repos[n] for n in group['repos']]
        weights=np.asarray([r['weight'] for r in selected_repos],float);weights/=weights.sum()
        # Draw a bounded set of unique source episodes; retries also have a hard bound.
        episode_keys=set()
        attempts=0
        while len(episode_keys)<max_episodes_per_profile and attempts<max_episodes_per_profile*16:
            attempts+=1;repo=selected_repos[int(rng.choice(len(selected_repos),p=weights))]
            ranges=repo['ranges'];duration=np.asarray([b-a for _,a,b in ranges],float)
            episode=ranges[int(rng.choice(len(ranges),p=duration/duration.sum()))][0]
            identity=(repo['repo'],episode)
            if identity in episode_keys:continue
            episode_keys.add(identity)
            m=dataset._episode(repo,episode);ranges=[r for r in ranges if r[0]==episode]
            quota=max(1,max_windows_per_profile//max_episodes_per_profile)
            for _ in range(min(quota,max_windows_per_profile-len(population))):
                try:
                    option,anchor,level=dataset._choose(m,ranges,rng)
                    _,raw,mask,state,sm,_=dataset.raw_window(m,option,anchor)
                except ValueError as exc:
                    rejected.append(dict(repo=repo['repo'],episode=episode,reason=str(exc)));break
                action_parts.append((raw,mask));state_parts.append((state,sm))
                population.append(dict(repo=repo['repo'],episode=episode,anchor=anchor,granularity=level,
                    selected_range=[option['selected_start'],option['selected_end_exclusive']],
                    instruction_interval=[option['instruction']['start'],option['instruction']['end_exclusive']]))
        if not population:raise ValueError(f'{group["repos"][0]}: stats_insufficient_valid_population')
        action=summarize(action_parts);state=summarize(state_parts)
        required=[20+7*side+i for side,name in enumerate(('left','right')) for i in range(len(group['contract']['joint_indices'][name]))]
        for stream,stats in [('action',action),('state',state)]:
            missing=[i for i in required if not stats['count'][i]]
            if missing:raise ValueError(f'{group["repos"][0]}:{group["contract"]["joint_state_key"]}: stats_insufficient_valid_population {stream} {missing}')
        entries[key]=dict(contract=group['contract'],action=action,state=state,population=population,rejected=rejected,
                          scope='explicit profile-shared deterministic sampled population',sampled_episodes=len(episode_keys))
    payload=dict(contract=dataset.stats_contract(),preparation=dict(seed=sample_seed,max_episodes_per_profile=max_episodes_per_profile,
        max_windows_per_profile=max_windows_per_profile,population_rule='profile balanced; weighted selected repos; selected-duration episodes; 1:1 task/subtask eligible anchors'),
        sources=audit['sources'],entries=entries)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    path=output/('registry-'+hashlib.sha256(raw).hexdigest()+'.json')
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('immutable registry content mismatch')
    else:
        with path.open('xb') as stream:stream.write(raw)
    return str(path)


def read_registry(path,contract):
    path=Path(path);raw=path.read_bytes()
    if path.name!='registry-'+hashlib.sha256(raw).hexdigest()+'.json':raise ValueError('stats fingerprint_mismatch: immutable registry filename/content')
    payload=json.loads(raw)
    if payload['contract']!=contract:raise ValueError(f'stats selection/contract fingerprint_mismatch: {path}')
    return payload


def lookup_stats(registry,repo,metadata):
    source=registry['sources'].get(repo['repo'])
    if source is None:raise ValueError(f'{repo["repo"]}: stats_entry_missing')
    key=fingerprint(metadata['representation'].contract)
    if source['info_sha256']!=metadata['info_hash'] or source['entry']!=key or source.get('calibration_sha256')!=metadata['calibration_sha256']:
        raise ValueError(f'{repo["repo"]}: stats source/schema fingerprint_mismatch')
    entry=registry['entries'].get(key)
    if entry is None:raise ValueError(f'{repo["repo"]}: stats_entry_missing {key}')
    if entry['contract']!=metadata['representation'].contract:raise ValueError(f'{repo["repo"]}: stats contract fingerprint_mismatch')
    return entry
