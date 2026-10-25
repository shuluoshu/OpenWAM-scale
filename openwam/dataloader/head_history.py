"""Strictly past head observations using validated source row/video ordinals."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch


@dataclass(frozen=True)
class HistorySamplingConfig:
    enabled: bool = False
    num_frames: int = 10
    mean_interval_s: float = 1.0
    jitter_fraction: float = 0.5

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.num_frames) is not int or self.num_frames != 10:
            raise ValueError('history requires enabled bool and ten slots')
        if self.mean_interval_s != 1.0 or self.jitter_fraction != .5:
            raise ValueError('history requires one-second mean and 0.5 jitter')


def select_history_frames(anchor, fps, start, config, rng, training):
    """Return oldest-first ordinals, realized ages and mask; never fill duplicates."""
    if not math.isfinite(fps) or fps <= 0 or not 0 <= start <= anchor:
        raise ValueError('invalid history bounds/fps')
    indices=np.full(config.num_frames,-1,np.int64)
    ages=np.zeros(config.num_frames,np.float32)
    valid=np.zeros(config.num_frames,bool)
    if not config.enabled:return indices,ages,valid
    gaps=(rng.uniform(.5,1.5,config.num_frames) if training else np.ones(config.num_frames))
    candidates=anchor-np.rint(np.cumsum(gaps)*fps).astype(np.int64)
    seen=set()
    for slot,index in enumerate(candidates[::-1]):
        if start<=index<anchor and int(index) not in seen:
            seen.add(int(index));indices[slot]=index;ages[slot]=(anchor-index)/fps;valid[slot]=True
    return indices,ages,valid


def materialize_head_history(root, info, ep, head, anchor, start, config, rng, training,
                             sequential=False, dtype=torch.float32, source_valid=None,video_ordinals=None):
    """Decode past rows; ages use source row / nominal FPS, not sensor time."""
    if not config.enabled:return {},{}
    if sequential and video_ordinals is None:raise ValueError('sequential history requires explicit video ordinals')
    from scripts.data.prepare_subtask_goals import decode_frames
    from openwam.model.goal.runtime import clean_goal_head
    prefix='videos/'+head+'/'
    try:
        fps=float(info['fps']);offset=float(ep[prefix+'from_timestamp'])
        chunk,file=(ep[prefix+k] for k in ('chunk_index','file_index'))
        if any(isinstance(v,bool) or int(v)!=v or v<0 for v in (chunk,file)):
            raise ValueError('invalid chunk/file')
        if not math.isfinite(offset) or offset<0 or not math.isfinite(fps) or fps<=0:
            raise ValueError('invalid timestamp/fps')
        if sequential and offset!=0:raise ValueError('VFR nonzero time offset needs an explicit ordinal origin')
        if abs(offset*fps-round(offset*fps))>1e-3:raise ValueError('nonintegral video offset')
        if not 0<=start<=anchor<int(ep['length']):raise ValueError('invalid episode bounds')
        path=Path(root)/info['video_path'].format(video_key=head,episode_index=int(ep['episode_index']),chunk_index=int(chunk),file_index=int(file))
        end=ep.get(prefix+'to_timestamp')
        if end is not None and (not math.isfinite(float(end)) or float(end)<offset+anchor/fps):
            raise ValueError('invalid segment bounds')
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        raise ValueError(f'malformed present head video metadata: {head}: {exc}') from exc
    if not path.is_file():raise FileNotFoundError(path)
    indices,ages,valid=select_history_frames(anchor,fps,start,config,rng,training)
    if source_valid is not None:
        source_valid=np.asarray(source_valid)
        if source_valid.shape != (int(ep['length']),) or source_valid.dtype != np.bool_:
            raise ValueError('history source_valid must be bool [episode length]')
        slots=np.flatnonzero(valid)
        rejected=slots[~source_valid[indices[slots]]]
        valid[rejected]=False;indices[rejected]=-1;ages[rejected]=0
    pixels=torch.zeros((config.num_frames,3,1,256,320),dtype=dtype)
    requested=indices[valid].tolist();alignment={}
    if requested:
        images,alignment=decode_frames(path,offset,fps,requested,sequential=sequential,video_ordinals=video_ordinals)
        for slot in np.flatnonzero(valid):pixels[slot]=clean_goal_head(images[int(indices[slot])]).to(dtype=dtype)
    trace=dict(source_video=str(path),head=head,anchor=int(anchor),selected_start=int(start),
        timestamp_source=('source_row_ordinal_over_nominal_fps' if sequential else 'validated_video_cfr_ordinal_over_fps'),fps=fps,video_offset_s=offset,
        indices=indices.tolist(),ages_s=ages.tolist(),valid=valid.tolist(),alignment=alignment)
    return dict(history_images=pixels,history_ages=torch.from_numpy(ages),history_valid=torch.from_numpy(valid)),trace
