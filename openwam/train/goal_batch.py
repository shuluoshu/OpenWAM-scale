"""True Stage1 batches and bounded, ordered CPU-only spawn readers.

Workers receive only a dataset and (index, seed) requests. Text inference and
all device transfers stay in the training process. Prefetch never owns the
committed cursor; reconstructing a loader from that cursor discards unread work.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields
from pathlib import Path
import random
import uuid

import numpy as np
import torch


def prepare_goal_batch(pairs, *, device, dtype, context_cache=None):
    """Collate singleton dataset samples; preserve target/history/view masks."""
    from openwam.train.goal_curated import prepare_batch
    from openwam.model.goal.history import HISTORY_FIELDS
    pairs=list(pairs)
    if not pairs:raise ValueError('cannot collate an empty goal batch')
    batches=[prepare_batch(record,batch,device=device,dtype=dtype,context_cache=context_cache)
             for record,batch in pairs]
    if any(b['observed'].shape[0]!=1 or b['context'].shape[0]!=1 for b in batches):
        raise ValueError('goal collation requires singleton dataset samples')
    result=dict(batches[0])
    result['observed']=torch.cat([b['observed'] for b in batches])
    target_type=type(batches[0]['targets'])
    result['targets']=target_type(**{f.name:torch.cat([getattr(b['targets'],f.name) for b in batches])
                                   for f in fields(target_type)})
    # Keep stored context lengths, including contexts with pre-existing padding.
    result['context']=torch.nn.utils.rnn.pad_sequence([b['context'][0] for b in batches],batch_first=True)
    lens=[]
    for batch in batches:
        value=batch.get('context_lens')
        lens.append(int(torch.as_tensor(value).reshape(-1)[0]) if value is not None else batch['context'].shape[1])
    result['context_lens']=torch.tensor(lens,device=device,dtype=torch.long)
    if any('observed_view_valid' in b for b in batches):
        template=next(b['observed_view_valid'] for b in batches if 'observed_view_valid' in b)
        result['observed_view_valid']=torch.cat([b.get('observed_view_valid',torch.ones_like(template)) for b in batches])
    for key in HISTORY_FIELDS:
        present=[key in b for b in batches]
        if any(present):
            if not all(present):raise ValueError('inconsistent goal history fields: '+key)
            result[key]=torch.cat([b[key] for b in batches])
    result['sources']=[source for b in batches for source in b['sources']]
    result['records']=[r for r,_ in pairs]
    return result


def rank_batch(cursor,total,rank,world,batch_size):
    """Interleaved real positions, real count, DDP sample weight, global advance.

    Partial ranks use smaller batches, never padding inside a real batch. Only
    an empty rank executes a singleton dummy with zero loss. DDP averages local
    masked objectives weighted by real sample count; each local objective keeps
    the model's existing valid-image/valid-point normalization.
    """
    if (any(type(v) is not int for v in (cursor,total,rank,world,batch_size)) or
            not 0<=cursor<total or not 0<=rank<world or batch_size<1):
        raise ValueError('invalid finite batch cursor/rank/size')
    consumed=min(world*batch_size,total-cursor)
    positions=list(range(cursor+rank,cursor+consumed,world))
    count=len(positions)
    return positions or [total-1],count,world*count/consumed,consumed


def _identity_collate(samples):return samples


def _worker_init(worker_id):
    """Each worker owns its readers and scratch image/NPZ slots."""
    torch.set_num_threads(1)
    wrapper=torch.utils.data.get_worker_info().dataset
    dataset=wrapper.dataset
    while hasattr(dataset,'dataset'):dataset=dataset.dataset
    if hasattr(dataset,'cache_dir'):
        dataset.cache_dir=Path(dataset.cache_dir)/'workers'/wrapper.run_id/f'worker_{worker_id}'
        dataset.cache_dir.mkdir(parents=True,exist_ok=True)


class _CPURequests(torch.utils.data.Dataset):
    def __init__(self,dataset,cache_size,run_id):
        self.dataset=dataset;self.cache_size=cache_size;self.cache=OrderedDict();self.run_id=run_id
    def __getitem__(self,request):
        index,seed=request
        key=(index,seed)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        # Isolate data RNG from model RNG, including synchronous reading. Seed
        # only the CPU generator: worker code must never touch CUDA state.
        python_state=random.getstate();numpy_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(seed);np.random.seed(seed%(2**32));torch.random.default_generator.manual_seed(seed)
                result=self.dataset.sample(index)
        finally:
            random.setstate(python_state);np.random.set_state(numpy_state)
        if self.cache_size:
            self.cache[key]=result
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return result


class GoalBatchLoader:
    """Lazy ordered sample readers; the parent assembles true training batches.

    Worker tasks are single samples, so even the first large training batch uses
    every reader. Outstanding worker work is bounded to workers * prefetch_factor
    samples, in addition to the batch being assembled/processed in the parent.
    """
    def __init__(self,dataset,order,*,cursor,rank,world,batch_size=1,num_workers=0,prefetch_factor=2,
                 seed=42,repeat_population=False,max_batches=None,sample_cache_size=0):
        for name,value,minimum in (('batch_size',batch_size,1),('num_workers',num_workers,0),
                                  ('prefetch_factor',prefetch_factor,1),('sample_cache_size',sample_cache_size,0)):
            if type(value) is not int or value<minimum:raise ValueError(name+' has invalid value')
        self.dataset=dataset;self.order=order;self.cursor=cursor;self.rank=rank;self.world=world
        self.batch_size=batch_size;self.seed=seed;self.repeat_population=repeat_population;self.max_batches=max_batches
        self.read_cursor=cursor;self.read_batches=0
        self.wrapper=_CPURequests(dataset,sample_cache_size,uuid.uuid4().hex)
        kwargs=dict(dataset=self.wrapper,batch_size=None,sampler=self._requests(),num_workers=num_workers,
                    collate_fn=_identity_collate,generator=torch.Generator().manual_seed(seed))
        if num_workers:kwargs.update(multiprocessing_context='spawn',worker_init_fn=_worker_init,prefetch_factor=prefetch_factor)
        self.loader=torch.utils.data.DataLoader(**kwargs);self.iterator=None
    def _requests(self):
        cursor=self.cursor;update=0;total=len(self.dataset)
        while (cursor<total or self.repeat_population) and (self.max_batches is None or update<self.max_batches):
            positions,_,_,consumed=rank_batch(cursor%total,total,self.rank,self.world,self.batch_size)
            for position in positions:
                index=self.order[position]
                yield index,self.seed+index+(cursor//total)*total
            cursor+=consumed;update+=1
    def __iter__(self):return self
    def __next__(self):
        total=len(self.dataset)
        if ((self.read_cursor>=total and not self.repeat_population) or
                (self.max_batches is not None and self.read_batches>=self.max_batches)):
            raise StopIteration
        positions,_,_,consumed=rank_batch(self.read_cursor%total,total,self.rank,self.world,self.batch_size)
        if self.iterator is None:self.iterator=iter(self.loader)
        # A failed partial assembly never changes the trainer's consumed cursor.
        result=[next(self.iterator) for _ in positions]
        self.read_cursor+=consumed;self.read_batches+=1
        return result
    def close(self):
        if self.iterator is not None and hasattr(self.iterator,'_shutdown_workers'):self.iterator._shutdown_workers()
        self.iterator=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
