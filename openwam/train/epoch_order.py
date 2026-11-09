"""Deterministic global finite-epoch order, independent of model/data RNG state."""
from __future__ import annotations

import hashlib
import numpy as np


def sampling_contract(config, total):
    """Bind sample order to checkpoints; old unversioned cursors cannot resume."""
    seed = config.get('seed', 42)
    mode = config.get('sample_order', 'global_shuffle')
    if isinstance(total, bool) or not isinstance(total, int) or not 0 < total <= 2**63 - 1:
        raise ValueError('finite population must be a positive integer below 2**63')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError('sampling seed must be an unsigned 64-bit integer')
    if mode == 'filtered_global_shuffle':
        parent = config.get('parent_total_samples')
        if type(parent) is not int or not total <= parent <= 2**63 - 1:
            raise ValueError('invalid parent population')
        ranges = config.get('excluded_sample_ranges')
        if not isinstance(ranges, list):
            raise ValueError('excluded_sample_ranges must be a list')
        end = removed = 0
        canonical = []
        for pair in ranges:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(v) is not int for v in pair)
                    or not end <= pair[0] < pair[1] <= parent):
                raise ValueError('invalid or overlapping exclusion interval')
            start, end = pair
            removed += end - start
            canonical.append([start, end])
        if parent - removed != total:
            raise ValueError('filtered population count mismatch')
        return dict(version=1, mode=mode, seed=seed, total_samples=total,
                    algorithm='numpy.PCG64.shuffle.filter_compact.v1',
                    numpy_version=np.__version__, dtype='uint32' if total <= 2**32 else 'uint64',
                    parent_total_samples=parent, excluded_sample_ranges=canonical)
    if mode not in ('global_shuffle', 'sequential'):
        raise ValueError('sample_order must be global_shuffle or explicit sequential')
    return dict(version=1, mode=mode, seed=seed, total_samples=total,
                algorithm='numpy.PCG64.shuffle' if mode == 'global_shuffle' else 'identity',
                numpy_version=np.__version__ if mode == 'global_shuffle' else None,
                dtype='uint32' if total <= 2**32 else 'uint64')


def _filtered_parent(contract):
    parent = contract['parent_total_samples']
    order = np.arange(parent, dtype='uint32' if parent <= 2**32 else 'uint64')
    np.random.Generator(np.random.PCG64(contract['seed'])).shuffle(order)
    keep = np.ones(parent, dtype=bool)
    for start, end in contract['excluded_sample_ranges']:
        keep[start:end] = False
    return order, keep


def retained_parent_cursor(contract, cursor):
    """Count retained parent draws before a committed checkpoint position."""
    expected = sampling_contract({**contract, 'sample_order': contract.get('mode')},
                                 contract.get('total_samples'))
    if expected != contract or contract['mode'] != 'filtered_global_shuffle':
        raise ValueError('filtered sampling contract mismatch')
    if type(cursor) is not int or not 0 <= cursor <= contract['parent_total_samples']:
        raise ValueError('invalid parent cursor')
    order, keep = _filtered_parent(contract)
    return int(np.count_nonzero(keep[order[:cursor]]))


class EpochOrder:
    """Map a committed global cursor position to its immutable dataset ID.

    All ranks construct the same permutation once using a private RNG. A 53.8M
    population uses 205 MiB of host RAM per rank, no GPU RAM or CPFS permutation
    file. The checkpoint stores a position, not a dataset ID; rebuilding this
    exact order resumes without skipping or repeating positive-weight samples.
    """
    def __init__(self, contract):
        if not isinstance(contract, dict):
            raise ValueError('missing finite sampling contract; start a new shuffled run')
        expected = sampling_contract({**contract, 'sample_order': contract.get('mode')},
                                     contract.get('total_samples'))
        if contract != expected:
            raise ValueError('finite sampling contract mismatch (algorithm/runtime changed)')
        self.contract = expected
        self.total = expected['total_samples']
        self._order = None
        if expected['mode'] == 'filtered_global_shuffle':
            parent_order, keep = _filtered_parent(expected)
            compact = np.cumsum(keep, dtype=expected['dtype']) - 1
            self._order = compact[parent_order[keep[parent_order]]]
            self._order.flags.writeable = False
            self.sha256 = hashlib.sha256(memoryview(self._order).cast('B')).hexdigest()
        elif expected['mode'] == 'global_shuffle':
            self._order = np.arange(self.total, dtype=expected['dtype'])
            np.random.Generator(np.random.PCG64(expected['seed'])).shuffle(self._order)
            self._order.flags.writeable = False
            self.sha256 = hashlib.sha256(memoryview(self._order).cast('B')).hexdigest()
        else:
            self.sha256 = hashlib.sha256(f'identity:{self.total}'.encode()).hexdigest()

    @property
    def nbytes(self):
        return self._order.nbytes if self._order is not None else 0

    def __getitem__(self, position):
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)) or not 0 <= position < self.total:
            raise IndexError(position)
        return int(self._order[position]) if self._order is not None else int(position)
