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
    if mode not in ('global_shuffle', 'sequential'):
        raise ValueError('sample_order must be global_shuffle or explicit sequential')
    return dict(version=1, mode=mode, seed=seed, total_samples=total,
                algorithm='numpy.PCG64.shuffle' if mode == 'global_shuffle' else 'identity',
                numpy_version=np.__version__ if mode == 'global_shuffle' else None,
                dtype='uint32' if total <= 2**32 else 'uint64')


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
        expected = sampling_contract(dict(sample_order=contract.get('mode'), seed=contract.get('seed')),
                                     contract.get('total_samples'))
        if contract != expected:
            raise ValueError('finite sampling contract mismatch (algorithm/runtime changed)')
        self.contract = expected
        self.total = expected['total_samples']
        self._order = None
        if expected['mode'] == 'global_shuffle':
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
