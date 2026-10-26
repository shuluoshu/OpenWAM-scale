"""Exercise the actual trainer, including shuffled resume and missing TCP GT."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch


def test_trainer_uses_shuffle_logs_supervision_and_resumes_same_samples(tmp_path):
    from openwam.train.goal_epoch import train_epoch
    from openwam.model.goal.dependencies import dependency_identity
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    torch.set_num_threads(2)
    torch.manual_seed(12)
    expert = make_query()
    obs, ctx, target, valid = multiview_batch(1)

    class Data:
        def __init__(self, fail_after=None):
            self.seen = []
            self.fail_after = fail_after

        def __len__(self): return 7

        def row(self, index):
            targets = target if index % 2 else replace(target, point_valid=torch.zeros_like(target.point_valid))
            return dict(id=str(index), family=f'source{index // 2}', instruction='move',
                        granularity='subtask' if index % 2 else 'task'), dict(
                observed=obs, context=ctx, targets=targets, observed_view_valid=valid)

        def sample(self, index):
            if self.fail_after is not None and len(self.seen) == self.fail_after:
                raise RuntimeError('injected interruption')
            self.seen.append(index)
            return self.row(index)

        def evaluation_samples(self): return [self.row(1)]
        def scope_summary(self): return {'total_samples': 7}

    identity = dict(index_sha256='a'*64, source_yaml_sha256='b'*64, total_samples=7,
                    world_size=1, seed=42, config_sha256='c'*64,
                    sampling=dict(version=1, mode='global_shuffle', seed=42, total_samples=7,
                                  algorithm='numpy.PCG64.shuffle', numpy_version=np.__version__, dtype='uint32'))
    deps = dependency_identity(expert.codec.vae, text_dim=expert.text_dim, precomputed_text_fingerprint='d'*64)
    kwargs = dict(identity=identity, dependencies=deps, checkpoint_every=3, eval_every=7, sampling_steps=2)
    interrupted = Data(fail_after=3)
    with pytest.raises(RuntimeError, match='injected interruption'):
        train_epoch(expert, interrupted, None, tmp_path/'run', **kwargs)
    assert interrupted.seen != [0, 1, 2], 'the real trainer still consumes source-contiguous indices'
    resumed = Data()
    clone = make_query()
    train_epoch(clone, resumed, None, tmp_path/'run', resume=True, **kwargs)
    full = Data()
    torch.manual_seed(12)
    uninterrupted = make_query()
    train_epoch(uninterrupted, full, None, tmp_path/'full', **kwargs)
    assert interrupted.seen + resumed.seen == full.seen
    assert sorted(full.seen) == list(range(7))
    for name, tensor in uninterrupted.state_dict().items():
        torch.testing.assert_close(tensor, clone.state_dict()[name], rtol=0, atol=0)
    rows = [json.loads(s) for s in (tmp_path/'full/progress.jsonl').read_text().splitlines()]
    updates = [r for r in rows if r['event'] == 'update']
    for index, update in zip(full.seen, updates):
        assert update['family_counts'] == {f'source{index // 2}': 1}
        assert update['granularity_counts'] == {'subtask' if index % 2 else 'task': 1}
        assert update['valid_tcp_points'] == (2 if index % 2 else 0)
        assert update['samples_with_tcp'] == (1 if index % 2 else 0)
        if index % 2 == 0:
            assert update['tcp_loss'] == update['query_tcp_loss'] == 0
    assert rows[0]['sampling']['mode'] == 'global_shuffle'
