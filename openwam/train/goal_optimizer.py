"""Stage1 optimizer contracts and GPU ZeRO-1 with FP32 master updates.

DDP still reduces BF16 model gradients. PyTorch partitions parameter ownership;
only the owning rank allocates FP32 masters/moments and updates that parameter.
The resulting BF16 weights are broadcast by ZeroRedundancyOptimizer.step().
"""
from __future__ import annotations

import copy
import math

import torch
import torch.distributed as dist


def optimizer_config(config):
    """Validate options without creating CUDA state or a process group."""
    sharding = config.get('optimizer_sharding', 'none')
    device = config.get('optimizer_device', 'model')
    if sharding not in ('none', 'zero1'):
        raise ValueError('optimizer_sharding must be none or zero1')
    if device not in ('model', 'cpu'):
        raise ValueError('optimizer_device must be model or cpu')
    if sharding == 'zero1' and device != 'model':
        raise ValueError('zero1 requires optimizer_device=model (GPU optimizer)')
    betas = config.get('adam_betas', (.9, .999))
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or not 0 <= v < 1 for v in betas)):
        raise ValueError('adam_betas must contain two finite values in [0,1)')
    result = dict(sharding=sharding, device=device, betas=tuple(betas))
    for key, default, positive in (('adam_eps', 1e-8, True), ('weight_decay', .01, False)):
        value = config.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or (value <= 0 if positive else value < 0)):
            raise ValueError(f'invalid {key}')
        result['eps' if key == 'adam_eps' else key] = float(value)
    return result


class _FP32ShardAdamW(torch.optim.Optimizer):
    """Local optimizer for a PyTorch-owned shard; checkpoint state stays FP32."""

    def __init__(self, params, lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            for parameter in group['params']:
                self.state[parameter]['master'] = parameter.detach().float().clone()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Stage1 sharded AdamW does not use closures')
        from torch.optim._functional import adamw
        for group in self.param_groups:
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError('Stage1 sharded AdamW requires dense gradients')
                state = self.state[parameter]
                master = state['master']
                if 'step' not in state:
                    state.update(step=torch.tensor(0., device='cpu'),
                                 exp_avg=torch.zeros_like(master), exp_avg_sq=torch.zeros_like(master))
                # One FP32 gradient at a time bounds transient device memory.
                gradient = gradient.float()
                parameter.grad = None
                adamw([master], [gradient], [state['exp_avg']], [state['exp_avg_sq']], [],
                      [state['step']], foreach=False, capturable=False, differentiable=False,
                      fused=False, amsgrad=False, beta1=group['betas'][0], beta2=group['betas'][1],
                      lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'], maximize=False)
                parameter.copy_(master)


class ShardedFP32MasterAdamW:
    """FP32 master weights and Adam moments partitioned across DDP ranks.

    ``prepare_checkpoint`` is collective; ``checkpoint_payload`` is rank-zero
    only. Restoring never casts saved FP32 masters/moments through BF16 model
    dtype. CPU/Gloo is supported for exact small-model distributed tests.
    """
    format = 'openwam.zero1_fp32.v1'

    def __init__(self, parameter_groups, *, betas=(.9, .95), eps=1e-8, weight_decay=.01):
        if not dist.is_initialized():
            raise ValueError('zero1 optimizer requires an initialized distributed process group')
        from torch.distributed.optim import ZeroRedundancyOptimizer
        self.optimizer = ZeroRedundancyOptimizer(parameter_groups, optimizer_class=_FP32ShardAdamW,
                                                betas=betas, eps=eps, weight_decay=weight_decay,
                                                overlap_with_ddp=False)

    @property
    def local_optimizer(self):
        return self.optimizer.optim

    def zero_grad(self, set_to_none=True):
        if not set_to_none:
            raise ValueError('FP32 master optimizer requires set_to_none=True')
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()
        self.zero_grad()

    def prepare_checkpoint(self):
        from openwam.train.optimizer_checkpoint import consolidate_fp32_optimizer
        consolidate_fp32_optimizer(self.optimizer)

    def checkpoint_payload(self):
        return dict(format=self.format, adam=self.optimizer.state_dict())

    def restore_payload(self, payload):
        if payload.get('format') != self.format:
            raise ValueError('optimizer checkpoint format differs; start a new run')
        state = copy.deepcopy(payload['adam'])
        for entry in state['state'].values():
            if entry['master'].dtype != torch.float32:
                raise ValueError('invalid FP32 master checkpoint')
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in entry and entry[key].dtype != torch.float32:
                    raise ValueError('invalid FP32 Adam state checkpoint')
        # PyTorch ZRO loads local states without the dtype conversion performed
        # by Optimizer.load_state_dict for BF16 parameters. Its outer state is
        # only an unused global copy; remove that copy after restoring.
        self.optimizer.load_state_dict(state)
        self.optimizer.state.clear()
        with torch.no_grad():
            for parameter, entry in self.local_optimizer.state.items():
                if entry['master'].shape != parameter.shape:
                    raise ValueError('optimizer parameter shape mismatch')
                # ZRO treats every scalar state tensor like Adam's step and
                # moves it to CPU. Scalar trainable gates also have scalar
                # masters/moments, which must instead follow their parameter.
                for key in ('master', 'exp_avg', 'exp_avg_sq'):
                    if key in entry:
                        entry[key] = entry[key].to(device=parameter.device, dtype=torch.float32)
                parameter.copy_(entry['master'])


def build_goal_optimizer(parameter_groups, config):
    options = optimizer_config(config)
    if options['sharding'] == 'zero1':
        return ShardedFP32MasterAdamW(parameter_groups, betas=options['betas'],
                                     eps=options['eps'], weight_decay=options['weight_decay'])
    from openwam.train.goal_overfit import FP32MasterAdamW
    groups = [{**group, 'betas': options['betas'], 'eps': options['eps']} for group in parameter_groups]
    return FP32MasterAdamW(groups, weight_decay=options['weight_decay'],
                           master_device='cpu' if options['device'] == 'cpu' else None)
