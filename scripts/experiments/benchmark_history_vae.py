#!/usr/bin/env python3
"""Bounded DSW baseline/candidate measurement; never launches a DLC job.

Prepare real shuffled source samples once, then run each checkout in a fresh
process against identical serialized CPU tensors. No T5 or optimizer is timed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--variant', choices=['baseline', 'candidate'], default='candidate')
    ap.add_argument('--count', type=int, default=8)
    ap.add_argument('--encode-batch-size', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--condition-reference', type=Path)
    ap.add_argument('--stage1-bundle', type=Path)
    a = ap.parse_args()
    if min(a.count, a.encode_batch_size, a.repeats) < 1 or a.warmup < 0:
        ap.error('positive batch/repetition counts and nonnegative warmup required')
    sys.path.insert(0, str(a.repo.resolve()))
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from openwam.train.goal_curated import load_config
    cfg = load_config(a.experiment/'h100_joint_stage1.yaml')
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def emit(event, **kw):
        print(json.dumps(dict(event=event, **kw)), flush=True)

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):h.update(chunk)
        return h.hexdigest()

    def gpu_snapshot():
        return subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=10).stdout.strip()

    if a.prepare:
        from openwam.dataloader.curated_goal import CuratedGoalDataset
        from openwam.dataloader.goal_epoch import FiniteGoalDataset
        from openwam.train.epoch_order import EpochOrder
        options = dict(cfg['dataset'])
        options['cache_dir'] = str(a.output.parent/'reader_cache')
        data = CuratedGoalDataset(**options, seed=cfg['seed'])
        stage1 = a.experiment/'h100_joint/stage1'
        pop = json.loads((stage1/'population.json').read_text())
        identity = json.loads((stage1/'identity.json').read_text())
        finite = FiniteGoalDataset(data, a.experiment/'h100_joint/population.sqlite', _validated_summary=pop)
        order = EpochOrder(identity['sampling'])
        cursor = json.loads((stage1/'summary.json').read_text())['seen_samples']
        selected = []
        episodes = set()
        for position in range(cursor, min(len(order._order), cursor + 32 * 256), 32):
            record, batch = finite.sample(order[position])
            episode = (record['source_repo'], record['source_episode'])
            if int(batch['history_valid'].sum()) < 8 or episode in episodes:
                continue
            episodes.add(episode)
            selected.append((record, batch))
            emit('prepared_sample', count=len(selected), family=record['family'],
                 valid_history=int(batch['history_valid'].sum()), anchor=record['anchor_source_frame'])
            if len(selected) == a.count:
                break
        if len(selected) != a.count:
            raise RuntimeError('not enough complete real histories in bounded source draws')
        names = ['observed', 'observed_view_valid', 'history_images', 'history_valid', 'history_ages', 'history_episode_id']
        tensors = {k: torch.cat([b[k] for _, b in selected]).contiguous() for k in names}
        for k in ['image', 'image_valid', 'uv', 'point_valid', 'in_frame', 'visibility']:
            tensors['target_' + k] = torch.cat([getattr(b['targets'], k) for _, b in selected]).contiguous()
        save_file(tensors, str(a.fixture))
        a.fixture.with_suffix('.json').write_text(json.dumps([r for r, _ in selected], ensure_ascii=False, indent=2))
        a.output.write_text(json.dumps(dict(fixture=str(a.fixture), sha256=digest(a.fixture),
            cases=a.count, sampling_order_sha256=order.sha256), indent=2) + '\n')
        emit('fixture_saved', path=str(a.fixture), cases=a.count)
        return

    from openwam.model.goal.codec import GoalCodec
    from openwam.model.goal.history import HeadFeatureCache
    from openwam.train.goal_trainer import _construct, load_component_weights
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for throughput measurement')
    device, dtype = torch.device('cuda'), torch.bfloat16
    fixture = {k: v[:a.count] for k, v in load_file(str(a.fixture)).items()}
    if len(fixture['observed']) != a.count:
        raise ValueError('fixture is smaller than requested batch')
    gpu_at_start = gpu_snapshot()
    emit('loading_vae', gpu_at_start=gpu_at_start)
    vae = _construct(cfg['components']['vae'])
    load_component_weights(vae, cfg['baseline_checkpoint'], 'video_backbone.vae.')
    vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
    codec = GoalCodec(vae)
    if a.variant == 'candidate':
        codec.max_encode_batch_size = a.encode_batch_size
    codec.enable_cache(max_bytes=536870912)
    head = HeadFeatureCache(256)
    counts = {'encode_calls': 0, 'batch_encode_calls': 0, 'encoded_frames': 0}
    original_encode, original_batch = vae.encode, vae.batch_encode

    def encode(videos, *args, **kw):
        counts['encode_calls'] += 1
        counts['encoded_frames'] += len(videos)
        return original_encode(videos, *args, **kw)

    def batch_encode(videos, *args, **kw):
        counts['batch_encode_calls'] += 1
        counts['encoded_frames'] += len(videos)
        return original_batch(videos, *args, **kw)

    vae.encode, vae.batch_encode = encode, batch_encode
    valid = fixture['history_valid'].to(device)
    ids = fixture['history_episode_id'].to(device)
    pixels = fixture['history_images'].to(dtype=dtype)
    result = dict(variant=a.variant, repo=str(a.repo), count=a.count,
                  dtype=str(dtype), encode_batch_size=a.encode_batch_size,
                  valid_history_frames=int(valid.sum()), warmup=a.warmup, repeats=a.repeats,
                  gpu=torch.cuda.get_device_name(), scope='real VAE and exported real source histories')
    result['gpu_at_process_start'] = gpu_at_start
    result['fixture_sha256'] = digest(a.fixture)
    result['repo_head'] = subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', str(a.repo), 'diff', '10dc74e', '--', 'openwam'])
    result['implementation_diff_sha256'] = hashlib.sha256(diff).hexdigest()
    result['foundation_checkpoint'] = str(cfg['baseline_checkpoint'])
    result['gpu_before'] = gpu_snapshot()

    def history_call():
        if a.variant == 'baseline':
            return head.encode(codec, pixels.to(device), valid, episode_ids=ids)
        return head.encode(codec, pixels, valid, episode_ids=ids, device=device)

    def measure(call, clear, label):
        times = []
        if clear:
            head.clear(); codec.clear_cache()
        else:
            call()  # Prime outside the measured warm-cache interval.
        before = dict(counts)
        torch.cuda.reset_peak_memory_stats()
        output = None
        for i in range(a.warmup + a.repeats):
            if clear:
                head.clear(); codec.clear_cache()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = call()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if i >= a.warmup:
                times.append(elapsed)
            emit('measurement', phase=label, iteration=i, seconds=elapsed)
        value = dict(seconds=times, median_s=statistics.median(times),
                     samples_per_s=a.count/statistics.median(times),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     calls_including_warmup={k: counts[k]-before[k] for k in counts})
        return value, output

    result['history_cold'], latent = measure(history_call, True, 'history_cold')
    result['history_warm'], _ = measure(history_call, False, 'history_warm')
    latent_path = a.output.with_suffix('.latents.safetensors')
    save_file({'history': latent.detach().cpu().contiguous(), 'valid': fixture['history_valid']}, str(latent_path))
    if a.reference:
        reference = load_file(str(a.reference))['history'].float()
        actual = latent.detach().cpu().float()
        delta = (actual-reference)[fixture['history_valid']]
        source = reference[fixture['history_valid']]
        relative_l2 = float(delta.norm()/source.norm().clamp_min(1e-12))
        result['numerics'] = dict(max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()),
                                  relative_l2=relative_l2, acceptance_relative_l2=0.01,
                                  passed=relative_l2 <= 0.01 and bool(torch.isfinite(actual).all()))

    if a.stage1_bundle:
        from openwam.train.goal_visual_history import load_visual_history_bundle
        from openwam.model.goal.types import GoalTargets
        emit('loading_stage1')
        with safe_open(str(a.stage1_bundle), framework='pt', device='cpu') as f:
            meta = json.loads(f.metadata()['visual_history'])
        expert = load_visual_history_bundle(a.stage1_bundle, vae=vae,
            text_dim=meta['dependencies']['text_dim'],
            precomputed_text_fingerprint=meta['dependencies']['text']['fingerprint'], device=device, dtype=dtype)
        expert.train_stage_a()
        expert.use_gradient_checkpointing = True
        expert.codec.enable_cache(max_bytes=536870912)
        if a.variant == 'candidate':expert.codec.max_encode_batch_size = a.encode_batch_size
        head, codec = expert._history_cache, expert.codec
        context = torch.zeros(a.count, 512, meta['dependencies']['text_dim'], device=device, dtype=dtype)
        last_condition = {}

        def training_call():
            expert.zero_grad(set_to_none=True)
            torch.manual_seed(42)
            history = pixels.to(device) if a.variant == 'baseline' else pixels
            targets = GoalTargets(**{k: fixture['target_' + k].to(device)
                for k in ['image', 'image_valid', 'uv', 'point_valid', 'in_frame', 'visibility']})
            output = expert(fixture['observed'].to(device), context, targets,
                observed_view_valid=fixture['observed_view_valid'].to(device),
                history_images=history, history_valid=valid,
                history_ages=fixture['history_ages'].to(device), history_episode_id=ids)
            output['loss'].backward()
            if not torch.isfinite(output['loss']):raise FloatingPointError('nonfinite benchmark loss')
            last_condition['tokens'] = output['condition_tokens'].detach()
            return float(output['loss'].detach())

        result['stage1_forward_backward_cold'], loss = measure(training_call, True, 'stage1_forward_backward_cold')
        result['stage1_loss'] = loss
        condition = last_condition['tokens'].cpu().contiguous()
        save_file({'condition': condition}, str(a.output.with_suffix('.condition.safetensors')))
        if a.condition_reference:
            expected = load_file(str(a.condition_reference))['condition'].float()
            delta = condition.float() - expected
            relative_l2 = float(delta.norm()/expected.norm().clamp_min(1e-12))
            result['condition_numerics'] = dict(max_abs=float(delta.abs().max()),
                mean_abs=float(delta.abs().mean()), relative_l2=relative_l2,
                acceptance_relative_l2=0.01, passed=relative_l2 <= 0.01)
        result['stage1_bundle'] = str(a.stage1_bundle)
        result['stage1_scope'] = 'real 30-layer checkpoint, real RGB/TCP/masks, fixed synthetic 512-token text; no optimizer, T5, DataLoader or distributed communication'
    result['gpu_after'] = gpu_snapshot()
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    emit('complete', output=str(a.output), result=result)


if __name__ == '__main__':
    main()
