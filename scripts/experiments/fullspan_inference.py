"""Resumable fresh inference at every full-span anchor; never loads future GT into models.

Stage1 image/UV denoising and Stage2 visual C* conditioning are separate outputs.
The latter follows each saved checkpoint's native single-pass teacher policy.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import shutil
import sys
import time

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from scripts.experiments.infer_stage1_samecase import (
    dump, infer, make_batch, prepare_contexts, sample_identity, sha, tensor_sha,
)


def file_stamp(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def cached_sha(path, cache):
    """Reuse a locally verified hash only while the complete file stat is unchanged."""
    path = Path(path).resolve()
    stamp = file_stamp(path)
    entry = cache.get(str(path), {})
    if entry.get('stat') == stamp:
        return entry['sha256']
    digest = sha(path)
    if file_stamp(path) != stamp:
        raise ValueError(f'File changed while hashing: {path}')
    cache[str(path)] = dict(stat=stamp, sha256=digest)
    return digest


def wait_for_sample(record, manifest_path, *, wait):
    """The producer publishes each PT atomically before marking its manifest row ready."""
    announced = 0.
    while True:
        manifest = json.loads(Path(manifest_path).read_text())
        current = next(row for row in manifest['anchors'] if row['id'] == record['id'])
        if current.get('status') == 'ready' and Path(record['sample_path']).is_file():
            if current['sample_path'] != record['sample_path'] or current['anchor'] != record['anchor']:
                raise ValueError('Manifest anchor identity changed while waiting')
            return
        if not wait or manifest.get('status') in ('failed', 'complete'):
            raise ValueError(f"Sample not ready: {record['id']}; producer status {manifest.get('status')}")
        if time.monotonic() - announced > 60:
            emit('waiting_for_sample', id=record['id'], anchor=record['anchor'])
            announced = time.monotonic()
        time.sleep(5)


def latest_checkpoint(folder):
    candidates = []
    for path in Path(folder).glob('checkpoint_step_*.safetensors'):
        match = re.fullmatch(r'checkpoint_step_(\d+)\.safetensors', path.name)
        if match:
            candidates.append((int(match[1]), path))
    if not candidates:
        raise FileNotFoundError(f'No Stage2 checkpoint in {folder}')
    return max(candidates)[1]


def conditioning_identity(sample):
    identity = sample_identity(sample)
    identity['conditioning_sha256'].update({k: tensor_sha(sample[k]) for k in ('proprio', 'proprio_mask')})
    identity['conditioning_sha256']['active_action_mask'] = tensor_sha(torch.as_tensor(sample['_eval']['active_action_mask']))
    identity['first_frame_sha256'] = [hashlib.sha256(image.tobytes()).hexdigest() for image in sample['first_frame_image']]
    identity['goal_seed'] = int(sample['goal_seed'])
    identity['goal_head_valid'] = bool(sample['goal_head_valid'])
    return identity


def resume_complete(folder, identity):
    folder = Path(folder)
    if not folder.exists():
        return False
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata.get('identity') != identity:
        raise ValueError(f'Refusing mismatched resume identity: {folder}')
    for name, expected in metadata['artifacts_sha256'].items():
        if not (folder / name).is_file() or sha(folder / name) != expected:
            raise ValueError(f'Resume artifact changed: {folder / name}')
    return True


def publish_result(folder, identity, prediction, images, **metadata):
    """Publish a directory only after every payload and SHA is complete."""
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    work = folder.with_name('.' + folder.name + '.partial')
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    torch.save(prediction, work / 'prediction.pt')
    for name, image in images.items():
        image.save(work / name)
    hashes = {p.name: sha(p) for p in work.iterdir() if p.is_file()}
    dump(work / 'metadata.json', dict(schema='openwam.fullspan_prediction.v1', identity=identity,
         artifacts_sha256=hashes, **metadata))
    if folder.exists():
        raise FileExistsError(folder)
    work.rename(folder)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def load_anchor(record):
    path = Path(record['sample_path'])
    sample = torch.load(path, map_location='cpu', weights_only=False)
    if int(sample['trace']['anchor']) != int(record['anchor']):
        raise ValueError('Manifest anchor disagrees with sample')
    return sample


def load_verified_contexts(output, prompts, text_fingerprint):
    path = Path(output) / 'text_contexts.pt'
    source_path = Path(output) / 'text_contexts_source.json'
    if not path.is_file() or not source_path.is_file():
        return None
    source = json.loads(source_path.read_text())
    if source['text_dependency_fingerprint'] != text_fingerprint:
        raise ValueError('Cached T5 dependency fingerprint mismatch')
    if sha(path) != source['output_sha256']:
        raise ValueError('Cached text context SHA mismatch')
    contexts = torch.load(path, map_location='cpu', weights_only=True)
    if any(prompt not in contexts for prompt in prompts):
        return None
    return {prompt: contexts[prompt] for prompt in prompts}


def stage1_config_sha(run):
    """Bind native reload inputs and cached text bytes; preserve legacy identity."""
    folder=Path(run)/'stage1'
    if not (folder/'resolved_config.yaml').is_file():
        return sha(folder/'protocol.json')
    artifacts=json.loads((folder/'artifacts.json').read_text())
    names=('resolved_config.yaml','dependencies.json','text_contexts.pt')
    for name in names:
        expected=artifacts.get('files',{}).get(name)
        if expected is None or sha(folder/name)!=expected:
            raise ValueError(f'Stage1 reload artifact identity mismatch: {name}')
    return hashlib.sha256(json.dumps(artifacts,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def stage1_dependencies(run):
    """Read unified training artifacts, retaining compatibility with older runs."""
    import yaml
    folder=Path(run)/'stage1'
    if (folder/'resolved_config.yaml').is_file():
        cfg=yaml.safe_load((folder/'resolved_config.yaml').read_text())
        deps=json.loads((folder/'dependencies.json').read_text())
        return folder,cfg,deps,Path(cfg['baseline_checkpoint'])
    protocol=json.loads((folder/'protocol.json').read_text())
    smoke=Path(protocol['smoke_data'])
    return (smoke,yaml.safe_load((smoke/'config.yaml').read_text()),
            json.loads((smoke/'dependencies.json').read_text()),Path(protocol['official_dependencies']))


def load_stage1(run, records, output):
    from openwam.train.goal_trainer import _construct, load_component_weights
    from openwam.train.goal_visual_history import load_visual_history_bundle
    smoke,cfg,deps,official=stage1_dependencies(run)
    # Context preparation needs only prompt strings, not all decoded samples.
    prompts = [{'prompt': record['prompt']} for record in records]
    contexts = load_verified_contexts(output, [row['prompt'] for row in prompts], deps['text']['fingerprint'])
    if contexts is None:
        contexts = prepare_contexts(prompts, smoke, cfg, deps, official, output)
    vae = _construct(cfg['components']['vae'])
    load_component_weights(vae, official, 'video_backbone.vae.')
    vae.to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    expert = load_visual_history_bundle(run / 'stage1/visual_history.safetensors', vae=vae,
        text_dim=deps['text_dim'], precomputed_text_fingerprint=deps['text']['fingerprint'],
        device='cuda', dtype=torch.bfloat16)
    expert.eval().requires_grad_(False)
    return expert, contexts


def load_stage2(checkpoint):
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    _, architecture, _ = build_architecture_from_ckpt_dir(str(checkpoint), weights_required=True)
    architecture.set_dtype_device(torch.bfloat16, torch.device('cuda'))
    architecture.eval().requires_grad_(False)
    return architecture


@torch.inference_mode()
def infer_stage2(architecture, sample, seed, steps, mode):
    from openwam.deploy.denoise_schedule import make_schedule
    from openwam.train.stage2_eval import generation_conditions, _conditions_sha
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = generation_conditions(sample, seed=seed, num_steps=steps,
        observation_mode=architecture.goal_expert.observation_mode, observation_dtype=architecture.dtype)
    kwargs['decode_video'] = mode == 'joint'
    shift = float(getattr(architecture.action_backbone, 'shift_action', 5.))
    schedule = make_schedule('sync', architecture.video_scheduler, architecture.action_scheduler,
        num_steps=steps, shift=shift, shift_video=getattr(architecture.video_backbone, 'shift_video', None))
    result = architecture.generate(schedule, sample['prompt'], shift=shift, **kwargs)
    action = torch.as_tensor(np.asarray(result['normalized_actions'])).cpu()
    if tuple(action.shape) != (32, 112) or not torch.isfinite(action).all():
        raise ValueError('Invalid normalized action prediction')
    video = result.get('video')
    if (video is None) != (mode == 'none'):
        raise ValueError('Generated video availability disagrees with checkpoint mode')
    diagnostics = architecture.goal_diagnostics
    payload = dict(normalized_actions=action,
        teacher_visual_memory=diagnostics['predicted_latent'].detach().cpu(),
        teacher_image_token_valid=diagnostics['image_token_valid'].detach().cpu(),
        teacher_predicted_uv=diagnostics['predicted_uv'].detach().cpu())
    images = {} if video is None else {f'video_{i:02d}.png': frame for i, frame in enumerate(video)}
    return payload, images, dict(generation_condition_sha256=_conditions_sha(sample, kwargs),
        model_input_fields=sorted(kwargs), teacher_policy='native visual C*; not standalone decoded Stage1 image',
        teacher_goal_config=dict(architecture.goal_config))


def ordered_work(records, stages, sequences, *, anchor_major):
    if anchor_major:
        records = sorted(records, key=lambda row: (len(sequences[row['sequence_id']]['anchors']),
                                                   row['sequence_id'], row['anchor']))
        return ((record, stage) for record in records for stage in stages)
    return ((record, stage) for stage in stages for record in records)


def prediction_identity(stage, record, sample_sha, checkpoints, configs, seed, steps):
    return dict(stage=stage, sample_sha256=sample_sha, checkpoint_sha256=checkpoints[stage],
        stage1_sha256=checkpoints['stage1'], configuration_sha256=configs[stage],
        seed=seed + int(record['id']), steps=steps,
        sequence_id=record['sequence_id'], anchor=int(record['anchor']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed RGB Stage1/Stage2 training run')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stages', default='stage1,joint,none')
    parser.add_argument('--anchor-major', action='store_true', help='Retain all models and finish short sequences first')
    parser.add_argument('--first-anchor-gate', type=Path, help='Wait after first complete anchor until this release file exists')
    parser.add_argument('--wait-ready', action='store_true', help='Stream atomic samples while producer is running')
    parser.add_argument('--hash-cache', type=Path, help='Local checkpoint hash cache with strict stat identity')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=100042)
    parser.add_argument('--limit', type=int, default=0, help='Warmup subset; 0 means all selected anchors')
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    stages = args.stages.split(',')
    if not stages or any(stage not in ('stage1', 'joint', 'none') for stage in stages):
        raise ValueError('Invalid stages')
    if args.steps < 1 or args.limit < 0 or args.offset < 0:
        raise ValueError('Positive steps and nonnegative limit/offset required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    sequences = {sequence['sequence_id']: sequence for sequence in manifest['sequences']}
    records = [dict(record, prompt=sequences[record['sequence_id']]['prompt']) for record in manifest['anchors']]
    if args.anchor_major:
        records.sort(key=lambda row: (len(sequences[row['sequence_id']]['anchors']), row['sequence_id'], row['anchor']))
    records = records[args.offset:args.offset + args.limit if args.limit else None]
    if not records:
        raise ValueError('No selected anchors')
    if not args.wait_ready and any(record.get('status') != 'ready' or not Path(record['sample_path']).is_file() for record in records):
        raise ValueError('All selected samples must be ready before inference')
    frozen = args.run / 'stage1/visual_history.safetensors'
    hash_cache_path = args.hash_cache or args.output / 'checkpoint_hash_cache.json'
    hash_cache = json.loads(hash_cache_path.read_text()) if hash_cache_path.exists() else {}
    frozen_sha = cached_sha(frozen, hash_cache)
    checkpoints = {'stage1': frozen}
    for mode in ('joint', 'none'):
        checkpoints[mode] = latest_checkpoint(args.run / 'stage2' / mode / 'checkpoints')
        if cached_sha(checkpoints[mode].parent / 'goal/expert.safetensors', hash_cache) != frozen_sha:
            raise ValueError(f'{mode} bundled teacher differs from frozen Stage1')
    checkpoint_hashes = {'stage1': frozen_sha}
    for mode in ('joint', 'none'):
        emit('hashing_checkpoint', stage=mode, path=str(checkpoints[mode]))
        checkpoint_hashes[mode] = cached_sha(checkpoints[mode], hash_cache)
    dump(hash_cache_path, hash_cache)
    config_hashes = {'stage1': stage1_config_sha(args.run)}
    config_hashes.update({mode: sha(checkpoints[mode].parent / 'config.yaml') for mode in ('joint', 'none')})
    progress = dict(schema='openwam.fullspan_inference.v1', status='running', pid=os.getpid(),
        manifest=str(args.manifest), selected_count=len(records), total_count=len(manifest['anchors']),
        checkpoint_sha256=checkpoint_hashes, stages=stages, seed=args.seed, steps=args.steps, completed={}, timings={})
    started = time.monotonic()
    status_file = args.output / 'inference_status.json'
    dump(status_file, progress)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    models = {}
    contexts = None
    durations = {stage: [] for stage in stages}
    progress['traversal'] = 'anchor_major_shortest_sequence_first' if args.anchor_major else 'stage_major'
    progress['completed'] = {stage: sum((args.output / 'inference' / stage / f"{int(r['id']):06d}").is_dir()
                                       for r in records) for stage in stages}
    progress['existing_results_unverified'] = dict(progress['completed'])

    def get_model(stage):
        nonlocal contexts
        if stage in models:
            return models[stage]
        emit('loading', stage=stage, remaining=len(records) - progress['completed'][stage])
        loaded = time.monotonic()
        if stage == 'stage1':
            model, contexts = load_stage1(args.run, records, args.output)
        else:
            model = load_stage2(checkpoints[stage])
            if model.future_video_mode != stage:
                raise ValueError('Checkpoint future-video mode mismatch')
        models[stage] = model
        emit('loaded', stage=stage, seconds=time.monotonic() - loaded,
             cuda_allocated_gib=torch.cuda.memory_allocated() / 1024**3)
        if args.anchor_major and torch.cuda.memory_allocated() > 120 * 1024**3:
            raise RuntimeError('Resident GPU memory exceeds 120 GiB safety budget')
        return model

    dump(status_file, progress)
    if args.anchor_major:
        # First-anchor validation measures all three resident models, even when an
        # earlier Stage1-only run already published that anchor's Stage1 artifact.
        for stage in stages:
            get_model(stage)
    cached_record_id = None
    sample = None
    input_identity = None
    sample_sha = None
    previous_stage = None
    first_anchor = records[0]['id']
    gated = False
    for record, stage in ordered_work(records, stages, sequences, anchor_major=args.anchor_major):
        if not args.anchor_major and previous_stage is not None and stage != previous_stage:
            model = None
            models.clear()
            contexts = None
            gc.collect()
            torch.cuda.empty_cache()
        previous_stage = stage
        if record['id'] != cached_record_id:
            wait_for_sample(record, args.manifest, wait=args.wait_ready)
            cached_record_id = record['id']
            sample_sha = sha(record['sample_path'])
            sample = None
            input_identity = None
        identity = prediction_identity(stage, record, sample_sha, checkpoint_hashes,
                                       config_hashes, args.seed, args.steps)
        folder = args.output / 'inference' / stage / f"{int(record['id']):06d}"
        already_complete = resume_complete(folder, identity)
        if already_complete:
            progress['existing_results_unverified'][stage] -= 1
        else:
            model = get_model(stage)
            tick = time.monotonic()
            if sample is None:
                sample = load_anchor(record)
                input_identity = conditioning_identity(sample)
            if stage == 'stage1':
                batch = make_batch(sample, contexts, device='cuda', dtype=torch.bfloat16)
                prediction, uv = infer(model, batch, seed=identity['seed'], steps=args.steps)
                if not np.isfinite(np.asarray(uv)).all():
                    raise ValueError('Nonfinite Stage1 UV prediction')
                payload = dict(predicted_uv=torch.tensor(uv))
                images = {'prediction.png': prediction}
                details = dict(predicted_uv=uv, model_input_fields=sorted(batch),
                    context_sha256=tensor_sha(contexts[sample['prompt']][0]))
                del batch
            else:
                payload, images, details = infer_stage2(model, sample, identity['seed'], args.steps, stage)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            if sha(record['sample_path']) != identity['sample_sha256']:
                raise ValueError('Input sample changed during inference')
            publish_result(folder, identity, payload, images, input_identity=input_identity,
                sample_path=record['sample_path'], family=sample['_eval']['family'],
                elapsed_seconds=elapsed, **details)
            durations[stage].append(elapsed)
            progress['completed'][stage] += 1
            progress['timings'][stage] = dict(count=len(durations[stage]), mean_seconds=float(np.mean(durations[stage])),
                last_seconds=elapsed, estimated_all_anchors_seconds=float(np.mean(durations[stage])) * len(manifest['anchors']))
            emit('anchor_complete', stage=stage, id=record['id'], anchor=record['anchor'],
                elapsed_seconds=elapsed, completed=progress['completed'][stage], selected=len(records))
            del payload, images
        progress['elapsed_seconds'] = time.monotonic() - started
        progress['cuda_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        dump(status_file, progress)
        at_boundary = not args.anchor_major or stage == stages[-1]
        if at_boundary and args.anchor_major:
            emit('all_stages_anchor_complete', id=record['id'], sequence_id=record['sequence_id'],
                 cuda_peak_allocated_gib=progress['cuda_peak_allocated_gib'])
        if at_boundary and stop_requested:
            progress['status'] = 'stopped_at_anchor_boundary'
            dump(status_file, progress)
            return
        if at_boundary and args.anchor_major and record['id'] == first_anchor and args.first_anchor_gate and not gated:
            progress.update(status='waiting_first_anchor_validation', first_complete_anchor=first_anchor)
            dump(status_file, progress)
            emit('waiting_first_anchor_validation', anchor_id=first_anchor, release_file=str(args.first_anchor_gate))
            while not args.first_anchor_gate.exists():
                if stop_requested:
                    progress['status'] = 'stopped_at_anchor_boundary'
                    dump(status_file, progress)
                    return
                time.sleep(2)
            gated = True
            progress['status'] = 'running'
            dump(status_file, progress)
    model = None
    models.clear()
    gc.collect()
    torch.cuda.empty_cache()
    progress.update(status='complete', elapsed_seconds=time.monotonic() - started,
        checkpoint_unchanged=all(cached_sha(path, hash_cache) == checkpoint_hashes[stage] for stage, path in checkpoints.items()))
    if not progress['checkpoint_unchanged']:
        raise ValueError('Frozen checkpoint changed')
    dump(status_file, progress)
    emit('complete', selected=len(records), stages=stages, elapsed_seconds=progress['elapsed_seconds'])


if __name__ == '__main__':
    main()
