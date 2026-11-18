#!/usr/bin/env python3
"""Bounded DSW validation: version seven endpoint labels, then infer unchanged weights.

No optimizer, trainer.main(), DLC API, or source-data writes are used. Each
phase refuses to replace its completed receipt. Original prepared rows and
diagnostics are read-only baselines, not relabeled in place.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
INDICES = (8, 13, 14, 20, 34, 57, 60)
VERS = 'openwam.goal-observation-fix.dsw-seven.v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def runtime():
    sys.path.insert(0, str(ROOT))
    path = ROOT / 'training/stage1_longtrain_16gpu/entrypoint.py'
    spec = importlib.util.spec_from_file_location('goal_fix_entry', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    trainer = module.load_trainer()
    trainer.d.load_reviewed().base.bind_code(ROOT)
    import validation16
    return trainer, validation16


def current_git():
    return dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip())


def preserve_inputs(old, new):
    names = ('current_latents', 'memory_latents', 'memory_valid', 'memory_view_valid',
             'memory_ages', 'memory_frame_ids', 'observed_view_valid', 'raw_current_rgb',
             'observed_latent', 'history_latents', 'history_valid', 'history_ages', 'history_episode_id')
    checked = []
    for key in names:
        if key in old:
            if not torch.equal(old[key], new[key]): raise ValueError('input changed: ' + key)
            checked.append(key)
    for key in ('instruction', 'granularity'):
        if old[key] != new[key]: raise ValueError('input changed: ' + key)
    return checked


def publish_target_encoding(prepared, manifest, entry, row, writer):
    """A failed manifest publication must leave its previous row readable."""
    relative = entry['file'].replace('.pt', '.encoded.pt')
    writer.atomic_torch(prepared / relative, row)
    manifest['files'][relative] = sha(prepared / relative)
    entry['file'] = relative
    writer.atomic_json(prepared / 'manifest.json', manifest)


def prepare(args):
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.goal_validity import GOAL_LABEL_POLICY
    trainer, validation = runtime()
    args.output.mkdir(parents=True, exist_ok=False)
    oldm = {arch: json.loads((path / 'manifest.json').read_text())
            for arch, path in [('G', args.prepared), ('Cstar', args.prepared_cstar)]}
    source = oldm['G']['sources']['val']
    assert sha(source['path']) == source['sha256'], 'validation source selection changed'
    reader = CuratedGoalDataset(source['path'], args.output / 'reader',
        observation_mode='multiview', materialize_multiview_target=True, dtype=torch.float32)
    reader._evaluating = True
    cases = []; versions = {}
    for arch in ('G', 'Cstar'):
        versions[arch] = dict(format=VERS, dependencies=oldm[arch]['dependencies'],
            config=oldm[arch]['config'], goal_label_policy=GOAL_LABEL_POLICY,
            records={'val': []}, files={}, external_files={},
            parent=dict(path=str(args.prepared if arch == 'G' else args.prepared_cstar),
                        manifest_sha256=sha((args.prepared if arch == 'G' else args.prepared_cstar) / 'manifest.json')))
    for index in INDICES:
        old = validation.checked_row(args.prepared, oldm['G'], oldm['G']['records']['val'][index])
        cr = validation.checked_row(args.prepared_cstar, oldm['Cstar'], oldm['Cstar']['records']['val'][index])
        record = old['record']; episode = record['source_episode']; goal = record['goal_source_frame']
        repo = next(r for r in reader.repos if Path(r['path']).resolve() == Path(record['source_repo']).resolve())
        metadata = reader._episode(repo, episode)
        segment = next(s for s in metadata['tasks' if record['granularity'] == 'task' else 'subtasks']
                       if s['start'] == record['instruction_start'] and s['end_exclusive'] == record['instruction_end_exclusive'])
        newrecord, batch = reader._materialize(repo, episode, metadata, segment,
            record['anchor_source_frame'], goal, record['granularity'])
        assert newrecord['id'] == record['id'] and goal == segment['end_exclusive'] - 1
        assert newrecord['instruction'] == record['instruction']
        assert newrecord['provenance']['goal_video']['source_rgb_sha256'] == record['provenance']['goal_video']['source_rgb_sha256']
        assert newrecord['provenance']['observed_video']['source_rgb_sha256'] == record['provenance']['observed_video']['source_rgb_sha256']
        assert not newrecord['success_verified']
        # The original prepared pipeline used float32 reader pixels. Preserve it.
        rgb = ((batch['target_image'][0].float().clamp(-1, 1) + 1) * 127.5).round().byte()
        current = ((batch['observed'][0].float().clamp(-1, 1) + 1) * 127.5).round().byte()
        assert torch.equal(current, old['raw_current_rgb']), 'source current/preprocessing differs'
        target = batch['targets']
        new = deepcopy(old); child = deepcopy(cr)
        for row in (new, child):
            row['record'] = deepcopy(newrecord)
            row.update(target_uv=target.uv[0].cpu(), target_point_valid=target.point_valid[0].cpu(),
                       target_in_frame=target.in_frame[0].cpu())
        new.update(target_image_valid=batch['target_image_valid'][0],
                   target_view_valid=batch['target_view_valid'][0], raw_target_rgb=rgb,
                   goal_label_identity=newrecord['goal_label_identity'], target_encoding_pending=True)
        child.update(target_image_valid=target.image_valid[0], raw_target_rgb=rgb[:, :, :256],
                     target_view_valid=batch['target_view_valid'][0],
                     goal_label_identity=newrecord['goal_label_identity'], target_encoding_pending=True)
        # Discard only affected target latents; inference phase encodes each model's own canvas.
        new['target_latents'] = torch.zeros_like(old['target_latents'])
        child['target_latent'] = torch.zeros_like(cr['target_latent'])
        assert bool(new['target_image_valid']) == (episode == 193)
        assert bool(child['target_image_valid']) == (episode == 193)
        entry = oldm['G']['records']['val'][index]
        for arch, parent, oldrow, row in [('G', args.prepared, old, new), ('Cstar', args.prepared_cstar, cr, child)]:
            context = Path(oldrow['context_file'])
            if not context.is_absolute(): context = parent / context
            row['context_file'] = str(context.resolve())
            versions[arch]['external_files'][row['context_file']] = sha(context)
            preserved = preserve_inputs(oldrow, row)
            path = args.output / 'prepared' / arch / entry['file']; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(row, path)
            versions[arch]['records']['val'].append(dict(entry, source_index=index))
            versions[arch]['files'][entry['file']] = sha(path)
            if arch == 'G': input_fields = preserved
        import pyarrow.parquet as pq
        masks = pq.read_table(metadata['data'], filters=[('frame_index', '=', goal)], columns=[
            'canonical.frame_valid', 'training.valid', 'source.frame_validity',
            'raw.robot0.sensor.camera2.video_alignment_valid', 'observation.states.eef.mask', 'action.mask']).to_pylist()[0]
        case = dict(source_index=index, id=entry['id'], instruction=record['instruction'], episode=episode,
            anchor=record['anchor_source_frame'], goal=goal, goal_sha256=newrecord['provenance']['goal_video']['source_rgb_sha256'],
            old_image_valid=bool(old['target_image_valid']), new_image_valid=bool(new['target_image_valid']),
            old_point_valid=old['target_point_valid'].tolist(), new_point_valid=new['target_point_valid'].tolist(),
            new_in_frame=new['target_in_frame'].tolist(), new_view_valid=new['target_view_valid'].tolist(),
            source_masks=masks, old_source_valid=bool(metadata['source_valid'][goal]),
            goal_label_identity=newrecord['goal_label_identity'], preserved_input_tensors=input_fields,
            source_policy=metadata['goal_validity_policy'], granularity=record['granularity'])
        cases.append(case)
        print(json.dumps(dict(phase='labels', **{k: case[k] for k in ('source_index', 'episode', 'goal', 'new_image_valid', 'new_point_valid')})), flush=True)
    for arch in versions: dump(args.output / 'prepared' / arch / 'manifest.json', versions[arch])
    dump(args.output / 'labels.json', dict(format=VERS, cases=cases, indices=INDICES, policy=GOAL_LABEL_POLICY,
        git=current_git(), stage2_changed=False, target_frames_changed=False, checkpoints_changed=False,
        scope='seven selected task examples; fitting diagnostics, not held-out evaluation'))


def infer(args, *, baseline=False):
    from safetensors import safe_open
    from safetensors.torch import load_file
    trainer, validation = runtime(); shared = trainer.d; base = shared.load_reviewed().base
    helper = validation._generation_helpers()
    from openwam.model.goal.codec import GoalCodec, GoalCodecSpec
    from openwam.model.goal.dependencies import module_identity
    from openwam.train.goal_image_metrics import goal_image_metrics, aggregate_goal_image_metrics
    receipt = args.output / ('baseline.json' if baseline else 'inference.json')
    if receipt.exists(): raise FileExistsError('completed inference is immutable: ' + str(receipt))
    evidence = dict(format=VERS, checkpoints={}, labels_sha256=sha(args.output / 'labels.json'),
                    weights_updated=False, optimizer_loaded=False, device=args.device, seed=42, steps=20,
                    source=trainer.source_identity(ROOT), runner=trainer.runner_identity(),
                    generation_policy=('archived-gt-masked-generation' if baseline else 'current-view-kf1-fixed0-no-gt-v1'),
                    models={}, script_sha256=sha(Path(__file__)))
    if baseline:
        path = ROOT / 'training/stage1_longtrain_16gpu/reviewed/frozen_v1/rfc_compare.py'
        spec = importlib.util.spec_from_file_location('goal_fix_archived_generation', path)
        archived = importlib.util.module_from_spec(spec); spec.loader.exec_module(archived)
    for arch, step in [('G', 5515), ('Cstar', 3321)]:
        pointerfile = args.runs_root / arch / 'checkpoint.json'
        pointer = json.loads(pointerfile.read_text()); assert pointer['update'] == step
        manifestfile = Path(pointer['generation']) / 'manifest.json'
        assert sha(manifestfile) == pointer['manifest_sha256']
        checkpoint = json.loads(manifestfile.read_text()); weights = manifestfile.parent / checkpoint['weights']['file']
        print(json.dumps(dict(phase='hashing_weights', arch=arch)), flush=True)
        assert sha(weights) == checkpoint['weights']['sha256']
        evidence['checkpoints'][arch] = dict(update=step, weights=str(weights), weights_sha256=checkpoint['weights']['sha256'],
            manifest_sha256=sha(manifestfile), checkpoint_source=checkpoint['contract']['source'],
            old_source_differs=checkpoint['contract']['source'] != evidence['source'])
        config = trainer.read_config(ROOT / 'training/stage1_longtrain_16gpu/runtime' / (arch + '.json'))
        config['code_dir'] = str(ROOT)
        if arch == 'G': config['model_config'] = str(ROOT / 'training/stage1_longtrain_16gpu/runtime/G_model.json')
        print(json.dumps(dict(phase='loading_model', arch=arch, update=step)), flush=True)
        model, groups, cfg = trainer.build_native(config, arch, torch.device('cpu')); del groups
        key = 'goal_memory_bundle' if arch == 'G' else 'visual_history'
        with safe_open(str(weights), framework='pt', device='cpu') as source:
            metadata = json.loads(source.metadata()[key])
        assert metadata['training']['update'] == step and metadata['dit_config'] == model.dit_config
        if arch == 'G': assert metadata['goal_memory']['goal_horizon_mode'] == model.config.goal_horizon_mode == 'kf1_endpoint'
        model.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
        model.to(device=args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        vae = base.construct_component(cfg, 'vae', device=args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        assert module_identity(vae) == metadata['dependencies']['vae']
        codec = GoalCodec(vae, GoalCodecSpec(name='wan22_multiview_t1', image_shape=(3, 1, 384, 320),
                            latent_shape=(48, 1, 24, 20))) if arch == 'G' else GoalCodec(vae)
        prepared = ((args.prepared if arch == 'G' else args.prepared_cstar) if baseline
                    else args.output / 'prepared' / arch)
        pm = json.loads((prepared / 'manifest.json').read_text())
        entries = ([dict(pm['records']['val'][i], source_index=i) for i in INDICES] if baseline
                   else pm['records']['val'])
        assert pm['dependencies']['vae'] == module_identity(vae)
        directory = args.output / ('before_dsw' if baseline else 'after') / arch
        directory.mkdir(parents=True, exist_ok=True)
        records = []; start = time.monotonic()
        with torch.inference_mode():
            for entry in entries:
                index = entry['source_index']; row = validation.checked_row(prepared, pm, entry)
                if not baseline and row.get('target_encoding_pending'):
                    field = 'target_latents' if arch == 'G' else 'target_latent'
                    image = row['raw_target_rgb'].float() / 127.5 - 1
                    # Encode the G composite and C* head independently with the original VAE.
                    row[field] = (codec.encode(image[None].to(device=args.device, dtype=torch.bfloat16))[0].cpu()
                                  if bool(row['target_image_valid']) else torch.zeros_like(row[field]))
                    row['target_encoding_pending'] = False
                    row['target_head_encoding' if arch == 'Cstar' else 'target_encoding'] = dict(
                        policy=pm['goal_label_policy'], codec=codec.spec.name,
                        independent_T1=True, latent_shape=list(row[field].shape), label_identity=row['goal_label_identity'])
                    # Publish a new blob before atomically moving its manifest reference.
                    # An interruption leaves the old pending row readable, never a stale checksum.
                    publish_target_encoding(prepared, pm, entry, row, shared)
                if arch == 'G':
                    batch, _ = base.collate_prepared([row], prepared, device=args.device, dtype=torch.bfloat16)
                    assert (batch['memory_ages'][batch['memory_valid']] > 0).all()
                    generator = archived.generate_latents if baseline else base.generate_latents
                    prediction = generator(model, batch, arm='default', seed=42, index=index, steps=20)
                    latent, uv = prediction['latents'], prediction['uv'][0].float().cpu()
                else:
                    batch = shared.load_cstar().collate([row], prepared, device=args.device, dtype=torch.bfloat16)
                    assert (batch['history_ages'][batch['history_valid']] > 0).all()
                    prediction = model.generate(**{k: v for k, v in batch.items() if k != 'targets'},
                                                num_steps=20, seed=42 + index + 1000000)
                    latent, uv = prediction.latent, prediction.uv[0].float().cpu()
                head = ((codec.decode(latent)[0].float().clamp(-1, 1) + 1) / 2)[:, :, :256].cpu()
                assert torch.isfinite(latent).all()
                assert bool(latent.abs().sum() > 0) == (not baseline or arch == 'Cstar')
                score_row = row
                if baseline and arch == 'Cstar':
                    gm = json.loads((args.prepared / 'manifest.json').read_text())
                    score_row = validation.checked_row(args.prepared, gm, gm['records']['val'][index])
                valid, point_valid = helper.valid_targets(score_row)
                measured = helper.measure(head, uv, score_row)
                region = goal_image_metrics(head, score_row['raw_current_rgb'][:, :, :256], score_row['raw_target_rgb'][:, :, :256],
                    image_valid=valid, target_uv=score_row['target_uv'], point_valid=point_valid)
                result = dict(index=index, id=entry['id'], grain=row['granularity'], family=entry['family'],
                    arch=arch, update=step, generated_uv=uv.tolist(), target_uv=row['target_uv'].tolist(),
                    region_errors=region, inference_seconds=time.monotonic() - start,
                    generated_latent_abs_sum=float(latent.abs().sum()), **measured)
                torch.save(dict(head=head, uv=uv, latent=latent.cpu()), directory / f'{index:04d}.pt')
                validation.save_image(directory / f'{index:04d}.png', head, uv, score_row, arch=arch, index=index, update=step)
                dump(directory / f'{index:04d}.json', result); records.append(result)
                print(json.dumps(dict(phase='inferred', arch=arch, index=index, metrics=measured['metrics'])), flush=True)
                del row, batch, latent, head, prediction, uv
        if not baseline: shared.atomic_json(prepared / 'manifest.json', pm)
        result = dict(arch=arch, step=step, seconds=time.monotonic() - start, cases=records,
            region_errors=aggregate_goal_image_metrics([r['region_errors'] for r in records]), **helper.aggregate(records))
        dump(directory / 'manifest.json', result); evidence['models'][arch] = result
        evidence['checkpoints'][arch]['pointer_unchanged'] = json.loads(pointerfile.read_text()) == pointer
        assert evidence['checkpoints'][arch]['pointer_unchanged']
        del model, codec, vae
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps(dict(phase='complete', arch=arch, metrics=result['metrics'], seconds=result['seconds'])), flush=True)
    evidence['complete'] = True; dump(receipt, evidence)


def baseline(args):
    return infer(args, baseline=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'infer', 'baseline'])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--prepared-cstar', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT))
    globals()[args.phase](args)


if __name__ == '__main__': main()
