#!/usr/bin/env python3
"""Matched real Wan Stage1 training on an audited singleton subtask, all artifacts on CPFS."""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time,statistics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,yaml,numpy as np
from safetensors import safe_open
from openwam.train.goal_trainer import _construct,load_component_weights,_seed
from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.query_expert import QueryGoalExpert
from openwam.model.goal.history import history_kwargs
from openwam.model.goal.dependencies import dependency_identity,resolve_tokenizer_config
from openwam.model.video_backbone.wan.encode import encode_text
from openwam.train.goal_curated import prepare_batch,goal_parameter_groups
from openwam.train.goal_overfit import FP32MasterAdamW,_display_image,_write_json
from openwam.train.goal_query import save_query_bundle
from openwam.train.goal_patch import evaluate_patch,_prediction_overlay
from openwam.train.conditioning_report import render,NAMES


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def fingerprint(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(bytes.fromhex(digest(v)))
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--steps',type=int,default=500);parser.add_argument('--mode',choices=list(NAMES))
    args=parser.parse_args();p=Path(args.output).resolve();cfg=yaml.safe_load(Path(args.config).read_text())
    if not str(p).startswith('/mnt/data/cpfs/'):raise ValueError('experiment artifacts must be on CPFS')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dtype=torch.bfloat16;device='cuda';record=json.loads((p/'record.json').read_text())
    if record['granularity']!='subtask':raise ValueError('subtask sample required')
    def event(kind,**kw):
        row=dict(event=kind,time=time.time(),**kw)
        with (p/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    protocol=dict(base_commit='02edd62',mode_order=list(NAMES),steps=args.steps,evaluation_steps=[0,50,100,250,args.steps],
        seed=42,query_seed=777,noise_seed='10000+step; identical per arm',eval_seed=100042,sampling_steps=20,
        baseline_checkpoint=cfg['baseline_checkpoint'],query_config=cfg['model']['query_goal'],
        learning_rate=1e-5,adapter_learning_rate=1e-4,history_learning_rate=1e-4,
        weight_decay=.01,gradient_clip_norm=1.,optimizer='BF16 forward + FP32 master AdamW',
        input_cache='frozen VAE encodings memoized by exact tensor hash; never trainable query/history tokens',
        history_dropout=.2,observation_augmentation=False,checkpoint_scope='new optimizer; pretrained Wan; fresh shared query encoder',
        limitation='one sample/seed; does not establish grounding/generalization; C is Wan prefix ablation, not FLUX')
    _write_json(p/'protocol.json',protocol)
    event('loading_vae')
    vae=_construct(cfg['components']['vae']);load_component_weights(vae,cfg['baseline_checkpoint'],'video_backbone.vae.')
    vae.to(device=device,dtype=dtype).eval().requires_grad_(False)
    text_path=p/'text_contexts.pt'
    if not text_path.exists():
        event('loading_text_encoder')
        text=_construct(cfg['components']['text_encoder']);load_component_weights(text,cfg['baseline_checkpoint'],'video_backbone.text_encoder.')
        text.to(device=device,dtype=dtype).eval().requires_grad_(False)
        tc=resolve_tokenizer_config(cfg);tokenizer=_construct(dict(model_class=tc['model_class'],extra_kwargs=tc['kwargs']))
        with torch.no_grad():
            ctx,lens=encode_text([record['instruction']],tokenizer=tokenizer,text_encoder=text,device=device)
            alt,altlens=encode_text(['Pick up the stamp from the table.'],tokenizer=tokenizer,text_encoder=text,device=device)
        deps=dependency_identity(vae,text_dim=4096,text_encoder=text,tokenizer=tokenizer)
        torch.save(dict(context=ctx.cpu(),context_lens=lens.cpu(),alt=alt.cpu(),altlens=altlens.cpu()),text_path)
        _write_json(p/'dependencies.json',deps)
        del text,tokenizer,ctx,lens,alt,altlens;gc.collect();torch.cuda.empty_cache()
    textdata=torch.load(text_path,weights_only=True,map_location='cpu');deps=json.loads((p/'dependencies.json').read_text())
    batch0=torch.load(p/'batch.pt',weights_only=False,map_location='cpu')
    batch0.update(context=textdata['context'],context_lens=textdata['context_lens'])
    batch=prepare_batch(record,batch0,device=device,dtype=dtype);del batch0
    # Only a finite singleton's frozen features are memoized. Trainable outputs are never cached.
    original_encode=vae.encode;encoded={}
    def cached_encode(videos,device,**kw):
        tensors=list(videos);key=tuple((tuple(v.shape),str(v.dtype),digest(v)) for v in tensors)
        if key not in encoded:
            with torch.no_grad():encoded[key]=original_encode(tensors,device=device,**kw).detach()
        return encoded[key]
    vae.encode=cached_encode
    codec=GoalCodec(vae)
    with torch.no_grad():
        target_latent=codec.encode(batch['targets'].image)
        reconstruction=codec.decode(target_latent);_display_image(reconstruction).save(p/'report/inputs/vae_reconstruction.png')
    event('frozen_inputs_ready',instruction=record['instruction'],target_uv=batch['targets'].uv.cpu().tolist())
    common_hash=None;adapter_hash=None
    for mode in ([args.mode] if args.mode else NAMES):
        arm=p/mode
        if arm.exists():raise ValueError(f'preserve existing arm: {arm}')
        arm.mkdir();event('loading_arm',mode=mode)
        _seed(42);dit=_construct(cfg['components']['dit'])
        load_component_weights(dit,cfg['baseline_checkpoint'],'video_backbone.dit.')
        _seed(777)
        e=QueryGoalExpert(dit,codec,dit_config=cfg['components']['dit']['extra_kwargs'],
            query_config=cfg['model']['query_goal'],history_config=dict(enabled=True),
            conditioning_mode=mode,use_gradient_checkpointing=True).to(device=device,dtype=dtype)
        del dit
        qhash=fingerprint(e.encoder.state_dict())
        if common_hash is None:common_hash=qhash
        assert qhash==common_hash
        if mode!='legacy_text':
            ah=fingerprint(e.condition_projection.state_dict())
            if adapter_hash is None:adapter_hash=ah
            assert adapter_hash==ah
        summary=dict(mode=mode,status='running',completed_steps=0,evaluations=[],sensitivity={},
            query_initial_sha256=qhash,protocol=protocol,condition_adapter_initial_sha256=fingerprint(e.condition_projection.state_dict()))
        durations=[];peaks=[]
        def publish():
            _write_json(arm/'summary.json',summary);render(p)
        def call(b):
            return e(b['observed'],b['context'],b['targets'],context_lens=b['context_lens'],
                observed_view_valid=b['observed_view_valid'],**history_kwargs(b))
        @torch.no_grad()
        def sensitivity():
            e.eval();args0=dict(context_lens=batch['context_lens'],observed_view_valid=batch['observed_view_valid'],**history_kwargs(batch))
            z,blank,c=e._condition(batch['observed'],batch['context'],**args0)
            other=e._condition(batch['observed'],textdata['alt'].to(device=device),**{**args0,'context_lens':textdata['altlens'].to(device)})[2]
            no_visual=e._condition(torch.zeros_like(batch['observed']),batch['context'],**{**args0,'history_valid':torch.zeros_like(batch['history_valid'])})[2]
            raw=e.decoder.encode_observation(blank)
            torch.manual_seed(909)
            image=torch.randn_like(target_latent);tcp=torch.randn(1,192,device=device);t=torch.tensor([500.],device=device,dtype=dtype)
            a=e.decoder.velocity(raw,image,tcp,t,c)
            b=e.decoder.velocity(raw,image,tcp,t,other)
            d=e.decoder.velocity(raw,image,tcp,t,no_visual)
            rms=lambda x:float(x.float().square().mean().sqrt())
            embedded=e.decoder.embed_context(c)
            return dict(z_rms=rms(z),condition_rms=rms(embedded),
                changed_instruction_condition_rms=rms(embedded-e.decoder.embed_context(other)),
                changed_instruction_image_velocity_rms=rms(a[0]-b[0]),changed_instruction_tcp_velocity_rms=rms(a[1]-b[1]),
                blank_visual_image_velocity_rms=rms(a[0]-d[0]),blank_visual_tcp_velocity_rms=rms(a[1]-d[1]),
                note='fixed noisy input; sensitivity is not semantic correctness; blank visual is an intervention')
        @torch.no_grad()
        def evaluate(step):
            e.eval();torch.cuda.synchronize();begin=time.perf_counter()
            result=evaluate_patch(e,[batch],arm/f'eval_{step:06d}',step=step,seed=100042,num_steps=20)
            card=result['review'][0];errors=card['tcp_pixel_errors']
            goal=_display_image(batch['targets'].image)
            from openwam.train.goal_trainer import overlay_goal_targets
            goal=overlay_goal_targets(goal,card['target_uv_px'],card['target_point_valid'])
            _prediction_overlay(goal,card['predicted_uv_px']).save(arm/f'eval_{step:06d}/gt_vs_pred.png')
            fixed=[]
            for idx in (100,300,500,700,900):
                torch.manual_seed(9000+idx)
                fixed.append(float(e(batch['observed'],batch['context'],batch['targets'],
                    context_lens=batch['context_lens'],observed_view_valid=batch['observed_view_valid'],
                    timestep_indices=torch.tensor([idx],device=device),**history_kwargs(batch))['loss']))
            entry=dict(step=step,image_mae_uint8=result['image_mae_uint8'],tcp_mean_px=float(np.mean(errors)),
                tcp_left_px=errors[0],tcp_right_px=errors[1],fixed_flow_loss=float(np.mean(fixed)))
            summary['evaluations'].append(entry)
            if step in (0,args.steps):summary['sensitivity'][str(step)]=sensitivity()
            publish();event('evaluation',mode=mode,seconds=time.perf_counter()-begin,**entry)
        e.train_stage_a()
        groups=goal_parameter_groups(e,1e-5,1e-4);ids={id(v) for v in e.condition_projection.parameters()}
        for group in groups:group['params']=[v for v in group['params'] if id(v) not in ids]
        groups.append(dict(params=list(e.condition_projection.parameters()),lr=1e-4,name='condition_adapter'))
        opt=FP32MasterAdamW(groups)
        summary['trainable_parameters']=sum(v.numel() for v in e.parameters() if v.requires_grad)
        evaluate(0)
        for step in range(1,args.steps+1):
            e.train_stage_a();opt.zero_grad();_seed(10000+step)
            b=dict(batch);keep=torch.rand_like(batch['history_ages'])>=.2
            b['history_valid']=batch['history_valid']&keep
            b['history_ages']=torch.where(b['history_valid'],batch['history_ages'],0.)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            out=call(b)
            if not torch.isfinite(out['loss']):raise FloatingPointError('nonfinite loss')
            out['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(e.parameters(),1.,error_if_nonfinite=True,foreach=False)
            adapter_grad=sum(float(v.grad.float().square().sum()) for v in e.condition_projection.parameters() if v.grad is not None)**.5
            metrics=dict(step=step,loss=float(out['loss']),fm_loss=float(out['fm_loss']),tcp_loss=float(out['tcp_loss']),
                gradient_norm=float(norm),adapter_gradient_norm=adapter_grad,history_valid=int(b['history_valid'].sum()))
            opt.step();del out
            torch.cuda.synchronize();duration=time.perf_counter()-begin;peak=torch.cuda.max_memory_allocated()/2**30
            durations.append(duration);peaks.append(peak);metrics.update(seconds=duration,peak_allocated_gib=peak)
            with (arm/'train.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            summary.update(completed_steps=step,median_train_seconds=statistics.median(durations[10:] or durations),peak_train_allocated_gib=max(peaks))
            if step%25==0:publish();event('progress',mode=mode,**metrics)
            if step in {50,100,250,args.steps}:evaluate(step)
        # Extra independent sampling seeds at the final checkpoint.
        e.eval();seed_results=[]
        with torch.no_grad():
            for seed in (100043,100044):
                result=evaluate_patch(e,[batch],arm/f'final_seed_{seed}',step=args.steps,seed=seed,num_steps=20)
                seed_results.append(dict(seed=seed,image_mae_uint8=result['image_mae_uint8'],tcp_mean_px=float(np.mean(result['review'][0]['tcp_pixel_errors']))))
        summary['extra_eval_seeds']=seed_results
        save_query_bundle(e,arm/'query_goal.safetensors',training_metadata=summary,dependency_identities=deps,
                          dependency_metadata=cfg['components'])
        summary['status']='complete';publish();event('arm_complete',mode=mode,steps=args.steps)
        del opt,groups,e;encoded.clear();gc.collect();torch.cuda.empty_cache()
    event('comparison_complete');render(p)

if __name__=='__main__':main()
