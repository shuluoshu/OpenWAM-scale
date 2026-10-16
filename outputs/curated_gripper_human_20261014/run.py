"""Curated training with robot commands and human openness targets, retaining measured inputs and state-spike rejection."""
import os,sys,json,time,math,gc,argparse,logging,collections
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from data import CuratedDataset,OUT,ROOT
from scripts.train import _build_accelerator
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.observation_cache import prepare_observation

def dump(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def move(x,device):
 if torch.is_tensor(x):return x.detach().to(device)
 if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
 if isinstance(x,list):return [move(v,device) for v in x]
 return x
def tensor_bytes(x):
 if torch.is_tensor(x):return x.numel()*x.element_size()
 if isinstance(x,np.ndarray):return x.nbytes
 if isinstance(x,dict):return sum(tensor_bytes(v) for v in x.values())
 if isinstance(x,(list,tuple)):return sum(tensor_bytes(v) for v in x)
 return 0
def combine(rows):
 result={}
 for k in set().union(*(r.keys() for r in rows)):
  values=[r.get(k) for r in rows];v=values[0]
  if k=='fk_supervision':result[k]=sum([x if x is not None else [None] for x in values],[])
  elif torch.is_tensor(v):result[k]=torch.cat(values,dim=0)
  elif isinstance(v,list):result[k]=sum(values,[])
  else:
   assert all(x==v for x in values), (k,[type(x) for x in values])
   result[k]=v
 return result

class Trainer(OpenWAMTrainer):
 def __init__(self,cfg,accelerator,dataset,args):
  self.args=args;self.run_dir=OUT/args.name;self.run_dir.mkdir(exist_ok=True)
  self.writer=SummaryWriter(str(Path(args.tensorboard_root)/args.name),flush_secs=5)
  self.cache=collections.OrderedDict();self.cache_bytes=0;self.eval_history=[]
  assert all(r.GRIPPER_NORMALIZATION == "clip_0_1" for r in dataset.readers)
  super().__init__(cfg,accelerator,dataset)
  dump(self.run_dir/"label_contract.json", {"gripper_normalization":"clip_0_1", "unified_gripper_slots":[9,43], "state_input":"measured_state[t]", "gripper_gt":{"robot":"aligned cmd[t:t+32]", "human":"stored actions.eef[t:t+32,9/19], already future-aligned"}, "human_families":["LightWheel","mecka","egodex","egoverse"], "human_target_timing":"LightWheel/Mecka state+1; EgoDex/EgoVerse state+3 in audited clips", "joint_gt":"q[t+1:t+33]-q[t]", "filter_report":str(OUT/"gripper_filter_report.json"), "missing_target":"mask_gripper_gt", "other_channels":"next-state pose/joint targets; regenerated filtered-population statistics", "initialization":"foundation; no reuse of old gripper-scale fine-tuned weights"})
  enc=self.architecture.action_backbone.action_encoder
  with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.manual_seed(20261009);torch.cuda.manual_seed_all(20261009)
   enc.weight[:,80:].uniform_(-1/math.sqrt(80),1/math.sqrt(80))
  dump(self.run_dir/'initialization.json',{'checkpoint':str(cfg.training.finetune_ckpt_path),'new_encoder_columns':'80:112 initialized uniform, seed20261009; existing80 untouched; decoder expansion unchanged','trainable_parameters':sum(p.numel() for p in self.architecture.parameters() if p.requires_grad)})
  OmegaConf.save(cfg,self.run_dir/'config.yaml')
 def emit(self,split,step,values):
  values={k:float(v) for k,v in values.items() if np.isfinite(float(v))}
  row={'split':split,'step':step,'wall_time':time.time(),**values}
  with (self.run_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  for k,v in values.items():self.writer.add_scalar(split+'/'+k,v,step)
  self.writer.flush();print('METRICS',json.dumps(row),flush=True)
 def arch(self):return self.accelerator.unwrap_model(self.architecture) if self.accelerator is not None else self.architecture
 def prepared(self,batch):
  rows=[]
  for sample in batch:
   key=sample['_curated_index']
   existed=key in self.cache
   row=prepare_observation(sample,lambda batch:move(self.arch().prepare_inputs(batch),'cpu'),self.cache)
   if not existed and key in self.cache:
    self.cache_bytes+=tensor_bytes(self.cache[key])
   if key in self.cache:self.cache.move_to_end(key)
   while self.cache_bytes>16*2**30 and len(self.cache)>1:
    _,removed=self.cache.popitem(last=False);self.cache_bytes-=tensor_bytes(removed)
   rows.append(move(row,self.arch().device))
  return combine(rows)
 def result(self,inputs):
  return self.architecture.compute_loss(**inputs,lambda_video=self.lambda_video,lambda_action=self.lambda_action,lambda_fk_position=self.lambda_fk_position,lambda_fk_rotation=self.lambda_fk_rotation)
 def compute_loss(self,batch):
  r=self.result(self.prepared(batch));assert all(torch.isfinite(v).all() for v in r.values())
  return {'total':r['loss'],'video':r['loss_video'],'action':r['loss_action'],'fk_position':r['loss_fk_position'],'fk_rotation':r['loss_fk_rotation'],'fk_valid_count':r['fk_valid_count']}
 def build_dataloader(self,batch_size):
  generator=torch.Generator().manual_seed(42)
  sampler=torch.utils.data.WeightedRandomSampler(self.dataset.weights,len(self.dataset),replacement=True,generator=generator)
  return torch.utils.data.DataLoader(self.dataset,batch_size=batch_size,sampler=sampler,collate_fn=list,num_workers=0,drop_last=True)
 def build_optimizer(self):
  groups={'video':[],'action':[],'projection':[]}
  for name,p in self.architecture.named_parameters():
   if not p.requires_grad:continue
   key='projection' if name.startswith(('action_backbone.action_encoder.','action_backbone.action_decoder.')) else 'action' if name.startswith('action_backbone.') else 'video'
   groups[key].append(p)
  rates={'video':1e-5,'action':self.args.lr,'projection':self.args.lr*10}
  dump(self.run_dir/'optimizer_groups.json',{k:{'lr':rates[k],'numel':sum(x.numel() for x in v)} for k,v in groups.items()})
  return torch.optim.AdamW([{'params':v,'lr':rates[k]} for k,v in groups.items() if v],betas=(.9,.95),weight_decay=.01)
 def prepare_accelerate(self,*args):
  result=super().prepare_accelerate(*args)
  rates=[g['lr'] for g in result[0].param_groups]
  assert rates==[1e-5,self.args.lr,self.args.lr*10],rates
  dump(self.run_dir/'effective_optimizer_lrs.json',rates)
  self.evaluate(0);return result
 @torch.no_grad()
 def evaluate(self,step):
  arch=self.arch();training=arch.training;arch.eval();totals=collections.defaultdict(float);counts=collections.defaultdict(int)
  try:
   with self.dataset.unaugmented(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
    for n,index in enumerate(self.dataset.eval_indices()):
     sample=self.dataset[index];inputs=self.prepared([sample]);mask=~inputs['action_is_pad'];gt=inputs['actions'].float();captures={}
     def hook(module,args,kwargs,output):
      noisy,timestep=args[:2];sigma=timestep.float().reshape(-1,1,1)/arch.action_backbone.scheduler.num_train_timesteps
      clean=noisy.float()-sigma*output[1].float()
      for name,slots in [('action',list(range(112))),('tcp',list(range(9))+list(range(34,43))),('joint',list(range(80,94))),('head',list(range(94,112))),('gripper',[9,43])]:
       valid=mask[...,slots]
       if valid.any():captures['clean_'+name+'_mse']=float((clean[...,slots]-gt[...,slots]).square()[valid].mean())
     handle=arch.register_forward_hook(hook,with_kwargs=True)
     try:
      torch.manual_seed(17001+n);torch.cuda.manual_seed_all(17001+n);r=self.result(inputs)
     finally:handle.remove()
     vals={k:float(v) for k,v in r.items()};vals.update(captures)
     for k,v in vals.items():
      if k.startswith('loss_fk') and vals['fk_valid_count']==0:continue
      totals[k]+=v;counts[k]+=1
     for k,v in captures.items():self.writer.add_scalar('eval_dataset/'+sample['_curated_id']+'/'+k,v,step)
    averaged={k:v/counts[k] for k,v in totals.items()};self.emit('eval_fixed_noise',step,averaged)
    self.eval_history.append({'step':step,**averaged});dump(self.run_dir/'evaluation.json',self.eval_history)
  finally:arch.train(training)
 def log_step(self,**kw):
  super().log_step(**kw);self.emit('train',kw['global_step'],{**kw['metrics'],'lr':kw['lr'],'steps_per_sec':kw['steps_per_sec'],
   'lr_action':self.args.lr,'lr_projection':self.args.lr*10,'frozen_cache_gib':self.cache_bytes/2**30,'frozen_cache_windows':len(self.cache),
   'batch_size':self.cfg.training.batch_size,'max_memory_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'max_memory_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
  if kw['global_step']%self.args.eval_every==0 or kw['global_step']==self.args.steps:self.evaluate(kw['global_step'])
 def probe(self):
  entries=[];best=0;total=torch.cuda.get_device_properties(0).total_memory
  samples=[self.dataset[i] for i in self.dataset.eval_indices()]
  for b in [64,128,256,320,384,448,512,640]:
   self.architecture.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.time()
   try:
    inputs=self.prepared([samples[i%len(samples)] for i in range(b)])
    loss=self.result(inputs)['loss'];loss.backward();torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated();record={'batch':b,'status':'ok','peak_allocated_gib':peak/2**30,'seconds':time.time()-t}
    if peak<total*.78:best=b
    entries.append(record);print('BATCH_PROBE',record,flush=True)
    del loss,inputs
    if peak>=total*.78:break
   except torch.OutOfMemoryError as e:
    entries.append({'batch':b,'status':'oom','error':str(e)[:250]});break
   finally:self.architecture.zero_grad(set_to_none=True)
  assert best>0, entries
  dump(OUT/'batch_probe.json',{'selected_batch':best,'reserve_fraction':.22,'note':'forward/backward probe; actual optimizer run validates remaining memory','results':entries})
  self.writer.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--name',default='gripper_human_lr0.0001_b320');p.add_argument('--batch',type=int,default=320);p.add_argument('--steps',type=int,default=1000);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--eval-every',type=int,default=25);p.add_argument('--tensorboard-root',default=str(ROOT/'outputs/curated_train_20261011/tensorboard'));p.add_argument('--augmentation-config',type=Path);args=p.parse_args()
 assert json.loads((OUT/'validation.json').read_text())['all_passed']
 assert not (OUT/args.name/'metrics.jsonl').exists(), 'Use a new run name; never mix label contracts'
 assert not list((Path(args.tensorboard_root)/args.name).glob('events.out.tfevents.*')), 'TensorBoard run already has events'
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');torch.set_num_threads(8)
 with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):cfg=compose(config_name='openwam_alpha_v0_1_smoke')
 cfg.dataloader=OmegaConf.create({'type':'curated_fixture','manifest':str(OUT/'data_manifest.json'),'gripper_normalization':'clip_0_1','gripper_target_source':'controller_command_or_human_openness_target','gripper_state_filter':True})
 augmentation=OmegaConf.load(args.augmentation_config) if args.augmentation_config else None
 if augmentation is not None:OmegaConf.update(cfg,'dataloader.observation_augmentation',augmentation,force_add=True)
 cfg.training.batch_size=args.batch;cfg.training.max_steps=args.steps;cfg.training.save_steps=200 if args.steps>=200 else 0
 cfg.training.output_path=str(OUT/args.name/'checkpoints');cfg.training.dataset_num_workers=0
 cfg.training.learning_rate=1e-5;cfg.training.lr_scheduler=None;cfg.training.weight_decay=.01
 cfg.project.output_dir=str(OUT/args.name);cfg.project.wandb=None
 accelerator=None if args.probe else _build_accelerator(cfg)
 trainer=Trainer(cfg,accelerator,CuratedDataset(augmentation=augmentation),args)
 if args.probe:trainer.probe()
 else:
  trainer.train();trainer.writer.close();dump(OUT/args.name/'finished.json',{'steps':args.steps,'time':time.time()})
 if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':main()
