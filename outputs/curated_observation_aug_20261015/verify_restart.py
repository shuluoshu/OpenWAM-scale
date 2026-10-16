"""Verify actual augmentation training, historical preservation and TB overlay."""
from pathlib import Path
import hashlib
import json
import math
import urllib.parse
import urllib.request
from omegaconf import OmegaConf

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'outputs/curated_gripper_human_20261014'
NAME='gripper_human_aug_lr0.0001_b320'
OLD='gripper_human_lr0.0001_b320'
RUN=SOURCE/NAME
status=json.loads((OUT/'status.json').read_text())
assert status['phase']=='training',status
for key in ('pid','supervisor_pid'):
 p=Path('/proc')/str(status[key]);assert p.exists() and (p/'stat').read_text().split()[2]!='Z',key
records=json.loads((OUT/'restart.json').read_text())
for item in records['preserved_events']:
 blob=Path(item['path']).read_bytes()
 assert len(blob)==item['bytes'] and hashlib.sha256(blob).hexdigest()==item['sha256'],item['path']
for name,digest in records['source_hashes'].items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
cfg=OmegaConf.load(RUN/'config.yaml')
previous=OmegaConf.load(SOURCE/OLD/'config.yaml')
for key in ('model','training','project','dataloader'):
 a=OmegaConf.to_container(cfg[key],resolve=True);b=OmegaConf.to_container(previous[key],resolve=True)
 if key=='training':
  for path in ('output_path',):a.pop(path,None);b.pop(path,None)
 if key=='project':a.pop('output_dir',None);b.pop('output_dir',None)
 if key=='dataloader':a.pop('observation_augmentation',None);b.pop('observation_augmentation',None)
 assert a==b, f'Unexpected configuration difference: {key}'
assert cfg.training.batch_size==320 and cfg.training.max_steps==1000
aug=OmegaConf.to_container(cfg.dataloader.observation_augmentation,resolve=True)
assert aug==OmegaConf.to_container(OmegaConf.load(OUT/'augmentation.yaml'),resolve=True)
assert json.loads((RUN/'effective_optimizer_lrs.json').read_text())==[1e-5,1e-4,1e-3]
assert json.loads((RUN/'initialization.json').read_text())==json.loads((SOURCE/OLD/'initialization.json').read_text())
assert json.loads((RUN/'label_contract.json').read_text())==json.loads((SOURCE/OLD/'label_contract.json').read_text())
rows=[json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines()]
training=[r for r in rows if r['split']=='train']
assert training and training[-1]['step']>=1,'No completed optimizer step yet'
for row in training:
 for key,value in row.items():
  if isinstance(value,(int,float)):assert math.isfinite(value),(row['step'],key,value)
 assert row['batch_size']==320
new_eval=json.loads((RUN/'evaluation.json').read_text())
old_eval=json.loads((SOURCE/OLD/'evaluation.json').read_text())
a,b=new_eval[0],old_eval[0]
assert a['step']==b['step']==0
initial={}
for key,value in a.items():
 if key=='step':continue
 assert math.isclose(value,b[key],rel_tol=1e-5,abs_tol=1e-7),(key,value,b[key])
 initial[key]={'new':value,'old':b[key],'difference':value-b[key]}

def fetch(port,path,**params):
 url=f'http://127.0.0.1:{port}'+path
 if params:url+='?'+urllib.parse.urlencode(params)
 with urllib.request.urlopen(url,timeout=25) as response:
  assert response.status==200
  return json.load(response)

checks={}
for port in (18082,18083):
 tags=fetch(port,'/data/plugin/scalars/tags')
 assert NAME in tags and OLD in tags,(port,list(tags))
 wanted=['train/loss_total','train/loss_action','train/loss_video',
         'eval_fixed_noise/clean_joint_mse','eval_fixed_noise/clean_gripper_mse','eval_fixed_noise/loss_fk_position']
 points={}
 for tag in wanted:
  assert tag in tags[NAME],(port,tag)
  data=fetch(port,'/data/plugin/scalars/scalars',run=NAME,tag=tag)
  assert data and all(math.isfinite(p[2]) for p in data),tag
  points[tag]={'points':len(data),'last_step':data[-1][1],'last_value':data[-1][2]}
 checks[str(port)]={'runs':list(tags),'scalars':points}
report={'all_passed':True,'run':NAME,'baseline':OLD,'last_train_step':training[-1]['step'],
        'last_train':training[-1],'initial_clean_evaluation_matches':initial,
        'unchanged_model_training_labels_verified':True,'batch':320,'augmentation':aug,
        'old_events_preserved':len(records['preserved_events']),'tensorboards':checks}
(OUT/'restart_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'all_passed':True,'last_train_step':training[-1]['step'],'old_events_preserved':len(records['preserved_events']),
                 'initial_eval_matches':True,'last_loss':training[-1]['loss_total'],'batch':320,'tensorboard_ports':list(checks)},indent=2))
