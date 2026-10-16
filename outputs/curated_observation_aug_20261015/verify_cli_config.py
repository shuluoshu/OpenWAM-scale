"""Exercise the real CLI config path without creating models or using CUDA."""
import importlib.util
import json
from pathlib import Path
import sys
from omegaconf import OmegaConf
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'outputs/curated_gripper_human_20261014'
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('augmentation_entrypoint',SOURCE/'run.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
seen={}
class FakeTrainer:
 def __init__(self,cfg,accelerator,dataset,args):
  assert accelerator is None
  assert cfg.training.batch_size==320
  assert cfg.training.max_steps==1000
  aug=OmegaConf.to_container(cfg.dataloader.observation_augmentation,resolve=True)
  assert aug==OmegaConf.to_container(dataset,resolve=True)
  assert aug['color_jitter']['p']==.8 and aug['patch_mask']['p']==.5
  assert aug['view_dropout']['p']==.1 and aug['state_dropout']['p']==.1
  seen['augmentation']=aug
 def probe(self):seen['probe_called']=True
module.Trainer=FakeTrainer
module.CuratedDataset=lambda augmentation:augmentation
sys.argv=['run.py','--probe','--name','config_preflight_only','--batch','320','--steps','1000',
          '--augmentation-config',str(ROOT/'configs/augmentation/multiview.yaml')]
module.main()
assert seen['probe_called']
report={'all_passed':True,'real_main_cli_config_checked':True,'cuda_model_loaded':False,**seen}
(OUT/'cli_config_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
