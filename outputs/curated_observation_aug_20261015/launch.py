"""Launch the explicitly authorized augmentation run; keep old TensorBoards."""
from pathlib import Path
import json
import os
import subprocess
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PY = '/root/miniconda3/envs/openwam/bin/python'
pidfile = OUT/'supervisor.pid'
if pidfile.exists():
    proc = Path('/proc')/pidfile.read_text().strip()
    assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z', 'Already running'
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
           OMP_NUM_THREADS='8', NCCL_DEBUG='WARN', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
with (OUT/'supervisor.log').open('ab',buffering=0) as log:
    proc=subprocess.Popen([PY,str(OUT/'supervise.py')],cwd=ROOT,env=env,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(proc.pid))
print(json.dumps({'started_supervisor_pid':proc.pid,'run':'gripper_human_aug_lr0.0001_b320'}))
