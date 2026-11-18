import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import psutil

ROOT = Path('/mnt/data/cpfs/b5big/yeqingda/CodeX/storage/longtrain_16gpu_20261112_20261115/control/kf1_endpoint_20261115')
pod = os.uname().nodename
assert os.environ['DLC_JOB_ID'] == 'dlc17vidcbxadyj2'
path = Path(json.loads(Path('/mnt/data/cpfs/b5/yeqingda/CodeX/docs/2026-11-18/longtrain_status_recovery/launch_specs.json').read_text())[pod])
spec = json.loads(path.read_text())
receipt = Path(spec['receipt'])
assert not receipt.exists(), 'Inspect receipt/processes before any retry'
assert not Path(spec['exit_file']).exists()
for filename, expected in spec['files'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
assert subprocess.check_output(['git', '-C', spec['cwd'], 'rev-parse', 'HEAD'], text=True).strip() == spec['code_commit']
assert not subprocess.check_output(['git', '-C', spec['cwd'], 'status', '--porcelain'], text=True).strip()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip(), 'GPU occupancy must be inspected before launching'
assert psutil.virtual_memory().available > 250 * 2**30
output = Path(spec['command'][spec['command'].index('--output') + 1])
assert not (output / 'STOP_REQUESTED').exists()
pointer = json.loads((output / 'checkpoint.json').read_text())
assert pointer['manifest_sha256'] == spec['source_checkpoint_manifest_sha256']
if spec['node_rank'] == 0:
    port = int(spec['command'][spec['command'].index('--master_port') + 1])
    with socket.socket() as probe:
        probe.bind(('0.0.0.0', port))
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1', PYTHONPATH=spec['cwd'],
                   LONGTRAIN_CONTROL=str(ROOT), LONGTRAIN_ARCH=spec['arch'])
environment.update(spec.get('env', {}))
with Path(spec['log']).open('xb') as log:
    process = subprocess.Popen([sys.executable, str(ROOT / 'lifecycle.py'), str(path)], cwd=spec['cwd'], env=environment,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, close_fds=True)
value = dict(phase=spec['phase'], arch=spec['arch'], job=os.environ['DLC_JOB_ID'], pod=pod,
             node_rank=spec['node_rank'], pid=process.pid, create_time=psutil.Process(process.pid).create_time(),
             world=16, local_gpus=8, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             log=spec['log'], code_commit=spec['code_commit'], source_checkpoint=pointer['update'])
temporary = receipt.with_suffix('.tmp')
temporary.write_text(json.dumps(value, indent=2) + '\n')
temporary.replace(receipt)
print(json.dumps(dict(pod=pod, arch=spec['arch'], node_rank=spec['node_rank'], launched=True)), flush=True)
print('DIAG_COMPLETE_9021', flush=True)
