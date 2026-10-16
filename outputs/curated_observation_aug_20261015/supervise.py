"""Run the authorized augmentation comparison while preserving historical events."""
from pathlib import Path
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import traceback
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'outputs/curated_gripper_human_20261014'
BASELINE = 'gripper_human_lr0.0001_b320'
NAME = 'gripper_human_aug_lr0.0001_b320'
TB_ROOT = SOURCE / 'tensorboard'
SHARED = ROOT / 'outputs/curated_train_20261011/tensorboard'
RUN = SOURCE / NAME
PY = '/root/miniconda3/envs/openwam/bin/python'
child = None
burn = None


def dump(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def status(phase, **values):
    dump(OUT/'status.json', dict(phase=phase, run=NAME, supervisor_pid=os.getpid(),
         time=time.time(), tensorboard_root=str(TB_ROOT), tensorboard_port=18083, **values))


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A live writer can be midway through the last JSON line.
    return result


baseline = rows(SOURCE/BASELINE/'metrics.jsonl')
baseline_eval = {r['step']: r for r in baseline if r['split'] == 'eval_fixed_noise'}
baseline_train = {r['step']: r for r in baseline if r['split'] == 'train'}


def comparison():
    current = rows(RUN/'metrics.jsonl')
    train = [r for r in current if r['split'] == 'train']
    evaluation = [r for r in current if r['split'] == 'eval_fixed_noise']
    keys = ['loss', 'loss_action', 'loss_video', 'clean_action_mse', 'clean_gripper_mse',
            'clean_tcp_mse', 'clean_joint_mse', 'clean_head_mse', 'loss_fk_position', 'loss_fk_rotation']
    matched = []
    for r in evaluation:
        if r['step'] not in baseline_eval:
            continue
        b = baseline_eval[r['step']]
        matched.append(dict(step=r['step'], metrics={k: dict(baseline=b[k], augmented=r[k],
            difference=r[k]-b[k], ratio=r[k]/b[k] if b[k] else None)
            for k in keys if k in b and k in r}))
    last_train = train[-1] if train else None
    same_step = baseline_train.get(last_train['step']) if last_train else None
    dump(OUT/'comparison.json', dict(updated=time.time(), baseline=BASELINE, augmented=NAME,
         baseline_final=baseline_eval[max(baseline_eval)], matched_clean_evaluations=matched,
         latest_train=last_train, baseline_train_at_same_step=same_step,
         note='Compare clean eval at equal steps; train RGB/video targets are augmented. Step 0 validates initialization, not learning.'))
    return train, evaluation


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    assert json.loads((SOURCE/BASELINE/'finished.json').read_text())['steps'] == 1000
    for filename in ('validation.json', 'human_validation.json'):
        assert json.loads((SOURCE/filename).read_text())['all_passed']
    assert json.loads((ROOT/'outputs/observation_augmentation_20261015/review/validation.json').read_text())['all_passed']
    assert not (RUN/'metrics.jsonl').exists(), 'Refusing to mix starts'
    assert not list((TB_ROOT/NAME).glob('events.out.tfevents.*')), 'Run must be new'
    for port in (18082, 18083):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/data/plugin/scalars/tags', timeout=20) as response:
            assert BASELINE in json.load(response), 'Baseline missing from comparison TensorBoard'
    events = {}
    for directory in (SHARED, TB_ROOT):
        for path in directory.rglob('events.out.tfevents.*'):
            events[str(path.resolve())] = dict(path=str(path.resolve()), bytes=path.stat().st_size,
                                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    sources = ['openwam/dataloader/transforms/observation.py', 'openwam/train/utils/observation_cache.py',
               'openwam/dataloader/bases/lerobot_v3_reader.py', 'openwam/dataloader/curated_fixture.py',
               'openwam/dataloader/mora_relative_lerobot.py', 'openwam/dataloader/utils/mora_relative.py',
               'openwam/dataloader/utils/gripper.py', 'openwam/train/openwam_trainer.py',
               'outputs/curated_gripper_human_20261014/run.py', 'outputs/curated_gripper_human_20261014/data.py']
    hashes = {}
    for name in sources:
        data = (ROOT/name).read_bytes()
        dest = OUT/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    aug = OUT/'augmentation.yaml'
    aug.write_bytes((ROOT/'configs/augmentation/multiview.yaml').read_bytes())
    dump(OUT/'restart.json', dict(baseline=BASELINE, new_run=NAME, preserved_events=list(events.values()),
         source_hashes=hashes, data_manifest_sha256=hashlib.sha256((SOURCE/'data_manifest.json').read_bytes()).hexdigest(),
         augmentation_sha256=hashlib.sha256(aug.read_bytes()).hexdigest(), restart_time=time.time()))
    RUN.mkdir(exist_ok=True)
    (TB_ROOT/NAME).mkdir(exist_ok=True)
    link = SHARED/NAME
    if link.is_symlink():
        assert link.resolve() == (TB_ROOT/NAME).resolve(), 'Wrong comparison link'
    else:
        assert not link.exists(), 'Comparison path already used'
        link.symlink_to(TB_ROOT/NAME, target_is_directory=True)
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = [x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
            if len(args)>1 and Path(args[1]).name=='burn_gpu.py' and (process/'cwd').resolve()==ROOT:
                assert burn is None, 'Multiple repository burn processes'
                if (process/'stat').read_text().split()[2] not in ('T','t','Z'):
                    burn = int(process.name)
                    os.kill(burn, signal.SIGSTOP)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    cmd = ['/root/miniconda3/envs/openwam/bin/torchrun','--standalone','--nproc_per_node=1',
           str(SOURCE/'run.py'),'--name',NAME,'--batch','320','--steps','1000',
           '--lr','0.0001','--eval-every','25','--tensorboard-root',str(TB_ROOT),
           '--augmentation-config',str(aug)]
    with (RUN/'train.log').open('x') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status('training', pid=child.pid, command=cmd, paused_burn_pid=burn,
               baseline=BASELINE, old_events_preserved=True)
        dump(OUT/'handoff.json', dict(command=cmd, run_dir=str(RUN), tensorboard_port=18083,
             baseline=BASELINE, new_run=NAME, python=PY, steps=1000, batch=320))
        while child.poll() is None:
            comparison()
            time.sleep(10)
        comparison()
        if child.returncode:
            raise RuntimeError(f'Training exited {child.returncode}; see {RUN / "train.log"}')
    status('complete', steps=1000)
except BaseException:
    status('failed', traceback=traceback.format_exc(), pid=child.pid if child else None)
    raise
finally:
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    if burn is not None:
        try:
            os.kill(burn, signal.SIGCONT)
        except ProcessLookupError:
            pass
