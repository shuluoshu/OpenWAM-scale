"""Restart the authorized smoke experiment and serve its new TensorBoard."""
from pathlib import Path
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / "outputs/curated_gripper_cmd_20261013"
SHARED = ROOT / "outputs/curated_train_20261011/tensorboard"
NAME = "gripper_human_lr0.0001_b320"
PY = "/root/miniconda3/envs/openwam/bin/python"
TB_ROOT = OUT / "tensorboard"
RUN = OUT / NAME
child = None
burn = None
tb = None


def dump(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def argv(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")


def status(phase, **values):
    dump(OUT / "status.json", dict(phase=phase, run=NAME, supervisor_pid=os.getpid(),
         time=time.time(), tensorboard_root=str(TB_ROOT), tensorboard_port=18083, **values))


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    assert json.loads((OUT / "validation.json").read_text())["all_passed"]
    assert json.loads((OUT / "human_validation.json").read_text())["all_passed"]
    assert not (RUN / "metrics.jsonl").exists(), "Refusing to mix training starts"
    assert not list((TB_ROOT / NAME).glob("events.out.tfevents.*")), "New run must be empty"
    RUN.mkdir(exist_ok=True)
    TB_ROOT.mkdir(exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 18083))
    with (OUT / "tensorboard.log").open("ab", buffering=0) as log:
        tb = subprocess.Popen([PY, "-m", "tensorboard.main", "--logdir", str(TB_ROOT),
             "--host", "0.0.0.0", "--port", "18083", "--load_fast=false"],
             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / "tensorboard.pid").write_text(str(tb.pid))
    deadline = time.monotonic() + 40
    while True:
        assert tb.poll() is None, "New TensorBoard exited"
        try:
            with urllib.request.urlopen("http://127.0.0.1:18083/", timeout=2) as response:
                assert response.status == 200
            break
        except OSError:
            assert time.monotonic() < deadline, "New TensorBoard did not become ready"
            time.sleep(.5)
    link = SHARED / NAME
    assert not link.exists() and not link.is_symlink(), "Comparison run already exists"
    (TB_ROOT / NAME).mkdir(exist_ok=True)
    link.symlink_to(TB_ROOT / NAME, target_is_directory=True)
    old = json.loads((OLD / "status.json").read_text())
    preservation = []
    for path in SHARED.rglob("events.out.tfevents.*"):
        blob = path.read_bytes()
        preservation.append(dict(path=str(path), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
    dump(OUT / "restart.json", dict(previous_status=old, preserved_event_prefixes=preservation,
         new_run=NAME, tensorboard_root=str(TB_ROOT), restart_time=time.time()))
    old_supervisor, old_child = old["supervisor_pid"], old["pid"]
    status("stopping_previous", previous_pid=old_child, tensorboard_pid=tb.pid)
    if alive(old_supervisor):
        assert str(OLD / "supervise.py").encode() in argv(old_supervisor)
        if alive(old_child):
            assert str(OLD / "run.py").encode() in argv(old_child)
        os.kill(old_supervisor, signal.SIGTERM)
        deadline = time.monotonic() + 55
        while alive(old_supervisor) and time.monotonic() < deadline:
            time.sleep(.25)
        assert not alive(old_supervisor), "Old supervisor did not exit"
    assert not alive(old_child), "Old torchrun still exists"
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = argv(int(p.name))
            assert not (str(OLD / "run.py").encode() in args and alive(int(p.name))), "Old worker remains"
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    candidate = old.get("paused_burn_pid")
    if candidate and alive(candidate):
        args = argv(candidate)
        assert args[1] in (b"burn_gpu.py", str(ROOT / "burn_gpu.py").encode())
        assert Path(f"/proc/{candidate}/cwd").resolve() == ROOT
        burn = candidate
        os.kill(burn, signal.SIGSTOP)
    for item in preservation:
        with Path(item["path"]).open("rb") as handle:
            assert hashlib.sha256(handle.read(item["bytes"])).hexdigest() == item["sha256"]
    metrics = [json.loads(line) for line in (OLD / old["run"] / "metrics.jsonl").read_text().splitlines()]
    last_step = max(r["step"] for r in metrics if r["split"] == "train")
    dump(OLD / "status.json", dict(old, phase="stopped_for_human_gt_restart", last_train_step=last_step,
         replacement_run=NAME, stopped=time.time()))
    cmd = ["/root/miniconda3/envs/openwam/bin/torchrun", "--standalone", "--nproc_per_node=1",
           str(OUT / "run.py"), "--name", NAME, "--batch", "320", "--steps", "1000",
           "--lr", "0.0001", "--eval-every", "25", "--tensorboard-root", str(TB_ROOT)]
    with (RUN / "train.log").open("x") as log:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status("training", pid=child.pid, command=cmd, previous_last_step=last_step,
               paused_burn_pid=burn, tensorboard_pid=tb.pid, old_events_preserved=True)
        dump(OUT / "handoff.json", dict(training_started=True, run=NAME, command=cmd,
             tensorboard_root=str(TB_ROOT), tensorboard_port=18083, previous_last_step=last_step))
        rc = child.wait()
    if rc:
        raise RuntimeError(f"Training exited {rc}")
    status("complete", steps=1000, tensorboard_pid=tb.pid)
except BaseException:
    status("failed", traceback=traceback.format_exc())
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
