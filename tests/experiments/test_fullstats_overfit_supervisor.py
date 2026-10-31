import json
from types import SimpleNamespace
import pytest

def test_readiness_requires_complete_hash_and_population(tmp_path):
 from scripts.experiments.run_fullstats_overfit import checked_readiness
 import hashlib
 p=tmp_path/'registry.json';p.write_text(json.dumps({'preparation':{'complete':True},'contract':{'horizon':32}}))
 h=hashlib.sha256(p.read_bytes()).hexdigest();q=tmp_path/f'registry-{h}.json';p.rename(q)
 ready=tmp_path/'readiness.json';ready.write_text(json.dumps(dict(status='complete',stats_registry=str(q),registry_sha256=h,source_yaml='source',horizon=32,dimensions=112)))
 assert checked_readiness(ready)['registry_sha256']==h
 q.write_text('{}')
 with pytest.raises(ValueError,match='hash'):checked_readiness(ready)

def test_timed_stage2_has_no_exact_update_budget():
 from scripts.experiments.run_fullstats_overfit import stage2_command
 from pathlib import Path
 cmd=stage2_command(Path('/tmp/run'),[1,4,9],3600,'joint')
 assert '--max-updates' not in cmd
 assert cmd[cmd.index('--seconds-per-arm')+1]=='3600'
 assert cmd[-2:]==['--arm','joint']

def test_training_completion_requires_both_frozen_arms():
 from scripts.experiments.run_fullstats_overfit import verify_arm
 verify_arm({'status':'complete','step':2,'teacher_frozen':True,'budget_seconds':3600},3600)
 for value in ({'status':'training','step':2,'teacher_frozen':True,'budget_seconds':3600},{'status':'complete','step':0,'teacher_frozen':True,'budget_seconds':3600},{'status':'complete','step':2,'teacher_frozen':False,'budget_seconds':3600}):
  with pytest.raises(ValueError):verify_arm(value,3600)
