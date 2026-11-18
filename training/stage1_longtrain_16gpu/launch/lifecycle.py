import datetime,json,os,subprocess,sys
from pathlib import Path
spec=json.loads(Path(sys.argv[1]).read_text())
try:code=subprocess.call(spec['command'],cwd=spec['cwd'],env=os.environ.copy())
except Exception as e:
    code=125
    print(type(e).__name__,str(e),flush=True)
p=Path(spec['exit_file']);temp=p.with_suffix('.tmp')
temp.write_text(json.dumps(dict(exit_code=code,phase=spec['phase'],arch=spec['arch'],node_rank=spec['node_rank'],finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');temp.replace(p)
raise SystemExit(code)
