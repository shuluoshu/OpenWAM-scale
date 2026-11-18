"""Execute a receipt-producing script with the verified DLC training Python."""
import asyncio
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import websockets

async def main():
    job,pod,script,output=sys.argv[1:5]
    data=json.loads(subprocess.check_output(['/etc/dsw/runtime/export_bin/aliyun','pai-dlc',
        'get-web-terminal','--job-id',job,'--pod-id',pod,'--region','cn-shanghai','--is-shared','true']))
    payload=base64.b64encode(Path(script).read_bytes()).decode()
    code='import base64;exec(base64.b64decode('+repr(payload)+'))'
    command=shlex.quote('/tmp/openwam-runtime-20261031/env/bin/python')+' -c '+shlex.quote(code)+'\n'
    collected=[]
    async with websockets.connect(data['WebTerminalUrl'],open_timeout=15,max_size=2**22) as ws:
        await asyncio.wait_for(ws.recv(),15)
        await ws.send(json.dumps(dict(operation='stdin',data=command,rows=40,cols=160)))
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            try:
                obj=json.loads(await asyncio.wait_for(ws.recv(),10))
                text=obj.get('data','');collected.append(text)
                if 'DIAG_COMPLETE_9021' in text:break
            except asyncio.TimeoutError:continue
    Path(output).write_text(''.join(collected))
    print(pod,'terminal_output_saved',flush=True)

asyncio.run(main())
