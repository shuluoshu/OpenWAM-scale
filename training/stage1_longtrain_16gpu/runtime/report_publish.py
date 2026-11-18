#!/usr/bin/env python3
"""Single-host CPU publisher for combined reports. Never calls a trainer."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);runs=args.runs_root.resolve();output=args.output.resolve()
    if output in (runs,runs/'G',runs/'Cstar'):raise ValueError('report output must be a separate report directory')
    # Only this CPU subprocess is masked. No parent/job environment is changed.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from validation16 import build_report
    from distributed16 import atomic_json
    result=build_report(runs,output)
    root=Path(__file__).resolve().parent
    receipt=dict(format='openwam.stage1.report-publication.v1',published_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 runs_root=str(runs),output=str(output),gpu_disabled=True,training_mutations=False,
                 code={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('report_publish.py','validation16.py')},**result)
    atomic_json(output/'report_publish.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,allow_nan=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
