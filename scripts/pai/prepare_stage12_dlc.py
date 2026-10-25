#!/usr/bin/env python3
"""Render a 4-node 32-GPU DLC request; never submits a job.

Site values must be explicitly resolved. Uses the installed PAI skill validator
and optionally executes the CLI's serialization-only dry run.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess


def build_request(site):
    required=('workspace_id','quota_id','image','cpu','memory','shared_memory','python','code_dir','commit')
    missing=[k for k in required if not site.get(k)]
    if missing:
        raise ValueError('Resolve site fields before rendering: '+', '.join(missing))
    if site.get('pod_count')!=4 or site.get('gpus_per_pod')!=8:
        raise ValueError('This training entry supports exactly 4 Pods x 8 local GPUs')
    if not site.get('data_sources') or not site.get('user_vpc'):
        raise ValueError('Resolve data_sources and user_vpc')
    env={'OPENWAM_CODE_DIR':site['code_dir'], 'OPENWAM_PYTHON':site['python'],
         'OPENWAM_EXPECTED_COMMIT':(subprocess.check_output(['git','-C',site['code_dir'],'rev-parse','HEAD'],text=True).strip() if site['commit']=='auto' else site['commit']),
         'STAGE12_CONFIG':site.get('pipeline_config','configs/goal/stage12_query_history_posttrain_3fps_4x8gpu.yaml')}
    if site.get('preparation_status'):
        env['STAGE12_PREPARATION_STATUS']=site['preparation_status']
        env['STAGE12_PREPARED_MARKER']=site['prepared_marker']
    command=['bash', str(Path(site['code_dir'])/'scripts/pai/stage12_entry.sh'), 'train']
    if site.get('resume'):
        command.append('--resume')
    if site.get('stage2_resume'):
        if not site.get('resume'): raise ValueError('stage2_resume requires resume=true')
        command.extend(['--stage2-resume',site['stage2_resume']])
    return dict(Region=site.get('region','cn-shanghai'),WorkspaceId=site['workspace_id'],
        ResourceId=site['quota_id'],DisplayName=site['display_name'],JobType='PyTorchJob',
        Accessibility='PRIVATE',Priority=1,
        JobSpecs=[dict(Type='Worker',PodCount=4,Image=site['image'],ResourceConfig=dict(
            CPU=str(site['cpu']),GPU='8',Memory=site['memory'],SharedMemory=site['shared_memory']))],
        Envs=env,UserCommand=shlex.join(command),DataSources=site['data_sources'],
        UserVpc=site['user_vpc'],Settings={'EnableRDMA':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',default='configs/pai/stage12_3fps_32gpu.site.json')
    p.add_argument('--out',default='outputs/dlc_stage12_3fps_32gpu')
    p.add_argument('--validator',default='/root/.codex-chatgpt/skills/pai/scripts/validate_submission.py')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    site=json.loads(Path(args.site).read_text())
    request=build_request(site)
    spec=importlib.util.spec_from_file_location('pai_validate',args.validator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    errors=module.validate_request(request)
    if errors: raise ValueError('\n'.join(errors))
    # User explicitly requested queued submission without capacity polling.
    # Capacity is not a precondition: scheduler decides when 4 nodes are free.
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    dry=module.build_command(request,cli_dry_run=True)
    submit=module.build_command(request)
    (out/'dry_run.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(dry)+'\n')
    (out/'submit.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(submit)+'\n')
    print(f'Prepared {out}/request.json; 4 Pods x 8 GPU. No job submitted.')
    print('Queued submission; quota availability was not queried as requested. Verify image compatibility and CPU preparation before submit.sh.')
    if args.dry_run:subprocess.run(dry,check=True)


if __name__=='__main__':
    main()
