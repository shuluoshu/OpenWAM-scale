#!/usr/bin/env python3
"""Render a 4-node 32-GPU DLC request; never submits a job.

Site values must be explicitly resolved. Uses the installed PAI skill validator
and optionally executes the CLI's serialization-only dry run.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def runtime_command(site, command):
    """Unpack a trusted, SHA-pinned conda archive into its rewritten prefix."""
    archive=site.get('runtime_archive')
    if archive is None:return shlex.join(command)
    if (not isinstance(archive,dict) or set(archive)!={'path','sha256','prefix'}
        or any(not isinstance(archive.get(k),str) or not archive[k] for k in ('path','sha256','prefix'))):
        raise ValueError('runtime_archive requires path, sha256, prefix strings')
    path,prefix,checksum=archive['path'],archive['prefix'],archive['sha256']
    if (not Path(path).is_absolute() or not Path(prefix).is_absolute()
        or Path(prefix)==Path('/') or not re.fullmatch('[0-9a-fA-F]{64}',checksum)):
        raise ValueError('runtime_archive requires absolute paths and a pinned SHA256')
    if site['python']!=str(Path(prefix)/'bin/python'):
        raise ValueError('runtime_archive prefix must match site.python exactly')
    # Hash stdin so even filenames with spaces/newlines cannot alter checksum
    # parsing. Every substituted path is shell-quoted independently.
    check=(f'[ "$(sha256sum < {shlex.quote(path)} | cut -d " " -f 1)" = {shlex.quote(checksum.lower())} ]'
           ' || { echo "Runtime archive SHA256 mismatch or unreadable archive" >&2; exit 1; }')
    return '\n'.join(['set -eu',check,shlex.join(['mkdir','-p','--',prefix]),
        shlex.join(['tar','-xf',path,'-C',prefix]),shlex.join(command)])


def hardware_preflight(site):
    """Shared lightweight checks run before either training startup path."""
    minimum_memory=site.get('expected_min_gpu_memory_gib')
    if minimum_memory is None:return []
    return [
        'echo "GPU inventory (name, total MiB):"',
        'gpu_inventory="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"',
        'printf "%s\\n" "$gpu_inventory"',
        'printf "%s\\n" "$gpu_inventory" | awk -F, '
        f'-v required_mib={minimum_memory*1024:g} '
        "'{ memory=$NF; gsub(/^[[:space:]]+|[[:space:]]+$/, \"\", memory); "
        'if (NF<2 || memory !~ /^[0-9]+([.][0-9]+)?$/ || memory+0<required_mib) bad=1 } '
        'END { if (NR!=8) bad=1; exit bad }\' || '
        '{ echo "GPU memory requirement failed: need 8 GPUs each with at least '
        f'{minimum_memory:g} GiB" >&2; exit 1; }}',
    ]


def build_request(site):
    required=('workspace_id','quota_id','image','cpu','memory','shared_memory','python','code_dir','commit')
    missing=[k for k in required if not site.get(k)]
    if missing:
        raise ValueError('Resolve site fields before rendering: '+', '.join(missing))
    if site.get('pod_count')!=4 or site.get('gpus_per_pod')!=8:
        raise ValueError('This training entry supports exactly 4 Pods x 8 local GPUs')
    if not site.get('data_sources') or not site.get('user_vpc'):
        raise ValueError('Resolve data_sources and user_vpc')
    priority=site.get('priority',1)
    if type(priority) is not int or not 1<=priority<=9:raise ValueError('priority must be an integer from 1 to 9')
    startup_mode=site.get('startup_mode','hold')
    if startup_mode not in ('hold','train'):
        raise ValueError('startup_mode must be hold or train')
    minimum_memory=site.get('expected_min_gpu_memory_gib')
    if minimum_memory is not None and (type(minimum_memory) not in (int,float)
            or not math.isfinite(minimum_memory) or minimum_memory<=0):
        raise ValueError('expected_min_gpu_memory_gib must be a positive finite number')
    env={'OPENWAM_CODE_DIR':site['code_dir'], 'OPENWAM_PYTHON':site['python'],
         'OPENWAM_EXPECTED_COMMIT':(subprocess.check_output(['git','-C',site['code_dir'],'rev-parse','HEAD'],text=True).strip() if site['commit']=='auto' else site['commit']),
         'STAGE12_CONFIG':site.get('pipeline_config','configs/goal/stage12_query_history_posttrain_3fps_4x8gpu.yaml')}
    if site.get('prepared_marker'):
        env['STAGE12_PREPARED_MARKER']=site['prepared_marker']
    for key in ('TMPDIR','XDG_CACHE_HOME','HF_HOME','TORCH_HOME','TRITON_CACHE_DIR'):
        if site.get('cache_env',{}).get(key):env[key]=site['cache_env'][key]
    command=['bash', str(Path(site['code_dir'])/'scripts/pai/stage12_entry.sh'), 'train']
    if site.get('resume'):
        command.append('--resume')
    if site.get('stage2_resume'):
        if not site.get('resume'): raise ValueError('stage2_resume requires resume=true')
        command.extend(['--stage2-resume',site['stage2_resume']])
    training_command=runtime_command(site,command)
    preflight=hardware_preflight(site)
    if preflight:
        training_command='\n'.join(['set -euo pipefail',*preflight,training_command])
    return dict(Region=site.get('region','cn-shanghai'),WorkspaceId=site['workspace_id'],
        ResourceId=site['quota_id'],DisplayName=site['display_name'],JobType='PyTorchJob',
        Accessibility='PRIVATE',Priority=priority,
        JobSpecs=[dict(Type='Worker',PodCount=4,Image=site['image'],ResourceConfig=dict(
            CPU=str(site['cpu']),GPU='8',Memory=site['memory'],SharedMemory=site['shared_memory']))],
        Envs=env,UserCommand=(training_command if startup_mode=='train' else 'sleep 36000000'),DataSources=site['data_sources'],
        UserVpc=site['user_vpc'],Settings={'EnableRDMA':True})


def training_launcher(site, env):
    """Render an explicit, foreground launch with a node-local lifetime lock."""
    command=['bash',str(Path(site['code_dir'])/'scripts/pai/stage12_entry.sh'),'train']
    if site.get('resume'):command.append('--resume')
    if site.get('stage2_resume'):command.extend(['--stage2-resume',site['stage2_resume']])
    runtime=runtime_command(site,command)
    identity=json.dumps([site['code_dir'],env['STAGE12_CONFIG']],sort_keys=True).encode()
    key=hashlib.sha256(identity).hexdigest()[:20]
    log_dir=str(Path(site['code_dir'])/'outputs/dlc_launch')
    lines=['#!/usr/bin/env bash','set -euo pipefail','umask 077']
    lines.extend(f'export {key}={shlex.quote(str(value))}' for key,value in env.items())
    lines.extend([
        ': "${STAGE12_PREPARED_MARKER:?Set the prepared marker before launching training}"',
        f'exec 9> /tmp/openwam-stage12-{key}.lock',
        'flock -n 9 || { echo "Training already running on this pod for this code/pipeline" >&2; exit 75; }',
        f'log_dir={shlex.quote(log_dir)}',
        'mkdir -p -- "$log_dir"',
        'log_file="$log_dir/$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"',
        ': > "$log_file"',
        'echo "Training log: $log_file"',
        'exec > >(tee -a "$log_file") 2>&1',
        "trap 'status=$?; echo \"Training finished with exit status $status\"; exit \"$status\"' EXIT",
        '[[ -f "$STAGE12_PREPARED_MARKER" ]] || { echo "Prepared marker missing: $STAGE12_PREPARED_MARKER" >&2; exit 1; }',
    ])
    lines.extend(hardware_preflight(site))
    # Run in this foreground shell: it owns fd 9 through runtime extraction
    # and the training child's complete lifetime. flock releases on exit.
    lines.append(runtime)
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',default='configs/pai/stage12_3fps_32gpu.site.json')
    p.add_argument('--out',default='outputs/dlc_stage12_3fps_32gpu')
    p.add_argument('--validator',default='/root/.codex-chatgpt/skills/pai/scripts/validate_submission.py')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    site=json.loads(Path(args.site).read_text())
    if not site.get('prepared_marker'):
        raise ValueError('prepared_marker path is required; preparation must complete before explicit training launch')
    pipeline=Path(site.get('pipeline_config','configs/goal/stage12_query_history_posttrain_3fps_4x8gpu.yaml'))
    if not pipeline.is_absolute():pipeline=Path(site['code_dir'])/pipeline
    request=build_request(site)
    launcher=training_launcher(site,request['Envs'])
    startup_mode=site.get('startup_mode','hold')
    if startup_mode=='train':
        from scripts.pai.preparation_gate import require_pipeline_prepared
        require_pipeline_prepared(site['prepared_marker'],pipeline)
    spec=importlib.util.spec_from_file_location('pai_validate',args.validator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    errors=module.validate_request(request)
    if errors: raise ValueError('\n'.join(errors))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    dry=module.build_command(request,cli_dry_run=True)
    submit=module.build_command(request)
    (out/'dry_run.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(dry)+'\n')
    # Legacy auto-training checks readiness on the submitting host. Hold mode
    # defers readiness checks and runtime extraction to the explicit launcher.
    host_python=sys.executable if site.get('runtime_archive') is not None else site['python']
    check=[host_python,str(Path(site['code_dir'])/'scripts/pai/preparation_gate.py'),
           '--check','--marker',site['prepared_marker'],'--config',str(pipeline)]
    preflight=shlex.join(check)+'\n' if startup_mode=='train' else ''
    (out/'submit.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+preflight+shlex.join(submit)+'\n')
    (out/'launch_train.sh').write_text(launcher)
    (out/'launch_train.sh').chmod(0o755)
    (out/'LAUNCH.md').write_text(
        '# Explicit per-pod training launch\n\n'
        f'Startup mode: `{startup_mode}`. In hold mode the scheduled command is exactly `sleep 36000000`; '
        'rendering and submitting do not extract the runtime or start training.\n\n'
        'Copy this submission output directory to a path mounted in the DLC containers. '
        'After preparation completes, open a terminal on each of the 4 worker pods and run the same launcher. '
        'DLC supplies the rank/rendezvous environment; preserve it in each terminal. '
        'All four launches are required for distributed training.\n\n'
        '```bash\ncd /shared/path/to/submission-output\nbash ./launch_train.sh\n```\n\n'
        'For a separate detached invocation in each pod terminal:\n\n'
        '```bash\nnohup bash ./launch_train.sh > /tmp/openwam-launch-terminal.log 2>&1 < /dev/null &\n'
        'tail -f /tmp/openwam-launch-terminal.log\n```\n\n'
        f'Training logs are written under `{str(Path(site["code_dir"])/"outputs/dlc_launch")}` '
        'with hostname, UTC timestamp and process ID. The launcher records the child exit status and returns it. '
        'Duplicate starts on the same pod/code/pipeline exit 75. The node-local flock is held until training exits; '
        'an existing unlocked lock file does not prevent a retry. '
        'The sleep reservation remains running after training exits; manage job lifetime separately.\n')
    print(f'Prepared {out}/request.json; 4 Pods x 8 GPU. No job submitted.')
    print(f'Startup mode: {startup_mode}. Explicit training launcher: {out}/launch_train.sh; see {out}/LAUNCH.md.')
    print('Request rendering does not establish quota availability; the scheduler determines placement. Complete CPU preparation before starting training.')
    if args.dry_run:subprocess.run(dry,check=True)


if __name__=='__main__':
    main()
