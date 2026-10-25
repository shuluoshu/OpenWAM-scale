#!/usr/bin/env python3
"""Job-scoped OpenWAM metrics and 3-hour Feishu reporting; no job mutation."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone, timedelta
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.request

BJ=timezone(timedelta(hours=8))
TERMINAL={'Succeeded','Failed','Stopped','Deleted','Terminated'}
SKILL=Path('/root/.codex/skills/dlc-loss-monitor/scripts/monitor.py')


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));pending.replace(path)


def read_progress(path):
    """Stream the log, retaining a bounded curve plus its complete recent window."""
    if not Path(path).exists():return []
    points=[];recent=deque(maxlen=300);stride=1;seen=0
    with Path(path).open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('event')!='update':continue
            if not isinstance(r.get('step'),int):continue
            if seen%stride==0:points.append(r)
            recent.append(r);seen+=1
            if len(points)>4000:points=points[::2];stride*=2
    # Order is chronological, not sorted by step; a resume may rewind the curve.
    if not recent:return []
    cutoff=recent[0]['step']
    return [r for r in points if r['step']<cutoff]+list(recent)


def stage_summary(rows,*,total_hint,stage,now):
    latest=rows[-1] if rows else {};step=latest.get('step',0)
    total=latest.get('total_steps',total_hint)
    if stage=='stage1' and latest.get('total_samples'):
        total=math.ceil(latest['total_samples']/32)
    recent=rows[-100:];remaining=None
    # Only use a contiguous run segment; elapsed_s restarts after resume.
    for i in range(len(recent)-1,0,-1):
        if recent[i].get('elapsed_s',0)<=recent[i-1].get('elapsed_s',0) or recent[i]['step']<=recent[i-1]['step']:
            recent=recent[i:];break
    if len(recent)>=5 and total:
        elapsed=recent[-1].get('elapsed_s',0)-recent[0].get('elapsed_s',0)
        updates=recent[-1]['step']-recent[0]['step']
        if elapsed>0 and updates>0:remaining=max(total-step,0)*elapsed/updates
    stale=bool(latest.get('wall_time') and now-latest['wall_time']>3600)
    if stale:remaining=None
    return dict(step=step,total=total,percent=100*step/total if total else None,
        latest={k:v for k,v in latest.items() if 'loss' in k and isinstance(v,(float,int)) and math.isfinite(v)},
        remaining_s=remaining,eta_unix=now+remaining if remaining is not None else None,stale=stale,rows=rows)


def collect(job,root,*,now=None):
    now=time.time() if now is None else now;root=Path(root)
    state=read_json(root/'pipeline_state.json',{})
    first=read_progress(root/'stage1/progress.jsonl')
    candidates=list((root/'stage2').rglob('progress.jsonl')) if (root/'stage2').exists() else []
    latest=max(candidates,key=lambda p:p.stat().st_mtime) if candidates else None
    second=read_progress(latest) if latest else []
    one=stage_summary(first,total_hint=None,stage='stage1',now=now)
    two=stage_summary(second,total_hint=100000,stage='stage2',now=now)
    phase=state.get('phase','preparation_or_queue');status=job.get('Status','Unknown')
    eta=two['eta_unix'] if phase=='stage2' and status=='Running' else None
    if status!='Running':
        for s in (one,two):s['eta_unix']=s['remaining_s']=None
    return dict(job_id=job['JobId'],name=job.get('DisplayName',job['JobId']),status=status,phase=phase,
        url=f"https://pai.console.aliyun.com/?regionId=cn-shanghai&workspaceId={job.get('WorkspaceId','1140257')}#/dlc/jobs/{job['JobId']}/overview",
        stage1=one,stage2=two,eta_unix=eta,observed_at=now,
        reason=job.get('ReasonMessage') or '',pipeline_error=state.get('error',''))


def format_time(t):
    return datetime.fromtimestamp(t,BJ).strftime('%m-%d %H:%M 北京时间') if t is not None else '暂无法估计'


def message_text(s):
    queue=s['status'] in {'Queuing','Queueing','Pending','Created','Creating','Waiting'}
    lines=['OpenWAM 32 卡训练进度',f"名称：{s['name']}",f"DLC：{s['url']}",
        f"状态：{s['status']}"+('（排队中）' if queue else ''),f"阶段：{s['phase']}",
        '配置：4 节点 × 8 卡；Stage1 3 FPS 一轮 → 冻结 → Stage2 100000 步']
    for name,key in (('Stage1','stage1'),('Stage2','stage2')):
        stage=s[key];total=stage['total'];pct=stage['percent']
        lines.append(f"{name}：{stage['step']}/{total if total else '待索引完成'} 步"+(f'（{pct:.2f}%）' if pct is not None else ''))
        if stage['latest']:lines.append('最新 '+name+' loss：'+', '.join(f'{k}={v:.6g}' for k,v in stage['latest'].items()))
        else:lines.append(name+'：尚无 loss，不填充模拟值。')
        if s['phase']==key:lines.append('当前阶段预计完成：'+format_time(stage['eta_unix']))
        if stage['stale']:lines.append(name+' 超过 1 小时无更新，暂停外推 ETA。')
    lines.append('全部训练：已完成' if s['status']=='Succeeded' else '全部训练预计完成：'+format_time(s['eta_unix']))
    if s['eta_unix'] is None and s['status'] not in TERMINAL:lines.append('排队、初始化或 Stage2 吞吐尚未建立时，不推算整场完成时间。')
    prep=s.get('preparation')
    if prep:lines.append('CPU 数据准备：'+str(prep.get('status','unknown')))
    if s['reason']:lines.append('DLC 原因：'+s['reason'][:400])
    if s['pipeline_error']:lines.append('训练错误：'+s['pipeline_error'][:400])
    lines+=['loss 使用跨 rank 归约后的训练指标；ETA 按当前阶段最近更新耗时估算，包含该窗口内评估/保存耗时。',
        '更新时间：'+format_time(s['observed_at']),'每 3 小时推送；任务结束时补发结果并结束监控。']
    return '\n'.join(lines)


def loss_points(rows,metric):
    return [(r['step'],r[metric]) for r in rows if isinstance(r.get(metric),(float,int)) and math.isfinite(r[metric])]


def plot(s,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,key in zip(axes,('stage1','stage2')):
        rows=s[key]['rows'];keys=sorted({k for r in rows for k,v in r.items() if 'loss' in k and isinstance(v,(float,int))})
        for metric in keys:
            points=loss_points(rows,metric)
            if points:ax.plot(*zip(*points),label=metric,linewidth=.9)
        if keys:ax.legend(fontsize=7)
        else:ax.text(.5,.5,'No training metrics yet',ha='center',transform=ax.transAxes)
        ax.set_title(key+' | step '+str(s[key]['step']));ax.set_xlabel('Optimizer update');ax.set_ylabel('Loss');ax.grid(alpha=.25)
    fig.suptitle(s['name']+'\n'+s['job_id']+' | '+s['status']);fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def feishu_module():
    spec=importlib.util.spec_from_file_location('pai_feishu',SKILL);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def push(s,png):
    m=feishu_module();c=m.load_feishu_config(SKILL.parent)
    token=m.feishu_token(c['app_id'],c['app_secret'])
    req=urllib.request.Request('https://open.feishu.cn/open-apis/im/v1/chats/'+c['chat_id'],headers={'Authorization':'Bearer '+token})
    response=json.load(urllib.request.urlopen(req,timeout=30))
    if response.get('code')!=0 or response.get('data',{}).get('name')!='通知我！':
        raise RuntimeError('Configured Feishu destination is not the requested group 通知我！')
    image_key=m.feishu_upload_image(token,png)
    image=m.feishu_send(token,c['chat_id'],'image',{'image_key':image_key})
    text=m.feishu_send(token,c['chat_id'],'text',{'text':message_text(s)})
    return dict(image_message_id=image.get('data',{}).get('message_id'),text_message_id=text.get('data',{}).get('message_id'))


def get_job(job):
    query='{JobId:JobId,DisplayName:DisplayName,Status:Status,WorkspaceId:WorkspaceId,ReasonMessage:ReasonMessage}'
    result=subprocess.run(['aliyun','pai-dlc','get-job','--region','cn-shanghai','--job-id',job,'--cli-query',query],capture_output=True,text=True,check=True,timeout=120)
    return json.loads(result.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preparation',type=Path);p.add_argument('--interval',type=int,default=10800)
    p.add_argument('--watch',action='store_true');p.add_argument('--no-feishu',action='store_true')
    args=p.parse_args()
    if args.interval<60:raise ValueError('interval must be >=60 seconds')
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=read_json(args.out/'state.json',{})
            if state.get('terminal_sent'):return 0
            try:
                job=get_job(args.job);now=time.time()
                terminal=job['Status'] in TERMINAL
                due=now-state.get('last_sent',0)>=args.interval or terminal
                if due or not args.watch:
                    s=collect(job,args.root,now=now)
                    if args.preparation:s['preparation']=read_json(args.preparation,{})
                    png=args.out/'loss_overview.png';plot(s,png);write_json(args.out/'summary.json',s)
                    if not args.no_feishu:
                        receipt=push(s,png)
                        write_json(args.out/'state.json',dict(last_sent=now,terminal_sent=terminal,job_id=args.job,interval=args.interval,**receipt))
                        print(json.dumps(dict(event='feishu_sent',job=args.job,status=job['Status'],next_due=format_time(now+args.interval)),ensure_ascii=False),flush=True)
                    else:print(message_text(s),flush=True)
                if terminal or not args.watch:return 0
            except Exception as exc:
                # Do not print API/credential payloads on failure.
                print(json.dumps(dict(event='monitor_error',kind=type(exc).__name__,at=time.time())),flush=True)
                if not args.watch:raise
            time.sleep(60)


if __name__=='__main__':raise SystemExit(main())
