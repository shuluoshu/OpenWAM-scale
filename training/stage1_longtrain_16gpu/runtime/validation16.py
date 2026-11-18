"""Matched fixed64 native FM and20-step image/TCP validation; Chinese report."""
from __future__ import annotations
from collections import defaultdict
import html
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
import distributed16 as d


def _generation_helpers():
    spec=importlib.util.spec_from_file_location('longtrain_reviewed_generation',d.REVIEWED/'full_generation_eval.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def checked_row(directory,manifest,entry):
    directory=Path(directory);path=directory/entry['file']
    if d.sha256(path)!=manifest['files'][entry['file']]:raise ValueError('validation row checksum differs')
    row=torch.load(path,map_location='cpu',weights_only=False)
    context=Path(row['context_file']);context=context if context.is_absolute() else directory/context
    hashes={**manifest['files'],**manifest.get('external_files',{})}
    key=str(context) if str(context) in hashes else row['context_file']
    if key not in hashes or d.sha256(context)!=hashes[key]:raise ValueError('validation context checksum differs')
    return row

def save_image(path,prediction,uv,row,*,arch,index,update):
    from PIL import Image,ImageDraw
    helper=_generation_helpers();_,valid=helper.valid_targets(row);truth=row['target_uv'].float()
    def pil(value):return Image.fromarray(value[:,0].permute(1,2,0).contiguous().numpy())
    current=pil(row['raw_current_rgb'][:,:,:256]);target=pil(row['raw_target_rgb'][:,:,:256]);generated=pil((prediction.clamp(0,1)*255).round().byte())
    for image,coords in ((target,truth),(generated,uv)):
        draw=ImageDraw.Draw(image)
        for hand,color in ((0,'cyan'),(1,'magenta')):
            if valid[hand]:
                x,y=(coords[hand]*torch.tensor([319.,255.])).tolist();draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
    canvas=Image.new('RGB',(960,296),'white');draw=ImageDraw.Draw(canvas)
    draw.text((5,5),f'{arch} update{update} case{index}: current head | terminal head | native20-step prediction',fill='black')
    draw.text((5,21),str(row['record']['id']),fill='black')
    for x,image in ((0,current),(320,target),(640,generated)):canvas.paste(image,(x,40))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def _sum_records(records):
    total=defaultdict(float)
    for r in records:
        for k,v in r['sums'].items():total[k]+=v
    return dict(total)


def region_table(regions):
    labels = dict(full='全图', changed='变化区域', static='静态区域', tcp='GT TCP 邻域')
    rows = []
    def number(value):
        return '无有效区域' if value is None else f'{value:.5f}'
    for name, label in labels.items():
        item = regions[name]
        cells = [label, number(item['model_mae']), number(item['copy_mae']), number(item['gain_over_copy'])]
        rows.append('<tr>' + ''.join('<td>' + value + '</td>' for value in cells) + '</tr>')
    return ('<table><tr><th>区域</th><th>预测 MAE（0–1）↓</th><th>复制当前帧 MAE↓</th>'
            '<th>相对复制改进↑</th></tr>' + ''.join(rows) + '</table>')

@torch.no_grad()
def assess(model,arch,config,output,update,*,device,rank,world,vae=None,limit=None):
    """Use all64 for production; explicit smaller limit only for profile probe."""
    error=None;owned=False
    try:
        base=d.load_reviewed().base;helper=_generation_helpers()
        parent=Path(config.get('prepared_g','/mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_20261111/data'))
        manifest=json.loads((parent/'manifest.json').read_text());entries=manifest['records']['val']
        if len(entries)!=64:raise ValueError('fixed validation must contain64 records')
        child=Path(config.get('prepared_cstar',d.REVIEWED/'baseline_cstar/data'))
        cm=json.loads((child/'manifest.json').read_text()) if arch=='Cstar' else None
        if cm and [r['id'] for r in cm['records']['val']]!=[r['id'] for r in entries]:raise ValueError('fixed G/Cstar validation IDs differ')
        from openwam.model.goal.codec import GoalCodec,GoalCodecSpec
        from openwam.model.goal.dependencies import module_identity
        owned=vae is None
        if owned:vae=base.construct_component(manifest['config'],'vae',device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
        if module_identity(vae)!=manifest['dependencies']['vae']:raise ValueError('validation VAE identity differs')
        codec=GoalCodec(vae) if arch=='Cstar' else GoalCodec(vae,GoalCodecSpec(name='wan22_multiview_t1',image_shape=(3,1,384,320),latent_shape=(48,1,24,20)))
        output=Path(output);directory=output/('profile_validation' if limit else 'validation')/f'step_{update:09d}'
        directory.mkdir(parents=True,exist_ok=True);started=time.monotonic();records=[];native=[]
        count=min(len(entries),limit) if limit else len(entries)
        with d.preserved_rng():
            model.eval()
            for index in range(rank,count,world):
                d.atomic_json(Path(output)/'health'/f'rank_{rank:02d}.json',dict(rank=rank,phase='validation-case',update=update,index=index,unix_time=time.time()))
                entry=entries[index];row=checked_row(parent,manifest,entry)
                if arch=='G':
                    batch,_=base.collate_prepared([row],parent,device=device,dtype=torch.bfloat16)
                    sigma,noise=base.common_noise(batch['target_latents'],seed=42,cursor=index,validation=True)
                    native.append(dict(index=index,id=entry['id'],sums=base.evaluate_batch(model,batch,arm='default',sigma=sigma,noise=noise)['sums']))
                    prediction=base.generate_latents(model,batch,arm='default',seed=42,index=index,steps=20)
                    decoded=codec.decode(prediction['latents'])[0].float()
                    head=((decoded.clamp(-1,1)+1)/2)[:,:,:256].cpu();uv=prediction['uv'][0].float().cpu()
                else:
                    c=d.load_cstar();cr=checked_row(child,cm,cm['records']['val'][index])
                    helper.verify_cstar_record(row,cr);batch=c.collate([cr],child,device=device,dtype=torch.bfloat16)
                    native.append(dict(index=index,id=entry['id'],sums=c.evaluate_record(model,batch,index=index)))
                    prediction=model.generate(**{k:v for k,v in batch.items() if k!='targets'},num_steps=20,seed=42+index+1000000)
                    head=((codec.decode(prediction.latent)[0].float().clamp(-1,1)+1)/2).cpu();uv=prediction.uv[0].float().cpu()
                measured=helper.measure(head,uv,row)
                from openwam.train.goal_image_metrics import goal_image_metrics
                head_valid, point_valid = helper.valid_targets(row)
                region_errors = goal_image_metrics(head, row['raw_current_rgb'][:, :, :256],
                                                   row['raw_target_rgb'][:, :, :256],
                                                   image_valid=head_valid, target_uv=row['target_uv'],
                                                   point_valid=point_valid)
                record=dict(index=index,id=entry['id'],family=entry['family'],grain=entry['granularity'],arm=arch,update=update,steps=20,
                            instruction=row['instruction'],ground_truth_identity=helper.ground_truth_identity(row),generated_uv=uv.tolist(),
                            region_errors=region_errors, **measured)
                image=f'images/{index:04d}.png';save_image(directory/image,head,uv,row,arch=arch,index=index,update=update);record['image']=image
                d.atomic_json(directory/f'records/{index:04d}.json',record);records.append(record)
                del row,batch,prediction,head,uv
            if arch=='Cstar':model.train_stage_a()
            else:model.train()
    except Exception as exc:error=repr(exc)
    d.collective_status(error,device)
    gathered=[None]*world;dist.all_gather_object(gathered,dict(records=records,native=native))
    result=None
    error=None
    if rank==0:
        try:
            records=sorted([r for part in gathered for r in part['records']],key=lambda r:r['index'])
            native=sorted([r for part in gathered for r in part['native']],key=lambda r:r['index'])
            if len(records)!=count or [r['id'] for r in records]!=[e['id'] for e in entries[:count]]:raise ValueError('validation missing or duplicated IDs')
            aggregate=helper.aggregate(records);sums=_sum_records(native)
            from openwam.train.goal_image_metrics import aggregate_goal_image_metrics
            region_summary = aggregate_goal_image_metrics([r['region_errors'] for r in records])
            native_summary=base.summarize_sums(sums) if arch=='G' else d.load_cstar().summarize(sums)
            result=dict(format='openwam.stage1_longtrain.fixed_validation.v1',arch=arch,update=update,full64=limit is None,steps=20,seconds=time.monotonic()-started,
                        validation_scope='fixed64 fitting diagnostics; may overlap full training; NOT heldout',generated_protocol='native20step;headRGB0to1;TCPeuclidean-pixel319x255;G-clean-goalUV-Cstar-jointTCP',cases=records,
                        goal_horizon_mode=getattr(getattr(model, 'config', None), 'goal_horizon_mode', None),
                        region_errors=region_summary,
                        native_FM=dict(protocol='source-native-fixed-noise-objective-separate-between-architectures',**native_summary),**aggregate)
            d.atomic_json(directory/'manifest.json',result)
            d.append_jsonl(output/('profile_validation.jsonl' if limit else 'validation.jsonl'),{k:v for k,v in result.items() if k!='cases'})
            cards=''.join('<details><summary>'+html.escape(r['id'])+' · '+html.escape(r['instruction'])+'</summary><p>当前头图 / 真实目标头图 / 原生20步预测</p><img width="960" src="'+r['image']+'"></details>' for r in records)
            (directory/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><h1>'+arch+' 第'+str(update)+'次更新 · 固定64条拟合诊断</h1><p>固定样例可能与全量训练总体重叠。头图 MAE 单位0–1，TCP 为有效端点平均欧氏像素误差。变化区域为当前与目标平均通道差大于0.1，TCP区域为有效GT端点半径24像素的圆并集。相对复制改进为复制MAE减预测MAE。</p>'+region_table(region_summary)+cards+'</html>')
        except Exception as exc:error=repr(exc)
    d.collective_status(error,device)
    if owned:del codec,vae;torch.cuda.empty_cache()
    dist.barrier();return result


def _read_jsonl(path):
    if not path.exists():return []
    lines=path.read_text().splitlines();rows=[]
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
    return rows

def _read_scalar_history(path,max_rows=20000):
    fields=('update','samples','wall_seconds','seconds','samples_per_second','loss','fm_loss','uv_loss','tcp_loss')
    if not path.exists():return [],0,1
    rows=[];total=0;stride=1;latest=None
    with path.open() as source:
        for line in source:
            if not line.strip():continue
            try:value=json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith('\n'):break
                raise
            latest={k:value[k] for k in fields if k in value}
            if total%stride==0:rows.append(latest)
            total+=1
            if len(rows)>max_rows*2:rows=rows[::2];stride*=2
    while len(rows)>max_rows:rows=rows[::2];stride*=2
    if latest is not None and (not rows or rows[-1]!=latest):rows.append(latest)
    return rows,total,stride

def _atomic_text(path,value):
    import tempfile
    path=Path(path);fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(value);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def build_report(runs_root,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _build_report_locked(runs_root,output)

def _build_report_locked(runs_root,output):
    """Reuse audited Chinese visible-legend/hover chart view, leaving values raw."""
    root=Path(runs_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('longtrain_chinese_view',d.REVIEWED/'reports_zh/build_reports.py')
    view=importlib.util.module_from_spec(spec);spec.loader.exec_module(view)
    charts={};runs={};rows=[];links=[];region_sections=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/'contract.json').exists():continue
        contract=json.loads((directory/'contract.json').read_text());arch=contract.get('arch')
        if arch not in ('G','Cstar'):continue
        key='default' if arch=='G' else 'causal_cstar';train,total_train_rows,stride=_read_scalar_history(directory/'train.jsonl');validation=_read_jsonl(directory/'validation.jsonl')
        charts[key]=train;runs[arch]=dict(train_scalars=train,source_train_rows=total_train_rows,display_stride=stride,validation=validation,contract=contract)
        last=train[-1] if train else {};val=validation[-1] if validation else {};metrics=val.get('metrics',{})
        if val.get('region_errors'):
            region_sections.append('<section class="card"><h2>'+arch+' 第'+str(val['update'])+'步区域拟合诊断</h2>'+region_table(val['region_errors'])+'</section>')
        def number(x):return '待生成' if x is None else f'{x:.6g}'
        rows.append('<tr><td>'+('默认 G' if arch=='G' else '旧因果 C*')+'</td><td>'+str(last.get('update',0))+'</td><td>'+number(last.get('samples_per_second'))+'</td><td>'+number(metrics.get('head_image_mae'))+'</td><td>'+number(metrics.get('tcp_px'))+'</td><td>'+str(val.get('update','待生成'))+'</td></tr>')
        if validation:
            link=os.path.relpath(directory/'validation'/f'step_{val["update"]:09d}'/'index.html',output)
            links.append('<a href="'+html.escape(link)+'">'+arch+' 最新64条验证图像</a>')
    matched={}
    if all(arch in runs for arch in ('G','Cstar')):
        g,c=runs['G']['contract'],runs['Cstar']['contract']
        matched=dict(foundation=bool(g.get('foundation')) and g.get('foundation')==c.get('foundation'),seed=g.get('fresh_foundation_seed')==c.get('fresh_foundation_seed')==42,
                     global_batch=bool(g.get('execution',{}).get('global_batch_size')) and g.get('execution',{}).get('global_batch_size')==c.get('execution',{}).get('global_batch_size'),
                     data=bool(g.get('data',{}).get('fingerprint')) and g.get('data',{}).get('fingerprint')==c.get('data',{}).get('fingerprint'))
    evidence='两路契约一致：基础模型、种子42、全局batch与数据流。' if matched and all(matched.values()) else '跨架构对照契约尚未全部匹配；请查看metrics.json中的comparison_identity。'
    controls='<p id="term-note">G 与旧 C* 原生目标定义不同，分别绘图。</p><section class="card"><div class="controls"><label>横轴<select id="axis"><option value="update">训练更新步数</option><option value="samples">累计样本数</option><option value="wall_seconds">训练耗时（秒）</option></select></label><label>指标<select id="term"><option value="loss">原生总损失</option><option value="fm_loss">原生图像 FM</option><option value="auxiliary">G UV / C* TCP patch FM</option></select></label><label>平滑窗口<input id="smooth" type="number" min="1" value="25"></label><label><input id="raw-lines" type="checkbox">显示原始值</label></div></section>'
    panels=view.chart_panel('chart-g',['default'],'默认 G 原生训练目标','G 的 FM 与 UV 按全局整个优化更新的有效值和有效点归一化。')+view.chart_panel('chart-cstar',['causal_cstar'],'旧因果 C* 原生训练目标','C* 保留原生头图 FM 与 TCP patch FM。与 G 的损失定义不同，曲线分别绘制。')
    data=json.dumps(charts,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    script=(d.REVIEWED/'reports_zh/charts.js').read_text()
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage1 16卡长训</title><style>'+view.CSS+'</style><body><h1>Stage1 双架构16卡持续长训</h1><p>'+html.escape(evidence)+'每1000次更新保存、固定64条训练拟合反馈与原生20步生成；固定样例可能与训练总体重叠，不是独立测试集，收到停止标记后在优化器边界保存。</p><section class="card"><table><tr><th>架构</th><th>更新</th><th>样本/秒</th><th>头图 MAE（0–1）↓</th><th>TCP（像素）↓</th><th>验证更新</th></tr>'+''.join(rows)+'</table><p>'+ ' · '.join(links)+'</p></section>'+controls+panels+'<script id="chart-data" type="application/json">'+data+'</script><script>'+script+'</script></body></html>'
    page=page.replace(controls,''.join(region_sections)+controls)
    _atomic_text(output/'index.html',page);d.atomic_json(output/'metrics.json',dict(runs=runs,comparison_identity=matched,view_source_sha256=d.sha256(d.REVIEWED/'reports_zh/charts.js')))
    return dict(report=str(output/'index.html'),architectures=list(runs))
