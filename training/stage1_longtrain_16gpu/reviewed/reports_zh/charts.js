(() => {
  'use strict';
  const source = JSON.parse(document.getElementById('chart-data').textContent);
  const palette = {default:'#1762c4',no_history:'#c2630b',no_g:'#15825d',causal_cstar:'#8741bd'};
  const labels = {default:'默认 G',no_history:'无历史',no_g:'屏蔽 G→图像路径',causal_cstar:'旧因果 C*'};
  const dash = {default:[],no_history:[9,5],no_g:[3,4],causal_cstar:[]};
  const settings = {axis:'update',term:'loss',window:25,raw:false};
  const panels = [...document.querySelectorAll('.chart-panel')].map(el => ({el,canvas:el.querySelector('canvas'),names:el.dataset.arms.split(','),enabled:new Set(el.dataset.arms.split(',')),view:null,pointer:null}));
  function key(arm) {return settings.term==='auxiliary'?(arm==='causal_cstar'?'tcp_loss':'uv_loss'):settings.term;}
  function points(arm) {
    const rows=source[arm]||[], k=key(arm), values=[];let sum=0;
    rows.forEach((r,i)=>{
      sum+=r[k];if(i>=settings.window)sum-=rows[i-settings.window][k];
      values.push({x:r[settings.axis],raw:r[k],y:sum/Math.min(i+1,settings.window),step:r.update});
    });return values;
  }
  const number=v=>Math.abs(v)<.001&&v!==0?v.toExponential(2):v.toLocaleString('zh-CN',{maximumFractionDigits:4});
  function nearest(rows,x){let lo=0,hi=rows.length-1;while(lo<hi){const m=Math.floor((lo+hi)/2);if(rows[m].x<x)lo=m+1;else hi=m;}return lo>0&&Math.abs(rows[lo-1].x-x)<Math.abs(rows[lo].x-x)?rows[lo-1]:rows[lo];}
  function draw(p){
    const width=p.canvas.getBoundingClientRect().width,height=350,dpr=window.devicePixelRatio||1;
    p.canvas.width=Math.round(width*dpr);p.canvas.height=height*dpr;
    const g=p.canvas.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,width,height);
    const box={x:72,y:24,w:Math.max(30,width-98),h:266};
    const series=p.names.filter(a=>p.enabled.has(a)).map(a=>({arm:a,rows:points(a)})).filter(s=>s.rows.length);
    if(!series.length){g.fillStyle='#526173';g.font='15px sans-serif';g.fillText('请在图例中选择至少一组实验',35,100);p.view=null;return;}
    const all=series.flatMap(s=>s.rows),xmin=Math.min(...all.map(r=>r.x)),xmax=Math.max(...all.map(r=>r.x));
    const top=Math.max(...all.map(r=>settings.raw?Math.max(r.y,r.raw):r.y));const ymax=Math.max(top*1.1,1e-8);
    const xx=v=>box.x+(v-xmin)/(xmax-xmin||1)*box.w, yy=v=>box.y+box.h-v/ymax*box.h;
    g.font='12px sans-serif';g.textBaseline='middle';
    for(let i=0;i<=5;i++){
      const y=box.y+box.h-i/5*box.h;g.strokeStyle='#e7edf5';g.lineWidth=1;g.setLineDash([]);g.beginPath();g.moveTo(box.x,y);g.lineTo(box.x+box.w,y);g.stroke();
      g.fillStyle='#59697d';g.textAlign='right';g.fillText(number(ymax*i/5),box.x-10,y);
      const x=box.x+i/5*box.w,tick=xmin+(xmax-xmin)*i/5;g.textAlign='center';g.fillText(number(settings.axis==='wall_seconds'?tick:Math.round(tick)),x,box.y+box.h+22);
    }
    g.textAlign='left';g.fillStyle='#526173';g.fillText('损失值',box.x,10);
    g.textAlign='center';g.fillText({update:'训练更新步数',samples:'累计训练样本数',wall_seconds:'当前执行阶段耗时（秒，含评估与保存）'}[settings.axis],box.x+box.w/2,338);
    function line(s,field,alpha,lw){g.save();g.strokeStyle=palette[s.arm];g.globalAlpha=alpha;g.lineWidth=lw;g.setLineDash(dash[s.arm]);g.beginPath();s.rows.forEach((r,i)=>i?g.lineTo(xx(r.x),yy(r[field])):g.moveTo(xx(r.x),yy(r[field])));g.stroke();g.restore();}
    series.forEach(s=>{if(settings.raw&&settings.window>1)line(s,'raw',.2,1);line(s,'y',1,2.7);});
    p.view={series,box,xmin,xmax,xx,yy};
    if(p.pointer!==null&&p.pointer>=box.x&&p.pointer<=box.x+box.w){
      const x=xmin+(p.pointer-box.x)/box.w*(xmax-xmin);g.setLineDash([4,4]);g.strokeStyle='#64758d';g.lineWidth=1;g.beginPath();g.moveTo(p.pointer,box.y);g.lineTo(p.pointer,box.y+box.h);g.stroke();g.setLineDash([]);
      const rows=series.map(s=>({arm:s.arm,...nearest(s.rows,x)}));
      rows.forEach(r=>{g.fillStyle=palette[r.arm];g.beginPath();g.arc(xx(r.x),yy(r.y),4,0,Math.PI*2);g.fill();});
      const tooltip=p.el.querySelector('.chart-tooltip');
      tooltip.innerHTML='<strong>当前位置读数</strong><table><tr><th>实验</th><th>步数</th><th>原始值</th><th>'+settings.window+'步均值</th></tr>'+rows.map(r=>'<tr><td><i style="background:'+palette[r.arm]+'"></i>'+labels[r.arm]+'</td><td>'+r.step+'</td><td>'+number(r.raw)+'</td><td>'+number(r.y)+'</td></tr>').join('')+'</table>';
      tooltip.hidden=false;
    }
  }
  panels.forEach(p=>{
    p.el.querySelectorAll('.legend-item').forEach(b=>b.addEventListener('click',()=>{const a=b.dataset.arm;p.enabled.has(a)?p.enabled.delete(a):p.enabled.add(a);b.setAttribute('aria-pressed',String(p.enabled.has(a)));draw(p);}));
    p.canvas.addEventListener('pointermove',e=>{p.pointer=e.clientX-p.canvas.getBoundingClientRect().left;draw(p);});
    p.canvas.addEventListener('pointerleave',()=>{p.pointer=null;p.el.querySelector('.chart-tooltip').hidden=true;draw(p);});
    new ResizeObserver(()=>draw(p)).observe(p.canvas);
  });
  const redraw=()=>panels.forEach(p=>{p.pointer=null;p.el.querySelector('.chart-tooltip').hidden=true;draw(p);});
  document.getElementById('axis').addEventListener('change',e=>{settings.axis=e.target.value;redraw();});
  document.getElementById('term').addEventListener('change',e=>{
    settings.term=e.target.value;
    document.getElementById('term-note').textContent={loss:'G：图像 FM＋UV 回归；旧 C*：0.5×头图 FM＋TCP patch FM。两类模型的绝对 loss 不直接比较。',fm_loss:'G 的 FM 覆盖多视图目标；旧 C* 的 FM 覆盖头图并使用原生时间步权重。两图分别缩放。',auxiliary:'G 的辅助项是二维 UV 回归；旧 C* 的辅助项是 192 维 TCP patch 流匹配。定义不同，分别观察趋势。'}[settings.term];redraw();
  });
  document.getElementById('smooth').addEventListener('change',e=>{settings.window=Number(e.target.value);redraw();});
  document.getElementById('raw-lines').addEventListener('change',e=>{settings.raw=e.target.checked;redraw();});
  redraw();
})();
