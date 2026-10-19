import json
import shutil
import subprocess

import pytest


def test_history_is_per_dataset_chronological_and_visible(tmp_path):
    from openwam.train.goal_history import render_goal_history
    cards = [dict(id='case_0000', family='egodex', valid_points=[True, True]),
             dict(id='case_0001', family='missing', valid_points=[False, False])]
    summary = dict(status='running', completed_steps=1600, steps=5000, eval_every=500,
                   evaluations=[dict(step=s, directory=f'eval_step_{s:06d}') for s in (0, 500, 1000, 1500)])
    render_goal_history(tmp_path, summary, cards)
    page = (tmp_path/'timeline.html').read_text()
    first = page.split('data-family="egodex"')[1].split('</article>')[0]
    assert '<details' not in first
    positions = [first.index(f'src="eval_step_{s:06d}/0000_00_generated.png"')
                 for s in (0, 500, 1000, 1500)]
    assert positions == sorted(positions)
    assert 'inputs/case_0000_target_points.png' in first
    second = page.split('data-family="missing"')[1].split('</article>')[0]
    assert 'src="eval_step_000500/0001_00_generated_clean.png"' in second
    assert 'src="eval_step_000500/0001_00_generated.png"' not in second
    assert 'appendChild' in page and 'fetch(' in page
    assert 'http-equiv="refresh"' not in page


def test_history_escapes_family_and_embedded_state(tmp_path):
    from openwam.train.goal_history import render_goal_history
    family = '</script><img src=x onerror=alert(1)>'
    render_goal_history(tmp_path, dict(status='running', completed_steps=0, steps=5000,
        evaluations=[]), [dict(id='case_0000', family=family, valid_points=[True, True])])
    page = (tmp_path/'timeline.html').read_text()
    assert family not in page
    assert '\\u003c/script>' in page


def test_live_updates_append_without_replacing_previous_panels(tmp_path):
    if not shutil.which('node'):
        pytest.skip('node required for live history behavior check')
    from openwam.train.goal_history import render_goal_history
    summary = dict(status='running', steps=5000, completed_steps=500,
        evaluations=[dict(step=s,directory=f'eval_step_{s:06d}') for s in (0,500)])
    cards = [dict(id='case_0000',family='valid',valid_points=[True,True]),
             dict(id='case_0001',family='missing',valid_points=[False,False])]
    render_goal_history(tmp_path, summary, cards)
    page=(tmp_path/'timeline.html').read_text()
    runtime=page.rsplit('<script>',1)[1].split('</script>',1)[0]
    state=json.dumps(dict(summary=summary,cards=cards))
    harness=r'''
const assert=require('node:assert/strict');
class Element {
 constructor(){this.children=[];this.dataset={};this.checked=true;}
 appendChild(child){this.children.push(child);return child;}
 addEventListener(){}
}
const initialData=INITIAL;
const stripsMock=[new Element(),new Element()];
const original=stripsMock.map(strip=>{const a={step:0},b={step:500};strip.children.push(a,b);return [a,b];});
const ids={'initial-state':{textContent:JSON.stringify(initialData)},'live-status':new Element(),sync:new Element(),check:new Element()};
const document={getElementById:id=>ids[id],createElement:()=>new Element(),
 querySelectorAll:()=>stripsMock,querySelector:selector=>stripsMock[Number(selector.match(/data-index="(\d+)"/)[1])]};
const location={protocol:'https:'};const setInterval=()=>{};const requestAnimationFrame=callback=>callback();
let latest=JSON.parse(JSON.stringify(initialData.summary));
let fail=false;
const fetch=async url=>{if(fail)throw new Error('offline');return {ok:true,json:async()=>url.endsWith('/metrics.json')?{review:[{source:'valid',image_mae_uint8:2.5},{source:'missing',image_mae_uint8:8.25}]}:latest};};
RUNTIME
(async()=>{
 latest.evaluations.push({step:1000,directory:'eval_step_001000',point_metrics:{'source:valid':{left:{count:1,pixel_error:3},right:{count:1,pixel_error:7}}}});
 await update();await update();
 stripsMock.forEach((strip,i)=>{assert.equal(strip.children.length,3);assert.equal(strip.children[0],original[i][0]);assert.equal(strip.children[1],original[i][1]);assert.equal(strip.children[2].dataset.step,'1000');});
 assert.equal(stripsMock[1].children[2].children[1].src,'eval_step_001000/0001_00_generated_clean.png');
 assert.equal(stripsMock[0].children[2].children[2].textContent,'图像 MAE：2.50 / 255');
 assert.equal(stripsMock[0].children[2].children[3].textContent,'TCP 平均误差：5.00 px');
 assert.equal(stripsMock[1].children[2].children[2].textContent,'图像 MAE：8.25 / 255');
 assert.equal(stripsMock[1].children[2].children[3].textContent,'TCP 平均误差：—（无有效 GT）');
 latest.evaluations.push({step:1500,directory:'eval_step_001500'});await update();
 stripsMock.forEach(strip=>{assert.equal(strip.children.length,4);assert.equal(strip.children[3].dataset.step,'1500');});
 fail=true;await update();stripsMock.forEach(strip=>assert.equal(strip.children.length,4));
 console.log('append, deduplication, original-node preservation, missing-GT policy and fetch-failure retention passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''.replace('INITIAL',state).replace('RUNTIME',runtime)
    script=tmp_path/'history_behavior.cjs';script.write_text(harness)
    result=subprocess.run(['node',str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_history_per_family_image_mae_and_weighted_tcp_mean(tmp_path):
    from openwam.train.goal_history import render_goal_history
    metrics={"source:valid":{"left":{"count":2,"pixel_error":3.},"right":{"count":1,"pixel_error":9.}},
             "source:missing":{"left":{"count":0,"pixel_error":None},"right":{"count":0,"pixel_error":None}}}
    directory=tmp_path/'eval_step_000500';directory.mkdir()
    (directory/'metrics.json').write_text(json.dumps({"review":[
        {"source":"valid","image_mae_uint8":2.345}, {"source":"missing","image_mae_uint8":7.891}]}))
    render_goal_history(tmp_path,dict(status='running',steps=5000,completed_steps=500,
        evaluations=[dict(step=500,directory=directory.name,point_metrics=metrics,image_mae_uint8=99.)]),
        [dict(id='case_0000',family='valid',valid_points=[True,True]),
         dict(id='case_0001',family='missing',valid_points=[False,False])])
    page=(tmp_path/'timeline.html').read_text()
    valid=page.split('data-family="valid"')[1].split('</article>')[0]
    missing=page.split('data-family="missing"')[1].split('</article>')[0]
    assert '图像 MAE：2.35 / 255' in valid
    assert 'TCP 平均误差：5.00 px' in valid
    assert '图像 MAE：7.89 / 255' in missing
    assert 'TCP 平均误差：—（无有效 GT）' in missing
    assert '99.00' not in valid
