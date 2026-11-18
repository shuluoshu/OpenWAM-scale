"""Real native CPU digest + renderer regression for every frozen root."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy,json,sqlite3,threading,time
from pathlib import Path
import data_pipeline as dp
from frozen_metadata_view import GENERATION_ROOT,load_manifest
from metadata_generation_v2 import reader
from openwam.dataloader.goal_epoch import _episode_digest

manifest=load_manifest(GENERATION_ROOT/'manifest.json');d=reader(GENERATION_ROOT/'manifest.json')
old=sqlite3.connect(f'{dp.DEFAULT_ORIGINAL_INDEX.as_uri()}?mode=ro',uri=True)
scope={v['repo_id'] for v in manifest['sources'].values()};requests=[];expected={}
for ri in sorted(scope):
 episodes=old.execute('SELECT episode,digest FROM episodes WHERE repo_id=? ORDER BY episode',(ri,)).fetchall()
 for episode,digest in (episodes[0],episodes[-1]):requests.append((ri,episode,digest))
first={}
for ri,episode,start in old.execute('SELECT repo_id,episode,sample_start FROM runs ORDER BY sample_start'):
 if ri in scope and ri not in first:first[ri]=start
render_rows=[old.execute('SELECT * FROM runs WHERE sample_start=?',(first[ri],)).fetchone() for ri in sorted(scope)]
local=threading.local();t=time.perf_counter()
def dataset():
 if not hasattr(local,'dataset'):
  local.dataset=copy.copy(d);local.dataset._episodes=OrderedDict();local.dataset._evaluations=None
 return local.dataset
def probe(request):
 ri,episode,digest=request;ds=dataset();repo=ds.repos[ri];entry=manifest['sources'][repo['repo']];m=ds._episode(repo,episode)
 if _episode_digest(m)!=digest:raise ValueError('representative original native digest differs '+str((ri,episode)))
 if str(m['root'])!=entry['physical_root'] or str(m['metadata_root'])!=entry['view_root'] or m['cameras']!=entry['cameras']:raise ValueError('metadata/physical camera paths differ')
 return dict(repo_id=ri,episode=episode,digest_equal_original=True,physical_root=str(m['root']),metadata_root=str(m['metadata_root']),cameras=m['cameras'],physical_data=str(m['data']))
def render(row):
 start,count,ri,episode,level,segment,grid,num,den=row;ds=dataset();repo=ds.repos[ri];entry=manifest['sources'][repo['repo']];m=ds._episode(repo,episode);segment=json.loads(segment)
 record,batch=ds._materialize(repo,episode,m,segment,grid*num//den,segment['end_exclusive']-1,level)
 if record['source_repo']!=entry['physical_root'] or record['head']!=entry['cameras'][0]:raise ValueError('renderer escaped original camera/source generation')
 return dict(repo_id=ri,episode=episode,id=record['id'],head=record['head'],source_repo=record['source_repo'],source_video=record['provenance']['source_video'],
  observed_shape=list(batch['observed'].shape),target_shape=list(batch['target_image'].shape),target_image_valid=bool(batch['target_image_valid'][0]),
  actual_native_rgb_provenance=record['provenance']['goal_video'])
with ThreadPoolExecutor(4) as pool:probes=list(pool.map(probe,requests))
with ThreadPoolExecutor(4) as pool:renders=list(pool.map(render,render_rows))
report=dict(format='openwam.longtrain.metadata-view-regression.v2',manifest_sha256=dp.sha256_file(GENERATION_ROOT/'manifest.json'),
 native_digest_probes=probes,native_render_probes=renders,all_frozen_roots_tested=len(scope),elapsed_seconds=time.perf_counter()-t)
dp.atomic_json(GENERATION_ROOT/'native_probe.json',report);print(dp.canonical(dict(roots=len(scope),digest_probes=len(probes),real_native_render_probes=len(renders),elapsed=report['elapsed_seconds'])))
