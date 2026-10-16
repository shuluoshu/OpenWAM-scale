from pathlib import Path
import json,av
out=Path(__file__).resolve().parent;rows=json.loads((out/'manifest.json').read_text())['results'];results=[]
for r in rows:
 info=json.loads((Path(r['source'])/'meta/info.json').read_text());meta=json.loads((out/'clips'/r['id']/'source_episode.json').read_text());cams=[]
 for cam in [r['head'],r['left'],r['right']]:
  if not cam:continue
  prefix=f'videos/{cam}/';path=Path(r['source'])/info['video_path'].format(video_key=cam,chunk_index=meta[prefix+'chunk_index'],file_index=meta[prefix+'file_index'])
  try:
   with av.open(str(path)) as c:
    s=c.streams.video[0];cams.append({'camera':cam,'average_rate':str(s.average_rate),'base_rate':str(s.base_rate),'time_base':str(s.time_base),'frames':s.frames})
  except Exception as e:cams.append({'camera':cam,'error':f'{type(e).__name__}: {e}'})
 results.append({'id':r['id'],'cameras':cams})
(out/'frequency_audit_videos.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
errors=[(r['id'],c['error']) for r in results for c in r['cameras'] if 'error' in c];rates=sorted({c['average_rate'] for r in results for c in r['cameras'] if 'average_rate' in c})
print(json.dumps({'clip_count':len(results),'camera_count':sum(len(r['cameras']) for r in results),'average_rates':rates,'errors':errors},ensure_ascii=False))
if errors:raise SystemExit(1)
