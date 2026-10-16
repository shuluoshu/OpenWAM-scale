from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import urllib.request,urllib.error,hashlib,json
root=Path(__file__).resolve().parent;p=root/'index.html'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[];self.ids=[];self.videos=0
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag=='video':self.videos+=1
  if 'id' in attrs:self.ids.append(attrs['id'])
  for key in ['src','href','poster']:
   if attrs.get(key):self.urls.append(attrs[key])
x=Links();x.feed(p.read_text());bad=[]
for u in x.urls:
 s=urlsplit(u)
 if s.scheme:continue
 if s.path and not (root/unquote(s.path)).is_file():bad.append(u)
 if s.fragment and not s.path and s.fragment not in x.ids:bad.append(u)
assert x.videos==57 and not bad,(x.videos,bad)
assert len(x.ids)==len(set(x.ids)), 'duplicate HTML ids'
assert 'mora-comparison' in x.ids and 'paired-visuals' in x.ids and 'forward-inputs' in x.ids
final_input_rows=json.loads((root/'comparison/forward_input_results.json').read_text())
assert len(final_input_rows)==46 and all(r['display_paste_verified_without_resize'] for r in final_input_rows)
motion_rows=json.loads((root/'comparison/abc_wrist_motion_results.json').read_text())
assert len(motion_rows)==4 and all(r['moving_background_verified'] and r['frames']==75 for r in motion_rows)
assert all(c['unique_background_frames']>1 for r in motion_rows for c in r['cameras'].values())
url='http://127.0.0.1:18081/curated_smoke_20261010/index.html'
with urllib.request.urlopen(url,timeout=20) as r:
 b=r.read();assert b==p.read_bytes();print('HTML HTTP',r.status,len(b),'all asset links exist',len(x.urls),'videos',x.videos)
with urllib.request.urlopen('http://127.0.0.1:18081/curated_smoke_20261010/clips/20_interndata_a1/trajectory.mp4',timeout=20) as r:print('MP4 HTTP',r.status,r.headers.get('Content-Type'),len(r.read()))
for name in ['comparison/20_interndata_a1/gt_projection_compare.mp4','comparison/20_interndata_a1/image_comparison.jpg','comparison/08_abc_130k/gt_comparison.png','comparison/20_interndata_a1/forward_input_comparison.png','comparison/24_mecka/forward_input_comparison.png','comparison/08_abc_130k/abc_wrist_tcp_motion.mp4']:
 with urllib.request.urlopen('http://127.0.0.1:18081/curated_smoke_20261010/'+name,timeout=20) as r:
  assert r.read()==(root/name).read_bytes();print('Paired asset HTTP',r.status,name)
result={'html_http_status':200,'served_html_sha256':hashlib.sha256(b).hexdigest(),'html_asset_link_count':len(x.urls),'video_count':x.videos,'broken_relative_links':bad}
archive=root.parent/'fk_consistency/curated_smoke_20261010.zip'
with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18081/curated_smoke_20261010.zip',method='HEAD'),timeout=20) as response:
 assert int(response.headers['Content-Length'])==archive.stat().st_size
 result['archive_http_status']=response.status;result['archive_bytes']=archive.stat().st_size
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
public='https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/curated_smoke_20261010/index.html'
try:
 with urllib.request.build_opener(NoRedirect).open(public,timeout=30) as r:result['gateway_status']=r.status
except urllib.error.HTTPError as e:
 result['gateway_status']=e.code;loc=urlsplit(e.headers.get('Location',''));result['gateway_redirect_host']=loc.netloc;result['gateway_redirect_path']=loc.path
except Exception as e:result['gateway_error']=str(e)
(root/'delivery_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
