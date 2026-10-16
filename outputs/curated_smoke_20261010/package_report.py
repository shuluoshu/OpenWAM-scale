from pathlib import Path
import subprocess,json,zipfile
import av
OUT=Path(__file__).resolve().parent
rows=[]
paths=sorted((OUT/'clips').glob('*/trajectory.mp4'))+sorted((OUT/'comparison').glob('*/gt_projection_compare.mp4'))
paths+=sorted((OUT/'comparison').glob('*/abc_wrist_tcp_motion.mp4'))
assert len(paths)==57
for path in paths:
 data=path.read_bytes()
 if data.index(b'moov')>data.index(b'mdat'):
  tmp=path.with_name(path.stem+'_faststart.mp4')
  subprocess.run(['/usr/bin/ffmpeg','-v','error','-y','-i',str(path),'-c','copy','-movflags','+faststart',str(tmp)],check=True)
  tmp.replace(path)
 with av.open(str(path)) as c:
  frames=sum(1 for _ in c.decode(video=0));assert frames==(75 if path.name in ('trajectory.mp4','abc_wrist_tcp_motion.mp4') else 52)
 data=path.read_bytes();assert data.index(b'moov')<data.index(b'mdat')
 rows.append({'id':path.parent.name,'path':str(path.relative_to(OUT)),'frames':frames,'bytes':path.stat().st_size,'faststart':True})
(OUT/'media_verification.json').write_text(json.dumps(rows,indent=2)+'\n')
archive=OUT.parent/'fk_consistency/curated_smoke_20261010.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
 for p in sorted(OUT.rglob('*')):
  if not p.is_file() or p.is_symlink() or 'dataset' in p.relative_to(OUT).parts or '__pycache__' in p.parts:continue
  z.write(p,Path('curated_smoke_20261010')/p.relative_to(OUT))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print('Verified 57 H264 MP4s: 50x75 + 7x52 frames, faststart; ZIP CRC passes',archive,archive.stat().st_size)
