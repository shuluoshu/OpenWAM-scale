"""Display actual final pixel organization; no inferred Mora spatial canvas."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent
sys.path.insert(0,str(REPORT.parents[1]))
from openwam.model.video_backbone.wan.preprocess import preprocess_video

def dump(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def render(clip):
 ident=clip['id'];dest=HERE/ident;src=REPORT/'clips'/ident
 imeta=json.loads((dest/'image_metrics.json').read_text())
 frames=[Image.open(src/f'model_frame_{k}.png').convert('RGB') for k in range(9)]
 tensor=preprocess_video(frames,dtype=torch.bfloat16,device='cpu')
 assert list(tensor.shape)==[1,3,9,384,320]
 # Invert actual bf16 pixel preprocessing for display, not a simulated image layout.
 rgb=(tensor[0,:,0].float().permute(1,2,0)+1)*127.5
 restored=np.rint(rgb.clamp(0,255).numpy()).astype(np.uint8)
 image=Image.fromarray(restored);image.save(dest/'forward_openwam_canvas.png')
 canvas=Image.new('RGB',(1536,700),'#111c2b');d=ImageDraw.Draw(canvas)
 def text(x,y,s,size=18,color='#dce8f5'):d.text((x,y),s,font=font(size),fill=color)
 text(24,16,f'{ident} | SAME source anchor {clip["start"]} | Final visual input organization',24)
 text(24,66,'OpenWAM: ONE spatial canvas',22,'#ffbf63')
 text(414,66,'MoraVLA: SEPARATE camera images -> concatenate patches',22,'#69dbe9')
 text(24,110,'t+0 | 320 x 384 | pre-VAE pixels',17)
 canvas.paste(image,(24,150))
 d.rectangle((23,149,344,534),outline='#ffbf63',width=1)
 roles=['head','left','right'];patches=0;tokens=0;views=[];rects=[]
 for i,role in enumerate(roles):
  x=414+i*370;c=imeta['cameras'].get(role)
  text(x,110,f'image{i}: {role}',19,'#69dbe9')
  if c is None:
   text(x,180,'ABSENT: omitted',19);text(x,213,'No image / no image tokens',16)
   continue
  final=Image.open(dest/c['files']['final']).convert('RGB');assert list(final.size)==c['final_size_wh']
  canvas.paste(final,(x,150));d.rectangle((x-1,149,x+final.width,150+final.height),outline='#69dbe9')
  text(x,490,f'{final.width} x {final.height} | {c["patch_count"]} patches',17)
  text(x,516,f'{c["visual_token_count"]} visual token positions',16)
  views.append({'role':role,'image_slot':i,'size_wh':list(final.size),'grid_thw':c['grid_thw'],
    'patch_shape':[c['patch_count'],c['patch_width']],'visual_token_count':c['visual_token_count'],
    'patch_sequence_range':[patches,patches+c['patch_count']]})
  patches+=c['patch_count'];tokens+=c['visual_token_count'];rects.append((x,150,final))
 text(24,565,'Head above; left/right wrists below.',16)
 text(24,591,'Missing wrist slots stay black.',16)
 text(414,565,f'pixel_values [{patches}, 1536]  |  image_grid_thw [{len(views)}, 3]  |  {tokens} visual token positions',18)
 text(414,598,'Camera tiles above are REPORT LAYOUT ONLY. Mora has no stitched RGB canvas.',17)
 text(24,642,'Images are display reconstructions of processed pixels. Full model / VAE forward was not executed.',17)
 text(24,671,'OpenWAM training video: t+[0,4,...,32], tensor [1,3,9,384,320]. Mora: current camera images only.',16)
 canvas.save(dest/'forward_input_comparison.png')
 # Native resolution paste must not introduce another resize or cover image pixels.
 assert np.array_equal(np.asarray(canvas.crop((24,150,344,534))),restored)
 for x,y,im in rects:assert np.array_equal(np.asarray(canvas.crop((x,y,x+im.width,y+im.height))),np.asarray(im))
 data={'id':ident,'source_anchor':clip['start'],'scope':'final pixel/patch organization; no full model or VAE forward',
  'openwam':{'canvas_size_wh':[320,384],'video_shape_bcthw':list(tensor.shape),'dtype':str(tensor.dtype),
   'video_source_offsets':list(range(0,33,4)),'displayed_time_offset':0,'pixel_range':[float(tensor.min()),float(tensor.max())],
   'inverse_bf16_display_max_difference_uint8':int(np.max(np.abs(restored.astype(int)-np.asarray(frames[0]).astype(int)))),
   'source_canvas_sha256':hashlib.sha256((src/'model_frame_0.png').read_bytes()).hexdigest()},
  'mora':{'spatially_stitched':False,'display_tiles_are_report_layout_only':True,'views':views,
   'concatenated_patch_shape':[patches,1536],'image_grid_thw_shape':[len(views),3],
   'visual_token_positions':tokens,'missing_views_omitted':imeta['missing_views_omitted_in_mora'],
   'patch_boundary_provenance':'Derived from previously executed image processor outputs and actual torch.cat code; full batch/model not executed',
   'display_rule':'Previously verified inverse patch + inverse normalize; clip RGB to [0,1] and round uint8'},
  'display_paste_verified_without_resize':True}
 dump(dest/'forward_input_metrics.json',data);return data

if __name__=='__main__':
 torch.set_num_threads(2)
 results=[render(x) for x in json.loads((REPORT/'manifest.json').read_text())['results']]
 dump(HERE/'forward_input_results.json',results)
 print('Rendered and verified',len(results),'final input comparisons')
