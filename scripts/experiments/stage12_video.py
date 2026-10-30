"""Visible TCP markers on the actual horizon video, with per-frame geometry."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
GT='#00dc78'
PRED='#f02727'


def render_tcp_video_frame(gt_image,predicted_image,data,index,title):
    """Head-tile UV only; wrist tiles are retained without invented calibration."""
    panels=[gt_image] if predicted_image is None else [gt_image,predicted_image]
    if any(im.size!=(320,384) for im in panels):
        raise ValueError('TCP video requires the actual 320x384 multiview frames')
    result=Image.new('RGB',(320*len(panels),480),'white')
    font=ImageFont.truetype(FONT,11);small=ImageFont.truetype(FONT,10)
    gt=np.asarray(data['gt_uv'],float)[index];pred=np.asarray(data['pred_uv'],float)[index]
    gt_inside=np.asarray(data['gt_in_frame'],bool)[index]
    pred_inside=np.asarray(data['pred_in_frame'],bool)[index]
    projected=np.asarray(data['pred_projectable'],bool)[index]
    action=int(data['action_indices'][index]);frame=int(data['source_frames'][index])
    for col,im in enumerate(panels):
        tile=im.convert('RGB').copy();draw=ImageDraw.Draw(tile)
        for side in range(2):
            name=('L','R')[side]
            if gt_inside[side] and np.isfinite(gt[side]).all():
                x,y=gt[side];draw.rectangle((x-5,y-5,x+5,y+5),outline=GT,width=3)
                label_x=x-34 if side==0 else x+8
                draw.text((max(0,min(270,label_x)),max(0,min(242,y-20))),'GT '+name,font=small,fill=GT,stroke_width=1,stroke_fill='black')
            if action>=0 and projected[side] and pred_inside[side] and np.isfinite(pred[side]).all():
                x,y=pred[side];draw.ellipse((x-6,y-6,x+6,y+6),outline=PRED,width=3)
                draw.text((max(0,min(280,x+8)),max(0,min(242,y+8))),'P '+name,font=small,fill=PRED,stroke_width=1,stroke_fill='white')
        result.paste(tile,(320*col,96))
    d=ImageDraw.Draw(result)
    d.text((6,2),title,font=font,fill='black')
    time_label=f'source f{frame} | '+(f'action[{action}] = future step {action+1}' if action>=0 else 'observed GT only; no future prediction')
    d.text((6,18),time_label,font=small,fill='black')
    d.text((6,34),'GT: green square | predicted TCP: red circle | L/R',font=small,fill='black')
    d.text((6,49),'Source video + TCP (head camera)',font=small,fill='black')
    if len(panels)==2:d.text((326,49),'Generated video + same source-camera pixels',font=small,fill='black')
    for side in range(2):
        name=('L','R')[side]
        if action<0:state='observed GT' if gt_inside[side] else 'GT unavailable/outside'
        elif not projected[side]:state='prediction unavailable / masked'
        elif not pred_inside[side]:state=f'prediction outside: ({pred[side,0]:.1f}, {pred[side,1]:.1f})'
        else:
            error=float(np.asarray(data['pixel_error'],float)[index,side])
            state=f'error {error:.2f}px' if np.isfinite(error) else 'prediction visible; GT unavailable'
        d.text((6,64+15*side),name+': '+state,font=small,fill='black')
    reasons=data.get('frame_reasons',data.get('reasons',[]))
    if reasons and reasons[index] and len(panels)==2:
        d.text((326,65),'N/A: '+str(reasons[index])[:48],font=small,fill='#9b4800')
    return result
