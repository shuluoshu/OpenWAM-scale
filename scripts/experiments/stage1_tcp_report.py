"""Report-only Stage1 2-D TCP overlays; never changes supervision masks."""
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
COLORS = dict(gt='#00e676', supervised='#ff3b30', untrusted='#ff9f1c')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def tcp_overlay_data(record, target, prediction):
    """Coordinates are decoded normalized UV; tile pixels = UV * [319,255]."""
    def field(key):
        return _array(target[key] if isinstance(target, dict) else getattr(target, key))
    uv = field('uv').reshape(-1, 2, 2)[0]
    point = field('point_valid').reshape(-1, 2)[0].astype(bool)
    inside = field('in_frame').reshape(-1, 2)[0].astype(bool)
    image_valid = bool(field('image_valid').reshape(-1)[0])
    pred = np.asarray(prediction['uv'], float).reshape(-1, 2, 2)[0]
    provenance = record.get('provenance', {})
    reasons = list(provenance.get('endpoint', {}).get('missing_reasons', []))
    reasons += list(provenance.get('camera', {}).get('missing_reasons', []))
    if not image_valid: reasons.append('目标图像无效')
    if np.any(point & ~inside): reasons.append('有效三维端点的二维投影不在目标图像内')
    if not point.all() and not reasons: reasons.append('端点/投影有效 mask 为 false')
    arms = []
    for side in range(2):
        gt_valid = bool(image_valid and point[side] and inside[side] and np.isfinite(uv[side]).all())
        finite = bool(np.isfinite(pred[side]).all())
        drawable = finite and bool(((pred[side] >= 0) & (pred[side] <= 1)).all())
        arms.append(dict(side=('L', 'R')[side], gt_valid=gt_valid, point_valid=bool(point[side]),
            in_frame=bool(inside[side]), image_valid=image_valid,
            gt_uv=uv[side].tolist() if gt_valid else None,
            gt_px=(uv[side] * [319., 255.]).tolist() if gt_valid else None,
            pred_uv=pred[side].tolist() if finite else None,
            pred_px=(pred[side] * [319., 255.]).tolist() if finite else None,
            pred_finite=finite, pred_drawable=drawable,
            prediction_status='supervised_comparison' if gt_valid else 'unsupervised_untrusted'))
    count = sum(a['gt_valid'] for a in arms)
    return dict(arms=arms, valid_gt_count=count, reason='; '.join(dict.fromkeys(reasons)) if count < 2 else '',
        prediction_coordinate_kind='learned_normalized_uv',
        gt_coordinate_kind='source_geometry_projection',
        physical_contact_verification='not_established_by_this_renderer',
        endpoint_definition=provenance.get('endpoint', {}).get('definition_id'),
        camera_provenance=provenance.get('camera', {}),
        units='normalized UV; 320x256 tile pixels = UV * [319,255]',
        caveat='Orange: no valid GT supervision for this sample/arm; untrusted 2-D forecast, not calibrated projection.')


def draw_tcp_overlay(image, data, path, title, *, show_predictions=True):
    """Draw every eligible point on the supplied GT or generated background."""
    image = image.convert('RGB')
    if image.size != (320, 256):
        raise ValueError('Stage1 TCP overlay requires the actual 320x256 goal tile')
    result = Image.new('RGB', (320, 344), '#142238'); result.paste(image, (0, 0))
    draw = ImageDraw.Draw(result); font = ImageFont.truetype(FONT, 11)
    for a in data['arms']:
        if a['gt_valid']:
            x, y = a['gt_px']; draw.rectangle((x-6, y-6, x+6, y+6), outline=COLORS['gt'], width=3)
            draw.text((max(0, min(275, x+8)), max(0, min(240, y-14))), 'GT '+a['side'], font=font, fill=COLORS['gt'], stroke_width=1, stroke_fill='black')
        if show_predictions and a['pred_drawable']:
            x, y = a['pred_px']; color = COLORS['supervised' if a['gt_valid'] else 'untrusted']
            draw.ellipse((x-5, y-5, x+5, y+5), outline=color, width=3)
            draw.text((max(0, min(280, x+8)), max(0, min(240, y+6))), 'P '+a['side'], font=font, fill=color, stroke_width=1, stroke_fill='black')
    draw.text((7, 261), title, fill='white', font=font)
    draw.text((7, 277), f"GT: {data['valid_gt_count']}/2 valid | green squares", fill=COLORS['gt'], font=font)
    if show_predictions:
        draw.text((7, 293), 'Red: prediction with GT | Orange: NO GT', fill='white', font=font)
        draw.text((7, 309), 'Orange is untrusted, not calibrated projection.', fill=COLORS['untrusted'], font=font)
        draw.text((7, 325), 'P = learned 2D UV; not camera/FK projection.', fill='white', font=font)
    else:
        draw.text((7, 297), 'Masked GT = N/A; (0,0) placeholders hidden.', fill='white', font=font)
    result.save(path)


def coverage_rows(cases):
    groups = defaultdict(list)
    for case in cases: groups[case['record']['family']].append(case)
    rows = []
    for family, items in sorted(groups.items()):
        rows.append(dict(family=family, cases=len(items), possible_gt_points=2*len(items),
            valid_gt_points=sum(x['tcp']['valid_gt_count'] for x in items),
            valid_cases=sum(x['tcp']['valid_gt_count'] > 0 for x in items),
            fit_points=sum(x['tcp']['valid_gt_count'] for x in items if x['split']=='fit'),
            alternate_points=sum(x['tcp']['valid_gt_count'] for x in items if x['split']!='fit'),
            reasons=list(dict.fromkeys(x['tcp']['reason'] for x in items if x['tcp']['reason']))))
    return rows
