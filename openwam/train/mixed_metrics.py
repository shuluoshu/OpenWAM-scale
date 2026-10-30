"""Aggregate per-case evaluation errors without conflating language granularities."""
from collections import defaultdict
import html


def aggregate_mixed_metrics(cards):
    groups=defaultdict(list)
    for card in cards:groups[card['family']+' / '+card['granularity']].append(card)
    result={}
    for key,rows in groups.items():
        errors=[e for r in rows for e in r['tcp_pixel_errors'] if e is not None]
        image_errors=[r['image_mae_uint8'] for r in rows if r['image_mae_uint8'] is not None]
        result[key]=dict(samples=len(rows),image_mae_uint8=sum(image_errors)/len(image_errors) if image_errors else None,
            valid_points=len(errors),tcp_pixel_error=sum(errors)/len(errors) if errors else None)
    return result


def render_group_table(groups):
    rows=[]
    for key,values in groups.items():
        tcp='—' if values['tcp_pixel_error'] is None else f"{values['tcp_pixel_error']:.2f}"
        image='—' if values['image_mae_uint8'] is None else f"{values['image_mae_uint8']:.2f}"
        rows.append(f'<tr><td>{html.escape(key)}</td><td>{values["samples"]}</td><td>{image}</td><td>{tcp}</td><td>{values["valid_points"]}</td></tr>')
    return '<table border="1" cellpadding="6"><thead><tr><th>数据集 / 粒度</th><th>样例数</th><th>图像 MAE /255</th><th>TCP 均值 px</th><th>有效点数</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
