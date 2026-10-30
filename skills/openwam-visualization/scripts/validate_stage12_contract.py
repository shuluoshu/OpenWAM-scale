"""Check report completeness, not camera correctness or actual overlay pixels."""
import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


class Media(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = set()
        self.videos = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ('src', 'poster', 'href'):
            if attrs.get(key):
                self.assets.add(unquote(urlsplit(attrs[key]).path))
        if tag == 'video' and attrs.get('src'):
            self.videos[unquote(urlsplit(attrs['src']).path)] = 'controls' in attrs


def validate(data, root, decode=False):
    root = Path(root).resolve()
    errors, limitations, decoded = [], [], {}
    parser = Media()

    def asset(path, label, linked=False, video=False):
        if not isinstance(path, str) or not path or urlsplit(path).scheme or Path(path).is_absolute():
            errors.append(f'{label}: expected relative asset path'); return
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            errors.append(f'{label}: missing/outside asset {path}'); return
        if linked and path not in parser.assets:
            errors.append(f'{label}: asset not shown in page: {path}')
        if video:
            if not parser.videos.get(path):
                errors.append(f'{label}: video requires controls and direct src: {path}')
            if decode and path not in decoded:
                try:
                    import av
                    with av.open(str(target)) as container:
                        stream = container.streams.video[0]
                        frames = list(container.decode(video=0))
                        times = [f.time for f in frames]
                        if len(frames) < 2 or any(t is None for t in times) or any(a >= b for a, b in zip(times, times[1:])):
                            raise ValueError('Need >=2 frames with increasing timestamps')
                        if stream.codec_context.name != 'h264' or frames[0].format.name != 'yuv420p':
                            raise ValueError('Expected browser-compatible H264/yuv420p')
                        decoded[path] = dict(frames=len(frames), first_time=times[0], last_time=times[-1])
                except Exception as exc:
                    errors.append(f'{label}: cannot decode {path}: {exc}')

    page = data.get('page', '')
    asset(page, 'page')
    if page and (root / page).is_file() and (root / page).resolve().is_relative_to(root):
        parser.feed((root / page).read_text())
    if data.get('schema_version') != 1:
        errors.append('schema_version must be 1')
    expected = data.get('expected_cases', [])
    variants = data.get('expected_variants', [])
    if not expected or not variants:
        errors.append('expected_cases and expected_variants must come from evaluation manifest')
    cases = data.get('cases', [])
    actual = [c.get('case_id') for c in cases]
    if Counter(actual) != Counter(expected) or len(set(actual)) != len(actual):
        errors.append(f'case coverage mismatch: expected {expected}, actual {actual}')

    def lane(row, name, key, is_video=False, optional=False):
        obj = row.get(key, {})
        label = f'{name}/{key}'
        status = obj.get('status')
        if status == 'ready':
            paths = obj.get('assets', [])
            if not paths:
                errors.append(f'{label}: ready without assets')
            if key == 'stage1' and len(paths) < 2:
                errors.append(f'{label}: need predicted image and TCP overlay')
            for path in paths:
                asset(path, label, linked=True, video=is_video)
            if key in ('stage1', 'direct', 'fk') and not obj.get('evidence'):
                errors.append(f'{label}: ready requires coordinate/provenance evidence')
        elif status == 'unavailable':
            if not obj.get('reason') or not obj.get('evidence'):
                errors.append(f'{label}: unavailable requires reason and evidence')
            limitations.append(dict(lane=label, reason=obj.get('reason')))
        elif status == 'not_applicable' and optional:
            if not obj.get('reason'):
                errors.append(f'{label}: not_applicable requires reason')
        else:
            errors.append(f'{label}: incomplete or invalid status {status!r}')
        for path in obj.get('evidence', []):
            asset(path, label)

    for case in cases:
        name = str(case.get('case_id'))
        identity = case.get('identity', {})
        for key in ('dataset', 'episode', 'anchor_frame', 'conditioning_text'):
            if identity.get(key) is None or (key != 'anchor_frame' and identity.get(key) == ''):
                errors.append(f'{name}: missing identity.{key}')
        for key in ('task', 'subtask', 'granularity'):
            if key not in case:
                errors.append(f'{name}: missing input {key}')
        frame = case.get('frame_contract', {})
        numbers = ('input_frame', 'instruction_start_frame', 'instruction_end_exclusive', 'target_frame', 'stage2_horizon_last_frame')
        if not all(type(frame.get(k)) is int for k in numbers):
            errors.append(f'{name}: frame_contract requires explicit integer input/instruction/target/horizon frames')
        else:
            if frame['input_frame'] != identity.get('anchor_frame'):
                errors.append(f'{name}: frame_contract input frame differs from identity')
            if not frame['instruction_start_frame'] <= frame['input_frame'] < frame['instruction_end_exclusive']:
                errors.append(f'{name}: frame_contract input outside selected instruction interval')
            if frame['target_frame'] != frame['instruction_end_exclusive'] - 1:
                errors.append(f'{name}: target frame must equal instruction end_exclusive - 1')
            if frame['stage2_horizon_last_frame'] < frame['input_frame']:
                errors.append(f'{name}: frame_contract Stage2 horizon precedes input')
        if frame.get('instruction_kind') not in ('task', 'subtask') or frame.get('instruction_kind') != case.get('granularity'):
            errors.append(f'{name}: frame_contract selected instruction kind mismatch')
        asset(case.get('current_image'), f'{name}/current_image', linked=True)
        lane(case, name, 'source_video', is_video=True)
        lane(case, name, 'stage1')
        if case.get('stage1', {}).get('status') == 'ready' and case['stage1'].get('identity') != identity:
            errors.append(f'{name}/stage1: identity differs from actual input')
        if case.get('stage1', {}).get('status') == 'ready':
            stage1 = case['stage1']
            if stage1.get('frame_contract') != frame or stage1.get('goal_frame') != frame.get('target_frame'):
                errors.append(f'{name}/stage1: frame_contract/goal differs from selected instruction target')
            if stage1.get('coordinate_output_kind') != 'stage1_2d_model_output':
                errors.append(f'{name}/stage1: coordinate_output_kind must identify learned 2D model output, not camera/FK projection')
            if type(stage1.get('valid_gt_count')) is not int or stage1['valid_gt_count'] not in (0, 1, 2):
                errors.append(f'{name}/stage1: require explicit valid_gt_count')
            if stage1.get('valid_gt_count') == 0 and stage1.get('prediction_status') != 'unverified_no_gt':
                errors.append(f'{name}/stage1: no-GT prediction must remain unverified_no_gt')
        stage1_sha = case.get('stage1', {}).get('checkpoint_sha256')
        if case.get('stage1', {}).get('status') == 'ready' and not re.fullmatch(r'[0-9a-f]{64}', str(stage1_sha)):
            errors.append(f'{name}/stage1: missing actual checkpoint_sha256')
        if set(case.get('variants', {})) != set(variants):
            errors.append(f'{name}: variant coverage mismatch')
        for variant, row in case.get('variants', {}).items():
            label = f'{name}/{variant}'
            if row.get('step') is None:
                errors.append(f'{label}: missing checkpoint step')
            if case.get('stage1', {}).get('status') == 'ready' and row.get('stage1_checkpoint_sha256') != stage1_sha:
                errors.append(f'{label}: Stage1 checkpoint differs from shared Stage1 prediction')
            lane(row, label, 'direct', is_video=True)
            lane(row, label, 'fk', is_video=True, optional=True)
            lane(row, label, 'generated', is_video=True, optional=True)
    return dict(ok=not errors, errors=errors, limitations=limitations,
                all_requested_projections_available=not errors and not limitations,
                checked_cases=len(cases), checked_variants=variants, decoded_videos=decoded,
                scope='Completeness and media checks only; does not prove geometry or overlay pixels')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--decode-videos', action='store_true')
    p.add_argument('--output-json', type=Path)
    args = p.parse_args()
    result = validate(json.loads(args.manifest.read_text()), args.manifest.parent, args.decode_videos)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.write_text(rendered + '\n')
    print(rendered)
    raise SystemExit(0 if result['ok'] else 1)
