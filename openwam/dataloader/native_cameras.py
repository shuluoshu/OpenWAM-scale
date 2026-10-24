"""Canonical native camera names, independent of robot state/geometry semantics."""

HEAD_CAMERAS = {f: 'observation.images.' + c for f, c in {
    'Genrobot': 'cam_high', 'LightWheel': 'head_left', 'abc_130k': 'cam_high',
    'egodex': 'egocentric', 'egoverse': 'front_1', 'interndata_a1': 'cam_high',
    'mecka': 'egocentric', 'morphirobot0930': 'cam_high', 'robochallenge_v1': 'cam_high',
    'robochallenge_v2': 'cam_high', 'robomind_v1': 'cam_front', 'robomind_v2': 'camera_front'}.items()}


def resolve_native_cameras(repo, info):
    """Return ordered head/left-wrist/right-wrist keys or None when absent.

    This is the Stage2 profile's camera preference order. A declared but malformed
    camera is an error, rather than evidence that a view is unavailable.
    """
    family = repo.split('/')[0]
    native=repo.startswith('/') and any(k in info['features'] for k in ('observation.images.head','observation.images.head_undistorted'))
    if family not in HEAD_CAMERAS and not native:
        raise ValueError(f'{repo}: unsupported_family')
    features = info['features']

    def choose(candidates):
        for key in candidates:
            if key in features and features[key].get('dtype') != 'video':
                raise ValueError(f'{repo}: malformed present camera feature: {key}')
            if key in features:
                shape=features[key].get('shape')
                if (not isinstance(shape,(list,tuple)) or len(shape)!=3 or shape[-1]!=3 or
                        any(type(v) is not int or v<1 for v in shape)):
                    raise ValueError(f'{repo}: malformed present camera shape: {key}: {shape}')
        return next((key for key in candidates if key in features), None)

    if native:
        return [choose([f'observation.images.{role}_undistorted',f'observation.images.{role}']) for role in ('head','left','right')]
    return [choose([HEAD_CAMERAS[family]])] + [choose([
        f'observation.images.cam_{side}_wrist', f'observation.images.camera_{side}',
        f'observation.images.{side}_wrist', f'observation.images.cam_{side}'
    ]) for side in ('left', 'right')]
