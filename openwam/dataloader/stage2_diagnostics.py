"""Report-only endpoint pixels/geometry shared by native preflight and evaluation.

Nothing returned by this module belongs in model conditioning or cache keys.
"""
from __future__ import annotations


def report_goal_target(dataset, repo, episode, metadata, sample):
    import pyarrow.parquet as pq
    from scripts.data.prepare_subtask_goals import recorded_tcp_transform, camera_eligibility, decode_frames
    from scripts.data.prepare_goal_labels import goal_camera
    from scripts.data.trajectory_projection import goal_source_tcp, project_goal_tcp
    target = dict(sample['report_target'])
    image = None
    try:
        goal = target['goal_source_frame']; camera = metadata['profile']['cameras'][0]
        if camera is None:
            raise ValueError('head camera unavailable')
        prefix = 'videos/' + camera + '/'; ep = metadata['ep']; info = metadata['info']
        video = metadata['root'] / info['video_path'].format(video_key=camera, episode_index=episode,
            chunk_index=int(ep[prefix+'chunk_index']), file_index=int(ep[prefix+'file_index']))
        frames, _ = decode_frames(video, float(ep[prefix+'from_timestamp']), float(info['fps']), [goal],
            sequential=repo['family']=='mecka',
            video_ordinals=metadata['video_ordinals'] if repo['family']=='mecka' else None)
        image = frames[goal]
        target['source_wh'] = list(image.size)
        from openwam.dataloader.episode_data import read_episode_rows
        rows = read_episode_rows(metadata, episode, filters=[('frame_index','=',goal)]).to_pylist()
        if len(rows) != 1:
            raise ValueError('report goal row not unique')
        endpoint = goal_source_tcp(repo['family'], rows[0], tcp_from_eef=recorded_tcp_transform(repo['family'], info))
        eligible, qa = camera_eligibility(repo['family'], metadata['root'], episode)
        calibration, camera_metadata = goal_camera(repo['family'], rows[0], camera, info, image.size, geometry_eligible=eligible)
        target.update(endpoint=endpoint, camera=camera_metadata, calibration_eligibility=qa)
        if calibration is None:
            raise ValueError('goal camera geometry unavailable')
        k, e, model, distortion, wh = calibration
        # project_goal_tcp already applies the exact pixel-center stretch; never resize UV twice.
        target['projection'] = project_goal_tcp(endpoint['tcp_world'], e, k, wh, camera_model=model,
            distortion=distortion, tcp_valid=endpoint['tcp_valid'])
    except Exception as exc:
        target['diagnostic_unavailable'] = f'{type(exc).__name__}: {exc}'
    return image, target
