"""An indexed seek must reproduce presentation-order RGB and PTS exactly."""
from fractions import Fraction
import os
import hashlib

import numpy as np
import pytest


def video(path, count=144):
    av = pytest.importorskip('av')
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('libx264', rate=30)
        stream.width = 48
        stream.height = 32
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.time_base = Fraction(1, 30000)
        stream.options = {'g': '12', 'bf': '3', 'sc_threshold': '0'}
        pts = 660
        for i in range(count):
            rgb = np.random.default_rng(i).integers(0, 256, (32, 48, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts = pts
            frame.time_base = Fraction(1, 30000)
            for packet in stream.encode(frame):
                out.mux(packet)
            pts += 1000 + (73 if i % 5 == 0 else 0)
        for packet in stream.encode():
            out.mux(packet)
    with av.open(str(path)) as source:
        reference = [(f.pts, np.asarray(f.to_image().convert('RGB')))
                     for f in source.decode(video=0)]
    return reference


def test_indexed_vfr_b_frames_match_reference_and_skip_prefix(tmp_path):
    import openwam.dataloader.video_ordinal_index as module
    path = tmp_path / 'source.mp4'
    reference = video(path)
    cache = tmp_path / 'cache'
    index = module.build_video_index(path, cache_dir=cache)
    assert index['pts'].tolist() == [pts for pts, _ in reference]
    for ordinals in ([0, 1, 13], [70, 71, 73], [140, 143], [9, 57, 102, 138]):
        result = module.read_indexed_frames(path, ordinals, cache_dir=cache, build=False)
        assert result is not None
        for ordinal, frame in result['frames'].items():
            assert frame.pts == reference[ordinal][0]
            np.testing.assert_array_equal(np.asarray(frame.to_image().convert('RGB')), reference[ordinal][1])
        if min(ordinals) >= 70:
            assert result['decoded_frames'] < 40


def test_missing_corrupt_and_changed_source_indexes_fail_safe(tmp_path):
    import openwam.dataloader.video_ordinal_index as module
    path = tmp_path / 'source.mp4'
    video(path, 24)
    cache = tmp_path / 'cache'
    assert module.read_indexed_frames(path, [20], cache_dir=cache, build=False) is None
    module.build_video_index(path, cache_dir=cache)
    artifact = next(cache.glob('*.npz'))
    artifact.write_bytes(b'partial index')
    assert module.read_indexed_frames(path, [20], cache_dir=cache, build=False) is None
    module.build_video_index(path, cache_dir=cache)
    assert module.read_indexed_frames(path, [20], cache_dir=cache, build=False) is not None
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    assert module.read_indexed_frames(path, [20], cache_dir=cache, build=False) is None


def test_truncated_zip_index_falls_back_instead_of_crashing(tmp_path):
    import openwam.dataloader.video_ordinal_index as module
    path = tmp_path / 'source.mp4'
    video(path, 24)
    cache = tmp_path / 'cache'
    module.build_video_index(path, cache_dir=cache)
    artifact = next(cache.glob('*.npz'))
    artifact.write_bytes(artifact.read_bytes()[:100])
    assert module.read_indexed_frames(path, [20], cache_dir=cache, build=False) is None
    assert module.build_video_index(path, cache_dir=cache) is not None


def test_wrong_rgb_or_ambiguous_pts_cannot_escape_as_indexed_result(tmp_path, monkeypatch):
    import openwam.dataloader.video_ordinal_index as module
    path = tmp_path / 'source.mp4'
    video(path, 36)
    index = module.build_video_index(path, cache_dir=tmp_path / 'cache')
    monkeypatch.setattr(module, 'get_video_index', lambda *a, **k: index)
    index['rgb_sha256'][20] = b'0' * 64
    assert module.read_indexed_frames(path, [20]) is None
    index['pts'][10] = index['pts'][9]
    assert module.read_indexed_frames(path, [5]) is None


def test_busy_builder_and_unwritable_cache_leave_sequential_fallback(tmp_path):
    import fcntl
    import openwam.dataloader.video_ordinal_index as module
    path = tmp_path / 'source.mp4'
    video(path, 12)
    cache = tmp_path / 'cache'
    cache.mkdir()
    artifact = module.index_path(path, cache_dir=cache)
    with artifact.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module.read_indexed_frames(path, [8], cache_dir=cache) is None
    blocked = tmp_path / 'not_a_directory'
    blocked.touch()
    assert module.read_indexed_frames(path, [8], cache_dir=blocked) is None


def test_decode_frames_uses_index_and_preserves_explicit_source_mapping(tmp_path, monkeypatch):
    from scripts.data.prepare_subtask_goals import decode_frames
    path = tmp_path / 'source.mp4'
    reference = video(path)
    monkeypatch.setenv('OPENWAM_VIDEO_INDEX_CACHE', str(tmp_path / 'cache'))
    images, alignment = decode_frames(path, 0., 30, [0, 1, 2], sequential=True,
                                     video_ordinals=[120, 127, 143])
    for row, ordinal in enumerate([120, 127, 143]):
        np.testing.assert_array_equal(np.asarray(images[row]), reference[ordinal][1])
        assert alignment[row]['pts'] == reference[ordinal][0]
        assert alignment[row]['video_ordinal'] == ordinal
        assert alignment[row]['decode_method'] == 'indexed_verified_presentation_ordinal'
    monkeypatch.setenv('OPENWAM_VIDEO_INDEX_MODE', 'off')
    _, alignment = decode_frames(path, 0., 30, [140], sequential=True)
    assert alignment[140]['decode_method'] == 'sequential_decoded_ordinal'


def test_curated_inventory_deduplicates_selected_video_files(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    from scripts.data.build_mecka_video_index import selected_videos
    root = tmp_path / 'mecka' / 'shard_0000'
    (root / 'meta' / 'episodes').mkdir(parents=True)
    camera = 'observation.images.egocentric'
    (root / 'meta' / 'info.json').write_text(json.dumps({
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}))
    pq.write_table(pa.Table.from_pylist([
        {'episode_index': i, f'videos/{camera}/chunk_index': 0,
         f'videos/{camera}/file_index': 0 if i < 2 else 1}
        for i in range(3)]), root / 'meta' / 'episodes' / 'part.parquet')
    config = tmp_path / 'selection.yaml'
    config.write_text(yaml.safe_dump({'data_root_dir': str(tmp_path), 'resolver': {
        'segments_by_repo': {'mecka/shard_0000': [[0, 0, 10], [0, 10, 20], [1, 0, 20]]},
        'repo_weights': {'mecka/shard_0000': 1.}}}))
    records = selected_videos(config)
    assert len(records) == 1
    assert records[0]['episodes'] == [0, 1]
    assert records[0]['path'].endswith('file-000.mp4')


def test_cached_index_reused_when_mount_device_number_changes(tmp_path, monkeypatch):
    from pathlib import Path
    import openwam.dataloader.video_ordinal_index as module
    path = tmp_path / 'source.mp4'
    reference = video(path, 24)
    cache = tmp_path / 'cache'
    module.build_video_index(path, cache_dir=cache)
    artifact = module.index_path(path, cache_dir=cache)
    original_stat = Path.stat
    def changed_device(self, **kwargs):
        value = original_stat(self, **kwargs)
        if self == path:
            class OtherMount:
                st_dev = value.st_dev + 1
                def __getattr__(self, name):
                    return getattr(value, name)
            return OtherMount()
        return value
    monkeypatch.setattr(Path, 'stat', changed_device)
    assert module.index_path(path, cache_dir=cache) == artifact
    result = module.read_indexed_frames(path, [20], cache_dir=cache, build=False)
    assert result is not None
    np.testing.assert_array_equal(np.asarray(result['frames'][20].to_image()), reference[20][1])


def test_explicit_json_inventory_accepts_paths_or_previous_report(tmp_path):
    import json
    from scripts.data.build_mecka_video_index import read_inventory
    path = tmp_path / 'inventory.json'
    path.write_text(json.dumps([str(tmp_path / 'one.mp4'), str(tmp_path / 'two.mp4')]))
    records = read_inventory(path)
    assert [item['path'] for item in records] == [str(tmp_path / 'one.mp4'), str(tmp_path / 'two.mp4')]
    path.write_text(json.dumps({'inventory': records}))
    assert read_inventory(path) == records
    path.write_text(json.dumps([{'path': 42}]))
    with pytest.raises(ValueError, match='path'):
        read_inventory(path)


def test_read_only_counters_track_cache_misses_and_verified_reads(tmp_path, monkeypatch):
    import openwam.dataloader.video_ordinal_index as module
    path = tmp_path / 'source.mp4'
    video(path, 24)
    cache = tmp_path / 'cache'
    monkeypatch.setenv('OPENWAM_VIDEO_INDEX_MODE', 'read')
    module.video_index_counters(reset=True)
    assert module.read_indexed_frames(path, [20], cache_dir=cache) is None
    assert not cache.exists()
    module.build_video_index(path, cache_dir=cache)
    result = module.read_indexed_frames(path, [20], cache_dir=cache)
    assert result is not None
    counts = module.video_index_counters()
    assert counts['requests'] == 2
    assert counts['cache_misses'] == 1
    assert counts['cache_hits'] == 1
    assert counts['indexed_reads'] == 1
    assert counts['fallbacks'] == 1
    assert counts['decoded_frames'] == result['decoded_frames']


def test_builder_json_inventory_parallel_workers_publish_complete_artifacts(tmp_path, monkeypatch):
    import json
    import sys
    from scripts.data.build_mecka_video_index import main
    paths = [tmp_path / 'one.mp4', tmp_path / 'two.mp4']
    for path in paths:
        video(path, 24)
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps([{'path': str(path), 'episodes': [i]} for i, path in enumerate(paths)]))
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['build_mecka_video_index.py', '--inventory-json', str(inventory),
        '--video', str(paths[0]), '--workers', '2', '--cache-dir', str(tmp_path / 'cache'), '--output', str(output)])
    main()
    report = json.loads(output.read_text())
    assert report['selected_unique_videos'] == 2
    assert report['selected_episodes'] == 2
    assert len(report['builds']) == 2
    assert all(item['status'] == 'ready' for item in report['builds'])
    assert len(list((tmp_path / 'cache').glob('*.npz'))) == 2


def test_decoder_threads_preserve_rgb_and_reuse_index_across_thread_counts(tmp_path, monkeypatch):
    import openwam.dataloader.video_ordinal_index as module
    from scripts.data.prepare_subtask_goals import decode_frames
    from tests.dataloader.test_video_alignment import encode_video
    path = tmp_path / 'source.mp4'
    reference = video(path, 36)
    cache = tmp_path / 'cache'
    monkeypatch.setenv('OPENWAM_VIDEO_INDEX_CACHE', str(cache))
    monkeypatch.delenv('OPENWAM_VIDEO_DECODE_THREADS', raising=False)
    assert module.video_decode_threads() == 2
    monkeypatch.setenv('OPENWAM_VIDEO_DECODE_THREADS', '2')
    assert module.video_decode_threads() == 2
    module.build_video_index(path, cache_dir=cache)
    artifact = module.index_path(path, cache_dir=cache)
    cfr_path = tmp_path / 'cfr.mp4'
    cfr_reference = encode_video(cfr_path, count=24, variable=False)
    for threads in (1, 2):
        monkeypatch.setenv('OPENWAM_VIDEO_DECODE_THREADS', str(threads))
        assert module.video_decode_threads() == threads
        assert module.index_path(path, cache_dir=cache) == artifact
        images, alignment = decode_frames(cfr_path, 0., 30, [0, 23])
        for ordinal in (0, 23):
            np.testing.assert_array_equal(np.asarray(images[ordinal]), cfr_reference[ordinal][1])
            assert alignment[ordinal]['pts'] == cfr_reference[ordinal][0]
        for mode in ('sequential', 'indexed'):
            monkeypatch.setenv('OPENWAM_MECKA_DECODE_MODE', mode)
            images, alignment = decode_frames(path, 0., 30, [13, 35], sequential=True)
            for ordinal in (13, 35):
                np.testing.assert_array_equal(np.asarray(images[ordinal]), reference[ordinal][1])
                assert alignment[ordinal]['pts'] == reference[ordinal][0]
            if mode == 'indexed':
                assert alignment[35]['decode_method'] == 'indexed_verified_presentation_ordinal'


@pytest.mark.parametrize('value', ['0', '-1', '9', '2.0', 'bad', ''])
def test_invalid_decoder_thread_configuration_is_rejected(tmp_path, monkeypatch, value):
    import openwam.dataloader.video_ordinal_index as module
    from scripts.data.prepare_subtask_goals import decode_frames
    monkeypatch.setenv('OPENWAM_VIDEO_DECODE_THREADS', value)
    for call in (lambda: module.video_decode_threads(),
                 lambda: module.build_video_index(tmp_path / 'absent.mp4'),
                 lambda: module.read_indexed_frames(tmp_path / 'absent.mp4', [0]),
                 lambda: decode_frames(tmp_path / 'absent.mp4', 0., 30, [0])):
        with pytest.raises(ValueError, match='OPENWAM_VIDEO_DECODE_THREADS'):
            call()


@pytest.mark.parametrize('variable', [False, True])
def test_rgb_hash_uses_array_buffer_with_exact_pil_reference_bytes(tmp_path, variable):
    import av
    from tests.dataloader.test_video_alignment import encode_video
    from openwam.dataloader.video_ordinal_index import _rgb_hash
    path = tmp_path / 'source.mp4'
    encode_video(path, count=24, variable=variable)
    with av.open(str(path)) as source:
        for frame in source.decode(video=0):
            expected = hashlib.sha256(np.asarray(frame.to_image().convert('RGB')).tobytes()).hexdigest().encode()
            class ArrayOnly:
                def __getattr__(self, name):
                    return getattr(frame, name)
                def to_image(self):
                    raise AssertionError('fast hashing must not copy through PIL')
            assert _rgb_hash(ArrayOnly()) == expected


def test_rgb_hash_retains_pil_fallback_for_unsupported_array_conversion():
    from PIL import Image
    from openwam.dataloader.video_ordinal_index import _rgb_hash
    rgb = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    class Unsupported:
        def to_ndarray(self, **kwargs):
            raise NotImplementedError('array conversion unavailable')
        def to_image(self):
            return Image.fromarray(rgb)
    assert _rgb_hash(Unsupported()) == hashlib.sha256(rgb.tobytes()).hexdigest().encode()
