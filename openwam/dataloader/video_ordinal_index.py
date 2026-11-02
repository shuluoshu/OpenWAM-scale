"""Verified presentation-order random access for immutable VFR source videos.

The index is built by decoding *every* frame from stream start. Neither nominal
FPS nor timestamp rounding defines an ordinal. Every requested seek result must
match both indexed PTS and the sequential RGB hash; otherwise callers use their
ordinary sequential decoder. No cache files are written beside source videos.

OPENWAM_VIDEO_INDEX_CACHE selects a shared cache (default ~/.cache/openwam/video_index).
OPENWAM_MECKA_DECODE_MODE=indexed (default) enables this path; sequential bypasses it.
OPENWAM_VIDEO_INDEX_MODE is lazy (build missing indexes), read (prebuilt only), or
off. Builders hold a nonblocking flock and publish with atomic rename. Other
workers keep working sequentially while an index is being built.
OPENWAM_VIDEO_DECODE_THREADS sets decoder CPU threads (1..8, default 2), including
the ordinary sequential/CFR decoder. Thread count does not change index identity.
"""
from __future__ import annotations

import fcntl
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

VERSION = 2
CACHE_ERRORS = (OSError, ValueError, KeyError, EOFError, BadZipFile)
_COUNTERS = Counter()


def video_decode_threads():
    """One bounded CPU thread policy for sequential, indexed, and index builds."""
    value = os.environ.get('OPENWAM_VIDEO_DECODE_THREADS', '2')
    try:
        threads = int(value)
    except ValueError as exc:
        raise ValueError('OPENWAM_VIDEO_DECODE_THREADS must be an integer between 1 and 8') from exc
    if not 1 <= threads <= 8:
        raise ValueError('OPENWAM_VIDEO_DECODE_THREADS must be an integer between 1 and 8')
    return threads


def video_index_counters(*, reset=False):
    """Snapshot this process's diagnostics; workers have independent counters."""
    result = dict(_COUNTERS)
    if reset:
        _COUNTERS.clear()
    return result


def _fingerprint(path):
    import av
    path = Path(path).resolve()
    stat = path.stat()
    return dict(version=VERSION, path=str(path), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                inode=stat.st_ino, pyav=av.__version__,
                libraries={k: list(v) for k, v in av.library_versions.items()})


def _cache_dir(cache_dir):
    if cache_dir is None:
        cache_dir = os.environ.get('OPENWAM_VIDEO_INDEX_CACHE')
    if cache_dir is None:
        cache_dir = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'openwam/video_index'
    return Path(cache_dir)


def index_path(path, *, cache_dir=None):
    """Return the artifact path for this exact source stat and decoder version."""
    identity = json.dumps(_fingerprint(path), sort_keys=True).encode()
    return _cache_dir(cache_dir) / (hashlib.sha256(identity).hexdigest() + '.npz')


def _rgb_hash(frame):
    # PyAV to_image() also reformats to rgb24, then copies every packed row
    # through PIL. Hash the same packed RGB bytes without those extra copies.
    try:
        rgb = frame.to_ndarray(format='rgb24')
        if rgb.dtype != np.uint8 or rgb.shape != (frame.height, frame.width, 3):
            raise ValueError('unsupported RGB array layout')
        return hashlib.sha256(memoryview(np.ascontiguousarray(rgb))).hexdigest().encode()
    except (AttributeError, NotImplementedError, ValueError, TypeError, BufferError):
        pass
    return hashlib.sha256(np.asarray(frame.to_image().convert('RGB')).tobytes()).hexdigest().encode()


def _validate(index, fingerprint):
    if index['metadata']['fingerprint'] != fingerprint:
        raise ValueError('source fingerprint changed')
    pts, keys, hashes = index['pts'], index['keyframes'], index['rgb_sha256']
    if (pts.ndim != 1 or pts.dtype != np.int64 or len(pts) == 0
            or keys.ndim != 1 or keys.dtype != np.int64
            or hashes.shape != pts.shape or hashes.dtype != np.dtype('S64')
            or np.any(keys < 0) or np.any(keys >= len(pts))
            or np.any(keys[1:] <= keys[:-1])):
        raise ValueError('invalid ordinal index')


def _load(artifact, fingerprint):
    with np.load(artifact, allow_pickle=False) as data:
        index = {key: data[key] for key in ('pts', 'keyframes', 'rgb_sha256')}
        index['metadata'] = json.loads(str(data['metadata'].item()))
    _validate(index, fingerprint)
    return index


def build_video_index(path, *, cache_dir=None):
    """Build or load an immutable index; return None if another builder owns it.

    Full decoding and RGB hashing are intentionally paid once. The source must
    remain immutable: stat identity is checked before and after decoding and on
    every read. A complete source SHA256 is also recorded for provenance.
    """
    import av
    threads = video_decode_threads()
    fingerprint = _fingerprint(path)
    artifact = index_path(path, cache_dir=cache_dir)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            return _load(artifact, fingerprint)
        except CACHE_ERRORS:
            pass
        digest = hashlib.sha256()
        with open(path, 'rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        pts, keys, hashes = [], [], []
        missing_pts = False
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = threads
            metadata = dict(fingerprint=fingerprint, source_sha256=digest.hexdigest(),
                            time_base=str(stream.time_base), average_rate=str(stream.average_rate))
            for ordinal, frame in enumerate(container.decode(stream)):
                missing_pts |= frame.pts is None
                pts.append(0 if frame.pts is None else frame.pts)
                hashes.append(_rgb_hash(frame))
                if frame.key_frame:
                    keys.append(ordinal)
        if _fingerprint(path) != fingerprint:
            raise ValueError('source changed while building index')
        metadata['seek_safe'] = bool(not missing_pts and pts and keys
                                     and all(a < b for a, b in zip(pts, pts[1:])))
        index = dict(pts=np.asarray(pts, dtype=np.int64),
                     keyframes=np.asarray(keys, dtype=np.int64),
                     rgb_sha256=np.asarray(hashes, dtype='S64'), metadata=metadata)
        _validate(index, fingerprint)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=artifact.parent, suffix='.tmp', delete=False) as out:
                temporary = out.name
                np.savez(out, pts=index['pts'], keyframes=index['keyframes'],
                         rgb_sha256=index['rgb_sha256'], metadata=json.dumps(metadata, sort_keys=True))
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, artifact)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return index


def get_video_index(path, *, cache_dir=None, build=True):
    fingerprint = _fingerprint(path)
    try:
        index = _load(index_path(path, cache_dir=cache_dir), fingerprint)
        _COUNTERS['cache_hits'] += 1
        return index
    except CACHE_ERRORS:
        _COUNTERS['cache_misses'] += 1
        if build:
            _COUNTERS['build_requests'] += 1
        return build_video_index(path, cache_dir=cache_dir) if build else None


def read_indexed_frames(path, ordinals, *, cache_dir=None, build=True):
    """Read verified frames and record process-local hit/fallback counters."""
    video_decode_threads()  # Invalid launch configuration must not be hidden by fallback.
    _COUNTERS['requests'] += 1
    if (os.environ.get('OPENWAM_MECKA_DECODE_MODE', 'indexed') == 'sequential'
            or os.environ.get('OPENWAM_VIDEO_INDEX_MODE', 'lazy') == 'off'):
        _COUNTERS['bypasses'] += 1
        return None
    result = _read_indexed_frames(path, ordinals, cache_dir=cache_dir, build=build)
    if result is None:
        _COUNTERS['fallbacks'] += 1
    else:
        _COUNTERS['indexed_reads'] += 1
        _COUNTERS['decoded_frames'] += result['decoded_frames']
        _COUNTERS['requested_frames'] += len(result['frames'])
    return result


def _read_indexed_frames(path, ordinals, *, cache_dir=None, build=True):
    """Return verified AVFrames and decode count, or None for safe fallback.

    Unique strictly increasing PTS allow an unambiguous seek landing ordinal.
    B-frame reordering is handled by the decoder's presentation-order output.
    Hash validation catches open GOPs, incorrect reference frames, and any seek
    output that has the right PTS but different pixels.
    """
    import av
    decode_mode = os.environ.get('OPENWAM_MECKA_DECODE_MODE', 'indexed')
    if decode_mode == 'sequential':
        return None
    if decode_mode != 'indexed':
        raise ValueError('OPENWAM_MECKA_DECODE_MODE must be sequential or indexed')
    mode = os.environ.get('OPENWAM_VIDEO_INDEX_MODE', 'lazy')
    if mode == 'off':
        return None
    if mode not in ('lazy', 'read'):
        raise ValueError('OPENWAM_VIDEO_INDEX_MODE must be lazy, read, or off')
    requested = sorted(set(ordinals))
    if not requested:
        return None
    try:
        fingerprint = _fingerprint(path)
        index = get_video_index(path, cache_dir=cache_dir, build=build and mode == 'lazy')
        if index is None:
            return None
        _validate(index, fingerprint)
        pts, keys = index['pts'], index['keyframes']
        if (not index['metadata']['seek_safe'] or np.any(pts[1:] <= pts[:-1])
                or requested[0] < 0 or requested[-1] >= len(pts)):
            return None
        frames, decoded_count = {}, 0
        remaining = set(requested)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = video_decode_threads()
            if str(stream.time_base) != index['metadata']['time_base']:
                return None
            while remaining:
                target = min(remaining)
                key_pos = int(np.searchsorted(keys, target, side='right')) - 1
                if key_pos < 0:
                    return None
                anchor = int(keys[key_pos])
                container.seek(int(pts[anchor]), stream=stream, backward=True, any_frame=False)
                ordinal = None
                for frame in container.decode(stream):
                    decoded_count += 1
                    if frame.pts is None:
                        return None
                    if ordinal is None:
                        ordinal = int(np.searchsorted(pts, frame.pts))
                        if ordinal > target:
                            return None
                    if ordinal >= len(pts) or frame.pts != int(pts[ordinal]):
                        return None
                    if ordinal in remaining:
                        if _rgb_hash(frame) != index['rgb_sha256'][ordinal]:
                            return None
                        frames[ordinal] = frame
                        remaining.remove(ordinal)
                    if not remaining:
                        break
                    # Skip whole GOPs between sparse requested regions.
                    next_key_pos = int(np.searchsorted(keys, min(remaining), side='right')) - 1
                    if (target not in remaining and next_key_pos >= 0
                            and int(keys[next_key_pos]) > ordinal + 1):
                        break
                    ordinal += 1
                if target in remaining:
                    return None
        if _fingerprint(path) != fingerprint:
            return None
        return dict(frames=frames, decoded_frames=decoded_count, metadata=index['metadata'])
    except (*CACHE_ERRORS, av.error.FFmpegError):
        # Cache/seek errors must never replace the source decoder's semantics.
        return None
