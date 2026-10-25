"""Source row to released video ordinal contracts (independent of video PTS)."""
import numpy as np


def mecka_video_ordinals(columns, episode):
    """Validate Mecka's raw-frame pointers before indexing its untrimmed MP4.

    The raw-fields converter keeps video bytes and writes zero-based row times,
    while source_frame_index retains the original video frame number. A missing
    or inconsistent pointer must not silently become a zero video offset.
    """
    length=episode['length'];offset=episode.get('source_frame_offset')
    if (isinstance(offset,(bool,np.bool_)) or not isinstance(offset,(int,np.integer)) or offset<0):
        raise ValueError('Mecka requires nonnegative integer source_frame_offset')
    frames=np.asarray(columns.get('frame_index'))
    source=np.asarray(columns.get('source_frame_index'))
    if (frames.shape!=(length,) or frames.dtype.kind not in 'iu'
            or not np.array_equal(frames,np.arange(length))):
        raise ValueError('Mecka frame_index must be dense unique complete')
    if (source.shape!=(length,) or source.dtype.kind not in 'iu'
            or not np.array_equal(source,np.arange(length)+offset)):
        raise ValueError('Mecka source_frame_index must equal frame_index + source_frame_offset')
    return source
