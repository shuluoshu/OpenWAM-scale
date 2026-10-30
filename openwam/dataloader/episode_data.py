"""Projected reads of complete episodes, including episodes spanning data shards."""
from pathlib import Path
from string import Formatter


def episode_data_path(root, info, episode):
    return Path(root) / info['data_path'].format(
        episode_index=int(episode['episode_index']),
        chunk_index=int(episode['data/chunk_index']),
        file_index=int(episode['data/file_index']))


def _candidate_paths(root, info, episode):
    # The metadata points to the first shard. File and chunk boundaries need not
    # coincide with episode boundaries; do not infer them from chunks_size.
    pattern=''
    for literal,field,spec,conversion in Formatter().parse(info['data_path']):
        pattern+=literal
        if field is not None:
            if field in ('chunk_index','file_index'):pattern+='*'
            elif field=='episode_index':pattern+=format(int(episode['episode_index']),spec)
            else:raise ValueError(f'unsupported data path field: {field}')
    return sorted(Path(root).glob(pattern))


def _may_contain_episode(parquet, episode):
    """Skip unrelated shards using row-group statistics, never approximate rows."""
    metadata=parquet.metadata
    column=next((i for i in range(metadata.num_columns)
                 if metadata.schema.column(i).path=='episode_index'),None)
    if column is None:raise ValueError('data shard lacks episode_index')
    for i in range(metadata.num_row_groups):
        stats=metadata.row_group(i).column(column).statistics
        if stats is None or not stats.has_min_max or stats.min<=episode<=stats.max:
            return True
    return False


def _validate_indices(table, episode, root):
    import pyarrow as pa
    length=int(episode['length']);context=f"{root}:{episode['episode_index']}"
    if (not pa.types.is_integer(table.schema.field('frame_index').type)
            or table['frame_index'].to_pylist()!=list(range(length))):
        raise ValueError(f'{context}: source frame indices must be dense unique complete')
    if 'index' in table.column_names and 'dataset_from_index' in episode and 'dataset_to_index' in episode:
        start=int(episode['dataset_from_index']);end=int(episode['dataset_to_index'])
        if (end-start!=length or not pa.types.is_integer(table.schema.field('index').type)
                or table['index'].to_pylist()!=list(range(start,end))):
            raise ValueError(f'{context}: global source indices disagree with episode metadata')


def read_episode_table(root, info, episode, columns=None, *, max_bytes=None):
    """Return the full episode and its actual files, with strict row identity.

    The common single-file case performs no directory scan. Only an incomplete
    episode triggers discovery of other shards. Reads are filtered by episode
    and projected to requested columns plus the row identity columns.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=episode_data_path(root,info,episode);parquet=pq.ParquetFile(path)
    schema=parquet.schema_arrow;number=int(episode['episode_index']);length=int(episode['length'])
    keys=None if columns is None else list(dict.fromkeys([
        *columns,'frame_index',*(['index'] if 'index' in schema.names else [])]))
    table=pq.read_table(path,columns=keys,filters=[('episode_index','=',number)])
    parts=[table];files=[path];size=table.nbytes
    if max_bytes is not None and size>max_bytes:raise ValueError('cache_byte_budget')
    if not pa.types.is_integer(table.schema.field('frame_index').type) or table['frame_index'].null_count:
        raise ValueError(f'{root}:{number}: source frame indices must be dense unique complete')
    frames=sorted(table['frame_index'].to_pylist())
    if frames!=list(range(length)):
        if not frames or frames[0]!=0:
            raise ValueError(f'{root}:{number}: metadata does not point to first episode part')
        for candidate in _candidate_paths(root,info,episode):
            if candidate==path:continue
            other=pq.ParquetFile(candidate)
            if not _may_contain_episode(other,number):continue
            part=pq.read_table(candidate,columns=keys,filters=[('episode_index','=',number)])
            if not len(part):continue
            if not schema.equals(other.schema_arrow,check_metadata=False):
                raise ValueError(f'{root}:{number}: inconsistent episode shard schema: {candidate}')
            size+=part.nbytes
            if max_bytes is not None and size>max_bytes:raise ValueError('cache_byte_budget')
            parts.append(part);files.append(candidate)
    table=pa.concat_tables(parts).sort_by('frame_index')
    _validate_indices(table,episode,root)
    return (table if columns is None else table.select(columns)),tuple(files)


def read_episode_rows(metadata, episode, *, columns=None, filters=()):
    """Read selected rows from every previously validated episode part."""
    import pyarrow.parquet as pq
    files=metadata.get('data_files',(metadata['data'],))
    return pq.read_table([str(path) for path in files],columns=columns,
                         filters=[('episode_index','=',int(episode)),*filters])
