"""Read audited external frame intervals without rewriting source datasets."""
from pathlib import Path


def external_subtasks(repo,episode,length):
    """Return None without a sidecar, or the sidecar's genuine subtask intervals."""
    import pyarrow.parquet as pq
    meta=repo.get('native_metadata',{})
    path=meta.get('annotation_file')
    if not path:return None
    if not Path(path).is_file():raise FileNotFoundError(f'external annotation source missing: {path}')
    source=meta.get('annotation_filter',{}).get('repo',repo['repo'])
    columns=['episode_index','subtask_index','start_frame','end_frame','subtask']
    rows=pq.read_table(path,columns=columns,filters=[('repo','=',source),('episode_index','=',episode)]).to_pylist()
    result=[];last=0
    for row in sorted(rows,key=lambda row:row['start_frame']):
        a,b=row['start_frame'],row['end_frame'];text=row['subtask'];index=row['subtask_index']
        if type(a) is not int or type(b) is not int or not 0<=a<b<=length or a<last:
            raise ValueError(f'malformed external annotation interval: {source}:{episode}')
        if type(index) is not int or not isinstance(text,str) or not text.strip():raise ValueError('malformed external subtask text/index')
        result.append(dict(start=a,end_exclusive=b,text=text,subtask_index=index));last=b
    return result
