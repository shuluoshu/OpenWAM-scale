import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.episode_data import read_episode_table,read_episode_rows


def parts(tmp_path,first=(0,1),second=(2,3),*,statistics=True):
    info={'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'}
    ep={'episode_index':7,'length':4,'data/chunk_index':0,'data/file_index':0,
        'dataset_from_index':100,'dataset_to_index':104}
    paths=[]
    for chunk,indices in enumerate((first,second)):
        path=tmp_path/f'data/chunk-{chunk:03d}/file-000.parquet'
        path.parent.mkdir(parents=True)
        rows=[{'episode_index':7,'frame_index':i,'index':None if i is None else 100+i,'value':i} for i in indices]
        pq.write_table(pa.Table.from_pylist(rows),path,write_statistics=statistics)
        paths.append(path)
    return info,ep,paths


@pytest.mark.parametrize('statistics',[True,False])
def test_projected_split_read_and_terminal_row(tmp_path,statistics):
    info,ep,paths=parts(tmp_path,statistics=statistics)
    table,files=read_episode_table(tmp_path,info,ep,['value'])
    assert table.column_names==['value'] and table['value'].to_pylist()==[0,1,2,3]
    assert files==tuple(paths)
    selected=read_episode_rows({'data':paths[0],'data_files':files},7,columns=['frame_index','index'],filters=[('frame_index','=',3)])
    assert selected.to_pylist()==[{'frame_index':3,'index':103}]


@pytest.mark.parametrize('first,second',[
    ((0,None),(2,3)), ((0,1),(2,None)), ((0,1),(2,4)),
    ((0,1),(1,2,3)), ((0,1),(3,)),
])
def test_malformed_rows_are_rejected_without_dropping(tmp_path,first,second):
    info,ep,_=parts(tmp_path,first,second)
    with pytest.raises(ValueError,match='dense unique complete'):
        read_episode_table(tmp_path,info,ep,['value'])


def test_metadata_must_point_to_first_episode_part(tmp_path):
    info,ep,_=parts(tmp_path,(2,3),(0,1))
    with pytest.raises(ValueError,match='first episode part'):
        read_episode_table(tmp_path,info,ep,['value'])


def test_single_file_does_not_enumerate_other_shards(tmp_path,monkeypatch):
    import openwam.dataloader.episode_data as module
    info,ep,paths=parts(tmp_path,(0,1,2,3),(10,))
    def forbidden(*args):raise AssertionError('single-file reads must not enumerate shards')
    monkeypatch.setattr(module,'_candidate_paths',forbidden)
    table,files=read_episode_table(tmp_path,info,ep,['value'])
    assert len(table)==4 and files==(paths[0],)


def test_split_read_respects_total_byte_budget(tmp_path):
    info,ep,paths=parts(tmp_path)
    one=pq.read_table(paths[0],columns=['value','frame_index','index']).nbytes
    with pytest.raises(ValueError,match='cache_byte_budget'):
        read_episode_table(tmp_path,info,ep,['value'],max_bytes=one+1)
