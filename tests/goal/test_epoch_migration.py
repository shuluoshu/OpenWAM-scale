import json
import sqlite3
import torch
import pytest


def test_subset_index_keeps_episode_digests_and_source_stamps(tmp_path):
    from openwam.train.epoch_migration import subset_index
    parent = tmp_path/'parent.sqlite'
    with sqlite3.connect(parent) as c:
        c.executescript('''
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE episodes(repo_id INTEGER,episode INTEGER,excluded INTEGER,digest TEXT,PRIMARY KEY(repo_id,episode));
        CREATE TABLE sources(path TEXT PRIMARY KEY,size INTEGER,mtime_ns INTEGER);
        CREATE TABLE episode_sources(repo_id INTEGER,episode INTEGER,path TEXT,PRIMARY KEY(repo_id,episode,path));
        CREATE TABLE runs(sample_start INTEGER PRIMARY KEY,count INTEGER NOT NULL,repo_id INTEGER,episode INTEGER,level TEXT,segment TEXT,grid_start INTEGER,numerator INTEGER,denominator INTEGER);
        ''')
        c.executemany('INSERT INTO metadata VALUES (?,?)', [('complete','true'),('total_samples','9')])
        for i in range(3):
            c.execute('INSERT INTO episodes VALUES (?,?,?,?)',(i,0,i,f'digest-{i}'))
            c.execute('INSERT INTO sources VALUES (?,?,?)',(f'file-{i}',i,123+i))
            c.execute('INSERT INTO episode_sources VALUES (?,?,?)',(i,0,f'file-{i}'))
            c.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)',(i*3,3,i,0,'task','{}',8,10,1))
    child = tmp_path/'child.sqlite'
    result = subset_index(parent,child,{0:0,2:1},{'source_yaml_sha256':'new'})
    assert result['excluded_sample_ranges']==[[3,6]]
    assert result['total_samples']==6
    with sqlite3.connect(child) as c:
        assert c.execute('SELECT * FROM episodes ORDER BY repo_id').fetchall()==[(0,0,0,'digest-0'),(1,0,2,'digest-2')]
        assert c.execute('SELECT * FROM sources ORDER BY path').fetchall()==[('file-0',0,123),('file-2',2,125)]
        assert c.execute('SELECT sample_start,count,repo_id,grid_start FROM runs ORDER BY sample_start').fetchall()==[(0,3,0,8),(3,3,1,8)]
    with pytest.raises(FileExistsError):
        subset_index(parent,child,{0:0},{})


def test_migration_preserves_optimizer_and_original_checkpoint(tmp_path):
    from openwam.train.epoch_migration import migrate_checkpoint
    from openwam.train.goal_epoch import sha256
    from openwam.train.epoch_order import sampling_contract, retained_parent_cursor
    parent = tmp_path/'parent';parent.mkdir()
    old=dict(total_samples=10,world_size=1,seed=42,representation='visual_history_masked_sa',data_loading={'batch_size':2},sampling=sampling_contract({},10))
    saved=dict(identity=old,cursor=4,update=2,optimizer={'masters':[torch.tensor([1.25])],'adam':{'step':torch.tensor(2.),'moment':torch.tensor([.7])}})
    torch.save(saved,parent/'training.pt');(parent/'visual_history.safetensors').write_bytes(b'unchanged-weight-fixture')
    files={p.name:sha256(p) for p in parent.iterdir()}
    (parent/'manifest.json').write_text(json.dumps(dict(identity=old,cursor=4,update=2,files=files)))
    contract=sampling_contract(dict(sample_order='filtered_global_shuffle',parent_total_samples=10,excluded_sample_ranges=[[2,5]]),7)
    identity={**old,'total_samples':7,'sampling':contract}
    child=migrate_checkpoint(parent,tmp_path/'new-run',identity)
    actual=torch.load(child/'training.pt',weights_only=True)
    assert actual['identity']==identity and actual['update']==2
    assert actual['cursor']==retained_parent_cursor(contract,4)
    torch.testing.assert_close(actual['optimizer']['masters'][0],saved['optimizer']['masters'][0],rtol=0,atol=0)
    torch.testing.assert_close(actual['optimizer']['adam']['moment'],saved['optimizer']['adam']['moment'],rtol=0,atol=0)
    assert all(sha256(parent/name)==digest for name,digest in files.items())
    assert sha256(child/'visual_history.safetensors')==files['visual_history.safetensors']
    assert json.loads((child/'migration.json').read_text())['parent_update']==2


def test_migration_rejects_changed_checkpoint_before_publish(tmp_path):
    from openwam.train.epoch_migration import migrate_checkpoint
    parent=tmp_path/'parent';parent.mkdir();(parent/'training.pt').write_bytes(b'bad')
    (parent/'manifest.json').write_text(json.dumps(dict(files={'training.pt':'0'*64})))
    with pytest.raises(ValueError,match='checksum'):
        migrate_checkpoint(parent,tmp_path/'child',{})
    assert not (tmp_path/'child/latest.json').exists()
