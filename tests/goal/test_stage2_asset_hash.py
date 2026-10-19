import hashlib
from pathlib import Path

def test_goal_asset_hash_reads_bounded_chunks(tmp_path,monkeypatch):
    from openwam.model.goal.runtime import goal_asset_sha256
    data=bytes(range(256))*16385
    path=tmp_path/'expert.safetensors';path.write_bytes(data)
    original=Path.open;reads=[]
    class Guard:
        def __enter__(self):self.stream=original(path,'rb');return self
        def __exit__(self,*args):self.stream.close()
        def read(self,size=-1):
            assert 0<size<=1024*1024,'large goal asset must never be read as one byte buffer'
            reads.append(size);return self.stream.read(size)
    monkeypatch.setattr(Path,'open',lambda self,*a,**kw:Guard() if self==path else original(self,*a,**kw))
    assert goal_asset_sha256(path)==hashlib.sha256(data).hexdigest()
    assert len(reads)>=5
