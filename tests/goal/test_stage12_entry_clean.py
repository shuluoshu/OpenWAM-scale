"""DLC admission rejects launcher changes and untracked executable sources."""
import os
from pathlib import Path
import subprocess
import pytest

ENTRY=Path(__file__).resolve().parents[2]/'scripts/pai/stage12_entry.sh'


@pytest.mark.parametrize('change', ['tracked_launcher','untracked_python','ignored_python','root_python'])
def test_entry_rejects_non_exact_runtime_checkout(tmp_path,change):
    (tmp_path/'scripts/pai').mkdir(parents=True)
    entry=tmp_path/'scripts/pai/stage12_entry.sh';entry.write_bytes(ENTRY.read_bytes())
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    subprocess.run(['git','-C',str(tmp_path),'add','.'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Test','-c','user.email=test@example.com','commit','-qm','initial'],check=True)
    revision=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip()
    if change=='tracked_launcher':entry.write_text(entry.read_text()+'\n# changed\n')
    elif change=='root_python':(tmp_path/'sitecustomize.py').write_text('changed = True\n')
    else:
        if change=='ignored_python':(tmp_path/'.git/info/exclude').write_text('scripts/extra.py\n')
        (tmp_path/'scripts/extra.py').write_text('changed = True\n')
    env=dict(os.environ,OPENWAM_CODE_DIR=str(tmp_path),OPENWAM_PYTHON='/bin/true',OPENWAM_EXPECTED_COMMIT=revision)
    result=subprocess.run(['bash',str(entry),'check'],env=env,capture_output=True,text=True)
    assert result.returncode!=0
    assert 'uncommitted' in result.stderr or 'Untracked' in result.stderr


def test_entry_never_regenerates_selection():
    assert 'prepare_stage12_selection.py' not in ENTRY.read_text()
