"""Install the reviewed repository skill into this runtime's personal skills."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "skills/openwam-visualization"
TARGET = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "skills/openwam-visualization"
assert (SOURCE / "SKILL.md").is_file()
assert not TARGET.exists(), f"Skill already exists; review before replacing: {TARGET}"
shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
files = []
for source in sorted(SOURCE.rglob("*")):
    if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
        continue
    relative = source.relative_to(SOURCE)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256((TARGET / relative).read_bytes()).hexdigest() == digest
    files.append(dict(path=str(relative), sha256=digest))
report = dict(installed=True, source=str(SOURCE), target=str(TARGET), files=files)
out = ROOT / "outputs/skill_validation_20261015"
out.mkdir(exist_ok=True)
(out / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(dict(installed=True, target=str(TARGET), files=len(files)), indent=2))
