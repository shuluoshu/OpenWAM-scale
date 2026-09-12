# OpenWAM Code Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Chinese HTML guide that explains OpenWAM's current code paths, entrypoints, benchmark coverage, and compatibility of the Conda environments installed under `/root/miniconda3/envs/`.

**Architecture:** The deliverable is one offline HTML document generated from evidence collected directly from repository source/configuration and read-only environment probes. Its embedded CSS provides the layout and status language, while a small embedded JavaScript block provides navigation highlighting, section folding, and benchmark/environment filtering without external dependencies.

**Tech Stack:** HTML5, embedded CSS, vanilla JavaScript, Python standard-library validation, shell/Conda read-only inspection.

## Global Constraints

- Create one self-contained HTML file; do not load fonts, CSS, scripts, images, or APIs from the network.
- Write explanations in Chinese while preserving source identifiers, configuration keys, paths, and shell commands in English.
- Do not alter OpenWAM runtime code or configuration.
- Do not install packages, download weights/data, launch training, or launch benchmark simulators.
- Distinguish built-in evaluation clients from training-only dataset adapters and externally evaluated integrations.
- Base environment recommendations on both declared version constraints and lightweight import/version probes.
- Do not claim full training or benchmark success from static or import-only checks.

---

## File Structure

- Create: `docs/openwam_code_guide.html` — complete offline report, including content, styling, navigation, filtering, and source references.
- Modify: `docs/superpowers/plans/2026-09-12-openwam-code-guide.md` — mark completed checklist items during execution.
- Read only: `README.md`, `pyproject.toml`, `configs/**`, `scripts/**`, `openwam/**`, `benchmarks/**`, `/root/miniconda3/envs/**` — evidence sources; no edits.

### Task 1: Build the verified evidence inventory

**Files:**
- Read: `README.md`
- Read: `pyproject.toml`
- Read: `configs/train.yaml`
- Read: `configs/deploy.yaml`
- Read: `openwam/dataloader/registry.py`
- Read: `openwam/model/architectures/registry.py`
- Read: `openwam/model/video_backbone/registry.py`
- Read: `openwam/train/openwam_trainer.py`
- Read: `openwam/deploy/server.py`
- Read: `benchmarks/README.md`

**Interfaces:**
- Consumes: repository source/configuration at commit `42a61cb` or its descendant, plus Python executables under `/root/miniconda3/envs/*/bin/python`.
- Produces: verified facts to embed directly into `docs/openwam_code_guide.html`: call-chain nodes, entrypoint commands, registry keys, benchmark classifications, dependency constraints, installed versions, and import-probe outcomes.

- [ ] **Step 1: Enumerate executable and registry surfaces**

Run:

```bash
rg -n "hydra.main|def main|if __name__|project.scripts|register_dataset|register_architecture|register_video_backbone" scripts openwam benchmarks pyproject.toml
```

Expected: hits identify the training launcher, deployment CLI/server, asset/statistics tools, benchmark entrypoints, and registry declarations.

- [ ] **Step 2: Reconcile benchmark documentation with actual files**

Run:

```bash
find benchmarks -mindepth 1 -maxdepth 2 -type f | sort
find configs/dataloader -type f | sort
```

Expected: built-in evaluation directories for RoboTwin, LIBERO, LIBERO-plus, RoboCasa365, RoboCasa GR1, VLABench, and EBench; a RoboDojo guide that delegates evaluation to XPolicyLab; additional pretraining dataloaders that are not mislabeled as benchmark clients.

- [ ] **Step 3: Probe each existing Conda environment without mutation**

Run:

```bash
/root/miniconda3/bin/conda env list
for env_dir in /root/miniconda3/envs/*; do
  test -x "$env_dir/bin/python" || continue
  "$env_dir/bin/python" -c 'import importlib.metadata as m, sys; names=("torch","torchvision","transformers","diffusers","hydra-core","accelerate","deepspeed","websockets"); print(sys.version.split()[0], {n:m.version(n) for n in names if any((d.metadata.get("Name") or "").lower()==n for d in m.distributions())})'
done
```

Expected: version inventories for `fastwam`, `hamer`, `lerobot`, `lerobot_052`, and `wilor`, with no package installation or environment changes.

- [ ] **Step 4: Run focused import probes in the leading candidate environment**

Run:

```bash
/root/miniconda3/envs/fastwam/bin/python -c 'import torch, torchvision, hydra, accelerate, deepspeed, websockets; print("core imports OK")'
PYTHONPATH=. /root/miniconda3/envs/fastwam/bin/python -c 'import openwam; from openwam.dataloader.registry import list_datasets; print(list_datasets())'
```

Expected: either explicit successful output or a concrete missing/incompatible import error to record in the report; no claim beyond the tested imports.

### Task 2: Author the standalone HTML report

**Files:**
- Create: `docs/openwam_code_guide.html`

**Interfaces:**
- Consumes: Task 1's verified facts and source paths.
- Produces: one valid UTF-8 HTML5 document with IDs `overview`, `repository-map`, `call-chains`, `modules`, `entrypoints`, `benchmarks`, `environment`, `quickstart`, and `pitfalls`; filter controls use `data-filter-group` and rows use `data-search`.

- [ ] **Step 1: Create the semantic report shell and visual system**

Add a complete document containing `<!doctype html>`, `<meta charset="utf-8">`, responsive viewport metadata, embedded `<style>`, a skip link, fixed desktop navigation, mobile navigation, `<main>`, and the nine required section IDs. Use CSS custom properties for background, panels, text, muted text, borders, accent colors, and status colors. Include print styles that remove navigation and interactive controls while expanding report content.

- [ ] **Step 2: Encode the repository map and three source-derived call chains**

Add source-linked cards for the top-level directories and CSS flow diagrams for:

```text
scripts/train.sh → torchrun → scripts/train.py::main → Hydra compose → build_dataset → OpenWAMTrainer → build_architecture → train loop/checkpoints
scripts/deploy.sh | openwam-serve → openwam.deploy.server::main → load_model → inference engine → WAMPolicy/executor → PolicyServer WebSocket
benchmark simulator → benchmark adapter → WSPolicyClient → PolicyServer → action response → action conversion/controller
```

Each node must state its responsibility and reference at least one concrete source path.

- [ ] **Step 3: Add module, entrypoint, and benchmark matrices**

Document architecture variants (`single_system`, `dual_system`, `tri_system`), video backbones, dataset registry keys, training/deployment subsystems, shell/Python/console entrypoints, and benchmark status. Explicitly separate:

```text
Built-in clients: RoboTwin, LIBERO, LIBERO-plus, RoboCasa365, RoboCasa GR1, VLABench, EBench
External evaluation: RoboDojo via XPolicyLab
Planned only: SimplerEnv, Calvin
Training/pretraining dataset adapters: registry entries including AgiBotWorld, InternData A1, mixture, MUKA Franka, OXE-DROID, RoboCOIN
```

- [ ] **Step 4: Add the environment compatibility matrix and actionable setup**

For every discovered environment, show Python, Torch/CUDA build, Transformers, Diffusers, Accelerate, DeepSpeed, compatibility judgment, and the reason. Recommend either reusing `fastwam` after correcting concrete version gaps or creating a dedicated `openwam` Python 3.10 environment, based on Task 1 results. Include commands for dependency installation, editable install, optional Cosmos setup, asset download, training, deployment, and one client evaluation, while clearly marking unexecuted commands.

- [ ] **Step 5: Add offline interactions**

Embed vanilla JavaScript that:

```javascript
document.querySelectorAll('[data-filter-input]').forEach((input) => {
  input.addEventListener('input', () => {
    const query = input.value.trim().toLowerCase();
    const group = input.dataset.filterInput;
    document.querySelectorAll(`[data-filter-group="${group}"]`).forEach((row) => {
      row.hidden = query !== '' && !row.dataset.search.toLowerCase().includes(query);
    });
  });
});
```

Also implement section folding with `aria-expanded`, copy buttons for command blocks, and current-section navigation highlighting using `IntersectionObserver`; retain usable content when JavaScript is disabled.

### Task 3: Validate report correctness and offline behavior

**Files:**
- Test: `docs/openwam_code_guide.html`

**Interfaces:**
- Consumes: Task 2's completed HTML.
- Produces: validation evidence covering syntax, required structure, offline assets, internal references, and content invariants.

- [ ] **Step 1: Parse HTML and assert required sections**

Run:

```bash
python - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

path = Path("docs/openwam_code_guide.html")
text = path.read_text(encoding="utf-8")

class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])

parser = Parser()
parser.feed(text)
required = {"overview", "repository-map", "call-chains", "modules", "entrypoints", "benchmarks", "environment", "quickstart", "pitfalls"}
assert required <= parser.ids, required - parser.ids
assert "<html" in text.lower() and "</html>" in text.lower()
print("HTML structure OK")
PY
```

Expected: `HTML structure OK`.

- [ ] **Step 2: Reject external runtime dependencies and placeholders**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

text = Path("docs/openwam_code_guide.html").read_text(encoding="utf-8")
assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\x27]https?://', text, re.I)
for marker in ("T" + "BD", "TO" + "DO", "待补充", "lorem ipsum"):
    assert marker.lower() not in text.lower(), marker
print("Offline and placeholder checks OK")
PY
```

Expected: `Offline and placeholder checks OK`.

- [ ] **Step 3: Check anchors, benchmark labels, and environment coverage**

Run:

```bash
python - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

text = Path("docs/openwam_code_guide.html").read_text(encoding="utf-8")
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=set(); self.refs=[]
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if attrs.get("id"): self.ids.add(attrs["id"])
        if attrs.get("href", "").startswith("#"): self.refs.append(attrs["href"][1:])
p=Links(); p.feed(text)
assert set(p.refs) <= p.ids, set(p.refs)-p.ids
for name in ("RoboTwin", "LIBERO", "LIBERO-plus", "RoboCasa365", "RoboCasa GR1", "VLABench", "EBench", "RoboDojo", "fastwam", "hamer", "lerobot", "lerobot_052", "wilor"):
    assert name in text, name
print("Content invariants OK")
PY
```

Expected: `Content invariants OK`.

- [ ] **Step 4: Review the final diff and commit**

Run:

```bash
git diff --check
git status --short
git diff --stat
git add docs/openwam_code_guide.html docs/superpowers/plans/2026-09-12-openwam-code-guide.md
git commit -m "docs: add OpenWAM source guide"
```

Expected: no whitespace errors; the commit contains the HTML report and completed plan checklist only.
