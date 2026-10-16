"""Behavior checks for the portable skill helpers; no datasets or GPU needed."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills/openwam-visualization"


def invoke(script, *args):
    return subprocess.run([sys.executable, str(SKILL / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True)


def test_plan_does_not_execute_renderer_and_keeps_argument_boundaries(tmp_path):
    repo = tmp_path / "OpenWAM"
    (repo / "openwam/dataloader").mkdir(parents=True)
    (repo / "scripts").mkdir()
    marker = repo / "executed"
    (repo / "scripts/visualize_model_inputs.py").write_text(f"open({str(marker)!r},'w').write('bad')")
    result = invoke("tools.py", "--repo", repo, "plan", "model-inputs", "--",
                    "--profile", "a profile.yaml", "--output-dir", "a report")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["command"][-4:] == ["--profile", "a profile.yaml", "--output-dir", "a report"]
    assert not marker.exists()


def test_no_cli_legacy_renderer_rejects_invented_flags_without_execution():
    result = invoke("tools.py", "--repo", ROOT, "plan", "forward-inputs", "--", "--ids", "08_abc_130k")
    assert result.returncode != 0
    assert "no CLI" in result.stderr


def test_catalog_check_finds_existing_tools_without_importing_them():
    result = invoke("tools.py", "--repo", ROOT, "list", "--check")
    assert result.returncode == 0, result.stderr
    tools = json.loads(result.stdout)["tools"]
    assert tools and all(item["exists"] for item in tools)
    assert any(item["historical"] for item in tools)


def test_local_report_accepts_data_video_and_relative_assets(tmp_path):
    (tmp_path / "trace.json").write_text("{}")
    page = tmp_path / "index.html"
    page.write_text('<video controls src="data:video/mp4;base64,AAAA"></video><a href="trace.json#x">trace</a>')
    result = invoke("validate_report.py", page, "--require-video", "--require-offline")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["video_sources"] == 1 and data["embedded_video_sources"] == 1


def test_report_reports_broken_assets_external_runtime_and_no_video(tmp_path):
    page = tmp_path / "index.html"
    page.write_text('<img src="missing.png"><script src="https://example.com/runtime.js"></script>')
    result = invoke("validate_report.py", page, "--require-video", "--require-offline")
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert data["missing"] and data["external_runtime"] and data["video_sources"] == 0


def test_report_does_not_accept_existing_parent_files_as_portable_assets(tmp_path):
    (tmp_path / "outside.json").write_text("{}")
    folder = tmp_path / "report"
    folder.mkdir()
    page = folder / "index.html"
    page.write_text('<a href="../outside.json">outside</a>')
    result = invoke("validate_report.py", page)
    assert result.returncode != 0
    assert json.loads(result.stdout)["outside_root"]
