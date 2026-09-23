from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_usage_guide_documents_exact_end_to_end_commands():
    guide = ROOT / "assets/openwam_usage_docs/openwam-alpha-data-adaptation.md"
    text = guide.read_text(encoding="utf-8")
    for token in (
        "openwam_alpha_v0_1_smoke",
        "prepare_openwam_eef_smoke.py",
        "open_loop_inference.py",
        "visualize_model_inputs.py",
        "canonical_eef_robodojo_openloop.yaml",
        "canonical_eef_robotwin_openloop.yaml",
        "absolute EEF20",
        "0:10",
        "34:44",
    ):
        assert token in text
    for flag in ("--bucket", "--episode-index", "--profile", "--checkpoint-dir", "--denoise-steps"):
        assert flag in text


def test_comparison_html_records_verified_moravla_and_openwam_contracts():
    page = ROOT / "docs/pretrain_vs_openwam_data_pipeline.html"
    text = page.read_text(encoding="utf-8")
    for token in ("qwena1_nogen", "36D", "head_camera", "FRAME_SAMPLE_FPS=5", "EEF20", "80D", "96D"):
        assert token in text
    for relative in re.findall(r"(?:src|href)=[\"']([^\"'#]+)[\"']", text):
        if "://" not in relative and not relative.startswith("javascript:"):
            assert (page.parent / relative).resolve().exists(), relative


def test_original_external_comparison_was_not_overwritten():
    original = Path("/mnt/data/cpfs/b5/yeqingda/pretrain_vs_openwam_comparison.html")
    assert original.is_file()
    assert original.resolve() != (ROOT / "docs/pretrain_vs_openwam_data_pipeline.html").resolve()
