from pathlib import Path

import pandas as pd
import pytest

from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader


def _reader(tmp_path: Path, tasks: pd.DataFrame | None = None) -> LeRobotV3Reader:
    if tasks is not None:
        (tmp_path / "meta").mkdir()
        tasks.to_parquet(tmp_path / "meta" / "tasks.parquet")
    reader = LeRobotV3Reader.__new__(LeRobotV3Reader)
    reader._dataset_dir = tmp_path
    reader._dataset_id = "prompt-schema-test"
    return reader


def test_explicit_task_column_with_range_index_resolves_prompt(tmp_path: Path):
    reader = _reader(tmp_path, pd.DataFrame({"task_index": [7, 3], "task": [" pick cup ", "place cup"]}))
    reader._load_prompts()

    assert reader._resolve_prompt({}, pd.DataFrame({"task_index": [7]})) == "pick cup"
    assert reader._resolve_prompt({}, pd.DataFrame({"task_index": [3]})) == "place cup"


def test_legacy_text_index_resolves_prompt(tmp_path: Path):
    reader = _reader(tmp_path, pd.DataFrame({"task_index": [7]}, index=pd.Index([" pick cup "], name="task")))
    reader._load_prompts()

    assert reader._resolve_prompt({}, pd.DataFrame({"task_index": [7]})) == "pick cup"


def test_explicit_task_column_takes_precedence_over_text_index(tmp_path: Path):
    reader = _reader(tmp_path, pd.DataFrame({"task_index": [7], "task": ["pick cup"]}, index=["stale prompt"]))
    reader._load_prompts()

    assert reader._resolve_prompt({}, pd.DataFrame({"task_index": [7]})) == "pick cup"


@pytest.mark.parametrize("text", [None, 42, "", " \t\n"])
@pytest.mark.parametrize("schema", ["column", "index"])
def test_invalid_prompt_text_is_rejected_at_load(tmp_path: Path, text, schema: str):
    tasks = (
        pd.DataFrame({"task_index": [7], "task": [text]}, index=["valid but stale fallback"])
        if schema == "column"
        else pd.DataFrame({"task_index": [7]}, index=[text])
    )
    reader = _reader(tmp_path, tasks)

    with pytest.raises(ValueError, match=r"tasks\.parquet.*task_index=7.*non-empty string"):
        reader._load_prompts()


@pytest.mark.parametrize("schema", ["column", "index"])
def test_conflicting_duplicate_task_ids_are_rejected(tmp_path: Path, schema: str):
    tasks = pd.DataFrame({"task_index": [7, 7], "task": ["pick cup", "place cup"]})
    if schema == "index":
        tasks = tasks.set_index("task")
    reader = _reader(tmp_path, tasks)

    with pytest.raises(ValueError, match=r"tasks\.parquet.*conflicting.*task_index=7"):
        reader._load_prompts()


@pytest.mark.parametrize("schema", ["column", "index"])
def test_identical_duplicate_task_ids_are_allowed(tmp_path: Path, schema: str):
    tasks = pd.DataFrame({"task_index": [7, 7], "task": ["pick cup", "pick cup"]})
    if schema == "index":
        tasks = tasks.set_index("task")
    reader = _reader(tmp_path, tasks)
    reader._load_prompts()

    assert reader._resolve_prompt({}, pd.DataFrame({"task_index": [7]})) == "pick cup"


def test_missing_task_id_still_raises(tmp_path: Path):
    reader = _reader(tmp_path, pd.DataFrame({"task_index": [7], "task": ["pick cup"]}))
    reader._load_prompts()

    with pytest.raises(KeyError, match="task_index=8 not present"):
        reader._resolve_prompt({}, pd.DataFrame({"task_index": [8]}))


def test_required_missing_tasks_file_still_raises(tmp_path: Path):
    reader = _reader(tmp_path)

    with pytest.raises(FileNotFoundError):
        reader._load_prompts()


def test_optional_missing_tasks_file_still_loads_empty_lookup(tmp_path: Path):
    reader = _reader(tmp_path)
    reader.PROMPT_FILE_REQUIRED = False
    reader._load_prompts()

    assert reader._task_idx_to_text == {}
