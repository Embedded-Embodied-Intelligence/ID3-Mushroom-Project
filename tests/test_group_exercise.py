"""Workshop: synthetic data properties, participant notebook without answers, solution passing all checks."""
import json
import re
import sys

import nbformat
import pandas as pd
import pytest
from nbclient import NotebookClient

from tests.conftest import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_notebook import SOLUTION_NOTEBOOK, STUDENT_NOTEBOOK, WORKSHOP_CSV, injected_modules  # noqa: E402
from notebook_workshop import FEATURES, NEW_PARCEL, TARGET, expected_answers, leaked_answers  # noqa: E402

from src.id3 import ID3Classifier, information_gain_table  # noqa: E402

CSV_TEXT = WORKSHOP_CSV.read_text(encoding="utf-8")


def _execute(nb, path):
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path)}}).execute()
    return "".join(o.get("text", "") for c in nb.cells for o in c.get("outputs", []))


def test_dataset_follows_the_workshop_specification():
    df = pd.read_csv(WORKSHOP_CSV)
    assert 24 <= len(df) <= 32 and list(df.columns) == [*FEATURES, TARGET]
    assert set(df[TARGET]) == {"on_time", "delayed"}
    missing = df.isna().sum()
    assert 2 <= missing.sum() <= 4 and (missing > 0).sum() == 2
    assert set(df["distance"].dropna()) == {"short", "medium", "long"}
    assert set(df["shipping"].dropna()) == {"standard", "express"}


def test_root_is_unique_and_tree_is_non_trivial():
    answers = expected_answers(CSV_TEXT)
    gains = sorted(answers["gains"].values(), reverse=True)
    assert gains[0] - gains[1] > 0.1
    df = pd.read_csv(WORKSHOP_CSV)
    filled = df.fillna({c: df[c].mode()[0] for c in df.columns if df[c].isna().any()})
    assert not filled[FEATURES].duplicated().any()
    model = ID3Classifier().fit(filled[FEATURES], filled[TARGET])
    assert model.depth_ >= 2 and model.node_count_ <= 15
    assert not (filled[FEATURES] == pd.Series(NEW_PARCEL)).all(axis=1).any()
    assert information_gain_table(filled[FEATURES], filled[TARGET]).iloc[0]["feature"] == answers["root"]


def test_participant_notebook_contains_no_answer_key():
    nb = nbformat.read(STUDENT_NOTEBOOK, as_version=4)
    text = nbformat.writes(nb)
    assert leaked_answers(text, CSV_TEXT) == []
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    hidden = [c for c in nb.cells if c.metadata.get("cellView") == "form" or c.metadata.get("jupyter", {}).get("source_hidden")]
    assert len(hidden) == 1 and "helpers" in hidden[0].metadata.get("tags", [])
    assert leaked_answers(hidden[0].source, CSV_TEXT) == []
    todos = {m for c in nb.cells if c.cell_type == "code" for m in re.findall(r"# TODO (\d)", c.source)}
    assert todos == {"1", "2", "3", "4"}


def test_participant_notebook_is_self_contained():
    nb = nbformat.read(STUDENT_NOTEBOOK, as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert CSV_TEXT.strip() in code
    assert not re.search(r"^\s*(from\s+src\b|import\s+src\b|drive\.mount)", code, re.M)
    id3_body = injected_modules(["id3"])[0][1].split("\n", 2)[2].strip()
    assert id3_body in code, "workshop mini-ID3 must be the project ID3 code (src/id3.py)"


def test_participant_notebook_runs_with_unfilled_todos(tmp_path):
    nb = nbformat.read(STUDENT_NOTEBOOK, as_version=4)
    streams = _execute(nb, tmp_path)
    assert streams.count("⏳ TODO") == 4 and "✅" not in streams


def test_solution_notebook_passes_every_check(tmp_path):
    nb = nbformat.read(SOLUTION_NOTEBOOK, as_version=4)
    streams = _execute(nb, tmp_path)
    assert all(f"✅ TODO {i} zaliczone" in streams for i in range(1, 5))


def test_answer_key_matches_the_computation():
    key = (SOLUTION_NOTEBOOK.parent / "ANSWER_KEY.md").read_text(encoding="utf-8")
    answers = expected_answers(CSV_TEXT)
    assert f"**Korzeń: `{answers['root']}`**" in key
    assert f"Predykcja: **{answers['prediction']}**" in key
    assert f"{answers['entropy']:.4f}" in key
