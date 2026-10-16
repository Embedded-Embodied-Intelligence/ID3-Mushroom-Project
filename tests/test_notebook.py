"""The main notebook: metadata, synchronisation with src/, Colab-readiness and an isolated run without src/."""
import copy
import json
import re
import shutil
import sys

import nbformat
import pytest
from nbclient import NotebookClient

from tests.conftest import DATA_ZIP, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_notebook import (CONTENT_MARKERS, MAIN_NOTEBOOK, MODULE_ORDER, PLACEHOLDER,  # noqa: E402
                            injected_modules, strip_internal_imports)


@pytest.fixture(scope="module")
def notebook():
    if not MAIN_NOTEBOOK.exists():
        pytest.fail("notebook missing - run: python scripts/build_notebook.py")
    return nbformat.read(MAIN_NOTEBOOK, as_version=4)


def test_metadata_and_schema(notebook):
    nbformat.validate(notebook)
    assert notebook.metadata["kernelspec"]["name"] == "python3"
    assert notebook.metadata["language_info"]["name"] == "python"


def test_twelve_chapters_in_order(notebook):
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    positions = [markdown.find(f'<a id="rozdzial-{i:02d}"></a>') for i in range(12)]
    assert all(p >= 0 for p in positions) and positions == sorted(positions)


def test_no_package_imports_and_modules_in_sync_with_src(notebook):
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert not re.search(r"^\s*(from\s+src\b|import\s+src\b|from\s+\.\w*\s+import)", code, re.M)
    cells = [c for c in notebook.cells if "injected-module" in c.metadata.get("tags", [])]
    assert [c.metadata["id3_module"] for c in cells] == MODULE_ORDER
    assert [c.source for c in cells] == [code for _, code in injected_modules()], \
        "notebook out of sync with src/ - run: python scripts/build_notebook.py"


def test_import_stripping_handles_multiline_and_absolute_forms():
    source = ("from __future__ import annotations\nimport numpy as np\nfrom .id3 import (\n    ID3Classifier,\n"
              "    entropy,\n)\nfrom src.data_io import load_mushroom\nimport src.preprocessing\nX = 1\n")
    code, names = strip_internal_imports(source)
    assert "from ." not in code and "src" not in code and "import numpy as np" in code
    assert names == {"ID3Classifier", "entropy", "load_mushroom", "src.preprocessing"}


def test_narrative_is_rendered_and_checked(notebook):
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    assert not PLACEHOLDER.search(markdown)
    assert not [m for m in CONTENT_MARKERS if m in nbformat.writes(notebook)]
    facts_cell = [c.source for c in notebook.cells if c.cell_type == "code" and "NARRATIVE_FACTS = {" in c.source]
    assert len(facts_cell) == 1
    literal = facts_cell[0].split("NARRATIVE_FACTS = ", 1)[1].split("  # wygenerowane", 1)[0]
    assert len(json.loads(literal)) > 100


def test_saved_outputs_have_no_errors(notebook):
    outputs = [o for c in notebook.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    if not outputs:
        pytest.skip("notebook not executed yet - run: python scripts/validate_notebook.py")
    assert not [o for o in outputs if o.get("output_type") == "error"]
    assert sum("image/png" in o.get("data", {}) for o in outputs) >= 12


def test_runs_end_to_end_without_src_folder(tmp_path, notebook):
    clean = copy.deepcopy(notebook)
    for cell in clean.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    (tmp_path / "data").mkdir()
    (tmp_path / "notebooks").mkdir()
    shutil.copy2(DATA_ZIP, tmp_path / "data" / "mushroom.zip")
    assert not (tmp_path / "src").exists() and not (tmp_path / "tests").exists()
    NotebookClient(clean, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(tmp_path / "notebooks")}}).execute()
    assert (tmp_path / "outputs" / "experiment_results.csv").exists()
    assert (tmp_path / "outputs" / "figures" / "final_tree.png").stat().st_size > 20_000
    errors = [o for c in clean.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors
