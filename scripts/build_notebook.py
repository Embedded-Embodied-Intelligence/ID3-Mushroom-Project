#!/usr/bin/env python
"""Build the project notebooks from ``src/`` (single source of truth for the code).

    python scripts/build_notebook.py              # main notebook + workshop notebooks
    python scripts/build_notebook.py --no-facts   # skip pass 1 (placeholders stay visible; dev only)

Synchronisation mechanism
-------------------------
1. Code. Every module of ``src/`` is copied - in dependency order data_io -> preprocessing -> id3 ->
   evaluation -> experiments -> visualization - into its own code cell. Package-internal imports
   (``from .x import``, ``from src.x import``, ``import src``) are removed with :mod:`ast`; the build
   verifies that every removed name is defined by an earlier module and that no two modules define the
   same top-level name. The notebook therefore runs in a clean Google Colab without the ``src/`` folder,
   and there is exactly one implementation of ID3 (``tests/test_notebook.py`` fails on any drift).
2. Narrative numbers. Markdown cells use placeholders ``{{name}}`` or ``{{name:.3f}}``. Pass 1 executes
   a draft of the notebook in a temporary directory and collects the values recorded there with
   ``note(...)``; the placeholders are rendered from these values and the rendered strings are embedded
   in the last cell, which recomputes and compares them on every run of the notebook.
3. Workshop. The participant and instructor notebooks embed ``group_exercise/delivery_data.csv`` as text
   and the same ``src/id3.py`` code.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
MAIN_NOTEBOOK = PROJECT_ROOT / "notebooks" / "ID3_Mushroom_Project.ipynb"
STUDENT_NOTEBOOK = PROJECT_ROOT / "group_exercise" / "ID3_Delivery_Workshop.ipynb"
SOLUTION_NOTEBOOK = PROJECT_ROOT / "group_exercise" / "instructor_private" / "ID3_Delivery_Solution.ipynb"
WORKSHOP_CSV = PROJECT_ROOT / "group_exercise" / "delivery_data.csv"
MODULE_ORDER = ["data_io", "preprocessing", "id3", "evaluation", "experiments", "visualization"]
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*([^}]*?))?\s*\}\}")
FACTS_MARKER = "NARRATIVE_FACTS = None  # filled in by scripts/build_notebook.py"
CONTENT_MARKERS = ("__TEST_INTERPRETATION__", "__ERROR_ANALYSIS__", "__CSV__")  # must never reach a notebook

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Module injection
# ---------------------------------------------------------------------------

def _is_internal_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.level > 0 or (node.module or "").split(".")[0] == "src"
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "src" for alias in node.names)
    return False


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def strip_internal_imports(source: str) -> tuple[str, set[str]]:
    """Remove package-internal import statements; return the code and the names they imported."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    drop: set[int] = set()
    imported: set[str] = set()
    for node in tree.body:
        if _is_internal_import(node):
            drop.update(range(node.lineno - 1, node.end_lineno))
            imported.update(alias.asname or alias.name for alias in node.names)
    nested = [n for n in ast.walk(tree) if _is_internal_import(n) and n not in tree.body]
    if nested:
        raise RuntimeError(f"internal imports must be at module level (line {nested[0].lineno})")
    code = "".join(line for i, line in enumerate(lines) if i not in drop)
    ast.parse(code)
    return code, imported


def injected_modules(names: list[str] = MODULE_ORDER) -> list[tuple[str, str]]:
    """``(module, code)`` pairs ready for notebook cells, with namespace consistency checks."""
    defined: dict[str, str] = {}
    result = []
    for name in names:
        path = SRC / f"{name}.py"
        code, imported = strip_internal_imports(path.read_text(encoding="utf-8"))
        missing = sorted(n for n in imported if n not in defined)
        if missing:
            raise RuntimeError(f"src/{name}.py imports {missing} that no earlier module defines")
        own = _top_level_names(ast.parse(code))
        clashes = sorted(n for n in own if n in defined)
        if clashes:
            raise RuntimeError(f"src/{name}.py redefines {clashes} already defined in "
                               f"{sorted({defined[n] for n in clashes})} - names must be unique in the notebook")
        defined.update(dict.fromkeys(own, name))
        header = (f"# @title src/{name}.py — kod modułu wstrzyknięty automatycznie\n"
                  f"# Źródło: src/{name}.py · generator: scripts/build_notebook.py · nie edytuj ręcznie.\n")
        result.append((name, header + code.strip() + "\n"))
    return result


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

def notebook_metadata(title: str) -> dict:
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": [], "toc_visible": True, "name": title},
        "title": title,
    }


def make_cells(spec: list[dict], modules: list[tuple[str, str]]) -> list[nbformat.NotebookNode]:
    cells = []
    for item in spec:
        if item["type"] == "modules":
            for name, code in modules:
                cell = nbformat.v4.new_code_cell(code)
                cell.metadata.update({"tags": ["injected-module"], "jupyter": {"source_hidden": True},
                                      "cellView": "form", "id3_module": name})
                cells.append(cell)
            continue
        if item["type"] == "markdown":
            cell = nbformat.v4.new_markdown_cell(item["source"])
        else:
            cell = nbformat.v4.new_code_cell(item["source"])
        if item.get("tags"):
            cell.metadata["tags"] = list(item["tags"])
        cell.metadata.update(item.get("metadata", {}))
        cells.append(cell)
    for i, cell in enumerate(cells):  # stable ids keep diffs of the generated file readable
        cell["id"] = f"cell-{i:03d}"
    return cells


def format_fact(value, spec: str) -> str:
    return format(value, spec) if spec else str(value)


def render_placeholders(nb: nbformat.NotebookNode, facts: dict) -> dict[str, str]:
    """Replace placeholders in markdown cells; return ``{"name|spec": rendered}`` for the check cell."""
    used: dict[str, str] = {}

    def substitute(match: re.Match) -> str:
        name, spec = match.group(1), (match.group(2) or "").strip()
        if name not in facts:
            raise KeyError(f"placeholder '{name}' has no value recorded with note(...) in the notebook")
        rendered = format_fact(facts[name], spec)
        used[f"{name}|{spec}"] = rendered
        return rendered

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = PLACEHOLDER.sub(substitute, cell.source)
            if "{{" in cell.source:
                raise ValueError(f"unrendered placeholder left in markdown: {cell.source[:120]!r}")
    return dict(sorted(used.items()))


def embed_facts(nb: nbformat.NotebookNode, rendered: dict[str, str]) -> None:
    literal = "NARRATIVE_FACTS = " + json.dumps(rendered, ensure_ascii=False, indent=1)
    hits = [c for c in nb.cells if c.cell_type == "code" and FACTS_MARKER in c.source]
    if len(hits) != 1:
        raise RuntimeError("expected exactly one narrative-facts cell")
    hits[0].source = hits[0].source.replace(FACTS_MARKER, literal + "  # wygenerowane przez build_notebook.py")


def execute_draft(nb: nbformat.NotebookNode, workdir: Path, timeout: int = 1200) -> dict:
    """Pass 1: run the draft in ``workdir`` (copy of the data only) and return the recorded facts."""
    from nbclient import NotebookClient

    (workdir / "data").mkdir(parents=True)
    (workdir / "notebooks").mkdir()
    shutil.copy2(PROJECT_ROOT / "data" / "mushroom.zip", workdir / "data" / "mushroom.zip")
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(workdir / "notebooks")}})
    client.execute()
    return json.loads((workdir / "outputs" / "narrative_facts.json").read_text(encoding="utf-8"))


def build_main(with_facts: bool = True) -> Path:
    import notebook_main

    modules = injected_modules()
    spec = notebook_main.cells(listings=notebook_main.listings(SRC / "id3.py"),
                               workshop_csv=WORKSHOP_CSV.read_text(encoding="utf-8"))
    nb = nbformat.v4.new_notebook(cells=make_cells(spec, modules),
                                  metadata=notebook_metadata("ID3_Mushroom_Project"))
    if with_facts:
        draft = nbformat.reads(nbformat.writes(nb), as_version=4)
        with tempfile.TemporaryDirectory(prefix="id3_build_") as tmp:
            facts = execute_draft(draft, Path(tmp))
        rendered = render_placeholders(nb, facts)
        embed_facts(nb, rendered)
        print(f"  narrative facts: {len(rendered)} placeholders rendered from pass 1")
    nbformat.validate(nb)
    MAIN_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, MAIN_NOTEBOOK)
    return MAIN_NOTEBOOK


def build_workshop() -> tuple[Path, Path]:
    import notebook_workshop

    id3_code = dict(injected_modules(["id3"]))["id3"]
    csv_text = WORKSHOP_CSV.read_text(encoding="utf-8")
    paths = []
    for variant, path in (("student", STUDENT_NOTEBOOK), ("solution", SOLUTION_NOTEBOOK)):
        spec = notebook_workshop.cells(variant=variant, csv_text=csv_text, id3_code=id3_code)
        nb = nbformat.v4.new_notebook(cells=make_cells(spec, []),
                                      metadata=notebook_metadata(path.stem))
        nbformat.validate(nb)
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, path)
        paths.append(path)
    (SOLUTION_NOTEBOOK.parent / "ANSWER_KEY.md").write_text(notebook_workshop.answer_key(csv_text), encoding="utf-8")
    return paths[0], paths[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-facts", action="store_true", help="skip pass 1 (development only)")
    args = parser.parse_args()
    print("Building notebooks from src/ ...")
    print(f"  modules injected in order: {' -> '.join(MODULE_ORDER)}")
    print(f"  main notebook: {build_main(with_facts=not args.no_facts).relative_to(PROJECT_ROOT)}")
    student, solution = build_workshop()
    print(f"  workshop (participant): {student.relative_to(PROJECT_ROOT)}")
    print(f"  workshop (instructor):  {solution.relative_to(PROJECT_ROOT)}")
    print("Done. Next: python scripts/validate_notebook.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
