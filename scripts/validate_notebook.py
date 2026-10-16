#!/usr/bin/env python
"""Execute and validate the notebooks, export HTML and record the results in outputs/QA_REPORT.md.

    python scripts/validate_notebook.py              # full validation
    python scripts/validate_notebook.py --skip-isolated

1. Main notebook: static checks (schema, kernelspec, 12 chapters, no imports from src/, injected modules
   identical to src/, no unrendered placeholders), execution in place with nbclient (outputs are saved),
   checks of errors, figures and saved outputs, artefacts on disk, HTML export (outputs/notebook.html).
2. Isolated run: the notebook copied together with data/mushroom.zip only - no src/, no tests/ - which
   reproduces a clean Google Colab session (local ZIP instead of the download).
3. Workshop: the participant notebook is executed in a temporary copy (the distributed file stays without
   outputs) and checked for leaked answers; the instructor notebook is executed in place and must pass
   all four checks.
4. Requirement map (8 slide requirements -> evidence) in the QA report.
Exit code 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import re
import shutil
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(PROJECT_ROOT))

from build_notebook import (CONTENT_MARKERS, MAIN_NOTEBOOK, MODULE_ORDER, PLACEHOLDER, SOLUTION_NOTEBOOK,  # noqa: E402
                            STUDENT_NOTEBOOK, WORKSHOP_CSV, injected_modules)
from notebook_workshop import leaked_answers  # noqa: E402
from qa_report import update_section  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
HTML_PATH = OUTPUT_DIR / "notebook.html"
REQUIRED_FIGURES = [
    "01_entropy_curve.png", "02_dataset_overview.png", "03_missing_values.png", "03_category_distributions.png",
    "03_cramers_v.png", "04_split_balance.png", "05_root_information_gain.png", "05_root_split_odor.png",
    "06_experiment_heatmap.png", "06_depth_curve.png", "06_train_vs_validation.png", "06_strategy_comparison.png",
    "07_feature_importance.png", "08_tree_presentation.png", "08_tree_presentation.svg", "08_tree_odor_none.png",
    "08_tree_odor_none.svg", "final_tree.png", "final_tree.svg", "final_tree.dot", "09_confusion_matrix.png",
    "09_prediction_path.png",
]
REQUIRED_ARTIFACTS = [
    "experiment_results.csv", "split_indices.csv", "test_predictions.csv", "run_summary.json",
    "narrative_facts.json", "final_model/final_tree.json", "final_model/final_tree.dot",
    "final_model/final_config.json",
]
MIN_IMAGES = 12
SECTION_TITLES = {"notebook": "Walidacja notebooka głównego", "workshop": "Warsztat dla grupy",
                  "environment": "Środowisko", "requirements_map": "Audyt 8 wymagań prowadzącego"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.rows)

    def markdown(self) -> str:
        lines = [f"- Status: **{'PASS' if self.ok else 'FAIL'}** ({sum(o for _, o, _ in self.rows)}/{len(self.rows)})",
                 "", "| kontrola | wynik | szczegóły |", "|---|---|---|"]
        lines += [f"| {name} | {'✅' if ok else '❌'} | {detail} |" for name, ok, detail in self.rows]
        return "\n".join(lines)


def execute(nb: nbformat.NotebookNode, cwd: Path, timeout: int = 1200) -> float:
    start = time.perf_counter()
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}).execute()
    return time.perf_counter() - start


def output_stats(nb: nbformat.NotebookNode) -> dict:
    stats = {"cells": len(nb.cells), "code": 0, "errors": 0, "images": 0, "code_without_output": 0}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        stats["code"] += 1
        outputs = cell.get("outputs", [])
        if not outputs and "injected-module" not in cell.metadata.get("tags", []):
            stats["code_without_output"] += 1
        for output in outputs:
            if output.get("output_type") == "error":
                stats["errors"] += 1
            data = output.get("data", {})
            stats["images"] += int("image/png" in data or "image/svg+xml" in data)
    return stats


def code_text(nb: nbformat.NotebookNode) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def static_checks(nb: nbformat.NotebookNode, report: Report) -> None:
    try:
        nbformat.validate(nb)
        valid = True
    except nbformat.ValidationError as error:  # pragma: no cover - reported below
        valid, _ = False, error
    report.check("schemat nbformat 4", valid)
    report.check("kernelspec python3", nb.metadata.get("kernelspec", {}).get("name") == "python3")
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    anchors = [f'<a id="rozdzial-{i:02d}"></a>' for i in range(12)]
    positions = [markdown.find(a) for a in anchors]
    report.check("12 rozdziałów (00–11) w kolejności", all(p >= 0 for p in positions) and positions == sorted(positions))
    imports = re.findall(r"^\s*(?:from\s+src(?:\.\w+)*\s+import|import\s+src\b|from\s+\.\w*\s+import)", code_text(nb), re.M)
    report.check("brak importów z src/ (Colab-ready)", not imports, f"znaleziono: {imports}" if imports else "")
    module_cells = [c for c in nb.cells if "injected-module" in c.metadata.get("tags", [])]
    expected = injected_modules()
    in_sync = [c.metadata.get("id3_module") for c in module_cells] == MODULE_ORDER and all(
        c.source == code for c, (_, code) in zip(module_cells, expected))
    report.check("moduły wstrzyknięte w kolejności zależności i zgodne z src/", in_sync,
                 " → ".join(c.metadata.get("id3_module", "?") for c in module_cells))
    leftovers = PLACEHOLDER.findall(markdown) + [m for m in CONTENT_MARKERS if m in nbformat.writes(nb)]
    report.check("narracja bez niewypełnionych wstawek", not leftovers, str(leftovers[:5]) if leftovers else "")
    report.check("kontrola spójności narracji osadzona", "NARRATIVE_FACTS = {" in code_text(nb))


def validate_main(report: Report, skip_isolated: bool) -> dict:
    print("Main notebook:", MAIN_NOTEBOOK.relative_to(PROJECT_ROOT))
    nb = nbformat.read(MAIN_NOTEBOOK, as_version=4)
    static_checks(nb, report)
    for stale in FIG_DIR.glob("*"):
        stale.unlink()
    try:
        seconds = execute(nb, MAIN_NOTEBOOK.parent)
        executed = True
    except Exception as error:  # CellExecutionError and kernel failures
        seconds, executed = float("nan"), False
        report.check("wykonanie od początku do końca (nbclient)", False, str(error).splitlines()[0][:200])
    if executed:
        report.check("wykonanie od początku do końca (nbclient)", True, f"{seconds:.0f} s")
    nbformat.write(nb, MAIN_NOTEBOOK)
    stats = output_stats(nb)
    report.check("brak błędów w komórkach", stats["errors"] == 0, f"błędy: {stats['errors']}")
    report.check(f"osadzone wykresy (≥ {MIN_IMAGES})", stats["images"] >= MIN_IMAGES, f"obrazy: {stats['images']}")
    report.check("zapisane outputy komórek kodu", stats["code_without_output"] == 0,
                 f"komórki kodu bez outputu: {stats['code_without_output']} z {stats['code']}")
    missing_figures = [f for f in REQUIRED_FIGURES if not (FIG_DIR / f).is_file() or (FIG_DIR / f).stat().st_size < 1000]
    report.check("pliki wykresów (PNG/SVG/DOT) niepuste", not missing_figures,
                 f"{len(REQUIRED_FIGURES) - len(missing_figures)}/{len(REQUIRED_FIGURES)}"
                 + (f"; brak: {missing_figures}" if missing_figures else ""))
    missing_artifacts = [a for a in REQUIRED_ARTIFACTS if not (OUTPUT_DIR / a).is_file()]
    report.check("artefakty (CSV, JSON, DOT)", not missing_artifacts, str(missing_artifacts) if missing_artifacts else
                 ", ".join(REQUIRED_ARTIFACTS))
    rows = sum(1 for _ in open(OUTPUT_DIR / "experiment_results.csv", encoding="utf-8")) - 1 \
        if (OUTPUT_DIR / "experiment_results.csv").exists() else 0
    report.check("experiment_results.csv: 36 modeli", rows == 36, f"wiersze: {rows}")
    summary = json.loads((OUTPUT_DIR / "run_summary.json").read_text()) if (OUTPUT_DIR / "run_summary.json").exists() else {}
    html_ok = export_html(nb)
    report.check("eksport HTML z osadzonymi obrazami", html_ok,
                 f"{HTML_PATH.relative_to(PROJECT_ROOT)} ({HTML_PATH.stat().st_size / 1e6:.1f} MB)" if html_ok else "")
    if not skip_isolated:
        isolated_seconds, isolated_error = isolated_run(nb)
        report.check("uruchomienie izolowane bez src/ i tests/ (symulacja Colab)", isolated_error is None,
                     f"{isolated_seconds:.0f} s" if isolated_error is None else isolated_error)
    return {"stats": stats, "seconds": seconds, "summary": summary}


def export_html(nb: nbformat.NotebookNode) -> bool:
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import TagRemovePreprocessor

    exporter = HTMLExporter(template_name="lab")
    exporter.register_preprocessor(TagRemovePreprocessor(remove_input_tags={"injected-module"}), enabled=True)
    body, _ = exporter.from_notebook_node(nb)
    HTML_PATH.write_text(body, encoding="utf-8")
    return "data:image/png;base64" in body and body.count("<img") >= MIN_IMAGES


def isolated_run(nb: nbformat.NotebookNode) -> tuple[float, str | None]:
    clean = copy.deepcopy(nb)
    for cell in clean.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    with tempfile.TemporaryDirectory(prefix="id3_isolated_") as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        (root / "notebooks").mkdir()
        shutil.copy2(PROJECT_ROOT / "data" / "mushroom.zip", root / "data" / "mushroom.zip")
        try:
            return execute(clean, root / "notebooks"), None
        except Exception as error:
            return float("nan"), str(error).splitlines()[0][:200]


def validate_workshop(report: Report) -> None:
    print("Workshop notebooks")
    csv_text = WORKSHOP_CSV.read_text(encoding="utf-8")
    student = nbformat.read(STUDENT_NOTEBOOK, as_version=4)
    student_text = nbformat.writes(student)
    report.check("uczestnik: brak klucza / gotowych odpowiedzi", not leaked_answers(student_text, csv_text),
                 str(leaked_answers(student_text, csv_text)))
    todo_cells = [c for c in student.cells if c.cell_type == "code" and "# TODO" in c.source]
    report.check("uczestnik: 3–4 komórki TODO", 3 <= len({re.search(r"TODO (\d)", c.source).group(1) for c in todo_cells}) <= 4,
                 f"komórki z TODO: {len(todo_cells)}")
    report.check("uczestnik: zapisany bez outputów", all(not c.get("outputs") for c in student.cells if c.cell_type == "code"))
    report.check("uczestnik: dane wbudowane = delivery_data.csv", csv_text.strip() in code_text(student))
    work = nbformat.reads(student_text, as_version=4)
    with tempfile.TemporaryDirectory(prefix="id3_workshop_") as tmp:
        try:
            seconds = execute(work, Path(tmp), timeout=300)
            error = None
        except Exception as exc:
            seconds, error = float("nan"), str(exc).splitlines()[0][:200]
    report.check("uczestnik: uruchamia się bez błędów (niewypełnione TODO)", error is None,
                 f"{seconds:.0f} s" if error is None else error)
    solution = nbformat.read(SOLUTION_NOTEBOOK, as_version=4)
    try:
        seconds = execute(solution, SOLUTION_NOTEBOOK.parent, timeout=300)
        error = None
    except Exception as exc:
        seconds, error = float("nan"), str(exc).splitlines()[0][:200]
    nbformat.write(solution, SOLUTION_NOTEBOOK)
    streams = "".join(o.get("text", "") for c in solution.cells for o in c.get("outputs", []))
    passed = sum(f"✅ TODO {i} zaliczone" in streams for i in range(1, 5))
    report.check("rozwiązanie: wykonane z zapisanymi wynikami", error is None, f"{seconds:.0f} s" if error is None else error)
    report.check("rozwiązanie: wszystkie 4 kontrolki zaliczone", passed == 4, f"{passed}/4")
    report.check("ANSWER_KEY.md wygenerowany", (SOLUTION_NOTEBOOK.parent / "ANSWER_KEY.md").is_file())


def environment_section() -> str:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "nbformat", "nbclient",
                "nbconvert", "ipykernel", "pytest"]
    versions = []
    for name in packages:
        try:
            versions.append(f"`{name}` {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            versions.append(f"`{name}` —")
    graphviz = shutil.which("dot")
    return "\n".join([
        f"- Python {platform.python_version()} ({sys.executable}) na {platform.platform(terse=True)}",
        f"- Pakiety: {', '.join(versions)}",
        f"- Graphviz `dot`: {'dostępny — ' + graphviz if graphviz else 'niedostępny → drzewo renderowane fallbackiem matplotlib'}",
    ])


def requirements_section(main: dict) -> str:
    figures = len([f for f in REQUIRED_FIGURES if (FIG_DIR / f).exists()])
    summary = main.get("summary", {})
    metrics = summary.get("test_metrics", {})
    rows = [
        ("1", "Wprowadzenie, pseudokod, biblioteki z ID3", "rozdz. 01 (wzory, pseudokod, przykład ręczny, tabela 7 bibliotek)"),
        ("2", "Dane, EDA, balans, korelacje, braki, błędne dane, czyszczenie, uzupełnianie, zbędne atrybuty",
         "rozdz. 02–03: kontrakt UCI, audyt jakości (wyniki negatywne), braki + χ², V Craméra, decyzja o niebalansowaniu, usunięcie `veil-type`"),
        ("3", "Przygotowanie danych, `???`, train/val/test", "rozdz. 04: X/y, kodowanie i dyskretyzacja, 60/20/20 stratyfikowane, 3 strategie braków, audyt duplikatów, `split_indices.csv`"),
        ("4", "Modele próbne, strojenie, dobór parametrów, uwaga o lasach", "rozdz. 06: 36 modeli, krzywa głębokości, reguła wyboru, akapit o lasach/OOB, `experiment_results.csv`"),
        ("5", "Model finalny", f"rozdz. 07: konfiguracja {summary.get('selected_configuration', '?')}, refit train+val, `final_tree.json`"),
        ("6", "Grafika drzewa + biblioteki grafiki", f"rozdz. 08: diagramy PNG/SVG + DOT, tabela 7 bibliotek; wykresów w outputs/figures: {figures}"),
        ("7", "Ocena, macierz pomyłek, predykcje", f"rozdz. 09: accuracy {metrics.get('accuracy', '?')}, p→e {metrics.get('p_to_e', '?')}, 5 ścieżek predykcji"),
        ("8", "Zadanie dla grupy, dostęp tylko dla grupy", "rozdz. 10 + `group_exercise/` (uczestnik) + `instructor_private/` (klucz); instrukcja publikacji w README — publikacja do wykonania przez autora"),
    ]
    lines = ["| Nr | Wymóg | Dowód |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-isolated", action="store_true", help="skip the run without src/ (faster)")
    args = parser.parse_args()
    update_section("environment", SECTION_TITLES["environment"], environment_section())
    main_report = Report()
    main_info = validate_main(main_report, args.skip_isolated)
    update_section("notebook", SECTION_TITLES["notebook"], main_report.markdown())
    workshop_report = Report()
    validate_workshop(workshop_report)
    update_section("workshop", SECTION_TITLES["workshop"], workshop_report.markdown())
    update_section("requirements_map", SECTION_TITLES["requirements_map"], requirements_section(main_info))
    ok = main_report.ok and workshop_report.ok
    print("VALIDATION", "PASSED" if ok else "FAILED", "- see outputs/QA_REPORT.md")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
