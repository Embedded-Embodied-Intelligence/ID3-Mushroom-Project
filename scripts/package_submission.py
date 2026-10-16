#!/usr/bin/env python
"""Create the delivery archives and record their audit in outputs/QA_REPORT.md.

    python scripts/package_submission.py
    python scripts/package_submission.py --include-instructor-key   # adds the answer key to the main ZIP

submission/ID3_Mushroom_Submission.zip           for the course instructor: executed notebook, data, src/,
                                                 scripts/, tests/, outputs/, requirements, README and the
                                                 participant workshop materials (no answer key by default)
submission/ID3_Group_Workshop_Students.zip       for the group: participant notebook, CSV, README only
submission/ID3_Workshop_Instructor_PRIVATE.zip   private: solution notebook and ANSWER_KEY.md - never share

Excluded everywhere: .venv/, __pycache__/, .pytest_cache/, .git/, .ipynb_checkpoints/, temporary files.
The main archive must stay below 15 MB (university mail attachment filters).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_report import read_sections, update_section  # noqa: E402

SUBMISSION_DIR = PROJECT_ROOT / "submission"
ARCHIVE_ROOT = "ID3-Mushroom-Project"
MAX_BYTES = 15 * 1024 * 1024
MAIN_INCLUDE = [
    "README.md", "RUNBOOK.md", "requirements.txt", ".gitignore", "pytest.ini",
    "data/mushroom.zip", "data/README.md", "notebooks/ID3_Mushroom_Project.ipynb",
    "src", "scripts", "tests", "outputs",
    "group_exercise/README.md", "group_exercise/delivery_data.csv", "group_exercise/ID3_Delivery_Workshop.ipynb",
]
STUDENT_INCLUDE = ["group_exercise/README.md", "group_exercise/delivery_data.csv",
                   "group_exercise/ID3_Delivery_Workshop.ipynb"]
INSTRUCTOR_INCLUDE = ["group_exercise/instructor_private/ID3_Delivery_Solution.ipynb",
                      "group_exercise/instructor_private/ANSWER_KEY.md"]
EXCLUDED_PARTS = {".venv", "__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints", "submission",
                  "instructor_private", ".mypy_cache", ".ruff_cache"}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".part", ".log", ".swp"}


def is_excluded(relative: Path, allow_private: bool = False) -> bool:
    parts = set(relative.parts)
    if allow_private:
        parts.discard("instructor_private")
    return bool(parts & EXCLUDED_PARTS) or relative.name in EXCLUDED_NAMES or relative.suffix in EXCLUDED_SUFFIXES \
        or relative.name.startswith("~$")


def collect(include: list[str], allow_private: bool = False) -> list[Path]:
    files: list[Path] = []
    for entry in include:
        path = PROJECT_ROOT / entry
        if path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f"required path is missing: {entry}")
        files += [p for p in candidates if not is_excluded(p.relative_to(PROJECT_ROOT), allow_private)]
    return files


def write_zip(target: Path, files: list[Path], strip_prefix: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if strip_prefix and relative.startswith(strip_prefix):
                relative = relative[len(strip_prefix):]
            archive.write(path, f"{ARCHIVE_ROOT if not strip_prefix else target.stem}/{relative}")
    return target


def notebook_is_executed(path: Path) -> tuple[bool, str]:
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code" and "injected-module" not in c.metadata.get("tags", [])]
    errors = sum(o.get("output_type") == "error" for c in code for o in c.get("outputs", []))
    without = sum(not c.get("outputs") for c in code)
    images = sum("image/png" in o.get("data", {}) for c in code for o in c.get("outputs", []))
    return without == 0 and errors == 0, f"komórki bez outputu: {without}, błędy: {errors}, obrazy: {images}"


def audit_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        corrupt = archive.testzip()
    return {"names": names, "size": path.stat().st_size, "corrupt": corrupt,
            "forbidden": sorted({n for n in names if is_excluded(Path(n))}),
            "private": [n for n in names if "instructor_private" in n or "ANSWER_KEY" in n]}


def build_all(out_dir: Path = SUBMISSION_DIR, include_key: bool = False,
              require_executed: bool = True) -> dict[str, Path]:
    notebook = PROJECT_ROOT / "notebooks" / "ID3_Mushroom_Project.ipynb"
    executed, detail = notebook_is_executed(notebook)
    if require_executed and not executed:
        raise SystemExit(f"Notebook is not fully executed ({detail}). Run: python scripts/validate_notebook.py")
    main_files = collect(MAIN_INCLUDE) + (collect(INSTRUCTOR_INCLUDE, allow_private=True) if include_key else [])
    return {
        "main": write_zip(out_dir / "ID3_Mushroom_Submission.zip", main_files),
        "students": write_zip(out_dir / "ID3_Group_Workshop_Students.zip", collect(STUDENT_INCLUDE),
                              strip_prefix="group_exercise/"),
        "instructor": write_zip(out_dir / "ID3_Workshop_Instructor_PRIVATE.zip",
                                collect(INSTRUCTOR_INCLUDE, allow_private=True),
                                strip_prefix="group_exercise/instructor_private/"),
    }


def dod_section() -> str:
    sections = read_sections()

    def status(key: str) -> str:
        body = sections.get(key, "")
        if "Status: **PASS**" in body:
            return "✅"
        return "❌" if "Status: **FAIL**" in body else "—"

    items = [
        ("Osiem punktów ze slajdu + rozwinięcie `???` (pkt 3 i 4), narracja po polsku", status("notebook")),
        ("Dane UCI: schemat 8124×23 i 2480 braków w `stalk-root` zweryfikowane", status("pytest")),
        ("Własny, nominalny, wielogałęziowy ID3; testy matematyczne", status("pytest")),
        ("EDA: klasy, braki, duplikaty przed/po, błędne dane, V Craméra odporne na `veil-type`", status("notebook")),
        ("Podział 60/20/20 odtwarzalny i rozłączny; imputacja dopasowana tylko na train", status("pytest")),
        ("3 strategie braków, 36 modeli, jawna reguła remisów", status("notebook")),
        ("Model wybrany na walidacji, refit train+val, test użyty raz", status("notebook")),
        ("Drzewo jako grafika (SVG/PNG) + tabele bibliotek ID3 i grafiki", status("notebook")),
        ("Macierz pomyłek, metryki i ścieżki predykcji z interpretacją", status("notebook")),
        ("Notebook samowystarczalny (bez `src/`), wykonany, z zapisanymi outputami", status("notebook")),
        ("Dwa pakiety warsztatu (uczestnik bez klucza / instruktor)", status("workshop")),
        ("`ID3_Mushroom_Submission.zip` < 15 MB, z danymi i wykonanym notebookiem", status("packages")),
        ("`pytest` i `validate_notebook.py` w 100%", "✅" if status("pytest") == status("notebook") == "✅" else "❌"),
    ]
    lines = ["| Kryterium (RUNBOOK §13) | Status |", "|---|---|"] + [f"| {a} | {b} |" for a, b in items]
    lines += ["", "Do wykonania ręcznie przez autora: publikacja materiałów warsztatu w miejscu dostępnym tylko dla "
              "grupy (instrukcja: `group_exercise/README.md`) i wysłanie ZIP-a mailem co najmniej dzień przed zajęciami."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--include-instructor-key", action="store_true",
                        help="add instructor_private/ to the instructor's archive (not shared with the group)")
    args = parser.parse_args()
    archives = build_all(include_key=args.include_instructor_key)
    rows, ok = [], True
    for kind, path in archives.items():
        audit = audit_zip(path)
        checks = {
            "integralność ZIP": audit["corrupt"] is None,
            "brak .venv/__pycache__/.pytest_cache/.git/plików tymczasowych": not audit["forbidden"],
        }
        if kind == "main":
            checks["rozmiar < 15 MB"] = audit["size"] < MAX_BYTES
            checks["zawiera dane, notebook z outputami, src/scripts/tests"] = all(
                any(n.endswith(suffix) for n in audit["names"]) for suffix in
                ("data/mushroom.zip", "notebooks/ID3_Mushroom_Project.ipynb", "src/id3.py",
                 "scripts/build_notebook.py", "tests/test_id3.py", "outputs/notebook.html", "outputs/QA_REPORT.md"))
            if not args.include_instructor_key:
                checks["bez klucza odpowiedzi"] = not audit["private"]
        if kind == "students":
            checks["tylko notebook, CSV i README; bez klucza"] = len(audit["names"]) == 3 and not audit["private"]
            with zipfile.ZipFile(path) as archive:
                name = next(n for n in audit["names"] if n.endswith(".ipynb"))
                nb = nbformat.reads(archive.read(name).decode("utf-8"), as_version=4)
            checks["notebook uczestnika bez outputów"] = all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
        if kind == "instructor":
            checks["zawiera rozwiązanie i ANSWER_KEY"] = len(audit["names"]) == 2
        passed = all(checks.values())
        ok &= passed
        rows.append(f"| `{path.relative_to(PROJECT_ROOT)}` | {audit['size'] / 1024 / 1024:.2f} MB | {len(audit['names'])} | "
                    + "; ".join(f"{'✅' if v else '❌'} {k}" for k, v in checks.items()) + " |")
        print(f"  {path.relative_to(PROJECT_ROOT)}: {audit['size'] / 1024 / 1024:.2f} MB, {len(audit['names'])} files, "
              f"{'OK' if passed else 'FAILED'}")
    body = "\n".join([f"- Status: **{'PASS' if ok else 'FAIL'}**",
                      f"- Klucz odpowiedzi w paczce prowadzącego: {'tak' if args.include_instructor_key else 'nie (domyślnie)'}",
                      "- Rozmiary zmierzone przed dołączeniem tego raportu do archiwum; archiwum jest potem budowane "
                      "ponownie z kompletnym raportem i jeszcze raz sprawdzane (integralność, limit 15 MB).",
                      "", "| archiwum | rozmiar | pliki | kontrole |", "|---|---:|---:|---|", *rows])
    update_section("packages", "Paczki ZIP", body)
    update_section("dod", "Definition of Done (podsumowanie automatyczne)", dod_section())
    # Second pass: the instructor's archive must carry the complete QA report (with the sections above).
    final = build_all(include_key=args.include_instructor_key)["main"]
    final_audit = audit_zip(final)
    final_ok = final_audit["corrupt"] is None and final_audit["size"] < MAX_BYTES and not final_audit["forbidden"]
    ok &= final_ok
    print(f"  final {final.relative_to(PROJECT_ROOT)} (with complete QA report): "
          f"{final_audit['size'] / 1024 / 1024:.2f} MB, {'OK' if final_ok else 'FAILED'}")
    print("PACKAGING", "PASSED" if ok else "FAILED", "- see outputs/QA_REPORT.md")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
