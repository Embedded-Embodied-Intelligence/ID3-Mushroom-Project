"""Section-based writer for outputs/QA_REPORT.md - every QA step owns one section."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "QA_REPORT.md"
SECTION_ORDER = ["environment", "pytest", "notebook", "colab_compat", "workshop", "requirements_map", "packages", "dod"]
HEADER = (
    "# QA_REPORT — ID3 × Mushroom\n\n"
    "Raport generowany automatycznie: sekcję testów zapisuje `tests/conftest.py` (pełne uruchomienie "
    "`python -m pytest -q`), sekcje notebooka i warsztatu — `scripts/validate_notebook.py`, "
    "sekcję paczek ZIP — `scripts/package_submission.py`. Każda sekcja ma własny znacznik czasu.\n"
)
_SECTION = re.compile(r"<!-- BEGIN (?P<key>[\w-]+) -->\n(?P<body>.*?)\n<!-- END (?P=key) -->", re.S)


def read_sections(path: Path = REPORT_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    return {m.group("key"): m.group("body") for m in _SECTION.finditer(path.read_text(encoding="utf-8"))}


def update_section(key: str, title: str, body: str, path: Path = REPORT_PATH) -> Path:
    """Replace (or add) one section and rewrite the report in the canonical order."""
    sections = read_sections(path)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    sections[key] = f"## {title}\n\n_Aktualizacja: {stamp}_\n\n{body.strip()}"
    ordered = [k for k in SECTION_ORDER if k in sections] + sorted(k for k in sections if k not in SECTION_ORDER)
    text = HEADER + "".join(f"\n<!-- BEGIN {k} -->\n{sections[k]}\n<!-- END {k} -->\n" for k in ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
