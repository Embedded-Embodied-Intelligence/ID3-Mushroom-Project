"""Shared fixtures and the hook that records the pytest summary in outputs/QA_REPORT.md."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP = PROJECT_ROOT / "data" / "mushroom.zip"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import load_mushroom  # noqa: E402
from src.preprocessing import make_stratified_split, split_features_target  # noqa: E402

PLAY_TENNIS = pd.DataFrame(
    [
        ("Sunny", "Hot", "High", "Weak", "No"),
        ("Sunny", "Hot", "High", "Strong", "No"),
        ("Overcast", "Hot", "High", "Weak", "Yes"),
        ("Rain", "Mild", "High", "Weak", "Yes"),
        ("Rain", "Cool", "Normal", "Weak", "Yes"),
        ("Rain", "Cool", "Normal", "Strong", "No"),
        ("Overcast", "Cool", "Normal", "Strong", "Yes"),
        ("Sunny", "Mild", "High", "Weak", "No"),
        ("Sunny", "Cool", "Normal", "Weak", "Yes"),
        ("Rain", "Mild", "Normal", "Weak", "Yes"),
        ("Sunny", "Mild", "Normal", "Strong", "Yes"),
        ("Overcast", "Mild", "High", "Strong", "Yes"),
        ("Overcast", "Hot", "Normal", "Weak", "Yes"),
        ("Rain", "Mild", "High", "Strong", "No"),
    ],
    columns=["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"],
)


@pytest.fixture()
def play_tennis():
    """Quinlan's (1986) / Mitchell's (1997) textbook example, checked by hand."""
    return PLAY_TENNIS.drop(columns="PlayTennis").copy(), PLAY_TENNIS["PlayTennis"].copy()


@pytest.fixture(scope="session")
def mushroom_df():
    return load_mushroom(DATA_ZIP)


@pytest.fixture(scope="session")
def mushroom_split(mushroom_df):
    X, y = split_features_target(mushroom_df)
    return X, y, make_stratified_split(y)


# ---------------------------------------------------------------------------
# QA report hook: a full run (no explicit test paths) updates outputs/QA_REPORT.md
# ---------------------------------------------------------------------------
_STARTED = time.time()


def pytest_sessionfinish(session, exitstatus):
    explicit_paths = [a for a in session.config.args if Path(a).resolve() != PROJECT_ROOT / "tests"]
    if explicit_paths or os.environ.get("ID3_QA_SKIP_REPORT") == "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    counts = {key: len(reporter.stats.get(key, [])) for key in ("passed", "failed", "error", "skipped")}
    by_file: dict[str, list[int]] = {}
    for report in reporter.stats.get("passed", []) + reporter.stats.get("failed", []):
        if getattr(report, "when", "call") != "call":
            continue
        name = report.nodeid.split("::", 1)[0]
        by_file.setdefault(name, [0, 0])[0 if report.passed else 1] += 1
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from qa_report import update_section
    except ImportError:
        return
    status = "PASS" if exitstatus == 0 and counts["failed"] == 0 and counts["error"] == 0 else "FAIL"
    lines = [
        f"- Status: **{status}** (exit code {int(exitstatus)})",
        f"- Wynik: {counts['passed']} passed, {counts['failed']} failed, {counts['error']} errors, "
        f"{counts['skipped']} skipped",
        f"- Czas: {time.time() - _STARTED:.1f} s; Python {sys.version.split()[0]}; pandas {pd.__version__}",
        "",
        "| plik testów | passed | failed |",
        "|---|---:|---:|",
        *[f"| `{name}` | {p} | {f} |" for name, (p, f) in sorted(by_file.items())],
    ]
    update_section("pytest", "Testy automatyczne (pytest)", "\n".join(lines))
