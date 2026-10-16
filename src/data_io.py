"""Access to the UCI Mushroom data: local ZIP (offline) or download (Colab).

The original ``agaricus-lepiota.data`` is a header-less CSV with the class in
the first column followed by 22 nominal attributes. Values are read as strings
and the documented missing-value marker ``?`` is converted to NaN explicitly.
"""
from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/73/mushroom"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/73/mushroom.zip"
UCI_DOI = "https://doi.org/10.24432/C5959T"
DATA_LICENSE = "CC BY 4.0"
DATA_CITATION = ("Mushroom [Dataset]. (1981). UCI Machine Learning Repository. "
                 "https://doi.org/10.24432/C5959T")

DATA_MEMBER = "agaricus-lepiota.data"
NAMES_MEMBER = "agaricus-lepiota.names"
MISSING_CODE = "?"
MISSING_TOKEN = "__MISSING__"

FEATURE_NAMES = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat',
]
TARGET_NAME = 'class'
COLUMN_NAMES = [TARGET_NAME, *FEATURE_NAMES]

EXPECTED_N_ROWS = 8124
EXPECTED_N_COLUMNS = 23
EXPECTED_CLASS_COUNTS = {"e": 4208, "p": 3916}
EXPECTED_MISSING = {"stalk-root": 2480}

CLASS_NAMES = {"e": "edible", "p": "poisonous"}
CLASS_NAMES_PL = {"e": "jadalny", "p": "trujący"}

# Category dictionary from section 7 of agaricus-lepiota.names (code -> name).
# tests/test_preprocessing.py checks it against the .names file in the ZIP.
CATEGORY_NAMES: dict[str, dict[str, str]] = {
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "p": "pink",
                  "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "bruises": {"t": "bruises", "f": "no"},
    "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul", "m": "musty",
             "n": "none", "p": "pungent", "s": "spicy"},
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green",
                   "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted"},
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
                               "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
                               "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "veil-type": {"p": "partial", "u": "universal"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large", "n": "none",
                  "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore-print-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green",
                          "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several",
                   "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste",
                "d": "woods"},
}

FEATURE_NAMES_PL = {
    "cap-shape": "kształt kapelusza",
    "cap-surface": "powierzchnia kapelusza",
    "cap-color": "kolor kapelusza",
    "bruises": "siniaczenie po uszkodzeniu",
    "odor": "zapach",
    "gill-attachment": "przyrośnięcie blaszek",
    "gill-spacing": "rozstaw blaszek",
    "gill-size": "szerokość blaszek",
    "gill-color": "kolor blaszek",
    "stalk-shape": "kształt trzonu",
    "stalk-root": "nasada trzonu",
    "stalk-surface-above-ring": "powierzchnia trzonu nad pierścieniem",
    "stalk-surface-below-ring": "powierzchnia trzonu pod pierścieniem",
    "stalk-color-above-ring": "kolor trzonu nad pierścieniem",
    "stalk-color-below-ring": "kolor trzonu pod pierścieniem",
    "veil-type": "typ osłony (velum)",
    "veil-color": "kolor osłony",
    "ring-number": "liczba pierścieni",
    "ring-type": "typ pierścienia",
    "spore-print-color": "kolor wysypu zarodników",
    "population": "sposób występowania (populacja)",
    "habitat": "siedlisko",
}

MANUAL_DOWNLOAD_HINT = """\
Nie znaleziono pliku {path} i nie udało się pobrać archiwum z UCI ({reason}).
Jak wczytać dane ręcznie:
  1. Pobierz {url} w przeglądarce (UCI Machine Learning Repository, Mushroom).
  2. Lokalnie: zapisz plik jako {path} (bez rozpakowywania).
  3. Google Colab: uruchom w komórce
         from google.colab import files; files.upload()
     wybierz mushroom.zip, a następnie przenieś plik do {path}
     (np. !mkdir -p data && mv mushroom.zip data/).
"""


class DataContractError(ValueError):
    """Source data does not match the documented UCI Mushroom contract."""


# ---------------------------------------------------------------------------
# Locating / downloading the archive
# ---------------------------------------------------------------------------

def find_project_root(start: str | Path | None = None, marker: str = "data/mushroom.zip") -> Path:
    """Directory (``start`` or its parent) containing ``marker``; else ``start``."""
    start = Path.cwd() if start is None else Path(start)
    for candidate in (start, start.parent):
        if (candidate / marker).exists():
            return candidate.resolve()
    return start.resolve()


def ensure_mushroom_zip(zip_path: str | Path, *, url: str = UCI_ZIP_URL,
                        allow_download: bool = True, timeout: float = 60.0) -> Path:
    """Return the local archive, downloading the original UCI ZIP only if it is absent.

    An existing file is never re-downloaded, so the local/offline mode does not
    touch the network. A downloaded payload is checked to be a ZIP containing
    ``agaricus-lepiota.data`` before it is written to disk.
    """
    zip_path = Path(zip_path)
    if zip_path.exists():
        return zip_path
    if not allow_download:
        raise FileNotFoundError(MANUAL_DOWNLOAD_HINT.format(path=zip_path, url=url, reason="pobieranie wyłączone"))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            _find_member(archive, DATA_MEMBER)
    except Exception as exc:  # network errors, HTTP errors, corrupted payload
        raise FileNotFoundError(MANUAL_DOWNLOAD_HINT.format(path=zip_path, url=url, reason=repr(exc))) from exc
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_name(zip_path.name + ".part")
    partial.write_bytes(payload)
    partial.replace(zip_path)
    return zip_path


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _find_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if name.rsplit("/", 1)[-1] == basename]
    if len(matches) != 1:
        raise DataContractError(
            f"expected exactly one '{basename}' in the archive, found {len(matches)}; "
            f"archive members: {archive.namelist()}"
        )
    return matches[0]


def zip_members(zip_path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        return pd.DataFrame([{"plik": info.filename, "rozmiar [B]": info.file_size}
                             for info in archive.infolist()])


# ---------------------------------------------------------------------------
# Reading and validating
# ---------------------------------------------------------------------------

def read_raw_mushroom(zip_path: str | Path) -> pd.DataFrame:
    """The data file exactly as published: all values as strings, ``?`` kept."""
    with zipfile.ZipFile(zip_path) as archive:
        member = _find_member(archive, DATA_MEMBER)
        with archive.open(member) as handle:
            raw = pd.read_csv(handle, header=None, dtype=str, keep_default_na=False, na_filter=False)
    if raw.shape[1] != EXPECTED_N_COLUMNS:
        raise DataContractError(f"expected {EXPECTED_N_COLUMNS} columns, found {raw.shape[1]} in {member}")
    raw.columns = COLUMN_NAMES
    return raw


def validate_raw_contract(raw: pd.DataFrame) -> pd.DataFrame:
    """Check the *unmodified* data against the UCI documentation.

    Returns the table of checks; raises :class:`DataContractError` if any fails.
    """
    question_marks = {col: int((raw[col] == MISSING_CODE).sum()) for col in raw.columns}
    other_marks = {col: n for col, n in question_marks.items() if n and col not in EXPECTED_MISSING}
    class_counts = {str(k): int(v) for k, v in raw[TARGET_NAME].value_counts().sort_index().items()}
    checks = [
        ("liczba wierszy", EXPECTED_N_ROWS, int(raw.shape[0])),
        ("liczba kolumn (klasa + 22 cechy)", EXPECTED_N_COLUMNS, int(raw.shape[1])),
        ("rozkład klas e / p", EXPECTED_CLASS_COUNTS, class_counts),
        ("znaki '?' w stalk-root", EXPECTED_MISSING["stalk-root"], question_marks["stalk-root"]),
        ("znaki '?' w pozostałych kolumnach", {}, other_marks),
    ]
    report = pd.DataFrame(checks, columns=["kontrola", "oczekiwane (UCI)", "zaobserwowane"])
    report["zgodne"] = [expected == observed for _, expected, observed in checks]
    if not report["zgodne"].all():
        failed = report.loc[~report["zgodne"], "kontrola"].tolist()
        raise DataContractError(f"data contract violated for: {failed}\n{report.to_string(index=False)}")
    return report


def load_mushroom(zip_path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Validated dataset with ``?`` replaced by NaN (no other modification)."""
    raw = read_raw_mushroom(zip_path)
    if validate:
        validate_raw_contract(raw)
    data = raw.mask(raw.eq(MISSING_CODE))
    if validate:
        observed = {col: int(n) for col, n in data.isna().sum().items() if n}
        if observed != EXPECTED_MISSING:
            raise DataContractError(f"missing values after '?' -> NaN: {observed}, expected {EXPECTED_MISSING}")
    return data


def read_names_text(zip_path: str | Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(_find_member(archive, NAMES_MEMBER)).decode("latin-1")


def parse_attribute_dictionary(names_text: str) -> dict[str, dict[str, str]]:
    """Parse section 7 ("Attribute Information") of agaricus-lepiota.names."""
    start = names_text.index("7. Attribute Information")
    end = names_text.index("8. Missing Attribute Values")
    section = names_text[start:end].split("\n", 1)[1]
    pattern = re.compile(r"^\s*\d+\.\s+([a-z?-]+):\s+(.*?)(?=^\s*\d+\.\s+[a-z?-]+:|\Z)", re.M | re.S)
    dictionary = {}
    for name, body in pattern.findall(section):
        pairs = [item.strip() for item in re.sub(r"\s+", "", body).split(",") if item.strip()]
        dictionary[name.rstrip("?")] = {code: label for label, code in (pair.split("=") for pair in pairs)}
    return dictionary


def compare_with_names_file(zip_path: str | Path) -> pd.DataFrame:
    """Per-feature comparison of ``CATEGORY_NAMES`` with the .names file."""
    parsed = parse_attribute_dictionary(read_names_text(zip_path))
    rows = []
    for feature in FEATURE_NAMES:
        documented = {k: v for k, v in parsed.get(feature, {}).items() if k != MISSING_CODE}
        rows.append({"cecha": feature, "kategorie w .names": len(documented),
                     "kategorie w kodzie": len(CATEGORY_NAMES[feature]),
                     "zgodne": documented == CATEGORY_NAMES[feature]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------

def expand_value(feature: str, value: Any) -> str:
    """``'f'`` -> ``'foul (f)'``; missing markers -> ``'brak danych (?)'``."""
    if value is None or value == MISSING_TOKEN or value == MISSING_CODE or (
            not isinstance(value, str) and pd.isna(value)):
        return "brak danych (?)"
    name = CATEGORY_NAMES.get(feature, {}).get(value)
    return f"{name} ({value})" if name else str(value)


def format_condition(feature: str, value: Any) -> str:
    """``odor = foul (f)``."""
    return f"{feature} = {expand_value(feature, value)}"


def describe_class(label: Any) -> str:
    return f"{CLASS_NAMES_PL.get(label, label)} ({label})"


def dictionary_frame() -> pd.DataFrame:
    """Feature dictionary for presentation: name, Polish meaning and categories."""
    return pd.DataFrame([
        {"cecha": feature, "znaczenie": FEATURE_NAMES_PL[feature],
         "liczba kategorii (UCI)": len(CATEGORY_NAMES[feature]),
         "kategorie (kod = nazwa)": ", ".join(f"{code}={name}" for code, name in CATEGORY_NAMES[feature].items())}
        for feature in FEATURE_NAMES
    ])


def expand_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with codes replaced by ``name (code)`` labels."""
    out = df.copy().astype(object)
    for column in out.columns:
        if column in CATEGORY_NAMES:
            out[column] = [expand_value(column, v) for v in out[column]]
        elif column == TARGET_NAME:
            out[column] = [describe_class(v) for v in out[column]]
    return out


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna()


def as_object_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Plain object-dtype copy (identical behaviour across pandas 2.x and 3.x)."""
    return df.astype(object).where(df.notna(), np.nan)
