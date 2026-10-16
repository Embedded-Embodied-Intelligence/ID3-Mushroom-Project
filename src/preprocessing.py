"""Data preparation for ID3: quality audit, associations, split protocol, missing values.

Every transformation that learns something from data (constant columns,
imputation values, known categories) is fitted on the training rows only and
then applied unchanged to validation/test (fit on train, transform elsewhere).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split

from .data_io import CATEGORY_NAMES, MISSING_CODE, MISSING_TOKEN, TARGET_NAME

MISSING_STRATEGIES = ("missing_category", "most_frequent", "drop_stalk_root")
STALK_ROOT = "stalk-root"
RANDOM_STATE = 42
SPLIT_SIZES = {"train": 0.6, "validation": 0.2, "test": 0.2}
PART_NAMES_PL = {"train": "uczący", "validation": "walidacyjny", "test": "testowy"}


def split_features_target(df: pd.DataFrame, target: str = TARGET_NAME) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix X (all columns but ``target``) and target vector y."""
    return df.drop(columns=[target]), df[target].rename(target)


# ---------------------------------------------------------------------------
# Data quality audit
# ---------------------------------------------------------------------------

def find_constant_features(X: pd.DataFrame) -> list[str]:
    """Columns with at most one distinct observed (non-missing) value."""
    return [column for column in X.columns if X[column].nunique(dropna=True) <= 1]


def out_of_domain_values(X: pd.DataFrame) -> dict[str, list[str]]:
    """Observed values that the UCI dictionary does not declare (NaN excluded)."""
    found = {}
    for column in X.columns:
        observed = set(X[column].dropna().unique().tolist())
        extra = sorted(observed - set(CATEGORY_NAMES.get(column, {})))
        if extra:
            found[column] = extra
    return found


def unobserved_categories(X: pd.DataFrame) -> dict[str, list[str]]:
    """Categories declared in the UCI dictionary that never occur in ``X``."""
    found = {}
    for column in X.columns:
        observed = set(X[column].dropna().unique().tolist())
        absent = [code for code in CATEGORY_NAMES.get(column, {}) if code not in observed]
        if absent:
            found[column] = absent
    return found


def _fill_for_comparison(X: pd.DataFrame) -> pd.DataFrame:
    """Object copy with NaN shown as '?' so that rows can be compared for equality."""
    return X.astype(object).where(X.notna(), MISSING_CODE)


def duplicate_stats(X: pd.DataFrame, y: pd.Series) -> dict[str, int]:
    """Duplicated feature vectors, full duplicates (features + class) and label conflicts."""
    frame = _fill_for_comparison(X)
    labelled = frame.assign(__label__=y.to_numpy(dtype=object))
    labels_per_vector = labelled.groupby(list(frame.columns), sort=False)["__label__"].nunique()
    return {
        "duplicate_feature_vectors": int(frame.duplicated().sum()),
        "duplicate_rows": int(labelled.duplicated().sum()),
        "conflicting_vectors": int((labels_per_vector > 1).sum()),
    }


def overlap_stats(X_ref: pd.DataFrame, y_ref: pd.Series, X_new: pd.DataFrame, y_new: pd.Series) -> dict[str, int]:
    """Rows of ``X_new`` whose exact feature vector also occurs in ``X_ref`` (and label conflicts)."""
    ref_labels: dict[tuple, set] = {}
    for key, label in zip(_fill_for_comparison(X_ref).itertuples(index=False, name=None), y_ref.to_numpy()):
        ref_labels.setdefault(key, set()).add(label)
    overlap = conflicting = 0
    for key, label in zip(_fill_for_comparison(X_new).itertuples(index=False, name=None), y_new.to_numpy()):
        if key in ref_labels:
            overlap += 1
            conflicting += int(ref_labels[key] != {label})
    return {"overlap": overlap, "conflicting": conflicting}


def data_quality_audit(df: pd.DataFrame, target: str = TARGET_NAME) -> pd.DataFrame:
    """Table of quality checks; negative findings ("nothing found") are reported as results too."""
    X, y = split_features_target(df, target)
    values = pd.Series(X.to_numpy(dtype=object).ravel()).dropna().astype(str)
    numeric_columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    bad_format = int((~values.str.fullmatch(r"[a-z]")).sum())
    blank = int((values.str.strip() != values).sum() + (values == "").sum())
    domain = out_of_domain_values(X)
    unknown_labels = sorted(set(y.dropna().unique().tolist()) - {"e", "p"})
    dups = duplicate_stats(X, y)
    missing = X.isna().sum()
    missing = missing[missing > 0]
    constant = find_constant_features(X)
    unobserved = unobserved_categories(X)

    def status(ok: bool, action: str = "do obsłużenia") -> str:
        return "OK - brak problemu" if ok else action

    rows = [
        ("typy kolumn", f"{X.shape[1] - len(numeric_columns)}/{X.shape[1]} kolumn tekstowych (nominalnych)",
         status(not numeric_columns)),
        ("wartości spoza słownika UCI", f"{sum(len(v) for v in domain.values())} {domain or ''}".strip(),
         status(not domain)),
        ("format kodu (1 mała litera)", f"{bad_format} niezgodnych wartości", status(bad_format == 0)),
        ("puste napisy / białe znaki", f"{blank} wartości", status(blank == 0)),
        ("etykiety klas spoza {e, p}", f"{len(unknown_labels)} {unknown_labels or ''}".strip(),
         status(not unknown_labels)),
        ("pełne duplikaty wierszy (cechy + klasa)", f"{dups['duplicate_rows']}", status(dups["duplicate_rows"] == 0)),
        ("duplikaty wektorów cech", f"{dups['duplicate_feature_vectors']}",
         status(dups["duplicate_feature_vectors"] == 0)),
        ("sprzeczne etykiety (te same cechy, inna klasa)", f"{dups['conflicting_vectors']}",
         status(dups["conflicting_vectors"] == 0)),
        ("braki danych ('?' -> NaN)",
         ", ".join(f"{c}: {n} ({n / len(X):.2%})" for c, n in missing.items()) or "0",
         status(missing.empty, "do obsłużenia (3 strategie, rozdz. 04)")),
        ("cechy stałe (1 wartość)", ", ".join(constant) or "brak",
         status(not constant, "do usunięcia (zerowy IG)")),
        ("kategorie ze słownika niewystępujące w danych",
         "; ".join(f"{c}: {', '.join(v)}" for c, v in unobserved.items()) or "brak",
         "informacyjnie (nie jest błędem)"),
    ]
    return pd.DataFrame(rows, columns=["kontrola", "wynik", "status"])


def feature_profile(X: pd.DataFrame) -> pd.DataFrame:
    """Cardinality, missing values and dominant category of every feature."""
    rows = []
    for column in X.columns:
        counts = X[column].value_counts(dropna=True)
        rows.append({
            "cecha": column,
            "kategorie obserwowane": int(counts.size),
            "kategorie w słowniku": len(CATEGORY_NAMES.get(column, {})),
            "braki": int(X[column].isna().sum()),
            "dominanta": counts.index[0] if counts.size else None,
            "udział dominanty": float(counts.iloc[0] / len(X)) if counts.size else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Association measures for nominal variables
# ---------------------------------------------------------------------------

def _as_object_series(values: Any) -> pd.Series:
    arr = values.to_numpy(dtype=object) if isinstance(values, pd.Series) else np.asarray(values, dtype=object)
    return pd.Series(arr)


def cramers_v(x: Any, y: Any) -> float:
    """Cramér's V = sqrt(chi2 / (n * min(r - 1, c - 1))) of two nominal variables.

    A constant variable (r == 1 or c == 1) gives min(r - 1, c - 1) == 0; V is then
    defined as 0.0 (no association can be measured) instead of dividing by zero.
    Missing values are rejected - encode them as a category first.
    """
    xs, ys = _as_object_series(x), _as_object_series(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if xs.isna().any() or ys.isna().any():
        raise ValueError("cramers_v does not accept missing values; encode them as a category first")
    observed = pd.crosstab(xs, ys).to_numpy()
    k = min(observed.shape[0] - 1, observed.shape[1] - 1)
    if k == 0:
        return 0.0
    chi2 = chi2_contingency(observed, correction=False)[0]
    return float(min(1.0, math.sqrt(chi2 / (observed.sum() * k))))


def cramers_v_matrix(df: pd.DataFrame, columns: list[str] | None = None,
                     missing_as_category: bool = True) -> pd.DataFrame:
    """Symmetric matrix of Cramér's V (diagonal = 1 for non-constant variables)."""
    columns = list(df.columns) if columns is None else list(columns)
    data = df[columns].astype(object)
    if missing_as_category:
        data = data.where(data.notna(), MISSING_TOKEN)
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for i, a in enumerate(columns):
        if data[a].nunique() <= 1:
            matrix.loc[a, a] = 0.0
        for b in columns[i + 1:]:
            matrix.loc[a, b] = matrix.loc[b, a] = cramers_v(data[a], data[b])
    return matrix


def association_with_target(X: pd.DataFrame, y: pd.Series, missing_as_category: bool = True) -> pd.DataFrame:
    """Cramér's V, mutual information I(A; Y) [bits] and Theil's U(Y|A) of every feature.

    Mutual information is computed with scikit-learn (an implementation
    independent of src/id3.py); I(A; Y) is numerically the ID3 information gain.
    """
    data = X.astype(object)
    if missing_as_category:
        data = data.where(data.notna(), MISSING_TOKEN)
    labels = y.to_numpy(dtype=object)
    h_y = mutual_info_score(labels, labels) / math.log(2)
    rows = []
    for column in data.columns:
        values = data[column].to_numpy(dtype=object)
        mi = mutual_info_score(labels, values) / math.log(2)
        rows.append({"cecha": column, "kategorie": int(pd.Series(values).nunique()),
                     "V Craméra": cramers_v(values, labels), "I(A;Y) [bit]": max(mi, 0.0),
                     "U Theila": max(mi, 0.0) / h_y if h_y > 0 else 0.0})
    return pd.DataFrame(rows).sort_values(["I(A;Y) [bit]", "cecha"], ascending=[False, True],
                                          kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSplit:
    """Disjoint index labels of the train / validation / test parts."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    random_state: int = RANDOM_STATE

    def parts(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def train_validation(self) -> np.ndarray:
        return np.sort(np.concatenate([self.train, self.validation]))

    def assert_disjoint(self, n_total: int | None = None) -> None:
        sets = {name: set(idx.tolist()) for name, idx in self.parts().items()}
        names = list(sets)
        for i, a in enumerate(names):
            if len(sets[a]) != len(self.parts()[a]):
                raise AssertionError(f"part '{a}' contains repeated indices")
            for b in names[i + 1:]:
                common = sets[a] & sets[b]
                if common:
                    raise AssertionError(f"parts '{a}' and '{b}' share {len(common)} indices")
        if n_total is not None and sum(len(s) for s in sets.values()) != n_total:
            raise AssertionError("the parts do not cover the whole dataset")

    def summary(self, y: pd.Series) -> pd.DataFrame:
        """Sizes and class proportions of every part (documents the stratification)."""
        total = sum(len(idx) for idx in self.parts().values())
        rows = []
        for name, idx in self.parts().items():
            counts = y.loc[idx].value_counts()
            row = {"zbiór": PART_NAMES_PL[name], "n": len(idx), "udział": len(idx) / total}
            for label in sorted(y.unique()):
                row[f"n {label}"] = int(counts.get(label, 0))
                row[f"% {label}"] = counts.get(label, 0) / len(idx)
            rows.append(row)
        return pd.DataFrame(rows)

    def assignment(self) -> pd.DataFrame:
        """Row index -> part, for saving the split for audit."""
        frames = [pd.DataFrame({"row_index": idx, "part": name}) for name, idx in self.parts().items()]
        return pd.concat(frames).sort_values("row_index").reset_index(drop=True)


def make_stratified_split(y: pd.Series, *, train_size: float = 0.6, validation_size: float = 0.2,
                          test_size: float = 0.2, random_state: int = RANDOM_STATE) -> DataSplit:
    """Two stratified, shuffled splits: (train+val | test), then (train | val)."""
    if not math.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1")
    index = y.index.to_numpy()
    rest, test = train_test_split(index, test_size=test_size, shuffle=True,
                                  stratify=y.to_numpy(dtype=object), random_state=random_state)
    train, validation = train_test_split(rest, test_size=validation_size / (train_size + validation_size),
                                         shuffle=True, stratify=y.loc[rest].to_numpy(dtype=object),
                                         random_state=random_state)
    split = DataSplit(np.sort(train), np.sort(validation), np.sort(test), random_state)
    split.assert_disjoint(len(index))
    return split


class HoldoutVault:
    """Keeps the test split sealed until one explicit, final evaluation.

    ``audit_view`` gives logged, read-only access for data-hygiene audits that
    involve no model (e.g. duplicates across splits); ``unlock`` may be called
    exactly once - after the final model has been fixed.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, name: str = "test") -> None:
        self._X, self._y, self.name = X.copy(), y.copy(), name
        self._unlocked = False
        self.access_log: list[dict[str, str]] = []

    def __len__(self) -> int:
        return len(self._y)

    def __repr__(self) -> str:
        state = "otwarty" if self._unlocked else "zapieczętowany"
        return f"HoldoutVault({self.name!r}, n={len(self)}, {state}, dostępy={len(self.access_log)})"

    @property
    def index(self) -> pd.Index:
        return self._X.index.copy()

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked

    def audit_view(self, purpose: str) -> tuple[pd.DataFrame, pd.Series]:
        if self._unlocked:
            raise RuntimeError("the test split has already been used for the final evaluation")
        self.access_log.append({"typ dostępu": "audyt danych (bez modelu)", "cel": purpose})
        return self._X.copy(), self._y.copy()

    def unlock(self, purpose: str) -> tuple[pd.DataFrame, pd.Series]:
        if self._unlocked:
            raise RuntimeError("the test split can be unlocked only once - no further tuning allowed")
        self._unlocked = True
        self.access_log.append({"typ dostępu": "jednorazowa ocena modelu", "cel": purpose})
        return self._X.copy(), self._y.copy()

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.access_log, columns=["typ dostępu", "cel"])


# ---------------------------------------------------------------------------
# Missing-value strategies (fit on train, transform elsewhere)
# ---------------------------------------------------------------------------

def _deterministic_mode(series: pd.Series) -> str:
    """Most frequent observed value; ties broken alphabetically."""
    counts = series.value_counts(dropna=True)
    if counts.empty:
        raise ValueError(f"column {series.name!r} has no observed values to compute a mode")
    return min(counts.index[counts == counts.max()].tolist())


class CategoricalPreprocessor:
    """Fit-on-train preparation of nominal data for ID3.

    ``fit`` learns from the training rows only: constant features to drop, the
    missing-value treatment of ``strategy`` and the known categories of every
    kept feature. ``transform`` applies exactly these parameters to any split
    and guarantees an output without missing values.

    Strategies
    ----------
    ``missing_category``  NaN -> explicit category ``'__MISSING__'``;
    ``most_frequent``     NaN -> mode of the column in the fitting data;
    ``drop_stalk_root``   the ``stalk-root`` column is removed (no imputation).
    """

    def __init__(self, strategy: str = "missing_category", *, drop_constant: bool = True,
                 missing_token: str = MISSING_TOKEN) -> None:
        self.strategy = strategy
        self.drop_constant = drop_constant
        self.missing_token = missing_token

    def __repr__(self) -> str:
        return f"CategoricalPreprocessor(strategy={self.strategy!r}, drop_constant={self.drop_constant})"

    def fit(self, X: pd.DataFrame) -> CategoricalPreprocessor:
        if self.strategy not in MISSING_STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}; choose one of {MISSING_STRATEGIES}")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame")
        self.feature_names_in_ = list(X.columns)
        self.constant_features_ = find_constant_features(X) if self.drop_constant else []
        dropped = list(self.constant_features_)
        if self.strategy == "drop_stalk_root" and STALK_ROOT in X.columns and STALK_ROOT not in dropped:
            dropped.append(STALK_ROOT)
        self.dropped_features_ = dropped
        self.feature_names_out_ = [c for c in X.columns if c not in dropped]
        if self.strategy == "most_frequent":
            self.fill_values_ = {c: _deterministic_mode(X[c]) for c in self.feature_names_out_}
        elif self.strategy == "missing_category":
            self.fill_values_ = {c: self.missing_token for c in self.feature_names_out_}
        else:
            self.fill_values_ = {}
        missing = X[self.feature_names_out_].isna().sum()
        self.missing_counts_fit_ = {c: int(n) for c, n in missing.items() if n}
        transformed = self._apply(X)
        self.categories_ = {c: sorted(transformed[c].unique().tolist()) for c in self.feature_names_out_}
        self.n_samples_fit_ = len(X)
        self.fit_index_ = X.index.to_numpy().copy()
        return self

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.feature_names_out_].copy()
        if self.fill_values_:
            out = out.fillna(self.fill_values_)
        leftover = out.isna().sum()
        leftover = leftover[leftover > 0]
        if not leftover.empty:
            raise ValueError(f"strategy {self.strategy!r} leaves missing values in {leftover.to_dict()}")
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("CategoricalPreprocessor is not fitted; call fit(X_train) first")
        absent = [c for c in self.feature_names_out_ if c not in X.columns]
        if absent:
            raise ValueError(f"X lacks columns seen during fit: {absent}")
        return self._apply(X)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def unseen_categories(self, X: pd.DataFrame) -> dict[str, list[str]]:
        """Categories of the transformed ``X`` that were never seen during fit."""
        transformed = self.transform(X)
        found = {}
        for column in self.feature_names_out_:
            extra = sorted(set(transformed[column].unique().tolist()) - set(self.categories_[column]))
            if extra:
                found[column] = extra
        return found

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "constant_features_dropped": list(self.constant_features_),
            "dropped_features": list(self.dropped_features_),
            "n_features_out": len(self.feature_names_out_),
            "missing_in_fit_data": dict(self.missing_counts_fit_),
            "imputation": {c: self.fill_values_[c] for c in self.missing_counts_fit_ if c in self.fill_values_},
            "n_samples_fit": self.n_samples_fit_,
        }


def duplicate_audit(parts: dict[str, tuple[pd.DataFrame, pd.Series]],
                    strategies: tuple[str, ...] = MISSING_STRATEGIES) -> pd.DataFrame:
    """Duplicates before/after every missing-value strategy and exact-vector overlap across splits.

    ``parts`` maps ``train``/``validation``/``test`` to ``(X, y)``. Each
    strategy is fitted on ``train`` only, as in the modelling pipeline.
    """
    variants: dict[str, dict[str, tuple[pd.DataFrame, pd.Series]]] = {"surowe (22 cechy, '?' jako symbol)": parts}
    for strategy in strategies:
        pre = CategoricalPreprocessor(strategy).fit(parts["train"][0])
        variants[strategy] = {name: (pre.transform(X), y) for name, (X, y) in parts.items()}
    rows = []
    for name, data in variants.items():
        X_all = pd.concat([X for X, _ in data.values()])
        y_all = pd.concat([y for _, y in data.values()])
        stats = duplicate_stats(X_all, y_all)
        val_train = overlap_stats(*data["train"], *data["validation"])
        X_trval = pd.concat([data["train"][0], data["validation"][0]])
        y_trval = pd.concat([data["train"][1], data["validation"][1]])
        test_trval = overlap_stats(X_trval, y_trval, *data["test"])
        rows.append({
            "wariant": name,
            "liczba cech": X_all.shape[1],
            "duplikaty wektorów cech": stats["duplicate_feature_vectors"],
            "pełne duplikaty (cechy+klasa)": stats["duplicate_rows"],
            "wektory ze sprzeczną klasą": stats["conflicting_vectors"],
            "walidacja ∩ trening": val_train["overlap"],
            "test ∩ (trening+walidacja)": test_trval["overlap"],
            "w tym sprzeczne etykiety": test_trval["conflicting"] + val_train["conflicting"],
        })
    return pd.DataFrame(rows)
