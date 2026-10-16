"""Classic ID3 decision tree (Quinlan, 1986) for nominal attributes.

The tree is grown with the original recursive procedure: greedy choice of the
attribute with maximal information gain, a multiway split on every category
observed in the node, and no re-use of an attribute along the same path.

``max_depth``, ``min_samples_split`` and ``min_information_gain`` are didactic
pre-pruning extensions of that growth procedure - they are *not* parameters of
Quinlan's original algorithm. With their defaults the classifier grows the
classic, unrestricted ID3 tree (zero-gain splits are never created).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Gains closer than this are treated as equal (deterministic tie-breaking) and a
# best gain not exceeding it is treated as "no information" (no split is made).
GAIN_TOLERANCE = 1e-12

STOP_REASONS = {
    "pure": "węzeł czysty (jedna klasa)",
    "no_features": "wyczerpane atrybuty",
    "max_depth": "osiągnięto max_depth",
    "min_samples_split": "za mało obserwacji (min_samples_split)",
    "zero_gain": "żaden atrybut nie daje przyrostu informacji",
    "min_information_gain": "IG poniżej min_information_gain",
}

ValueFormatter = Callable[[str, Any], str]


class NotFittedError(ValueError, AttributeError):
    """Raised when a fitted attribute is used before ``fit``."""


# ---------------------------------------------------------------------------
# Information-theoretic building blocks
# ---------------------------------------------------------------------------

def _as_1d_array(values: Any, name: str) -> np.ndarray:
    """Return ``values`` as a 1-D object array, rejecting missing entries."""
    if isinstance(values, (pd.Series, pd.Index)):
        arr = values.to_numpy(dtype=object)
    elif isinstance(values, np.ndarray):
        arr = values.astype(object, copy=False)
    else:
        arr = np.asarray(list(values), dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size and pd.isna(arr).any():
        raise ValueError(
            f"{name} contains missing values (NaN/None/pd.NA); ID3 needs every "
            "missing value to be converted into an explicit category first"
        )
    return arr


def entropy_from_counts(counts: Any) -> float:
    """Shannon entropy (in bits) of a vector of class counts.

    Zero counts are discarded *before* the logarithm is evaluated, which
    implements the convention ``0 * log2(0) := 0`` without ever computing
    ``log2(0)``. Empty and single-class inputs return exactly ``0.0``.
    """
    if not isinstance(counts, np.ndarray):
        counts = list(counts)
    arr = np.asarray(counts, dtype=float).ravel()
    if arr.size and (not np.all(np.isfinite(arr)) or np.any(arr < 0)):
        raise ValueError("class counts must be finite and non-negative")
    arr = arr[arr > 0]
    if arr.size <= 1:
        return 0.0
    p = arr / arr.sum()
    return max(float(-np.sum(p * np.log2(p))), 0.0)


def class_counts(labels: Any) -> dict[Any, int]:
    """Number of occurrences of every label, in sorted label order."""
    arr = _as_1d_array(labels, "labels")
    if arr.size == 0:
        return {}
    values, counts = np.unique(arr, return_counts=True)
    return {value: int(count) for value, count in zip(values.tolist(), counts.tolist())}


def entropy(labels: Any) -> float:
    """H(S) = -sum_i p_i * log2(p_i) of a collection of labels.

    Convention: ``H([]) == 0.0`` - an empty set carries no uncertainty, so an
    (impossible in ID3) empty branch would contribute nothing to the remainder.
    """
    return entropy_from_counts(list(class_counts(labels).values()))


def _gain_from_table(table: np.ndarray) -> float:
    """Information gain from a contingency table n(v, c) (rows: categories)."""
    n = table.sum()
    if n == 0:
        return 0.0
    parent = entropy_from_counts(table.sum(axis=0))
    remainder = sum((row.sum() / n) * entropy_from_counts(row) for row in table)
    gain = parent - remainder
    # Absorb floating-point noise only; genuine violations would surface in tests.
    if -1e-9 < gain < 0.0:
        gain = 0.0
    if parent < gain < parent + 1e-9:
        gain = parent
    return float(gain)


def contingency_table(feature_values: Any, labels: Any) -> pd.DataFrame:
    """Counts n(v, c) for every observed category v and class c."""
    x = _as_1d_array(feature_values, "feature_values")
    y = _as_1d_array(labels, "labels")
    if x.shape != y.shape:
        raise ValueError("feature_values and labels must have the same length")
    values, x_idx = np.unique(x, return_inverse=True)
    classes, y_idx = np.unique(y, return_inverse=True)
    table = np.zeros((values.size, classes.size), dtype=np.int64)
    np.add.at(table, (x_idx.ravel(), y_idx.ravel()), 1)
    return pd.DataFrame(table, index=values, columns=classes)


def information_gain(feature_values: Any, labels: Any) -> float:
    """IG(S, A) = H(S) - sum_v |S_v|/|S| * H(S_v) over the observed values of A."""
    return _gain_from_table(contingency_table(feature_values, labels).to_numpy())


def gain_breakdown(feature_values: Any, labels: Any) -> pd.DataFrame:
    """Terms of IG(S, A) per category v: |S_v|, class counts, H(S_v), |S_v|/|S| and their product."""
    table = contingency_table(feature_values, labels)
    n = int(table.to_numpy().sum())
    rows = []
    for value, counts in table.iterrows():
        n_v = int(counts.sum())
        h_v = entropy_from_counts(counts.to_numpy())
        rows.append({"value": value, "n": n_v, **{f"n_{label}": int(c) for label, c in counts.items()},
                     "entropy": h_v, "weight": n_v / n, "weighted_entropy": n_v / n * h_v})
    return pd.DataFrame(rows)


def split_information(feature_values: Any) -> float:
    """SplitInfo(S, A): entropy of the partition induced by A (C4.5)."""
    return entropy(feature_values)


def gain_ratio(feature_values: Any, labels: Any) -> float:
    """GainRatio(S, A) = IG(S, A) / SplitInfo(S, A); 0.0 for a single-valued A."""
    split_info = split_information(feature_values)
    if split_info <= GAIN_TOLERANCE:
        return 0.0
    return information_gain(feature_values, labels) / split_info


def information_gain_table(X: pd.DataFrame, y: Any, features: Sequence[str] | None = None) -> pd.DataFrame:
    """IG, SplitInfo and GainRatio of every attribute, ranked like ID3 ranks them.

    The ranking uses the classifier's rule: decreasing IG, ties (within
    ``GAIN_TOLERANCE``) broken alphabetically by attribute name.
    """
    labels = _as_1d_array(y, "y")
    rows = []
    for feature in features if features is not None else X.columns:
        values = _as_1d_array(X[feature], feature)
        rows.append({
            "feature": feature,
            "n_values": int(len(set(values.tolist()))),
            "information_gain": information_gain(values, labels),
            "split_information": split_information(values),
            "gain_ratio": gain_ratio(values, labels),
        })
    table = pd.DataFrame(rows).set_index("feature", drop=False)
    remaining = dict(zip(table["feature"], table["information_gain"]))
    order = []
    while remaining:  # repeated argmax with the classifier's own tie-breaking rule
        feature, _ = select_best_feature(remaining)
        order.append(feature)
        del remaining[feature]
    table = table.loc[order].reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def select_best_feature(gains: Mapping[str, float]) -> tuple[str, float]:
    """argmax of information gain; gains within ``GAIN_TOLERANCE`` tie and are
    resolved alphabetically by attribute name (deterministic, order-independent)."""
    if not gains:
        raise ValueError("no candidate attributes")
    best_gain = max(gains.values())
    best_feature = min(f for f, g in gains.items() if g >= best_gain - GAIN_TOLERANCE)
    return best_feature, gains[best_feature]


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

@dataclass
class ID3Node:
    """One node of an ID3 tree.

    Leaves have ``is_leaf=True`` and ``feature=None``; internal nodes keep one
    child per category observed in the node (``children``). Every node stores its
    class distribution and majority class, which serves as the prediction of a
    leaf and as the fallback for categories never seen at an internal node.
    """

    node_id: int
    depth: int
    n_samples: int
    class_counts: dict[Any, int]
    majority_class: Any
    entropy: float
    is_leaf: bool = True
    feature: str | None = None
    information_gain: float | None = None
    children: dict[str, ID3Node] = field(default_factory=dict)
    candidate_gains: dict[str, float] = field(default_factory=dict)
    stop_reason: str | None = None
    branch_value: str | None = None

    @property
    def purity(self) -> float:
        """Share of the majority class in the node."""
        return self.class_counts.get(self.majority_class, 0) / self.n_samples if self.n_samples else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "n_samples": self.n_samples,
            "class_counts": {str(k): v for k, v in self.class_counts.items()},
            "majority_class": str(self.majority_class),
            "entropy": self.entropy,
            "is_leaf": self.is_leaf,
            "feature": self.feature,
            "information_gain": self.information_gain,
            "stop_reason": self.stop_reason,
            "branch_value": self.branch_value,
            "candidate_gains": dict(self.candidate_gains),
            "children": {value: child.to_dict() for value, child in self.children.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ID3Node:
        node = cls(
            node_id=int(payload["node_id"]),
            depth=int(payload["depth"]),
            n_samples=int(payload["n_samples"]),
            class_counts={k: int(v) for k, v in payload["class_counts"].items()},
            majority_class=payload["majority_class"],
            entropy=float(payload["entropy"]),
            is_leaf=bool(payload["is_leaf"]),
            feature=payload.get("feature"),
            information_gain=payload.get("information_gain"),
            candidate_gains={k: float(v) for k, v in payload.get("candidate_gains", {}).items()},
            stop_reason=payload.get("stop_reason"),
            branch_value=payload.get("branch_value"),
        )
        node.children = {value: cls.from_dict(child) for value, child in payload.get("children", {}).items()}
        return node


@dataclass(frozen=True)
class DecisionStep:
    """One test on the prediction path: node ``node_id`` tests ``feature``."""

    node_id: int
    depth: int
    feature: str
    value: str
    n_samples: int
    class_counts: dict[Any, int]
    entropy: float
    information_gain: float
    matched: bool


@dataclass(frozen=True)
class PredictionExplanation:
    """Full, human-readable trace of a single prediction."""

    prediction: Any
    steps: tuple[DecisionStep, ...]
    final_node_id: int
    final_n_samples: int
    final_class_counts: dict[Any, int]
    used_fallback: bool
    fallback_value: str | None = None

    def to_frame(self, value_formatter: ValueFormatter | None = None) -> pd.DataFrame:
        fmt = value_formatter or (lambda feature, value: str(value))
        return pd.DataFrame([
            {
                "krok": i,
                "węzeł": step.node_id,
                "test": step.feature,
                "wartość próbki": fmt(step.feature, step.value),
                "n w węźle": step.n_samples,
                "rozkład klas": _format_counts(step.class_counts),
                "H [bit]": round(step.entropy, 4),
                "IG [bit]": round(step.information_gain, 4),
                "gałąź istnieje": step.matched,
            }
            for i, step in enumerate(self.steps, start=1)
        ])

    def to_text(self, value_formatter: ValueFormatter | None = None) -> str:
        fmt = value_formatter or (lambda feature, value: str(value))
        lines = []
        for i, step in enumerate(self.steps, start=1):
            lines.append(
                f"{i}. węzeł #{step.node_id} (n={step.n_samples}, {_format_counts(step.class_counts)}, "
                f"H={step.entropy:.3f}) testuje {step.feature!r} -> {fmt(step.feature, step.value)}"
            )
        counts = _format_counts(self.final_class_counts)
        if self.used_fallback:
            lines.append(
                f"   brak gałęzi dla tej wartości -> klasa większościowa węzła #{self.final_node_id} "
                f"({counts}): {self.prediction}"
            )
        else:
            lines.append(f"=> liść #{self.final_node_id} (n={self.final_n_samples}, {counts}): predykcja {self.prediction}")
        return "\n".join(lines)


def _format_counts(counts: Mapping[Any, int]) -> str:
    return ", ".join(f"{label}={count}" for label, count in counts.items())


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ID3Classifier:
    """ID3 decision tree for nominal (categorical) attributes.

    Parameters
    ----------
    max_depth:
        ``None`` (unlimited) or a non-negative int; the root has depth 0, so
        ``max_depth=0`` yields a single leaf and ``max_depth=1`` a decision stump.
    min_samples_split:
        Minimal number of samples (>= 2) a node needs to be considered for a split.
    min_information_gain:
        Minimal information gain (bits, >= 0) of the best split. Independently of
        this value, splits with gain <= ``GAIN_TOLERANCE`` are never made.
    class_priority:
        Order of preference used when classes tie for the majority, e.g.
        ``("p", "e")`` to resolve ties towards "poisonous". ``None`` means
        alphabetical order of the labels.

    Ties between attributes with equal gain are broken alphabetically by name.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_information_gain: float = 0.0,
        class_priority: Sequence[Any] | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.class_priority = class_priority

    # -- parameters ---------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_information_gain": self.min_information_gain,
            "class_priority": None if self.class_priority is None else list(self.class_priority),
        }

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"ID3Classifier({params})"

    def _validate_params(self) -> None:
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool) or not isinstance(self.max_depth, (int, np.integer)) or self.max_depth < 0
        ):
            raise ValueError("max_depth must be None or a non-negative integer (root depth = 0)")
        if isinstance(self.min_samples_split, bool) or not isinstance(self.min_samples_split, (int, np.integer)) \
                or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not np.isfinite(self.min_information_gain) or self.min_information_gain < 0:
            raise ValueError("min_information_gain must be a finite number >= 0")

    # -- validation ---------------------------------------------------------
    def _check_fitted(self) -> None:
        if getattr(self, "root_", None) is None:
            raise NotFittedError("this ID3Classifier is not fitted yet; call fit(X, y) first")

    def _check_frame(self, X: Any, *, fitting: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame with one named column per nominal attribute")
        if X.columns.has_duplicates:
            raise ValueError("X has duplicated column names")
        if fitting:
            if X.shape[1] == 0 or X.shape[0] == 0:
                raise ValueError("X must contain at least one row and one feature column")
            if not all(isinstance(c, str) for c in X.columns):
                raise TypeError("feature names (column labels) must be strings")
            frame = X
        else:
            absent = [f for f in self.feature_names_ if f not in X.columns]
            if absent:
                raise ValueError(f"X lacks features seen during fit: {absent}")
            frame = X[self.feature_names_]
        numeric = [c for c in frame.columns
                   if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
        if numeric:
            raise ValueError(
                f"classic ID3 handles nominal attributes only; numeric columns {numeric} must be "
                "discretised first (e.g. pd.cut) or explicitly cast to strings"
            )
        na_counts = frame.isna().sum()
        na_counts = na_counts[na_counts > 0]
        if not na_counts.empty:
            raise ValueError(
                "ID3Classifier does not accept missing values (NaN/pd.NA); found "
                f"{ {str(k): int(v) for k, v in na_counts.items()} }. Convert them explicitly first, "
                "e.g. into a '__MISSING__' category or by imputation fitted on the training split"
            )
        return frame.astype(str)

    def _check_sample(self, sample: Any) -> dict[str, str]:
        if isinstance(sample, pd.DataFrame):
            if len(sample) != 1:
                raise ValueError("predict_one expects exactly one record")
            sample = sample.iloc[0]
        if isinstance(sample, pd.Series):
            sample = sample.to_dict()
        if not isinstance(sample, Mapping):
            raise TypeError("a sample must be a mapping {feature: category} or a pandas Series")
        absent = [f for f in self.feature_names_ if f not in sample]
        if absent:
            raise ValueError(f"sample lacks features seen during fit: {absent}")
        missing = [f for f in self.feature_names_ if pd.isna(sample[f])]
        if missing:
            raise ValueError(f"sample has missing values in {missing}; convert them explicitly first")
        return {f: str(sample[f]) for f in self.feature_names_}

    # -- fitting ------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any) -> ID3Classifier:
        """Grow the tree on nominal features ``X`` and labels ``y``."""
        self._validate_params()
        frame = self._check_frame(X, fitting=True)
        labels = _as_1d_array(y, "y")
        if labels.shape[0] != frame.shape[0]:
            raise ValueError(f"X has {frame.shape[0]} rows but y has {labels.shape[0]} labels")

        self.feature_names_ = list(frame.columns)
        self.n_features_in_ = len(self.feature_names_)
        self.n_samples_fit_ = int(labels.shape[0])
        self.classes_, y_codes = np.unique(labels, return_inverse=True)
        self.classes_ = self.classes_.tolist()
        priority = [c for c in (self.class_priority or []) if c in self.classes_]
        ordered = priority + [c for c in self.classes_ if c not in priority]
        self._class_rank = {label: rank for rank, label in enumerate(ordered)}

        self._columns = {f: frame[f].to_numpy(dtype=object) for f in self.feature_names_}
        self._y_codes = y_codes.ravel()
        self._next_node_id = 0
        try:
            self.root_ = self._build_node(np.arange(self.n_samples_fit_), list(self.feature_names_), depth=0)
        finally:
            del self._columns, self._y_codes
        return self

    def _build_node(self, idx: np.ndarray, features: list[str], depth: int,
                    branch_value: str | None = None) -> ID3Node:
        """Recursive ID3 step for the samples ``idx`` and the unused ``features``."""
        counts = self._count_classes(idx)
        node = ID3Node(
            node_id=self._new_node_id(), depth=depth, n_samples=int(idx.size),
            class_counts=counts, majority_class=self._majority_class(counts),
            entropy=entropy_from_counts(list(counts.values())), branch_value=branch_value,
        )
        # 1-2. Leaves: pure node, exhausted attributes, or a pre-pruning limit.
        if sum(c > 0 for c in counts.values()) == 1:
            return self._as_leaf(node, "pure")
        if not features:
            return self._as_leaf(node, "no_features")
        if self.max_depth is not None and depth >= self.max_depth:
            return self._as_leaf(node, "max_depth")
        if node.n_samples < self.min_samples_split:
            return self._as_leaf(node, "min_samples_split")

        # 3. Greedy choice: attribute with maximal information gain.
        node.candidate_gains = {f: self._gain(f, idx) for f in features}
        best_feature, best_gain = select_best_feature(node.candidate_gains)
        if best_gain <= GAIN_TOLERANCE:
            return self._as_leaf(node, "zero_gain")
        if best_gain < self.min_information_gain - GAIN_TOLERANCE:
            return self._as_leaf(node, "min_information_gain")

        # 4-5. Multiway split on every observed category; the attribute is not reused below.
        node.is_leaf, node.feature, node.information_gain = False, best_feature, best_gain
        remaining = [f for f in features if f != best_feature]
        values = self._columns[best_feature][idx]
        for value in sorted(set(values.tolist())):
            node.children[value] = self._build_node(idx[values == value], remaining, depth + 1, value)
        return node

    def _new_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _count_classes(self, idx: np.ndarray) -> dict[Any, int]:
        counts = np.bincount(self._y_codes[idx], minlength=len(self.classes_))
        return {label: int(count) for label, count in zip(self.classes_, counts)}

    def _majority_class(self, counts: Mapping[Any, int]) -> Any:
        """Most frequent class; ties resolved by ``class_priority`` (default: alphabetical)."""
        return min(self.classes_, key=lambda label: (-counts.get(label, 0), self._class_rank[label]))

    def _gain(self, feature: str, idx: np.ndarray) -> float:
        _, value_codes = np.unique(self._columns[feature][idx], return_inverse=True)
        value_codes = value_codes.ravel()
        table = np.zeros((int(value_codes.max()) + 1, len(self.classes_)), dtype=np.int64)
        np.add.at(table, (value_codes, self._y_codes[idx]), 1)
        return _gain_from_table(table)

    @staticmethod
    def _as_leaf(node: ID3Node, reason: str) -> ID3Node:
        node.is_leaf, node.feature, node.stop_reason = True, None, reason
        return node

    # -- prediction ---------------------------------------------------------
    def _trace(self, sample: Mapping[str, str]) -> PredictionExplanation:
        node, steps = self.root_, []
        while not node.is_leaf:
            value = sample[node.feature]
            child = node.children.get(value)
            steps.append(DecisionStep(
                node_id=node.node_id, depth=node.depth, feature=node.feature, value=value,
                n_samples=node.n_samples, class_counts=dict(node.class_counts), entropy=node.entropy,
                information_gain=float(node.information_gain), matched=child is not None,
            ))
            if child is None:  # category unseen in this node -> majority class stored in the node
                return PredictionExplanation(node.majority_class, tuple(steps), node.node_id, node.n_samples,
                                             dict(node.class_counts), used_fallback=True, fallback_value=value)
            node = child
        return PredictionExplanation(node.majority_class, tuple(steps), node.node_id, node.n_samples,
                                     dict(node.class_counts), used_fallback=False)

    def predict_one(self, sample: Mapping[str, Any] | pd.Series) -> Any:
        """Class predicted for a single record ``{feature: category}``."""
        self._check_fitted()
        return self._trace(self._check_sample(sample)).prediction

    def explain_prediction(self, sample: Mapping[str, Any] | pd.Series) -> PredictionExplanation:
        """Prediction together with every test made on the way from the root."""
        self._check_fitted()
        return self._trace(self._check_sample(sample))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted classes for every row of ``X`` (1-D object array)."""
        self._check_fitted()
        frame = self._check_frame(X, fitting=False)
        records = frame.to_dict(orient="records")
        return np.array([self._trace(record).prediction for record in records], dtype=object)

    def predict_details(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prediction, final node, path length and fallback flag for every row."""
        self._check_fitted()
        frame = self._check_frame(X, fitting=False)
        rows = []
        for record in frame.to_dict(orient="records"):
            trace = self._trace(record)
            rows.append({"prediction": trace.prediction, "final_node_id": trace.final_node_id,
                         "path_length": len(trace.steps), "used_fallback": trace.used_fallback})
        return pd.DataFrame(rows, index=X.index)

    def score(self, X: pd.DataFrame, y: Any) -> float:
        """Accuracy of ``predict(X)`` against ``y``."""
        labels = _as_1d_array(y, "y")
        return float(np.mean(self.predict(X) == labels))

    # -- tree inspection ----------------------------------------------------
    def iter_nodes(self) -> Iterator[ID3Node]:
        """All nodes in pre-order (parent before children, children by category)."""
        self._check_fitted()
        stack = [self.root_]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(list(node.children.values())))

    def get_node(self, node_id: int) -> ID3Node:
        for node in self.iter_nodes():
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def node_paths(self) -> dict[int, tuple[tuple[str, str], ...]]:
        """Conditions (feature, value) leading from the root to every node."""
        self._check_fitted()
        paths: dict[int, tuple[tuple[str, str], ...]] = {}

        def walk(node: ID3Node, conditions: tuple[tuple[str, str], ...]) -> None:
            paths[node.node_id] = conditions
            for value, child in node.children.items():
                walk(child, conditions + ((node.feature, value),))

        walk(self.root_, ())
        return paths

    @property
    def node_count_(self) -> int:
        """Total number of nodes (internal nodes + leaves)."""
        return sum(1 for _ in self.iter_nodes())

    @property
    def leaf_count_(self) -> int:
        return sum(1 for node in self.iter_nodes() if node.is_leaf)

    @property
    def depth_(self) -> int:
        """Depth of the deepest leaf (root depth = 0)."""
        return max(node.depth for node in self.iter_nodes())

    @property
    def feature_importance_bits_(self) -> pd.Series:
        """Weighted total information gain per attribute, in bits.

        imp(A) = sum over internal nodes t testing A of (n_t / N) * IG(t).
        Summed over all attributes this equals H(root) minus the weighted
        entropy of the leaves, i.e. the uncertainty removed by the whole tree.
        """
        self._check_fitted()
        totals = dict.fromkeys(self.feature_names_, 0.0)
        for node in self.iter_nodes():
            if not node.is_leaf:
                totals[node.feature] += node.n_samples / self.root_.n_samples * node.information_gain
        return pd.Series(totals, name="importance_bits").sort_values(ascending=False, kind="mergesort")

    @property
    def feature_importance_(self) -> pd.Series:
        """``feature_importance_bits_`` normalised to sum to 1 (all zeros for a single leaf)."""
        bits = self.feature_importance_bits_
        total = bits.sum()
        normalised = bits / total if total > 0 else bits * 0.0
        return normalised.rename("importance")

    def used_features(self) -> list[str]:
        """Attributes tested anywhere in the tree, in pre-order of first use."""
        seen: list[str] = []
        for node in self.iter_nodes():
            if not node.is_leaf and node.feature not in seen:
                seen.append(node.feature)
        return seen

    def rules(self) -> pd.DataFrame:
        """One decision rule per leaf: conjunction of tests on the path -> class."""
        self._check_fitted()
        rows = []

        def walk(node: ID3Node, conditions: tuple[tuple[str, str], ...]) -> None:
            if node.is_leaf:
                rows.append({"leaf_id": node.node_id, "conditions": conditions, "prediction": node.majority_class,
                             "n_samples": node.n_samples, "class_counts": dict(node.class_counts),
                             "purity": node.purity, "depth": node.depth, "stop_reason": node.stop_reason})
                return
            for value, child in node.children.items():
                walk(child, conditions + ((node.feature, value),))

        walk(self.root_, ())
        return pd.DataFrame(rows)

    def rules_text(self, value_formatter: ValueFormatter | None = None) -> list[str]:
        fmt = value_formatter or (lambda feature, value: str(value))
        lines = []
        for rule in self.rules().itertuples(index=False):
            condition = " ORAZ ".join(f"{f} = {fmt(f, v)}" for f, v in rule.conditions) or "(zawsze)"
            lines.append(f"JEŻELI {condition} TO {rule.prediction}   [n={rule.n_samples}, "
                         f"{_format_counts(rule.class_counts)}]")
        return lines

    def structure_signature(self) -> str:
        """Hash of the tree structure (tests, branches, leaf classes) to detect identical trees."""
        self._check_fitted()

        def shape(node: ID3Node) -> Any:
            if node.is_leaf:
                return ["leaf", str(node.majority_class)]
            return [node.feature, {value: shape(child) for value, child in node.children.items()}]

        canonical = json.dumps(shape(self.root_), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def summary(self) -> dict[str, Any]:
        self._check_fitted()
        return {"node_count": self.node_count_, "leaf_count": self.leaf_count_, "depth": self.depth_,
                "n_samples_fit": self.n_samples_fit_, "n_features_in": self.n_features_in_,
                "used_features": self.used_features(), "classes": list(self.classes_), **self.get_params()}

    # -- persistence (safe formats only: JSON, no pickle) -------------------
    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {"format": "id3-tree", "version": 1, "params": self.get_params(),
                "classes": [str(c) for c in self.classes_], "feature_names": list(self.feature_names_),
                "n_samples_fit": self.n_samples_fit_, "root": self.root_.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ID3Classifier:
        if payload.get("format") != "id3-tree":
            raise ValueError("not a serialised ID3 tree")
        model = cls(**payload["params"])
        model.feature_names_ = list(payload["feature_names"])
        model.n_features_in_ = len(model.feature_names_)
        model.n_samples_fit_ = int(payload["n_samples_fit"])
        model.classes_ = list(payload["classes"])
        priority = [c for c in (model.class_priority or []) if c in model.classes_]
        ordered = priority + [c for c in model.classes_ if c not in priority]
        model._class_rank = {label: rank for rank, label in enumerate(ordered)}
        model.root_ = ID3Node.from_dict(payload["root"])
        return model

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> ID3Classifier:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def export_text(model: ID3Classifier, value_formatter: ValueFormatter | None = None,
                max_depth: int | None = None) -> str:
    """Indented text view of the tree (auxiliary - not a substitute for the diagram)."""
    fmt = value_formatter or (lambda feature, value: str(value))
    model._check_fitted()
    lines = [f"[{model.root_.feature or model.root_.majority_class}]  n={model.root_.n_samples}  "
             f"{_format_counts(model.root_.class_counts)}"]

    def walk(node: ID3Node, prefix: str) -> None:
        items = list(node.children.items())
        for i, (value, child) in enumerate(items):
            last = i == len(items) - 1
            connector, extension = ("└── ", "    ") if last else ("├── ", "│   ")
            label = f"{node.feature} = {fmt(node.feature, value)}"
            if child.is_leaf:
                lines.append(f"{prefix}{connector}{label} -> {child.majority_class}  "
                             f"(n={child.n_samples}; {_format_counts(child.class_counts)})")
            else:
                lines.append(f"{prefix}{connector}{label} -> [{child.feature}]  "
                             f"(n={child.n_samples}; {_format_counts(child.class_counts)})")
                if max_depth is None or child.depth < max_depth:
                    walk(child, prefix + extension)
                else:
                    lines.append(f"{prefix}{extension}└── ...")

    if not model.root_.is_leaf:
        walk(model.root_, "")
    return "\n".join(lines)
