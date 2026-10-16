"""Classification metrics with an explicit class order and the critical p -> e error."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

CLASS_ORDER = ("e", "p")
# The costly mistake on Mushroom: a poisonous mushroom predicted as edible.
CRITICAL_ERROR = ("p", "e")


def _labels(values: Any) -> np.ndarray:
    return values.to_numpy(dtype=object) if isinstance(values, pd.Series) else np.asarray(values, dtype=object)


def count_errors(y_true: Any, y_pred: Any, true_label: Any, predicted_label: Any) -> int:
    """Number of samples of class ``true_label`` predicted as ``predicted_label``."""
    t, p = _labels(y_true), _labels(y_pred)
    return int(np.sum((t == true_label) & (p == predicted_label)))


def confusion_matrix_frame(y_true: Any, y_pred: Any, labels: Sequence[Any] = CLASS_ORDER) -> pd.DataFrame:
    """Confusion matrix with rows = true class and columns = predicted class, in ``labels`` order."""
    matrix = confusion_matrix(_labels(y_true), _labels(y_pred), labels=list(labels))
    return pd.DataFrame(matrix, index=pd.Index(list(labels), name="klasa rzeczywista"),
                        columns=pd.Index(list(labels), name="klasa przewidziana"))


def classification_metrics(y_true: Any, y_pred: Any, labels: Sequence[Any] = CLASS_ORDER,
                           critical: tuple[Any, Any] = CRITICAL_ERROR) -> dict[str, float | int]:
    """Accuracy, macro/weighted F1, per-class precision/recall/F1 and critical errors."""
    t, p = _labels(y_true), _labels(y_pred)
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(t, p, labels=labels, zero_division=0)
    metrics: dict[str, float | int] = {
        "n": int(t.size),
        "accuracy": float(accuracy_score(t, p)),
        "macro_f1": float(f1_score(t, p, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(t, p, labels=labels, average="weighted", zero_division=0)),
        "n_errors": int(np.sum(t != p)),
    }
    for i, label in enumerate(labels):
        metrics[f"precision_{label}"] = float(precision[i])
        metrics[f"recall_{label}"] = float(recall[i])
        metrics[f"f1_{label}"] = float(f1[i])
        metrics[f"support_{label}"] = int(support[i])
    true_label, predicted_label = critical
    n_critical = count_errors(t, p, true_label, predicted_label)
    n_true = int(np.sum(t == true_label))
    metrics[f"{true_label}_to_{predicted_label}"] = n_critical
    metrics[f"{true_label}_to_{predicted_label}_rate_of_{true_label}"] = n_critical / n_true if n_true else 0.0
    metrics[f"{true_label}_to_{predicted_label}_rate_of_all"] = n_critical / t.size if t.size else 0.0
    metrics[f"{predicted_label}_to_{true_label}"] = count_errors(t, p, predicted_label, true_label)
    return metrics


def classification_report_frame(y_true: Any, y_pred: Any, labels: Sequence[Any] = CLASS_ORDER) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support plus macro and weighted averages."""
    t, p = _labels(y_true), _labels(y_pred)
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(t, p, labels=labels, zero_division=0)
    frame = pd.DataFrame({"precision": precision, "recall": recall, "F1": f1, "support": support},
                         index=pd.Index(labels, name="klasa"))
    for average, name in (("macro", "średnia makro"), ("weighted", "średnia ważona")):
        pr, rc, f, _ = precision_recall_fscore_support(t, p, labels=labels, average=average, zero_division=0)
        frame.loc[name] = [pr, rc, f, int(support.sum())]
    frame["support"] = frame["support"].astype(int)
    return frame


def prediction_table(y_true: Any, y_pred: Any, index: Any = None) -> pd.DataFrame:
    """Side-by-side true vs predicted labels with an error-type column."""
    t, p = _labels(y_true), _labels(y_pred)
    kinds = np.where(t == p, "poprawna", np.char.add(np.char.add(t.astype(str), " -> "), p.astype(str)))
    return pd.DataFrame({"klasa rzeczywista": t, "predykcja": p, "wynik": kinds}, index=index)
