"""Controlled tuning: the 36-configuration grid, the depth curve and a deterministic selection rule.

None of the functions below accepts the test split - model selection can
only see the training and validation parts.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import CLASS_ORDER, classification_metrics
from .data_io import format_condition
from .id3 import ID3Classifier, information_gain_table
from .preprocessing import MISSING_STRATEGIES, CategoricalPreprocessor

PARAM_GRID: dict[str, list[Any]] = {
    "strategy": list(MISSING_STRATEGIES),
    "max_depth": [2, 4, None],
    "min_samples_split": [2, 20],
    "min_information_gain": [0.0, 0.01],
}
DEPTH_CURVE = [1, 2, 3, 4, 5, 6, None]
STRATEGY_PREFERENCE = {"missing_category": 0, "most_frequent": 1, "drop_stalk_root": 2}
SAFETY_CLASS_PRIORITY = ("p", "e")  # majority ties are resolved towards "poisonous"

# Lexicographic selection rule, fixed before looking at any result:
# (column, ascending, description). Levels 6c-6e only make the order total.
SELECTION_RULE: list[tuple[str, bool, str]] = [
    ("val_p_to_e", True, "1. min. liczba błędów krytycznych p → e (walidacja)"),
    ("val_macro_f1", False, "2. maks. macro-F1 (walidacja)"),
    ("leaf_count", True, "3. min. liczba liści"),
    ("node_count", True, "4. min. liczba węzłów"),
    ("depth", True, "5. min. rzeczywista głębokość drzewa"),
    ("strategy_rank", True, "6a. strategia: missing_category > most_frequent > drop_stalk_root"),
    ("max_depth_rank", True, "6b. mniejsze max_depth (None = ∞)"),
    ("min_samples_split", False, "6c. większe min_samples_split (ostrożniejszy wzrost)"),
    ("min_information_gain", False, "6d. większe min_information_gain (ostrożniejszy wzrost)"),
    ("config_id", True, "6e. identyfikator konfiguracji (porządek całkowity)"),
]
PRIMARY_KEYS = ["val_p_to_e", "val_macro_f1", "leaf_count", "node_count", "depth"]
DETERMINISTIC_COLUMNS = [
    "config_id", "strategy", "max_depth_label", "min_samples_split", "min_information_gain",
    "train_accuracy", "val_accuracy", "val_macro_f1", "val_recall_p", "val_p_to_e", "val_e_to_p",
    "depth", "node_count", "leaf_count", "tree_signature",
]


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str
    strategy: str
    max_depth: int | None
    min_samples_split: int
    min_information_gain: float

    def model_params(self) -> dict[str, Any]:
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_information_gain": self.min_information_gain}

    def label(self) -> str:
        return (f"{self.strategy} | max_depth={self.max_depth} | min_samples_split={self.min_samples_split}"
                f" | min_information_gain={self.min_information_gain}")


def _is_none(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def depth_label(value: Any) -> str:
    return "None" if _is_none(value) else str(int(value))


def build_configurations(grid: dict[str, list[Any]] = PARAM_GRID) -> list[ExperimentConfig]:
    """Cartesian product of the grid axes with stable identifiers C01, C02, ..."""
    combos = itertools.product(grid["strategy"], grid["max_depth"], grid["min_samples_split"],
                               grid["min_information_gain"])
    return [ExperimentConfig(f"C{i:02d}", *combo) for i, combo in enumerate(combos, start=1)]


def config_from_row(row: pd.Series | dict[str, Any]) -> ExperimentConfig:
    max_depth = None if _is_none(row["max_depth"]) else int(row["max_depth"])
    return ExperimentConfig(str(row["config_id"]), str(row["strategy"]), max_depth,
                            int(row["min_samples_split"]), float(row["min_information_gain"]))


def _tree_row(model: ID3Classifier) -> dict[str, Any]:
    return {"depth": model.depth_, "node_count": model.node_count_, "leaf_count": model.leaf_count_,
            "n_features_used": len(model.used_features()), "root_feature": model.root_.feature,
            "tree_signature": model.structure_signature()}


def run_experiment_grid(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                        configs: list[ExperimentConfig] | None = None,
                        class_priority: tuple[Any, ...] = SAFETY_CLASS_PRIORITY) -> pd.DataFrame:
    """Train one ID3 per configuration on ``train`` and score it on ``validation``.

    Each missing-value strategy is fitted once on the raw training part and
    applied to the validation part, so all configurations share the same split
    and the same protocol. ``fit_time_ms`` is the only non-deterministic column.
    """
    configs = build_configurations() if configs is None else configs
    prepared: dict[str, tuple[CategoricalPreprocessor, pd.DataFrame, pd.DataFrame]] = {}
    rows = []
    for config in configs:
        if config.strategy not in prepared:
            pre = CategoricalPreprocessor(config.strategy).fit(X_train)
            prepared[config.strategy] = (pre, pre.transform(X_train), pre.transform(X_val))
        pre, X_tr, X_va = prepared[config.strategy]
        model = ID3Classifier(**config.model_params(), class_priority=class_priority)
        start = time.perf_counter()
        model.fit(X_tr, y_train)
        fit_ms = (time.perf_counter() - start) * 1000.0
        train_m = classification_metrics(y_train, model.predict(X_tr), CLASS_ORDER)
        val_m = classification_metrics(y_val, model.predict(X_va), CLASS_ORDER)
        rows.append({
            **asdict(config),
            "max_depth_label": depth_label(config.max_depth),
            "n_features": len(pre.feature_names_out_),
            "train_accuracy": train_m["accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_recall_p": val_m["recall_p"],
            "val_p_to_e": val_m["p_to_e"],
            "val_e_to_p": val_m["e_to_p"],
            **_tree_row(model),
            "val_fallbacks": int(model.predict_details(X_va)["used_fallback"].sum()),
            "fit_time_ms": fit_ms,
        })
    results = pd.DataFrame(rows)
    results["max_depth"] = results["max_depth"].astype(object).where(results["max_depth"].notna(), None)
    return results


def rank_configurations(results: pd.DataFrame) -> pd.DataFrame:
    """Sort configurations by ``SELECTION_RULE``; rank 1 is the selected model."""
    ranked = results.copy()
    ranked["strategy_rank"] = ranked["strategy"].map(STRATEGY_PREFERENCE)
    ranked["max_depth_rank"] = [math.inf if _is_none(d) else float(d) for d in ranked["max_depth"]]
    keys = [column for column, _, _ in SELECTION_RULE]
    ascending = [asc for _, asc, _ in SELECTION_RULE]
    sort_view = ranked.assign(val_macro_f1=ranked["val_macro_f1"].round(12))  # float-noise-proof ties
    order = sort_view.sort_values(keys, ascending=ascending, kind="mergesort").index
    ranked = ranked.loc[order].reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def select_best_configuration(results: pd.DataFrame) -> ExperimentConfig:
    return config_from_row(rank_configurations(results).iloc[0])


def selection_rule_frame() -> pd.DataFrame:
    return pd.DataFrame([{"poziom": description, "kolumna": column, "kierunek": "rosnąco" if asc else "malejąco"}
                         for column, asc, description in SELECTION_RULE])


def metric_tie_groups(ranked: pd.DataFrame) -> pd.DataFrame:
    """Groups of configurations with identical primary criteria (levels 1-5 of the rule)."""
    view = ranked.assign(val_macro_f1=ranked["val_macro_f1"].round(12))
    rows = []
    for key, group in view.groupby(PRIMARY_KEYS, sort=False):
        rows.append({
            "najlepsza pozycja": int(group["rank"].min()),
            "liczba konfiguracji": len(group),
            "konfiguracje": ", ".join(group.sort_values("rank")["config_id"]),
            "p → e": int(key[0]), "macro-F1": float(key[1]), "liście": int(key[2]), "węzły": int(key[3]),
            "głębokość": int(key[4]),
            "różne drzewa": int(group["tree_signature"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("najlepsza pozycja").reset_index(drop=True)


def depth_curve(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                depths: list[int | None] = DEPTH_CURVE, strategies: tuple[str, ...] = MISSING_STRATEGIES,
                min_samples_split: int = 2, min_information_gain: float = 0.0,
                class_priority: tuple[Any, ...] = SAFETY_CLASS_PRIORITY) -> pd.DataFrame:
    """Train/validation quality as a function of ``max_depth`` (other parameters at defaults)."""
    rows = []
    for strategy in strategies:
        pre = CategoricalPreprocessor(strategy).fit(X_train)
        X_tr, X_va = pre.transform(X_train), pre.transform(X_val)
        for max_depth in depths:
            model = ID3Classifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_information_gain=min_information_gain,
                                  class_priority=class_priority).fit(X_tr, y_train)
            train_m = classification_metrics(y_train, model.predict(X_tr), CLASS_ORDER)
            val_m = classification_metrics(y_val, model.predict(X_va), CLASS_ORDER)
            rows.append({"strategy": strategy, "max_depth": max_depth, "max_depth_label": depth_label(max_depth),
                         "train_accuracy": train_m["accuracy"], "val_accuracy": val_m["accuracy"],
                         "val_macro_f1": val_m["macro_f1"], "train_p_to_e": train_m["p_to_e"],
                         "val_p_to_e": val_m["p_to_e"], "val_e_to_p": val_m["e_to_p"], **_tree_row(model)})
    curve = pd.DataFrame(rows)
    curve["max_depth"] = curve["max_depth"].astype(object).where(curve["max_depth"].notna(), None)
    return curve


def fit_final_model(X_train_val: pd.DataFrame, y_train_val: pd.Series, config: ExperimentConfig,
                    class_priority: tuple[Any, ...] = SAFETY_CLASS_PRIORITY
                    ) -> tuple[CategoricalPreprocessor, ID3Classifier]:
    """Refit the selected preprocessing *and* ID3 from scratch on train + validation."""
    pre = CategoricalPreprocessor(config.strategy).fit(X_train_val)
    model = ID3Classifier(**config.model_params(), class_priority=class_priority)
    model.fit(pre.transform(X_train_val), y_train_val)
    return pre, model


def results_fingerprint(results: pd.DataFrame) -> str:
    """SHA-256 of the deterministic columns (timings excluded) - equal across reruns."""
    view = results[DETERMINISTIC_COLUMNS].sort_values("config_id").copy()
    for column in ("train_accuracy", "val_accuracy", "val_macro_f1", "val_recall_p"):
        view[column] = view[column].map(lambda v: f"{v:.12f}")
    return hashlib.sha256(view.to_csv(index=False).encode("utf-8")).hexdigest()


def feature_gain_report(X_train: pd.DataFrame, y_train: pd.Series, model: ID3Classifier, feature: str = "stalk-root",
                        strategies: tuple[str, ...] = ("missing_category", "most_frequent")) -> pd.DataFrame:
    """For every internal node: the winning attribute vs ``feature`` under each missing-value strategy.

    The node subsets are recomputed from the path conditions, so the report
    shows whether a different treatment of ``feature`` could have changed any
    split of ``model`` (it could only if ``feature`` overtook the winner).
    """
    transformed = {s: CategoricalPreprocessor(s).fit(X_train).transform(X_train) for s in strategies}
    rows = []
    for node_id, conditions in model.node_paths().items():
        node = model.get_node(node_id)
        if node.is_leaf:
            continue
        mask = pd.Series(True, index=X_train.index)
        for f, value in conditions:
            mask &= transformed[strategies[0]][f] == value
        label = "korzeń" if not conditions else f"głęb. {len(conditions)}: " + format_condition(*conditions[-1])
        row = {"node": label, "n": int(mask.sum()), "winner": node.feature, "winner_gain": node.information_gain}
        for strategy in strategies:
            table = information_gain_table(transformed[strategy].loc[mask], y_train.loc[mask])
            hit = table[table["feature"] == feature].iloc[0]
            row[f"gain_{strategy}"] = float(hit["information_gain"])
            row[f"rank_{strategy}"] = int(hit["rank"])
            row[f"winner_{strategy}"] = table.iloc[0]["feature"]
        rows.append(row)
    return pd.DataFrame(rows)
