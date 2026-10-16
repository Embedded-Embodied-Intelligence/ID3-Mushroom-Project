"""The 36-model tuning protocol, the selection rule and the final refit."""
import inspect

import numpy as np
import pandas as pd
import pytest

from src.evaluation import classification_metrics, classification_report_frame, confusion_matrix_frame
from src.experiments import (
    PARAM_GRID,
    SELECTION_RULE,
    build_configurations,
    depth_curve,
    fit_final_model,
    metric_tie_groups,
    rank_configurations,
    results_fingerprint,
    run_experiment_grid,
    select_best_configuration,
)

REQUIRED_COLUMNS = ["config_id", "strategy", "max_depth", "min_samples_split", "min_information_gain",
                    "train_accuracy", "val_accuracy", "val_macro_f1", "val_recall_p", "val_p_to_e",
                    "depth", "node_count", "leaf_count", "fit_time_ms"]


@pytest.fixture(scope="module")
def grid_results(mushroom_split):
    X, y, split = mushroom_split
    return run_experiment_grid(X.loc[split.train], y.loc[split.train], X.loc[split.validation],
                               y.loc[split.validation])


def test_grid_has_36_unique_configurations():
    configs = build_configurations()
    assert len(configs) == 3 * 3 * 2 * 2 == 36
    assert len({(c.strategy, c.max_depth, c.min_samples_split, c.min_information_gain) for c in configs}) == 36
    assert [c.config_id for c in configs[:3]] == ["C01", "C02", "C03"]
    assert set(PARAM_GRID["max_depth"]) == {2, 4, None}


def test_report_is_complete_and_reproducible(mushroom_split, grid_results):
    assert len(grid_results) == 36
    assert not grid_results[[c for c in REQUIRED_COLUMNS if c != "max_depth"]].isna().any().any()
    X, y, split = mushroom_split
    again = run_experiment_grid(X.loc[split.train], y.loc[split.train], X.loc[split.validation],
                                y.loc[split.validation])
    assert results_fingerprint(again) == results_fingerprint(grid_results)
    assert (grid_results["fit_time_ms"] > 0).all()


def test_tuning_cannot_see_the_test_split(mushroom_split, grid_results):
    for function in (run_experiment_grid, depth_curve, rank_configurations, select_best_configuration):
        assert not any("test" in name for name in inspect.signature(function).parameters)
    X, y, split = mushroom_split
    X_corrupt, y_corrupt = X.copy(), y.copy()
    y_corrupt.loc[split.test] = np.where(y.loc[split.test] == "e", "p", "e")
    X_corrupt.loc[split.test, "odor"] = "n"
    rerun = run_experiment_grid(X_corrupt.loc[split.train], y_corrupt.loc[split.train],
                                X_corrupt.loc[split.validation], y_corrupt.loc[split.validation])
    assert results_fingerprint(rerun) == results_fingerprint(grid_results)
    assert select_best_configuration(rerun) == select_best_configuration(grid_results)


def _synthetic(rows):
    base = {"strategy": "missing_category", "max_depth": None, "min_samples_split": 2,
            "min_information_gain": 0.0, "val_p_to_e": 0, "val_macro_f1": 1.0, "leaf_count": 10,
            "node_count": 12, "depth": 3, "tree_signature": "t"}
    return pd.DataFrame([{**base, "config_id": f"X{i}", **row} for i, row in enumerate(rows)])


@pytest.mark.parametrize("winner, loser", [
    ({"val_p_to_e": 0, "val_macro_f1": 0.90}, {"val_p_to_e": 1, "val_macro_f1": 1.0}),
    ({"val_macro_f1": 0.99}, {"val_macro_f1": 0.98}),
    ({"leaf_count": 9, "node_count": 20}, {"leaf_count": 10, "node_count": 11}),
    ({"node_count": 11}, {"node_count": 12}),
    ({"depth": 2}, {"depth": 3}),
    ({"strategy": "missing_category", "max_depth": None}, {"strategy": "most_frequent", "max_depth": 2}),
    ({"strategy": "most_frequent"}, {"strategy": "drop_stalk_root"}),
    ({"max_depth": 4}, {"max_depth": None}),
    ({"max_depth": 4, "min_samples_split": 20}, {"max_depth": 4, "min_samples_split": 2}),
    ({"min_information_gain": 0.01}, {"min_information_gain": 0.0}),
])
def test_selection_rule_is_lexicographic(winner, loser):
    ranked = rank_configurations(_synthetic([loser, winner]))
    assert ranked.iloc[0]["config_id"] == "X1"
    assert ranked["rank"].tolist() == [1, 2]


def test_selection_rule_is_fixed_and_documented():
    assert [column for column, _, _ in SELECTION_RULE][:5] == ["val_p_to_e", "val_macro_f1", "leaf_count",
                                                                "node_count", "depth"]


def test_selected_configuration_and_ties_on_mushroom(grid_results):
    ranked = rank_configurations(grid_results)
    best = select_best_configuration(grid_results)
    top = ranked.iloc[0]
    assert top["config_id"] == best.config_id and top["val_p_to_e"] == ranked["val_p_to_e"].min()
    ties = metric_tie_groups(ranked)
    assert ties["liczba konfiguracji"].sum() == 36
    assert ties.iloc[0]["najlepsza pozycja"] == 1


def test_depth_curve_covers_full_range(mushroom_split):
    X, y, split = mushroom_split
    curve = depth_curve(X.loc[split.train], y.loc[split.train], X.loc[split.validation],
                        y.loc[split.validation], strategies=("missing_category",))
    assert curve["max_depth_label"].tolist() == ["1", "2", "3", "4", "5", "6", "None"]
    assert curve["train_accuracy"].is_monotonic_increasing
    assert curve["depth"].iloc[-1] == curve["depth"].max()


def test_final_model_is_refitted_on_train_plus_validation(mushroom_split, grid_results):
    X, y, split = mushroom_split
    config = select_best_configuration(grid_results)
    trval = split.train_validation()
    pre, model = fit_final_model(X.loc[trval], y.loc[trval], config)
    assert pre.n_samples_fit_ == model.n_samples_fit_ == len(split.train) + len(split.validation) == 6499
    assert not set(pre.fit_index_) & set(split.test)
    assert model.get_params()["max_depth"] == config.max_depth


def test_metrics_are_consistent_with_the_confusion_matrix():
    y_true = ["e", "e", "p", "p", "p", "e"]
    y_pred = ["e", "p", "e", "p", "p", "e"]
    cm = confusion_matrix_frame(y_true, y_pred)
    metrics = classification_metrics(y_true, y_pred)
    assert cm.loc["p", "e"] == metrics["p_to_e"] == 1 and cm.loc["e", "p"] == metrics["e_to_p"] == 1
    assert cm.to_numpy().sum() == metrics["n"] == 6
    assert metrics["accuracy"] == pytest.approx(np.trace(cm.to_numpy()) / 6)
    assert metrics["recall_p"] == pytest.approx(2 / 3)
    report = classification_report_frame(y_true, y_pred)
    assert report.loc["p", "recall"] == pytest.approx(metrics["recall_p"])
    assert report.loc["średnia makro", "F1"] == pytest.approx(metrics["macro_f1"])
