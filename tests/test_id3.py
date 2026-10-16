"""Behaviour of the ID3 classifier: structure, stopping rules, prediction and robustness."""
import inspect
import math

import numpy as np
import pandas as pd
import pytest

import src.id3 as id3_module
from src.id3 import ID3Classifier, NotFittedError, entropy, export_text


def fitted(play_tennis, **params):
    X, y = play_tennis
    return ID3Classifier(**params).fit(X, y)


def test_root_and_structure_match_hand_checked_example(play_tennis):
    model = fitted(play_tennis)
    root = model.root_
    assert root.feature == "Outlook" and not root.is_leaf
    assert set(root.children) == {"Overcast", "Rain", "Sunny"}
    assert root.children["Overcast"].is_leaf and root.children["Overcast"].majority_class == "Yes"
    assert root.children["Sunny"].feature == "Humidity"
    assert root.children["Rain"].feature == "Wind"
    assert (model.node_count_, model.leaf_count_, model.depth_) == (8, 5, 2)
    assert model.root_.information_gain == pytest.approx(0.2467498, abs=1e-6)


def test_node_stores_required_information(play_tennis):
    root = fitted(play_tennis).root_
    assert root.class_counts == {"No": 5, "Yes": 9}
    assert root.majority_class == "Yes" and root.n_samples == 14
    assert root.entropy == pytest.approx(entropy(play_tennis[1]))
    assert set(root.candidate_gains) == {"Outlook", "Temperature", "Humidity", "Wind"}
    leaf = root.children["Overcast"]
    assert leaf.is_leaf and leaf.feature is None and leaf.stop_reason == "pure" and leaf.entropy == 0.0


def test_pure_data_gives_a_single_leaf():
    X = pd.DataFrame({"a": ["x", "y", "z"], "b": ["u", "u", "v"]})
    model = ID3Classifier().fit(X, ["e", "e", "e"])
    assert model.node_count_ == 1 and model.root_.is_leaf and model.root_.stop_reason == "pure"
    assert model.predict(X).tolist() == ["e", "e", "e"]


def test_exhausted_features_return_majority_class():
    X = pd.DataFrame({"f": ["x", "x", "x", "y"]})
    model = ID3Classifier().fit(X, ["e", "e", "p", "p"])
    child = model.root_.children["x"]
    assert child.is_leaf and child.stop_reason == "no_features" and child.majority_class == "e"


def test_majority_ties_are_resolved_deterministically_by_class_priority():
    X = pd.DataFrame({"f": ["x", "x", "y", "y"]})
    y = ["e", "p", "e", "e"]
    alphabetical = ID3Classifier().fit(X, y).root_.children["x"]
    safety_first = ID3Classifier(class_priority=("p", "e")).fit(X, y).root_.children["x"]
    assert alphabetical.class_counts == {"e": 1, "p": 1}
    assert alphabetical.majority_class == "e"
    assert safety_first.majority_class == "p"


def test_unseen_category_falls_back_to_majority_of_last_known_node(play_tennis):
    model = fitted(play_tennis)
    fog = {"Outlook": "Fog", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}
    explanation = model.explain_prediction(fog)
    assert explanation.used_fallback and explanation.final_node_id == model.root_.node_id
    assert explanation.prediction == model.root_.majority_class == "Yes"
    sunny_extreme = {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "Extreme", "Wind": "Weak"}
    explanation = model.explain_prediction(sunny_extreme)
    sunny = model.root_.children["Sunny"]
    assert explanation.used_fallback and explanation.final_node_id == sunny.node_id
    assert explanation.prediction == sunny.majority_class == "No"
    assert model.predict(pd.DataFrame([fog, sunny_extreme])).tolist() == ["Yes", "No"]


@pytest.mark.parametrize("missing", [np.nan, None, pd.NA])
def test_missing_values_without_preprocessing_raise_value_error(play_tennis, missing):
    X, y = play_tennis
    X_bad = X.astype(object)
    X_bad.loc[3, "Wind"] = missing
    with pytest.raises(ValueError, match="missing values"):
        ID3Classifier().fit(X_bad, y)
    model = ID3Classifier().fit(X, y)
    with pytest.raises(ValueError, match="missing values"):
        model.predict(X_bad)
    with pytest.raises(ValueError, match="missing values"):
        model.predict_one(X_bad.loc[3])
    with pytest.raises(ValueError, match="missing values"):
        ID3Classifier().fit(X, list(y[:-1]) + [missing])


def test_schema_validation(play_tennis):
    X, y = play_tennis
    with pytest.raises(ValueError, match="nominal"):
        ID3Classifier().fit(X.assign(temp_c=np.linspace(5.0, 30.0, len(X))), y)
    with pytest.raises(TypeError):
        ID3Classifier().fit(X.to_numpy(), y)
    with pytest.raises(ValueError, match="rows"):
        ID3Classifier().fit(X, y[:-1])
    model = ID3Classifier().fit(X, y)
    with pytest.raises(ValueError, match="lacks features"):
        model.predict(X.drop(columns="Wind"))
    # Extra columns are ignored, column order does not matter at prediction time.
    shuffled = X[["Wind", "Humidity", "Temperature", "Outlook"]].assign(extra="z")
    assert model.predict(shuffled).tolist() == model.predict(X).tolist()


def test_max_depth_convention_root_depth_zero(play_tennis):
    stump0 = fitted(play_tennis, max_depth=0)
    assert stump0.node_count_ == 1 and stump0.root_.stop_reason == "max_depth"
    stump1 = fitted(play_tennis, max_depth=1)
    assert (stump1.depth_, stump1.node_count_, stump1.leaf_count_) == (1, 4, 3)
    assert stump1.root_.children["Sunny"].stop_reason == "max_depth"
    assert stump1.root_.children["Sunny"].majority_class == "No"


def test_min_samples_split_stops_small_nodes(play_tennis):
    model = fitted(play_tennis, min_samples_split=6)
    assert (model.node_count_, model.leaf_count_) == (4, 3)
    assert model.root_.children["Rain"].stop_reason == "min_samples_split"
    assert fitted(play_tennis, min_samples_split=15).root_.stop_reason == "min_samples_split"


def test_min_information_gain_threshold(play_tennis):
    blocked = fitted(play_tennis, min_information_gain=0.3)
    assert blocked.root_.is_leaf and blocked.root_.stop_reason == "min_information_gain"
    assert blocked.root_.majority_class == "Yes"
    allowed = fitted(play_tennis, min_information_gain=0.2)
    assert allowed.structure_signature() == fitted(play_tennis).structure_signature()


def test_zero_gain_never_creates_branches():
    X = pd.DataFrame({"a": ["0", "0", "1", "1"], "b": ["0", "1", "0", "1"]})
    model = ID3Classifier().fit(X, ["n", "y", "y", "n"])  # XOR: every single attribute has IG = 0
    assert model.root_.is_leaf and model.root_.stop_reason == "zero_gain"


def test_multiway_split_uses_observed_values_and_never_reuses_an_attribute(mushroom_split):
    X, y, split = mushroom_split
    X_train = X.loc[split.train].fillna("__MISSING__")
    model = ID3Classifier().fit(X_train, y.loc[split.train])
    for node in model.iter_nodes():
        if node.is_leaf:
            continue
        assert len(node.children) >= 2
        assert sum(child.n_samples for child in node.children.values()) == node.n_samples

    def walk(node, used):
        if node.is_leaf:
            return
        assert node.feature not in used
        for child in node.children.values():
            walk(child, used | {node.feature})

    walk(model.root_, set())
    assert model.root_.feature == "odor"
    assert set(model.root_.children) == set(X_train["odor"].unique())


def test_predict_one_and_batch_agree_and_have_valid_format(play_tennis):
    X, y = play_tennis
    model = ID3Classifier().fit(X, y)
    batch = model.predict(X)
    assert isinstance(batch, np.ndarray) and batch.shape == (len(X),) and batch.dtype == object
    assert set(batch) <= set(model.classes_)
    assert [model.predict_one(row) for _, row in X.iterrows()] == batch.tolist()
    assert model.predict_one(X.iloc[[0]]) == batch[0]
    assert model.score(X, y) == 1.0


def test_fit_is_deterministic_and_order_invariant(play_tennis):
    X, y = play_tennis
    a, b = ID3Classifier().fit(X, y), ID3Classifier().fit(X, y)
    assert a.to_dict() == b.to_dict()
    reversed_columns = ID3Classifier().fit(X[X.columns[::-1]], y)
    permutation = np.random.default_rng(3).permutation(len(X))
    shuffled_rows = ID3Classifier().fit(X.iloc[permutation], y.iloc[permutation])
    assert a.structure_signature() == reversed_columns.structure_signature() == shuffled_rows.structure_signature()
    assert (a.predict(X) == shuffled_rows.predict(X)).all()


def test_explain_prediction_is_consistent_with_predict(play_tennis):
    X, y = play_tennis
    model = ID3Classifier().fit(X, y)
    for _, row in X.iterrows():
        explanation = model.explain_prediction(row)
        assert explanation.prediction == model.predict_one(row)
        assert [step.feature for step in explanation.steps][0] == "Outlook"
        assert not explanation.used_fallback
        assert "liść" in explanation.to_text()
        assert len(explanation.to_frame()) == len(explanation.steps)


def test_feature_importance_follows_its_definition(play_tennis):
    model = fitted(play_tennis)
    bits = model.feature_importance_bits_
    h_23 = -(2 / 5) * math.log2(2 / 5) - (3 / 5) * math.log2(3 / 5)
    assert bits["Outlook"] == pytest.approx(model.root_.information_gain)
    assert bits["Humidity"] == pytest.approx(5 / 14 * h_23)
    assert bits["Temperature"] == 0.0
    leaves = [n for n in model.iter_nodes() if n.is_leaf]
    removed = model.root_.entropy - sum(n.n_samples / 14 * n.entropy for n in leaves)
    assert bits.sum() == pytest.approx(removed)
    assert model.feature_importance_.sum() == pytest.approx(1.0)


def test_json_round_trip_preserves_predictions(play_tennis, tmp_path):
    X, y = play_tennis
    model = ID3Classifier(class_priority=("No", "Yes")).fit(X, y)
    restored = ID3Classifier.load_json(model.save_json(tmp_path / "tree.json"))
    assert restored.predict(X).tolist() == model.predict(X).tolist()
    assert restored.structure_signature() == model.structure_signature()
    assert restored.node_count_ == model.node_count_


def test_rules_and_text_export(play_tennis):
    model = fitted(play_tennis)
    rules = model.rules()
    assert len(rules) == model.leaf_count_ and rules["n_samples"].sum() == 14
    assert any(line.startswith("JEŻELI Outlook = Sunny ORAZ Humidity = High TO No") for line in model.rules_text())
    text = export_text(model)
    assert "Outlook = Overcast -> Yes" in text and text.count("\n") == model.node_count_ - 1


@pytest.mark.parametrize("params", [{"max_depth": -1}, {"max_depth": 1.5}, {"min_samples_split": 1},
                                    {"min_information_gain": -0.1}, {"min_information_gain": float("nan")}])
def test_invalid_parameters_are_rejected(play_tennis, params):
    with pytest.raises(ValueError):
        fitted(play_tennis, **params)


def test_unfitted_model_raises():
    with pytest.raises(NotFittedError):
        ID3Classifier().predict(pd.DataFrame({"a": ["x"]}))
    with pytest.raises(NotFittedError):
        _ = ID3Classifier().node_count_


def test_implementation_is_independent_of_sklearn_trees():
    source = inspect.getsource(id3_module)
    assert "sklearn" not in source and "DecisionTreeClassifier" not in source
