"""Data loading contract, quality audit, associations, split protocol and missing-value strategies."""
import io
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy.stats.contingency import association

from tests.conftest import DATA_ZIP
from src.data_io import (
    COLUMN_NAMES,
    DataContractError,
    MISSING_TOKEN,
    compare_with_names_file,
    ensure_mushroom_zip,
    expand_value,
    format_condition,
    load_mushroom,
    read_raw_mushroom,
    validate_raw_contract,
)
from src.preprocessing import (
    MISSING_STRATEGIES,
    CategoricalPreprocessor,
    HoldoutVault,
    association_with_target,
    cramers_v,
    cramers_v_matrix,
    data_quality_audit,
    duplicate_audit,
    duplicate_stats,
    find_constant_features,
    make_stratified_split,
    overlap_stats,
    split_features_target,
)


def test_loader_reads_official_zip_without_header():
    raw = read_raw_mushroom(DATA_ZIP)
    assert raw.shape == (8124, 23) and list(raw.columns) == COLUMN_NAMES
    first_line = "p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u".split(",")
    assert raw.iloc[0].tolist() == first_line
    assert raw["class"].value_counts().to_dict() == {"e": 4208, "p": 3916}


def test_question_marks_are_counted_on_raw_data_then_become_nan():
    raw = read_raw_mushroom(DATA_ZIP)
    report = validate_raw_contract(raw)
    assert report["zgodne"].all()
    assert int((raw["stalk-root"] == "?").sum()) == 2480
    data = load_mushroom(DATA_ZIP)
    assert data.isna().sum().to_dict() == {c: (2480 if c == "stalk-root" else 0) for c in COLUMN_NAMES}
    assert not (data.astype(object) == "?").any().any()


def _zip_with(tmp_path, text, member="agaricus-lepiota.data"):
    path = tmp_path / "variant.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, text)
    return path


def test_contract_violations_raise_clear_errors(tmp_path):
    lines = zipfile.ZipFile(DATA_ZIP).read("agaricus-lepiota.data").decode().splitlines()
    with pytest.raises(DataContractError, match="liczba wierszy"):
        load_mushroom(_zip_with(tmp_path, "\n".join(lines[:100]) + "\n"))
    with pytest.raises(DataContractError, match="columns"):
        read_raw_mushroom(_zip_with(tmp_path, "\n".join(line + ",x" for line in lines) + "\n"))
    with pytest.raises(DataContractError, match="exactly one"):
        read_raw_mushroom(_zip_with(tmp_path, "a,b\n", member="other.csv"))


def test_local_zip_is_used_without_network(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network must not be used when the local ZIP exists")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert ensure_mushroom_zip(DATA_ZIP) == DATA_ZIP


def test_failed_download_explains_manual_upload(monkeypatch, tmp_path):
    def offline(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(FileNotFoundError, match="files.upload"):
        ensure_mushroom_zip(tmp_path / "data" / "mushroom.zip")


def test_category_dictionary_matches_names_file_and_labels_are_expanded():
    assert compare_with_names_file(DATA_ZIP)["zgodne"].all()
    assert expand_value("odor", "f") == "foul (f)"
    assert format_condition("spore-print-color", "r") == "spore-print-color = green (r)"
    assert expand_value("stalk-root", MISSING_TOKEN) == expand_value("stalk-root", np.nan) == "brak danych (?)"


def test_quality_audit_reports_negative_results(mushroom_df):
    audit = data_quality_audit(mushroom_df).set_index("kontrola")
    assert audit.loc["wartości spoza słownika UCI", "wynik"] == "0"
    assert audit.loc["pełne duplikaty wierszy (cechy + klasa)", "wynik"] == "0"
    assert audit.loc["cechy stałe (1 wartość)", "wynik"] == "veil-type"
    assert "2480 (30.53%)" in audit.loc["braki danych ('?' -> NaN)", "wynik"]


def test_constant_veil_type_is_detected_dropped_and_safe_for_cramers_v(mushroom_df, mushroom_split):
    X, y, split = mushroom_split
    assert find_constant_features(X) == ["veil-type"]
    assert find_constant_features(X.loc[split.train]) == ["veil-type"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cramers_v(X["veil-type"], y) == 0.0
        matrix = cramers_v_matrix(mushroom_df)
    assert not matrix.isna().any().any()
    assert matrix.loc["veil-type"].abs().sum() == 0.0
    pre = CategoricalPreprocessor("missing_category").fit(X.loc[split.train])
    assert pre.constant_features_ == ["veil-type"] and "veil-type" not in pre.transform(X).columns


def test_cramers_v_properties():
    rng = np.random.default_rng(0)
    x = rng.choice(list("abc"), size=3000)
    assert cramers_v(x, x) == pytest.approx(1.0)
    assert cramers_v(x, rng.choice(list("uv"), size=3000)) < 0.06
    y = np.where(rng.random(3000) < 0.8, x, "z")
    assert cramers_v(x, y) == pytest.approx(cramers_v(y, x))
    table = pd.crosstab(pd.Series(x), pd.Series(y)).to_numpy()
    assert cramers_v(x, y) == pytest.approx(association(table, method="cramer"), abs=1e-12)
    with pytest.raises(ValueError):
        cramers_v(["a", None], ["e", "p"])


def test_mutual_information_equals_root_information_gain(mushroom_split):
    from src.id3 import information_gain
    X, y, _ = mushroom_split
    table = association_with_target(X, y).set_index("cecha")
    assert table.index[0] == "odor"
    assert table.loc["odor", "I(A;Y) [bit]"] == pytest.approx(information_gain(X["odor"], y), abs=1e-10)


def test_split_is_60_20_20_disjoint_stratified_and_reproducible(mushroom_split):
    X, y, split = mushroom_split
    assert (len(split.train), len(split.validation), len(split.test)) == (4874, 1625, 1625)
    split.assert_disjoint(len(y))
    assert set(split.train) | set(split.validation) | set(split.test) == set(y.index)
    overall = (y == "p").mean()
    for idx in split.parts().values():
        assert abs((y.loc[idx] == "p").mean() - overall) < 0.002
    again = make_stratified_split(y)
    assert all(np.array_equal(a, b) for a, b in zip(split.parts().values(), again.parts().values()))
    assert len(split.assignment()) == len(y)


def test_preprocessing_is_fitted_on_train_only(mushroom_split):
    X, y, split = mushroom_split
    X_train = X.loc[split.train]
    pre = CategoricalPreprocessor("most_frequent").fit(X_train)
    assert set(pre.fit_index_) == set(split.train) and not set(pre.fit_index_) & set(split.test)
    train_mode = X_train["stalk-root"].value_counts().idxmax()
    assert pre.fill_values_["stalk-root"] == train_mode
    # Manipulating validation/test cannot influence what was learned on train.
    tampered = X.copy()
    tampered.loc[split.test, "stalk-root"] = "r"
    tampered.loc[split.validation, "stalk-root"] = "r"
    assert CategoricalPreprocessor("most_frequent").fit(tampered.loc[split.train]).fill_values_ == pre.fill_values_
    X_test = X.loc[split.test]
    filled = pre.transform(X_test)["stalk-root"]
    assert (filled[X_test["stalk-root"].isna()] == train_mode).all()


@pytest.mark.parametrize("strategy", MISSING_STRATEGIES)
def test_strategies_leave_no_missing_values_and_a_consistent_schema(mushroom_split, strategy):
    X, y, split = mushroom_split
    pre = CategoricalPreprocessor(strategy).fit(X.loc[split.train])
    parts = [pre.transform(X.loc[idx]) for idx in split.parts().values()]
    for part in parts:
        assert not part.isna().any().any()
        assert list(part.columns) == pre.feature_names_out_
        assert all(isinstance(v, str) for v in part.iloc[:50].to_numpy().ravel())
    columns = pre.feature_names_out_
    assert "veil-type" not in columns
    if strategy == "drop_stalk_root":
        assert "stalk-root" not in columns
    elif strategy == "missing_category":
        assert (parts[0]["stalk-root"] == MISSING_TOKEN).sum() == X.loc[split.train, "stalk-root"].isna().sum()
    else:
        assert MISSING_TOKEN not in set(parts[0]["stalk-root"])


def test_unseen_categories_are_reported_not_hidden(mushroom_split):
    X, y, split = mushroom_split
    pre = CategoricalPreprocessor("missing_category").fit(X.loc[split.train])
    probe = X.loc[split.validation].iloc[:3].copy()
    probe.loc[probe.index[0], "odor"] = "q"
    assert pre.unseen_categories(probe) == {"odor": ["q"]}
    with pytest.raises(ValueError):
        CategoricalPreprocessor("bogus").fit(X)


def test_holdout_vault_allows_exactly_one_final_unlock(mushroom_split):
    X, y, split = mushroom_split
    vault = HoldoutVault(X.loc[split.test], y.loc[split.test])
    X_audit, _ = vault.audit_view("duplikaty")
    assert len(X_audit) == len(vault) == 1625
    X_test, y_test = vault.unlock("final evaluation")
    assert vault.is_unlocked and len(y_test) == 1625
    with pytest.raises(RuntimeError):
        vault.unlock("second look")
    with pytest.raises(RuntimeError):
        vault.audit_view("late audit")
    assert vault.log_frame()["typ dostępu"].tolist() == ["audyt danych (bez modelu)", "jednorazowa ocena modelu"]


def test_duplicate_tools_detect_planted_duplicates_and_conflicts():
    X = pd.DataFrame({"a": ["x", "x", "y", "y", "z"], "b": ["u", "u", np.nan, np.nan, "v"]})
    y = pd.Series(["e", "e", "e", "p", "p"])
    assert duplicate_stats(X, y) == {"duplicate_feature_vectors": 2, "duplicate_rows": 1, "conflicting_vectors": 1}
    assert overlap_stats(X.iloc[:2], y.iloc[:2], X.iloc[2:], y.iloc[2:]) == {"overlap": 0, "conflicting": 0}
    assert overlap_stats(X.iloc[[2]], y.iloc[[2]], X.iloc[[3]], y.iloc[[3]]) == {"overlap": 1, "conflicting": 1}


def test_duplicate_audit_on_mushroom(mushroom_split):
    X, y, split = mushroom_split
    parts = {name: (X.loc[idx], y.loc[idx]) for name, idx in split.parts().items()}
    audit = duplicate_audit(parts)
    assert len(audit) == 1 + len(MISSING_STRATEGIES)
    assert audit["liczba cech"].tolist() == [22, 21, 21, 20]
    numeric = audit.drop(columns=["wariant", "liczba cech"])
    assert (numeric >= 0).all().all()
