"""Information gain, split information and the ID3 ranking of attributes."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mutual_info_score

from src.id3 import (
    GAIN_TOLERANCE,
    entropy,
    gain_ratio,
    information_gain,
    information_gain_table,
    select_best_feature,
    split_information,
)

# Hand computation for the PlayTennis example (Quinlan 1986, Mitchell 1997).
H_S = -(9 / 14) * math.log2(9 / 14) - (5 / 14) * math.log2(5 / 14)
H_23 = -(2 / 5) * math.log2(2 / 5) - (3 / 5) * math.log2(3 / 5)
EXPECTED_GAIN = {
    "Outlook": H_S - (10 / 14) * H_23,
    "Humidity": H_S - 0.5 * (-(3 / 7) * math.log2(3 / 7) - (4 / 7) * math.log2(4 / 7))
                - 0.5 * (-(6 / 7) * math.log2(6 / 7) - (1 / 7) * math.log2(1 / 7)),
    "Wind": H_S - (8 / 14) * (-(6 / 8) * math.log2(6 / 8) - (2 / 8) * math.log2(2 / 8)) - (6 / 14) * 1.0,
    "Temperature": H_S - (4 / 14) * 1.0 - (6 / 14) * (-(4 / 6) * math.log2(4 / 6) - (2 / 6) * math.log2(2 / 6))
                   - (4 / 14) * (-(3 / 4) * math.log2(3 / 4) - (1 / 4) * math.log2(1 / 4)),
}


def test_hand_computed_gains_on_play_tennis(play_tennis):
    X, y = play_tennis
    for feature, expected in EXPECTED_GAIN.items():
        assert information_gain(X[feature], y) == pytest.approx(expected, abs=1e-12)
    assert EXPECTED_GAIN["Outlook"] == pytest.approx(0.2467498, abs=1e-6)


def test_perfect_split_gain_equals_parent_entropy():
    labels = ["e", "e", "p", "p", "p", "x"]
    assert information_gain(["a", "a", "b", "b", "b", "c"], labels) == pytest.approx(entropy(labels), abs=1e-12)


def test_constant_feature_has_zero_gain():
    assert information_gain(["k"] * 6, ["e", "p", "e", "p", "e", "e"]) == 0.0


def test_gain_is_bounded_and_matches_independent_mutual_information():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(2, 80))
        x = rng.integers(0, int(rng.integers(1, 6)), size=n).astype(str)
        y = rng.integers(0, int(rng.integers(1, 4)), size=n).astype(str)
        gain = information_gain(x, y)
        assert -GAIN_TOLERANCE <= gain <= entropy(y) + GAIN_TOLERANCE
        assert gain == pytest.approx(mutual_info_score(y, x) / math.log(2), abs=1e-10)


def test_identifier_like_attribute_shows_high_cardinality_bias():
    labels = ["e", "p", "e", "p", "e", "e", "p", "e"]
    identifier = [f"id{i}" for i in range(len(labels))]
    assert information_gain(identifier, labels) == pytest.approx(entropy(labels), abs=1e-12)
    assert split_information(identifier) == pytest.approx(math.log2(len(labels)), abs=1e-12)
    assert gain_ratio(identifier, labels) < information_gain(identifier, labels)


def test_gain_ratio_of_single_valued_attribute_is_zero():
    assert gain_ratio(["a", "a", "a"], ["e", "p", "e"]) == 0.0


def test_ranking_uses_alphabetical_tie_break():
    X = pd.DataFrame({"zeta": ["a", "a", "b", "b"], "alpha": ["a", "a", "b", "b"], "noise": ["a", "b", "a", "b"]})
    y = ["e", "e", "p", "p"]
    table = information_gain_table(X, y)
    assert table["feature"].tolist() == ["alpha", "zeta", "noise"]
    assert select_best_feature({"zeta": 0.5, "alpha": 0.5 - GAIN_TOLERANCE / 10, "beta": 0.1}) == ("alpha", 0.5 - GAIN_TOLERANCE / 10)
    # The result does not depend on the order in which candidates are listed.
    assert select_best_feature({"b": 0.3, "a": 0.3}) == select_best_feature({"a": 0.3, "b": 0.3})


def test_odor_has_the_highest_gain_on_mushroom(mushroom_split):
    X, y, split = mushroom_split
    table = information_gain_table(X.loc[split.train].fillna("__MISSING__"), y.loc[split.train])
    assert table.iloc[0]["feature"] == "odor"
    assert table.set_index("feature").loc["veil-type", "information_gain"] == 0.0
