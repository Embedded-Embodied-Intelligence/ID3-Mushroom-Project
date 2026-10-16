"""Mathematical properties of the entropy H(S) used by ID3."""
import math
import warnings

import numpy as np
import pytest

from src.id3 import entropy, entropy_from_counts


def test_empty_set_has_zero_entropy_by_documented_convention():
    assert entropy([]) == 0.0
    assert entropy_from_counts([]) == 0.0


def test_pure_set_has_exactly_zero_entropy():
    h = entropy(["e", "e", "e"])
    assert h == 0.0
    assert math.copysign(1.0, h) == 1.0  # +0.0, never -0.0


def test_balanced_two_classes_give_one_bit():
    assert entropy(["e", "p"]) == 1.0
    assert entropy(["e", "p", "e", "p"]) == 1.0


@pytest.mark.parametrize("k", [2, 3, 4, 8, 13])
def test_uniform_distribution_reaches_log2_k(k):
    assert entropy([f"c{i}" for i in range(k)]) == pytest.approx(math.log2(k), abs=1e-12)


def test_entropy_is_non_negative_and_bounded_on_random_samples():
    rng = np.random.default_rng(42)
    for _ in range(300):
        k = int(rng.integers(1, 7))
        labels = rng.integers(0, k, size=int(rng.integers(1, 60))).astype(str)
        h = entropy(labels)
        assert -1e-12 <= h <= math.log2(k) + 1e-12


def test_zero_probabilities_never_reach_log2():
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        assert entropy_from_counts([5, 0, 0]) == 0.0
        assert entropy_from_counts([3, 0, 3]) == 1.0
        assert entropy_from_counts(np.array([0, 0, 7])) == 0.0


def test_counts_and_labels_versions_agree():
    expected = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
    assert entropy(["a", "a", "b"]) == pytest.approx(expected, abs=1e-12)
    assert entropy_from_counts([2, 1]) == pytest.approx(expected, abs=1e-12)
    assert entropy_from_counts({"a": 2, "b": 1}.values()) == pytest.approx(expected, abs=1e-12)


def test_entropy_rejects_missing_labels_and_negative_counts():
    with pytest.raises(ValueError):
        entropy(["e", None])
    with pytest.raises(ValueError):
        entropy(["e", float("nan")])
    with pytest.raises(ValueError):
        entropy_from_counts([3, -1])


def test_mushroom_class_entropy(mushroom_df):
    p = 4208 / 8124
    expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    assert entropy(mushroom_df["class"]) == pytest.approx(expected, abs=1e-12)
