#!/usr/bin/env python
"""Generate group_exercise/delivery_data.csv - the synthetic "Delivery Decision" workshop data.

The 28 shipments follow explicit, plausible rules (no randomness in the labels):
  * clear weather: on time unless the courier workload is high;
  * rain: express parcels arrive on time; standard ones only over a short distance;
  * snow: every parcel is delayed.
Three controlled missing values are planted: ``weather`` in two shipments (true weather: rain,
both on time) and ``workload`` in one snowy shipment. Row order is shuffled with a fixed seed.

The script does not assume the answer: after writing the CSV it *computes* the information gain
ranking, checks that the root is unique with a clear margin, that no feature vector is duplicated
and that the prediction case is not a training row, and prints the result.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.id3 import ID3Classifier, export_text, information_gain_table  # noqa: E402

CSV_PATH = PROJECT_ROOT / "group_exercise" / "delivery_data.csv"
FEATURES = ["distance", "weather", "workload", "shipping"]
TARGET = "on_time"
NEW_PARCEL = {"distance": "long", "weather": "rain", "workload": "high", "shipping": "express"}
SEED = 2026

ROWS = [  # distance, weather, workload, shipping, label  (comment: planted missing value)
    ("short", "clear", "low", "standard", "on_time"),
    ("long", "clear", "low", "express", "on_time"),
    ("medium", "clear", "low", "standard", "on_time"),
    ("short", "clear", "medium", "express", "on_time"),
    ("long", "clear", "medium", "standard", "on_time"),
    ("medium", "clear", "medium", "express", "on_time"),
    ("long", "clear", "medium", "express", "on_time"),
    ("long", "clear", "high", "standard", "delayed"),
    ("medium", "clear", "high", "express", "delayed"),
    ("short", "clear", "high", "standard", "delayed"),
    ("short", "rain", "high", "express", "on_time"),
    ("medium", None, "low", "express", "on_time"),        # weather missing (true: rain)
    ("medium", "rain", "medium", "express", "on_time"),
    ("long", "rain", "low", "express", "on_time"),
    ("short", None, "medium", "standard", "on_time"),     # weather missing (true: rain)
    ("short", "rain", "high", "standard", "on_time"),
    ("medium", "rain", "low", "standard", "delayed"),
    ("medium", "rain", "high", "standard", "delayed"),
    ("long", "rain", "medium", "standard", "delayed"),
    ("long", "rain", "low", "standard", "delayed"),
    ("short", "snow", "medium", "express", "delayed"),
    ("medium", "snow", "medium", "standard", "delayed"),
    ("long", "snow", "high", "express", "delayed"),
    ("short", "snow", None, "standard", "delayed"),       # workload missing
    ("long", "snow", "low", "standard", "delayed"),
    ("medium", "snow", "high", "express", "delayed"),
    ("short", "snow", "high", "standard", "delayed"),
    ("long", "snow", "medium", "express", "delayed"),
]


def build_frame() -> pd.DataFrame:
    rows = list(ROWS)
    random.Random(SEED).shuffle(rows)
    return pd.DataFrame(rows, columns=[*FEATURES, TARGET])


def mode_fill(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for column in FEATURES:
        counts = df[column].value_counts()
        filled[column] = df[column].fillna(min(counts.index[counts == counts.max()]))
    return filled


def verify(df: pd.DataFrame) -> dict:
    assert 24 <= len(df) <= 32 and list(df.columns) == [*FEATURES, TARGET]
    missing = df[FEATURES].isna().sum()
    assert 2 <= missing.sum() <= 4 and (missing > 0).sum() == 2, missing.to_dict()
    for column in FEATURES:  # unique mode -> unambiguous imputation for the participants
        counts = df[column].value_counts()
        assert (counts == counts.max()).sum() == 1, f"mode of {column} is tied"
    filled = mode_fill(df)
    assert not filled[FEATURES].duplicated().any(), "duplicated feature vectors"
    assert not (filled[FEATURES] == pd.Series(NEW_PARCEL)).all(axis=1).any(), "prediction case is a training row"
    ranking = information_gain_table(filled[FEATURES], filled[TARGET])
    margin = ranking.iloc[0]["information_gain"] - ranking.iloc[1]["information_gain"]
    assert margin > 0.1, f"root not clear enough (margin {margin:.3f})"
    model = ID3Classifier().fit(filled[FEATURES], filled[TARGET])
    assert 2 <= model.depth_ <= 3 and model.node_count_ <= 15, "tree too trivial or too large for 20 minutes"
    return {"ranking": ranking, "margin": margin, "model": model}


def main() -> int:
    df = build_frame()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    result = verify(pd.read_csv(CSV_PATH))
    print(f"Written {CSV_PATH.relative_to(PROJECT_ROOT)}: {len(df)} rows, missing {df.isna().sum().to_dict()}")
    print(result["ranking"].to_string(index=False))
    print(f"root = {result['ranking'].iloc[0]['feature']} (margin {result['margin']:.3f} bit)")
    print(export_text(result["model"]))
    print("prediction case:", NEW_PARCEL, "->", result["model"].predict_one(NEW_PARCEL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
