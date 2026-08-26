"""
Feature extraction for the synthetic fashion SKU demand dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from dependancies.params_utils import load_params
from dependancies.Features_Functions import get_train_cutoff, add_abc_classification, add_xyz_classification, add_calendar_features, \
    add_lifecycle_features, add_price_features, add_target_lag_features, add_endogenous_lagged_features, add_cross_product_features, encode_categoricals, time_based_split  

_p = load_params()["features"]
GRANULARITY = _p["granularity"]

FORECAST_HORIZON = _p["forecast_horizon"]
VALIDATION_HORIZON = _p["validation_horizon"]

# setting lags to 0, 1, 2, 4, 44 for weekly data (or 0, 6, 13, 29, 364 for daily)
_LAG_OFFSETS = [0, 1, 2, 4, 44] if GRANULARITY == "W" else [0, 6, 13, 29, 364]
LAGS = [FORECAST_HORIZON + off for off in _LAG_OFFSETS]  # e.g. [8, 9, 10, 12, 52]

# setting rolling windows to 4, 12 for weekly data (or 7, 30 for daily)
ROLLING_WINDOWS = [4, 12] if GRANULARITY == "W" else [7, 30]


if __name__ == "__main__":
    input_dir = Path(__file__).resolve().parent / "input"
    print(input_dir)
    df = pd.read_csv(input_dir / "synthetic_fashion_demand.csv", parse_dates=["date"])

    train_cutoff = get_train_cutoff(df)
    print(f"train cutoff for ABC/XYZ classification: {train_cutoff.date()}")

    df = add_abc_classification(df, as_of=train_cutoff)
    df = add_xyz_classification(df, as_of=train_cutoff)
    df = add_calendar_features(df)
    df = add_lifecycle_features(df)
    df = add_price_features(df)
    df = add_target_lag_features(df)
    df = add_endogenous_lagged_features(df)
    df = add_cross_product_features(df)
    df = encode_categoricals(df)
    train, val, test = time_based_split(df)

    df.to_csv(input_dir / "synthetic_fashion_demand_features.csv", index=False)
    train.to_csv(input_dir / "train_features.csv", index=False)
    val.to_csv(input_dir / "val_features.csv", index=False)
    test.to_csv(input_dir / "test_features.csv", index=False)
    print(f"train/val/test feature files written to {input_dir}")
    print(df.head())
