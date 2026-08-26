"""
Unit tests for Features.py -- focused on the leakage-sensitive pieces
(assert_no_leakage, the train/val/test split boundaries, and the ABC/XYZ
as_of cutoff) rather than every feature-engineering function, since those
are the parts a silent regression here would be most costly.
"""

import numpy as np
import pandas as pd
import pytest

from Features import (
    assert_no_leakage, time_based_split, get_train_cutoff,
    add_abc_classification, add_xyz_classification,
    FORECAST_HORIZON, VALIDATION_HORIZON,
)


def _toy_panel(n_weeks=120):
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
    return pd.DataFrame({
        "sku_id": "SKU1",
        "date": dates,
        "sales_qty": np.arange(n_weeks, dtype=float),
    })


def test_assert_no_leakage_flags_a_future_copy(capsys):
    # sales_qty has to be genuinely noisy here, not a smooth trend/ramp -- a
    # perfectly deterministic ramp makes EVERY shifted copy of it near-
    # perfectly correlated with both the current and future target alike
    # (correlation is ~invariant to shifting a linear sequence), which masks
    # exactly the gap assert_no_leakage is trying to detect.
    rng = np.random.default_rng(0)
    df = _toy_panel()
    df["sales_qty"] = rng.normal(size=len(df))
    # a feature that IS next period's target -- the exact pattern
    # assert_no_leakage exists to catch.
    df["suspicious_feature"] = df.groupby("sku_id")["sales_qty"].shift(-1)
    df["suspicious_feature"] = df["suspicious_feature"].fillna(df["sales_qty"])
    assert_no_leakage(df, ["suspicious_feature"])
    out = capsys.readouterr().out
    assert "suspicious_feature" in out
    assert "investigate for leakage" in out


def test_assert_no_leakage_silent_on_clean_feature(capsys):
    df = _toy_panel()
    rng = np.random.default_rng(0)
    # pure noise, uncorrelated with past or future target -- should not trip
    df["clean_feature"] = rng.normal(size=len(df))
    assert_no_leakage(df, ["clean_feature"])
    out = capsys.readouterr().out
    assert "clean_feature" not in out


def test_assert_no_leakage_skips_constant_columns():
    df = _toy_panel()
    df["constant_col"] = 1  # zero variance -- must not raise/warn on .corr()
    assert_no_leakage(df, ["constant_col"])  # should simply not raise


def test_time_based_split_boundaries_match_horizon_config():
    # enough weeks that all three splits are non-empty under the real
    # FORECAST_HORIZON/VALIDATION_HORIZON from params.yaml
    n_weeks = VALIDATION_HORIZON + FORECAST_HORIZON + 20
    df = _toy_panel(n_weeks=n_weeks)
    train, val, test = time_based_split(df)

    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(weeks=VALIDATION_HORIZON)
    test_start = max_date - pd.Timedelta(weeks=FORECAST_HORIZON)

    assert train["date"].max() < val_start
    assert val["date"].min() >= val_start
    assert val["date"].max() < test_start
    assert test["date"].min() >= test_start
    # every row accounted for exactly once
    assert len(train) + len(val) + len(test) == len(df)


def test_get_train_cutoff_is_one_day_before_val_start():
    n_weeks = VALIDATION_HORIZON + FORECAST_HORIZON + 20
    df = _toy_panel(n_weeks=n_weeks)
    cutoff = get_train_cutoff(df)
    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(weeks=VALIDATION_HORIZON)
    assert cutoff == val_start - pd.Timedelta(days=1)
    assert cutoff < val_start


def test_abc_classification_uses_only_history_up_to_as_of():
    # A, C, D, E all have real pre-cutoff volume (so A's cumulative Pareto
    # share stays comfortably under the 80% class-A cutoff -- a single SKU
    # holding 100% of volume would itself land in class C under a strict
    # cumulative-share reading, which isn't what this test is checking). B
    # sells a lot only AFTER the cutoff: if as_of leaked future data, B
    # would show up alongside A/C/D/E instead of contributing ~0 share.
    dates = pd.date_range("2020-01-06", periods=20, freq="W-MON")
    cutoff = dates[9]  # 10 weeks of "history", 10 weeks "after"
    df = pd.concat([
        pd.DataFrame({"sku_id": "A", "date": dates, "sales_qty": 40.0}),   # 400 pre-cutoff
        pd.DataFrame({"sku_id": "C", "date": dates, "sales_qty": 20.0}),   # 200 pre-cutoff
        pd.DataFrame({"sku_id": "D", "date": dates, "sales_qty": 20.0}),   # 200 pre-cutoff
        pd.DataFrame({"sku_id": "E", "date": dates, "sales_qty": 20.0}),   # 200 pre-cutoff
        pd.DataFrame({
            "sku_id": "B", "date": dates,
            "sales_qty": [0.0] * 10 + [1000.0] * 10,  # all volume is post-cutoff
        }),
    ], ignore_index=True)

    result = add_abc_classification(df, as_of=cutoff)
    b_class = result.loc[result["sku_id"] == "B", "abc_class"].iloc[0]
    a_class = result.loc[result["sku_id"] == "A", "abc_class"].iloc[0]
    assert a_class == 1   # A's pre-cutoff share (0.4) is well within the class-A cutoff
    assert b_class == 3   # B's volume is invisible before the cutoff -> class C


def test_xyz_classification_separates_stable_from_erratic():
    dates = pd.date_range("2020-01-06", periods=10, freq="W-MON")
    df = pd.concat([
        # perfectly stable demand -> CV = 0 -> X
        pd.DataFrame({"sku_id": "STABLE", "date": dates, "sales_qty": 50.0}),
        # wildly variable demand -> high CV -> Z
        pd.DataFrame({"sku_id": "ERRATIC", "date": dates,
                       "sales_qty": [0, 200, 0, 300, 0, 250, 0, 400, 0, 100]}),
    ], ignore_index=True)

    result = add_xyz_classification(df, as_of=dates[-1])
    stable_class = result.loc[result["sku_id"] == "STABLE", "xyz_class"].iloc[0]
    erratic_class = result.loc[result["sku_id"] == "ERRATIC", "xyz_class"].iloc[0]
    assert stable_class == "X"
    assert erratic_class == "Z"
