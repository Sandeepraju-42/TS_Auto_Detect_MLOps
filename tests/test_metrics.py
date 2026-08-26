"""
Unit tests for Metrics_Functions.py. Every expected value here is hand-
computed (see comments), not just "whatever the code currently returns" --
the point is catching a regression, which a test that just mirrors the
implementation can't do.
"""

import numpy as np
import pandas as pd
import pytest

import dependancies.Metrics_Functions


def test_wape_basic():
    # |10-12| + |20-18| + |30-33| = 2+2+3 = 7; sum(actual) = 60
    assert dependancies.Metrics_Functions.wape([10, 20, 30], [12, 18, 33]) == pytest.approx(7 / 60)


def test_wape_zero_denominator_is_nan():
    assert np.isnan(dependancies.Metrics_Functions.wape([0, 0], [1, 2]))


def test_bias_pct_sign_matches_under_over_forecasting():
    # under-forecast: actual > predicted -> positive
    assert dependancies.Metrics_Functions.bias_pct([10, 20], [8, 15]) == pytest.approx(7 / 30)
    # over-forecast: actual < predicted -> negative
    assert dependancies.Metrics_Functions.bias_pct([10, 20], [12, 25]) == pytest.approx(-7 / 30)


def test_tracking_signal_known_value():
    # errors = [0, 2, 1, 3]; mad = 1.5; sum = 6 -> ts = 6 / 1.5 = 4.0
    assert dependancies.Metrics_Functions.tracking_signal([10, 12, 11, 13], [10, 10, 10, 10]) == pytest.approx(4.0)


def test_tracking_signal_zero_mad_is_nan():
    assert np.isnan(dependancies.Metrics_Functions.tracking_signal([5, 5], [5, 5]))


def test_tracking_signal_by_sku_scoped_per_series():
    df = pd.DataFrame({
        "sku_id": ["A", "A", "B", "B"],
        "y_true": [10, 12, 5, 5],
        "y_pred": [10, 10, 5, 5],
    })
    result = dependancies.Metrics_Functions.tracking_signal_by_sku(df, "y_true", "y_pred")
    assert result["A"] == pytest.approx(2 / 1.0)  # errors [0,2], mad=1, sum=2
    assert np.isnan(result["B"])                   # zero mad


def test_tracking_signal_summary_threshold():
    # For a run of same-signed errors, tracking_signal always equals exactly
    # the row count (sum(errors) == sum(|errors|) when every error shares a
    # sign, so ts = n regardless of magnitude) -- so A needs n=5 (ts=5.0) to
    # clear a threshold of 4.0 strictly. B and C mix signs so their errors
    # largely cancel, keeping them well inside the control limit.
    df = pd.DataFrame({
        "sku_id": ["A"] * 5 + ["B"] * 4 + ["C"] * 4,
        "y_true": [10, 20, 30, 40, 15] + [10, 11, 9, 10] + [5, 6, 4, 5],
        "y_pred": [0, 0, 0, 0, 0] + [10, 10, 10, 10] + [6, 5, 5, 4],
    })
    summary = dependancies.Metrics_Functions.tracking_signal_summary(df, "y_true", "y_pred", threshold=4.0)
    assert summary["pct_skus_out_of_control"] == pytest.approx(1 / 3)


def _mase_train_df():
    # SKU A: 60 weeks, sales_qty = 1..60 (a clean ramp). For season_length=52,
    # y[52:] - y[:-52] = (i+53) - (i+1) = 52 for every one of the 8 overlap
    # points -> in-sample scale is exactly 52, not an approximation.
    dates_a = pd.date_range("2020-01-06", periods=60, freq="W-MON")
    df_a = pd.DataFrame({"sku_id": "A", "date": dates_a, "sales_qty": range(1, 61)})

    # SKU B: only 10 weeks -- too little history for its own scale (needs
    # > season_length rows), so mase() must fall back to the global median
    # scale rather than dropping B or dividing by NaN.
    dates_b = pd.date_range("2020-01-06", periods=10, freq="W-MON")
    df_b = pd.DataFrame({"sku_id": "B", "date": dates_b, "sales_qty": range(1, 11)})

    return pd.concat([df_a, df_b], ignore_index=True)


def test_mase_known_scale():
    train_df = _mase_train_df()
    eval_df = pd.DataFrame({"sku_id": ["A"], "y_true": [100.0], "y_pred": [48.0]})
    # |100 - 48| = 52; scale = 52 -> scaled error = 1.0 exactly
    assert dependancies.Metrics_Functions.mase(eval_df, "y_true", "y_pred", train_df, season_length=52) == pytest.approx(1.0)


def test_mase_falls_back_to_global_scale_for_short_history_sku():
    train_df = _mase_train_df()
    # SKU B has no own scale (only 10 weeks of history); global median scale
    # across SKUs with a computable scale is A's 52, so B's rows should be
    # scaled by 52 too, not silently dropped or NaN.
    eval_df = pd.DataFrame({"sku_id": ["B"], "y_true": [100.0], "y_pred": [48.0]})
    assert dependancies.Metrics_Functions.mase(eval_df, "y_true", "y_pred", train_df, season_length=52) == pytest.approx(1.0)


def test_evaluate_overall_and_group_rows():
    train_df = _mase_train_df()
    df = pd.DataFrame({
        "sku_id": ["A", "A", "B", "B"],
        "category": ["Tops", "Tops", "Bottoms", "Bottoms"],
        "y_true": [100.0, 200.0, 100.0, 200.0],
        "y_pred": [48.0, 148.0, 48.0, 148.0],
    })
    result = dependancies.Metrics_Functions.evaluate(df, "y_true", "y_pred", train_df, group_col="category")
    assert "overall" in result.index
    assert set(result.index) == {"overall", "Tops", "Bottoms"}
    assert result.loc["overall", "n_rows"] == 4
    # Every row here under-forecasts by exactly 52 -- bias_pct should be
    # strictly positive (under-forecasting), and identical in both segments
    # since the errors are symmetric across them by construction.
    assert result.loc["Tops", "bias_pct"] > 0
    assert result.loc["Tops", "bias_pct"] == pytest.approx(result.loc["Bottoms", "bias_pct"])
