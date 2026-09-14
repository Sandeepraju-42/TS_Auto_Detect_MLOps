"""
Shared evaluation metrics
"""

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error: sum(|error|) / sum(actual)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan


def bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Signed counterpart to WAPE: sum(actual - forecast) / sum(actual).
    Positive => under-forecasting (actual exceeded forecast, i.e. the
    "we under-produced" case); negative => over-forecasting. Deliberately
    NOT a per-row percentage error (which blows up near actual=0 on
    zero-inflated SKU demand) -- this aggregates first, divides once.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum(y_true)
    return float(np.sum(y_true - y_pred) / denom) if denom > 0 else np.nan


def tracking_signal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Classic demand-planning bias signal for ONE series: cumulative error /
    mean absolute error, over that series' own periods. Values outside
    roughly [-4, 4] are the usual rule-of-thumb trigger for "this forecast
    is biased, not just noisy."

    Deliberately NOT meant to be called on a pool of many SKUs x many
    periods at once -- cumulative error keeps growing with every row you
    throw in, so a pooled tracking signal across thousands of (sku, period)
    rows is dominated by n, not by how biased any individual series
    actually is. Call this per SKU (see tracking_signal_by_sku) and
    summarize the resulting distribution instead.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    errors = y_true - y_pred
    mad = np.mean(np.abs(errors))
    return float(np.sum(errors) / mad) if mad > 0 else np.nan


def tracking_signal_by_sku(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                             sku_col: str = "sku_id") -> pd.Series:
    """
    tracking_signal(), correctly scoped: one value per SKU, computed over
    that SKU's own rows in the given window (sorted by date isn't required
    since sum/mean are order-independent, but the semantics only make
    sense as "this SKU's cumulative bias", not pooled across SKUs).
    """
    # select only the two needed columns before grouping -- keeps the
    # applied function from touching the grouping column itself, which is
    # both slightly faster and avoids pandas' "operated on the grouping
    # columns" deprecation warning without depending on the
    # version-specific include_groups= kwarg.
    return df.groupby(sku_col)[[y_true_col, y_pred_col]].apply(
        lambda g: tracking_signal(g[y_true_col], g[y_pred_col])
    )


def tracking_signal_summary(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                              sku_col: str = "sku_id", threshold: float = 4.0) -> dict:
    """
    Turns the per-SKU tracking signal distribution into two numbers that
    are actually readable at a glance: the share of SKUs currently outside
    the classic +/-4 control limit (candidates for the model/regressor
    switch your plan describes), and the median so one or two extreme SKUs
    don't dominate the read the way a pooled sum does.
    """
    ts = tracking_signal_by_sku(df, y_true_col, y_pred_col, sku_col).dropna()
    if len(ts) == 0:
        return {"pct_skus_out_of_control": np.nan, "median_tracking_signal": np.nan}
    return {
        "pct_skus_out_of_control": float((ts.abs() > threshold).mean()),
        "median_tracking_signal": float(ts.median()),
    }


def mase(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
          train_df: pd.DataFrame, season_length: int = 52,
          sku_col: str = "sku_id") -> float:
    """
    Mean Absolute Scaled Error, scaled per SKU using the seasonal-naive
    in-sample error from TRAIN only (never from the eval window itself --
    that would leak eval-period information into the scale). A SKU with
    too little training history to compute a seasonal scale falls back to
    the global median scale across all other SKUs, rather than being
    silently dropped from the metric.
    """
    def sku_scale(g: pd.DataFrame) -> float:
        y = g.sort_values("date")["sales_qty"]
        if len(y) <= season_length:
            return np.nan
        return float(np.mean(np.abs(y.values[season_length:] - y.values[:-season_length])))

    scales = train_df.groupby(sku_col)[["date", "sales_qty"]].apply(sku_scale)
    global_fallback = scales.median()
    scales = scales.fillna(global_fallback)

    merged = df.merge(scales.rename("scale"), left_on=sku_col, right_index=True, how="left")
    merged["scale"] = merged["scale"].fillna(global_fallback).replace(0, global_fallback)

    scaled_errors = np.abs(merged[y_true_col] - merged[y_pred_col]) / merged["scale"]
    return float(scaled_errors.mean())


def evaluate(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
             train_df: pd.DataFrame, group_col: str = None,
             sku_col: str = "sku_id") -> pd.DataFrame:
    """
    One-shot evaluation: overall row, plus one row per group (e.g. per
    category) if group_col is given. Returns a tidy dataframe so results
    from different models/segments are trivial to concatenate and compare.

    tracking_signal replaced with pct_skus_out_of_control / median_tracking_signal
    -- the old pooled tracking_signal scaled with row count (a group with
    28k rows and a group with 300 rows aren't comparable on that number),
    which is exactly the "huge values" that shouldn't be interpreted as-is.
    """
    def _row(g):
        ts = tracking_signal_summary(g, y_true_col, y_pred_col, sku_col)
        return pd.Series({
            "n_rows": len(g),
            "wape": wape(g[y_true_col], g[y_pred_col]),
            "mase": mase(g, y_true_col, y_pred_col, train_df),
            "bias_pct": bias_pct(g[y_true_col], g[y_pred_col]),
            "pct_skus_out_of_control": ts["pct_skus_out_of_control"],
            "median_tracking_signal": ts["median_tracking_signal"],
        })

    rows = [pd.Series(_row(df), name="overall")]
    if group_col:
        # observed=True: group_col is often a pandas "category" dtype
        # column (see baseline_model.py's categorical cast) -- without this,
        # pandas warns it will change its default and, worse, would include
        # a phantom row for any category level present in the dtype but not
        # actually in this particular split.
        for key, g in df.groupby(group_col, observed=True):
            rows.append(pd.Series(_row(g), name=key))
    return pd.DataFrame(rows)