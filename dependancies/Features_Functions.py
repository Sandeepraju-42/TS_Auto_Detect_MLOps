import numpy as np
import pandas as pd

from .params_utils import load_params

_params = load_params()["features"]
FORECAST_HORIZON = _params["forecast_horizon"]
VALIDATION_HORIZON = _params["validation_horizon"]
GRANULARITY = _params["granularity"]

_LAG_OFFSETS = [0, 1, 2, 4, 44] if GRANULARITY == "W" else [0, 6, 13, 29, 364]
LAGS = [FORECAST_HORIZON + offset for offset in _LAG_OFFSETS]
ROLLING_WINDOWS = [4, 12] if GRANULARITY == "W" else [7, 30]

# Cyclical (sin/cos) encoding of week_of_year/month
# a raw integer week_of_year makes week 52 and week 1 look maximally 
# far apart to a model, when they're actually adjacent.
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

# Product lifecycle stage from days_since_launch
# bucketing it gives models an explicit signal instead of hoping 
# days_since_launch alone captures the nonlinearity.
def add_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [-1, 14, 90, 270, np.inf]
    labels = ["new", "growth", "mature", "decline"]
    df["lifecycle_stage"] = pd.cut(df["days_since_launch"], bins=bins, labels=labels)
    return df


# adding:
# price change percentage, 
# relative price vs category, 
# discount depth bucket: "none", "light", "moderate", "deep", "clearance"
# and weeks since last promo: if a SKU has never had a promo, this is NaN (not 0), 
#   else it's the number of weeks since the last promo. 
def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sku_id", "date"])

    df["price_change_pct"] = df.groupby("sku_id")["price"].pct_change()

    cat_avg_price = df.groupby(["category", "date"])["price"].transform("mean")
    df["relative_price_vs_category"] = df["price"] / cat_avg_price

    df["discount_bucket"] = pd.cut(
        df["discount_pct"], bins=[-0.01, 0, 0.1, 0.25, 0.5, 1.0],
        labels=["none", "light", "moderate", "deep", "clearance"],
    )

    df["had_promo"] = df["promo_flag"].astype(int)

    def _weeks_since_last(s: pd.Series) -> pd.Series:
        out = np.full(len(s), np.nan)
        last_idx = None
        for i, v in enumerate(s.values):
            if v:
                last_idx = i
                out[i] = 0
            elif last_idx is not None:
                out[i] = i - last_idx
        return pd.Series(out, index=s.index)

    df["weeks_since_last_promo"] = (
        df.groupby("sku_id")["had_promo"].apply(_weeks_since_last).reset_index(level=0, drop=True)
    )
    return df


# adding lag functions for the target (sales_qty) and 
# other endogenous features 
# (review_count, avg_rating, wishlist_adds, social_trend_score, inventory_on_hand, stockout_flag).
def add_target_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sku_id", "date"])
    g = df.groupby("sku_id")["sales_qty"]

    for lag in LAGS:
        df[f"sales_qty_lag_{lag}"] = g.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"sales_qty_rollmean_{window}"] = (
            g.shift(FORECAST_HORIZON).rolling(window, min_periods=max(2, window // 2)).mean()
        )
        df[f"sales_qty_rollstd_{window}"] = (
            g.shift(FORECAST_HORIZON).rolling(window, min_periods=max(2, window // 2)).std()
        )
    return df


def add_endogenous_lagged_features(df: pd.DataFrame, lag: int = FORECAST_HORIZON) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sku_id", "date"])
    endogenous_cols = [
        "review_count", "avg_rating", "wishlist_adds", "social_trend_score",
        "inventory_on_hand", "stockout_flag",
    ]

    for col in endogenous_cols:
        if col in df.columns:
            shifted = df.groupby("sku_id")[col].shift(lag)
            df[f"{col}_lag{lag}"] = shifted.astype(float)
            df = df.drop(columns=[col])
    return df


# cross-product features: e.g. substitute_cheaper_flag = (substitute_price_ratio < 1)
def add_cross_product_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "substitute_price_ratio" in df.columns:
        df["substitute_cheaper_flag"] = (df["substitute_price_ratio"] < 1).astype(int)
    return df

# ABC/XYZ classification: segment SKUs by sales contribution (ABC) and demand variability (XYZ)
def add_abc_classification(
    df: pd.DataFrame,
    sales_col: str = "sales_qty",
    group_col: str = "sku_id",
    a_cut: float = 0.8,
    b_cut: float = 0.95,
    result_col: str = "abc_class",
    as_of: str | None = None,
) -> pd.DataFrame:
    
    df = df.copy()
    if as_of is not None:
        cutoff = pd.to_datetime(as_of)
        sales_df = df[df["date"] <= cutoff]
    else:
        sales_df = df

    sku_sales = sales_df.groupby(group_col)[sales_col].sum()
    total = sku_sales.sum()

    if total == 0 or len(sku_sales) == 0:
        abc_map = pd.Series(3, index=sku_sales.index, name=result_col)
    else:
        contrib = (sku_sales / total).sort_values(ascending=False)
        cumsum = contrib.cumsum()
        abc_labels = pd.Series(index=contrib.index, dtype="object")
        abc_labels[cumsum <= a_cut] = 1
        abc_labels[(cumsum > a_cut) & (cumsum <= b_cut)] = 2
        abc_labels[cumsum > b_cut] = 3
        abc_map = abc_labels.rename(result_col)

    abc_map = abc_map.reset_index()
    df = df.merge(abc_map, on=group_col, how='left')
    if result_col in df.columns:
        df[result_col] = df[result_col].infer_objects(copy=False).fillna(3).astype('category')
    else:
        df[result_col] = pd.Categorical([3] * len(df))
    return df


def add_xyz_classification(
    df: pd.DataFrame,
    sales_col: str = "sales_qty",
    group_col: str = "sku_id",
    date_col: str = "date",
    x_cut: float = 0.5,
    y_cut: float = 1.0,
    result_col: str = "xyz_class",
    as_of: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    if as_of is not None:
        cutoff = pd.to_datetime(as_of)
        sales_df = df[df[date_col] <= cutoff]
    else:
        sales_df = df

    stats = sales_df.groupby(group_col)[sales_col].agg(["mean", "std"])
    cv = (stats["std"] / stats["mean"]).replace([np.inf, -np.inf], np.nan)

    xyz_labels = pd.Series(index=cv.index, dtype="object")
    xyz_labels[stats["mean"] <= 0] = "3"
    xyz_labels[cv.isna() & (stats["mean"] > 0)] = "1"  # zero variance, positive mean -> perfectly stable
    remaining = xyz_labels.isna()
    xyz_labels[remaining & (cv <= x_cut)] = "1"
    xyz_labels[remaining & (cv > x_cut) & (cv <= y_cut)] = "2"
    xyz_labels[remaining & (cv > y_cut)] = "3"

    xyz_map = xyz_labels.rename(result_col).reset_index()
    df = df.merge(xyz_map, on=group_col, how="left")
    df[result_col] = df[result_col].fillna("3").astype("category")
    return df

# One-hot for low-cardinality categoricals (category, gender, material)
# Keeps an unencoded `category_copy` of `category` alongside the one-hot dummies
def encode_categoricals(df: pd.DataFrame, target_encode_cols=None) -> pd.DataFrame:
    df = df.copy()
    df["category_copy"] = df["category"]
    df["lifecycle_stage_copy"] = df["lifecycle_stage"]
    low_card = ["category_copy", "sub_category", "gender", "lifecycle_stage_copy", "discount_bucket"]
    low_card = [c for c in low_card if c in df.columns]
    df = pd.get_dummies(df, columns=low_card, dummy_na=True)
    return df

# train, validation, test split based on time, not random
# demand forecasting always needs a walk-forward split
def time_based_split(df: pd.DataFrame, forecast_horizon=FORECAST_HORIZON,
                      validation_horizon=VALIDATION_HORIZON):
    """
    Matches the existing forecast-framework config (FORECAST_HORIZON=8,
    VALIDATION_HORIZON=104) rather than a random split -- demand forecasting
    always needs a walk-forward/time-based split, since a random split lets
    the model train on data from *after* the point it's meant to predict.
    """
    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(weeks=validation_horizon)
    test_start = max_date - pd.Timedelta(weeks=forecast_horizon)

    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, val, test

# The last date that belongs to TRAIN under time_based_split's boundaries
def get_train_cutoff(df: pd.DataFrame, forecast_horizon=FORECAST_HORIZON,
                      validation_horizon=VALIDATION_HORIZON):

    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(weeks=validation_horizon)
    return val_start - pd.Timedelta(days=1)


def assert_no_leakage(df: pd.DataFrame, feature_cols: list, target_col="sales_qty") -> None:
    """
    Cheap automated guard, not just a manual checklist item: for each
    feature, check its correlation with the *future* target (shifted -1)
    isn't suspiciously higher than its correlation with the current target.
    Doesn't catch everything, but catches the common accidental-leak
    pattern where a feature is basically a copy of next period's outcome.
    """
    future_target = df.groupby("sku_id")[target_col].shift(-1)
    for col in feature_cols:
        if df[col].dtype.kind not in "if":
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue  # constant column (e.g. is_weekend at weekly grain) -- .corr() on
            # zero variance is a harmless 0/0 -> NaN, just noisy RuntimeWarning spam
        curr_corr = df[col].corr(df[target_col])
        future_corr = df[col].corr(future_target)
        if pd.notna(future_corr) and abs(future_corr) > abs(curr_corr) + 0.15:
            print(f"[check] {col}: corr with FUTURE target ({future_corr:.2f}) notably "
                  f"higher than with current ({curr_corr:.2f}) -- investigate for leakage")