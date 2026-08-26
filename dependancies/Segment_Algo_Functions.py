"""
Shared candidate-algorithm definitions for the per-segment algorithm
selection step (6_optuna_tuning.py / 7_tuned_vs_baseline_segments.py).

WHY PER SEGMENT, NOT ONE GLOBAL MODEL
--------------------------------------
A single global LightGBM model (the original step 4/5 approach) pools loss
across every SKU-week row at once. High-volume A-class rows dominate that
pooled loss just by weight of numbers, so the model quietly optimizes for
them at the expense of low-volume, intermittent C-class series -- exactly
the segment where a single global tree model is structurally weakest. This
module lets each of the 6 segments (ABC class x promo/no-promo) pick
whichever candidate actually wins ON THAT SEGMENT, instead of assuming one
algorithm and one hyperparameter set is right everywhere.

TWO FAMILIES OF CANDIDATES, EVALUATED DIFFERENTLY
--------------------------------------------------
1. Tabular ML models (LightGBM, XGBoost, CatBoost) -- trained AND evaluated
   on the segment's own rows (abc_class == X AND promo_flag == Y), tuned
   per segment via Optuna. These can exploit the full feature set, but need
   enough rows in the segment to fit meaningfully -- see MIN_SEGMENT_ROWS.

2. Per-SKU demand-history methods (Croston, TSB, seasonal-naive, moving
   average) -- these have no concept of "rows" or exogenous features; they
   need one SKU's own continuous, evenly-spaced demand series. They're fit
   on each SKU's FULL train history (NOT promo-filtered -- Croston/TSB
   would break on a series with artificial gaps punched in it), producing
   one constant forecast per SKU, then EVALUATED only on the segment's own
   rows. That's a deliberate asymmetry, not an oversight: these are cheap,
   promo-blind per-SKU baselines answering "does even a dumb method that
   ignores promotions entirely already beat the tuned tree model on this
   segment" -- not full competitors tuned the same way as the ML models.

   Their tiny parameter spaces (one or two smoothing constants) are grid-
   searched directly, not run through Optuna/MLflow-per-trial -- that
   machinery is overkill for a 4-5 point grid and would multiply the
   already-large number of MLflow runs for no benefit.
"""

import numpy as np
import pandas as pd

from dependancies.Metrics_Functions import wape


# abc_class is 1 (A) / 2 (B) / 3 (C) -- see Features_Functions.py's
# add_abc_classification. promo_flag is boolean.
SEGMENTS = [
    (1, True, "A_promo"),
    (1, False, "A_no_promo"),
    (2, True, "B_promo"),
    (2, False, "B_no_promo"),
    (3, True, "C_promo"),
    (3, False, "C_no_promo"),
]

# Below this many training rows, an ML tree model is more likely to
# memorize noise than learn anything -- skip tuning it for that segment
# and rely on the classical per-SKU candidates instead (they don't need
# many rows since they operate per-SKU on full history, not on the
# segment's row count).
MIN_SEGMENT_ROWS = 200

# Grids for the classical candidates' smoothing constants / window sizes.
# Small on purpose -- these are meant to be cheap sanity-check baselines,
# not a tuning target in their own right.
GRID_ALPHA = [0.05, 0.1, 0.15, 0.2, 0.3]
GRID_WINDOW = [4, 8, 12, 16]
SEASONAL_NAIVE_LENGTH = 52  # weekly data -- one year back; not grid-searched,
                             # there's no other seasonally-meaningful choice
                             # given ~5 years of weekly history


def segment_mask(df: pd.DataFrame, abc_val: int, promo_val: bool) -> pd.Series:
    """
    Boolean mask for one (abc_class, promo_flag) segment.

    Uses pd.to_numeric on abc_class rather than a direct == comparison --
    after a CSV round-trip, abc_class becomes a category of STRINGS
    ("1"/"2"/"3"), and comparing that directly against a Python int
    silently matches nothing (the same bug that was making
    test_bias_pct__A/B/C come back NaN in 5_MLFlow_Dagshub.py).
    """
    abc_numeric = pd.to_numeric(df["abc_class"], errors="coerce")
    return (abc_numeric == abc_val) & (df["promo_flag"] == promo_val)


# ---------------------------------------------------------------------------
# Classical per-SKU demand-history candidates
# ---------------------------------------------------------------------------

def croston_forecast(g: pd.DataFrame, alpha: float = 0.1) -> float:
    """
    Classic Croston's method for ONE SKU's train history (g: that SKU's
    rows, must contain 'date'/'sales_qty'). Splits history into nonzero
    demand SIZES and the INTERVALS between them, single-exponential-smooths
    each separately, forecast = smoothed size / smoothed interval -- held
    constant across the whole eval horizon, which is the standard way
    Croston/TSB output is used (they estimate a current LEVEL, not a
    multi-step trajectory).

    Returns 0.0 for a SKU with no nonzero demand in its train history --
    nothing to extrapolate a level from.
    """
    history = g.sort_values("date")["sales_qty"].to_numpy(dtype=float)
    nonzero_idx = np.nonzero(history > 0)[0]
    if len(nonzero_idx) == 0:
        return 0.0

    sizes = history[nonzero_idx]
    # periods since the previous nonzero demand; the very first nonzero
    # observation's "interval" is just its own position + 1 periods since
    # the start of history.
    intervals = np.diff(nonzero_idx, prepend=-1).astype(float)

    z = sizes[0]
    p = intervals[0] if intervals[0] > 0 else 1.0
    for i in range(1, len(sizes)):
        z = alpha * sizes[i] + (1 - alpha) * z
        p = alpha * intervals[i] + (1 - alpha) * p

    return float(z / p) if p > 0 else float(z)


def tsb_forecast(g: pd.DataFrame, alpha_d: float = 0.1, alpha_p: float = 0.1) -> float:
    """
    Teunter-Syntetos-Babai variant: smooths demand SIZE and the
    PROBABILITY of a nonzero period each period (instead of the interval
    between nonzero periods), so a long run of zero demand correctly drags
    the forecast toward zero. Croston's interval estimate never updates
    during a zero-run, so it keeps forecasting the old level even after a
    SKU has effectively gone dead ("obsolescence" -- a known Croston
    weakness TSB was specifically designed to fix).
    """
    history = g.sort_values("date")["sales_qty"].to_numpy(dtype=float)
    if len(history) == 0 or not np.any(history > 0):
        return 0.0

    z = float(history[history > 0][0])
    p = float(np.mean(history > 0))
    for h in history:
        occurred = h > 0
        if occurred:
            z = alpha_d * float(h) + (1 - alpha_d) * z
        p = alpha_p * float(occurred) + (1 - alpha_p) * p

    return float(z * p)


def seasonal_naive_forecast(g: pd.DataFrame, season_length: int = SEASONAL_NAIVE_LENGTH) -> float:
    """This SKU's sales_qty value from `season_length` periods before the
    end of its train history. Falls back to the plain mean if the SKU
    doesn't have that much history yet (e.g. a newer SKU)."""
    y = g.sort_values("date")["sales_qty"].to_numpy(dtype=float)
    if len(y) == 0:
        return 0.0
    if len(y) > season_length:
        return float(y[-season_length])
    return float(np.mean(y))


def moving_average_forecast(g: pd.DataFrame, window: int = 8) -> float:
    """Mean of this SKU's last `window` periods of train history."""
    y = g.sort_values("date")["sales_qty"].to_numpy(dtype=float)
    if len(y) == 0:
        return 0.0
    return float(np.mean(y[-window:]))


def per_sku_constant_forecast(train_full: pd.DataFrame, eval_df: pd.DataFrame,
                               forecast_fn, **kwargs) -> np.ndarray:
    """
    Applies a per-SKU constant-forecast function (one of the four above)
    to each SKU's FULL train history, then broadcasts that one number
    across every row for that SKU in eval_df. A SKU present in eval_df but
    absent from train_full (shouldn't happen given the walk-forward split,
    but guarded anyway) gets 0.0.
    """
    preds_by_sku = train_full.groupby("sku_id").apply(lambda g: forecast_fn(g, **kwargs), include_groups=False)
    return eval_df["sku_id"].map(preds_by_sku).fillna(0.0).to_numpy()


def evaluate_stat_candidates(train_full: pd.DataFrame, val_seg: pd.DataFrame) -> dict:
    """
    Grid-searches each classical candidate's tiny parameter space and
    returns, for each of croston/tsb/seasonal_naive/moving_average:
    {"params": {...}, "val_wape": float}.

    Fit on train_full (every SKU's full train history -- see module
    docstring for why), scored against val_seg (this segment's own val
    rows only).
    """
    results = {}

    best = None
    for alpha in GRID_ALPHA:
        pred = per_sku_constant_forecast(train_full, val_seg, croston_forecast, alpha=alpha)
        w = wape(val_seg["sales_qty"], pred)
        if best is None or (not np.isnan(w) and w < best["val_wape"]):
            best = {"params": {"alpha": alpha}, "val_wape": w}
    results["croston"] = best

    best = None
    for alpha in GRID_ALPHA:
        pred = per_sku_constant_forecast(train_full, val_seg, tsb_forecast, alpha_d=alpha, alpha_p=alpha)
        w = wape(val_seg["sales_qty"], pred)
        if best is None or (not np.isnan(w) and w < best["val_wape"]):
            best = {"params": {"alpha_d": alpha, "alpha_p": alpha}, "val_wape": w}
    results["tsb"] = best

    pred = per_sku_constant_forecast(train_full, val_seg, seasonal_naive_forecast,
                                      season_length=SEASONAL_NAIVE_LENGTH)
    results["seasonal_naive"] = {
        "params": {"season_length": SEASONAL_NAIVE_LENGTH},
        "val_wape": wape(val_seg["sales_qty"], pred),
    }

    best = None
    for window in GRID_WINDOW:
        pred = per_sku_constant_forecast(train_full, val_seg, moving_average_forecast, window=window)
        w = wape(val_seg["sales_qty"], pred)
        if best is None or (not np.isnan(w) and w < best["val_wape"]):
            best = {"params": {"window": window}, "val_wape": w}
    results["moving_average"] = best

    return results


def predict_stat_candidate(algo: str, params: dict, train_full: pd.DataFrame,
                            eval_df: pd.DataFrame) -> np.ndarray:
    """Re-runs one classical candidate's per-SKU forecast with a specific
    (already-chosen) params dict, for scoring on val OR test."""
    fn = {
        "croston": croston_forecast,
        "tsb": tsb_forecast,
        "seasonal_naive": seasonal_naive_forecast,
        "moving_average": moving_average_forecast,
    }[algo]
    return per_sku_constant_forecast(train_full, eval_df, fn, **params)


# ---------------------------------------------------------------------------
# Tabular ML candidates -- Optuna search spaces + fit/predict wrappers
# ---------------------------------------------------------------------------

def lgb_search_space(trial, seed: int) -> dict:
    import lightgbm  # noqa: F401 -- import only to fail fast if missing
    return dict(
        objective="regression", metric="mae", verbose=-1, seed=seed,
        # see 6_optuna_tuning.py's original comment: reused Datasets across
        # trials with a varying min_data_in_leaf need this disabled.
        feature_pre_filter=False,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 100),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    )


def fit_predict_lightgbm(params: dict, train: pd.DataFrame, val: pd.DataFrame,
                          feature_cols: list, cat_cols: list):
    import lightgbm as lgb
    y_train = np.log1p(train["sales_qty"])
    y_val = np.log1p(val["sales_qty"])
    train_set = lgb.Dataset(train[feature_cols], label=y_train,
                             categorical_feature=cat_cols, free_raw_data=False)
    val_set = lgb.Dataset(val[feature_cols], label=y_val, categorical_feature=cat_cols,
                           reference=train_set, free_raw_data=False)
    model = lgb.train(
        params, train_set, num_boost_round=1000, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred = np.expm1(model.predict(val[feature_cols]))
    return model, pred, model.best_iteration


def xgb_search_space(trial, seed: int) -> dict:
    return dict(
        objective="reg:squarederror", eval_metric="mae", seed=seed,
        tree_method="hist",
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def fit_predict_xgboost(params: dict, train: pd.DataFrame, val: pd.DataFrame,
                         feature_cols: list, cat_cols: list):
    import xgboost as xgb
    y_train = np.log1p(train["sales_qty"])
    y_val = np.log1p(val["sales_qty"])
    # enable_categorical=True + DataFrame columns already pandas 'category'
    # dtype (see load_data() in the calling script) -- XGBoost's native
    # categorical split support, same idea as LightGBM's categorical_feature.
    dtrain = xgb.DMatrix(train[feature_cols], label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(val[feature_cols], label=y_val, enable_categorical=True)
    booster = xgb.train(
        params, dtrain, num_boost_round=1000, evals=[(dval, "val")],
        early_stopping_rounds=50, verbose_eval=False,
    )
    pred = np.expm1(booster.predict(dval, iteration_range=(0, booster.best_iteration + 1)))
    return booster, pred, booster.best_iteration


def catboost_search_space(trial, seed: int) -> dict:
    return dict(
        loss_function="MAE", random_seed=seed, verbose=False,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        depth=trial.suggest_int("depth", 3, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        # bagging_temperature (Bayesian bootstrap intensity), not
        # subsample -- CatBoost's default bootstrap_type is Bayesian, which
        # doesn't accept a subsample param; switching bootstrap_type just
        # to add one more knob isn't worth the extra surface here.
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
    )


def fit_predict_catboost(params: dict, train: pd.DataFrame, val: pd.DataFrame,
                          feature_cols: list, cat_cols: list):
    from catboost import CatBoostRegressor, Pool
    y_train = np.log1p(train["sales_qty"])
    y_val = np.log1p(val["sales_qty"])

    # CatBoost's cat_features wants string/int categories, not a pandas
    # Categorical dtype directly -- cast just the categorical columns.
    train_x = train[feature_cols].copy()
    val_x = val[feature_cols].copy()
    for c in cat_cols:
        train_x[c] = train_x[c].astype(str)
        val_x[c] = val_x[c].astype(str)
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    train_pool = Pool(train_x, label=y_train, cat_features=cat_idx)
    val_pool = Pool(val_x, label=y_val, cat_features=cat_idx)

    model = CatBoostRegressor(iterations=1000, early_stopping_rounds=50, **params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    pred = np.expm1(model.predict(val_x))
    return model, pred, model.get_best_iteration()


ML_CANDIDATES = {
    "lightgbm": (lgb_search_space, fit_predict_lightgbm),
    "xgboost": (xgb_search_space, fit_predict_xgboost),
    "catboost": (catboost_search_space, fit_predict_catboost),
}


def build_full_ml_params(algo: str, tuned_params: dict, seed: int) -> dict:
    """
    Recombines an Optuna study's best_params (search-space keys ONLY --
    trial.suggest_* values) with that algorithm's fixed/non-tuned keys, so
    a segment's winning params can be retrained outside of an Optuna trial
    (e.g. in 7_tuned_vs_baseline_segments.py). study.best_params never
    includes objective/metric/seed/etc. because those are plain dict
    literals in *_search_space(), never reached via trial.suggest_*.
    """
    if algo == "lightgbm":
        base = dict(objective="regression", metric="mae", verbose=-1, seed=seed, feature_pre_filter=False)
    elif algo == "xgboost":
        base = dict(objective="reg:squarederror", eval_metric="mae", seed=seed, tree_method="hist")
    elif algo == "catboost":
        base = dict(loss_function="MAE", random_seed=seed, verbose=False)
    else:
        raise ValueError(f"unknown ML algo: {algo}")
    base.update(tuned_params)
    return base


def predict_ml(algo: str, model, df: pd.DataFrame, feature_cols: list, cat_cols: list) -> np.ndarray:
    """Score a fitted ML candidate on new rows (log1p target, back-transformed here)."""
    if algo == "lightgbm":
        return np.expm1(model.predict(df[feature_cols]))
    if algo == "xgboost":
        import xgboost as xgb
        dmat = xgb.DMatrix(df[feature_cols], enable_categorical=True)
        return np.expm1(model.predict(dmat))
    if algo == "catboost":
        x = df[feature_cols].copy()
        for c in cat_cols:
            x[c] = x[c].astype(str)
        return np.expm1(model.predict(x))
    raise ValueError(f"unknown ML algo: {algo}")
