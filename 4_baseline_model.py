"""
baseline model.

Deliberately simple and un-tuned. The point of a baseline is to get the whole pipeline 
*(features -> train -> predict -> evaluate)*
  1. Seasonal naive:
  2. LightGBM regressor
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb

from dependancies.Features_Functions import assert_no_leakage
from dependancies.Metrics_Functions import evaluate, wape, mase, bias_pct
from dependancies.params_utils import load_params

_params = load_params()
FORECAST_HORIZON = _params["features"]["forecast_horizon"]
VALIDATION_HORIZON = _params["features"]["validation_horizon"]
SEASON_LENGTH = 52
_baseline_p = _params["baseline_model"]

# Columns that must never reach the model
NON_FEATURE_COLS = ["sku_id", "date", "sales_qty", "true_demand", "pattern_type"]

CATEGORICAL_COLS = [
    "category", "sub_category", "color", "material", "brand", "gender",
    "size_range", "channel", "markdown_stage", "lifecycle_stage",
    "discount_bucket", "abc_class", "xyz_class",
]

input_dir = Path(__file__).resolve().parent / "input"
print(input_dir)

train = pd.read_csv(input_dir / "train_features.csv", parse_dates=["date"])
val = pd.read_csv(input_dir / "val_features.csv", parse_dates=["date"])
test = pd.read_csv(input_dir / "test_features.csv", parse_dates=["date"])


for col in CATEGORICAL_COLS:
    if col not in train.columns:
        continue
    all_categories = pd.concat([train[col], val[col], test[col]]).astype(str).unique()
    cat_dtype = pd.CategoricalDtype(categories=sorted(all_categories))
    train[col] = train[col].astype(str).astype(cat_dtype)
    val[col] = val[col].astype(str).astype(cat_dtype)
    test[col] = test[col].astype(str).astype(cat_dtype)


def seasonal_naive_predict(df: pd.DataFrame) -> pd.Series:
    pred = df["sales_qty_lag_52"].copy()
    sku_mean = df.groupby("sku_id")["sales_qty"].transform("mean")
    pred = pred.fillna(sku_mean)
    return pred


def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list,
                    categorical_cols: list) -> lgb.Booster:

    train_set = lgb.Dataset(
        train[feature_cols], label=np.log1p(train["sales_qty"]),
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        val[feature_cols], label=np.log1p(val["sales_qty"]),
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
        reference=train_set, free_raw_data=False,
    )

    params = dict(
        objective="regression",
        metric="mae",
        learning_rate=_baseline_p["learning_rate"],

        # LightGBM default and untouched deliberately
        num_leaves=_baseline_p["num_leaves"],      

        # a bit above default given SKU-week grain is noisy    
        min_data_in_leaf=_baseline_p["min_data_in_leaf"],  
        verbose=-1,
    )
    model = lgb.train(
        params, train_set, num_boost_round=_baseline_p["num_boost_round"], valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=_baseline_p["early_stopping_rounds"], verbose=False)],
    )
    return model


def main():
    # --- IGNORE ---
    if not (set(train.columns) == set(val.columns) == set(test.columns)):
        raise ValueError(
            "train/val/test feature files have different columns -- "
            f"train-only: {set(train.columns) - set(val.columns) - set(test.columns)}, "
            f"val/test-only: {(set(val.columns) | set(test.columns)) - set(train.columns)}"
        )
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]

    assert_no_leakage(train, [c for c in feature_cols if train[c].dtype.kind in "if"])
    print(f"train: {train.shape}, val: {val.shape}, test: {test.shape}")

    # -- Baseline 1: seasonal naive
    for split_name, split in [("val", val), ("test", test)]:
        split = split.copy()
        split["pred_naive"] = seasonal_naive_predict(split)
        res = evaluate(split, "sales_qty", "pred_naive", train, group_col="category")
        print(f"\n=== Seasonal naive -- {split_name} ===")
        print(res)

    # -- Baseline 2: LightGBM
    model = train_lightgbm(train, val, feature_cols, CATEGORICAL_COLS)
    split_results = {}
    for split_name, split in [("val", val), ("test", test)]:
        split = split.copy()
        split["pred_lgb"] = np.expm1(model.predict(split[feature_cols]))
        res = evaluate(split, "sales_qty", "pred_lgb", train, group_col="category")
        split_results[split_name] = res
        print(f"\n=== LightGBM baseline -- {split_name} ===")
        print(res)

    # promo vs non-promo AND ABC segmentation breakdown, test set only
    test_scored = test.copy()
    test_scored["pred_lgb"] = np.expm1(model.predict(test_scored[feature_cols]))

    # promo vs non-promo
    for promo_val, label in [(True, "with promo"), (False, "without promo")]:
        seg = test_scored[test_scored["promo_flag"] == promo_val]
        print(f"\n{label}: n={len(seg)}  wape={wape(seg['sales_qty'], seg['pred_lgb']):.3f}  "
              f"bias%={bias_pct(seg['sales_qty'], seg['pred_lgb']):.3f}")
        
    # ABC segmentation breakdown
    abc_values = pd.to_numeric(test_scored["abc_class"], errors="coerce")
    for abc_val, label in [(1, "A"), (2, "B"), (3, "C")]:
        seg = test_scored[abc_values == abc_val]
        print(f"\n{label}: n={len(seg)}  wape={wape(seg['sales_qty'], seg['pred_lgb']):.3f}  "
              f"bias%={bias_pct(seg['sales_qty'], seg['pred_lgb']):.3f}")   


    # feature importance 
    importance = pd.Series(
        model.feature_importance(importance_type="gain"), index=feature_cols
    ).sort_values(ascending=False)
    print("\ntop 15 features by gain:")
    print(importance.head(15))

    # save model and metrics
    model.save_model(Path(__file__).resolve().parent / "Output" / "Models" / "baseline_lightgbm.txt")
    print("\nsaved model to baseline_lightgbm.txt")

    # Flat metrics.json
    metrics_out = {
        "val_wape": float(split_results["val"].loc["overall", "wape"]),
        "val_mase": float(split_results["val"].loc["overall", "mase"]),
        "test_wape": float(split_results["test"].loc["overall", "wape"]),
        "test_mase": float(split_results["test"].loc["overall", "mase"]),
        "test_bias_pct": float(split_results["test"].loc["overall", "bias_pct"]),
    }
    with open(Path(__file__).resolve().parent / "Output" / "Metrics" / "metrics_baseline.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("wrote metrics_baseline.json")


if __name__ == "__main__":
    main()
