"""
MLflow-tracked training run, logged to DagsHub.

Using DagsHub's built-in MLflow integration, this script will log the following to the DagsHub experiment:
- model parameters (from params.yaml)
- model artifacts (LightGBM model file, feature importance CSV)
- evaluation metrics (overall and per-category WAPE, MASE, bias%, pct SKUs out of control, median tracking signal)
- evaluation artifacts (CSV of per-category metrics for val and test splits)
- run tags (model type, feature set version, stage)
The run can be viewed in the DagsHub UI at:
https://dagshub.com/Sandeepraju-42/Forecast_MLFlow_Fashion_Dataset/experiments

Deliberately not using MLflow's sklearn wrapper, because it doesn't support 
categorical features and doesn't allow logging per-category metrics. 
Instead, we log the model with mlflow.lightgbm.log_model() and log metrics manually.

Also not using AWS or GCP for artifact storage, because DagsHub's MLflow integration 
handles that automatically. 

Also considering cost and complexity of setting up S3 or GCS buckets, IAM roles, etc. 
for a small project.
Consider AWS or GCP if you want to scale up to a larger project with multiple team members,
or if you want to use MLflow's model registry, which requires a remote artifact store.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import dagshub

from dependancies.Features_Functions import assert_no_leakage
from dependancies.Metrics_Functions import evaluate, wape, mase, bias_pct
from dependancies.params_utils import load_params


_params = load_params()
FORECAST_HORIZON = _params["features"]["forecast_horizon"]
VALIDATION_HORIZON = _params["features"]["validation_horizon"]
EXPERIMENT_NAME = "fashion-demand-forecast-baseline-lgbm"

DAGSHUB_REPO_OWNER = "Sandeepraju-42"
DAGSHUB_REPO_NAME = "Forecast_MLFlow_Fashion_Dataset"

NON_FEATURE_COLS = ["sku_id", "date", "sales_qty", "true_demand", "pattern_type"]

CATEGORICAL_COLS = [
    "category", "sub_category", "color", "material", "brand", "gender",
    "size_range", "channel", "markdown_stage", "lifecycle_stage",
    "discount_bucket", "abc_class", "xyz_class",
]

input_dir = Path(__file__).resolve().parent / "input"
output_dir = Path(__file__).resolve().parent / "output"


_baseline_p = _params["baseline_model"]
LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=_baseline_p["learning_rate"],
    num_leaves=_baseline_p["num_leaves"],
    min_data_in_leaf=_baseline_p["min_data_in_leaf"],
    verbose=-1,
)


def load_data():
    train = pd.read_csv(input_dir / "train_features.csv", parse_dates=["date"])
    val = pd.read_csv(input_dir / "val_features.csv", parse_dates=["date"])
    test = pd.read_csv(input_dir / "test_features.csv", parse_dates=["date"])

    if not (set(train.columns) == set(val.columns) == set(test.columns)):
        raise ValueError("train/val/test feature files have different columns")

    for col in CATEGORICAL_COLS:
        if col not in train.columns:
            continue
        all_categories = pd.concat([train[col], val[col], test[col]]).astype(str).unique()
        cat_dtype = pd.CategoricalDtype(categories=sorted(all_categories))
        for split in (train, val, test):
            split[col] = split[col].astype(str).astype(cat_dtype)

    return train, val, test

# only sending wape, mase, bias_pct to mlflow, not pct_skus_out_of_control or median_tracking_signal
def log_split_metrics(prefix: str, results: pd.DataFrame) -> None:

    for metric in ["wape", "mase", "bias_pct"]:
        overall_val = results.loc["overall", metric]
        if pd.notna(overall_val):
            mlflow.log_metric(f"{prefix}_{metric}", float(overall_val))
        for idx in results.index:
            if idx == "overall":
                continue
            val = results.loc[idx, metric]
            if pd.notna(val):
                mlflow.log_metric(f"{prefix}_{metric}__{idx}", float(val))


def main():
    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking experiment '{EXPERIMENT_NAME}' to DagsHub")

    train, val, test = load_data()
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]
    assert_no_leakage(train, [c for c in feature_cols if train[c].dtype.kind in "if"])
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    with mlflow.start_run(run_name="lgbm-baseline"):
        mlflow.set_tags({
            "model_type": "lightgbm",
            "feature_set_version": "horizon_consistent_v2",
            "stage": "baseline",
        })

        mlflow.log_params(LGB_PARAMS)
        mlflow.log_params({
            "forecast_horizon": FORECAST_HORIZON,
            "validation_horizon": VALIDATION_HORIZON,
            "n_features": len(feature_cols),
            "n_train_rows": len(train),
            "target_transform": "log1p",
        })

        train_set = lgb.Dataset(train[feature_cols], label=np.log1p(train["sales_qty"]),
                                 categorical_feature=cat_cols, free_raw_data=False)
        val_set = lgb.Dataset(val[feature_cols], label=np.log1p(val["sales_qty"]),
                               categorical_feature=cat_cols, reference=train_set, free_raw_data=False)

        model = lgb.train(
            LGB_PARAMS, train_set, num_boost_round=_baseline_p["num_boost_round"], valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=_baseline_p["early_stopping_rounds"], verbose=False)],
        )
        mlflow.log_param("best_iteration", model.best_iteration)

        # sending predictions to evaluate() for val and test splits, 
        # logging overall and per-category, ABC, Promo metrics to mlflow
        for split_name, split in [("val", val), ("test", test)]:
            split = split.copy()
            split["pred"] = np.expm1(model.predict(split[feature_cols]))
            results = evaluate(split, "sales_qty", "pred", train, group_col="category")
            log_split_metrics(split_name, results)
            results.to_csv( output_dir / f"eval_results/{split_name}_eval_results.csv")
            mlflow.log_artifact(output_dir / f"eval_results/{split_name}_eval_results.csv")

        # sending promo vs non-promo metrics to mlflow, test split only
        test_scored = test.copy()
        test_scored["pred"] = np.expm1(model.predict(test_scored[feature_cols]))
        for promo_val, label in [(True, "promo"), (False, "no_promo")]:
            seg = test_scored[test_scored["promo_flag"] == promo_val]
            mlflow.log_metric(f"test_wape__{label}", wape(seg["sales_qty"], seg["pred"]))
            mlflow.log_metric(f"test_bias_pct__{label}", bias_pct(seg["sales_qty"], seg["pred"]))

        # sending ABC class metrics to mlflow, test split only
        test_scored = test.copy()
        test_scored["pred"] = np.expm1(model.predict(test_scored[feature_cols]))

        abc_values = pd.to_numeric(test_scored["abc_class"], errors="coerce")
        for abc_val, label in [(1, "A"), (2, "B"), (3, "C")]:
            seg = test_scored[abc_values == abc_val]
            mlflow.log_metric(f"test_wape__{label}", wape(seg["sales_qty"], seg["pred"]))
            mlflow.log_metric(f"test_bias_pct__{label}", bias_pct(seg["sales_qty"], seg["pred"]))

        # logging feature importance to mlflow
        importance = pd.Series(
            model.feature_importance(importance_type="gain"), index=feature_cols
        ).sort_values(ascending=False)
        importance.to_csv(output_dir / "Feature_Importance" / "feature_importance.csv", header=["gain"])
        mlflow.log_artifact(output_dir / "Feature_Importance" / "feature_importance.csv")

        # logging the model to mlflow
        mlflow.lightgbm.log_model(model, name="model")

        run = mlflow.active_run()
        print(f"\nlogged run: {run.info.run_id}")
        print(f"view at: https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/experiments")


if __name__ == "__main__":
    main()
