"""
Step 5b: baseline vs. single global tuned model vs. per-segment algorithm
ensemble.

Reads TWO upstream artifacts:
  - optuna_best_params.json -- the single-global-LightGBM tuned params (from
    the ORIGINAL 6_optuna_tuning.py run; the reworked 6_optuna_tuning.py no
    longer produces this file -- see its module docstring). Kept here
    unchanged so this script still has a "what we had before" arm to
    compare against.
  - segment_algo_selection.json -- the per-segment winning algorithm +
    params (from the REWORKED 6_optuna_tuning.py). This is the new part.

Three models get compared, overall AND broken down by the 6 (abc_class x
promo_flag) segments:
  1. baseline    -- fresh LightGBM, un-tuned params, retrained here.
  2. tuned       -- single global LightGBM, tuned params, retrained here.
  3. segment_ensemble -- for each segment, whichever algorithm won that
     segment in segment_algo_selection.json (could be LightGBM, XGBoost,
     CatBoost, or a classical per-SKU method), retrained on that segment's
     own rows and used ONLY for that segment's test rows. This is what
     actually answers "does letting each segment pick its own algorithm
     beat one global tuned model" -- not just "did tuning help overall."

Any test row that somehow falls outside all 6 segments (shouldn't happen --
abc_class in {1,2,3} and promo_flag is boolean, so the segments are
exhaustive and non-overlapping) falls back to the global tuned model's
prediction for that row, so the ensemble's coverage is always complete.

STILL NOT TOUCHING THE REGISTRY OR app.py -- see 6_optuna_tuning.py's
module docstring. This script's job is to produce the comparison numbers
(printed, segment_ensemble_report.csv, metrics_segment_ensemble.json, and
an MLflow run) so you can decide whether productionizing per-segment
routing is worth the added complexity -- not to make that call for you.

Both the baseline and tuned models are logged to DagsHub MLflow exactly as
before (log_model_run, unchanged). The segment ensemble is logged
separately: one summary run with the comparison table and, per segment, a
tag recording which algorithm won -- not one full nested run per segment
per algorithm, which would be 6x the run count for information that's
already in segment_algo_selection.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import dagshub


from dependancies.Metrics_Functions import evaluate, wape, bias_pct
from dependancies.params_utils import load_params
from dependancies.Features_Functions import assert_no_leakage
from dependancies.Segment_Algo_Functions import (
    SEGMENTS, ML_CANDIDATES, segment_mask,
    build_full_ml_params, predict_ml, predict_stat_candidate,
)

EXPERIMENT_NAME = "fashion-demand-forecast-Tuned-vs-baseline-comparison"
DAGSHUB_REPO_OWNER = "Sandeepraju-42"
DAGSHUB_REPO_NAME = "Forecast_MLFlow_Fashion_Dataset"

NON_FEATURE_COLS = ["sku_id", "date", "sales_qty", "true_demand", "pattern_type"]
CATEGORICAL_COLS = [
    "category", "sub_category", "color", "material", "brand", "gender",
    "size_range", "channel", "markdown_stage", "lifecycle_stage",
    "discount_bucket", "abc_class", "xyz_class",
]

_params = load_params()
_baseline_p = _params["baseline_model"]
_tuning_p = _params["tuning"]
SEED = _tuning_p["seed"]
BASELINE_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=_baseline_p["learning_rate"],
    num_leaves=_baseline_p["num_leaves"],
    min_data_in_leaf=_baseline_p["min_data_in_leaf"],
    verbose=-1,
)

input_dir = Path(__file__).resolve().parent / "input"
output_dir = Path(__file__).resolve().parent / "output"


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


def train_model(params, train, val, feature_cols, cat_cols):
    train_set = lgb.Dataset(
        train[feature_cols], label=np.log1p(train["sales_qty"]),
        categorical_feature=cat_cols, free_raw_data=False,
    )
    val_set = lgb.Dataset(
        val[feature_cols], label=np.log1p(val["sales_qty"]),
        categorical_feature=cat_cols, reference=train_set, free_raw_data=False,
    )
    model = lgb.train(
        params, train_set, num_boost_round=_baseline_p["num_boost_round"], valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=_baseline_p["early_stopping_rounds"], verbose=False)],
    )
    return model


def segment_table(test_scored, train, pred_col, group_cols):
    """Unchanged from the original script -- category/abc_class/xyz_class/
    promo_flag breakdown for the baseline/tuned comparison."""
    tables = {}
    for gc in group_cols:
        tables[gc] = evaluate(test_scored, "sales_qty", pred_col, train, group_col=gc)

    promo_rows = []
    for promo_val, label in [(True, "promo"), (False, "no_promo")]:
        seg = test_scored[test_scored["promo_flag"] == promo_val]
        promo_rows.append({
            "segment": label,
            "n_rows": len(seg),
            "wape": wape(seg["sales_qty"], seg[pred_col]),
            "bias_pct": bias_pct(seg["sales_qty"], seg[pred_col]),
        })
    tables["promo_flag"] = pd.DataFrame(promo_rows).set_index("segment")
    return tables


def log_model_run(label, params, model, test_scored, train, pred_col, group_cols,
                   overall_test_wape, model_out_path, feature_cols, cat_cols):
    """Unchanged from the original script."""
    with mlflow.start_run(run_name=label, nested=True):
        mlflow.set_tags({"stage": label, "feature_set_version": "horizon_consistent_v2"})
        mlflow.log_params(params)
        mlflow.log_param("best_iteration", model.best_iteration)
        mlflow.log_metric("test_wape", overall_test_wape)

        tables = segment_table(test_scored, train, pred_col, group_cols)
        for gc, tbl in tables.items():
            for idx in tbl.index:
                row = tbl.loc[idx]
                if pd.notna(row.get("wape")):
                    mlflow.log_metric(f"test_wape__{gc}__{idx}", float(row["wape"]))
                if pd.notna(row.get("bias_pct")):
                    mlflow.log_metric(f"test_bias_pct__{gc}__{idx}", float(row["bias_pct"]))

        importance = pd.Series(
            model.feature_importance(importance_type="gain"), index=feature_cols,
        )
        importance_path = Path(__file__).resolve().parent / f"feature_importance_{label}.csv"
        importance.sort_values(ascending=False).to_csv(importance_path, header=["gain"])
        mlflow.log_artifact(str(importance_path))

        schema = {
            "feature_cols": feature_cols,
            "categorical_categories": {
                col: train[col].cat.categories.tolist() for col in cat_cols
            },
        }
        schema_path = Path(__file__).resolve().parent / f"feature_schema_{label}.json"
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        mlflow.log_artifact(str(schema_path), artifact_path="schema")

        model.save_model(str(model_out_path))
        mlflow.lightgbm.log_model(model, name="model")
        print(f"logged '{label}' run to DagsHub (test WAPE {overall_test_wape:.4f})")


def train_segment_winners(seg_state, train, val, feature_cols, cat_cols):
    """
    Retrains each segment's winning candidate from segment_algo_selection.json.

    ML winners (lightgbm/xgboost/catboost) are retrained on that segment's
    OWN rows (train/val filtered to abc_class==X AND promo_flag==Y) using
    the exact best params the Optuna study found. Classical winners
    (croston/tsb/seasonal_naive/moving_average) have no "model" to fit here
    -- they're just replayed at predict time against each SKU's full train
    history, same as during selection.

    Returns {segment_label: {"algo": str, "kind": "ml"|"stat", "model": obj|None, "params": dict}}.
    A segment missing from segment_algo_selection.json (not yet run, or no
    valid candidate found) is simply absent from the returned dict --
    predict_segment_ensemble() below falls back to the global tuned model
    for that segment's rows.
    """
    winners = {}
    for abc_val, promo_val, seg_label in SEGMENTS:
        seg_entry = seg_state.get("segments", {}).get(seg_label)
        if not seg_entry or not seg_entry.get("winner"):
            print(f"  {seg_label}: no winner in segment_algo_selection.json -- "
                  f"will fall back to the global tuned model for this segment")
            continue

        algo = seg_entry["winner"]
        params = seg_entry["candidates"][algo]["params"]

        if algo in ML_CANDIDATES:
            train_seg = train[segment_mask(train, abc_val, promo_val)]
            val_seg = val[segment_mask(val, abc_val, promo_val)]
            full_params = build_full_ml_params(algo, params, SEED)
            _, fit_fn = ML_CANDIDATES[algo]
            model, _, best_iter = fit_fn(full_params, train_seg, val_seg, feature_cols, cat_cols)
            winners[seg_label] = {"algo": algo, "kind": "ml", "model": model, "params": full_params}
            print(f"  {seg_label}: retrained winner {algo} ({len(train_seg)} train rows)")
        else:
            winners[seg_label] = {"algo": algo, "kind": "stat", "model": None, "params": params}
            print(f"  {seg_label}: winner {algo} is a classical method, no retraining needed")

    return winners


def predict_segment_ensemble(winners, train_full, eval_df, feature_cols, cat_cols):
    """
    Builds the per-segment ensemble's prediction for every row in eval_df.
    Returns (pred: np.ndarray, assigned: np.ndarray[bool]) -- `assigned`
    marks which rows actually got a segment-specific prediction, so the
    caller can fall back the rest (should be none) to the global model.
    """
    pred = np.zeros(len(eval_df))
    assigned = np.zeros(len(eval_df), dtype=bool)

    for abc_val, promo_val, seg_label in SEGMENTS:
        mask = segment_mask(eval_df, abc_val, promo_val).to_numpy()
        if seg_label not in winners or mask.sum() == 0:
            continue
        w = winners[seg_label]
        sub = eval_df[mask]
        if w["kind"] == "ml":
            sub_pred = predict_ml(w["algo"], w["model"], sub, feature_cols, cat_cols)
        else:
            sub_pred = predict_stat_candidate(w["algo"], w["params"], train_full, sub)
        pred[mask] = sub_pred
        assigned[mask] = True

    return pred, assigned


def main():
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_params_path = Path(__file__).resolve().parent / "optuna_best_params.json"
    with open(best_params_path) as f:
        best = json.load(f)
    tuned_params = dict(BASELINE_PARAMS)
    tuned_params.update(best["best_params"])
    tuned_params["objective"] = "regression"
    tuned_params["metric"] = "mae"
    tuned_params["verbose"] = -1
    print(f"loaded global tuned params from {best_params_path.name} "
          f"(val WAPE at tuning time: {best['best_value_val_wape']:.4f})")

    seg_state_path = Path(__file__).resolve().parent / "segment_algo_selection.json"
    if not seg_state_path.exists():
        raise FileNotFoundError(
            f"{seg_state_path.name} not found -- run 6_optuna_tuning.py first "
            f"(it needs to have selected at least one segment's winner)."
        )
    with open(seg_state_path) as f:
        seg_state = json.load(f)

    train, val, test = load_data()
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]
    assert_no_leakage(train, [c for c in feature_cols if train[c].dtype.kind in "if"])
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    print("\ntraining baseline (default-ish params)...")
    baseline_model = train_model(BASELINE_PARAMS, train, val, feature_cols, cat_cols)
    print("training global tuned (optuna best params)...")
    tuned_model = train_model(tuned_params, train, val, feature_cols, cat_cols)

    test_scored = test.copy()
    test_scored["pred_baseline"] = np.expm1(baseline_model.predict(test_scored[feature_cols]))
    test_scored["pred_tuned"] = np.expm1(tuned_model.predict(test_scored[feature_cols]))

    print("\nretraining each segment's winning candidate...")
    winners = train_segment_winners(seg_state, train, val, feature_cols, cat_cols)
    seg_pred, seg_assigned = predict_segment_ensemble(winners, train, test_scored, feature_cols, cat_cols)
    if not seg_assigned.all():
        n_missing = int((~seg_assigned).sum())
        print(f"\nWARNING: {n_missing} test rows fell outside all 6 segments "
              f"-- filled with the global tuned model's prediction")
        seg_pred[~seg_assigned] = test_scored.loc[~seg_assigned, "pred_tuned"].to_numpy()
    test_scored["pred_segment_ensemble"] = seg_pred

    baseline_wape = wape(test_scored["sales_qty"], test_scored["pred_baseline"])
    tuned_wape = wape(test_scored["sales_qty"], test_scored["pred_tuned"])
    ensemble_wape = wape(test_scored["sales_qty"], test_scored["pred_segment_ensemble"])
    baseline_bias = bias_pct(test_scored["sales_qty"], test_scored["pred_baseline"])
    tuned_bias = bias_pct(test_scored["sales_qty"], test_scored["pred_tuned"])
    ensemble_bias = bias_pct(test_scored["sales_qty"], test_scored["pred_segment_ensemble"])

    print(f"\noverall test WAPE -- baseline: {baseline_wape:.4f}   "
          f"tuned (global): {tuned_wape:.4f}   segment_ensemble: {ensemble_wape:.4f}")

    # --- unchanged: category/abc_class/xyz_class/promo breakdown for baseline vs tuned ---
    group_cols = [c for c in ["category", "abc_class", "xyz_class"] if c in test_scored.columns]
    baseline_tables = segment_table(test_scored, train, "pred_baseline", group_cols)
    tuned_tables = segment_table(test_scored, train, "pred_tuned", group_cols)

    out_path = Path(__file__).resolve().parent / "tuned_vs_baseline_segments.csv"
    all_rows = []
    for gc in list(group_cols) + ["promo_flag"]:
        b = baseline_tables[gc][["wape", "bias_pct"]].add_suffix("_baseline")
        t = tuned_tables[gc][["wape", "bias_pct"]].add_suffix("_tuned")
        combined = b.join(t)
        combined["wape_delta"] = combined["wape_tuned"] - combined["wape_baseline"]
        combined["segment_type"] = gc
        all_rows.append(combined.reset_index().rename(columns={"index": "segment"}))
    pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
    print(f"wrote {out_path.name}")

    # --- NEW: the (abc_class x promo_flag) three-way comparison this rework is about ---
    ensemble_rows = []
    for abc_val, promo_val, seg_label in SEGMENTS:
        mask = segment_mask(test_scored, abc_val, promo_val)
        seg = test_scored[mask]
        winner_algo = winners[seg_label]["algo"] if seg_label in winners else "(fallback: global tuned)"
        row = {
            "segment": seg_label,
            "n_test_rows": int(mask.sum()),
            "winner_algo": winner_algo,
            "test_wape_baseline": wape(seg["sales_qty"], seg["pred_baseline"]),
            "test_wape_tuned_global": wape(seg["sales_qty"], seg["pred_tuned"]),
            "test_wape_segment_ensemble": wape(seg["sales_qty"], seg["pred_segment_ensemble"]),
        }
        row["wape_delta_ensemble_vs_tuned_global"] = (
            row["test_wape_segment_ensemble"] - row["test_wape_tuned_global"]
        )
        ensemble_rows.append(row)

    ensemble_report = pd.DataFrame(ensemble_rows).set_index("segment")
    ensemble_report_path = Path(__file__).resolve().parent / "segment_ensemble_report.csv"
    ensemble_report.to_csv(ensemble_report_path)
    print(f"\n=== segment ensemble vs global tuned model ===")
    print(ensemble_report.round(4))
    print(f"wrote {ensemble_report_path.name}")

    output_dir_models = output_dir / "Models"
    output_dir_models.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="tuned-vs-baseline-comparison") as parent_run:
        mlflow.set_tags({"feature_set_version": "horizon_consistent_v2"})
        mlflow.log_metric("test_wape_baseline", baseline_wape)
        mlflow.log_metric("test_wape_tuned", tuned_wape)
        mlflow.log_metric("test_wape_delta", tuned_wape - baseline_wape)
        mlflow.log_artifact(str(out_path))

        log_model_run("baseline", BASELINE_PARAMS, baseline_model, test_scored, train,
                       "pred_baseline", group_cols, baseline_wape,
                       output_dir_models / "baseline_lightgbm_rerun.txt", feature_cols, cat_cols)
        log_model_run("tuned", tuned_params, tuned_model, test_scored, train,
                       "pred_tuned", group_cols, tuned_wape,
                       output_dir_models / "tuned_lightgbm.txt", feature_cols, cat_cols)

        # One summary run for the segment ensemble -- not one nested run per
        # segment per algorithm (that detail already lives in
        # segment_algo_selection.json / the segment-algo-tuning-batch runs
        # from 6_optuna_tuning.py).
        with mlflow.start_run(run_name="segment_ensemble", nested=True):
            mlflow.set_tags({"stage": "segment_ensemble", "feature_set_version": "horizon_consistent_v2"})
            mlflow.log_metric("test_wape", ensemble_wape)
            mlflow.log_metric("test_wape_delta_vs_tuned_global", ensemble_wape - tuned_wape)
            for row in ensemble_rows:
                seg_label = row["segment"]
                mlflow.set_tag(f"winner_algo__{seg_label}", row["winner_algo"])
                mlflow.log_metric(f"test_wape__{seg_label}__baseline", row["test_wape_baseline"])
                mlflow.log_metric(f"test_wape__{seg_label}__tuned_global", row["test_wape_tuned_global"])
                mlflow.log_metric(f"test_wape__{seg_label}__segment_ensemble", row["test_wape_segment_ensemble"])
            mlflow.log_artifact(str(ensemble_report_path))

    print(f"\nlogged comparison to DagsHub, parent run: {parent_run.info.run_id}")
    print(f"view at: https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/experiments")

    metrics_out = {
        "test_wape_baseline": float(baseline_wape),
        "test_wape_tuned_global": float(tuned_wape),
        "test_wape_segment_ensemble": float(ensemble_wape),
        "test_wape_delta_ensemble_vs_tuned_global": float(ensemble_wape - tuned_wape),
        "test_bias_pct_baseline": float(baseline_bias),
        "test_bias_pct_tuned_global": float(tuned_bias),
        "test_bias_pct_segment_ensemble": float(ensemble_bias),
        "segments": ensemble_rows,
    }
    with open(Path(__file__).resolve().parent / "metrics_segment_ensemble.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("wrote metrics_segment_ensemble.json")


if __name__ == "__main__":
    main()
