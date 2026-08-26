"""
Per-segment algorithm selection + tuning (reworked step 5).

The original version of this script ran ONE Optuna study tuning ONE
LightGBM model against the whole dataset. That's a reasonable first pass,
but it silently assumes one algorithm and one hyperparameter set is right
for every SKU -- and a single global model pools loss across all of them,
so high-volume A-class rows dominate that pooled loss and the model ends
up weakest exactly where it matters least visibly: low-volume, intermittent
C-class series.

This version instead runs the search PER SEGMENT (ABC class x
promo/no-promo, 6 segments -- see dependancies/Segment_Algo_Functions.py),
across SEVEN candidates per segment:
  - 3 tabular ML models, each tuned via its own Optuna study: LightGBM,
    XGBoost, CatBoost.
  - 4 classical per-SKU demand-history baselines, each grid-searched over
    a tiny parameter space: Croston, TSB, seasonal-naive, moving-average.
    These are promo-blind by construction (they only see one SKU's own
    demand history) but cost almost nothing to compute, so they're a
    useful floor: if a tuned tree model can't beat a dumb per-SKU average
    on some segment, that's worth knowing.

For every segment, whichever candidate has the lowest val WAPE gets
recorded as that segment's "winner" in segment_algo_selection.json.
7_tuned_vs_baseline_segments.py reads that file, retrains each segment's
winner, and compares the resulting per-segment ensemble against the
single global tuned model from before.

THIS SCRIPT DOES NOT TOUCH THE REGISTRY OR app.py. It's an analysis step:
"which algorithm wins on which segment," not a decision about how serving
should work. Turning a per-segment winner set into something app.py
actually routes predictions through is a bigger design change (multiple
models to version, a routing layer, per-segment drift monitoring) that's
deliberately out of scope here -- see segment_algo_selection.json and
7_tuned_vs_baseline_segments.py's comparison output, then decide from
there whether it's worth it.

RUNTIME / RESUMABILITY
-----------------------
Same resumable-batch design as the original script, just multiplied across
segments x ML algorithms (18 Optuna studies instead of 1, all sharing one
SQLite file, keyed by study_name=f"{segment}__{algo}"). Because 18 studies
is a lot to run in one blocking call, two extra env vars let you scope a
single invocation down:

  N_TRIALS        -- trials to run per (segment, algorithm) pair this call
                     (default from params.yaml's tuning.n_trials_default).
  SEGMENT_FILTER   -- comma-separated segment labels to run this call, e.g.
                     "C_promo,C_no_promo" to focus on the segment you
                     actually care about right now. Default: all 6.
  ALGO_FILTER      -- comma-separated ML algorithm names, e.g. "catboost".
                     Default: all 3 (lightgbm,xgboost,catboost).

The classical candidates are always fully recomputed every call regardless
of these filters -- they're cheap (a few seconds for all 4, all 6
segments) and deterministic, so there's no "resuming" concept for them.

Every trial is logged to DagsHub MLflow, one level less granular than the
original script: one parent run per invocation
("segment-algo-tuning-batch"), and ONE child metric per (segment,
algorithm) for THIS call's best-so-far -- not one nested run per individual
Optuna trial. With 18 studies x potentially dozens of trials each, logging
every trial as its own MLflow run would make the DagsHub experiment
unreadable; per-trial detail still lives in the Optuna SQLite study
(optuna_segment_studies.db) if you want to inspect it directly.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
import mlflow
import dagshub

from dependancies.Features_Functions import assert_no_leakage
from dependancies.Metrics_Functions import wape
from dependancies.params_utils import load_params
from dependancies.Segment_Algo_Functions import (
    SEGMENTS, MIN_SEGMENT_ROWS, ML_CANDIDATES,
    segment_mask, evaluate_stat_candidates,
)

_tuning_p = load_params()["tuning"]
N_TRIALS = int(os.environ.get("N_TRIALS", str(_tuning_p["n_trials_default"])))
SEED = _tuning_p["seed"]
EXPERIMENT_NAME = "fashion-demand-forecast-segment-algo-selection"
DAGSHUB_REPO_OWNER = "Sandeepraju-42"
DAGSHUB_REPO_NAME = "Forecast_MLFlow_Fashion_Dataset"

NON_FEATURE_COLS = ["sku_id", "date", "sales_qty", "true_demand", "pattern_type"]
CATEGORICAL_COLS = [
    "category", "sub_category", "color", "material", "brand", "gender",
    "size_range", "channel", "markdown_stage", "lifecycle_stage",
    "discount_bucket", "abc_class", "xyz_class",
]

input_dir = Path(__file__).resolve().parent / "input"

_all_segment_labels = {label for _, _, label in SEGMENTS}
_seg_filter_env = os.environ.get("SEGMENT_FILTER")
SEGMENTS_TO_RUN = (
    [s for s in SEGMENTS if s[2] in {x.strip() for x in _seg_filter_env.split(",")}]
    if _seg_filter_env else SEGMENTS
)
_algo_filter_env = os.environ.get("ALGO_FILTER")
ALGOS_TO_RUN = (
    {x.strip() for x in _algo_filter_env.split(",")} & set(ML_CANDIDATES)
    if _algo_filter_env else set(ML_CANDIDATES)
)


def load_data():
    train = pd.read_csv(input_dir / "train_features.csv", parse_dates=["date"])
    val = pd.read_csv(input_dir / "val_features.csv", parse_dates=["date"])

    for col in CATEGORICAL_COLS:
        if col not in train.columns:
            continue
        all_categories = pd.concat([train[col], val[col]]).astype(str).unique()
        cat_dtype = pd.CategoricalDtype(categories=sorted(all_categories))
        for split in (train, val):
            split[col] = split[col].astype(str).astype(cat_dtype)

    return train, val


def main():
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train, val = load_data()
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]
    assert_no_leakage(train, [c for c in feature_cols if train[c].dtype.kind in "if"])
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    out_path = Path(__file__).resolve().parent / "segment_algo_selection.json"
    if out_path.exists():
        with open(out_path) as f:
            state = json.load(f)
        state.setdefault("segments", {})
    else:
        state = {"segments": {}}

    # Same OPTUNA_DB_PATH override as the original script -- matters when
    # this runs through a network/FUSE-style mount rather than a native
    # filesystem (SQLite locking can throw "disk I/O error" there).
    db_path = Path(os.environ.get(
        "OPTUNA_DB_PATH", str(Path(__file__).resolve().parent / "optuna_segment_studies.db")
    ))

    print(f"running segments: {[s[2] for s in SEGMENTS_TO_RUN]}")
    print(f"running ML algorithms: {sorted(ALGOS_TO_RUN)}")
    print(f"N_TRIALS per (segment, algorithm) this call: {N_TRIALS}\n")

    with mlflow.start_run(run_name="segment-algo-tuning-batch") as parent_run:
        mlflow.set_tags({
            "stage": "segment_algo_tuning",
            "feature_set_version": "horizon_consistent_v2",
        })
        mlflow.log_param("n_trials_per_algo_this_call", N_TRIALS)
        mlflow.log_param("segments_this_call", ",".join(s[2] for s in SEGMENTS_TO_RUN))
        mlflow.log_param("algos_this_call", ",".join(sorted(ALGOS_TO_RUN)))

        for abc_val, promo_val, seg_label in SEGMENTS_TO_RUN:
            train_seg = train[segment_mask(train, abc_val, promo_val)]
            val_seg = val[segment_mask(val, abc_val, promo_val)]
            n_train_seg = len(train_seg)
            print(f"=== segment {seg_label}: {n_train_seg} train rows, {len(val_seg)} val rows ===")

            seg_entry = state["segments"].setdefault(seg_label, {"candidates": {}})
            seg_entry["n_train_rows"] = n_train_seg
            seg_entry["n_val_rows"] = len(val_seg)

            # Classical candidates fit on FULL train history per SKU (not
            # train_seg) -- see Segment_Algo_Functions.py's module
            # docstring for why -- scored on this segment's own val rows.
            # Always fully recomputed: cheap, deterministic, no batching.
            stat_results = evaluate_stat_candidates(train, val_seg)
            for algo, res in stat_results.items():
                seg_entry["candidates"][algo] = res
                mlflow.log_metric(f"val_wape__{seg_label}__{algo}", res["val_wape"])
                print(f"  {algo:16s} val WAPE {res['val_wape']:.4f}  params={res['params']}")

            if n_train_seg < MIN_SEGMENT_ROWS:
                seg_entry["ml_tuning_skipped"] = True
                seg_entry["ml_tuning_skip_reason"] = (
                    f"only {n_train_seg} train rows (< MIN_SEGMENT_ROWS={MIN_SEGMENT_ROWS})"
                )
                print(f"  skipping ML tuning: {seg_entry['ml_tuning_skip_reason']}")
            else:
                seg_entry["ml_tuning_skipped"] = False
                for algo in sorted(ALGOS_TO_RUN):
                    space_fn, fit_fn = ML_CANDIDATES[algo]
                    study_name = f"{seg_label}__{algo}"
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=f"sqlite:///{db_path}",
                        direction="minimize",
                        sampler=TPESampler(),  # unseeded -- see original script's comment
                        load_if_exists=True,
                    )
                    n_done = len(study.trials)

                    def objective(trial, _space_fn=space_fn, _fit_fn=fit_fn):
                        params = _space_fn(trial, seed=SEED)
                        _, pred, best_iter = _fit_fn(params, train_seg, val_seg, feature_cols, cat_cols)
                        trial.set_user_attr("best_iteration", best_iter)
                        return wape(val_seg["sales_qty"], pred)

                    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

                    seg_entry["candidates"][algo] = {
                        "params": study.best_params,
                        "val_wape": study.best_value,
                        "n_trials_total": len(study.trials),
                    }
                    mlflow.log_metric(f"val_wape__{seg_label}__{algo}", study.best_value)
                    print(f"  {algo:16s} val WAPE {study.best_value:.4f}  "
                          f"({n_done} done before this call, {len(study.trials)} total)")

            valid = {
                k: v for k, v in seg_entry["candidates"].items()
                if v is not None and v.get("val_wape") is not None and not np.isnan(v["val_wape"])
            }
            if valid:
                winner = min(valid, key=lambda k: valid[k]["val_wape"])
                seg_entry["winner"] = winner
                seg_entry["winner_val_wape"] = valid[winner]["val_wape"]
                print(f"  -> winner: {winner} (val WAPE {seg_entry['winner_val_wape']:.4f})\n")
            else:
                seg_entry["winner"] = None
                print("  -> no valid candidate this call\n")

            # Written after EVERY segment, not just once at the end -- a
            # segment's Optuna trials are already durable in
            # optuna_segment_studies.db the moment they finish, but this
            # JSON snapshot is only rebuilt when the script runs to
            # completion. Segments x algorithms is enough work that a call
            # can run long (each ML algorithm's Optuna trials on a large
            # segment take real time); if it gets killed or times out
            # partway through, saving here means everything up to the
            # segment in progress is still reflected on disk, instead of
            # silently reverting to whatever the last fully-completed call
            # left behind.
            state["min_segment_rows"] = MIN_SEGMENT_ROWS
            with open(out_path, "w") as f:
                json.dump(state, f, indent=2)

        mlflow.log_param("min_segment_rows", MIN_SEGMENT_ROWS)

    print(f"logged batch to DagsHub, parent run: {parent_run.info.run_id}")
    print(f"view at: https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/experiments")
    print(f"\nwrote {out_path.name}")
    print("\ncurrent winners by segment:")
    for label, entry in state["segments"].items():
        w = entry.get("winner")
        wv = entry.get("winner_val_wape")
        if w is not None:
            print(f"  {label:12s} -> {w:16s} (val WAPE {wv:.4f})")
        else:
            print(f"  {label:12s} -> (not yet run)")


if __name__ == "__main__":
    main()
