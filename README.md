# Fashion Demand Forecast -- MLOps Pipeline

End-to-end demand forecasting for fashion SKUs: synthetic data generation,
leakage-checked feature engineering, a LightGBM baseline, Optuna tuning,
DagsHub-tracked MLflow experiments, a DVC pipeline, a model registry with a
Flask serving API, a Power BI export, and CI/CD/Docker/cloud-deployment
scaffolding.

## Project plan and where each step lives

| # | Step | Files |
|---|------|-------|
| 1 | Data collection | `0_synthetic_Data.py` |
| 2 | Preprocessing + EDA | `Features.py`, `1_EDA_Jupyter.ipynb` |
| 3 | Baseline model | `2_baseline_model.py` |
| 4 | MLflow tracking (DagsHub) | `3_MLFlow_Dagshub.py` |
| 5 | Optuna tuning + segment analysis | `4_optuna_tuning.py`, `5_tuned_vs_baseline_segments.py` |
| 6 | DVC pipeline | `dvc.yaml`, `params.yaml`, `params_utils.py` |
| 7 | Model registry + Flask API | `register_model.py`, `app.py` |
| 8 | Power BI consumption layer | `export_predictions.py` |
| 9 | CI/CD | `.github/workflows/ci.yml`, `.github/workflows/cd.yml` |
| 10 | Dockerization | `Dockerfile`, `.dockerignore` |
| 11 | Cloud deployment | `.github/workflows/cd.yml` (AWS ECR/ECS -- template, needs your AWS setup) |
| 12 | GitHub management | this file, `.gitignore` |

## Running the pipeline

Config lives in one place, `params.yaml` (forecast horizon, validation
horizon, synthetic-data size, baseline LightGBM hyperparameters, tuning
seed). Change a value there and the affected stages know to rerun.

**With DVC** (recommended -- tracks what's stale and skips what isn't):

```bash
pip install -r requirements.txt dvc
dvc repro
```

**By hand**, same order DVC runs them in:

```bash
python 0_synthetic_Data.py
python Features.py
python 2_baseline_model.py
python 4_optuna_tuning.py          # repeat a few times -- see below
python 5_tuned_vs_baseline_segments.py
python register_model.py
```

`4_optuna_tuning.py` is meant to be invoked more than once: it resumes from
`optuna_study.db` (or wherever `OPTUNA_DB_PATH` points) and runs `N_TRIALS`
more each call (env var, default from `params.yaml`'s `tuning.n_trials_default`).
Run it as many times as you want search depth, then run
`5_tuned_vs_baseline_segments.py` once to compare the best result found so
far against the baseline.

## Experiment tracking

Every training/tuning run logs to DagsHub MLflow
(`https://dagshub.com/Sandeepraju-42/Forecast_MLFlow_Fashion_Dataset/experiments`):
`3_MLFlow_Dagshub.py`'s baseline run, each Optuna trial (nested under a
per-invocation parent run), and the final baseline-vs-tuned comparison
(nested baseline/tuned runs under one parent, with per-segment metrics,
feature importance, and the model artifact each).

`dvc metrics diff` compares `metrics_baseline.json` / `metrics_tuned.json`
across git commits -- the lightweight, git-native complement to MLflow's
richer per-run UI.

## Model registry + serving API

`register_model.py` finds the latest DagsHub MLflow run tagged
`stage=tuned`, registers its model under `fashion-demand-forecaster`, and
promotes it to the `champion` alias -- but only if it beats the current
champion's test WAPE (pass `--force` to override).

`app.py` serves whatever is aliased `champion`, with a local-file fallback
(`Output/tuned_lightgbm.txt` + `feature_schema_tuned.json`) if the registry
can't be reached. Run locally:

```bash
python app.py                 # dev server on :5000
# or, matching what the Dockerfile runs:
gunicorn --bind 0.0.0.0:5000 app:app
```

```bash
curl -X POST localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"category": "Dresses", "sku_id": "SKU00001", ...}]}'
```

Request rows should already be feature-engineered (i.e. what
`Features.py` produces) -- the endpoint scores a feature row, it doesn't
rebuild lag/rolling features from raw history.

## Power BI

```bash
python export_predictions.py
```

writes `predictions_for_powerbi.csv` (actual vs. predicted, by SKU/category/
ABC/XYZ/promo) using the exact same model-loading and encoding path as
`app.py`. In Power BI: **Get Data -> Text/CSV** -> point at that file.

## Docker

```bash
docker build -t fashion-demand-forecaster .
docker run -p 5000:5000 fashion-demand-forecaster
```

Only `app.py` + its dependencies + one fallback model/schema pair go into
the image (see `.dockerignore`) -- the training/tuning scripts and the full
dataset don't need to ship with the serving container.

## CI/CD

`.github/workflows/ci.yml` runs `tests/` (leakage checks, metrics, split
boundaries -- pure unit tests, no data generation needed) plus a
report-only lint pass on every push/PR to `main`.

`.github/workflows/cd.yml` builds the Docker image and deploys it to AWS
ECS on push to `main`. It's a template: it needs an ECR repo, an ECS
cluster/service, and these GitHub secrets before it does anything --
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
`ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`. See the comments at the top
of that file for the full setup.

## Tests

```bash
pytest tests/ -v
```

Covers the parts where a silent regression would be most costly:
`assert_no_leakage`, the metrics functions (hand-computed expected values,
not just mirrors of the implementation), the train/val/test split
boundaries, and the ABC/XYZ `as_of` cutoff (the exact bug class this
project already found once -- a hardcoded cutoff that leaked into the
validation window).
