# Fashion Demand Forecasting -- MLOps Pipeline

A weekly SKU-level demand forecasting project on GCP, built specifically t go deep on the parts of MLOps that are easy to skip. 

The steps include 
- feature-engineering
- correctness
- experiment tracking
- a real champion/challenger retraining pipeline
- monitoring, and 
- CI/CD

**Important**: It's a portfolio project, not a production system with a real business behind it yet (see "What's real vs. placeholder" below before
assuming everything here runs against live data.)

**The dataset**

Synthetic data (300 SKUs, weekly, ~2.5 years) stands in for a real fashion retailer's sales history: 
prices, discounts, promotions, marketing spend, weather, product lifecycle, and a handful of endogenous demand signals (reviews, wishlist adds, social trend score, inventory/stockouts), all generated with a real underlying data-generating process (trend + seasonality + price/promo/weather/marketing effects + lifecycle ramp-and-decline +noise) in 1_Generate_Synthetic_Data.ipynb. Not using np.random with no structure to actually learn from.

## Notebooks

| Notebook | What it does |
|---|---|
| `0_Setup.ipynb` | One-time GCP project setup: enable APIs, create the GCS bucket, IAM, package installs. |
| `1_Generate_Synthetic_Data.ipynb` | Generates the synthetic dataset described above and uploads `config/params.yaml`. This is the single source of truth for `forecast_horizon` (8 weeks), `validation_horizon` (104 weeks), and `granularity` ("W"), read everywhere else via `functions/params_utils.load_params()`. |
| `2_Features.ipynb` | Runs the shared feature-engineering functions (`functions/Features_Functions.py`) over the raw data: calendar features, lifecycle stage, price features, target/endogenous lags and rolling stats, ABC/XYZ classification, one-hot encoding, and the time-based train/validation/test split. Also builds `future_df.csv, the `forecast_horizon` future rows per SKU that Vertex AI batch prediction scores against. |
| `3_EDA.ipynb` | Exploratory analysis: target distribution, STL seasonal decomposition, stationarity (ADF/ACF/PACF), price/promo/weather relationships. |
| `4_Baseline_Model.ipynb` | Trains the first model: Vertex AI **AutoML Forecasting** on the full feature set, evaluates it (RMSE/MAE/MAPE/WAPE), backtests against the TEST split, and pulls Shapley-value feature attributions. |
| `5_Model_Comparison_ABC_Category.ipynb` | Sweeps three forecasting approaches. AutoML Forecasting, a local `statsforecast` AutoARIMA (`RUN_LOCAL_ARIMA`, default on (free), replaces BQML ARIMA_PLUS to save credits), and darts N-BEATS/N-HiTS across context windows, scored by WAPE at the overall/category/ABC-class level. AutoML itself stays gated behind `RUN_AUTOML = False` since it costs real money per run. |
| `5.1_Experiment_Tracking.ipynb` | Logs every run from the comparison above to Vertex AI Experiments (the MLflow equivalent on GCP) so the sweep results are queryable later, not just printed once and lost. |
| `6_Kubeflow_Pipelines.ipynb` | The retraining pipeline that actually runs in production: refresh data -> train an **XGBoost** candidate -> compare it against the current champion's real stored WAPE -> conditionally deploy to a Vertex AI Endpoint -> write an 8-week batch forecast to BigQuery -> notify by email/Pub-Sub. Also where the pipeline's weekly schedule lives. See "Two model tracks" below for why this trains its own XGBoost model rather than reusing AutoML. |
| `7_Monitoring_and_Consumption.ipynb` | Reads what notebook 6 writes: querying the forecast tables (what a BI tool connects to), skew/drift monitoring on the deployed endpoint, and a scheduled realized-vs-predicted accuracy query. |
| `8_CICD_and_Packaging.ipynb` | Writes out `src/`, `tests/`, `ci/`, `.github/workflows/`, and this README/requirements/architecture doc . Run once (or after a real change to the pipeline) to (re)generate the repo scaffold; not part of the pipeline itself. |

## The `functions/` package

Shared, single-sourced logic imported by more than one notebook, instead of each notebook keeping its own copy:

- **`Features_Functions.py`** : every feature-engineering function (`add_calendar_features`, `add_lifecycle_features`, `add_price_features`, `add_target_lag_features`, `add_endogenous_lagged_features`, `add_cross_product_features`, `add_abc_classification`, `add_xyz_classification`, `encode_categoricals`, `time_based_split`), plus the covariate typing Vertex AI Forecasting needs: `AVAILABLE_AT_FORECAST` / `UNAVAILABLE_AT_FORECAST` (which columns must vs. must not carry a real value on a future batch-prediction row) and `build_column_specs()` (the categorical/numeric/timestamp map every AutoML Forecasting training job needs). `2_Features.ipynb`, `4_Baseline_Model.ipynb`, and `5_Model_Comparison_ABC_Category.ipynb` all import these rather than each hand-typing the same ~80-column lists
- **`Metrics_Functions.py`**:`wape`, `bias_pct`, `tracking_signal` (+ per-SKU/summary variants), `mase`, and a combined `evaluate()`.
- **`params_utils.py`**:`load_params()`, the one function that reads `gs://<bucket>/config/params.yaml`. Everything that needs `forecast_horizon`/`validation_horizon`/`granularity` reads it from here, not a locally hardcoded number.


## Two model tracks -- and why there are two

**AutoML Forecasting** (notebooks 4 and 5) is the "which approach wins" experimentation track. It's the fastest way to get a strong forecasting model with zero feature-selection or hyperparameter work, and it's the benchmark the comparison sweep in notebook 5 measures every alternativem against.

**A self-trained XGBoost model** (`train_xgboost_candidate_op` inside `6_Kubeflow_Pipelines.ipynb`) is what the actual retraining *pipeline* uses, not AutoML. This is a deliberate, practical choice, not an oversight: a KFP
pipeline needs full programmatic control over training, evaluation, and the
champion/challenger comparison on every run, and AutoML Forecasting is
designed to be driven interactively (or as one long-running job you poll),
not as a fast, cheap, fully scriptable step inside a DAG that also needs to
compare against a stored prior WAPE and conditionally deploy. Training a
plain `XGBRegressor` directly inside the component gives that control (and
costs nothing beyond the container's own compute) at the price of AutoML's
own search over model architectures

## Repo layout

```
functions/              shared package -- feature engineering, metrics, params.yaml loader
config/params.yaml       forecast_horizon / validation_horizon / granularity (uploaded by notebook 1)
0.._8..ipynb             the pipeline, one numbered notebook per stage (see table above)
src/forecasting/        shared package for the CI/CD scaffold 
tests/                   pytest, unit tests + a pipeline-compiles check
ci/                      scripts the GitHub Actions workflows call
.github/workflows/       CI (lint+test) and deploy-pipeline (recompile+push on merge)
docs/ARCHITECTURE.md     how the pieces fit together and why some choices were made
```

`forecasting_pipeline.yaml` (the compiled KFP pipeline template) is a build artifact, not source. `ci/compile_pipeline.py` regenerates it from
`6_Kubeflow_Pipelines.ipynb` on every CI run, so it's gitignored rather than committed; a committed copy would just go stale the next time the notebook
changes.

## Running things locally

```
pip install -r requirements.txt        # to run the notebooks themselves
pip install -r requirements-dev.txt    # to run the CI/CD scaffold's own tests
ruff check src/ tests/ ci/
PYTHONPATH=src pytest tests/ -v
```

`test_pipeline_compiles.py` is worth understanding on its own: it extracts
notebook 6's actual component + pipeline code and feeds it to the real `kfp`
compiler, so a change that breaks the pipeline's DAG (the exact class of bug
this project hit early on a `PipelineArtifactChannel` with no `.uri`)
fails CI, not just a future notebook run.


## What's real vs. placeholder


**Real and tested:** the pipeline's plumbing end to end like DAG dependencies,
artifact passing, the conditional-deploy logic, the notification wiring.
`train_xgboost_candidate_op` fits a real `XGBRegressor` on the feature data
and saves a real model artifact (verified by reloading it with a fresh
`xgb.Booster()`, the same load path the serving container uses).
`deploy_model_op` registers and deploys that model to a live Vertex AI
Endpoint, with retry logic around a transient backend error class hit on a
real run. `evaluate_and_compare_op` compares the candidate against the
*actual* current champion's stored validation WAPE (a Model Registry label
set at deploy time), not a hardcoded number. `batch_forecast_to_bq_op`
downloads the *actual* current champion's `model.bst` from its Model
Registry artifact URI, loads it with a real `xgb.Booster`, and produces a
genuine 8-step-ahead forecast per SKU with every lag/rolling feature
correctly anchored to match training (see "Recently fixed" above). It also
writes a direct-look CSV (`gs://<bucket>/data/output/actual_vs_predicted.csv`)
with every historical row's real actual value next to the champion's own
in-sample prediction for that row, followed by the forecast horizon rows, so
the model's fit and its forecast are visible in one file without writing SQL
against BigQuery first.

**Still placeholder:** `prepare_data_op`'s data-refresh step
(`# df = fetch_fresh_data()`) training and batch-forecasting both still
read a static CSV from GCS, so nothing here reflects genuinely new data. The
one remaining honest simplification in both `2_Features.ipynb`'s future rows
and `batch_forecast_to_bq_op`: covariates with no real future plan in this
synthetic dataset (price, discount_pct, marketing_spend, ...) are carried
forward from each SKU's last known row rather than a real plan and the thing
to revisit once `prepare_data_op` does real refreshes. A real `actuals` feed
for the accuracy job, and Terraform for the infra pieces, are the other
concrete next steps -- not "someday," genuinely the next things to build.

See `docs/ARCHITECTURE.md` for how the pieces fit together and why a few of
the less-obvious choices were made.
