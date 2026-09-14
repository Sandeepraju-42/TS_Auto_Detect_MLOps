# Architecture

## The project, end to end

```mermaid
flowchart TD
    N0[0_Setup] --> N1[1_Generate_Synthetic_Data]
    N1 -->|writes config/params.yaml| P[(params.yaml)]
    N1 --> N2[2_Features]
    P -.read by.-> N2
    N2 --> N3[3_EDA]
    N2 --> N4[4_Baseline_Model: AutoML]
    N2 --> N5[5_Model_Comparison: AutoML vs local AutoARIMA vs darts]
    N5 --> N51[5.1_Experiment_Tracking]
    N2 --> N6

    subgraph N6[6_Kubeflow_Pipelines -- the retraining pipeline]
        direction TD
        A[prepare_data_op] --> B[train_xgboost_candidate_op]
        B --> C[evaluate_and_compare_op]
        C -->|candidate wins| D[deploy_model_op]
        D --> E1[notify_promotion_op -> Pub/Sub]
        D --> F1[batch_forecast_to_bq_op]
        C -->|candidate loses| F2[batch_forecast_to_bq_op]
        F1 --> G[(latest_forecasts / forecast_history)]
        F2 --> G
    end

    G --> N7[7_Monitoring_and_Consumption]
    N7 --> H[Power BI]
    N7 --> I[forecast_accuracy scheduled query]
    J[ModelDeploymentMonitoringJob] -.watches live endpoint traffic.-> D
    N8[8_CICD_and_Packaging] -.scaffolds src/tests/ci/.github, not part of the DAG.-> N6
```

Everything inside the `6_Kubeflow_Pipelines` box is one KFP pipeline,
wrapped in a `dsl.ExitHandler` that emails on completion either way. `J`
(skew/drift monitoring) sits outside the pipeline entirely -- it watches
the *endpoint*, not a pipeline run. Notebooks 0-3 and 5/5.1 are upstream of
the pipeline (data generation, feature engineering, and the offline
algorithm comparison that justified training XGBoost inside the pipeline
rather than something else); notebook 7 is downstream of it.

## `config/params.yaml` -- one source of truth for the horizon

`1_Generate_Synthetic_Data.ipynb` uploads `gs://<bucket>/config/params.yaml`
with three values: `forecast_horizon` (8 weeks), `validation_horizon` (104
weeks), and `granularity` ("W"). `functions/params_utils.load_params()` is
the only function that reads it, and `functions/Features_Functions.py`
reads it once at import time into `FORECAST_HORIZON` /
`VALIDATION_HORIZON` / `GRANULARITY` (and derives `LAGS` /
`ROLLING_WINDOWS` from `GRANULARITY`). Every notebook that needs the
horizon -- `2_Features.ipynb`'s future-row construction,
`4_Baseline_Model.ipynb` and `5_Model_Comparison_ABC_Category.ipynb`'s
training-job calls, `6_Kubeflow_Pipelines.ipynb`'s batch-forecast component
-- either imports `FORECAST_HORIZON` from `Features_Functions` or receives
it as a KFP component parameter defaulted to the same value. This replaced
three independently hardcoded `forecast_horizon = 8` literals that had
already started to drift from each other before being consolidated -- see
the README's "Recently fixed" section.

## Feature engineering: what's available at forecast time, and why it matters

Vertex AI Forecasting (and this project's own batch-forecast logic) needs
to know, for every column, whether it can have a real value on a *future*
row or not:

- **`AVAILABLE_AT_FORECAST`** (`functions/Features_Functions.py`) -- static
  product attributes (category, color, brand, ...), calendar features, and
  covariates this synthetic dataset has no real forward-looking plan for
  (price, discount_pct, marketing_spend, ...). That last group is carried
  forward from each SKU's last known value on future rows rather than a
  real plan -- an explicit, honest simplification, not a hidden one; see
  "Known gaps" below.
- **`UNAVAILABLE_AT_FORECAST`** -- the target itself, plus every lag/rolling
  feature and endogenous business signal (reviews, weather, inventory, ...)
  that genuinely isn't known in advance. These stay `null` on future rows;
  that null is what tells Vertex AI "forecast this row."

Before this was centralized, `2_Features.ipynb`'s future-row cell only
populated date-derived columns and left the rest -- including `abc_class`
-- null, which produced a real `Missing struct property: abc_class`
batch-prediction failure. It's fixed now (see README), but the underlying
lesson is why these two lists live in one file (`Features_Functions.py`)
that every consumer imports, instead of three separate hand-typed copies
that can silently stop matching each other.

## Two model tracks, and why `6_Kubeflow_Pipelines.ipynb` doesn't use AutoML

`4_Baseline_Model.ipynb` and `5_Model_Comparison_ABC_Category.ipynb` are
the experimentation track: they exist to answer "which forecasting
approach is actually best for this data," sweeping Vertex AI AutoML
Forecasting against a local `statsforecast` AutoARIMA and darts
N-BEATS/N-HiTS by WAPE, broken out by category and ABC class.

The retraining *pipeline* (`6_Kubeflow_Pipelines.ipynb`) instead trains a
plain `XGBRegressor` inside `train_xgboost_candidate_op`. This is a
deliberate tradeoff, not an inconsistency: a KFP pipeline needs full
programmatic control on every run -- train, compare against the *actual*
current champion's stored WAPE, and conditionally deploy, all inside one
scripted DAG. AutoML Forecasting is built to be driven interactively (or
polled as one long job), not as a fast, cheap, fully scriptable step that
also needs to read back a prior run's metric and branch on it. Training
XGBoost directly inside the component gives that control, at the cost of
AutoML's own architecture search -- worth stating plainly rather than
leaving the two tracks looking like an oversight.

## Decisions worth explaining, not just stating

**Why `batch_forecast_to_bq_op` is called from both branches of the
conditional, instead of once after it.** KFP's control-flow groups
(`dsl.If`/`dsl.Else`) don't allow a task outside the group to depend on
one defined inside it -- the inside task might not have run at all. "Run
this exactly once, whichever way the condition went" has to mean calling
it from both branches, each correctly ordered relative to whatever else
that branch does. Confirmed by compiling a minimal throwaway pipeline
with this exact shape and inspecting the resulting DAG before touching
the real pipeline.

**Why the champion lookup is `Model.list(..., order_by="create_time
desc")[0]`, not just `Model.list(...)[0]`.** `Model.list()`'s default
ordering is unspecified -- confirmed via the SDK's own docstring. Every
retraining run registers a new model under the same `display_name`
(no `parent_model=` is passed, so each is its own resource, not a new
version of one), so without an explicit order, "the champion" would have
been an arbitrary past model, not reliably the current one. A real bug,
caught while reusing this exact lookup for the batch-forecast component.

**Why local `statsforecast`/darts replace BQML/AutoML by default.**
Straightforward cost tradeoff, not a technical constraint -- BQML
training and Vertex AI AutoML Forecasting both cost real money per run
(AutoML: ~$21/node-hour; BQML: per-TB-scanned x candidate models x
backtest windows), and this project's GCP credits are limited. AutoML
stays available behind `RUN_AUTOML = False`, and local AutoARIMA behind
`RUN_LOCAL_ARIMA = True` (it's free, so it defaults on), for when it's
worth spending on that comparison point again.

**Why notifications split into email vs. Pub/Sub instead of one
channel.** `dsl.ExitHandler` + `VertexNotificationEmailOp` only knows
"the pipeline finished" (success or failure) -- it can't distinguish "the
candidate lost, nothing changed" from "the candidate won and is now
live." Those are genuinely different events for different audiences (one
is an incident signal, one is a deployment signal), so they're two
separate mechanisms rather than overloading one.

**Why `aiplatform.PipelineJobSchedule` instead of Cloud Scheduler + a
Cloud Function.** Vertex has a native scheduling resource for pipeline
jobs -- one thing to create and maintain, billed the same as a manually
submitted run (nothing extra for the schedule itself), instead of a
Scheduler job whose only purpose is to fire an HTTP endpoint that a
separate Cloud Function turns into a pipeline submission.

**Why the realized-vs-predicted accuracy job is a BigQuery Scheduled
Query, not another KFP pipeline.** It's one SQL join and aggregation over
two tables. Spinning up a full pipeline container to run one query would
be paying container-startup overhead for something BigQuery already
schedules natively, for free. Same reasoning as the point above: match
the tool to the actual job, don't reach for the heaviest one by default.

**Why CI/CD authenticates with Workload Identity Federation, not a
service account key.** A downloaded JSON key is a long-lived credential
that has to be stored as a GitHub secret and rotated manually forever;
WIF lets GitHub Actions impersonate a service account per-run via a
short-lived token, with no standing credential to leak or rotate. This is
Google's own current recommendation for CI/CD auth, not a preference.

**Why the champion's WAPE is stored as a Model Registry label, not read
from MLflow or a metrics file.** `evaluate_and_compare_op` needs the
*previous* winning run's WAPE at comparison time, and the Model resource
it already looks up (to find the current champion) is the simplest place
to keep it -- no second system to query, no extra artifact to pass
around. Labels can't hold a `.` (lowercase letters/digits/underscores/
dashes only, per the SDK's own docstring), so the value is stored as basis
points (`round(wape * 10000)`) and divided back down on read. A real bug
this replaced: the comparison used to be against a hardcoded `0.100`,
which a real (non-synthetic) candidate's WAPE will often lose to
regardless of whether it actually improved on the champion.

**Why `batch_forecast_to_bq_op` builds its own future rows instead of
reading them from the feature file, and why its lag/rolling features are
anchored the way they are.** The static feature file has no genuine
future dates -- there's no row to predict *against* beyond what already
exists, so the component constructs `forecast_horizon` (default 8) rows
per SKU itself. Date-derived features (`week_of_year`, `month`, sin/cos
pairs) are recomputed for the real future date, not carried forward. Lag
features (`sales_qty_lag_8` and up, plus the four `*_lag8` business-signal
columns) and the rolling features (`rollmean_4/12`, `rollstd_4/12`) both
have to land on the *same anchor training used*: `add_target_lag_features`
shifts the target by `forecast_horizon` before computing lags directly
(`shift(lag)`, `lag >= forecast_horizon` always) and before rolling
(`shift(forecast_horizon).rolling(window)`), so a training-time rolling
feature at row `p` reflects the window ending at `p - forecast_horizon`,
not `p - 1`. The inference-time formula for both is
`len(history_so_far) - forecast_horizon` as the anchor point -- worked out
by hand and confirmed by backtesting the fixed formula against pandas'
own training-time calculation (32/32 checks matched exactly; the earlier
naive "last `w` values seen so far" version matched essentially none of
them). Each step's own prediction is still appended to a running history
afterwards: under this project's actual `forecast_horizon` (8, matching
what's baked into training), every step's anchor resolves within real
history and the appended predictions are never actually read back for the
target column -- but the append keeps the code correct in general, for a
hypothetical future run where the component's `forecast_horizon` argument
exceeds the one training used. The one covariate group still carried
forward flatly (price, discount_pct, marketing_spend, ...) -- in both this
component and `2_Features.ipynb`'s future rows -- is the thing to revisit
once `prepare_data_op` does real refreshes and a genuine forward plan for
those exists.

## Known gaps -- what's still ahead, stated plainly

- **`prepare_data_op` doesn't refresh data.** Still the original
  placeholder (`# df = fetch_fresh_data()`). Training and batch-forecasting
  both read a static CSV from GCS regardless of what this component does.
  This is now the single biggest gap -- training, evaluation, deployment,
  and batch inference are all real; only the data feeding all of them is
  static.
- **No `actuals` feed.** The realized-vs-predicted accuracy query reads
  from a table nothing populates. Needs a real source for demand actuals
  once they're known.
- **No Terraform / IaC.** Every resource created in these notebooks
  (Pub/Sub topic, BQ dataset, the pipeline schedule, the monitoring job)
  was created by hand-run notebook cells, not declared as code. That's
  fine for a portfolio project's current stage; it's the natural next
  phase before this could be called a real production setup.
- **`src/forecasting/` (the CI/CD scaffold's own package) has exactly one
  function in it.** `data_helpers.py` is the first piece of what the
  roadmap calls a shared package for that scaffold specifically -- separate
  from `functions/`, which is the real, already-multi-file package the
  notebooks themselves import. Notebook 5's own `fill_series_gaps` still
  isn't wired to import `src/forecasting/data_helpers.py`'s copy;
  migrating it is a deliberate follow-up, not done as part of this pass.
- **`5.1_Experiment_Tracking.ipynb`'s `get_champion_metrics` is still a
  draft, not a wired-in KFP component.** It's commented out
  (`# @dsl.component(...)`) and references `PROJECT_ID`/`REGION` as if
  they were notebook globals, which won't resolve inside an actual
  isolated component container. Turning it into a real
  `@dsl.component` with those passed as parameters (matching every other
  component in `6_Kubeflow_Pipelines.ipynb`) is the next step before it
  could be added to the pipeline's DAG.
