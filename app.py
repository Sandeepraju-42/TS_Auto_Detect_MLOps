"""
Step 7b: Flask serving app -- the other half of "add model to registry
(Flask API?)". Doesn't read a model file off disk by default; it loads
whichever version register_model.py aliased "champion" in the DagsHub MLflow
Model Registry, plus that exact run's feature_schema.json (feature column
order + categorical category lists) so incoming requests get encoded
identically to how the model was trained -- see 5_tuned_vs_baseline_segments.py's
log_model_run() docstring for why that has to match exactly, not just
approximately.

Endpoints:
  GET  /health           -- liveness + which model version is loaded
  POST /predict           -- score one or more feature rows

Falls back to the local Output/tuned_lightgbm.txt (+ a locally-saved schema)
if the registry can't be reached -- e.g. no network, or DagsHub credentials
aren't configured in this environment -- so the API still comes up for local
dev/testing rather than refusing to start.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "fashion-demand-forecaster"
CHAMPION_ALIAS = "champion"
DAGSHUB_REPO_OWNER = "Sandeepraju-42"
DAGSHUB_REPO_NAME = "Forecast_MLFlow_Fashion_Dataset"

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = PROJECT_DIR / "Output" / "tuned_lightgbm.txt"
LOCAL_SCHEMA_PATH = PROJECT_DIR / "feature_schema_tuned.json"

app = Flask(__name__)

_model = None            # lgb.Booster
_schema = None           # {"feature_cols": [...], "categorical_categories": {...}}
_model_source = None     # "registry:v<N>" or "local_file", for /health


def _load_from_registry():
    import mlflow
    import mlflow.lightgbm
    import dagshub
    from mlflow.tracking import MlflowClient

    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    client = MlflowClient()
    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    model = mlflow.lightgbm.load_model(model_uri)

    # Same run the champion model came from, so the schema matches exactly
    # what that specific model version was trained against -- not whatever
    # the *latest* training run happened to produce.
    schema_local_dir = mlflow.artifacts.download_artifacts(
        run_id=champion.run_id, artifact_path="schema"
    )
    with open(Path(schema_local_dir) / "feature_schema_tuned.json") as f:
        schema = json.load(f)

    return model, schema, f"registry:v{champion.version}"


def _load_from_local_file():
    import lightgbm as lgb

    model = lgb.Booster(model_file=str(LOCAL_MODEL_PATH))
    with open(LOCAL_SCHEMA_PATH) as f:
        schema = json.load(f)
    return model, schema, "local_file"


def load_model():
    """
    Called once at startup (see __main__) and cached in the _model/_schema
    globals -- registry lookups and artifact downloads are network calls,
    not something to repeat on every request.
    """
    global _model, _schema, _model_source
    try:
        _model, _schema, _model_source = _load_from_registry()
        logger.info(f"loaded model from registry ({_model_source})")
    except Exception as e:
        logger.warning(f"registry load failed ({e!r}), falling back to local file")
        _model, _schema, _model_source = _load_from_local_file()
        logger.info(f"loaded model from local file ({_model_source})")


def encode_request_rows(rows: list[dict]) -> pd.DataFrame:
    """
    Builds a DataFrame matching the model's training-time schema exactly:
    same column set and order (missing fields become NaN -- LightGBM splits
    on missing natively, same as training), and every categorical column
    cast to the SAME pd.CategoricalDtype (same categories, same order) the
    model was trained with, not one rebuilt from whatever values happen to
    show up in this request.
    """
    df = pd.DataFrame(rows)
    for col in _schema["feature_cols"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[_schema["feature_cols"]]

    for col, categories in _schema["categorical_categories"].items():
        cat_dtype = pd.CategoricalDtype(categories=categories)
        df[col] = df[col].astype(str).astype(cat_dtype)

    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if _model is not None else "model not loaded",
        "model_source": _model_source,
        "registered_model": REGISTERED_MODEL_NAME,
        "n_features": len(_schema["feature_cols"]) if _schema else None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Body: {"rows": [{...feature dict...}, ...]}  -- one dict per SKU-week to
    forecast, using the same feature names Features.py produces (lag/rolling
    columns, calendar features, abc_class/xyz_class, etc.). Unknown/missing
    fields are fine (treated as missing); the caller is responsible for
    having already computed the lag/rolling features (this endpoint scores
    a feature row, it doesn't rebuild feature engineering from raw history).
    """
    if _model is None:
        return jsonify({"error": "model not loaded"}), 503

    payload = request.get_json(silent=True)
    if not payload or "rows" not in payload:
        return jsonify({"error": "expected JSON body {'rows': [...]}"}), 400
    if not isinstance(payload["rows"], list) or len(payload["rows"]) == 0:
        return jsonify({"error": "'rows' must be a non-empty list"}), 400

    try:
        df = encode_request_rows(payload["rows"])
    except Exception as e:
        return jsonify({"error": f"could not encode input rows: {e!r}"}), 400

    # Model trains on log1p(sales_qty); invert with expm1 to get back to
    # actual units, same as every training/eval script does.
    try:
        preds = np.expm1(_model.predict(df))
    except Exception as e:
        logger.exception("prediction failed")
        return jsonify({"error": f"prediction failed: {e!r}"}), 500
    return jsonify({"predictions": preds.tolist(), "model_source": _model_source})


# Called at IMPORT time, not just inside the __main__ guard below -- a
# production server (gunicorn app:app, per the Dockerfile) imports this
# module rather than executing it as __main__, so load_model() would never
# run and every request would 503 forever if this were only in __main__.
# `python app.py` for local dev hits this exact same call, not a separate
# one, so there's no risk of the two entry points loading the model
# differently.
load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
