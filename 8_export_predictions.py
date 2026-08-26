"""
Step 8: consumption layer for Power BI.

Two ways Power BI can consume this model's output, and this script covers
the simpler/more reliable one:

  1. Static export (what this script does): score the test set with the
     champion model, write one flat CSV with actuals + predictions +
     segment columns. Power BI: Get Data -> Text/CSV -> point at
     predictions_for_powerbi.csv -> build visuals (WAPE by category, actual
     vs. predicted over time, error by ABC/XYZ segment, etc.) straight from
     the columns.

  2. Live scoring via the API: Power Query's Web.Contents can POST to
     app.py's /predict endpoint and pull back predictions on refresh. That
     needs the API actually reachable when Power BI refreshes (i.e. step 11
     -- cloud-deployed, not just running on your laptop), so it's the
     better fit later, not now. The static CSV here works immediately with
     no deployment dependency.

Re-run this after every retraining (or wire it as a 6th DVC stage once
you're comfortable with the pipeline) to keep the export current.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import app as serving_app  # reuses app.py's model-loading + encoding logic
                            # so the export scores with EXACTLY the same
                            # code path the API uses -- no risk of the export
                            # and the live API silently disagreeing.

OUTPUT_PATH = Path(__file__).resolve().parent / "predictions_for_powerbi.csv"
# sub_category isn't included here even though it'd be a natural Power BI
# slicer: Features.py's encode_categoricals() one-hot-collapses it (no
# category_copy-style plain column kept, unlike `category`), so the raw
# sub_category string doesn't survive into train/val/test_features.csv. Add
# a sub_category_copy column there (same pattern as category_copy) if you
# want it in this export.
IDENTIFYING_COLS = ["sku_id", "date", "category", "abc_class",
                     "xyz_class", "promo_flag"]


def main():
    # app.py now loads its model at import time (not just inside its
    # __main__ guard, so gunicorn picking it up as a module still works) --
    # the `import app as serving_app` above already triggered that load, so
    # calling load_model() again here would just be a redundant reload.
    print(f"scoring with model from: {serving_app._model_source}")

    test = pd.read_csv(
        serving_app.PROJECT_DIR / "input" / "test_features.csv", parse_dates=["date"]
    )

    feature_rows = test.to_dict(orient="records")
    encoded = serving_app.encode_request_rows(feature_rows)
    predicted = np.expm1(serving_app._model.predict(encoded))

    out = test[[c for c in IDENTIFYING_COLS if c in test.columns]].copy()
    out["actual"] = test["sales_qty"]
    out["predicted"] = np.round(predicted, 1)
    out["error"] = out["predicted"] - out["actual"]
    # abs_pct_error left as NaN on zero-actual rows rather than a divide-by-
    # zero inf -- Power BI's SUM(abs_error)/SUM(actual) for a segment-level
    # WAPE handles zero-actual rows correctly on its own; a per-row inf here
    # would just poison any visual that averages this column directly.
    out["abs_pct_error"] = np.where(
        out["actual"] > 0, (out["error"].abs() / out["actual"]), np.nan
    )

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH.name}: {len(out)} rows")
    print(f"overall test WAPE from this export: "
          f"{out['error'].abs().sum() / out['actual'].sum():.4f}")


if __name__ == "__main__":
    main()
