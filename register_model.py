"""
Step 7a: promote a trained model into DagsHub's MLflow Model Registry.

This is the "registry" half of the plan's step 7 ("add model to registry
(Flask API?)") -- app.py (the Flask half) doesn't load a model file off
disk, it loads whichever model this script has aliased "champion" in the
registry. That split matters operationally: redeploying a better model
later is "run this script again," not "rebuild and redeploy the API."

What it does:
  1. Finds the most recent run tagged stage="tuned" in the
     fashion-demand-forecast experiment (written by
     5_tuned_vs_baseline_segments.py's log_model_run()).
  2. Registers that run's logged model under one registered model name
     (REGISTERED_MODEL_NAME), creating a new version.
  3. Compares its test_wape against whatever version currently holds the
     "champion" alias, if any, and only moves the alias if the new one is
     actually better -- registering a worse model shouldn't silently
     replace a better one in production. Pass --force to override.

Uses MLflow's alias API (set_registered_model_alias), not the older
Staging/Production "stage" transitions -- stages are deprecated as of
MLflow 2.9+; aliases (arbitrary tags like "champion") are the current
mechanism and read more clearly here anyway ("give me @champion") than a
fixed Staging/Production/Archived vocabulary that doesn't really fit a
single-model registry like this one.
"""

import argparse

import mlflow
import dagshub
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "fashion-demand-forecast"
REGISTERED_MODEL_NAME = "fashion-demand-forecaster"
CHAMPION_ALIAS = "champion"
DAGSHUB_REPO_OWNER = "Sandeepraju-42"
DAGSHUB_REPO_NAME = "Forecast_MLFlow_Fashion_Dataset"


def find_latest_tuned_run(client: MlflowClient) -> mlflow.entities.Run:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"experiment '{EXPERIMENT_NAME}' not found -- run "
            "5_tuned_vs_baseline_segments.py at least once first"
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'tuned'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            "no run tagged stage='tuned' found -- run "
            "5_tuned_vs_baseline_segments.py first"
        )
    return runs[0]


def get_champion_test_wape(client: MlflowClient) -> float | None:
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        return None  # no champion yet, or model doesn't exist yet
    run = client.get_run(champion.run_id)
    return run.data.metrics.get("test_wape")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="promote to champion even if the candidate's test_wape isn't better",
    )
    args = parser.parse_args()

    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    client = MlflowClient()

    candidate_run = find_latest_tuned_run(client)
    candidate_wape = candidate_run.data.metrics.get("test_wape")
    print(f"candidate run: {candidate_run.info.run_id}  test_wape={candidate_wape:.4f}")

    model_uri = f"runs:/{candidate_run.info.run_id}/model"
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    print(f"registered as {REGISTERED_MODEL_NAME} v{version.version}")

    champion_wape = get_champion_test_wape(client)
    if champion_wape is None:
        print("no existing champion -- promoting unconditionally")
        should_promote = True
    else:
        should_promote = candidate_wape < champion_wape
        print(f"current champion test_wape={champion_wape:.4f}")
        if not should_promote and not args.force:
            print(
                f"candidate ({candidate_wape:.4f}) is not better than champion "
                f"({champion_wape:.4f}) -- NOT promoting. Registered as a new "
                f"version anyway (v{version.version}); re-run with --force to "
                f"promote it regardless."
            )

    if should_promote or args.force:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
        print(f"promoted v{version.version} to @{CHAMPION_ALIAS}")

    print(f"\napp.py loads: models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")


if __name__ == "__main__":
    main()
