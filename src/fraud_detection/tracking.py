"""Experiment tracking and model registry with MLflow.

One MLflow run per pipeline execution: parameters, metrics, plots/reports, lineage tags, and the
final model registered as a new version of `params.mlflow.registered_model`.
Storage is local: run history in a SQLite file (mlflow.db), files in mlruns/ (both git-ignored).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.models import infer_signature

from fraud_detection.config import (
    FINAL_MODEL_PATH,
    METADATA_PATH,
    POLICY_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    Params,
)


def artifact_root(tracking_uri: str) -> Path:
    """Artifacts live next to the SQLite tracking database (<project>/mlruns for the real one)."""
    prefix = "sqlite:///"
    if tracking_uri.startswith(prefix):
        return Path(tracking_uri[len(prefix):]).parent / "mlruns"
    return PROJECT_ROOT / "mlruns"


def _experiment_id(params: Params) -> str:
    """Create the experiment on first use, with its artifacts anchored next to the tracking database."""
    mlflow.set_tracking_uri(params.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(params.mlflow.experiment)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(params.mlflow.experiment,
                                    artifact_location=artifact_root(params.mlflow.tracking_uri).as_uri())


def log_run(params: Params, metadata: dict, metrics: dict, feature_sample: pd.DataFrame) -> dict:
    """Log one run and register the final model. Returns the run id and registered version."""
    experiment_id = _experiment_id(params)
    test = metrics["test"]
    with mlflow.start_run(experiment_id=experiment_id, run_name=metadata["model_version"]) as run:
        mlflow.set_tags({
            "model_version": metadata["model_version"],
            "git_commit": metadata["lineage"]["git_commit"],
            "data_md5": metadata["lineage"]["data_md5"],
        })
        mlflow.log_params({
            **params.model.lgbm_params(),
            "n_trees_early_stopping": metadata["training"]["n_trees_early_stopping"],
            "n_trees_final": metadata["training"]["n_trees"],
            "recency_weight": params.model.recency_weight,
            "recency_months": params.model.recency_months,
            "false_block_cost": params.policy.false_block_cost,
            "review_recall_target": params.policy.review_recall_target,
            "train_months": params.split.train_months,
            "validation_months": params.split.validation_months,
            "test_months": params.split.test_months,
        })
        mlflow.log_metrics({
            "val_pr_auc": metrics["validation"]["pr_auc"],
            "val_roc_auc": metrics["validation"]["roc_auc"],
            "test_pr_auc": test["pr_auc"],
            "test_roc_auc": test["roc_auc"],
            "test_recall_at_block": test["recall"],
            "test_precision_at_block": test["precision"],
            "test_block_rate": test["block_rate"],
            "test_net_benefit_brl": test["net_benefit"],
            "baseline_rule_test_pr_auc": metrics["baseline_rule"]["test_pr_auc"],
            "block_threshold": metrics["policy"]["block_threshold"],
            "review_threshold": metrics["policy"]["review_threshold"],
        })
        for path in (REPORTS_DIR / "metrics.json", REPORTS_DIR / "pr_curve.png", POLICY_PATH, METADATA_PATH):
            mlflow.log_artifact(str(path))

        # Final model in MLflow's format, with an input/output signature, registered as a new version.
        booster = lgb.Booster(model_file=str(FINAL_MODEL_PATH))
        signature = infer_signature(feature_sample, booster.predict(feature_sample))
        model_info = mlflow.lightgbm.log_model(
            booster, name="model", signature=signature,
            registered_model_name=params.mlflow.registered_model,
        )
    return {"run_id": run.info.run_id, "registered_model": params.mlflow.registered_model,
            "registered_version": model_info.registered_model_version}
