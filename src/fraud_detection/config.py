"""Load and validate params.yaml, plus the project's fixed paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

EVAL_MODEL_PATH = MODELS_DIR / "eval_model.txt"      # trained on train months only, for honest evaluation
FINAL_MODEL_PATH = MODELS_DIR / "final_model.txt"    # trained on all labelled data, used for scoring
TRAINING_INFO_PATH = MODELS_DIR / "training_info.json"
POLICY_PATH = MODELS_DIR / "policy.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
REFERENCE_PATH = MODELS_DIR / "reference.json"        # training-time distributions + performance baseline

# Live operation (not versioned, git-ignored): the API's prediction log and the scheduled monitor's outputs
LOGS_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG_PATH = LOGS_DIR / "predictions.jsonl"                 # written by the API
MONITORING_REPORT_PATH = LOGS_DIR / "monitoring" / "monitoring.json"
HEARTBEAT_PATH = LOGS_DIR / "monitoring" / "heartbeat.json"


class DataConfig(BaseModel):
    history_path: Path
    scoring_path: Path
    timestamp_col: str
    target: str


class FeatureConfig(BaseModel):
    columns: list[str]        # ordered model inputs
    categorical: list[str]

    @model_validator(mode="after")
    def categorical_are_columns(self) -> FeatureConfig:
        missing = set(self.categorical) - set(self.columns)
        if missing:
            raise ValueError(f"categorical features not in columns: {sorted(missing)}")
        return self


class SplitConfig(BaseModel):
    train_months: list[int]
    validation_months: list[int]
    test_months: list[int]

    @model_validator(mode="after")
    def months_do_not_overlap(self) -> SplitConfig:
        sets = [set(self.train_months), set(self.validation_months), set(self.test_months)]
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
            raise ValueError("train, validation and test months must not overlap")
        in_order = (max(self.train_months) < min(self.validation_months)
                    and max(self.validation_months) < min(self.test_months))
        if not in_order:
            raise ValueError("splits must be in time order: train < validation < test")
        return self


class ModelConfig(BaseModel):
    learning_rate: float = Field(gt=0, le=1)
    num_leaves: int = Field(ge=2)
    min_child_samples: int = Field(ge=1)
    subsample: float = Field(gt=0, le=1)
    subsample_freq: int = Field(ge=0)
    colsample_bytree: float = Field(gt=0, le=1)
    reg_lambda: float = Field(ge=0)
    random_state: int
    max_estimators: int = Field(ge=1)
    early_stopping_rounds: int = Field(ge=1)
    final_trees_multiplier: float = Field(ge=1)
    recency_months: list[int]
    recency_weight: float = Field(ge=1)

    def lgbm_params(self) -> dict:
        """Keyword arguments for lightgbm.LGBMClassifier (without n_estimators)."""
        return dict(
            learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples, subsample=self.subsample,
            subsample_freq=self.subsample_freq, colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda, random_state=self.random_state, verbose=-1,
        )


class ThresholdGrid(BaseModel):
    start: float
    stop: float
    step: float


class PolicyConfig(BaseModel):
    false_block_cost: float = Field(gt=0)
    threshold_grid: ThresholdGrid
    review_recall_target: float = Field(gt=0, lt=1)


class QualityGateConfig(BaseModel):
    min_test_pr_auc: float = Field(gt=0, lt=1)
    min_lift_over_rule: float = Field(ge=1)
    min_recall_at_block: float = Field(gt=0, lt=1)
    min_precision_at_block: float = Field(gt=0, lt=1)


class MonitoringConfig(BaseModel):
    numeric_bins: int = Field(ge=2, le=100)
    top_categories: int = Field(ge=1)
    psi_warn: float = Field(gt=0)
    psi_alert: float = Field(gt=0)
    score_percentiles: list[float]
    segments: list[str] = Field(min_length=1)
    label_maturity_days: int = Field(ge=0)
    min_segment_frauds: int = Field(ge=1)
    max_pr_auc_drop: float = Field(gt=0, lt=1)
    max_recall_drop: float = Field(gt=0, lt=1)
    max_fraud_share_shift: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def warn_below_alert(self) -> MonitoringConfig:
        if self.psi_warn >= self.psi_alert:
            raise ValueError("psi_warn must be lower than psi_alert")
        if not all(0 < p < 1 for p in self.score_percentiles):
            raise ValueError("score_percentiles must be between 0 and 1")
        return self


class MlflowConfig(BaseModel):
    tracking_uri: str
    experiment: str
    registered_model: str

    @field_validator("tracking_uri")
    @classmethod
    def anchor_sqlite_to_project(cls, uri: str) -> str:
        # A relative sqlite path would depend on the working directory; pin it to the project root.
        prefix = "sqlite:///"
        if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
            return prefix + (PROJECT_ROOT / uri[len(prefix):]).as_posix()
        return uri


class Params(BaseModel):
    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    model: ModelConfig
    policy: PolicyConfig
    quality_gate: QualityGateConfig
    monitoring: MonitoringConfig
    mlflow: MlflowConfig

    @model_validator(mode="after")
    def segments_are_features(self) -> Params:
        missing = set(self.monitoring.segments) - set(self.features.columns)
        if missing:
            raise ValueError(f"monitoring segments not in features.columns: {sorted(missing)}")
        return self

    def path(self, relative: Path) -> Path:
        return relative if relative.is_absolute() else PROJECT_ROOT / relative


@lru_cache
def load_params(path: Path = PARAMS_PATH) -> Params:
    """Read params.yaml and validate it. Fails fast with a clear message on a bad setting."""
    with open(path, encoding="utf-8") as f:
        return Params.model_validate(yaml.safe_load(f))
