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
        if max(self.train_months) >= min(self.validation_months) or max(self.validation_months) >= min(self.test_months):
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
    mlflow: MlflowConfig

    def path(self, relative: Path) -> Path:
        return relative if relative.is_absolute() else PROJECT_ROOT / relative


@lru_cache
def load_params(path: Path = PARAMS_PATH) -> Params:
    """Read params.yaml and validate it. Fails fast with a clear message on a bad setting."""
    with open(path, encoding="utf-8") as f:
        return Params.model_validate(yaml.safe_load(f))
