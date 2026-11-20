"""The decision policy: how a fraud probability becomes an action.

Kept apart from the model on purpose: thresholds can be reviewed or changed (config, not code),
and every retrained model must come with a re-checked policy. Batch scoring and the API both
use `Policy.band` / `Policy.decide`, so decisions are identical everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

BANDS = ("High", "Medium", "Low")
ACTIONS = {"High": "block", "Medium": "review", "Low": "approve"}


class Policy(BaseModel):
    block_threshold: float = Field(gt=0, lt=1)    # p >= this -> High -> block / step-up authentication
    review_threshold: float = Field(gt=0, lt=1)   # review <= p < block -> Medium -> manual review
    false_block_cost: float
    review_recall_target: float
    chosen_on: str                                # which data the thresholds were selected on

    @model_validator(mode="after")
    def review_below_block(self) -> Policy:
        if self.review_threshold > self.block_threshold:
            raise ValueError("review_threshold must not exceed block_threshold")
        return self

    def band(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        return np.select([probs >= self.block_threshold, probs >= self.review_threshold],
                         ["High", "Medium"], default="Low")

    def decide(self, probs: np.ndarray) -> np.ndarray:
        """Binary fraud classification: True when the transaction is blocked."""
        return np.asarray(probs, dtype=float) >= self.block_threshold

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Policy:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
