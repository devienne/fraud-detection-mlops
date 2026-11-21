"""Build a tiny model in the exact production artifact format, from synthetic transactions.

Used where the real model (a DVC output) is not available: the API tests, and the CI job that
builds and smoke-tests the Docker image. It tests the SERVICE, never the model's quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_detection.data import DEVICE_TYPES, MERCHANT_TYPES
from fraud_detection.features import CategoryLevels, make_features
from fraud_detection.policy import Policy

MERCHANT_TYPE = {"Clothing Store": "Retail", "Electronics Store": "Retail", "Gas Station": "Fuel",
                 "Online Shop 1": "E-commerce", "Online Shop 2": "E-commerce", "Pharmacy Z": "Pharmacy",
                 "Restaurant X": "Food & Beverage", "Restaurant Y": "Food & Beverage",
                 "Supermarket A": "Supermarket", "Supermarket B": "Supermarket"}
MERCHANTS = sorted(MERCHANT_TYPE)
CITIES = ["Belo Horizonte", "Curitiba", "Porto Alegre", "Recife", "Rio de Janeiro", "Salvador", "São Paulo"]
COLUMNS = ["hour", "tipo_estabelecimento", "nome_estabelecimento", "cidade", "device_type",
           "is_new_device", "valor_transacao"]
TEST_MODEL_VERSION = "test-0001"


def test_category_levels() -> CategoryLevels:
    return {"tipo_estabelecimento": sorted(MERCHANT_TYPES), "nome_estabelecimento": MERCHANTS,
            "cidade": sorted(CITIES), "device_type": sorted(DEVICE_TYPES)}


def synthetic_history(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Random transactions where fraud is likelier at night on new devices (a learnable toy signal)."""
    rng = np.random.default_rng(seed)
    merchants = rng.choice(MERCHANTS, n)
    df = pd.DataFrame({
        "datahora_transacao": pd.Timestamp("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 60 * 24 * 90, n), unit="min"),
        "nome_estabelecimento": merchants,
        "tipo_estabelecimento": [MERCHANT_TYPE[m] for m in merchants],
        "valor_transacao": rng.uniform(1, 300, n).round(2),
        "cidade": rng.choice(CITIES, n),
        "device_type": rng.choice(DEVICE_TYPES, n),
        "is_new_device": rng.random(n) < 0.3,
    })
    night = df["datahora_transacao"].dt.hour >= 22
    df["label"] = (rng.random(n) < np.where(night & df["is_new_device"], 0.8, 0.05)).astype(int)
    return df


def build_test_model(folder: Path) -> Path:
    """Write final_model.txt, metadata.json and policy.json (block 0.5, review 0.2) into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    levels = test_category_levels()
    hist = synthetic_history()
    model = lgb.LGBMClassifier(n_estimators=20, num_leaves=4, min_child_samples=5, verbose=-1)
    model.fit(make_features(hist, levels, COLUMNS), hist["label"])
    model.booster_.save_model(str(folder / "final_model.txt"))
    metadata = {
        "model_version": TEST_MODEL_VERSION, "algorithm": "LightGBM binary classifier",
        "lineage": {"git_commit": "test"},
        "features": {"columns": COLUMNS, "category_levels": levels, "timestamp_col": "datahora_transacao"},
        "evaluation": {"test_pr_auc": None},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    Policy(block_threshold=0.5, review_threshold=0.2, false_block_cost=10.0,
           review_recall_target=0.95, chosen_on="test").save(folder / "policy.json")
    return folder
