"""Shared test fixtures: small, hand-made (synthetic) transactions.

Code tests use these instead of the real data, so they run anywhere - including CI machines that
have no access to the DVC-stored datasets. Tests that need the real data/model are marked `model`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.config import load_params
from fraud_detection.features import CategoryLevels
from tests.model_fixture import test_category_levels


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def category_levels() -> CategoryLevels:
    """The category vocabulary as it would be learned from training data (sorted)."""
    return test_category_levels()


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """Six realistic raw transactions with every column of the data contract."""
    rows = [
        # time,               merchant,        type,             amount, city,            device,   new
        ("2025-03-10 03:15:00", "Gas Station",   "Fuel",            41.70, "São Paulo",      "mobile",  True),
        ("2025-03-10 12:40:00", "Restaurant X",  "Food & Beverage", 28.90, "Rio de Janeiro", "desktop", False),
        ("2025-03-10 20:05:00", "Online Shop 2", "E-commerce",      35.10, "Recife",         "tablet",  True),
        ("2025-03-10 22:30:00", "Pharmacy Z",    "Pharmacy",         9.99, "Curitiba",       "mobile",  False),
        ("2025-03-11 10:00:00", "Supermarket A", "Supermarket",    152.30, "Salvador",       "desktop", False),
        ("2025-03-11 23:59:00", "Clothing Store", "Retail",         64.00, "Porto Alegre",   "mobile",  True),
    ]
    df = pd.DataFrame(rows, columns=["datahora_transacao", "nome_estabelecimento", "tipo_estabelecimento",
                                     "valor_transacao", "cidade", "device_type", "is_new_device"])
    df["datahora_transacao"] = pd.to_datetime(df["datahora_transacao"])
    # Remaining contract columns (not used by the model, but part of the raw files)
    df["nome_cliente"] = [f"Cliente_{i}" for i in range(len(df))]
    df["temperatura_celsius"] = [21.5, 28.0, 30.1, 17.2, 29.9, 15.0]
    df["codigo_postal"] = ["01310-100", "20040-002", "50010-000", "80010-000", "40010-000", "90010-000"]
    df["score_interno"] = [120, 540, 999, 0, 300, 777]
    df["turno"] = ["madrugada", "tarde", "noite", "noite", "manha", "noite"]
    df["valor_categoria"] = ["medio", "baixo", "medio", "baixo", "alto", "medio"]
    df["cidade_regiao"] = ["Sudeste", "Sudeste", "Nordeste", "Sul", "Nordeste", "Sul"]
    df["flag_fraude"] = [True, False, False, False, False, False]   # 1/6 = 16.7%, inside the contract's 0.1%-20%
    return df
