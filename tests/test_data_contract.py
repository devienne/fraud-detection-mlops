"""Data contract tests.

1. The training contract (pandera, fraud_detection.data) accepts good tables and catches each kind
   of bad data - so the contract itself can never silently weaken.
2. Consistency: the training contract and the API contract (Pydantic, api.schemas) must accept and
   reject exactly the same values for the fields they share. If someone edits one and forgets the
   other, this test fails (a mismatch would let the API score inputs the model never saw).
"""

from __future__ import annotations

import pandas as pd
import pandera.errors as pe
import pytest
from pydantic import ValidationError

from api.schemas import Transaction
from fraud_detection.data import SCORING_SCHEMA, history_schema, load_history

TARGET = "flag_fraude"


def failed_columns(err: pe.SchemaErrors) -> set[str]:
    return set(err.failure_cases["column"].dropna())


# --------------------------------------------------------------------------- training contract
def test_valid_table_passes(raw_transactions):
    history_schema(TARGET).validate(raw_transactions, lazy=True)
    SCORING_SCHEMA.validate(raw_transactions.drop(columns=TARGET), lazy=True)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("valor_transacao", -10.0),                 # impossible amount
        ("valor_transacao", 250_000.0),             # above the maximum
        ("tipo_estabelecimento", "Casino"),         # unknown business category
        ("device_type", "smartwatch"),              # unknown device
        ("codigo_postal", "ABC"),                   # malformed postal code
        ("score_interno", 5000),                    # out of range
        ("tipo_estabelecimento", None),             # missing required value
    ],
)
def test_each_violation_is_caught(raw_transactions, column, bad_value):
    df = raw_transactions.copy()
    df[column] = df[column].astype(object)
    df.loc[2, column] = bad_value
    with pytest.raises(pe.SchemaErrors) as err:
        history_schema(TARGET).validate(df, lazy=True)
    assert column in failed_columns(err.value)


def test_missing_column_is_caught(raw_transactions):
    with pytest.raises(pe.SchemaErrors) as err:
        history_schema(TARGET).validate(raw_transactions.drop(columns="device_type"), lazy=True)
    assert "device_type" in err.value.failure_cases["failure_case"].astype(str).tolist()


def test_all_violations_are_reported_at_once(raw_transactions):
    df = raw_transactions.copy()
    df.loc[0, "valor_transacao"] = -1.0
    df.loc[1, "device_type"] = "smartwatch"
    df.loc[2, "codigo_postal"] = "ABC"
    with pytest.raises(pe.SchemaErrors) as err:
        history_schema(TARGET).validate(df, lazy=True)
    assert {"valor_transacao", "device_type", "codigo_postal"} <= failed_columns(err.value)


@pytest.mark.parametrize("n_frauds", [0, 3])        # 0% and 50%: both look like a broken extract
def test_implausible_fraud_rate_is_rejected(raw_transactions, n_frauds):
    df = raw_transactions.copy()
    df[TARGET] = [True] * n_frauds + [False] * (len(df) - n_frauds)
    with pytest.raises(pe.SchemaErrors, match="fraud rate"):
        history_schema(TARGET).validate(df, lazy=True)


def test_loader_validates_and_returns_integer_label(raw_transactions, params, tmp_path):
    csv = tmp_path / "history.csv"
    raw_transactions.to_csv(csv, index=False)
    test_params = params.model_copy(update={"data": params.data.model_copy(update={"history_path": csv})})
    df = load_history(test_params)
    assert df[TARGET].tolist() == [1, 0, 0, 0, 0, 0]


# --------------------------------------------------------------------------- consistency
SHARED_FIELD_CASES = [
    ("valor_transacao", 30.5), ("valor_transacao", 100_000.0), ("valor_transacao", 0.0),
    ("valor_transacao", -5.0), ("valor_transacao", 100_000.01),
    ("tipo_estabelecimento", "Fuel"), ("tipo_estabelecimento", "E-commerce"), ("tipo_estabelecimento", "Casino"),
    ("device_type", "tablet"), ("device_type", "smartwatch"),
    ("is_new_device", True), ("is_new_device", False), ("is_new_device", "yes"), ("is_new_device", 1),
]


def training_contract_accepts(row: pd.DataFrame) -> bool:
    try:
        SCORING_SCHEMA.validate(row, lazy=True)
        return True
    except pe.SchemaErrors:
        return False


def _plain(value):
    """numpy scalars -> plain Python values, as they would arrive from JSON."""
    return value.item() if hasattr(value, "item") else value


def api_contract_accepts(row: pd.DataFrame) -> bool:
    r = {k: _plain(v) for k, v in row.iloc[0].items()}
    payload = {
        "transaction_id": "t", "datahora_transacao": r["datahora_transacao"].isoformat(),
        "nome_estabelecimento": r["nome_estabelecimento"], "tipo_estabelecimento": r["tipo_estabelecimento"],
        "valor_transacao": r["valor_transacao"], "cidade": r["cidade"], "device_type": r["device_type"],
        "is_new_device": r["is_new_device"],
    }
    try:
        Transaction.model_validate(payload)
        return True
    except ValidationError:
        return False


@pytest.mark.parametrize("field, value", SHARED_FIELD_CASES)
def test_training_and_api_contracts_agree(raw_transactions, field, value):
    row = raw_transactions.drop(columns=TARGET).iloc[[0]].reset_index(drop=True)
    row[field] = pd.Series([value], dtype=object if isinstance(value, str) else None)
    training, api = training_contract_accepts(row), api_contract_accepts(row)
    assert training == api, f"{field}={value!r}: training contract accepts={training}, API contract accepts={api}"
