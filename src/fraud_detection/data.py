"""Load raw transaction files and enforce the data contract.

Every load goes through a pandera schema, so a broken upstream file (missing column,
wrong type, impossible value) stops the pipeline instead of silently training a bad model.

Run as a pipeline stage:  python -m fraud_detection.data
"""

from __future__ import annotations

import json

import pandas as pd
import pandera.pandas as pa

from fraud_detection.config import REPORTS_DIR, Params, load_params

MERCHANT_TYPES = ["E-commerce", "Food & Beverage", "Fuel", "Pharmacy", "Retail", "Supermarket"]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
SHIFTS = ["madrugada", "manha", "tarde", "noite"]
AMOUNT_BUCKETS = ["baixo", "medio", "alto"]

# Transaction amount in reais: strictly above AMOUNT_MIN, at most AMOUNT_MAX.
# Shared with the API contract (api/schemas.py) so training and serving accept the same amounts.
AMOUNT_MIN = 0.0
AMOUNT_MAX = 100_000.0

# Contract shared by the labelled history and the unlabelled scoring file.
# Business enums (merchant type, device) are strict; free-text dimensions (merchant name, city)
# only need to be present, because new merchants and cities legitimately appear in production.
_BASE_COLUMNS = {
    "datahora_transacao": pa.Column("datetime64[ns]", nullable=False),
    "nome_estabelecimento": pa.Column(str, nullable=False),
    "tipo_estabelecimento": pa.Column(str, pa.Check.isin(MERCHANT_TYPES), nullable=False),
    "valor_transacao": pa.Column(float, [pa.Check.gt(AMOUNT_MIN), pa.Check.le(AMOUNT_MAX)], nullable=False),
    "nome_cliente": pa.Column(str, nullable=False),
    "cidade": pa.Column(str, nullable=False),
    "device_type": pa.Column(str, pa.Check.isin(DEVICE_TYPES), nullable=False),
    "is_new_device": pa.Column(bool, nullable=False),
    "temperatura_celsius": pa.Column(float, pa.Check.in_range(-50, 60), nullable=True),
    "codigo_postal": pa.Column(str, pa.Check.str_matches(r"^\d{5}-\d{3}$"), nullable=True),
    "score_interno": pa.Column(int, pa.Check.in_range(0, 1000), nullable=True),
    "turno": pa.Column(str, pa.Check.isin(SHIFTS), nullable=True),
    "valor_categoria": pa.Column(str, pa.Check.isin(AMOUNT_BUCKETS), nullable=True),
    "cidade_regiao": pa.Column(str, nullable=True),
}

SCORING_SCHEMA = pa.DataFrameSchema(_BASE_COLUMNS, strict=False, coerce=False, name="scoring")


def history_schema(target: str) -> pa.DataFrameSchema:
    """Scoring contract plus the label, with a dataset-level sanity check on the fraud rate."""
    return pa.DataFrameSchema(
        {**_BASE_COLUMNS, target: pa.Column(bool, nullable=False)},
        checks=[
            # A label rate outside this range means a broken extract, not a real change in fraud.
            pa.Check(lambda df: 0.001 <= df[target].mean() <= 0.20, error="fraud rate outside 0.1%-20%"),
        ],
        strict=False, coerce=False, name="history",
    )


def _read(path, timestamp_col: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[timestamp_col])


def load_history(params: Params | None = None) -> pd.DataFrame:
    """Labelled history, validated. The target is returned as 0/1 integers."""
    params = params or load_params()
    df = _read(params.path(params.data.history_path), params.data.timestamp_col)
    df = history_schema(params.data.target).validate(df, lazy=True)   # lazy: report ALL violations at once
    df[params.data.target] = df[params.data.target].astype(int)
    return df


def load_scoring(params: Params | None = None) -> pd.DataFrame:
    """Unlabelled transactions to score, validated."""
    params = params or load_params()
    df = _read(params.path(params.data.scoring_path), params.data.timestamp_col)
    return SCORING_SCHEMA.validate(df, lazy=True)


def _summary(df: pd.DataFrame, timestamp_col: str, target: str | None = None) -> dict:
    out = {
        "rows": int(len(df)),
        "first_timestamp": str(df[timestamp_col].min()),
        "last_timestamp": str(df[timestamp_col].max()),
        "missing_values": int(df.isna().sum().sum()),
    }
    if target:
        out["fraud_rate"] = round(float(df[target].mean()), 6)
    return out


def main() -> None:
    params = load_params()
    history, scoring = load_history(params), load_scoring(params)   # raises SchemaErrors on any violation
    report = {
        "status": "passed",
        "history": _summary(history, params.data.timestamp_col, params.data.target),
        "scoring": _summary(scoring, params.data.timestamp_col),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
