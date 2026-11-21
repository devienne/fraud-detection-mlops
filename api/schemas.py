"""Request/response contract of the fraud API.

This is the online twin of the training data contract (fraud_detection.data): requests that do
not match what the model was trained on are rejected at the boundary with a clear error, instead
of being silently scored wrong (train/serve skew).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from fraud_detection.data import DEVICE_TYPES, MERCHANT_TYPES

TRAINING_TIMEZONE = ZoneInfo("America/Sao_Paulo")   # training timestamps are local Brazil time
MAX_BATCH_SIZE = 1000

MerchantType = Literal[tuple(MERCHANT_TYPES)]
DeviceType = Literal[tuple(DEVICE_TYPES)]
RiskBand = Literal["High", "Medium", "Low"]
Decision = Literal["block", "review", "approve"]

EXAMPLE_TRANSACTION = {
    "transaction_id": "tx-000123",
    "datahora_transacao": "2026-01-15T22:14:00",
    "nome_estabelecimento": "Online Shop 1",
    "tipo_estabelecimento": "E-commerce",
    "valor_transacao": 32.90,
    "cidade": "São Paulo",
    "device_type": "mobile",
    "is_new_device": True,
}


class Transaction(BaseModel):
    """One card transaction to score. Unknown extra fields are ignored."""

    model_config = ConfigDict(extra="ignore", json_schema_extra={"examples": [EXAMPLE_TRANSACTION]})

    transaction_id: str = Field(min_length=1, max_length=64, description="Caller's unique id, echoed back for tracing.")
    datahora_transacao: datetime = Field(
        description="Transaction time. Naive timestamps are read as local Brazil time (as in training); "
                    "timestamps with a time zone (e.g. UTC '...Z') are converted to Brazil time.")
    nome_estabelecimento: str = Field(min_length=1, max_length=100, description="Merchant name; new merchants allowed.")
    tipo_estabelecimento: MerchantType = Field(description="Merchant type; must be one of the known types.")
    valor_transacao: float = Field(gt=0, le=100_000, strict=True, allow_inf_nan=False,
                                   description="Amount in REAIS (e.g. 32.90), as a JSON number - never cents, never text.")
    cidade: str = Field(min_length=1, max_length=100, description="City; new cities allowed (scored as unknown).")
    device_type: DeviceType = Field(description="Device used for the transaction.")
    is_new_device: StrictBool = Field(description="True/False only; 'yes', 1 or 'true' as text are rejected.")

    @field_validator("datahora_transacao")
    @classmethod
    def to_training_timezone(cls, ts: datetime) -> datetime:
        # Avoid the time-zone skew: 22h in Brazil must not be scored as 01h UTC.
        if ts.tzinfo is not None:
            ts = ts.astimezone(TRAINING_TIMEZONE).replace(tzinfo=None)
        return ts


class FraudScore(BaseModel):
    """Model output for one transaction."""

    transaction_id: str
    fraud_probability: float = Field(ge=0, le=1)
    risk_band: RiskBand
    decision: Decision
    model_version: str


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    model_version: str
    results: list[FraudScore]
