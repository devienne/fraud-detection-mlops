"""The ONE place where raw transaction columns become model inputs.

Training, batch scoring and the API all call `make_features`, so features can never be
computed two different ways (train/serve skew).

Category vocabulary
-------------------
LightGBM sees each category as a numeric code, and the code depends on the list of known
values. `fit_category_levels` learns that list from TRAINING data only; it is saved with the
model and passed back in at serving time. A value never seen in training (e.g. a new city)
becomes missing (NaN) instead of being silently mapped to another category's code.
"""

from __future__ import annotations

import pandas as pd

CategoryLevels = dict[str, list[str]]


def fit_category_levels(df: pd.DataFrame, categorical: list[str]) -> CategoryLevels:
    """Known values of each categorical column, sorted for a stable code order."""
    return {col: sorted(df[col].dropna().unique().tolist()) for col in categorical}


def _to_binary(series: pd.Series, name: str) -> pd.Series:
    """Accept booleans or 0/1 integers only; anything else (e.g. "yes") is an error, not a guess."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_integer_dtype(series) and series.isin([0, 1]).all():
        return series.astype(int)
    raise ValueError(f"{name} must be boolean or 0/1, got dtype {series.dtype}")


def make_features(
    df: pd.DataFrame,
    category_levels: CategoryLevels,
    columns: list[str],
    timestamp_col: str = "datahora_transacao",
) -> pd.DataFrame:
    """Turn raw transactions into the model's input matrix, columns in `columns` order.

    Expects the raw columns: the timestamp (local Brazil time, as in training), the categorical
    columns, `is_new_device` and `valor_transacao` (in reais).
    """
    timestamps = pd.to_datetime(df[timestamp_col])
    X = pd.DataFrame(index=df.index)
    X["hour"] = timestamps.dt.hour
    for col, levels in category_levels.items():
        # Fixed categories from training -> identical codes in every environment; unseen values -> NaN.
        X[col] = pd.Categorical(df[col], categories=levels)
    X["is_new_device"] = _to_binary(df["is_new_device"], "is_new_device")
    X["valor_transacao"] = df["valor_transacao"].astype(float)

    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise KeyError(f"make_features cannot build columns: {missing}")
    return X[columns]
