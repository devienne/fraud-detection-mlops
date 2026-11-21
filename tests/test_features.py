"""Feature parity tests: the guard against train/serve skew.

If anyone changes the feature code, the category handling or the API input path so that training
and serving would see different inputs, these tests fail.
"""

from __future__ import annotations

import pandas as pd
import pytest

from api.schemas import Transaction
from fraud_detection.features import fit_category_levels, make_features


def serving_path(raw: pd.DataFrame, category_levels, columns) -> pd.DataFrame:
    """Rebuild the features the way the API does: JSON -> Transaction -> table -> make_features."""
    requests = []
    for i, row in raw.reset_index(drop=True).iterrows():
        payload = {
            "transaction_id": f"tx-{i}",
            "datahora_transacao": row["datahora_transacao"].isoformat(),
            "nome_estabelecimento": row["nome_estabelecimento"],
            "tipo_estabelecimento": row["tipo_estabelecimento"],
            "valor_transacao": float(row["valor_transacao"]),
            "cidade": row["cidade"],
            "device_type": row["device_type"],
            "is_new_device": bool(row["is_new_device"]),
        }
        requests.append(Transaction.model_validate(payload).model_dump())
    return make_features(pd.DataFrame(requests), category_levels, columns)


def test_training_and_serving_paths_give_identical_features(raw_transactions, category_levels, params):
    training = make_features(raw_transactions, category_levels, params.features.columns).reset_index(drop=True)
    serving = serving_path(raw_transactions, category_levels, params.features.columns)
    pd.testing.assert_frame_equal(training, serving)


def test_columns_follow_params_order(raw_transactions, category_levels, params):
    X = make_features(raw_transactions, category_levels, params.features.columns)
    assert list(X.columns) == params.features.columns


def test_category_codes_do_not_depend_on_the_batch(raw_transactions, category_levels, params):
    batch = make_features(raw_transactions, category_levels, params.features.columns)
    for i in range(len(raw_transactions)):
        alone = make_features(raw_transactions.iloc[[i]], category_levels, params.features.columns)
        for col in category_levels:
            assert alone[col].cat.codes.iloc[0] == batch[col].cat.codes.iloc[i], f"{col} code changed for row {i}"


def test_fuel_always_gets_its_training_code(raw_transactions, category_levels, params):
    fuel_only = raw_transactions[raw_transactions["tipo_estabelecimento"] == "Fuel"]
    X = make_features(fuel_only, category_levels, params.features.columns)
    assert X["tipo_estabelecimento"].cat.codes.iloc[0] == category_levels["tipo_estabelecimento"].index("Fuel")


def test_unseen_category_becomes_missing_not_another_category(raw_transactions, category_levels, params):
    raw = raw_transactions.copy()
    raw.loc[0, "cidade"] = "Manaus"            # never seen in training
    X = make_features(raw, category_levels, params.features.columns)
    assert pd.isna(X.loc[0, "cidade"])
    assert X.loc[1, "cidade"] == raw.loc[1, "cidade"]   # other rows untouched


@pytest.mark.parametrize("bad_value", ["yes", "true", 2])
def test_non_boolean_new_device_is_rejected(raw_transactions, category_levels, params, bad_value):
    raw = raw_transactions.copy()
    raw["is_new_device"] = bad_value
    with pytest.raises(ValueError):
        make_features(raw, category_levels, params.features.columns)


def test_zero_one_integers_are_accepted_for_new_device(raw_transactions, category_levels, params):
    raw = raw_transactions.copy()
    raw["is_new_device"] = raw["is_new_device"].astype(int)
    X = make_features(raw, category_levels, params.features.columns)
    assert X["is_new_device"].tolist() == raw["is_new_device"].tolist()


def test_utc_timestamp_gives_same_features_as_brazil_time(category_levels, params):
    base = {"transaction_id": "t", "nome_estabelecimento": "Online Shop 1", "tipo_estabelecimento": "E-commerce",
            "valor_transacao": 32.9, "cidade": "São Paulo", "device_type": "mobile", "is_new_device": True}
    local = Transaction.model_validate({**base, "datahora_transacao": "2026-01-15T22:14:00"})
    utc = Transaction.model_validate({**base, "datahora_transacao": "2026-01-16T01:14:00Z"})
    X = make_features(pd.DataFrame([local.model_dump(), utc.model_dump()]), category_levels, params.features.columns)
    assert X["hour"].tolist() == [22, 22]


def test_missing_input_column_fails_loudly(raw_transactions, category_levels, params):
    with pytest.raises(KeyError):
        make_features(raw_transactions.drop(columns="valor_transacao"), category_levels, params.features.columns)


def test_category_levels_are_learned_sorted(raw_transactions):
    levels = fit_category_levels(raw_transactions, ["device_type"])
    assert levels["device_type"] == sorted(raw_transactions["device_type"].unique())
