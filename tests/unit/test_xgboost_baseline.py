"""Tests for the XGBoost baseline model."""

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_baseline import (
    XGBoostBaseline,
    XGBoostConfig,
    XGBoostModelError,
)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "amount": [
                10.0,
                20.0,
                30.0,
                100.0,
                120.0,
                150.0,
                15.0,
                25.0,
                110.0,
                130.0,
            ],
            "velocity": [
                1.0,
                2.0,
                1.0,
                8.0,
                9.0,
                10.0,
                2.0,
                1.0,
                7.0,
                9.0,
            ],
        }
    )

    target = pd.Series(
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        name="isFraud",
    )

    return features, target


def test_model_can_be_fitted(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_data

    model = XGBoostBaseline(
        XGBoostConfig(
            n_estimators=10,
            max_depth=2,
        )
    )

    result = model.fit(features, target)

    assert result is model
    assert model.model is not None
    assert model.feature_names == (
        "amount",
        "velocity",
    )


def test_predict_proba_returns_valid_probabilities(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_data

    model = XGBoostBaseline(
        XGBoostConfig(n_estimators=10)
    )
    model.fit(features, target)

    probabilities = model.predict_proba(features)

    assert len(probabilities) == len(features)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_predict_returns_binary_values(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_data

    model = XGBoostBaseline(
        XGBoostConfig(n_estimators=10)
    )
    model.fit(features, target)

    predictions = model.predict(features)

    assert len(predictions) == len(features)
    assert set(predictions).issubset({0, 1})


def test_custom_threshold_changes_predictions(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_data

    model = XGBoostBaseline(
        XGBoostConfig(n_estimators=10)
    )
    model.fit(features, target)

    low_threshold = model.predict(
        features,
        threshold=0.2,
    )

    high_threshold = model.predict(
        features,
        threshold=0.8,
    )

    assert low_threshold.sum() >= high_threshold.sum()


def test_unfitted_model_is_rejected(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _ = training_data

    model = XGBoostBaseline()

    with pytest.raises(
        XGBoostModelError,
        match="not been fitted",
    ):
        model.predict_proba(features)


def test_missing_values_are_rejected(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_data

    features = features.copy()
    features.loc[0, "amount"] = np.nan

    model = XGBoostBaseline()

    with pytest.raises(
        XGBoostModelError,
        match="missing values",
    ):
        model.fit(features, target)


def test_non_numeric_features_are_rejected(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_data

    features = features.copy()
    features["category"] = "A"

    model = XGBoostBaseline()

    with pytest.raises(
        XGBoostModelError,
        match="numeric",
    ):
        model.fit(features, target)


def test_invalid_target_is_rejected(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _ = training_data

    target = pd.Series(
        [0, 1, 2, 0, 1, 0, 1, 0, 1, 0]
    )

    model = XGBoostBaseline()

    with pytest.raises(
        XGBoostModelError,
        match="binary",
    ):
        model.fit(features, target)


def test_invalid_threshold_is_rejected(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_data

    model = XGBoostBaseline(
        XGBoostConfig(n_estimators=10)
    )
    model.fit(features, target)

    with pytest.raises(
        XGBoostModelError,
        match="between 0 and 1",
    ):
        model.predict(features, threshold=0.0)


def test_feature_schema_mismatch_is_rejected(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_data

    model = XGBoostBaseline(
        XGBoostConfig(n_estimators=10)
    )
    model.fit(features, target)

    reordered = features[
        ["velocity", "amount"]
    ]

    with pytest.raises(
        XGBoostModelError,
        match="feature schema",
    ):
        model.predict_proba(reordered)