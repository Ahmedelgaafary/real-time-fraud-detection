"""Tests for categorical feature engineering."""

import pandas as pd
import pytest

from src.features.categorical_features import (
    CategoricalFeatureError,
    FrequencyEncoder,
)


def test_frequency_encoder_fit() -> None:
    """Test frequency mapping creation."""
    data = pd.DataFrame(
        {
            "ProductCD": [
                "W",
                "W",
                "C",
                "H",
            ]
        }
    )

    encoder = FrequencyEncoder.fit(
        data,
        ["ProductCD"],
    )

    assert encoder.columns == ("ProductCD",)

    assert encoder.mappings["ProductCD"]["W"] == 0.5
    assert encoder.mappings["ProductCD"]["C"] == 0.25
    assert encoder.mappings["ProductCD"]["H"] == 0.25


def test_frequency_encoder_transform() -> None:
    """Test frequency transformation."""
    train = pd.DataFrame(
        {
            "ProductCD": [
                "W",
                "W",
                "C",
                "H",
            ]
        }
    )

    encoder = FrequencyEncoder.fit(
        train,
        ["ProductCD"],
    )

    result = encoder.transform(train)

    assert "ProductCD_frequency" in result.columns

    assert result["ProductCD_frequency"].tolist() == [
        0.5,
        0.5,
        0.25,
        0.25,
    ]


def test_unknown_category_gets_zero() -> None:
    """Test unseen categories."""
    train = pd.DataFrame(
        {
            "ProductCD": [
                "W",
                "C",
            ]
        }
    )

    test = pd.DataFrame(
        {
            "ProductCD": [
                "W",
                "H",
            ]
        }
    )

    encoder = FrequencyEncoder.fit(
        train,
        ["ProductCD"],
    )

    result = encoder.transform(test)

    assert result["ProductCD_frequency"].tolist() == [
        0.5,
        0.0,
    ]


def test_missing_category_frequency_is_learned() -> None:
    """Test missing-value frequency."""
    train = pd.DataFrame(
        {
            "ProductCD": [
                "W",
                None,
                "W",
                None,
            ]
        }
    )

    encoder = FrequencyEncoder.fit(
        train,
        ["ProductCD"],
    )

    result = encoder.transform(train)

    assert result["ProductCD_frequency"].tolist() == [
        0.5,
        0.5,
        0.5,
        0.5,
    ]


def test_missing_column_is_rejected_during_fit() -> None:
    """Test missing categorical column."""
    data = pd.DataFrame(
        {
            "ProductCD": ["W", "C"],
        }
    )

    with pytest.raises(
        CategoricalFeatureError,
        match="do not exist",
    ):
        FrequencyEncoder.fit(
            data,
            ["card4"],
        )


def test_missing_column_is_rejected_during_transform() -> None:
    """Test missing categorical column during transformation."""
    train = pd.DataFrame(
        {
            "ProductCD": ["W", "C"],
        }
    )

    test = pd.DataFrame(
        {
            "card4": ["visa"],
        }
    )

    encoder = FrequencyEncoder.fit(
        train,
        ["ProductCD"],
    )

    with pytest.raises(
        CategoricalFeatureError,
        match="do not exist",
    ):
        encoder.transform(test)


def test_empty_columns_are_rejected() -> None:
    """Test empty categorical column configuration."""
    data = pd.DataFrame(
        {
            "ProductCD": ["W", "C"],
        }
    )

    with pytest.raises(
        CategoricalFeatureError,
        match="At least one",
    ):
        FrequencyEncoder.fit(
            data,
            [],
        )


def test_empty_training_data_is_rejected() -> None:
    """Test empty training data."""
    data = pd.DataFrame(
        {
            "ProductCD": pd.Series(dtype="object"),
        }
    )

    with pytest.raises(
        CategoricalFeatureError,
        match="empty dataset",
    ):
        FrequencyEncoder.fit(
            data,
            ["ProductCD"],
        )


def test_encoder_does_not_modify_input() -> None:
    """Test non-mutating transformation."""
    data = pd.DataFrame(
        {
            "ProductCD": ["W", "C"],
        }
    )

    encoder = FrequencyEncoder.fit(
        data,
        ["ProductCD"],
    )

    result = encoder.transform(data)

    assert "ProductCD_frequency" not in data.columns
    assert "ProductCD_frequency" in result.columns