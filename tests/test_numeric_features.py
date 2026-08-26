"""Tests for numeric feature preprocessing."""

import pandas as pd
import pytest

from src.features.numeric_features import (
    NumericFeatureError,
    NumericImputer,
)


def test_numeric_imputer_fit() -> None:
    """Test median calculation."""
    data = pd.DataFrame(
        {
            "amount": [10.0, None, 30.0, 40.0],
            "count": [1.0, 2.0, None, 4.0],
        }
    )

    imputer = NumericImputer.fit(data)

    assert imputer.medians["amount"] == 30.0
    assert imputer.medians["count"] == 2.0


def test_numeric_imputer_transform() -> None:
    """Test median imputation."""
    train = pd.DataFrame(
        {
            "amount": [10.0, None, 30.0],
        }
    )

    imputer = NumericImputer.fit(train)

    result = imputer.transform(train)

    assert result["amount"].tolist() == [
        10.0,
        20.0,
        30.0,
    ]


def test_missing_indicator_is_created() -> None:
    """Test missing-value indicator."""
    train = pd.DataFrame(
        {
            "amount": [10.0, None, 30.0],
        }
    )

    imputer = NumericImputer.fit(train)

    result = imputer.transform(train)

    assert result["amount_was_missing"].tolist() == [
        0,
        1,
        0,
    ]


def test_training_statistics_are_reused() -> None:
    """Test validation data uses training median."""
    train = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0],
        }
    )

    validation = pd.DataFrame(
        {
            "amount": [100.0, None, 200.0],
        }
    )

    imputer = NumericImputer.fit(train)

    result = imputer.transform(validation)

    assert result["amount"].tolist() == [
        100.0,
        20.0,
        200.0,
    ]


def test_unknown_numeric_column_is_rejected() -> None:
    """Test missing column handling."""
    train = pd.DataFrame(
        {
            "amount": [10.0, 20.0],
        }
    )

    imputer = NumericImputer.fit(train)

    validation = pd.DataFrame(
        {
            "other": [1.0, 2.0],
        }
    )

    with pytest.raises(
        NumericFeatureError,
        match="do not exist",
    ):
        imputer.transform(validation)


def test_non_numeric_column_is_rejected() -> None:
    """Test non-numeric configuration."""
    data = pd.DataFrame(
        {
            "amount": [10.0, 20.0],
            "category": ["A", "B"],
        }
    )

    with pytest.raises(
        NumericFeatureError,
        match="must be numeric",
    ):
        NumericImputer.fit(
            data,
            columns=("category",),
        )


def test_empty_data_is_rejected() -> None:
    """Test empty dataset."""
    data = pd.DataFrame()

    with pytest.raises(
        NumericFeatureError,
        match="empty data",
    ):
        NumericImputer.fit(data)


def test_all_missing_column_is_rejected() -> None:
    """Test column containing only missing values."""
    data = pd.DataFrame(
        {
            "amount": [None, None, None],
        }
    )

    with pytest.raises(
        NumericFeatureError,
        match="no valid",
    ):
        NumericImputer.fit(data)


def test_input_is_not_modified() -> None:
    """Test transformation does not mutate input."""
    data = pd.DataFrame(
        {
            "amount": [10.0, None, 30.0],
        }
    )

    original = data.copy()

    imputer = NumericImputer.fit(data)
    imputer.transform(data)

    pd.testing.assert_frame_equal(data, original)