"""Tests for transaction feature engineering."""

import pandas as pd
import pytest

from src.features.transaction_features import (
    FeatureEngineeringError,
    add_missingness_features,
    add_transaction_amount_features,
    add_transaction_time_features,
)


def test_transaction_time_features_are_added() -> None:
    """Test time-derived feature creation."""
    data = pd.DataFrame(
        {
            "TransactionDT": [
                0,
                3_600,
                86_400,
            ]
        }
    )

    result = add_transaction_time_features(data)

    assert result["transaction_day"].tolist() == [0, 0, 1]
    assert result["transaction_hour"].tolist() == [0, 1, 0]
    assert result["transaction_minute"].tolist() == [0, 0, 0]
    assert result["transaction_second"].tolist() == [0, 0, 0]


def test_transaction_minute_and_second() -> None:
    """Test minute and second extraction."""
    data = pd.DataFrame(
        {
            "TransactionDT": [3_725],
        }
    )

    result = add_transaction_time_features(data)

    assert result["transaction_day"].iloc[0] == 0
    assert result["transaction_hour"].iloc[0] == 1
    assert result["transaction_minute"].iloc[0] == 2
    assert result["transaction_second"].iloc[0] == 5


def test_weekday_is_relative_to_transaction_day() -> None:
    """Test relative weekday feature."""
    data = pd.DataFrame(
        {
            "TransactionDT": [
                0,
                86_400 * 6,
                86_400 * 7,
            ]
        }
    )

    result = add_transaction_time_features(data)

    assert result["transaction_weekday"].tolist() == [
        0,
        6,
        0,
    ]


def test_missing_time_column_is_rejected() -> None:
    """Test missing TransactionDT."""
    data = pd.DataFrame(
        {"amount": [10.0]}
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="does not exist",
    ):
        add_transaction_time_features(data)


def test_missing_time_value_is_rejected() -> None:
    """Test missing temporal value."""
    data = pd.DataFrame(
        {
            "TransactionDT": [100, None],
        }
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="contains missing values",
    ):
        add_transaction_time_features(data)


def test_non_numeric_time_is_rejected() -> None:
    """Test invalid temporal type."""
    data = pd.DataFrame(
        {
            "TransactionDT": ["100", "200"],
        }
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="must be numeric",
    ):
        add_transaction_time_features(data)


def test_original_data_is_not_modified() -> None:
    """Test that time feature engineering does not mutate input."""
    data = pd.DataFrame(
        {
            "TransactionDT": [100],
        }
    )

    original_columns = data.columns.tolist()

    result = add_transaction_time_features(data)

    assert data.columns.tolist() == original_columns
    assert "transaction_hour" not in data.columns
    assert "transaction_hour" in result.columns


def test_transaction_amount_features_are_added() -> None:
    """Test transaction amount feature creation."""
    data = pd.DataFrame(
        {
            "TransactionAmt": [
                10.00,
                25.50,
                100.25,
            ]
        }
    )

    result = add_transaction_amount_features(data)

    assert "transaction_amount_log" in result.columns
    assert "transaction_amount_decimal" in result.columns
    assert "transaction_amount_is_round" in result.columns


def test_transaction_amount_decimal() -> None:
    """Test decimal component extraction."""
    data = pd.DataFrame(
        {
            "TransactionAmt": [
                10.00,
                25.50,
                100.25,
            ]
        }
    )

    result = add_transaction_amount_features(data)

    assert result["transaction_amount_decimal"].tolist() == [
        0.0,
        0.5,
        0.25,
    ]


def test_round_transaction_amount_flag() -> None:
    """Test round amount detection."""
    data = pd.DataFrame(
        {
            "TransactionAmt": [
                10.00,
                25.50,
                100.00,
            ]
        }
    )

    result = add_transaction_amount_features(data)

    assert result["transaction_amount_is_round"].tolist() == [
        1,
        0,
        1,
    ]


def test_transaction_amount_log_is_non_negative() -> None:
    """Test log-transformed amounts."""
    data = pd.DataFrame(
        {
            "TransactionAmt": [
                0.0,
                10.0,
                100.0,
            ]
        }
    )

    result = add_transaction_amount_features(data)

    assert (
        result["transaction_amount_log"] >= 0
    ).all()


def test_missing_amount_is_rejected() -> None:
    """Test missing transaction amounts."""
    data = pd.DataFrame(
        {
            "TransactionAmt": [10.0, None],
        }
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="contains missing values",
    ):
        add_transaction_amount_features(data)


def test_negative_amount_is_rejected() -> None:
    """Test negative transaction amounts."""
    data = pd.DataFrame(
        {
            "TransactionAmt": [10.0, -5.0],
        }
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="cannot contain",
    ):
        add_transaction_amount_features(data)


def test_missingness_features_are_added() -> None:
    """Test missingness indicator creation."""
    data = pd.DataFrame(
        {
            "card1": [1, 2, 3, 4],
            "card2": [10, None, 20, None],
            "card3": [1, 2, 3, 4],
        }
    )

    result = add_missingness_features(
        data,
        threshold=0.25,
    )

    assert "card2_is_missing" in result.columns
    assert "card1_is_missing" not in result.columns
    assert "card3_is_missing" not in result.columns


def test_missingness_indicator_values() -> None:
    """Test missingness indicator values."""
    data = pd.DataFrame(
        {
            "feature": [10, None, 20, None],
        }
    )

    result = add_missingness_features(
        data,
        threshold=0.25,
    )

    assert result["feature_is_missing"].tolist() == [
        0,
        1,
        0,
        1,
    ]


def test_missingness_threshold() -> None:
    """Test that threshold controls indicator creation."""
    data = pd.DataFrame(
        {
            "feature": [10, None, 20, 30],
        }
    )

    result = add_missingness_features(
        data,
        threshold=0.50,
    )

    assert "feature_is_missing" not in result.columns


def test_missingness_threshold_includes_exact_boundary() -> None:
    """Test inclusion at the exact threshold."""
    data = pd.DataFrame(
        {
            "feature": [10, None, 20, None],
        }
    )

    result = add_missingness_features(
        data,
        threshold=0.50,
    )

    assert "feature_is_missing" in result.columns


def test_excluded_columns_do_not_receive_indicators() -> None:
    """Test explicit missingness exclusions."""
    data = pd.DataFrame(
        {
            "feature_a": [10, None],
            "feature_b": [20, None],
        }
    )

    result = add_missingness_features(
        data,
        threshold=0.50,
        exclude_columns=("feature_a",),
    )

    assert "feature_a_is_missing" not in result.columns
    assert "feature_b_is_missing" in result.columns


def test_invalid_missingness_threshold_is_rejected() -> None:
    """Test invalid missingness threshold."""
    data = pd.DataFrame(
        {
            "feature": [1, None],
        }
    )

    with pytest.raises(
        FeatureEngineeringError,
        match="between 0 and 1",
    ):
        add_missingness_features(
            data,
            threshold=1.5,
        )


def test_missingness_does_not_modify_input() -> None:
    """Test missingness feature engineering is non-mutating."""
    data = pd.DataFrame(
        {
            "feature": [1, None],
        }
    )

    original_columns = data.columns.tolist()

    result = add_missingness_features(
        data,
        threshold=0.5,
    )

    assert data.columns.tolist() == original_columns
    assert "feature_is_missing" in result.columns
    assert "feature_is_missing" not in data.columns