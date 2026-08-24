import pandas as pd
import pytest

from src.data.ieee_cis import (
    IEEECISValidationError,
    validate_ieee_cis,
    validate_ieee_cis_numeric_fields,
    validate_ieee_cis_schema,
    validate_ieee_cis_target,
    validate_ieee_cis_transaction_id,
)


def create_ieee_cis_sample() -> pd.DataFrame:
    """Create a small IEEE-CIS-like dataset."""
    return pd.DataFrame(
        {
            "TransactionID": [2987000, 2987001, 2987002],
            "isFraud": [0, 1, 0],
            "TransactionDT": [86400, 86401, 86469],
            "TransactionAmt": [68.5, 29.0, 59.0],
            "ProductCD": ["W", "W", "H"],
        }
    )


def test_valid_ieee_cis_schema() -> None:
    """Test the required IEEE-CIS schema."""
    data = create_ieee_cis_sample()

    validate_ieee_cis_schema(data)


def test_missing_required_column_is_rejected() -> None:
    """Test missing required columns."""
    data = create_ieee_cis_sample()
    data = data.drop(columns=["ProductCD"])

    with pytest.raises(
        IEEECISValidationError,
        match="Missing required IEEE-CIS columns",
    ):
        validate_ieee_cis_schema(data)


def test_valid_target() -> None:
    """Test valid binary fraud labels."""
    data = create_ieee_cis_sample()

    validate_ieee_cis_target(data)


def test_missing_target_is_rejected() -> None:
    """Test missing fraud labels."""
    data = create_ieee_cis_sample()
    data.loc[0, "isFraud"] = None

    with pytest.raises(
        IEEECISValidationError,
        match="target contains missing values",
    ):
        validate_ieee_cis_target(data)


def test_non_binary_target_is_rejected() -> None:
    """Test invalid fraud labels."""
    data = create_ieee_cis_sample()
    data.loc[0, "isFraud"] = 2

    with pytest.raises(
        IEEECISValidationError,
        match="binary values",
    ):
        validate_ieee_cis_target(data)


def test_transaction_id_must_be_unique() -> None:
    """Test TransactionID uniqueness."""
    data = create_ieee_cis_sample()
    data.loc[1, "TransactionID"] = data.loc[0, "TransactionID"]

    with pytest.raises(
        IEEECISValidationError,
        match="must be unique",
    ):
        validate_ieee_cis_transaction_id(data)


def test_transaction_id_cannot_be_missing() -> None:
    """Test missing TransactionID."""
    data = create_ieee_cis_sample()
    data.loc[0, "TransactionID"] = None

    with pytest.raises(
        IEEECISValidationError,
        match="TransactionID contains missing values",
    ):
        validate_ieee_cis_transaction_id(data)


def test_numeric_fields_are_valid() -> None:
    """Test numeric IEEE-CIS fields."""
    data = create_ieee_cis_sample()

    validate_ieee_cis_numeric_fields(data)


def test_transaction_amount_must_be_numeric() -> None:
    """Test TransactionAmt type."""
    data = create_ieee_cis_sample()
    data["TransactionAmt"] = ["bad", "data", "here"]

    with pytest.raises(
        IEEECISValidationError,
        match="TransactionAmt must be numeric",
    ):
        validate_ieee_cis_numeric_fields(data)


def test_transaction_time_must_be_numeric() -> None:
    """Test TransactionDT type."""
    data = create_ieee_cis_sample()
    data["TransactionDT"] = ["a", "b", "c"]

    with pytest.raises(
        IEEECISValidationError,
        match="TransactionDT must be numeric",
    ):
        validate_ieee_cis_numeric_fields(data)


def test_negative_transaction_amount_is_rejected() -> None:
    """Test negative transaction amounts."""
    data = create_ieee_cis_sample()
    data.loc[0, "TransactionAmt"] = -10.0

    with pytest.raises(
        IEEECISValidationError,
        match="cannot contain negative values",
    ):
        validate_ieee_cis_numeric_fields(data)


def test_complete_ieee_cis_validation() -> None:
    """Test the complete IEEE-CIS validation pipeline."""
    data = create_ieee_cis_sample()

    validate_ieee_cis(data)