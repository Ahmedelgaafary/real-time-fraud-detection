"""IEEE-CIS Fraud Detection dataset definitions and validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class IEEECISSchema:
    """Schema definition for the IEEE-CIS transaction dataset."""

    transaction_id: str = "TransactionID"
    target: str = "isFraud"
    timestamp: str = "TransactionDT"
    amount: str = "TransactionAmt"
    product: str = "ProductCD"

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the minimum required columns."""
        return (
            self.transaction_id,
            self.target,
            self.timestamp,
            self.amount,
            self.product,
        )


IEEE_CIS_SCHEMA = IEEECISSchema()


class IEEECISValidationError(ValueError):
    """Raised when IEEE-CIS-specific validation fails."""


def validate_ieee_cis_schema(
    data: pd.DataFrame,
    schema: IEEECISSchema = IEEE_CIS_SCHEMA,
) -> None:
    """Validate the required IEEE-CIS columns."""
    if not isinstance(data, pd.DataFrame):
        raise IEEECISValidationError(
            "IEEE-CIS data must be a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in schema.required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise IEEECISValidationError(
            "Missing required IEEE-CIS columns: "
            f"{missing_columns}"
        )


def validate_ieee_cis_target(
    data: pd.DataFrame,
    schema: IEEECISSchema = IEEE_CIS_SCHEMA,
) -> None:
    """Validate the IEEE-CIS fraud target."""
    validate_ieee_cis_schema(data, schema)

    target = data[schema.target]

    if target.isna().any():
        raise IEEECISValidationError(
            "IEEE-CIS target contains missing values."
        )

    values = set(target.unique())

    if not values.issubset({0, 1}):
        raise IEEECISValidationError(
            "IEEE-CIS target must contain only binary values "
            "{0, 1}."
        )


def validate_ieee_cis_transaction_id(
    data: pd.DataFrame,
    schema: IEEECISSchema = IEEE_CIS_SCHEMA,
) -> None:
    """Validate the transaction identifier."""
    validate_ieee_cis_schema(data, schema)

    transaction_id = data[schema.transaction_id]

    if transaction_id.isna().any():
        raise IEEECISValidationError(
            "TransactionID contains missing values."
        )

    if not transaction_id.is_unique:
        raise IEEECISValidationError(
            "TransactionID must be unique."
        )


def validate_ieee_cis_numeric_fields(
    data: pd.DataFrame,
    schema: IEEECISSchema = IEEE_CIS_SCHEMA,
) -> None:
    """Validate required numeric transaction fields."""
    validate_ieee_cis_schema(data, schema)

    for column in (
        schema.timestamp,
        schema.amount,
    ):
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise IEEECISValidationError(
                f"{column} must be numeric."
            )

    if (data[schema.amount] < 0).any():
        raise IEEECISValidationError(
            "TransactionAmt cannot contain negative values."
        )


def validate_ieee_cis(
    data: pd.DataFrame,
    schema: IEEECISSchema = IEEE_CIS_SCHEMA,
) -> None:
    """Run all IEEE-CIS-specific validation checks."""
    validate_ieee_cis_schema(data, schema)
    validate_ieee_cis_target(data, schema)
    validate_ieee_cis_transaction_id(data, schema)
    validate_ieee_cis_numeric_fields(data, schema)