"""Transaction-level feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineeringError(ValueError):
    """Raised when feature engineering fails."""


def add_transaction_time_features(
    data: pd.DataFrame,
    time_column: str = "TransactionDT",
) -> pd.DataFrame:
    """Add time-derived transaction features."""
    if not isinstance(data, pd.DataFrame):
        raise FeatureEngineeringError(
            "Input data must be a pandas DataFrame."
        )

    if data.empty:
        raise FeatureEngineeringError(
            "Cannot engineer features from an empty dataset."
        )

    if time_column not in data.columns:
        raise FeatureEngineeringError(
            f"Time column '{time_column}' does not exist."
        )

    if data[time_column].isna().any():
        raise FeatureEngineeringError(
            f"Time column '{time_column}' contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(data[time_column]):
        raise FeatureEngineeringError(
            f"Time column '{time_column}' must be numeric."
        )

    result = data.copy()

    seconds = data[time_column]

    result["transaction_day"] = seconds // 86_400
    result["transaction_hour"] = (
        (seconds % 86_400) // 3_600
    )
    result["transaction_minute"] = (
        (seconds % 3_600) // 60
    )
    result["transaction_second"] = seconds % 60
    result["transaction_weekday"] = (
        result["transaction_day"] % 7
    )

    return result


def add_transaction_amount_features(
    data: pd.DataFrame,
    amount_column: str = "TransactionAmt",
) -> pd.DataFrame:
    """Add transaction amount features."""
    if not isinstance(data, pd.DataFrame):
        raise FeatureEngineeringError(
            "Input data must be a pandas DataFrame."
        )

    if data.empty:
        raise FeatureEngineeringError(
            "Cannot engineer features from an empty dataset."
        )

    if amount_column not in data.columns:
        raise FeatureEngineeringError(
            f"Amount column '{amount_column}' does not exist."
        )

    if data[amount_column].isna().any():
        raise FeatureEngineeringError(
            f"Amount column '{amount_column}' contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(data[amount_column]):
        raise FeatureEngineeringError(
            f"Amount column '{amount_column}' must be numeric."
        )

    if (data[amount_column] < 0).any():
        raise FeatureEngineeringError(
            f"Amount column '{amount_column}' cannot contain "
            "negative values."
        )

    result = data.copy()

    amount = data[amount_column]

    result["transaction_amount_log"] = np.log1p(amount)

    result["transaction_amount_decimal"] = amount % 1

    result["transaction_amount_is_round"] = (
        (amount % 1) == 0
    ).astype("int8")

    return result


def get_missingness_columns(
    data: pd.DataFrame,
    threshold: float = 0.05,
    exclude_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """
    Determine which columns should receive missingness indicators.

    This selection should normally be performed on training data only.
    """
    if not isinstance(data, pd.DataFrame):
        raise FeatureEngineeringError(
            "Input data must be a pandas DataFrame."
        )

    if data.empty:
        raise FeatureEngineeringError(
            "Cannot analyze missingness of empty data."
        )

    if not 0 <= threshold <= 1:
        raise FeatureEngineeringError(
            "threshold must be between 0 and 1."
        )

    excluded = set(exclude_columns)

    missing_rates = data.isna().mean()

    return tuple(
        column
        for column in data.columns
        if column not in excluded
        and missing_rates[column] >= threshold
    )


def add_missingness_features(
    data: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
    threshold: float = 0.05,
    exclude_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Add missingness indicator features.

    A ``*_is_missing`` indicator is created when the missing-value
    ratio of a selected column is greater than or equal to
    ``threshold``.

    Parameters
    ----------
    data:
        Input DataFrame. The input is never modified.

    columns:
        Columns to evaluate. If None, all columns are considered.

    threshold:
        Minimum missing-value ratio required to create an indicator.
        The boundary is inclusive.

    exclude_columns:
        Columns that must not receive missingness indicators.

    Returns
    -------
    pd.DataFrame
        Copy of the input data with missingness indicators added.

    Raises
    ------
    FeatureEngineeringError
        If the input or threshold is invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise FeatureEngineeringError(
            "data must be a pandas DataFrame."
        )

    if not 0 <= threshold <= 1:
        raise FeatureEngineeringError(
            "threshold must be between 0 and 1."
        )

    result = data.copy()

    if columns is None:
        selected_columns = list(result.columns)
    else:
        selected_columns = list(columns)

    missing_columns = [
        column
        for column in selected_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise FeatureEngineeringError(
            "Columns do not exist: "
            f"{missing_columns}"
        )

    excluded = set(exclude_columns)

    indicator_columns: dict[str, pd.Series] = {}

    for column in selected_columns:
        if column in excluded:
            continue

        missing_ratio = result[column].isna().mean()

        if missing_ratio >= threshold:
            indicator_columns[f"{column}_is_missing"] = (
                result[column].isna().astype("int8")
            )

    if indicator_columns:
        indicators = pd.DataFrame(
            indicator_columns,
            index=result.index,
        )

        result = pd.concat(
            [result, indicators],
            axis=1,
        )

    return result