from dataclasses import dataclass

import numpy as np
import pandas as pd


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""


@dataclass(frozen=True)
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_dataset(
    data: pd.DataFrame,
    required_columns: list[str] | None = None,
    target_column: str | None = None,
) -> ValidationResult:
    """Validate the basic structure and contents of a dataset."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, pd.DataFrame):
        errors.append("Dataset must be a pandas DataFrame.")
        return ValidationResult(False, errors, warnings)

    if data.empty:
        errors.append("Dataset is empty.")
        return ValidationResult(False, errors, warnings)

    if required_columns:
        missing_columns = [
            column for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            errors.append(
                f"Missing required columns: {missing_columns}"
            )

    if target_column is not None:
        if target_column not in data.columns:
            errors.append(
                f"Target column '{target_column}' is missing."
            )
        else:
            target = data[target_column]

            if target.isna().any():
                errors.append(
                    f"Target column '{target_column}' contains "
                    "missing values."
                )

            unique_values = set(target.dropna().unique())

            if not unique_values.issubset({0, 1}):
                errors.append(
                    f"Target column '{target_column}' must contain "
                    f"only binary values 0 and 1. "
                    f"Found: {sorted(unique_values)}"
                )

    duplicate_count = int(data.duplicated().sum())

    if duplicate_count > 0:
        warnings.append(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    missing_values = int(data.isna().sum().sum())

    if missing_values > 0:
        warnings.append(
            f"Dataset contains {missing_values} missing values."
        )

    numeric_data = data.select_dtypes(include="number")

    if not numeric_data.empty:
        infinite_values = int(
            np.isinf(numeric_data.to_numpy()).sum()
        )

        if infinite_values > 0:
            warnings.append(
                f"Dataset contains {infinite_values} infinite values."
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_dataset_or_raise(
    data: pd.DataFrame,
    required_columns: list[str] | None = None,
    target_column: str | None = None,
) -> ValidationResult:
    """Validate a dataset and raise an exception if it is invalid."""
    result = validate_dataset(
        data=data,
        required_columns=required_columns,
        target_column=target_column,
    )

    if not result.is_valid:
        message = "Dataset validation failed:\n" + "\n".join(
            f"- {error}" for error in result.errors
        )

        raise DatasetValidationError(message)

    return result