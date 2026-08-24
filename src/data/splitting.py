from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplittingError(Exception):
    """Raised when dataset splitting fails."""


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train, validation, and test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplit:
    """
    Split a dataset into train, validation, and test sets.

    The validation and test sizes are expressed as fractions
    of the complete dataset.

    Parameters
    ----------
    data:
        Input dataset.
    target_column:
        Name of the target column.
    test_size:
        Fraction of the complete dataset reserved for testing.
    validation_size:
        Fraction of the complete dataset reserved for validation.
    random_state:
        Random seed for reproducibility.
    stratify:
        Whether to preserve the target class distribution.

    Returns
    -------
    DatasetSplit
        Train, validation, and test datasets.

    Raises
    ------
    DataSplittingError
        If the input or split configuration is invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataSplittingError(
            "Input data must be a pandas DataFrame."
        )

    if data.empty:
        raise DataSplittingError("Cannot split an empty dataset.")

    if target_column not in data.columns:
        raise DataSplittingError(
            f"Target column '{target_column}' does not exist."
        )

    if not 0 < test_size < 1:
        raise DataSplittingError(
            "test_size must be between 0 and 1."
        )

    if not 0 < validation_size < 1:
        raise DataSplittingError(
            "validation_size must be between 0 and 1."
        )

    if test_size + validation_size >= 1:
        raise DataSplittingError(
            "test_size + validation_size must be less than 1."
        )

    target = data[target_column]

    if target.isna().any():
        raise DataSplittingError(
            f"Target column '{target_column}' contains missing values."
        )

    stratify_values = target if stratify else None

    train_validation, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    # Convert validation size from a fraction of the full dataset
    # to a fraction of the remaining train-validation dataset.
    validation_fraction = validation_size / (1 - test_size)

    train_stratify = (
        train_validation[target_column]
        if stratify
        else None
    )

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=train_stratify,
    )

    return DatasetSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def verify_no_row_overlap(split: DatasetSplit) -> bool:
    """
    Verify that no identical rows appear across dataset splits.

    Returns
    -------
    bool
        True if no rows overlap between splits.
    """
    train_rows = set(map(tuple, split.train.to_numpy()))
    validation_rows = set(map(tuple, split.validation.to_numpy()))
    test_rows = set(map(tuple, split.test.to_numpy()))

    return (
        train_rows.isdisjoint(validation_rows)
        and train_rows.isdisjoint(test_rows)
        and validation_rows.isdisjoint(test_rows)
    )


def get_split_sizes(split: DatasetSplit) -> dict[str, int]:
    """
    Return the number of rows in each split.
    """
    return {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }