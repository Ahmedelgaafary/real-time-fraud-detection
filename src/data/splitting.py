"""Dataset splitting utilities."""

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

    This is a generic random split and should not be used when
    temporal ordering is important.
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


def temporal_split_dataset(
    data: pd.DataFrame,
    time_column: str,
    train_size: float = 0.70,
    validation_size: float = 0.15,
) -> DatasetSplit:
    """
    Split a dataset chronologically into train, validation, and test.

    The input data is sorted by the supplied time column before
    splitting. The test set therefore contains the latest observations.

    Parameters
    ----------
    data:
        Input dataset.
    time_column:
        Column containing the temporal ordering.
    train_size:
        Fraction of rows assigned to training.
    validation_size:
        Fraction of rows assigned to validation.

    Returns
    -------
    DatasetSplit
        Chronologically ordered train, validation, and test sets.

    Raises
    ------
    DataSplittingError
        If the input or configuration is invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise DataSplittingError(
            "Input data must be a pandas DataFrame."
        )

    if data.empty:
        raise DataSplittingError(
            "Cannot split an empty dataset."
        )

    if time_column not in data.columns:
        raise DataSplittingError(
            f"Time column '{time_column}' does not exist."
        )

    if not 0 < train_size < 1:
        raise DataSplittingError(
            "train_size must be between 0 and 1."
        )

    if not 0 < validation_size < 1:
        raise DataSplittingError(
            "validation_size must be between 0 and 1."
        )

    if train_size + validation_size >= 1:
        raise DataSplittingError(
            "train_size + validation_size must be less than 1."
        )

    if data[time_column].isna().any():
        raise DataSplittingError(
            f"Time column '{time_column}' contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(data[time_column]):
        raise DataSplittingError(
            f"Time column '{time_column}' must be numeric."
        )

    sorted_data = data.sort_values(
        by=time_column,
        kind="mergesort",
    ).reset_index(drop=True)

    total_rows = len(sorted_data)

    train_end = int(total_rows * train_size)
    validation_end = int(
        total_rows * (train_size + validation_size)
    )

    if train_end == 0:
        raise DataSplittingError(
            "Training split contains zero rows."
        )

    if validation_end <= train_end:
        raise DataSplittingError(
            "Validation split contains zero rows."
        )

    if validation_end >= total_rows:
        raise DataSplittingError(
            "Test split contains zero rows."
        )

    train = sorted_data.iloc[:train_end]
    validation = sorted_data.iloc[train_end:validation_end]
    test = sorted_data.iloc[validation_end:]

    return DatasetSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def verify_no_row_overlap(split: DatasetSplit) -> bool:
    """
    Verify that no identical rows appear across dataset splits.
    """
    train_rows = set(map(tuple, split.train.to_numpy()))
    validation_rows = set(map(tuple, split.validation.to_numpy()))
    test_rows = set(map(tuple, split.test.to_numpy()))

    return (
        train_rows.isdisjoint(validation_rows)
        and train_rows.isdisjoint(test_rows)
        and validation_rows.isdisjoint(test_rows)
    )


def verify_temporal_order(
    split: DatasetSplit,
    time_column: str,
) -> bool:
    """
    Verify chronological ordering between dataset splits.

    Returns True when every training observation occurs no later
    than validation observations, and every validation observation
    occurs no later than test observations.
    """
    train_max = split.train[time_column].max()
    validation_min = split.validation[time_column].min()
    validation_max = split.validation[time_column].max()
    test_min = split.test[time_column].min()

    return (
        train_max <= validation_min
        and validation_max <= test_min
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