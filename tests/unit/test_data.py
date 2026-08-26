import pandas as pd
import pytest

from src.data.splitting import (
    DataSplittingError,
    get_split_sizes,
    temporal_split_dataset,
    verify_temporal_order,
)


def create_temporal_dataset() -> pd.DataFrame:
    """Create a small chronological dataset."""
    return pd.DataFrame(
        {
            "TransactionDT": [5, 1, 8, 3, 10, 2, 7, 4, 9, 6],
            "isFraud": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        }
    )


def test_temporal_split_sorts_by_time() -> None:
    """Test chronological sorting."""
    data = create_temporal_dataset()

    split = temporal_split_dataset(
        data,
        time_column="TransactionDT",
        train_size=0.70,
        validation_size=0.15,
    )

    assert split.train["TransactionDT"].tolist() == [
        1, 2, 3, 4, 5, 6, 7
    ]
    assert split.validation["TransactionDT"].tolist() == [8]
    assert split.test["TransactionDT"].tolist() == [9, 10]


def test_temporal_order_is_preserved() -> None:
    """Test that future observations never enter earlier splits."""
    data = create_temporal_dataset()

    split = temporal_split_dataset(
        data,
        time_column="TransactionDT",
    )

    assert verify_temporal_order(
        split,
        time_column="TransactionDT",
    )


def test_temporal_split_sizes() -> None:
    """Test temporal split sizes."""
    data = create_temporal_dataset()

    split = temporal_split_dataset(
        data,
        time_column="TransactionDT",
    )

    assert get_split_sizes(split) == {
        "train": 7,
        "validation": 1,
        "test": 2,
    }


def test_missing_time_column_is_rejected() -> None:
    """Test missing temporal column."""
    data = create_temporal_dataset()

    with pytest.raises(
        DataSplittingError,
        match="does not exist",
    ):
        temporal_split_dataset(
            data,
            time_column="missing",
        )


def test_missing_time_values_are_rejected() -> None:
    """Test missing temporal values."""
    data = create_temporal_dataset()
    data.loc[0, "TransactionDT"] = None

    with pytest.raises(
        DataSplittingError,
        match="contains missing values",
    ):
        temporal_split_dataset(
            data,
            time_column="TransactionDT",
        )


def test_non_numeric_time_column_is_rejected() -> None:
    """Test invalid temporal data type."""
    data = create_temporal_dataset()
    data["TransactionDT"] = ["a"] * len(data)

    with pytest.raises(
        DataSplittingError,
        match="must be numeric",
    ):
        temporal_split_dataset(
            data,
            time_column="TransactionDT",
        )


def test_invalid_split_configuration_is_rejected() -> None:
    """Test invalid split proportions."""
    data = create_temporal_dataset()

    with pytest.raises(
        DataSplittingError,
        match="must be less than 1",
    ):
        temporal_split_dataset(
            data,
            time_column="TransactionDT",
            train_size=0.80,
            validation_size=0.30,
        )