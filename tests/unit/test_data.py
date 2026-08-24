from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import (
    DatasetIngestionError,
    get_basic_metadata,
    load_dataset,
    load_dataset_with_metadata,
    validate_file_path,
)
from src.data.splitting import (
    DataSplittingError,
    get_split_sizes,
    split_dataset,
    verify_no_row_overlap,
)
from src.data.validation import (
    DatasetValidationError,
    validate_dataset,
    validate_dataset_or_raise,
)


def create_sample_dataset(rows: int = 100) -> pd.DataFrame:
    """Create a deterministic dataset for testing."""
    return pd.DataFrame(
        {
            "transaction_id": range(rows),
            "amount": [float(i + 1) for i in range(rows)],
            "is_fraud": [1 if i % 10 == 0 else 0 for i in range(rows)],
        }
    )


# ============================================================
# Ingestion tests
# ============================================================


def test_load_csv(tmp_path: Path) -> None:
    """Test loading a CSV dataset."""
    file_path = tmp_path / "transactions.csv"

    expected = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "amount": [100.0, 250.5, 75.25],
            "is_fraud": [0, 1, 0],
        }
    )

    expected.to_csv(file_path, index=False)

    result = load_dataset(file_path)

    pd.testing.assert_frame_equal(result, expected)


def test_load_parquet(tmp_path: Path) -> None:
    """Test loading a Parquet dataset."""
    file_path = tmp_path / "transactions.parquet"

    expected = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "amount": [100.0, 250.5, 75.25],
            "is_fraud": [0, 1, 0],
        }
    )

    expected.to_parquet(file_path, index=False)

    result = load_dataset(file_path)

    pd.testing.assert_frame_equal(result, expected)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    """Test that a missing dataset file raises an error."""
    file_path = tmp_path / "missing.csv"

    with pytest.raises(
        DatasetIngestionError,
        match="does not exist",
    ):
        load_dataset(file_path)


def test_directory_is_rejected(tmp_path: Path) -> None:
    """Test that a directory cannot be loaded as a dataset."""
    directory = tmp_path / "transactions.csv"
    directory.mkdir()

    with pytest.raises(
        DatasetIngestionError,
        match="not a file",
    ):
        load_dataset(directory)


def test_unsupported_file_format_is_rejected(
    tmp_path: Path,
) -> None:
    """Test that unsupported file formats are rejected."""
    file_path = tmp_path / "transactions.txt"
    file_path.write_text(
        "transaction_id,amount\n1,100",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetIngestionError,
        match="Unsupported dataset format",
    ):
        load_dataset(file_path)


def test_validate_file_path_accepts_supported_format(
    tmp_path: Path,
) -> None:
    """Test that supported file formats pass validation."""
    file_path = tmp_path / "transactions.csv"
    file_path.write_text(
        "transaction_id,amount\n1,100",
        encoding="utf-8",
    )

    validate_file_path(file_path)


def test_basic_metadata() -> None:
    """Test basic dataset metadata."""
    data = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3],
            "amount": [100.0, 250.5, 75.25],
            "is_fraud": [0, 1, 0],
        }
    )

    metadata = get_basic_metadata(data)

    assert metadata["rows"] == 3
    assert metadata["columns"] == 3
    assert metadata["column_names"] == [
        "transaction_id",
        "amount",
        "is_fraud",
    ]
    assert metadata["missing_values"] == 0
    assert metadata["duplicate_rows"] == 0
    assert isinstance(metadata["memory_usage_mb"], float)


def test_basic_metadata_detects_missing_values_and_duplicates() -> None:
    """Test metadata for missing values and duplicate rows."""
    data = pd.DataFrame(
        {
            "transaction_id": [1, 2, 2],
            "amount": [100.0, None, None],
            "is_fraud": [0, 1, 1],
        }
    )

    metadata = get_basic_metadata(data)

    assert metadata["rows"] == 3
    assert metadata["columns"] == 3
    assert metadata["missing_values"] == 2
    assert metadata["duplicate_rows"] == 1


def test_load_dataset_with_metadata(tmp_path: Path) -> None:
    """Test loading a dataset together with its metadata."""
    file_path = tmp_path / "transactions.csv"

    expected = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "amount": [100.0, 200.0],
            "is_fraud": [0, 1],
        }
    )

    expected.to_csv(file_path, index=False)

    result, metadata = load_dataset_with_metadata(file_path)

    pd.testing.assert_frame_equal(result, expected)

    assert metadata["rows"] == 2
    assert metadata["columns"] == 3
    assert metadata["missing_values"] == 0


# ============================================================
# Validation tests
# ============================================================


def test_valid_dataset_passes_validation() -> None:
    """Test that a valid dataset passes validation."""
    data = create_sample_dataset()

    result = validate_dataset(
        data=data,
        required_columns=[
            "transaction_id",
            "amount",
            "is_fraud",
        ],
        target_column="is_fraud",
    )

    assert result.is_valid is True
    assert result.errors == []


def test_missing_required_column_fails_validation() -> None:
    """Test that missing required columns are reported."""
    data = create_sample_dataset().drop(columns=["amount"])

    result = validate_dataset(
        data=data,
        required_columns=[
            "transaction_id",
            "amount",
            "is_fraud",
        ],
    )

    assert result.is_valid is False
    assert any(
        "amount" in error
        for error in result.errors
    )


def test_missing_target_column_fails_validation() -> None:
    """Test that a missing target column is reported."""
    data = create_sample_dataset().drop(columns=["is_fraud"])

    result = validate_dataset(
        data=data,
        target_column="is_fraud",
    )

    assert result.is_valid is False
    assert any(
        "Target column" in error
        for error in result.errors
    )


def test_invalid_target_values_fail_validation() -> None:
    """Test that target values other than 0 and 1 are rejected."""
    data = create_sample_dataset()
    data.loc[0, "is_fraud"] = 2

    result = validate_dataset(
        data=data,
        target_column="is_fraud",
    )

    assert result.is_valid is False
    assert any(
        "binary values" in error
        for error in result.errors
    )


def test_missing_target_values_fail_validation() -> None:
    """Test that missing target values are rejected."""
    data = create_sample_dataset()
    data.loc[0, "is_fraud"] = None

    result = validate_dataset(
        data=data,
        target_column="is_fraud",
    )

    assert result.is_valid is False
    assert any(
        "missing values" in error
        for error in result.errors
    )


def test_duplicate_rows_generate_warning() -> None:
    """Test that duplicate rows generate a warning."""
    data = create_sample_dataset()

    data = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True,
    )

    result = validate_dataset(data)

    assert result.is_valid is True
    assert any(
        "duplicate rows" in warning
        for warning in result.warnings
    )


def test_missing_values_generate_warning() -> None:
    """Test that missing values generate a warning."""
    data = create_sample_dataset()
    data.loc[0, "amount"] = None

    result = validate_dataset(data)

    assert result.is_valid is True
    assert any(
        "missing values" in warning
        for warning in result.warnings
    )


def test_infinite_values_generate_warning() -> None:
    """Test that infinite numeric values generate a warning."""
    data = create_sample_dataset()
    data.loc[0, "amount"] = float("inf")

    result = validate_dataset(data)

    assert result.is_valid is True
    assert any(
        "infinite values" in warning
        for warning in result.warnings
    )


def test_empty_dataset_fails_validation() -> None:
    """Test that an empty dataset fails validation."""
    data = pd.DataFrame(
        columns=[
            "transaction_id",
            "amount",
            "is_fraud",
        ]
    )

    result = validate_dataset(data)

    assert result.is_valid is False
    assert "Dataset is empty." in result.errors


def test_non_dataframe_input_fails_validation() -> None:
    """Test that non-DataFrame input fails validation."""
    result = validate_dataset(
        data="not a dataframe",  # type: ignore[arg-type]
    )

    assert result.is_valid is False
    assert (
        "Dataset must be a pandas DataFrame."
        in result.errors
    )


def test_validate_dataset_or_raise() -> None:
    """Test that invalid data raises DatasetValidationError."""
    data = create_sample_dataset().drop(columns=["is_fraud"])

    with pytest.raises(
        DatasetValidationError,
        match="Dataset validation failed",
    ):
        validate_dataset_or_raise(
            data=data,
            target_column="is_fraud",
        )


# ============================================================
# Splitting tests
# ============================================================


def test_dataset_split_sizes() -> None:
    """Test train, validation, and test split sizes."""
    data = create_sample_dataset()

    split = split_dataset(
        data=data,
        target_column="is_fraud",
        test_size=0.20,
        validation_size=0.20,
        random_state=42,
    )

    sizes = get_split_sizes(split)

    assert sizes["train"] == 60
    assert sizes["validation"] == 20
    assert sizes["test"] == 20


def test_dataset_split_contains_all_rows() -> None:
    """Test that splitting does not lose or duplicate rows."""
    data = create_sample_dataset()

    split = split_dataset(
        data=data,
        target_column="is_fraud",
        random_state=42,
    )

    total_rows = (
        len(split.train)
        + len(split.validation)
        + len(split.test)
    )

    assert total_rows == len(data)


def test_dataset_split_has_no_overlap() -> None:
    """Test that train, validation, and test rows do not overlap."""
    data = create_sample_dataset()

    split = split_dataset(
        data=data,
        target_column="is_fraud",
        random_state=42,
    )

    assert verify_no_row_overlap(split) is True


def test_dataset_split_is_reproducible() -> None:
    """Test that the same random seed produces the same split."""
    data = create_sample_dataset()

    split_a = split_dataset(
        data=data,
        target_column="is_fraud",
        random_state=42,
    )

    split_b = split_dataset(
        data=data,
        target_column="is_fraud",
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        split_a.train,
        split_b.train,
    )

    pd.testing.assert_frame_equal(
        split_a.validation,
        split_b.validation,
    )

    pd.testing.assert_frame_equal(
        split_a.test,
        split_b.test,
    )


def test_dataset_split_preserves_target_distribution() -> None:
    """Test that stratification preserves fraud rate."""
    data = create_sample_dataset(rows=1000)

    original_rate = data["is_fraud"].mean()

    split = split_dataset(
        data=data,
        target_column="is_fraud",
        random_state=42,
        stratify=True,
    )

    assert split.train["is_fraud"].mean() == original_rate
    assert split.validation["is_fraud"].mean() == original_rate
    assert split.test["is_fraud"].mean() == original_rate


def test_missing_target_column_is_rejected() -> None:
    """Test that a missing target column raises an error."""
    data = create_sample_dataset()

    data = data.drop(columns=["is_fraud"])

    with pytest.raises(
        DataSplittingError,
        match="does not exist",
    ):
        split_dataset(
            data=data,
            target_column="is_fraud",
        )


def test_empty_dataset_is_rejected() -> None:
    """Test that an empty dataset raises an error."""
    data = pd.DataFrame(
        columns=[
            "transaction_id",
            "amount",
            "is_fraud",
        ]
    )

    with pytest.raises(
        DataSplittingError,
        match="empty",
    ):
        split_dataset(
            data=data,
            target_column="is_fraud",
        )


def test_missing_target_values_are_rejected() -> None:
    """Test that missing target values are rejected."""
    data = create_sample_dataset()
    data.loc[0, "is_fraud"] = None

    with pytest.raises(
        DataSplittingError,
        match="contains missing values",
    ):
        split_dataset(
            data=data,
            target_column="is_fraud",
        )


def test_invalid_test_size_is_rejected() -> None:
    """Test that invalid test size is rejected."""
    data = create_sample_dataset()

    with pytest.raises(
        DataSplittingError,
        match="test_size",
    ):
        split_dataset(
            data=data,
            target_column="is_fraud",
            test_size=1.0,
        )


def test_invalid_validation_size_is_rejected() -> None:
    """Test that invalid validation size is rejected."""
    data = create_sample_dataset()

    with pytest.raises(
        DataSplittingError,
        match="validation_size",
    ):
        split_dataset(
            data=data,
            target_column="is_fraud",
            validation_size=1.0,
        )


def test_split_sizes_cannot_exceed_dataset() -> None:
    """Test that validation and test sizes cannot consume all data."""
    data = create_sample_dataset()

    with pytest.raises(
        DataSplittingError,
        match="must be less than 1",
    ):
        split_dataset(
            data=data,
            target_column="is_fraud",
            test_size=0.60,
            validation_size=0.40,
        )