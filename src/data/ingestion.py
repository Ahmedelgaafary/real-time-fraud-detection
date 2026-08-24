from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet"}


class DatasetIngestionError(Exception):
    """Raised when a dataset cannot be loaded."""


def validate_file_path(file_path: Path) -> None:
    """
    Validate that the dataset file exists and has a supported format.

    Parameters
    ----------
    file_path:
        Path to the dataset file.

    Raises
    ------
    DatasetIngestionError
        If the path is invalid, does not exist, is not a file,
        or has an unsupported extension.
    """
    if not file_path.exists():
        raise DatasetIngestionError(
            f"Dataset file does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise DatasetIngestionError(
            f"Dataset path is not a file: {file_path}"
        )

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetIngestionError(
            f"Unsupported dataset format: {file_path.suffix}. "
            f"Supported formats: {supported}"
        )


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load a supported dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path:
        Path to a CSV or Parquet dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    DatasetIngestionError
        If the dataset cannot be validated or loaded.
    """
    path = Path(file_path)

    validate_file_path(path)

    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)

        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DatasetIngestionError(
            f"Failed to load dataset: {path}"
        ) from exc

    raise DatasetIngestionError(
        f"Unsupported dataset format: {path.suffix}"
    )


def get_basic_metadata(data: pd.DataFrame) -> dict[str, object]:
    """
    Return basic metadata about a loaded dataset.

    Parameters
    ----------
    data:
        Loaded dataset.

    Returns
    -------
    dict
        Basic dataset metadata.
    """
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "column_names": list(data.columns),
        "memory_usage_mb": round(
            data.memory_usage(deep=True).sum() / (1024**2),
            2,
        ),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
    }


def load_dataset_with_metadata(
    file_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Load a dataset and return it together with basic metadata.
    """
    data = load_dataset(file_path)
    metadata = get_basic_metadata(data)

    return data, metadata


if __name__ == "__main__":
    print("Dataset ingestion module.")
    print("Use load_dataset() to load a dataset.")