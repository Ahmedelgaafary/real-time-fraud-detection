"""Tests for the unified feature pipeline."""

import pandas as pd
import pytest

from src.features.pipeline import (
    FeaturePipeline,
    FeaturePipelineError,
)


def create_training_data() -> pd.DataFrame:
    """Create representative training data."""
    return pd.DataFrame(
        {
            "TransactionDT": [
                100,
                200,
                300,
                400,
            ],
            "TransactionAmt": [
                10.0,
                25.5,
                50.0,
                100.25,
            ],
            "ProductCD": [
                "W",
                "W",
                "C",
                "H",
            ],
            "card4": [
                "visa",
                "visa",
                "mastercard",
                "visa",
            ],
            "card6": [
                "debit",
                "credit",
                "debit",
                "debit",
            ],
            "P_emaildomain": [
                "gmail.com",
                "gmail.com",
                "yahoo.com",
                "gmail.com",
            ],
            "R_emaildomain": [
                None,
                None,
                "gmail.com",
                None,
            ],
            "optional_feature": [
                1.0,
                None,
                3.0,
                None,
            ],
            "isFraud": [
                0,
                1,
                0,
                0,
            ],
        }
    )


def test_pipeline_fit() -> None:
    """Test pipeline fitting."""
    data = create_training_data()

    pipeline = FeaturePipeline.fit(data)

    assert pipeline.categorical_columns == (
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
    )

    assert "optional_feature" in pipeline.missingness_columns


def test_pipeline_transform_adds_features() -> None:
    """Test complete feature transformation."""
    data = create_training_data()

    pipeline = FeaturePipeline.fit(data)

    result = pipeline.transform(data)

    assert "transaction_day" in result.columns
    assert "transaction_hour" in result.columns
    assert "transaction_amount_log" in result.columns
    assert "transaction_amount_decimal" in result.columns
    assert "transaction_amount_is_round" in result.columns
    assert "optional_feature_is_missing" in result.columns
    assert "ProductCD_frequency" in result.columns


def test_pipeline_preserves_original_columns() -> None:
    """Test raw columns remain available."""
    data = create_training_data()

    pipeline = FeaturePipeline.fit(data)

    result = pipeline.transform(data)

    for column in data.columns:
        assert column in result.columns


def test_pipeline_uses_training_missingness_schema() -> None:
    """Test missingness indicators come from training data."""
    train = create_training_data()

    validation = train.copy()

    validation["new_feature"] = [
        None,
        None,
        None,
        None,
    ]

    pipeline = FeaturePipeline.fit(train)

    result = pipeline.transform(validation)

    assert "optional_feature_is_missing" in result.columns
    assert "new_feature_is_missing" not in result.columns


def test_pipeline_handles_unseen_categories() -> None:
    """Test unseen validation categories."""
    train = create_training_data()

    validation = train.copy()

    validation.loc[0, "ProductCD"] = "NEW_CATEGORY"

    pipeline = FeaturePipeline.fit(train)

    result = pipeline.transform(validation)

    assert (
        result.loc[0, "ProductCD_frequency"] == 0.0
    )


def test_pipeline_rejects_empty_training_data() -> None:
    """Test empty training data."""
    data = pd.DataFrame()

    with pytest.raises(
        FeaturePipelineError,
        match="empty data",
    ):
        FeaturePipeline.fit(data)


def test_pipeline_rejects_missing_categorical_column() -> None:
    """Test missing categorical configuration."""
    data = create_training_data()

    with pytest.raises(
        FeaturePipelineError,
        match="do not exist",
    ):
        FeaturePipeline.fit(
            data,
            categorical_columns=("missing_column",),
        )


def test_pipeline_does_not_modify_training_data() -> None:
    """Test non-mutating pipeline behavior."""
    data = create_training_data()

    original_columns = data.columns.tolist()

    pipeline = FeaturePipeline.fit(data)
    pipeline.transform(data)

    assert data.columns.tolist() == original_columns