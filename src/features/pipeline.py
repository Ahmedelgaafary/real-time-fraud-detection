"""Unified feature engineering pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.categorical_features import FrequencyEncoder
from src.features.numeric_features import NumericImputer
from src.features.transaction_features import (
    add_missingness_features,
    add_transaction_amount_features,
    add_transaction_time_features,
    get_missingness_columns,
)


class FeaturePipelineError(ValueError):
    """Raised when feature pipeline operations fail."""


DEFAULT_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
)


@dataclass(frozen=True)
class FeaturePipeline:
    """Feature engineering pipeline fitted on training data."""

    categorical_encoder: FrequencyEncoder
    categorical_columns: tuple[str, ...]
    missingness_columns: tuple[str, ...]
    numeric_imputer: NumericImputer
    numeric_columns: tuple[str, ...]
    missingness_threshold: float

    @classmethod
    def fit(
        cls,
        train_data: pd.DataFrame,
        categorical_columns: tuple[str, ...] = (
            DEFAULT_CATEGORICAL_COLUMNS
        ),
        missingness_threshold: float = 0.05,
    ) -> FeaturePipeline:
        """
        Fit the feature pipeline using training data only.

        All learned statistics are derived exclusively from the
        training dataset to prevent data leakage.
        """
        if not isinstance(train_data, pd.DataFrame):
            raise FeaturePipelineError(
                "train_data must be a pandas DataFrame."
            )

        if train_data.empty:
            raise FeaturePipelineError(
                "Cannot fit feature pipeline on empty data."
            )

        if not 0 <= missingness_threshold <= 1:
            raise FeaturePipelineError(
                "missingness_threshold must be between 0 and 1."
            )

        selected_categorical_columns = tuple(categorical_columns)

        if not selected_categorical_columns:
            raise FeaturePipelineError(
                "At least one categorical column is required."
            )

        missing_categorical = [
            column
            for column in selected_categorical_columns
            if column not in train_data.columns
        ]

        if missing_categorical:
            raise FeaturePipelineError(
                "Categorical columns do not exist: "
                f"{missing_categorical}"
            )

        try:
            categorical_encoder = FrequencyEncoder.fit(
                train_data,
                columns=selected_categorical_columns,
            )
        except Exception as exc:
            raise FeaturePipelineError(
                "Failed to fit categorical encoder."
            ) from exc

        try:
            missingness_columns = get_missingness_columns(
                train_data,
                threshold=missingness_threshold,
                exclude_columns=("isFraud",),
            )
        except Exception as exc:
            raise FeaturePipelineError(
                "Failed to determine missingness columns."
            ) from exc

        numeric_columns = tuple(
            column
            for column in train_data.select_dtypes(
                include="number"
            ).columns
            if column != "isFraud"
        )

        try:
            numeric_imputer = NumericImputer.fit(
                train_data,
                columns=numeric_columns,
            )
        except Exception as exc:
            raise FeaturePipelineError(
                "Failed to fit numeric imputer."
            ) from exc

        return cls(
            categorical_encoder=categorical_encoder,
            categorical_columns=selected_categorical_columns,
            missingness_columns=missingness_columns,
            numeric_imputer=numeric_imputer,
            numeric_columns=numeric_columns,
            missingness_threshold=missingness_threshold,
        )

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform data using statistics learned from training data.

        Original columns are preserved. Additional engineered columns
        are added by the feature engineering stages.

        The pipeline itself does not require the final output to be
        entirely numeric. Model-specific code is responsible for
        selecting the numeric features required by the estimator.
        """
        if not isinstance(data, pd.DataFrame):
            raise FeaturePipelineError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise FeaturePipelineError(
                "Cannot transform empty data."
            )

        missing_categorical = [
            column
            for column in self.categorical_columns
            if column not in data.columns
        ]

        if missing_categorical:
            raise FeaturePipelineError(
                "Categorical columns do not exist: "
                f"{missing_categorical}"
            )

        missing_numeric = [
            column
            for column in self.numeric_columns
            if column not in data.columns
        ]

        if missing_numeric:
            raise FeaturePipelineError(
                "Numeric columns do not exist: "
                f"{missing_numeric}"
            )

        try:
            result = add_transaction_time_features(data)

            result = add_transaction_amount_features(result)

            result = add_missingness_features(
                result,
                columns=self.missingness_columns,
            )

            # Add frequency-encoded representations while preserving
            # the original categorical columns.
            result = self.categorical_encoder.transform(result)

            # Apply the training-fitted numeric imputation strategy.
            result = self.numeric_imputer.transform(result)

        except Exception as exc:
            if isinstance(exc, FeaturePipelineError):
                raise

            raise FeaturePipelineError(
                "Feature transformation failed."
            ) from exc

        return result

    def fit_transform(
        self,
        train_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit the pipeline on training data and transform it.

        This method creates a new fitted pipeline using only
        the supplied training data.
        """
        pipeline = type(self).fit(
            train_data,
            categorical_columns=self.categorical_columns,
            missingness_threshold=self.missingness_threshold,
        )

        return pipeline.transform(train_data)