"""Categorical feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class CategoricalFeatureError(ValueError):
    """Raised when categorical feature processing fails."""


@dataclass(frozen=True)
class FrequencyEncoder:
    """
    Frequency encoder fitted on training data.

    The encoder stores the proportion of observations belonging
    to each category for every selected column.
    """

    mappings: dict[str, dict[object, float]]
    columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        columns: list[str] | tuple[str, ...],
    ) -> FrequencyEncoder:
        """
        Fit frequency mappings using training data.
        """
        if not isinstance(data, pd.DataFrame):
            raise CategoricalFeatureError(
                "Input data must be a pandas DataFrame."
            )

        if data.empty:
            raise CategoricalFeatureError(
                "Cannot fit encoder on an empty dataset."
            )

        selected_columns = tuple(columns)

        if not selected_columns:
            raise CategoricalFeatureError(
                "At least one categorical column is required."
            )

        missing_columns = [
            column
            for column in selected_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise CategoricalFeatureError(
                "Categorical columns do not exist: "
                f"{missing_columns}"
            )

        mappings: dict[str, dict[object, float]] = {}

        for column in selected_columns:
            frequencies = data[column].value_counts(
                normalize=True,
                dropna=False,
            )

            mappings[column] = frequencies.to_dict()

        return cls(
            mappings=mappings,
            columns=selected_columns,
        )

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform categorical columns using learned frequencies.

        Unknown categories receive frequency 0.0.
        """
        if not isinstance(data, pd.DataFrame):
            raise CategoricalFeatureError(
                "Input data must be a pandas DataFrame."
            )

        missing_columns = [
            column
            for column in self.columns
            if column not in data.columns
        ]

        if missing_columns:
            raise CategoricalFeatureError(
                "Categorical columns do not exist: "
                f"{missing_columns}"
            )

        result = data.copy()

        for column in self.columns:
            mapping = self.mappings[column]

            result[f"{column}_frequency"] = (
                result[column]
                .map(mapping)
                .fillna(0.0)
                .astype("float32")
            )

        return result