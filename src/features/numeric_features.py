"""Numeric feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class NumericFeatureError(ValueError):
    """Raised when numeric feature preprocessing fails."""


@dataclass(frozen=True)
class NumericImputer:
    """Training-fitted median imputer for numeric features."""

    medians: dict[str, float]
    columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        columns: tuple[str, ...] | None = None,
    ) -> NumericImputer:
        """
        Fit median values using training data only.

        All imputation statistics are learned exclusively from
        the supplied training dataset.
        """
        if not isinstance(data, pd.DataFrame):
            raise NumericFeatureError(
                "Input data must be a pandas DataFrame."
            )

        if data.empty:
            raise NumericFeatureError(
                "Cannot fit imputer on empty data."
            )

        if columns is None:
            selected_columns = tuple(
                column
                for column in data.columns
                if (
                    pd.api.types.is_numeric_dtype(
                        data[column]
                    )
                    or data[column].isna().all()
                )
            )
        else:
            selected_columns = tuple(columns)

        if not selected_columns:
            raise NumericFeatureError(
                "No numeric columns were selected."
            )

        missing_columns = [
            column
            for column in selected_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise NumericFeatureError(
                "Numeric columns do not exist: "
                f"{missing_columns}"
            )

        non_numeric = [
            column
            for column in selected_columns
            if (
                not pd.api.types.is_numeric_dtype(
                    data[column]
                )
                and not data[column].isna().all()
            )
        ]

        if non_numeric:
            raise NumericFeatureError(
                "Selected columns must be numeric: "
                f"{non_numeric}"
            )

        medians: dict[str, float] = {}

        for column in selected_columns:
            if data[column].isna().all():
                raise NumericFeatureError(
                    f"Column '{column}' contains no valid numeric values."
                )

            median = data[column].median()

            if pd.isna(median):
                raise NumericFeatureError(
                    f"Column '{column}' contains no valid numeric values."
                )

            medians[column] = float(median)

        return cls(
            medians=medians,
            columns=selected_columns,
        )

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply training-fitted median imputation.

        Missing-value indicators are created before imputation
        so the model retains information about original missingness.
        """
        if not isinstance(data, pd.DataFrame):
            raise NumericFeatureError(
                "Input data must be a pandas DataFrame."
            )

        if data.empty:
            raise NumericFeatureError(
                "Cannot transform empty data."
            )

        missing_columns = [
            column
            for column in self.columns
            if column not in data.columns
        ]

        if missing_columns:
            raise NumericFeatureError(
                "Numeric columns do not exist: "
                f"{missing_columns}"
            )

        result = data.copy()

        for column in self.columns:
            result[f"{column}_was_missing"] = (
                result[column]
                .isna()
                .astype("int8")
            )

            result[column] = (
                result[column]
                .fillna(self.medians[column])
            )

        return result