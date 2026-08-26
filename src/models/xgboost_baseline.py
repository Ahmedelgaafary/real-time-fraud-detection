"""XGBoost baseline model for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class XGBoostModelError(ValueError):
    """Raised when XGBoost model operations fail."""


@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration for the baseline XGBoost classifier."""

    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1
    eval_metric: str = "logloss"


class XGBoostBaseline:
    """Train and serve a baseline XGBoost fraud classifier."""

    def __init__(
        self,
        config: XGBoostConfig | None = None,
    ) -> None:
        self.config = config or XGBoostConfig()
        self.model: XGBClassifier | None = None
        self.feature_names: tuple[str, ...] | None = None

    def _validate_features(
        self,
        features: pd.DataFrame,
    ) -> None:
        if not isinstance(features, pd.DataFrame):
            raise XGBoostModelError(
                "Features must be a pandas DataFrame."
            )

        if features.empty:
            raise XGBoostModelError(
                "Features cannot be empty."
            )

        non_numeric = features.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        if non_numeric:
            raise XGBoostModelError(
                "Features must be numeric. "
                f"Non-numeric columns: {non_numeric}"
            )

        if features.isna().any().any():
            raise XGBoostModelError(
                "Features cannot contain missing values."
            )

    def _validate_target(
        self,
        target: pd.Series,
    ) -> None:
        if not isinstance(target, pd.Series):
            raise XGBoostModelError(
                "Target must be a pandas Series."
            )

        if target.empty:
            raise XGBoostModelError(
                "Target cannot be empty."
            )

        if target.isna().any():
            raise XGBoostModelError(
                "Target cannot contain missing values."
            )

        unique_values = set(target.unique())

        if not unique_values.issubset({0, 1}):
            raise XGBoostModelError(
                "Target must contain only binary values 0 and 1."
            )

        if len(unique_values) < 2:
            raise XGBoostModelError(
                "Target must contain both classes."
            )

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> XGBoostBaseline:
        """Fit the baseline model on training data."""
        self._validate_features(features)
        self._validate_target(target)

        if len(features) != len(target):
            raise XGBoostModelError(
                "Features and target must contain the same "
                "number of rows."
            )

        self.feature_names = tuple(features.columns)

        self.model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            eval_metric=self.config.eval_metric,
            objective="binary:logistic",
        )

        self.model.fit(features, target)

        return self

    def _validate_fitted(self) -> None:
        if self.model is None:
            raise XGBoostModelError(
                "Model has not been fitted."
            )

    def _validate_prediction_features(
        self,
        features: pd.DataFrame,
    ) -> None:
        self._validate_features(features)

        if self.feature_names is None:
            raise XGBoostModelError(
                "Feature schema is not available."
            )

        if tuple(features.columns) != self.feature_names:
            raise XGBoostModelError(
                "Prediction features do not match the "
                "training feature schema."
            )

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Return fraud probabilities."""
        self._validate_fitted()
        self._validate_prediction_features(features)

        assert self.model is not None

        return self.model.predict_proba(features)[:, 1]

    def predict(
        self,
        features: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary fraud predictions."""
        if not 0 < threshold < 1:
            raise XGBoostModelError(
                "threshold must be between 0 and 1."
            )

        probabilities = self.predict_proba(features)

        return (probabilities >= threshold).astype(int)