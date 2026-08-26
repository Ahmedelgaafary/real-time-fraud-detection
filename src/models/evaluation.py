"""Evaluation utilities for fraud detection models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EvaluationError(ValueError):
    """Raised when model evaluation fails."""


@dataclass(frozen=True)
class EvaluationResult:
    """Container for binary fraud-model evaluation metrics."""

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    fraud_rate: float
    predicted_fraud_count: int
    total_samples: int

    def as_dict(self) -> dict[str, float | int]:
        """Return metrics as a serializable dictionary."""
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_positives": self.true_positives,
            "fraud_rate": self.fraud_rate,
            "predicted_fraud_count": self.predicted_fraud_count,
            "total_samples": self.total_samples,
        }


def _validate_inputs(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    y_prediction: np.ndarray,
) -> None:
    """Validate evaluation inputs."""
    if not isinstance(y_true, np.ndarray):
        raise EvaluationError("y_true must be a NumPy array.")

    if not isinstance(y_probability, np.ndarray):
        raise EvaluationError(
            "y_probability must be a NumPy array."
        )

    if not isinstance(y_prediction, np.ndarray):
        raise EvaluationError(
            "y_prediction must be a NumPy array."
        )

    if len(y_true) == 0:
        raise EvaluationError(
            "Evaluation data cannot be empty."
        )

    if not (
        len(y_true)
        == len(y_probability)
        == len(y_prediction)
    ):
        raise EvaluationError(
            "Evaluation arrays must have the same length."
        )

    if np.isnan(y_probability).any():
        raise EvaluationError(
            "y_probability cannot contain NaN values."
        )

    if not np.isfinite(y_probability).all():
        raise EvaluationError(
            "y_probability must contain finite values."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise EvaluationError(
            "y_true must contain only binary values 0 and 1."
        )

    if not np.isin(y_prediction, [0, 1]).all():
        raise EvaluationError(
            "y_prediction must contain only binary values 0 and 1."
        )

    if len(np.unique(y_true)) < 2:
        raise EvaluationError(
            "y_true must contain both classes."
        )

    if ((y_probability < 0) | (y_probability > 1)).any():
        raise EvaluationError(
            "y_probability values must be between 0 and 1."
        )


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    y_prediction: np.ndarray,
) -> EvaluationResult:
    """
    Evaluate a binary fraud classifier.

    ROC-AUC and PR-AUC use probability scores, while
    precision, recall, F1, and the confusion matrix use
    thresholded predictions.
    """
    _validate_inputs(
        y_true,
        y_probability,
        y_prediction,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_true,
        y_probability,
    )

    precision = precision_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_prediction,
        labels=[0, 1],
    ).ravel()

    total_samples = len(y_true)

    fraud_rate = float(
        np.mean(y_true == 1)
    )

    predicted_fraud_count = int(
        np.sum(y_prediction == 1)
    )

    return EvaluationResult(
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        fraud_rate=fraud_rate,
        predicted_fraud_count=predicted_fraud_count,
        total_samples=total_samples,
    )