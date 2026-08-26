"""Tests for fraud-model evaluation."""

import numpy as np
import pytest

from src.models.evaluation import (
    EvaluationError,
    evaluate_binary_classifier,
)


def test_evaluation_returns_expected_metrics() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])

    y_probability = np.array(
        [0.05, 0.10, 0.20, 0.80, 0.90, 0.95]
    )

    y_prediction = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    result = evaluate_binary_classifier(
        y_true,
        y_probability,
        y_prediction,
    )

    assert result.roc_auc == pytest.approx(1.0)
    assert result.pr_auc == pytest.approx(1.0)

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)

    assert result.true_negatives == 3
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.true_positives == 3

    assert result.fraud_rate == pytest.approx(0.5)
    assert result.predicted_fraud_count == 3
    assert result.total_samples == 6


def test_as_dict_contains_all_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])

    y_probability = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    y_prediction = np.array(
        [0, 0, 1, 1]
    )

    result = evaluate_binary_classifier(
        y_true,
        y_probability,
        y_prediction,
    )

    metrics = result.as_dict()

    expected_keys = {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "fraud_rate",
        "predicted_fraud_count",
        "total_samples",
    }

    assert set(metrics) == expected_keys


def test_mismatched_lengths_are_rejected() -> None:
    y_true = np.array([0, 1])
    y_probability = np.array([0.1])
    y_prediction = np.array([0, 1])

    with pytest.raises(
        EvaluationError,
        match="same length",
    ):
        evaluate_binary_classifier(
            y_true,
            y_probability,
            y_prediction,
        )


def test_empty_data_is_rejected() -> None:
    with pytest.raises(
        EvaluationError,
        match="cannot be empty",
    ):
        evaluate_binary_classifier(
            np.array([]),
            np.array([]),
            np.array([]),
        )


def test_invalid_target_values_are_rejected() -> None:
    y_true = np.array([0, 1, 2, 1])

    y_probability = np.array(
        [0.1, 0.8, 0.9, 0.7]
    )

    y_prediction = np.array(
        [0, 1, 1, 1]
    )

    with pytest.raises(
        EvaluationError,
        match="binary",
    ):
        evaluate_binary_classifier(
            y_true,
            y_probability,
            y_prediction,
        )


def test_invalid_prediction_values_are_rejected() -> None:
    y_true = np.array([0, 0, 1, 1])

    y_probability = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    y_prediction = np.array(
        [0, 2, 1, 1]
    )

    with pytest.raises(
        EvaluationError,
        match="binary",
    ):
        evaluate_binary_classifier(
            y_true,
            y_probability,
            y_prediction,
        )


def test_nan_probabilities_are_rejected() -> None:
    y_true = np.array([0, 0, 1, 1])

    y_probability = np.array(
        [0.1, np.nan, 0.8, 0.9]
    )

    y_prediction = np.array(
        [0, 0, 1, 1]
    )

    with pytest.raises(
        EvaluationError,
        match="NaN",
    ):
        evaluate_binary_classifier(
            y_true,
            y_probability,
            y_prediction,
        )


def test_probability_out_of_range_is_rejected() -> None:
    y_true = np.array([0, 0, 1, 1])

    y_probability = np.array(
        [0.1, 1.2, 0.8, 0.9]
    )

    y_prediction = np.array(
        [0, 0, 1, 1]
    )

    with pytest.raises(
        EvaluationError,
        match="between 0 and 1",
    ):
        evaluate_binary_classifier(
            y_true,
            y_probability,
            y_prediction,
        )


def test_single_class_target_is_rejected() -> None:
    y_true = np.array([0, 0, 0, 0])

    y_probability = np.array(
        [0.1, 0.2, 0.3, 0.4]
    )

    y_prediction = np.array(
        [0, 0, 0, 0]
    )

    with pytest.raises(
        EvaluationError,
        match="both classes",
    ):
        evaluate_binary_classifier(
            y_true,
            y_probability,
            y_prediction,
        )