"""Train and evaluate the first XGBoost fraud-detection baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.models.evaluation import evaluate_binary_classifier
from src.models.xgboost_baseline import XGBoostBaseline, XGBoostConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = RAW_DATA_DIR / "train_identity.csv"

TARGET_COLUMN = "isFraud"
JOIN_COLUMN = "TransactionID"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def load_training_data() -> pd.DataFrame:
    """Load and merge the IEEE-CIS training transaction and identity data."""
    if not TRAIN_TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            f"Training transaction file not found: "
            f"{TRAIN_TRANSACTION_PATH}"
        )

    if not TRAIN_IDENTITY_PATH.exists():
        raise FileNotFoundError(
            f"Training identity file not found: "
            f"{TRAIN_IDENTITY_PATH}"
        )

    print("Loading training transaction data...")
    transactions = pd.read_csv(TRAIN_TRANSACTION_PATH)

    print("Loading training identity data...")
    identity = pd.read_csv(TRAIN_IDENTITY_PATH)

    if JOIN_COLUMN not in transactions.columns:
        raise ValueError(
            f"'{JOIN_COLUMN}' is missing from transaction data."
        )

    if JOIN_COLUMN not in identity.columns:
        raise ValueError(
            f"'{JOIN_COLUMN}' is missing from identity data."
        )

    if TARGET_COLUMN not in transactions.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' is missing from transaction data."
        )

    if transactions[JOIN_COLUMN].duplicated().any():
        raise ValueError(
            "Transaction data contains duplicate TransactionID values."
        )

    if identity[JOIN_COLUMN].duplicated().any():
        raise ValueError(
            "Identity data contains duplicate TransactionID values."
        )

    data = transactions.merge(
        identity,
        on=JOIN_COLUMN,
        how="left",
        validate="one_to_one",
    )

    print(f"Merged dataset shape: {data.shape}")

    return data


def split_train_validation(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the existing training dataset into train and validation sets.

    The official IEEE-CIS test set is not used or modified here.
    """
    from sklearn.model_selection import train_test_split

    train_data, validation_data = train_test_split(
        data,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
    )


def prepare_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the fraud target and identifiers."""
    target = data[TARGET_COLUMN].copy()

    features = data.drop(
        columns=[
            TARGET_COLUMN,
            JOIN_COLUMN,
        ],
        errors="ignore",
    )

    return features, target


def print_dataset_summary(
    target: pd.Series,
    name: str,
) -> None:
    """Print basic class-distribution information."""
    fraud_count = int((target == 1).sum())
    legitimate_count = int((target == 0).sum())
    total = len(target)

    fraud_rate = (
        fraud_count / total
        if total > 0
        else 0.0
    )

    print()
    print(f"=== {name.upper()} DISTRIBUTION ===")
    print(f"Total transactions:    {total:,}")
    print(f"Fraud transactions:    {fraud_count:,}")
    print(f"Legitimate:            {legitimate_count:,}")
    print(f"Fraud rate:            {fraud_rate:.4%}")


def main() -> None:
    """Run the baseline training and validation evaluation."""
    print("=== XGBOOST FRAUD BASELINE ===")
    print()

    data = load_training_data()

    if data.empty:
        raise ValueError("Merged training dataset is empty.")

    train_data, validation_data = split_train_validation(
        data
    )

    print(
        f"Training rows:   {len(train_data):,}"
    )
    print(
        f"Validation rows: {len(validation_data):,}"
    )

    _, train_target = prepare_features(train_data)
    _, validation_target = prepare_features(
        validation_data
    )

    print_dataset_summary(
        train_target,
        "training",
    )

    print_dataset_summary(
        validation_target,
        "validation",
    )

    train_features, _ = prepare_features(train_data)
    validation_features, _ = prepare_features(
        validation_data
    )

    print()
    print("Fitting feature pipeline...")

    feature_pipeline = FeaturePipeline.fit(
        train_features,
    )

    print("Transforming training data...")
    transformed_train = feature_pipeline.transform(
        train_features
    )

    print("Transforming validation data...")
    transformed_validation = feature_pipeline.transform(
        validation_features
    )

    print(
        f"Training feature shape: "
        f"{transformed_train.shape}"
    )

    print(
        f"Validation feature shape: "
        f"{transformed_validation.shape}"
    )

    print()
    print("Training XGBoost baseline...")

    model = XGBoostBaseline(
        XGBoostConfig(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    model.fit(
        transformed_train,
        train_target,
    )

    print("Generating validation predictions...")

    probabilities = model.predict_proba(
        transformed_validation
    )

    predictions = model.predict(
        transformed_validation,
        threshold=0.5,
    )

    result = evaluate_binary_classifier(
        validation_target.to_numpy(),
        probabilities,
        predictions,
    )

    print()
    print("=== BASELINE RESULTS ===")
    print(f"ROC-AUC:              {result.roc_auc:.4f}")
    print(f"PR-AUC:               {result.pr_auc:.4f}")
    print(f"Precision:            {result.precision:.4f}")
    print(f"Recall:               {result.recall:.4f}")
    print(f"F1:                   {result.f1:.4f}")

    print()
    print("=== CONFUSION MATRIX ===")
    print(
        f"True negatives:       "
        f"{result.true_negatives:,}"
    )
    print(
        f"False positives:      "
        f"{result.false_positives:,}"
    )
    print(
        f"False negatives:      "
        f"{result.false_negatives:,}"
    )
    print(
        f"True positives:       "
        f"{result.true_positives:,}"
    )

    print()
    print("=== PREDICTION SUMMARY ===")
    print(
        f"Actual fraud rate:    "
        f"{result.fraud_rate:.4%}"
    )
    print(
        f"Predicted fraud:      "
        f"{result.predicted_fraud_count:,}"
    )
    print(
        f"Validation samples:   "
        f"{result.total_samples:,}"
    )

    print()
    print("Baseline evaluation completed successfully.")


if __name__ == "__main__":
    main()