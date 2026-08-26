# XGBoost Fraud Detection Baseline

## 1. Purpose

This report establishes the first machine-learning baseline for the
Real-Time Fraud Detection project.

The baseline provides a reproducible reference point for evaluating
future improvements such as:

- class-weighted XGBoost
- threshold optimization
- anomaly detection
- ensemble models
- explainability
- real-time inference optimization

All future model improvements should be compared against this baseline
using the same validation protocol where applicable.

---

## 2. Dataset

Dataset: IEEE-CIS Fraud Detection

Source:

- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

The transaction and identity datasets are joined using:

```text
TransactionID