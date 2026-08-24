# Real-Time Fraud & Anomaly Detection

A production-oriented fraud detection system combining **supervised machine learning, unsupervised anomaly detection, explainable AI, real-time inference, streaming, and MLOps**.

The goal is to demonstrate how a modern FinTech fraud detection pipeline can identify suspicious transactions under extreme class imbalance while providing an understandable explanation for every high-risk decision.

---

## Project Status

**Version:** 0.1.0
**Status:** 🚧 Initial architecture and development

The project is being developed incrementally from data ingestion through real-time deployment and monitoring.

---

## Objectives

This project focuses on four major challenges in real-time financial fraud detection:

### 1. Extreme Class Imbalance

Fraudulent transactions are typically a very small fraction of all transactions.

The project will investigate:

* Class weighting
* SMOTE
* SMOTE + Tomek Links
* Threshold optimization
* Precision/Recall trade-offs
* PR-AUC as a primary evaluation metric

The resampling strategy will be applied only to the training data to prevent data leakage.

### 2. Hybrid Fraud & Anomaly Detection

The system will combine complementary approaches:

```text
                Transaction
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       XGBoost              Autoencoder
          │                     │
          ▼                     ▼
   Fraud Probability      Anomaly Score
          │                     │
          └──────────┬──────────┘
                     ▼
                Risk Engine
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        ALLOW      REVIEW      BLOCK
```

The supervised model learns from known fraud cases, while the anomaly detector identifies transactions that differ significantly from normal behavior.

### 3. Explainable AI

Every suspicious transaction should be accompanied by an explanation.

SHAP will be used to identify the features that contributed most strongly to the model's prediction.

Example:

```text
Decision: BLOCK
Fraud Probability: 0.94

Top contributing factors:

transaction_amount       +0.31
transaction_velocity     +0.24
device_change            +0.19
location_anomaly         +0.14
merchant_risk            +0.08
```

The objective is to move beyond:

> "This transaction is fraudulent."

toward:

> "This transaction was flagged because these specific factors increased its risk."

### 4. Real-Time Fraud Detection

The final system will support simulated transaction streaming:

```text
Transaction Source
       │
       ▼
     Kafka
       │
       ▼
   Consumer
       │
       ▼
   FastAPI
       │
       ▼
 Fraud Detection
       │
       ├──────────────┐
       ▼              ▼
   Decision        Explanation
       │              │
       └───────┬──────┘
               ▼
          Monitoring
```

The system will measure inference latency and target:

```text
P50 < 50 ms
P95 < 50 ms
```

These targets will be validated through benchmarking rather than assumed.

---

## System Architecture

```text
                         ┌──────────────────────┐
                         │ Transaction Simulator│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Kafka / Redis        │
                         │ Streaming Layer      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ FastAPI              │
                         │ Real-Time Scoring     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering   │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
              ┌──────────────┐             ┌──────────────┐
              │ XGBoost      │             │ Autoencoder  │
              │ Fraud Model  │             │ Anomaly      │
              └──────┬───────┘             └──────┬───────┘
                     │                            │
                     └────────────┬───────────────┘
                                  ▼
                         ┌──────────────────┐
                         │ Risk Engine      │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                  ALLOW         REVIEW         BLOCK
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ SHAP Explainability│
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Monitoring / MLOps│
                         └──────────────────┘
```

---

## Project Structure

```text
real-time-fraud-detection/
│
├── .github/
│   └── workflows/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── models/
│   ├── xgboost/
│   ├── autoencoder/
│   └── artifacts/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── explainability/
│   ├── streaming/
│   ├── monitoring/
│   └── api/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
│
├── docker/
├── scripts/
│
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

---

## Technology Stack

### Machine Learning

* Python 3.12
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* SHAP

### API & Real-Time Processing

* FastAPI
* Uvicorn
* Kafka
* Redis

### MLOps

* MLflow
* Prometheus
* Docker
* GitHub Actions

### Testing & Code Quality

* Pytest
* Ruff
* MyPy

---

## Data Pipeline

The data pipeline will follow:

```text
Raw Data
   │
   ▼
Ingestion
   │
   ▼
Validation
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Feature Engineering
   │
   ▼
Training Dataset
```

Raw datasets will remain outside version control.

---

## Modeling Pipeline

The modeling workflow will be:

```text
Baseline
   │
   ▼
XGBoost
   │
   ▼
Class Weighting
   │
   ▼
SMOTE
   │
   ▼
SMOTE + Tomek
   │
   ▼
Threshold Optimization
   │
   ▼
Best Supervised Model
```

Anomaly detection will then be evaluated separately using models such as:

* Autoencoder
* Isolation Forest

The final risk engine will combine the strongest components.

---

## Evaluation

Because fraud detection is highly imbalanced, accuracy will not be the primary metric.

The project will report:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score
* Confusion matrix
* False-positive rate
* False-negative rate
* Detection threshold
* P50 latency
* P95 latency
* Throughput

Special attention will be given to the cost of false positives and false negatives.

---

## Explainability

The system will provide transaction-level explanations using SHAP.

Example API response:

```json
{
  "transaction_id": "TX_102938",
  "fraud_probability": 0.94,
  "anomaly_score": 0.87,
  "risk_score": 0.93,
  "decision": "BLOCK",
  "model_version": "fraud-xgb-v1",
  "explanation": [
    {
      "feature": "transaction_amount",
      "impact": 0.31
    },
    {
      "feature": "transaction_velocity",
      "impact": 0.24
    }
  ]
}
```

---

## MLOps

The project will eventually support:

```text
Experiment Tracking
        ↓
Model Versioning
        ↓
Model Deployment
        ↓
Prediction Monitoring
        ↓
Latency Monitoring
        ↓
Data Drift Detection
        ↓
Model Performance Monitoring
        ↓
Retraining
```

MLflow will be used for experiment and model lifecycle management.

---

## Testing Strategy

Testing will cover three levels:

### Unit Tests

Individual components:

* Data validation
* Feature engineering
* Models
* Threshold logic
* Risk engine
* SHAP explanations

### Integration Tests

End-to-end components:

```text
Data → Features → Model → Risk Engine
```

and:

```text
HTTP Request → FastAPI → Prediction
```

### Performance Tests

The real-time scoring service will be benchmarked for:

* P50 latency
* P95 latency
* Throughput
* Concurrent requests

---

## Development Roadmap

### Phase 1 — Foundation

* [x] Repository architecture
* [x] Project directory structure
* [x] `.gitignore`
* [x] `pyproject.toml`
* [x] Initial README
* [ ] CI workflow
* [ ] Development environment verification

### Phase 2 — Data Pipeline

* [ ] Dataset acquisition
* [ ] Data ingestion
* [ ] Schema validation
* [ ] Data quality checks
* [ ] Train/validation/test splitting
* [ ] Leakage checks

### Phase 3 — Exploratory Data Analysis

* [ ] Fraud distribution
* [ ] Missing-value analysis
* [ ] Feature analysis
* [ ] Fraud vs legitimate behavior
* [ ] Temporal analysis
* [ ] Potential leakage investigation

### Phase 4 — Baseline Models

* [ ] Logistic Regression
* [ ] Baseline XGBoost
* [ ] Evaluation framework

### Phase 5 — Imbalance Handling

* [ ] Class weighting
* [ ] SMOTE
* [ ] Tomek Links
* [ ] Threshold optimization
* [ ] Comparative evaluation

### Phase 6 — Anomaly Detection

* [ ] Autoencoder
* [ ] Isolation Forest
* [ ] Anomaly threshold selection

### Phase 7 — Hybrid Risk Engine

* [ ] Fraud probability
* [ ] Anomaly score
* [ ] Risk aggregation
* [ ] Allow / Review / Block decisions

### Phase 8 — Explainability

* [ ] SHAP integration
* [ ] Feature contribution extraction
* [ ] Human-readable explanations

### Phase 9 — Real-Time API

* [ ] FastAPI
* [ ] Request validation
* [ ] Model loading
* [ ] Prediction endpoint
* [ ] Explanation endpoint
* [ ] Health checks

### Phase 10 — Streaming

* [ ] Transaction simulator
* [ ] Kafka producer
* [ ] Kafka consumer
* [ ] Redis integration

### Phase 11 — MLOps

* [ ] MLflow
* [ ] Model versioning
* [ ] Monitoring
* [ ] Drift detection
* [ ] Docker
* [ ] CI/CD

### Phase 12 — Production Benchmark

* [ ] Latency benchmark
* [ ] Throughput benchmark
* [ ] End-to-end test
* [ ] Final model comparison
* [ ] Final documentation

---

## Design Principles

This project follows several engineering principles:

1. **No data leakage**
2. **Reproducible experiments**
3. **Explainable predictions**
4. **Modular architecture**
5. **Testable components**
6. **Measured performance**
7. **Versioned models**
8. **Production-oriented deployment**
9. **Monitoring and observability**
10. **Security-conscious handling of financial data**

---

## Disclaimer

This project is an educational and portfolio implementation of a fraud detection system. It is not intended for direct production use in financial decision-making without appropriate validation, security controls, regulatory review, and domain-specific risk management.
